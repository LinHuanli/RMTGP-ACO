"""从已完成控制实验和定向重放提取历史路径强化的作用链。"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np

from .common import atomic_json, atomic_npz, digest, file_hash, now, read_json
from .diagnostic_forks import transition_distribution
from .evaluate import program_entries
from .explanation_analysis import canonical_edges
from .explanation_results import publish_matched_states
from .mechanism_visualization import (
    BEHAVIORS,
    CHAMPION,
    REGIMES,
    SAMPLE_ITERATIONS,
    STORAGE_CAP_BYTES,
)
from .prepare import batch
from .statistics import write_csv

PHASES = ("更新前", "蒸发和下界保护后", "沉积后", "重启和可选上界处理后")
SOURCE_POLICIES = {
    "restart_best_only": "历史重启阶段最优路径",
    "global_best_only": "历史全局最优路径",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float(value):
    if value in (None, "", "None", "—"):
        return np.nan
    return float(value)


def _point_interval(values: np.ndarray, seed: int = 2026091803) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("点态区间需要至少两个完整实例")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def _aggregate_instances(
    rows: list[dict], dimensions: tuple[str, ...], metrics: tuple[str, ...]
) -> list[dict]:
    """先在实例内平均种子和表达式；不把它们当作独立样本。"""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[name] for name in dimensions)
        instance = int(row["instance"])
        for metric in metrics:
            value = row.get(metric)
            if value is not None and np.isfinite(float(value)):
                grouped[(key, instance)][metric].append(float(value))
    per_instance = []
    keys = sorted({key for key, _ in grouped})
    for key in keys:
        instances = sorted(instance for group_key, instance in grouped if group_key == key)
        for instance in instances:
            values = grouped[(key, instance)]
            per_instance.append(
                {
                    **dict(zip(dimensions, key, strict=True)),
                    "instance": instance,
                    **{
                        metric: float(np.mean(values[metric])) if values[metric] else np.nan
                        for metric in metrics
                    },
                }
            )
    return per_instance


def _summarize_instances(
    rows: list[dict], dimensions: tuple[str, ...], metrics: tuple[str, ...]
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in dimensions)].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        base = dict(zip(dimensions, key, strict=True))
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in group if np.isfinite(float(row[metric]))]
            )
            if len(values) < 2:
                continue
            mean, low, high = _point_interval(values, seed=2026091803 + len(result))
            result.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "point_low": low,
                    "point_high": high,
                    "instances": len(values),
                }
            )
    return result


def _edge_codes(tour: np.ndarray, n: int) -> np.ndarray:
    tour = np.asarray(tour, dtype=np.int64)
    return np.minimum(tour[:-1], tour[1:]) * n + np.maximum(tour[:-1], tour[1:])


def _directed_mask(codes: set[int], n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if codes:
        values = np.fromiter(codes, dtype=np.int64)
        u, v = values // n, values % n
        mask[u, v] = True
        mask[v, u] = True
    return mask


def _parent_probe_arrays(
    job: Path, iteration: int, owner: int, instances: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    index = read_json(job / "diagnostics/index.json")
    with np.load(job / "raw.npz", allow_pickle=False) as archive:
        representative = int(archive["behavior_alias"][owner])
    contexts = np.full((instances, 16, 10), -1, dtype=np.int32)
    visited = None
    found = np.zeros(instances, dtype=bool)
    for name, record in index["files"].items():
        meta = record["metadata"]
        if (
            meta.get("kind") != "permanent"
            or meta.get("phase") != "post_ls"
            or meta.get("iteration") != iteration
        ):
            continue
        flat = np.asarray(meta["flat_indices"], dtype=int)
        selected = np.flatnonzero(flat // instances == representative)
        if not len(selected):
            continue
        with np.load(job / "diagnostics" / name, allow_pickle=False) as archive:
            current_context = archive["audit_context"][selected]
            current_visited = archive["audit_visited"][selected]
        if visited is None:
            visited = np.zeros((instances, *current_visited.shape[1:]), dtype=current_visited.dtype)
        order = flat[selected] % instances
        contexts[order] = current_context
        visited[order] = current_visited
        found[order] = True
    if visited is None or not found.all():
        raise ValueError("同状态父快照缺少固定构造探针")
    return contexts, visited


def _same_state_tasks(source: Path) -> list[dict]:
    tasks = []
    for path in (source / "queue").glob("*.json"):
        task = read_json(path)["task"]
        if task.get("kind") != "explanation_fork":
            continue
        parent = read_json(source / "queue" / f"{task['parent']}.json")["task"]
        if parent["mechanism"]["terminal_statistics"] == "centered_fp32":
            tasks.append(task)
    if len(tasks) != 108:
        raise ValueError("中心化 FP32 同状态任务应完整包含 108 项")
    return tasks


def analyze_same_state(source: Path, target: Path) -> dict:
    """分解来源支持集变化、GP 源内重加权、概率传播和 2-opt 差异存活。"""
    tasks = _same_state_tasks(source)
    grouped = defaultdict(list)
    for task in tasks:
        grouped[(task["parent"], task["snapshot_iteration"], task["owner"])].append(task)
    source_rows, update_rows, probability_rows = [], [], []
    programs = {
        entry["seed"]: entry["program"] for entry in program_entries("mmas") if entry["seed"]
    }
    problem_cache = {}
    geometry_cache = {}
    for group_index, ((parent_id, iteration, owner), group) in enumerate(
        sorted(grouped.items()), 1
    ):
        parent_job = source / "jobs" / parent_id
        parent_task = read_json(source / "queue" / f"{parent_id}.json")["task"]
        problem = problem_cache.setdefault(
            parent_id, batch(parent_task["split"], parent_task["indices"], source)
        )
        if parent_id not in geometry_cache:
            distance = problem.distances.numpy()
            order = np.argsort(distance, axis=-1, kind="stable")
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.arange(problem.n)[None, None, :], axis=-1)
            geometry_cache[parent_id] = {
                "distances": distance,
                "coords": problem.coords.numpy(),
                "nearest": problem.nn_indices.numpy(),
                "ranks": ranks,
            }
        geometry = geometry_cache[parent_id]
        contexts, visited = _parent_probe_arrays(parent_job, iteration, owner)
        aco = None
        from .common import experiment

        aco, _ = experiment("mmas")
        n = problem.n
        code_matrix = np.minimum(np.arange(n)[:, None], np.arange(n)[None, :]) * n + np.maximum(
            np.arange(n)[:, None], np.arange(n)[None, :]
        )
        for task in group:
            job = source / "jobs" / task["id"]
            status = read_json(job / "status.json", {})
            if status.get("status") != "completed":
                raise ValueError(f"同状态任务未完成：{task['id']}")
            tr_program, _ = programs[task["champion"]]
            replicate = task["replicate"]
            state_label = "不使用 GP 产生的状态" if owner == 0 else "GP 表达式自身产生的状态"
            states = {}
            for policy in ("iteration_best", *SOURCE_POLICIES):
                for update in ("uniform_update", "learned_update"):
                    key = f"{policy}_{update}"
                    path = job / f"immediate-{key}.npz"
                    if file_hash(path) != status["files"][path.name]:
                        raise ValueError(f"同状态即时结果损坏：{path}")
                    with np.load(path, allow_pickle=False) as archive:
                        states[key] = {
                            name: archive[name].copy()
                            for name in (
                                "pheromone_workspace",
                                "audit_tau",
                                "audit_sources",
                                "deposit_workspace",
                                "trace",
                                "stagnation",
                            )
                        }
            for update in ("uniform_update", "learned_update"):
                current = states[f"iteration_best_{update}"]
                for policy, source_label in SOURCE_POLICIES.items():
                    history = states[f"{policy}_{update}"]
                    for instance in range(8):
                        candidate_mask = np.zeros((n, n), dtype=bool)
                        candidate_mask[
                            np.arange(n)[:, None],
                            geometry["nearest"][instance, :, : aco.candidate_size],
                        ] = True
                        np.fill_diagonal(candidate_mask, False)
                        current_codes = set(
                            map(int, _edge_codes(current["audit_sources"][instance, 0], n))
                        )
                        history_codes = set(
                            map(int, _edge_codes(history["audit_sources"][instance, 0], n))
                        )
                        categories = {
                            "两条来源路径共有边": current_codes & history_codes,
                            "仅历史来源路径包含的边": history_codes - current_codes,
                            "仅本轮最优路径包含的边": current_codes - history_codes,
                        }
                        budget = float(current["trace"][instance, 4])
                        for phase, phase_name in enumerate(PHASES):
                            tau_current = (
                                current["audit_tau"][instance, phase]
                                if phase < 3
                                else current["pheromone_workspace"][instance]
                            )
                            tau_history = (
                                history["audit_tau"][instance, phase]
                                if phase < 3
                                else history["pheromone_workspace"][instance]
                            )
                            delta = tau_history.astype(float) - tau_current.astype(float)
                            for category, codes in categories.items():
                                mask = _directed_mask(codes, n)
                                values = delta[mask]
                                source_rows.append(
                                    {
                                        "replicate": replicate,
                                        "champion": task["champion"],
                                        "instance": instance,
                                        "state_owner": state_label,
                                        "snapshot_iteration": iteration,
                                        "historical_source": source_label,
                                        "update_rule": "GP 源内重加权"
                                        if update == "learned_update"
                                        else "来源内均匀沉积",
                                        "phase": phase_name,
                                        "edge_category": category,
                                        "source_overlap": len(current_codes & history_codes) / n,
                                        "edge_count": int(mask.sum()),
                                        "signed_tau_change_over_two_budget": float(
                                            values.sum() / max(2 * budget, 1e-30)
                                        ),
                                        "absolute_tau_change_over_two_budget": float(
                                            np.abs(values).sum() / max(2 * budget, 1e-30)
                                        ),
                                        "mean_signed_tau_change": float(values.mean())
                                        if len(values)
                                        else 0.0,
                                        "mean_absolute_tau_change": float(np.abs(values).mean())
                                        if len(values)
                                        else 0.0,
                                    }
                                )
                            neither = candidate_mask & ~np.isin(
                                code_matrix,
                                np.fromiter(current_codes | history_codes, dtype=np.int64),
                            )
                            values = delta[neither]
                            source_rows.append(
                                {
                                    "replicate": replicate,
                                    "champion": task["champion"],
                                    "instance": instance,
                                    "state_owner": state_label,
                                    "snapshot_iteration": iteration,
                                    "historical_source": source_label,
                                    "update_rule": "GP 源内重加权"
                                    if update == "learned_update"
                                    else "来源内均匀沉积",
                                    "phase": phase_name,
                                    "edge_category": "其他候选弧",
                                    "source_overlap": len(current_codes & history_codes) / n,
                                    "edge_count": int(neither.sum()),
                                    "signed_tau_change_over_two_budget": float(
                                        values.sum() / max(2 * budget, 1e-30)
                                    ),
                                    "absolute_tau_change_over_two_budget": float(
                                        np.abs(values).sum() / max(2 * budget, 1e-30)
                                    ),
                                    "mean_signed_tau_change": float(values.mean())
                                    if len(values)
                                    else 0.0,
                                    "mean_absolute_tau_change": float(np.abs(values).mean())
                                    if len(values)
                                    else 0.0,
                                }
                            )
                        affected_tv, unaffected_tv = [], []
                        symmetric = current_codes ^ history_codes
                        for context, visit in zip(
                            contexts[instance], visited[instance], strict=True
                        ):
                            if context[0] < 0:
                                continue
                            results = []
                            for state in (current, history):
                                results.append(
                                    transition_distribution(
                                        state["pheromone_workspace"][instance],
                                        problem,
                                        instance,
                                        context,
                                        visit,
                                        tr_program,
                                        aco,
                                        iteration + 1,
                                        int(state["stagnation"][instance]),
                                        geometry,
                                    )
                                )
                            np.testing.assert_array_equal(results[0][0], results[1][0])
                            candidates = results[0][0]
                            city = int(context[3])
                            codes = np.minimum(city, candidates) * n + np.maximum(city, candidates)
                            value = float(0.5 * np.abs(results[1][1] - results[0][1]).sum())
                            (
                                affected_tv
                                if np.isin(codes, list(symmetric)).any()
                                else unaffected_tv
                            ).append(value)
                        probability_rows.append(
                            {
                                "replicate": replicate,
                                "champion": task["champion"],
                                "instance": instance,
                                "state_owner": state_label,
                                "snapshot_iteration": iteration,
                                "historical_source": source_label,
                                "update_rule": "GP 源内重加权"
                                if update == "learned_update"
                                else "来源内均匀沉积",
                                "affected_probe_count": len(affected_tv),
                                "unaffected_probe_count": len(unaffected_tv),
                                "affected_probability_tv": float(np.mean(affected_tv))
                                if affected_tv
                                else np.nan,
                                "unaffected_probability_tv": float(np.mean(unaffected_tv))
                                if unaffected_tv
                                else np.nan,
                            }
                        )
            for policy, label in {"iteration_best": "本轮最优路径", **SOURCE_POLICIES}.items():
                uniform = states[f"{policy}_uniform_update"]
                learned = states[f"{policy}_learned_update"]
                for instance in range(8):
                    source_codes = set(
                        map(int, _edge_codes(uniform["audit_sources"][instance, 0], n))
                    )
                    source_mask = _directed_mask(source_codes, n)
                    budget = float(uniform["trace"][instance, 4])
                    deposit_change = float(
                        np.abs(
                            learned["deposit_workspace"][instance, 0].astype(float)
                            - uniform["deposit_workspace"][instance, 0].astype(float)
                        ).sum()
                        / max(budget, 1e-30)
                    )
                    for phase, phase_name in enumerate(PHASES):
                        a = (
                            uniform["audit_tau"][instance, phase]
                            if phase < 3
                            else uniform["pheromone_workspace"][instance]
                        )
                        b = (
                            learned["audit_tau"][instance, phase]
                            if phase < 3
                            else learned["pheromone_workspace"][instance]
                        )
                        delta = b.astype(float) - a.astype(float)
                        off_diag = ~np.eye(n, dtype=bool)
                        update_rows.append(
                            {
                                "replicate": replicate,
                                "champion": task["champion"],
                                "instance": instance,
                                "state_owner": state_label,
                                "snapshot_iteration": iteration,
                                "source": label,
                                "phase": phase_name,
                                "deposit_l1_over_budget": deposit_change,
                                "source_edge_tau_l1_over_two_budget": float(
                                    np.abs(delta[source_mask]).sum() / max(2 * budget, 1e-30)
                                ),
                                "global_tau_relative_l1": float(
                                    np.abs(delta[off_diag]).sum()
                                    / max(np.abs(a[off_diag]).sum(), 1e-30)
                                ),
                            }
                        )
        print(f"同状态细分 {group_index}/{len(grouped)}", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "same_state_source_support_raw.csv", source_rows)
    write_csv(target / "same_state_probability_raw.csv", probability_rows)
    write_csv(target / "same_state_update_tree_raw.csv", update_rows)
    source_metrics = (
        "source_overlap",
        "signed_tau_change_over_two_budget",
        "absolute_tau_change_over_two_budget",
        "mean_signed_tau_change",
        "mean_absolute_tau_change",
    )
    source_dimensions = (
        "state_owner",
        "snapshot_iteration",
        "historical_source",
        "update_rule",
        "phase",
        "edge_category",
    )
    source_instance = _aggregate_instances(source_rows, source_dimensions, source_metrics)
    probability_metrics = ("affected_probability_tv", "unaffected_probability_tv")
    probability_dimensions = (
        "state_owner",
        "snapshot_iteration",
        "historical_source",
        "update_rule",
    )
    probability_instance = _aggregate_instances(
        probability_rows, probability_dimensions, probability_metrics
    )
    update_metrics = (
        "deposit_l1_over_budget",
        "source_edge_tau_l1_over_two_budget",
        "global_tau_relative_l1",
    )
    update_dimensions = ("state_owner", "snapshot_iteration", "source", "phase")
    update_instance = _aggregate_instances(update_rows, update_dimensions, update_metrics)
    write_csv(target / "same_state_source_support_per_instance.csv", source_instance)
    write_csv(target / "same_state_probability_per_instance.csv", probability_instance)
    write_csv(target / "same_state_update_tree_per_instance.csv", update_instance)
    summaries = {
        "source_support": _summarize_instances(source_instance, source_dimensions, source_metrics),
        "probability": _summarize_instances(
            probability_instance, probability_dimensions, probability_metrics
        ),
        "update_tree": _summarize_instances(update_instance, update_dimensions, update_metrics),
    }
    for name, rows in summaries.items():
        write_csv(target / f"same_state_{name}_summary.csv", rows)
    return summaries


def _candidate_metrics(tau: np.ndarray, nearest: np.ndarray, tau_min: float) -> dict:
    n, candidate_size = nearest.shape
    values = tau[np.arange(n)[:, None], nearest].astype(np.float64).ravel()
    total = values.sum()
    probability = values / max(total, 1e-300)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-300)))
    ordered = np.sort(values)
    count = len(ordered)
    gini = (
        2 * np.dot(np.arange(1, count + 1), ordered) / max(count * total, 1e-300)
        - (count + 1) / count
    )
    top_500 = min(500, count)
    top_2500 = min(2500, count)
    top = np.partition(values, count - top_2500)
    return {
        "candidate_entropy": float(entropy),
        "candidate_effective_arcs": float(np.exp(entropy)),
        "candidate_gini": float(gini),
        "top500_mass": float(top[-top_500:].sum() / max(total, 1e-300)),
        "top2500_mass": float(top[-top_2500:].sum() / max(total, 1e-300)),
        "floor_fraction": float(np.mean(values <= tau_min * (1 + 2e-6))),
    }


def _path_lift(tau: np.ndarray, tour: np.ndarray, candidate_mean: float) -> float:
    values = tau[tour[:-1], tour[1:]].astype(float)
    return float(values.mean() / max(candidate_mean, 1e-300))


def analyze_pheromone_trajectories(source: Path, target: Path) -> dict:
    """中心化 FP32 重型父任务；自然轨迹只作描述，不替代同状态干预。"""
    logical_rows, event_rows = [], []
    reference = np.load(source / "inputs/diagnosis_dev.npz", allow_pickle=False)["reference_tour"][
        :8
    ]
    parent_ids = [f"matched-state-mmas-centered_fp32-seed{replicate}" for replicate in range(3)]
    for replicate, parent_id in enumerate(parent_ids):
        job = source / "jobs" / parent_id
        status = read_json(job / "status.json", {})
        if status.get("status") != "completed":
            raise ValueError(f"重型父任务未完成：{parent_id}")
        index = read_json(job / "diagnostics/index.json")
        with np.load(job / "diagnostics/geometry.npz", allow_pickle=False) as archive:
            nearest = archive["nearest"].copy()
        records = sorted(
            (
                (record["metadata"]["iteration"], name, record)
                for name, record in index["files"].items()
                if record["metadata"].get("kind") == "sample"
            ),
            key=lambda row: (row[0], row[1]),
        )
        previous_top = {}
        for record_index, (iteration, name, record) in enumerate(records, 1):
            raw = (job / "diagnostics" / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"重型采样文件损坏：{name}")
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                arrays = {
                    key: archive[key]
                    for key in (
                        "tau_matrices",
                        "source_tours",
                        "source_valid",
                        "source_info",
                        "post_tours",
                        "pre_tours",
                        "colony_lengths",
                        "best_tours",
                        "trace",
                        "deposit",
                        "ph",
                    )
                }
            flat = np.asarray(record["metadata"]["flat_indices"], dtype=int)
            for row, task_index in enumerate(flat):
                program = int(task_index) // 8
                instance = int(task_index) % 8
                tau = arrays["tau_matrices"][row, 3]
                candidates = tau[np.arange(500)[:, None], nearest[instance]].astype(float)
                metrics = _candidate_metrics(
                    tau, nearest[instance], float(arrays["trace"][row, 11])
                )
                current = arrays["post_tours"][row, int(np.argmin(arrays["colony_lengths"][row]))]
                source_tour = arrays["source_tours"][row, 0]
                global_tour = arrays["best_tours"][row]
                candidate_mean = float(candidates.mean())
                top_indices = set(np.argpartition(candidates.ravel(), -500)[-500:].tolist())
                persistence = np.nan
                if task_index in previous_top:
                    before = previous_top[task_index]
                    persistence = len(before & top_indices) / len(before | top_indices)
                previous_top[task_index] = top_indices
                row_values = {
                    "replicate": replicate,
                    "program": program,
                    "instance": instance,
                    "behavior": "不使用 GP" if program == 0 else "三个固定 GP 表达式的均值",
                    "iteration": iteration,
                    **metrics,
                    "source_edge_lift": _path_lift(tau, source_tour, candidate_mean),
                    "current_best_edge_lift": _path_lift(tau, current, candidate_mean),
                    "global_best_edge_lift": _path_lift(tau, global_tour, candidate_mean),
                    "reference_edge_lift": _path_lift(tau, reference[instance], candidate_mean),
                    "top500_persistence": persistence,
                    "source_age": float(arrays["trace"][row, 14]),
                    "source_kind": int(arrays["trace"][row, 0]),
                    "post_unique_tours": len(
                        np.unique(canonical_edges(arrays["post_tours"][row], 500), axis=0)
                    ),
                    "pre_mean_length": float(
                        arrays["pre_tours"][row].shape[0] and arrays["trace"][row, 18]
                    ),
                    "post_mean_length": float(arrays["trace"][row, 19]),
                }
                logical_rows.append(row_values)
                source_codes = set(map(int, _edge_codes(source_tour, 500)))
                current_codes = set(map(int, _edge_codes(current, 500)))
                if int(arrays["trace"][row, 0]) > 0 and source_codes != current_codes:
                    phase_lifts = [
                        _path_lift(
                            arrays["tau_matrices"][row, phase],
                            source_tour,
                            float(
                                arrays["tau_matrices"][row, phase][
                                    np.arange(500)[:, None], nearest[instance]
                                ].mean()
                            ),
                        )
                        for phase in range(4)
                    ]
                    event_rows.append(
                        {
                            "replicate": replicate,
                            "program": program,
                            "instance": instance,
                            "behavior": row_values["behavior"],
                            "iteration": iteration,
                            "source_kind": int(arrays["trace"][row, 0]),
                            "source_current_overlap": len(source_codes & current_codes) / 500,
                            **{f"source_lift_{phase}": phase_lifts[phase] for phase in range(4)},
                            "post_unique_tours": row_values["post_unique_tours"],
                        }
                    )
            if record_index % 100 == 0:
                print(f"信息素轨迹 {parent_id}: {record_index}/{len(records)}", flush=True)
    metrics = (
        "candidate_entropy",
        "candidate_effective_arcs",
        "candidate_gini",
        "top500_mass",
        "top2500_mass",
        "floor_fraction",
        "source_edge_lift",
        "current_best_edge_lift",
        "global_best_edge_lift",
        "reference_edge_lift",
        "top500_persistence",
        "source_age",
        "post_unique_tours",
    )
    dimensions = ("behavior", "iteration")
    per_instance = _aggregate_instances(logical_rows, dimensions, metrics)
    summary = _summarize_instances(per_instance, dimensions, metrics)
    write_csv(target / "pheromone_trajectory_raw.csv", logical_rows)
    write_csv(target / "pheromone_trajectory_per_instance.csv", per_instance)
    write_csv(target / "pheromone_trajectory_summary.csv", summary)
    write_csv(target / "historical_source_events_raw.csv", event_rows)
    event_metrics = (
        "source_current_overlap",
        "source_lift_0",
        "source_lift_1",
        "source_lift_2",
        "source_lift_3",
        "post_unique_tours",
    )
    event_instance = _aggregate_instances(event_rows, ("behavior", "source_kind"), event_metrics)
    event_summary = _summarize_instances(event_instance, ("behavior", "source_kind"), event_metrics)
    write_csv(target / "historical_source_events_per_instance.csv", event_instance)
    write_csv(target / "historical_source_events_summary.csv", event_summary)
    return {"trajectory": summary, "events": event_summary}


def analyze_quality_and_local_search(source: Path, target: Path) -> dict:
    publish_matched_states(source)
    matched = source / "reports/matched_states"
    if read_json(matched / "summary.json", {}).get("status") != "complete":
        raise ValueError("同状态汇总未完成")
    quality = _read_csv(matched / "quality_per_instance.csv")
    quality = [row for row in quality if row["数值实现"] == "中心化 FP32 补偿求和"]
    quality_metrics = (
        "原生来源下 GP 增益（百分点）",
        "仅本轮最优来源下 GP 增益（百分点）",
        "移除历史来源后 GP 增益变化（百分点）",
        "仅构造决策树的增益（百分点）",
        "仅信息素更新树的增益（百分点）",
    )
    normalized = []
    for row in quality:
        normalized.append(
            {
                "instance": int(row["实例编号"]),
                "state_owner": row["状态来源"],
                "snapshot_iteration": int(row["快照轮次"]),
                "continuation_iterations": int(row["后续轮数"]),
                "replicate": int(row["求解种子编号"]),
                "expression": row["表达式"],
                **{metric: _float(row[metric]) for metric in quality_metrics},
            }
        )
    dimensions = ("state_owner", "snapshot_iteration", "continuation_iterations")
    quality_instance = _aggregate_instances(normalized, dimensions, quality_metrics)
    quality_summary = _summarize_instances(quality_instance, dimensions, quality_metrics)
    across_dimensions = ("state_owner", "continuation_iterations")
    quality_across_instance = _aggregate_instances(
        normalized, across_dimensions, quality_metrics
    )
    quality_across_summary = _summarize_instances(
        quality_across_instance, across_dimensions, quality_metrics
    )
    write_csv(target / "continuation_quality_per_instance.csv", quality_instance)
    write_csv(target / "continuation_quality_summary.csv", quality_summary)
    write_csv(
        target / "continuation_quality_across_snapshots_per_instance.csv",
        quality_across_instance,
    )
    write_csv(
        target / "continuation_quality_across_snapshots_summary.csv",
        quality_across_summary,
    )
    structure = _read_csv(matched / "local_search_structure.csv")
    structure = [row for row in structure if row["数值实现"] == "中心化 FP32 补偿求和"]
    structure_rows = [
        {
            "instance": int(row["实例编号"]),
            "state_owner": row["状态来源"],
            "snapshot_iteration": int(row["快照轮次"]),
            "continuation_iterations": int(row["后续轮数"]),
            "replicate": int(row["求解种子编号"]),
            "expression": row["表达式"],
            "pre_symmetric_difference": _float(row["局部搜索前边集对称差"]),
            "post_symmetric_difference": _float(row["局部搜索后边集对称差"]),
            "defined_survival_ratio": _float(row["已定义比率均值"]),
            "undefined_ants": int(row["分母为零的蚂蚁数"]),
        }
        for row in structure
    ]
    structure_metrics = (
        "pre_symmetric_difference",
        "post_symmetric_difference",
        "defined_survival_ratio",
        "undefined_ants",
    )
    structure_instance = _aggregate_instances(structure_rows, dimensions, structure_metrics)
    structure_summary = _summarize_instances(structure_instance, dimensions, structure_metrics)
    structure_across_instance = _aggregate_instances(
        structure_rows, across_dimensions, structure_metrics
    )
    structure_across_summary = _summarize_instances(
        structure_across_instance, across_dimensions, structure_metrics
    )
    write_csv(target / "two_opt_survival_per_instance.csv", structure_instance)
    write_csv(target / "two_opt_survival_summary.csv", structure_summary)
    write_csv(
        target / "two_opt_survival_across_snapshots_per_instance.csv",
        structure_across_instance,
    )
    write_csv(
        target / "two_opt_survival_across_snapshots_summary.csv",
        structure_across_summary,
    )
    return {
        "quality": quality_summary,
        "quality_across_snapshots": quality_across_summary,
        "two_opt": structure_summary,
        "two_opt_across_snapshots": structure_across_summary,
    }


def analyze_source_performance(source: Path, target: Path) -> list[dict]:
    path = source / "reports/development/results.csv"
    rows = _read_csv(path)
    normalized = []
    for row in rows:
        normalized.append(
            {
                "execution_configuration": row["执行配置"],
                "expression_origin": row["表达式来源"],
                "baseline_gap_percent": _float(row["不使用 GP 的 gap（%）"]),
                "gp_gap_percent": _float(row["使用 GP 的 gap（%）"]),
                "gp_benefit_pp": _float(row["GP 带来的改进（百分点）"]),
                "wins": int(row["胜"]),
                "ties": int(row["平"]),
                "losses": int(row["负"]),
            }
        )
    write_csv(target / "source_policy_performance.csv", normalized)
    return normalized


def analyze_expressions(source: Path, target: Path) -> list[dict]:
    from .common import models

    update = _read_csv(target / "same_state_update_tree_raw.csv")
    rows = []
    for model in models("mmas"):
        tr, ph = model["program"]
        relevant = [
            row
            for row in update
            if int(row["champion"]) == model["seed"]
            if row["source"] == "历史全局最优路径"
            and row["phase"] == "沉积后"
        ]
        by_instance = defaultdict(list)
        for row in relevant:
            by_instance[int(row["instance"])].append(
                float(row["source_edge_tau_l1_over_two_budget"])
            )
        rows.append(
            {
                "seed": model["seed"],
                "transition_tree_active": not tr.is_exact_zero,
                "pheromone_tree_active": not ph.is_exact_zero,
                "selected_for_animation": model["seed"] == CHAMPION,
                "selection_passed_noninferiority": False,
                "final_deployed": False,
                "transition_expression": model["expression"]
                .splitlines()[0]
                .removeprefix("transition: "),
                "pheromone_expression": model["expression"]
                .splitlines()[1]
                .removeprefix("pheromone: "),
                "aggregate_global_history_source_edge_change": float(
                    np.mean([np.mean(values) for values in by_instance.values()])
                ),
            }
        )
    write_csv(target / "expression_behavior.csv", rows)
    return rows


def analyze_replay(source: Path, output: Path, target: Path) -> dict:
    """重放若已完成则提取动画帧指标；未完成不阻止静态控制实验报告。"""
    summary = read_json(output / "replay/summary.json", {})
    if summary.get("status") != "completed":
        result = {"status": "pending", "reason": "八个定向重放尚未全部完成"}
        atomic_json(target / "replay_analysis.json", result)
        return result
    rows = []
    selection = read_json(output / "selection.json")
    with np.load(source / "inputs/diagnosis_dev.npz", allow_pickle=False) as archive:
        reference_tour = archive["reference_tour"][selection["instance"]]
    for regime in REGIMES:
        for behavior in BEHAVIORS:
            directory = output / "replay" / regime / behavior / "samples"
            index = read_json(directory / "index.json")
            previous_top = None
            for name, record in sorted(
                index["files"].items(), key=lambda item: item[1]["iteration"]
            ):
                path = directory / name
                if file_hash(path) != record["sha256"]:
                    raise ValueError(f"重放帧损坏：{path}")
                with np.load(path, allow_pickle=False) as archive:
                    tau = archive["tau"]
                    trace = archive["trace"]
                    source_tour = archive["source_tours"][0]
                    current = archive["post_tours"][int(np.argmin(archive["colony_lengths"]))]
                    best = archive["best_tour"]
                    tau_min = float(archive["tau_min"][0])
                nearest_path = output / "replay/geometry.npz"
                if not nearest_path.exists():
                    problem = batch("diagnosis_dev", [selection["instance"]], source)
                    atomic_npz(
                        nearest_path,
                        coords=problem.coords.numpy()[0],
                        nearest=problem.nn_indices.numpy()[0],
                    )
                with np.load(nearest_path, allow_pickle=False) as geometry:
                    nearest = geometry["nearest"]
                metrics = _candidate_metrics(tau[3], nearest, tau_min)
                candidate_mean = float(tau[3][np.arange(500)[:, None], nearest].mean())
                candidate_values = tau[3][np.arange(500)[:, None], nearest]
                top = set(np.argpartition(candidate_values.ravel(), -500)[-500:].tolist())
                persistence = (
                    np.nan
                    if previous_top is None
                    else len(previous_top & top) / len(previous_top | top)
                )
                previous_top = top
                rows.append(
                    {
                        "regime": REGIMES[regime][0],
                        "regime_key": regime,
                        "behavior": BEHAVIORS[behavior],
                        "behavior_key": behavior,
                        "iteration": record["iteration"],
                        **metrics,
                        "source_edge_lift": _path_lift(tau[3], source_tour, candidate_mean),
                        "current_best_edge_lift": _path_lift(tau[3], current, candidate_mean),
                        "global_best_edge_lift": _path_lift(tau[3], best, candidate_mean),
                        "reference_edge_lift": _path_lift(tau[3], reference_tour, candidate_mean),
                        "top500_persistence": persistence,
                        "source_kind": int(trace[0]),
                        "source_age": float(trace[14]),
                        "source_current_overlap": len(
                            set(_edge_codes(source_tour, 500)) & set(_edge_codes(current, 500))
                        )
                        / 500,
                    }
                )
    write_csv(target / "replay_trajectory.csv", rows)
    result = {
        "status": "complete",
        "rows": len(rows),
        "selection": selection,
        "storage_bytes": summary["bytes"],
    }
    atomic_json(target / "replay_analysis.json", result)
    return result


def analyze(source: Path, output: Path, report: Path) -> Path:
    report.mkdir(parents=True, exist_ok=True)
    source_performance = analyze_source_performance(source, report)
    same_state = analyze_same_state(source, report)
    trajectories = analyze_pheromone_trajectories(source, report)
    continuation = analyze_quality_and_local_search(source, report)
    expressions = analyze_expressions(source, report)
    replay = analyze_replay(source, output, report)
    manifest = {
        "status": "analysis_complete",
        "generated_at": now(),
        "source": str(source),
        "output": str(output),
        "report": str(report),
        "development_tasks": 35,
        "matched_state_parents": 6,
        "matched_interventions": 216,
        "confirmation_tasks_completed": 0,
        "confirmation_tasks_pending": 280,
        "source_performance_rows": len(source_performance),
        "same_state_summary_rows": {key: len(value) for key, value in same_state.items()},
        "trajectory_summary_rows": {key: len(value) for key, value in trajectories.items()},
        "continuation_summary_rows": {key: len(value) for key, value in continuation.items()},
        "expression_rows": len(expressions),
        "replay": replay,
        "inference_unit": "实例；先平均同实例的 ACO 种子和固定表达式",
        "evidence_hierarchy": [
            "最终质量来源对照",
            "同状态即时与续跑干预",
            "自然轨迹描述",
            "单实例说明性重放",
        ],
        "analysis_sha256": file_hash(__file__),
    }
    atomic_json(report / "analysis_manifest.json", manifest)
    return report / "analysis_manifest.json"


def verify(source: Path, output: Path, report: Path) -> Path:
    errors = []
    counts = defaultdict(int)
    for path in (source / "queue").glob("*.json"):
        task = read_json(path)["task"]
        status = read_json(source / "jobs" / task["id"] / "status.json", {})
        if status.get("status") == "completed":
            counts[task["stage"]] += 1
    expected = {"development": 35, "matched_states": 6, "matched_interventions": 216}
    for stage, count in expected.items():
        if counts[stage] != count:
            errors.append(f"{stage} 完成数 {counts[stage]}，预期 {count}")
    if counts["confirmation"]:
        errors.append("本报告不得混入未完成的独立确认子集")
    selection = read_json(output / "selection.json", {})
    if selection.get("forbidden_selection_fields") != [
        "GP gap",
        "GP 增益",
        "处理效应",
        "冠军间差异",
    ]:
        errors.append("代表实例选择审计字段缺失")
    replay_summary = read_json(output / "replay/summary.json", {})
    if replay_summary.get("status") == "completed":
        if replay_summary.get("bytes", STORAGE_CAP_BYTES + 1) > STORAGE_CAP_BYTES:
            errors.append("重放存储超过硬上限")
        for regime in REGIMES:
            for behavior in BEHAVIORS:
                target = output / "replay" / regime / behavior
                status = read_json(target / "status.json", {})
                if status.get("status") != "completed":
                    errors.append(f"重放未完成：{regime}/{behavior}")
                    continue
                index = read_json(target / "samples/index.json")
                observed = sorted(row["iteration"] for row in index["files"].values())
                if observed != list(SAMPLE_ITERATIONS):
                    errors.append(f"重放帧缺失：{regime}/{behavior}")
                with np.load(target / "result.npz", allow_pickle=False) as archive:
                    if np.any(np.diff(archive["anytime"]) > 0):
                        errors.append(f"anytime 非单调：{regime}/{behavior}")
    required = (
        "source_policy_performance.csv",
        "same_state_source_support_per_instance.csv",
        "same_state_probability_per_instance.csv",
        "same_state_update_tree_per_instance.csv",
        "pheromone_trajectory_per_instance.csv",
        "historical_source_events_per_instance.csv",
        "continuation_quality_per_instance.csv",
        "continuation_quality_across_snapshots_per_instance.csv",
        "two_opt_survival_per_instance.csv",
        "two_opt_survival_across_snapshots_per_instance.csv",
        "expression_behavior.csv",
        "analysis_manifest.json",
    )
    for name in required:
        if not (report / name).exists():
            errors.append(f"缺少分析产物：{name}")
    html = report / "mechanism_dashboard.html"
    if html.exists():
        text = html.read_text(errors="replace")
        if "https://" in text or "http://" in text:
            errors.append("交互 HTML 含外部网络依赖")
    report_manifest = read_json(report / "report_manifest.json", {})
    if report_manifest.get("status") == "complete":
        for name, expected_hash in report_manifest.get("files", {}).items():
            if not (report / name).exists() or file_hash(report / name) != expected_hash:
                errors.append(f"报告文件哈希不匹配：{name}")
    result = {
        "status": "passed" if not errors else "failed",
        "checked_at": now(),
        "completed_counts": dict(counts),
        "errors": errors,
        "confirmation_scope": "280 项独立确认仍因共享盘空间暂停；未选择性使用其子集",
        "analysis_identity": digest(
            {name: file_hash(report / name) for name in required if (report / name).exists()}
        ),
    }
    atomic_json(report / "verification.json", result)
    if errors:
        raise ValueError("；".join(errors))
    return report / "verification.json"
