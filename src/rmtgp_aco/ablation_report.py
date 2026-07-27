"""多方法消融 study 的配对统计、训练曲线与中文学术报告。"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
from scipy import stats

from .ablation import (
    CORE_METHODS,
    MECHANISM_METHODS,
    AblationStudySpec,
    evaluated_methods_for_partition,
    export_ablation_contract,
    method_run_path,
)
from .artifacts import git_state
from .evaluation import EvaluationRecord, load_champion, read_records
from .study import load_study_spec

CURVE_FIELDS = (
    "train_candidate_gap_percent",
    "train_baseline_gap_percent",
    "train_delta_pp",
    "validation_candidate_gap_percent",
    "validation_baseline_gap_percent",
    "validation_delta_pp",
    "generation_wall_time_sec",
    "validation_monitor_wall_time_sec",
)

ALL_EVALUATED_METHODS = (*CORE_METHODS, *MECHANISM_METHODS)
BASELINE_METHOD = "baseline-aco"

METHOD_LABELS = {
    "legacy": "Legacy-GP",
    "matched-replace": "Matched-Replace",
    "tr-rgp": "TR-RGP",
    "ph-rgp": "PH-RGP",
    "rmtgp-core-f0": "RMTGP-Core-F0",
    "rmtgp-core-f1": "RMTGP-Core-F1",
    "rmtgp-full-f0": "RMTGP-Full-F0",
    "rmtgp-full-f1": "RMTGP-Full-F1",
    "rmtgp-full-f1-drop-transition": "Full-F1 drop-TR",
    "rmtgp-full-f1-drop-pheromone": "Full-F1 drop-PH",
    "rmtgp-full-f1-shuffle-r1": "Full-F1 shuffle-r1",
    "rmtgp-full-f1-shuffle-r2": "Full-F1 shuffle-r2",
    BASELINE_METHOD: "原始 ACO",
}

PARTITION_LABELS = {
    "tsp50_uniform": "TSP50-U",
    "tsp100_uniform": "TSP100-U",
    "tsp500_uniform": "TSP500-U",
    "tsp1000_uniform": "TSP1000-U",
    "tsp500_cluster": "TSP500-C",
    "tsp500_gaussian": "TSP500-G",
    "tsplib_le500": "TSPLIB≤500",
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """原子写入同构字典长表。"""

    if not rows:
        raise ValueError(f"{path}: 不允许写空 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    fieldnames = list(rows[0])
    for row in rows:
        if list(row) != fieldnames:
            raise ValueError(f"{path}: CSV rows 字段顺序不一致")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_records(study: AblationStudySpec) -> list[EvaluationRecord]:
    """按预注册作用域合并结果，忽略历史全 OOD 消融 artifact。"""

    source_partitions = set(load_study_spec(study.reuse.config).partitions)
    all_records: list[EvaluationRecord] = []
    for variant in study.variants:
        expected_roots = set(variant.seeds)
        for partition in study.partitions:
            context: list[EvaluationRecord] = []
            if partition in study.ablation_partitions:
                paths = [
                    study.output_root
                    / "test"
                    / variant.name
                    / partition
                    / group
                    / "records.csv"
                    for group in ("residual", "replacement")
                ]
            elif partition in source_partitions:
                paths = [
                    study.reuse.root
                    / "test"
                    / variant.name
                    / partition
                    / "records.csv"
                ]
            else:
                paths = [
                    study.output_root
                    / "test"
                    / variant.name
                    / partition
                    / "final"
                    / "records.csv"
                ]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f"作用域内测试结果缺失: {path}")
                context.extend(read_records([path]))
            if (
                partition in study.ablation_partitions
                and partition in source_partitions
            ):
                reused = (
                    study.reuse.root
                    / "test"
                    / variant.name
                    / partition
                    / "records.csv"
                )
                context.extend(read_records([reused]))

            methods = {record.method for record in context}
            expected_methods = evaluated_methods_for_partition(study, partition)
            if methods != set(expected_methods):
                raise ValueError(
                    f"{variant.name}/{partition}: methods={sorted(methods)}，"
                    f"预期 {sorted(expected_methods)}"
                )
            if {record.variant for record in context} != {variant.name}:
                raise ValueError(f"{variant.name}/{partition}: variant provenance 错误")
            if {record.partition for record in context} != {partition}:
                raise ValueError(f"{variant.name}/{partition}: partition provenance 错误")
            for method in expected_methods:
                roots = {
                    record.gp_root_seed
                    for record in context
                    if record.method == method
                }
                if roots != expected_roots:
                    raise ValueError(
                        f"{variant.name}/{partition}/{method}: GP roots={sorted(roots)}"
                    )
            keys = {
                (
                    record.method,
                    record.gp_root_seed,
                    record.instance_id,
                    record.seed,
                )
                for record in context
            }
            if len(keys) != len(context):
                raise ValueError(f"{variant.name}/{partition}: 合并记录存在重复键")

            # baseline 在全部方法之间应是相同的共享 paired realization。
            baseline_values: dict[tuple[str, int], tuple[float, float]] = {}
            for record in context:
                key = (record.instance_id, record.seed)
                value = (record.baseline_length, record.baseline_gap_percent)
                previous = baseline_values.setdefault(key, value)
                if not np.allclose(previous, value, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"{variant.name}/{partition}: baseline cache 不一致 key={key}"
                    )
            all_records.extend(context)
    return all_records


def _candidate_blocks(
    records: list[EvaluationRecord],
) -> dict[tuple[int, str], list[EvaluationRecord]]:
    blocks: dict[tuple[int, str], list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        blocks[(record.gp_root_seed, record.instance_id)].append(record)
    return blocks


def _tail_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, ceil(fraction * values.size))
    return float(np.sort(values)[-count:].mean())


def _quality_row(
    *,
    variant: str,
    partition: str,
    method: str,
    records: list[EvaluationRecord],
) -> dict[str, Any]:
    """以 GP-run×instance 为描述 block，先平均三个 ACO seeds。"""

    blocks = _candidate_blocks(records)
    gaps = np.asarray(
        [fmean(item.gap_percent for item in values) for values in blocks.values()],
        dtype=np.float64,
    )
    deltas = np.asarray(
        [fmean(item.delta_pp for item in values) for values in blocks.values()],
        dtype=np.float64,
    )
    auc = np.asarray(
        [fmean(item.anytime_gap_auc for item in values) for values in blocks.values()],
        dtype=np.float64,
    )
    best_iteration = np.asarray(
        [fmean(item.best_iteration for item in values) for values in blocks.values()],
        dtype=np.float64,
    )
    scales = sorted({item.scale for item in records})
    tolerance = 1e-12
    return {
        "variant": variant,
        "partition": partition,
        "distribution": records[0].distribution,
        "scale_min": min(scales),
        "scale_max": max(scales),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "scope": "posthoc" if method in MECHANISM_METHODS else "selected-candidate",
        "gp_runs": len({key[0] for key in blocks}),
        "instances": len({key[1] for key in blocks}),
        "aco_seeds_per_instance": len(records) // len(blocks),
        "blocks": len(blocks),
        "mean_gap_percent": float(gaps.mean()),
        "median_gap_percent": float(np.median(gaps)),
        "standard_deviation": float(gaps.std(ddof=1) if gaps.size > 1 else 0.0),
        "q1_gap_percent": float(np.quantile(gaps, 0.25)),
        "q3_gap_percent": float(np.quantile(gaps, 0.75)),
        "mean_delta_pp": float(deltas.mean()),
        "median_delta_pp": float(np.median(deltas)),
        "win_rate": float(np.mean(deltas < -tolerance)),
        "tie_rate": float(np.mean(np.abs(deltas) <= tolerance)),
        "loss_rate": float(np.mean(deltas > tolerance)),
        "worse_than_baseline_rate": float(np.mean(deltas > tolerance)),
        "cvar_worst_10_percent": _tail_mean(gaps),
        "reference_hit_rate": float(np.mean(gaps <= 1e-9)),
        "mean_anytime_gap_auc": float(auc.mean()),
        "mean_best_iteration": float(best_iteration.mean()),
    }


def _baseline_quality_row(
    *,
    variant: str,
    partition: str,
    records: list[EvaluationRecord],
) -> dict[str, Any]:
    """baseline 不随 GP run 复制，统计单位仅为 instance。"""

    by_instance: dict[str, dict[int, EvaluationRecord]] = defaultdict(dict)
    # 任选一个方法与 GP root；baseline 字段已在合并阶段验证完全一致。
    anchor_method = "rmtgp-full-f1"
    anchor_root = min(
        record.gp_root_seed
        for record in records
        if record.method == anchor_method
    )
    for record in records:
        if record.method == anchor_method and record.gp_root_seed == anchor_root:
            by_instance[record.instance_id][record.seed] = record
    gaps = np.asarray(
        [
            fmean(item.baseline_gap_percent for item in seed_map.values())
            for seed_map in by_instance.values()
        ],
        dtype=np.float64,
    )
    auc = np.asarray(
        [
            fmean(item.baseline_anytime_gap_auc for item in seed_map.values())
            for seed_map in by_instance.values()
        ],
        dtype=np.float64,
    )
    best_iteration = np.asarray(
        [
            fmean(item.baseline_best_iteration for item in seed_map.values())
            for seed_map in by_instance.values()
        ],
        dtype=np.float64,
    )
    scales = sorted({item.scale for item in records})
    return {
        "variant": variant,
        "partition": partition,
        "distribution": records[0].distribution,
        "scale_min": min(scales),
        "scale_max": max(scales),
        "method": BASELINE_METHOD,
        "method_label": METHOD_LABELS[BASELINE_METHOD],
        "scope": "baseline",
        "gp_runs": 0,
        "instances": len(by_instance),
        "aco_seeds_per_instance": len(next(iter(by_instance.values()))),
        "blocks": len(by_instance),
        "mean_gap_percent": float(gaps.mean()),
        "median_gap_percent": float(np.median(gaps)),
        "standard_deviation": float(gaps.std(ddof=1) if gaps.size > 1 else 0.0),
        "q1_gap_percent": float(np.quantile(gaps, 0.25)),
        "q3_gap_percent": float(np.quantile(gaps, 0.75)),
        "mean_delta_pp": 0.0,
        "median_delta_pp": 0.0,
        "win_rate": 0.0,
        "tie_rate": 1.0,
        "loss_rate": 0.0,
        "worse_than_baseline_rate": 0.0,
        "cvar_worst_10_percent": _tail_mean(gaps),
        "reference_hit_rate": float(np.mean(gaps <= 1e-9)),
        "mean_anytime_gap_auc": float(auc.mean()),
        "mean_best_iteration": float(best_iteration.mean()),
    }


def _quality_summaries(
    study: AblationStudySpec,
    records: list[EvaluationRecord],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in study.variants:
        for partition in study.partitions:
            context = [
                record
                for record in records
                if record.variant == variant.name and record.partition == partition
            ]
            for method in evaluated_methods_for_partition(study, partition):
                selected = [record for record in context if record.method == method]
                summaries.append(
                    _quality_row(
                        variant=variant.name,
                        partition=partition,
                        method=method,
                        records=selected,
                    )
                )
            summaries.append(
                _baseline_quality_row(
                    variant=variant.name,
                    partition=partition,
                    records=context,
                )
            )
    return summaries


def _tsplib_band(scale: int) -> str:
    if scale <= 100:
        return "n<=100"
    if scale <= 200:
        return "101<=n<=200"
    return "201<=n<=500"


def _tsplib_band_summaries(
    study: AblationStudySpec,
    records: list[EvaluationRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in study.variants:
        context = [
            record
            for record in records
            if record.variant == variant.name
            and record.partition == "tsplib_le500"
        ]
        for band in ("n<=100", "101<=n<=200", "201<=n<=500"):
            band_records = [
                record for record in context if _tsplib_band(record.scale) == band
            ]
            if not band_records:
                continue
            for method in evaluated_methods_for_partition(
                study,
                "tsplib_le500",
            ):
                row = _quality_row(
                    variant=variant.name,
                    partition=f"tsplib:{band}",
                    method=method,
                    records=[
                        record for record in band_records if record.method == method
                    ],
                )
                row["size_band"] = band
                rows.append(row)
            baseline = _baseline_quality_row(
                variant=variant.name,
                partition=f"tsplib:{band}",
                records=band_records,
            )
            baseline["size_band"] = band
            rows.append(baseline)
    # 让 size_band 位于 CSV 的固定尾列，同时保持每行字段同构。
    return [
        {**{key: value for key, value in row.items() if key != "size_band"},
         "size_band": row["size_band"]}
        for row in rows
    ]


def _read_training_curves(
    study: AblationStudySpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for variant in study.variants:
        for method in CORE_METHODS:
            for root_seed in variant.seeds:
                run = method_run_path(
                    study,
                    method,
                    variant.name,
                    root_seed,
                )
                curve_path = run / "training_validation_curve.csv"
                with curve_path.open("r", encoding="utf-8", newline="") as handle:
                    raw_rows = list(csv.DictReader(handle))
                generations = {int(row["generation"]) for row in raw_rows}
                if generations != set(range(1, 51)):
                    raise ValueError(f"{curve_path}: 必须恰好包含 generation 1..50")
                run_rows: list[dict[str, Any]] = []
                for raw in raw_rows:
                    row: dict[str, Any] = {
                        "variant": variant.name,
                        "method": method,
                        "gp_root_seed": root_seed,
                        "generation": int(raw["generation"]),
                        "scale": int(raw["scale"]),
                    }
                    row.update({field: float(raw[field]) for field in CURVE_FIELDS})
                    curves.append(row)
                    run_rows.append(row)
                run_rows.sort(key=lambda row: row["generation"])
                final = run_rows[-1]
                decision = json.loads(
                    (run / "deployment_decision.json").read_text(encoding="utf-8")
                )
                champion = load_champion(run / "selected_candidate.pkl")
                runs.append(
                    {
                        "variant": variant.name,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "artifact_source": (
                            "reused-main-study"
                            if method == "rmtgp-full-f1"
                            else "new-ablation-run"
                        ),
                        "gp_root_seed": root_seed,
                        "generations": len(run_rows),
                        "mean_generation_wall_time_sec": fmean(
                            row["generation_wall_time_sec"] for row in run_rows
                        ),
                        "total_generation_wall_time_sec": sum(
                            row["generation_wall_time_sec"] for row in run_rows
                        ),
                        "total_validation_monitor_wall_time_sec": sum(
                            row["validation_monitor_wall_time_sec"]
                            for row in run_rows
                        ),
                        "final_train_gap_percent": final[
                            "train_candidate_gap_percent"
                        ],
                        "final_train_baseline_gap_percent": final[
                            "train_baseline_gap_percent"
                        ],
                        "final_train_delta_pp": final["train_delta_pp"],
                        "final_validation_gap_percent": final[
                            "validation_candidate_gap_percent"
                        ],
                        "final_validation_baseline_gap_percent": final[
                            "validation_baseline_gap_percent"
                        ],
                        "final_validation_delta_pp": final["validation_delta_pp"],
                        "transition_nodes": champion.transition_nodes,
                        "pheromone_nodes": champion.pheromone_nodes,
                        "total_nodes": champion.total_nodes,
                        "transition_expression": str(champion.transition_tree),
                        "pheromone_expression": str(champion.pheromone_tree),
                        "selected_candidate_hash": decision[
                            "selected_candidate_hash"
                        ],
                        "selection_passed_noninferiority": bool(
                            decision["selection_passed_noninferiority"]
                        ),
                        "cpu_fp64_audit_passed": bool(
                            decision["cpu_fp64_audit_passed"]
                        ),
                        "final_passed_noninferiority": bool(
                            decision["final_passed_noninferiority"]
                        ),
                        "deployed_method": str(decision["deployed_method"]),
                    }
                )
    return curves, runs


def _aggregate_curves(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in curves:
        grouped[
            (row["variant"], row["method"], row["generation"], row["scale"])
        ].append(row)
    aggregate: list[dict[str, Any]] = []
    for (variant, method, generation, scale), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "variant": variant,
            "method": method,
            "generation": generation,
            "scale": scale,
            "gp_runs": len(values),
        }
        for field in CURVE_FIELDS:
            data = np.asarray([item[field] for item in values], dtype=np.float64)
            row[f"{field}_mean"] = float(data.mean())
            row[f"{field}_median"] = float(np.median(data))
            row[f"{field}_min"] = float(data.min())
            row[f"{field}_max"] = float(data.max())
        aggregate.append(row)
    return aggregate


def _plot_curves(
    study: AblationStudySpec,
    aggregate: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        method: plt.get_cmap("tab10")(index % 10)
        for index, method in enumerate(CORE_METHODS)
    }
    artifacts: list[str] = []
    for variant in study.variants:
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(10.5, 8.0),
            sharex=True,
            constrained_layout=True,
        )
        for method in CORE_METHODS:
            selected = sorted(
                (
                    row
                    for row in aggregate
                    if row["variant"] == variant.name and row["method"] == method
                ),
                key=lambda row: row["generation"],
            )
            x = np.asarray([row["generation"] for row in selected])
            validation = np.asarray(
                [
                    row["validation_delta_pp_median"]
                    for row in selected
                ]
            )
            validation_min = np.asarray(
                [row["validation_delta_pp_min"] for row in selected]
            )
            validation_max = np.asarray(
                [row["validation_delta_pp_max"] for row in selected]
            )
            generation_time = np.asarray(
                [
                    row["generation_wall_time_sec_median"]
                    for row in selected
                ]
            )
            axes[0].plot(
                x,
                validation,
                color=colors[method],
                linewidth=1.7,
                label=METHOD_LABELS[method],
            )
            axes[0].fill_between(
                x,
                validation_min,
                validation_max,
                color=colors[method],
                alpha=0.05,
                linewidth=0,
            )
            axes[1].plot(
                x,
                generation_time,
                color=colors[method],
                linewidth=1.5,
                label=METHOD_LABELS[method],
            )
        axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        axes[0].set_ylabel("Validation Δ (pp)")
        axes[0].set_title(f"{variant.name.upper()}：三 GP seeds 中位数与 min–max")
        axes[0].grid(alpha=0.20)
        axes[0].legend(ncol=2, fontsize=8)
        axes[1].set_xlabel("GP generation")
        axes[1].set_ylabel("Generation wall time (s)")
        axes[1].grid(alpha=0.20)
        stem = f"ablation_training_curve_{variant.name}"
        for suffix in ("png", "svg"):
            path = output / f"{stem}.{suffix}"
            figure.savefig(path, dpi=180 if suffix == "png" else None)
            artifacts.append(path.name)
        plt.close(figure)
    return artifacts


def _aligned_quality_cube(
    records: list[EvaluationRecord],
    *,
    methods: tuple[str, ...],
    roots: tuple[int, ...],
) -> tuple[np.ndarray, tuple[str, ...], int]:
    """生成 [method, GP run, instance, ACO seed] 的严格配对 gap cube。"""

    candidate: dict[
        str,
        dict[int, dict[str, dict[int, float]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    baseline: dict[int, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        if record.method in methods:
            target = candidate[record.method][record.gp_root_seed][record.instance_id]
            if record.seed in target:
                raise ValueError("candidate paired cube 存在重复 seed")
            target[record.seed] = record.gap_percent
        # baseline 可从任意核心方法取一次；按 root 复制以保留 run 层级。
        if record.method == "rmtgp-full-f1":
            target_baseline = baseline[record.gp_root_seed][record.instance_id]
            previous = target_baseline.setdefault(
                record.seed,
                record.baseline_gap_percent,
            )
            if not np.isclose(
                previous,
                record.baseline_gap_percent,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("paired cube 的 baseline 不一致")

    common_instances: set[str] | None = None
    for root in roots:
        run_instances = set(baseline[root])
        for method in methods:
            run_instances &= set(candidate[method][root])
        common_instances = (
            run_instances
            if common_instances is None
            else common_instances & run_instances
        )
    instances = tuple(sorted(common_instances or ()))
    if not instances:
        raise ValueError("方法之间没有共同 run×instance blocks")

    seed_count: int | None = None
    method_order = (*methods, BASELINE_METHOD)
    values = np.empty(
        (len(method_order), len(roots), len(instances), 3),
        dtype=np.float64,
    )
    for run_index, root in enumerate(roots):
        for instance_index, instance in enumerate(instances):
            common_seeds = set(baseline[root][instance])
            for method in methods:
                common_seeds &= set(candidate[method][root][instance])
            ordered_seeds = tuple(sorted(common_seeds))
            if seed_count is None:
                seed_count = len(ordered_seeds)
                if seed_count != 3:
                    raise ValueError("正式消融每个 block 必须恰好有三个 ACO seeds")
            if len(ordered_seeds) != seed_count:
                raise ValueError("ACO seed 数量不平衡")
            for method_index, method in enumerate(methods):
                values[method_index, run_index, instance_index] = [
                    candidate[method][root][instance][seed]
                    for seed in ordered_seeds
                ]
            values[-1, run_index, instance_index] = [
                baseline[root][instance][seed] for seed in ordered_seeds
            ]
    return values, method_order, len(instances)


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    return (positive - negative) / float(ranks.sum())


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, original in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[int(original)])
        running = max(running, candidate)
        adjusted[int(original)] = running
    return adjusted.tolist()


def _joint_hierarchical_bootstrap(
    contrast_cube: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """联合重采样多个线性 contrast，返回 estimate 与 [K,2] 区间。

    ACO seed 层只有三个值。先精确枚举 3^3=27 个有序 bootstrap
    均值，再为每个 sampled run×instance 抽一个类别，可把临时数据量降为
    直接四维高级索引的三分之一。
    """

    if contrast_cube.ndim != 4:
        raise ValueError("contrast cube 必须为 [K,R,I,S]")
    contrasts, runs, instances, seeds = contrast_cube.shape
    if runs < 2 or instances < 1 or seeds < 2:
        raise ValueError("层次 bootstrap 的 run/instance/seed 层不完整")
    estimate = contrast_cube.mean(axis=(1, 2, 3))
    seed_draws = np.asarray(
        list(itertools.product(range(seeds), repeat=seeds)),
        dtype=np.int16,
    )
    # [K,R,I,C]，C 为全部等概率有序 seed bootstrap 样本。
    seed_means = contrast_cube[..., seed_draws].mean(axis=-1)
    categories = seed_draws.shape[0]
    rng = np.random.default_rng(seed)
    estimates = np.empty((contrasts, replicates), dtype=np.float64)
    # 每个 chunk 的 gathered float64 控制在约 32 MiB。
    chunk_size = max(
        1,
        min(
            replicates,
            4_000_000 // max(contrasts * runs * instances, 1),
        ),
    )
    contrast_index = np.arange(contrasts)[:, None, None, None]
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        count = stop - start
        sampled_runs = rng.integers(0, runs, size=(count, runs))
        sampled_instances = rng.integers(
            0,
            instances,
            size=(count, runs, instances),
        )
        sampled_categories = rng.integers(
            0,
            categories,
            size=(count, runs, instances),
        )
        sampled = seed_means[
            contrast_index,
            sampled_runs[None, :, :, None],
            sampled_instances[None, :, :, :],
            sampled_categories[None, :, :, :],
        ]
        estimates[:, start:stop] = sampled.mean(axis=(2, 3))
    intervals = np.quantile(estimates, [0.025, 0.975], axis=1).T
    return estimate, intervals


def _contrast_definitions() -> tuple[
    tuple[tuple[str, str, str, str], ...],
    tuple[tuple[str, str, str, str], ...],
]:
    """返回 (family, name, first, second)；first−second<0 表示 first 更好。"""

    primary = (
        ("rq1-components", "TR-RGP − ACO", "tr-rgp", BASELINE_METHOD),
        ("rq1-components", "PH-RGP − ACO", "ph-rgp", BASELINE_METHOD),
        (
            "rq1-components",
            "RMTGP-Full-F1 − ACO",
            "rmtgp-full-f1",
            BASELINE_METHOD,
        ),
        (
            "rq2-dual-tree",
            "RMTGP-Full-F1 − TR-RGP",
            "rmtgp-full-f1",
            "tr-rgp",
        ),
        (
            "rq2-dual-tree",
            "RMTGP-Full-F1 − PH-RGP",
            "rmtgp-full-f1",
            "ph-rgp",
        ),
        (
            "rq3-residual",
            "TR-RGP − Matched-Replace",
            "tr-rgp",
            "matched-replace",
        ),
        (
            "prior-study",
            "RMTGP-Full-F1 − Legacy-GP",
            "rmtgp-full-f1",
            "legacy",
        ),
    )
    mechanism = (
        (
            "posthoc-mechanism",
            "Full-F1 − drop-TR",
            "rmtgp-full-f1",
            "rmtgp-full-f1-drop-transition",
        ),
        (
            "posthoc-mechanism",
            "Full-F1 − drop-PH",
            "rmtgp-full-f1",
            "rmtgp-full-f1-drop-pheromone",
        ),
        (
            "posthoc-coadaptation",
            "Full-F1 − shuffle-r1",
            "rmtgp-full-f1",
            "rmtgp-full-f1-shuffle-r1",
        ),
        (
            "posthoc-coadaptation",
            "Full-F1 − shuffle-r2",
            "rmtgp-full-f1",
            "rmtgp-full-f1-shuffle-r2",
        ),
    )
    return primary, mechanism


def _factorial_coefficients(method_order: tuple[str, ...]) -> list[tuple[str, np.ndarray]]:
    index = {method: position for position, method in enumerate(method_order)}

    def coefficient(**weights: float) -> np.ndarray:
        result = np.zeros(len(method_order), dtype=np.float64)
        for method, weight in weights.items():
            result[index[method]] = weight
        return result

    return [
        (
            "terminal_full_minus_core",
            coefficient(
                **{
                    "rmtgp-core-f0": -0.5,
                    "rmtgp-core-f1": -0.5,
                    "rmtgp-full-f0": 0.5,
                    "rmtgp-full-f1": 0.5,
                }
            ),
        ),
        (
            "function_f1_minus_f0",
            coefficient(
                **{
                    "rmtgp-core-f0": -0.5,
                    "rmtgp-core-f1": 0.5,
                    "rmtgp-full-f0": -0.5,
                    "rmtgp-full-f1": 0.5,
                }
            ),
        ),
        (
            "terminal_function_interaction",
            coefficient(
                **{
                    "rmtgp-core-f0": 1.0,
                    "rmtgp-core-f1": -1.0,
                    "rmtgp-full-f0": -1.0,
                    "rmtgp-full-f1": 1.0,
                }
            ),
        ),
    ]


def _paired_statistics(
    study: AblationStudySpec,
    records: list[EvaluationRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    primary_defs, mechanism_defs = _contrast_definitions()
    primary_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    factorial_rows: list[dict[str, Any]] = []

    for variant in study.variants:
        for partition in study.ablation_partitions:
            context = [
                record
                for record in records
                if record.variant == variant.name and record.partition == partition
            ]
            cube, method_order, instances = _aligned_quality_cube(
                context,
                methods=ALL_EVALUATED_METHODS,
                roots=variant.seeds,
            )
            method_index = {
                method: position for position, method in enumerate(method_order)
            }
            definitions = (*primary_defs, *mechanism_defs)
            coefficients: list[np.ndarray] = []
            for _, _, first, second in definitions:
                vector = np.zeros(len(method_order), dtype=np.float64)
                vector[method_index[first]] = 1.0
                vector[method_index[second]] = -1.0
                coefficients.append(vector)
            factorial = _factorial_coefficients(method_order)
            coefficients.extend(vector for _, vector in factorial)
            matrix = np.stack(coefficients, axis=0)
            contrast_cube = np.einsum("km,mris->kris", matrix, cube)
            context_seed = int.from_bytes(
                sha256(
                    f"{study.test_root_seed}:{variant.name}:{partition}".encode()
                ).digest()[:8],
                byteorder="little",
                signed=False,
            )
            estimates, intervals = _joint_hierarchical_bootstrap(
                contrast_cube,
                replicates=study.bootstrap_replicates,
                seed=context_seed,
            )

            for index, (family, name, first, second) in enumerate(definitions):
                block_differences = contrast_cube[index].mean(axis=2).reshape(-1)
                if np.all(block_differences == 0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = stats.wilcoxon(
                        block_differences,
                        zero_method="pratt",
                        alternative="two-sided",
                        method="auto",
                    )
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                tolerance = 1e-12
                row = {
                    "variant": variant.name,
                    "partition": partition,
                    "family": family,
                    "contrast": name,
                    "first_method": first,
                    "second_method": second,
                    "estimand": "first_gap_minus_second_gap_pp",
                    "runs": len(variant.seeds),
                    "instances": instances,
                    "aco_seeds": cube.shape[-1],
                    "run_instance_blocks": int(block_differences.size),
                    "estimate_pp": float(estimates[index]),
                    "lower_95": float(intervals[index, 0]),
                    "upper_95": float(intervals[index, 1]),
                    "wins_first": int(np.sum(block_differences < -tolerance)),
                    "ties": int(np.sum(np.abs(block_differences) <= tolerance)),
                    "losses_first": int(np.sum(block_differences > tolerance)),
                    "wilcoxon_statistic": statistic,
                    "raw_p_value": p_value,
                    "holm_p_value": float("nan"),
                    "rank_biserial": _rank_biserial(block_differences),
                    "bootstrap_replicates": study.bootstrap_replicates,
                    "directionally_first_better": bool(estimates[index] < 0.0),
                    "ci_supports_first_better": bool(intervals[index, 1] < 0.0),
                }
                if index < len(primary_defs):
                    primary_rows.append(row)
                else:
                    mechanism_rows.append(row)

            offset = len(definitions)
            for factorial_index, (name, _) in enumerate(factorial):
                index = offset + factorial_index
                factorial_rows.append(
                    {
                        "variant": variant.name,
                        "partition": partition,
                        "contrast": name,
                        "estimand": "gap_contrast_pp",
                        "runs": len(variant.seeds),
                        "instances": instances,
                        "aco_seeds": cube.shape[-1],
                        "run_instance_blocks": len(variant.seeds) * instances,
                        "estimate_pp": float(estimates[index]),
                        "lower_95": float(intervals[index, 0]),
                        "upper_95": float(intervals[index, 1]),
                        "bootstrap_replicates": study.bootstrap_replicates,
                        "directionally_reduces_gap": bool(estimates[index] < 0.0),
                        "ci_supports_reduction": bool(intervals[index, 1] < 0.0),
                    }
                )

    # Holm family：每个 ACO variant×RQ family 内跨 partition/contrast 校正。
    for rows in (primary_rows, mechanism_rows):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["variant"], row["family"])].append(row)
        for family_rows in grouped.values():
            adjusted = _holm([row["raw_p_value"] for row in family_rows])
            for row, corrected in zip(family_rows, adjusted, strict=True):
                row["holm_p_value"] = corrected
    return primary_rows, mechanism_rows, factorial_rows


def _efficiency_rows(study: AblationStudySpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in study.variants:
        path = study.output_root / "efficiency" / f"{variant.name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"{path}: efficiency artifact 未完成")
        for summary in payload["summaries"]:
            rows.append(
                {
                    "variant": str(summary["variant"]),
                    "method": str(summary["method"]),
                    "partition": str(summary["partition"]),
                    "scale": int(summary["scale"]),
                    "batch_size": int(payload["batch_sizes"][summary["partition"]]),
                    "observations": int(summary["observations"]),
                    "median_wall_time_sec": float(
                        summary["median_wall_time_sec"]
                    ),
                    "median_baseline_wall_time_sec": float(
                        summary["median_baseline_wall_time_sec"]
                    ),
                    "median_overhead_percent": float(
                        summary["median_overhead_percent"]
                    ),
                    "median_tours_per_second": float(
                        summary["median_tours_per_second"]
                    ),
                    "timing_scope": str(payload["timing_scope"]),
                }
            )
    return rows


def _training_aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["variant"], row["method"])].append(row)
    results: list[dict[str, Any]] = []
    for (variant, method), values in sorted(grouped.items()):
        results.append(
            {
                "variant": variant,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "gp_runs": len(values),
                "mean_generation_wall_time_sec": fmean(
                    row["mean_generation_wall_time_sec"] for row in values
                ),
                "total_gpu_hours_generation": sum(
                    row["total_generation_wall_time_sec"] for row in values
                )
                / 3600.0,
                "mean_final_train_gap_percent": fmean(
                    row["final_train_gap_percent"] for row in values
                ),
                "mean_final_train_delta_pp": fmean(
                    row["final_train_delta_pp"] for row in values
                ),
                "mean_final_validation_gap_percent": fmean(
                    row["final_validation_gap_percent"] for row in values
                ),
                "mean_final_validation_delta_pp": fmean(
                    row["final_validation_delta_pp"] for row in values
                ),
                "median_total_nodes": median(row["total_nodes"] for row in values),
                "gates_passed": sum(
                    bool(row["final_passed_noninferiority"]) for row in values
                ),
            }
        )
    return results


def _rq_audit(
    primary: list[dict[str, Any]],
    mechanism: list[dict[str, Any]],
    factorial: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """只生成证据计数，不把 3-seed pilot 自动升级为确认性结论。"""

    definitions = {
        "RQ1-TR-vs-ACO": (
            primary,
            lambda row: row["contrast"] == "TR-RGP − ACO",
        ),
        "RQ1-PH-vs-ACO": (
            primary,
            lambda row: row["contrast"] == "PH-RGP − ACO",
        ),
        "RQ1-Full-vs-ACO": (
            primary,
            lambda row: row["contrast"] == "RMTGP-Full-F1 − ACO",
        ),
        "RQ2-Full-vs-single": (
            primary,
            lambda row: row["family"] == "rq2-dual-tree",
        ),
        "RQ3-residual-vs-replacement": (
            primary,
            lambda row: row["family"] == "rq3-residual",
        ),
        "Factorial-terminal": (
            factorial,
            lambda row: row["contrast"] == "terminal_full_minus_core",
        ),
        "Factorial-function": (
            factorial,
            lambda row: row["contrast"] == "function_f1_minus_f0",
        ),
        "Posthoc-coadaptation": (
            mechanism,
            lambda row: row["family"] == "posthoc-coadaptation",
        ),
    }
    audit: list[dict[str, Any]] = []
    for question, (rows, predicate) in definitions.items():
        selected = [row for row in rows if predicate(row)]
        audit.append(
            {
                "question": question,
                "comparisons": len(selected),
                "negative_estimates": sum(
                    float(row["estimate_pp"]) < 0.0 for row in selected
                ),
                "ci_upper_below_zero": sum(
                    float(row["upper_95"]) < 0.0 for row in selected
                ),
                "positive_estimates": sum(
                    float(row["estimate_pp"]) > 0.0 for row in selected
                ),
                "ci_lower_above_zero": sum(
                    float(row["lower_95"]) > 0.0 for row in selected
                ),
            }
        )
    return audit


def _format_ci(row: dict[str, Any]) -> str:
    return f"{row['estimate_pp']:+.4f} [{row['lower_95']:+.4f}, {row['upper_95']:+.4f}]"


def _render_report(
    *,
    study: AblationStudySpec,
    training: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    mechanism: list[dict[str, Any]],
    factorial: list[dict[str, Any]],
    efficiency: list[dict[str, Any]],
    rq_audit: list[dict[str, Any]],
    plots: list[str],
) -> str:
    lines = [
        f"# {study.study_id}：Multi-Tree GP–ACO 消融实验报告",
        "",
        "## 实验合同与解释边界",
        "",
        (
            "本实验在纯 TSP100 上分别为 AS、ACS、MMAS 进化规则。每个"
            " ACO×方法使用 3 个独立 GP root seeds；GP population=100、"
            "50 generations；每代 32 个训练实例；每次 ACO simulation 为"
            " 32 ants×500 iterations。RMTGP-Full-F1 的 9 个已锁定 run 从"
            "主 study 逐文件哈希复用，另外训练 63 个 run。每个 run 仅由"
            " validation 选择一个 `selected_candidate`；三个 GP runs 全部"
            "保留，禁止再挑选其中表现最好的一次。"
        ),
        "",
        (
            "RMTGP-Full-F1 的最终测试包含 TSP50/100/500/1000 uniform、"
            "TSP500 cluster、TSP500 Gaussian 与 TSPLIB n≤500；核心方法、"
            "residual/replacement、terminal/function 与 post-hoc 机制"
            "消融只在预注册的 TSP100-U 和 TSP500-U 上统计。历史目录中"
            "其他分区的全消融结果不进入本报告。"
        ),
        "",
        (
            "所有质量指标均为相对数据集中 reference tour 的 gap%。"
            "contrast 定义为 first gap−second gap，故负值表示 first 更好。"
            "Wilcoxon 先在 ACO seeds 内平均，以 GP-run×instance 为 block；"
            "Holm 在每个 ACO variant×RQ family 内校正。95% CI 来自 10,000"
            " 次 GP run→instance→ACO seed 层次 bootstrap。"
        ),
        "",
        (
            "这是 3-seed pilot。方向、区间和 p 值用于筛查机制与估计方差，"
            "不能替代冻结配置后的 30-run confirmatory experiment，也不能"
            "把最优 GP seed 当作方法性能，或把核心分区消融外推为全部 OOD"
            "分布上的普遍因果优势。"
        ),
        "",
        "## 方法与可识别问题",
        "",
        "| 方法 | Transition | Pheromone | 主要用途 |",
        "|---|---|---|---|",
        "| Legacy-GP | replacement，上一篇表示 | 无 | 上一研究 baseline |",
        "| Matched-Replace | replacement，Full/F1 | 无 | 控制表示方式 |",
        "| TR-RGP | residual，Full/F1 | 无 | transition 单树 |",
        "| PH-RGP | 无 | residual，Full/F1 | pheromone 单树 |",
        "| Core/Full × F0/F1 | residual | residual | terminal/function factorial |",
        "| RMTGP-Full-F1 | residual | residual | 主方法 |",
        "",
        (
            "Residual 的直接 estimand 是 TR-RGP−Matched-Replace：二者都是"
            "单 transition tree，使用同一 Full/F1 primitives、相同节点总预算、"
            "相同 schedule 与 baseline，仅 integration 不同。双树增益要求"
            " Full-F1 同时优于 TR-RGP 与 PH-RGP；这些 estimands 仅定义于"
            " TSP100-U 与 TSP500-U，不能只比较 Full-F1 与 ACO。"
        ),
        "",
        "## 训练与 validation",
        "",
        (
            "| ACO | 方法 | GP runs | 每代均时(s) | 最终 train gap% | "
            "最终 val gap% | val Δ(pp) | gate | 节点中位数 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training:
        lines.append(
            f"| {row['variant'].upper()} | {row['method_label']} | "
            f"{row['gp_runs']} | {row['mean_generation_wall_time_sec']:.3f} | "
            f"{row['mean_final_train_gap_percent']:.4f} | "
            f"{row['mean_final_validation_gap_percent']:.4f} | "
            f"{row['mean_final_validation_delta_pp']:+.4f} | "
            f"{row['gates_passed']}/{row['gp_runs']} | "
            f"{row['median_total_nodes']:.1f} |"
        )
    lines.extend(["", "三种 ACO 的 validation Δ 与单代时间曲线：", ""])
    for variant in study.variants:
        lines.append(
            f"![{variant.name} ablation curves]"
            f"(ablation_training_curve_{variant.name}.png)"
        )

    lines.extend(
        [
            "",
            "## 主方法相对原始 ACO 的锁定测试",
            "",
            "| ACO | Test | ACO gap% | Full-F1 gap% | Δ(pp) | W/T/L | CVaR10% | Anytime AUC |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    quality_map = {
        (row["variant"], row["partition"], row["method"]): row for row in quality
    }
    for variant in study.variants:
        for partition in study.partitions:
            baseline = quality_map[(variant.name, partition, BASELINE_METHOD)]
            full = quality_map[(variant.name, partition, "rmtgp-full-f1")]
            wtl = (
                f"{full['win_rate']:.1%}/"
                f"{full['tie_rate']:.1%}/"
                f"{full['loss_rate']:.1%}"
            )
            lines.append(
                f"| {variant.name.upper()} | {PARTITION_LABELS[partition]} | "
                f"{baseline['mean_gap_percent']:.4f} | "
                f"{full['mean_gap_percent']:.4f} | "
                f"{full['mean_delta_pp']:+.4f} | {wtl} | "
                f"{full['cvar_worst_10_percent']:.4f} | "
                f"{full['mean_anytime_gap_auc']:.4f} |"
            )

    def append_contrast_table(
        title: str,
        description: str,
        rows: list[dict[str, Any]],
        predicate: Any,
    ) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                description,
                "",
                "| ACO | Test | Contrast (pp, 95% CI) | Holm p | Rank-biserial | first W/T/L |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            if not predicate(row):
                continue
            lines.append(
                f"| {row['variant'].upper()} | "
                f"{PARTITION_LABELS[row['partition']]} | "
                f"{row['contrast']}: {_format_ci(row)} | "
                f"{row['holm_p_value']:.4g} | "
                f"{row['rank_biserial']:+.4f} | "
                f"{row['wins_first']}/{row['ties']}/{row['losses_first']} |"
            )

    append_contrast_table(
        "RQ1：单 residual 与主方法是否改善 ACO",
        "负值表示相应 learned rule 的 reference gap 低于同随机流的原始 ACO。",
        primary,
        lambda row: row["family"] == "rq1-components",
    )
    append_contrast_table(
        "RQ2：双树是否优于单树",
        (
            "只有 Full-F1−TR 与 Full-F1−PH 两个 contrast 均支持负效应，"
            "才构成该上下文中双树增量价值的完整证据。"
        ),
        primary,
        lambda row: row["family"] == "rq2-dual-tree",
    )
    append_contrast_table(
        "RQ3：Residual 是否优于完整替换",
        (
            "主 estimand 为 TR-RGP−Matched-Replace；Legacy-GP 另作为上一篇"
            "表示的外部 baseline，不用于隔离 residual 本身。"
        ),
        primary,
        lambda row: row["family"] in {"rq3-residual", "prior-study"},
    )

    lines.extend(
        [
            "",
            "## Terminal/function set 的 2×2 factorial",
            "",
            (
                "`terminal_full_minus_core<0` 表示增加 context terminals 降低"
                " gap；`function_f1_minus_f0<0` 表示加入 MIN/MAX/ABS 降低 gap；"
                "interaction 衡量两者是否非加性。"
            ),
            "",
            "| ACO | Test | Contrast | Estimate (pp, 95% CI) |",
            "|---|---|---|---:|",
        ]
    )
    for row in factorial:
        lines.append(
            f"| {row['variant'].upper()} | "
            f"{PARTITION_LABELS[row['partition']]} | {row['contrast']} | "
            f"{_format_ci(row)} |"
        )

    append_contrast_table(
        "Post-hoc：组件删除与树配对协同",
        (
            "drop-tree 不重新训练，用于测量已进化 Full-F1 champion 内组件的"
            "必要性；shuffle 保留两棵树的边际分布、打破同一 run 的配对，"
            "用于探索 coadaptation。它们不能替代独立训练的 TR/PH 对照。"
        ),
        mechanism,
        lambda row: True,
    )

    lines.extend(
        [
            "",
            "## 孤立推理效率",
            "",
            (
                "质量测试把多个 programs 拼成一个 CUDA task matrix，campaign"
                " 时间无法无偏分摊到单个方法；因此下表只采用 warm program、"
                "固定 batch、逐 champion 孤立运行的中位数。"
            ),
            "",
            "| ACO | Test | 方法 | Batch | 中位时间(s) | 相对 ACO 开销 | tours/s |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in efficiency:
        if row["method"] not in {
            "legacy",
            "matched-replace",
            "tr-rgp",
            "ph-rgp",
            "rmtgp-full-f1",
        }:
            continue
        lines.append(
            f"| {row['variant'].upper()} | "
            f"{PARTITION_LABELS[row['partition']]} | "
            f"{METHOD_LABELS[row['method']]} | {row['batch_size']} | "
            f"{row['median_wall_time_sec']:.4f} | "
            f"{row['median_overhead_percent']:+.2f}% | "
            f"{row['median_tours_per_second']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 证据审计（不自动生成“证明成立”的结论）",
            "",
            "| 问题 | comparisons | 负向估计 | CI 全负 | 正向估计 | CI 全正 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rq_audit:
        lines.append(
            f"| {row['question']} | {row['comparisons']} | "
            f"{row['negative_estimates']} | {row['ci_upper_below_zero']} | "
            f"{row['positive_estimates']} | {row['ci_lower_above_zero']} |"
        )
    lines.extend(
        [
            "",
            (
                "建议表述规则：若某 contrast 的均值为负、95% CI 上界低于 0，"
                "且在三个 GP runs 与两个预注册核心分区方向一致，可表述为"
                "“本 pilot 提供一致支持”；否则只能表述为混合证据或未观察"
                "到稳定改善。"
                "最终论文的显著性主张应等待 30-run confirmatory experiment。"
            ),
            "",
            "## 可复现 artifact",
            "",
            "- `training_runs.csv`：72 个 selected champions、表达式、节点和 gate",
            "- `training_summary.csv`：variant×method 训练汇总",
            "- `training_validation_curves_all.csv` 与 `..._aggregate.csv`",
            "- `quality_summary.csv`：主方法全分区与核心分区消融汇总",
            "- `tsplib_size_band_summary.csv`：主方法的 TSPLIB 预定义规模带",
            "- `primary_contrasts.csv`：RQ1/RQ2/RQ3 与上一篇 baseline",
            "- `factorial_contrasts.csv`：Core/Full × F0/F1",
            "- `mechanism_contrasts.csv`：drop-tree 与 shuffled pairing",
            "- `efficiency_summary.csv`：孤立 warm timing",
            "- `ablation_summary.json`：完整机器可读结果与 provenance",
        ]
    )
    lines.extend(f"- `{name}`" for name in plots)
    lines.append("")
    return "\n".join(lines)


def generate_ablation_report(study: AblationStudySpec) -> Path:
    """验证全部 artifact 后生成论文表格、统计、曲线与中文报告。"""

    output = study.output_root / "report"
    output.mkdir(parents=True, exist_ok=True)
    records = _load_records(study)
    quality = _quality_summaries(study, records)
    tsplib_bands = _tsplib_band_summaries(study, records)
    curves, training_runs = _read_training_curves(study)
    curve_aggregate = _aggregate_curves(curves)
    plots = _plot_curves(study, curve_aggregate, output)
    training = _training_aggregate(training_runs)
    primary, mechanism, factorial = _paired_statistics(study, records)
    efficiency = _efficiency_rows(study)
    rq_audit = _rq_audit(primary, mechanism, factorial)

    _write_dict_csv(output / "training_runs.csv", training_runs)
    _write_dict_csv(output / "training_summary.csv", training)
    _write_dict_csv(output / "training_validation_curves_all.csv", curves)
    _write_dict_csv(
        output / "training_validation_curves_aggregate.csv",
        curve_aggregate,
    )
    _write_dict_csv(output / "quality_summary.csv", quality)
    _write_dict_csv(output / "tsplib_size_band_summary.csv", tsplib_bands)
    _write_dict_csv(output / "primary_contrasts.csv", primary)
    _write_dict_csv(output / "factorial_contrasts.csv", factorial)
    _write_dict_csv(output / "mechanism_contrasts.csv", mechanism)
    _write_dict_csv(output / "efficiency_summary.csv", efficiency)
    _write_dict_csv(output / "rq_evidence_audit.csv", rq_audit)

    summary = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "study": export_ablation_contract(study),
        "git": git_state(Path.cwd()),
        "interpretation": {
            "phase": "three-seed-pilot",
            "confirmatory": False,
            "fitness_target": "reference_gap_percent",
            "contrast_sign": "first_minus_second; negative_is_first_better",
            "pairing_unit": "gp_root_seed-instance_id-aco_seed",
            "wilcoxon_block": "gp_run_x_instance_after_aco_seed_mean",
            "holm_family": "aco_variant_x_research_question_family",
            "bootstrap_hierarchy": ["gp_run", "instance", "aco_seed"],
            "bootstrap_replicates": study.bootstrap_replicates,
            "champion_selection": "one_validation_selected_candidate_per_gp_run",
            "best_gp_seed_selection": False,
            "final_test_partitions": list(study.partitions),
            "primary_ablation_partitions": list(study.ablation_partitions),
            "legacy_full_ood_ablation_artifacts": "excluded_from_inference",
            "tsp1000_role": "locked-test-only-supplementary-extrapolation",
            "quality_timing": "excluded_batched_campaign_allocation",
            "efficiency_timing": "isolated-warm-program",
        },
        "artifact_counts": {
            "training_runs": len(training_runs),
            "evaluation_records": len(records),
            "quality_summaries": len(quality),
            "primary_contrasts": len(primary),
            "factorial_contrasts": len(factorial),
            "mechanism_contrasts": len(mechanism),
            "efficiency_rows": len(efficiency),
        },
        "training_summary": training,
        "quality_summary": quality,
        "primary_contrasts": primary,
        "factorial_contrasts": factorial,
        "mechanism_contrasts": mechanism,
        "efficiency_summary": efficiency,
        "rq_evidence_audit": rq_audit,
        "plots": plots,
    }
    _atomic_text(
        output / "ablation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
    )
    target = output / "ablation_report.md"
    _atomic_text(
        target,
        _render_report(
            study=study,
            training=training,
            quality=quality,
            primary=primary,
            mechanism=mechanism,
            factorial=factorial,
            efficiency=efficiency,
            rq_audit=rq_audit,
            plots=plots,
        ),
    )
    return target
