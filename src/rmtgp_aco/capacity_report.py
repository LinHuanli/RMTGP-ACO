"""31/62 节点结构 × 容量敏感性实验的配对统计与中文报告。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
from scipy import stats

from .ablation_report import (
    CURVE_FIELDS,
    _atomic_text,
    _holm,
    _joint_hierarchical_bootstrap,
    _rank_biserial,
    _write_dict_csv,
)
from .artifacts import git_state
from .capacity import (
    CAPACITY_METHODS,
    SOURCE_CAPACITY,
    TARGET_CAPACITY,
    CapacityStudySpec,
    capacity_method,
    capacity_run_path,
    export_capacity_contract,
    source_ready,
    source_run_path,
)
from .evaluation import EvaluationRecord, load_champion, read_records
from .study import load_study_spec

BASELINE_METHOD = "baseline-aco"
CAPACITY_EVALUATED_METHODS: tuple[str, ...] = tuple(
    capacity_method(method, nodes)
    for method in CAPACITY_METHODS
    for nodes in (SOURCE_CAPACITY, TARGET_CAPACITY)
)

METHOD_LABELS = {
    capacity_method("tr-rgp", 31): "TR-RGP (B=31)",
    capacity_method("tr-rgp", 62): "TR-RGP (B=62)",
    capacity_method("ph-rgp", 31): "PH-RGP (B=31)",
    capacity_method("ph-rgp", 62): "PH-RGP (B=62)",
    capacity_method("rmtgp-full-f1", 31): "双树 Full-F1 (B=31)",
    capacity_method("rmtgp-full-f1", 62): "双树 Full-F1 (B=62)",
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


def _source_records_for_method(
    study: CapacityStudySpec,
    *,
    variant: str,
    partition: str,
    method: str,
) -> list[EvaluationRecord]:
    """读取源消融中的一个 31 节点方法，兼容 Full-F1 uniform 复用路径。"""

    main_partitions = set(load_study_spec(study.source.study.reuse.config).partitions)
    if method == "rmtgp-full-f1" and partition in main_partitions:
        path = study.source.study.reuse.root / "test" / variant / partition / "records.csv"
    else:
        path = study.source.root / "test" / variant / partition / "residual" / "records.csv"
    if not path.is_file():
        raise FileNotFoundError(f"31 节点测试结果缺失: {path}")
    label = capacity_method(method, SOURCE_CAPACITY)
    return [
        replace(
            record,
            method=label,
            gp_run_id=f"{variant}:{label}:seed-{record.gp_root_seed}",
        )
        for record in read_records([path])
        if record.method == method
    ]


def _load_records(study: CapacityStudySpec) -> list[EvaluationRecord]:
    """严格合并 31 节点只读结果和新 62 节点结果。"""

    ready, detail = source_ready(study)
    if not ready:
        raise RuntimeError(f"容量报告要求源消融完成: {detail}")
    all_records: list[EvaluationRecord] = []
    for variant in study.variants:
        expected_roots = set(variant.seeds)
        for partition in study.partitions:
            context: list[EvaluationRecord] = []
            for method in CAPACITY_METHODS:
                context.extend(
                    _source_records_for_method(
                        study,
                        variant=variant.name,
                        partition=partition,
                        method=method,
                    )
                )
            new_path = study.output_root / "test" / variant.name / partition / "records.csv"
            if not new_path.is_file():
                raise FileNotFoundError(f"62 节点测试结果缺失: {new_path}")
            context.extend(read_records([new_path]))

            methods = {record.method for record in context}
            if methods != set(CAPACITY_EVALUATED_METHODS):
                raise ValueError(
                    f"{variant.name}/{partition}: methods={sorted(methods)}，"
                    f"预期 {sorted(CAPACITY_EVALUATED_METHODS)}"
                )
            if {record.variant for record in context} != {variant.name}:
                raise ValueError(f"{variant.name}/{partition}: variant provenance 错误")
            if {record.partition for record in context} != {partition}:
                raise ValueError(f"{variant.name}/{partition}: partition provenance 错误")
            for method in CAPACITY_EVALUATED_METHODS:
                roots = {record.gp_root_seed for record in context if record.method == method}
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
                raise ValueError(f"{variant.name}/{partition}: 合并记录存在重复配对键")

            baseline: dict[tuple[str, int], tuple[float, float]] = {}
            for record in context:
                key = (record.instance_id, record.seed)
                value = (record.baseline_length, record.baseline_gap_percent)
                previous = baseline.setdefault(key, value)
                if not np.allclose(previous, value, rtol=0.0, atol=1e-12):
                    raise ValueError(f"{variant.name}/{partition}: baseline 不一致 key={key}")
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
    count = max(1, ceil(values.size * fraction))
    return float(np.sort(values)[-count:].mean())


def _quality_row(
    *,
    variant: str,
    partition: str,
    method: str,
    records: list[EvaluationRecord],
) -> dict[str, Any]:
    blocks = _candidate_blocks(records)
    gaps = np.asarray(
        [fmean(item.gap_percent for item in values) for values in blocks.values()],
        dtype=np.float64,
    )
    deltas = np.asarray(
        [fmean(item.delta_pp for item in values) for values in blocks.values()],
        dtype=np.float64,
    )
    tolerance = 1e-12
    return {
        "variant": variant,
        "partition": partition,
        "distribution": records[0].distribution,
        "scale_min": min(record.scale for record in records),
        "scale_max": max(record.scale for record in records),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "gp_runs": len({key[0] for key in blocks}),
        "instances": len({key[1] for key in blocks}),
        "aco_seeds_per_instance": len(records) // len(blocks),
        "blocks": len(blocks),
        "mean_gap_percent": float(gaps.mean()),
        "median_gap_percent": float(np.median(gaps)),
        "standard_deviation": float(gaps.std(ddof=1) if gaps.size > 1 else 0.0),
        "mean_delta_pp": float(deltas.mean()),
        "median_delta_pp": float(np.median(deltas)),
        "win_rate": float(np.mean(deltas < -tolerance)),
        "tie_rate": float(np.mean(np.abs(deltas) <= tolerance)),
        "loss_rate": float(np.mean(deltas > tolerance)),
        "cvar_worst_10_percent": _tail_mean(gaps),
        "reference_hit_rate": float(np.mean(gaps <= 1e-9)),
    }


def _baseline_quality_row(
    *,
    variant: str,
    partition: str,
    records: list[EvaluationRecord],
) -> dict[str, Any]:
    anchor_method = capacity_method("rmtgp-full-f1", SOURCE_CAPACITY)
    anchor_root = min(record.gp_root_seed for record in records if record.method == anchor_method)
    by_instance: dict[str, dict[int, EvaluationRecord]] = defaultdict(dict)
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
    return {
        "variant": variant,
        "partition": partition,
        "distribution": records[0].distribution,
        "scale_min": min(record.scale for record in records),
        "scale_max": max(record.scale for record in records),
        "method": BASELINE_METHOD,
        "method_label": METHOD_LABELS[BASELINE_METHOD],
        "gp_runs": 0,
        "instances": len(by_instance),
        "aco_seeds_per_instance": len(next(iter(by_instance.values()))),
        "blocks": len(by_instance),
        "mean_gap_percent": float(gaps.mean()),
        "median_gap_percent": float(np.median(gaps)),
        "standard_deviation": float(gaps.std(ddof=1) if gaps.size > 1 else 0.0),
        "mean_delta_pp": 0.0,
        "median_delta_pp": 0.0,
        "win_rate": 0.0,
        "tie_rate": 1.0,
        "loss_rate": 0.0,
        "cvar_worst_10_percent": _tail_mean(gaps),
        "reference_hit_rate": float(np.mean(gaps <= 1e-9)),
    }


def _quality_summaries(
    study: CapacityStudySpec,
    records: list[EvaluationRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in study.variants:
        for partition in study.partitions:
            context = [
                record
                for record in records
                if record.variant == variant.name and record.partition == partition
            ]
            for method in CAPACITY_EVALUATED_METHODS:
                rows.append(
                    _quality_row(
                        variant=variant.name,
                        partition=partition,
                        method=method,
                        records=[record for record in context if record.method == method],
                    )
                )
            rows.append(
                _baseline_quality_row(
                    variant=variant.name,
                    partition=partition,
                    records=context,
                )
            )
    return rows


def _read_training_curves(
    study: CapacityStudySpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curves: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for variant in study.variants:
        for method in CAPACITY_METHODS:
            for nodes in (SOURCE_CAPACITY, TARGET_CAPACITY):
                label = capacity_method(method, nodes)
                for root_seed in variant.seeds:
                    run = (
                        source_run_path(study, method, variant.name, root_seed)
                        if nodes == SOURCE_CAPACITY
                        else capacity_run_path(study, method, variant.name, root_seed)
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
                            "method": label,
                            "base_method": method,
                            "node_budget": nodes,
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
                    if champion.total_nodes > nodes:
                        raise ValueError(f"{run}: champion 节点数 {champion.total_nodes}>{nodes}")
                    runs.append(
                        {
                            "variant": variant.name,
                            "method": label,
                            "method_label": METHOD_LABELS[label],
                            "base_method": method,
                            "architecture": ("single" if method != "rmtgp-full-f1" else "multi"),
                            "node_budget": nodes,
                            "artifact_source": (
                                "reused-31-node-ablation"
                                if nodes == SOURCE_CAPACITY
                                else "new-62-node-run"
                            ),
                            "gp_root_seed": root_seed,
                            "generations": len(run_rows),
                            "mean_generation_wall_time_sec": fmean(
                                row["generation_wall_time_sec"] for row in run_rows
                            ),
                            "total_generation_wall_time_sec": sum(
                                row["generation_wall_time_sec"] for row in run_rows
                            ),
                            "final_train_gap_percent": final["train_candidate_gap_percent"],
                            "final_train_delta_pp": final["train_delta_pp"],
                            "final_validation_gap_percent": final[
                                "validation_candidate_gap_percent"
                            ],
                            "final_validation_delta_pp": final["validation_delta_pp"],
                            "transition_nodes": champion.transition_nodes,
                            "pheromone_nodes": champion.pheromone_nodes,
                            "total_nodes": champion.total_nodes,
                            "budget_utilization": champion.total_nodes / nodes,
                            "transition_expression": str(champion.transition_tree),
                            "pheromone_expression": str(champion.pheromone_tree),
                            "selected_candidate_hash": decision["selected_candidate_hash"],
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
            (
                row["variant"],
                row["method"],
                row["generation"],
                row["scale"],
            )
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


def _training_aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["variant"], row["method"])].append(row)
    rows: list[dict[str, Any]] = []
    for (variant, method), values in sorted(grouped.items()):
        rows.append(
            {
                "variant": variant,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "gp_runs": len(values),
                "node_budget": int(values[0]["node_budget"]),
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
                "mean_final_train_delta_pp": fmean(row["final_train_delta_pp"] for row in values),
                "mean_final_validation_gap_percent": fmean(
                    row["final_validation_gap_percent"] for row in values
                ),
                "mean_final_validation_delta_pp": fmean(
                    row["final_validation_delta_pp"] for row in values
                ),
                "median_total_nodes": median(row["total_nodes"] for row in values),
                "mean_budget_utilization": fmean(row["budget_utilization"] for row in values),
                "at_budget": sum(row["total_nodes"] == row["node_budget"] for row in values),
                "gates_passed": sum(bool(row["final_passed_noninferiority"]) for row in values),
            }
        )
    return rows


def _plot_curves(
    study: CapacityStudySpec,
    aggregate: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "tr-rgp": "#1f77b4",
        "ph-rgp": "#ff7f0e",
        "rmtgp-full-f1": "#2ca02c",
    }
    artifacts: list[str] = []
    for variant in study.variants:
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(12.0, 10.0),
            sharex=True,
            constrained_layout=True,
        )
        for row_index, method in enumerate(CAPACITY_METHODS):
            for nodes, linestyle in ((31, "-"), (62, "--")):
                label = capacity_method(method, nodes)
                selected = sorted(
                    (
                        row
                        for row in aggregate
                        if row["variant"] == variant.name and row["method"] == label
                    ),
                    key=lambda row: row["generation"],
                )
                x = np.asarray([row["generation"] for row in selected])
                validation = np.asarray([row["validation_delta_pp_median"] for row in selected])
                lower = np.asarray([row["validation_delta_pp_min"] for row in selected])
                upper = np.asarray([row["validation_delta_pp_max"] for row in selected])
                wall = np.asarray([row["generation_wall_time_sec_median"] for row in selected])
                axes[row_index, 0].plot(
                    x,
                    validation,
                    color=colors[method],
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=f"B={nodes}",
                )
                axes[row_index, 0].fill_between(
                    x,
                    lower,
                    upper,
                    color=colors[method],
                    alpha=0.05,
                    linewidth=0,
                )
                axes[row_index, 1].plot(
                    x,
                    wall,
                    color=colors[method],
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=f"B={nodes}",
                )
            axes[row_index, 0].axhline(0.0, color="black", linestyle=":", linewidth=0.9)
            axes[row_index, 0].set_ylabel(f"{method}\nValidation Δ (pp)")
            axes[row_index, 0].grid(alpha=0.20)
            axes[row_index, 1].set_ylabel("Generation time (s)")
            axes[row_index, 1].grid(alpha=0.20)
            axes[row_index, 0].legend(fontsize=8)
            axes[row_index, 1].legend(fontsize=8)
        axes[0, 0].set_title(f"{variant.name.upper()}：质量")
        axes[0, 1].set_title(f"{variant.name.upper()}：计算时间")
        axes[-1, 0].set_xlabel("GP generation")
        axes[-1, 1].set_xlabel("GP generation")
        stem = f"capacity_training_curve_{variant.name}"
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
) -> tuple[np.ndarray, int]:
    """生成严格配对的 [method, GP run, instance, ACO seed] gap cube。"""

    candidate: dict[str, dict[int, dict[str, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    baseline: dict[int, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        if record.method not in methods:
            continue
        target = candidate[record.method][record.gp_root_seed][record.instance_id]
        if record.seed in target:
            raise ValueError("capacity paired cube 存在重复 candidate seed")
        target[record.seed] = record.gap_percent
        target_baseline = baseline[record.gp_root_seed][record.instance_id]
        previous = target_baseline.setdefault(record.seed, record.baseline_gap_percent)
        if not np.isclose(previous, record.baseline_gap_percent, rtol=0.0, atol=1e-12):
            raise ValueError("capacity paired cube 的 baseline 不一致")

    common_instances: set[str] | None = None
    for root in roots:
        run_instances = set(baseline[root])
        for method in methods:
            run_instances &= set(candidate[method][root])
        common_instances = (
            run_instances if common_instances is None else common_instances & run_instances
        )
    instances = tuple(sorted(common_instances or ()))
    if not instances:
        raise ValueError("容量方法之间没有共同 run×instance blocks")

    values = np.empty((len(methods), len(roots), len(instances), 3), dtype=np.float64)
    for run_index, root in enumerate(roots):
        for instance_index, instance in enumerate(instances):
            common_seeds = set(baseline[root][instance])
            for method in methods:
                common_seeds &= set(candidate[method][root][instance])
            ordered_seeds = tuple(sorted(common_seeds))
            if len(ordered_seeds) != 3:
                raise ValueError("容量 pilot 每个 block 必须恰有三个 ACO seeds")
            for method_index, method in enumerate(methods):
                values[method_index, run_index, instance_index] = [
                    candidate[method][root][instance][seed] for seed in ordered_seeds
                ]
    return values, len(instances)


def _contrast_definitions(
    method_order: tuple[str, ...],
) -> list[tuple[str, str, str, np.ndarray]]:
    index = {method: position for position, method in enumerate(method_order)}

    def vector(**weights: float) -> np.ndarray:
        result = np.zeros(len(method_order), dtype=np.float64)
        for method, weight in weights.items():
            result[index[method]] = weight
        return result

    tr31 = capacity_method("tr-rgp", 31)
    tr62 = capacity_method("tr-rgp", 62)
    ph31 = capacity_method("ph-rgp", 31)
    ph62 = capacity_method("ph-rgp", 62)
    mt31 = capacity_method("rmtgp-full-f1", 31)
    mt62 = capacity_method("rmtgp-full-f1", 62)
    return [
        (
            "capacity-effect",
            "TR: B62 − B31",
            "tr62-tr31",
            vector(**{tr62: 1.0, tr31: -1.0}),
        ),
        (
            "capacity-effect",
            "PH: B62 − B31",
            "ph62-ph31",
            vector(**{ph62: 1.0, ph31: -1.0}),
        ),
        (
            "capacity-effect",
            "双树: B62 − B31",
            "mt62-mt31",
            vector(**{mt62: 1.0, mt31: -1.0}),
        ),
        (
            "architecture-at-capacity",
            "B31: 双树 − TR",
            "mt31-tr31",
            vector(**{mt31: 1.0, tr31: -1.0}),
        ),
        (
            "architecture-at-capacity",
            "B62: 双树 − TR",
            "mt62-tr62",
            vector(**{mt62: 1.0, tr62: -1.0}),
        ),
        (
            "architecture-at-capacity",
            "B31: 双树 − PH",
            "mt31-ph31",
            vector(**{mt31: 1.0, ph31: -1.0}),
        ),
        (
            "architecture-at-capacity",
            "B62: 双树 − PH",
            "mt62-ph62",
            vector(**{mt62: 1.0, ph62: -1.0}),
        ),
        (
            "capacity-architecture-interaction",
            "(双树−TR)B62 − (双树−TR)B31",
            "did-mt-vs-tr",
            vector(**{mt62: 1.0, tr62: -1.0, mt31: -1.0, tr31: 1.0}),
        ),
        (
            "capacity-architecture-interaction",
            "(双树−PH)B62 − (双树−PH)B31",
            "did-mt-vs-ph",
            vector(**{mt62: 1.0, ph62: -1.0, mt31: -1.0, ph31: 1.0}),
        ),
    ]


def _capacity_contrasts(
    study: CapacityStudySpec,
    records: list[EvaluationRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = _contrast_definitions(CAPACITY_EVALUATED_METHODS)
    coefficient_matrix = np.stack([definition[3] for definition in definitions], axis=0)
    for variant in study.variants:
        for partition in study.partitions:
            context = [
                record
                for record in records
                if record.variant == variant.name and record.partition == partition
            ]
            cube, instances = _aligned_quality_cube(
                context,
                methods=CAPACITY_EVALUATED_METHODS,
                roots=variant.seeds,
            )
            contrast_cube = np.einsum("km,mris->kris", coefficient_matrix, cube)
            context_seed = int.from_bytes(
                sha256(
                    f"{study.test_root_seed}:{variant.name}:{partition}:capacity".encode()
                ).digest()[:8],
                byteorder="little",
                signed=False,
            )
            estimates, intervals = _joint_hierarchical_bootstrap(
                contrast_cube,
                replicates=study.bootstrap_replicates,
                seed=context_seed,
            )
            for index, (family, name, formula, _) in enumerate(definitions):
                # 推断 block 是 GP-run×instance；三个 ACO seeds 先在 block 内平均。
                differences = contrast_cube[index].mean(axis=-1).reshape(-1)
                if np.all(differences == 0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = stats.wilcoxon(
                        differences,
                        zero_method="pratt",
                        alternative="two-sided",
                        method="auto",
                    )
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                tolerance = 1e-12
                rows.append(
                    {
                        "variant": variant.name,
                        "partition": partition,
                        "family": family,
                        "contrast": name,
                        "formula": formula,
                        "estimand": "gap_linear_contrast_pp",
                        "runs": len(variant.seeds),
                        "instances": instances,
                        "aco_seeds": cube.shape[-1],
                        "run_instance_blocks": int(differences.size),
                        "estimate_pp": float(estimates[index]),
                        "lower_95": float(intervals[index, 0]),
                        "upper_95": float(intervals[index, 1]),
                        "wins_negative": int(np.sum(differences < -tolerance)),
                        "ties": int(np.sum(np.abs(differences) <= tolerance)),
                        "losses_positive": int(np.sum(differences > tolerance)),
                        "wilcoxon_statistic": statistic,
                        "raw_p_value": p_value,
                        "holm_p_value": float("nan"),
                        "rank_biserial": _rank_biserial(differences),
                        "bootstrap_replicates": study.bootstrap_replicates,
                        "directionally_negative": bool(estimates[index] < 0.0),
                        "ci_below_zero": bool(intervals[index, 1] < 0.0),
                        "ci_above_zero": bool(intervals[index, 0] > 0.0),
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["family"])].append(row)
    for family_rows in grouped.values():
        adjusted = _holm([row["raw_p_value"] for row in family_rows])
        for row, corrected in zip(family_rows, adjusted, strict=True):
            row["holm_p_value"] = corrected
    return rows


def _efficiency_rows(study: CapacityStudySpec) -> list[dict[str, Any]]:
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
                    "method_label": METHOD_LABELS[str(summary["method"])],
                    "partition": str(summary["partition"]),
                    "scale": int(summary["scale"]),
                    "batch_size": int(payload["batch_sizes"][summary["partition"]]),
                    "observations": int(summary["observations"]),
                    "median_wall_time_sec": float(summary["median_wall_time_sec"]),
                    "median_baseline_wall_time_sec": float(
                        summary["median_baseline_wall_time_sec"]
                    ),
                    "median_overhead_percent": float(summary["median_overhead_percent"]),
                    "median_tours_per_second": float(summary["median_tours_per_second"]),
                    "timing_scope": str(payload["timing_scope"]),
                }
            )
    return rows


def _evidence_audit(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in (
        "capacity-effect",
        "architecture-at-capacity",
        "capacity-architecture-interaction",
    ):
        selected = [row for row in contrasts if row["family"] == family]
        rows.append(
            {
                "family": family,
                "comparisons": len(selected),
                "negative_estimates": sum(row["estimate_pp"] < 0.0 for row in selected),
                "ci_below_zero": sum(row["upper_95"] < 0.0 for row in selected),
                "positive_estimates": sum(row["estimate_pp"] > 0.0 for row in selected),
                "ci_above_zero": sum(row["lower_95"] > 0.0 for row in selected),
            }
        )
    return rows


def _format_ci(row: dict[str, Any]) -> str:
    return f"{row['estimate_pp']:+.4f} [{row['lower_95']:+.4f}, {row['upper_95']:+.4f}]"


def _render_report(
    *,
    study: CapacityStudySpec,
    training: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    efficiency: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    plots: list[str],
) -> str:
    lines = [
        f"# {study.study_id}：GP 节点容量敏感性报告",
        "",
        "## 实验目的与可识别量",
        "",
        (
            "主消融采用每个个体两棵树合计最多 31 个有效节点。本补充实验"
            "将总预算提高为 62，并对 TR 单树、PH 单树和双树 Full-F1 全部"
            "重新训练。单树 B=62 允许唯一活动树最多 62 个节点；双树 B=62"
            " 保持每棵最多 31、两棵合计最多 62。因此结构与总表达容量可被"
            "分别比较，而不是只给双树增加搜索空间。"
        ),
        "",
        (
            "31 与 62 条件共享 ACO 参数、population=100、50 generations、"
            "训练/验证实例、GP root seeds、逐代 instance schedule、baseline"
            " cache 与锁定测试随机流。质量统一为相对 reference tour 的 gap%。"
            "所有 contrast 均以“前项−后项”或公式所示线性组合定义；负值表示"
            "公式前侧具有更低 gap。"
        ),
        "",
        (
            "三个关键 estimand 为：(1) 同结构 B62−B31 的容量效应；"
            "(2) 固定 B 下双树−单树的结构效应；(3) difference-in-differences"
            "，即容量增加是否改变双树相对单树的优势。所有容量与结构"
            " estimands 只在预注册的 TSP100-U 与 TSP500-U 核心分区计算，"
            "不把容量消融扩展为全 OOD 测试。"
        ),
        "",
        (
            "本研究仍是 3 个 GP seeds 的 pilot。95% CI 采用 GP run→instance→"
            "ACO seed 的 10,000 次层次 bootstrap；Wilcoxon 以 GP-run×instance"
            " 为 block，并先平均 block 内 3 个 ACO seeds；Holm 在每个"
            " ACO variant×contrast family 内校正。"
        ),
        "",
        "## 训练、验证与实际模型大小",
        "",
        (
            "| ACO | 方法 | runs | 每代(s) | train Δ(pp) | val Δ(pp) | "
            "节点中位数/预算 | 平均预算利用率 | 撞上预算 | gate |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training:
        lines.append(
            f"| {row['variant'].upper()} | {row['method_label']} | "
            f"{row['gp_runs']} | {row['mean_generation_wall_time_sec']:.3f} | "
            f"{row['mean_final_train_delta_pp']:+.4f} | "
            f"{row['mean_final_validation_delta_pp']:+.4f} | "
            f"{row['median_total_nodes']:.1f}/{row['node_budget']} | "
            f"{row['mean_budget_utilization']:.1%} | "
            f"{row['at_budget']}/{row['gp_runs']} | "
            f"{row['gates_passed']}/{row['gp_runs']} |"
        )
    lines.extend(["", "训练曲线（实线 B=31，虚线 B=62）：", ""])
    for variant in study.variants:
        lines.append(
            f"![{variant.name} capacity curves](capacity_training_curve_{variant.name}.png)"
        )

    quality_map = {(row["variant"], row["partition"], row["method"]): row for row in quality}
    lines.extend(
        [
            "",
            "## 双树主方法的核心分区测试",
            "",
            "| ACO | Test | 原始 ACO gap% | 双树 B31 gap% | 双树 B62 gap% | B62−B31(pp) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    contrast_map = {(row["variant"], row["partition"], row["formula"]): row for row in contrasts}
    mt31 = capacity_method("rmtgp-full-f1", 31)
    mt62 = capacity_method("rmtgp-full-f1", 62)
    for variant in study.variants:
        for partition in study.partitions:
            baseline = quality_map[(variant.name, partition, BASELINE_METHOD)]
            first = quality_map[(variant.name, partition, mt31)]
            second = quality_map[(variant.name, partition, mt62)]
            effect = contrast_map[(variant.name, partition, "mt62-mt31")]
            lines.append(
                f"| {variant.name.upper()} | "
                f"{PARTITION_LABELS[partition]} | "
                f"{baseline['mean_gap_percent']:.4f} | "
                f"{first['mean_gap_percent']:.4f} | "
                f"{second['mean_gap_percent']:.4f} | "
                f"{_format_ci(effect)} |"
            )

    def append_contrasts(
        title: str,
        family: str,
        description: str,
    ) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                description,
                "",
                "| ACO | Test | Contrast (pp, 95% CI) | Holm p | Rank-biserial | −/0/+ blocks |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for row in contrasts:
            if row["family"] != family:
                continue
            lines.append(
                f"| {row['variant'].upper()} | "
                f"{PARTITION_LABELS[row['partition']]} | "
                f"{row['contrast']}: {_format_ci(row)} | "
                f"{row['holm_p_value']:.4g} | "
                f"{row['rank_biserial']:+.4f} | "
                f"{row['wins_negative']}/{row['ties']}/"
                f"{row['losses_positive']} |"
            )

    append_contrasts(
        "RQ-C1：增加节点预算是否改善同一结构",
        "capacity-effect",
        (
            "B62−B31<0 表示额外表达容量降低 reference gap。若效应接近 0，"
            "说明 31 节点上限在该方法/数据上下文中不是主要瓶颈。"
        ),
    )
    append_contrasts(
        "RQ-C2：固定总容量时双树是否优于单树",
        "architecture-at-capacity",
        (
            "分别在 B=31 与 B=62 下比较双树−TR、双树−PH。只有同一容量下"
            "两项均稳定为负，才支持双树角色分解优于两个单树对照。"
        ),
    )
    append_contrasts(
        "RQ-C3：结构优势是否由容量约束驱动",
        "capacity-architecture-interaction",
        (
            "difference-in-differences<0 表示从 31 增至 62 后，双树相对"
            "相应单树变得更有利；>0 则表示单树从额外容量中获益更多。"
        ),
    )

    lines.extend(
        [
            "",
            "## 孤立推理效率",
            "",
            (
                "31/62 champions 在同一进程、相同 warm-up、batch 与随机种子"
                "下逐个运行。该表用于描述表达式求值开销，不对 timing 做显著性"
                "推断，也不把批量质量 campaign 时间分摊给单个 program。"
            ),
            "",
            "| ACO | Test | 方法 | Batch | 中位时间(s) | 相对 ACO 开销 | tours/s |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in efficiency:
        lines.append(
            f"| {row['variant'].upper()} | "
            f"{PARTITION_LABELS[row['partition']]} | "
            f"{row['method_label']} | {row['batch_size']} | "
            f"{row['median_wall_time_sec']:.4f} | "
            f"{row['median_overhead_percent']:+.2f}% | "
            f"{row['median_tours_per_second']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 证据审计",
            "",
            "| Family | comparisons | 负向估计 | CI 全负 | 正向估计 | CI 全正 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit:
        lines.append(
            f"| {row['family']} | {row['comparisons']} | "
            f"{row['negative_estimates']} | {row['ci_below_zero']} | "
            f"{row['positive_estimates']} | {row['ci_above_zero']} |"
        )
    lines.extend(
        [
            "",
            (
                "解释规则：若 B62−B31 没有稳定负效应，则保留 B=31 作为主实验"
                "容量是合理的简约选择；若双树仅在 B=62 才稳定优于两个单树，"
                "主实验可能受到容量上限约束；若双树在 B=31 与 B=62 都稳定"
                "优于两个单树且交互接近 0，则证据更符合角色分解本身，而非"
                "额外节点数造成的优势。"
            ),
            "",
            "## 可复现 artifacts",
            "",
            "- `training_runs.csv`：54 个 31/62 selected champions、表达式与节点",
            "- `training_summary.csv`：variant×method×capacity 训练汇总",
            "- `training_validation_curves_all.csv` 与 `..._aggregate.csv`",
            "- `quality_summary.csv`：六个容量方法及原始 ACO",
            "- `capacity_contrasts.csv`：容量、结构与交互的全部 paired estimands",
            "- `efficiency_summary.csv`：同条件 warm timing",
            "- `evidence_audit.csv`：方向与区间计数",
            "- `capacity_summary.json`：完整机器可读结果与 provenance",
        ]
    )
    lines.extend(f"- `{name}`" for name in plots)
    lines.append("")
    return "\n".join(lines)


def generate_capacity_report(study: CapacityStudySpec) -> Path:
    """验证所有 artifacts，输出统计长表、曲线和中文学术报告。"""

    output = study.output_root / "report"
    output.mkdir(parents=True, exist_ok=True)
    records = _load_records(study)
    quality = _quality_summaries(study, records)
    curves, training_runs = _read_training_curves(study)
    curve_aggregate = _aggregate_curves(curves)
    training = _training_aggregate(training_runs)
    plots = _plot_curves(study, curve_aggregate, output)
    contrasts = _capacity_contrasts(study, records)
    efficiency = _efficiency_rows(study)
    audit = _evidence_audit(contrasts)

    _write_dict_csv(output / "training_runs.csv", training_runs)
    _write_dict_csv(output / "training_summary.csv", training)
    _write_dict_csv(output / "training_validation_curves_all.csv", curves)
    _write_dict_csv(
        output / "training_validation_curves_aggregate.csv",
        curve_aggregate,
    )
    _write_dict_csv(output / "quality_summary.csv", quality)
    _write_dict_csv(output / "capacity_contrasts.csv", contrasts)
    _write_dict_csv(output / "efficiency_summary.csv", efficiency)
    _write_dict_csv(output / "evidence_audit.csv", audit)

    summary = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "study": export_capacity_contract(study),
        "git": git_state(Path.cwd()),
        "interpretation": {
            "phase": "three-seed-capacity-sensitivity-pilot",
            "confirmatory": False,
            "fitness_target": "reference_gap_percent",
            "contrast_sign": "linear gap contrast; negative follows formula",
            "pairing_unit": "gp_root_seed-instance_id-aco_seed",
            "wilcoxon_block": "gp_run_x_instance_after_aco_seed_mean",
            "holm_family": "aco_variant_x_capacity_question_family",
            "bootstrap_hierarchy": ["gp_run", "instance", "aco_seed"],
            "bootstrap_replicates": study.bootstrap_replicates,
            "champion_selection": "one_validation_selected_candidate_per_gp_run",
            "best_gp_seed_selection": False,
            "primary_capacity_partitions": list(study.partitions),
            "quality_timing": "excluded_batched_campaign_allocation",
            "efficiency_timing": "same-process-isolated-warm-program",
        },
        "artifact_counts": {
            "training_runs": len(training_runs),
            "evaluation_records": len(records),
            "quality_summaries": len(quality),
            "contrasts": len(contrasts),
            "efficiency_rows": len(efficiency),
        },
        "training_summary": training,
        "quality_summary": quality,
        "capacity_contrasts": contrasts,
        "efficiency_summary": efficiency,
        "evidence_audit": audit,
        "plots": plots,
    }
    _atomic_text(
        output / "capacity_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
    )
    target = output / "capacity_report.md"
    _atomic_text(
        target,
        _render_report(
            study=study,
            training=training,
            quality=quality,
            contrasts=contrasts,
            efficiency=efficiency,
            audit=audit,
            plots=plots,
        ),
    )
    return target
