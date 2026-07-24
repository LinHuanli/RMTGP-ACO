"""三算法三种子 study 的训练曲线、paired 统计与中文报告。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from scipy import stats

from .artifacts import git_state
from .evaluation import EvaluationRecord, read_records
from .stats import hierarchical_bootstrap_delta
from .study import StudySpec, export_study_contract

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


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path}: 不允许写空 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_curves(study: StudySpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in study.variants:
        for root_seed in variant.seeds:
            path = (
                study.output_root
                / "train"
                / variant.name
                / f"seed-{root_seed}"
                / "training_validation_curve.csv"
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                run_rows = list(csv.DictReader(handle))
            generations = {int(row["generation"]) for row in run_rows}
            if generations != set(range(1, 51)):
                raise ValueError(
                    f"{path}: 正式曲线必须恰好包含 generation 1..50"
                )
            for raw in run_rows:
                row: dict[str, Any] = {
                    "variant": variant.name,
                    "gp_root_seed": root_seed,
                    "generation": int(raw["generation"]),
                    "scale": int(raw["scale"]),
                }
                row.update({name: float(raw[name]) for name in CURVE_FIELDS})
                rows.append(row)
    return rows


def _aggregate_curves(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["generation"], row["scale"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for (variant, generation, scale), values in sorted(grouped.items()):
        result: dict[str, Any] = {
            "variant": variant,
            "generation": generation,
            "scale": scale,
            "gp_runs": len(values),
        }
        for field in CURVE_FIELDS:
            data = np.asarray([row[field] for row in values], dtype=np.float64)
            result[f"{field}_mean"] = float(np.nanmean(data))
            result[f"{field}_median"] = float(np.nanmedian(data))
            result[f"{field}_min"] = float(np.nanmin(data))
            result[f"{field}_max"] = float(np.nanmax(data))
        aggregate.append(result)
    return aggregate


def _plot_curves(
    study: StudySpec,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts: list[str] = []
    colors = {
        "train_candidate_gap_percent": "#1f77b4",
        "train_baseline_gap_percent": "#7f7f7f",
        "validation_candidate_gap_percent": "#d62728",
        "validation_baseline_gap_percent": "#9467bd",
    }
    labels = {
        "train_candidate_gap_percent": "Train candidate",
        "train_baseline_gap_percent": "Train baseline",
        "validation_candidate_gap_percent": "Validation candidate",
        "validation_baseline_gap_percent": "Validation baseline",
    }
    for variant in study.variants:
        selected = [row for row in rows if row["variant"] == variant.name]
        summary = [row for row in aggregate if row["variant"] == variant.name]
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(8.4, 7.2),
            sharex=True,
            constrained_layout=True,
        )
        for root_seed in variant.seeds:
            run = sorted(
                (
                    row
                    for row in selected
                    if row["gp_root_seed"] == root_seed
                ),
                key=lambda row: row["generation"],
            )
            x = [row["generation"] for row in run]
            axes[0].plot(
                x,
                [row["train_candidate_gap_percent"] for row in run],
                color=colors["train_candidate_gap_percent"],
                alpha=0.18,
                linewidth=0.9,
            )
            axes[0].plot(
                x,
                [row["validation_candidate_gap_percent"] for row in run],
                color=colors["validation_candidate_gap_percent"],
                alpha=0.18,
                linewidth=0.9,
            )
            axes[1].plot(
                x,
                [row["train_delta_pp"] for row in run],
                color="#1f77b4",
                alpha=0.18,
                linewidth=0.9,
            )
            axes[1].plot(
                x,
                [row["validation_delta_pp"] for row in run],
                color="#d62728",
                alpha=0.18,
                linewidth=0.9,
            )
        summary.sort(key=lambda row: row["generation"])
        x = np.asarray([row["generation"] for row in summary])
        for field in colors:
            median = np.asarray([row[f"{field}_median"] for row in summary])
            lower = np.asarray([row[f"{field}_min"] for row in summary])
            upper = np.asarray([row[f"{field}_max"] for row in summary])
            axes[0].plot(
                x,
                median,
                color=colors[field],
                linewidth=2.0,
                label=labels[field],
            )
            axes[0].fill_between(
                x,
                lower,
                upper,
                color=colors[field],
                alpha=0.08,
                linewidth=0,
            )
        for field, color, label in (
            ("train_delta_pp", "#1f77b4", "Train candidate − baseline"),
            ("validation_delta_pp", "#d62728", "Validation candidate − baseline"),
        ):
            median = [row[f"{field}_median"] for row in summary]
            lower = [row[f"{field}_min"] for row in summary]
            upper = [row[f"{field}_max"] for row in summary]
            axes[1].plot(x, median, color=color, linewidth=2.0, label=label)
            axes[1].fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        axes[0].set_title(f"{variant.name.upper()} — pure TSP100, 3 GP seeds")
        axes[0].set_ylabel("Reference gap (%)")
        axes[1].set_ylabel("Paired delta (pp)")
        axes[1].set_xlabel("Generation")
        axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, ncol=2)
        for suffix in ("png", "svg"):
            path = output / f"train_validation_curve_{variant.name}.{suffix}"
            figure.savefig(path, dpi=180 if suffix == "png" else None)
            artifacts.append(path.name)
        plt.close(figure)
    return artifacts


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    return float(
        (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum())
        / ranks.sum()
    )


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, original in enumerate(order):
        running = max(
            running,
            min(1.0, (len(p_values) - rank) * p_values[int(original)]),
        )
        adjusted[int(original)] = running
    return adjusted.tolist()


def _tail_mean(values: np.ndarray) -> float:
    count = max(1, ceil(0.10 * values.size))
    return float(np.sort(values)[-count:].mean())


def _basic_summary(
    *,
    variant: str,
    partition: str,
    scale: int,
    method: str,
    gaps: np.ndarray,
    deltas: np.ndarray,
    auc: np.ndarray,
    wall: np.ndarray,
    throughput: np.ndarray,
    best_iteration: np.ndarray,
    gp_runs: int,
    instances: int,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "partition": partition,
        "scale": scale,
        "method": method,
        "gp_runs": gp_runs,
        "instances": instances,
        "blocks": int(gaps.size),
        "mean_gap_percent": float(gaps.mean()),
        "median_gap_percent": float(np.median(gaps)),
        "standard_deviation": float(
            gaps.std(ddof=1) if gaps.size > 1 else 0.0
        ),
        "q1_gap_percent": float(np.quantile(gaps, 0.25)),
        "q3_gap_percent": float(np.quantile(gaps, 0.75)),
        "mean_delta_pp": float(deltas.mean()),
        "median_delta_pp": float(np.median(deltas)),
        "win_rate": float(np.mean(deltas < -1e-12)),
        "tie_rate": float(np.mean(np.abs(deltas) <= 1e-12)),
        "loss_rate": float(np.mean(deltas > 1e-12)),
        "cvar_worst_10_percent": _tail_mean(gaps),
        "reference_hit_rate": float(np.mean(gaps <= 1e-9)),
        "mean_anytime_gap_auc": float(np.nanmean(auc)),
        "mean_wall_time_sec": float(np.nanmean(wall)),
        "mean_tours_per_second": float(np.nanmean(throughput)),
        "mean_best_iteration": float(np.nanmean(best_iteration)),
    }


def _test_summaries(
    study: StudySpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    tests_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bootstraps: list[dict[str, Any]] = []
    for variant in study.variants:
        decisions = {
            root_seed: json.loads(
                (
                    study.output_root
                    / "train"
                    / variant.name
                    / f"seed-{root_seed}"
                    / "deployment_decision.json"
                ).read_text(encoding="utf-8")
            )
            for root_seed in variant.seeds
        }
        for partition in study.partitions:
            path = (
                study.output_root
                / "test"
                / variant.name
                / partition
                / "records.csv"
            )
            records = read_records([path])
            if not records:
                raise ValueError(f"{path}: 测试长表为空")
            scale = records[0].scale
            instances = len({record.instance_id for record in records})

            # candidate 与 baseline 均先在 ACO seeds 内平均；candidate 保留
            # GP-run×instance 层，正好对应层次 bootstrap 的实验单位。
            candidate_blocks: dict[
                tuple[int, str], list[EvaluationRecord]
            ] = defaultdict(list)
            baseline_blocks: dict[str, dict[int, EvaluationRecord]] = defaultdict(
                dict
            )
            deployed_blocks: dict[
                tuple[int, str], list[EvaluationRecord]
            ] = defaultdict(list)
            for record in records:
                candidate_blocks[
                    (record.gp_root_seed, record.instance_id)
                ].append(record)
                baseline_blocks[record.instance_id].setdefault(
                    record.seed, record
                )
                deployed_blocks[
                    (record.gp_root_seed, record.instance_id)
                ].append(record)

            candidate_gap = np.asarray(
                [
                    fmean(item.gap_percent for item in block)
                    for block in candidate_blocks.values()
                ]
            )
            candidate_delta = np.asarray(
                [
                    fmean(item.delta_pp for item in block)
                    for block in candidate_blocks.values()
                ]
            )
            candidate_auc = np.asarray(
                [
                    fmean(item.anytime_gap_auc for item in block)
                    for block in candidate_blocks.values()
                ]
            )
            candidate_wall = np.asarray(
                [
                    fmean(item.wall_time_sec for item in block)
                    for block in candidate_blocks.values()
                ]
            )
            candidate_tps = np.asarray(
                [
                    fmean(item.tours_per_second for item in block)
                    for block in candidate_blocks.values()
                ]
            )
            candidate_iteration = np.asarray(
                [
                    fmean(item.best_iteration for item in block)
                    for block in candidate_blocks.values()
                ]
            )
            summaries.append(
                _basic_summary(
                    variant=variant.name,
                    partition=partition,
                    scale=scale,
                    method=study.method_profile,
                    gaps=candidate_gap,
                    deltas=candidate_delta,
                    auc=candidate_auc,
                    wall=candidate_wall,
                    throughput=candidate_tps,
                    best_iteration=candidate_iteration,
                    gp_runs=len(variant.seeds),
                    instances=instances,
                )
            )

            baseline_rows = [
                list(seed_map.values())
                for seed_map in baseline_blocks.values()
            ]
            baseline_gap = np.asarray(
                [
                    fmean(item.baseline_gap_percent for item in block)
                    for block in baseline_rows
                ]
            )
            baseline_auc = np.asarray(
                [
                    fmean(item.baseline_anytime_gap_auc for item in block)
                    for block in baseline_rows
                ]
            )
            baseline_wall = np.asarray(
                [
                    fmean(item.baseline_wall_time_sec for item in block)
                    for block in baseline_rows
                ]
            )
            baseline_tps = np.asarray(
                [
                    fmean(item.baseline_tours_per_second for item in block)
                    for block in baseline_rows
                ]
            )
            baseline_iteration = np.asarray(
                [
                    fmean(item.baseline_best_iteration for item in block)
                    for block in baseline_rows
                ]
            )
            summaries.append(
                _basic_summary(
                    variant=variant.name,
                    partition=partition,
                    scale=scale,
                    method="baseline-aco",
                    gaps=baseline_gap,
                    deltas=np.zeros_like(baseline_gap),
                    auc=baseline_auc,
                    wall=baseline_wall,
                    throughput=baseline_tps,
                    best_iteration=baseline_iteration,
                    gp_runs=0,
                    instances=instances,
                )
            )

            deployed_gap = np.asarray(
                [
                    fmean(
                        (
                            item.gap_percent
                            if decisions[root_seed][
                                "final_passed_noninferiority"
                            ]
                            else item.baseline_gap_percent
                        )
                        for item in block
                    )
                    for (root_seed, _), block in deployed_blocks.items()
                ]
            )
            deployed_delta = np.asarray(
                [
                    fmean(
                        (
                            item.delta_pp
                            if decisions[root_seed][
                                "final_passed_noninferiority"
                            ]
                            else 0.0
                        )
                        for item in block
                    )
                    for (root_seed, _), block in deployed_blocks.items()
                ]
            )
            deployed_auc = np.asarray(
                [
                    fmean(
                        (
                            item.anytime_gap_auc
                            if decisions[root_seed][
                                "final_passed_noninferiority"
                            ]
                            else item.baseline_anytime_gap_auc
                        )
                        for item in block
                    )
                    for (root_seed, _), block in deployed_blocks.items()
                ]
            )
            deployed_wall = np.asarray(
                [
                    fmean(
                        (
                            item.wall_time_sec
                            if decisions[root_seed][
                                "final_passed_noninferiority"
                            ]
                            else item.baseline_wall_time_sec
                        )
                        for item in block
                    )
                    for (root_seed, _), block in deployed_blocks.items()
                ]
            )
            deployed_tps = np.asarray(
                [
                    fmean(
                        (
                            item.tours_per_second
                            if decisions[root_seed][
                                "final_passed_noninferiority"
                            ]
                            else item.baseline_tours_per_second
                        )
                        for item in block
                    )
                    for (root_seed, _), block in deployed_blocks.items()
                ]
            )
            deployed_iteration = np.asarray(
                [
                    fmean(
                        (
                            item.best_iteration
                            if decisions[root_seed][
                                "final_passed_noninferiority"
                            ]
                            else item.baseline_best_iteration
                        )
                        for item in block
                    )
                    for (root_seed, _), block in deployed_blocks.items()
                ]
            )
            summaries.append(
                _basic_summary(
                    variant=variant.name,
                    partition=partition,
                    scale=scale,
                    method="gate-deployed",
                    gaps=deployed_gap,
                    deltas=deployed_delta,
                    auc=deployed_auc,
                    wall=deployed_wall,
                    throughput=deployed_tps,
                    best_iteration=deployed_iteration,
                    gp_runs=len(variant.seeds),
                    instances=instances,
                )
            )

            if np.all(candidate_delta == 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = stats.wilcoxon(
                    candidate_delta,
                    zero_method="pratt",
                    alternative="two-sided",
                    method="auto",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            tests_by_variant[variant.name].append(
                {
                    "variant": variant.name,
                    "partition": partition,
                    "scale": scale,
                    "blocks": int(candidate_delta.size),
                    "difference": "candidate_minus_baseline",
                    "mean_difference_pp": float(candidate_delta.mean()),
                    "wilcoxon_statistic": statistic,
                    "raw_p_value": p_value,
                    "holm_p_value": float("nan"),
                    "rank_biserial": _rank_biserial(candidate_delta),
                }
            )
            interval = hierarchical_bootstrap_delta(
                records,
                replicates=study.bootstrap_replicates,
                seed=study.test_root_seed + scale,
            )[0]
            bootstraps.append(
                {
                    "variant": variant.name,
                    "partition": partition,
                    "scale": scale,
                    **asdict(interval),
                }
            )

    tests: list[dict[str, Any]] = []
    for variant in study.variants:
        rows = tests_by_variant[variant.name]
        adjusted = _holm([row["raw_p_value"] for row in rows])
        for row, corrected in zip(rows, adjusted, strict=True):
            row["holm_p_value"] = corrected
            tests.append(row)
    return summaries, tests, bootstraps


def _training_run_summaries(
    study: StudySpec,
    curves: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in curves:
        grouped[(row["variant"], row["gp_root_seed"])].append(row)
    for (variant, root_seed), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["generation"])
        final = rows[-1]
        decision_path = (
            study.output_root
            / "train"
            / variant
            / f"seed-{root_seed}"
            / "deployment_decision.json"
        )
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        results.append(
            {
                "variant": variant,
                "gp_root_seed": root_seed,
                "generations": len(rows),
                "mean_generation_wall_time_sec": fmean(
                    row["generation_wall_time_sec"] for row in rows
                ),
                "total_generation_wall_time_sec": sum(
                    row["generation_wall_time_sec"] for row in rows
                ),
                "mean_validation_monitor_wall_time_sec": fmean(
                    row["validation_monitor_wall_time_sec"] for row in rows
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
                "selected_candidate_hash": decision["selected_candidate_hash"],
                "selection_passed_noninferiority": decision[
                    "selection_passed_noninferiority"
                ],
                "cpu_fp64_audit_passed": decision["cpu_fp64_audit_passed"],
                "final_passed_noninferiority": decision[
                    "final_passed_noninferiority"
                ],
                "deployed_method": decision["deployed_method"],
            }
        )
    return results


def _render_report(
    *,
    study: StudySpec,
    training: list[dict[str, Any]],
    test: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    plots: list[str],
) -> str:
    lines = [
        f"# {study.study_id} 完整预算 pilot 报告",
        "",
        "## 实验合同",
        "",
        (
            "本报告对应纯 TSP100 训练与验证；AS、ACS、MMAS 各 3 个独立 "
            "GP root seed，GP population=100、50 代，ACO 为 32 ants×500 "
            "iterations。最终测试统一使用与 GP seed 解耦的 3 个 ACO seeds。"
        ),
        "",
        (
            "TSP50/100/500 为尺度内与尺度外测试；TSP1000 仅为锁定模型后的"
            "补充外推测试，不参与训练、验证、候选选择或 gate。"
        ),
        "",
        (
            "这是 3-seed pilot：置信区间和显著性检验用于描述与筛查，不能"
            "替代预注册的 30-run confirmatory experiment。"
        ),
        "",
        "## 训练结果",
        "",
        "| ACO | GP seed | 每代均时(s) | 最终 train gap% | 最终 val gap% | val Δ(pp) | gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in training:
        lines.append(
            f"| {row['variant'].upper()} | {row['gp_root_seed']} | "
            f"{row['mean_generation_wall_time_sec']:.3f} | "
            f"{row['final_train_gap_percent']:.4f} | "
            f"{row['final_validation_gap_percent']:.4f} | "
            f"{row['final_validation_delta_pp']:+.4f} | "
            f"{'candidate' if row['final_passed_noninferiority'] else 'baseline fallback'} |"
        )
    lines.extend(["", "训练/验证曲线（细线为独立 seed，粗线为中位数，阴影为 min–max）：", ""])
    for variant in study.variants:
        lines.append(
            f"![{variant.name} train-validation](train_validation_curve_{variant.name}.png)"
        )

    lines.extend(
        [
            "",
            "## 最终测试",
            "",
            (
                "下表同时报告锁定的 selected candidate、原始 baseline ACO 和"
                "经 validation gate 后的 deployed 行为。Δ=候选 gap−baseline gap，"
                "负值表示学习方法更好。"
            ),
            "",
            "| ACO | Test | 方法 | Mean gap% | Δ(pp) | W/T/L | CVaR10% | Anytime AUC |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in test:
        wtl = (
            f"{row['win_rate']:.1%}/"
            f"{row['tie_rate']:.1%}/"
            f"{row['loss_rate']:.1%}"
        )
        lines.append(
            f"| {row['variant'].upper()} | TSP{row['scale']} | "
            f"{row['method']} | {row['mean_gap_percent']:.4f} | "
            f"{row['mean_delta_pp']:+.4f} | {wtl} | "
            f"{row['cvar_worst_10_percent']:.4f} | "
            f"{row['mean_anytime_gap_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 配对统计",
            "",
            (
                "Wilcoxon 以 GP-run×instance 为 block，在 ACO seeds 内先平均；"
                "每个 ACO 变体内对四个尺度作 Holm 校正。层次 bootstrap 依次"
                "重采样 GP run、instance、ACO seed。"
            ),
            "",
            "| ACO | Test | blocks | Mean Δ(pp) | Holm p | Rank-biserial | Bootstrap 95% CI |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    bootstrap_map = {
        (row["variant"], row["partition"]): row for row in bootstrap
    }
    for row in paired:
        interval = bootstrap_map[(row["variant"], row["partition"])]
        lines.append(
            f"| {row['variant'].upper()} | TSP{row['scale']} | "
            f"{row['blocks']} | {row['mean_difference_pp']:+.4f} | "
            f"{row['holm_p_value']:.4g} | {row['rank_biserial']:+.4f} | "
            f"[{interval['lower_95']:+.4f}, {interval['upper_95']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            (
                "仅当 selected candidate 在独立测试上的 gap、层次 bootstrap 和"
                "配对效应方向一致时，才可描述为本 pilot 中优于对应 baseline；"
                "gate-deployed 行为用于报告实际部署性能，不能替代对学习组件本身"
                "的 selected-candidate 分析。"
            ),
            "",
            (
                "本 study 比较的是相对参考最优/高质量标签的 gap%，而不是把"
                "baseline-relative 值作为 fitness。Residual vs replacement、双树"
                " vs 单树以及 function/terminal set 的因果归因仍须由既定 ablation"
                " 矩阵回答，本次九个主方法 run 本身不能回答这些机制问题。"
            ),
            "",
            "## Artifact",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in plots)
    lines.extend(
        [
            "- `training_runs.csv`",
            "- `training_validation_curves_all.csv`",
            "- `training_validation_curves_aggregate.csv`",
            "- `test_summary.csv`",
            "- `paired_tests.csv`",
            "- `bootstrap_intervals.csv`",
            "- `study_summary.json`",
            "",
        ]
    )
    return "\n".join(lines)


def generate_study_report(study: StudySpec) -> Path:
    """验证全部输入后生成 CSV、JSON、PNG/SVG 与中文 Markdown。"""

    output = study.output_root / "report"
    output.mkdir(parents=True, exist_ok=True)
    curves = _read_curves(study)
    aggregate = _aggregate_curves(curves)
    _write_dict_csv(output / "training_validation_curves_all.csv", curves)
    _write_dict_csv(
        output / "training_validation_curves_aggregate.csv",
        aggregate,
    )
    plots = _plot_curves(study, curves, aggregate, output)
    training = _training_run_summaries(study, curves)
    _write_dict_csv(output / "training_runs.csv", training)

    test, paired, bootstrap = _test_summaries(study)
    _write_dict_csv(output / "test_summary.csv", test)
    _write_dict_csv(output / "paired_tests.csv", paired)
    _write_dict_csv(output / "bootstrap_intervals.csv", bootstrap)

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "study": export_study_contract(study),
        "git": git_state(Path.cwd()),
        "interpretation": {
            "phase": "three-seed-pilot",
            "confirmatory": False,
            "fitness_target": "reference_gap_percent",
            "test_seed_independent_of_gp_seed": True,
            "tsp1000_role": "test-only-supplementary-extrapolation",
            "selected_candidate_reported_separately": True,
            "gate_deployed_reported_separately": True,
        },
        "training_runs": training,
        "test_summaries": test,
        "paired_tests": paired,
        "bootstrap_intervals": bootstrap,
        "plots": plots,
    }
    _atomic_text(
        output / "study_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
    )
    target = output / "study_report.md"
    _atomic_text(
        target,
        _render_report(
            study=study,
            training=training,
            test=test,
            paired=paired,
            bootstrap=bootstrap,
            plots=plots,
        ),
    )
    return target
