"""从冻结的主实验、消融和容量报告生成论文图表。

脚本会验证 study id、完成状态与任务总数。任一正式报告未完成时立即失败，
因此论文资产不会混入中间 shard 或旧版本 study 的结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from build_extended_assets import (
    build_extended_assets,
    load_extended_sources,
)

plt.switch_backend("Agg")


VARIANT_ORDER = ("as", "acs", "mmas")
VARIANT_LABEL = {"as": "AS", "acs": "ACS", "mmas": "MMAS"}
SCALE_ORDER = (50, 100, 500, 1000)
ABLATION_PARTITION_ORDER = ("tsp100_uniform", "tsp500_uniform")
OOD_PARTITION_ORDER = (
    "tsp500_cluster",
    "tsp500_gaussian",
    "tsplib_le500",
)
INFERENCE_PARTITION_ORDER = (
    "tsp50_uniform",
    "tsp100_uniform",
    "tsp500_uniform",
    "tsp1000_uniform",
)
PARTITION_LABEL = {
    "tsp50_uniform": "TSP50-U",
    "tsp100_uniform": "TSP100-U",
    "tsp500_uniform": "TSP500-U",
    "tsp1000_uniform": "TSP1000-U",
    "tsp500_cluster": "TSP500-C",
    "tsp500_gaussian": "TSP500-G",
    "tsplib_le500": r"TSPLIB$\leq$500",
}
PLOT_PARTITION_LABEL = {
    "tsp100_uniform": "TSP100-U",
    "tsp500_uniform": "TSP500-U",
    "tsp500_cluster": "TSP500-C",
    "tsp500_gaussian": "TSP500-G",
    "tsplib_le500": "TSPLIB≤500",
}
TRANSITION_TERMINALS = (
    "RTau",
    "REta",
    "BaseConf",
    "DistRank",
    "Entropy",
    "ConstructProg",
    "ACOProg",
    "Stagnation",
)
PHEROMONE_TERMINALS = (
    "EdgeEta",
    "EdgeTau",
    "NNRank",
    "ColonyFreq",
    "SourceQuality",
    "ACOProg",
    "Stagnation",
)
ABLATION_METHOD_LABEL = {
    "legacy": "Legacy-GP",
    "matched-replace": "Matched-Replace",
    "tr-rgp": "TR-RGP",
    "ph-rgp": "PH-RGP",
    "rmtgp-core-f0": "Core-F0",
    "rmtgp-core-f1": "Core-F1",
    "rmtgp-full-f0": "Full-F0",
    "rmtgp-full-f1": "Full-F1",
}
COLORS = {
    "as": "#0072B2",
    "acs": "#D55E00",
    "mmas": "#009E73",
    "candidate": "#0072B2",
    "validation": "#D55E00",
    "baseline": "#6B7280",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV，并在缺少输入时给出清晰错误。"""

    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结报告文件: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON object，并拒绝缺失或非 object 输入。"""

    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结报告文件: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: 顶层必须是 JSON object")
    return payload


def require_completed_study(
    root: Path,
    *,
    study_id: str,
    total_tasks: int,
    summary_name: str,
    summary_schema: int = 2,
) -> dict[str, Any]:
    """只允许完整且任务数匹配的 study 进入论文资产。"""

    state = read_json(root / "study_state.json")
    completed = state.get("completed_count")
    if completed is None:
        completed = len(state.get("completed_tasks", ()))
    actual_total = state.get("total_tasks")
    if not (
        state.get("study_id") == study_id
        and state.get("status") == "completed"
        and int(completed) == total_tasks
        and int(actual_total) == total_tasks
    ):
        raise RuntimeError(
            f"{root}: study 尚未冻结完成；"
            f"id={state.get('study_id')!r}, status={state.get('status')!r}, "
            f"tasks={completed}/{actual_total}"
        )
    summary = read_json(root / "report" / summary_name)
    if summary.get("schema_version") != summary_schema:
        raise ValueError(
            f"{summary_name}: schema_version 不是 {summary_schema}"
        )
    return summary


def indexed_rows(
    rows: list[dict[str, str]],
    *fields: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    """用指定字段建立唯一键索引。"""

    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"CSV 出现重复键: {key}")
        result[key] = row
    return result


def context_order() -> tuple[tuple[str, str], ...]:
    """返回论文 forest plot 的固定 ACO×partition 顺序。"""

    return tuple(
        (variant, partition)
        for variant in VARIANT_ORDER
        for partition in ABLATION_PARTITION_ORDER
    )


def context_plot_label(variant: str, partition: str) -> str:
    return f"{VARIANT_LABEL[variant]} · {PLOT_PARTITION_LABEL[partition]}"


def tex_ci(row: dict[str, str], *, digits: int = 3) -> str:
    """格式化 estimate 与 95% CI；显式保留正负号。"""

    estimate = float(row["estimate_pp"])
    lower = float(row["lower_95"])
    upper = float(row["upper_95"])
    return (
        f"{estimate:+.{digits}f} "
        f"[{lower:+.{digits}f}, {upper:+.{digits}f}]"
    )


def write_tex_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_plot_style() -> None:
    """使用适合 IEEE 双栏缩放的字体、线宽和配色。"""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_main_result_rows(
    test_rows: list[dict[str, str]],
    interval_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成主结果表的 LaTeX 数据行。"""

    method_rows = {
        (row["variant"], int(row["scale"])): row
        for row in test_rows
        if row["method"] == "rmtgp-full-f1"
    }
    baseline_rows = {
        (row["variant"], int(row["scale"])): row
        for row in test_rows
        if row["method"] == "baseline-aco"
    }
    intervals = {
        (row["variant"], int(row["scale"])): row for row in interval_rows
    }

    lines = [
        "% 本文件由 build_paper_assets.py 生成，请勿手工修改。",
    ]
    for variant in VARIANT_ORDER:
        for scale in SCALE_ORDER:
            key = (variant, scale)
            method = method_rows[key]
            baseline = baseline_rows[key]
            interval = intervals[key]
            win = 100.0 * float(method["win_rate"])
            tie = 100.0 * float(method["tie_rate"])
            loss = 100.0 * float(method["loss_rate"])
            lines.append(
                f"{VARIANT_LABEL[variant]} & {scale} & "
                f"{float(baseline['mean_gap_percent']):.4f} & "
                f"{float(method['mean_gap_percent']):.4f} & "
                f"{float(method['mean_delta_pp']):+.4f} & "
                f"[{float(interval['lower_95']):.4f}, "
                f"{float(interval['upper_95']):.4f}] & "
                f"{win:.1f}/{tie:.1f}/{loss:.1f} \\\\"
            )
    # booktabs 的 \noalign 必须紧跟最后一个换行，放在同一输入文件最稳妥。
    lines.append(r"\bottomrule")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_time_rows(
    training_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """汇总三次独立 GP run 的每代时间。"""

    by_variant: dict[str, list[float]] = defaultdict(list)
    for row in training_rows:
        by_variant[row["variant"]].append(
            float(row["mean_generation_wall_time_sec"])
        )

    lines = [
        "% 本文件由 build_paper_assets.py 生成，请勿手工修改。",
    ]
    for variant in VARIANT_ORDER:
        values = by_variant[variant]
        deviation = stdev(values) if len(values) > 1 else 0.0
        lines.append(
            f"{VARIANT_LABEL[variant]} & {mean(values):.2f} $\\pm$ "
            f"{deviation:.2f} & {min(values):.2f}--{max(values):.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_main_gap_delta(
    interval_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制 Full-F1 相对原始 ACO 的 reference-gap 差值。"""

    lookup = {
        (row["variant"], int(row["scale"])): row for row in interval_rows
    }
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
    x = np.arange(len(SCALE_ORDER))
    for axis, variant in zip(axes, VARIANT_ORDER, strict=True):
        rows = [lookup[(variant, scale)] for scale in SCALE_ORDER]
        estimate = np.asarray([float(row["estimate"]) for row in rows])
        lower = np.asarray([float(row["lower_95"]) for row in rows])
        upper = np.asarray([float(row["upper_95"]) for row in rows])
        error = np.vstack((estimate - lower, upper - estimate))
        axis.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
        axis.errorbar(
            x,
            estimate,
            yerr=error,
            color=COLORS[variant],
            marker="o",
            markersize=4,
            capsize=2.5,
        )
        axis.set_title(VARIANT_LABEL[variant])
        axis.set_xticks(x, [str(scale) for scale in SCALE_ORDER])
        axis.set_xlabel("Number of cities")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel(
        r"$\Delta g$ (percentage points)" "\nFull-F1 $-$ baseline ACO"
    )
    figure.tight_layout(w_pad=1.1)
    figure.savefig(output)
    plt.close(figure)


def plot_training_curves(
    curve_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制三种 ACO 下的训练与验证曲线。"""

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in curve_rows:
        if int(row["scale"]) == 100:
            grouped[row["variant"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["generation"]))

    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    for axis, variant in zip(axes, VARIANT_ORDER, strict=True):
        rows = grouped[variant]
        generation = np.asarray([int(row["generation"]) for row in rows])
        train_median = np.asarray(
            [float(row["train_candidate_gap_percent_median"]) for row in rows]
        )
        train_min = np.asarray(
            [float(row["train_candidate_gap_percent_min"]) for row in rows]
        )
        train_max = np.asarray(
            [float(row["train_candidate_gap_percent_max"]) for row in rows]
        )
        validation_median = np.asarray(
            [
                float(row["validation_candidate_gap_percent_median"])
                for row in rows
            ]
        )
        validation_min = np.asarray(
            [
                float(row["validation_candidate_gap_percent_min"])
                for row in rows
            ]
        )
        validation_max = np.asarray(
            [
                float(row["validation_candidate_gap_percent_max"])
                for row in rows
            ]
        )
        validation_baseline = np.asarray(
            [
                float(row["validation_baseline_gap_percent_median"])
                for row in rows
            ]
        )

        axis.fill_between(
            generation,
            train_min,
            train_max,
            color=COLORS["candidate"],
            alpha=0.12,
            linewidth=0,
        )
        axis.fill_between(
            generation,
            validation_min,
            validation_max,
            color=COLORS["validation"],
            alpha=0.10,
            linewidth=0,
        )
        axis.plot(
            generation,
            train_median,
            color=COLORS["candidate"],
            label="Train (generation-best)",
        )
        axis.plot(
            generation,
            validation_median,
            color=COLORS["validation"],
            label="Validation (same program)",
        )
        axis.plot(
            generation,
            validation_baseline,
            color=COLORS["baseline"],
            linestyle="--",
            label="Baseline ACO",
        )
        axis.set_title(VARIANT_LABEL[variant])
        axis.set_xlabel("GP generation")
        axis.set_xlim(1, 50)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel("Reference gap (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.0)
    figure.savefig(output)
    plt.close(figure)


def plot_accelerator_throughput(output: Path) -> None:
    """绘制独占 RTX 4000 Ada 正式基准的吞吐量。"""

    labels = ("CPU8", "CPU16", "GPU0", "GPU1", "Dual\nshard", "Dual\ncampaign")
    throughput = np.asarray(
        (101_776, 132_911, 1_943_957, 1_941_110, 3_165_928, 3_833_840),
        dtype=float,
    )
    colors = ("#9CA3AF", "#6B7280", "#0072B2", "#56B4E9", "#009E73", "#D55E00")
    figure, axis = plt.subplots(figsize=(3.45, 2.35))
    bars = axis.bar(np.arange(len(labels)), throughput / 1e6, color=colors, width=0.72)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel("Throughput (million tours/s)")
    axis.set_ylim(0.0, 4.35)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.8)
    for bar, value in zip(bars, throughput, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{value / 1e6:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def write_ablation_primary_rows(
    primary_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成 residual 与双树/单树的共同主表。"""

    lookup = indexed_rows(primary_rows, "variant", "partition", "contrast")
    contrasts = (
        "TR-RGP − Matched-Replace",
        "RMTGP-Full-F1 − TR-RGP",
        "RMTGP-Full-F1 − PH-RGP",
    )
    lines = ["% 自动生成；每个单元格为 estimate [95\\% CI]，单位为百分点。"]
    for variant, partition in context_order():
        values = [
            tex_ci(lookup[(variant, partition, contrast)])
            for contrast in contrasts
        ]
        lines.append(
            f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
            + " & ".join(values)
            + r" \\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_legacy_comparison_rows(
    primary_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成当前 Full-F1 与上一篇研究 Legacy-GP 的同预算对比。"""

    contrast_lookup = indexed_rows(
        primary_rows,
        "variant",
        "partition",
        "contrast",
    )
    quality_lookup = indexed_rows(
        quality_rows,
        "variant",
        "partition",
        "method",
    )
    contrast = "RMTGP-Full-F1 − Legacy-GP"
    lines = [
        "% 自动生成；Legacy-GP 与 Full-F1 均按当前实验预算重新训练。",
    ]
    for variant, partition in context_order():
        legacy = quality_lookup[(variant, partition, "legacy")]
        full = quality_lookup[(variant, partition, "rmtgp-full-f1")]
        effect = contrast_lookup[(variant, partition, contrast)]
        lines.append(
            f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
            f"{float(legacy['mean_gap_percent']):.4f} & "
            f"{float(full['mean_gap_percent']):.4f} & "
            f"{tex_ci(effect)} \\\\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_residual_rows(
    primary_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成 RQ2 的 residual 与完整替换独立表。"""

    lookup = indexed_rows(primary_rows, "variant", "partition", "contrast")
    contrast = "TR-RGP − Matched-Replace"
    lines = ["% 自动生成；负值表示 residual 接口具有更低 gap。"]
    for variant, partition in context_order():
        value = tex_ci(lookup[(variant, partition, contrast)])
        lines.append(
            f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
            f"{value} \\\\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_architecture_rows(
    primary_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成 RQ3 在 31 节点下的双树与单树独立表。"""

    lookup = indexed_rows(primary_rows, "variant", "partition", "contrast")
    contrasts = (
        "RMTGP-Full-F1 − TR-RGP",
        "RMTGP-Full-F1 − PH-RGP",
    )
    lines = ["% 自动生成；负值表示 31 节点双树具有更低 gap。"]
    for variant, partition in context_order():
        values = [
            tex_ci(lookup[(variant, partition, contrast)])
            for contrast in contrasts
        ]
        lines.append(
            f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
            + " & ".join(values)
            + r" \\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_factorial_rows(
    factorial_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成 terminal、function 与交互效应表。"""

    lookup = indexed_rows(factorial_rows, "variant", "partition", "contrast")
    contrasts = (
        "terminal_full_minus_core",
        "function_f1_minus_f0",
        "terminal_function_interaction",
    )
    lines = ["% 自动生成；负值表示相应线性 contrast 降低 gap。"]
    for variant, partition in context_order():
        values = [
            tex_ci(lookup[(variant, partition, contrast)])
            for contrast in contrasts
        ]
        lines.append(
            f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
            + " & ".join(values)
            + r" \\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_mechanism_rows(
    mechanism_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成 drop-tree 与 shuffled-pairing 表。"""

    lookup = indexed_rows(mechanism_rows, "variant", "partition", "contrast")
    contrasts = (
        "Full-F1 − drop-TR",
        "Full-F1 − drop-PH",
        "Full-F1 − shuffle-r1",
        "Full-F1 − shuffle-r2",
    )
    lines = ["% 自动生成；负值表示原始 Full-F1 配对更好。"]
    for variant, partition in context_order():
        values = [
            tex_ci(lookup[(variant, partition, contrast)])
            for contrast in contrasts
        ]
        lines.append(
            f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
            + " & ".join(values)
            + r" \\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_ood_rows(
    quality_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成 OOD 描述性质量表。"""

    lookup = indexed_rows(quality_rows, "variant", "partition", "method")
    lines = ["% 自动生成；OOD 不进行 terminal/function 因果推断。"]
    for variant in VARIANT_ORDER:
        for partition in OOD_PARTITION_ORDER:
            baseline = lookup[(variant, partition, "baseline-aco")]
            full = lookup[(variant, partition, "rmtgp-full-f1")]
            win = 100.0 * float(full["win_rate"])
            tie = 100.0 * float(full["tie_rate"])
            loss = 100.0 * float(full["loss_rate"])
            lines.append(
                f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
                f"{float(baseline['mean_gap_percent']):.4f} & "
                f"{float(full['mean_gap_percent']):.4f} & "
                f"{float(full['mean_delta_pp']):+.4f} & "
                f"{win:.1f}/{tie:.1f}/{loss:.1f} \\\\"
            )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_capacity_rows(
    capacity_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成 31/62 节点的容量、结构与交互主表。"""

    lookup = indexed_rows(capacity_rows, "variant", "partition", "formula")
    formulas = (
        "mt62-mt31",
        "mt31-tr31",
        "mt31-ph31",
        "mt62-tr62",
        "mt62-ph62",
    )
    lines = ["% 自动生成；负值表示公式前项具有更低 gap。"]
    for variant, partition in context_order():
        values = [
            tex_ci(lookup[(variant, partition, formula)])
            for formula in formulas
        ]
        lines.append(
            f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
            + " & ".join(values)
            + r" \\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_inference_overhead_rows(
    efficiency_rows: list[dict[str, str]],
    output: Path,
    *,
    comment: str = "% 自动生成；每项为 isolated warm median overhead。",
) -> None:
    """生成 Full-F1 相对原始 ACO 的孤立 warm 推理开销。"""

    lookup = indexed_rows(efficiency_rows, "variant", "partition", "method")
    lines = [comment]
    for variant in VARIANT_ORDER:
        values = [
            float(
                lookup[
                    (variant, partition, "rmtgp-full-f1")
                ]["median_overhead_percent"]
            )
            for partition in INFERENCE_PARTITION_ORDER
        ]
        lines.append(
            f"{VARIANT_LABEL[variant]} & "
            + " & ".join(f"{value:+.2f}\\%" for value in values)
            + r" \\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def write_capacity_node_rows(
    training_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """生成三种结构在 B31/B62 下的 champion 节点中位数。"""

    lookup = indexed_rows(training_rows, "variant", "method")
    methods = ("tr-rgp", "ph-rgp", "rmtgp-full-f1")
    lines = ["% 自动生成；每项为 B31 champion / B62 champion 节点中位数。"]
    for variant in VARIANT_ORDER:
        cells = []
        for method in methods:
            first = float(
                lookup[(variant, f"{method}-n31")]["median_total_nodes"]
            )
            second = float(
                lookup[(variant, f"{method}-n62")]["median_total_nodes"]
            )
            cells.append(f"{first:.0f}/{second:.0f}")
        lines.append(
            f"{VARIANT_LABEL[variant]} & " + " & ".join(cells) + r" \\"
        )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def selected_full_f1_expressions(
    training_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """返回九个 Full-F1 champion，并验证表达式字段完整。"""

    selected = [
        row for row in training_rows if row.get("method") == "rmtgp-full-f1"
    ]
    if len(selected) != 9:
        raise ValueError(f"预期 9 个 Full-F1 champions，实际 {len(selected)}")
    for row in selected:
        for field in ("transition_expression", "pheromone_expression"):
            if not row.get(field):
                raise ValueError(f"Full-F1 champion 缺少 {field}")
    return sorted(
        selected,
        key=lambda row: (
            VARIANT_ORDER.index(row["variant"]),
            int(row["gp_root_seed"]),
        ),
    )


def terminal_presence(expression: str, terminals: tuple[str, ...]) -> list[int]:
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", expression))
    return [int(terminal in tokens) for terminal in terminals]


def write_terminal_usage_rows(
    training_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """按 ACO 变体汇总 Full-F1 champion 的语法引用次数。"""

    champions = selected_full_f1_expressions(training_rows)
    lines = ["% 自动生成；计数表示三个 champion 中至少语法引用一次。"]
    for role, field, terminals in (
        ("TR", "transition_expression", TRANSITION_TERMINALS),
        ("PH", "pheromone_expression", PHEROMONE_TERMINALS),
    ):
        for terminal in terminals:
            values = []
            for variant in VARIANT_ORDER:
                count = sum(
                    terminal_presence(row[field], terminals)[
                        terminals.index(terminal)
                    ]
                    for row in champions
                    if row["variant"] == variant
                )
                values.append(count)
            lines.append(
                f"{role} & \\texttt{{{terminal}}} & "
                f"{values[0]}/3 & {values[1]}/3 & {values[2]}/3 & "
                f"{sum(values)}/9 \\\\"
            )
    lines.append(r"\bottomrule")
    write_tex_lines(output, lines)


def errorbar_values(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    estimate = np.asarray([float(row["estimate_pp"]) for row in rows])
    lower = np.asarray([float(row["lower_95"]) for row in rows])
    upper = np.asarray([float(row["upper_95"]) for row in rows])
    return estimate, np.vstack((estimate - lower, upper - estimate))


def style_forest_axis(
    axis: plt.Axes,
    *,
    title: str,
    xlabel: str,
    show_context_labels: bool = True,
) -> None:
    y = np.arange(len(context_order()))
    axis.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_yticks(
        y,
        [
            context_plot_label(variant, partition)
            for variant, partition in context_order()
        ]
        if show_context_labels
        else [],
    )
    axis.invert_yaxis()
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.5, alpha=0.8)


def contrast_series(
    rows: list[dict[str, str]],
    *,
    contrast: str | None = None,
    formula: str | None = None,
) -> list[dict[str, str]]:
    """按固定上下文顺序提取一个 contrast 或 formula。"""

    if (contrast is None) == (formula is None):
        raise ValueError("contrast 与 formula 必须且只能指定一个")
    key_field = "contrast" if contrast is not None else "formula"
    key_value = contrast if contrast is not None else formula
    lookup = indexed_rows(rows, "variant", "partition", key_field)
    return [
        lookup[(variant, partition, str(key_value))]
        for variant, partition in context_order()
    ]


def plot_ablation_primary_forest(
    primary_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制 residual 接口和双树结构的主配对区间。"""

    y = np.arange(len(context_order()))
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))

    residual = contrast_series(
        primary_rows,
        contrast="TR-RGP − Matched-Replace",
    )
    estimate, error = errorbar_values(residual)
    axes[0].errorbar(
        estimate,
        y,
        xerr=error,
        color="#0072B2",
        marker="o",
        markersize=4,
        capsize=2.5,
        linestyle="none",
    )
    style_forest_axis(
        axes[0],
        title="(a) Residual interface",
        xlabel=r"$\Delta g$: TR-RGP $-$ Matched-Replace (pp)",
    )

    for contrast, color, marker, label, offset in (
        (
            "RMTGP-Full-F1 − TR-RGP",
            "#009E73",
            "o",
            "Dual − TR",
            -0.11,
        ),
        (
            "RMTGP-Full-F1 − PH-RGP",
            "#D55E00",
            "s",
            "Dual − PH",
            0.11,
        ),
    ):
        rows = contrast_series(primary_rows, contrast=contrast)
        estimate, error = errorbar_values(rows)
        axes[1].errorbar(
            estimate,
            y + offset,
            xerr=error,
            color=color,
            marker=marker,
            markersize=3.8,
            capsize=2.3,
            linestyle="none",
            label=label,
        )
    style_forest_axis(
        axes[1],
        title="(b) Dual versus independently trained single tree",
        xlabel=r"$\Delta g$: Dual $-$ single tree (pp)",
        show_context_labels=False,
    )
    axes[1].legend(loc="best", frameon=False)
    figure.tight_layout(w_pad=1.0)
    figure.savefig(output)
    plt.close(figure)


def plot_legacy_comparison_forest(
    primary_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制 Full-F1 与同预算 Legacy-GP 的配对差距区间。"""

    rows = contrast_series(
        primary_rows,
        contrast="RMTGP-Full-F1 − Legacy-GP",
    )
    estimate, error = errorbar_values(rows)
    y = np.arange(len(context_order()))
    figure, axis = plt.subplots(figsize=(3.45, 2.65))
    axis.errorbar(
        estimate,
        y,
        xerr=error,
        color="#7B2CBF",
        marker="o",
        markersize=4,
        capsize=2.5,
        linestyle="none",
    )
    style_forest_axis(
        axis,
        title="Full-F1 versus retrained Legacy-GP",
        xlabel=r"$\Delta g$: Full-F1 $-$ Legacy-GP (pp)",
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_residual_forest(
    primary_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制 RQ2 residual 接口的独立森林图。"""

    y = np.arange(len(context_order()))
    rows = contrast_series(
        primary_rows,
        contrast="TR-RGP − Matched-Replace",
    )
    estimate, error = errorbar_values(rows)
    figure, axis = plt.subplots(figsize=(3.45, 2.65))
    axis.errorbar(
        estimate,
        y,
        xerr=error,
        color="#0072B2",
        marker="o",
        markersize=4,
        capsize=2.5,
        linestyle="none",
    )
    style_forest_axis(
        axis,
        title="Residual interface",
        xlabel=r"$\Delta g$: TR-RGP $-$ Matched-Replace (pp)",
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_architecture_forest(
    primary_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制 RQ3 在 31 节点下的双树与单树独立森林图。"""

    y = np.arange(len(context_order()))
    figure, axis = plt.subplots(figsize=(3.45, 2.65))
    for contrast, color, marker, label, offset in (
        (
            "RMTGP-Full-F1 − TR-RGP",
            "#009E73",
            "o",
            "Dual − TR",
            -0.11,
        ),
        (
            "RMTGP-Full-F1 − PH-RGP",
            "#D55E00",
            "s",
            "Dual − PH",
            0.11,
        ),
    ):
        rows = contrast_series(primary_rows, contrast=contrast)
        estimate, error = errorbar_values(rows)
        axis.errorbar(
            estimate,
            y + offset,
            xerr=error,
            color=color,
            marker=marker,
            markersize=3.8,
            capsize=2.3,
            linestyle="none",
            label=label,
        )
    style_forest_axis(
        axis,
        title="Dual versus independently trained single tree",
        xlabel=r"$\Delta g$: Dual $-$ single tree (pp)",
    )
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_factorial_forest(
    factorial_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制 Core/Full × F0/F1 的三个预注册线性效应。"""

    y = np.arange(len(context_order()))
    definitions = (
        (
            "terminal_full_minus_core",
            "Terminal effect",
            r"Full $-$ Core (pp)",
            "#0072B2",
        ),
        (
            "function_f1_minus_f0",
            "Function effect",
            r"F1 $-$ F0 (pp)",
            "#D55E00",
        ),
        (
            "terminal_function_interaction",
            "Interaction",
            "Difference-in-differences (pp)",
            "#009E73",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 3.0))
    for index, (contrast, title, xlabel, color) in enumerate(definitions):
        rows = contrast_series(factorial_rows, contrast=contrast)
        estimate, error = errorbar_values(rows)
        axes[index].errorbar(
            estimate,
            y,
            xerr=error,
            color=color,
            marker="o",
            markersize=3.8,
            capsize=2.2,
            linestyle="none",
        )
        style_forest_axis(
            axes[index],
            title=title,
            xlabel=xlabel,
            show_context_labels=index == 0,
        )
    figure.tight_layout(w_pad=0.7)
    figure.savefig(output)
    plt.close(figure)


def plot_mechanism_forest(
    mechanism_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制组件删除与跨 run 打乱配对的 post-hoc 结果。"""

    y = np.arange(len(context_order()))
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    panels = (
        (
            axes[0],
            (
                ("Full-F1 − drop-TR", "#0072B2", "o", "Full − drop-TR"),
                ("Full-F1 − drop-PH", "#D55E00", "s", "Full − drop-PH"),
            ),
            "(a) Component deletion",
            r"$\Delta g$: Full $-$ dropped component (pp)",
        ),
        (
            axes[1],
            (
                ("Full-F1 − shuffle-r1", "#009E73", "o", "Shuffle r1"),
                ("Full-F1 − shuffle-r2", "#CC79A7", "s", "Shuffle r2"),
            ),
            "(b) Across-run pairing shuffle",
            r"$\Delta g$: matched Full $-$ shuffled pair (pp)",
        ),
    )
    for panel_index, (axis, definitions, title, xlabel) in enumerate(panels):
        for offset, (contrast, color, marker, label) in zip(
            (-0.11, 0.11),
            definitions,
            strict=True,
        ):
            rows = contrast_series(mechanism_rows, contrast=contrast)
            estimate, error = errorbar_values(rows)
            axis.errorbar(
                estimate,
                y + offset,
                xerr=error,
                color=color,
                marker=marker,
                markersize=3.8,
                capsize=2.2,
                linestyle="none",
                label=label,
            )
        style_forest_axis(
            axis,
            title=title,
            xlabel=xlabel,
            show_context_labels=panel_index == 0,
        )
        axis.legend(loc="best", frameon=False)
    figure.tight_layout(w_pad=1.0)
    figure.savefig(output)
    plt.close(figure)


def plot_ood_delta(
    quality_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制 OOD 上 Full-F1 相对 ACO 的描述性平均差值。"""

    lookup = indexed_rows(quality_rows, "variant", "partition", "method")
    x = np.arange(len(OOD_PARTITION_ORDER))
    width = 0.23
    figure, axis = plt.subplots(figsize=(3.45, 2.55))
    for index, variant in enumerate(VARIANT_ORDER):
        values = [
            float(
                lookup[
                    (variant, partition, "rmtgp-full-f1")
                ]["mean_delta_pp"]
            )
            for partition in OOD_PARTITION_ORDER
        ]
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            color=COLORS[variant],
            label=VARIANT_LABEL[variant],
        )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set_xticks(
        x,
        ["Cluster\nTSP500", "Gaussian\nTSP500", "TSPLIB\n≤500"],
    )
    axis.set_ylabel(r"Mean $\Delta g$ (pp), Full-F1 $-$ ACO")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.8)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_terminal_usage(
    training_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制九个 Full-F1 champion 的 terminal 二值语法使用矩阵。"""

    champions = selected_full_f1_expressions(training_rows)
    column_labels = [
        f"{VARIANT_LABEL[row['variant']]}-{index + 1}"
        for variant in VARIANT_ORDER
        for index, row in enumerate(
            [item for item in champions if item["variant"] == variant]
        )
    ]
    figure, axes = plt.subplots(2, 1, figsize=(5.8, 4.0))
    for axis, field, terminals, title in (
        (
            axes[0],
            "transition_expression",
            TRANSITION_TERMINALS,
            "(a) Transition tree terminals",
        ),
        (
            axes[1],
            "pheromone_expression",
            PHEROMONE_TERMINALS,
            "(b) Pheromone tree terminals",
        ),
    ):
        matrix = np.asarray(
            [
                terminal_presence(row[field], terminals)
                for row in champions
            ],
            dtype=float,
        ).T
        axis.imshow(
            matrix,
            cmap=matplotlib.colors.ListedColormap(("#F3F4F6", "#0072B2")),
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_yticks(np.arange(len(terminals)), terminals)
        axis.set_xticks(np.arange(len(champions)), column_labels)
        axis.set_title(title, loc="left")
        axis.tick_params(axis="both", length=0)
        for x_index in range(len(champions) + 1):
            axis.axvline(x_index - 0.5, color="white", linewidth=0.8)
        for y_index in range(len(terminals) + 1):
            axis.axhline(y_index - 0.5, color="white", linewidth=0.8)
    figure.tight_layout(h_pad=1.0)
    figure.savefig(output)
    plt.close(figure)


def plot_ablation_validation_curves(
    curve_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制接口/结构方法与 factorial 方法的 validation 曲线。"""

    groups = (
        (
            "Interface and architecture",
            (
                "legacy",
                "matched-replace",
                "tr-rgp",
                "ph-rgp",
                "rmtgp-full-f1",
            ),
        ),
        (
            "Terminal/function factorial",
            (
                "rmtgp-core-f0",
                "rmtgp-core-f1",
                "rmtgp-full-f0",
                "rmtgp-full-f1",
            ),
        ),
    )
    palette = {
        method: plt.get_cmap("tab10")(index)
        for index, method in enumerate(ABLATION_METHOD_LABEL)
    }
    lookup: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in curve_rows:
        lookup[(row["variant"], row["method"])].append(row)
    for rows in lookup.values():
        rows.sort(key=lambda row: int(row["generation"]))

    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.25), sharex=True)
    for row_index, (group_title, methods) in enumerate(groups):
        for column_index, variant in enumerate(VARIANT_ORDER):
            axis = axes[row_index, column_index]
            for method in methods:
                rows = lookup[(variant, method)]
                generation = np.asarray(
                    [int(row["generation"]) for row in rows]
                )
                validation = np.asarray(
                    [float(row["validation_delta_pp_median"]) for row in rows]
                )
                axis.plot(
                    generation,
                    validation,
                    color=palette[method],
                    label=ABLATION_METHOD_LABEL[method],
                )
            axis.axhline(0.0, color="#444444", linewidth=0.7, linestyle="--")
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.8)
            if row_index == 0:
                axis.set_title(VARIANT_LABEL[variant])
            if column_index == 0:
                axis.set_ylabel(f"{group_title}\nValidation Δ (pp)")
            if row_index == len(groups) - 1:
                axis.set_xlabel("GP generation")
            axis.set_xlim(1, 50)
    handles_top, labels_top = axes[0, -1].get_legend_handles_labels()
    handles_bottom, labels_bottom = axes[1, -1].get_legend_handles_labels()
    unique_legend: dict[str, Any] = {}
    for handle, label in zip(
        handles_top + handles_bottom,
        labels_top + labels_bottom,
        strict=True,
    ):
        unique_legend.setdefault(label, handle)
    figure.legend(
        list(unique_legend.values()),
        list(unique_legend),
        ncol=4,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), h_pad=0.8, w_pad=0.7)
    figure.savefig(output)
    plt.close(figure)


def plot_capacity_forest(
    capacity_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """绘制节点容量、固定容量结构效应及其交互。"""

    y = np.arange(len(context_order()))
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 3.25))
    panels = (
        (
            axes[0],
            (
                ("tr62-tr31", "#0072B2", "o", "TR"),
                ("ph62-ph31", "#D55E00", "s", "PH"),
                ("mt62-mt31", "#009E73", "^", "Dual"),
            ),
            "Capacity effect",
            r"B62 $-$ B31 (pp)",
        ),
        (
            axes[1],
            (
                ("mt31-tr31", "#0072B2", "o", "B31: Dual−TR"),
                ("mt31-ph31", "#56B4E9", "s", "B31: Dual−PH"),
                ("mt62-tr62", "#D55E00", "^", "B62: Dual−TR"),
                ("mt62-ph62", "#E69F00", "D", "B62: Dual−PH"),
            ),
            "Architecture at fixed capacity",
            r"Dual $-$ single tree (pp)",
        ),
        (
            axes[2],
            (
                ("did-mt-vs-tr", "#009E73", "o", "vs TR"),
                ("did-mt-vs-ph", "#CC79A7", "s", "vs PH"),
            ),
            "Capacity × architecture",
            "Difference-in-differences (pp)",
        ),
    )
    for panel_index, (axis, definitions, title, xlabel) in enumerate(panels):
        offsets = np.linspace(-0.18, 0.18, len(definitions))
        for offset, (formula, color, marker, label) in zip(
            offsets,
            definitions,
            strict=True,
        ):
            rows = contrast_series(capacity_rows, formula=formula)
            estimate, error = errorbar_values(rows)
            axis.errorbar(
                estimate,
                y + offset,
                xerr=error,
                color=color,
                marker=marker,
                markersize=3.4,
                capsize=2.0,
                linestyle="none",
                label=label,
            )
        style_forest_axis(
            axis,
            title=title,
            xlabel=xlabel,
            show_context_labels=panel_index == 0,
        )
        axis.legend(loc="best", frameon=False, fontsize=6.2)
    figure.tight_layout(w_pad=0.6)
    figure.savefig(output)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="MTGP_ACO 仓库根目录",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验证全部冻结数据源，不生成论文资产",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    report_root = repo_root / "runs" / "tsp100-gpu0-3seed" / "report"
    ablation_root = repo_root / "runs" / "tsp100-ablation-gpu1-3seed"
    capacity_root = repo_root / "runs" / "tsp100-capacity-sensitivity-3seed"
    output_root = Path(__file__).resolve().parent

    main_summary = require_completed_study(
        repo_root / "runs" / "tsp100-gpu0-3seed",
        study_id="tsp100-gpu0-3seed-v1",
        total_tasks=40,
        summary_name="study_summary.json",
        summary_schema=1,
    )

    ablation_summary = require_completed_study(
        ablation_root,
        study_id="tsp100-ablation-gpu1-3seed-v2-selected-champions",
        total_tasks=112,
        summary_name="ablation_summary.json",
    )
    capacity_summary = require_completed_study(
        capacity_root,
        study_id="tsp100-capacity-sensitivity-3seed-v2-core-partitions",
        total_tasks=46,
        summary_name="capacity_summary.json",
    )
    extended_data = load_extended_sources(repo_root)

    test_rows = read_csv(report_root / "test_summary.csv")
    interval_rows = read_csv(report_root / "bootstrap_intervals.csv")
    training_rows = read_csv(report_root / "training_runs.csv")
    curve_rows = read_csv(report_root / "training_validation_curves_aggregate.csv")
    ablation_report_root = ablation_root / "report"
    ablation_primary_rows = read_csv(
        ablation_report_root / "primary_contrasts.csv"
    )
    factorial_rows = read_csv(
        ablation_report_root / "factorial_contrasts.csv"
    )
    mechanism_rows = read_csv(
        ablation_report_root / "mechanism_contrasts.csv"
    )
    ablation_quality_rows = read_csv(
        ablation_report_root / "quality_summary.csv"
    )
    ablation_training_rows = read_csv(
        ablation_report_root / "training_runs.csv"
    )
    ablation_curve_rows = read_csv(
        ablation_report_root / "training_validation_curves_aggregate.csv"
    )
    ablation_efficiency_rows = read_csv(
        ablation_report_root / "efficiency_summary.csv"
    )
    capacity_rows = read_csv(
        capacity_root / "report" / "capacity_contrasts.csv"
    )
    capacity_training_rows = read_csv(
        capacity_root / "report" / "training_summary.csv"
    )

    if args.verify_only:
        print(
            "Verified completed paper sources: "
            "main=40/40, ablation=112/112, capacity=46/46, "
            "instance-budget=9 runs, signal-audit=3 variants, "
            "TSP100-fitness-controls=18/18 runs, racing-audit=3 variants, "
            "TSP500-LS=27/27 final shards."
        )
        return

    configure_plot_style()
    write_main_result_rows(
        test_rows,
        interval_rows,
        output_root / "main_results_rows.tex",
    )
    write_training_time_rows(
        training_rows,
        output_root / "training_time_rows.tex",
    )
    plot_main_gap_delta(
        interval_rows,
        output_root / "main_gap_delta.pdf",
    )
    plot_training_curves(
        curve_rows,
        output_root / "training_validation_curves.pdf",
    )
    plot_accelerator_throughput(
        output_root / "accelerator_throughput.pdf",
    )
    write_ablation_primary_rows(
        ablation_primary_rows,
        output_root / "ablation_primary_rows.tex",
    )
    write_legacy_comparison_rows(
        ablation_primary_rows,
        ablation_quality_rows,
        output_root / "legacy_comparison_rows.tex",
    )
    write_residual_rows(
        ablation_primary_rows,
        output_root / "residual_effect_rows.tex",
    )
    write_architecture_rows(
        ablation_primary_rows,
        output_root / "architecture_effect_rows.tex",
    )
    write_factorial_rows(
        factorial_rows,
        output_root / "factorial_effect_rows.tex",
    )
    write_mechanism_rows(
        mechanism_rows,
        output_root / "mechanism_rows.tex",
    )
    write_ood_rows(
        ablation_quality_rows,
        output_root / "ood_results_rows.tex",
    )
    write_capacity_rows(
        capacity_rows,
        output_root / "capacity_rows.tex",
    )
    write_terminal_usage_rows(
        ablation_training_rows,
        output_root / "terminal_usage_rows.tex",
    )
    write_inference_overhead_rows(
        ablation_efficiency_rows,
        output_root / "inference_overhead_rows.tex",
    )
    write_inference_overhead_rows(
        ablation_efficiency_rows,
        output_root / "inference_overhead_rows_en.tex",
        comment="% Generated; each entry is isolated warm median overhead.",
    )
    write_capacity_node_rows(
        capacity_training_rows,
        output_root / "capacity_node_rows.tex",
    )
    plot_ablation_primary_forest(
        ablation_primary_rows,
        output_root / "ablation_primary_forest.pdf",
    )
    plot_legacy_comparison_forest(
        ablation_primary_rows,
        output_root / "legacy_comparison.pdf",
    )
    plot_residual_forest(
        ablation_primary_rows,
        output_root / "residual_effects.pdf",
    )
    plot_architecture_forest(
        ablation_primary_rows,
        output_root / "architecture_effects.pdf",
    )
    plot_factorial_forest(
        factorial_rows,
        output_root / "factorial_effects.pdf",
    )
    plot_mechanism_forest(
        mechanism_rows,
        output_root / "mechanism_effects.pdf",
    )
    plot_ood_delta(
        ablation_quality_rows,
        output_root / "ood_delta.pdf",
    )
    plot_terminal_usage(
        ablation_training_rows,
        output_root / "terminal_usage_heatmap.pdf",
    )
    plot_ablation_validation_curves(
        ablation_curve_rows,
        output_root / "ablation_validation_curves.pdf",
    )
    plot_capacity_forest(
        capacity_rows,
        output_root / "capacity_effects.pdf",
    )
    extended_provenance = build_extended_assets(
        repo_root,
        output_root,
        extended_data,
    )
    provenance = {
        "schema_version": 2,
        "main": {
            "study_id": main_summary["study"]["study_id"],
            "generated_at": main_summary["generated_at"],
            "git": main_summary["git"],
            "report": str(report_root.relative_to(repo_root)),
        },
        "ablation": {
            "study_id": ablation_summary["study"]["study_id"],
            "generated_at": ablation_summary["generated_at"],
            "git": ablation_summary["git"],
        },
        "capacity": {
            "study_id": capacity_summary["study"]["study_id"],
            "generated_at": capacity_summary["generated_at"],
            "git": capacity_summary["git"],
        },
        "extended": extended_provenance,
    }
    (output_root / "paper_result_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "generated_from.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
