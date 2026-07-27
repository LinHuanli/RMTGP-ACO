"""从冻结的实验报告生成论文图表。

该脚本只读取完整的主实验报告。消融和容量实验在正式报告生成前不会被读取，
因此预备稿不会混入不完整任务的中间结果。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER = ("as", "acs", "mmas")
VARIANT_LABEL = {"as": "AS", "acs": "ACS", "mmas": "MMAS"}
SCALE_ORDER = (50, 100, 500, 1000)
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
            label="Training",
        )
        axis.plot(
            generation,
            validation_median,
            color=COLORS["validation"],
            label="Validation",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="MTGP_ACO 仓库根目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    report_root = repo_root / "runs" / "tsp100-gpu0-3seed" / "report"
    output_root = Path(__file__).resolve().parent

    test_rows = read_csv(report_root / "test_summary.csv")
    interval_rows = read_csv(report_root / "bootstrap_intervals.csv")
    training_rows = read_csv(report_root / "training_runs.csv")
    curve_rows = read_csv(report_root / "training_validation_curves_aggregate.csv")

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


if __name__ == "__main__":
    main()
