"""生成训练预算、局部搜索、学习信号和 TSP500 LS-aware 论文资产。

本模块只读取已经冻结完成的机器可读结果。它由主资产脚本调用，不提供第二套
论文统计口径。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("as", "acs", "mmas")
VARIANT_LABEL = {"as": "AS", "acs": "ACS", "mmas": "MMAS"}
PARTITIONS = ("tsp500_uniform", "tsp500_cluster", "tsp500_gaussian")
PARTITION_LABEL = {
    "tsp50_uniform": "TSP50-U",
    "tsp100_uniform": "TSP100-U",
    "tsp500_uniform": "Uniform",
    "tsp1000_uniform": "TSP1000-U",
    "tsp500_cluster": "Cluster",
    "tsp500_gaussian": "Gaussian",
}
COLORS = {
    "aco2": "#6B7280",
    "rmtgp": "#0072B2",
    "aco3": "#D55E00",
    "train": "#0072B2",
    "validation": "#D55E00",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少论文数据源: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: 顶层必须是 JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少论文数据源: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_tex(path: Path, lines: list[str]) -> None:
    """写出可直接置于 booktabs 表格中的行，并在同一输入流内闭合表底线。"""

    output = [*lines, r"\bottomrule"]
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _tex_bool(value: str | bool) -> str:
    flag = value if isinstance(value, bool) else value.lower() == "true"
    return "Yes" if flag else "No"


def load_extended_sources(repo_root: Path) -> dict[str, Any]:
    """读取并验证所有后续实验；此函数不写文件。"""

    instance_root = repo_root / "runs" / "tsp100-instance-budget-single-gpu"
    instance_state_path = instance_root / "campaign_state.json"
    instance_summary_path = instance_root / "parallel-test" / "summary.json"
    instance_state = _read_json(instance_state_path)
    instance_summary = _read_json(instance_summary_path)
    _require(instance_state.get("status") == "completed", "instance-budget campaign 未完成")
    _require(len(instance_state.get("completed", ())) == 9, "instance-budget 应有 9 次训练")
    _require(not instance_state.get("failures"), "instance-budget campaign 含失败任务")
    _require(instance_summary.get("schema_version") == 1, "instance-budget schema 不匹配")
    _require(len(instance_summary.get("partitions", ())) == 12, "instance-budget 汇总不完整")

    local_root = repo_root / "runs" / "tsp100-local-search-3seed" / "final-test"
    local_summary_path = local_root / "summary.json"
    local_quality_path = local_root / "quality_summary.csv"
    local_runtime_path = local_root / "runtime_summary.csv"
    local_paired_path = local_root / "paired_vs_aco3.csv"
    local_summary = _read_json(local_summary_path)
    local_quality = _read_csv(local_quality_path)
    local_runtime = _read_csv(local_runtime_path)
    local_paired = _read_csv(local_paired_path)
    _require(local_summary.get("status") == "completed", "初始 local-search final test 未完成")
    _require(
        local_summary.get("population_individuals_tested") is False,
        "初始 LS 测试口径不是最终候选",
    )
    _require(len(local_quality) == 24, "初始 LS quality_summary 应有 24 行")

    signal_root = repo_root / "runs" / "tsp100-2opt-signal-v2"
    audit_state_path = signal_root / "audit_state.json"
    pilot_state_path = signal_root / "pilot_state.json"
    audit_state = _read_json(audit_state_path)
    pilot_state = _read_json(pilot_state_path)
    _require(audit_state.get("status") == "completed", "2-opt signal audit 未完成")
    _require(len(audit_state.get("completed", ())) == 3, "2-opt signal audit 应有 3 个 variant")
    _require(not audit_state.get("failures"), "2-opt signal audit 含失败任务")
    _require(pilot_state.get("status") == "completed", "2-opt signal pilot 未完成")
    audit_paths = [
        signal_root / "audit" / variant / "summary.json" for variant in VARIANTS
    ]
    audit_summaries = {
        variant: _read_json(path) for variant, path in zip(VARIANTS, audit_paths)
    }
    for variant, summary in audit_summaries.items():
        _require(summary.get("variant") == variant, f"signal audit variant 不匹配: {variant}")
        _require(
            summary.get("horizons") == [500, 2000, 5000],
            f"signal audit horizon 不完整: {variant}",
        )

    fitness_control_specs = {
        "anytime_final": (
            repo_root / "runs" / "tsp100-2opt-anytime" / "formal",
            "paired_final_anytime_ucb",
        ),
        "basin_final": (
            signal_root / "formal",
            "paired_combined_ucb",
        ),
    }
    fitness_control_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    fitness_control_paths: list[Path] = []
    for condition, (formal_root, expected_fitness) in fitness_control_specs.items():
        state_path = formal_root / "formal_state.json"
        state = _read_json(state_path)
        _require(
            state.get("status") == "completed"
            and len(state.get("completed", ())) == 9
            and not state.get("failures"),
            f"TSP100 fitness control 不完整: {condition}",
        )
        fitness_control_paths.append(state_path)
        by_variant: dict[str, list[dict[str, str]]] = {}
        for variant in VARIANTS:
            rows = []
            for seed in (71001, 71002, 71003):
                run_root = formal_root / "train" / variant / f"seed-{seed}"
                config_path = run_root / "config.yaml"
                summary_path = run_root / "validation_summary.csv"
                _require(
                    f"fitness_mode: {expected_fitness}"
                    in config_path.read_text(encoding="utf-8"),
                    f"{condition}-{variant}-{seed}: fitness 配置不匹配",
                )
                summary_rows = _read_csv(summary_path)
                _require(
                    len(summary_rows) == 1
                    and int(summary_rows[0]["instances"]) == 512
                    and int(summary_rows[0]["observations"]) == 2560,
                    f"{condition}-{variant}-{seed}: validation 汇总不完整",
                )
                rows.append(summary_rows[0])
                fitness_control_paths.extend((config_path, summary_path))
            by_variant[variant] = rows
        fitness_control_rows[condition] = by_variant

    racing_root = repo_root / "runs" / "tsp500-2opt-racing"
    racing_state_path = racing_root / "audit_state.json"
    racing_state = _read_json(racing_state_path)
    _require(
        racing_state.get("status") == "completed"
        and len(racing_state.get("completed", ())) == 3
        and not racing_state.get("failures"),
        "TSP500 racing audit 不完整",
    )
    racing_paths = [
        racing_root / "audit" / variant / "summary.json"
        for variant in VARIANTS
    ]
    racing_summaries = {
        variant: _read_json(path)
        for variant, path in zip(VARIANTS, racing_paths)
    }
    for variant, summary in racing_summaries.items():
        _require(summary.get("variant") == variant, f"racing variant 不匹配: {variant}")
        _require(
            summary.get("formal_training_allowed") is False
            and all(not row.get("gate_passed") for row in summary.get("reports", ())),
            f"racing gate 状态不匹配: {variant}",
        )

    latest_root = repo_root / "runs" / "tsp500-2opt-ls-v2"
    latest_manifest_path = latest_root / "final-test" / "manifest.json"
    latest_summary_path = latest_root / "final-test" / "summary.json"
    latest_main_path = latest_root / "final-test" / "main_results.csv"
    latest_runtime_path = latest_root / "final-test" / "runtime_results.csv"
    latest_manifest = _read_json(latest_manifest_path)
    latest_summary = _read_json(latest_summary_path)
    latest_main = _read_csv(latest_main_path)
    latest_runtime = _read_csv(latest_runtime_path)
    _require(
        latest_manifest.get("status") == "completed",
        "TSP500 LS-aware final test 未完成",
    )
    progress = latest_manifest.get("progress", {})
    _require(
        len(progress.get("completed", ())) == 27,
        "TSP500 LS-aware final test 应有 27 个 shard",
    )
    _require(not progress.get("failures"), "TSP500 LS-aware final test 含失败 shard")
    protocol = latest_summary.get("protocol", {})
    _require(
        protocol.get("candidate_policy") == "selected_candidate_only",
        "TSP500 测试不是 selected-candidate-only",
    )
    _require(
        protocol.get("aco_iterations") == 5000,
        "TSP500 final test 不是 5000 iterations",
    )
    _require(
        len(latest_main) == 9 and len(latest_runtime) == 9,
        "TSP500 主结果应各有 9 行",
    )

    curves: dict[tuple[str, int], list[dict[str, str]]] = {}
    training_metrics: dict[tuple[str, int], list[dict[str, Any]]] = {}
    expressions: dict[tuple[str, int], dict[str, str]] = {}
    curve_paths: list[Path] = []
    training_metric_paths: list[Path] = []
    expression_paths: list[Path] = []
    for variant in VARIANTS:
        for seed in (81001, 81002, 81003):
            run_root = latest_root / "formal" / "train" / variant / f"seed-{seed}"
            curve_path = run_root / "training_validation_curve.csv"
            metrics_path = run_root / "training_metrics.json"
            expression_path = run_root / "selected_candidate_expression.txt"
            curve_rows = _read_csv(curve_path)
            metric_rows = json.loads(metrics_path.read_text(encoding="utf-8"))
            _require(len(curve_rows) == 50, f"{variant}-{seed}: 训练曲线不是 50 代")
            _require(
                isinstance(metric_rows, list)
                and len(metric_rows) == 50
                and [int(row["generation"]) for row in metric_rows]
                == list(range(1, 51)),
                f"{variant}-{seed}: timing metrics 不是完整 50 代",
            )
            raw_expression = expression_path.read_text(encoding="utf-8").splitlines()
            expression = {}
            for line in raw_expression:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    expression[key] = value
            _require(
                "transition" in expression and "pheromone" in expression,
                f"{expression_path}: 表达式缺失",
            )
            curves[(variant, seed)] = curve_rows
            training_metrics[(variant, seed)] = metric_rows
            expressions[(variant, seed)] = expression
            curve_paths.append(curve_path)
            training_metric_paths.append(metrics_path)
            expression_paths.append(expression_path)

    benchmark_path = (
        repo_root
        / "experiments"
        / "tsp100_local_search_3seed"
        / "benchmark_summary.json"
    )
    benchmark = _read_json(benchmark_path)
    _require(
        benchmark.get("hardware", {}).get("device")
        == "NVIDIA RTX PRO 5000 Blackwell",
        "Blackwell benchmark 硬件不匹配",
    )
    instance_budget_note_path = (
        repo_root
        / "docs"
        / "experiments"
        / "tsp100_instance_budget_single_gpu_3seed.md"
    )
    ls_v2_readme_path = (
        repo_root / "experiments" / "tsp500_2opt_ls_v2" / "README.md"
    )
    _require(instance_budget_note_path.is_file(), f"缺少 {instance_budget_note_path}")
    _require(ls_v2_readme_path.is_file(), f"缺少 {ls_v2_readme_path}")

    source_paths = [
        instance_state_path,
        instance_summary_path,
        local_summary_path,
        local_quality_path,
        local_runtime_path,
        local_paired_path,
        audit_state_path,
        pilot_state_path,
        *audit_paths,
        *fitness_control_paths,
        racing_state_path,
        *racing_paths,
        latest_manifest_path,
        latest_summary_path,
        latest_main_path,
        latest_runtime_path,
        *curve_paths,
        *training_metric_paths,
        *expression_paths,
        benchmark_path,
        instance_budget_note_path,
        ls_v2_readme_path,
    ]
    for path in source_paths:
        _require(
            ".nfs-checkpoint" not in str(path),
            f"禁止使用 NFS checkpoint: {path}",
        )

    return {
        "instance_state": instance_state,
        "instance_summary": instance_summary,
        "local_summary": local_summary,
        "local_quality": local_quality,
        "local_runtime": local_runtime,
        "local_paired": local_paired,
        "audit_summaries": audit_summaries,
        "fitness_control_rows": fitness_control_rows,
        "racing_summaries": racing_summaries,
        "latest_summary": latest_summary,
        "latest_manifest": latest_manifest,
        "latest_main": latest_main,
        "latest_runtime": latest_runtime,
        "curves": curves,
        "training_metrics": training_metrics,
        "expressions": expressions,
        "benchmark": benchmark,
        "source_paths": source_paths,
    }


def _write_instance_budget_assets(data: dict[str, Any], output_root: Path) -> None:
    summary = data["instance_summary"]
    rows = summary["partitions"]
    lookup = {
        (row["partition"], int(row["budget"])): row
        for row in rows
    }
    order = (
        "tsp50_uniform",
        "tsp100_uniform",
        "tsp500_uniform",
        "tsp1000_uniform",
    )
    instance_labels = {
        "tsp50_uniform": "TSP50-U",
        "tsp100_uniform": "TSP100-U",
        "tsp500_uniform": "TSP500-U",
        "tsp1000_uniform": "TSP1000-U",
    }
    lines = ["% Generated from the completed ACS instance-budget study."]
    for partition in order:
        values = [lookup[(partition, budget)] for budget in (32, 64, 128)]
        baseline = values[0]["mean_gap_percent"] - values[0]["mean_delta_pp"]
        cells = [
            f"{row['mean_gap_percent']:.4f} / {row['mean_delta_pp']:+.4f}"
            for row in values
        ]
        lines.append(
            f"{instance_labels[partition]} & {baseline:.4f} & "
            + " & ".join(cells)
            + r" \\"
        )
    _write_tex(output_root / "instance_budget_rows.tex", lines)

    # Frozen summary of the nine completed training runs. The source report is
    # docs/experiments/tsp100_instance_budget_single_gpu_3seed.md.
    timing = {
        32: (7.63, 1.48, 7.71, 1.24, 6.44, 0.61),
        64: (10.43, 2.68, 10.04, 2.26, 7.57, 0.32),
        128: (17.21, 2.56, 15.69, 2.15, 9.36, 0.08),
    }
    time_lines = [
        "% Frozen summary of nine completed single-GPU runs; mean plus/minus sample SD."
    ]
    for budget, values in timing.items():
        (
            generation,
            generation_sd,
            end_to_end,
            end_to_end_sd,
            throughput,
            throughput_sd,
        ) = values
        time_lines.append(
            f"{budget} & {generation:.2f} $\\pm$ {generation_sd:.2f} & "
            f"{end_to_end:.2f} $\\pm$ {end_to_end_sd:.2f} & "
            f"{throughput:.2f} $\\pm$ {throughput_sd:.2f} \\\\"
        )
    _write_tex(output_root / "instance_budget_time_rows.tex", time_lines)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
    for partition, marker in zip(order, ("o", "s", "^", "D")):
        y = [
            lookup[(partition, budget)]["mean_delta_pp"]
            for budget in (32, 64, 128)
        ]
        axes[0].plot(
            (32, 64, 128),
            y,
            marker=marker,
            label=PARTITION_LABEL[partition],
        )
    axes[0].axhline(0.0, color="#333333", linewidth=0.7)
    axes[0].set_xlabel("Training instances per generation")
    axes[0].set_ylabel(r"Mean $\Delta g$ (pp)")
    axes[0].set_xticks((32, 64, 128))
    axes[0].legend(frameon=False, ncol=2, fontsize=6.3)
    axes[0].grid(axis="y", alpha=0.2)

    budgets = np.array((32, 64, 128))
    axes[1].plot(
        budgets,
        [timing[int(budget)][2] for budget in budgets],
        "o-",
        color="#0072B2",
        label="End-to-end time",
    )
    axes[1].set_xlabel("Training instances per generation")
    axes[1].set_ylabel("End-to-end time (min)", color="#0072B2")
    axes[1].tick_params(axis="y", colors="#0072B2")
    twin = axes[1].twinx()
    twin.plot(
        budgets,
        [timing[int(budget)][4] for budget in budgets],
        "s--",
        color="#D55E00",
        label="Throughput",
    )
    twin.set_ylabel("Throughput (M tours/s)", color="#D55E00")
    twin.tick_params(axis="y", colors="#D55E00")
    axes[1].set_xticks(budgets)
    axes[1].grid(axis="x", alpha=0.2)
    figure.tight_layout(w_pad=1.2)
    figure.savefig(output_root / "instance_budget_tradeoff.pdf")
    plt.close(figure)


def _write_initial_ls_assets(data: dict[str, Any], output_root: Path) -> None:
    rows = data["local_quality"]
    methods = (
        "aco-2opt",
        "rmtgp-nols-2opt",
        "rmtgp-ls-2opt",
        "aco-3opt",
    )
    lookup = {
        (row["variant"], row["partition"], row["method"]): row
        for row in rows
    }
    lines = ["% Generated from the completed initial local-search study."]
    for variant in VARIANTS:
        values = [
            float(
                lookup[(variant, "tsp500_uniform", method)][
                    "mean_gap_percent"
                ]
            )
            for method in methods
        ]
        lines.append(
            f"{VARIANT_LABEL[variant]} & "
            + " & ".join(f"{value:.4f}" for value in values)
            + r" \\"
        )
    _write_tex(output_root / "initial_ls_tsp500_rows.tex", lines)

    all_lines = [
        "% Full initial local-search results; final candidates only."
    ]
    for variant in VARIANTS:
        for partition in ("tsp100_uniform", "tsp500_uniform"):
            values = [
                float(
                    lookup[(variant, partition, method)]["mean_gap_percent"]
                )
                for method in methods
            ]
            all_lines.append(
                f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
                + " & ".join(f"{value:.4f}" for value in values)
                + r" \\"
            )
    _write_tex(output_root / "initial_ls_all_rows.tex", all_lines)

    x = np.arange(len(VARIANTS))
    width = 0.19
    figure, axis = plt.subplots(figsize=(3.45, 2.55))
    labels = (
        "ACO+2-opt",
        "RMTGP(no-LS)+2-opt",
        "RMTGP(LS)+2-opt",
        "ACO+3-opt",
    )
    colors = ("#6B7280", "#56B4E9", "#0072B2", "#D55E00")
    for index, (method, label, color) in enumerate(
        zip(methods, labels, colors)
    ):
        y = [
            float(
                lookup[(variant, "tsp500_uniform", method)][
                    "mean_gap_percent"
                ]
            )
            for variant in VARIANTS
        ]
        axis.bar(
            x + (index - 1.5) * width,
            y,
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(x, [VARIANT_LABEL[variant] for variant in VARIANTS])
    axis.set_ylabel("Mean reference gap (%)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=5.8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_root / "initial_ls_tsp500_gap.pdf")
    plt.close(figure)


def _horizon_summary(
    summary: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    for item in summary["horizon_summaries"]:
        if int(item["horizon"]) == horizon:
            return item
    raise KeyError(horizon)


def _write_signal_assets(data: dict[str, Any], output_root: Path) -> None:
    summaries = data["audit_summaries"]
    lines = [
        "% Generated from the completed TSP100 2-opt signal audit at 500 ACO iterations."
    ]
    metrics = {}
    for variant in VARIANTS:
        item = _horizon_summary(summaries[variant], 500)
        values = {
            "headroom": item["baseline_headroom"]["final_gap_percent"]["mean"],
            "compression": item["compression"]["ratio_of_mean_variances"],
            "spearman": item["compression"]["spearman_pre_post"]["mean"],
            "final_snr": item["signal_to_noise"]["final"][
                "ratio_of_mean_variances"
            ],
            "basin_snr": item["signal_to_noise"]["post_basin_auc"][
                "ratio_of_mean_variances"
            ],
            "introduced": item["edge_credit"]["introduced_fraction"]["mean"],
            "survival": item["edge_credit"]["difference_survival"]["mean"],
        }
        metrics[variant] = values
        lines.append(
            f"{VARIANT_LABEL[variant]} & {values['headroom']:.4f} & "
            f"{values['compression']:.4f} & {values['spearman']:.3f} & "
            f"{values['final_snr']:.3f} & {values['basin_snr']:.3f} & "
            f"{100.0 * values['introduced']:.1f} & "
            f"{100.0 * values['survival']:.1f} \\\\"
        )
    _write_tex(output_root / "two_opt_signal_rows.tex", lines)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(3)
    axes[0].bar(
        x - 0.16,
        [metrics[variant]["compression"] for variant in VARIANTS],
        0.32,
        label="Post/pre variance",
        color="#0072B2",
    )
    axes[0].bar(
        x + 0.16,
        [metrics[variant]["final_snr"] for variant in VARIANTS],
        0.32,
        label="Final-gap SNR",
        color="#D55E00",
    )
    axes[0].set_xticks(x, [VARIANT_LABEL[variant] for variant in VARIANTS])
    axes[0].set_ylabel("Variance ratio")
    axes[0].set_title("2-opt compression and final signal")
    axes[0].legend(frameon=False, fontsize=6.2)
    axes[0].grid(axis="y", alpha=0.2)

    final = np.array([metrics[variant]["final_snr"] for variant in VARIANTS])
    basin = np.array([metrics[variant]["basin_snr"] for variant in VARIANTS])
    axes[1].bar(
        x - 0.16,
        final,
        0.32,
        label="Final gap",
        color="#D55E00",
    )
    axes[1].bar(
        x + 0.16,
        basin,
        0.32,
        label="Post-basin AUC",
        color="#009E73",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, [VARIANT_LABEL[variant] for variant in VARIANTS])
    axes[1].set_ylabel("Signal-to-noise ratio (log scale)")
    axes[1].set_title("Alternative learning signals")
    axes[1].legend(frameon=False, fontsize=6.2)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout(w_pad=1.1)
    figure.savefig(output_root / "two_opt_signal_audit.pdf")
    plt.close(figure)


def _write_development_control_assets(
    data: dict[str, Any],
    output_root: Path,
) -> None:
    """汇总已完成但没有独立 final test 的适应度对照与 racing 审计。"""

    control_rows = data["fitness_control_rows"]
    conditions = (
        ("anytime_final", "Anytime+Final"),
        ("basin_final", "Basin+Final"),
    )
    lines = [
        "% Completed TSP100 fitness controls; independent 512-instance gate validation."
    ]
    control_metrics: dict[tuple[str, str], dict[str, float]] = {}
    for variant in VARIANTS:
        for condition, label in conditions:
            rows = control_rows[condition][variant]
            baseline = float(
                np.mean([float(row["baseline_mean_gap_percent"]) for row in rows])
            )
            candidate = float(
                np.mean([float(row["candidate_mean_gap_percent"]) for row in rows])
            )
            delta = float(np.mean([float(row["mean_delta_pp"]) for row in rows]))
            reduction = 100.0 * (baseline - candidate) / baseline
            passed = sum(row["deployed"].lower() == "true" for row in rows)
            control_metrics[(variant, condition)] = {
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
                "reduction": reduction,
                "passed": float(passed),
            }
            lines.append(
                f"{VARIANT_LABEL[variant]} & {label} & {baseline:.4f} & "
                f"{candidate:.4f} & {delta:+.4f} & {reduction:+.1f} & "
                f"{passed}/3 \\\\"
            )
    _write_tex(output_root / "tsp100_fitness_control_rows.tex", lines)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(3)
    width = 0.34
    for offset, (condition, label), color in zip(
        (-0.5, 0.5),
        conditions,
        ("#0072B2", "#D55E00"),
    ):
        axes[0].bar(
            x + offset * width,
            [control_metrics[(variant, condition)]["delta"] for variant in VARIANTS],
            width,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + offset * width,
            [control_metrics[(variant, condition)]["passed"] for variant in VARIANTS],
            width,
            label=label,
            color=color,
        )
    for axis in axes:
        axis.set_xticks(x, [VARIANT_LABEL[variant] for variant in VARIANTS])
        axis.grid(axis="y", alpha=0.2)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Mean validation delta (pp)")
    axes[0].set_title("Selected-candidate gate quality")
    axes[1].set_ylim(0, 3.25)
    axes[1].set_yticks((0, 1, 2, 3))
    axes[1].set_ylabel("Runs passing gate (of 3)")
    axes[1].set_title("Independent deployment gate")
    axes[1].legend(frameon=False, fontsize=6.3)
    figure.tight_layout(w_pad=1.2)
    figure.savefig(output_root / "tsp100_fitness_controls.pdf")
    plt.close(figure)

    racing_summaries = data["racing_summaries"]
    racing_lines = [
        "% TSP500 racing audit at the fixed 16-instance, 500-iteration screen."
    ]
    racing_metrics = {}
    for variant in VARIANTS:
        report = next(
            row
            for row in racing_summaries[variant]["reports"]
            if int(row["horizon"]) == 500 and int(row["instances"]) == 16
        )
        racing_metrics[variant] = report
        racing_lines.append(
            f"{VARIANT_LABEL[variant]} & "
            f"{report['final_nonzero_fraction']:.3f} & "
            f"{report['anytime_nonzero_fraction']:.3f} & "
            f"{report['combined_signal_to_noise']:.3f} & "
            f"{report['top32_recall_at_target']:.3f} & "
            f"{report['spearman_at_target']:.3f} & "
            f"{_tex_bool(report['gate_passed'])} \\\\"
        )
    _write_tex(output_root / "tsp500_racing_audit_rows.tex", racing_lines)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(3)
    snr = [
        racing_metrics[variant]["combined_signal_to_noise"]
        for variant in VARIANTS
    ]
    axes[0].bar(x, snr, color=("#0072B2", "#D55E00", "#009E73"))
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Gate = 1.0")
    axes[0].set_xticks(x, [VARIANT_LABEL[variant] for variant in VARIANTS])
    axes[0].set_ylabel("Combined signal-to-noise")
    axes[0].set_title("16 instances, 500 iterations")
    axes[0].legend(frameon=False, fontsize=6.3)
    axes[0].grid(axis="y", alpha=0.2)

    recall = [
        racing_metrics[variant]["top32_recall_at_target"]
        for variant in VARIANTS
    ]
    correlation = [
        racing_metrics[variant]["spearman_at_target"]
        for variant in VARIANTS
    ]
    axes[1].bar(x - 0.17, recall, 0.34, label="Top-32 recall", color="#56B4E9")
    axes[1].bar(x + 0.17, correlation, 0.34, label="Spearman", color="#CC79A7")
    axes[1].axhline(0.8, color="#56B4E9", linestyle="--", linewidth=0.7)
    axes[1].axhline(0.7, color="#CC79A7", linestyle=":", linewidth=0.8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, [VARIANT_LABEL[variant] for variant in VARIANTS])
    axes[1].set_ylabel("Screen-to-target agreement")
    axes[1].set_title("Ranking fidelity")
    axes[1].legend(frameon=False, fontsize=6.3)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout(w_pad=1.2)
    figure.savefig(output_root / "tsp500_racing_audit.pdf")
    plt.close(figure)


def _write_ls_v2_result_assets(
    data: dict[str, Any],
    output_root: Path,
) -> None:
    rows = data["latest_main"]
    lookup = {
        (row["variant"], row["partition"]): row
        for row in rows
    }
    lines = [
        "% Generated from the completed TSP500 LS-aware v2 final test."
    ]
    for variant in VARIANTS:
        for partition in PARTITIONS:
            row = lookup[(variant, partition)]
            lines.append(
                f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
                f"{float(row['baseline_2opt_mean_gap_percent']):.4f} & "
                f"{float(row['mean_gap_percent']):.4f} & "
                f"{float(row['mean_delta_vs_2opt_pp']):+.4f} & "
                f"{float(row['one_sided_upper_95_vs_2opt']):+.4f} & "
                f"{float(row['baseline_3opt_mean_gap_percent']):.4f} & "
                f"{float(row['mean_delta_vs_3opt_pp']):+.4f} & "
                f"{_tex_bool(row['success_vs_2opt'])}/"
                f"{_tex_bool(row['better_than_3opt'])} \\\\"
            )
    _write_tex(output_root / "ls_v2_main_rows.tex", lines)

    pooled_lines = [
        "% Descriptive arithmetic means across the three TSP500 distributions."
    ]
    for variant in VARIANTS:
        selected = [lookup[(variant, partition)] for partition in PARTITIONS]
        rmtgp = float(
            np.mean(
                [float(row["mean_gap_percent"]) for row in selected]
            )
        )
        aco2 = float(
            np.mean(
                [
                    float(row["baseline_2opt_mean_gap_percent"])
                    for row in selected
                ]
            )
        )
        aco3 = float(
            np.mean(
                [
                    float(row["baseline_3opt_mean_gap_percent"])
                    for row in selected
                ]
            )
        )
        reduction = 100.0 * (aco2 - rmtgp) / aco2
        pooled_lines.append(
            f"{VARIANT_LABEL[variant]} & {aco2:.4f} & {rmtgp:.4f} & "
            f"{aco3:.4f} & {rmtgp-aco2:+.4f} & {rmtgp-aco3:+.4f} & "
            f"{reduction:+.1f} \\\\"
        )
    _write_tex(output_root / "ls_v2_pooled_rows.tex", pooled_lines)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.45),
        sharex=True,
    )
    x = np.arange(3)
    width = 0.25
    for axis, variant in zip(axes, VARIANTS):
        selected = [lookup[(variant, partition)] for partition in PARTITIONS]
        aco2 = [
            float(row["baseline_2opt_mean_gap_percent"])
            for row in selected
        ]
        rmtgp = [float(row["mean_gap_percent"]) for row in selected]
        aco3 = [
            float(row["baseline_3opt_mean_gap_percent"])
            for row in selected
        ]
        axis.bar(
            x - width,
            aco2,
            width,
            label="ACO+2-opt",
            color=COLORS["aco2"],
        )
        axis.bar(
            x,
            rmtgp,
            width,
            label="RMTGP+2-opt",
            color=COLORS["rmtgp"],
        )
        axis.bar(
            x + width,
            aco3,
            width,
            label="ACO+3-opt",
            color=COLORS["aco3"],
        )
        axis.set_title(VARIANT_LABEL[variant])
        axis.set_xticks(x, ("U", "C", "G"))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean reference gap (%)")
    axes[1].set_xlabel("TSP500 distribution (U/C/G)")
    axes[0].legend(frameon=False, fontsize=6.0)
    figure.tight_layout(w_pad=0.6)
    figure.savefig(output_root / "ls_v2_gap_comparison.pdf")
    plt.close(figure)


def _write_ls_v2_runtime_assets(
    data: dict[str, Any],
    output_root: Path,
) -> None:
    rows = data["latest_runtime"]
    lookup = {
        (row["variant"], row["partition"]): row
        for row in rows
    }
    lines = [
        "% Generated from the TSP500 LS-aware v2 latency audit; one selected program."
    ]
    for variant in VARIANTS:
        for partition in PARTITIONS:
            row = lookup[(variant, partition)]
            lines.append(
                f"{VARIANT_LABEL[variant]} & {PARTITION_LABEL[partition]} & "
                f"{float(row['aco_2opt_wall_time_sec_mean']):.1f} & "
                f"{float(row['rmtgp_2opt_individual_wall_time_sec_mean']):.1f} & "
                f"{float(row['aco_3opt_wall_time_sec_mean']):.1f} & "
                f"{float(row['aco_3opt_over_rmtgp_2opt_time_ratio']):.2f} \\\\"
            )
    _write_tex(output_root / "ls_v2_runtime_rows.tex", lines)

    quality = {
        (row["variant"], row["partition"]): row
        for row in data["latest_main"]
    }
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.45))
    methods = (
        (
            "ACO+2-opt",
            "aco_2opt_wall_time_sec_mean",
            "baseline_2opt_mean_gap_percent",
            COLORS["aco2"],
            "o",
        ),
        (
            "RMTGP+2-opt",
            "rmtgp_2opt_individual_wall_time_sec_mean",
            "mean_gap_percent",
            COLORS["rmtgp"],
            "s",
        ),
        (
            "ACO+3-opt",
            "aco_3opt_wall_time_sec_mean",
            "baseline_3opt_mean_gap_percent",
            COLORS["aco3"],
            "^",
        ),
    )
    for axis, variant in zip(axes, VARIANTS):
        for method, time_key, gap_key, color, marker in methods:
            times = [
                float(lookup[(variant, partition)][time_key])
                for partition in PARTITIONS
            ]
            gaps = [
                float(quality[(variant, partition)][gap_key])
                for partition in PARTITIONS
            ]
            axis.scatter(
                np.mean(times),
                np.mean(gaps),
                color=color,
                marker=marker,
                s=28,
                label=method,
            )
        axis.set_title(VARIANT_LABEL[variant])
        axis.set_xlabel("Mean wall time (s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean gap across distributions (%)")
    axes[0].legend(frameon=False, fontsize=5.9)
    figure.tight_layout(w_pad=0.7)
    figure.savefig(output_root / "ls_v2_runtime_quality.pdf")
    plt.close(figure)


def _write_ls_v2_training_time_assets(
    data: dict[str, Any],
    output_root: Path,
) -> None:
    metrics = data["training_metrics"]
    stages = ((0, 15), (15, 35), (35, 50))
    summaries: dict[str, dict[str, Any]] = {}
    lines = [
        "% TSP500 LS-aware v2 training time; mean plus/minus sample SD over three GP runs."
    ]
    for variant in VARIANTS:
        total_hours = []
        stage_seconds = [[], [], []]
        late_throughput = []
        for seed in (81001, 81002, 81003):
            rows = metrics[(variant, seed)]
            total_hours.append(float(rows[-1]["cumulative_wall_time"]) / 3600.0)
            for index, (start, stop) in enumerate(stages):
                stage_seconds[index].append(
                    float(
                        np.mean(
                            [
                                float(row["generation_wall_time"])
                                for row in rows[start:stop]
                            ]
                        )
                    )
                )
            late_throughput.append(
                float(
                    np.mean(
                        [
                            float(row["tours_per_second"])
                            for row in rows[35:50]
                        ]
                    )
                    / 1.0e6
                )
            )
        summaries[variant] = {
            "total_hours": total_hours,
            "stage_seconds": stage_seconds,
            "late_throughput": late_throughput,
        }

        def mean_sd(values: list[float]) -> tuple[float, float]:
            return float(np.mean(values)), float(np.std(values, ddof=1))

        total_mean, total_sd = mean_sd(total_hours)
        stage_stats = [mean_sd(values) for values in stage_seconds]
        throughput_mean, throughput_sd = mean_sd(late_throughput)
        lines.append(
            f"{VARIANT_LABEL[variant]} & "
            f"{total_mean:.2f} $\\pm$ {total_sd:.2f} & "
            + " & ".join(
                f"{mean:.1f} $\\pm$ {sd:.1f}"
                for mean, sd in stage_stats
            )
            + f" & {throughput_mean:.2f} $\\pm$ {throughput_sd:.2f} \\\\"
        )
    _write_tex(output_root / "ls_v2_training_time_rows.tex", lines)

    total_gpu_hours = sum(
        sum(summaries[variant]["total_hours"]) for variant in VARIANTS
    )
    manifest = data["latest_manifest"]
    start = datetime.fromisoformat(manifest["started_at"])
    end = datetime.fromisoformat(manifest["ended_at"])
    final_wall_hours = (end - start).total_seconds() / 3600.0
    gpu_count = len(manifest["arguments"]["gpu_devices"])
    campaign_lines = [
        "% End-to-end completed TSP500 LS-aware campaign accounting.",
        f"Training & 9 GP runs & {total_gpu_hours:.2f} GPU-h & "
        "Sum of recorded run wall times \\\\",
        f"Final test & 27 shards & {final_wall_hours:.2f} wall-h & "
        f"{gpu_count} GPUs; includes latency audit \\\\",
    ]
    _write_tex(output_root / "ls_v2_campaign_time_rows.tex", campaign_lines)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(3)
    width = 0.24
    stage_labels = ("G1--15", "G16--35", "G36--50")
    stage_colors = ("#56B4E9", "#0072B2", "#D55E00")
    for index, (label, color) in enumerate(zip(stage_labels, stage_colors)):
        means = [
            float(np.mean(summaries[variant]["stage_seconds"][index]))
            for variant in VARIANTS
        ]
        axes[0].bar(x + (index - 1) * width, means, width, label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [VARIANT_LABEL[variant] for variant in VARIANTS])
    axes[0].set_ylabel("Mean seconds per generation (log)")
    axes[0].set_title("Multi-fidelity stage cost")
    axes[0].legend(frameon=False, fontsize=6.2)
    axes[0].grid(axis="y", alpha=0.2)

    for index, variant in enumerate(VARIANTS):
        values = summaries[variant]["total_hours"]
        axes[1].scatter(
            np.full(3, index) + np.array((-0.08, 0.0, 0.08)),
            values,
            color=("#0072B2", "#D55E00", "#009E73")[index],
            s=24,
        )
        axes[1].hlines(
            np.mean(values),
            index - 0.18,
            index + 0.18,
            color="black",
            linewidth=1.0,
        )
    axes[1].set_xticks(x, [VARIANT_LABEL[variant] for variant in VARIANTS])
    axes[1].set_ylabel("End-to-end hours per GP run")
    axes[1].set_title("Completed 50-generation runs")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout(w_pad=1.1)
    figure.savefig(output_root / "ls_v2_training_time.pdf")
    plt.close(figure)


def _aggregate_curve_rows(
    curves: dict[tuple[str, int], list[dict[str, str]]],
    variant: str,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    list[tuple[int, tuple[int, ...]]],
]:
    by_generation: dict[int, list[dict[str, str]]] = defaultdict(list)
    for seed in (81001, 81002, 81003):
        for row in curves[(variant, seed)]:
            by_generation[int(row["generation"])].append(row)
    train = []
    validation = []
    excluded = []
    for generation in sorted(by_generation):
        rows = by_generation[generation]
        train_values = np.array(
            [float(row["train_fitness_delta_pp"]) for row in rows]
        )
        train.append(
            {
                "generation": generation,
                "median": float(np.median(train_values)),
                "low": float(np.min(train_values)),
                "high": float(np.max(train_values)),
            }
        )
        val_rows = [
            row
            for row in rows
            if math.isfinite(float(row["validation_delta_pp"]))
        ]
        if not val_rows:
            continue
        horizons = tuple(
            sorted(
                {
                    int(float(row["validation_aco_iterations"]))
                    for row in val_rows
                }
            )
        )
        if len(val_rows) != 3 or len(horizons) != 1:
            excluded.append((generation, horizons))
            continue
        val_values = np.array(
            [float(row["validation_delta_pp"]) for row in val_rows]
        )
        validation.append(
            {
                "generation": generation,
                "median": float(np.median(val_values)),
                "low": float(np.min(val_values)),
                "high": float(np.max(val_values)),
                "horizon": float(horizons[0]),
            }
        )
    return train, validation, excluded


def _write_ls_v2_curve_assets(
    data: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.05, 2.55),
        sharex=True,
    )
    excluded_manifest = []
    for axis, variant in zip(axes, VARIANTS):
        train, validation, excluded = _aggregate_curve_rows(
            data["curves"],
            variant,
        )
        x = np.array([row["generation"] for row in train])
        median = np.array([row["median"] for row in train])
        low = np.array([row["low"] for row in train])
        high = np.array([row["high"] for row in train])
        axis.plot(
            x,
            median,
            color=COLORS["train"],
            label="Generation champion",
        )
        axis.fill_between(
            x,
            low,
            high,
            color=COLORS["train"],
            alpha=0.16,
        )
        if validation:
            vx = np.array([row["generation"] for row in validation])
            vm = np.array([row["median"] for row in validation])
            vl = np.array([row["low"] for row in validation])
            vh = np.array([row["high"] for row in validation])
            axis.errorbar(
                vx,
                vm,
                yerr=np.vstack((vm - vl, vh - vm)),
                color=COLORS["validation"],
                marker="o",
                markersize=2.6,
                capsize=1.8,
                linestyle="--",
                label="Comparable validation",
            )
        axis.axhline(0.0, color="#333333", linewidth=0.7)
        axis.axvline(10, color="#777777", linestyle=":", linewidth=0.8)
        axis.axvline(30, color="#777777", linestyle=":", linewidth=0.8)
        axis.set_title(VARIANT_LABEL[variant])
        axis.set_xlabel("Generation")
        axis.grid(axis="y", alpha=0.2)
        excluded_manifest.append(
            {
                "variant": variant,
                "excluded_validation_points": excluded,
            }
        )
    axes[0].set_ylabel(r"Paired fitness $\Delta g$ (pp)")
    axes[0].legend(frameon=False, fontsize=5.8)
    figure.tight_layout(w_pad=0.7)
    figure.savefig(output_root / "ls_v2_training_curves.pdf")
    plt.close(figure)
    return excluded_manifest


def _expression_terminals(expression: str) -> list[str]:
    known = (
        "RTau",
        "REta",
        "BaseConf",
        "DistRank",
        "Entropy",
        "ConstructProg",
        "ACOProg",
        "Stagnation",
        "MutualRank",
        "TurnCos",
        "EdgeEta",
        "EdgeTau",
        "NNRank",
        "ColonyFreq",
        "SourceQuality",
        "Origin",
        "LSGain",
        "PreFreq",
        "PostFreq",
        "TauHeadroom",
    )
    return [
        name
        for name in known
        if re.search(rf"\b{re.escape(name)}\b", expression)
    ]


def _tex_escape_expression(expression: str) -> str:
    escaped = expression.replace("_", r"\_")
    escaped = escaped.replace(",", r",\allowbreak{}")
    escaped = escaped.replace("(", r"(\allowbreak{}")
    escaped = escaped.replace(")", r"\allowbreak{})")
    return escaped


def _write_expression_assets(
    data: dict[str, Any],
    output_root: Path,
) -> None:
    expressions = data["expressions"]
    lines = [
        "% Generated from selected_candidate_expression.txt for all nine TSP500 LS-aware runs."
    ]
    terminal_lines = [
        "% Compact terminal-use audit for the nine TSP500 LS-aware selected candidates."
    ]
    terminal_names = (
        "RTau",
        "REta",
        "BaseConf",
        "DistRank",
        "Entropy",
        "ConstructProg",
        "ACOProg",
        "Stagnation",
        "MutualRank",
        "TurnCos",
        "EdgeEta",
        "EdgeTau",
        "NNRank",
        "ColonyFreq",
        "SourceQuality",
        "Origin",
        "LSGain",
        "PreFreq",
        "PostFreq",
        "TauHeadroom",
    )
    matrix = []
    labels = []
    for variant in VARIANTS:
        for seed in (81001, 81002, 81003):
            item = expressions[(variant, seed)]
            transition = item["transition"]
            pheromone = item["pheromone"]
            hash_value = item.get("selected_candidate_hash", "")[:12]
            lines.append(
                f"{VARIANT_LABEL[variant]}-{seed} & {hash_value} & "
                f"\\parbox[t]{{0.73\\textwidth}}{{\\ttfamily\\scriptsize "
                f"TR: {_tex_escape_expression(transition)}\\\\ "
                f"PH: {_tex_escape_expression(pheromone)}}} \\\\"
            )
            used = set(
                _expression_terminals(transition + " " + pheromone)
            )
            terminal_lines.append(
                f"{VARIANT_LABEL[variant]}-{seed} & "
                + ", ".join(sorted(used))
                + r" \\"
            )
            matrix.append(
                [1 if terminal in used else 0 for terminal in terminal_names]
            )
            labels.append(f"{VARIANT_LABEL[variant]}-{str(seed)[-1]}")
    _write_tex(output_root / "ls_v2_expression_rows.tex", lines)
    _write_tex(output_root / "ls_v2_terminal_rows.tex", terminal_lines)

    figure, axis = plt.subplots(figsize=(7.0, 2.35))
    axis.imshow(
        np.asarray(matrix),
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(len(terminal_names)),
        terminal_names,
        rotation=55,
        ha="right",
        fontsize=5.6,
    )
    axis.set_yticks(
        np.arange(len(labels)),
        labels,
        fontsize=6.2,
    )
    axis.set_xlabel("Terminal referenced syntactically")
    for row in range(len(labels)):
        for column in range(len(terminal_names)):
            if matrix[row][column]:
                axis.text(
                    column,
                    row,
                    "●",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=5,
                )
    figure.tight_layout()
    figure.savefig(output_root / "ls_v2_terminal_usage.pdf")
    plt.close(figure)


def _write_engineering_assets(
    data: dict[str, Any],
    output_root: Path,
) -> None:
    benchmark = data["benchmark"]
    tuning = benchmark["local_search_launch_tuning"]
    two_opt = tuning["two_opt"]
    three_opt = tuning["three_opt"]
    # This semantic-terminal microbenchmark is frozen in the registered
    # experiment README. It is implementation evidence, not quality evidence.
    micro = {
        "EdgeEta": (1.602, 1.0),
        "PreFreq": (1.606, 1.602 / 1.606),
        "PostFreq": (1.613, 1.602 / 1.613),
        "LSGain": (2.217, 25.64),
    }
    lines = [
        "% Blackwell implementation microbenchmarks; protocols differ and are not combined statistically."
    ]
    lines.append(
        f"2-opt launch: 8 vs. 4 warps & {two_opt['speedup']:.3f}$\\times$ & "
        f"{two_opt['workload']} \\\\"
    )
    lines.append(
        "3-opt block: 512 threads, TSP100 & "
        f"{three_opt['speedup_over_old_one_warp_per_tour']['tsp100']:.3f}"
        "$\\times$ & old one-warp kernel \\\\"
    )
    lines.append(
        "3-opt block: 512 threads, TSP500 & "
        f"{three_opt['speedup_over_old_one_warp_per_tour']['tsp500']:.3f}"
        "$\\times$ & old one-warp kernel \\\\"
    )
    lines.append(
        "Edge LSGain matrix path & 25.640$\\times$ & old per-edge loop \\\\"
    )
    _write_tex(output_root / "blackwell_kernel_rows.tex", lines)

    semantic_lines = [
        "% Frozen semantic-terminal benchmark: 65 programs, 32 TSP500 instances, 10 ACO iterations."
    ]
    for name, (seconds, ratio) in micro.items():
        semantic_lines.append(
            f"{name} & {seconds:.3f} & {ratio:.3f} \\\\"
        )
    _write_tex(
        output_root / "semantic_terminal_benchmark_rows.tex",
        semantic_lines,
    )

    labels = (
        "2-opt\nlaunch",
        "3-opt\nTSP100",
        "3-opt\nTSP500",
        "edge LSGain\nvs loop",
    )
    speedups = (
        two_opt["speedup"],
        three_opt["speedup_over_old_one_warp_per_tour"]["tsp100"],
        three_opt["speedup_over_old_one_warp_per_tour"]["tsp500"],
        25.64,
    )
    figure, axis = plt.subplots(figsize=(3.45, 2.45))
    bars = axis.bar(
        np.arange(4),
        speedups,
        color=("#56B4E9", "#0072B2", "#0072B2", "#009E73"),
    )
    axis.set_yscale("log")
    axis.set_xticks(np.arange(4), labels, fontsize=6.2)
    axis.set_ylabel("Speedup (log scale)")
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, speedups):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.08,
            f"{value:.2f}×",
            ha="center",
            va="bottom",
            fontsize=6.3,
        )
    figure.tight_layout()
    figure.savefig(output_root / "blackwell_kernel_speedups.pdf")
    plt.close(figure)


def build_extended_assets(
    repo_root: Path,
    output_root: Path,
    data: dict[str, Any],
) -> dict[str, Any]:
    """生成后续实验资产，并返回总 provenance 所需的信息。"""

    _write_instance_budget_assets(data, output_root)
    _write_initial_ls_assets(data, output_root)
    _write_signal_assets(data, output_root)
    _write_development_control_assets(data, output_root)
    _write_ls_v2_result_assets(data, output_root)
    _write_ls_v2_runtime_assets(data, output_root)
    _write_ls_v2_training_time_assets(data, output_root)
    excluded = _write_ls_v2_curve_assets(data, output_root)
    _write_expression_assets(data, output_root)
    _write_engineering_assets(data, output_root)

    source_records = []
    for path in data["source_paths"]:
        source_records.append(
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256(path),
            }
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(repo_root),
        "sources": source_records,
        "tsp500_ls_protocol": data["latest_summary"]["protocol"],
        "curve_exclusions": excluded,
        "frozen_engineering_sources": [
            "docs/experiments/tsp100_instance_budget_single_gpu_3seed.md",
            "experiments/tsp500_2opt_ls_v2/README.md",
        ],
    }
