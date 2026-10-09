"""从已提交记录增量出表出图；不把调优数据或单次容量探索混入留出统计。"""

import argparse
import csv
import fcntl
import json
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpaco.data import ROOT, write_json
from gpaco.hardware_campaign import clean_record
from gpaco.hardware_inputs import safe_directory


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def export(directory, name, rows):
    write_json(directory / f"{name}.json", rows)
    if rows:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (directory / f"{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def paired_interval(numerator, denominator):
    """以完整配对 block 重采样；5-block 区间仅用于先导，不代表跨设备总体。"""
    ratios = np.asarray(numerator) / np.asarray(denominator)
    if len(ratios) < 2:
        return float(np.median(ratios)), None, None
    samples = np.random.default_rng(5301).choice(ratios, size=(20000, len(ratios)))
    low, high = np.quantile(np.median(samples, axis=1), [0.025, 0.975])
    return float(np.median(ratios)), float(low), float(high)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def report(campaign):
    summary = campaign / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    config = read(campaign / "campaign.json")["config"]
    cells, histories, trainings, audits, statuses = [], [], [], [], []
    by_key = {}
    for model in config["targets"]:
        device = campaign / "devices" / model
        statuses.append(
            {
                "model": model,
                "phase": (read(device / "status.json") or {}).get("phase", "not_started"),
                "status": read(device / "status.json"),
                "failed": read(device / "FAILED.json"),
                "complete": (device / "COMPLETE.json").exists(),
            }
        )
        for path in sorted(device.glob("tsp*/*/*/record.json")):
            record = read(path)
            row = {"model": model, **record, "clean": clean_record(record)}
            cells.append(row)
            if row["stage"] == "holdout" and row["clean"]:
                role = "selected" if row["cell"].endswith("selected") else "default"
                by_key[model, row["n"], row["block"], role] = row
        for path in sorted(device.glob("tsp*/training/seed-*/history.json")):
            run = path.parent
            manifest = read(run / "run_manifest.json")
            n, seed = manifest["n"], manifest["root_seed"]
            history = read(path)
            for row in history:
                histories.append({"model": model, "n": n, "seed": seed, **row})
            complete = read(run / "COMPLETE.json")
            telemetry = read(run / "telemetry_summary.json") or {}
            champion = read(run / "champion.json") or {}
            trainings.append(
                {
                    "model": model,
                    "n": n,
                    "seed": seed,
                    "completed": bool(complete),
                    "generations": len(history),
                    "training_wall_s": (complete or {}).get("training_wall_s"),
                    "local_validation_gap_percent": champion.get("validation_gap_percent"),
                    "canonical_baseline_gap_percent": champion.get(
                        "validation_baseline_gap_percent"
                    ),
                    "local_delta_pp": champion.get("validation_delta_pp"),
                    "energy_j": telemetry.get("energy_j"),
                    "energy_method": telemetry.get("energy_method"),
                    "contended": telemetry.get("contended"),
                    "median_generation_s": float(
                        np.median([r["generation_wall_s"] for r in history])
                    ),
                    "champion": champion.get("expression"),
                }
            )
    for path in sorted((campaign / "canonical_audit").glob("tsp*/*/seed-*/COMPLETE.json")):
        audits.append(read(path))
    comparisons, numerical = [], []
    for model in config["targets"]:
        for n in config["sizes"]:
            for kind in (
                "tuned_vs_default",
                "cross_gpu_default_vs_a5000",
                "cross_gpu_selected_vs_a5000",
            ):
                if model == "a5000" and kind != "tuned_vs_default":
                    continue
                pairs = []
                for block in range(config["paired_blocks"]):
                    if kind == "tuned_vs_default":
                        a, b = (
                            by_key.get((model, n, block, "default")),
                            by_key.get((model, n, block, "selected")),
                        )
                    else:
                        role = "selected" if "selected" in kind else "default"
                        a, b = (
                            by_key.get(("a5000", n, block, role)),
                            by_key.get((model, n, block, role)),
                        )
                    if a and b:
                        # 两端必须来自同一冻结输入和随机流；切勿跨 I 或 B 配对。
                        if (
                            a["seed"],
                            a["program_hashes"],
                            a["search"],
                            a["b"],
                            a.get("input_manifest_sha256"),
                        ) != (
                            b["seed"],
                            b["program_hashes"],
                            b["search"],
                            b["b"],
                            b.get("input_manifest_sha256"),
                        ):
                            raise ValueError("性能配对身份不一致")
                        pairs.append((a, b))
                if not pairs:
                    continue
                numerical.append(
                    {
                        "model": model,
                        "n": n,
                        "comparison": kind,
                        "paired_blocks": len(pairs),
                        "bitwise_equal_length_blocks": sum(
                            a.get("lengths_array_sha256") == b.get("lengths_array_sha256")
                            for a, b in pairs
                        ),
                        "bitwise_equal_tour_blocks": sum(
                            a.get("tours_array_sha256") == b.get("tours_array_sha256")
                            for a, b in pairs
                        ),
                        "mean_gap_difference_pp_denominator_minus_numerator": float(
                            np.mean(
                                [b["mean_gap_percent"] - a["mean_gap_percent"] for a, b in pairs]
                            )
                        ),
                    }
                )
                for metric in ("eval_wall_s", "device_search_s"):
                    speed, low, high = paired_interval(
                        [a[metric] for a, b in pairs], [b[metric] for a, b in pairs]
                    )
                    comparisons.append(
                        {
                            "model": model,
                            "n": n,
                            "comparison": kind,
                            "metric": metric,
                            "paired_blocks": len(pairs),
                            "expected_blocks": config["paired_blocks"],
                            "complete": len(pairs) == config["paired_blocks"],
                            "median_speedup": speed,
                            "paired_bootstrap_95_low": low,
                            "paired_bootstrap_95_high": high,
                            "denominator_median_s": float(np.median([b[metric] for a, b in pairs])),
                        }
                    )
    for name, rows in (
        ("cells", cells),
        ("generations", histories),
        ("training", trainings),
        ("audits", audits),
        ("comparisons", comparisons),
        ("numerical_comparisons", numerical),
        ("status", statuses),
    ):
        export(summary, name, rows)
    for n in config["sizes"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        any_points = False
        for model in config["targets"]:
            for role, marker in (("default", "o"), ("selected", "s")):
                rows = [
                    v
                    for (m, size, b, r), v in by_key.items()
                    if m == model and size == n and r == role
                ]
                if not rows:
                    continue
                any_points = True
                x = float(np.median([r["eval_wall_s"] for r in rows]))
                energies = [r["energy_j"] for r in rows if r.get("energy_j") is not None]
                if energies:
                    axes[0].scatter(x, np.median(energies), marker=marker, label=f"{model}/{role}")
                axes[1].scatter(
                    [model] * len(rows),
                    [r["eval_wall_s"] for r in rows],
                    marker=marker,
                    label=f"{model}/{role}",
                )
        if any_points:
            axes[0].set(
                xlabel="Warm evaluation wall (s)", ylabel="Whole-GPU energy per evaluation (J)"
            )
            axes[1].set(
                ylabel="Warm evaluation wall (s)", title="Raw held-out blocks; not training runs"
            )
            axes[0].legend(fontsize=7)
            fig.suptitle(f"TSP{n}: pilot paired measurements")
            save_figure(fig, summary, f"tsp{n}-time-energy")
        else:
            plt.close(fig)
        active = [
            r
            for r in cells
            if r["n"] == n and r["stage"] == "capacity_exploratory_single_block" and r["clean"]
        ]
        if active:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for model in config["targets"]:
                rows = sorted(
                    [r for r in active if r["model"] == model],
                    key=lambda r: r["plan"]["active_tasks"],
                )
                if not rows:
                    continue
                axes[0].plot(
                    [r["active_tasks"] for r in rows],
                    [r["eval_wall_s"] for r in rows],
                    "o-",
                    label=model,
                )
                memory = [r for r in rows if r.get("sampled_peak_gpu_used_bytes") is not None]
                axes[1].plot(
                    [r["active_tasks"] for r in memory],
                    [r["sampled_peak_gpu_used_bytes"] / 2**30 for r in memory],
                    "o-",
                    label=model,
                )
            axes[0].set(ylabel="Evaluation wall (s)", xlabel="Actual active tasks")
            axes[1].set(ylabel="Sampled total GPU memory peak (GiB)", xlabel="Actual active tasks")
            axes[0].legend()
            fig.suptitle(f"TSP{n}: B=128; one block/config, exploratory only")
            save_figure(fig, summary, f"tsp{n}-capacity")
        for model in config["targets"]:
            groups = defaultdict(list)
            for row in histories:
                if row["model"] == model and row["n"] == n:
                    groups[row["seed"]].append(row)
            if not groups:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            for seed, rows in groups.items():
                x = [r["generation"] for r in rows]
                axes[0].plot(
                    x, [r["train_best_gap_percent"] for r in rows], "o-", label=f"best/{seed}"
                )
                axes[0].plot(
                    x, [r["train_median_gap_percent"] for r in rows], ":", label=f"median/{seed}"
                )
                axes[0].plot(
                    x,
                    [r["train_baseline_gap_percent"] for r in rows],
                    "--",
                    alpha=0.6,
                    label=f"ACO/{seed}",
                )
                axes[1].plot(x, [r["generation_wall_s"] for r in rows], "o-", label=str(seed))
                validated = [r for r in rows if r["validation_champion_gap_percent"] is not None]
                axes[2].plot(
                    [r["generation"] for r in validated],
                    [r["validation_champion_gap_percent"] for r in validated],
                    "o",
                    label=f"local/{seed}",
                )
                for audit in audits:
                    if audit["model"] == model and audit["n"] == n and audit["root_seed"] == seed:
                        axes[2].scatter(
                            [config["generations"]],
                            [audit["canonical_validation_gap_percent"]],
                            marker="x",
                            label=f"A5000 audit/{seed}",
                        )
            for axis in axes:
                axis.set_xlabel("GP generation")
                axis.grid(alpha=0.2)
                axis.legend(fontsize=6)
            axes[0].set_ylabel("Training gap (%)")
            axes[1].set_ylabel("Generation wall (s)")
            axes[2].set_ylabel("Validation gap (%); no test")
            fig.suptitle(f"{model} / TSP{n}: three-seed exploratory short training")
            save_figure(fig, summary, f"{model}-tsp{n}-training")
    lines = [
        "# 跨卡先导实验（自动增量汇总）",
        "",
        "未完成的数据明确标记。5 个配对 block 与 3 个 GP 根种子是不同统计单位。",
        "bootstrap 为配对 block 重采样的探索性 95% 区间；单台机器/单张物理卡不能代表整个型号总体。",
        "主机 CPU、ECC、功耗限制和架构均有差异，跨卡结果不是单独某个硬件特性的因果效应。",
        "标准测试集未开启；validation 只有第 3 代一个选优点，不构造虚假的多代验证曲线。",
        "",
        "| GPU | TSP | 对比 | 完整配对数 | wall 加速比中位数 | 探索性 95% 区间 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in comparisons:
        if row["metric"] == "eval_wall_s":
            interval = (
                "待补"
                if row["paired_bootstrap_95_low"] is None
                else f"[{row['paired_bootstrap_95_low']:.3f}, {row['paired_bootstrap_95_high']:.3f}]"
            )
            lines.append(
                f"| {row['model']} | {row['n']} | {row['comparison']} | {row['paired_blocks']}/5 | {row['median_speedup']:.3f} | {interval} |"
            )
    lines += [
        "",
        "## 短训练与统一 A5000 验证复评",
        "",
        "| GPU | TSP | seed | 训练 wall/s | 本卡 val gap/% | A5000 复评 gap/% | 相对 ACO/pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trainings:
        audit = next(
            (
                a
                for a in audits
                if (a["model"], a["n"], a["root_seed"]) == (row["model"], row["n"], row["seed"])
            ),
            {},
        )

        def number(value):
            return "待完成" if value is None else f"{value:.4f}"

        lines.append(
            f"| {row['model']} | {row['n']} | {row['seed']} | {number(row['training_wall_s'])} | {number(row['local_validation_gap_percent'])} | {number(audit.get('canonical_validation_gap_percent'))} | {number(audit.get('canonical_delta_pp'))} |"
        )
    (summary / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cells": len(cells), "training_runs": len(trainings), "audits": len(audits)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="hardware-pilot-v1")
    args = parser.parse_args()
    campaign = safe_directory(ROOT / "artifacts" / args.campaign)
    with (campaign / "report.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report(campaign)


if __name__ == "__main__":
    main()
