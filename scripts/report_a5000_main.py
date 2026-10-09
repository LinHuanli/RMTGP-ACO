"""A5000主基线队列的进度与配对结果；不把当前阶段称为全部论文实验完成。"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from report_hardware_pilot import export, paired_interval, save_figure

from gpaco.hardware_campaign import clean_record
from gpaco.hardware_inputs import safe_directory


def report(campaign):
    manifest = json.loads((campaign / "campaign.json").read_text())
    tasks = json.loads((campaign / "queue.json").read_text())
    directory = campaign / "summary"
    directory.mkdir(parents=True, exist_ok=True)
    results, pairs, profiles, states = [], [], [], []
    groups = defaultdict(list)
    for task in tasks:
        states.append({k: v for k, v in task.items() if k != "attempts"})
        for attempt in task["attempts"]:
            base = Path(attempt["job_path"]).parent
            for path in sorted((base / "measurements").glob("*/record.json")):
                row = json.loads(path.read_text())
                results.append(
                    {
                        "task": task["id"],
                        "generation": task.get("generation"),
                        "variant": task.get("variant"),
                        "gpu_uuid": attempt["gpu_uuid"],
                        "host": attempt["host"],
                        "attempt": base.name,
                        "record_path": str(path),
                        **row,
                    }
                )
        if task["status"] != "completed" or task["kind"] == "prepare":
            continue
        base = Path(task["attempts"][-1]["job_path"]).parent
        if task["kind"] == "profile":
            path = base / "measurements" / manifest["config"]["profile_mode"] / "record.json"
            row = json.loads(path.read_text())
            profiles.append(
                {
                    "task": task["id"],
                    "variant": task["variant"],
                    "generation": task["generation"],
                    "clean": clean_record(row),
                    **row,
                }
            )
            continue
        generated = json.loads((base / "measurements/generated/record.json").read_text())
        interpreted = json.loads((base / "measurements/interpreted/record.json").read_text())
        for key in ("input_manifest_sha256", "seed", "program_hashes", "search", "requested_tasks"):
            if generated[key] != interpreted[key]:
                raise ValueError(f"配对任务身份不一致：{task['id']}, {key}")
        row = {
            "task": task["id"],
            "n": task["n"],
            "variant": task["variant"],
            "generation": task["generation"],
            "block": task["block"],
            "gpu_uuid": task["attempts"][-1]["gpu_uuid"],
            "host": task["attempts"][-1]["host"],
            "clean": clean_record(generated) and clean_record(interpreted),
            "generated_eval_s": generated["eval_wall_s"],
            "interpreted_eval_s": interpreted["eval_wall_s"],
            "generated_device_s": generated["device_search_s"],
            "interpreted_device_s": interpreted["device_search_s"],
            "generated_warmup_s": generated["warmup"]["wall_s"],
            "interpreted_warmup_s": interpreted["warmup"]["wall_s"],
            "bitwise_equal_lengths": generated["lengths_array_sha256"]
            == interpreted["lengths_array_sha256"],
            "bitwise_equal_tours": generated["tours_array_sha256"]
            == interpreted["tours_array_sha256"],
            "generated_gap_percent": generated["mean_gap_percent"],
            "interpreted_gap_percent": interpreted["mean_gap_percent"],
            "generated_energy_j": generated.get("energy_j"),
            "interpreted_energy_j": interpreted.get("energy_j"),
        }
        pairs.append(row)
        if row["clean"]:
            groups[task["n"], task["generation"], task["variant"]].append(row)
    stats = []
    for (n, generation, variant), rows in sorted(groups.items()):
        ratio, low, high = paired_interval(
            [r["interpreted_eval_s"] for r in rows], [r["generated_eval_s"] for r in rows]
        )
        stats.append(
            {
                "n": n,
                "generation": generation,
                "variant": variant,
                "clean_paired_blocks": len(rows),
                "expected_blocks": manifest["config"]["paired_blocks"],
                "generated_median_s": float(np.median([r["generated_eval_s"] for r in rows])),
                "interpreted_median_s": float(np.median([r["interpreted_eval_s"] for r in rows])),
                "median_interpreter_over_jit": ratio,
                "paired_bootstrap_low": low,
                "paired_bootstrap_high": high,
                "physical_gpu_count": len({r["gpu_uuid"] for r in rows}),
                "bitwise_equal_tour_blocks": sum(r["bitwise_equal_tours"] for r in rows),
                "pilot_not_confirmatory": True,
            }
        )
    for name, values in (
        ("tasks", states),
        ("measurements", results),
        ("paired_blocks", pairs),
        ("performance", stats),
        ("profiles", profiles),
    ):
        export(directory, name, values)
    counts = dict(Counter(t["status"] for t in tasks))
    lines = [
        "# 空闲 A5000 主基线队列",
        "",
        f"任务状态：`{json.dumps(counts, ensure_ascii=False)}`。调度器每60秒扫描一次。",
        "",
        "范围：E01 GPU-Existing基线与E04解释/树JIT已有路径、三宿主执行迁移。",
        "这不是全部E00–E13完成声明。优化CPU、共享计算优化、优化GPU、正式10-seed与最终测试仍有独立前置条件。",
        "本轮cohort均来自AS训练的seed1002，第1/25/50代；在AS/同步ACS/MMAS宿主中重放，不是三宿主分别训练。",
        "5个性能block不是5个GP训练seed。配对两端同一物理卡；不同block可以使用不同A5000，保留GPU/CPU身份。",
        "配对bootstrap区间为当前硬件分配条件下的探索性block区间，不视为GPU型号总体置信区间。",
        "",
        "| TSP | cohort代 | 宿主 | 无争用配对数 | JIT中位秒 | 解释器中位秒 | 解释/JIT比值 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| {row['n']} | {row['generation']} | {row['variant']} | "
            f"{row['clean_paired_blocks']}/{row['expected_blocks']} | "
            f"{row['generated_median_s']:.3f} | {row['interpreted_median_s']:.3f} | "
            f"{row['median_interpreter_over_jit']:.3f} |"
        )
    lines += ["", "## 尚待依赖", "", "| 任务 | 状态/依赖 |", "|---|---|"]
    for task in tasks:
        if task["status"] not in ("completed", "running"):
            lines.append(f"| {task['id']} | {task['status']}: {task.get('dependency_state', '')} |")
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    for n in manifest["config"]["scales"]:
        rows = [r for r in stats if r["n"] == n]
        if rows:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            labels = [f"{r['variant']}/g{r['generation']}" for r in rows]
            x = np.arange(len(rows))
            axes[0].bar(x - 0.18, [r["generated_median_s"] for r in rows], 0.36, label="Tree JIT")
            axes[0].bar(
                x + 0.18, [r["interpreted_median_s"] for r in rows], 0.36, label="Interpreter"
            )
            axes[1].bar(x, [r["median_interpreter_over_jit"] for r in rows])
            axes[1].axhline(1, color="gray", linewidth=0.8)
            for axis in axes:
                axis.set_xticks(x, labels, rotation=45, ha="right")
            axes[0].set_ylabel("Warm evaluation wall (s)")
            axes[1].set_ylabel("Paired interpreter / JIT ratio")
            axes[0].legend()
            fig.suptitle(f"A5000 / TSP{n}: pilot; partial cells retained")
            save_figure(fig, directory, f"tsp{n}-paired-performance")
        rows = [r for r in profiles if r["n"] == n and r["clean"]]
        if rows:
            fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
            labels = [f"{r['variant']}/g{r['generation']}" for r in rows]
            construct = [r["construct_device_s"] for r in rows]
            axis.bar(labels, construct, label="Construct incl. features, GP, selection")
            axis.bar(
                labels,
                [r["update_device_s"] for r in rows],
                bottom=construct,
                label="Pheromone update",
            )
            axis.set(
                ylabel="Instrumented device time (s)", title=f"TSP{n}: separate diagnostic runs"
            )
            axis.tick_params(axis="x", rotation=45)
            axis.legend(fontsize=8)
            save_figure(fig, directory, f"tsp{n}-stage-profile")
    print(json.dumps({"counts": counts, "completed_pairs": len(pairs), "profiles": len(profiles)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    report(safe_directory(args.campaign))


if __name__ == "__main__":
    main()
