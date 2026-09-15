"""从原始测量生成报告图表、硬件示意与双语逐页提纲。"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .presentation_bench import DEFAULT_OUTPUT, atomic_json, read_json
from .presentation_bench import write_csv as write_measurement_csv

COLORS = {
    "cpu1": "#778899",
    "cpu8": "#155f93",
    "v1": "#d68b25",
    "v2": "#008a70",
    "v2-interp4": "#a85d93",
    "v2-gen4": "#569ccc",
}
LABELS = {
    "cpu1": "CPU-1 (FP64)",
    "cpu8": "CPU-8 (FP64)",
    "v1": "GPU-v1 (FP32)",
    "v2": "GPU-v2 generated / 8 lanes",
    "v2-interp4": "v2 interpreter / 4 lanes",
    "v2-gen4": "v2 generated / 4 lanes",
}


def write_csv(path, rows):
    """重建派生 CSV；没有有效行时不能遗留上次汇总的数据。"""
    if rows:
        write_measurement_csv(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{os.getpid()}.empty.tmp")
        temporary.write_text("")
        temporary.replace(path)


def save(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(directory / f"{name}.{suffix}", bbox_inches="tight", dpi=170)
    plt.close(figure)


def stats(values):
    x = np.asarray(values, dtype=float)
    middle = float(np.median(x))
    return middle, middle - float(x.min()), float(x.max()) - middle


def new_plot(title, ylabel="Seconds", width=9, height=4.7):
    figure, ax = plt.subplots(figsize=(width, height), layout="constrained")
    ax.set_title(title, loc="left", fontsize=14, pad=16)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    return figure, ax


def collect(output):
    records, sanity = [], []
    for state_path in sorted((output / "jobs").glob("*/status.json")):
        state = read_json(state_path)
        if state.get("status") != "completed" or state.get("contaminated"):
            continue
        destination = Path(state["destination"])
        result_path = destination / "result.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        task = state["task"]
        if result.get("kind") == "sanity":
            sanity.append({"task": task, "host": state["host"], **result})
            continue
        if result.get("kind") not in ("train", "replay") or not task.startswith(
            ("E1-", "E2-", "E4-", "E5-")
        ):
            continue
        repeat_match = re.search(r"-r(\d+)$", task)
        repeat = int(repeat_match[1]) if repeat_match else None
        env = result.get("environment", {})
        samples = destination / f"telemetry-{state.get('attempt', 0)}.json"
        sampled_peak = None
        if samples.is_file():
            memory = []
            for sample in read_json(samples):
                for line in sample["sample"].splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4 and parts[1] == state["gpu_uuid"]:
                        try:
                            memory.append(float(parts[3]) * 1024**2)
                        except ValueError:
                            pass
            sampled_peak = max(memory) if memory else None
        for original in result["records"]:
            row = dict(original)
            row.update(
                task_id=task,
                experiment_id=task.split("-")[0],
                repeat_id=repeat,
                host=state["host"],
                gpu_uuid=state["gpu_uuid"],
                gpu_name=env.get("gpu", {}).get("name"),
                source_hash=state["source_hash"],
                sampled_peak_device_memory_bytes=sampled_peak,
                memory_sampling_interval_s=2,
                provenance=str(result_path.relative_to(output)),
            )
            records.append(row)
    return records, sanity


def matched_summary(rows):
    """只在同机、同输入、同实际工作量的完整重复之间计算速度比。"""
    by_run = defaultdict(list)
    for row in rows:
        if row["experiment_id"] == "E2":
            by_run[(row["host"], row["repeat_id"], row["backend"], row["source_hash"])].append(row)
    totals, speedups = [], []
    for (host, repeat, backend, source_hash), values in sorted(by_run.items()):
        signature = [
            (r["workload_id"], r["tasks_executed"], r["tours_executed"])
            for r in sorted(values, key=lambda r: r["evaluation_call_id"])
        ]
        value = sum(r["evaluation_wall_s"] for r in values)
        totals.append(
            {
                "host": host,
                "repeat_id": repeat,
                "backend": backend,
                "source_hash": source_hash,
                "evaluation_wall_s": value,
                "calls": len(values),
                "signature": signature,
            }
        )
    for row in totals:
        reference = next(
            (
                r
                for r in totals
                if r["host"] == row["host"]
                and r["repeat_id"] == row["repeat_id"]
                and r["backend"] == "cpu8"
                and r["source_hash"] == row["source_hash"]
            ),
            None,
        )
        if reference and reference["signature"] == row["signature"]:
            speedups.append(
                {
                    "host": row["host"],
                    "repeat_id": row["repeat_id"],
                    "backend": row["backend"],
                    "speedup_vs_cpu8": reference["evaluation_wall_s"] / row["evaluation_wall_s"],
                }
            )
    return totals, speedups


def performance_figures(output, rows):
    figures = output / "figures"
    totals, speedups = matched_summary(rows)
    write_csv(output / "trace_totals.csv", totals)
    write_csv(output / "speedups.csv", speedups)
    if totals:
        figure, ax = new_plot("Matched 5-generation evaluation trace")
        labels = [b for b in ("cpu1", "cpu8", "v1", "v2") if any(r["backend"] == b for r in totals)]
        for x, backend in enumerate(labels):
            values = [r["evaluation_wall_s"] for r in totals if r["backend"] == backend]
            middle, low, high = stats(values)
            ax.bar(x, middle, color=COLORS[backend])
            ax.errorbar(x, middle, yerr=np.array([[low], [high]]), color="black", capsize=4)
            ax.annotate(
                f"{middle:.2f} s\nn={len(values)}",
                (x, middle),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(
            range(len(labels)), [LABELS[b].replace(" generated / 8 lanes", "") for b in labels]
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.set_xlabel("First use of trace; includes new-program compilation; median and min–max")
        save(figure, figures, "C3_matched_backends")
    for backend in ("cpu8", "v2"):
        data = [r for r in rows if r["experiment_id"] == "E1" and r["backend"] == backend]
        if not data:
            continue
        # 取中位总时间的一次完整运行；堆叠互斥墙钟，不把事件时间与编译时间相加。
        repeats = sorted(
            set(r["repeat_id"] for r in data),
            key=lambda repeat: sum(
                r["generation_wall_s"] for r in data if r["repeat_id"] == repeat
            ),
        )
        chosen = repeats[len(repeats) // 2]
        selected = sorted(
            (r for r in data if r["repeat_id"] == chosen), key=lambda r: r["generation"]
        )
        figure, ax = new_plot(f"Real evolution: {LABELS[backend]} (repeat {chosen})")
        xs = np.array([r["generation"] for r in selected])
        bottom = np.zeros(len(selected))
        for key, label, color in (
            ("evaluation_request_wall_s", "Evaluation request", COLORS[backend]),
            ("evolve_s", "Selection / variation", "#db9860"),
            ("logging_s", "Logging", "#9e79b5"),
            ("other_wall_s", "Other", "#c7c7c7"),
        ):
            values = np.array([r.get(key, 0) for r in selected])
            ax.bar(xs, values, bottom=bottom, label=label, color=color)
            bottom += values
        ax.set_xticks(xs)
        ax.set_xlabel("Generation (inputs preloaded in startup)")
        ax.legend(fontsize=9)
        save(figure, figures, f"C4_generation_stages_{backend}")
        typical = selected[min(2, len(selected) - 1)]
        figure, ax = new_plot(f"Generation {typical['generation']}: {LABELS[backend]}")
        keys = ["evaluation_request_wall_s", "evolve_s", "logging_s", "other_wall_s"]
        ax.bar(
            ["Evaluation", "Evolution", "Logging", "Other"],
            [typical[k] for k in keys],
            color=[COLORS[backend], "#db9860", "#9e79b5", "#c7c7c7"],
        )
        save(figure, figures, f"C1_stage_time_{backend}")
    replay = [r for r in rows if r["experiment_id"] == "E2"]
    if replay:
        figure, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained", sharex=True)
        for backend in ("cpu1", "cpu8", "v1", "v2"):
            data = [r for r in replay if r["backend"] == backend]
            if not data:
                continue
            xs = sorted(set(r["generation"] for r in data))
            ys = [stats([r["evaluation_wall_s"] for r in data if r["generation"] == g]) for g in xs]
            axes[0].errorbar(
                xs,
                [v[0] for v in ys],
                yerr=np.array([[v[1] for v in ys], [v[2] for v in ys]]),
                marker="o",
                color=COLORS[backend],
                label=LABELS[backend],
                capsize=3,
            )
        reference = next(
            (
                [r for r in replay if r["backend"] == b and r["repeat_id"] == 0]
                for b in ("v2", "cpu8", "cpu1")
                if any(r["backend"] == b for r in replay)
            ),
            [],
        )
        for key, label in (
            ("population_requested", "Requested"),
            ("programs_unique", "Structural unique"),
            ("programs_executed", "Executed"),
        ):
            axes[1].plot(
                [r["generation"] for r in reference], [r[key] for r in reference], "o-", label=label
            )
        axes[0].set_title("Matched trace: time and actual workload", loc="left")
        axes[0].set_ylabel("Evaluation seconds")
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("Programs")
        axes[1].set_xlabel("Trace generation")
        axes[1].legend()
        save(figure, figures, "C5_matched_generations")
    scans = [r for r in rows if r["experiment_id"] == "E4"]
    for mode, name, title in (
        ("population", "C6_population", "Population scaling (TSP100, 32 instances)"),
        ("cities", "C7_cities", "Simulation size (100 programs, 32 instances)"),
    ):
        subset = [
            r
            for r in scans
            if (r["cities"] == 100 if mode == "population" else r["population_requested"] == 100)
        ]
        if not subset:
            continue
        key = "population_requested" if mode == "population" else "cities"
        figure, axes = plt.subplots(
            1,
            2 if mode == "population" else 1,
            figsize=(10, 4.8),
            layout="constrained",
            squeeze=False,
        )
        for backend in ("cpu8", "v2"):
            data = [r for r in subset if r["backend"] == backend]
            xs = sorted(set(r[key] for r in data))
            if not xs:
                continue
            for ax_index, field in enumerate(
                ("time", "throughput") if mode == "population" else ("time",)
            ):
                values = [
                    stats(
                        [
                            r["evaluation_wall_s"]
                            if field == "time"
                            else r["tasks_executed"] / r["evaluation_wall_s"]
                            for r in data
                            if r[key] == x
                        ]
                    )
                    for x in xs
                ]
                axes[0, ax_index].errorbar(
                    xs,
                    [v[0] for v in values],
                    yerr=np.array([[v[1] for v in values], [v[2] for v in values]]),
                    marker="o",
                    capsize=3,
                    label=LABELS[backend],
                    color=COLORS[backend],
                )
                axes[0, ax_index].set_ylabel(
                    "Evaluation seconds" if field == "time" else "Actual tasks / second"
                )
                axes[0, ax_index].set_xlabel(
                    "Requested population" if mode == "population" else "Cities"
                )
                axes[0, ax_index].legend(fontsize=8)
        figure.suptitle(title)
        save(figure, figures, name)
    for generation in (3, 5):
        data = [r for r in rows if r["experiment_id"] == "E5" and r["generation"] == generation]
        if not data:
            continue
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
        order = [
            b for b in ("v1", "v2-interp4", "v2-gen4", "v2") if any(r["backend"] == b for r in data)
        ]
        for ax, key, title in zip(
            axes,
            ("evaluation_wall_s", "compile_load_s", "gpu_span_s"),
            ("Evaluation wall", "Compile + module load", "GPU event span"),
            strict=True,
        ):
            for x, backend in enumerate(order):
                values = [r[key] for r in data if r["backend"] == backend and r[key] is not None]
                mid, low, high = stats(values)
                ax.bar(x, mid, color=COLORS[backend])
                ax.errorbar(x, mid, yerr=np.array([[low], [high]]), capsize=3, color="black")
            ax.set_title(title)
            ax.set_ylabel("Seconds")
            ax.set_xticks(
                range(len(order)), [LABELS[b] for b in order], rotation=25, ha="right", fontsize=8
            )
        figure.suptitle(
            f"GPU ablation: generation {generation}; nested timing metrics shown separately"
        )
        save(figure, figures, f"C8_ablations_g{generation}")
    jit = output / "jobs/E3-jit/result.json"
    if jit.exists():
        data = read_json(jit)["records"]
        figure, ax = new_plot("Individual JIT: compilation amortization (sum over 8 trees)")
        for method, field, name in (
            ("python", "execute_s", "Python numeric loop"),
            ("individual_jit", "total_s", "Individual JIT + compilation"),
            ("individual_jit", "execute_s", "Individual JIT, compiled"),
            ("postfix_numba", "execute_s", "Numba postfix, compiled"),
        ):
            xs = sorted(set(r["call_count"] for r in data))
            ys = []
            for count in xs:
                by_repeat = defaultdict(float)
                for row in data:
                    if row["method"] == method and row["call_count"] == count:
                        by_repeat[row["repeat_id"]] += row[field]
                ys.append(np.median(list(by_repeat.values())))
            ax.plot(xs, ys, "o-", label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Calls per tree; hot loop is inside the Numba boundary")
        ax.legend(fontsize=8)
        save(figure, figures, "C2_jit_amortization")
    return totals, speedups


def canvas(title):
    figure, ax = plt.subplots(figsize=(12, 6.4))
    ax.set(xlim=(0, 12), ylim=(0, 6.4))
    ax.axis("off")
    ax.text(0.2, 6.05, title, fontsize=18, weight="bold", va="top", color="#15364a")
    return figure, ax


def box(ax, x, y, w, h, text, color="#e4eff7", size=12):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.1",
            linewidth=1.2,
            edgecolor="#607784",
            facecolor=color,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, start, end, text=None):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, color="#607784", linewidth=1.5
        )
    )
    if text:
        ax.text(
            (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.15, text, fontsize=10, ha="center"
        )


def diagrams(output):
    directory = output / "figures"
    figure, ax = canvas("Where simulation-based GP spends its time")
    levels = [
        ("GP generation", "Selection depends on returned fitness"),
        ("Programs × instances", "Independent simulation tasks"),
        ("ACO iteration", "Pheromone state links successive iterations"),
        ("Ants", "Parallel construction from a shared state snapshot"),
        ("Construction step", "Next step depends on the selected city"),
        ("Candidate scores → GP expression", "Parallel scoring; reduction and selection"),
    ]
    for i, (label, note) in enumerate(levels):
        x, y = 0.25 + i * 0.27, 4.65 - i * 0.77
        box(ax, x, y, 4.8, 0.5, label, color="#daf1e9" if i in (1, 3, 5) else "#e4eff7", size=11)
        ax.text(6.85, y + 0.25, note, fontsize=10, va="center")
        if i:
            arrow(ax, (x + 2, y + 1.0), (x + 2, y + 0.55))
    save(figure, directory, "D1_compute_hierarchy")

    figure, ax = canvas("DEAP manages trees; compiled code evaluates numbers")
    box(ax, 0.2, 2.3, 2.0, 1.3, "DEAP\npopulation")
    for y, first, second in (
        (4.2, "Executable numeric function", "Numba individual JIT"),
        (1.1, "Postfix opcode arrays", "Numba interpreter + simulator"),
    ):
        box(ax, 3.2, y, 3.3, 0.9, first)
        box(ax, 7.6, y, 4.0, 0.9, second, "#daf1e9")
        arrow(ax, (2.2, 2.95), (3.2, y + 0.45))
        arrow(ax, (6.5, y + 0.45), (7.6, y + 0.45))
    ax.text(3.25, 3.65, "Expression microbenchmark", fontsize=11)
    ax.text(3.25, 0.55, "Current CPU simulation backend", fontsize=11)
    save(figure, directory, "D2_deap_numba")

    figure, ax = canvas("RTX A5000: hardware and data movement")
    box(ax, 0.2, 3.2, 2.2, 1.5, "CPU\nSystem RAM")
    box(ax, 3.3, 1.8, 8.1, 3.5, "", "#f2f6f9")
    ax.text(3.5, 5.02, "GPU chip: 64 SMs · compute capability 8.6", fontsize=12)
    box(ax, 3.7, 4.0, 7.3, 0.55, "L2 cache: 6 MiB", "#f8ebd7")
    for i in range(6):
        box(ax, 3.75 + i * 1.21, 2.85, 1.0, 0.75, f"SM {i}" if i < 5 else "…", size=10)
    box(ax, 3.6, 0.5, 7.6, 0.75, "24 GB GDDR6 · 768 GB/s specification bandwidth", "#daf1e9")
    arrow(ax, (2.4, 3.95), (3.25, 3.95))
    ax.text(2.75, 4.55, "PCIe\n4.0 ×16", fontsize=9, ha="center")
    arrow(ax, (7.2, 1.3), (7.2, 1.78))
    ax.text(3.65, 2.13, "8,192 CUDA cores across SMs; CUDA threads are scheduled work", fontsize=10)
    ax.text(
        0.2,
        0.03,
        "Sources: NVIDIA RTX A5000 datasheet; CUDA device properties measured on target GPUs",
        fontsize=9,
    )
    save(figure, directory, "D3_a5000_hardware")

    figure, ax = canvas("One SM: execution resources and storage")
    box(ax, 0.25, 1.0, 7.0, 4.5, "", "#f2f6f9")
    box(ax, 0.6, 4.4, 6.3, 0.65, "Warp schedulers → execution units", "#daf1e9")
    box(ax, 0.6, 3.25, 6.3, 0.75, "Register file: 64K × 32-bit registers")
    box(
        ax,
        0.6,
        1.55,
        6.3,
        1.2,
        "Combined L1 / shared resource: 128 KiB\nShared-memory capacity: up to 100 KiB / SM",
        "#f8ebd7",
    )
    ax.text(7.8, 4.7, "Registers\nIndices and intermediate scalars", fontsize=11, va="top")
    ax.text(7.8, 3.35, "Shared memory\nExplicit block cooperation", fontsize=11, va="top")
    ax.text(7.8, 2.0, "L1 / L2 caches\nHardware-managed data reuse", fontsize=11, va="top")
    ax.text(
        0.4,
        0.4,
        "Global memory: geometry, candidate lists, pheromone, tours, working buffers",
        fontsize=12,
    )
    save(figure, directory, "D4_sm_memory")

    figure, ax = canvas("Mapping independent tasks and candidate lanes")
    ax.text(0.3, 5.2, "Grid: unique programs × instances", fontsize=13)
    for row in range(4):
        for column in range(4):
            box(ax, 0.3 + column * 0.8, 1.6 + row * 0.73, 0.64, 0.55, "task", size=8)
    arrow(ax, (3.6, 3.4), (4.6, 3.4))
    box(ax, 4.8, 1.35, 6.65, 3.5, "", "#f2f6f9")
    ax.text(5.0, 4.4, "1 construction block = 32 ants × 8 lanes", fontsize=12)
    for ant in range(4):
        for lane in range(8):
            ax.add_patch(
                Rectangle(
                    (5.05 + ant * 1.54 + lane * 0.18, 3.25),
                    0.15,
                    0.65,
                    facecolor=["#155f93", "#008a70", "#d68b25", "#a85d93"][ant],
                )
            )
        ax.text(5.72 + ant * 1.54, 2.85, f"Ant {ant}", fontsize=10, ha="center")
    ax.text(5.0, 2.15, "One warp: 32 threads = four 8-lane ant groups", fontsize=11)
    ax.text(5.0, 1.65, "Block: 256 threads = 8 warps", fontsize=11)
    ax.text(
        0.4,
        0.55,
        "8 lanes cooperate over 20 candidates; construction steps remain sequential",
        fontsize=12,
    )
    save(figure, directory, "D5_task_mapping")

    figure, ax = canvas("CPU–GPU workflow for one GP generation")
    steps = [
        ("CPU: selection / variation", "#e4eff7"),
        ("Encode trees / generate CUDA", "#e4eff7"),
        ("Host: compile + load modules", "#e4eff7"),
        ("GPU: simulate all tasks", "#daf1e9"),
        ("CPU: FP64 score returned tours", "#e4eff7"),
        ("Write fitness → next generation", "#e4eff7"),
    ]
    for i, (label, color) in enumerate(steps):
        row, col = divmod(i, 3)
        x, y = 0.3 + col * 4.0, 3.7 - row * 2.0
        box(ax, x, y, 3.3, 1.0, label, color, size=11)
        if col < 2:
            arrow(ax, (x + 3.35, y + 0.5), (x + 3.95, y + 0.5))
    arrow(ax, (10.0, 3.65), (2.0, 2.75), "Submit GPU work")
    ax.text(0.4, 0.65, "DEAP remains on CPU; numerical simulation uses GPU blocks", fontsize=12)
    save(figure, directory, "D6_generation_pipeline")

    figure, ax = canvas("Resident data and repeated CUDA work")
    box(ax, 0.3, 4.2, 3.0, 0.8, "Upload static problem data")
    box(ax, 4.0, 4.2, 7.3, 0.8, "Reuse geometry while the same batch is active", "#f8ebd7")
    arrow(ax, (3.35, 4.6), (3.95, 4.6))
    ax.text(0.3, 3.1, "v2", fontsize=14)
    for i in range(3):
        x = 1.1 + 3.15 * i
        box(ax, x, 2.6, 1.65, 0.8, "construct", "#daf1e9")
        box(ax, x + 1.7, 2.6, 1.1, 0.8, "update", "#e4eff7")
        if i < 2:
            arrow(ax, (x + 2.85, 3.0), (x + 3.1, 3.0))
    ax.text(10.85, 2.97, "…", fontsize=18)
    ax.text(
        0.35,
        1.75,
        "Each iteration has state dependencies; v2 does not fuse all iterations into one kernel",
        fontsize=11,
    )
    ax.text(
        0.35,
        0.85,
        "New instance batch → new uploads as needed       Final tours → CPU FP64 scoring",
        fontsize=11,
    )
    save(figure, directory, "D7_residency_timeline")


PAGES = [
    (
        "GPU Acceleration for Simulation-based GP",
        0.5,
        "D6",
        "用 population、simulation、fitness 三个环节定义本次加速对象。"
        "报告围绕计算成本，不展开 GP-ACO 的算法创新。",
    ),
    (
        "How much work is inside one evaluation?",
        2,
        "D1",
        "从个体、实例、ACO iteration、蚂蚁、路径步骤和候选逐层展开。"
        "指出独立任务与状态依赖，结合日志中的实际 tasks 和 tours 解释预算。",
    ),
    (
        "Where does generation time go?",
        1.5,
        "C1",
        "展示 CPU 的一个注明编号的代表代。"
        "用秒数说明评估、遗传操作和其它开销的比例。"
        "评价是否占主导以实测为准。",
    ),
    (
        "Earlier approaches: workers, arrays and masks",
        1.5,
        "D1",
        "回顾 2024 年的多进程、循环数组化和 visited mask。"
        "说明当时解决的 Python 调度与逐元素循环开销，历史倍率不用于本次比较。",
    ),
    (
        "DEAP manages the population; Numba computes",
        2,
        "D2",
        "区分两条路径：将每棵树编译成函数；将树编码为指令交给已编译解释器。"
        "当前 CPU solver 采用后者。"
        "DEAP 的 gp.compile 本身不生成 Numba 机器码。",
    ),
    (
        "Individual JIT: when does compilation pay off?",
        2,
        "C2",
        "用同一数值函数展示 Python、JIT 含编译和已编译执行的累计时间。"
        "强调热点循环也在 JIT 边界内，微基准倍率不能直接外推为完整仿真倍率。",
    ),
    (
        "From CPU workers to GPU throughput",
        1.5,
        "D5",
        "CPU 用有限数量的核处理任务，GPU 需要足够多的并行工作来隐藏延迟。"
        "任务规模小时，准备和提交开销可能占较大比例。",
    ),
    (
        "Inside an RTX A5000",
        2,
        "D3",
        "说明 CPU 内存、PCIe、显存与 GPU 芯片的关系。"
        "64 个 SM 和 6 MiB L2 来自设备查询，8192 CUDA cores 来自规格表。"
        "线程不是固定绑定的一颗 core。",
    ),
    (
        "Memory hierarchy and data reuse",
        2,
        "D4",
        "说明寄存器、shared memory、L1、L2 和显存各自保存或缓存什么。"
        "L1/shared 是组合资源。"
        "线程局部数组可能溢出到 local memory，不能一概当作寄存器。",
    ),
    (
        "Grid, block, warp and thread",
        2,
        "D5",
        "用一个 program–instance task 放大到 block。"
        "32 ants 乘 8 lanes 是 256 threads，即 8 warps；一个 warp 内有四个 ant groups。"
        "8 lanes 不等于只评估8个候选。",
    ),
    (
        "Why GPU code can still be slow",
        1.5,
        "D4",
        "解释任务不足、warp 内分支分歧、寄存器压力、随机访问和主机提交间隙。"
        "只有实际诊断数据能支持具体瓶颈结论，GPU 利用率并不等于计算单元效率。",
    ),
    (
        "Our CPU–GPU evaluation pipeline",
        2,
        "D6",
        "DEAP 负责选择、交叉和变异。"
        "CPU 编码或生成代码，工具链编译，GPU 执行仿真，最后 CPU 用 FP64 重算返回 tour。"
        "编译成本计入首次使用。",
    ),
    (
        "Programs × instances × ants × candidates",
        2,
        "D5",
        "说明跨任务和任务内两层并行。"
        "当前 Numba 按 instance 并行、instance 内遍历 programs。"
        "GPU 将 task 映射到 block，并在 ant 内使用 lanes 协作。",
    ),
    (
        "Interpreted GP versus generated CUDA",
        1.5,
        "C8",
        "同一棵树，一边是 postfix 指令循环，一边是生成后的标量语句。"
        "解释减少指令解释开销的原因，同时展示编译和装载的成本。",
    ),
    (
        "Resident data and kernel organization",
        1.5,
        "D7",
        "相同问题批次可复用显存数据；更换批次需要新上传。"
        "v2 每个 iteration 有 construct/update。"
        "kernel 数量增加或减少本身不能决定快慢。",
    ),
    (
        "The same evaluations on four backends",
        2.5,
        "C3",
        "用完整冻结 trace 比较 CPU-1、CPU-8、GPU-v1 和 GPU-v2。"
        "相同 GP seed 不能保证跨精度演化轨迹相同，因此严格加速比采用回放。",
    ),
    (
        "Generation time and compilation overhead",
        2,
        "C4 / C5",
        "分开讲真实进化和匹配回放。"
        "观察首代与后续代的编译、实际唯一程序数、树规模和耗时关系。"
        "堆叠图只叠加互斥的墙钟阶段。",
    ),
    (
        "Scaling the population",
        2,
        "C6",
        "同时看 evaluation 秒数和实际 tasks/s。"
        "确认扩大 population 后实际执行程序确实增加，再讨论 GPU 利用和吞吐是否趋于稳定。",
    ),
    (
        "Simulation size and program complexity",
        1.5,
        "C7",
        "比较 TSP50、100、500 的计算时间。"
        "更大的 tour 带来更多步骤和更大状态；不同规模的 task 不是相同计算量。"
        "树复杂度使用节点统计解释。",
    ),
    (
        "Which optimizations matter?",
        1.5,
        "C8",
        "按 fused→v2 解释器→生成式 GP→8 lanes 的顺序解释对照。"
        "第一个变化包含多项内核组织调整，不能归因于单条优化。"
        "其余比较固定其它配置。",
    ),
    (
        "Conclusions and practical limits",
        1,
        "C3 / C6",
        "只填写已经完成的匹配测量结论：整体加速、编译摊销和适合 GPU 的工作量范围。"
        "说明硬件、精度、共享主机和 profiling 权限的实际限制。",
    ),
]


def outline(output):
    text = [
        "# GPU Acceleration for Simulation-based GP",
        "",
        "英文页面与图表；中文讲解稿。正文 36 分钟，讨论 4 分钟。",
        "",
        "结果页只引用本目录已完成的测量；缺失数据保持待补。",
        "",
    ]
    for i, (title, duration, figures, notes) in enumerate(PAGES, start=1):
        text.extend(
            [f"## {i}. {title}", "", f"时间：{duration:g} 分钟。图：{figures}。", "", notes, ""]
        )
    text.extend(
        [
            "## Sources",
            "",
            "- [NVIDIA RTX A5000 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf)",
            "- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)",
            "- [DEAP GP API](https://deap.readthedocs.io/en/master/api/gp.html)",
            "- [Numba Performance Tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)",
        ]
    )
    (output / "presentation_outline_zh.md").write_text("\n".join(text) + "\n")


def profile_figures(output):
    metrics = []
    for path in (output / "profiles").glob("*/timeline.sqlite"):
        with sqlite3.connect(path) as connection:
            names = {
                r[0]
                for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "CUPTI_ACTIVITY_KIND_KERNEL" not in names:
                continue
            intervals = connection.execute(
                "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
            ).fetchall()
        if not intervals:
            continue
        origin = intervals[0][0]
        metrics.append(
            {
                "backend": path.parent.name,
                "kernel_calls": len(intervals),
                "kernel_sum_s": sum(end - start for start, end in intervals) / 1e9,
                "span_s": (intervals[-1][1] - origin) / 1e9,
                "source": str(path.relative_to(output)),
                "diagnostic_only": True,
            }
        )
        figure, ax = new_plot(f"{path.parent.name}: first 30 kernels (diagnostic)", "GPU")
        for start, end in intervals[:30]:
            ax.broken_barh(
                [((start - origin) / 1e6, (end - start) / 1e6)],
                (0, 0.8),
                facecolors=COLORS[path.parent.name],
            )
        ax.set_xlabel("Milliseconds from first kernel")
        ax.set_yticks([])
        save(figure, output / "figures", f"C9_timeline_{path.parent.name}")
    write_csv(output / "profiles/kernel_metrics.csv", metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
    rows, sanity = collect(output)
    write_csv(output / "generations.csv", [r for r in rows if r["experiment_id"] == "E1"])
    write_csv(output / "evaluations.csv", [r for r in rows if r["experiment_id"] != "E1"])
    write_csv(output / "sanity.csv", sanity)
    totals, speedups = performance_figures(output, rows)
    if not (output / "figures/D7_residency_timeline.svg").exists():
        diagrams(output)
    outline(output)
    profile_figures(output)
    lines = [
        "# Presentation experiment results",
        "",
        "自动汇总完成且未检测到同卡外部进程干扰的任务。",
        "",
        f"当前有效测量行：{len(rows)}；功能检查结果：{len(sanity)}。",
        "",
        "## Matched trace results",
        "",
        "| Backend | Repeats | Median evaluation (s) | Speedup vs CPU-8 |",
        "|---|---:|---:|---:|",
    ]
    if (output / "cohort.json").exists():
        lines[2:2] = [
            "本页是独立的 GPU 先行测量组。沿用原冻结源码、输入和预算，不等待 CPU 对照。",
            "不与另一主机的 CPU 时间拼接计算加速比；原同机对照仍独立运行。",
            "",
        ]
    for backend in ("cpu1", "cpu8", "v1", "v2"):
        values = [r["evaluation_wall_s"] for r in totals if r["backend"] == backend]
        ratios = [r["speedup_vs_cpu8"] for r in speedups if r["backend"] == backend]
        if values:
            ratio = f"{np.median(ratios):.2f}×" if ratios else "pending matched reference"
            lines.append(
                f"| {LABELS[backend]} | {len(values)} | {np.median(values):.4f} | {ratio} |"
            )
    lines.extend(
        [
            "",
            "速度比仅使用同主机、同输入哈希、同实际 tasks/tours 的完整回放。",
            "CPU 为 FP64；GPU 搜索为普通 FP32，返回 tour 后 CPU FP64 计分。",
            "图表显示中位数与最小–最大范围；时间重复不是算法效果的独立种子。",
            "",
            "## Available figures",
            "",
        ]
    )
    # 真实演化的时间单独列出，不与固定 trace 的速度比混合。
    position = lines.index("## Available figures")
    progress_lines = [
        "## Real 5-generation runs",
        "",
        "| Backend | Repeats | Median startup (s) | Median 5-generation wall (s) |",
        "|---|---:|---:|---:|",
    ]
    for backend in ("cpu8", "v2"):
        runs = defaultdict(list)
        for row in rows:
            if row["experiment_id"] == "E1" and row["backend"] == backend:
                runs[(row["host"], row["source_hash"], row["repeat_id"])].append(row)
        if runs:
            startup = np.median([v[0]["startup_wall_s"] for v in runs.values()])
            total = np.median([sum(r["generation_wall_s"] for r in v) for v in runs.values()])
            progress_lines.append(
                f"| {LABELS[backend]} | {len(runs)} | {startup:.3f} | {total:.3f} |"
            )
    progress_lines.extend(
        ["", "不足 3 次重复时只是初步数据；不据此宣称稳定加速倍率。", "", "## Queue status", ""]
    )
    state_path = output / "status.json"
    if state_path.exists():
        queue = read_json(state_path)
        progress_lines.extend(
            [
                f"更新时间（UTC）：{queue['updated_at']}",
                "",
                "| Group | Status | Host | Current task |",
                "|---|---|---|---|",
            ]
        )
        for group, state in queue["groups"].items():
            progress_lines.append(
                f"| {group} | {state['status']} | {state.get('host', '—')} | "
                f"{state.get('current_task') or '—'} |"
            )
    progress_lines.append("")
    lines[position:position] = progress_lines
    priority_state = output / "gpu_first/status.json"
    if priority_state.exists():
        queue = read_json(priority_state)
        section = [
            "## GPU-first queue",
            "",
            "[GPU 先行结果与图表](gpu_first/RESULTS.md)独立汇总，不受 CPU 队列进度限制。",
            "",
            "| Group | Status | Host / GPU | Current task |",
            "|---|---|---|---|",
        ]
        for group, state in queue["groups"].items():
            section.append(
                f"| {group} | {state['status']} | {state.get('host', '—')} / "
                f"{state.get('gpu', '—')} | {state.get('current_task') or '—'} |"
            )
        section.append("")
        position = lines.index("## Available figures")
        lines[position:position] = section
    for path in sorted((output / "figures").glob("*.svg")):
        lines.append(f"- [{path.stem}](figures/{path.name})")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        output / "report_manifest.json",
        {
            "measurement_rows": len(rows),
            "sanity_rows": len(sanity),
            "sources": sorted(set(r["provenance"] for r in rows)),
        },
    )


if __name__ == "__main__":
    main()
