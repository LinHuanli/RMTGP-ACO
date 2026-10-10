"""先整理真实已有数据；历史协议、当前性能、质量基线和插桩诊断严格分表。"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpaco.artifact_registry import resolve
from gpaco.data import ROOT, validate_tours, write_json
from gpaco.hardware_inputs import file_hash


def export(directory, name, rows):
    directory = directory / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"{name}.json", rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (directory / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        if keys:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )


def save(fig, directory, name):
    directory = directory / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        target = directory / f"{name}.{suffix}"
        fig.savefig(target, dpi=160)
        if suffix == "svg":
            # Matplotlib的路径默认留行尾空格；统一格式后再计算报告SHA。
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def history(directory, old):
    """仅显式导入已知导出文件，原始工作目录只读；不加载旧 Python 代码。"""
    paths = [
        "RESULTS.md",
        "config.yaml",
        "trace_totals.csv",
        "review_v1/RESULTS.md",
        "review_v1/tsp500-pair_totals.csv",
        "review_v1/exports/measurements.csv",
        "review_v1/exports/jit_microbench.csv",
        "review_v1/exports/profile_captures.csv",
        "review_v1/exports/profile_kernels.csv",
    ]
    records = []
    for name in paths:
        source = old / name
        if not source.is_file():
            continue
        target = resolve("E02-p01-historical-inputs") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_hash(target) != file_hash(source):
            raise ValueError("历史导出已改变；必须登记新输入版本，禁止覆盖原始副本")
        if not target.exists():
            shutil.copyfile(source, target)
        records.append(
            {
                "source": str(source),
                "local": str(target.relative_to(ROOT)),
                "sha256": file_hash(source),
            }
        )
    write_json(directory / "historical_provenance.json", records)
    groups = defaultdict(list)
    for relative, label in [
        ("trace_totals.csv", "TSP100 / 5-generation trace"),
        ("review_v1/tsp500-pair_totals.csv", "TSP500 / one evaluation"),
    ]:
        with (old / relative).open() as handle:
            for row in csv.DictReader(handle):
                groups[label, row["backend"]].append(float(row["evaluation_wall_s"]))
    result = [
        {
            "workload": workload,
            "backend": backend,
            "repeats": len(times),
            "median_s": float(np.median(times)),
            "min_s": min(times),
            "max_s": max(times),
            "current_protocol_compatible": False,
            "reason": "双树；CPU FP64；GPU FP32 后主机 FP64 计分；去重/结果复用；不同计时边界",
        }
        for (workload, backend), times in groups.items()
    ]
    export(directory, "historical_timing", result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, workload in zip(axes, dict.fromkeys(r["workload"] for r in result), strict=True):
        rows = [r for r in result if r["workload"] == workload]
        axis.bar([r["backend"] for r in rows], [r["median_s"] for r in rows])
        axis.set(title=workload, ylabel="Historical evaluation wall (s)", yscale="log")
    fig.suptitle("Historical protocol only; not ratios against current GPU results")
    save(fig, directory, "historical_timings")
    return result


def current(directory, diagnostic_directory, campaign):
    queue = json.loads((campaign / "queue.json").read_text())
    measurements, aco, profiles = [], [], []
    sources, aco_seen, exclusions = [], set(), []
    for task in queue:
        if task["kind"] == "prepare":
            continue
        for attempt in task["attempts"]:
            base = Path(attempt["job_path"]).parent
            if not (base / "cohort.json").exists():
                continue
            cohort = json.loads((base / "cohort.json").read_text())
            zero_indices = [i for i, p in enumerate(cohort) if p["expression"] == "ZERO"]
            for path in sorted((base / "measurements").glob("*/record.json")):
                row = json.loads(path.read_text())
                if (
                    row.get("status") != "completed"
                    or row.get("contended")
                    or row.get("telemetry_errors")
                ):
                    exclusions.append(
                        {
                            "record_path": str(path.relative_to(ROOT)),
                            "status": row.get("status"),
                            "contended": row.get("contended"),
                            "telemetry_errors": row.get("telemetry_errors"),
                            "reason": "未完成、检测到争用或监测错误；保留原始记录，不纳入本报告",
                        }
                    )
                    continue
                if row["program_hashes"] != [p["semantic_hash"] for p in cohort]:
                    raise ValueError(f"测量程序与真实 cohort 不一致：{path}")
                result_path = path.parent / "result.npz"
                if file_hash(result_path) != row["result_sha256"]:
                    raise ValueError(f"结果文件哈希不匹配：{result_path}")
                sources.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": file_hash(path),
                        "result_sha256": row["result_sha256"],
                    }
                )
                with np.load(result_path, allow_pickle=False) as arrays:
                    counts = arrays["diagnostics"].sum(axis=(0, 1), dtype=np.uint64)
                    lengths, tours = arrays["lengths"], arrays["tours"]
                    validate_tours(tours, row["n"])
                tours_count = (
                    row["executed_tasks"] * row["search"]["ants"] * row["search"]["iterations"]
                )
                transitions = tours_count * (row["n"] - 1)
                fields = {
                    "task": task["id"],
                    "n": row["n"],
                    "variant": task["variant"],
                    "generation": task["generation"],
                    "mode": path.parent.name,
                    "block": row["block"],
                    "host": attempt["host"],
                    "gpu_uuid": attempt["gpu_uuid"],
                    "eval_wall_s": row["eval_wall_s"],
                    "device_search_s": row["device_search_s"],
                    "executed_tasks": row["executed_tasks"],
                    "tasks_per_s": row["executed_tasks"] / row["eval_wall_s"],
                    "tours_per_device_s": tours_count / row["device_search_s"],
                    "effective_transitions": transitions,
                    "transitions_per_device_s": transitions / row["device_search_s"],
                    "fallback_events": int(counts[0]),
                    "fallback_event_percent": float(100 * counts[0] / transitions),
                    "legacy_mixed_guard_events": int(counts[1]),
                    "mmas_clipped_edges": int(counts[2]),
                    "mmas_restarts": int(counts[3]),
                    "mean_nodes": float(np.mean([p["node_count"] for p in cohort])),
                    "mean_depth": float(np.mean([p["depth"] for p in cohort])),
                    "dist_rank_program_fraction": float(
                        np.mean([bool(p["required_mask"] & (1 << 3)) for p in cohort])
                    ),
                    "unique_programs": len({p["semantic_hash"] for p in cohort}),
                    "unique_structures": len({p["structural_hash"] for p in cohort}),
                    "pool_reserved_bytes": row.get("gpu_pool_reserved_bytes"),
                    "sampled_gpu_used_bytes": row.get("sampled_peak_gpu_used_bytes"),
                    "energy_j": row.get("energy_j"),
                    "energy_method": row.get("energy_method"),
                    "energy_scope": row.get("energy_scope"),
                    "power_w": row.get("mean_power_w"),
                    "sm_clock_mhz": row.get("mean_sm_clock_mhz"),
                    "active_tasks": row.get("active_tasks"),
                    "task_waves": row.get("waves"),
                    "construct_registers_per_thread": row["kernel_resources"]["v2_construct"][
                        "num_regs"
                    ],
                    "compile_load_warm_s": row["compile_s"],
                    "warmup_s": row["warmup"]["wall_s"],
                    "instrumented": row["instrumented"],
                    "occupancy_measured": None,
                    "dram_bandwidth_measured": None,
                    "record_path": str(path.relative_to(ROOT)),
                }
                if row["instrumented"]:
                    c, u = row["construct_device_s"], row["update_device_s"]
                    profiles.append(
                        {
                            **fields,
                            "construct_device_s": c,
                            "update_device_s": u,
                            "construct_percent_of_recorded_stages": 100 * c / (c + u),
                        }
                    )
                else:
                    measurements.append(fields)
                if (
                    zero_indices
                    and task["generation"] == 1
                    and path.parent.name == "generated"
                    and not row["instrumented"]
                    and task["status"] == "completed"
                ):
                    values = lengths[zero_indices]
                    if not np.all(values == values[0]) or not np.all(
                        tours[zero_indices] == tours[zero_indices[0]]
                    ):
                        raise ValueError("相同 ZERO 程序与共同随机数却产生不同输出，不能合并")
                    reference_path = (
                        campaign
                        / f"inputs/tsp{row['n']}/{task['variant']}/geometry/holdout/reference.npy"
                    )
                    reference = np.load(reference_path)
                    for i, length in enumerate(values[0]):
                        identity = (row["input_manifest_sha256"], row["seed"], i)
                        if identity in aco_seen:
                            continue
                        aco_seen.add(identity)
                        aco.append(
                            {
                                "n": row["n"],
                                "variant": task["variant"],
                                "split": "performance_holdout_32",
                                "block": row["block"],
                                "seed": row["seed"],
                                "instance_index": i,
                                "length_fp32": float(length),
                                "reference_fp32": float(reference[i]),
                                "gap_percent": float(100 * (length - reference[i]) / reference[i]),
                                "replicate_unit": "instance × ACO seed; duplicated ZERO programs removed",
                                "record_path": str(path.relative_to(ROOT)),
                            }
                        )
    export(directory, "gpu_work_metrics", measurements)
    export(diagnostic_directory, "gpu_stage_profiles", profiles)
    export(directory, "excluded_measurements", exclusions)
    write_json(directory / "measurement_sources.json", sources)
    return measurements, aco, profiles


def cached_aco(rows):
    for n in (100, 500):
        base = resolve("E12-p01-cross-gpu") / f"inputs/tsp{n}"
        if not (base / "READY.json").exists():
            continue
        manifest = json.loads((base / "manifest.json").read_text())
        if (
            file_hash(base / "manifest.json")
            != json.loads((base / "READY.json").read_text())["manifest_sha256"]
        ):
            raise ValueError("基线 manifest 校验失败")
        for name, scenario in manifest["scenarios"].items():
            if not scenario["baseline"]:
                continue
            prefix = manifest["geometry"][scenario["geometry"]]["path"]
            rel = f"scenarios/{name}/baseline.npz"
            if (
                file_hash(base / rel) != manifest["files"][rel]
                or file_hash(base / f"{prefix}/reference.npy")
                != manifest["files"][f"{prefix}/reference.npy"]
            ):
                raise ValueError("基线缓存或标签被修改")
            with np.load(base / rel, allow_pickle=False) as data:
                lengths = data["lengths"]
            reference = np.load(base / f"{prefix}/reference.npy")
            for i, length in enumerate(lengths):
                rows.append(
                    {
                        "n": n,
                        "variant": manifest["search"]["variant"],
                        "split": scenario["geometry"].split("-")[0],
                        "scenario": name,
                        "seed": scenario["seed"],
                        "instance_index": i,
                        "instance_id": manifest["geometry"][scenario["geometry"]]["instances"][i],
                        "length_fp32": float(length),
                        "reference_fp32": float(reference[i]),
                        "gap_percent": float(100 * (length - reference[i]) / reference[i]),
                        "source_manifest_sha256": file_hash(base / "manifest.json"),
                        "record_path": str((base / rel).relative_to(ROOT)),
                    }
                )
    return rows


def plots(directory, diagnostic_directory, rows, profiles):
    for n in (100, 500):
        selected = [r for r in rows if r["n"] == n]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for mode, marker in [("generated", "o"), ("interpreted", "x")]:
            sub = [r for r in selected if r["mode"] == mode]
            axes[0].scatter(
                [r["fallback_event_percent"] for r in sub],
                [r["device_search_s"] for r in sub],
                marker=marker,
                label=mode,
            )
            axes[1].scatter(
                [r["mean_nodes"] for r in sub], [r["eval_wall_s"] for r in sub], marker=marker
            )
            axes[2].scatter(
                [r["sampled_gpu_used_bytes"] / 1024**3 for r in sub if r["sampled_gpu_used_bytes"]],
                [r["tasks_per_s"] for r in sub if r["sampled_gpu_used_bytes"]],
                marker=marker,
            )
        axes[0].set(
            xlabel="Fallback events / effective transitions (%)", ylabel="Device search (s)"
        )
        axes[1].set(xlabel="Mean tree nodes", ylabel="Warm backend wall (s)")
        axes[2].set(xlabel="Sampled whole-GPU memory (GiB)", ylabel="Actual tasks / s")
        axes[0].legend()
        fig.suptitle(f"TSP{n}: descriptive associations, not causal attribution")
        save(fig, directory, f"tsp{n}_work_and_resources")
        energy = [r for r in selected if r["energy_j"] is not None]
        if energy:
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            for mode in ("generated", "interpreted"):
                sub = [r for r in energy if r["mode"] == mode]
                ax.scatter(
                    [r["eval_wall_s"] for r in sub], [r["energy_j"] for r in sub], label=mode
                )
            ax.set(
                xlabel="Warm backend wall (s)",
                ylabel="Whole-GPU outer evaluation energy (J)",
                title=f"TSP{n}; not whole-system energy",
            )
            ax.legend()
            save(fig, directory, f"tsp{n}_energy")
        sub = [r for r in profiles if r["n"] == n]
        if sub:
            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            labels = [f"{r['variant']}/g{r['generation']}" for r in sub]
            c = [r["construct_device_s"] for r in sub]
            ax.bar(labels, c, label="Construct incl. features, GP, selection, ACS local update")
            ax.bar(labels, [r["update_device_s"] for r in sub], bottom=c, label="Global update")
            ax.set(
                ylabel="Separate instrumented device time (s)", title=f"TSP{n}: stage attribution"
            )
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=8)
            save(fig, diagnostic_directory, f"tsp{n}_stages")


def cpu_plots(directory, cpu_rows, gpu_rows):
    """有真实完整CPU测量后才生成主加速图；按同一输入身份配对，不混历史协议。"""
    groups = defaultdict(list)
    pairs = []
    for row in cpu_rows:
        key = row["n"], row["search"]["variant"], row["generation"], row["cache_state"], row["host"]
        groups[key].append(row)
        if row["cache_state"] != "warm":
            continue
        for gpu in gpu_rows:
            if (gpu["n"], gpu["variant"], gpu["generation"], gpu["block"]) != (
                row["n"],
                row["search"]["variant"],
                row["generation"],
                row["block"],
            ):
                continue
            record = json.loads((ROOT / gpu["record_path"]).read_text())
            if any(
                row[k] != record[k]
                for k in (
                    "input_manifest_sha256",
                    "program_hashes",
                    "seed",
                    "search",
                    "requested_tasks",
                )
            ):
                continue
            pairs.append(
                {
                    "n": row["n"],
                    "variant": row["search"]["variant"],
                    "generation": row["generation"],
                    "block": row["block"],
                    "cpu_backend": row["backend"],
                    "cores": row["cores"],
                    "gpu_mode": gpu["mode"],
                    "cpu_backend_s": row["eval_wall_s"],
                    "gpu_backend_s": gpu["eval_wall_s"],
                    "cpu_over_gpu": row["eval_wall_s"] / gpu["eval_wall_s"],
                    "cpu_host": row["host"],
                    "gpu_host": gpu["host"],
                    "gpu_uuid": gpu["gpu_uuid"],
                    "pairing": "same workload/seed; different hosts allowed and recorded; not same-host hardware attribution",
                }
            )
    export(directory, "cpu_gpu_matched_pairs", pairs)
    for (n, variant, generation, state, host), rows in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for backend in ("cpu_python", "cpu_existing"):
            sub = [r for r in rows if r["backend"] == backend]
            if not sub:
                continue
            core_counts = sorted({r["cores"] for r in sub})
            median = [
                float(np.median([r["eval_wall_s"] for r in sub if r["cores"] == c]))
                for c in core_counts
            ]
            axes[0].plot(core_counts, median, "o-", label=backend)
            if 1 in core_counts:
                paired_cores, speedups = [], []
                for core in core_counts:
                    ratios = [
                        one["eval_wall_s"] / other["eval_wall_s"]
                        for one in sub
                        if one["cores"] == 1
                        for other in sub
                        if other["cores"] == core
                        and all(
                            one[k] == other[k]
                            for k in (
                                "input_manifest_sha256",
                                "program_hashes",
                                "seed",
                                "requested_tasks",
                            )
                        )
                    ]
                    if ratios:
                        paired_cores.append(core)
                        speedups.append(float(np.median(ratios)))
                axes[1].plot(paired_cores, speedups, "o-", label=backend)
                axes[2].plot(paired_cores, np.asarray(speedups) / paired_cores, "o-", label=backend)
        for ax, ylabel in zip(
            axes,
            ("Evaluation wall (s)", "Same-host paired speedup vs 1-core", "Parallel efficiency"),
            strict=True,
        ):
            ax.set(xlabel="Physical CPU cores", ylabel=ylabel, xticks=[1, 8, 16])
            ax.legend(fontsize=8)
        axes[0].set_yscale("log")
        fig.suptitle(f"{host}: TSP{n}/{variant}/g{generation}/{state}; partial measurements")
        save(fig, directory, f"cpu_tsp{n}_{variant}_g{generation}_{state}_{host}")


def deep_diagnostics(directory):
    rows = []
    for path in sorted(resolve("E01-p01-work-diagnostics").glob("**/record.json")):
        attempt = path.parent.parent
        if (attempt / "job.json").exists():
            complete = attempt / "COMPLETE.json"
            if not complete.exists() or not json.loads(complete.read_text()).get("clean"):
                continue
        r = json.loads(path.read_text())
        if (
            r.get("reduced_budget_smoke")
            or r.get("status") != "completed"
            or r.get("contended")
            or r.get("telemetry_errors")
        ):
            continue
        if file_hash(path.parent / "result.npz") != r["result_sha256"]:
            raise ValueError(f"诊断结果哈希不匹配：{path}")
        pair = json.loads((path.parent / "instrumentation_pair.json").read_text())
        count = r["counts"]
        rows.append(
            {
                "n": r["n"],
                "variant": r["variant"],
                "generation": r["generation"],
                "block": r["block"],
                **count,
                "device_search_instrumented_s": r["device_search_s"],
                "snapshots": r["state_replay"]["snapshots"],
                "fallback_snapshots": r["state_replay"]["fallback_snapshots"],
                "instrumentation_bitwise_equal": pair["bitwise_equal_tours"]
                and pair["bitwise_equal_lengths"],
                "pair_clean": not any(
                    pair[mode].get("contended") or pair[mode].get("telemetry_errors")
                    for mode in ("plain", "instrumented")
                ),
                "record_path": str(path.relative_to(ROOT)),
            }
        )
    export(directory, "detailed_diagnostics", rows)
    if rows:
        groups = (
            [rows]
            if len(rows) <= 12
            else [
                [r for r in rows if (r["n"], r["variant"]) == key]
                for key in sorted({(r["n"], r["variant"]) for r in rows})
            ]
        )
        fig, grid = plt.subplots(
            len(groups), 2, figsize=(12, 4 * len(groups)), squeeze=False, constrained_layout=True
        )
        for group, axes in zip(groups, grid, strict=True):
            diagnostic_panel(group, axes)
        save(fig, directory, "detailed_work_and_sampled_cycles")
    return rows


def diagnostic_panel(rows, axes):
    """按规模/宿主分面；保留每个独立block，不能在同名类别位置重叠。"""
    if rows:
        labels = [f"{r['n']}/{r['variant']}/g{r['generation']}/b{r['block']}" for r in rows]
        x = np.arange(len(rows))
        for index, field in enumerate(
            (
                "candidate_list_probe_positions",
                "fallback_rank_visited_checks",
                "fallback_rank_distance_pairs",
            )
        ):
            axes[0].bar(
                x + (index - 1) * 0.25,
                [r[field] / r["effective_transitions"] for r in rows],
                width=0.25,
                label=field,
            )
        axes[0].set(
            xticks=x, xticklabels=labels, ylabel="Logical source work per effective transition"
        )
        bottom = np.zeros(len(rows))
        fields = (
            "candidate_lane0_cycles_sum",
            "stats_lane0_cycles_sum",
            "feature_gp_score_lane0_cycles_sum",
            "selection_sync_acs_lane0_cycles_sum",
        )
        totals = np.asarray([sum(r[f] for f in fields) for r in rows], float)
        for field in fields:
            height = np.asarray([r[field] for r in rows]) / totals * 100
            # 独立随机block使用数值横坐标；重复类别字符串会把柱子叠在同一位置。
            axes[1].bar(x, height, bottom=bottom, label=field)
            bottom += height
        axes[1].set(
            xticks=x,
            xticklabels=labels,
            ylabel="Instrumented lane-0 cycle distribution (%)",
            title="Not device wall-time shares",
        )
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=6)


def publish(directory, title, tier, experiment, lines, source_rows):
    """每份报告只对应一种证据等级；追溯输入和全部小型衍生输出。"""
    figures = sorted((directory / "figures").glob("*.svg"))
    text = [f"# {title}", "", f"证据等级：`{tier}`；实验：{experiment}；协议：p01。", "", *lines]
    text += ["", "## 图表和原始表", ""]
    text += [f"- [{p.stem}](figures/{p.name})" for p in figures]
    text += [
        "",
        "CSV/JSON在 `tables/`；PDF/SVG/PNG在 `figures/`。来源、筛选规则和文件校验见 `provenance.json`。",
        "",
        "正式结果与本报告分开；未采集值不填零。",
    ]
    (directory / "README.md").write_text("\n".join(text) + "\n")
    inputs = []
    for name in sorted({r["record_path"] for r in source_rows if r.get("record_path")}):
        path = ROOT / name
        inputs.append({"path": name, "sha256": file_hash(path)})
    write_json(
        directory / "provenance.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_tier": tier,
            "experiment_id": experiment,
            "protocol_id": "p01",
            "formal_result": False,
            "script_sha256": file_hash(Path(__file__)),
            "registry_sha256": file_hash(ROOT / "configs/experiments/registry.yaml"),
            "input_records": inputs,
            "filter_policy": "显式登记来源；已完成且未检测到争用/监测错误；smoke与历史协议不进入当前测量；不同split分表",
            "tests_opened": False,
            "exports": {
                str(p.relative_to(directory)): file_hash(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file() and p.name != "provenance.json"
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root")
    args = parser.parse_args()
    directory = ROOT / "docs/results/pilot/E01/p01"
    diagnostic = ROOT / "docs/results/diagnostic/E01/p01"
    historical_dir = ROOT / "docs/results/historical/E02/p01"
    for target in (directory, diagnostic, historical_dir):
        target.mkdir(parents=True, exist_ok=True)
    historical = history(historical_dir, Path(args.historical_root)) if args.historical_root else []
    measurements, aco, profiles = current(directory, diagnostic, resolve("E01-p01-gpu-baselines"))
    cached_aco(aco)
    export(directory, "aco_by_instance", aco)
    groups = defaultdict(list)
    for row in aco:
        groups[row["n"], row["variant"], row["split"]].append(row)
    summaries = []
    for (n, variant, split), rows in sorted(groups.items()):
        summaries.append(
            {
                "n": n,
                "variant": variant,
                "split": split,
                "instance_seed_records": len(rows),
                "aco_seeds": len({r["seed"] for r in rows}),
                "mean_gap_percent": float(np.mean([r["gap_percent"] for r in rows])),
                "standard_test": False,
                "ants": 32,
                "aco_iterations": 500,
                "local_search": False,
            }
        )
    export(directory, "aco_summary", summaries)
    plots(directory, diagnostic, measurements, profiles)
    missing = [
        {
            "backend": backend,
            "physical_cores": cores,
            "current_protocol_status": "not_measured",
            "time_s": None,
        }
        for backend in ("cpu_python", "cpu_existing")
        for cores in (1, 8, 16)
    ]
    cpu_rows = []
    for path in resolve("E01-p01-cpu-baselines").glob("**/record.json"):
        row = json.loads(path.read_text())
        if row.get("status") == "completed":
            if row.get("evidence_tier") != "pilot" or row.get("instrumented"):
                raise ValueError(f"CPU测量证据等级不符：{path}")
            if file_hash(path.parent / "result.npz") != row["result_sha256"]:
                raise ValueError(f"CPU结果哈希不符：{path}")
            cpu_rows.append({**row, "record_path": str(path.relative_to(ROOT))})
    for entry in missing:
        matches = [
            r
            for r in cpu_rows
            if r["backend"] == entry["backend"] and r["cores"] == entry["physical_cores"]
        ]
        if matches:
            entry["current_protocol_status"] = (
                f"{len(matches)} cells available; see cpu_measurements"
            )
    export(directory, "cpu_coverage", missing)
    export(directory, "cpu_measurements", cpu_rows)
    cpu_plots(directory, cpu_rows, measurements)
    detailed = deep_diagnostics(diagnostic)
    lines = [
        "本报告只包含先导测量。5个性能block不是5个GP训练seed。没有CPU时间时不计算GPU/CPU加速比。",
        "",
        "## 普通 ACO 质量基线",
        "",
        "| TSP | 宿主 | 数据范围 | 实例×随机种子记录数 | 平均 gap/% |",
        "|---|---|---|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"| {r['n']} | {r['variant']} | {r['split']} | {r['instance_seed_records']} | {r['mean_gap_percent']:.4f} |"
        )
    lines += [
        "",
        "均为32只蚂蚁、500次迭代、无局部搜索。gap 相对于同一 FP32 最优标签路径长度。ZERO重复个体已合并，不是独立重复。训练场景、验证与性能holdout不混合平均；标准测试未开启。",
        "",
        "## CPU 覆盖",
        "",
        "| 后端 | 物理核 | 当前协议 |",
        "|---|---:|---|",
    ]
    for r in missing:
        lines.append(f"| {r['backend']} | {r['physical_cores']} | {r['current_protocol_status']} |")
    lines += [
        "",
        "历史CPU不在本报告；见 `docs/results/historical/E02/p01`。未采集的六组CPU计时不能用历史数据补齐。",
        "",
        "## 当前证据边界",
        "",
        f"- 已汇总 {len(measurements)} 条无插桩GPU测量。插桩记录另存诊断报告，不作为速度分母。",
        "- 图中的程序规模、fallback与时间关系是描述性关联，不是因果结论。",
        "- 显存为NVML采样的整卡值；内存池保留量单列。能耗是整卡外层评价区间，不是整机能耗。",
        "- 现有compile字段是预热评价中的compile/load调用成本，不是真冷编译。",
    ]
    publish(
        directory,
        "E01／p01：ACO、CPU与GPU基线先导",
        "pilot",
        "E01",
        lines,
        measurements + aco + cpu_rows,
    )
    publish(
        diagnostic,
        "E01／p01：独立瓶颈诊断",
        "diagnostic",
        "E01",
        [
            f"已整理 {len(profiles)} 条独立阶段诊断和 {len(detailed)} 条完整预算细粒度诊断。",
            "",
            "- 构造阶段包含终端、GP、候选选择和同步ACS局部更新，不等于纯GP时间。",
            "- 逻辑工作计数不是硬件指令。插桩lane-0周期分布不是整卡墙钟比例。",
            "- occupancy、DRAM带宽、stall和spill流量尚未实测，不用NVML利用率替代。",
            "- 三迭代smoke位于E00功能验证区，不纳入此处的完整预算诊断。",
            "- 配对路径一致性和快照重放用于检查插桩；不能证明最终求解质量提升。",
        ],
        profiles + detailed,
    )
    if historical:
        provenance = json.loads((historical_dir / "historical_provenance.json").read_text())
        publish(
            historical_dir,
            "E02／p01：旧GP-ACO实验的协议复核",
            "historical",
            "E02",
            [
                "这些是旧双树研究的导出。CPU为FP64，GPU为FP32后主机FP64计分，存在去重和结果复用。",
                "",
                "不能与本轮单树FP32计时直接配对。TSP100五代累积与TSP500一次评价分别绘图。",
                "",
                "历史CPU是Numba，不是纯Python。没有的CPU16/完整Python数据保持缺失。",
                "",
                "导出原件的副本放在 `artifacts/provenance/historical/E02/p01`，不混在当前结果表或图目录。",
            ],
            [{"record_path": r["local"]} for r in provenance],
        )
    print(
        json.dumps(
            {
                "output": [
                    str(p.relative_to(ROOT)) for p in (directory, diagnostic, historical_dir)
                ],
                "gpu_records": len(measurements),
                "profiles": len(profiles),
                "aco_records": len(aco),
            }
        )
    )


if __name__ == "__main__":
    main()
