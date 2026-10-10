"""按证据等级发布已有结果；不打开测试集，不用未完成seed补正式均值。"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from report_baseline_diagnostics import export, plt, save
from report_hardware_pilot import paired_interval

from gpaco.artifact_registry import resolve
from gpaco.data import ROOT, validate_tours, write_json
from gpaco.hardware_campaign import clean_record
from gpaco.hardware_inputs import file_hash


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def interval(values):
    """以独立根seed重采样；不把代数、个体、实例当独立重复。"""
    values = np.asarray(values, float)
    samples = np.random.default_rng(5301).choice(values, (20000, len(values)))
    low, high = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    return float(values.mean()), float(values.std(ddof=1)), float(low), float(high)


def publish(directory, title, tier, lines, sources):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "figures").mkdir(exist_ok=True)
    figures = sorted((directory / "figures").glob("*.svg"))
    text = [
        f"# {title}",
        "",
        f"证据等级：{tier}。生成时间：{datetime.now(timezone.utc).isoformat()}。",
        "",
        *lines,
    ]
    text += ["", "## 图表", ""] + [f"- [{p.stem}](figures/{p.name})" for p in figures]
    text += [
        "",
        "CSV/JSON在tables；图同时提供PDF/SVG/PNG。输入身份和筛选规则见provenance.json。标准测试未打开。",
    ]
    (directory / "README.md").write_text("\n".join(text) + "\n")
    write_json(
        directory / "provenance.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_tier": tier,
            "formal_result": tier == "formal",
            "tests_opened": False,
            "script_sha256": file_hash(Path(__file__)),
            "input_records": [
                {"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)}
                for p in sorted(set(sources))
            ],
            "filter_policy": "完整结束标记及哈希核验；计时排除争用/监测异常；缺失、不可行和原始失败全部交代；不同证据等级分开",
            "exports": {
                str(p.relative_to(directory)): file_hash(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file() and p.name != "provenance.json"
            },
        },
    )


def checked_record(path):
    row = read(path)
    if row and row.get("status") == "completed":
        if file_hash(path.parent / "result.npz") != row["result_sha256"]:
            raise ValueError(f"输出哈希不匹配：{path}")
        with np.load(path.parent / "result.npz", allow_pickle=False) as data:
            validate_tours(data["tours"], row["n"])
    return row


def paired_gpu():
    root = resolve("E01-p01-gpu-baselines")
    out = ROOT / "docs/results/pilot/E04/p01"
    rows, sources, groups = [], [root / "queue.json"], defaultdict(list)
    for task in read(root / "queue.json"):
        if task["status"] != "completed" or task["kind"] != "pair":
            continue
        base = Path(task["attempts"][-1]["job_path"]).parent
        paths = [base / f"measurements/{mode}/record.json" for mode in ("interpreted", "generated")]
        a, b = [checked_record(p) for p in paths]
        for key in ("input_manifest_sha256", "seed", "program_hashes", "search", "requested_tasks"):
            if a[key] != b[key]:
                raise ValueError(f"解释/JIT配对身份不匹配：{task['id']}/{key}")
        clean = clean_record(a) and clean_record(b)
        row = {
            "task": task["id"],
            "n": task["n"],
            "variant": task["variant"],
            "generation": task["generation"],
            "block": task["block"],
            "clean": clean,
            "interpreter_s": a["eval_wall_s"],
            "jit_s": b["eval_wall_s"],
            "ratio": a["eval_wall_s"] / b["eval_wall_s"],
            "same_tours": a["tours_array_sha256"] == b["tours_array_sha256"],
            "warmup_jit_s": b["warmup"]["wall_s"],
        }
        rows.append(row)
        sources.extend(paths)
        if clean:
            groups[row["n"], row["variant"], row["generation"]].append(row)
    stats = []
    for (n, variant, generation), values in sorted(groups.items()):
        ratio, low, high = paired_interval(
            [v["interpreter_s"] for v in values], [v["jit_s"] for v in values]
        )
        stats.append(
            dict(
                n=n,
                variant=variant,
                generation=generation,
                clean_blocks=len(values),
                expected_blocks=5,
                interpreter_median_s=float(np.median([v["interpreter_s"] for v in values])),
                jit_median_s=float(np.median([v["jit_s"] for v in values])),
                paired_ratio=ratio,
                ci_low=low,
                ci_high=high,
            )
        )
    export(out, "paired_blocks", rows)
    export(out, "performance", stats)
    for n in (100, 500):
        values = [r for r in stats if r["n"] == n]
        if not values:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        x = np.arange(len(values))
        labels = [f"{r['variant']}/g{r['generation']}" for r in values]
        for offset, field, label in [
            (-0.18, "interpreter_median_s", "Interpreter"),
            (0.18, "jit_median_s", "Tree JIT"),
        ]:
            axes[0].bar(x + offset, [v[field] for v in values], 0.36, label=label)
        axes[1].errorbar(
            x,
            [v["paired_ratio"] for v in values],
            yerr=[
                [v["paired_ratio"] - v["ci_low"] for v in values],
                [v["ci_high"] - v["paired_ratio"] for v in values],
            ],
            fmt="o",
            capsize=3,
        )
        axes[1].axhline(1, color="gray", linewidth=0.8)
        for ax in axes:
            ax.set_xticks(x, labels, rotation=40, ha="right")
        axes[0].set_ylabel("Warm evaluation wall (s)")
        axes[0].legend()
        axes[1].set_ylabel("Paired interpreter / JIT ratio (pilot)")
        save(fig, out, f"tsp{n}_jit_pairs")
    lines = [
        "真实AS先导seed1002的第1/25/50代种群，在三种宿主中重放。不是三宿主独立训练，也不是CPU/GPU加速比。",
        "",
        "| n | 宿主 | 代 | 干净block | JIT秒 | 解释/JIT |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {v['n']} | {v['variant']} | {v['generation']} | {v['clean_blocks']}/5 | {v['jit_median_s']:.2f} | {v['paired_ratio']:.3f} |"
        for v in stats
    ]
    publish(out, "E04：已有GPU解释器与树JIT的配对先导", "pilot", lines, sources)


def training(formal):
    root = resolve("E09-p01-formal-gpu-existing" if formal else "E09-p01-training")
    out = ROOT / f"docs/results/{'formal' if formal else 'pilot'}/E09/p01"
    expected = 10 if formal else 3
    rows, histories, sources = [], defaultdict(list), []
    for base in sorted(root.glob("as-tsp*-seed*")):
        run = base / "training" if formal else base
        manifest = read(run / "run_manifest.json")
        if not manifest:
            continue
        complete = read(base / "COMPLETE.json")
        history = read(run / "history.json") or []
        valid = bool(complete) and len(history) == 50
        if valid and formal:
            if (
                file_hash(run / "history.json") != complete["history_sha256"]
                or file_hash(run / "COMPLETE.json") != complete["training_complete_sha256"]
            ):
                raise ValueError("正式训练结束证据不匹配")
            job = read(base / "job.json")
            input_manifest = Path(job["inputs"]) / "manifest.json"
            if file_hash(input_manifest) != complete["input_manifest_sha256"]:
                raise ValueError("正式训练输入身份不匹配")
            sources.extend([base / "job.json", input_manifest])
        val = [h for h in history if h.get("validation_champion_gap_percent") is not None]
        record = {
            "run": base.name,
            "n": manifest["n"],
            "seed": manifest["root_seed"],
            "completed": valid,
            "generations": len(history),
            "timing_eligible": bool(
                complete
                and complete.get(
                    "formal_timing_eligible", complete.get("uninterrupted_timing_sample", False)
                )
            ),
            "training_s": complete.get("training_wall_s") if complete else None,
            "preparation_s": complete.get("input_preparation_wall_s") if complete else None,
            "validation_gap": val[-1]["validation_champion_gap_percent"] if valid else None,
            "validation_baseline_gap": val[-1]["validation_baseline_gap_percent"]
            if valid
            else None,
            "validation_delta_pp": val[-1]["validation_delta_pp"] if valid else None,
        }
        rows.append(record)
        if valid:
            histories[record["n"]].append((record, history))
            sources += [run / "run_manifest.json", run / "history.json", base / "COMPLETE.json"]
    export(out, "runs", rows)
    summary = []
    lines = [
        "固定GPU-Existing对照；无局部搜索。标准测试未开启。验证gap不是最终测试性能，不能据此声明新优化方法加速成立。",
        "",
        "| n | 已完成/预设 | 可用连续计时 | 状态 |",
        "|---|---:|---:|---|",
    ]
    for n in (100, 500):
        runs = [r for r in rows if r["n"] == n]
        values = histories[n]
        complete_set = len(values) == expected
        lines.append(
            f"| {n} | {len(values)}/{expected} | {sum(r['timing_eligible'] for r in runs)} | {'重复齐全' if complete_set else '部分完成，不发布正式总体均值'} |"
        )
        if not complete_set:
            if runs:
                fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
                ax.bar([str(r["seed"]) for r in runs], [r["generations"] for r in runs])
                ax.set(
                    xlabel="Independent root seed",
                    ylabel="Recorded generations",
                    ylim=(0, 52),
                    title=f"TSP{n}: incomplete set; no formal mean",
                )
                ax.axhline(50, color="gray", linestyle="--")
                save(fig, out, f"tsp{n}_progress")
            continue
        for metric in (
            "validation_gap",
            "validation_baseline_gap",
            "validation_delta_pp",
            "training_s",
            "preparation_s",
        ):
            selected = [
                r[metric]
                for r, h in values
                if r[metric] is not None
                and (metric not in ("training_s", "preparation_s") or r["timing_eligible"])
            ]
            if len(selected) < 2:
                continue
            mean, sd, lo, hi = interval(selected)
            summary.append(
                dict(
                    n=n,
                    metric=metric,
                    independent_seeds=len(selected),
                    mean=mean,
                    sd=sd,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for key, label in (
            ("train_best_gap_percent", "Generation best"),
            ("train_median_gap_percent", "Population median"),
            ("train_baseline_gap_percent", "Fixed ACO"),
        ):
            a = np.array([[h[key] for h in history] for r, history in values])
            x = np.arange(1, 51)
            line = axes[0, 0].plot(x, a.mean(0), label=label)[0]
            axes[0, 0].fill_between(
                x,
                a.mean(0) - a.std(0, ddof=1),
                a.mean(0) + a.std(0, ddof=1),
                color=line.get_color(),
                alpha=0.12,
            )
        for key, label in (
            ("validation_champion_gap_percent", "Selected champion"),
            ("validation_baseline_gap_percent", "Fixed ACO"),
        ):
            a = np.array([[h[key] for h in history if h[key] is not None] for r, history in values])
            x = np.arange(5, 51, 5)
            axes[0, 1].errorbar(
                x, a.mean(0), yerr=a.std(0, ddof=1), marker="o", label=label, capsize=2
            )
        for r, history in values:
            if r["timing_eligible"]:
                axes[1, 0].plot(
                    np.arange(1, 51),
                    [h["generation_wall_s"] for h in history],
                    alpha=0.5,
                    label=str(r["seed"]),
                )
            v = [h for h in history if h.get("validation_champion_gap_percent") is not None]
            axes[1, 1].step(
                [h["elapsed_s"] / 60 for h in v],
                [h["validation_champion_gap_percent"] for h in v],
                where="post",
                alpha=0.6,
                label=str(r["seed"]),
            )
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        axes[0, 0].set(
            xlabel="Generation", ylabel="Train gap (%)", title="Mean +/- seed SD; changing batches"
        )
        axes[0, 1].set(
            xlabel="Generation", ylabel="Validation gap (%)", title="Actual validation points only"
        )
        axes[1, 0].set(
            xlabel="Generation", ylabel="Generation wall (s)", title="Continuous timing samples"
        )
        axes[1, 1].set(
            xlabel="Elapsed training (min)",
            ylabel="Validation champion gap (%)",
            title="Not a test / equal-time comparison",
        )
        save(fig, out, f"tsp{n}_training")
    export(out, "summary", summary)
    lines += [
        "",
        "| n | 指标 | 独立seed | 均值 | SD | 95%根seed bootstrap区间 |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines += [
        f"| {r['n']} | {r['metric']} | {r['independent_seeds']} | {r['mean']:.4f} | {r['sd']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |"
        for r in summary
    ]
    publish(
        out, "E09：连续训练、计时边界与验证质量", "formal" if formal else "pilot", lines, sources
    )


def mapping():
    root = resolve("E07-p01-fixed-mapping")
    out = ROOT / "docs/results/pilot/E07/p01"
    rows = []
    sources = []
    groups = defaultdict(list)
    excluded = []
    # a02仅在明确补测后替代同一block的a01，不把attempt当作额外重复。
    by_task = defaultdict(list)
    for base in sorted(root.glob("*-a*")):
        if read(base / "COMPLETE.json"):
            by_task[base.name.rsplit("-a", 1)[0]].append(base)
    for task, attempts in sorted(by_task.items()):
        clean_attempts = [d for d in attempts if read(d / "COMPLETE.json").get("clean")]
        selected = clean_attempts[0] if clean_attempts else attempts[-1]
        job = read(selected / "job.json")
        complete = read(selected / "COMPLETE.json")
        t = job["task"]
        values = {}
        for path in sorted((selected / "measurements").glob("*/record.json")):
            r = checked_record(path)
            sources.append(path)
            if file_hash(path) != complete["record_hashes"][path.parent.name]:
                raise ValueError("映射结束哈希不一致")
            row = dict(
                task=task,
                attempt=selected.name,
                n=t["n"],
                generation=t["generation"],
                block=t["block"],
                lanes=r["plan"]["candidate_lanes"],
                active=r["plan"]["active_tasks"],
                status=r["status"],
                block_clean=bool(complete["clean"]),
                eval_s=r.get("eval_wall_s"),
                reason=r.get("reason"),
                gpu_bytes=r.get("sampled_peak_gpu_used_bytes"),
            )
            rows.append(row)
            values[row["lanes"], row["active"]] = row
        for d in attempts:
            c = read(d / "COMPLETE.json")
            if not c.get("clean"):
                excluded.append(
                    dict(
                        task=task,
                        attempt=d.name,
                        reason="同卡争用或遥测异常，整个配对block排除",
                        replacement=selected.name if complete["clean"] else None,
                    )
                )
        if not complete["clean"]:
            continue
        baseline = values[8, 3200]
        for row in values.values():
            if row["status"] == "completed":
                groups[t["n"], t["generation"], row["lanes"], row["active"]].append(
                    (baseline["eval_s"], row["eval_s"])
                )
    stats = []
    for (n, g, lanes, active), v in sorted(groups.items()):
        ratio, lo, hi = paired_interval([a for a, b in v], [b for a, b in v])
        stats.append(
            dict(
                n=n,
                generation=g,
                lanes=lanes,
                active=active,
                clean_blocks=len(v),
                reference_over_plan=ratio,
                ci_low=lo,
                ci_high=hi,
                median_s=float(np.median([b for a, b in v])),
            )
        )
    export(out, "measurements", rows)
    export(out, "paired_summary", stats)
    export(out, "excluded_attempts", excluded)
    for n in (100, 500):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, g in zip(axes, (1, 25, 50), strict=True):
            a = np.full((4, 3), np.nan)
            for r in stats:
                if (r["n"], r["generation"]) == (n, g):
                    a[(4, 8, 16, 32).index(r["lanes"]), (256, 1024, 3200).index(r["active"])] = r[
                        "reference_over_plan"
                    ]
            im = ax.imshow(np.ma.masked_invalid(a), vmin=0.2, vmax=2, aspect="auto")
            for i in range(4):
                for j in range(3):
                    ax.text(
                        j,
                        i,
                        "not feasible"
                        if np.isnan(a[i, j]) and i == 3
                        else "missing"
                        if np.isnan(a[i, j])
                        else f"{a[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if i < 3 else "black",
                    )
            ax.set(
                xticks=range(3),
                xticklabels=(256, 1024, 3200),
                yticks=range(4),
                yticklabels=(4, 8, 16, 32),
                xlabel="Active task cap",
                ylabel="Candidate lanes",
                title=f"TSP{n} / cohort g{g}",
            )
        fig.colorbar(im, ax=axes, label="Paired reference (8,3200) / plan time")
        save(fig, out, f"tsp{n}_fixed_mapping")
    publish(
        out,
        "E07：固定线程映射和活跃状态先导",
        "pilot",
        [
            f"18个预设block；记录{len(rows)}个计划结果，{len(excluded)}个受污染attempt。32 lanes的不可行原因保留，不当作无限加速。",
            "相同block同卡配对；整个受污染block排除。三个ACO block不是三个GP seed。区间仅探索性。没有在holdout上选择正式执行计划。",
        ],
        sources,
    )


def cross_gpu():
    root = resolve("E12-p01-cross-gpu")
    out = ROOT / "docs/results/pilot/E12/p01"
    sources = []
    rows = []
    lookup = {}
    for p in sorted(root.glob("devices/*/tsp*/holdout/*/record.json")):
        r = checked_record(p)
        model = p.relative_to(root).parts[1]
        sources.append(p)
        row = dict(model=model, **r, clean=clean_record(r))
        rows.append(row)
        if row["clean"]:
            lookup[
                model,
                r["n"],
                r["block"],
                "selected" if r["cell"].endswith("selected") else "default",
            ] = row
    stats = []
    models = sorted({r["model"] for r in rows})
    for model in models:
        for n in (100, 500):
            pairs = []
            for block in range(5):
                a, b = (
                    lookup.get((model, n, block, "default")),
                    lookup.get((model, n, block, "selected")),
                )
                if not a or not b:
                    continue
                if any(
                    a[k] != b[k]
                    for k in ("seed", "program_hashes", "search", "input_manifest_sha256")
                ):
                    raise ValueError("跨卡报告内部配对身份不一致")
                pairs.append((a, b))
            if not pairs:
                continue
            ratio, lo, hi = paired_interval(
                [a["eval_wall_s"] for a, b in pairs], [b["eval_wall_s"] for a, b in pairs]
            )
            stats.append(
                dict(
                    model=model,
                    n=n,
                    clean_blocks=len(pairs),
                    default_s=float(np.median([a["eval_wall_s"] for a, b in pairs])),
                    selected_s=float(np.median([b["eval_wall_s"] for a, b in pairs])),
                    default_over_selected=ratio,
                    ci_low=lo,
                    ci_high=hi,
                    energy_j=float(np.median([b["energy_j"] for a, b in pairs]))
                    if all(b.get("energy_j") is not None for a, b in pairs)
                    else None,
                )
            )
    export(out, "holdout_records", rows)
    export(out, "within_model_pairs", stats)
    hardware, audits, numerical = [], [], []
    for p in sorted(root.glob("devices/*/hardware.json")):
        value = read(p)
        props = value["device_properties"]
        hardware.append(
            dict(
                model=p.parent.name,
                name=value["name"],
                host=value["hostname"],
                gpu_uuid=value["gpu_visible"],
                sm_count=props["multiProcessorCount"],
                compute_capability=f"{props['major']}.{props['minor']}",
                memory_bytes=props["totalGlobalMem"],
                l2_bytes=props["l2CacheSize"],
                nvml_inventory=value["nvml_inventory"],
                counters_measured=False,
            )
        )
        sources.append(p)
    for p in sorted(root.glob("canonical_audit/tsp*/*/seed-*/COMPLETE.json")):
        audit = read(p)
        measurements = [checked_record(q) for q in sorted(p.parent.glob("repeat-*/record.json"))]
        if len(measurements) != 3 or any(
            r["program_hashes"] != [audit["champion_hash"]] for r in measurements
        ):
            raise ValueError("跨卡冠军审计的重复数或候选身份不一致")
        audits.append({**audit, "clean": all(clean_record(r) for r in measurements)})
        sources.extend([p, *p.parent.glob("repeat-*/record.json")])
    for model in models:
        for n in (100, 500):
            for role in ("default", "selected"):
                pairs = [
                    (lookup.get(("a5000", n, block, role)), lookup.get((model, n, block, role)))
                    for block in range(5)
                ]
                pairs = [(a, b) for a, b in pairs if a and b]
                if not pairs:
                    continue
                if any(
                    any(
                        a[k] != b[k]
                        for k in ("seed", "program_hashes", "search", "input_manifest_sha256")
                    )
                    for a, b in pairs
                ):
                    raise ValueError("跨型号配对身份不一致")
                ratio, low, high = paired_interval(
                    [a["eval_wall_s"] for a, b in pairs], [b["eval_wall_s"] for a, b in pairs]
                )
                numerical.append(
                    dict(
                        model=model,
                        n=n,
                        plan_role=role,
                        clean_blocks=len(pairs),
                        a5000_over_device=ratio,
                        ci_low=low,
                        ci_high=high,
                        bitwise_equal_tours=sum(
                            a["tours_array_sha256"] == b["tours_array_sha256"] for a, b in pairs
                        ),
                        bitwise_equal_lengths=sum(
                            a["lengths_array_sha256"] == b["lengths_array_sha256"] for a, b in pairs
                        ),
                        mean_gap_difference_pp=float(
                            np.mean(
                                [b["mean_gap_percent"] - a["mean_gap_percent"] for a, b in pairs]
                            )
                        ),
                    )
                )
    export(out, "hardware", hardware)
    export(out, "canonical_champion_audits", audits)
    export(out, "cross_model_numerical_and_time", numerical)
    for n in (100, 500):
        values = [r for r in stats if r["n"] == n]
        if not values:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        x = np.arange(len(values))
        for off, key, label in [
            (-0.18, "default_s", "Default"),
            (0.18, "selected_s", "Tuning-selected"),
        ]:
            axes[0].bar(x + off, [r[key] for r in values], 0.36, label=label)
        axes[1].bar(x, [r["default_over_selected"] for r in values])
        axes[1].axhline(1, color="gray")
        for ax in axes:
            ax.set_xticks(x, [r["model"] for r in values], rotation=25)
        axes[0].set_ylabel("Holdout warm evaluation (s)")
        axes[0].legend()
        axes[1].set_ylabel("Paired default / selected time")
        save(fig, out, f"tsp{n}_cross_gpu")
    lines = [
        "各型号内部为同输入配对；每型号仅一张指定物理卡，不能把5个ACO随机block解释成5张独立显卡。调优与holdout分开。",
        "",
        "| GPU | n | 干净block | 默认秒 | 调优秒 | 默认/调优 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['model']} | {r['n']} | {r['clean_blocks']}/5 | {r['default_s']:.2f} | {r['selected_s']:.2f} | {r['default_over_selected']:.3f} |"
        for r in stats
    ]
    lines += [
        "",
        "硬件资源与NVML配置见tables/hardware.csv；能耗见within_model_pairs.csv；跨型号相对时间及逐位一致性见cross_model_numerical_and_time.csv；规范执行器的冠军复核见canonical_champion_audits.csv。缺失计数器不填零。",
    ]
    publish(out, "E12：五类GPU留出性能先导", "pilot", lines, sources)


def main():
    paired_gpu()
    training(False)
    training(True)
    mapping()
    cross_gpu()
    print("分类报告已生成；没有打开测试集，没有运行新科学样本。")


if __name__ == "__main__":
    main()
