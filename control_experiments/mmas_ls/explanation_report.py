"""以组件归因为主线生成中文报告；旧报告归档，所有图表使用完整名称。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import numpy as np

from .common import ROOT, atomic_json, file_hash, now, read_json
from .explanation_analysis import REPORT, WINDOW_FIELDS, SAMPLE_FIELDS
from .explanation_labels import CONDITION_LABELS, MODEL_LABELS, PERIODS, component_columns, benefit_interval
from .report_render import markdown_table, save, interval_band
from .statistics import CONDITIONS, write_csv

OLD_REPORT = ROOT/"control_experiments/mmas_ls/reports/numerical-v1"
ARCHIVE = ROOT/"control_experiments/mmas_ls/reports/numerical-v1-original"
LABELS = {**CONDITION_LABELS, "as_native": "完整 AS"}
METRICS = {
    "source_gap": "实际强化路径的平均 gap（%）",
    "source_advantage_over_current_best_pp": "强化路径优于本轮最优路径的幅度（百分点）",
    "source_age": "强化路径年龄（轮）", "historical_budget_fraction": "用于历史来源的预算比例",
    "pre_mean_gap": "构造路径平均 gap（%）", "post_mean_gap": "2-opt 后路径平均 gap（%）",
    "pre_best_gap": "构造路径最优 gap（%）", "post_best_gap": "2-opt 后本轮最优 gap（%）",
    "pre_top7_gap": "构造路径较优七条平均 gap（%）", "post_top7_gap": "2-opt 后较优七条平均 gap（%）",
    "within_source_tanh_std": "同轮同来源内更新树输出的标准差",
    "ph_saturation": "更新树输出饱和比例", "ls_gain_pp": "2-opt 平均改进（百分点）",
    "retained_fraction": "同一路径经 2-opt 后的边保留率", "improvements_per_100": "每百轮改进全局最优的次数",
    "floor_fraction": "信息素下界触发比例", "restarts_per_100": "每百轮重启次数", "upper_clip_count": "每轮硬上界裁剪次数",
    "source_current_best_overlap": "强化来源与本轮最优路径的边重合率",
    "source_global_best_overlap": "强化来源与历史全局最优路径的边重合率",
    "source_current_best_equal_fraction": "与本轮最优路径完全相同的来源预算比例",
    "source_global_best_equal_fraction": "与历史全局最优路径完全相同的来源预算比例",
    "post_unique_tours": "2-opt 后不同路径数量", "pre_edge_disagreement": "构造路径之间的边不一致率",
    "post_edge_disagreement": "2-opt 后路径之间的边不一致率", "post_effective_edges": "2-opt 后边频率的有效边数",
    "deposit_change_same_sources_budget": "同来源同预算下 GP 沉积相对变化量",
    "deposit_effective_edges": "总沉积的有效边数", "construction_probability_change": "关闭构造树后的选择概率差",
    "construction_argmax_change": "关闭构造树后最大概率城市改变比例",
    "normalized_choice_entropy": "按可行城市数归一化的选择熵", "greedy_fallback_fraction": "候选耗尽后贪心回退比例",
    "source_factor_spatial_std": "同一来源内残差乘数的标准差",
    "history_source_is_different_fraction": "历史来源且不同于本轮最优路径的预算比例",
}


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["FandolHei", "Droid Sans Fallback", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def verified_old_report():
    """归档旧版时验证完整清单；只复制发布产物，不复制大型中间缓存。"""
    source = ARCHIVE if ARCHIVE.exists() else OLD_REPORT
    manifest = read_json(source/"report_manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("旧报告完整性核验未完成")
    for name, expected in manifest["files"].items():
        if file_hash(source/name) != expected:
            raise ValueError(f"旧报告产物已变化：{name}")
    if not ARCHIVE.exists():
        ARCHIVE.mkdir()
        for name in (*manifest["files"], "report_manifest.json"):
            shutil.copy2(source/name, ARCHIVE/name)
    return read_json(ARCHIVE/"statistics.json")


def component_tables(tables):
    rows = []
    for r in tables["factorial"]:
        mean, lo, hi = benefit_interval(r, "delta_pp")
        rows.append({"配置": CONDITION_LABELS[r["condition"]], **component_columns(r["condition"]),
                     "不使用 GP 的 gap（%）": r["baseline_gap"], "使用 GP 的 gap（%）": r["mean_gap"],
                     "GP 带来的改进（百分点）": mean, "同时区间下限": lo, "同时区间上限": hi,
                     "不使用 GP 时移除组件的退化（百分点）": r["baseline_change_pp"],
                     "使用 GP 时移除组件的退化（百分点）": r["gp_change_pp"]})
    effects = {r["contrast"]: r for r in tables["contrasts"]}
    main = []
    for code, label in (("E_R", "重启"), ("E_F", "信息素下界保护"), ("E_H", "历史路径强化")):
        mean, lo, hi = benefit_interval(effects[code])
        main.append({"移除的组件": label, "移除后 GP 增益的变化（百分点）": mean,
                     "同时区间下限": lo, "同时区间上限": hi})
    interactions = []
    names = {
        "J_RF|H=1": "保留历史路径强化时，重启与下界的交互",
        "J_RH|F=1": "保留下界保护时，重启与历史强化的交互",
        "J_FH|R=1": "保留重启时，下界与历史强化的交互",
        "J_RFH": "三个组件的交互",
    }
    for key, title in names.items():
        mean, lo, hi = benefit_interval(effects[key])
        interactions.append({"比较": title, "改进口径的交互效应（百分点）": mean,
                             "同时区间下限": lo, "同时区间上限": hi})
    return rows, main, interactions


def load_deep(target):
    state = read_json(target/"deep_progress.json", {})
    if state.get("status") != "completed":
        return None
    windows = np.full((9, 4, 32, 5, 200, len(WINDOW_FIELDS)), np.nan)
    samples = np.full((9, 4, 32, 5, 201, len(SAMPLE_FIELDS)), np.nan)
    order = (*CONDITIONS, "as_native")
    seen = set()
    for result in state["results"]:
        task = result["task"]; path = Path(result["cache"])
        if file_hash(path) != read_json(path.with_suffix(".json"))["sha256"]:
            raise ValueError("中间分析缓存损坏")
        c = order.index(task["condition"]); r = task["replicate"]
        with np.load(path) as a:
            np.testing.assert_array_equal(a["window_fields"], WINDOW_FIELDS)
            np.testing.assert_array_equal(a["sample_fields"], SAMPLE_FIELDS)
            for j, i in enumerate(task["indices"]):
                if (c, i, r) in seen:
                    raise ValueError("同一配对求解被重复纳入")
                seen.add((c, i, r))
                windows[c, :, i, r] = a["windows"][:, j]
                samples[c, :, i, r] = a["samples"][:, j]
    if not np.isfinite(windows).all() or not np.isfinite(samples).all():
        raise ValueError("深度分析缺少完整实例配对")
    return windows, samples


def deep_tables(target, data):
    summaries = []; stages = []; traces = []; paired = []
    for values, names, times in ((data[0], WINDOW_FIELDS, np.arange(25, 5001, 25)),
                                (data[1], SAMPLE_FIELDS, np.array([1]+list(range(25, 5001, 25))))):
        for c, code in enumerate((*CONDITIONS, "as_native")):
            for p, label in enumerate(("baseline", "81001", "81002", "81003", "GP_mean")):
                v = values[c, p] if p < 4 else values[c, 1:].mean(0)
                for k, metric in enumerate(names):
                    base = {"配置": LABELS[code], "模型": MODEL_LABELS[label], "指标": METRICS[metric]}
                    summaries.append({**base, "均值": float(v[..., k].mean())})
                    for lo, hi in PERIODS:
                        selected = (times >= lo) & (times <= hi)
                        stages.append({**base, "阶段": f"第 {lo}–{hi} 轮", "均值": float(v[:, :, selected, k].mean())})
                    traces.extend({**base, "迭代轮次": int(t), "均值": float(x)}
                                  for t, x in zip(times, v[..., k].mean((0, 1))))
            # 实例配对的历史来源移除效应：三个表达式先平均，不增加样本量。
        for k, metric in enumerate(names):
            for p, label in ((0, "不使用 GP"), (4, "三个 GP 表达式均值")):
                v = values[:, p] if p == 0 else values[:, 1:].mean(1)
                change = (v[1, ..., k]-v[0, ..., k]).mean((1, 2))
                low, high = interval_band(change[:, None])[:, 0]
                paired.append({"模型": label, "指标": METRICS[metric], "关闭历史强化后的变化": float(change.mean()),
                               "点态区间下限": float(low), "点态区间上限": float(high)})
    for name, rows in (("process_summary", summaries), ("process_stages", stages),
                       ("process_curves", traces), ("process_paired_changes", paired)):
        write_csv(target/(name+".csv"), rows)
    return summaries, stages, paired


def plot_quality(target, tables):
    plt = pyplot()
    native = [r for r in tables["main_results"] if r["mode"] == "legacy" and r["model"] in ("baseline", "GP_mean")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for ax, variant in zip(axes, ("as", "mmas")):
        rows = [r for r in native if r["variant"] == variant]
        ax.bar(["不使用 GP", "三个表达式均值"], [r["mean_gap"] for r in rows], color=["#4979a7", "#ca702b"])
        ax.set(title=variant.upper()+" + 2-opt", ylabel="最终参考 gap（%）")
        for i, r in enumerate(rows):
            ax.text(i, r["mean_gap"], f"{r['mean_gap']:.5f}", ha="center", va="bottom")
        ax.set_ylim(0, max(r["mean_gap"] for r in rows)*1.2)
    save(plt, fig, target/"native_quality")
    rows, effects, _ = component_tables(tables)
    fig, ax = plt.subplots(figsize=(8.5, 3.5), layout="constrained")
    x = np.arange(3); mean = np.array([r["移除后 GP 增益的变化（百分点）"] for r in effects])
    lo = np.array([r["同时区间下限"] for r in effects]); hi = np.array([r["同时区间上限"] for r in effects])
    ax.errorbar(mean, x, xerr=[mean-lo, hi-mean], fmt="o", capsize=3)
    ax.axvline(0, color="black", lw=.7); ax.axvspan(-.01, .01, color="gray", alpha=.15)
    ax.set(yticks=x, yticklabels=[r["移除的组件"] for r in effects], xlabel="关闭该组件后，GP 增益的变化（百分点）",
           title="其余组件保持原生配置；原有 18 项比较的 95% 同时区间")
    save(plt, fig, target/"component_effects")
    # 使用旧缓存的完整配对曲线；最终质量表来自已核验发布统计，不重新选择模型。
    with np.load(OLD_REPORT/".cache/quality_arrays.npz") as a:
        curves = a["f_curve"]; native_curves = a["n_curve"]
    times = np.array([1]+list(range(25, 5001, 25)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for v, ax in enumerate(axes):
        for label, values in (("不使用 GP", native_curves[v, 0, 0].mean(1)),
                              ("三个表达式均值", native_curves[v, 0, 1:].mean((0, 2)))):
            lo, hi = interval_band(values)
            ax.plot(times, values.mean(0), label=label); ax.fill_between(times, lo, hi, alpha=.15)
        ax.set(title=("AS", "MMAS")[v]+" + 2-opt", xlabel="迭代轮次", ylabel="截至本轮最优 gap（%）", xscale="log")
        ax.legend()
    save(plt, fig, target/"native_trajectories")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), layout="constrained")
    for c, ax in enumerate(axes.flat):
        for label, v in (("不使用 GP", curves[c, 0].mean(1)), ("三个表达式均值", curves[c, 1:].mean((0, 2)))):
            ax.plot(times, v.mean(0), label=label)
        ax.set(title=CONDITION_LABELS[CONDITIONS[c]], xlabel="迭代轮次", ylabel="截至本轮最优 gap（%）", xscale="log")
    axes.flat[0].legend(); save(plt, fig, target/"all_component_trajectories")


def plot_process(target, data):
    plt = pyplot()
    groups = {
        "reinforcement_sources": ("source_advantage_over_current_best_pp", "source_age", "historical_budget_fraction",
                                  "history_source_is_different_fraction"),
        "actual_rule_effect": ("within_source_tanh_std", "deposit_change_same_sources_budget", "construction_probability_change",
                               "normalized_choice_entropy"),
        "local_search_process": ("pre_mean_gap", "post_mean_gap", "post_edge_disagreement", "improvements_per_100"),
    }
    for name, fields in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for ax, field in zip(axes.flat, fields):
            values, names = (data[0], WINDOW_FIELDS) if field in WINDOW_FIELDS else (data[1], SAMPLE_FIELDS)
            times = np.arange(25, 5001, 25) if field in WINDOW_FIELDS else np.array([1]+list(range(25, 5001, 25)))
            for c in (0, 1):
                for p in (0, 4):
                    v = values[c, p, ..., names.index(field)] if p == 0 else values[c, 1:, ..., names.index(field)].mean(0)
                    label = ("完整配置" if c == 0 else "关闭历史强化") + "；" + ("不使用 GP" if p == 0 else "使用 GP")
                    ax.plot(times, v.mean((0, 1)), label=label, lw=1.1)
            ax.set(title=METRICS[field], xlabel="迭代轮次")
        axes.flat[0].legend(fontsize=7); save(plt, fig, target/name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
    for ax, field in zip(axes, ("deposit_change_same_sources_budget", "construction_probability_change")):
        for p in range(1, 4):
            ax.plot(np.array([1]+list(range(25, 5001, 25))), data[1][0, p, ..., SAMPLE_FIELDS.index(field)].mean((0, 1)),
                    label=MODEL_LABELS[str(81000+p)])
        ax.set(title=METRICS[field], xlabel="迭代轮次"); ax.legend()
    save(plt, fig, target/"individual_expression_effects")


def restart_analysis(target):
    """对齐真实重启事件，先求每次求解内平均，再求实例均值；不把事件当独立样本。"""
    import csv
    from collections import defaultdict
    with (ARCHIVE/"restart_events.csv").open() as stream:
        events = [r for r in csv.DictReader(stream) if int(r["executed"]) == 1]
    with np.load(OLD_REPORT/".cache/quality_arrays.npz") as arrays:
        curves = arrays["f_curve"]
    offsets = (-100, -50, -25, 0, 25, 50, 100, 500)
    times = np.array([1]+list(range(25, 5001, 25)))
    groups = defaultdict(lambda: defaultdict(list))
    for event in events:
        c = CONDITIONS.index(event["condition"]); p = int(event["model"])
        i = int(event["instance"]); r = int(event["replicate"]); iteration = int(event["iteration"])
        anchor = curves[c, p, i, r, np.searchsorted(times, iteration)]
        for offset in offsets:
            t = iteration+offset
            if 1 <= t <= 5000:
                change = curves[c, p, i, r, np.searchsorted(times, t)]-anchor
                groups[(c, p, offset)][(i, r)].append(float(change))
    rows = []
    for (c, p, offset), by_solve in sorted(groups.items()):
        instances = defaultdict(list)
        for (i, _), values in by_solve.items():
            instances[i].append(np.mean(values))
        rows.append({"配置": CONDITION_LABELS[CONDITIONS[c]], "模型": MODEL_LABELS["baseline" if p == 0 else str(81000+p)],
                     "相对重启轮次": offset, "相对重启时最优 gap 的变化（百分点）": float(np.mean([np.mean(v) for v in instances.values()])),
                     "有观测实例数": len(instances), "有观测求解次数": len(by_solve), "重启事件数": sum(map(len, by_solve.values()))})
    write_csv(target/"restart_aligned.csv", rows)
    plt = pyplot(); fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, condition in zip(axes, ("C111", "C110", "C101")):
        for p in (0, 1, 2, 3):
            label = MODEL_LABELS["baseline" if p == 0 else str(81000+p)]
            selected = [r for r in rows if r["配置"] == CONDITION_LABELS[condition] and r["模型"] == label]
            ax.plot([r["相对重启轮次"] for r in selected], [r["相对重启时最优 gap 的变化（百分点）"] for r in selected], label=label)
        ax.axvline(0, color="gray", lw=.7); ax.axhline(0, color="gray", lw=.7)
        ax.set(title=CONDITION_LABELS[condition], xlabel="相对真实重启的轮次", ylabel="最优 gap 相对变化（百分点）")
    axes[0].legend(fontsize=7); save(plt, fig, target/"restart_aligned")


def render(target, tables, data, process):
    components, effects, interactions = component_tables(tables)
    native = []
    for r in tables["main_results"]:
        if r["mode"] != "legacy":
            continue
        native.append({"框架": r["variant"].upper(), "模型": MODEL_LABELS[r["model"]], "gap（%）": r["mean_gap"],
                       "GP 带来的改进（百分点）": 0.0 if r["model"] == "baseline" else -r["delta_pp"],
                       "胜": r["wins"], "平": r["ties"], "负": r["losses"]})
    for name, rows in (("native_results", native), ("component_conditions", components),
                       ("component_effects", effects), ("component_interactions", interactions)):
        write_csv(target/(name+".csv"), rows)
    def table(rows, keys=None):
        return markdown_table(rows, [(k, k) for k in (keys or rows[0].keys())])
    factor = tables["factorial"]
    history = effects[2]
    differences = []
    for key, label in (("E_R-E_H", "历史路径强化的移除效应减重启的移除效应"),
                       ("E_F-E_H", "历史路径强化的移除效应减下界保护的移除效应")):
        r = next(r for r in tables["contrasts"] if r["contrast"] == key)
        # 新口径下的“历史减重启”恰等于旧口径的“重启减历史”，这里不再额外取负。
        differences.append({"直接比较": label, "GP 增益变化之差（百分点）": r["mean_pp"],
                            "同时区间下限": r["ci_low_pp"], "同时区间上限": r["ci_high_pp"]})
    write_csv(target/"component_effect_differences.csv", differences)
    lines = ["# MMAS 的哪些设计影响了 GP 在结合 2-opt 时的增量收益？", "",
        "## 摘要与直接回答", "",
        "现有组件干预中，最强证据指向**使用历史优良路径进行信息素强化**。在保留重启和信息素下界保护时，"
        "关闭历史路径强化，会使不使用 GP 的方法退化得更多，因而扩大 GP 的相对增益。"
        "单独关闭重启或下界保护，没有观察到同等幅度的增益变化。后两者的区间跨零，不代表它们没有作用。", "",
        f"关闭历史路径强化后，GP 增益增加 {history['移除后 GP 增益的变化（百分点）']:.5f} 个百分点，"
        f"95% 同时区间为 [{history['同时区间下限']:.5f}, {history['同时区间上限']:.5f}]。"
        f"但不使用 GP 和使用 GP 的 gap 分别增加 {factor[1]['baseline_change_pp']:.5f} 和 {factor[1]['gp_change_pp']:.5f} 个百分点。"
        "因此，这不是移除组件提高绝对质量的结果。", "",
        "已有证据定位了一个重要组件，但尚未完整证明作用机制。历史强化与 GP 是否提供相似的边偏好、"
        "单路径强化是否限制 GP 的作用范围、2-opt 是否削弱两种策略间的结构差异，需要中间日志与同状态干预共同检验。"
        "报告不把这些解释写成已证实结论。", "",
        "当前结论来自 32 个开发实例、5 个配对求解种子和每框架三个固定表达式。没有重新训练。"
        "独立确认和新增控制实验的结果尚未纳入。", "",
        "## 1. 待解释的现象：增量收益不同，不等于绝对性能排序相同", "", table(native), "",
        "![原生配置最终质量](native_quality.png)", "",
        "AS 中 GP 改善了这些开发实例的平均质量。MMAS 的不使用 GP 对照本身已经更好，固定 GP 表达式的最终均值反而略差。"
        "两种框架使用了各自训练的表达式，蒸发率和强化来源也不同。因此，框架之间的这一结果本身不是单组件因果实验。", "",
        "![随迭代变化的质量](native_trajectories.png)", "",
        f"MMAS 中，使用 GP 的全过程平均最优 gap 为 {factor[0]['gp_auc']:.5f}%，不使用 GP 为 {factor[0]['baseline_auc']:.5f}%。"
        "平均过程表现和最终结果的排序不同，不能写成 GP 在每个阶段都更差。阴影为实例级点态区间，不表示逐轮显著。", "",
        "## 2. 实际实现与控制变量", "",
        "所有运行都是 TSP500、32 只蚂蚁、5000 轮；构造和局部搜索候选数均为 20，每轮全部蚂蚁执行 ACOTSP 风格 2-opt。"
        "AS 的蒸发率为 0.5，MMAS 为 0.2；信息素与距离启发式指数分别为 1 和 2。两棵树的残差系数均为 1/3。", "",
        table([
            {"设计": "强化来源", "AS": "本轮全部 32 条局部搜索后路径", "MMAS": "每轮一条路径；按原生日程选择本轮最优、重启后最优或历史全局最优"},
            {"设计": "信息素下界", "AS": "没有 MMAS 下界保护", "MMAS": "候选有向弧蒸发后执行下界保护"},
            {"设计": "信息素硬上界", "AS": "不执行", "MMAS": "本研究使用的 2-opt 分支不执行硬上界裁剪"},
            {"设计": "重启", "AS": "不执行 MMAS 原生重启", "MMAS": "重置信息素及重启周期的部分历史状态和时钟"},
        ]), "",
        "实际更新顺序是：确定强化来源与预算 → 候选弧蒸发及下界保护 → 沉积 → 重启判断和执行。"
        "因此，不能解释成 GP 刚写入的沉积立即被本轮信息素下界裁掉。后续轮次下界和重启的作用需要另行分析。", "",
        "关闭历史路径强化时仍只强化本轮最优的一条路径，并保留历史最优记录和原生影子来源的预算规则。"
        "它不是删除所有记忆，也不是把 MMAS 变成 AS。来源元数据来自实际强化路径，不来自预算参考路径。", "",
        "两棵 GP 树改变的内容也必须区分。构造树调整当前可行城市的选择分数。更新树只重新分配给定来源路径内部的沉积，"
        "不选择强化来源，也不改变该来源的总沉积预算。对一条来源路径，其边上的沉积可写为：", "",
        r"\[\text{边的沉积}=\text{来源预算}\times\frac{\text{该边的权重}}{\text{来源路径上所有边的权重之和}}.\]", "",
        "边权等于 1 加上更新树输出的双曲正切值的三分之一，范围为 2/3 到 4/3。"
        "关闭更新树后，同一来源的每条边均分预算。多来源时，各来源对同一边的沉积相加。"
        "因此，更新树不能直接强化所有来源都不包含的边。构造树仍可通过后续路径间接改变来源。"
        "这是当前 GP 的作用边界，不是单路径强化一定不利于 GP 的实验证明。", "",
        "gap 等于 100 ×（路径长度／参考长度 − 1）。参考路径是可行参考解，不声称已证明最优。最终长度用 CPU FP64 重算。"
        "下文的 GP 增益等于不使用 GP 的 gap 减使用 GP 的 gap，正值代表改善。", "",
        "## 3. 用组件干预定位差异", "",
        "### 3.1 先比较完整配置附近的单组件变化", "", table(effects), "",
        "![关闭组件对GP增益的影响](component_effects.png)", "",
        "历史路径强化的移除效应大于另外两项的移除效应，原有同时比较支持这个条件下的差别。"
        "这定位的是完整配置附近的条件效应，不是所有背景下普遍不变的组件排名。", "",
        table(differences), "",
        "上表直接检验两个效应之差，不是用“一个显著、另一个不显著”代替效应大小比较。", "",
        "### 3.2 检查绝对退化与组件交互", "",
        table(components, ["配置", "重启", "信息素下界保护", "历史路径强化", "不使用 GP 的 gap（%）",
                           "使用 GP 的 gap（%）", "GP 带来的改进（百分点）", "同时区间下限", "同时区间上限"]), "",
        f"关闭历史路径强化后，GP 的 gap 为 {factor[1]['mean_gap']:.5f}%，仍差于完整 MMAS 不使用 GP 的 {factor[0]['baseline_gap']:.5f}%。"
        "相对优势扩大主要对应对照方法退化更多，不能据此推荐删除历史强化。", "",
        table(interactions), "",
        "下界保护与历史强化、重启与历史强化存在条件交互。三个组件全部关闭时，GP 没有保持关闭历史强化时的平均优势。"
        "因此，结论必须保留其他组件的背景条件。", "",
        "这里的交互是：先计算关闭一个组件改变了多少 GP 增益，再比较另一个组件开启和关闭时，这一变化是否相同。"
        f"例如，只关闭历史强化时 GP 的平均增益为 {-factor[1]['delta_pp']:.5f} 个百分点；"
        f"同时关闭下界保护后，这个增益为 {-factor[3]['delta_pp']:.5f} 个百分点。不能忽略下界保护的背景作用。", "",
        "![全部组件组合的过程曲线](all_component_trajectories.png)", "",
        "## 4. 从中间记录检验作用过程", "",
    ]
    if data is None:
        lines += ["完整日志的新增指标正在重新提取。本节暂不填入未完成的分析；已有最终质量和组件统计已在上文列出。", ""]
    else:
        summary, stages, paired = process
        lookup = {(r["配置"], r["模型"], r["指标"]): r["均值"] for r in summary}
        stage_lookup = {(r["配置"], r["模型"], r["指标"], r["阶段"]): r["均值"] for r in stages}
        def metric(field, model="GP_mean", code="C111"):
            return lookup[(LABELS[code], MODEL_LABELS[model], METRICS[field])]
        stage_rows = []
        for lo, hi in PERIODS:
            period = f"第 {lo}–{hi} 轮"
            row = {"阶段": period}
            for field, label in (("post_best_gap", "2-opt 后本轮最优 gap（%）"),
                                 ("post_unique_tours", "2-opt 后不同路径数"),
                                 ("improvements_per_100", "每百轮改进次数")):
                for model, title in (("baseline", "不使用 GP"), ("GP_mean", "使用 GP")):
                    row[label+"；"+title] = stage_lookup[(LABELS["C111"], MODEL_LABELS[model], METRICS[field], period)]
            stage_rows.append(row)
        write_csv(target/"native_mmas_stages.csv", stage_rows)
        native_process = []
        for field in ("source_gap", "source_advantage_over_current_best_pp", "post_best_gap", "post_unique_tours",
                      "deposit_change_same_sources_budget", "construction_probability_change"):
            row = {"过程指标": METRICS[field]}
            for code, label in (("as_native", "AS"), ("C111", "MMAS")):
                for model, title in (("baseline", "不使用 GP"), ("GP_mean", "使用 GP")):
                    row[label+"；"+title] = metric(field, model, code)
            native_process.append(row)
        write_csv(target/"native_framework_process.csv", native_process)
        # 旧汇总中的连续重复计数已经跨提交块验证。只复用这两项，不复用混合了时间变化的输出方差。
        import csv
        with (ARCHIVE/"mechanism_summary.csv").open() as stream:
            old_process = list(csv.DictReader(stream))
        repeats = []
        for code in ("C111", "C110"):
            for model in ("baseline", "GP_mean"):
                old = next(r for r in old_process if r["condition"] == code and r["model"] == model)
                repeats.append({"配置": LABELS[code], "模型": MODEL_LABELS[model],
                    "实际来源路径改变的轮次比例": float(old["source_switch_fraction"]),
                    "窗口末尾已连续强化同一路径的轮数": float(old["source_repeat_duration_end"])})
        write_csv(target/"source_repetition.csv", repeats)
        def process_table(fields):
            rows = []
            for field in fields:
                r = {"过程指标": METRICS[field]}
                for code, model, title in (("C111", "baseline", "完整配置：不使用 GP"), ("C110", "baseline", "关闭历史强化：不使用 GP"),
                                           ("C111", "GP_mean", "完整配置：使用 GP"), ("C110", "GP_mean", "关闭历史强化：使用 GP")):
                    r[title] = lookup[(LABELS[code], MODEL_LABELS[model], METRICS[field])]
                rows.append(r)
            return table(rows)
        lines += ["### 4.1 历史路径强化提供了怎样的路径来源？", "",
            process_table(("source_advantage_over_current_best_pp", "source_age", "historical_budget_fraction", "history_source_is_different_fraction")), "",
            "![强化来源的变化](reinforcement_sources.png)", "",
            "来源优势为本轮最优路径长度减实际来源长度，再除参考长度并乘 100。正值表示强化来源比当前种群最优路径更好。"
            "多来源指标按实际沉积预算加权。历史标签与不同路径是两回事：表中另外列出历史来源确实不同于本轮最优路径的比例。", "",
            f"完整 MMAS 不使用 GP 时，历史来源获得 {100*metric('historical_budget_fraction','baseline'):.2f}% 的预算，"
            f"但历史来源且与本轮最优路径不同的预算仅占 {100*metric('history_source_is_different_fraction','baseline'):.2f}%。"
            "历史记录可能与本轮重新找到的路径相同。因此，来源年龄大不能直接说明算法一直强化当前种群之外的旧路径。"
            "需要同时检查路径身份、边重合率和质量，才能区分历史来源标签与实际边集差异。", "",
            table(repeats), "",
            "来源切换通过实际来源路径的边集哈希比较，不用来源类型标签代替。连续强化轮数在每个 25 轮窗口末尾取值后平均，"
            "不是完整重复片段的平均持续时间。即使关闭历史强化，本轮种群也可能反复产生同一条最优路径，因而仍会连续强化同一路径。", "",
            "### 4.2 GP 的输出是否变成了实际更新和选择差异？", "",
            process_table(("within_source_tanh_std", "deposit_change_same_sources_budget", "construction_probability_change", "normalized_choice_entropy")), "",
            "![GP的实际作用](actual_rule_effect.png)", "",
            "同来源沉积变化量是实际沉积与关闭更新树后均匀源内沉积的边差绝对值之和，除以相同总预算。"
            "AS 的多个来源先合并到无向边再计算。构造概率差为同一保存上下文下，两组概率差绝对值之和的一半。"
            "关闭构造树的概率是保存基础分数的 CPU FP64 参考归一化，不是另一条实际运行轨迹。", "",
            "![分别观察三个表达式](individual_expression_effects.png)", "",
            "输出饱和不等于 GP 没有作用；共同的边权倍数可能在预算归一化时抵消。三个表达式分开展示，避免总体平均掩盖零构造树或不同更新方式。", "",
            f"关闭历史强化前后，GP 的同来源沉积变化量为 {metric('deposit_change_same_sources_budget'):.5f} 和 "
            f"{metric('deposit_change_same_sources_budget',code='C110'):.5f}；固定上下文的构造概率差为 "
            f"{metric('construction_probability_change'):.5f} 和 {metric('construction_probability_change',code='C110'):.5f}。"
            "两种配置下都存在实际调整，其平均幅度没有出现与最终增益相当的突变。"
            "这不支持把现象简单解释为“历史强化使 GP 输出完全失效”。"
            "但这些量来自各自运行到达的状态，不能替代相同状态下的干预，也不能排除后续更新削弱调整。", "",
            "### 4.3 局部搜索前后发生了什么？", "",
            process_table(("pre_mean_gap", "post_mean_gap", "post_best_gap", "post_unique_tours", "pre_edge_disagreement", "post_edge_disagreement", "improvements_per_100")), "",
            "![局部搜索与后续改进](local_search_process.png)", "",
            "路径之间的边不一致率衡量同一轮蚂蚁之间的多样性，不是 GP 与不使用 GP 之间的差异存活率。"
            "较小的 2-opt 改进可能来自更好的构造路径，也可能来自过早集中；必须与搜索后质量、路径多样性和后续改进共同解释。", "",
            f"关闭历史强化后，不使用 GP 的局部搜索后平均 gap 从 "
            f"{lookup[(LABELS['C111'],MODEL_LABELS['baseline'],METRICS['post_mean_gap'])]:.5f}% 变为 "
            f"{lookup[(LABELS['C110'],MODEL_LABELS['baseline'],METRICS['post_mean_gap'])]:.5f}%；"
            f"使用 GP 时，从 {lookup[(LABELS['C111'],MODEL_LABELS['GP_mean'],METRICS['post_mean_gap'])]:.5f}% 变为 "
            f"{lookup[(LABELS['C110'],MODEL_LABELS['GP_mean'],METRICS['post_mean_gap'])]:.5f}%。"
            "这把最终结果中的差异进一步定位到每轮局部搜索后的路径种群，而不只是最后一条最优路径。"
            "它仍是完整组件干预产生的过程变化，不能据此确定其中某个过程量是唯一中介。", "",
            "### 4.4 为什么每轮路径更好，最终最优解却没有更好？", "", table(stage_rows), "",
            "完整 MMAS 中，使用 GP 的本轮最优路径在四个阶段的平均 gap 都更小，但 2-opt 后不同路径的数量也更少。"
            "最终成绩取 5000 轮中最好的一条路径，不取当前种群均值。较好的典型路径不保证出现更好的极端最优路径。"
            "这两个评价对象的区别，可以解释为什么仅查看训练中的平均过程指标会得出不完整判断。", "",
            "第 251–1000 轮和第 1001–2500 轮，使用 GP 的全局最优改进次数更少。"
            "第 2501–5000 轮则略多。因此，不能概括成 GP 在所有阶段都更少改进，"
            "也不能仅用改进次数推断改进幅度。路径质量提高与多样性减少同时出现，"
            "与搜索过于集中这一解释相容，但尚不能证明多样性下降造成了最终退化。", "",
            "要检验这个解释，需要从同一起点开关两棵树，观察差异能否经过 2-opt 保留，并比较后续最优质量。"
            "若同状态下质量差异与结构差异的变化不一致，就需要否定或修正该解释，而不能只根据相关曲线作结论。", "",
            "### 4.5 历史强化移除后的配对过程变化", "",
            table([r for r in paired if r["指标"] in [METRICS[k] for k in ("post_best_gap", "post_edge_disagreement", "deposit_change_same_sources_budget", "improvements_per_100")]]), "",
            "上述区间按实例配对计算，为事后过程分析的点态区间，不继承组件分析的同时覆盖率。"
            "完整的[阶段统计](process_stages.csv)、[逐模型汇总](process_summary.csv)、[时间曲线数据](process_curves.csv)和"
            "[全部配对变化](process_paired_changes.csv)一并提供。阶段固定为 1–250、251–1000、1001–2500、2501–5000 轮。", "",
            "### 4.6 为什么还需要在 AS 内做反向对照？", "", table(native_process), "",
            "原生 AS 强化全部当前路径，因此实际来源的预算加权平均质量差于本轮最优路径。"
            "原生 MMAS 使用单条择优路径，且部分轮次可以直接使用历史优良路径。"
            "这提供了一个具体解释方向：MMAS 在进入 GP 的边权分配之前，已经执行了更强的路径选择。", "",
            "但是，这张跨框架表不能单独证明路径选择是原因。两种框架的来源数量、蒸发率和已学表达式都不同。"
            "因此，新增实验先在同一个框架内比较全部当前路径与单条当前最优路径，再比较单条当前路径与单条历史路径。"
            "六个表达式跨框架运行，以区分执行框架与表达式来源；不会把迁移结果等同于重新训练结果。", ""]
    lines += ["### 4.7 重启前后的实际改进", "", "![重启事件对齐](restart_aligned.png)", "",
              "每条曲线以真实重启轮次为零点，先在每次求解内平均事件，再按实例平均。三个表达式分别列出。"
              "不同横坐标的可用事件数可能不同，[事件对齐统计](restart_aligned.csv)同时列出实例数和事件数。"
              "这是发生重启条件下的描述，不能把重启后下降的曲线直接解释成重启的因果收益。", ""]
    lines += ["## 5. 哪些解释还需要控制实验？", "",
        table([
            {"解释": "历史强化为不使用 GP 的方法提供了更多质量改进，因此压缩 GP 的增量空间", "当前证据": "组件干预支持收益差异；不是功能重叠的充分证据", "决定性补充": "固定状态改变强化来源；在 AS 内加入相同历史来源规则"},
            {"解释": "MMAS 只强化一条路径，限制了更新树的支持范围", "当前证据": "现有历史来源消融始终是单路径，尚未分离数量", "决定性补充": "同框架比较全部当前路径与单条当前最优路径，匹配预算规则"},
            {"解释": "GP 的调整被归一化、后续下界保护或重启削弱", "当前证据": "可测实际沉积；硬上界裁剪不是当前实现的解释", "决定性补充": "从相同完整状态比较开关更新树，逐阶段追踪信息素与选择概率"},
            {"解释": "2-opt 削弱了 GP 与对照的结构差异", "当前证据": "各自路径的边保留率不能回答这个问题", "决定性补充": "同状态、配对随机数下比较两种策略局部搜索前后的边集差异"},
        ]), "",
        "新增实验按[冻结执行协议](../../MECHANISM_EXPLANATION.md)开展。开发阶段让六个固定表达式跨框架运行；"
        "同状态实验保留局部搜索前后结构和 500 轮续跑；独立确认使用 128 个实例、10 个求解种子。"
        "这些结果未完成前，不把计划写成证据。", "",
        "## 6. 结论", "",
        "第一，在已经完成的开发集组件实验中，历史优良路径强化是影响固定 GP 相对增益最明显的已测组件。"
        "这一结论成立于当前 MMAS 配置附近，不能忽略与下界保护、重启的交互。", "",
        "第二，关闭历史强化没有提高 GP 的绝对质量。它主要使不使用 GP 的方法退化更多。"
        "因此，目前更准确的表述是该组件减少了这些已学表达式的增量收益，而不是该组件本身有害。", "",
        *(["第三，中间日志表明，历史强化移除后的差异已经出现在每轮局部搜索后的路径质量和种群结构中。"
           "完整 MMAS 中，GP 同时伴随较好的典型路径与较低的路径多样性。"
           "这些结果将机制问题缩小到路径来源选择、边权调整与后续搜索之间的关系，"
           "但还没有证明某一个过程量是最终差异的唯一原因。", ""] if data is not None else []),
        "最后，现有结果不能证明所有 GP 规则都不适合 MMAS，也不能证明重新训练无效。"
        "具体作用过程、来源数量的独立作用和 AS 反向干预仍需与新增对照结果一起判断。", "",
        "据此，下一步应优先检验强化来源的选择方式及其与更新树的配合，而不是直接删除重启或下界保护。"
        "本报告尚不支持宣称学习来源选择一定优于固定规则，也不把扩大 GP 相对增益当作提高绝对求解质量。", "",
        "## 附录：统计、数值环境与复现范围", "",
        "实例是统计单位。每实例先平均 5 个求解种子，再平均三个固定表达式。不使用 GP 的对照只有一份。"
        "胜平负以 ±0.01 个百分点划分；误差区间跨零不等于证明无作用。组件结果保留原有 18 项比较的 30,000 次配对 bootstrap 同时区间。", "",
        "本报告上述已完成的组件结果来自原数值实现。原实现存在近常量终端输入的数值稳定性问题。"
        "稳定修正已通过独立输入核验，但没有改善旧表达式的平均质量；这不是稳定输入下重新训练的实验。"
        "新增正式对照采用中心化 FP32，原实现与稳定实现的结果不得混合。", "",
        "新增跨框架验收还发现，500 条来源边的 FP32 顺序累加可产生小幅均值偏移。"
        "首次验收有 13 个未被对应表达式使用的输入超过既定阈值，最大误差为 2.115716×10⁻⁵。"
        "该批次已暂停并保留。后续实现采用显式舍入的补偿求和，不改变终端数学定义或误差阈值，"
        "并重新执行逐型号验收。此修改以独立批次和源码哈希区分，不回写已有质量结果。", "",
        "三个 MMAS 表达式在历史部署检查中均未通过，因此这里报告的是未回退表达式的机制结果，而不是回退部署策略的质量。", "",
        "[原始完整报告与核验记录](../numerical-v1-original/report_zh.md)保留旧发布版本，供审计使用；正文及新图不再使用内部组件简称。", "",
        f"报告生成时间：{now()}。", ""]
    text = "\n".join(lines)
    import re
    if re.search(r"\bC[01]{3}\b|\b[RFH]\s*[=/]|\b(?:TR|PH|IB|RB|GB)\b", text):
        raise ValueError("正文仍含未解释的内部缩写")
    (target/"report_zh.md").write_text(text)


def generate(target=REPORT):
    target = Path(target); target.mkdir(parents=True, exist_ok=True)
    tables = verified_old_report()
    data = load_deep(target)
    process = deep_tables(target, data) if data is not None else None
    plot_quality(target, tables)
    restart_analysis(target)
    if data is not None:
        plot_process(target, data)
    render(target, tables, data, process)
    atomic_json(target/"report_manifest.json", {"status": "existing_evidence_complete" if data is not None else "quality_ready_process_pending",
        "new_experiments": "not_yet_included", "generated_at": now(), "old_statistics_sha256": file_hash(ARCHIVE/"statistics.json"),
        "analysis_sha256": file_hash(Path(__file__).with_name("explanation_analysis.py")), "renderer_sha256": file_hash(__file__),
        "files": {p.name: file_hash(p) for p in sorted(target.iterdir()) if p.is_file() and p.suffix in (".md", ".csv", ".png", ".pdf")}})
    # 原入口只做跳转，不让读者继续读含有旧内部编号的旧正文。
    (OLD_REPORT/"report_zh.md").write_text("# MMAS 组件与 GP 增益的机制分析\n\n"
        "报告已按组件干预与作用过程重写。\n\n"
        "[阅读新版中文报告](../mechanism-explanation-v1/report_zh.md)\n\n"
        "[原发布版本归档](../numerical-v1-original/report_zh.md)\n")
    manifest = read_json(OLD_REPORT/"report_manifest.json")
    manifest["files"]["report_zh.md"] = file_hash(OLD_REPORT/"report_zh.md")
    manifest["report_redirect"] = "../mechanism-explanation-v1/report_zh.md"
    atomic_json(OLD_REPORT/"report_manifest.json", manifest)
    print(target/"report_zh.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--report-dir", type=Path, default=REPORT)
    generate(parser.parse_args().report_dir)
