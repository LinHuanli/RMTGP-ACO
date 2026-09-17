"""新增对照的完整配对统计；阶段未完成时不做选择性显著性分析。"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .common import atomic_json, atomic_npz, digest, evaluation_seed, file_hash, now, read_json, validate_tours
from .explanation_campaign import CONDITION_NAMES, CONFIRMATION_ORDER, confirmation_contrasts
from .report_inputs import checked_status, diagnostic_coverage
from .report_render import markdown_table
from .statistics import simultaneous_interval, write_csv


def load_stage(out, stage):
    out = Path(out)
    queue = [read_json(p) for p in (out/"queue").glob("*.json")]
    queue = [q for q in queue if q["task"]["kind"] == "explanation_"+stage]
    expected_tasks = 35 if stage == "development" else 280
    if len(queue) != expected_tasks:
        raise ValueError("阶段队列数量与冻结方案不符")
    split, instances, seeds, models = (("diagnosis_dev", 32, 5, 7) if stage == "development"
                                      else ("confirm_uniform", 128, 10, 4))
    names = tuple(sorted({q["task"]["condition"] for q in queue})) if stage == "development" else CONFIRMATION_ORDER
    gaps = np.full((7, models, instances, seeds), np.nan)
    curves = np.full((7, models, instances, seeds, 201), np.nan)
    times = np.array([1]+list(range(25, 5001, 25)))
    records = read_json(out/f"manifests/{split}.json")
    frozen_models = {m["id"]: m for m in read_json(out/"manifests/checkpoints.json")}
    with np.load(out/f"inputs/{split}.npz", allow_pickle=False) as a:
        coords = a["coords"]; reference = a["reference_length"]; instance_hashes = a["hashes"]
        points = coords[np.arange(instances)[:, None], a["reference_tour"]]
        np.testing.assert_allclose(np.linalg.norm(np.diff(points, axis=1), axis=-1).sum(-1), reference, atol=1e-12, rtol=0)
    input_hash = file_hash(out/f"inputs/{split}.npz")
    seen = set(); sources = {}
    for q in queue:
        task = q["task"]; job = out/"jobs"/task["id"]
        status = checked_status(job, ("raw.npz", "manifest.json", "diagnostics/index.json"))
        meta = read_json(job/"manifest.json")
        if (meta["task"] != task or meta["source"] != q["source_hash"] or meta["seed"] != evaluation_seed(split, task["replicate"])
                or meta["manifest"] != records["manifest_hash"] or meta["input_sha256"] != input_hash):
            raise ValueError("运行配置、数据或随机种子不匹配冻结队列")
        if meta["scientific_hash"] != status["scientific_hash"]:
            raise ValueError("完成状态科学身份不匹配")
        for entry in meta["models"][1:]:
            frozen = frozen_models[entry["id"]]
            if entry["file_hash"] != frozen["file_hash"] or entry["hash"] != frozen["structural_hash"]:
                raise ValueError("模型不是冻结表达式")
        expected_ids = ["baseline"] + [f"{v}-{s}" for v in (("as", "mmas") if stage == "development" else (task["variant"],))
                                         for s in (81001, 81002, 81003)]
        if [e["id"] for e in meta["models"]] != expected_ids:
            raise ValueError("模型顺序或跨框架来源错误")
        aco = meta["aco"]
        expected = {"ants": 32, "iterations": 5000, "candidate_size": 20, "local_search_candidate_size": 20,
                    "local_search": "two_opt", "rho": .5 if task["variant"] == "as" else .2}
        if any(aco.get(k) != v for k, v in expected.items()):
            raise ValueError("确认配置改变了被冻结的控制变量")
        index = read_json(job/"diagnostics/index.json")
        if index["scientific_hash"] != meta["scientific_hash"] or index["horizon"] != 5000:
            raise ValueError("诊断身份或时域不完整")
        diagnostic_coverage(index)
        indices = np.asarray(task["indices"])
        with np.load(job/"raw.npz", allow_pickle=False) as a:
            tour = a["tour"]; validate_tours(tour, 500)
            np.testing.assert_array_equal(a["model_ids"], expected_ids)
            np.testing.assert_array_equal(a["instance_hashes"], instance_hashes[indices])
            np.testing.assert_array_equal(a["reference"], reference[indices])
            points = coords[indices[None, :, None], tour]
            lengths = np.linalg.norm(np.diff(points, axis=2), axis=-1).sum(-1)
            np.testing.assert_allclose(a["length"], lengths, atol=1e-10, rtol=0)
            values = 100*(lengths/reference[indices][None, :]-1)
            np.testing.assert_allclose(a["gap"], values, atol=1e-10, rtol=0)
            anytime = a["anytime"]
            if anytime.shape != (models, len(indices), 5000) or not np.isfinite(anytime).all() or np.any(np.diff(anytime, axis=-1) > 0):
                raise ValueError("曲线长度错误、非有限或非单调")
            c = names.index(task["condition"]); r = task["replicate"]
            for j, i in enumerate(indices):
                if (c, int(i), r) in seen:
                    raise ValueError("结果重复占用实例配对位置")
                seen.add((c, int(i), r)); gaps[c, :, i, r] = values[:, j]
                curves[c, :, i, r] = 100*(anytime[:, j, times-1]/reference[i]-1)
        sources[task["id"]] = status["files"]
    if not np.isfinite(gaps).all() or not np.isfinite(curves).all():
        raise ValueError("阶段缺少完整配对结果")
    return names, gaps, curves, sources


def publish(out, stage):
    """确认数据必须齐全；开发结果中不同训练框架分组，不把迁移表达式混入原组。"""
    out = Path(out); target = out/"reports"/stage
    queue = [read_json(p)["task"] for p in (out/"queue").glob("*.json")]
    selected = [t for t in queue if t["kind"] == "explanation_"+stage]
    if not selected or any(read_json(out/"jobs"/t["id"]/"status.json", {}).get("status") != "completed" for t in selected):
        return None
    signature = digest({t["id"]: file_hash(out/"jobs"/t["id"]/"status.json") for t in selected})
    previous = read_json(target/"summary.json", {})
    if previous.get("signature") == signature and previous.get("status") == "complete":
        if stage == "development":
            publish_matched_states(out)
        return target
    names, gaps, curves, sources = load_stage(out, stage)
    rows = []; per_expression = []
    for c, name in enumerate(names):
        groups = (("AS 训练的三个表达式", slice(1, 4)), ("MMAS 训练的三个表达式", slice(4, 7))) if stage == "development" else (
            (name.split("_")[0].upper()+" 训练的三个表达式", slice(1, 4)),)
        for label, group in groups:
            benefit = (gaps[c, 0]-gaps[c, group].mean(0)).mean(-1)
            rows.append({"执行配置": CONDITION_NAMES[name], "表达式来源": label, "不使用 GP 的 gap（%）": float(gaps[c, 0].mean()),
                "使用 GP 的 gap（%）": float(gaps[c, group].mean()), "GP 带来的改进（百分点）": float(benefit.mean()),
                "胜": int((benefit > .01).sum()), "平": int((np.abs(benefit) <= .01).sum()), "负": int((benefit < -.01).sum())})
        for p in range(1, gaps.shape[1]):
            origin = ("AS" if p < 4 else "MMAS") if stage == "development" else name.split("_")[0].upper()
            per_expression.append({"执行配置": CONDITION_NAMES[name], "训练框架": origin, "表达式": f"第 {(p-1)%3+1} 个固定表达式",
                "gap（%）": float(gaps[c, p].mean()), "GP 带来的改进（百分点）": float((gaps[c, 0]-gaps[c, p]).mean())})
    contrasts = []
    if stage == "confirmation":
        labels, matrix = confirmation_contrasts()
        protocol = read_json(out/"protocol/explanation.json")
        np.testing.assert_array_equal(matrix, protocol["contrast_matrix"])
        if list(labels) != protocol["confirmation_contrasts"]:
            raise ValueError("确认比较不符合预先冻结的定义")
        benefit = (gaps[:, 0]-gaps[:, 1:].mean(1)).mean(-1).T
        mean, low, high, error, critical = simultaneous_interval(benefit@matrix.T, seed=2026091801)
        contrasts = [{"比较": label, "效应（百分点）": float(mean[i]), "同时区间下限": float(low[i]),
                      "同时区间上限": float(high[i])} for i, label in enumerate(labels)]
        write_csv(target/"confirmation_contrasts.csv", contrasts)
    write_csv(target/"results.csv", rows); write_csv(target/"individual_expressions.csv", per_expression)
    atomic_npz(target/"paired_arrays.npz", gap=gaps, curves=curves, conditions=np.array(names))
    def table(values):
        return markdown_table(values, [(k, k) for k in values[0]])
    text = ["# "+("独立确认结果" if stage == "confirmation" else "强化来源对照与固定表达式跨框架结果"), "",
            "GP 增益为不使用 GP 的 gap 减使用 GP 的 gap，正值表示改善。参考长度不声称已证明最优。", "",
            table(rows), "", "## 三个表达式分别统计", "", table(per_expression), ""]
    if contrasts:
        text += ["## 预先冻结的 14 项同时比较", "", table(contrasts), "",
                 "统计单位为 128 个实例；每实例先平均 10 个求解种子，再平均三个固定表达式。", ""]
    else:
        text += ["这组结果使用 32 个开发实例与 5 个种子，不是重新训练。跨框架部署不能替代该框架重新训练的性能。", ""]
    target.mkdir(parents=True, exist_ok=True); (target/"report_zh.md").write_text("\n".join(text))
    atomic_json(target/"summary.json", {"status": "complete", "signature": signature, "stage": stage,
        "source_files": sources, "results": rows, "contrasts": contrasts, "generated_at": now(),
        "analysis_sha256": file_hash(__file__), "diagnostic_scope": "逐任务完整性索引与覆盖已核对；所有原始诊断另行流式哈希审计",
        "fixed_expressions_only": True})
    if stage == "development":
        publish_matched_states(out)
    return target


def publish_matched_states(out):
    """先平均同一实例的种子与表达式；状态生成者和快照轮次分别报告。"""
    from collections import defaultdict
    out = Path(out); target = out/"reports/matched_states"
    all_tasks = [read_json(p)["task"] for p in (out/"queue").glob("*.json")]
    selected = [t for t in all_tasks if t["kind"] == "explanation_fork"]
    if len(selected) != 216 or any(read_json(out/"jobs"/t["id"]/"status.json", {}).get("status") != "completed" for t in selected):
        return None
    signature = digest({t["id"]: file_hash(out/"jobs"/t["id"]/"status.json") for t in selected})
    if read_json(target/"summary.json", {}).get("signature") == signature:
        return target
    parent_tasks = {t["id"]: t for t in all_tasks}
    detail = []; structure = []; immediate = []; grouped = defaultdict(lambda: defaultdict(list))
    for task in selected:
        job = out/"jobs"/task["id"]
        checked_status(job, ("result.npz", "local_search_difference.npz", "immediate_comparisons.json", "manifest.json"))
        meta = read_json(job/"manifest.json")
        if meta["task"] != task or meta["original_horizon"] != 5000:
            raise ValueError("同状态干预元数据不匹配")
        parent = parent_tasks[task["parent"]]
        mode = parent["mechanism"]["terminal_statistics"]
        mode_label = "原数值实现" if mode == "legacy" else "中心化 FP32 补偿求和"
        owner = "不使用 GP 产生的状态" if task["owner"] == 0 else "各 GP 表达式自身产生的状态"
        info = {"数值实现": mode_label, "状态来源": owner, "快照轮次": task["snapshot_iteration"],
                "表达式": f"第 {task['champion']-81000} 个固定表达式", "求解种子编号": task["replicate"]}
        with np.load(job/"result.npz", allow_pickle=False) as a:
            reference = a["reference"]
            for elapsed in task["measurement_steps"]:
                def gap(branch):
                    length = a[branch+"_length"][0] if elapsed == 500 else a[branch+"_anytime"][0, :, elapsed-1]
                    return 100*(length/reference-1)
                native = gap("native_without_gp")-gap("native_both_trees")
                current = gap("current_without_gp")-gap("current_both_trees")
                construction = gap("native_without_gp")-gap("native_construction_only")
                update = gap("native_without_gp")-gap("native_update_only")
                for j, instance in enumerate(task["indices"]):
                    values = {"原生来源下 GP 增益（百分点）": float(native[j]),
                              "仅本轮最优来源下 GP 增益（百分点）": float(current[j]),
                              "移除历史来源后 GP 增益变化（百分点）": float(current[j]-native[j]),
                              "仅构造决策树的增益（百分点）": float(construction[j]),
                              "仅信息素更新树的增益（百分点）": float(update[j])}
                    detail.append({**info, "后续轮数": elapsed, "实例编号": instance, **values})
                    for metric, value in values.items():
                        grouped[(mode_label, owner, task["snapshot_iteration"], elapsed, metric)][instance].append(value)
        with np.load(job/"local_search_difference.npz", allow_pickle=False) as a:
            for elapsed in task["measurement_steps"]:
                prefix = f"native_without_gp__native_both_trees__after{elapsed}"
                before = a[prefix+"_before"]; after = a[prefix+"_after"]; ratio = a[prefix+"_ratio"]
                for j, instance in enumerate(task["indices"]):
                    finite = np.isfinite(ratio[j])
                    structure.append({**info, "后续轮数": elapsed, "实例编号": instance,
                        "局部搜索前边集对称差": float(before[j].mean()), "局部搜索后边集对称差": float(after[j].mean()),
                        "比率定义的蚂蚁数": int(finite.sum()), "分母为零的蚂蚁数": int((~finite).sum()),
                        "已定义比率均值": float(ratio[j, finite].mean()) if finite.any() else None})
        for row in read_json(job/"immediate_comparisons.json")["rows"]:
            immediate.append({**info, **row})
    summaries = []
    rng = np.random.default_rng(2026091802)
    draws = rng.integers(0, 8, size=(10000, 8))
    for (mode, owner, snapshot, elapsed, metric), instances in sorted(grouped.items()):
        if set(instances) != set(range(8)):
            raise ValueError("同状态分析缺少完整八个实例")
        # baseline 状态虽被三个表达式共用，但始终只有八个实例，而不是 24 个。
        values = np.array([np.mean(instances[i]) for i in range(8)])
        lo, hi = np.quantile(values[draws].mean(1), [.025, .975])
        summaries.append({"数值实现": mode, "状态来源": owner, "快照轮次": snapshot, "后续轮数": elapsed,
                          "指标": metric, "均值": float(values.mean()), "点态区间下限": float(lo), "点态区间上限": float(hi)})
    write_csv(target/"quality_per_instance.csv", detail); write_csv(target/"quality_summary.csv", summaries)
    write_csv(target/"local_search_structure.csv", structure)
    atomic_json(target/"immediate_updates.json", {"rows": immediate, "probability_scope": "CPU FP64 固定上下文参考概率，不是 GPU 逐位概率"})
    shown = [r for r in summaries if r["后续轮数"] == 500 and r["指标"] == "移除历史来源后 GP 增益变化（百分点）"]
    target.mkdir(parents=True, exist_ok=True)
    text = "# 同一完整状态下的组件与 GP 干预\n\n"
    text += "所有分支保留原始 5000 轮的时间归一化分母。先平均每实例的配对种子及固定表达式，再统计八个实例。\n\n"
    text += markdown_table(shown, [(k, k) for k in shown[0]])
    text += "\n\n正值表示同一起点下，移除历史来源扩大了 GP 的增量收益。不同快照轮次与状态来源分开报告。"
    text += "区间是探索性点态区间，不属于独立确认实验的 14 项同时比较。\n\n"
    text += "[全部质量统计](quality_summary.csv)、[逐实例结果](quality_per_instance.csv)、"
    text += "[2-opt 前后结构差异](local_search_structure.csv)、[单次更新对照](immediate_updates.json)。\n"
    (target/"report_zh.md").write_text(text)
    atomic_json(target/"summary.json", {"status": "complete", "signature": signature, "tasks": len(selected),
                "instances": 8, "aco_seeds": 3, "analysis_sha256": file_hash(__file__), "generated_at": now(),
                "scope": "固定状态和固定表达式的 500 轮干预；不是重新训练或自然中介效应"})
    return target
