"""从完整已审计日志提取作用链，保持实例轴并支持 AS 多来源沉积。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
import hashlib
from pathlib import Path
import numpy as np

from .common import ROOT, atomic_json, atomic_npz, digest, file_hash, now, read_json
from .report_inputs import checked_status, diagnostic_coverage

OLD = ROOT / "control_experiments/mmas_ls/artifacts/numerical-v1"
REPORT = ROOT / "control_experiments/mmas_ls/reports/mechanism-explanation-v1"
WINDOW_FIELDS = (
    "source_gap", "source_advantage_over_current_best_pp", "source_age", "historical_budget_fraction",
    "pre_mean_gap", "post_mean_gap", "pre_best_gap", "post_best_gap", "pre_top7_gap", "post_top7_gap",
    "within_source_tanh_std", "ph_saturation", "ls_gain_pp", "retained_fraction",
    "improvements_per_100", "floor_fraction", "restarts_per_100", "upper_clip_count",
)
SAMPLE_FIELDS = (
    "source_current_best_overlap", "source_global_best_overlap", "source_current_best_equal_fraction",
    "source_global_best_equal_fraction", "post_unique_tours", "pre_edge_disagreement", "post_edge_disagreement",
    "post_effective_edges", "deposit_change_same_sources_budget", "deposit_effective_edges",
    "construction_probability_change", "construction_argmax_change", "normalized_choice_entropy",
    "greedy_fallback_fraction", "source_factor_spatial_std", "history_source_is_different_fraction",
)


def canonical_edges(tours, n):
    """无向边的无碰撞整数编码；排序后对旋转及反向保持不变。"""
    tours = np.asarray(tours, dtype=np.int64)
    a, b = tours[..., :-1], tours[..., 1:]
    return np.sort(np.minimum(a, b) * n + np.maximum(a, b), axis=-1)


def colony_structure(tours, n):
    """蚂蚁之间平均边不一致率；不是与另一算法的结构差异。"""
    edges = canonical_edges(tours, n)
    count = np.bincount(edges.ravel(), minlength=n*n)
    ants = len(tours)
    disagreement = 1 - np.sum(count * (count-1)) / (ants * (ants-1) * n) if ants > 1 else 0.
    mass = count[count > 0] / float(ants*n)
    return edges, len(np.unique(edges, axis=0)), float(disagreement), float(1 / np.square(mass).sum())


def fixed_context_probabilities(tr, context):
    """用保存的基础分数关闭构造树；沿用贪心分支与退化分支，不另造候选集合。

    CPU FP64 归一化是固定输入的参考重算，不声称逐位复现 GPU 浮点除法。
    """
    valid = np.isfinite(tr[..., 19])
    actual = np.where(valid, tr[..., 19], 0).astype(np.float64)
    scores = np.where(valid, tr[..., 17], 0).astype(np.float64)
    count = valid.sum(-1)
    if np.any(count < 1) or np.any(scores < 0) or np.any(actual < 0):
        raise ValueError("构造探针无效")
    np.testing.assert_array_equal(count, context[..., 7])
    np.testing.assert_allclose(actual.sum(-1), 1, atol=2e-6, rtol=0)
    total = scores.sum(-1, keepdims=True)
    base = np.divide(scores, total, out=valid.astype(float)/count[..., None], where=total > 1e-12)
    greedy = context[..., 8].astype(bool)
    # 零分数并列按保存的候选存储顺序取首个最大值，和 GPU 扫描顺序对应。
    best = np.where(valid, scores, -np.inf).argmax(-1)
    deterministic = np.zeros_like(base)
    np.put_along_axis(deterministic, best[..., None], 1., axis=-1)
    base = np.where(greedy[..., None], deterministic, base)
    distance = .5*np.abs(actual-base).sum(-1)
    changed = actual.argmax(-1) != base.argmax(-1)
    entropy = -np.sum(actual*np.log(np.maximum(actual, 1e-300)), axis=-1)
    normalized = np.divide(entropy, np.log(np.maximum(count, 2)), out=np.zeros_like(entropy), where=count > 1)
    return distance.mean(-1), changed.mean(-1), normalized.mean(-1)


def sample_metrics(data):
    """来源先按无向边合并再统计沉积，避免把 AS 的同一条边算作不同来源槽。"""
    tv, changed, entropy = fixed_context_probabilities(data["tr"], data["context"])
    rows = []
    for row in range(len(data["source_valid"])):
        mask = data["source_valid"][row]
        tours = data["source_tours"][row, mask]
        n = tours.shape[-1]-1
        info = data["source_info"][row, mask].astype(float)
        budget = info[:, 5]
        weights = budget / budget.sum()
        sources = canonical_edges(tours, n)
        post, unique, disagreement, effective = colony_structure(data["post_tours"][row], n)
        _, _, before, _ = colony_structure(data["pre_tours"][row], n)
        best = post[int(np.argmin(data["colony_lengths"][row]))]
        global_best = canonical_edges(data["best_tours"][row], n)
        current_overlap = np.isin(sources, best).mean(-1)
        global_overlap = np.isin(sources, global_best).mean(-1)
        same_current = np.all(sources == best, axis=-1)
        same_global = np.all(sources == global_best, axis=-1)
        # 沉积值必须与原始路径的边顺序配对，不能与排序后的边错位。
        a = tours[:, :-1].astype(np.int64); b = tours[:, 1:].astype(np.int64)
        edge = np.minimum(a, b)*n + np.maximum(a, b)
        deposits = data["deposit"][row, mask].astype(float)
        actual = np.bincount(edge.ravel(), weights=deposits.ravel(), minlength=n*n)
        zero = np.bincount(edge.ravel(), weights=np.broadcast_to(budget[:, None]/n, edge.shape).ravel(), minlength=n*n)
        np.testing.assert_allclose(actual.sum(), budget.sum(), rtol=1e-5, atol=1e-9)
        mass = actual / actual.sum()
        factor_std = data["ph"][row, mask, :, 14].astype(float).std(-1)
        rows.append((weights@current_overlap, weights@global_overlap, weights@same_current,
                     weights@same_global, unique, before, disagreement, effective,
                     np.abs(actual-zero).sum()/budget.sum(), 1/np.square(mass).sum(),
                     tv[row], changed[row], entropy[row], data["context"][row, :, 6].mean(),
                     weights@factor_std, weights@((info[:, 0] > 0) & ~same_current)))
    return np.asarray(rows)


def window_metrics(data, reference, previous_best):
    """先计算每一轮、每条来源内部的方差，再作窗口平均，避免混合时间变化。"""
    info = data["source_info"].astype(float)
    valid = np.isfinite(info[..., 5]) & (info[..., 5] > 0)
    info = np.where(valid[..., None], info, 0.)
    budget = info[..., 5]
    weights = budget / budget.sum(-1, keepdims=True)
    moments = np.where(valid[..., None], data["ph_moments"], 0.)
    count = np.maximum(moments[..., 0], 1)
    mean = moments[..., 4] / count
    std = np.sqrt(np.maximum(0, moments[..., 5]/count - mean**2))
    trace = data["trace"].astype(float)
    ls = data["ls"].astype(float)
    ref = reference[:, None]
    source_length = np.sum(weights * info[..., 2], axis=-1)
    def gap(x):
        return 100*(x/ref-1)
    curves = data["anytime"]
    earlier = np.concatenate((previous_best[:, None], curves[:, :-1]), axis=-1)
    improved = curves < earlier
    # 首轮没有既有搜索最优值，不把初始化计为一次搜索改进。
    if data["start"] == 1:
        improved[:, 0] = False
    iterations = np.arange(data["start"], data["start"]+trace.shape[1])
    values = (
        gap(source_length), 100*(trace[..., 17]-source_length)/ref,
        np.sum(weights*(iterations[None, :, None]-info[..., 3]), axis=-1),
        np.sum(weights*(info[..., 0] > 0), axis=-1),
        gap(trace[..., 18]), gap(trace[..., 19]), gap(trace[..., 16]), gap(trace[..., 17]),
        gap(trace[..., 24]), gap(trace[..., 25]), np.sum(weights*std, axis=-1),
        moments[..., 1].sum(-1)/count.sum(-1),
        100*(ls[..., 0]-ls[..., 1]).mean(-1)/ref, (ls[..., 2]/500).mean(-1),
        improved*100., trace[..., 6]/10000, trace[..., 8]*100., trace[..., 7],
    )
    stack = np.stack(values, axis=-1)
    if stack.shape[1] % 25 or not np.isfinite(stack).all():
        raise ValueError("逐轮指标覆盖不完整或出现非有限值")
    return stack.reshape(stack.shape[0], -1, 25, len(WINDOW_FIELDS)).mean(2), curves[:, -1]


def load_checked(directory, name, record, keys):
    """一遍读取同时核验哈希，不因提取新指标而省略原始数据校验。"""
    raw = (directory/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"诊断文件校验失败：{directory/name}")
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        return {k: archive[k] for k in keys}


def jobs(out):
    """全八种 MMAS 条件与原始 AS；不把数值重复对照当作额外独立实例。"""
    result = []
    for path in sorted((Path(out)/"queue").glob("*.json")):
        task = read_json(path)["task"]
        if task["kind"] == "historical_mechanism":
            result.append(task)
        elif task["kind"] == "numeric_pair" and task["variant"] == "as":
            result.append({**task, "id": task["id"]+"--legacy", "condition": "as_native"})
    if len(result) != 60:
        raise ValueError("旧结果中应有 40 个 MMAS 条件任务与 20 个 AS 原数值任务")
    return result


def summarize_one(args):
    out, target, task = args
    out, target = Path(out), Path(target)
    job = out/"jobs"/task["id"]
    checked_status(job, ("raw.npz", "manifest.json", "diagnostics/index.json"))
    directory = job/"diagnostics"
    index = read_json(directory/"index.json")
    flats = diagnostic_coverage(index)
    key = digest({"code": file_hash(__file__), "index": file_hash(directory/"index.json"),
                  "raw": file_hash(job/"raw.npz"), "task": task})
    cache = target/".cache/deep"/(task["id"]+".npz")
    prior = read_json(cache.with_suffix(".json"), {})
    if prior.get("key") == key and cache.exists() and file_hash(cache) == prior.get("sha256"):
        return {"task": task, "cache": str(cache), "reused": True}
    with np.load(job/"raw.npz", allow_pickle=False) as raw:
        alias = raw["behavior_alias"]; reference = raw["reference"]
    b = len(reference); p = int(alias.max())+1
    if flats != set(range(p*b)):
        raise ValueError("行为别名与诊断求解轴不一致")
    windows = np.full((p*b, 200, len(WINDOW_FIELDS)), np.nan)
    samples = np.full((p*b, 201, len(SAMPLE_FIELDS)), np.nan)
    points = np.array([1]+list(range(25, 5001, 25)))
    window_keys = ("source_info", "ph_moments", "trace", "ls", "anytime")
    sample_keys = ("source_valid", "source_tours", "source_info", "post_tours", "pre_tours", "colony_lengths",
                   "best_tours", "deposit", "ph", "tr", "context")
    for shard in index["shards"]:
        records = [(k, v) for k, v in index["files"].items() if v["metadata"].get("shard") == shard]
        prev = None
        for name, record in sorted((r for r in records if r[1]["metadata"]["kind"] == "iterations"),
                                   key=lambda r: r[1]["metadata"]["start"]):
            meta = record["metadata"]; selected = np.asarray(meta["flat_indices"])
            data = load_checked(directory, name, record, window_keys)
            data["start"] = meta["start"]
            if prev is None:
                prev = data["anytime"][:, 0]
            values, prev = window_metrics(data, reference[selected % b], prev)
            lo = (meta["start"]-1)//25
            windows[selected, lo:lo+values.shape[1]] = values
        for name, record in (r for r in records if r[1]["metadata"]["kind"] == "sample"):
            meta = record["metadata"]
            data = load_checked(directory, name, record, sample_keys)
            samples[np.asarray(meta["flat_indices"]), np.searchsorted(points, meta["iteration"])] = sample_metrics(data)
    if not np.isfinite(windows).all() or not np.isfinite(samples).all():
        raise ValueError("过程指标存在缺失；禁止用部分任务冒充完整报告")
    atomic_npz(cache, windows=windows.reshape(p, b, 200, -1)[alias],
               samples=samples.reshape(p, b, 201, -1)[alias], window_fields=np.array(WINDOW_FIELDS),
               sample_fields=np.array(SAMPLE_FIELDS), sample_iterations=points)
    atomic_json(cache.with_suffix(".json"), {"key": key, "sha256": file_hash(cache), "task": task,
        "scope": "已完成原实现开发集；固定上下文概率为保存基础分数的 CPU FP64 归一化参考重算"})
    return {"task": task, "cache": str(cache), "reused": False}


def run(out=OLD, target=REPORT, workers=4):
    tasks = jobs(out); results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(summarize_one, (out, target, task)) for task in tasks]
        for future in as_completed(pending):
            results.append(future.result())
            print(f"中间日志分析 {len(results)}/{len(tasks)} {results[-1]['task']['id']}", flush=True)
            atomic_json(Path(target)/"deep_progress.json", {"done": len(results), "total": len(tasks),
                        "status": "running", "updated_at": now()})
    atomic_json(Path(target)/"deep_progress.json", {"done": len(results), "total": len(tasks), "status": "completed",
                                                  "results": results, "completed_at": now()})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OLD)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.report_dir, args.workers)
