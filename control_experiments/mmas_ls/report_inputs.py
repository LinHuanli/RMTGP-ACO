"""报告只读输入校验。冻结队列是任务身份来源，当前代码只核对科学语义。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np

from .common import digest, evaluation_seed, file_hash, read_json


def frozen_factorial_tasks(out, split):
    """兼容旧 light 队列与 schema 3 历史队列，不接受其他科学干预。"""
    from rmtgp_aco.mechanisms import MechanismConfig, factorial_conditions
    out = Path(out)
    tasks = []
    for path in sorted((out / "queue").glob("*.json")):
        record = read_json(path)
        task = record["task"]
        if task.get("split") != split or task.get("condition") not in factorial_conditions():
            continue
        if task.get("kind") not in (None, "historical_mechanism"):
            continue
        if path.stem != task["id"] or task.get("variant") != "mmas":
            raise ValueError("全因子任务身份或算法不一致")
        expected = asdict(factorial_conditions()[task["condition"]])
        if asdict(MechanismConfig(**task["mechanism"])) != expected:
            raise ValueError("全因子条件对应的机制不一致")
        if task.get("iterations") != 5000 or task.get("modes") != ["full"]:
            raise ValueError("全因子预算或模型模式不一致")
        audit = task.get("instrumentation")
        if audit != "light" and not (isinstance(audit, dict) and
                audit.get("profile") == "mechanism_v3" and audit.get("schema_version") == 3):
            raise ValueError("未知全因子记录配置")
        tasks.append(task)
    if not tasks:
        raise ValueError("冻结队列中没有匹配的全因子任务")
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("重复任务身份")
    return tasks


def checked_status(path, required=()):
    path = Path(path)
    status = read_json(path / "status.json", {})
    if status.get("status") != "completed":
        raise ValueError(f"缺少完成结果: {path}")
    if not set(required).issubset(status.get("files", {})):
        raise ValueError(f"缺少结果哈希: {path}")
    for name, expected in status.get("files", {}).items():
        if file_hash(path / name) != expected:
            raise ValueError(f"缓存结果损坏: {path / name}")
    return status


def validate_manifest(out, task, meta, source_hash=None):
    """检查实际执行参数、输入和模型。既不加载 pickle，也不启动求解器。"""
    out = Path(out)
    if meta["task"] != task:
        raise ValueError("任务规格与冻结队列不一致")
    if meta["seed"] != evaluation_seed(task["split"], task["replicate"]):
        raise ValueError("配对随机种子不一致")
    split = read_json(out / f"manifests/{task['split']}.json")
    if meta["manifest"] != split["manifest_hash"]:
        raise ValueError("实例清单不一致")
    if meta["input_sha256"] != file_hash(out / f"inputs/{task['split']}.npz"):
        raise ValueError("输入文件不一致")
    if source_hash is not None and meta["source"] != source_hash:
        raise ValueError("冻结求解源码身份不一致")
    expected = {m["id"]: m for m in read_json(out / "manifests/checkpoints.json")}
    ids = ["baseline"] + [f"{task['variant']}-{s}" for s in (81001, 81002, 81003)]
    if [m["id"] for m in meta["models"]] != ids:
        raise ValueError("冠军身份或顺序不一致")
    for m in meta["models"][1:]:
        frozen = expected[m["id"]]
        if (m["file_hash"] != frozen["file_hash"] or m["hash"] != frozen["structural_hash"]
                or m["seed"] != frozen["seed"] or m["mode"] != "full"):
            raise ValueError("冠军内容不一致")
    aco = meta["aco"]
    fixed = {"variant": task["variant"], "ants": 32, "iterations": 5000,
             "candidate_size": 20, "local_search": "two_opt", "local_search_candidate_size": 20,
             "local_search_profile": "acotsp", "local_search_dlb": True,
             "rho": .2 if task["variant"] == "mmas" else .5,
             "alpha": 1., "beta": 2., "gamma_transition": 1/3, "gamma_pheromone": 1/3,
             "transition_integration": "residual", "pheromone_integration": "budget_residual"}
    if any(aco.get(k) != v for k, v in fixed.items()):
        raise ValueError("ACO 科学参数不一致")
    if meta["runtime"]["cuda_precision"] != "fp32_fast":
        raise ValueError("搜索精度不一致")
    scientific = {k: meta[k] for k in ("task", "seed", "source", "manifest", "models", "protocol",
                                      "trace_schema", "input_sha256")}
    if digest(scientific) != meta["scientific_hash"]:
        raise ValueError("科学配置摘要不一致")


def validate_raw(data, task, records):
    """验证配对轴、gap 与 AUC 定义；曲线是 GPU FP32 incumbent，不冒充 FP64 曲线。"""
    indices = task["indices"]
    n = len(indices)
    expected_ids = ["baseline"] + [f"{task['variant']}-{s}" for s in (81001, 81002, 81003)]
    np.testing.assert_array_equal(data["model_ids"], expected_ids)
    np.testing.assert_array_equal(data["modes"], ["baseline", "full", "full", "full"])
    np.testing.assert_array_equal(data["instance_hashes"], [records[i]["coordinate_hash"] for i in indices])
    for name, shape in (("gap", (4,n)), ("length", (4,n)), ("auc", (4,n)),
                        ("reference", (n,)), ("anytime", (4,n,5000)), ("tour", (4,n,501))):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f"原始数组维度或有限性错误: {name}")
    np.testing.assert_allclose(data["reference"], [records[i]["reference_length"] for i in indices], rtol=0, atol=1e-12)
    np.testing.assert_allclose(data["gap"], 100*(data["length"]/data["reference"]-1), rtol=0, atol=1e-10)
    np.testing.assert_allclose(data["auc"], (100*(data["anytime"]/data["reference"][None,:,None]-1)).mean(-1), rtol=0, atol=1e-10)
    if np.any(np.diff(data["anytime"], axis=-1) > 0):
        raise ValueError("incumbent 曲线非单调")
    from .common import validate_tours
    validate_tours(data["tour"], 500)


def diagnostic_coverage(index):
    """检查所有执行分块的连续覆盖和固定采样；禁止忽略重复/缺失记录。"""
    if index.get("version") != 3 or not index.get("completed"):
        raise ValueError("诊断没有完整提交")
    horizon = index["horizon"]
    flats = set()
    for shard in index["shards"]:
        records = [r["metadata"] for r in index["files"].values() if r["metadata"].get("shard") == shard]
        blocks = sorted((r for r in records if r["kind"] == "iterations"), key=lambda r:r["start"])
        if not blocks:
            raise ValueError("诊断分块无逐轮数据")
        selected = blocks[0]["flat_indices"]
        if len(set(selected)) != len(selected) or flats.intersection(selected):
            raise ValueError("诊断分块重复逻辑求解")
        flats.update(selected)
        cursor = 1
        for r in blocks:
            if r["start"] != cursor or r["end"] < cursor or r["flat_indices"] != selected:
                raise ValueError("诊断轮次重复、缺失或映射变化")
            cursor = r["end"] + 1
        if cursor != horizon + 1:
            raise ValueError("诊断轮次不完整")
        samples = [r for r in records if r["kind"] == "sample"]
        if sorted(r["iteration"] for r in samples) != [1] + list(range(25,horizon+1,25)):
            raise ValueError("诊断采样缺失或重复")
        if any(r["flat_indices"] != selected for r in samples):
            raise ValueError("诊断采样映射变化")
    actual = {r["metadata"]["shard"] for r in index["files"].values() if r["metadata"]["kind"] == "iterations"}
    if actual != set(index["shards"]):
        raise ValueError("诊断含额外分块")
    return flats
