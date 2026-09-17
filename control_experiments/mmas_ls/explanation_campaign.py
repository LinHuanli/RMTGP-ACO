"""组件解释实验的独立冻结队列；验收、开发、同状态实验与确认分阶段放行。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import numpy as np

from .common import ROOT, OUT, atomic_json, atomic_npz, digest, file_hash, now, read_json
from .campaign import ALLOWED_GPU_MODELS, snapshot
from rmtgp_aco.mechanisms import InstrumentationConfig, MechanismConfig

DEFAULT = ROOT/"control_experiments/mmas_ls/artifacts/mechanism-explanation-v2"
MODEL_KEYS = dict(zip(ALLOWED_GPU_MODELS, ("a5000", "ada4000", "a4000")))
CONFIRMATION_ORDER = (
    "mmas_native", "mmas_current_best", "mmas_no_restart", "mmas_no_floor",
    "as_all_current", "as_current_best", "as_history_calendar",
)
CONDITION_NAMES = {
    "mmas_native": "MMAS：完整原生配置", "mmas_current_best": "MMAS：仅强化本轮最优路径",
    "mmas_all_current": "MMAS：强化全部本轮路径", "mmas_history_calendar": "MMAS：固定日程历史路径强化",
    "mmas_no_restart": "MMAS：仅关闭重启", "mmas_no_floor": "MMAS：仅关闭信息素下界保护",
    "as_all_current": "AS：强化全部本轮路径", "as_current_best": "AS：仅强化本轮最优路径",
    "as_history_calendar": "AS：固定日程历史路径强化",
}
_READINESS_CACHE = ContextVar("component_readiness_cache", default=None)


@contextmanager
def readiness_batch():
    """同一次资源扫描共享只读门禁结果；退出后清空，不跨扫描沿用旧状态。"""
    token = _READINESS_CACHE.set({})
    try:
        yield
    finally:
        _READINESS_CACHE.reset(token)


def readiness_json(path, default=None):
    cache = _READINESS_CACHE.get()
    if cache is None:
        return read_json(path, default)
    key = ("json", str(path))
    if key not in cache:
        cache[key] = read_json(path, default)
    return cache[key]


def conditions(mode="centered_fp32"):
    base = MechanismConfig(terminal_statistics=mode)
    return {
        "mmas_native": base, "mmas_current_best": replace(base, source_policy="iteration_best"),
        "mmas_all_current": replace(base, source_policy="top_k", source_count=32),
        "mmas_history_calendar": replace(base, source_policy="global_calendar"),
        "mmas_no_restart": replace(base, restart_policy="off"), "mmas_no_floor": replace(base, floor_scale=0.),
        "as_all_current": base, "as_current_best": replace(base, source_policy="iteration_best"),
        "as_history_calendar": replace(base, source_policy="global_calendar"),
    }


def confirmation_contrasts():
    """14 项比较采用正值为 GP 改善的口径；矩阵顺序独立于文件调度顺序。"""
    eye = np.eye(7); basis = dict(zip(CONFIRMATION_ORDER, eye))
    family = {CONDITION_NAMES[k]+"的 GP 增益": basis[k] for k in CONFIRMATION_ORDER}
    history = basis["mmas_current_best"]-basis["mmas_native"]
    restart = basis["mmas_no_restart"]-basis["mmas_native"]
    floor = basis["mmas_no_floor"]-basis["mmas_native"]
    family.update({
        "MMAS 关闭历史路径强化后 GP 增益的变化": history,
        "MMAS 关闭重启后 GP 增益的变化": restart,
        "MMAS 关闭下界保护后 GP 增益的变化": floor,
        "AS 从全部路径改为本轮最优单路径后 GP 增益的变化": basis["as_current_best"]-basis["as_all_current"],
        "AS 从本轮最优改为固定日程历史来源后 GP 增益的变化": basis["as_history_calendar"]-basis["as_current_best"],
        "MMAS 历史强化移除效应减重启移除效应": history-restart,
        "MMAS 历史强化移除效应减下界移除效应": history-floor,
    })
    return tuple(family), np.stack(list(family.values()))


def tasks():
    result = []
    for model, label in MODEL_KEYS.items():
        result.append({"id": f"explanation-kernel-check-{label}", "kind": "explanation_kernel_validation",
                       "stage": "validation", "split": "diagnosis_dev", "required_gpu_model": model})
        for variant in ("as", "mmas"):
            result.append({"id": f"explanation-input-check-{label}-{variant}", "kind": "explanation_validation",
                           "stage": "validation", "split": "diagnosis_dev", "variant": variant,
                           "required_gpu_model": model, "instances": 32, "steps": 100})
    inst = asdict(InstrumentationConfig(profile="mechanism_v3", schema_version=3))
    dev = tuple(k for k in CONDITION_NAMES if k not in ("mmas_no_restart", "mmas_no_floor"))
    # 日程、来源语义与模型来源进入任务哈希；硬件不进入科学随机种子。
    for rep in range(5):
        order = list(dev)
        np.random.default_rng(2026091800+rep).shuffle(order)
        for name in order:
            result.append({"id": f"source-development-{name}-seed{rep}", "kind": "explanation_development",
                "stage": "development", "split": "diagnosis_dev", "indices": list(range(32)), "replicate": rep,
                "variant": name.split("_")[0], "condition": name, "display_name": CONDITION_NAMES[name],
                "mechanism": asdict(conditions()[name]), "training_variants": ["as", "mmas"],
                "iterations": 5000, "modes": ["full"], "instrumentation": inst})
    heavy = {**inst, "level": "heavy", "snapshot_iterations": [1, 100, 250, 500, 1000, 2500]}
    for mode in ("legacy", "centered_fp32"):
        for rep in range(3):
            parent = f"matched-state-mmas-{mode}-seed{rep}"
            result.append({"id": parent, "kind": "explanation_heavy", "stage": "matched_states", "split": "diagnosis_dev",
                "indices": list(range(8)), "replicate": rep, "variant": "mmas", "condition": "mmas_native",
                "mechanism": asdict(conditions(mode)["mmas_native"]), "iterations": 5000, "modes": ["full"],
                "instrumentation": heavy})
            for iteration in heavy["snapshot_iterations"]:
                for owner, champion in ((0, 81001), (0, 81002), (0, 81003), (1, 81001), (2, 81002), (3, 81003)):
                    result.append({"id": f"{parent}-iteration{iteration}-owner{owner}-expression{champion}",
                        "kind": "explanation_fork", "stage": "matched_interventions", "split": "diagnosis_dev",
                        "indices": list(range(8)), "replicate": rep, "variant": "mmas", "parent": parent,
                        "snapshot_iteration": iteration, "owner": owner, "champion": champion,
                        "continuation_iterations": 500, "measurement_steps": [1, 25, 100, 500]})
    for rep in range(10):
        for start in range(0, 128, 32):
            order = list(CONFIRMATION_ORDER)
            np.random.default_rng(2026091900+rep*10+start).shuffle(order)
            for name in order:
                result.append({"id": f"source-confirmation-{name}-seed{rep}-block{start:03d}",
                    "kind": "explanation_confirmation", "stage": "confirmation", "split": "confirm_uniform",
                    "indices": list(range(start, start+32)), "replicate": rep, "variant": name.split("_")[0],
                    "condition": name, "display_name": CONDITION_NAMES[name], "mechanism": asdict(conditions()[name]),
                    "iterations": 5000, "modes": ["full"], "instrumentation": inst})
    return result


def prepare(out=DEFAULT):
    out = Path(out).resolve()
    if (out/"queue").exists():
        raise ValueError("不得覆盖冻结队列；修改实验必须新建批次")
    for relative in ("inputs/diagnosis_dev.npz", "inputs/confirm_uniform.npz", "manifests/diagnosis_dev.json",
                     "manifests/confirm_uniform.json", "manifests/checkpoints.json", "validation/data_provenance.json"):
        src = OUT/relative; dst = out/relative
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    if read_json(out/"validation/data_provenance.json")["status"] != "passed":
        raise ValueError("数据来源核验未通过")
    records = [read_json(out/f"manifests/{s}.json")["records"] for s in ("diagnosis_dev", "confirm_uniform")]
    if tuple(map(len, records)) != (32, 128):
        raise ValueError("冻结开发集或确认集数量不符")
    hashes = [r["coordinate_hash"] for rows in records for r in rows]
    if len(set(hashes)) != 160:
        raise ValueError("开发与确认实例重复")
    names, matrix = confirmation_contrasts()
    # 保存已发现的偏斜来源分布作为固定数值回归输入；不放宽输入核验阈值。
    previous = ROOT/"control_experiments/mmas_ls/artifacts/mechanism-explanation-v1/jobs/explanation-input-check-a5000-mmas-recorded/diagnostics"
    if previous.exists():
        index = read_json(previous/"index.json")
        source_name, record = next((k, v) for k, v in index["files"].items()
            if v["metadata"]["kind"] == "sample" and v["metadata"]["iteration"] == 100
            and 32 in v["metadata"]["flat_indices"])
        if file_hash(previous/source_name) != record["sha256"]:
            raise ValueError("数值回归来源文件损坏")
        row = record["metadata"]["flat_indices"].index(32)
        with np.load(previous/source_name, allow_pickle=False) as arrays:
            regression = arrays["source_edge_tau_before"][row, 0]
        atomic_npz(out/"validation/edge_tau_regression.npz", values=regression)
        atomic_json(out/"validation/edge_tau_regression.json", {"source": str(previous/source_name),
            "source_sha256": record["sha256"], "flat_index": 32, "iteration": 100,
            "input_sha256": file_hash(out/"validation/edge_tau_regression.npz"),
            "failure": "FP32 顺序累加均值的舍入偏移；原误差 2.115716e-5，原阈值保持不变"})
    atomic_json(out/"protocol/explanation.json", {"authorized_scope": "fixed_expression_component_mechanisms",
        "created_at": now(), "retraining_allowed": False, "confirmation_allowed": True,
        "confirmation_order": CONFIRMATION_ORDER, "confirmation_contrasts": names, "contrast_matrix": matrix,
        "bootstrap_replicates": 30000, "bootstrap_seed": 2026091801, "epsilon_pp": .01,
        "development_logical_solves": 7840, "confirmation_logical_solves": 35840,
        "reporting": "正的 GP 增益表示优于同条件不使用 GP 的对照；不把旧数值与稳定数值结果混合",
        "numerical_refinement": "中心化 FP32 使用显式舍入的补偿求和；只改统计计算，不改终端数学定义或核验阈值；由源码哈希区别旧中心化实现",
        "confirmation_release": "全部开发来源对照与同状态实验完成，且本型号本框架验收通过；不按结果显著性调整样本",
        "disk_reserve_bytes": 150*1024**3, "source_policy": "多路径按长度倒数分配同状态原生影子总预算",
        "resource_policy": "只使用用户允许的空闲 A5000、RTX 4000 Ada、A4000；按任务配对保持型号一致"})
    destination, source = snapshot(out)
    frozen = tasks()
    for order, task in enumerate(frozen):
        atomic_json(out/"queue"/(task["id"]+".json"), {"task": task, "order": order,
                    "snapshot": str(destination), "source_hash": source["source_hash"]})
    atomic_json(out/"protocol/queue_freeze.json", {"source": source, "snapshot": str(destination),
        "task_count": len(frozen), "created_at": now(), "stages": dict(Counter(t["stage"] for t in frozen)),
        "confirmation_frozen": True, "task_hashes": {t["id"]: digest(t) for t in frozen}})
    return out


def completed(out, name, validated=False):
    out = Path(out); status = readiness_json(out/"jobs"/name/"status.json", {})
    if status.get("status") != "completed":
        return False
    if validated:
        queued = readiness_json(out/"queue"/(name+".json"), {})
        return (status.get("validation_status") == "passed" and
                status.get("source_hash") == queued.get("source_hash") and
                status.get("task_hash") == digest(queued.get("task")))
    return True


def ready(task, out, gpu_model=None):
    out = Path(out)
    if readiness_json(out/"protocol/explanation.json", {}).get("authorized_scope") != "fixed_expression_component_mechanisms":
        return False
    if readiness_json(out/"validation/data_provenance.json", {}).get("status") != "passed":
        return False
    if shutil.disk_usage(out).free < 150*1024**3:
        return False
    models = (gpu_model,) if gpu_model else ALLOWED_GPU_MODELS
    allowed = False
    for model in models:
        if model not in MODEL_KEYS or task.get("required_gpu_model", model) != model:
            continue
        label = MODEL_KEYS[model]
        if task["kind"] == "explanation_kernel_validation":
            allowed = True; break
        if not completed(out, f"explanation-kernel-check-{label}", True):
            continue
        if task["kind"] == "explanation_validation":
            allowed = True; break
        if not completed(out, f"explanation-input-check-{label}-{task['variant']}", True):
            continue
        allowed = True; break
    if not allowed:
        return False
    if task["kind"] == "explanation_fork":
        return completed(out, task["parent"])
    if task["kind"] == "explanation_confirmation":
        cache = _READINESS_CACHE.get(); key = ("confirmation_prerequisites", str(out))
        if cache is not None and key in cache:
            return cache[key]
        prereq = (readiness_json(p)["task"] for p in (out/"queue").glob("*.json"))
        allowed = all(completed(out, t["id"]) for t in prereq
                      if t["kind"] in ("explanation_development", "explanation_heavy", "explanation_fork"))
        if cache is not None:
            cache[key] = allowed
        return allowed
    return True


def execute(task, out):
    if task["kind"] in ("explanation_kernel_validation", "explanation_validation"):
        from .explanation_validation import validate
        return validate(task, out)
    if task["kind"] == "explanation_fork":
        from .explanation_forks import run
        return run(task, out)
    from .evaluate import run_task
    return run_task(task, out)


def update_progress(out):
    """只报告实际完成与可运行状态；不把验证通过等同于科学假设成立。"""
    out = Path(out); counts = Counter(); logical = Counter(); times = []; rows = []
    for path in sorted((out/"queue").glob("*.json")):
        task = read_json(path)["task"]; status = read_json(out/"jobs"/task["id"]/"status.json", {})
        state = status.get("status", "pending")
        counts[task["stage"]+":"+state] += 1
        if state == "completed":
            if task["kind"] in ("explanation_development", "explanation_confirmation", "explanation_heavy"):
                logical[task["stage"]] += len(task["indices"])*(7 if task.get("training_variants") else 4)
                meta = read_json(out/"jobs"/task["id"]/"manifest.json")
                times.append({"stage": task["stage"], "variant": task["variant"], "seconds_last_attempt": meta["wall_seconds"],
                              "task": task["id"], "hardware": meta["environment"].get("gpu"),
                              "scope": "含完整审计；恢复后最后一段计时不得冒充完整求解时间"})
        rows.append({"task": task["id"], "description": task.get("display_name", task["stage"]), "status": state})
    report = {"updated_at": now(), "counts": dict(counts), "completed_logical_solves": dict(logical),
              "disk_free_bytes": shutil.disk_usage(out).free, "timings": times, "tasks": rows}
    atomic_json(out/"monitor/scientific_progress.json", report)
    from .explanation_results import publish
    for stage in ("development", "confirmation"):
        publish(out,stage)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("prepare", "status"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(prepare(args.output) if args.action == "prepare" else update_progress(args.output), flush=True)
