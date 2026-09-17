"""用验收实测日志大小检查阶段空间；不删记录，不改变冻结实验配置。"""
from pathlib import Path
import math
import shutil

from .common import atomic_json, now, read_json

GIB = 1024**3
RESERVE = 150*GIB


def task_bytes(task, single_source, all_sources):
    """这是带余量的容量估算，不是压缩率或磁盘使用量的严格上界。"""
    kind = task["kind"]
    if kind == "explanation_fork":
        # 包括六个分支、端点矩阵、单次来源干预和逐轮记录。
        return GIB
    if kind not in ("explanation_development", "explanation_heavy", "explanation_confirmation"):
        return 0
    mechanism = task["mechanism"]
    multi = (mechanism.get("source_count", 1) == 32 or
             task["variant"] == "as" and mechanism.get("source_policy") == "native_schedule")
    logical = len(task["indices"])*(7 if task.get("training_variants") else 4)
    amount = logical*(all_sources if multi else single_source)*1.25
    if kind == "explanation_heavy":
        amount += logical*128*1024**2
    return math.ceil(amount)


def assess(tasks, statuses, single_source, all_sources, free_bytes):
    remaining = [t for t in tasks if statuses.get(t["id"]) != "completed"]
    stages = {"development": 0, "confirmation": 0}
    for task in remaining:
        stage = "confirmation" if task["kind"] == "explanation_confirmation" else "development"
        stages[stage] += task_bytes(task, single_source, all_sources)
    prerequisites = ("explanation_development", "explanation_heavy", "explanation_fork")
    stage = "development" if any(t["kind"] in prerequisites for t in remaining) else "confirmation"
    needed = stages[stage]+RESERVE
    return {"stage": stage, "remaining_estimated_bytes": stages, "reserve_bytes": RESERVE,
            "free_bytes": free_bytes, "current_stage_required_bytes": needed,
            "current_stage_fits_estimate": free_bytes >= needed,
            "all_remaining_stages_fit_estimate": free_bytes >= sum(stages.values())+RESERVE}


def enforce_capacity(out):
    """监控层独立执行。算子仍使用不可变快照；已有 worker 用 STOP 安全让出资源。"""
    out = Path(out)
    measurements = {"as": [], "mmas": []}
    for path in (out/"queue").glob("*.json"):
        task = read_json(path)["task"]
        if task["kind"] != "explanation_validation":
            continue
        status = read_json(out/"jobs"/task["id"]/"status.json", {})
        size = status.get("evidence", {}).get("projected_bytes_per_5000_iteration_logical_solve")
        if status.get("validation_status") == "passed" and size and math.isfinite(size) and size > 0:
            measurements[task["variant"]].append(size)
    if not all(measurements.values()):
        atomic_json(out/"monitor/storage_budget.json", {"status": "awaiting_validation_measurements", "updated_at": now()})
        return True
    tasks = [read_json(p)["task"] for p in (out/"queue").glob("*.json")]
    statuses = {t["id"]: read_json(out/"jobs"/t["id"]/"status.json", {}).get("status") for t in tasks}
    # 多来源日志至少按单来源大小估计，避免异常测量产生倒置的容量估计。
    single = max(measurements["mmas"]); multi = max(single, *measurements["as"])
    state = assess(tasks, statuses, single, multi, shutil.disk_usage(out).free)
    state.update(updated_at=now(), status="space_available" if state["current_stage_fits_estimate"] else "paused_for_space",
                 single_source_bytes_per_solve=single, all_sources_bytes_per_solve=multi,
                 estimation_scope="100 轮验收外推，正式求解加 25% 余量；已运行但未完成的任务按完整大小保守计入；压缩率可能变化")
    atomic_json(out/"monitor/storage_budget.json", state)
    if not state["current_stage_fits_estimate"]:
        # STOP 仅作用于本实验的监控和子进程。已有结果与检查点全部保留。
        if not (out/"STOP").exists():
            atomic_json(out/"STOP", {"reason": "本阶段完整诊断日志预计空间不足", **state})
        return False
    return True
