"""GPU 先行调度不引入 CPU 依赖，也不混合不同卡上的重复。"""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_presentation_gpu_first")


@pytest.mark.parametrize("group", ["gpu-main", "gpu-scaling"])
def test_gpu_groups_never_wait_for_cpu_tasks(scheduler, tmp_path, group):
    worker = scheduler.GPUWorker.__new__(scheduler.GPUWorker)
    worker.group, worker.output, worker.state = group, tmp_path, {}
    calls = []

    def run(task, action, label, **kwargs):
        calls.append((task, action, label))
        return {"records": [{"evaluation_wall_s": 1.0}]}

    worker.run = run
    worker.heartbeat = lambda: None
    worker.execute()
    assert calls
    assert all(not label.startswith("cpu") for _, _, label in calls)
    assert worker.state["status"] == "completed"
    if group == "gpu-main":
        assert sum(task.startswith("E2-") for task, _, _ in calls) == 6
        assert sum(task.startswith("E1-") for task, _, _ in calls) == 3
    else:
        assert sum(task.startswith("E4-") for task, _, _ in calls) == 21


def test_pending_group_keeps_its_gpu_and_other_card_can_run_independent_group(scheduler):
    states = {g: {"status": "pending"} for g in scheduler.GROUPS}
    states["gpu-main"].update(host="cuda12", gpu=0)
    assert scheduler.select_group(states, "cuda12", 0) == "gpu-main"
    assert scheduler.select_group(states, "cuda12", 1) == "gpu-scaling"
    assert scheduler.select_group(states, "cuda08", 0) == "gpu-scaling"


def test_nsight_descendant_is_not_external_even_with_new_process_group(scheduler, monkeypatch):
    base = scheduler.base
    monkeypatch.setattr(base.os, "getpgid", lambda pid: 999)
    monkeypatch.setattr(base.Path, "read_text", lambda self: "501 (nsys target) S 123 999")
    assert base.belongs_to_job(501, 123)


def test_reparented_child_requires_matching_job_token(scheduler, monkeypatch):
    base = scheduler.base
    monkeypatch.setattr(base.os, "getpgid", lambda pid: 999)
    monkeypatch.setattr(base.Path, "read_text", lambda self: "501 (target) S 1 999")
    monkeypatch.setattr(
        base, "process_job_token", lambda pid: "job-a" if pid in (123, 501) else "job-b"
    )
    assert base.belongs_to_job(501, 123)
    assert not base.belongs_to_job(502, 123)
