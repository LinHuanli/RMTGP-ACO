"""空间检查只影响调度，不减少实例数或日志。"""
from control_experiments.mmas_ls.storage_guard import GIB, RESERVE, assess, task_bytes, enforce_capacity
from control_experiments.mmas_ls.common import atomic_json


def task(name, kind="explanation_development", **extra):
    return {"id": name, "kind": kind, "indices": [0, 1], "variant": "mmas",
            "mechanism": {"source_policy": "iteration_best", "source_count": 1}, **extra}


def test_source_count_changes_capacity_estimate():
    one = task("one")
    many = task("many", mechanism={"source_policy": "top_k", "source_count": 32})
    assert task_bytes(many, 100, 500) == 5*task_bytes(one, 100, 500)


def test_confirmation_space_does_not_block_smaller_development_stage():
    tasks = [task("dev"), task("confirm", "explanation_confirmation", indices=list(range(128)))]
    state = assess(tasks, {}, GIB, GIB, RESERVE+20*GIB)
    assert state["stage"] == "development" and state["current_stage_fits_estimate"]
    assert not state["all_remaining_stages_fit_estimate"]
    state = assess(tasks, {"dev": "completed"}, GIB, GIB, RESERVE+20*GIB)
    assert state["stage"] == "confirmation" and not state["current_stage_fits_estimate"]


def test_running_tasks_still_reserve_space():
    t = task("running")
    a = assess([t], {t["id"]: "running"}, GIB, GIB, RESERVE)
    b = assess([t], {t["id"]: "completed"}, GIB, GIB, RESERVE)
    assert not a["current_stage_fits_estimate"] and b["current_stage_fits_estimate"]


def test_heavy_and_fork_states_have_extra_space():
    base = task_bytes(task("base"), GIB, GIB)
    assert task_bytes(task("heavy", "explanation_heavy"), GIB, GIB) > base
    assert task_bytes(task("fork", "explanation_fork"), GIB, GIB) == GIB


def test_unavailable_measurements_do_not_fabricate_an_estimate(tmp_path):
    assert enforce_capacity(tmp_path)
    assert not (tmp_path/"STOP").exists()


def test_guard_preserves_existing_stop_reason(tmp_path, monkeypatch):
    from types import SimpleNamespace
    for variant in ("as", "mmas"):
        t = task(variant, "explanation_validation", variant=variant)
        atomic_json(tmp_path/"queue"/(variant+".json"), {"task": t})
        atomic_json(tmp_path/"jobs"/variant/"status.json", {"status": "completed", "validation_status": "passed",
            "evidence": {"projected_bytes_per_5000_iteration_logical_solve": GIB}})
    atomic_json(tmp_path/"queue/dev.json", {"task": task("dev")})
    atomic_json(tmp_path/"STOP", {"reason": "已有的人工停止原因"})
    before = (tmp_path/"STOP").read_bytes()
    monkeypatch.setattr("control_experiments.mmas_ls.storage_guard.shutil.disk_usage", lambda _: SimpleNamespace(free=RESERVE))
    assert not enforce_capacity(tmp_path)
    assert (tmp_path/"STOP").read_bytes() == before


def test_readiness_cache_does_not_outlive_scan(tmp_path):
    from control_experiments.mmas_ls.explanation_campaign import completed, readiness_batch
    path = tmp_path/"jobs/task/status.json"
    atomic_json(path, {"status": "running"})
    with readiness_batch():
        assert not completed(tmp_path, "task")
        atomic_json(path, {"status": "completed"})
        assert not completed(tmp_path, "task")
    assert completed(tmp_path, "task")
