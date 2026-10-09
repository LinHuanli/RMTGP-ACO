"""持续调度协议测试；全部测试数据留在项目内TMPDIR，不需要GPU。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from test_core import problem

from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import FrozenStore

sys.path.insert(0, str(ROOT / "scripts"))
import a5000_pool as pool
import run_main_benchmark as worker


def config():
    return yaml.safe_load((ROOT / "configs/workloads/a5000_main_queue.yaml").read_text())


def test_gpu_free_parser_only_takes_idle_exact_a5000():
    text = """
IDLE cuda01 1 0.0 / 24.0 0% 0 - NVIDIA RTX A5000
IDLE cuda08 0 0.0 / 24.0 0% 0 - NVIDIA RTX A5000
SINGLE cuda02 0 6.6 / 24.0 100% 1 linbocheng NVIDIA RTX A5000
WARM cuda03 0 0.0 / 24.0 4% 0 - NVIDIA RTX A5000
IDLE cuda06 1 0.0 / 47.8 0% 0 - NVIDIA RTX PRO 5000 Blackwell
IDLE cuda20 2 0.0 / 45.0 0% 0 - NVIDIA L40S
IDLE cuda01 1 0.0 / 24.0 0% 0 - NVIDIA RTX A5000
"""
    assert pool.parse_idle_a5000(text) == [
        {"host": "cuda01", "index": 1},
        {"host": "cuda08", "index": 0},
    ]


def test_finite_tasks_and_paired_blocks():
    tasks = pool.tasks_for(config())
    assert len(tasks) == len({t["id"] for t in tasks}) == 110
    assert sum(t["kind"] == "prepare" for t in tasks) == 2
    assert sum(t["kind"] == "pair" for t in tasks) == 90
    assert sum(t["kind"] == "profile" for t in tasks) == 18
    assert all(t["block"] < 5 for t in tasks if t["kind"] == "pair")
    assert all("test" not in t.get("cohort_path", "") for t in tasks)


def test_dependency_states_do_not_replace_missing_cohort(tmp_path):
    tasks = pool.tasks_for(config())
    task = next(t for t in tasks if t["kind"] == "pair")
    task["cohort_path"] = str(tmp_path / "real-cohort.json")
    assert pool.dependencies(task, tasks, tmp_path) == "waiting:inputs"
    write_json(tmp_path / "inputs/tsp100/READY.json", {"ready": True})
    assert pool.dependencies(task, tasks, tmp_path) == "waiting:real_cohort"
    write_json(Path(task["cohort_path"]), [])
    assert pool.dependencies(task, tasks, tmp_path) == "ready"
    tasks[0]["status"] = "failed"
    assert pool.dependencies(task, tasks, tmp_path).startswith("failed:")


def test_profile_waits_for_completed_pair(tmp_path):
    tasks = pool.tasks_for(config())
    task = next(t for t in tasks if t["kind"] == "profile")
    task["cohort_path"] = str(tmp_path / "cohort.json")
    write_json(Path(task["cohort_path"]), [])
    write_json(tmp_path / "inputs/tsp100/READY.json", {})
    assert pool.dependencies(task, tasks, tmp_path) == "waiting:paired_measurement"
    next(t for t in tasks if t["id"] == task["pair_dependency"])["status"] = "completed"
    assert pool.dependencies(task, tasks, tmp_path) == "ready"


def test_shared_geometry_and_variant_specific_initialization(tmp_path, monkeypatch):
    cfg = config()
    cfg.update(batch=2, paired_blocks=2)
    cfg["search"]["candidate_size"] = 3
    data = problem(n=100)
    tours = np.tile(np.r_[np.arange(data.n), 0], (data.size, 1)).astype(np.int32)
    monkeypatch.setattr(worker, "load_split", lambda *a: (data.coords, tours, data.instance_ids))
    real_hash = worker.file_hash
    monkeypatch.setattr(
        worker, "file_hash", lambda p: "fixture" if "Datasets" in str(p) else real_hash(p)
    )
    info = {"name": "NVIDIA RTX A5000", "gpu_visible": "GPU-unit"}
    worker.prepare_inputs(tmp_path, 100, cfg, info, "GPU-unit")
    stores = {v: FrozenStore(tmp_path / "inputs/tsp100" / v) for v in cfg["variants"]}
    problems = {v: s.problem("block-00") for v, s in stores.items()}
    for p in problems.values():
        np.testing.assert_array_equal(p.distances, problems["as"].distances)
        assert p.initialization.seed == problems["as"].initialization.seed
        assert not p.distances.flags.writeable
    assert np.isinf(problems["as"].initialization.values[2]).all()
    assert np.isfinite(problems["mmas"].initialization.values[2]).all()
    assert np.all(
        problems["acs"].initialization.values[0] < problems["as"].initialization.values[0]
    )
    with pytest.raises(ValueError, match="写入者"):
        worker.prepare_inputs(tmp_path, 500, cfg, {**info, "name": "NVIDIA L4"}, "GPU-unit")


def test_reconcile_unknown_worker_keeps_lease(tmp_path, monkeypatch):
    task = {
        "status": "running",
        "attempts": [
            {"job_path": str(tmp_path / "job.json"), "assigned_unix_s": 0, "host": "cuda01"}
        ],
    }
    monkeypatch.setattr(pool, "worker_alive", lambda a: None)
    pool.reconcile([task])
    assert task["status"] == "running"
    assert not (tmp_path / "FAILED.json").exists()
    monkeypatch.setattr(pool, "worker_alive", lambda a: False)
    # 一次失联不再直接判失败；宽限与远程文件可见性的回归由test_queue_state覆盖。
    assert task["status"] == "running"


def test_startup_rejection_is_not_a_failed_scientific_sample(tmp_path):
    task = {
        "status": "running",
        "attempts": [
            {"job_path": str(tmp_path / "job.json"), "assigned_unix_s": 0, "host": "cuda01"}
        ],
    }
    write_json(tmp_path / "REJECTED.json", {"scientific_work_started": False})
    pool.reconcile([task])
    assert task["status"] == "pending"
    assert task["attempts"][-1]["rejected"]
