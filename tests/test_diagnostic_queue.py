"""完整诊断队列的预算、依赖、租约与迁移兼容性检查。"""

import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from gpaco.artifact_registry import resolve
from gpaco.data import ROOT, write_json

sys.path.insert(0, str(ROOT / "scripts"))
import diagnostic_pool as pool


def config():
    return yaml.safe_load((ROOT / "configs/workloads/diagnostic_queue.yaml").read_text())


def test_fixed_diagnostic_matrix():
    cfg = config()
    tasks = pool.tasks_for(cfg)
    assert len(tasks) == len({t["id"] for t in tasks}) == 54
    assert {t["block"] for t in tasks} == {0, 1, 2}
    assert cfg["scientific_budget"]["iterations"] == 500
    assert not cfg["automatic_standard_test"] and not cfg["automatic_formal_training"]


def test_missing_real_cohort_stays_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pool, "BenchmarkInputs", lambda _: SimpleNamespace(manifest={"cohorts": {"1": {}}})
    )
    original = pool.resolve
    monkeypatch.setattr(
        pool, "resolve", lambda key: tmp_path if key == "E09-p01-training" else original(key)
    )
    task = {"n": 500, "variant": "as", "generation": 25, "block": 0}
    assert pool.bundle_for(task, config(), prepare=False) is None
    assert not list(tmp_path.iterdir())


def test_dry_run_does_not_export_late_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pool, "BenchmarkInputs", lambda _: SimpleNamespace(manifest={"cohorts": {"1": {}}})
    )
    original = pool.resolve
    cohort = tmp_path / "training/as-tsp500-seed1002/cohorts/generation-025.json"
    write_json(cohort, [])
    paths = {
        "E09-p01-training": tmp_path / "training",
        "shared-diagnostic-cohorts-p01": tmp_path / "inputs",
    }
    monkeypatch.setattr(pool, "resolve", lambda key: paths[key] if key in paths else original(key))

    def forbidden(*args):
        raise AssertionError("dry-run不能写输入")

    monkeypatch.setattr(pool, "export_bundle", forbidden)
    result = pool.bundle_for(
        {"n": 500, "variant": "as", "generation": 25, "block": 0}, config(), prepare=False
    )
    assert result == tmp_path / "inputs/tsp500/g025/as"
    assert not (tmp_path / "inputs").exists()


def task(tmp_path):
    return {
        "status": "running",
        "attempts": [
            {"host": "example", "job_path": str(tmp_path / "job.json"), "assigned_unix_s": 0}
        ],
    }


def test_uncertain_worker_does_not_create_duplicate(tmp_path, monkeypatch):
    item = task(tmp_path)
    monkeypatch.setattr(pool, "worker_alive", lambda _: None)
    pool.reconcile([item])
    assert item["status"] == "running"
    assert not (tmp_path / "FAILED.json").exists()


def test_startup_rejection_and_contamination_are_different(tmp_path):
    item = task(tmp_path)
    write_json(tmp_path / "REJECTED.json", {"scientific_work_started": False})
    pool.reconcile([item])
    assert item["status"] == "pending" and item["attempts"][-1]["rejected"]
    item["status"] = "running"
    write_json(tmp_path / "COMPLETE.json", {"status": "completed", "clean": False})
    pool.reconcile([item])
    assert item["status"] == "excluded_contended"


def test_migrated_original_queue_preserves_original_commit():
    campaign = resolve("E01-p01-gpu-baselines")
    if not (campaign / "campaign.json").exists():
        pytest.skip("新检出仓库没有本机历史队列；不要求下载科研原始数据")
    manifest = json.loads((campaign / "campaign.json").read_text())
    assert manifest["commit"] == "c5b93ac3950f828beab5ca67579f526adba8e2dd"
    assert (ROOT / "artifacts/a5000-main-v1").resolve() == campaign.resolve()
