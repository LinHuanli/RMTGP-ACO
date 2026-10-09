"""NFS负缓存、退出竞争、历史误报与真正失败的状态边界。"""

import json
import sys

from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import file_hash

sys.path.insert(0, str(ROOT / "scripts"))
from queue_state import LEGACY_MISSING, reconcile_tasks, validate_diagnostic


def item(directory, status="running"):
    return {
        "id": "fixture",
        "status": status,
        "attempts": [
            {"job_path": str(directory / "job.json"), "host": "fixture", "assigned_unix_s": 0}
        ],
    }


def test_missing_needs_grace_and_multiple_observations(tmp_path):
    task = item(tmp_path)
    for now in (200, 260, 320):
        reconcile_tasks([task], lambda _: False, lambda *a: None, now=now, probe=lambda _: [])
        assert task["status"] == "running"
    reconcile_tasks([task], lambda _: False, lambda *a: None, now=381, probe=lambda _: [])
    assert task["status"] == "failed"
    assert json.loads((tmp_path / "FAILED.json").read_text())["detected_by_controller"]


def test_remote_complete_not_visible_locally_keeps_lease(tmp_path):
    task = item(tmp_path)
    for now in (200, 500, 800):
        reconcile_tasks(
            [task], lambda _: False, lambda *a: None, now=now, probe=lambda _: ["COMPLETE.json"]
        )
    assert task["status"] == "running" and not (tmp_path / "FAILED.json").exists()


def test_unknown_remote_keeps_lease(tmp_path):
    task = item(tmp_path)
    reconcile_tasks([task], lambda _: False, lambda *a: None, now=900, probe=lambda _: None)
    assert task["status"] == "running"


def test_exit_race_rechecks_completion(tmp_path):
    task = item(tmp_path)
    task["attempts"][0].update(missing_since=1, missing_observations=5)

    def probe(_):
        write_json(tmp_path / "COMPLETE.json", {"clean": True})
        return []

    reconcile_tasks([task], lambda _: False, lambda *a: None, now=900, probe=probe)
    assert not (tmp_path / "FAILED.json").exists()
    reconcile_tasks([task], lambda _: False, lambda *a: None, now=960, probe=probe)
    assert task["status"] == "completed"


def diagnostic_fixture(directory):
    measurement = directory / "measurement"
    write_json(directory / "job.json", {"task": {"id": "fixture"}})
    # SHA测试只依赖不透明文件字节；此文件不是生产NPZ数据。
    write_json(measurement / "result.npz", {"fixture": True})
    write_json(
        measurement / "record.json", {"result_sha256": file_hash(measurement / "result.npz")}
    )
    write_json(
        measurement / "instrumentation_pair.json",
        {"bitwise_equal_tours": True, "bitwise_equal_lengths": True},
    )
    write_json(
        directory / "COMPLETE.json",
        {
            "clean": True,
            "task_id": "fixture",
            "record_sha256": file_hash(measurement / "record.json"),
        },
    )


def test_historical_false_failure_recovers_without_deleting_evidence(tmp_path):
    diagnostic_fixture(tmp_path)
    write_json(tmp_path / "FAILED.json", {"reason": LEGACY_MISSING})
    original = (tmp_path / "FAILED.json").read_bytes()
    task = item(tmp_path, "failed")
    reconcile_tasks([task], lambda _: False, validate_diagnostic, now=900)
    assert task["status"] == "completed"
    assert (tmp_path / "FAILED.json").read_bytes() == original
    assert (
        json.loads((tmp_path / "STATE_RECONCILED.json").read_text())["scientific_work_repeated"]
        is False
    )


def test_real_worker_failure_is_not_silently_overridden(tmp_path):
    write_json(tmp_path / "FAILED.json", {"traceback": "actual error"})
    write_json(tmp_path / "COMPLETE.json", {"clean": True})
    task = item(tmp_path, "failed")
    reconcile_tasks([task], lambda _: False, lambda *a: None, now=900)
    assert task["status"] == "failed" and task["state_conflict"]


def test_corrupt_completion_cannot_recover(tmp_path):
    diagnostic_fixture(tmp_path)
    write_json(tmp_path / "FAILED.json", {"reason": LEGACY_MISSING})
    write_json(tmp_path / "measurement/result.npz", {"corrupt": True})
    task = item(tmp_path, "failed")
    reconcile_tasks([task], lambda _: False, validate_diagnostic, now=900)
    assert task["status"] == "failed" and task["completion_validation_error"]
