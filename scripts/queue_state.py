"""共享文件系统上的结束状态对账；完成证据优先，失联不等于计算失败。"""

import json
import shlex
import subprocess
import time
from pathlib import Path

from launch_pilots import remote

from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import file_hash

LEGACY_MISSING = "worker消失且没有结束标记；不自动重复科学样本"


def read_json(path):
    """直接open，不用exists的负缓存作为存在性的最终判据。"""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def controller_failure(value):
    return bool(value) and (
        value.get("detected_by_controller") or value.get("reason") == LEGACY_MISSING
    )


def remote_markers(attempt):
    """从worker所在主机二次观察；SSH失败返回未知，不释放运行租约。"""
    code = (
        "import json,pathlib; p=pathlib.Path("
        + repr(str(Path(attempt["job_path"]).parent))
        + "); print(json.dumps([n for n in ('COMPLETE.json','FAILED.json','REJECTED.json') "
        "if (p/n).exists()]))"
    )
    try:
        return json.loads(
            remote(
                attempt["host"],
                shlex.quote(str(ROOT / ".envs/main/bin/python")) + " -c " + shlex.quote(code),
            )
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def validate_diagnostic(directory, complete):
    """误报恢复也必须验证结果内容，不能只因看见COMPLETE文件就升级为成功。"""
    record_path = directory / "measurement/record.json"
    record = read_json(record_path)
    pair = read_json(directory / "measurement/instrumentation_pair.json")
    job = read_json(directory / "job.json")
    if (
        not record
        or not pair
        or not job
        or complete.get("task_id") != job["task"]["id"]
        or file_hash(record_path) != complete.get("record_sha256")
        or file_hash(directory / "measurement/result.npz") != record["result_sha256"]
        or not pair["bitwise_equal_tours"]
        or not pair["bitwise_equal_lengths"]
    ):
        raise ValueError(f"完成证据未通过内容校验：{directory}")


def validate_main(directory, complete):
    for relative in complete.get("record_paths", []):
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("完成记录引用越界")
        value = read_json(path)
        if not value or value.get("status") != "completed":
            raise ValueError("基线完成记录引用了未完成测量")


def reconcile_tasks(
    tasks,
    alive,
    validate,
    *,
    distinguish_contended=True,
    now=None,
    grace_s=180,
    probe=remote_markers,
):
    now = time.time() if now is None else now
    for task in tasks:
        if task["status"] not in ("running", "failed") or not task.get("attempts"):
            continue
        attempt = task["attempts"][-1]
        directory = Path(attempt["job_path"]).parent
        complete = read_json(directory / "COMPLETE.json")
        failure = read_json(directory / "FAILED.json")
        if complete is not None:
            # 真正的worker异常与完成标记冲突时，不自动掩盖；历史失联误报可以核验恢复。
            if failure and not controller_failure(failure):
                task["state_conflict"] = "worker_failure_and_completion; manual_review_required"
                task["status"] = "failed"
                continue
            try:
                validate(directory, complete)
            except FileNotFoundError as error:
                task["completion_validation_error"] = str(error)
                continue
            except (ValueError, KeyError) as error:
                task["completion_validation_error"] = str(error)
                task["status"] = "failed"
                continue
            status = (
                "excluded_contended"
                if distinguish_contended and not complete.get("clean", True)
                else "completed"
            )
            if task["status"] == "failed" or failure:
                recovery = directory / "STATE_RECONCILED.json"
                if read_json(recovery) is None:
                    write_json(
                        recovery,
                        {
                            "previous_status": task["status"],
                            "new_status": status,
                            "observed_unix_s": now,
                            "reason": "verified_completion_overrides_controller_liveness_report",
                            "complete_sha256": file_hash(directory / "COMPLETE.json"),
                            "failure_sha256": file_hash(directory / "FAILED.json")
                            if failure
                            else None,
                            "original_records_preserved": True,
                            "scientific_work_repeated": False,
                        },
                    )
                task["reconciliation"] = str(recovery)
            task["status"], task["result"] = status, str(directory / "COMPLETE.json")
            continue
        if task["status"] == "failed":
            continue
        if failure is not None:
            task["status"] = "failed"
            continue
        if read_json(directory / "REJECTED.json") is not None:
            task["status"], attempt["rejected"] = "pending", True
            continue
        if now - attempt["assigned_unix_s"] <= 120:
            continue
        running = alive(attempt)
        attempt["last_liveness"] = running
        if running is not False:
            attempt.pop("missing_since", None)
            attempt.pop("missing_observations", None)
            continue
        markers = probe(attempt)
        if markers is None or markers:
            attempt["completion_visibility"] = "unknown_remote" if markers is None else markers
            continue
        attempt.setdefault("missing_since", now)
        attempt["missing_observations"] = attempt.get("missing_observations", 0) + 1
        if now - attempt["missing_since"] < grace_s or attempt["missing_observations"] < 3:
            continue
        # 至少跨180秒的多次主机存活/文件双检查；再查一次，防止正常退出竞争。
        if any(
            read_json(directory / name) is not None
            for name in ("COMPLETE.json", "FAILED.json", "REJECTED.json")
        ):
            continue
        write_json(
            directory / "FAILED.json",
            {
                "reason": "worker_missing_after_grace",
                "detected_by_controller": True,
                "missing_since": attempt["missing_since"],
                "detected_unix_s": now,
                "observations": attempt["missing_observations"],
                "auto_retry": False,
            },
        )
        task["status"] = "failed"
