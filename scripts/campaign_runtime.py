"""固定源码快照与控制器升级；控制代码可升级，科学worker快照保持不变。"""

import argparse
import fcntl
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from gpaco.artifact_registry import resolve
from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import file_hash, safe_directory


def snapshot_head():
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "src", "scripts", "configs", "tests"],
        cwd=ROOT,
        text=True,
    )
    if dirty or untracked:
        raise ValueError("源码、配置和已跟踪文档须先提交，才能冻结运行快照")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    snapshot = resolve("shared-managed-source-snapshots") / commit
    if not snapshot.exists():
        archive = subprocess.check_output(
            ["git", "archive", commit, "src", "scripts", "configs", "tests", "pyproject.toml"],
            cwd=ROOT,
        )
        snapshot.mkdir(parents=True, exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(archive)) as package:
            package.extractall(snapshot, filter="data")
    return {"commit": commit, "snapshot": str(snapshot)}


def start_controller(campaign, runtime, script):
    """由单主机控制器锁阻止重复；从nohup脱离当前会话。"""
    snapshot = Path(runtime["snapshot"])
    env = {
        **os.environ,
        "PYTHONPATH": str(snapshot / "src"),
        "GPACO_COMMIT": runtime["commit"],
        "GPACO_SNAPSHOT": str(snapshot),
    }
    with (campaign / "dispatcher.log").open("a") as log:
        process = subprocess.Popen(
            [
                "nohup",
                sys.executable,
                str(snapshot / "scripts" / script),
                "--run",
                "--campaign",
                str(campaign),
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {**runtime, "pid": process.pid, "time_unix_s": time.time(), "script": script}
    write_json(campaign / "controller_launched.json", record)
    return record


def upgrade(campaign, script):
    runtime = snapshot_head()
    with (campaign / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (campaign / "STOP_DISPATCH").exists():
            raise ValueError("必须先显式请求旧控制器正常停止派发")
        revision = campaign / "controller-revisions" / str(time.time_ns())
        revision.mkdir(parents=True, exist_ok=False)
        manifest = json.loads((campaign / "campaign.json").read_text())
        # 备份原队列和身份；不改写campaign或历史FAILED标记。
        write_json(
            revision / "queue-before.json", json.loads((campaign / "queue.json").read_text())
        )
        write_json(
            revision / "upgrade.json",
            {
                **runtime,
                "worker_commit_unchanged": manifest["commit"],
                "campaign_sha256": file_hash(campaign / "campaign.json"),
                "previous_controller": json.loads(
                    (campaign / "controller_runtime.json").read_text()
                )
                if (campaign / "controller_runtime.json").exists()
                else manifest["commit"],
                "reason": "用户要求修复NFS完成状态同步；只升级调度，不重跑科学样本",
            },
        )
        (campaign / "STOP_DISPATCH").rename(revision / "stop-request.txt")
        write_json(campaign / "controller_runtime.json", runtime)
    result = start_controller(campaign, runtime, script)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upgrade", choices=["diagnostic", "main"], required=True)
    args = parser.parse_args()
    key, script = (
        ("shared-diagnostic-dispatch", "diagnostic_pool.py")
        if args.upgrade == "diagnostic"
        else ("E01-p01-gpu-baselines", "a5000_pool.py")
    )
    upgrade(safe_directory(resolve(key)), script)


if __name__ == "__main__":
    main()
