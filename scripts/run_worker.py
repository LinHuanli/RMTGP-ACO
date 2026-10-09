"""后台实验工作进程：一张物理卡、一个固定源码快照、独立日志与心跳。"""

import argparse
import csv
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from gpaco.data import ROOT, write_json


def gpu_state(uuid):
    device = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-gpu=uuid,name,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    ).strip()
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    )
    pids = [
        int(row[1])
        for row in csv.reader(applications.splitlines())
        if len(row) >= 2 and row[0].strip() == uuid
    ]
    return device, pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = Path(args.job).resolve()
    if not job_path.is_relative_to(ROOT):
        raise ValueError("job 必须位于项目内")
    job = json.loads(job_path.read_text())
    run = job_path.parent
    snapshot = Path(job["snapshot"]).resolve()
    if not snapshot.is_relative_to(ROOT) or not snapshot.is_dir():
        raise ValueError("快照目录无效")
    uuid = job["gpu_uuid"]
    locks = ROOT / "artifacts/locks"
    locks.mkdir(parents=True, exist_ok=True)
    # 本项目的工作进程互斥；不能阻止其他用户启动新进程，因此另记争用。
    with (locks / f"{uuid}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device, pids = gpu_state(uuid)
        row = next(csv.reader([device]))
        if pids or int(row[2]) > 5 or int(row[3]) > 1024:
            raise RuntimeError(f"启动前 GPU 已被占用，退出而不抢占：{device}, pids={pids}")
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=uuid,
            GPACO_COMMIT=job["commit"],
            GPACO_SNAPSHOT=str(snapshot),
            PYTHONPATH=str(snapshot / "src"),
        )
        outputs = []
        for step, arguments in enumerate(job["commands"]):
            log_path = run / f"step-{step:02d}.log"
            with log_path.open("a") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "gpaco.cli", *arguments],
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                contaminated = False
                while True:
                    try:
                        device, pids = gpu_state(uuid)
                        outsiders = [pid for pid in pids if pid != process.pid]
                        contaminated |= bool(outsiders)
                        telemetry = {"device": device, "other_pids": outsiders}
                    except (OSError, subprocess.SubprocessError) as error:
                        telemetry = {"telemetry_error": str(error)}
                    heartbeat = {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "host": platform.node(),
                        "worker_pid": os.getpid(),
                        "child_pid": process.pid,
                        "step": step,
                        "contended_during_step": contaminated,
                        **telemetry,
                    }
                    write_json(run / "heartbeat.json", heartbeat)
                    with (run / "telemetry.jsonl").open("a") as handle:
                        handle.write(json.dumps(heartbeat) + "\n")
                    code = process.poll()
                    if code is not None:
                        break
                    time.sleep(15)
                outputs.append(
                    {
                        "step": step,
                        "returncode": code,
                        "log": str(log_path),
                        "contended": contaminated,
                    }
                )
                write_json(run / "steps.json", outputs)
                if code:
                    raise RuntimeError(f"实验步骤 {step} 失败，见 {log_path}")
        write_json(run / "COMPLETE.json", {"status": "completed", "steps": outputs})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"worker failed: {error}", file=sys.stderr, flush=True)
        raise
