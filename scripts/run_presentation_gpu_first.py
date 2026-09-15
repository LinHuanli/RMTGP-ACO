#!/usr/bin/env python3
"""GPU 先行队列：不等待 CPU 对照，沿用同一份冻结计算源码与输入。

结果使用独立 cohort，不能与另一主机的 CPU 时间直接计算速度比。
同一 GPU 比较组固定卡；同机的不同 GPU 使用不同物理 CPU 核。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import shlex
import socket
import subprocess
import time
import traceback
from pathlib import Path

import run_presentation_benchmarks as base

ROOT = base.ROOT
PARENT = base.OUT
OUTPUT = PARENT / "gpu_first"
GROUPS = ("gpu-main", "gpu-scaling", "gpu-profile")


def link_resource(source, target):
    """使用仓库内相对链接，搬移仓库后仍能读取冻结资源。"""
    if (
        target.is_symlink()
        and target.readlink().is_absolute()
        and target.resolve() == source.resolve()
    ):
        target.unlink()  # 仅替换本队列创建的链接，不修改其目标文件。
    if not target.exists():
        target.symlink_to(
            os.path.relpath(source, target.parent), target_is_directory=source.is_dir()
        )


def prepare(output):
    """链接已冻结资源，不重新抽样，也不覆盖原同机对照的结果。"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    for name in ("config.yaml", "workload_manifest.json", "inputs", "baselines", "trace", "scans"):
        source = PARENT / name
        target = output / name
        if not source.exists():
            raise FileNotFoundError(source)
        link_resource(source, target)
    source = base.read(PARENT / "source.json")
    base.atomic(output / "source.json", source)
    base.atomic(
        output / "cohort.json",
        {
            "cohort": "gpu-first",
            "created_at": base.now(),
            "parent": str(PARENT),
            "measurement_source_hash": source["source_hash"],
            "scheduler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "groups": list(GROUPS),
            "gpu_only": True,
            "cpu_gpu_speedup_allowed": False,
            "reason": "Run GPU tasks without waiting for CPU controls; do not pool across cohorts",
        },
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for path in (PARENT / "figures").glob("D*.*"):
        target = figures / path.name
        link_resource(path, target)


class GPUWorker(base.Worker):
    def execute(self):
        if self.group == "gpu-main":
            # 先完成同一 trace 的 GPU-v1/v2 对照，再补真实演化重复。
            for label in ("v1", "v2"):
                self.run(f"sanity-{label}", "sanity", label)
            self.repeats("E2", "replay", ["v2", "v1"])
            self.repeats("E1", "train", ["v2"])
        elif self.group == "gpu-scaling":
            self.run("sanity-scaling-v2", "sanity", "v2")
            names = [f"p{p}-n100" for p in (1, 8, 32, 100, 256)]
            names += ["p100-n50", "p100-n500"]
            for name in names:
                self.repeats(
                    f"E4-{name}",
                    "replay",
                    ["v2"],
                    extra=["--trace", self.output / "scans" / name],
                )
        elif self.group == "gpu-profile":
            # 修正进程归属检测后重新诊断，原始未通过检查的记录继续保留。
            for label in ("v1", "v2"):
                self.profile(label)
        else:
            raise ValueError(self.group)
        self.state.update(status="completed", completed_at=base.now())
        self.heartbeat()


def worker(output, group, gpu):
    host = socket.gethostname().split(".")[0]
    (output / "locks").mkdir(exist_ok=True)
    # 原队列持有独占主机锁；GPU 先行任务之间可共享主机，但不能共享同一张卡。
    with (
        (PARENT / "locks" / f"{host}.lock").open("a") as host_lock,
        (output / "locks" / f"{host}-gpu{gpu}.lock").open("a") as device_lock,
    ):
        try:
            fcntl.flock(host_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            base.atomic(
                output / "groups" / f"{group}.json",
                {
                    "status": "pending",
                    "reason": "host/device benchmark lock unavailable",
                },
            )
            return
        runner = GPUWorker(output, group, gpu, cpu_core_offset=gpu, gpu_wait_timeout=45)
        runner.state["cohort"] = "gpu-first"
        runner.heartbeat()
        try:
            runner.execute()
        except base.GPUUnavailable as error:
            runner.state.update(status="pending", reason=str(error))
            runner.heartbeat()
        except Exception:
            runner.state.update(status="failed", error=traceback.format_exc())
            runner.heartbeat()
            raise


def status(output):
    states = {
        name: base.read(output / "groups" / f"{name}.json", {"status": "pending"})
        for name in GROUPS
    }
    counts = {}
    for state in states.values():
        counts[state["status"]] = counts.get(state["status"], 0) + 1
    tasks = [base.read(p, {}) for p in (output / "jobs").glob("*/status.json")]
    result = {
        "updated_at": base.now(),
        "cohort": "gpu-first",
        "groups": states,
        "counts": counts,
        "completed_tasks": sum(t.get("status") == "completed" for t in tasks),
        "running_tasks": [t.get("task") for t in tasks if t.get("status") == "running"],
    }
    base.atomic(output / "status.json", result)
    return result


def select_group(states, host, gpu):
    """有有效组归属后固定卡，不把不同主机的重复混成一条曲线。"""
    for group in GROUPS:
        state = states[group]
        if state["status"] != "pending":
            continue
        if "host" not in state or (state["host"], state["gpu"]) == (host, gpu):
            return group
    return None


def launch_group(output, group, host, gpu):
    command = [
        str(base.PYTHON),
        str(Path(__file__).resolve()),
        "worker",
        "--output-root",
        str(output),
        "--group",
        group,
        "--gpu",
        str(gpu),
    ]
    log = output / "logs" / f"{group}.log"
    remote = "nohup " + shlex.join(command) + " >> " + shlex.quote(str(log))
    remote += " 2>&1 < /dev/null & echo $!"
    base.atomic(
        output / "groups" / f"{group}.json",
        {
            "status": "launching",
            "host": host,
            "gpu": gpu,
            "launched_at": base.now(),
        },
    )
    process = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    print(f"GPU-first {group}: {host}:{gpu} pid={process.stdout.strip()}", flush=True)


def controller(output):
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.atomic(
            output / "controller.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": base.now(),
            },
        )
        while True:
            summary = status(output)
            if all(s["status"] in ("completed", "failed") for s in summary["groups"].values()):
                base.report(output)
                break
            parent_states = [base.read(p, {}) for p in (PARENT / "groups").glob("*.json")]
            busy_hosts = {
                s.get("host") for s in parent_states if s.get("status") in ("running", "launching")
            }
            busy_cards = {
                (s["host"], s["gpu"])
                for s in summary["groups"].values()
                if s["status"] in ("running", "launching")
            }
            for host, gpu in base.discover(output):
                if host in busy_hosts or (host, gpu) in busy_cards:
                    continue
                group = select_group(summary["groups"], host, gpu)
                if group is None:
                    continue
                try:
                    launch_group(output, group, host, gpu)
                except (subprocess.SubprocessError, OSError) as error:
                    state_path = output / "groups" / f"{group}.json"
                    if base.read(state_path, {}).get("status") == "launching":
                        base.atomic(state_path, {"status": "pending", "reason": str(error)})
                    continue
                busy_cards.add((host, gpu))
                summary["groups"][group] = {"status": "launching", "host": host, "gpu": gpu}
            status(output)
            base.report(output)
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "controller", "worker", "status", "report"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.action == "launch":
        prepare(output)
        with (output / "logs/controller.log").open("a") as log:
            proc = subprocess.Popen(
                [
                    "nohup",
                    str(base.PYTHON),
                    str(Path(__file__).resolve()),
                    "controller",
                    "--output-root",
                    str(output),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"GPU-first controller pid={proc.pid} output={output}")
    elif args.action == "controller":
        controller(output)
    elif args.action == "worker":
        worker(output, args.group, args.gpu)
    elif args.action == "status":
        print(status(output))
    else:
        raise SystemExit(base.report(output))


if __name__ == "__main__":
    main()
