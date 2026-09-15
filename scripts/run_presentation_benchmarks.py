#!/usr/bin/env python3
"""A5000 报告实验队列：冻结源码、跨主机调度、可恢复执行及自动汇总。

调度单位是完整比较组。同一组在同一主机串行执行，独立组可分布到多机。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
OUT = ROOT / "slides/presentation_benchmarks"


def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def now():
    return datetime.now(UTC).isoformat()


def groups():
    return [
        "main",
        "gpu-setup",
        "scans-prepare",
        "jit",
        "scaling",
        "ablation-g3",
        "ablation-g5",
        "profile-v1",
        "profile-v2",
    ]


def snapshot(output):
    """复制项目代码与配置，随后所有测量引用同一只读语义快照。"""
    source_files = []
    for folder in ("src", "configs", "scripts"):
        source_files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in (".pyc", ".nbc", ".nbi")
        )
    digest = hashlib.sha256()
    for path in sorted(source_files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    revision = digest.hexdigest()
    target = output / "snapshots" / revision[:16]
    if not target.exists():
        for path in source_files:
            destination = target / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        (target / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
        (target / "Datasets").symlink_to(ROOT / "Datasets", target_is_directory=True)
    patch = subprocess.run(
        ["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    (output / "code.patch").write_text(patch)
    metadata = {
        "snapshot": str(target),
        "source_hash": revision,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "created_at": now(),
        "source_files": [str(p.relative_to(ROOT)) for p in source_files],
    }
    atomic(output / "source.json", metadata)
    return target


def discover(output):
    try:
        proc = subprocess.run(
            ["/home/linbocheng/bin/gpu-free"], capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        print("gpu-free timed out; retry on next scheduling pass", flush=True)
        return []
    (output / "gpu-free-latest.txt").write_text(proc.stdout + proc.stderr)
    cards = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) > 4 and fields[0] == "IDLE" and "RTX A5000" in line:
            cards.append((fields[1], int(fields[2])))
    preferred = {"cuda02": 0, "cuda08": 1, "cuda04": 2}
    return sorted(cards, key=lambda card: (preferred.get(card[0], 10), card))


def ready(output, group):
    trace = (output / "jobs/trace/result.json").is_file()
    main_baselines = all(
        (output / "baselines" / label / "manifest.json").is_file() for label in ("cpu8", "v1", "v2")
    )
    if group in ("main", "gpu-setup"):
        return True
    if group in ("scans-prepare", "jit"):
        return trace
    if group == "scaling":
        return main_baselines and (output / "scans/manifest.json").is_file()
    if group.startswith("ablation-"):
        return (
            trace
            and main_baselines
            and read(output / "groups/gpu-setup.json", {}).get("status") == "completed"
        )
    return trace and main_baselines


def gpu_sample():
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm,power.draw",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(query, capture_output=True, text=True, timeout=10)
    return proc.stdout.strip()


def process_job_token(pid):
    """只读取本任务标记，不输出进程环境中的其它字段。"""
    try:
        values = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    prefix = b"PRESENTATION_JOB_TOKEN="
    return next((v[len(prefix) :] for v in values if v.startswith(prefix)), None)


def belongs_to_job(pid, root_pid):
    """Nsight 可能让被测进程另建进程组，因此同时检查祖先与继承标记。"""
    if pid == root_pid:
        return True
    try:
        if os.getpgid(pid) == root_pid:
            return True
    except ProcessLookupError:
        return False
    current = pid
    for _ in range(64):
        if current == root_pid:
            return True
        if current <= 1:
            break
        try:
            fields = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
            current = int(fields[1])
        except (OSError, ValueError, IndexError):
            break
    token = process_job_token(root_pid)
    return bool(token and process_job_token(pid) == token)


def foreign_pids(uuid, pgid=None):
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode:
        raise RuntimeError("无法检查 GPU 进程")
    result = []
    for line in proc.stdout.splitlines():
        parts = [v.strip() for v in line.split(",")]
        if len(parts) != 2 or parts[0] != uuid:
            continue
        pid = int(parts[1])
        if pgid is not None and belongs_to_job(pid, pgid):
            continue
        result.append(pid)
    return result


def physical_cores():
    text = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)
    selected = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, package = map(int, line.split(","))
        selected.setdefault((package, core), cpu)
    return list(selected.values())


class GPUUnavailable(RuntimeError):
    """GPU 被其它任务占用；允许调度器释放 worker，之后再检查。"""


class Worker:
    def __init__(self, output, group, gpu, *, cpu_core_offset=0, gpu_wait_timeout=None):
        self.output, self.group, self.gpu = output, group, gpu
        self.cpu_core_offset = cpu_core_offset
        self.gpu_wait_timeout = gpu_wait_timeout
        self.host = socket.gethostname().split(".")[0]
        self.source = read(output / "source.json")
        self.snapshot = Path(self.source["snapshot"])
        self.state_path = output / "groups" / f"{group}.json"
        self.uuid = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        self.base = [str(PYTHON), "-m", "rmtgp_aco.presentation_bench"]
        self.state = {
            "group": group,
            "host": self.host,
            "gpu": gpu,
            "gpu_uuid": self.uuid,
            "pid": os.getpid(),
            "status": "running",
            "started_at": now(),
            "source_hash": self.source["source_hash"],
        }

    def heartbeat(self, task=None):
        self.state.update(heartbeat_at=now(), current_task=task)
        atomic(self.state_path, self.state)

    def run(
        self, task, action, label="v2", *, extra=(), destination=None, timeout=None, profiler=None
    ):
        task_dir = self.output / "jobs" / task
        completed = read(task_dir / "status.json", {})
        if completed.get("status") == "completed":
            return read(Path(completed["destination"]) / "result.json", {})
        destination = destination or task_dir
        task_dir.mkdir(parents=True, exist_ok=True)
        is_gpu = not label.startswith("cpu") and action not in ("jit", "scans")
        cpu_count = 8 if label == "cpu8" else 1
        cores = physical_cores()
        offset = self.cpu_core_offset % len(cores)
        cores = cores[offset:] + cores[:offset]
        if len(cores) < cpu_count:
            raise RuntimeError("主机物理核心数不足")
        command = self.base + [
            action,
            "--output-root",
            str(self.output),
            "--backend",
            label,
            "--destination",
            str(destination),
            *map(str, extra),
        ]
        if profiler:
            command = profiler + command + ["--profile"]
        command = ["taskset", "-c", ",".join(map(str, cores[:cpu_count])), *command]
        timeout = timeout or (
            28800 if action in ("train", "trace", "baseline", "jit", "replay") else 7200
        )
        previous = [int(p.stem.split("-")[-1]) for p in task_dir.glob("attempt-*.json")]
        first_attempt = max(previous, default=-1) + 1
        for attempt in range(first_attempt, first_attempt + 3):
            self.heartbeat(task)
            waiting_started = time.monotonic()
            while is_gpu and foreign_pids(self.uuid):
                # 空闲是瞬时状态；等待不会占用 CUDA context。
                atomic(
                    task_dir / "status.json",
                    {
                        "status": "waiting_gpu",
                        "host": self.host,
                        "task": task,
                        "gpu_uuid": self.uuid,
                    },
                )
                self.heartbeat(task)
                if self.gpu_wait_timeout is not None and (
                    time.monotonic() - waiting_started > self.gpu_wait_timeout
                ):
                    raise GPUUnavailable(f"{self.host}:{self.gpu} is occupied")
                time.sleep(15)
            cache = (
                Path("/tmp")
                / "rmtgp-presentation"
                / self.source["source_hash"][:12]
                / task
                / f"{attempt}-{time.time_ns()}"
            )
            cache.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.update(
                CUDA_VISIBLE_DEVICES=self.uuid,
                CUDA_HOME="/opt/cuda",
                CUDA_PATH="/opt/cuda",
                PYTHONPATH=str(self.snapshot / "src"),
                PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                NUMBA_NUM_THREADS=str(cpu_count),
                CUPY_CACHE_DIR=str(cache / "cupy"),
                CUDA_CACHE_PATH=str(cache / "driver"),
                NUMBA_CACHE_DIR=str(cache / "numba"),
                MPLCONFIGDIR=str(cache / "matplotlib"),
                PRESENTATION_JOB_TOKEN=str(cache),
            )
            env["PATH"] = "/opt/cuda/bin:" + ":".join(
                p for p in env.get("PATH", "").split(":") if "cuda-12.6" not in p
            )
            env["LD_LIBRARY_PATH"] = ":".join(
                p for p in env.get("LD_LIBRARY_PATH", "").split(":") if "cuda-12.6" not in p
            )
            record = {
                "task": task,
                "action": action,
                "backend": label,
                "status": "running",
                "host": self.host,
                "gpu_uuid": self.uuid,
                "attempt": attempt,
                "started_at": now(),
                "command": command,
                "cache": str(cache),
                "destination": str(destination),
                "source_hash": self.source["source_hash"],
                "scheduler_path": str(Path(__file__).resolve()),
                "scheduler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            atomic(task_dir / "status.json", record)
            samples, contaminated, interfering_pids = [], False, set()
            started = time.monotonic()
            with (task_dir / f"attempt-{attempt}.log").open("w") as log:
                env["PRESENTATION_PROCESS_START"] = str(time.perf_counter())
                child = subprocess.Popen(
                    command,
                    cwd=self.snapshot,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while child.poll() is None:
                    self.heartbeat(task)
                    if is_gpu:
                        samples.append(
                            {"time_s": time.monotonic() - started, "sample": gpu_sample()}
                        )
                        interference = foreign_pids(self.uuid, child.pid)
                        if interference:
                            contaminated = True
                            interfering_pids.update(interference)
                    if time.monotonic() - started > timeout:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        record["status"] = "timeout"
                        break
                    time.sleep(2)
            record.update(
                returncode=child.returncode,
                completed_at=now(),
                task_wall_s=time.monotonic() - started,
                contaminated=contaminated,
                interfering_pids=sorted(interfering_pids),
            )
            if record["status"] != "timeout":
                record["status"] = (
                    "contaminated"
                    if contaminated
                    else ("completed" if child.returncode == 0 else "failed")
                )
            atomic(task_dir / f"telemetry-{attempt}.json", samples)
            atomic(task_dir / f"attempt-{attempt}.json", record)
            atomic(task_dir / "status.json", record)
            if record["status"] == "completed":
                return read(destination / "result.json", {})
            if record["status"] in ("failed", "timeout"):
                raise RuntimeError(
                    f"{task}: {record['status']}, log={task_dir / f'attempt-{attempt}.log'}"
                )
        raise RuntimeError(f"{task}: GPU 持续受到外部任务干扰")

    def repeats(self, prefix, action, labels, extra=()):
        totals = {label: [] for label in labels}
        for repeat in range(5):
            if repeat == 3:
                noisy = any(
                    (max(v) - min(v)) / max(sorted(v)[1], 1e-12) > 0.1 for v in totals.values()
                )
                if not noisy:
                    break
            # 固定轮换顺序，降低系统负载/温度趋势与某个后端的绑定。
            order = labels[repeat % len(labels) :] + labels[: repeat % len(labels)]
            for label in order:
                result = self.run(f"{prefix}-{label}-r{repeat}", action, label, extra=extra)
                seconds = sum(row["evaluation_wall_s"] for row in result["records"])
                totals[label].append(seconds)

    def execute(self):
        group = self.group
        if group == "main":
            for label in ("cpu1", "cpu8", "v1", "v2"):
                self.run(f"sanity-{label}", "sanity", label)
            for label in ("cpu8", "v1", "v2"):
                self.run(
                    f"baseline-{label}",
                    "baseline",
                    label,
                    destination=self.output / "baselines" / label,
                )
            self.run("trace", "trace", "v2")
            self.repeats("E1", "train", ["v2", "cpu8"])
            self.repeats("E2", "replay", ["cpu1", "cpu8", "v1", "v2"])
        elif group == "gpu-setup":
            for label in ("v2-interp4", "v2-gen4"):
                self.run(f"sanity-{label}", "sanity", label)
                self.run(
                    f"baseline-{label}",
                    "baseline",
                    label,
                    destination=self.output / "baselines" / label,
                )
            import numpy as np

            with (
                np.load(self.output / "jobs/sanity-v2-interp4/outputs.npz") as left,
                np.load(self.output / "jobs/sanity-v2-gen4/outputs.npz") as right,
            ):
                np.testing.assert_array_equal(left["tours"], right["tours"])
                np.testing.assert_array_equal(left["lengths"], right["lengths"])
        elif group == "scans-prepare":
            self.run("scans-prepare", "scans", "cpu1")
        elif group == "jit":
            self.run("E3-jit", "jit", "cpu1")
        elif group == "scaling":
            # 全部扫描点固定主机，保证 CPU 曲线没有混合不同型号。
            for name in [f"p{p}-n100" for p in (1, 8, 32, 100, 256)] + ["p100-n50", "p100-n500"]:
                self.repeats(
                    f"E4-{name}",
                    "replay",
                    ["cpu8", "v2"],
                    extra=["--trace", self.output / "scans" / name],
                )
        elif group.startswith("ablation-"):
            generation = group[-1]
            self.repeats(
                f"E5-g{generation}",
                "replay",
                ["v1", "v2-interp4", "v2-gen4", "v2"],
                extra=["--generations", generation],
            )
        elif group.startswith("profile-"):
            label = group.removeprefix("profile-")
            self.profile(label)
        self.state.update(status="completed", completed_at=now())
        self.heartbeat()

    def profile(self, label):
        destination = self.output / "profiles" / label
        destination.mkdir(parents=True, exist_ok=True)
        nsys = shutil.which("nsys")
        counter_restricted = (
            "RmProfilingAdminOnly: 1" in Path("/proc/driver/nvidia/params").read_text()
        )
        atomic(
            destination / "ncu-status.json",
            {
                "status": "permission_denied" if counter_restricted else "available",
                "reason": "GPU performance counters require administrator access"
                if counter_restricted
                else None,
            },
        )
        if not nsys:
            atomic(
                destination / "nsys-status.json",
                {"status": "unsupported", "reason": "nsys executable missing"},
            )
            return
        version = subprocess.check_output([nsys, "--version"], text=True).strip()
        profiler = [
            nsys,
            "profile",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace=cuda,nvtx",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=true",
            "--output",
            str(destination / "timeline"),
        ]
        try:
            self.run(
                f"E6-{label}",
                "replay",
                label,
                extra=["--generations", "3"],
                profiler=profiler,
                destination=destination / "diagnostic",
            )
            status = {"status": "completed", "version": version}
            subprocess.run(
                [
                    nsys,
                    "export",
                    "--type=sqlite",
                    "--force-overwrite=true",
                    "--output",
                    str(destination / "timeline.sqlite"),
                    str(destination / "timeline.nsys-rep"),
                ],
                check=True,
            )
        except (RuntimeError, subprocess.CalledProcessError) as error:
            status = {"status": "unsupported", "version": version, "reason": str(error)}
        atomic(destination / "nsys-status.json", status)


def worker(output, group, gpu):
    lock_path = output / "locks" / (socket.gethostname().split(".")[0] + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Worker(output, group, gpu)
        runner.heartbeat()
        try:
            runner.execute()
        except Exception:
            runner.state.update(status="failed", error=traceback.format_exc(), completed_at=now())
            runner.heartbeat()
            raise


def launch_group(output, group, host, gpu):
    source = read(output / "source.json")
    script = Path(source["snapshot"]) / "scripts/run_presentation_benchmarks.py"
    log = output / "logs" / f"{group}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(script),
        "worker",
        "--output-root",
        str(output),
        "--group",
        group,
        "--gpu",
        str(gpu),
    ]
    remote = (
        "nohup "
        + shlex.join(command)
        + " > "
        + shlex.quote(str(log))
        + " 2>&1 < /dev/null & echo $!"
    )
    atomic(
        output / "groups" / f"{group}.json",
        {"status": "launching", "host": host, "gpu": gpu, "launched_at": now()},
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    print(f"launched {group}: {host}:{gpu} pid={proc.stdout.strip()}", flush=True)


def status(output):
    states = {
        group: read(output / "groups" / f"{group}.json", {"status": "pending"})
        for group in groups()
    }
    counts = {}
    for value in states.values():
        counts[value["status"]] = counts.get(value["status"], 0) + 1
    tasks = [read(p, {}) for p in (output / "jobs").glob("*/status.json")]
    payload = {
        "updated_at": now(),
        "groups": states,
        "counts": counts,
        "completed_tasks": sum(t.get("status") == "completed" for t in tasks),
        "running_tasks": [t.get("task") for t in tasks if t.get("status") == "running"],
    }
    atomic(output / "status.json", payload)
    return payload


def controller(output):
    lock_path = output / "controller.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(
            output / "controller.json",
            {"pid": os.getpid(), "host": socket.gethostname(), "started_at": now()},
        )
        while True:
            summary = status(output)
            busy_hosts = {
                s["host"]
                for s in summary["groups"].values()
                if s["status"] in ("running", "launching")
            }
            cards = discover(output)
            for host, gpu in cards:
                if host in busy_hosts:
                    continue
                pending = [
                    g
                    for g in groups()
                    if summary["groups"][g]["status"] == "pending" and ready(output, g)
                ]
                if not pending:
                    continue
                # 优先采用预先核对过的主机；新增空闲 A5000 承接完整独立组。
                desired = (
                    "main"
                    if host == "cuda02"
                    else "gpu-setup"
                    if host == "cuda08"
                    else "scaling"
                    if host == "cuda04"
                    else None
                )
                selected = desired if desired in pending else pending[0]
                if selected == "main" and host != "cuda02" and ("cuda02", 0) in cards:
                    continue
                try:
                    launch_group(output, selected, host, gpu)
                except (subprocess.SubprocessError, OSError) as error:
                    # SSH 失败不能让整个 nohup 调度器退出。远端已写运行状态时保留它。
                    state_path = output / "groups" / f"{selected}.json"
                    if read(state_path, {}).get("status") == "launching":
                        atomic(state_path, {"status": "pending", "last_launch_error": str(error)})
                    print(f"launch retry needed for {selected}: {error}", flush=True)
                    continue
                busy_hosts.add(host)
                summary["groups"][selected] = {"status": "launching", "host": host}
            report(output)
            summary = status(output)
            if all(
                s["status"] in ("completed", "failed", "blocked")
                for s in summary["groups"].values()
            ):
                break
            # 主输入生产任务失败时明确阻断依赖组，不保持无意义等待。
            if (
                summary["groups"]["main"]["status"] == "failed"
                and not (output / "trace/trace.json").exists()
            ):
                for group, state in summary["groups"].items():
                    if state["status"] == "pending" and group != "gpu-setup":
                        atomic(
                            output / "groups" / f"{group}.json",
                            {"status": "blocked", "reason": "trace production failed"},
                        )
            for prerequisite, dependents in (
                ("scans-prepare", ("scaling",)),
                ("gpu-setup", ("ablation-g3", "ablation-g5")),
            ):
                if summary["groups"][prerequisite]["status"] in ("failed", "blocked"):
                    for dependent in dependents:
                        if summary["groups"][dependent]["status"] == "pending":
                            atomic(
                                output / "groups" / f"{dependent}.json",
                                {"status": "blocked", "reason": f"{prerequisite} failed"},
                            )
            time.sleep(30)
        report(output)


def report(output):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MPLCONFIGDIR": str(output / ".mplcache")}
    with (output / "logs/report.log").open("a") as log:
        return subprocess.run(
            [str(PYTHON), "-m", "rmtgp_aco.presentation_report", "--output-root", str(output)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "launch", "controller", "worker", "status", "report")
    )
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--group")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    if args.action == "prepare":
        subprocess.run(
            [
                str(PYTHON),
                "-m",
                "rmtgp_aco.presentation_bench",
                "prepare",
                "--output-root",
                str(output),
            ],
            check=True,
        )
        snapshot(output)
        atomic(
            output / "queue.json",
            {"groups": groups(), "repeat_policy": "3; extend to 5 if range/median > 0.1"},
        )
    elif args.action == "launch":
        if not (output / "source.json").exists():
            raise RuntimeError("先执行 prepare 冻结代码")
        with (output / "logs/controller.log").open("a") as log:
            proc = subprocess.Popen(
                [
                    "nohup",
                    str(PYTHON),
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
        print(f"controller pid={proc.pid} output={output}")
    elif args.action == "controller":
        controller(output)
    elif args.action == "worker":
        worker(output, args.group, args.gpu)
    elif args.action == "status":
        print(json.dumps(status(output), indent=2))
    else:
        raise SystemExit(report(output))


if __name__ == "__main__":
    main()
