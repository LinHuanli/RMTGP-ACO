"""登记后的有限诊断队列：每分钟发现空闲A5000，真实cohort就绪后自动接续。"""

import argparse
import fcntl
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from a5000_pool import confirm_idle, scan
from launch_pilots import remote
from queue_state import reconcile_tasks, validate_diagnostic

from gpaco.artifact_registry import identity, resolve
from gpaco.benchmark_inputs import BenchmarkInputs, export_bundle
from gpaco.config import config_hash
from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import file_hash, safe_directory

TERMINAL = {"completed", "failed", "excluded_contended"}


def tasks_for(config):
    return [
        {
            "id": f"tsp{n}-{variant}-g{generation:03d}-b{block:03d}",
            "n": n,
            "variant": variant,
            "generation": generation,
            "block": block,
            "status": "pending",
            "attempts": [],
        }
        for generation in config["cohort_generations"]
        for block in config["blocks"]
        for n in config["scales"]
        for variant in config["variants"]
    ]


def bundle_for(task, config, *, prepare=True):
    """只导出实际产生的晚期种群；每个新增输入包不可覆盖。"""
    base = resolve(config["input_registry_id"]) / f"tsp{task['n']}/{task['variant']}"
    data = BenchmarkInputs(base)
    if str(task["generation"]) in data.manifest["cohorts"]:
        return base
    cohort = resolve("E09-p01-training") / (
        f"as-tsp{task['n']}-seed{config['cohort_root_seed']}/cohorts/generation-{task['generation']:03d}.json"
    )
    if not cohort.exists():
        return None
    target = resolve("shared-diagnostic-cohorts-p01") / (
        f"tsp{task['n']}/g{task['generation']:03d}/{task['variant']}"
    )
    if not (target / "READY.json").exists():
        if not prepare:
            return target
        if target.exists():
            raise RuntimeError(f"输入包未完成；保留现场，停止自动覆盖：{target}")
        source = resolve("E01-p01-gpu-baselines") / f"inputs/tsp{task['n']}/{task['variant']}"
        export_bundle(source, {task["generation"]: cohort}, target)
    BenchmarkInputs(target)
    return target


def worker_alive(attempt):
    try:
        output = remote(attempt["host"], "ps -eo pid=,args=")
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        if "run_diagnostic_worker.py" not in line:
            continue
        try:
            args = shlex.split(line)
        except ValueError:
            continue
        if "--job" in args and attempt["job_path"] in args:
            return True
    return False


def reconcile(tasks):
    reconcile_tasks(tasks, worker_alive, validate_diagnostic)


def dispatch(task, device, campaign, manifest, tasks):
    attempt_id = len(task["attempts"]) + 1
    directory = resolve("E01-p01-work-diagnostics") / f"{task['id']}-a{attempt_id:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    job_path = directory / "job.json"
    bundle = Path(task["bundle"])
    job = {
        **device,
        **identity("E01-p01-work-diagnostics"),
        "config": manifest["config"],
        "commit": manifest["commit"],
        "snapshot": manifest["snapshot"],
        "campaign_directory": str(campaign),
        "bundle": str(bundle),
        "bundle_sha256": file_hash(bundle / "manifest.json"),
        "task": {k: task[k] for k in ("id", "n", "variant", "generation", "block")},
        "attempt": attempt_id,
    }
    write_json(job_path, job)
    attempt = {**device, "job_path": str(job_path), "assigned_unix_s": time.time()}
    task["attempts"].append(attempt)
    task["status"] = "running"
    write_json(campaign / "queue.json", tasks)
    snapshot = Path(manifest["snapshot"])
    script = (
        f"cd {shlex.quote(str(ROOT))}\nsource scripts/env.sh\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(device['gpu_uuid'])}\n"
        f"export GPACO_COMMIT={shlex.quote(manifest['commit'])} GPACO_SNAPSHOT={shlex.quote(str(snapshot))}\n"
        f"export PYTHONPATH={shlex.quote(str(snapshot / 'src'))}\n"
        f"nohup setsid python {shlex.quote(str(snapshot / 'scripts/run_diagnostic_worker.py'))} "
        f"--job {shlex.quote(str(job_path))} > {shlex.quote(str(directory / 'worker.log'))} "
        "2>&1 < /dev/null &\n"
        'GPACO_DIAGNOSTIC_PID=$!\nsleep 1\nps -p "$GPACO_DIAGNOSTIC_PID" -o pid=,args=\n'
    )
    try:
        attempt["launch_ack"] = remote(device["host"], "bash -c " + shlex.quote(script)).strip()
    except (OSError, subprocess.SubprocessError) as error:
        attempt["launch_uncertain"] = str(error)
    write_json(campaign / "queue.json", tasks)
    print(f"派发 {task['id']} -> {device['host']} GPU{device['index']}", flush=True)


def controller(campaign):
    manifest = json.loads((campaign / "campaign.json").read_text())
    config = manifest["config"]
    with (campaign / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = json.loads((campaign / "queue.json").read_text())
        while True:
            start = time.monotonic()
            reconcile(tasks)
            stop = (campaign / "STOP_DISPATCH").exists()
            done = all(t["status"] in TERMINAL for t in tasks)
            error = None
            devices, started = [], []
            if not stop and not done:
                try:
                    ready = []
                    for task in tasks:
                        if task["status"] != "pending":
                            continue
                        bundle = bundle_for(task, config)
                        task["dependency"] = "ready" if bundle else "waiting:real_cohort"
                        if bundle:
                            task["bundle"] = str(bundle)
                            ready.append(task)
                    candidates = scan(config, campaign)
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        devices = [d for d in pool.map(confirm_idle, candidates) if d]
                    reserved = {
                        t["attempts"][-1]["gpu_uuid"] for t in tasks if t["status"] == "running"
                    }
                    for device in devices:
                        if device["gpu_uuid"] in reserved:
                            continue
                        if (
                            campaign / "qualifications" / device["gpu_uuid"] / "FAILED.json"
                        ).exists():
                            continue
                        task = next(
                            (
                                t
                                for t in ready
                                if sum(
                                    a.get("rejected", False) and a["gpu_uuid"] == device["gpu_uuid"]
                                    for a in t["attempts"]
                                )
                                < config["max_startup_rejections_per_gpu"]
                            ),
                            None,
                        )
                        if task is None:
                            continue
                        dispatch(task, device, campaign, manifest, tasks)
                        reserved.add(device["gpu_uuid"])
                        ready.remove(task)
                        started.append(task["id"])
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exception:
                    error = repr(exception)
                    print(f"本轮调度异常，保留已登记任务：{error}", flush=True)
            status = {
                "time_unix_s": time.time(),
                "pid": os.getpid(),
                "counts": dict(Counter(t["status"] for t in tasks)),
                "waiting": dict(
                    Counter(t.get("dependency") for t in tasks if t["status"] == "pending")
                ),
                "available": devices,
                "dispatched": started,
                "error": error,
                "phase": "stopped" if stop else "finished" if done else "scanning",
            }
            write_json(campaign / "queue.json", tasks)
            write_json(campaign / "dispatcher_status.json", status)
            with (campaign / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(status) + "\n")
            if stop or done:
                if done:
                    write_json(campaign / "FINISHED.json", status)
                return
            time.sleep(max(1, config["scan_interval_s"] - (time.monotonic() - start)))


def launch(args):
    campaign = resolve("shared-diagnostic-dispatch")
    config = yaml.safe_load((ROOT / "configs/workloads/diagnostic_queue.yaml").read_text())
    tasks = tasks_for(config)
    if not args.execute:
        ready = sum(bundle_for(t, config, prepare=False) is not None for t in tasks)
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "ready": ready,
                    "waiting": len(tasks) - ready,
                    "campaign": str(campaign),
                }
            )
        )
        return
    if args.resume:
        manifest = json.loads((campaign / "campaign.json").read_text())
        if manifest["config_hash"] != config_hash(config):
            raise ValueError("恢复不可改变诊断协议")
        if (campaign / "STOP_DISPATCH").exists():
            raise ValueError("STOP_DISPATCH仍存在")
    else:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        )
        untracked = subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "src",
                "scripts",
                "configs",
                "tests",
            ],
            cwd=ROOT,
            text=True,
        )
        if dirty or untracked:
            raise ValueError("先提交源码、配置和测试，再创建不可变快照")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        snapshot = resolve("shared-managed-source-snapshots") / commit
        if not snapshot.exists():
            archive = subprocess.check_output(
                ["git", "archive", commit, "src", "scripts", "configs", "tests", "pyproject.toml"],
                cwd=ROOT,
            )
            snapshot.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(archive)) as package:
                package.extractall(snapshot, filter="data")
        campaign.mkdir(parents=True, exist_ok=False)
        manifest = {
            "config": config,
            "config_hash": config_hash(config),
            "commit": commit,
            "snapshot": str(snapshot),
            **identity("shared-diagnostic-dispatch"),
        }
        write_json(campaign / "campaign.json", manifest)
        write_json(campaign / "queue.json", tasks)
    with (campaign / "dispatcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("诊断控制器已在运行") from error
    runtime = campaign / "controller_runtime.json"
    control = json.loads(runtime.read_text()) if runtime.exists() else manifest
    snapshot = Path(control["snapshot"])
    env = {
        **os.environ,
        "PYTHONPATH": str(snapshot / "src"),
        "GPACO_SNAPSHOT": str(snapshot),
        "GPACO_COMMIT": control["commit"],
    }
    with (campaign / "dispatcher.log").open("a") as log:
        process = subprocess.Popen(
            [
                "nohup",
                sys.executable,
                str(snapshot / "scripts/diagnostic_pool.py"),
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
    write_json(
        campaign / "launched.json",
        {
            "pid": process.pid,
            "time_unix_s": time.time(),
            "commit": control["commit"],
            "worker_commit": manifest["commit"],
        },
    )
    print(json.dumps({"pid": process.pid, "campaign": str(campaign), "tasks": len(tasks)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--campaign")
    args = parser.parse_args()
    if args.run:
        controller(safe_directory(args.campaign))
    else:
        launch(args)


if __name__ == "__main__":
    main()
