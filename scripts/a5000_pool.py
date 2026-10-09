"""空闲 A5000 持续调度器：每分钟 gpu-free，依赖就绪才派发，不重复计算或抢占。"""

import argparse
import csv
import fcntl
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from launch_pilots import remote

from gpaco.config import config_hash
from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import file_hash, safe_directory

IDLE_ROW = re.compile(r"^\s*IDLE\s+([A-Za-z0-9._-]+)\s+(\d+)\s+.*\bNVIDIA RTX A5000\s*$")
TERMINAL = {"completed", "failed", "blocked_dependency"}


def parse_idle_a5000(output):
    devices = []
    for line in output.splitlines():
        match = IDLE_ROW.fullmatch(line)
        if match:
            device = {"host": match[1], "index": int(match[2])}
            if device not in devices:
                devices.append(device)
    return devices


def tasks_for(config):
    """固定有限任务集；缺少晚期 cohort 是待依赖，不是允许复制早期种群的理由。"""
    tasks = []
    for scale_index, n in enumerate(config["scales"]):
        tasks.append(
            {
                "id": f"prepare-tsp{n}",
                "kind": "prepare",
                "n": n,
                "priority": scale_index,
                "status": "pending",
                "attempts": [],
            }
        )
        for generation in config["cohort_generations"]:
            cohort = (
                ROOT
                / "artifacts"
                / config["cohort_source_campaign"]
                / (
                    f"as-tsp{n}-seed{config['cohort_source_seed']}/cohorts/generation-{generation:03d}.json"
                )
            )
            for variant in config["variants"]:
                base = f"e01-tsp{n}-g{generation:03d}-{variant}"
                for block in range(config["paired_blocks"]):
                    tasks.append(
                        {
                            "id": f"{base}-b{block:02d}",
                            "kind": "pair",
                            "n": n,
                            "variant": variant,
                            "generation": generation,
                            "block": block,
                            "cohort_path": str(cohort),
                            "status": "pending",
                            "attempts": [],
                            "priority": 10 + scale_index * 1000 + block * 100 + generation,
                        }
                    )
                tasks.append(
                    {
                        "id": f"{base}-profile",
                        "kind": "profile",
                        "n": n,
                        "variant": variant,
                        "generation": generation,
                        "block": 0,
                        "cohort_path": str(cohort),
                        "pair_dependency": f"{base}-b00",
                        "status": "pending",
                        "attempts": [],
                        "priority": 800 + scale_index * 1000 + generation,
                    }
                )
    return tasks


def dependencies(task, tasks, campaign):
    if task["kind"] == "prepare":
        return "ready"
    prepare = next(t for t in tasks if t["id"] == f"prepare-tsp{task['n']}")
    if prepare["status"] in ("failed", "blocked_dependency"):
        return "failed:input_preparation"
    if not (campaign / "inputs" / f"tsp{task['n']}" / "READY.json").exists():
        return "waiting:inputs"
    if not Path(task["cohort_path"]).exists():
        return "waiting:real_cohort"
    if task.get("pair_dependency"):
        other = next(t for t in tasks if t["id"] == task["pair_dependency"])
        if other["status"] in ("failed", "blocked_dependency"):
            return "failed:paired_measurement"
        if other["status"] != "completed":
            return "waiting:paired_measurement"
    return "ready"


def scan(config, campaign):
    """保存 gpu-free 原始快照；超时只终止本次扫描创建的进程组。"""
    env = {**os.environ, "NO_COLOR": "1"}
    process = subprocess.Popen(
        [config["gpu_free_command"]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=config["scan_timeout_s"])
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise RuntimeError("gpu-free 扫描超时；本轮不根据过期结果派发") from None
    if process.returncode:
        raise RuntimeError(f"gpu-free 扫描失败：{output[-2000:]}")
    target = campaign / "scans" / f"{time.time_ns()}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output)
    return parse_idle_a5000(output)


def confirm_idle(device):
    """gpu-free 的空闲行只是候选；再次读取 UUID 和实际进程，拒绝只看 utilization。"""
    try:
        rows = remote(
            device["host"],
            (
                f"nvidia-smi -i {device['index']} "
                "--query-gpu=uuid,name,utilization.gpu,memory.used --format=csv,noheader,nounits"
            ),
        )
        row = next(csv.reader(rows.splitlines()))
        uuid, model = row[0].strip(), row[1].strip()
        apps = remote(
            device["host"],
            "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits",
        )
        pids = [r[1] for r in csv.reader(apps.splitlines()) if len(r) >= 2 and r[0].strip() == uuid]
        if model == "NVIDIA RTX A5000" and not pids and int(row[2]) <= 5 and int(row[3]) <= 1024:
            return {**device, "gpu_uuid": uuid, "model": model}
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as error:
        print(f"空闲复查失败 {device}: {error}", flush=True)
    return None


def worker_alive(attempt):
    """无法连接时返回 unknown，绝不因心跳慢就派发第二份相同科学任务。"""
    try:
        rows = remote(attempt["host"], "ps -eo pid=,args=")
    except (OSError, subprocess.SubprocessError):
        return None
    for row in rows.splitlines():
        if "run_main_benchmark.py" not in row:
            continue
        try:
            parts = shlex.split(row)
        except ValueError:
            continue
        if attempt["job_path"] in parts and "--job" in parts:
            return True
    return False


def reconcile(tasks):
    for task in tasks:
        if task["status"] != "running":
            continue
        attempt = task["attempts"][-1]
        directory = Path(attempt["job_path"]).parent
        if (directory / "COMPLETE.json").exists():
            task["status"] = "completed"
            task["result"] = str(directory / "COMPLETE.json")
        elif (directory / "FAILED.json").exists():
            task["status"] = "failed"
            task["result"] = str(directory / "FAILED.json")
        elif (directory / "REJECTED.json").exists():
            task["status"] = "pending"
            attempt["rejected"] = True
        elif time.time() - attempt["assigned_unix_s"] > 120:
            alive = worker_alive(attempt)
            attempt["last_liveness"] = alive
            if alive is False:
                # 已启动或启动不确定的任务不自动重算，避免重复样本和覆盖半成品。
                write_json(
                    directory / "FAILED.json",
                    {
                        "error": "worker 已退出且没有完成标记；需检查原日志后显式恢复",
                        "detected_by_controller": True,
                    },
                )
                task["status"], task["result"] = "failed", str(directory / "FAILED.json")


def dispatch(task, device, manifest, campaign, tasks):
    index = len(task["attempts"]) + 1
    directory = campaign / "tasks" / task["id"] / f"attempt-{index:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    job_path = directory / "job.json"
    immutable_task = {
        key: value
        for key, value in task.items()
        if key not in ("attempts", "status", "dependency_state", "result")
    }
    write_json(
        job_path,
        {
            **device,
            "task": immutable_task,
            "config": manifest["config"],
            "snapshot": manifest["snapshot"],
            "commit": manifest["commit"],
            "campaign_directory": str(campaign),
        },
    )
    attempt = {**device, "job_path": str(job_path), "assigned_unix_s": time.time()}
    task["attempts"].append(attempt)
    task["status"] = "running"
    # 先登记租约再启动，控制器意外退出后仍能找到这个唯一 attempt。
    write_json(campaign / "queue.json", tasks)
    snapshot = Path(manifest["snapshot"])
    script = (
        f"cd {shlex.quote(str(ROOT))}\nsource scripts/env.sh\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(device['gpu_uuid'])}\n"
        f"export GPACO_COMMIT={shlex.quote(manifest['commit'])} GPACO_SNAPSHOT={shlex.quote(str(snapshot))}\n"
        f"export PYTHONPATH={shlex.quote(str(snapshot / 'src'))}\n"
        f"nohup setsid python {shlex.quote(str(snapshot / 'scripts/run_main_benchmark.py'))} "
        f"--job {shlex.quote(str(job_path))} > {shlex.quote(str(directory / 'worker.log'))} "
        "2>&1 < /dev/null &\n"
        'GPACO_POOL_WORKER_PID=$!\nsleep 1\nps -p "$GPACO_POOL_WORKER_PID" -o pid=,args=\n'
    )
    try:
        result = remote(device["host"], "bash -c " + shlex.quote(script))
        attempt["launch_ack"] = result.strip()
        print(
            f"派发 {task['id']} → {device['host']} GPU{device['index']}: {result.strip()}",
            flush=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        attempt["launch_uncertain"] = str(error)
        print(f"启动确认不确定，保留租约并复查：{task['id']}: {error}", flush=True)
    write_json(campaign / "queue.json", tasks)


def write_status(campaign, tasks, **extra):
    counts = dict(Counter(t["status"] for t in tasks))
    waiting = dict(
        Counter(t.get("dependency_state", "not_checked") for t in tasks if t["status"] == "pending")
    )
    value = {
        "pid": os.getpid(),
        "time": time.time(),
        "counts": counts,
        "pending_dependencies": waiting,
        **extra,
    }
    write_json(campaign / "dispatcher_status.json", value)
    write_json(campaign / "queue.json", tasks)
    with (campaign / "dispatcher_events.jsonl").open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def controller(campaign):
    manifest = json.loads((campaign / "campaign.json").read_text())
    config = manifest["config"]

    def refresh_report():
        with (campaign / "report.log").open("a") as handle:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(manifest["snapshot"]) / "scripts/report_a5000_main.py"),
                    "--campaign",
                    str(campaign),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

    with (campaign / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = json.loads((campaign / "queue.json").read_text())
        while True:
            started = time.monotonic()
            reconcile(tasks)
            if (campaign / "STOP_DISPATCH").exists():
                write_status(campaign, tasks, phase="stopped_by_file; running_workers_continue")
                refresh_report()
                break
            for task in tasks:
                if task["status"] == "pending":
                    task["dependency_state"] = dependencies(task, tasks, campaign)
                    if task["dependency_state"].startswith("failed:"):
                        task["status"] = "blocked_dependency"
            if all(t["status"] in TERMINAL for t in tasks):
                write_status(campaign, tasks, phase="finished")
                write_json(
                    campaign / "FINISHED.json",
                    {
                        "all_successful": all(t["status"] == "completed" for t in tasks),
                        "counts": dict(Counter(t["status"] for t in tasks)),
                    },
                )
                refresh_report()
                break
            try:
                candidates = scan(config, campaign)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    devices = [device for device in pool.map(confirm_idle, candidates) if device]
                reserved = {
                    t["attempts"][-1]["gpu_uuid"] for t in tasks if t["status"] == "running"
                }
                ready = sorted(
                    [
                        t
                        for t in tasks
                        if t["status"] == "pending" and t["dependency_state"] == "ready"
                    ],
                    key=lambda t: (t["priority"], t["id"]),
                )
                used = []
                for device in devices:
                    if device["gpu_uuid"] in reserved:
                        continue
                    if (campaign / "qualifications" / device["gpu_uuid"] / "FAILED.json").exists():
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
                    dispatch(task, device, manifest, campaign, tasks)
                    ready.remove(task)
                    reserved.add(device["gpu_uuid"])
                    used.append(device)
                write_status(campaign, tasks, phase="scanning", available=devices, dispatched=used)
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                write_status(campaign, tasks, phase="scan_error", error=str(error))
            refresh_report()
            # 后台调度器等待不占 GPU，主对话无需保持打开。
            time.sleep(max(1, config["scan_interval_s"] - (time.monotonic() - started)))


def launch(args):
    config_path = safe_directory(ROOT / args.config)
    config = yaml.safe_load(config_path.read_text())
    campaign = safe_directory(ROOT / "artifacts" / config["name"])
    if not args.execute:
        tasks = tasks_for(config)
        print(
            json.dumps(
                {
                    "campaign": str(campaign),
                    "task_counts": dict(Counter(t["kind"] for t in tasks)),
                    "scan_interval_s": config["scan_interval_s"],
                },
                indent=2,
            )
        )
        return
    if args.resume:
        manifest = json.loads((campaign / "campaign.json").read_text())
        if manifest["config_hash"] != config_hash(config):
            raise ValueError("恢复调度器不能改变任务协议")
        if (campaign / "STOP_DISPATCH").exists():
            raise ValueError("STOP_DISPATCH 仍存在；请明确撤销停止后再恢复")
        with (campaign / "dispatcher.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("调度器已经运行，无须重复启动") from error
    else:
        if campaign.exists():
            raise FileExistsError("已有主实验队列；只能显式 --resume，不覆盖任务")
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
            raise ValueError("请先提交代码/配置/测试，再启动不可变主实验队列")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        snapshot = ROOT / "artifacts/runtime" / commit
        if not snapshot.exists():
            archive = subprocess.check_output(
                ["git", "archive", commit, "src", "scripts", "configs", "tests", "pyproject.toml"],
                cwd=ROOT,
            )
            snapshot.mkdir(parents=True)
            with tarfile.open(fileobj=io.BytesIO(archive)) as package:
                package.extractall(snapshot, filter="data")
        campaign.mkdir(parents=True)
        manifest = {
            "config": config,
            "config_hash": config_hash(config),
            "commit": commit,
            "snapshot": str(snapshot),
            "gpu_free_sha256": file_hash(config["gpu_free_command"]),
            "standard_test_opened": False,
        }
        write_json(campaign / "campaign.json", manifest)
        write_json(campaign / "queue.json", tasks_for(config))
    # 本地控制器也运行已提交快照，不随工作树编辑变化。
    snapshot = Path(manifest["snapshot"])
    env = {
        **os.environ,
        "PYTHONPATH": str(snapshot / "src"),
        "GPACO_SNAPSHOT": str(snapshot),
        "GPACO_COMMIT": manifest["commit"],
    }
    with (campaign / "dispatcher.log").open("a") as log:
        process = subprocess.Popen(
            [
                "nohup",
                sys.executable,
                str(snapshot / "scripts/a5000_pool.py"),
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
        campaign / "dispatcher_launched.json",
        {"pid": process.pid, "time": time.time(), "commit": manifest["commit"]},
    )
    print(
        json.dumps(
            {"dispatcher_pid": process.pid, "campaign": str(campaign), "commit": manifest["commit"]}
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/workloads/a5000_main_queue.yaml")
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
