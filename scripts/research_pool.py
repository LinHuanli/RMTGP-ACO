"""正式基线优先的有限A5000队列；利用剩余卡跑配对映射实验。"""

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from a5000_pool import confirm_idle, scan
from campaign_runtime import snapshot_head, start_controller
from diagnostic_pool import bundle_for
from launch_pilots import remote
from queue_state import read_json, reconcile_tasks

from gpaco.artifact_registry import identity, require_output, resolve
from gpaco.config import config_hash
from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import file_hash, safe_directory

TERMINAL = {"completed", "failed", "excluded_contended"}


def tasks_for(config):
    formal, mapping = config["formal"], config["mapping"]
    tasks = [
        {
            "id": f"as-tsp{n}-seed{seed}",
            "kind": "formal_train",
            "n": n,
            "seed": seed,
            "registry_id": formal["registry_id"],
            "status": "pending",
            "attempts": [],
        }
        for n in formal["scales"]
        for seed in formal["seeds"]
    ]
    tasks += [
        {
            "id": f"as-tsp{n}-g{generation:03d}-b{block:03d}",
            "kind": "mapping",
            "n": n,
            "variant": "as",
            "generation": generation,
            "block": block,
            "registry_id": mapping["registry_id"],
            "status": "pending",
            "attempts": [],
        }
        for generation in mapping["cohort_generations"]
        for block in mapping["blocks"]
        for n in mapping["scales"]
    ]
    return tasks


def validate_completion(directory, complete):
    job = read_json(directory / "job.json")
    if complete.get("task_id") != job["task"]["id"]:
        raise ValueError("完成标记与job身份不一致")
    if job["task"]["kind"] == "formal_train":
        for relative, key in (
            ("training/COMPLETE.json", "training_complete_sha256"),
            ("training/history.json", "history_sha256"),
        ):
            if file_hash(directory / relative) != complete[key]:
                raise ValueError("正式训练完成证据哈希不一致")
        if file_hash(Path(job["inputs"]) / "manifest.json") != complete["input_manifest_sha256"]:
            raise ValueError("正式输入发生变化")
    else:
        for cell, expected in complete["record_hashes"].items():
            record = directory / "measurements" / cell / "record.json"
            if file_hash(record) != expected:
                raise ValueError("映射测量哈希不一致")
            value = read_json(record)
            if (
                value["status"] == "completed"
                and file_hash(record.parent / "result.npz") != value["result_sha256"]
            ):
                raise ValueError("映射原始输出哈希不一致")


def worker_alive(attempt):
    try:
        lines = remote(attempt["host"], "ps -eo pid=,args=").splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    for line in lines:
        if "run_research_worker.py" not in line:
            continue
        try:
            args = shlex.split(line)
        except ValueError:
            continue
        if "--job" in args and attempt["job_path"] in args:
            return True
    return False


def dispatch(task, device, campaign, manifest, tasks):
    count = len(task["attempts"]) + 1
    directory = require_output(
        resolve(task["registry_id"]) / f"{task['id']}-a{count:02d}", task["registry_id"]
    )
    directory.mkdir(parents=True, exist_ok=False)
    job_path = directory / "job.json"
    job = {
        **device,
        **identity(task["registry_id"]),
        "config": manifest["config"],
        "commit": manifest["commit"],
        "snapshot": manifest["snapshot"],
        "campaign_directory": str(campaign),
        "task": {k: v for k, v in task.items() if k not in ("attempts", "status")},
    }
    if task["kind"] == "formal_train":
        job["inputs"] = str(
            resolve(manifest["config"]["formal"]["input_registry_id"])
            / f"{task['id']}-a{count:02d}"
        )
    else:
        job.update(
            bundle=task["bundle"], bundle_sha256=file_hash(Path(task["bundle"]) / "manifest.json")
        )
    write_json(job_path, job)
    attempt = {**device, "job_path": str(job_path), "assigned_unix_s": time.time()}
    task["attempts"].append(attempt)
    task["status"] = "running"
    write_json(campaign / "queue.json", tasks)
    snapshot = Path(manifest["snapshot"])
    cache = directory / "runtime-cache"
    script = (
        f"cd {shlex.quote(str(ROOT))}\nsource scripts/env.sh\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(device['gpu_uuid'])}\n"
        f"export GPACO_COMMIT={manifest['commit']} GPACO_SNAPSHOT={shlex.quote(str(snapshot))}\n"
        f"export PYTHONPATH={shlex.quote(str(snapshot / 'src'))}\n"
        f"export CUPY_CACHE_DIR={shlex.quote(str(cache / 'cupy'))}\n"
        f"export CUDA_CACHE_PATH={shlex.quote(str(cache / 'driver'))}\n"
        f"mkdir -p {shlex.quote(str(cache / 'cupy'))} {shlex.quote(str(cache / 'driver'))}\n"
        f"nohup setsid python {shlex.quote(str(snapshot / 'scripts/run_research_worker.py'))} "
        f"--job {shlex.quote(str(job_path))} > {shlex.quote(str(directory / 'worker.log'))} "
        '2>&1 < /dev/null &\nGPACO_RESEARCH_PID=$!\nsleep 1\nps -p "$GPACO_RESEARCH_PID" -o pid=,args=\n'
    )
    try:
        attempt["launch_ack"] = remote(device["host"], "bash -c " + shlex.quote(script)).strip()
    except (OSError, subprocess.SubprocessError) as error:
        attempt["launch_uncertain"] = str(error)
    write_json(campaign / "queue.json", tasks)
    print(f"派发 {task['kind']}/{task['id']} -> {device['host']} GPU{device['index']}", flush=True)


def legacy_priority_waiting():
    """旧队列出现就绪任务且控制器健康时，新补充队列让其先拿空卡。"""
    for registry in ("E01-p01-gpu-baselines", "shared-diagnostic-dispatch"):
        root = resolve(registry)
        status = read_json(root / "dispatcher_status.json")
        if not status or time.time() - status.get("time_unix_s", status.get("time", 0)) > 180:
            continue
        if not status.get("phase", "").startswith("scanning"):
            continue
        key = "pending_dependencies" if registry == "E01-p01-gpu-baselines" else "waiting"
        if status.get(key, {}).get("ready", 0):
            return True
    return False


def controller(campaign):
    manifest = read_json(campaign / "campaign.json")
    config = manifest["config"]
    with (campaign / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = read_json(campaign / "queue.json")
        while True:
            start = time.monotonic()
            reconcile_tasks(tasks, worker_alive, validate_completion)
            stopped = (campaign / "STOP_DISPATCH").exists()
            done = all(t["status"] in TERMINAL for t in tasks)
            error, launched, devices = None, [], []
            try:
                ready = []
                for task in tasks:
                    if task["status"] != "pending":
                        continue
                    available = True
                    if task["kind"] == "mapping":
                        bundle = bundle_for(task, config["mapping"])
                        available = bundle is not None
                        if available:
                            task["bundle"] = str(bundle)
                    task["dependency"] = "ready" if available else "waiting:real_cohort"
                    if available:
                        ready.append(task)
                if not stopped and not done and ready and not legacy_priority_waiting():
                    candidates = scan(config, campaign)
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        devices = [d for d in pool.map(confirm_idle, candidates) if d]
                    reserved = {
                        t["attempts"][-1]["gpu_uuid"] for t in tasks if t["status"] == "running"
                    }
                    for device in devices:
                        if (
                            device["gpu_uuid"] in reserved
                            or (
                                campaign / "qualifications" / device["gpu_uuid"] / "FAILED.json"
                            ).exists()
                        ):
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
                        ready.remove(task)
                        reserved.add(device["gpu_uuid"])
                        launched.append(task["id"])
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exception:
                error = repr(exception)
                print(error, flush=True)
            status = {
                "pid": os.getpid(),
                "time_unix_s": time.time(),
                "counts": dict(Counter(t["status"] for t in tasks)),
                "by_kind": {
                    kind: dict(Counter(t["status"] for t in tasks if t["kind"] == kind))
                    for kind in ("formal_train", "mapping")
                },
                "waiting": dict(
                    Counter(t.get("dependency") for t in tasks if t["status"] == "pending")
                ),
                "dispatched": launched,
                "available": devices,
                "error": error,
                "phase": "stopped" if stopped else "finished" if done else "scanning",
            }
            write_json(campaign / "queue.json", tasks)
            write_json(campaign / "dispatcher_status.json", status)
            with (campaign / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(status) + "\n")
            if stopped or done:
                if done:
                    write_json(campaign / "FINISHED.json", status)
                return
            time.sleep(max(1, config["scan_interval_s"] - (time.monotonic() - start)))


def launch(args):
    config = yaml.safe_load((ROOT / "configs/workloads/research_queue.yaml").read_text())
    tasks = tasks_for(config)
    campaign = resolve(config["registry_id"])
    if not args.execute:
        print(
            json.dumps(
                {
                    "campaign": str(campaign),
                    "tasks": len(tasks),
                    "formal_training_runs": sum(t["kind"] == "formal_train" for t in tasks),
                    "mapping_paired_blocks": sum(t["kind"] == "mapping" for t in tasks),
                    "mapping_evaluations": sum(t["kind"] == "mapping" for t in tasks) * 12,
                }
            )
        )
        return
    if args.resume:
        manifest = read_json(campaign / "campaign.json")
        if manifest["config_hash"] != config_hash(config) or (campaign / "STOP_DISPATCH").exists():
            raise ValueError("配置改变或STOP_DISPATCH存在，拒绝恢复")
    else:
        runtime = snapshot_head()
        campaign.mkdir(parents=True, exist_ok=False)
        manifest = {
            **runtime,
            "config": config,
            "config_hash": config_hash(config),
            **identity(config["registry_id"]),
            "test_sets_opened": False,
        }
        write_json(campaign / "campaign.json", manifest)
        write_json(campaign / "queue.json", tasks)
    with (campaign / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = start_controller(campaign, manifest, "research_pool.py")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--campaign")
    args = parser.parse_args()
    controller(safe_directory(args.campaign)) if args.run else launch(args)


if __name__ == "__main__":
    main()
