"""三宿主2-opt/3-opt持续队列。有限清单、显式门禁、不可变源码、2:1交替派发。"""

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

REGISTRIES = {
    "tuning": "E12-p02-ls-tuning",
    "smoke": "E00-p02-ls-qualification",
    "training": "E09-p02-ls-training",
    "performance": "E12-p02-ls-performance",
    "profile": "E01-p02-ls-profile",
}
TERMINAL = {"completed", "failed", "excluded_contended", "blocked_dependency"}


def tasks_for(cfg):
    tasks = []

    def add(identifier, kind, n, variant, dependencies=(), **fields):
        tasks.append(
            {
                "id": identifier,
                "kind": kind,
                "n": n,
                "variant": variant,
                **fields,
                "dependencies": list(dependencies),
                "registry_id": REGISTRIES[kind],
                "status": "pending",
                "attempts": [],
            }
        )

    for n in cfg["scales"]:
        for variant in cfg["variants"]:
            smoke_gates = []
            for mode in cfg["modes"]:
                key = f"{variant}-tsp{n}-{mode}"
                tuning_id, smoke_id = "tune-" + key, "smoke-" + key
                add(tuning_id, "tuning", n, variant, mode=mode, generation=1, block=0)
                add(
                    smoke_id, "smoke", n, variant, [tuning_id], mode=mode, seed=cfg["smoke"]["seed"]
                )
                smoke_gates.append(smoke_id)
                for seed in cfg["seeds"]:
                    add(
                        f"train-{key}-seed{seed}",
                        "training",
                        n,
                        variant,
                        [tuning_id, smoke_id],
                        mode=mode,
                        seed=seed,
                    )
                add(
                    "profile-" + key,
                    "profile",
                    n,
                    variant,
                    [tuning_id, smoke_id],
                    mode=mode,
                    generation=1,
                    block=0,
                )
            for generation in cfg["cohort_generations"]:
                for block in cfg["blocks"]:
                    add(
                        f"pair-{variant}-tsp{n}-g{generation:03d}-b{block:03d}",
                        "performance",
                        n,
                        variant,
                        smoke_gates,
                        generation=generation,
                        block=block,
                    )
    return tasks


def mark_dependencies(tasks):
    """科学失败不无限重试，也不让其依赖一直伪装成即将开始。"""
    lookup = {t["id"]: t for t in tasks}
    for task in tasks:
        if task["status"] != "pending":
            continue
        dependencies = [lookup[key] for key in task["dependencies"]]
        blocked = [d["id"] for d in dependencies if d["status"] in TERMINAL - {"completed"}]
        if blocked:
            task["status"], task["blocked_by"] = "blocked_dependency", blocked
        task["dependency"] = (
            "ready"
            if all(d["status"] == "completed" for d in dependencies)
            else "waiting:qualification_tuning_smoke"
        )
    return [t for t in tasks if t["status"] == "pending" and t["dependency"] == "ready"]


def choose_task(ready, position, cfg):
    """2个训练槽、1个测量槽；一类没有就绪任务时借用，不故意让GPU空转。"""
    train_share = cfg["training_dispatch_share"]
    period = train_share + cfg["measurement_dispatch_share"]
    want_training = position % period < train_share
    selected = [t for t in ready if (t["kind"] in ("training", "smoke")) == want_training]
    return min(
        selected or ready,
        key=lambda t: (
            -t["n"],
            {"tuning": 0, "smoke": 1, "training": 2, "profile": 3, "performance": 4}[t["kind"]],
            t["id"],
        ),
    )


def validate_completion(directory, complete):
    job = read_json(directory / "job.json")
    if (
        complete["task_id"] != job["task"]["id"]
        or complete["contract_sha256"] != job["contract_sha256"]
    ):
        raise ValueError("局部搜索完成身份不一致")
    for relative, expected in complete.get("record_hashes", {}).items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or file_hash(path) != expected:
            raise ValueError("测量记录哈希不一致")
        row = read_json(path)
        if (
            row["status"] == "completed"
            and file_hash(path.parent / "result.npz") != row["result_sha256"]
        ):
            raise ValueError("测量原始数组发生改变")
    if job["task"]["kind"] == "tuning":
        path = directory / "tuning-input/manifest.json"
        if file_hash(path) != complete["tuning_input_sha256"]:
            raise ValueError("调优输入身份改变")
        for relative, expected in read_json(path)["files"].items():
            target = path.parent / relative
            if (
                not target.resolve().is_relative_to(path.parent.resolve())
                or file_hash(target) != expected
            ):
                raise ValueError("调优输入数组改变")
    if job["task"]["kind"] in ("training", "smoke"):
        for relative, field in (
            ("training/history.json", "history_sha256"),
            ("training/COMPLETE.json", "training_complete_sha256"),
        ):
            if file_hash(directory / relative) != complete[field]:
                raise ValueError("训练输出哈希不一致")
        if file_hash(Path(job["inputs"]) / "manifest.json") != complete["input_manifest_sha256"]:
            raise ValueError("预生成输入发生改变")


def worker_alive(attempt):
    try:
        output = remote(attempt["host"], "ps -eo pid=,args=")
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        if "run_local_search_worker.py" not in line:
            continue
        try:
            args = shlex.split(line)
        except ValueError:
            continue
        if "--job" in args and attempt["job_path"] in args:
            return True
    return False


def dispatch(task, device, campaign, manifest, tasks):
    cfg = manifest["config"]
    count = len(task["attempts"]) + 1
    directory = require_output(
        resolve(task["registry_id"]) / f"{task['id']}-a{count:02d}", task["registry_id"]
    )
    directory.mkdir(parents=True, exist_ok=False)
    job_path = directory / "job.json"
    job = {
        **device,
        **identity(task["registry_id"]),
        "config": cfg,
        "commit": manifest["commit"],
        "snapshot": manifest["snapshot"],
        "contract_sha256": manifest["contract_sha256"],
        "campaign_directory": str(campaign),
        "task": {k: v for k, v in task.items() if k not in ("attempts", "status")},
        "gate_hashes": {},
    }
    for dependency in task["dependencies"]:
        gate = next(t for t in tasks if t["id"] == dependency)
        path = Path(gate["result"])
        job["gate_hashes"][str(path)] = file_hash(path)
        if gate["kind"] == "tuning":
            job["selection"] = str(path)
    if task["kind"] in ("smoke", "training"):
        job["inputs"] = str(
            resolve("shared-ls-training-inputs-p02") / task["kind"] / f"{task['id']}-a{count:02d}"
        )
    elif task["kind"] != "tuning":
        job.update(
            bundle=task["bundle"], bundle_sha256=file_hash(Path(task["bundle"]) / "manifest.json")
        )
    write_json(job_path, job)
    attempt = {**device, "job_path": str(job_path), "assigned_unix_s": time.time()}
    task["attempts"].append(attempt)
    task["status"] = "running"
    write_json(campaign / "queue.json", tasks)
    snapshot, cache = Path(manifest["snapshot"]), directory / "runtime-cache"
    command = (
        f"cd {shlex.quote(str(ROOT))}\nsource scripts/env.sh\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(device['gpu_uuid'])}\n"
        f"export GPACO_COMMIT={manifest['commit']} GPACO_SNAPSHOT={shlex.quote(str(snapshot))}\n"
        f"export PYTHONPATH={shlex.quote(str(snapshot / 'src'))}\n"
        f"export CUPY_CACHE_DIR={shlex.quote(str(cache / 'cupy'))}\n"
        f"export CUDA_CACHE_PATH={shlex.quote(str(cache / 'driver'))}\n"
        f"mkdir -p {shlex.quote(str(cache / 'cupy'))} {shlex.quote(str(cache / 'driver'))}\n"
        f"nohup setsid python {shlex.quote(str(snapshot / 'scripts/run_local_search_worker.py'))} "
        f"--job {shlex.quote(str(job_path))} > {shlex.quote(str(directory / 'worker.log'))} "
        '2>&1 < /dev/null &\nGPACO_LS_PID=$!\nsleep 1\nps -p "$GPACO_LS_PID" -o pid=,args=\n'
    )
    try:
        attempt["launch_ack"] = remote(device["host"], "bash -c " + shlex.quote(command)).strip()
    except (OSError, subprocess.SubprocessError) as error:
        attempt["launch_uncertain"] = str(error)
    write_json(campaign / "queue.json", tasks)
    print(f"派发 {task['id']} -> {device['host']} GPU{device['index']}", flush=True)


def controller(campaign):
    manifest = read_json(campaign / "campaign.json")
    cfg = manifest["config"]
    with (campaign / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = read_json(campaign / "queue.json")
        position = sum(len(t["attempts"]) for t in tasks)
        while True:
            started = time.monotonic()
            reconcile_tasks(tasks, worker_alive, validate_completion)
            ready = mark_dependencies(tasks)
            error, launched, devices = None, [], []
            stopped = (campaign / "STOP_DISPATCH").exists()
            try:
                for task in list(ready):
                    if task["kind"] in ("performance", "profile"):
                        bundle = bundle_for(task, cfg)
                        if bundle is None:
                            ready.remove(task)
                            task["dependency"] = "waiting:real_cohort"
                        else:
                            task["bundle"] = str(bundle)
                if ready and not stopped:
                    candidates = scan(cfg, campaign)
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
                        available = [
                            t
                            for t in ready
                            if sum(
                                a.get("rejected", False) and a["gpu_uuid"] == device["gpu_uuid"]
                                for a in t["attempts"]
                            )
                            < cfg["max_startup_rejections_per_gpu"]
                        ]
                        if not available:
                            continue
                        task = choose_task(available, position, cfg)
                        dispatch(task, device, campaign, manifest, tasks)
                        position += 1
                        ready.remove(task)
                        reserved.add(device["gpu_uuid"])
                        launched.append(task["id"])
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exception:
                error = repr(exception)
                print(error, flush=True)
            done = all(t["status"] in TERMINAL for t in tasks)
            status = {
                "pid": os.getpid(),
                "time_unix_s": time.time(),
                "counts": dict(Counter(t["status"] for t in tasks)),
                "by_kind": {
                    k: dict(Counter(t["status"] for t in tasks if t["kind"] == k))
                    for k in REGISTRIES
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
            time.sleep(max(1, cfg["scan_interval_s"] - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--campaign")
    args = parser.parse_args()
    if args.run:
        controller(safe_directory(args.campaign))
        return
    cfg = yaml.safe_load((ROOT / "configs/workloads/local_search_queue.yaml").read_text())
    tasks = tasks_for(cfg)
    campaign = resolve(cfg["registry_id"])
    if not args.execute:
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "by_kind": dict(Counter(t["kind"] for t in tasks)),
                    "holdout_evaluations": 450,
                    "tuning_evaluations": 120,
                    "tests_opened": False,
                }
            )
        )
        return
    if args.resume:
        manifest = read_json(campaign / "campaign.json")
        if manifest["config_hash"] != config_hash(cfg) or (campaign / "STOP_DISPATCH").exists():
            raise ValueError("配置发生变化或存在停止请求，拒绝恢复")
    else:
        runtime = snapshot_head()
        campaign.mkdir(parents=True, exist_ok=False)
        manifest = {
            **runtime,
            **identity(cfg["registry_id"]),
            "config": cfg,
            "config_hash": config_hash(cfg),
            "contract_sha256": file_hash(
                Path(runtime["snapshot"]) / "configs/local_search_contract.yaml"
            ),
            "standard_test_opened": False,
        }
        write_json(campaign / "campaign.json", manifest)
        write_json(campaign / "queue.json", tasks)
    with (campaign / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(json.dumps(start_controller(campaign, manifest, "local_search_pool.py")))


if __name__ == "__main__":
    main()
