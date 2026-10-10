"""两项显式争用修复。保留a01，a02使用原科学快照、原输入和原配对顺序。"""

import argparse
import copy
import fcntl
import json
import os
import time

from a5000_pool import confirm_idle, scan
from campaign_runtime import snapshot_head, start_controller
from queue_state import read_json, reconcile_tasks
from research_pool import dispatch, validate_completion, worker_alive

from gpaco.artifact_registry import identity, resolve
from gpaco.data import write_json
from gpaco.hardware_inputs import file_hash, safe_directory

TARGETS = {"as-tsp500-g001-b002", "as-tsp100-g050-b002"}


def controller(campaign):
    manifest = read_json(campaign / "campaign.json")
    tasks = read_json(campaign / "queue.json")
    with (campaign / "dispatcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            started = time.monotonic()
            reconcile_tasks(tasks, worker_alive, validate_completion)
            done = all(t["status"] in ("completed", "failed", "excluded_contended") for t in tasks)
            error = None
            if not done and not (campaign / "STOP_DISPATCH").exists():
                try:
                    for candidate in scan(manifest["config"], campaign):
                        pending = [
                            t
                            for t in tasks
                            if t["status"] == "pending"
                            and sum(a.get("rejected", False) for a in t["attempts"]) < 6
                        ]
                        if not pending:
                            break
                        device = confirm_idle(candidate)
                        if (
                            device
                            and not (
                                campaign / "qualifications" / device["gpu_uuid"] / "FAILED.json"
                            ).exists()
                        ):
                            dispatch(pending[0], device, campaign, manifest, tasks)
                except Exception as exception:
                    error = repr(exception)
                    print(error, flush=True)
            write_json(campaign / "queue.json", tasks)
            state = {
                "pid": os.getpid(),
                "time_unix_s": time.time(),
                "tasks": {t["id"]: t["status"] for t in tasks},
                "error": error,
                "phase": "finished" if done else "scanning",
            }
            write_json(campaign / "dispatcher_status.json", state)
            if done or (campaign / "STOP_DISPATCH").exists():
                if done:
                    write_json(campaign / "FINISHED.json", state)
                return
            time.sleep(max(1, 60 - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--campaign")
    args = parser.parse_args()
    if args.run:
        controller(safe_directory(args.campaign))
        return
    source = resolve("shared-research-dispatch-p01")
    tasks = [copy.deepcopy(t) for t in read_json(source / "queue.json") if t["id"] in TARGETS]
    if len(tasks) != 2 or any(
        t["status"] != "excluded_contended" or len(t["attempts"]) != 1 for t in tasks
    ):
        raise ValueError("只允许两个预登记a01争用block，不自动扩充修复清单")
    if not args.execute:
        print(json.dumps({"repair_blocks": list(TARGETS), "additional_independent_replicates": 0}))
        return
    runtime = snapshot_head()
    campaign = resolve("shared-mapping-repair-p01")
    campaign.mkdir(parents=True, exist_ok=False)
    original = read_json(source / "campaign.json")
    manifest = {
        **original,
        **identity("shared-mapping-repair-p01"),
        "controller_runtime": runtime,
        "source_campaign_sha256": file_hash(source / "campaign.json"),
        "reason": "两个明确争用block各一次显式修复；不把a01与a02都当独立重复",
    }
    for task in tasks:
        task["status"] = "pending"
        task["repair_of"] = task.pop("result")
    write_json(campaign / "campaign.json", manifest)
    write_json(campaign / "queue.json", tasks)
    print(json.dumps(start_controller(campaign, runtime, "mapping_repair_pool.py")))


if __name__ == "__main__":
    main()
