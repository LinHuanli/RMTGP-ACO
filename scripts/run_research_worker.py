"""正式GPU基线及同卡映射配对worker；不打开测试集，不减少科学预算。"""

import argparse
import fcntl
import json
import os
import random
import time
import traceback
from pathlib import Path

from run_main_benchmark import DeviceUnavailable, qualify
from run_worker import gpu_state

from gpaco.artifact_registry import identity, require_output, resolve
from gpaco.benchmark_inputs import BenchmarkInputs
from gpaco.config import ExecutionPlan, SearchConfig
from gpaco.data import write_json
from gpaco.experiment import namespace_seed, train
from gpaco.formal_inputs import prepare
from gpaco.hardware_campaign import clean_record, hardware_info, measure_cell
from gpaco.hardware_inputs import TrainingInputs, file_hash, safe_directory
from gpaco.telemetry import Monitor


def formal_run(job, directory, monitor):
    cfg, task = job["config"]["formal"], job["task"]
    inputs_path = require_output(job["inputs"], cfg["input_registry_id"])
    write_json(directory / "status.json", {"phase": "prepare_inputs", "pid": os.getpid()})
    with monitor.measure() as prep_telemetry:
        store = prepare(inputs_path, task["n"], task["seed"], cfg, job["gpu_uuid"])
    write_json(directory / "preparation_telemetry.json", prep_telemetry)
    inputs = TrainingInputs(store, task["seed"])
    write_json(directory / "status.json", {"phase": "training", "pid": os.getpid()})
    with monitor.measure() as telemetry:
        train(
            directory / "training",
            task["n"],
            task["seed"],
            SearchConfig(**cfg["search"]),
            ExecutionPlan(**cfg["plan"]),
            population_size=cfg["population"],
            generations=cfg["generations"],
            batch_size=cfg["batch"],
            validation_interval=cfg["validation_interval"],
            validation_repeats=cfg["validation_repeats"],
            inputs=inputs,
            evidence={**identity(cfg["registry_id"]), "role": "fixed_gpu_existing_control"},
        )
    write_json(directory / "training_telemetry.json", telemetry)
    complete = json.loads((directory / "training/COMPLETE.json").read_text())
    history = json.loads((directory / "training/history.json").read_text())
    if len(history) != cfg["generations"] or not all(
        r["baseline_cache_hit"] and r["executed_tasks"] == cfg["population"] * cfg["batch"]
        for r in history
    ):
        raise ValueError("正式训练预算或预生成ACO缓存条件未满足")
    clean = all(
        not r.get("contended") and not r.get("telemetry_errors")
        for r in (prep_telemetry, telemetry)
    )
    return {
        "status": "completed",
        "clean": clean,
        "task_id": task["id"],
        "formal_timing_eligible": clean and complete["uninterrupted_timing_sample"],
        "training_wall_s": complete["training_wall_s"],
        "input_preparation_wall_s": store.manifest["preparation_wall_s"],
        "training_plus_input_preparation_s": complete["training_wall_s"]
        + store.manifest["preparation_wall_s"],
        "input_manifest_sha256": store.identity,
        "training_complete_sha256": file_hash(directory / "training/COMPLETE.json"),
        "history_sha256": file_hash(directory / "training/history.json"),
        "initial_cohort_matches_frozen": (
            json.loads((directory / "training/cohorts/generation-001.json").read_text())
            == json.loads((inputs_path / "initial_population.json").read_text())
        ),
        "standard_test_opened": False,
    }


def mapping_run(job, directory, monitor):
    task, cfg = job["task"], job["config"]["mapping"]
    inputs = BenchmarkInputs(job["bundle"])
    if inputs.identity != job["bundle_sha256"]:
        raise ValueError("映射输入身份不一致")
    programs, problem = inputs.load(task["generation"], task["block"])
    budget = cfg["scientific_budget"]
    if (
        len(programs),
        problem.size,
        inputs.search.ants,
        inputs.search.iterations,
        inputs.search.candidate_size,
    ) != (
        budget["population"],
        budget["batch"],
        budget["ants"],
        budget["iterations"],
        budget["candidate_size"],
    ) or inputs.search.variant != "as":
        raise ValueError("映射补充不能改变预算或ACO宿主")
    plans = [(lanes, active) for lanes in cfg["candidate_lanes"] for active in cfg["active_tasks"]]
    random.Random(
        namespace_seed(
            cfg["order_seed"], f"mapping-{task['n']}-{task['generation']}", task["block"]
        )
    ).shuffle(plans)
    write_json(directory / "plan_order.json", plans)
    records = []
    for index, (lanes, active) in enumerate(plans):
        cell = f"lanes{lanes:02d}-active{active:05d}"
        write_json(
            directory / "status.json",
            {
                "phase": "mapping",
                "cell": cell,
                "completed_cells": index,
                "total_cells": len(plans),
                "pid": os.getpid(),
            },
        )
        record = measure_cell(
            directory / "measurements",
            cell,
            programs,
            problem,
            inputs.search,
            ExecutionPlan(candidate_lanes=lanes, active_tasks=active, generated=True),
            monitor,
            stage="fixed_mapping_pilot",
            block=task["block"],
        )
        records.append({"cell": cell, "status": record["status"], "clean": clean_record(record)})
    return {
        "status": "completed",
        "clean": all(r["clean"] for r in records if r["status"] == "completed"),
        "task_id": task["id"],
        "cells": records,
        "formal_result": False,
        "record_hashes": {
            r["cell"]: file_hash(directory / "measurements" / r["cell"] / "record.json")
            for r in records
        },
        "completed_cells": sum(r["status"] == "completed" for r in records),
        "infeasible_cells": sum(r["status"] == "infeasible" for r in records),
        "tests_opened": False,
    }


def execute(job, directory):
    uuid = job["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("仅允许看到已分配的UUID")
    lock_path = resolve("shared-device-locks") / f"{uuid}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DeviceUnavailable("项目内另一worker已持有该GPU") from error
        state, pids = gpu_state(uuid)
        fields = [x.strip() for x in state.split(",")]
        if (
            pids
            or int(fields[2]) > 5
            or int(fields[3]) > 1024
            or fields[1] != job["config"]["gpu_model"]
        ):
            raise DeviceUnavailable(f"启动复查不满足空闲A5000条件：{state}; {pids}")
        qualification_start = time.perf_counter()
        qualify(Path(job["campaign_directory"]), job, directory)
        qualification_wall = time.perf_counter() - qualification_start
        write_json(directory / "hardware.json", hardware_info())
        monitor = Monitor(uuid, directory)
        try:
            result = (
                formal_run(job, directory, monitor)
                if job["task"]["kind"] == "formal_train"
                else mapping_run(job, directory, monitor)
            )
        finally:
            monitor.close()
        if result.get("initial_cohort_matches_frozen") is False:
            raise ValueError("正式训练初始种群与冻结输入不一致")
        return {
            **result,
            "qualification_wall_s": qualification_wall,
            "worker_commit": job["commit"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    path = safe_directory(args.job)
    job = json.loads(path.read_text())
    try:
        result = execute(job, path.parent)
    except DeviceUnavailable as error:
        write_json(
            path.parent / "REJECTED.json", {"reason": str(error), "scientific_work_started": False}
        )
    except Exception:
        write_json(path.parent / "FAILED.json", {"traceback": traceback.format_exc()})
        raise
    else:
        write_json(path.parent / "COMPLETE.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
