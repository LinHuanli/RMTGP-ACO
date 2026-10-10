"""p02有限任务worker：先资格检查，再调优/短训练/先导；标准测试始终封存。"""

import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from run_main_benchmark import DeviceUnavailable, qualify
from run_worker import gpu_state

from gpaco.artifact_registry import identity, require_output, resolve
from gpaco.backends.cpu import initial_parameters
from gpaco.benchmark_inputs import BenchmarkInputs
from gpaco.config import ExecutionPlan, SearchConfig
from gpaco.data import ROOT, FrozenInitialization, load_split, prepare_problem, write_json
from gpaco.evolution import initial_population
from gpaco.experiment import namespace_seed, source_hash, train
from gpaco.formal_inputs import prepare
from gpaco.hardware_campaign import clean_record, hardware_info, measure_cell
from gpaco.hardware_inputs import TrainingInputs, file_hash, safe_directory, save_problem
from gpaco.language import ProgramSpec
from gpaco.telemetry import Monitor


def ls_contract():
    root = Path(os.environ.get("GPACO_SNAPSHOT", ROOT))
    return file_hash(root / "configs/local_search_contract.yaml")


def qualify_ls(job, directory):
    """源码版本和设备UUID双重绑定。失败隔离设备，不自动重跑科学任务。"""
    campaign = Path(job["campaign_directory"])
    target = campaign / "qualifications" / job["gpu_uuid"]
    qualify(campaign, job, directory)
    complete = target / "LS_COMPLETE.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["source_hash"] != source_hash() or previous["contract_sha256"] != ls_contract():
            raise ValueError("局部搜索资格检查版本不一致")
        return
    try:
        with (directory / "E00-LS.log").open("w") as handle:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(Path(job["snapshot"]) / "tests/test_local_search.py"),
                ],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        write_json(
            target / "FAILED.json",
            {
                "reason": str(error),
                "log": str(directory / "E00-LS.log"),
                "scientific_work_started": False,
            },
        )
        raise DeviceUnavailable("局部搜索资格检查失败，该GPU已隔离") from error
    write_json(
        complete,
        {
            "commit": job["commit"],
            "source_hash": source_hash(),
            "contract_sha256": ls_contract(),
            "log": str(directory / "E00-LS.log"),
        },
    )


def configs(job, mode=None):
    task, cfg = job["task"], job["config"]["training"]
    search = SearchConfig(
        variant=task["variant"], local_search=mode or task["mode"], **cfg["search"]
    )
    plan = ExecutionPlan(**cfg["plan"])
    return search, plan


def record_set(directory):
    return {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(directory.glob("measurements/*/record.json"))
    }


def identical_pair(records):
    if len(records) != 2 or not all(r["status"] == "completed" for r in records):
        return False
    for field in (
        "program_hashes",
        "input_manifest_sha256",
        "seed",
        "search",
        "requested_tasks",
        "lengths_array_sha256",
        "tours_array_sha256",
    ):
        if records[0][field] != records[1][field]:
            raise ValueError(f"LS执行器或插桩改变了固定语义：{field}")
    return True


def tuning(job, directory, monitor):
    task, cfg = job["task"], job["config"]
    # tuning程序与性能cohort来源根种子分离，不读取性能holdout选执行器。
    previous_random_state = random.getstate()
    cohort_seed = namespace_seed(7403, "ls-tuning-programs", task["n"])
    random.seed(cohort_seed)
    programs = [ProgramSpec.from_tree(t) for t in initial_population(cfg["training"]["population"])]
    random.setstate(previous_random_state)
    search, base_plan = configs(job)
    problem = prepare_problem(
        *load_split(task["n"], "tuning", np.arange(cfg["training"]["batch"])), search.candidate_size
    )
    save_problem(directory / "tuning-input/geometry", problem)
    scenarios = {}
    for block in cfg["blocks"]:
        seed = namespace_seed(7401, "ls-tuning", block)
        initial = initial_parameters(
            problem.distances,
            problem.instance_keys,
            np.uint64(seed),
            search.variant_id,
            np.float32(search.rho),
        )
        name = f"initial-{block}.npz"
        np.savez(
            directory / f"tuning-input/{name}", tau0=initial[0], low=initial[1], high=initial[2]
        )
        scenarios[str(block)] = {"seed": seed, "initial": name}
    input_manifest = {
        "split": "tuning",
        "instances": problem.instance_ids,
        "cohort_root_seed": cohort_seed,
        "programs": [p.record() for p in programs],
        "search": asdict(search),
        "scenarios": scenarios,
        "files": {
            str(p.relative_to(directory / "tuning-input")): file_hash(p)
            for p in sorted((directory / "tuning-input").rglob("*"))
            if p.is_file()
        },
        "source_split_sha256": file_hash(
            ROOT / f"Datasets/processed/v1/tsp{task['n']}/tuning/manifest.json"
        ),
    }
    write_json(directory / "tuning-input/manifest.json", input_manifest)
    input_sha = file_hash(directory / "tuning-input/manifest.json")
    times = {"scalar": [], "cooperative": []}
    clean = True
    for block in cfg["blocks"]:
        seed = namespace_seed(7401, "ls-tuning", block)
        with np.load(directory / f"tuning-input/initial-{block}.npz", allow_pickle=False) as saved:
            initial = tuple(saved[name].copy() for name in ("tau0", "low", "high"))
        data = replace(
            problem,
            initialization=FrozenInitialization(
                seed,
                search.variant,
                search.rho,
                problem.instance_ids,
                initial,
                f"tuning-{block}",
                input_sha,
            ),
        )
        order = ["scalar", "cooperative"]
        random.Random(namespace_seed(cfg["order_seed"], task["id"], block)).shuffle(order)
        write_json(directory / f"order-b{block}.json", order)
        records = []
        for executor in order:
            write_json(
                directory / "status.json",
                {"phase": "tuning", "block": block, "executor": executor, "pid": os.getpid()},
            )
            record = measure_cell(
                directory / "measurements",
                f"b{block:02d}-{executor}",
                programs,
                data,
                search,
                replace(base_plan, ls_executor=executor),
                monitor,
                stage="ls_tuning",
                block=block,
            )
            records.append(record)
        if not identical_pair(records):
            raise ValueError("调优执行器不可行，保留原始记录，不缩小预算或放行训练")
        if all(clean_record(r) for r in records):
            for r in records:
                times[r["plan"]["ls_executor"]].append(r["eval_wall_s"])
        else:
            clean = False
    medians = {key: float(np.median(values)) if values else None for key, values in times.items()}
    eligible = clean and all(len(values) == len(cfg["blocks"]) for values in times.values())
    selected = None
    if eligible:
        selected = (
            "scalar"
            if medians["scalar"] <= medians["cooperative"] * (1 - cfg["selection_minimum_gain"])
            else "cooperative"
        )
    return {
        "clean": eligible,
        "selected_executor": selected,
        "median_eval_wall_s": medians,
        "paired_blocks": len(times["scalar"]),
        "record_hashes": record_set(directory),
        "tuning_input_sha256": input_sha,
        "tuning_only": True,
    }


def training(job, directory, monitor):
    task, cfg = job["task"], dict(job["config"]["training"])
    search, plan = configs(job)
    selection = json.loads(Path(job["selection"]).read_text())
    if not selection["clean"] or selection["selected_executor"] not in ("scalar", "cooperative"):
        raise ValueError("未通过执行器筛选，不能启动训练")
    plan = replace(plan, ls_executor=selection["selected_executor"])
    cfg.update(search=asdict(search), plan=asdict(plan))
    if task["kind"] == "smoke":
        cfg["generations"] = job["config"]["smoke"]["generations"]
    input_dir = require_output(job["inputs"], "shared-ls-training-inputs-p02")
    write_json(directory / "status.json", {"phase": "prepare_inputs", "pid": os.getpid()})
    with monitor.measure() as prep_telemetry:
        store = prepare(input_dir, task["n"], task["seed"], cfg, job["gpu_uuid"])
    write_json(directory / "preparation_telemetry.json", prep_telemetry)
    write_json(directory / "status.json", {"phase": "training", "pid": os.getpid()})
    with monitor.measure() as telemetry:
        train(
            directory / "training",
            task["n"],
            task["seed"],
            search,
            plan,
            population_size=cfg["population"],
            generations=cfg["generations"],
            batch_size=cfg["batch"],
            validation_interval=cfg["validation_interval"],
            validation_repeats=cfg["validation_repeats"],
            inputs=TrainingInputs(store, task["seed"]),
            evidence={**identity(task["registry_id"]), "role": "joint_ls_training"},
        )
    write_json(directory / "training_telemetry.json", telemetry)
    complete = json.loads((directory / "training/COMPLETE.json").read_text())
    history = json.loads((directory / "training/history.json").read_text())
    if len(history) != cfg["generations"] or not all(
        r["baseline_cache_hit"] and r["executed_tasks"] == cfg["population"] * cfg["batch"]
        for r in history
    ):
        raise ValueError("局部搜索训练预算或基准缓存未满足")
    initial_match = json.loads(
        (directory / "training/cohorts/generation-001.json").read_text()
    ) == json.loads((input_dir / "initial_population.json").read_text())
    if not initial_match:
        raise ValueError("初始种群与冻结输入不一致")
    clean = all(
        not t.get("contended") and not t.get("telemetry_errors")
        for t in (prep_telemetry, telemetry)
    )
    return {
        "clean": clean,
        "training_wall_s": complete["training_wall_s"],
        "input_preparation_wall_s": store.manifest["preparation_wall_s"],
        "training_plus_input_preparation_s": complete["training_wall_s"]
        + store.manifest["preparation_wall_s"],
        "continuous_timing_eligible": clean and complete["uninterrupted_timing_sample"],
        "initial_cohort_matches_frozen": initial_match,
        "input_manifest_sha256": store.identity,
        "training_complete_sha256": file_hash(directory / "training/COMPLETE.json"),
        "history_sha256": file_hash(directory / "training/history.json"),
        "selected_executor": plan.ls_executor,
        "selection_sha256": file_hash(Path(job["selection"])),
    }


def performance(job, directory, monitor):
    task, cfg = job["task"], job["config"]
    inputs = BenchmarkInputs(job["bundle"])
    programs, problem = inputs.load(task["generation"], task["block"])
    base_plan = ExecutionPlan(**cfg["training"]["plan"])
    if len(programs) != cfg["training"]["population"] or problem.size != cfg["training"]["batch"]:
        raise ValueError("完整种群/实例预算不一致")
    if task["kind"] == "profile":
        selected = json.loads(Path(job["selection"]).read_text())["selected_executor"]
        plans = [(task["mode"], selected, False), (task["mode"], selected, True)]
    else:
        plans = [("none", "cooperative", False)] + [
            (mode, executor, False)
            for mode in cfg["modes"]
            for executor in ("scalar", "cooperative")
        ]
        random.Random(namespace_seed(cfg["order_seed"], task["id"], task["block"])).shuffle(plans)
    write_json(directory / "plan_order.json", plans)
    records = []
    for mode, executor, profile in plans:
        cell = f"{mode}-{executor}" + ("-profile" if profile else "")
        write_json(
            directory / "status.json", {"phase": task["kind"], "cell": cell, "pid": os.getpid()}
        )
        search = replace(inputs.search, local_search=mode, ls_candidate_size=20)
        records.append(
            measure_cell(
                directory / "measurements",
                cell,
                programs,
                problem,
                search,
                replace(base_plan, ls_executor=executor, profile_stages=profile),
                monitor,
                stage="ls_" + task["kind"],
                block=task["block"],
            )
        )
    for mode in [task["mode"]] if task["kind"] == "profile" else cfg["modes"]:
        if not identical_pair([r for r in records if r["search"]["local_search"] == mode]):
            raise ValueError("局部搜索配对执行计划不可行，不缩小预算继续")
    return {
        "clean": all(clean_record(r) for r in records),
        "record_hashes": record_set(directory),
        "scalar_cooperative_or_instrumentation_exact": True,
        "source_bundle_sha256": inputs.identity,
    }


def execute(job, directory):
    uuid = job["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("CUDA设备身份不一致")
    if ls_contract() != job["contract_sha256"]:
        raise ValueError("局部搜索协议内容哈希不一致")
    if "bundle" in job and BenchmarkInputs(job["bundle"]).identity != job["bundle_sha256"]:
        raise ValueError("冻结种群输入哈希不一致")
    for path, expected in job.get("gate_hashes", {}).items():
        if file_hash(Path(path)) != expected:
            raise ValueError("依赖资格证据被改动")
    with (resolve("shared-device-locks") / f"{uuid}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DeviceUnavailable("同卡已有本项目worker") from error
        state, pids = gpu_state(uuid)
        fields = [x.strip() for x in state.split(",")]
        if (
            pids
            or int(fields[2]) > 5
            or int(fields[3]) > 1024
            or fields[1] != job["config"]["gpu_model"]
        ):
            raise DeviceUnavailable(f"启动前目标不再是空闲A5000：{state}; {pids}")
        started = time.perf_counter()
        qualify_ls(job, directory)
        qualification_wall = time.perf_counter() - started
        write_json(directory / "hardware.json", hardware_info())
        monitor = Monitor(uuid, directory)
        try:
            kind = job["task"]["kind"]
            result = (
                tuning
                if kind == "tuning"
                else training
                if kind in ("smoke", "training")
                else performance
            )(job, directory, monitor)
        finally:
            monitor.close()
    return {
        **result,
        "status": "completed",
        "task_id": job["task"]["id"],
        "worker_commit": job["commit"],
        "contract_sha256": ls_contract(),
        "qualification_wall_s": qualification_wall,
        "standard_test_opened": False,
        **identity(job["task"]["registry_id"]),
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
