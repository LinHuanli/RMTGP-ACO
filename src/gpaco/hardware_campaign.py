"""跨 GPU 先导流水线：独立调优/留出测量/短训练/规范执行器质量审计。"""

import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import ExecutionPlan, InfeasiblePlan, SearchConfig, config_hash
from .data import ROOT, validate_tours, write_json
from .experiment import evaluate, gap, metadata, namespace_seed, train
from .hardware_inputs import (
    DEFAULT_PLAN,
    FrozenStore,
    TrainingInputs,
    file_hash,
    prepare_scale,
    safe_directory,
    scenario_key,
)
from .language import ProgramSpec
from .telemetry import Monitor


def clean_record(row):
    return (
        row.get("status") == "completed"
        and not row.get("contended")
        and not row.get("telemetry_errors")
    )


def choose_plan(rows, default, threshold=0.03):
    """只按调优集的暖 evaluation wall 选执行计划；不足 3% 则保留默认值。"""
    if not clean_record(default):
        raise RuntimeError("默认配置缺少无争用有效测量，不能计算调优收益")
    valid = [r for r in rows if clean_record(r)]
    if not valid:
        raise RuntimeError("没有有效调优配置")
    best = min(valid, key=lambda r: r["eval_wall_s"])
    reduction = 1 - best["eval_wall_s"] / default["eval_wall_s"]
    selected = best if reduction >= threshold else default
    return {
        "plan": selected["plan"],
        "selection_metric": "warm_eval_wall_s",
        "selected_cell": selected["cell"],
        "best_cell": best["cell"],
        "best_time_reduction_fraction": reduction,
        "threshold": threshold,
        "default_retained": selected["plan"] == default["plan"],
        "tuning_only": True,
        "holdout_used_for_selection": False,
        "scientific_budget_changed": False,
    }


def paired_order(model, n, block):
    rng = np.random.default_rng(namespace_seed(9002, f"order-{model}-{n}", block))
    return list(rng.permutation(["default", "selected"]))


def hardware_info():
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    fields = (
        "multiProcessorCount",
        "major",
        "minor",
        "totalGlobalMem",
        "l2CacheSize",
        "maxThreadsPerBlock",
        "maxThreadsPerMultiProcessor",
        "regsPerBlock",
        "regsPerMultiprocessor",
        "sharedMemPerBlock",
        "sharedMemPerMultiprocessor",
    )
    query = (
        "uuid,name,pci.bus_id,driver_version,memory.total,ecc.mode.current,power.limit,"
        "clocks.max.sm,clocks.max.memory,compute_mode"
    )
    return {
        **metadata(),
        "name": props["name"].decode(),
        "device_properties": {key: int(props[key]) for key in fields if key in props},
        "nvml_inventory": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                os.environ["CUDA_VISIBLE_DEVICES"],
                f"--query-gpu={query}",
                "--format=csv",
            ],
            text=True,
        ),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "gpu_performance_counters": "not_collected; no occupancy/bandwidth claims",
    }


def measure_cell(directory, cell, programs, problem, search, plan, monitor, *, stage, block=0):
    """暖缓存测量不含输出校验/保存，编译及一次 I=1 预热另记；真实测量使用 I=500。"""
    import cupy as cp

    target = directory / cell
    target.mkdir(parents=True, exist_ok=False)
    request = {
        "cell": cell,
        "stage": stage,
        "block": block,
        "plan": asdict(plan),
        "search": asdict(search),
        "p": len(programs),
        "b": problem.size,
        "n": problem.n,
        "scenario": problem.initialization.scenario,
        "input_manifest_sha256": problem.initialization.input_manifest_sha256,
        "seed": problem.initialization.seed,
        "program_hashes": [p.semantic_hash for p in programs],
        "requested_tasks": len(programs) * problem.size,
        "fitness_cache_hits": 0,
    }
    write_json(target / "request.json", request)
    try:
        cp.get_default_memory_pool().free_all_blocks()
        started = perf_counter()
        warm = evaluate(
            programs,
            problem,
            replace(search, iterations=1),
            problem.initialization.seed,
            replace(plan, profile_stages=False),
        )
        warmup = {"wall_s": perf_counter() - started, "iterations": 1, **warm.timings}
        del warm
        cp.get_default_memory_pool().free_all_blocks()
        with monitor.measure() as telemetry:
            started = perf_counter()
            result = evaluate(programs, problem, search, problem.initialization.seed, plan)
            wall = perf_counter() - started
        if result.timings["executed_tasks"] != len(programs) * problem.size:
            raise RuntimeError("执行器减少了逻辑任务，性能样本无效")
        validate_tours(result.tours, problem.n)
        record = {
            **request,
            **result.timings,
            **telemetry,
            "status": "completed",
            "warmup": warmup,
            "outer_eval_wall_s": wall,
            "tour_valid": True,
            "mean_gap_percent": float(np.mean(gap(result.lengths, problem.reference[None]))),
            "compiled_before_measurement": True,
            "cold_compile_measured": False,
            "geometry_preparation_included": False,
            "lengths_array_sha256": sha256(result.lengths.tobytes()).hexdigest(),
            "tours_array_sha256": sha256(result.tours.tobytes()).hexdigest(),
        }
        np.savez(
            target / "result.npz",
            lengths=result.lengths,
            tours=result.tours,
            diagnostics=result.diagnostics,
            **(
                {"local_search_diagnostics": result.local_search_diagnostics}
                if result.local_search_diagnostics is not None
                else {}
            ),
        )
        record["result_sha256"] = file_hash(target / "result.npz")
    except (InfeasiblePlan, cp.cuda.memory.OutOfMemoryError) as error:
        record = {
            **request,
            "status": "infeasible",
            "reason": str(error),
            "error_type": type(error).__name__,
        }
    write_json(target / "record.json", record)
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("program_hashes", "warmup")}),
        flush=True,
    )
    cp.get_default_memory_pool().free_all_blocks()
    return record


def tune(directory, store, search, config, sms, monitor):
    cohort = store.cohort("tuning")
    problem = store.problem(scenario_key("tuning"), config["batch"])
    rows = []
    # 默认优先提供共同锚点；各方案预热后各测一次，收益仅用于探索性筛选。
    for lanes in [8, *[v for v in config["candidate_lanes"] if v != 8]]:
        plan = replace(DEFAULT_PLAN, candidate_lanes=lanes)
        rows.append(
            measure_cell(
                directory,
                f"lanes-{lanes}-active-3200",
                cohort,
                problem,
                search,
                plan,
                monitor,
                stage="tuning",
            )
        )
    valid = [r for r in rows if clean_record(r)]
    if not valid:
        raise RuntimeError("没有合法且无争用的 lanes 配置")
    best_lane = min(valid, key=lambda r: r["eval_wall_s"])["plan"]["candidate_lanes"]
    for factor in config["active_sm_multipliers"]:
        active = factor * sms
        plan = replace(DEFAULT_PLAN, candidate_lanes=best_lane, active_tasks=active)
        rows.append(
            measure_cell(
                directory,
                f"lanes-{best_lane}-active-{active}",
                cohort,
                problem,
                search,
                plan,
                monitor,
                stage="tuning",
            )
        )
    chosen = choose_plan(rows, rows[0], config["minimum_fractional_time_reduction"])
    chosen["input_manifest_sha256"] = store.identity
    chosen["chosen_at_unix_s"] = time.time()
    write_json(directory.parent / "selected_plan.json", chosen)
    return ExecutionPlan(**chosen["plan"])


def holdout(directory, store, search, selected, config, model, n, monitor):
    programs = store.cohort("holdout")
    order = [paired_order(model, n, block) for block in range(config["paired_blocks"])]
    write_json(directory / "paired_order.json", order)
    for block, roles in enumerate(order):
        data = store.problem(scenario_key("holdout", index=block), config["batch"])
        for role in roles:
            plan = DEFAULT_PLAN if role == "default" else selected
            # 即使保留默认配置，也分别测量两个角色，避免构造人为的零方差加速比。
            measure_cell(
                directory,
                f"block-{block:02d}-{role}",
                programs,
                data,
                search,
                plan,
                monitor,
                stage="holdout",
                block=block,
            )


def audit_available(campaign, store, config, monitor):
    """只重评已选冠军；既不改冠军，也不读取测试集，不将审计反馈给训练。"""
    n = store.manifest["n"]
    search = SearchConfig(**config["search"])
    for model in config["targets"]:
        for seed in config["seeds"]:
            run = campaign / "devices" / model / f"tsp{n}" / "training" / f"seed-{seed}"
            target = campaign / "canonical_audit" / f"tsp{n}" / model / f"seed-{seed}"
            if not (run / "COMPLETE.json").exists() or (target / "COMPLETE.json").exists():
                continue
            if target.exists():
                raise RuntimeError(f"不覆盖未完成的审计：{target}")
            target.mkdir(parents=True)
            champion = json.loads((run / "champion.json").read_text())
            program = ProgramSpec.parse(champion["expression"])
            records, values, baselines = [], [], []
            for repeat in range(config["validation_repeats"]):
                data = store.problem(scenario_key("validation", seed, repeat))
                records.append(
                    measure_cell(
                        target,
                        f"repeat-{repeat}",
                        [program],
                        data,
                        search,
                        DEFAULT_PLAN,
                        monitor,
                        stage="canonical_champion_audit",
                        block=repeat,
                    )
                )
                if records[-1]["status"] != "completed":
                    raise RuntimeError("规范质量审计不能跳过失败样本")
                values.append(records[-1]["mean_gap_percent"])
                base, _, _ = store.baseline(data, search, data.initialization.seed, DEFAULT_PLAN)
                baselines.append(float(np.mean(gap(base, data.reference))))
            canonical_gap = float(np.mean(np.asarray(values, np.float32), dtype=np.float32))
            canonical_baseline = float(np.mean(np.asarray(baselines, np.float32), dtype=np.float32))
            write_json(
                target / "COMPLETE.json",
                {
                    "model": model,
                    "n": n,
                    "root_seed": seed,
                    "expression": program.expression,
                    "champion_hash": program.semantic_hash,
                    "local_validation_gap_percent": champion["validation_gap_percent"],
                    "canonical_validation_gap_percent": canonical_gap,
                    "canonical_baseline_gap_percent": canonical_baseline,
                    "canonical_delta_pp": canonical_gap - canonical_baseline,
                    "reselected": False,
                    "test_opened": False,
                    "input_manifest_sha256": store.identity,
                    "executor_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
                    "contended": any(r["contended"] for r in records),
                },
            )


def worker(job):
    campaign = safe_directory(job["campaign_directory"])
    config, model = job["config"], job["model"]
    directory = campaign / "devices" / model
    directory.mkdir(parents=True, exist_ok=True)
    uuid = config["targets"][model]["gpu_uuid"]

    def refresh_report():
        # 在评估计时之外串行写报告；汇总进程只读 GPU 结果，不创建 CUDA context。
        with (directory / "report.log").open("a") as handle:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(job["snapshot"]) / "scripts/report_hardware_pilot.py"),
                    "--campaign",
                    config["name"],
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            print("汇总失败，原始测量仍保留，请查看 report.log", flush=True)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("必须只向进程暴露目标 GPU UUID")
    # 与旧 pilot 共用互斥锁；不同用户仍可能争用，因此保留持续采样。
    locks = ROOT / "artifacts/locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{uuid}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(Path(job["snapshot"]) / "scripts"))
        from run_worker import gpu_state

        state, pids = gpu_state(uuid)
        if pids or int(state.split(",")[2]) > 5 or int(state.split(",")[3]) > 1024:
            raise RuntimeError(f"GPU 已被占用，不抢占：{state}, pids={pids}")
        write_json(
            directory / "status.json", {"phase": "E00", "pid": os.getpid(), "host": platform.node()}
        )
        with (directory / "E00.log").open("w") as handle:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(Path(job["snapshot"]) / "tests/test_cuda.py"),
                    "-q",
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=ROOT,
            )
        monitor = Monitor(uuid, directory)
        try:
            hardware = hardware_info()
            if hardware["name"] != config["targets"][model]["model"]:
                raise RuntimeError("实际 GPU 型号不匹配冻结配置")
            write_json(directory / "hardware.json", hardware)
            stores = {}
            for n in config["sizes"]:
                frozen = campaign / "inputs" / f"tsp{n}"
                write_json(
                    directory / "status.json", {"phase": "inputs", "n": n, "pid": os.getpid()}
                )
                if model == "a5000":
                    prepare_scale(frozen, n, config, uuid)
                else:
                    while not (frozen / "READY.json").exists():
                        writer_failure = campaign / "devices/a5000/FAILED.json"
                        if writer_failure.exists():
                            raise RuntimeError("A5000 规范输入准备失败，停止等待")
                        time.sleep(15)
                store = FrozenStore(frozen)
                if store.manifest["config_hash"] != config_hash(config):
                    raise ValueError("跨卡配置不一致")
                stores[n] = store
                target = directory / f"tsp{n}"
                target.mkdir(parents=True, exist_ok=False)
                search = SearchConfig(**config["search"])
                write_json(
                    directory / "status.json", {"phase": "tuning", "n": n, "pid": os.getpid()}
                )
                plan = tune(
                    target / "tuning",
                    store,
                    search,
                    config,
                    hardware["device_properties"]["multiProcessorCount"],
                    monitor,
                )
                write_json(
                    directory / "status.json", {"phase": "holdout", "n": n, "pid": os.getpid()}
                )
                holdout(target / "holdout", store, search, plan, config, model, n, monitor)
                refresh_report()
                for seed in config["seeds"]:
                    write_json(
                        directory / "status.json",
                        {
                            "phase": "training",
                            "n": n,
                            "root_seed": seed,
                            "pid": os.getpid(),
                        },
                    )
                    run = target / "training" / f"seed-{seed}"
                    inputs = TrainingInputs(store, seed)
                    # 预核验全部文件，磁盘 SHA 审计与正式训练计时分离；读取开销仍单独记录。
                    for generation, indices in enumerate(store.manifest["schedules"][str(seed)], 1):
                        problem = inputs.training(generation, indices)
                        inputs.baseline(problem, search, problem.initialization.seed, plan)
                    for repeat in range(config["validation_repeats"]):
                        problem = inputs.validation(repeat)
                        inputs.baseline(problem, search, problem.initialization.seed, plan)
                    with monitor.measure() as telemetry:
                        train(
                            run,
                            n,
                            seed,
                            search,
                            plan,
                            population_size=config["population"],
                            generations=config["generations"],
                            batch_size=config["batch"],
                            validation_interval=config["validation_interval"],
                            validation_repeats=config["validation_repeats"],
                            inputs=inputs,
                        )
                    write_json(run / "telemetry_summary.json", telemetry)
                    if model == "a5000":
                        audit_available(campaign, store, config, monitor)
                    refresh_report()
                write_json(
                    directory / "status.json", {"phase": "diagnostics", "n": n, "pid": os.getpid()}
                )
                measure_cell(
                    target / "diagnostics",
                    "stages",
                    store.cohort("holdout"),
                    store.problem(scenario_key("holdout"), config["batch"]),
                    search,
                    replace(plan, profile_stages=True),
                    monitor,
                    stage="instrumented_not_performance",
                )
                for active in config["capacity_active_tasks"]:
                    write_json(
                        directory / "status.json",
                        {
                            "phase": "capacity",
                            "n": n,
                            "active": active,
                            "pid": os.getpid(),
                        },
                    )
                    measure_cell(
                        target / "capacity",
                        f"active-{active}",
                        store.cohort("holdout"),
                        store.problem(scenario_key("holdout"), config["capacity_batch"]),
                        search,
                        replace(plan, active_tasks=active),
                        monitor,
                        stage="capacity_exploratory_single_block",
                    )
                write_json(target / "COMPLETE.json", {"n": n, "status": "completed"})
                if model == "a5000":
                    for previous in stores.values():
                        audit_available(campaign, previous, config, monitor)
                refresh_report()
            write_json(directory / "COMPUTE_COMPLETE.json", {"status": "completed"})
            if model == "a5000":
                # 其他卡完成即审计。等待不增加 GPU 任务，不干预各卡的训练选择。
                while True:
                    for store in stores.values():
                        audit_available(campaign, store, config, monitor)
                    finished = all(
                        (campaign / "devices" / name / "COMPUTE_COMPLETE.json").exists()
                        or (campaign / "devices" / name / "FAILED.json").exists()
                        for name in config["targets"]
                    )
                    if finished:
                        break
                    write_json(
                        directory / "status.json",
                        {"phase": "waiting_for_audit", "pid": os.getpid()},
                    )
                    time.sleep(30)
            write_json(directory / "COMPLETE.json", {"status": "completed"})
            write_json(directory / "status.json", {"phase": "completed", "pid": os.getpid()})
            refresh_report()
        finally:
            monitor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job_path = safe_directory(args.job)
    job = json.loads(job_path.read_text())
    directory = safe_directory(job["campaign_directory"]) / "devices" / job["model"]
    try:
        worker(job)
    except Exception as error:
        write_json(
            directory / "FAILED.json",
            {
                "error": str(error),
                "traceback": traceback.format_exc(),
                "time": time.time(),
                "pid": os.getpid(),
                "host": platform.node(),
            },
        )
        raise


if __name__ == "__main__":
    main()
