"""A5000 主基线队列任务：冻结输入或执行一个完整配对 block，不改变科学预算。"""

import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from run_worker import gpu_state

from gpaco.backends.cpu import initial_parameters
from gpaco.config import ExecutionPlan, SearchConfig, config_hash
from gpaco.data import ROOT, load_split, prepare_problem, write_json
from gpaco.experiment import namespace_seed, source_hash
from gpaco.hardware_campaign import clean_record, hardware_info, measure_cell
from gpaco.hardware_inputs import FrozenStore, file_hash, safe_directory, save_problem
from gpaco.language import ProgramSpec
from gpaco.telemetry import Monitor


class DeviceUnavailable(RuntimeError):
    """尚未执行任何科学计算时的抢占/锁竞争，可在新空闲设备重试。"""


def prepare_inputs(campaign, n, config, hardware, uuid):
    """每规模只有一个明确指定的 A5000 写入者；三种宿主共享逐位相同几何。"""
    if hardware["name"] != config["gpu_model"] or hardware["gpu_visible"] != uuid:
        raise ValueError("输入写入者硬件/UUID不符合预先分配")
    target = campaign / "inputs" / f"tsp{n}"
    target.mkdir(parents=True, exist_ok=False)
    begin = perf_counter()
    problem = prepare_problem(
        *load_split(n, "holdout", np.arange(config["batch"])), config["search"]["candidate_size"]
    )
    if problem.n != n or problem.size != config["batch"]:
        raise ValueError("输入规模与科学工作负载不一致")
    input_hashes = {}
    for variant in config["variants"]:
        directory = target / variant
        search = SearchConfig(variant=variant, **config["search"])
        save_problem(directory / "geometry/holdout", problem)
        scenarios = {}
        for block in range(config["paired_blocks"]):
            # 随机输入不依赖宿主、硬件、cohort阶段或解释/JIT执行模式。
            seed = namespace_seed(9101, "main-e01-holdout", block)
            key = f"block-{block:02d}"
            scenario = directory / "scenarios" / key
            scenario.mkdir(parents=True)
            initial = initial_parameters(
                problem.distances,
                problem.instance_keys,
                np.uint64(seed),
                search.variant_id,
                np.float32(search.rho),
            )
            for field, values in zip(("tau0", "low", "high"), initial, strict=True):
                np.save(scenario / f"{field}.npy", values, allow_pickle=False)
            scenarios[key] = {"geometry": "holdout", "seed": seed, "baseline": False}
        manifest = {
            "n": n,
            "source_hash": source_hash(),
            "config_hash": config_hash(config),
            "search": asdict(search),
            "writer_uuid": uuid,
            "writer": hardware,
            "scenarios": scenarios,
            "geometry": {
                "holdout": {
                    "path": "geometry/holdout",
                    "instances": problem.instance_ids,
                    "source_split": "holdout",
                }
            },
            "source_manifest_sha256": file_hash(
                ROOT / f"Datasets/processed/v1/tsp{n}/holdout/manifest.json"
            ),
            "files": {
                str(p.relative_to(directory)): file_hash(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            },
            "tests_opened": False,
            "reference_cache_required": False,
        }
        write_json(directory / "manifest.json", manifest)
        input_hashes[variant] = file_hash(directory / "manifest.json")
        write_json(directory / "READY.json", {"manifest_sha256": input_hashes[variant]})
    # 只有全部宿主都冻结成功，调度器才释放依赖本规模输入的测量任务。
    write_json(
        target / "READY.json",
        {
            "n": n,
            "writer_uuid": uuid,
            "input_hashes": input_hashes,
            "preparation_wall_s": perf_counter() - begin,
        },
    )


def qualify(campaign, job, directory):
    """每个源码快照×物理卡执行一次基础 E00，避免每个 block 重复编译 CPU 测试。"""
    target = campaign / "qualifications" / job["gpu_uuid"]
    completed = target / "COMPLETE.json"
    if completed.exists():
        saved = json.loads(completed.read_text())
        if saved["commit"] != job["commit"] or saved["source_hash"] != source_hash():
            raise ValueError("GPU 资格检查不属于当前不可变版本")
        return
    target.mkdir(parents=True, exist_ok=True)
    test = Path(job["snapshot"]) / "tests/test_cuda.py"
    try:
        with (directory / "E00.log").open("w") as handle:
            subprocess.run(
                [sys.executable, "-m", "pytest", str(test), "-q"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=ROOT,
            )
    except subprocess.CalledProcessError as error:
        write_json(
            target / "FAILED.json",
            {
                "commit": job["commit"],
                "error": str(error),
                "log": str(directory / "E00.log"),
                "scientific_work_started": False,
            },
        )
        raise DeviceUnavailable("该卡基础检查失败并已隔离；科学任务尚未执行") from error
    write_json(
        completed,
        {
            "commit": job["commit"],
            "source_hash": source_hash(),
            "log": str(directory / "E00.log"),
            "host": platform.node(),
            "gpu_uuid": job["gpu_uuid"],
            "time": time.time(),
        },
    )


def run(job, directory):
    campaign = safe_directory(job["campaign_directory"])
    config, task = job["config"], job["task"]
    uuid = job["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("每个 worker 只能看到分配给它的 GPU UUID")
    locks = ROOT / "artifacts/locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{uuid}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DeviceUnavailable("项目内另一工作进程仍持有该GPU锁") from error
        state, pids = gpu_state(uuid)
        if pids or int(state.split(",")[2]) > 5 or int(state.split(",")[3]) > 1024:
            raise DeviceUnavailable(f"启动前设备已不空闲：{state}, pids={pids}")
        if state.split(",")[1].strip() != config["gpu_model"]:
            raise DeviceUnavailable("目标不再是 RTX A5000，拒绝替换硬件")
        qualify(campaign, job, directory)
        info = hardware_info()
        write_json(directory / "hardware.json", info)
        monitor = Monitor(uuid, directory)
        try:
            if task["kind"] == "prepare":
                prepare_inputs(campaign, task["n"], config, info, uuid)
                return {"status": "completed", "kind": "prepare", "n": task["n"]}
            store = FrozenStore(campaign / "inputs" / f"tsp{task['n']}" / task["variant"])
            if store.manifest["config_hash"] != config_hash(config):
                raise ValueError("共享输入不属于当前任务协议")
            cohort_path = safe_directory(task["cohort_path"])
            cohort_rows = json.loads(cohort_path.read_text())
            if len(cohort_rows) != config["population"]:
                raise ValueError("cohort 程序数量不符，禁止截取或复制后冒充原始种群")
            programs = [ProgramSpec.parse(row["expression"]) for row in cohort_rows]
            write_json(directory / "cohort.json", [p.record() for p in programs])
            data = store.problem(f"block-{task['block']:02d}")
            search = SearchConfig(variant=task["variant"], **config["search"])
            if task["kind"] == "pair":
                rng = np.random.default_rng(
                    namespace_seed(
                        9102,
                        f"main-order-{task['n']}-{task['variant']}-{task['generation']}",
                        task["block"],
                    )
                )
                order = list(rng.permutation(config["paired_modes"]))
            else:
                order = [config["profile_mode"]]
            write_json(
                directory / "manifest.json",
                {
                    "task": task,
                    "source_cohort_hash": file_hash(cohort_path),
                    "input_manifest_sha256": store.identity,
                    "order": order,
                    "commit": job["commit"],
                    "source_hash": source_hash(),
                    "cohort_training_variant": "as",
                    "execution_variant": task["variant"],
                    "gpu_uuid": uuid,
                    "host": platform.node(),
                    "tests_opened": False,
                    "fixed_plan": config["plan"],
                    "pair_same_physical_gpu": True,
                },
            )
            records = []
            for mode in order:
                plan = ExecutionPlan(
                    **config["plan"],
                    generated=(mode == "generated"),
                    profile_stages=(task["kind"] == "profile"),
                )
                write_json(
                    directory / "status.json",
                    {
                        "mode": mode,
                        "task": task["id"],
                        "phase": "evaluation",
                        "pid": os.getpid(),
                    },
                )
                record = measure_cell(
                    directory / "measurements",
                    mode,
                    programs,
                    data,
                    search,
                    plan,
                    monitor,
                    stage="main_e01_pair" if task["kind"] == "pair" else "main_e01_instrumented",
                    block=task["block"],
                )
                records.append(record)
                if record["status"] != "completed":
                    raise RuntimeError("预设主基线计划不可行；保留记录，不降低预算自动重试")
            return {
                "status": "completed",
                "kind": task["kind"],
                "task": task["id"],
                "clean": all(clean_record(r) for r in records),
                "source_cohort_hash": file_hash(cohort_path),
                "record_paths": [f"measurements/{mode}/record.json" for mode in order],
            }
        finally:
            monitor.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    path = safe_directory(args.job)
    job = json.loads(path.read_text())
    directory = path.parent
    write_json(
        directory / "STARTED.json",
        {
            "pid": os.getpid(),
            "host": platform.node(),
            "time": time.time(),
            "job": str(path),
        },
    )
    try:
        result = run(job, directory)
        write_json(directory / "COMPLETE.json", {**result, "finished_unix_s": time.time()})
    except DeviceUnavailable as error:
        write_json(
            directory / "REJECTED.json", {"error": str(error), "scientific_work_started": False}
        )
    except Exception as error:
        write_json(
            directory / "FAILED.json", {"error": str(error), "traceback": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    main()
