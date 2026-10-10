"""正式基线的逐根种子冻结输入；先付几何/初始化/ACO成本，再开始GP计时。"""

import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .backends.cpu import initial_parameters
from .config import ExecutionPlan, SearchConfig, config_hash
from .data import (
    ROOT,
    FrozenInitialization,
    load_split,
    prepare_problem,
    validate_tours,
    write_json,
)
from .evolution import initial_population
from .experiment import evaluate, metadata, namespace_seed, source_hash
from .hardware_inputs import (
    DEFAULT_PLAN,
    FrozenStore,
    file_hash,
    safe_directory,
    save_problem,
    scenario_key,
    training_schedule,
)
from .language import ProgramSpec


def prepare(directory, n, seed, config, gpu_uuid):
    """每run唯一写入者；READY提交后只读复用，失败现场不可覆盖。"""
    import cupy as cp

    directory = safe_directory(directory)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("输入写入者UUID不一致")
    if cp.cuda.runtime.getDeviceProperties(0)["name"].decode() != "NVIDIA RTX A5000":
        raise ValueError("正式参考仅用A5000生成")
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    search = SearchConfig(**config["search"])
    canonical = ExecutionPlan(**config["plan"]) if search.local_search != "none" else DEFAULT_PLAN
    schedule = training_schedule(n, seed, config["generations"], config["batch"])
    manifest = {
        "schema": "gpaco-formal-training-inputs-v1",
        "n": n,
        "root_seed": seed,
        "source_hash": source_hash(),
        "config_hash": config_hash(config),
        "search": asdict(search),
        "canonical_plan": asdict(canonical),
        "writer_uuid": gpu_uuid,
        "writer": metadata(),
        "geometry": {},
        "scenarios": {},
        "schedules": {str(seed): schedule.tolist()},
        "tests_opened": False,
    }
    if search.local_search != "none":
        manifest["local_search_contract_sha256"] = file_hash(
            Path(os.environ.get("GPACO_SNAPSHOT", ROOT)) / "configs/local_search_contract.yaml"
        )
    costs = {"geometry_s": 0.0, "initialization_s": 0.0, "baseline_s": 0.0, "geometry_save_s": 0.0}

    def geometry(key, split, indices=None):
        begin = perf_counter()
        problem = prepare_problem(*load_split(n, split, indices), search.candidate_size)
        costs["geometry_s"] += perf_counter() - begin
        begin = perf_counter()
        save_problem(directory / "geometry" / key, problem)
        costs["geometry_save_s"] += perf_counter() - begin
        manifest["geometry"][key] = {
            "path": f"geometry/{key}",
            "instances": problem.instance_ids,
            "source_split": split,
            "source_manifest_hash": file_hash(
                ROOT / f"Datasets/processed/v1/tsp{n}/{split}/manifest.json"
            ),
        }
        return problem

    def scenario(key, geometry_key, problem, aco_seed):
        target = directory / "scenarios" / key
        target.mkdir(parents=True)
        begin = perf_counter()
        initial = initial_parameters(
            problem.distances,
            problem.instance_keys,
            np.uint64(aco_seed),
            search.variant_id,
            np.float32(search.rho),
        )
        costs["initialization_s"] += perf_counter() - begin
        for name, values in zip(("tau0", "low", "high"), initial, strict=True):
            np.save(target / f"{name}.npy", values, allow_pickle=False)
        frozen = FrozenInitialization(
            aco_seed, search.variant, search.rho, problem.instance_ids, initial, key
        )
        begin = perf_counter()
        result = evaluate(
            [ProgramSpec.parse("ZERO")],
            replace(problem, initialization=frozen),
            search,
            aco_seed,
            canonical,
        )
        costs["baseline_s"] += perf_counter() - begin
        validate_tours(result.tours, n)
        np.savez(target / "baseline.npz", lengths=result.lengths[0], tours=result.tours[0])
        manifest["scenarios"][key] = {
            "geometry": geometry_key,
            "seed": aco_seed,
            "baseline": True,
            "baseline_timings": result.timings,
        }
        write_json(
            directory / "preparation_status.json",
            {
                "phase": "preparing",
                "scenario": key,
                "completed_scenarios": len(manifest["scenarios"]),
                "expected": config["generations"] + config["validation_repeats"],
            },
        )
        print(f"参考已冻结 tsp{n}/seed{seed}/{key}", flush=True)

    val = geometry("validation", "validation")
    for repeat in range(config["validation_repeats"]):
        scenario(
            scenario_key("validation", seed, repeat),
            "validation",
            val,
            namespace_seed(seed, "validation", repeat=repeat),
        )
    del val
    for generation, indices in enumerate(schedule, 1):
        key = scenario_key("train", seed, generation)
        problem = geometry(key, "train", indices)
        scenario(key, key, problem, namespace_seed(seed, "train", generation))
    state = random.getstate()
    random.seed(seed)
    population = [
        ProgramSpec.from_tree(tree).record() for tree in initial_population(config["population"])
    ]
    random.setstate(state)
    write_json(directory / "initial_population.json", population)
    np.save(directory / "schedule.npy", schedule, allow_pickle=False)
    cp.get_default_memory_pool().free_all_blocks()
    manifest["files"] = {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "preparation_status.json"
    }
    manifest["preparation_wall_s"] = perf_counter() - started
    manifest["preparation_costs"] = costs
    write_json(directory / "manifest.json", manifest)
    write_json(
        directory / "READY.json", {"manifest_sha256": file_hash(directory / "manifest.json")}
    )
    return FrozenStore(directory)
