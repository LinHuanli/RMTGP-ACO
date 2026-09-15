"""报告专用的冻结工作负载、真实短跑和匹配回放。

结果缓存不参与候选评估；baseline 必须在独立准备任务中生成。
大型原始记录与编译缓存由后台调度器分目录保存。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
import yaml
from deap import gp

from .baseline import backend_semantic_id
from .config import CudaPrecision, ExecutionBackend, GPUMode
from .data import TSPInstance, coordinate_hash, make_problem_batch, validate_reference_tour
from .genetic import RMTGPIndividual, compile_individual, evolve_generation, initialise_population
from .program import ERCValue, create_primitive_sets
from .runtime import configure_runtime
from .sampling import EvaluationCase, IndexedShard
from .spec import load_run_spec
from .training import BaselineCache, EvaluationPool

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "slides/presentation_benchmarks"
SCHEMA = 1
BACKENDS = {
    "cpu1": ("numba_batch", 1, False, 8),
    "cpu8": ("numba_batch", 8, False, 8),
    "v1": ("cuda_fused_fp32", 1, False, 1),
    "v2": ("cuda_tiled_v2", 1, True, 8),
    "v2-interp4": ("cuda_tiled_v2", 1, False, 4),
    "v2-gen4": ("cuda_tiled_v2", 1, True, 4),
}


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )
    temporary.replace(path)


def pack_individual(individual: RMTGPIndividual) -> dict:
    """保存前缀树节点，避免 from_string 丢失 strongly typed ERC。"""
    trees = []
    for tree in individual:
        nodes = []
        for node in tree:
            if isinstance(node, gp.Primitive):
                nodes.append({"kind": "primitive", "name": node.name})
            elif isinstance(node.value, ERCValue):
                nodes.append({"kind": "erc", "value": node.value.value})
            else:
                nodes.append({"kind": "terminal", "name": node.name})
        trees.append(nodes)
    return {
        "trees": trees,
        "baseline_passthrough": bool(individual.metadata.get("baseline_passthrough", False)),
        "structural_hash": individual.structural_hash,
    }


def unpack_individual(row: dict, psets) -> RMTGPIndividual:
    trees = []
    for nodes, pset in zip(row["trees"], psets, strict=True):
        values = []
        for node in nodes:
            if node["kind"] == "erc":
                values.append(gp.Terminal(ERCValue(float(node["value"])), False, pset.ret))
            else:
                values.append(deepcopy(pset.mapping[node["name"]]))
        trees.append(gp.PrimitiveTree(values))
    individual = RMTGPIndividual(*trees)
    if row["baseline_passthrough"]:
        individual.metadata["baseline_passthrough"] = True
    if individual.structural_hash != row["structural_hash"]:
        raise ValueError("trace 中的树哈希与重建结果不一致")
    return individual


def primitive_sets(config):
    return create_primitive_sets(
        transition_profile=config.transition_profile,
        function_profile=config.function_profile,
        transition_terminals=config.transition_terminals,
        pheromone_terminals=config.pheromone_terminals,
    )


def save_case(directory: Path, name: str, case: EvaluationCase) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npz"
    arrays = {
        key: getattr(case.batch, key).cpu().numpy()
        for key in ("coords", "reference_tour", "reference_length")
    }
    np.savez_compressed(path, **arrays)
    descriptor = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "instance_ids": list(case.batch.instance_ids),
        "coordinate_hashes": list(case.batch.coordinate_hashes),
        "scale": case.scale,
        "seed": int(case.seed),
        "candidate_size": int(case.batch.nn_indices.shape[-1]),
    }
    descriptor["case_id"] = canonical_hash(descriptor)
    return descriptor


def load_case(directory: Path, descriptor: dict) -> EvaluationCase:
    path = directory / descriptor["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"实例文件哈希不一致: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        instances = []
        for index, instance_id in enumerate(descriptor["instance_ids"]):
            coords = arrays["coords"][index].copy()
            tour = arrays["reference_tour"][index].copy()
            digest = coordinate_hash(coords)
            if digest != descriptor["coordinate_hashes"][index]:
                raise ValueError("坐标哈希不一致")
            validate_reference_tour(tour, coords.shape[0])
            instances.append(
                TSPInstance(
                    instance_id, coords, tour, float(arrays["reference_length"][index]), digest
                )
            )
    batch = make_problem_batch(instances, candidate_size=descriptor["candidate_size"])
    return EvaluationCase(descriptor["scale"], batch, descriptor["seed"])


class TraceWriter:
    """按 evaluation 调用顺序记录输入，序列化耗时在正式计时之外。"""

    def __init__(self, directory: Path, aco_config=None):
        self.directory = Path(directory)
        self.calls: list[dict] = []
        self.aco = aco_config.stable_dict() if aco_config is not None else None

    def capture(self, generation: int, population, cases) -> None:
        call_id = len(self.calls)
        descriptors = [
            save_case(self.directory, f"call-{call_id}-case-{i}", case)
            for i, case in enumerate(cases)
        ]
        call = {
            "generation": generation,
            "evaluation_call_id": call_id,
            "aco": self.aco,
            "programs": [pack_individual(p) for p in population],
            "cases": descriptors,
        }
        call["workload_id"] = canonical_hash(call)
        self.calls.append(call)
        atomic_json(
            self.directory / "trace.json",
            {
                "schema_version": SCHEMA,
                "calls": self.calls,
                "trace_hash": canonical_hash(self.calls),
            },
        )


def load_trace(path: Path, config):
    payload = read_json(path / "trace.json")
    if payload["trace_hash"] != canonical_hash(payload["calls"]):
        raise ValueError("trace 总哈希不一致")
    psets = primitive_sets(config)
    calls = []
    for call in payload["calls"]:
        core = {k: v for k, v in call.items() if k != "workload_id"}
        if canonical_hash(core) != call["workload_id"]:
            raise ValueError("trace workload 哈希不一致")
        calls.append(
            (
                call,
                [unpack_individual(p, psets) for p in call["programs"]],
                [load_case(path, c) for c in call["cases"]],
            )
        )
    return calls


def experiment(output: Path, label: str, *, iterations: int = 500):
    base = load_run_spec(output / "config.yaml").experiment
    backend, threads, generated, lanes = BACKENDS[label]
    runtime = replace(
        base.runtime,
        aco_backend=ExecutionBackend(backend),
        cpu_threads=threads,
        processes=1,
        gpu_devices=(0,),
        gpu_mode=GPUMode.SINGLE,
        cuda_precision=CudaPrecision.FP32,
        cuda_generated_gp=generated,
        cuda_candidate_lanes=lanes,
        cuda_tuning_manifest=None,
        gpu_block_threads=32 if label == "v1" else 0,
    )
    return replace(base, runtime=runtime, aco=replace(base.aco, iterations=iterations))


def native_solve(exp, case, programs):
    if exp.runtime.aco_backend is ExecutionBackend.NUMBA_BATCH:
        from .aco_numba import solve_population_numba

        return solve_population_numba(
            case.batch, exp.aco, programs, seed=case.seed, threads=exp.runtime.cpu_threads
        )
    from .aco_cuda import solve_population_cuda

    return solve_population_cuda(case.batch, exp.aco, programs, seed=case.seed, runtime=exp.runtime)


def warm(exp, case):
    """仅编译通用基础设施；不预编译待测 population。"""
    tiny = EvaluationCase(case.scale, case.batch.take([0]), case.seed)
    return native_solve(replace(exp, aco=replace(exp.aco, iterations=1)), tiny, [(None, None)])


def environment(exp) -> dict:
    result = {
        "schema_version": SCHEMA,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "versions": {},
        "backend": exp.runtime.aco_backend.value,
        "cpu_threads": exp.runtime.cpu_threads,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cache_paths": {
            k: os.environ.get(k) for k in ("CUPY_CACHE_DIR", "CUDA_CACHE_PATH", "NUMBA_CACHE_DIR")
        },
    }
    for package in ("deap", "numpy", "numba", "llvmlite", "torch", "cupy-cuda13x"):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][package] = None
    for name, command in (
        ("cpu", ["lscpu", "--json"]),
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("memory", ["free", "-b"]),
        (
            "gpu_inventory",
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,power.limit",
                "--format=csv,noheader",
            ],
        ),
    ):
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        result[name] = proc.stdout.strip() if proc.returncode == 0 else None
    if not exp.runtime.aco_backend.value.startswith("numba"):
        import cupy as cp

        d = cp.cuda.runtime.getDeviceProperties(0)
        result["gpu"] = {
            k: (v.decode() if isinstance(v, bytes) else v)
            for k, v in d.items()
            if k
            in (
                "name",
                "major",
                "minor",
                "multiProcessorCount",
                "l2CacheSize",
                "totalGlobalMem",
                "regsPerMultiprocessor",
                "sharedMemPerMultiprocessor",
            )
        }
        result["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
    return result


class CompilationAudit:
    """仅在报告基准中挂接 NVRTC 与源码生成计数，不修改数值计算。"""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.patches = []
        self.reset()

    def reset(self):
        self.compile_count = 0
        self.codegen_s = 0.0
        self.sources = []
        self.resource_attributes = []
        self.module_loads = 0
        self.module_hits = 0

    def __enter__(self):
        if not self.enabled:
            return self
        import cupy.cuda.compiler as compiler

        from . import aco_cuda

        original_compile = compiler.compile_using_nvrtc
        original_source = aco_cuda._generated_gp_source
        original_load = aco_cuda._load_v2_kernels
        original_v1 = aco_cuda._load_kernel

        def count_compile(*args, **kwargs):
            self.compile_count += 1
            return original_compile(*args, **kwargs)

        def source(*args, **kwargs):
            started = perf_counter()
            value = original_source(*args, **kwargs)
            self.codegen_s += perf_counter() - started
            self.sources.append(hashlib.sha256(value.encode()).hexdigest())
            return value

        def module(*args, **kwargs):
            kernels, seconds = original_load(*args, **kwargs)
            self.module_loads += int(seconds > 0)
            self.module_hits += int(seconds == 0)
            for kernel in kernels[:3]:
                self.resource_attributes.append(dict(kernel.attributes))
            return kernels, seconds

        def module_v1(*args, **kwargs):
            kernel, seconds = original_v1(*args, **kwargs)
            self.module_loads += int(seconds > 0)
            self.module_hits += int(seconds == 0)
            self.resource_attributes.append(dict(kernel.attributes))
            return kernel, seconds

        for obj, name, replacement in (
            (compiler, "compile_using_nvrtc", count_compile),
            (aco_cuda, "_generated_gp_source", source),
            (aco_cuda, "_load_v2_kernels", module),
            (aco_cuda, "_load_kernel", module_v1),
        ):
            manager = patch.object(obj, name, replacement)
            manager.start()
            self.patches.append(manager)
        return self

    def __exit__(self, *args):
        for manager in reversed(self.patches):
            manager.stop()


def prepare(output: Path, *, generations: int = 5):
    if not 1 <= generations <= 10:
        raise ValueError("generations 必须位于 1--10")
    if (output / "workload_manifest.json").exists():
        return read_json(output / "workload_manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/acs_tsp100_gpu0.yaml").read_text())
    config["gp"]["generations"] = generations
    config["runtime"].update(
        aco_backend="cuda_tiled_v2",
        cpu_threads=8,
        cuda_precision="fp32",
        cuda_generated_gp=True,
        cuda_candidate_lanes=8,
        cuda_task_order="instance_major",
        gpu_block_threads=0,
        cuda_tuning_manifest=None,
    )
    config["aco"]["local_search"] = "none"
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    descriptors = []
    data_directory = output / "inputs"
    rng = np.random.default_rng(20260915)
    for scale in (50, 100, 500):
        pattern = f"tsp_{scale}/tsp{scale}_uniform_*.txt"
        files = sorted((ROOT / "Datasets/TSP/train_dataset/tsp").glob(pattern))
        if not files:
            raise FileNotFoundError(pattern)
        shard = IndexedShard.open(files[0])
        count = 32 * generations if scale == 100 else 32
        indices = rng.choice(len(shard), count, replace=False)
        instances = [shard.get(int(i)) for i in indices]
        for start in range(0, count, 32):
            generation = start // 32 + 1
            case = EvaluationCase(
                scale, make_problem_batch(instances[start : start + 32]), 2026091500 + generation
            )
            descriptor = save_case(data_directory, f"tsp{scale}-g{generation}", case)
            descriptor["generation"] = generation
            descriptors.append(descriptor)
    manifest = {
        "schema_version": SCHEMA,
        "gp_seed": 2001,
        "generations": generations,
        "cases": descriptors,
        "config_sha256": hashlib.sha256((output / "config.yaml").read_bytes()).hexdigest(),
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    atomic_json(output / "workload_manifest.json", manifest)
    return manifest


def training_cases(output: Path):
    manifest = read_json(output / "workload_manifest.json")
    return [(d, load_case(output / "inputs", d)) for d in manifest["cases"] if d["scale"] == 100]


def baseline_case_key(case):
    return canonical_hash(
        {
            "seed": case.seed,
            "ids": list(case.batch.instance_ids),
            "coordinates": list(case.batch.coordinate_hashes),
        }
    )


def prepare_baseline(output: Path, label: str, destination: Path):
    exp = experiment(output, label)
    configure_runtime(exp.runtime)
    cases = [
        (d, load_case(output / "inputs", d))
        for d in read_json(output / "workload_manifest.json")["cases"]
    ]
    started = perf_counter()
    warm(exp, cases[0][1])
    arrays = {}
    for _descriptor, case in cases:
        arrays[baseline_case_key(case)] = (
            native_solve(exp, case, [(None, None)]).best_length[0].numpy()
        )
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "lengths.npz", **arrays)
    atomic_json(
        destination / "manifest.json",
        {
            "backend_semantic": backend_semantic_id(exp.runtime.aco_backend, exp.runtime),
            "baseline_behavior_hash": exp.aco.baseline_behavior_hash,
            "workload_hash": read_json(output / "workload_manifest.json")["manifest_hash"],
            "case_ids": list(arrays),
            "prepare_wall_s": perf_counter() - started,
            "environment": environment(exp),
        },
    )


def baseline_cache(output: Path, exp, label: str, cases):
    directory = output / "baselines" / ("cpu8" if label == "cpu1" else label)
    manifest = read_json(directory / "manifest.json")
    if manifest["backend_semantic"] != backend_semantic_id(exp.runtime.aco_backend, exp.runtime):
        raise ValueError("baseline 数值语义不匹配")
    if manifest["baseline_behavior_hash"] != exp.aco.baseline_behavior_hash:
        raise ValueError("baseline 配置不匹配")
    cache = BaselineCache()
    with np.load(directory / "lengths.npz", allow_pickle=False) as arrays:
        for _descriptor, case in cases:
            cache.put(case, exp, torch.from_numpy(arrays[baseline_case_key(case)].copy()))
    return cache


def evaluation_row(exp, label, population, cases, result, elapsed, audit, generation):
    metrics = result.benchmark_metrics
    calls = metrics.get("backend_calls", [])
    gpu = not label.startswith("cpu")
    def summed(key):
        return sum(c["metrics"].get(key, 0.0) for c in calls)
    per_program = sum(c.batch.batch_size * exp.aco.ants * exp.aco.iterations for c in cases)
    executed = result.constructed_tours // per_program
    nodes = [p.total_nodes for p in population]
    row = {
        "backend": label,
        "generation": generation,
        "cpu_threads": exp.runtime.cpu_threads,
        "precision": "fp32" if gpu else "fp64",
        "fast_math": False,
        "generated_gp": exp.runtime.cuda_generated_gp if gpu else False,
        "candidate_lanes": exp.runtime.cuda_candidate_lanes if gpu else None,
        "population_requested": len(population),
        "programs_unique": result.evaluated_unique,
        "programs_executed": executed,
        "instances": sum(c.batch.batch_size for c in cases),
        "cities": cases[0].scale,
        "ants": exp.aco.ants,
        "aco_iterations": exp.aco.iterations,
        "candidate_size": exp.aco.candidate_size,
        "nodes_mean": float(np.mean(nodes)),
        "nodes_max": max(nodes),
        "depth_mean": float(np.mean([max(t.height for t in p) for p in population])),
        "tasks_executed": result.constructed_tours // (exp.aco.ants * exp.aco.iterations),
        "tours_executed": result.constructed_tours,
        "evaluation_wall_s": result.evaluation_wall_time + metrics.get("structural_dedup_s", 0),
        "evaluation_request_wall_s": elapsed,
        "baseline_lookup_s": result.baseline_wall_time,
        "fitness_assign_s": metrics.get("fitness_assign_s", 0),
        "encode_codegen_s": metrics.get("encode_s", 0) + audit.codegen_s,
        "structural_dedup_s": metrics.get("structural_dedup_s", 0),
        "compile_load_s": summed("compile_seconds_sum") if gpu else 0.0,
        "modules_compiled": audit.compile_count if gpu else 0,
        "module_loads": audit.module_loads if gpu else 0,
        "compile_cache_hits": audit.module_hits if gpu else 0,
        "generated_source_hash": audit.sources,
        "gpu_span_s": summed("kernel_seconds_critical") if gpu else None,
        "kernel_sum_s": None,
        "cpu_compute_s": sum(c["native_reported_wall_s"] for c in calls) if not gpu else None,
        "h2d_s": summed("h2d_seconds_sum") if gpu else None,
        "d2h_s": summed("d2h_seconds_sum") if gpu else None,
        "fp64_score_s": summed("exact_fp64_scoring_seconds") if gpu else None,
        "chunks": int(summed("chunks")) if gpu else len(calls),
        "peak_device_memory_bytes": None,
        "kernel_resources": audit.resource_attributes,
        "backend_calls": calls,
        "cache_state": "trace_first_use",
        "status": "completed",
    }
    row["output_signature"] = canonical_hash([float(p.fitness.values[0]) for p in population])
    return row


def evaluate(exp, label, population, cases, pool, cache, audit, generation):
    # 每次显式失效，warm 回放也执行仿真；只允许同一次请求内去重。
    for individual in population:
        if individual.fitness.valid:
            del individual.fitness.values
    audit.reset()
    started = perf_counter()
    result = pool.evaluate_population(population, cases, cache)
    return evaluation_row(
        exp, label, population, cases, result, perf_counter() - started, audit, generation
    )


def train(output: Path, label: str, destination: Path, *, capture: bool = False):
    started = float(os.environ.get("PRESENTATION_PROCESS_START", perf_counter()))
    exp = experiment(output, label)
    configure_runtime(exp.runtime)
    cases = training_cases(output)
    if len(cases) != exp.gp.generations:
        raise ValueError("冻结训练 schedule 与 GP generations 不一致")
    cache = baseline_cache(output, exp, label, cases)
    random.seed(exp.root_seed)
    np.random.seed(exp.root_seed)
    torch.manual_seed(exp.root_seed)
    population, tr, ph = initialise_population(exp.gp)
    warm(exp, cases[0][1])
    env = environment(exp)
    startup = perf_counter() - started
    writer = TraceWriter(output / "trace", exp.aco) if capture else None
    records = []
    seen = set()
    with EvaluationPool(exp) as pool, CompilationAudit(not label.startswith("cpu")) as audit:
        pool.benchmark_metrics_enabled = True
        for generation, (_, case) in enumerate(cases, start=1):
            # trace 导出独立于所有正式性能重复。
            if writer:
                writer.capture(generation, population, [case])
            generation_started = perf_counter()
            row = evaluate(exp, label, population, [case], pool, cache, audit, generation)
            hashes = {p.structural_hash for p in population}
            row["new_programs"] = len(hashes - seen)
            seen.update(hashes)
            row["startup_wall_s"] = startup
            row["fitness_min_gap_percent"] = min(p.fitness.values[0] for p in population)
            breeding = perf_counter()
            if generation < len(cases):
                population = evolve_generation(population, tr, ph, exp.gp)
            row["evolve_s"] = perf_counter() - breeding
            row["input_prepare_s"] = 0.0  # 全部冻结输入已在 startup 载入。
            log_start = perf_counter()
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "backend",
                            "generation",
                            "evaluation_wall_s",
                            "programs_executed",
                            "modules_compiled",
                        )
                    }
                ),
                flush=True,
            )
            row["logging_s"] = perf_counter() - log_start
            row["generation_wall_s"] = perf_counter() - generation_started
            row["other_wall_s"] = max(
                0,
                row["generation_wall_s"]
                - row["evaluation_request_wall_s"]
                - row["evolve_s"]
                - row["logging_s"],
            )
            records.append(row)
            atomic_json(
                destination / "progress.json", {"generation": generation, "records": records}
            )
    atomic_json(
        destination / "result.json",
        {"kind": "trace" if capture else "train", "environment": env, "records": records},
    )
    write_csv(destination / "generations.csv", records)


def replay(
    output: Path,
    label: str,
    destination: Path,
    *,
    trace_directory: Path | None = None,
    generations: list[int] | None = None,
    profile: bool = False,
):
    started = float(os.environ.get("PRESENTATION_PROCESS_START", perf_counter()))
    exp = experiment(output, label)
    configure_runtime(exp.runtime)
    calls = load_trace(trace_directory or output / "trace", exp.gp)
    if generations:
        calls = [item for item in calls if item[0]["generation"] in generations]
    for call, _, _ in calls:
        if call.get("aco") is not None and call["aco"] != exp.aco.stable_dict():
            raise ValueError("回放 ACO 参数与冻结 trace 不一致")
    cache = baseline_cache(
        output, exp, label, [(None, case) for _, _, cases in calls for case in cases]
    )
    warm(exp, calls[0][2][0])
    env = environment(exp)
    startup = perf_counter() - started
    rows = []
    with EvaluationPool(exp) as pool, CompilationAudit(not label.startswith("cpu")) as audit:
        pool.benchmark_metrics_enabled = True
        if profile:
            import cupy as cp

            cp.cuda.runtime.profilerStart()
        try:
            for call, population, cases in calls:
                row = evaluate(
                    exp, label, population, cases, pool, cache, audit, call["generation"]
                )
                row.update(
                    workload_id=call["workload_id"],
                    evaluation_call_id=call["evaluation_call_id"],
                    startup_wall_s=startup,
                )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "backend",
                                "generation",
                                "evaluation_wall_s",
                                "tasks_executed",
                            )
                        }
                    ),
                    flush=True,
                )
                atomic_json(destination / "progress.json", {"records": rows})
        finally:
            if profile:
                cp.cuda.runtime.profilerStop()
    atomic_json(
        destination / "result.json", {"kind": "replay", "environment": env, "records": rows}
    )
    write_csv(destination / "evaluations.csv", rows)


def sanity(output: Path, label: str, destination: Path):
    exp = experiment(output, label, iterations=50)
    configure_runtime(exp.runtime)
    random.seed(2001)
    population, _, _ = initialise_population(exp.gp)
    population = [p for p in population if p.total_nodes > 0][:8]
    case = training_cases(output)[0][1]
    case = EvaluationCase(100, case.batch.take([0, 1, 2, 3]), case.seed)
    programs = [compile_individual(p) for p in population]
    result = native_solve(exp, case, programs)
    tours = result.best_tour.numpy()
    lengths = result.best_length.numpy()
    for i in range(len(programs)):
        for j in range(4):
            validate_reference_tour(tours[i, j], 100)
            tour = tours[i, j]
            exact = case.batch.distances[j, tour[:-1], tour[1:]].sum().item()
            if not np.isclose(exact, lengths[i, j], rtol=1e-12, atol=1e-12):
                raise AssertionError("FP64 tour 计分不一致")
    if not np.isfinite(lengths).all():
        raise AssertionError("长度非有限")
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "outputs.npz", tours=tours, lengths=lengths)
    atomic_json(
        destination / "result.json",
        {
            "kind": "sanity",
            "backend": label,
            "environment": environment(exp),
            "programs_requested": 8,
            "tours_executed": result.constructed_tours,
            "status": "completed",
            "output_signature": hashlib.sha256(tours.tobytes()).hexdigest(),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "baseline", "train", "trace", "replay", "sanity", "scans", "jit"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=BACKENDS, default="v2")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--generations", type=int, nargs="+")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    destination = args.destination or output / "manual" / args.action / args.backend
    if args.action == "prepare":
        prepare(output, generations=args.generations[0] if args.generations else 5)
    elif args.action == "baseline":
        prepare_baseline(output, args.backend, destination)
    elif args.action in ("train", "trace"):
        train(output, args.backend, destination, capture=args.action == "trace")
    elif args.action == "sanity":
        sanity(output, args.backend, destination)
    elif args.action == "replay":
        replay(
            output,
            args.backend,
            destination,
            trace_directory=args.trace,
            generations=args.generations,
            profile=args.profile,
        )
    elif args.action == "scans":
        from .presentation_jit import prepare_scans

        prepare_scans(output)
    else:
        from .presentation_jit import run_jit

        run_jit(output, destination)


if __name__ == "__main__":
    main()
