"""RMTGP-ACO 可复现训练、评测、数据审计与统计命令行。"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

import numpy as np
import torch

from .artifacts import (
    finalise_run_artifacts,
    initialise_run_artifacts,
    resume_run_artifacts,
)
from .baseline import (
    BaselineArchive,
    precompute_baseline_cases,
    write_baseline_shard,
)
from .config import (
    ExecutionBackend,
    GPUMode,
    PheromoneIntegration,
    TransitionIntegration,
)
from .data import make_problem_batch
from .evaluation import (
    compile_champion,
    evaluate_batches,
    load_champion,
    read_records,
    write_records,
)
from .experiment_plan import (
    PROTOCOL_ID,
    build_protocol_a_v05_pilot_plan,
    write_experiment_plan,
)
from .genetic import evolve_generation, initialise_population
from .manifest import (
    build_manifest,
    load_manifest,
    sampled_split_leakage,
    verify_manifest,
    write_manifest,
)
from .program import (
    CORE_PHEROMONE_TERMINALS,
    CORE_TRANSITION_TERMINALS,
)
from .runtime import configure_runtime
from .sampling import (
    iter_problem_batches,
    pools_from_paths,
)
from .schedule import (
    ScheduledTrainingSampler,
    build_protocol_schedule,
    load_schedule,
    validate_schedule_contract,
    validation_cases_from_schedule,
    write_schedule,
)
from .spec import load_run_spec
from .stats import (
    factorial_contrasts,
    friedman_test,
    hierarchical_bootstrap_delta,
    paired_wilcoxon_holm,
    summarize_quality,
    write_statistical_report,
)
from .training import BaselineCache, EvaluationPool, train


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _external_gpu_processes(
    target_devices: set[int],
    *,
    own_pid: int | None = None,
) -> list[dict[str, str | int]]:
    """只返回目标 GPU 上的外部计算进程。

    ``nvidia-smi --query-compute-apps`` 仅提供 GPU UUID，不能直接按设备
    索引过滤。因此先建立 UUID 到索引的映射，避免其他 GPU 上的无关作业
    使当前 benchmark 被误判为受到争用。
    """

    if not target_devices:
        return []
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        app_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    uuid_to_index: dict[str, int] = {}
    for line in gpu_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) != 2:
            continue
        try:
            uuid_to_index[fields[1]] = int(fields[0])
        except ValueError:
            continue

    current_pid = os.getpid() if own_pid is None else own_pid
    records: list[dict[str, str | int]] = []
    for line in app_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            continue
        device = uuid_to_index.get(fields[0])
        if device is None or device not in target_devices:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        records.append(
            {
                "device": device,
                "pid": pid,
                "process_name": fields[2],
                "used_memory_mib": fields[3],
            }
        )
    return records


def _apply_runtime_overrides(spec, args: argparse.Namespace):
    """为独立 run 覆盖 seed/process 数，不修改冻结的 YAML 模板。"""

    experiment = spec.experiment
    seed = getattr(args, "root_seed", None)
    if seed is not None:
        experiment = replace(experiment, root_seed=seed)
    processes = getattr(args, "processes", None)
    if processes is not None:
        selected_backend = experiment.runtime.aco_backend
        if processes > 1 and selected_backend is ExecutionBackend.NUMBA_BATCH:
            selected_backend = ExecutionBackend.NUMBA
        experiment = replace(
            experiment,
            runtime=replace(
                experiment.runtime,
                processes=processes,
                aco_backend=selected_backend,
            ),
        )
    cpu_threads = getattr(args, "cpu_threads", None)
    if cpu_threads is not None:
        experiment = replace(
            experiment,
            runtime=replace(experiment.runtime, cpu_threads=cpu_threads),
        )
    gpu_devices = getattr(args, "gpu_devices", None)
    gpu_mode = getattr(args, "gpu_mode", None)
    gpu_block_threads = getattr(args, "gpu_block_threads", None)
    gpu_task_chunk_size = getattr(args, "gpu_task_chunk_size", None)
    runtime_updates: dict[str, object] = {}
    if gpu_devices is not None:
        runtime_updates["gpu_devices"] = tuple(gpu_devices)
    if gpu_mode is not None:
        runtime_updates["gpu_mode"] = GPUMode(gpu_mode)
    if gpu_block_threads is not None:
        runtime_updates["gpu_block_threads"] = gpu_block_threads
    if gpu_task_chunk_size is not None:
        runtime_updates["gpu_task_chunk_size"] = gpu_task_chunk_size
    if runtime_updates:
        experiment = replace(
            experiment,
            runtime=replace(experiment.runtime, **runtime_updates),
        )
    backend = getattr(args, "backend", None)
    if backend is not None:
        experiment = replace(
            experiment,
            runtime=replace(
                experiment.runtime,
                aco_backend=ExecutionBackend(backend),
            ),
        )
    method = getattr(args, "method_profile", None)
    if method and method != "rmtgp":
        gp = experiment.gp
        aco = experiment.aco
        if method == "tr-rgp":
            gp = replace(
                gp,
                train_transition=True,
                train_pheromone=False,
                transition_profile="main",
                function_profile="f1",
                transition_terminals=None,
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.RESIDUAL,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        elif method == "ph-rgp":
            gp = replace(
                gp,
                train_transition=False,
                train_pheromone=True,
                transition_profile="main",
                function_profile="f1",
                pheromone_terminals=None,
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.RESIDUAL,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        elif method == "matched-replace":
            gp = replace(
                gp,
                train_transition=True,
                train_pheromone=False,
                transition_profile="main",
                function_profile="f1",
                transition_terminals=None,
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.REPLACEMENT,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        elif method == "legacy":
            gp = replace(
                gp,
                train_transition=True,
                train_pheromone=False,
                transition_profile="legacy",
                transition_terminals=None,
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.REPLACEMENT,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        elif method in {
            "rmtgp-core-f0",
            "rmtgp-core-f1",
            "rmtgp-full-f0",
            "rmtgp-full-f1",
        }:
            core = "core" in method
            function_profile = "f0" if method.endswith("f0") else "f1"
            gp = replace(
                gp,
                train_transition=True,
                train_pheromone=True,
                transition_profile="main",
                function_profile=function_profile,
                transition_terminals=(
                    CORE_TRANSITION_TERMINALS if core else None
                ),
                pheromone_terminals=(
                    CORE_PHEROMONE_TERMINALS if core else None
                ),
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.RESIDUAL,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        experiment = replace(
            experiment,
            experiment_id=f"{experiment.experiment_id}-{method}",
            gp=gp,
            aco=aco,
        )
    return replace(spec, experiment=experiment)


def _preflight_manifest(
    spec,
    manifest_path: str | Path,
    required_paths: list[tuple[Path, str]],
) -> None:
    """在昂贵运行前确认完整 manifest 与所需 split 文件一致。"""

    manifest = load_manifest(manifest_path)
    if manifest.hash_mode != "sha256-full":
        raise ValueError("正式运行要求 hash_mode=sha256-full 的数据 manifest")
    errors = verify_manifest(
        manifest,
        root=spec.data.root,
        verify_hashes=False,
        validate_first_record=True,
    )
    if errors:
        raise ValueError("数据 manifest 预检失败：" + "；".join(errors[:5]))
    declared = {record.path: record.split for record in manifest.files}
    missing: list[str] = []
    wrong_split: list[str] = []
    for path, expected_split in required_paths:
        try:
            relative = path.resolve().relative_to(spec.data.root).as_posix()
        except ValueError:
            missing.append(path.as_posix())
            continue
        if relative not in declared:
            missing.append(relative)
        elif declared[relative] != expected_split:
            wrong_split.append(
                f"{relative}: manifest={declared[relative]}, config={expected_split}"
            )
    if missing:
        raise ValueError(f"配置引用了 manifest 外的数据文件: {missing[:5]}")
    if wrong_split:
        raise ValueError(f"配置跨 split 使用数据: {wrong_split[:5]}")


def _command_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(
        args.root,
        full_hashes=args.full,
        workers=args.workers,
    )
    target = write_manifest(manifest, args.output)
    print(f"已写入 {target}：{len(manifest.files)} 个文件，模式={manifest.hash_mode}")
    return 0


def _command_verify_data(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    errors = verify_manifest(
        manifest,
        root=args.root,
        verify_hashes=not args.skip_hashes,
        validate_first_record=True,
    )
    duplicates = sampled_split_leakage(
        manifest,
        records_per_file=args.leakage_samples,
        root=args.root,
    )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    for coordinate_hash, first, second in duplicates:
        print(
            f"ERROR: 跨 split 重复 {coordinate_hash}: {first} <-> {second}",
            file=sys.stderr,
        )
    if errors or duplicates:
        return 1
    print(
        f"数据校验通过：{len(manifest.files)} 个文件；"
        f"每文件抽查 {args.leakage_samples} 条防泄漏记录"
    )
    return 0


def _protocol_schedule(
    spec,
    training_pools,
    validation_pools,
    *,
    path: str | Path | None,
    phase: str,
    replicate_id: int,
):
    """加载冻结 schedule；未配置路径时为开发运行生成内存 schedule。"""

    selected_path = (
        Path(path).resolve()
        if path is not None
        else spec.data.schedule_path
    )
    if phase == "formal" and selected_path is None:
        raise ValueError(
            "formal 训练禁止临时生成 schedule；请先运行 prepare-schedules，"
            "再通过 --schedule 或 data.schedule_manifest 指定冻结文件"
        )
    if selected_path is not None:
        if not selected_path.is_file():
            raise FileNotFoundError(
                f"schedule 不存在，请先运行 prepare-schedules: {selected_path}"
            )
        schedule = load_schedule(selected_path)
    else:
        schedule = build_protocol_schedule(
            training_pools,
            validation_pools,
            protocol_id=PROTOCOL_ID,
            phase=phase,
            root_seed=spec.experiment.root_seed,
            replicate_id=replicate_id,
            generations=spec.experiment.gp.generations,
            train_instances_per_scale=spec.data.train_instances_per_scale,
            validation_selection_instances_per_scale=(
                spec.data.validation_selection_instances_per_scale
            ),
            validation_gate_instances_per_scale=(
                spec.data.validation_gate_instances_per_scale
            ),
            validation_seeds=spec.experiment.validation_seeds,
        )
    validate_schedule_contract(
        schedule,
        protocol_id=PROTOCOL_ID,
        phase=phase,
        root_seed=spec.experiment.root_seed,
        replicate_id=replicate_id,
        generations=spec.experiment.gp.generations,
        train_scales=spec.experiment.train_scales,
        validation_scales=spec.experiment.validation_scales,
        train_instances_per_scale=spec.data.train_instances_per_scale,
        validation_selection_instances_per_scale=(
            spec.data.validation_selection_instances_per_scale
        ),
        validation_gate_instances_per_scale=(
            spec.data.validation_gate_instances_per_scale
        ),
        validation_seeds=spec.experiment.validation_seeds,
    )
    return schedule


def _command_prepare_schedules(args: argparse.Namespace) -> int:
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    if not args.skip_manifest_check:
        _preflight_manifest(
            spec,
            args.manifest,
            [
                *(
                    (path, "train")
                    for paths in spec.data.training_paths().values()
                    for path in paths
                ),
                *(
                    (path, "validation")
                    for paths in spec.data.validation_paths().values()
                    for path in paths
                ),
            ],
        )
    training_pools = pools_from_paths(spec.data.training_paths())
    validation_pools = pools_from_paths(spec.data.validation_paths())
    schedule = build_protocol_schedule(
        training_pools,
        validation_pools,
        protocol_id=args.protocol_id,
        phase=args.phase,
        root_seed=spec.experiment.root_seed,
        replicate_id=args.replicate_id,
        generations=spec.experiment.gp.generations,
        train_instances_per_scale=spec.data.train_instances_per_scale,
        validation_selection_instances_per_scale=(
            spec.data.validation_selection_instances_per_scale
        ),
        validation_gate_instances_per_scale=(
            spec.data.validation_gate_instances_per_scale
        ),
        validation_seeds=spec.experiment.validation_seeds,
    )
    target = write_schedule(schedule, args.output)
    print(
        json.dumps(
            {
                "output": str(target),
                "manifest_hash": schedule.manifest_hash,
                "records": len(schedule.records),
                "phase": schedule.phase,
                "replicate_id": args.replicate_id,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _command_precompute_baselines(args: argparse.Namespace) -> int:
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    configure_runtime(spec.experiment.runtime)
    training_pools = pools_from_paths(spec.data.training_paths())
    validation_pools = pools_from_paths(spec.data.validation_paths())
    schedule = load_schedule(args.schedule)
    validate_schedule_contract(
        schedule,
        protocol_id=PROTOCOL_ID,
        phase=schedule.phase,
        root_seed=spec.experiment.root_seed,
        replicate_id=args.replicate_id,
        generations=spec.experiment.gp.generations,
        train_scales=spec.experiment.train_scales,
        validation_scales=spec.experiment.validation_scales,
        train_instances_per_scale=spec.data.train_instances_per_scale,
        validation_selection_instances_per_scale=(
            spec.data.validation_selection_instances_per_scale
        ),
        validation_gate_instances_per_scale=(
            spec.data.validation_gate_instances_per_scale
        ),
        validation_seeds=spec.experiment.validation_seeds,
    )

    selected_splits = set(args.splits)
    cases = []
    if "train" in selected_splits:
        sampler = ScheduledTrainingSampler(
            schedule,
            training_pools,
            replicate_id=args.replicate_id,
            candidate_size=spec.experiment.aco.candidate_size,
            dtype=spec.experiment.aco.dtype,
            device=spec.experiment.aco.device,
        )
        for generation in range(1, spec.experiment.gp.generations + 1):
            cases.extend(sampler.cases_for_generation(generation))
    for role in ("selection", "gate"):
        if role in selected_splits:
            cases.extend(
                validation_cases_from_schedule(
                    schedule,
                    validation_pools,
                    role=role,
                    replicate_id=args.replicate_id,
                    batch_size=spec.data.evaluation_batch_size,
                    candidate_size=spec.experiment.aco.candidate_size,
                    dtype=spec.experiment.aco.dtype,
                    device=spec.experiment.aco.device,
                )
            )
    records = precompute_baseline_cases(
        cases,
        spec.experiment.aco,
        spec.experiment.runtime.aco_backend,
        threads=spec.experiment.runtime.cpu_threads,
        runtime=spec.experiment.runtime,
    )
    target = write_baseline_shard(
        records,
        args.output,
        metadata={
            "protocol_id": schedule.protocol_id,
            "phase": schedule.phase,
            "schedule_hash": schedule.manifest_hash,
            "replicate_id": args.replicate_id,
            "variant": spec.experiment.aco.variant.value,
            "aco_config_hash": spec.experiment.aco.config_hash,
            "splits": sorted(selected_splits),
        },
    )
    print(
        f"baseline 预计算完成：{len(records)} instance×seed records -> {target}"
    )
    return 0


def _command_benchmark_backends(args: argparse.Namespace) -> int:
    """比较旧 8-process 标量 Numba 与 16-thread population batching。"""

    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    training_pools = pools_from_paths(spec.data.training_paths())
    validation_pools = pools_from_paths(spec.data.validation_paths())
    schedule = _protocol_schedule(
        spec,
        training_pools,
        validation_pools,
        path=args.schedule,
        phase=args.phase,
        replicate_id=args.replicate_id,
    )
    sampler = ScheduledTrainingSampler(
        schedule,
        training_pools,
        replicate_id=args.replicate_id,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
    )
    cases = sampler.cases_for_generation(1)

    random.seed(spec.experiment.root_seed)
    torch.manual_seed(spec.experiment.root_seed)
    population, _, _ = initialise_population(spec.experiment.gp)
    population = population[: args.max_individuals]

    reference_experiment = replace(
        spec.experiment,
        runtime=replace(
            spec.experiment.runtime,
            aco_backend=ExecutionBackend.NUMBA,
            processes=args.reference_processes,
            cpu_threads=1,
        ),
    )
    reference_population = [individual.clone() for individual in population]
    with EvaluationPool(reference_experiment) as evaluator:
        evaluator.warm(cases[0])
        reference = evaluator.evaluate_population(
            reference_population,
            cases,
            BaselineCache(),
        )

    batch_experiment = replace(
        spec.experiment,
        runtime=replace(
            spec.experiment.runtime,
            aco_backend=ExecutionBackend.NUMBA_BATCH,
            processes=1,
            cpu_threads=args.cpu_threads or spec.experiment.runtime.cpu_threads,
        ),
    )
    configure_runtime(batch_experiment.runtime)
    batch_population = [individual.clone() for individual in population]
    with EvaluationPool(batch_experiment) as evaluator:
        evaluator.warm(cases[0])
        batched = evaluator.evaluate_population(
            batch_population,
            cases,
            BaselineCache(),
        )

    reference_by_hash = {
        individual.structural_hash: individual.fitness.values[0]
        for individual in reference_population
    }
    for individual in batch_population:
        expected = reference_by_hash[individual.structural_hash]
        observed = individual.fitness.values[0]
        if abs(expected - observed) > 1e-12:
            raise RuntimeError(
                f"backend fitness 不一致：{expected:.17g} != {observed:.17g}"
            )
    speedup = reference.evaluation_wall_time / max(
        batched.evaluation_wall_time,
        1e-12,
    )
    payload = {
        "variant": spec.experiment.aco.variant.value,
        "individuals": len(population),
        "instances": sum(case.batch.batch_size for case in cases),
        "iterations": spec.experiment.aco.iterations,
        "reference_processes": args.reference_processes,
        "batch_threads": batch_experiment.runtime.cpu_threads,
        "reference_seconds": reference.evaluation_wall_time,
        "batch_seconds": batched.evaluation_wall_time,
        "speedup": speedup,
        "batch_tours_per_second": (
            batched.constructed_tours
            / max(batched.evaluation_wall_time, 1e-12)
        ),
        "meets_1_5x_gate": speedup >= 1.5,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _command_benchmark_accelerators(args: argparse.Namespace) -> int:
    """在同一固定 GP generation 上比较 CPU、单 GPU、双 GPU 与 campaign。"""

    target_gpu_devices: set[int] = set()
    observed_contention: dict[
        tuple[int, int, str],
        dict[str, str | int],
    ] = {}

    def record_external_gpu_processes() -> None:
        for record in _external_gpu_processes(target_gpu_devices):
            key = (
                int(record["device"]),
                int(record["pid"]),
                str(record["process_name"]),
            )
            observed_contention[key] = record

    def start_gpu_monitor():
        stop = threading.Event()
        samples: list[dict[str, float | int]] = []

        def sample_loop() -> None:
            while not stop.is_set():
                record_external_gpu_processes()
                try:
                    completed = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,utilization.gpu,power.draw,"
                            "temperature.gpu,clocks.current.sm,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    for line in completed.stdout.splitlines():
                        fields = [
                            field.strip()
                            for field in line.split(",")
                        ]
                        if len(fields) != 6:
                            continue
                        device = int(fields[0])
                        if device not in target_gpu_devices:
                            continue
                        samples.append(
                            {
                                "device": device,
                                "utilization_percent": float(fields[1]),
                                "power_watts": float(fields[2]),
                                "temperature_c": float(fields[3]),
                                "sm_clock_mhz": float(fields[4]),
                                "memory_used_mib": float(fields[5]),
                            }
                        )
                except (
                    FileNotFoundError,
                    subprocess.SubprocessError,
                    ValueError,
                ):
                    pass
                stop.wait(0.5)

        thread = threading.Thread(target=sample_loop, daemon=True)
        thread.start()
        return stop, thread, samples

    def summarize_gpu_samples(
        samples: list[dict[str, float | int]],
    ) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        devices = sorted({int(sample["device"]) for sample in samples})
        for device in devices:
            selected = [
                sample
                for sample in samples
                if int(sample["device"]) == device
            ]
            entry: dict[str, float | int] = {"samples": len(selected)}
            for field in (
                "utilization_percent",
                "power_watts",
                "temperature_c",
                "sm_clock_mhz",
                "memory_used_mib",
            ):
                values = [float(sample[field]) for sample in selected]
                entry[f"{field}_mean"] = float(np.mean(values))
                entry[f"{field}_max"] = float(np.max(values))
            summary[str(device)] = entry
        return summary
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    experiment = spec.experiment
    gpu_devices = tuple(experiment.runtime.gpu_devices)
    target_gpu_devices.update(gpu_devices)
    if args.iterations is not None:
        experiment = replace(
            experiment,
            aco=replace(experiment.aco, iterations=args.iterations),
        )
    training_pools = pools_from_paths(spec.data.training_paths())
    validation_pools = pools_from_paths(spec.data.validation_paths())
    schedule = _protocol_schedule(
        replace(spec, experiment=experiment),
        training_pools,
        validation_pools,
        path=args.schedule,
        phase=args.phase,
        replicate_id=args.replicate_id,
    )
    sampler = ScheduledTrainingSampler(
        schedule,
        training_pools,
        replicate_id=args.replicate_id,
        candidate_size=experiment.aco.candidate_size,
        dtype=experiment.aco.dtype,
        device=experiment.aco.device,
    )
    cases = sampler.cases_for_generation(args.generation)

    random.seed(experiment.root_seed)
    np.random.seed(experiment.root_seed % (2**32))
    torch.manual_seed(experiment.root_seed)
    population, _, _ = initialise_population(experiment.gp)
    population = population[: args.max_individuals]
    from .genetic import compile_individual

    programs = [compile_individual(individual) for individual in population]

    def run_workload(runtime, *, seed_offset: int = 0):
        outputs = []
        constructed_tours = 0
        backend_metrics: list[dict[str, float | int | str]] = []
        started = perf_counter()
        for case in cases:
            if (
                runtime.aco_backend
                is ExecutionBackend.CUDA_FUSED_FP32
            ):
                from .aco_cuda import solve_population_cuda

                result = solve_population_cuda(
                    case.batch,
                    experiment.aco,
                    programs,
                    seed=case.seed + seed_offset,
                    runtime=runtime,
                )
            else:
                from .aco_numba import solve_population_numba

                result = solve_population_numba(
                    case.batch,
                    experiment.aco,
                    programs,
                    seed=case.seed + seed_offset,
                    threads=runtime.cpu_threads,
                )
            outputs.append(result)
            constructed_tours += result.constructed_tours
            backend_metrics.append(result.backend_metrics)
        elapsed = perf_counter() - started
        return elapsed, constructed_tours, outputs, backend_metrics

    def output_signature(outputs) -> str:
        digest = sha256()
        for result in outputs:
            digest.update(
                np.ascontiguousarray(result.best_tour.numpy()).tobytes()
            )
            digest.update(
                np.ascontiguousarray(result.best_length.numpy()).tobytes()
            )
            digest.update(
                np.ascontiguousarray(result.best_iteration.numpy()).tobytes()
            )
        return digest.hexdigest()

    def summarize_backend_metrics(
        samples: list[dict[str, float | int | str]],
    ) -> dict[str, float]:
        if not samples or "kernel_seconds_critical" not in samples[0]:
            return {}
        return {
            "kernel_seconds": sum(
                float(metric.get("kernel_seconds_critical", 0.0))
                for metric in samples
            ),
            "compile_seconds_sum": sum(
                float(metric.get("compile_seconds_sum", 0.0))
                for metric in samples
            ),
            "h2d_seconds_sum": sum(
                float(metric.get("h2d_seconds_sum", 0.0))
                for metric in samples
            ),
            "d2h_seconds_sum": sum(
                float(metric.get("d2h_seconds_sum", 0.0))
                for metric in samples
            ),
            "exact_fp64_scoring_seconds": sum(
                float(metric.get("exact_fp64_scoring_seconds", 0.0))
                for metric in samples
            ),
        }

    def measure(label: str, runtime) -> tuple[dict[str, object], list]:
        monitor_stop = monitor_thread = None
        monitor_samples: list[dict[str, float | int]] = []
        if runtime.aco_backend is ExecutionBackend.CUDA_FUSED_FP32:
            from .aco_cuda import (
                clear_cuda_kernel_cache,
                clear_cuda_problem_cache,
            )

            clear_cuda_problem_cache()
            clear_cuda_kernel_cache()
            monitor_stop, monitor_thread, monitor_samples = (
                start_gpu_monitor()
            )
        print(f"[benchmark] {label}: cold", file=sys.stderr, flush=True)
        cold, tours, cold_outputs, cold_metrics = run_workload(runtime)
        repeat_seconds: list[float] = []
        repeat_metrics: list[list[dict[str, float | int | str]]] = []
        repeat_outputs = cold_outputs
        metric_samples = cold_metrics
        for repeat in range(args.repeats):
            print(
                f"[benchmark] {label}: warm {repeat + 1}/{args.repeats}",
                file=sys.stderr,
                flush=True,
            )
            elapsed, observed_tours, repeat_outputs, metric_samples = (
                run_workload(runtime)
            )
            if observed_tours != tours:
                raise RuntimeError(f"{label}: constructed tour 数不稳定")
            repeat_seconds.append(elapsed)
            repeat_metrics.append(metric_samples)
        selected = median(repeat_seconds)
        median_index = min(
            range(len(repeat_seconds)),
            key=lambda index: abs(repeat_seconds[index] - selected),
        )
        metric_samples = repeat_metrics[median_index]
        payload: dict[str, object] = {
            "cold_seconds": cold,
            "repeat_seconds": repeat_seconds,
            "median_seconds": selected,
            "constructed_tours": tours,
            "tours_per_second": tours / max(selected, 1e-12),
            "signature": output_signature(repeat_outputs),
        }
        warm_summary = summarize_backend_metrics(metric_samples)
        cold_summary = summarize_backend_metrics(cold_metrics)
        if warm_summary:
            payload.update(warm_summary)
            payload["host_overhead_seconds"] = (
                selected - warm_summary["kernel_seconds"]
            )
        if cold_summary:
            payload.update(
                {
                    f"cold_{name}": value
                    for name, value in cold_summary.items()
                }
            )
            payload["cold_host_overhead_seconds"] = (
                cold - cold_summary["kernel_seconds"]
            )
        if monitor_stop is not None and monitor_thread is not None:
            monitor_stop.set()
            monitor_thread.join(timeout=2)
            payload["gpu_telemetry"] = summarize_gpu_samples(
                monitor_samples
            )
        return payload, repeat_outputs

    modes = set(args.modes)
    measurements: dict[str, dict[str, object]] = {}
    outputs_by_mode: dict[str, list] = {}
    if "cpu8" in modes:
        runtime = replace(
            experiment.runtime,
            aco_backend=ExecutionBackend.NUMBA_BATCH,
            processes=1,
            cpu_threads=8,
        )
        measurements["cpu8"], outputs_by_mode["cpu8"] = measure(
            "cpu8",
            runtime,
        )
    if "cpu16" in modes:
        runtime = replace(
            experiment.runtime,
            aco_backend=ExecutionBackend.NUMBA_BATCH,
            processes=1,
            cpu_threads=16,
        )
        measurements["cpu16"], outputs_by_mode["cpu16"] = measure(
            "cpu16",
            runtime,
        )

    for name, device in (("gpu0", 0), ("gpu1", 1)):
        if name not in modes:
            continue
        if device not in gpu_devices:
            raise ValueError(
                f"{name} 要求设备 {device} 出现在 runtime.gpu_devices"
            )
        runtime = replace(
            experiment.runtime,
            aco_backend=ExecutionBackend.CUDA_FUSED_FP32,
            processes=1,
            gpu_mode=GPUMode.SINGLE,
            gpu_devices=(device,),
        )
        measurements[name], outputs_by_mode[name] = measure(name, runtime)
    if "dual" in modes:
        if len(gpu_devices) < 2:
            raise ValueError("dual benchmark 至少需要两个 gpu_devices")
        runtime = replace(
            experiment.runtime,
            aco_backend=ExecutionBackend.CUDA_FUSED_FP32,
            processes=1,
            gpu_mode=GPUMode.DUAL,
            gpu_devices=gpu_devices[:2],
        )
        measurements["dual"], outputs_by_mode["dual"] = measure(
            "dual",
            runtime,
        )

    if "campaign" in modes:
        if len(gpu_devices) < 2:
            raise ValueError("campaign benchmark 至少需要两个 gpu_devices")
        from .aco_cuda import (
            clear_cuda_kernel_cache,
            clear_cuda_problem_cache,
        )

        clear_cuda_problem_cache()
        clear_cuda_kernel_cache()
        runtimes = [
            replace(
                experiment.runtime,
                aco_backend=ExecutionBackend.CUDA_FUSED_FP32,
                processes=1,
                gpu_mode=GPUMode.SINGLE,
                gpu_devices=(device,),
            )
            for device in gpu_devices[:2]
        ]
        monitor_stop, monitor_thread, monitor_samples = start_gpu_monitor()
        print("[benchmark] campaign: cold", file=sys.stderr, flush=True)
        cold_started = perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            cold_futures = [
                executor.submit(
                    run_workload,
                    runtime,
                    seed_offset=index * 10_000_019,
                )
                for index, runtime in enumerate(runtimes)
            ]
            cold_values = [future.result() for future in cold_futures]
        campaign_cold = perf_counter() - cold_started
        campaign_repeats: list[float] = []
        campaign_repeat_values: list[list[tuple]] = []
        tours_per_run = cold_values[0][1]
        for repeat in range(args.repeats):
            print(
                f"[benchmark] campaign: warm {repeat + 1}/{args.repeats}",
                file=sys.stderr,
                flush=True,
            )
            started = perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        run_workload,
                        runtime,
                        seed_offset=index * 10_000_019,
                    )
                    for index, runtime in enumerate(runtimes)
                ]
                values = [future.result() for future in futures]
            campaign_repeats.append(perf_counter() - started)
            campaign_repeat_values.append(values)
            tours_per_run = values[0][1]
        campaign_median = median(campaign_repeats)
        campaign_median_index = min(
            range(len(campaign_repeats)),
            key=lambda index: abs(
                campaign_repeats[index] - campaign_median
            ),
        )
        selected_campaign_values = campaign_repeat_values[
            campaign_median_index
        ]
        measurements["campaign"] = {
            "cold_seconds": campaign_cold,
            "repeat_seconds": campaign_repeats,
            "median_seconds": campaign_median,
            "constructed_tours": 2 * tours_per_run,
            "tours_per_second": (
                2 * tours_per_run / max(campaign_median, 1e-12)
            ),
            "signatures": [
                output_signature(value[2])
                for value in selected_campaign_values
            ],
        }
        warm_run_summaries = [
            summarize_backend_metrics(value[3])
            for value in selected_campaign_values
        ]
        cold_run_summaries = [
            summarize_backend_metrics(value[3])
            for value in cold_values
        ]
        if all(warm_run_summaries):
            warm_kernel = max(
                summary["kernel_seconds"]
                for summary in warm_run_summaries
            )
            measurements["campaign"].update(
                {
                    "kernel_seconds": warm_kernel,
                    "host_overhead_seconds": (
                        campaign_median - warm_kernel
                    ),
                    **{
                        name: sum(summary[name] for summary in warm_run_summaries)
                        for name in (
                            "compile_seconds_sum",
                            "h2d_seconds_sum",
                            "d2h_seconds_sum",
                            "exact_fp64_scoring_seconds",
                        )
                    },
                }
            )
        if all(cold_run_summaries):
            cold_kernel = max(
                summary["kernel_seconds"]
                for summary in cold_run_summaries
            )
            measurements["campaign"].update(
                {
                    "cold_kernel_seconds": cold_kernel,
                    "cold_host_overhead_seconds": (
                        campaign_cold - cold_kernel
                    ),
                    **{
                        f"cold_{name}": sum(
                            summary[name]
                            for summary in cold_run_summaries
                        )
                        for name in (
                            "compile_seconds_sum",
                            "h2d_seconds_sum",
                            "d2h_seconds_sum",
                            "exact_fp64_scoring_seconds",
                        )
                    },
                }
            )
        monitor_stop.set()
        monitor_thread.join(timeout=2)
        measurements["campaign"]["gpu_telemetry"] = summarize_gpu_samples(
            monitor_samples
        )

    gpu_signatures = {
        name: values["signature"]
        for name, values in measurements.items()
        if name in {"gpu0", "gpu1", "dual"}
    }
    gpu_device_invariant = len(set(gpu_signatures.values())) <= 1
    if not gpu_device_invariant:
        raise RuntimeError("单卡/双卡输出不一致，拒绝生成性能结论")

    cpu_key = "cpu16" if "cpu16" in measurements else "cpu8"
    single_keys = [
        key for key in ("gpu0", "gpu1") if key in measurements
    ]
    derived: dict[str, float | bool] = {
        "gpu_device_invariant": gpu_device_invariant,
    }
    if cpu_key in measurements and single_keys:
        single_seconds = min(
            float(measurements[key]["median_seconds"])
            for key in single_keys
        )
        cpu_seconds = float(measurements[cpu_key]["median_seconds"])
        derived["single_gpu_speedup"] = cpu_seconds / single_seconds
        derived["single_gpu_meets_3x_gate"] = (
            cpu_seconds / single_seconds >= 3.0
        )
    if cpu_key in measurements and "dual" in measurements:
        cpu_seconds = float(measurements[cpu_key]["median_seconds"])
        dual_seconds = float(measurements["dual"]["median_seconds"])
        derived["dual_gpu_speedup"] = cpu_seconds / dual_seconds
        derived["dual_gpu_meets_5x_gate"] = cpu_seconds / dual_seconds >= 5.0
    if single_keys and "dual" in measurements:
        single_seconds = min(
            float(measurements[key]["median_seconds"])
            for key in single_keys
        )
        dual_seconds = float(measurements["dual"]["median_seconds"])
        derived["dual_scaling_g2"] = single_seconds / dual_seconds
        derived["dual_efficiency_e2"] = single_seconds / (2.0 * dual_seconds)
        derived["dual_scaling_meets_1_7x_gate"] = (
            single_seconds / dual_seconds >= 1.7
        )
    if single_keys and "campaign" in measurements:
        single_seconds = min(
            float(measurements[key]["median_seconds"])
            for key in single_keys
        )
        campaign_seconds = float(
            measurements["campaign"]["median_seconds"]
        )
        derived["campaign_throughput_scaling"] = (
            2.0 * single_seconds / campaign_seconds
        )
        derived["campaign_meets_1_8x_gate"] = (
            2.0 * single_seconds / campaign_seconds >= 1.8
        )

    record_external_gpu_processes()
    derived["performance_gate_eligible"] = not observed_contention
    tours_per_semantic_program = sum(
        case.batch.batch_size
        * experiment.aco.resolve_ants(case.batch.n)
        * experiment.aco.iterations
        for case in cases
    )
    representative_measurement = next(
        (
            value
            for name, value in measurements.items()
            if name != "campaign"
        ),
        measurements.get("campaign"),
    )
    if representative_measurement is None:
        raise ValueError("benchmark 至少需要一个 mode")
    representative_tours = int(
        representative_measurement["constructed_tours"]
    )
    if not any(name != "campaign" for name in measurements):
        representative_tours //= 2
    semantic_programs = (
        representative_tours // tours_per_semantic_program
    )
    payload = {
        "schema_version": 1,
        "cold_definition": (
            "in-process resident/RawKernel cache cleared; "
            "CUDA context creation excluded"
        ),
        "variant": experiment.aco.variant.value,
        "generation": args.generation,
        "requested_individuals": len(programs),
        "semantic_programs": semantic_programs,
        "instances": sum(case.batch.batch_size for case in cases),
        "scales": [case.scale for case in cases],
        "ants": experiment.aco.resolve_ants(cases[0].batch.n),
        "iterations": experiment.aco.iterations,
        "repeats": args.repeats,
        "measurements": measurements,
        "derived": derived,
        "external_gpu_processes": list(observed_contention.values()),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _command_validate_cuda_quality(args: argparse.Namespace) -> int:
    """对 CPU FP64 与 CUDA FP32 搜索执行预注册的 paired 质量门控。"""

    if min(
        args.instances_per_scale,
        args.seeds,
        args.individuals,
        args.batch_size,
        args.cpu_threads,
    ) < 1:
        raise ValueError("实例、seed、individual、batch 和 thread 数必须为正")
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    experiment = spec.experiment
    validation_pools = pools_from_paths(spec.data.validation_paths())
    missing = set(experiment.validation_scales) - set(validation_pools)
    if missing:
        raise ValueError(f"validation pool 缺少规模: {sorted(missing)}")

    random.seed(experiment.root_seed)
    np.random.seed(experiment.root_seed % (2**32))
    torch.manual_seed(experiment.root_seed)
    programs = [(None, None)]
    if args.individuals > 1:
        from .genetic import compile_individual

        population, _, _ = initialise_population(experiment.gp)
        programs.extend(
            compile_individual(individual)
            for individual in population[: args.individuals - 1]
        )

    gpu_runtime = replace(
        experiment.runtime,
        aco_backend=ExecutionBackend.CUDA_FUSED_FP32,
        processes=1,
    )
    cpu_values: dict[int, list[np.ndarray]] = {}
    gpu_values: dict[int, list[np.ndarray]] = {}
    unit_keys: dict[int, list[np.ndarray]] = {}
    cpu_seconds = 0.0
    gpu_seconds = 0.0
    selected_instance_ids: dict[str, list[str]] = {}
    rng = np.random.default_rng(experiment.root_seed ^ 0x43554441)
    seed_values = [
        int(value)
        for value in rng.integers(
            0,
            2**63 - 1,
            size=args.seeds,
            dtype=np.int64,
        )
    ]
    for scale in experiment.validation_scales:
        pool = validation_pools[scale]
        if args.instances_per_scale > len(pool):
            raise ValueError(
                f"TSP{scale} validation 仅有 {len(pool)} 个实例，"
                f"无法抽取 {args.instances_per_scale}"
            )
        indices = rng.choice(
            len(pool),
            size=args.instances_per_scale,
            replace=False,
        )
        selected = [pool.get(int(index)) for index in indices]
        selected_instance_ids[str(scale)] = [
            instance.instance_id for instance in selected
        ]
        for start in range(0, len(selected), args.batch_size):
            print(
                f"[quality] TSP{scale}: batch "
                f"{start // args.batch_size + 1}/"
                f"{(len(selected) + args.batch_size - 1) // args.batch_size}",
                file=sys.stderr,
                flush=True,
            )
            batch = make_problem_batch(
                selected[start : start + args.batch_size],
                candidate_size=experiment.aco.candidate_size,
                dtype=torch.float64,
                device="cpu",
            )
            reference = batch.reference_length.numpy()[None, :]
            for seed in seed_values:
                from .aco_numba import solve_population_numba

                cpu_started = perf_counter()
                cpu_result = solve_population_numba(
                    batch,
                    experiment.aco,
                    programs,
                    seed=seed,
                    threads=args.cpu_threads,
                )
                cpu_seconds += perf_counter() - cpu_started

                from .aco_cuda import solve_population_cuda

                gpu_started = perf_counter()
                gpu_result = solve_population_cuda(
                    batch,
                    experiment.aco,
                    programs,
                    seed=seed,
                    runtime=gpu_runtime,
                )
                gpu_seconds += perf_counter() - gpu_started
                cpu_gap = (
                    100.0
                    * (cpu_result.best_length.numpy() - reference)
                    / reference
                )
                gpu_gap = (
                    100.0
                    * (gpu_result.best_length.numpy() - reference)
                    / reference
                )
                cpu_values.setdefault(scale, []).append(cpu_gap.reshape(-1))
                gpu_values.setdefault(scale, []).append(gpu_gap.reshape(-1))
                program_index = np.repeat(
                    np.arange(len(programs), dtype=np.int64),
                    batch.batch_size,
                )
                instance_id = np.tile(
                    np.asarray(batch.instance_ids, dtype=str),
                    len(programs),
                )
                unit_keys.setdefault(scale, []).append(
                    np.asarray(
                        [
                            f"{program}:{identifier}"
                            for program, identifier in zip(
                                program_index,
                                instance_id,
                                strict=True,
                            )
                        ],
                        dtype=str,
                    )
                )

    scale_payload: dict[str, dict[str, float | int]] = {}
    all_delta: list[np.ndarray] = []
    raw_observations = 0

    def aggregate_by_unit(
        values: np.ndarray,
        keys: np.ndarray,
    ) -> np.ndarray:
        grouped: dict[str, list[float]] = {}
        for value, key in zip(values, keys, strict=True):
            grouped.setdefault(str(key), []).append(float(value))
        return np.asarray(
            [fmean(grouped[key]) for key in sorted(grouped)],
            dtype=np.float64,
        )

    for scale in experiment.validation_scales:
        raw_cpu_gap = np.concatenate(cpu_values[scale])
        raw_gpu_gap = np.concatenate(gpu_values[scale])
        keys = np.concatenate(unit_keys[scale])
        raw_observations += int(raw_cpu_gap.size)
        cpu_gap = aggregate_by_unit(raw_cpu_gap, keys)
        gpu_gap = aggregate_by_unit(raw_gpu_gap, keys)
        delta = gpu_gap - cpu_gap
        all_delta.append(delta)
        standard_error = (
            float(delta.std(ddof=1) / np.sqrt(delta.size))
            if delta.size > 1
            else 0.0
        )
        scale_payload[str(scale)] = {
            "observations": int(delta.size),
            "raw_seed_observations": int(raw_cpu_gap.size),
            "cpu_mean_gap_percent": float(cpu_gap.mean()),
            "gpu_mean_gap_percent": float(gpu_gap.mean()),
            "mean_delta_pp": float(delta.mean()),
            "upper_bound_95_pp": float(
                delta.mean() + 1.645 * standard_error
            ),
            "wins": int((delta < -1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()),
            "losses": int((delta > 1e-12).sum()),
            "passed": bool(
                delta.mean() + 1.645 * standard_error
                <= args.tolerance_pp
            ),
        }
    pooled = np.concatenate(all_delta)
    pooled_standard_error = (
        float(pooled.std(ddof=1) / np.sqrt(pooled.size))
        if pooled.size > 1
        else 0.0
    )
    upper_bound = float(
        pooled.mean() + 1.645 * pooled_standard_error
    )
    passed = (
        upper_bound <= args.tolerance_pp
        and all(
            bool(summary["passed"])
            for summary in scale_payload.values()
        )
    )
    payload = {
        "schema_version": 1,
        "contract": "gpu-fp32-search-cpu-fp64-score",
        "statistical_unit": "program-instance (ACO seeds aggregated first)",
        "experiment_id": experiment.experiment_id,
        "root_seed": experiment.root_seed,
        "aco_config_hash": experiment.aco.config_hash,
        "aco_seed_values": seed_values,
        "selected_instance_ids": selected_instance_ids,
        "programs_manifest": [
            {
                "transition": (
                    None if transition is None else transition.expression
                ),
                "pheromone": (
                    None if pheromone is None else pheromone.expression
                ),
            }
            for transition, pheromone in programs
        ],
        "gpu_devices": list(gpu_runtime.gpu_devices),
        "gpu_mode": gpu_runtime.gpu_mode.value,
        "gpu_block_threads": gpu_runtime.gpu_block_threads or 32,
        "variant": experiment.aco.variant.value,
        "instances_per_scale": args.instances_per_scale,
        "seeds": args.seeds,
        "programs": len(programs),
        "observations": int(pooled.size),
        "raw_seed_observations": raw_observations,
        "tolerance_pp": args.tolerance_pp,
        "mean_delta_pp": float(pooled.mean()),
        "upper_bound_95_pp": upper_bound,
        "passed": passed,
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "speedup": cpu_seconds / max(gpu_seconds, 1e-12),
        "scales": scale_payload,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 1


def _command_benchmark_training(args: argparse.Namespace) -> int:
    """用正式每代计算规模执行 1--3 代、但不做 validation/checkpoint。"""

    if not 1 <= args.generations <= 3:
        raise ValueError("benchmark-training 的 --generations 仅允许 1--3")
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    training_paths = spec.data.training_paths()
    validation_paths = spec.data.validation_paths()
    if not args.skip_manifest_check:
        _preflight_manifest(
            spec,
            args.manifest,
            [
                *((path, "train") for paths in training_paths.values() for path in paths),
                *(
                    (path, "validation")
                    for paths in validation_paths.values()
                    for path in paths
                ),
            ],
        )
    configure_runtime(spec.experiment.runtime)
    training_pools = pools_from_paths(training_paths)
    validation_pools = pools_from_paths(validation_paths)
    schedule = _protocol_schedule(
        spec,
        training_pools,
        validation_pools,
        path=args.schedule,
        phase=args.phase,
        replicate_id=args.replicate_id,
    )
    sampler = ScheduledTrainingSampler(
        schedule,
        training_pools,
        replicate_id=args.replicate_id,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
    )
    baseline_path = (
        Path(args.baseline_archive).resolve()
        if args.baseline_archive
        else spec.data.baseline_path
    )
    baseline_archive = (
        BaselineArchive(
            baseline_path,
            spec.experiment.aco,
            spec.experiment.runtime.aco_backend,
            require=(spec.data.baseline_policy == "require"),
        )
        if baseline_path is not None
        else None
    )
    if spec.data.baseline_policy == "require" and baseline_archive is None:
        raise ValueError("baseline_policy=require 但未配置 baseline archive")

    random.seed(spec.experiment.root_seed)
    np.random.seed(spec.experiment.root_seed % (2**32))
    torch.manual_seed(spec.experiment.root_seed)
    population, transition_pset, pheromone_pset = initialise_population(
        spec.experiment.gp
    )
    cache = BaselineCache(baseline_archive)
    pending_cases = sampler.cases_for_generation(1)
    records: list[dict[str, object]] = []
    benchmark_started = perf_counter()

    with EvaluationPool(spec.experiment) as evaluator:
        evaluator.warm(pending_cases[0])
        for generation in range(1, args.generations + 1):
            generation_started = perf_counter()
            cases = (
                pending_cases
                if generation == 1
                else sampler.cases_for_generation(generation)
            )
            for individual in population:
                if individual.fitness.valid:
                    del individual.fitness.values
            evaluation = evaluator.evaluate_population(
                population,
                cases,
                cache,
            )
            evaluated_population = population
            fitness = np.asarray(
                [
                    individual.fitness.values[0]
                    for individual in evaluated_population
                ],
                dtype=np.float64,
            )
            best = min(
                evaluated_population,
                key=lambda item: (item.fitness.values[0], item.total_nodes),
            )
            breakdown = best.metadata["fitness_breakdown"]
            per_program_tours = sum(
                case.batch.batch_size
                * spec.experiment.aco.resolve_ants(case.batch.n)
                * spec.experiment.aco.iterations
                for case in cases
            )
            semantic_unique = (
                evaluation.constructed_tours // per_program_tours
                if per_program_tours
                else 0
            )

            breeding_started = perf_counter()
            if generation < args.generations:
                population = evolve_generation(
                    evaluated_population,
                    transition_pset,
                    pheromone_pset,
                    spec.experiment.gp,
                )
            breeding_seconds = perf_counter() - breeding_started
            generation_seconds = perf_counter() - generation_started
            record = {
                "generation": generation,
                "instances": sum(
                    case.batch.batch_size for case in cases
                ),
                "structural_unique": evaluation.evaluated_unique,
                "semantic_unique": semantic_unique,
                "fitness_min_gap_percent": float(fitness.min()),
                "fitness_median_gap_percent": float(np.median(fitness)),
                "fitness_mean_gap_percent": float(fitness.mean()),
                "best_nodes": best.total_nodes,
                "best_hash": best.structural_hash,
                "best_gap_percent_by_scale": breakdown.mean_gap_by_scale,
                "baseline_gap_percent_by_scale": (
                    breakdown.baseline_gap_by_scale
                ),
                "best_delta_pp_by_scale": breakdown.mean_delta_by_scale,
                "baseline_lookup_seconds": evaluation.baseline_wall_time,
                "evaluation_seconds": evaluation.evaluation_wall_time,
                "breeding_seconds": breeding_seconds,
                "generation_seconds": generation_seconds,
                "constructed_tours": evaluation.constructed_tours,
                "tours_per_second": (
                    evaluation.constructed_tours
                    / max(evaluation.evaluation_wall_time, 1e-12)
                ),
            }
            records.append(record)
            print(
                f"benchmark_generation={generation} "
                f"structural={evaluation.evaluated_unique} "
                f"semantic={semantic_unique} "
                f"fitness={fitness.min():.6f}% "
                f"evaluation={evaluation.evaluation_wall_time:.3f}s "
                f"generation={generation_seconds:.3f}s "
                f"delta={breakdown.mean_delta_by_scale}",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "purpose": "1--3 generation acceleration benchmark; not a final run",
        "config": str(Path(args.config).resolve()),
        "schedule_hash": schedule.manifest_hash,
        "variant": spec.experiment.aco.variant.value,
        "method_profile": args.method_profile,
        "root_seed": spec.experiment.root_seed,
        "population_size": spec.experiment.gp.population_size,
        "requested_generations": args.generations,
        "cpu_threads": spec.experiment.runtime.cpu_threads,
        "backend": spec.experiment.runtime.aco_backend.value,
        "total_seconds": perf_counter() - benchmark_started,
        "records": records,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _command_prepare_pilot_plan(args: argparse.Namespace) -> int:
    plan = build_protocol_a_v05_pilot_plan(
        runs_root=args.runs_root,
        python=args.python,
    )
    target, shell_target = write_experiment_plan(plan, args.output)
    print(
        json.dumps(
            {
                "output": str(target),
                "shell": str(shell_target),
                "setup_tasks": plan["setup_task_count"],
                "training_tasks": plan["training_task_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _command_train(args: argparse.Namespace) -> int:
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    training_paths = spec.data.training_paths()
    validation_paths = spec.data.validation_paths()
    if not args.skip_manifest_check:
        _preflight_manifest(
            spec,
            args.manifest,
            [
                *((path, "train") for paths in training_paths.values() for path in paths),
                *(
                    (path, "validation")
                    for paths in validation_paths.values()
                    for path in paths
                ),
            ],
        )
    configure_runtime(spec.experiment.runtime)
    training_pools = pools_from_paths(training_paths)
    validation_pools = pools_from_paths(validation_paths)
    schedule = _protocol_schedule(
        spec,
        training_pools,
        validation_pools,
        path=args.schedule,
        phase=args.phase,
        replicate_id=args.replicate_id,
    )
    sampler = ScheduledTrainingSampler(
        schedule,
        training_pools,
        replicate_id=args.replicate_id,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
    )
    validation_screening_cases = validation_cases_from_schedule(
        schedule,
        validation_pools,
        role="selection",
        replicate_id=args.replicate_id,
        batch_size=spec.data.evaluation_batch_size,
        seeds=spec.experiment.validation_screening_seeds,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
    )
    validation_cases = validation_cases_from_schedule(
        schedule,
        validation_pools,
        role="selection",
        replicate_id=args.replicate_id,
        batch_size=spec.data.evaluation_batch_size,
        seeds=spec.experiment.validation_seeds,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
    )
    validation_gate_cases = validation_cases_from_schedule(
        schedule,
        validation_pools,
        role="gate",
        replicate_id=args.replicate_id,
        batch_size=spec.data.evaluation_batch_size,
        seeds=spec.experiment.validation_seeds,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
    )
    baseline_path = (
        Path(args.baseline_archive).resolve()
        if args.baseline_archive
        else spec.data.baseline_path
    )
    baseline_archive = (
        BaselineArchive(
            baseline_path,
            spec.experiment.aco,
            spec.experiment.runtime.aco_backend,
            require=(spec.data.baseline_policy == "require"),
        )
        if baseline_path is not None
        else None
    )
    if spec.data.baseline_policy == "require" and baseline_archive is None:
        raise ValueError("baseline_policy=require 但未配置 baseline archive")
    if args.resume and not args.output:
        resume_path = Path(args.resume)
        output = resume_path if resume_path.is_dir() else resume_path.parent
    else:
        output = (
            Path(args.output)
            if args.output
            else Path("runs")
            / spec.experiment.experiment_id
            / f"seed-{spec.experiment.root_seed}"
        )
    artifact_payload = (
        resume_run_artifacts(output)
        if args.resume
        else initialise_run_artifacts(
            output,
            spec.experiment,
            repository=_repository_root(),
            data_manifest=args.manifest,
        )
    )
    if not args.resume:
        write_schedule(schedule, output / "schedule.json")
    try:
        result = train(
            spec.experiment,
            sampler.cases_for_generation,
            validation_cases,
            validation_screening_cases=validation_screening_cases,
            validation_gate_cases=validation_gate_cases,
            validation_monitor_cases=validation_screening_cases,
            baseline_archive=baseline_archive,
            output_directory=output,
            resume_from=args.resume,
            progress_callback=lambda record: print(
                (
                    f"generation={record.generation:03d} "
                    f"unique={record.evaluated_unique:03d} "
                    f"min={record.minimum:.6f} "
                    f"median={record.median:.6f} "
                    f"mean={record.mean:.6f} "
                    f"nodes={record.best_nodes} "
                    f"time={record.generation_wall_time:.2f}s "
                    f"eta={record.eta_seconds / 60.0:.1f}min "
                    f"train_delta={record.best_mean_delta_by_scale} "
                    f"val_delta={record.validation_monitor_delta_by_scale}"
                ),
                flush=True,
            ),
        )
    except Exception as exc:
        finalise_run_artifacts(
            output,
            artifact_payload,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        if args.traceback:
            traceback.print_exc()
        else:
            print(f"训练失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finalise_run_artifacts(output, artifact_payload, status="completed")
    audit_status = (
        ""
        if result.cpu_fp64_audit is None
        else (
            "；CPU/FP64-audit="
            + (
                "pass"
                if result.cpu_fp64_audit.passed_noninferiority
                else "fail"
            )
        )
    )
    print(
        f"训练完成：{output}；champion nodes={result.champion.total_nodes}；"
        f"non-inferiority={'pass' if result.passed_noninferiority else 'fallback'}"
        f"{audit_status}"
    )
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    paths = spec.data.test_paths(args.partition)
    if not args.skip_manifest_check:
        _preflight_manifest(
            spec,
            args.manifest,
            [(path, "test") for path in paths],
        )
    configure_runtime(spec.experiment.runtime)
    partition_spec = spec.data.test[args.partition]
    batches = iter_problem_batches(
        paths,
        batch_size=args.batch_size or spec.data.evaluation_batch_size,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
        min_scale=partition_spec.min_scale,
        max_scale=partition_spec.max_scale,
        max_instances=args.max_instances,
    )
    champion = load_champion(args.champion) if args.champion else None
    transition, pheromone = compile_champion(champion)
    records = evaluate_batches(
        batches,
        spec.experiment.aco,
        method=args.method,
        champion_id=args.champion_id,
        partition=args.partition,
        distribution=partition_spec.distribution,
        root_seed=spec.experiment.root_seed,
        seeds_per_batch=args.seeds,
        transition_program=transition,
        pheromone_program=pheromone,
        backend=spec.experiment.runtime.aco_backend,
        runtime=spec.experiment.runtime,
        gp_run_id=args.gp_run_id,
    )
    target = write_records(records, args.output)
    print(f"评测完成：{len(records)} 条记录 -> {target}")
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    records = read_records(args.inputs)
    contexts = {
        (record.variant, record.partition, record.distribution)
        for record in records
    }
    if len(contexts) != 1:
        raise ValueError(
            "一次 summarize 只能分析同一 variant/partition/distribution；"
            "请先拆分输入，避免把不可交换的 raw gaps 混合"
        )
    method_counts = Counter(record.method for record in records)
    summaries = summarize_quality(records)
    friedman = friedman_test(records) if len(method_counts) >= 3 else None
    pairwise = (
        paired_wilcoxon_holm(
            records,
            reference_method=args.reference_method,
        )
        if len(method_counts) >= 2
        else []
    )
    bootstrap = hierarchical_bootstrap_delta(
        records,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    factorial = (
        factorial_contrasts(
            records,
            core_f0=args.factorial_methods[0],
            core_f1=args.factorial_methods[1],
            full_f0=args.factorial_methods[2],
            full_f1=args.factorial_methods[3],
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        if args.factorial_methods
        else []
    )
    context = next(iter(contexts))
    target = write_statistical_report(
        args.output,
        summaries=summaries,
        friedman=friedman,
        pairwise=pairwise,
        bootstrap=bootstrap,
        factorial=factorial,
        metadata={
            "variant": context[0],
            "partition": context[1],
            "distribution": context[2],
            "scales": sorted({record.scale for record in records}),
            "input_files": [str(Path(item).resolve()) for item in args.inputs],
        },
    )
    print(
        json.dumps(
            {
                "output": str(target),
                "methods": dict(method_counts),
                "common_context": context,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _command_evaluate_study(args: argparse.Namespace) -> int:
    """执行一个可恢复的 variant×partition 正式测试任务。"""

    from .study import evaluate_study_partition, load_study_spec

    study = load_study_spec(args.study_config)
    target = evaluate_study_partition(
        study,
        variant_name=args.variant,
        partition=args.partition,
        skip_manifest_check=args.skip_manifest_check,
    )
    print(f"study 测试完成：{target}")
    return 0


def _command_report_study(args: argparse.Namespace) -> int:
    """汇总九个训练 run 和十二个 test partitions。"""

    from .study import load_study_spec
    from .study_report import generate_study_report

    study = load_study_spec(args.study_config)
    target = generate_study_report(study)
    print(f"study 报告完成：{target}")
    return 0


def _command_run_study(args: argparse.Namespace) -> int:
    """持有唯一锁并串行运行单 GPU0 study。"""

    from .study import load_study_spec, run_study_queue

    study = load_study_spec(args.study_config)
    run_study_queue(study)
    print(f"study 队列完成：{study.output_root}")
    return 0


def _command_study_status(args: argparse.Namespace) -> int:
    """打印后台 study 的机器可读状态。"""

    from .study import load_study_spec, study_status

    study = load_study_spec(args.study_config)
    print(json.dumps(study_status(study), ensure_ascii=False, indent=2))
    return 0


def _command_evaluate_ablation_study(args: argparse.Namespace) -> int:
    """批量执行一个消融 variant×partition×integration group。"""

    from .ablation import evaluate_ablation_group, load_ablation_spec

    study = load_ablation_spec(args.study_config)
    target = evaluate_ablation_group(
        study,
        variant_name=args.variant,
        partition=args.partition,
        group=args.group,
    )
    print(f"ablation 测试完成：{target}")
    return 0


def _command_benchmark_ablation_efficiency(args: argparse.Namespace) -> int:
    """孤立测量一个 ACO 变体全部核心方法的推理效率。"""

    from .ablation import (
        benchmark_ablation_efficiency,
        load_ablation_spec,
    )

    study = load_ablation_spec(args.study_config)
    target = benchmark_ablation_efficiency(
        study,
        variant_name=args.variant,
        output=args.output,
    )
    print(f"ablation 效率测试完成：{target}")
    return 0


def _command_report_ablation_study(args: argparse.Namespace) -> int:
    """生成跨方法 factorial、机制与 OOD 中文报告。"""

    from .ablation import load_ablation_spec
    from .ablation_report import generate_ablation_report

    study = load_ablation_spec(args.study_config)
    target = generate_ablation_report(study)
    print(f"ablation 报告完成：{target}")
    return 0


def _command_run_ablation_study(args: argparse.Namespace) -> int:
    """在一张或多张物理 GPU 上运行可恢复消融队列。"""

    from .ablation import (
        load_ablation_spec,
        run_ablation_parallel,
        run_ablation_queue,
    )

    study = load_ablation_spec(args.study_config)
    if args.physical_gpus:
        run_ablation_parallel(
            study,
            physical_devices=tuple(args.physical_gpus),
        )
    else:
        run_ablation_queue(study)
    print(f"ablation 队列完成：{study.output_root}")
    return 0


def _command_ablation_status(args: argparse.Namespace) -> int:
    """打印消融队列状态与当前训练进度。"""

    from .ablation import ablation_status, load_ablation_spec

    study = load_ablation_spec(args.study_config)
    print(json.dumps(ablation_status(study), ensure_ascii=False, indent=2))
    return 0


def _add_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    """为可执行 ACO 的命令加入一致的 CUDA 调度覆盖参数。"""

    parser.add_argument("--gpu-devices", nargs="+", type=int)
    parser.add_argument(
        "--gpu-mode",
        choices=[mode.value for mode in GPUMode],
    )
    parser.add_argument(
        "--gpu-block-threads",
        type=int,
        choices=[0, 32, 64],
    )
    parser.add_argument(
        "--gpu-task-chunk-size",
        type=int,
        help="0/省略表示按 20%% 显存保留策略自动分块",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rmtgp-aco",
        description="Strongly Typed Multi-Tree GP–ACO 研究工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="生成数据 manifest")
    manifest.add_argument("--root", default="Datasets/TSP")
    manifest.add_argument("--output", default="Datasets/manifest.json")
    manifest.add_argument("--full", action="store_true", help="计算完整 SHA-256 与行数")
    manifest.add_argument("--workers", type=int, default=4)
    manifest.set_defaults(handler=_command_manifest)

    verify = subparsers.add_parser("verify-data", help="校验数据 manifest")
    verify.add_argument("--manifest", default="Datasets/manifest.json")
    verify.add_argument("--root")
    verify.add_argument("--skip-hashes", action="store_true")
    verify.add_argument("--leakage-samples", type=int, default=1)
    verify.set_defaults(handler=_command_verify_data)

    schedule_parser = subparsers.add_parser(
        "prepare-schedules",
        help="生成紧凑 train/selection/gate schedule",
    )
    schedule_parser.add_argument("--config", required=True)
    schedule_parser.add_argument("--output", required=True)
    schedule_parser.add_argument("--phase", choices=["pilot", "formal"], required=True)
    schedule_parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    schedule_parser.add_argument("--replicate-id", type=int, default=0)
    schedule_parser.add_argument("--root-seed", type=int)
    schedule_parser.add_argument("--cpu-threads", type=int)
    schedule_parser.add_argument("--manifest", default="Datasets/manifest.json")
    schedule_parser.add_argument("--skip-manifest-check", action="store_true")
    schedule_parser.set_defaults(handler=_command_prepare_schedules)

    baseline_parser = subparsers.add_parser(
        "precompute-baselines",
        help="按冻结 schedule 提前计算不可变原始 ACO baseline",
    )
    baseline_parser.add_argument("--config", required=True)
    baseline_parser.add_argument("--schedule", required=True)
    baseline_parser.add_argument("--output", required=True)
    baseline_parser.add_argument("--replicate-id", type=int, default=0)
    baseline_parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "selection", "gate"],
        default=["train", "selection", "gate"],
    )
    baseline_parser.add_argument("--root-seed", type=int)
    baseline_parser.add_argument("--cpu-threads", type=int)
    baseline_parser.add_argument(
        "--backend",
        choices=[backend.value for backend in ExecutionBackend],
    )
    _add_gpu_arguments(baseline_parser)
    baseline_parser.set_defaults(handler=_command_precompute_baselines)

    benchmark_parser = subparsers.add_parser(
        "benchmark-backends",
        help="比较旧 process Numba 与 population-batched Numba",
    )
    benchmark_parser.add_argument("--config", required=True)
    benchmark_parser.add_argument("--schedule")
    benchmark_parser.add_argument(
        "--phase",
        choices=["pilot", "formal", "development"],
        default="pilot",
    )
    benchmark_parser.add_argument("--replicate-id", type=int, default=0)
    benchmark_parser.add_argument("--root-seed", type=int)
    benchmark_parser.add_argument("--cpu-threads", type=int, default=16)
    benchmark_parser.add_argument("--reference-processes", type=int, default=8)
    benchmark_parser.add_argument("--max-individuals", type=int, default=100)
    benchmark_parser.add_argument("--output")
    _add_gpu_arguments(benchmark_parser)
    benchmark_parser.set_defaults(handler=_command_benchmark_backends)

    accelerator_benchmark = subparsers.add_parser(
        "benchmark-accelerators",
        help="同代比较 CPU8/CPU16、单 GPU、双 GPU 与双 run campaign",
    )
    accelerator_benchmark.add_argument("--config", required=True)
    accelerator_benchmark.add_argument("--schedule")
    accelerator_benchmark.add_argument(
        "--phase",
        choices=["pilot", "formal", "development"],
        default="development",
    )
    accelerator_benchmark.add_argument("--replicate-id", type=int, default=0)
    accelerator_benchmark.add_argument("--root-seed", type=int)
    accelerator_benchmark.add_argument("--generation", type=int, default=1)
    accelerator_benchmark.add_argument("--iterations", type=int)
    accelerator_benchmark.add_argument("--max-individuals", type=int, default=100)
    accelerator_benchmark.add_argument("--repeats", type=int, default=3)
    accelerator_benchmark.add_argument(
        "--modes",
        nargs="+",
        choices=["cpu8", "cpu16", "gpu0", "gpu1", "dual", "campaign"],
        default=["cpu8", "cpu16", "gpu0", "gpu1", "dual", "campaign"],
    )
    accelerator_benchmark.add_argument("--output")
    accelerator_benchmark.add_argument("--cpu-threads", type=int)
    _add_gpu_arguments(accelerator_benchmark)
    accelerator_benchmark.set_defaults(
        handler=_command_benchmark_accelerators
    )

    cuda_quality = subparsers.add_parser(
        "validate-cuda-quality",
        help="CPU FP64 与 CUDA FP32 搜索的 paired 非劣质量门控",
    )
    cuda_quality.add_argument("--config", required=True)
    cuda_quality.add_argument("--instances-per-scale", type=int, default=128)
    cuda_quality.add_argument("--seeds", type=int, default=3)
    cuda_quality.add_argument("--individuals", type=int, default=1)
    cuda_quality.add_argument("--batch-size", type=int, default=16)
    cuda_quality.add_argument("--cpu-threads", type=int, default=16)
    cuda_quality.add_argument("--tolerance-pp", type=float, default=0.10)
    cuda_quality.add_argument("--root-seed", type=int)
    cuda_quality.add_argument("--output")
    _add_gpu_arguments(cuda_quality)
    cuda_quality.set_defaults(handler=_command_validate_cuda_quality)

    training_benchmark = subparsers.add_parser(
        "benchmark-training",
        help="按正式每代规模短跑 1--3 代，不执行 validation/checkpoint",
    )
    training_benchmark.add_argument("--config", required=True)
    training_benchmark.add_argument("--schedule")
    training_benchmark.add_argument("--baseline-archive")
    training_benchmark.add_argument("--generations", type=int, default=3)
    training_benchmark.add_argument("--output")
    training_benchmark.add_argument("--manifest", default="Datasets/manifest.json")
    training_benchmark.add_argument("--skip-manifest-check", action="store_true")
    training_benchmark.add_argument("--root-seed", type=int)
    training_benchmark.add_argument("--cpu-threads", type=int, default=16)
    training_benchmark.add_argument("--replicate-id", type=int, default=0)
    training_benchmark.add_argument(
        "--phase",
        choices=["pilot", "formal", "development"],
        default="pilot",
    )
    training_benchmark.add_argument(
        "--backend",
        choices=[backend.value for backend in ExecutionBackend],
    )
    training_benchmark.add_argument(
        "--method-profile",
        choices=[
            "rmtgp",
            "tr-rgp",
            "ph-rgp",
            "matched-replace",
            "legacy",
            "rmtgp-core-f0",
            "rmtgp-core-f1",
            "rmtgp-full-f0",
            "rmtgp-full-f1",
        ],
        default="rmtgp-full-f1",
    )
    _add_gpu_arguments(training_benchmark)
    training_benchmark.set_defaults(handler=_command_benchmark_training)

    plan_parser = subparsers.add_parser(
        "prepare-pilot-plan",
        help="生成 Protocol A v0.5 的 78-run pilot 任务图",
    )
    plan_parser.add_argument(
        "--output",
        default="runs/protocol-a-v0.5/pilot-plan.json",
    )
    plan_parser.add_argument(
        "--runs-root",
        default="runs/protocol-a-v0.5",
    )
    plan_parser.add_argument("--python")
    plan_parser.set_defaults(handler=_command_prepare_pilot_plan)

    train_parser = subparsers.add_parser("train", help="训练一个独立 GP run")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output")
    train_parser.add_argument("--manifest", default="Datasets/manifest.json")
    train_parser.add_argument("--skip-manifest-check", action="store_true")
    train_parser.add_argument("--root-seed", type=int)
    train_parser.add_argument("--processes", type=int)
    train_parser.add_argument("--cpu-threads", type=int)
    train_parser.add_argument("--schedule")
    train_parser.add_argument("--baseline-archive")
    train_parser.add_argument("--replicate-id", type=int, default=0)
    train_parser.add_argument(
        "--phase",
        choices=["pilot", "formal", "development"],
        default="pilot",
    )
    train_parser.add_argument(
        "--backend",
        choices=[backend.value for backend in ExecutionBackend],
    )
    train_parser.add_argument(
        "--resume",
        help="run 目录或 training_state.pkl；配置必须与 checkpoint 完全一致",
    )
    train_parser.add_argument(
        "--method-profile",
        choices=[
            "rmtgp",
            "tr-rgp",
            "ph-rgp",
            "matched-replace",
            "legacy",
            "rmtgp-core-f0",
            "rmtgp-core-f1",
            "rmtgp-full-f0",
            "rmtgp-full-f1",
        ],
        default="rmtgp",
        help="E1 组件/上一篇研究对照；默认训练双 residual",
    )
    train_parser.add_argument("--traceback", action="store_true")
    _add_gpu_arguments(train_parser)
    train_parser.set_defaults(handler=_command_train)

    evaluate = subparsers.add_parser("evaluate", help="锁定模型后的 paired 测试")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--partition", required=True)
    evaluate.add_argument("--champion", help="省略时评测原始 ACO")
    evaluate.add_argument("--method", required=True)
    evaluate.add_argument("--champion-id", default="baseline")
    evaluate.add_argument(
        "--gp-run-id",
        help="跨方法配对的 GP replicate 标识；默认使用 champion-id",
    )
    evaluate.add_argument("--seeds", type=int, required=True)
    evaluate.add_argument("--max-instances", type=int)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--manifest", default="Datasets/manifest.json")
    evaluate.add_argument("--skip-manifest-check", action="store_true")
    evaluate.add_argument("--root-seed", type=int)
    evaluate.add_argument("--processes", type=int)
    evaluate.add_argument(
        "--backend",
        choices=[backend.value for backend in ExecutionBackend],
    )
    _add_gpu_arguments(evaluate)
    evaluate.set_defaults(handler=_command_evaluate)

    summarize = subparsers.add_parser("summarize", help="统计检验与论文指标汇总")
    summarize.add_argument("--inputs", nargs="+", required=True)
    summarize.add_argument("--reference-method", required=True)
    summarize.add_argument("--bootstrap-replicates", type=int, default=10_000)
    summarize.add_argument("--seed", type=int, default=0)
    summarize.add_argument(
        "--factorial-methods",
        nargs=4,
        metavar="METHOD",
        help="依次给出 Core-F0 Core-F1 Full-F0 Full-F1 的方法名",
    )
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(handler=_command_summarize)

    evaluate_study = subparsers.add_parser(
        "evaluate-study",
        help="共享 baseline cache 的三 GP-seed 正式测试",
    )
    evaluate_study.add_argument("--study-config", required=True)
    evaluate_study.add_argument("--variant", choices=["as", "acs", "mmas"], required=True)
    evaluate_study.add_argument("--partition", required=True)
    evaluate_study.add_argument("--skip-manifest-check", action="store_true")
    evaluate_study.set_defaults(handler=_command_evaluate_study)

    report_study = subparsers.add_parser(
        "report-study",
        help="生成 train/validation 曲线、paired 统计与中文报告",
    )
    report_study.add_argument("--study-config", required=True)
    report_study.set_defaults(handler=_command_report_study)

    run_study = subparsers.add_parser(
        "run-study",
        help="在唯一可见物理 GPU 上串行执行可恢复 study 队列",
    )
    run_study.add_argument("--study-config", required=True)
    run_study.set_defaults(handler=_command_run_study)

    status_study = subparsers.add_parser(
        "study-status",
        help="查看后台 study 当前任务、训练代数与 ETA",
    )
    status_study.add_argument("--study-config", required=True)
    status_study.set_defaults(handler=_command_study_status)

    evaluate_ablation = subparsers.add_parser(
        "evaluate-ablation-study",
        help="批量评测消融 study 的一个 integration group",
    )
    evaluate_ablation.add_argument("--study-config", required=True)
    evaluate_ablation.add_argument(
        "--variant",
        choices=["as", "acs", "mmas"],
        required=True,
    )
    evaluate_ablation.add_argument("--partition", required=True)
    evaluate_ablation.add_argument(
        "--group",
        choices=["residual", "replacement"],
        required=True,
    )
    evaluate_ablation.set_defaults(
        handler=_command_evaluate_ablation_study
    )

    efficiency_ablation = subparsers.add_parser(
        "benchmark-ablation-efficiency",
        help="孤立测量消融 champions 的 GPU 推理效率",
    )
    efficiency_ablation.add_argument("--study-config", required=True)
    efficiency_ablation.add_argument(
        "--variant",
        choices=["as", "acs", "mmas"],
        required=True,
    )
    efficiency_ablation.add_argument("--output", required=True)
    efficiency_ablation.set_defaults(
        handler=_command_benchmark_ablation_efficiency
    )

    report_ablation = subparsers.add_parser(
        "report-ablation-study",
        help="生成消融、factorial、机制与 OOD 中文报告",
    )
    report_ablation.add_argument("--study-config", required=True)
    report_ablation.set_defaults(handler=_command_report_ablation_study)

    run_ablation = subparsers.add_parser(
        "run-ablation-study",
        help="在一张或多张物理 GPU 上执行可恢复消融队列",
    )
    run_ablation.add_argument("--study-config", required=True)
    run_ablation.add_argument(
        "--physical-gpus",
        nargs="+",
        type=int,
        help="并行 runner 使用的物理 GPU indexes，例如 0 1",
    )
    run_ablation.set_defaults(handler=_command_run_ablation_study)

    status_ablation = subparsers.add_parser(
        "ablation-status",
        help="查看消融后台任务和当前训练代数",
    )
    status_ablation.add_argument("--study-config", required=True)
    status_ablation.set_defaults(handler=_command_ablation_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
