"""RMTGP-ACO 可复现训练、评测、数据审计与统计命令行。"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
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
    PheromoneIntegration,
    TransitionIntegration,
)
from .evaluation import (
    compile_champion,
    evaluate_batches,
    load_champion,
    read_records,
    write_records,
)
from .experiment_plan import (
    build_protocol_a_v03_pilot_plan,
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
            protocol_id="protocol-a-v0.3",
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
        protocol_id="protocol-a-v0.3",
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
        protocol_id="protocol-a-v0.3",
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
    plan = build_protocol_a_v03_pilot_plan(
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
                    f"delta={record.best_mean_delta_by_scale}"
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
    print(
        f"训练完成：{output}；champion nodes={result.champion.total_nodes}；"
        f"non-inferiority={'pass' if result.passed_noninferiority else 'fallback'}"
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
    schedule_parser.add_argument("--protocol-id", default="protocol-a-v0.3")
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
    benchmark_parser.set_defaults(handler=_command_benchmark_backends)

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
    training_benchmark.set_defaults(handler=_command_benchmark_training)

    plan_parser = subparsers.add_parser(
        "prepare-pilot-plan",
        help="生成 Protocol A v0.3 的 78-run pilot 任务图",
    )
    plan_parser.add_argument(
        "--output",
        default="runs/protocol-a-v0.3/pilot-plan.json",
    )
    plan_parser.add_argument(
        "--runs-root",
        default="runs/protocol-a-v0.3",
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
