"""RMTGP fitness、持久并行训练、断点恢复与 validation selection。"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import platform
import random
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from time import perf_counter, sleep
from typing import TypeVar

import numpy as np
import torch
import yaml
from deap import tools

from .aco import solve
from .baseline import BaselineArchive
from .config import ExecutionBackend, ExperimentConfig
from .genetic import (
    RMTGPIndividual,
    compile_individual,
    evolve_generation,
    initialise_population,
    make_individual,
)
from .program import create_primitive_sets
from .sampling import EvaluationCase

_WORKER_EXPERIMENT: ExperimentConfig | None = None
_CHECKPOINT_SCHEMA_VERSION = 2
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class FitnessBreakdown:
    """一个个体在当前 mini-batch 上的可审计 fitness 组成。"""

    fitness: float
    mean_gap_by_scale: dict[int, float]
    median_gap_by_scale: dict[int, float]
    baseline_gap_by_scale: dict[int, float]
    mean_delta_by_scale: dict[int, float]
    degradation_by_scale: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationData:
    """候选在 validation 上的 absolute/baseline/paired gap 长表数组。"""

    candidate_gap_by_scale: dict[int, np.ndarray]
    baseline_gap_by_scale: dict[int, np.ndarray]
    delta_by_scale: dict[int, np.ndarray]
    instance_ids_by_scale: dict[int, np.ndarray]


@dataclass(slots=True)
class PopulationEvaluationResult:
    """一代 population evaluation 的结果和分阶段耗时。"""

    evaluated_unique: int
    breakdowns: dict[str, FitnessBreakdown]
    baseline_wall_time: float
    evaluation_wall_time: float
    constructed_tours: int = 0


@dataclass(slots=True)
class GenerationRecord:
    """一代 GP 的质量、复杂度和墙钟时间统计。"""

    generation: int
    evaluated_unique: int
    unique_genotypes: int
    minimum: float
    first_quartile: float
    median: float
    mean: float
    third_quartile: float
    standard_deviation: float
    best_nodes: int
    best_transition_nodes: int
    best_pheromone_nodes: int
    best_hash: str
    best_transition_expression: str
    best_pheromone_expression: str
    best_mean_gap_by_scale: dict[int, float]
    best_median_gap_by_scale: dict[int, float]
    baseline_mean_gap_by_scale: dict[int, float]
    best_mean_delta_by_scale: dict[int, float]
    baseline_wall_time: float
    evaluation_wall_time: float
    breeding_wall_time: float
    checkpoint_wall_time: float
    generation_wall_time: float
    cumulative_wall_time: float
    unique_individuals_per_second: float
    constructed_tours: int
    tours_per_second: float
    eta_seconds: float


@dataclass(frozen=True, slots=True)
class ValidationScaleSummary:
    """锁定候选在一个规模上的 paired validation 汇总。"""

    scale: int
    observations: int
    instances: int
    mean_delta_pp: float
    median_delta_pp: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    normal_upper_bound_95: float
    wins: int
    ties: int
    losses: int


@dataclass(slots=True)
class ValidationSelection:
    """validation 模型选择、门控和候选审计信息。"""

    champion: RMTGPIndividual
    passed_noninferiority: bool
    selected_candidate_hash: str
    selected_macro_gap_percent: float
    selected_macro_delta_pp: float
    screened_candidates: int
    finalist_candidates: int
    unique_candidates: int
    wall_time_sec: float
    scales: list[ValidationScaleSummary]


@dataclass(slots=True)
class TrainingResult:
    """一次独立 GP run 的返回值。"""

    champion: RMTGPIndividual
    history: list[GenerationRecord]
    checkpoints: list[RMTGPIndividual]
    passed_noninferiority: bool
    validation: ValidationSelection
    output_directory: Path | None = None


class BaselineCache:
    """按配置、后端、实例 IDs 和 seed 缓存原始 ACO 结果。"""

    def __init__(self, archive: BaselineArchive | None = None) -> None:
        self._values: dict[tuple[object, ...], torch.Tensor] = {}
        self.archive = archive

    @staticmethod
    def key(
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> tuple[object, ...]:
        return (
            experiment.aco.config_hash,
            experiment.runtime.aco_backend.value,
            tuple(case.batch.instance_ids),
            case.seed,
        )

    def get_cpu(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> torch.Tensor | None:
        cached = self._values.get(self.key(case, experiment))
        if cached is not None:
            return cached
        if self.archive is not None:
            archived = self.archive.lookup(case)
            if archived is not None:
                self._values[self.key(case, experiment)] = archived.detach().cpu()
                return self._values[self.key(case, experiment)]
        return None

    def put(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
        value: torch.Tensor,
    ) -> None:
        self._values[self.key(case, experiment)] = value.detach().cpu()

    def best_length(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> torch.Tensor:
        value = self.get_cpu(case, experiment)
        if value is None:
            result = solve(
                case.batch,
                experiment.aco,
                seed=case.seed,
                backend=experiment.runtime.aco_backend,
            )
            self.put(case, experiment, result.best_length)
            value = self.get_cpu(case, experiment)
            assert value is not None
        return value.to(case.batch.device)


def reference_gap_fitness(
    candidate_lengths: dict[int, list[torch.Tensor]],
    baseline_lengths: dict[int, list[torch.Tensor]],
    references: dict[int, list[torch.Tensor]],
) -> FitnessBreakdown:
    """实现 v0.3 的 scale-balanced absolute reference-gap fitness。"""

    mean_gap: dict[int, float] = {}
    median_gap: dict[int, float] = {}
    baseline_gap: dict[int, float] = {}
    mean_delta: dict[int, float] = {}
    degradation: dict[int, float] = {}
    terms: list[float] = []
    for scale in sorted(candidate_lengths):
        candidate = torch.cat(candidate_lengths[scale])
        baseline = torch.cat(baseline_lengths[scale])
        reference = torch.cat(references[scale])
        candidate_gap = 100.0 * (candidate - reference) / reference
        baseline_scale_gap = 100.0 * (baseline - reference) / reference
        delta = candidate_gap - baseline_scale_gap
        mean_gap[scale] = float(candidate_gap.mean().item())
        median_gap[scale] = float(candidate_gap.median().item())
        baseline_gap[scale] = float(baseline_scale_gap.mean().item())
        mean_delta[scale] = float(delta.mean().item())
        degradation[scale] = float(torch.clamp_min(delta, 0.0).mean().item())
        terms.append(mean_gap[scale])
    return FitnessBreakdown(
        fitness=fmean(terms),
        mean_gap_by_scale=mean_gap,
        median_gap_by_scale=median_gap,
        baseline_gap_by_scale=baseline_gap,
        mean_delta_by_scale=mean_delta,
        degradation_by_scale=degradation,
    )


def baseline_relative_fitness(
    candidate_lengths: dict[int, list[torch.Tensor]],
    baseline_lengths: dict[int, list[torch.Tensor]],
    references: dict[int, list[torch.Tensor]],
    *,
    degradation_penalty: float = 0.0,
) -> FitnessBreakdown:
    """v0.2 API 兼容别名；v0.3 忽略 degradation_penalty。"""

    current = reference_gap_fitness(
        candidate_lengths,
        baseline_lengths,
        references,
    )
    legacy_terms = [
        current.mean_delta_by_scale[scale]
        + degradation_penalty * current.degradation_by_scale[scale]
        for scale in sorted(current.mean_delta_by_scale)
    ]
    return FitnessBreakdown(
        fitness=fmean(legacy_terms),
        mean_gap_by_scale=current.mean_gap_by_scale,
        median_gap_by_scale=current.median_gap_by_scale,
        baseline_gap_by_scale=current.baseline_gap_by_scale,
        mean_delta_by_scale=current.mean_delta_by_scale,
        degradation_by_scale=current.degradation_by_scale,
    )


class IndividualEvaluator:
    """单进程便利 evaluator；正式训练使用持久 ``EvaluationPool``。"""

    def __init__(
        self,
        experiment: ExperimentConfig,
        cases: Sequence[EvaluationCase],
        baseline_cache: BaselineCache,
    ) -> None:
        self.experiment = experiment
        self.cases = tuple(cases)
        self.baseline_cache = baseline_cache
        self.last_breakdown: dict[str, FitnessBreakdown] = {}

    def __call__(self, individual: RMTGPIndividual) -> float:
        baseline_values = tuple(
            self.baseline_cache.best_length(case, self.experiment).detach().cpu()
            for case in self.cases
        )
        breakdown = _score_with_explicit_baselines(
            individual,
            self.experiment,
            self.cases,
            baseline_values,
        )
        self.last_breakdown[individual.structural_hash] = breakdown
        return breakdown.fitness


def _score_with_explicit_baselines(
    individual: RMTGPIndividual,
    experiment: ExperimentConfig,
    cases: Sequence[EvaluationCase],
    baseline_values: Sequence[torch.Tensor],
) -> FitnessBreakdown:
    """在 worker 内评估个体；baseline 已按 paired seed 明确给出。"""

    transition, pheromone = compile_individual(individual)
    candidate: dict[int, list[torch.Tensor]] = {}
    baseline: dict[int, list[torch.Tensor]] = {}
    references: dict[int, list[torch.Tensor]] = {}
    for case, baseline_length in zip(cases, baseline_values, strict=True):
        result = solve(
            case.batch,
            experiment.aco,
            transition_program=transition,
            pheromone_program=pheromone,
            seed=case.seed,
            backend=experiment.runtime.aco_backend,
        )
        candidate.setdefault(case.scale, []).append(result.best_length)
        baseline.setdefault(case.scale, []).append(
            baseline_length.to(case.batch.device)
        )
        references.setdefault(case.scale, []).append(case.batch.reference_length)
    return reference_gap_fitness(
        candidate,
        baseline,
        references,
    )


def _validation_data_with_explicit_baselines(
    individual: RMTGPIndividual,
    experiment: ExperimentConfig,
    cases: Sequence[EvaluationCase],
    baseline_values: Sequence[torch.Tensor],
) -> ValidationData:
    transition, pheromone = compile_individual(individual)
    candidate_gaps: dict[int, list[np.ndarray]] = {}
    baseline_gaps: dict[int, list[np.ndarray]] = {}
    deltas: dict[int, list[np.ndarray]] = {}
    instance_ids: dict[int, list[np.ndarray]] = {}
    for case, baseline in zip(cases, baseline_values, strict=True):
        result = solve(
            case.batch,
            experiment.aco,
            transition_program=transition,
            pheromone_program=pheromone,
            seed=case.seed,
            backend=experiment.runtime.aco_backend,
        )
        reference = case.batch.reference_length
        candidate_gap = 100.0 * (result.best_length - reference) / reference
        baseline_gap = (
            100.0
            * (baseline.to(case.batch.device) - reference)
            / reference
        )
        delta = candidate_gap - baseline_gap
        candidate_gaps.setdefault(case.scale, []).append(
            candidate_gap.detach().cpu().numpy()
        )
        baseline_gaps.setdefault(case.scale, []).append(
            baseline_gap.detach().cpu().numpy()
        )
        deltas.setdefault(case.scale, []).append(delta.detach().cpu().numpy())
        instance_ids.setdefault(case.scale, []).append(
            np.asarray(case.batch.instance_ids, dtype=str)
        )
    return ValidationData(
        candidate_gap_by_scale={
            scale: np.concatenate(values)
            for scale, values in candidate_gaps.items()
        },
        baseline_gap_by_scale={
            scale: np.concatenate(values)
            for scale, values in baseline_gaps.items()
        },
        delta_by_scale={
            scale: np.concatenate(values)
            for scale, values in deltas.items()
        },
        instance_ids_by_scale={
            scale: np.concatenate(values)
            for scale, values in instance_ids.items()
        },
    )


def _batched_population_breakdowns(
    individuals: Sequence[RMTGPIndividual],
    experiment: ExperimentConfig,
    cases: Sequence[EvaluationCase],
    baseline_values: Sequence[torch.Tensor],
) -> tuple[list[FitnessBreakdown], int]:
    """用一个 population×instance Numba 边界计算全部训练 fitness。"""

    from .aco_numba import solve_population_numba

    programs = [compile_individual(individual) for individual in individuals]
    candidate: dict[int, list[torch.Tensor]] = {}
    baseline: dict[int, list[torch.Tensor]] = {}
    references: dict[int, list[torch.Tensor]] = {}
    constructed_tours = 0
    for case, baseline_length in zip(cases, baseline_values, strict=True):
        result = solve_population_numba(
            case.batch,
            experiment.aco,
            programs,
            seed=case.seed,
            threads=experiment.runtime.cpu_threads,
        )
        candidate.setdefault(case.scale, []).append(result.best_length)
        baseline.setdefault(case.scale, []).append(baseline_length.detach().cpu())
        references.setdefault(case.scale, []).append(
            case.batch.reference_length.detach().cpu()
        )
        constructed_tours += result.constructed_tours

    scale_values: dict[
        int,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}
    for scale in sorted(candidate):
        candidate_length = torch.cat(candidate[scale], dim=1)
        baseline_length = torch.cat(baseline[scale]).unsqueeze(0)
        reference = torch.cat(references[scale]).unsqueeze(0)
        candidate_gap = 100.0 * (candidate_length - reference) / reference
        baseline_gap = 100.0 * (baseline_length - reference) / reference
        delta = candidate_gap - baseline_gap
        scale_values[scale] = (
            candidate_gap.mean(dim=1),
            candidate_gap.median(dim=1).values,
            baseline_gap.mean(dim=1).expand(len(individuals)),
            delta.mean(dim=1),
        )

    breakdowns: list[FitnessBreakdown] = []
    for index in range(len(individuals)):
        mean_gap = {
            scale: float(values[0][index].item())
            for scale, values in scale_values.items()
        }
        breakdowns.append(
            FitnessBreakdown(
                fitness=fmean(mean_gap.values()),
                mean_gap_by_scale=mean_gap,
                median_gap_by_scale={
                    scale: float(values[1][index].item())
                    for scale, values in scale_values.items()
                },
                baseline_gap_by_scale={
                    scale: float(values[2][index].item())
                    for scale, values in scale_values.items()
                },
                mean_delta_by_scale={
                    scale: float(values[3][index].item())
                    for scale, values in scale_values.items()
                },
            )
        )
    return breakdowns, constructed_tours


def _batched_validation_data(
    individuals: Sequence[RMTGPIndividual],
    experiment: ExperimentConfig,
    cases: Sequence[EvaluationCase],
    baseline_values: Sequence[torch.Tensor],
) -> dict[str, ValidationData]:
    """批量计算 validation absolute/baseline/delta gaps。"""

    from .aco_numba import solve_population_numba

    programs = [compile_individual(individual) for individual in individuals]
    candidates: dict[int, list[torch.Tensor]] = {}
    baselines: dict[int, list[torch.Tensor]] = {}
    references: dict[int, list[torch.Tensor]] = {}
    ids: dict[int, list[np.ndarray]] = {}
    for case, baseline_length in zip(cases, baseline_values, strict=True):
        result = solve_population_numba(
            case.batch,
            experiment.aco,
            programs,
            seed=case.seed,
            threads=experiment.runtime.cpu_threads,
        )
        candidates.setdefault(case.scale, []).append(result.best_length)
        baselines.setdefault(case.scale, []).append(baseline_length.detach().cpu())
        references.setdefault(case.scale, []).append(
            case.batch.reference_length.detach().cpu()
        )
        ids.setdefault(case.scale, []).append(
            np.asarray(case.batch.instance_ids, dtype=str)
        )

    scale_values: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for scale in sorted(candidates):
        candidate_length = torch.cat(candidates[scale], dim=1)
        baseline_length = torch.cat(baselines[scale]).unsqueeze(0)
        reference = torch.cat(references[scale]).unsqueeze(0)
        candidate_gap = 100.0 * (candidate_length - reference) / reference
        baseline_gap = 100.0 * (baseline_length - reference) / reference
        scale_values[scale] = (
            candidate_gap.numpy(),
            baseline_gap.expand_as(candidate_gap).numpy(),
            (candidate_gap - baseline_gap).numpy(),
            np.concatenate(ids[scale]),
        )

    return {
        individual.structural_hash: ValidationData(
            candidate_gap_by_scale={
                scale: values[0][index].copy()
                for scale, values in scale_values.items()
            },
            baseline_gap_by_scale={
                scale: values[1][index].copy()
                for scale, values in scale_values.items()
            },
            delta_by_scale={
                scale: values[2][index].copy()
                for scale, values in scale_values.items()
            },
            instance_ids_by_scale={
                scale: values[3].copy()
                for scale, values in scale_values.items()
            },
        )
        for index, individual in enumerate(individuals)
    }


def _initialise_evaluation_worker(experiment: ExperimentConfig) -> None:
    """初始化持久 worker，并禁止任何内层线程超额订阅。"""

    global _WORKER_EXPERIMENT
    _WORKER_EXPERIMENT = experiment
    os.environ["OMP_NUM_THREADS"] = str(experiment.runtime.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(experiment.runtime.torch_threads)
    os.environ["NUMBA_NUM_THREADS"] = "1"
    torch.set_num_threads(experiment.runtime.torch_threads)
    torch.use_deterministic_algorithms(
        experiment.runtime.deterministic_algorithms
    )


def _score_chunk_in_worker(
    payload: tuple[
        tuple[RMTGPIndividual, ...],
        tuple[EvaluationCase, ...],
        tuple[torch.Tensor, ...],
    ],
) -> list[FitnessBreakdown]:
    if _WORKER_EXPERIMENT is None:
        raise RuntimeError("评估 worker 尚未初始化")
    individuals, cases, baseline_values = payload
    return [
        _score_with_explicit_baselines(
            individual,
            _WORKER_EXPERIMENT,
            cases,
            baseline_values,
        )
        for individual in individuals
    ]


def _baseline_chunk_in_worker(
    cases: tuple[EvaluationCase, ...],
) -> list[torch.Tensor]:
    if _WORKER_EXPERIMENT is None:
        raise RuntimeError("评估 worker 尚未初始化")
    return [
        solve(
            case.batch,
            _WORKER_EXPERIMENT.aco,
            seed=case.seed,
            backend=_WORKER_EXPERIMENT.runtime.aco_backend,
        ).best_length.detach().cpu()
        for case in cases
    ]


def _validation_chunk_in_worker(
    payload: tuple[
        tuple[RMTGPIndividual, ...],
        tuple[EvaluationCase, ...],
        tuple[torch.Tensor, ...],
    ],
) -> list[tuple[str, ValidationData]]:
    if _WORKER_EXPERIMENT is None:
        raise RuntimeError("评估 worker 尚未初始化")
    individuals, cases, baseline_values = payload
    return [
        (
            individual.structural_hash,
            _validation_data_with_explicit_baselines(
                individual,
                _WORKER_EXPERIMENT,
                cases,
                baseline_values,
            ),
        )
        for individual in individuals
    ]


def _warm_worker(case: EvaluationCase) -> int:
    """触发 worker 的 Numba cache load，并返回 PID 供覆盖审计。"""

    if _WORKER_EXPERIMENT is None:
        raise RuntimeError("评估 worker 尚未初始化")
    solve(
        case.batch,
        _WORKER_EXPERIMENT.aco,
        seed=case.seed,
        backend=_WORKER_EXPERIMENT.runtime.aco_backend,
    )
    # 给 executor 足够时间启动全部 worker，避免一个进程吞掉全部 warm tasks。
    sleep(0.02)
    return os.getpid()


def _chunked(
    items: Sequence[_T],
    chunks: int,
) -> list[tuple[_T, ...]]:
    if not items:
        return []
    count = min(max(chunks, 1), len(items))
    chunk_size = (len(items) + count - 1) // count
    return [
        tuple(items[start : start + chunk_size])
        for start in range(0, len(items), chunk_size)
    ]


class EvaluationPool:
    """跨 generations 复用的确定性个体层进程池。"""

    def __init__(self, experiment: ExperimentConfig) -> None:
        self.experiment = experiment
        self.executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> EvaluationPool:
        if self.experiment.runtime.processes > 1:
            if self.experiment.aco.device != "cpu":
                raise ValueError("多进程评估仅支持 CPU ACO")
            context = mp.get_context(
                self.experiment.runtime.multiprocessing_start_method
            )
            self.executor = ProcessPoolExecutor(
                max_workers=self.experiment.runtime.processes,
                mp_context=context,
                initializer=_initialise_evaluation_worker,
                initargs=(self.experiment,),
            )
        return self

    def __exit__(self, *_: object) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None

    def warm(self, case: EvaluationCase) -> set[int]:
        """在正式计时前加载每个 worker 的已编译 Numba cache。"""

        if (
            self.experiment.runtime.aco_backend
            is ExecutionBackend.NUMBA_BATCH
        ):
            from .aco_numba import solve_population_numba

            solve_population_numba(
                case.batch,
                self.experiment.aco,
                [(None, None)],
                seed=case.seed,
                threads=self.experiment.runtime.cpu_threads,
            )
            return {os.getpid()}
        if (
            self.executor is None
            or self.experiment.runtime.aco_backend is not ExecutionBackend.NUMBA
        ):
            return set()
        expected = self.experiment.runtime.processes
        seen: set[int] = set()
        for _ in range(4):
            futures = [
                self.executor.submit(_warm_worker, case)
                for _ in range(expected * 2)
            ]
            seen.update(future.result() for future in futures)
            if len(seen) >= expected:
                break
        return seen

    def baseline_values(
        self,
        cases: Sequence[EvaluationCase],
        cache: BaselineCache,
        *,
        parallel: bool,
    ) -> tuple[torch.Tensor, ...]:
        missing = [
            case
            for case in cases
            if cache.get_cpu(case, self.experiment) is None
        ]
        if missing:
            if self.executor is not None and parallel:
                case_chunks = _chunked(
                    missing,
                    self.experiment.runtime.processes,
                )
                chunk_results = list(
                    self.executor.map(
                        _baseline_chunk_in_worker,
                        case_chunks,
                        chunksize=1,
                    )
                )
                values = [
                    value
                    for chunk in chunk_results
                    for value in chunk
                ]
            else:
                values = [
                    solve(
                        case.batch,
                        self.experiment.aco,
                        seed=case.seed,
                        backend=self.experiment.runtime.aco_backend,
                    ).best_length.detach().cpu()
                    for case in missing
                ]
            for case, value in zip(missing, values, strict=True):
                cache.put(case, self.experiment, value)
        result: list[torch.Tensor] = []
        for case in cases:
            value = cache.get_cpu(case, self.experiment)
            assert value is not None
            result.append(value)
        return tuple(result)

    def evaluate_population(
        self,
        population: Sequence[RMTGPIndividual],
        cases: Sequence[EvaluationCase],
        baseline_cache: BaselineCache,
    ) -> PopulationEvaluationResult:
        representatives: dict[str, RMTGPIndividual] = {}
        waiting: dict[str, list[RMTGPIndividual]] = {}
        for individual in population:
            if individual.fitness.valid:
                continue
            key = individual.structural_hash
            representatives.setdefault(key, individual)
            waiting.setdefault(key, []).append(individual)
        if not representatives:
            return PopulationEvaluationResult(0, {}, 0.0, 0.0)

        baseline_started = perf_counter()
        baseline_values = self.baseline_values(
            cases,
            baseline_cache,
            parallel=False,
        )
        baseline_elapsed = perf_counter() - baseline_started
        ordered_keys = list(representatives)
        individuals = [representatives[key] for key in ordered_keys]

        evaluation_started = perf_counter()
        constructed_tours = 0
        if (
            self.experiment.runtime.aco_backend
            is ExecutionBackend.NUMBA_BATCH
        ):
            breakdowns, constructed_tours = _batched_population_breakdowns(
                individuals,
                self.experiment,
                cases,
                baseline_values,
            )
        elif self.executor is None:
            breakdowns = [
                _score_with_explicit_baselines(
                    individual,
                    self.experiment,
                    cases,
                    baseline_values,
                )
                for individual in individuals
            ]
        else:
            individual_chunks = _chunked(
                individuals,
                self.experiment.runtime.processes,
            )
            payloads = [
                (chunk, tuple(cases), baseline_values)
                for chunk in individual_chunks
            ]
            chunk_results = list(
                self.executor.map(
                    _score_chunk_in_worker,
                    payloads,
                    chunksize=1,
                )
            )
            breakdowns = [
                breakdown
                for chunk in chunk_results
                for breakdown in chunk
            ]
        if constructed_tours == 0:
            constructed_tours = len(individuals) * sum(
                case.batch.batch_size
                * self.experiment.aco.resolve_ants(case.batch.n)
                * self.experiment.aco.iterations
                for case in cases
            )
        evaluation_elapsed = perf_counter() - evaluation_started

        by_hash: dict[str, FitnessBreakdown] = {}
        for key, breakdown in zip(ordered_keys, breakdowns, strict=True):
            by_hash[key] = breakdown
            for individual in waiting[key]:
                individual.fitness.values = (float(breakdown.fitness),)
                individual.metadata["fitness_breakdown"] = breakdown
        return PopulationEvaluationResult(
            evaluated_unique=len(ordered_keys),
            breakdowns=by_hash,
            baseline_wall_time=baseline_elapsed,
            evaluation_wall_time=evaluation_elapsed,
            constructed_tours=constructed_tours,
        )

    def validation_data(
        self,
        candidates: Sequence[RMTGPIndividual],
        cases: Sequence[EvaluationCase],
        baseline_cache: BaselineCache,
    ) -> dict[str, ValidationData]:
        baseline_values = self.baseline_values(
            cases,
            baseline_cache,
            parallel=True,
        )
        if (
            self.experiment.runtime.aco_backend
            is ExecutionBackend.NUMBA_BATCH
        ):
            return _batched_validation_data(
                candidates,
                self.experiment,
                cases,
                baseline_values,
            )
        if self.executor is None:
            return {
                individual.structural_hash: _validation_data_with_explicit_baselines(
                    individual,
                    self.experiment,
                    cases,
                    baseline_values,
                )
                for individual in candidates
            }
        candidate_chunks = _chunked(
            candidates,
            self.experiment.runtime.processes * 2,
        )
        payloads = [
            (chunk, tuple(cases), baseline_values)
            for chunk in candidate_chunks
        ]
        results = list(
            self.executor.map(
                _validation_chunk_in_worker,
                payloads,
                chunksize=1,
            )
        )
        return {
            key: arrays
            for chunk in results
            for key, arrays in chunk
        }


def evaluate_invalid_population(
    population: Sequence[RMTGPIndividual],
    experiment: ExperimentConfig,
    cases: Sequence[EvaluationCase],
    baseline_cache: BaselineCache,
) -> int:
    """兼容公共接口；一次性调用仍使用相同的并行实现。"""

    with EvaluationPool(experiment) as evaluator:
        return evaluator.evaluate_population(
            population,
            cases,
            baseline_cache,
        ).evaluated_unique


def _normal_upper_bound(values: np.ndarray) -> float:
    if values.size <= 1:
        return float(values.mean())
    standard_error = values.std(ddof=1) / np.sqrt(values.size)
    return float(values.mean() + 1.645 * standard_error)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, values.size, size=values.size)
        means[replicate] = values[indices].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _aggregate_by_instance(
    values: np.ndarray,
    instance_ids: np.ndarray,
) -> np.ndarray:
    """先聚合同一 TSP instance 的 ACO seeds，避免伪重复。"""

    if values.shape != instance_ids.shape:
        raise ValueError("validation values 与 instance IDs shape 不一致")
    grouped: dict[str, list[float]] = {}
    for value, instance_id in zip(values, instance_ids, strict=True):
        grouped.setdefault(str(instance_id), []).append(float(value))
    return np.asarray(
        [fmean(grouped[key]) for key in sorted(grouped)],
        dtype=np.float64,
    )


def validate_candidates(
    candidates: Iterable[RMTGPIndividual],
    experiment: ExperimentConfig,
    validation_cases: Sequence[EvaluationCase],
    baseline_cache: BaselineCache,
    *,
    screening_cases: Sequence[EvaluationCase] | None = None,
    gate_cases: Sequence[EvaluationCase] | None = None,
    evaluator_pool: EvaluationPool | None = None,
) -> ValidationSelection:
    """两阶段选 champion，再在独立 gate 上执行 non-inferiority。"""

    unique = {
        individual.structural_hash: individual
        for individual in candidates
    }
    if not unique:
        raise ValueError("validation candidates 不能为空")
    if evaluator_pool is None:
        with EvaluationPool(experiment) as temporary:
            return validate_candidates(
                unique.values(),
                experiment,
                validation_cases,
                baseline_cache,
                screening_cases=screening_cases,
                gate_cases=gate_cases,
                evaluator_pool=temporary,
            )

    screening_cases = (
        validation_cases if screening_cases is None else screening_cases
    )
    gate_cases = validation_cases if gate_cases is None else gate_cases
    started = perf_counter()
    all_candidates = list(unique.values())
    screening_data = evaluator_pool.validation_data(
        all_candidates,
        screening_cases,
        baseline_cache,
    )
    screening_scored: list[tuple[float, int, RMTGPIndividual]] = []
    for key, individual in unique.items():
        data = screening_data[key]
        macro = fmean(
            float(values.mean())
            for values in data.candidate_gap_by_scale.values()
        )
        screening_scored.append((macro, individual.total_nodes, individual))
    screening_scored.sort(key=lambda item: (item[0], item[1]))
    finalists = [
        item[2]
        for item in screening_scored[: min(experiment.validation_top_k, len(unique))]
    ]

    if screening_cases is validation_cases:
        selection_data = {
            individual.structural_hash: screening_data[individual.structural_hash]
            for individual in finalists
        }
    else:
        selection_data = evaluator_pool.validation_data(
            finalists,
            validation_cases,
            baseline_cache,
        )

    scored: list[tuple[float, int, RMTGPIndividual, ValidationData]] = []
    for individual in finalists:
        data = selection_data[individual.structural_hash]
        macro = fmean(
            float(values.mean())
            for values in data.candidate_gap_by_scale.values()
        )
        scored.append((macro, individual.total_nodes, individual, data))
    scored.sort(key=lambda item: (item[0], item[1]))
    best_macro = scored[0][0]
    near_ties = [item for item in scored if item[0] <= best_macro + 0.01]
    macro_gap, _, selected, selected_data = min(
        near_ties,
        key=lambda item: (item[1], item[0]),
    )

    if gate_cases is validation_cases:
        gate_data = selected_data
    else:
        gate_data = evaluator_pool.validation_data(
            [selected],
            gate_cases,
            baseline_cache,
        )[selected.structural_hash]

    instance_deltas = {
        scale: _aggregate_by_instance(
            values,
            gate_data.instance_ids_by_scale[scale],
        )
        for scale, values in gate_data.delta_by_scale.items()
    }
    passed = all(
        _normal_upper_bound(values) <= experiment.noninferiority_tolerance
        for values in instance_deltas.values()
    )

    scale_summaries: list[ValidationScaleSummary] = []
    for scale, values in sorted(instance_deltas.items()):
        ci_low, ci_high = _bootstrap_mean_ci(
            values,
            seed=int(
                np.random.default_rng(
                    np.random.SeedSequence(
                        [experiment.root_seed, scale, 0x424F4F54]
                    )
                ).integers(0, 2**63 - 1)
            ),
        )
        scale_summaries.append(
            ValidationScaleSummary(
                scale=scale,
                observations=int(gate_data.delta_by_scale[scale].size),
                instances=int(values.size),
                mean_delta_pp=float(values.mean()),
                median_delta_pp=float(np.median(values)),
                bootstrap_ci_low=ci_low,
                bootstrap_ci_high=ci_high,
                normal_upper_bound_95=_normal_upper_bound(values),
                wins=int(np.sum(values < -1e-12)),
                ties=int(np.sum(np.abs(values) <= 1e-12)),
                losses=int(np.sum(values > 1e-12)),
            )
        )

    if passed:
        champion = selected.clone()
    else:
        transition_pset, pheromone_pset = create_primitive_sets(
            transition_profile=experiment.gp.transition_profile,
            function_profile=experiment.gp.function_profile,
            transition_terminals=experiment.gp.transition_terminals,
            pheromone_terminals=experiment.gp.pheromone_terminals,
        )
        champion = make_individual(
            transition_pset,
            pheromone_pset,
            experiment.gp,
            mode="baseline",
        )
        champion.fitness.values = (0.0,)
    return ValidationSelection(
        champion=champion,
        passed_noninferiority=passed,
        selected_candidate_hash=selected.structural_hash,
        selected_macro_gap_percent=float(macro_gap),
        selected_macro_delta_pp=fmean(
            float(values.mean())
            for values in selected_data.delta_by_scale.values()
        ),
        screened_candidates=len(unique),
        finalist_candidates=len(finalists),
        unique_candidates=len(unique),
        wall_time_sec=perf_counter() - started,
        scales=scale_summaries,
    )


def _generation_record(
    generation: int,
    population: Sequence[RMTGPIndividual],
    evaluation: PopulationEvaluationResult,
    *,
    breeding_wall_time: float,
    cumulative_wall_time: float,
    generation_wall_time: float,
    eta_seconds: float,
) -> GenerationRecord:
    values = np.asarray(
        [item.fitness.values[0] for item in population],
        dtype=float,
    )
    best = min(
        population,
        key=lambda item: (item.fitness.values[0], item.total_nodes),
    )
    breakdown = evaluation.breakdowns.get(best.structural_hash)
    if breakdown is None:
        stored = best.metadata.get("fitness_breakdown")
        if isinstance(stored, FitnessBreakdown):
            breakdown = stored
    return GenerationRecord(
        generation=generation,
        evaluated_unique=evaluation.evaluated_unique,
        unique_genotypes=len({item.structural_hash for item in population}),
        minimum=float(values.min()),
        first_quartile=float(np.quantile(values, 0.25)),
        median=float(np.median(values)),
        mean=float(values.mean()),
        third_quartile=float(np.quantile(values, 0.75)),
        standard_deviation=float(values.std()),
        best_nodes=best.total_nodes,
        best_transition_nodes=best.transition_nodes,
        best_pheromone_nodes=best.pheromone_nodes,
        best_hash=best.structural_hash,
        best_transition_expression=str(best.transition_tree),
        best_pheromone_expression=str(best.pheromone_tree),
        best_mean_gap_by_scale=(
            {} if breakdown is None else dict(breakdown.mean_gap_by_scale)
        ),
        best_median_gap_by_scale=(
            {} if breakdown is None else dict(breakdown.median_gap_by_scale)
        ),
        baseline_mean_gap_by_scale=(
            {} if breakdown is None else dict(breakdown.baseline_gap_by_scale)
        ),
        best_mean_delta_by_scale=(
            {} if breakdown is None else dict(breakdown.mean_delta_by_scale)
        ),
        baseline_wall_time=evaluation.baseline_wall_time,
        evaluation_wall_time=evaluation.evaluation_wall_time,
        breeding_wall_time=breeding_wall_time,
        checkpoint_wall_time=0.0,
        generation_wall_time=generation_wall_time,
        cumulative_wall_time=cumulative_wall_time,
        unique_individuals_per_second=(
            evaluation.evaluated_unique
            / max(evaluation.evaluation_wall_time, 1e-12)
        ),
        constructed_tours=evaluation.constructed_tours,
        tours_per_second=(
            evaluation.constructed_tours
            / max(evaluation.evaluation_wall_time, 1e-12)
        ),
        eta_seconds=eta_seconds,
    )


def _environment_payload() -> dict[str, object]:
    try:
        import deap

        deap_version = deap.__version__
    except AttributeError:
        deap_version = "unknown"
    try:
        import llvmlite
        import numba

        numba_version = numba.__version__
        llvmlite_version = llvmlite.__version__
    except (ImportError, ModuleNotFoundError):
        numba_version = "unavailable"
        llvmlite_version = "unavailable"
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "deap": deap_version,
        "numba": numba_version,
        "llvmlite": llvmlite_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_incremental_metrics(
    history: Sequence[GenerationRecord],
    target: Path,
) -> None:
    payload = [asdict(record) for record in history]
    _atomic_write_text(
        target / "training_metrics.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    _atomic_write_text(
        target / "training_metrics.jsonl",
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in payload
        ),
    )


def _experiment_hash(experiment: ExperimentConfig) -> str:
    payload = json.dumps(
        experiment.stable_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _sampler_owner(
    provider: Callable[[int], Sequence[EvaluationCase]],
) -> object | None:
    owner = getattr(provider, "__self__", None)
    if (
        owner is not None
        and hasattr(owner, "state_dict")
        and hasattr(owner, "load_state_dict")
    ):
        return owner
    return None


def _save_resume_checkpoint(
    target: Path,
    *,
    experiment: ExperimentConfig,
    completed_generation: int,
    population: Sequence[RMTGPIndividual],
    checkpoints: Sequence[RMTGPIndividual],
    history: Sequence[GenerationRecord],
    sampler_owner: object | None,
) -> None:
    sampler_state = (
        sampler_owner.state_dict()  # type: ignore[attr-defined]
        if sampler_owner is not None
        else None
    )
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "experiment_hash": _experiment_hash(experiment),
        "completed_generation": completed_generation,
        "population": list(population),
        "checkpoints": list(checkpoints),
        "history": list(history),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "sampler_state": sampler_state,
    }
    temporary = target / "training_state.pkl.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target / "training_state.pkl")


def _load_resume_checkpoint(
    resume_from: str | Path,
    *,
    experiment: ExperimentConfig,
    sampler_owner: object | None,
) -> dict[str, object]:
    source = Path(resume_from)
    if source.is_dir():
        source = source / "training_state.pkl"
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("training checkpoint 根对象必须为 dict")
    if payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("training checkpoint schema 不兼容")
    if payload.get("experiment_hash") != _experiment_hash(experiment):
        raise ValueError("resume 配置与 checkpoint 不一致")
    sampler_state = payload.get("sampler_state")
    if sampler_state is not None:
        if sampler_owner is None:
            raise ValueError("当前 training case provider 不支持恢复 sampler")
        sampler_owner.load_state_dict(sampler_state)  # type: ignore[attr-defined]
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    return payload


def _evolution_summary(result: TrainingResult) -> str:
    lines = [
        "# 训练进化摘要",
        "",
        "| 代 | 用时(s) | 累计(s) | 最优 gap% | 均值 | 中位数 | 节点 | candidate/base/Δ(pp) |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in result.history:
        scale_delta = ", ".join(
            (
                f"TSP{scale}="
                f"{record.best_mean_gap_by_scale[scale]:.4f}/"
                f"{record.baseline_mean_gap_by_scale[scale]:.4f}/"
                f"{value:+.4f}"
            )
            for scale, value in sorted(record.best_mean_delta_by_scale.items())
        )
        lines.append(
            f"| {record.generation} | {record.generation_wall_time:.3f} | "
            f"{record.cumulative_wall_time:.3f} | {record.minimum:.6f} | "
            f"{record.mean:.6f} | {record.median:.6f} | {record.best_nodes} | "
            f"{scale_delta} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- 唯一候选数：{result.validation.unique_candidates}",
            f"- Finalists：{result.validation.finalist_candidates}",
            (
                "- 选中候选 macro reference gap："
                f"{result.validation.selected_macro_gap_percent:.6f}%"
            ),
            f"- 选中候选 macro Δ：{result.validation.selected_macro_delta_pp:+.6f} pp",
            (
                "- Non-inferiority："
                + ("通过" if result.passed_noninferiority else "失败，部署 baseline fallback")
            ),
        ]
    )
    for summary in result.validation.scales:
        lines.append(
            f"- TSP{summary.scale}: mean={summary.mean_delta_pp:+.6f} pp, "
            f"95% CI=[{summary.bootstrap_ci_low:+.6f}, "
            f"{summary.bootstrap_ci_high:+.6f}], "
            f"W/T/L={summary.wins}/{summary.ties}/{summary.losses}"
        )
    return "\n".join(lines) + "\n"


def save_training_result(
    result: TrainingResult,
    experiment: ExperimentConfig,
    output_directory: str | Path,
) -> Path:
    """保存可复现 artifact、validation 统计与全部候选。"""

    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        target / "config.yaml",
        yaml.safe_dump(
            experiment.stable_dict(),
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    _atomic_write_text(
        target / "environment.json",
        json.dumps(_environment_payload(), ensure_ascii=False, indent=2),
    )
    _write_incremental_metrics(result.history, target)
    _atomic_write_text(
        target / "champion_expression.txt",
        (
            f"transition: {result.champion.transition_tree}\n"
            f"pheromone: {result.champion.pheromone_tree}\n"
            f"passed_noninferiority: {result.passed_noninferiority}\n"
            f"selected_candidate_hash: "
            f"{result.validation.selected_candidate_hash}\n"
        ),
    )
    with (target / "champion.pkl").open("wb") as handle:
        pickle.dump(result.champion, handle, protocol=pickle.HIGHEST_PROTOCOL)

    validation_lines = [
        (
            "scale,observations,instances,mean_delta_pp,median_delta_pp,"
            "bootstrap_ci_low,bootstrap_ci_high,normal_upper_bound_95,"
            "wins,ties,losses,passed_noninferiority\n"
        )
    ]
    for summary in result.validation.scales:
        validation_lines.append(
            f"{summary.scale},{summary.observations},{summary.instances},"
            f"{summary.mean_delta_pp:.17g},{summary.median_delta_pp:.17g},"
            f"{summary.bootstrap_ci_low:.17g},{summary.bootstrap_ci_high:.17g},"
            f"{summary.normal_upper_bound_95:.17g},{summary.wins},"
            f"{summary.ties},{summary.losses},"
            f"{str(result.passed_noninferiority).lower()}\n"
        )
    _atomic_write_text(
        target / "validation_summary.csv",
        "".join(validation_lines),
    )
    _atomic_write_text(
        target / "evolution_summary.md",
        _evolution_summary(result),
    )

    checkpoint_directory = target / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    for index, checkpoint in enumerate(result.checkpoints):
        with (checkpoint_directory / f"candidate_{index:04d}.pkl").open(
            "wb"
        ) as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
    result.output_directory = target
    return target


def train(
    experiment: ExperimentConfig,
    training_cases_for_generation: Callable[[int], Sequence[EvaluationCase]],
    validation_cases: Sequence[EvaluationCase],
    *,
    validation_screening_cases: Sequence[EvaluationCase] | None = None,
    validation_gate_cases: Sequence[EvaluationCase] | None = None,
    baseline_archive: BaselineArchive | None = None,
    output_directory: str | Path | None = None,
    progress_callback: Callable[[GenerationRecord], None] | None = None,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """执行可逐代恢复的 Strongly Typed Multi-Tree GP run。"""

    target = Path(output_directory) if output_directory is not None else None
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
    sampler_owner = _sampler_owner(training_cases_for_generation)

    if resume_from is None:
        random.seed(experiment.root_seed)
        np.random.seed(experiment.root_seed % (2**32))
        torch.manual_seed(experiment.root_seed)
        (
            population,
            transition_pset,
            pheromone_pset,
        ) = initialise_population(experiment.gp)
        history: list[GenerationRecord] = []
        checkpoints: list[RMTGPIndividual] = []
        completed_generation = 0
    else:
        transition_pset, pheromone_pset = create_primitive_sets(
            transition_profile=experiment.gp.transition_profile,
            function_profile=experiment.gp.function_profile,
            transition_terminals=experiment.gp.transition_terminals,
            pheromone_terminals=experiment.gp.pheromone_terminals,
        )
        state = _load_resume_checkpoint(
            resume_from,
            experiment=experiment,
            sampler_owner=sampler_owner,
        )
        population = list(state["population"])
        checkpoints = list(state["checkpoints"])
        history = list(state["history"])
        completed_generation = int(state["completed_generation"])

    baseline_cache = BaselineCache(baseline_archive)
    first_generation = completed_generation + 1
    pending_cases: Sequence[EvaluationCase] | None = None
    if first_generation <= experiment.gp.generations:
        pending_cases = training_cases_for_generation(first_generation)
        warm_case = pending_cases[0]
    elif validation_cases:
        warm_case = validation_cases[0]
    else:
        raise ValueError("training 与 validation cases 不能同时为空")

    # 主进程先生成 Numba disk cache；随后 worker 只需加载，不计入每代时间。
    if experiment.runtime.aco_backend in {
        ExecutionBackend.NUMBA,
        ExecutionBackend.NUMBA_BATCH,
    }:
        baseline_cache.best_length(warm_case, experiment)

    cumulative = history[-1].cumulative_wall_time if history else 0.0
    with EvaluationPool(experiment) as evaluator:
        evaluator.warm(warm_case)
        for generation in range(
            first_generation,
            experiment.gp.generations + 1,
        ):
            generation_started = perf_counter()
            if generation == first_generation:
                assert pending_cases is not None
                cases = pending_cases
            else:
                cases = training_cases_for_generation(generation)

            for individual in population:
                if individual.fitness.valid:
                    del individual.fitness.values
            evaluation = evaluator.evaluate_population(
                population,
                cases,
                baseline_cache,
            )

            if generation % experiment.gp.checkpoint_interval == 0:
                selected = tools.selBest(
                    population,
                    experiment.gp.checkpoint_top_k,
                )
                checkpoints.extend(deepcopy(selected))

            before_breeding = perf_counter() - generation_started
            estimated_average = (
                (cumulative + before_breeding) / generation
            )
            record = _generation_record(
                generation,
                population,
                evaluation,
                breeding_wall_time=0.0,
                cumulative_wall_time=cumulative + before_breeding,
                generation_wall_time=before_breeding,
                eta_seconds=estimated_average
                * (experiment.gp.generations - generation),
            )

            breeding_started = perf_counter()
            if generation < experiment.gp.generations:
                next_population = evolve_generation(
                    population,
                    transition_pset,
                    pheromone_pset,
                    experiment.gp,
                )
            else:
                next_population = population
            breeding_elapsed = perf_counter() - breeding_started
            record.breeding_wall_time = breeding_elapsed
            record.generation_wall_time = perf_counter() - generation_started
            record.cumulative_wall_time = (
                cumulative + record.generation_wall_time
            )
            history.append(record)
            population = next_population

            if target is not None:
                checkpoint_started = perf_counter()
                _save_resume_checkpoint(
                    target,
                    experiment=experiment,
                    completed_generation=generation,
                    population=population,
                    checkpoints=checkpoints,
                    history=history,
                    sampler_owner=sampler_owner,
                )
                checkpoint_elapsed = perf_counter() - checkpoint_started
                record.checkpoint_wall_time = checkpoint_elapsed
                record.generation_wall_time = (
                    perf_counter() - generation_started
                )
                record.cumulative_wall_time = (
                    cumulative + record.generation_wall_time
                )
                record.eta_seconds = (
                    record.cumulative_wall_time
                    / generation
                    * (experiment.gp.generations - generation)
                )
                # 第二次原子写入使 checkpoint 内的 timing record 也完整。
                _save_resume_checkpoint(
                    target,
                    experiment=experiment,
                    completed_generation=generation,
                    population=population,
                    checkpoints=checkpoints,
                    history=history,
                    sampler_owner=sampler_owner,
                )
                _write_incremental_metrics(history, target)
            cumulative = record.cumulative_wall_time
            if progress_callback is not None:
                progress_callback(record)

        checkpoints.extend(
            deepcopy(
                tools.selBest(
                    population,
                    experiment.gp.checkpoint_top_k,
                )
            )
        )
        validation = validate_candidates(
            checkpoints,
            experiment,
            validation_cases,
            baseline_cache,
            screening_cases=validation_screening_cases,
            gate_cases=validation_gate_cases,
            evaluator_pool=evaluator,
        )

    result = TrainingResult(
        champion=validation.champion,
        history=history,
        checkpoints=checkpoints,
        passed_noninferiority=validation.passed_noninferiority,
        validation=validation,
    )
    if target is not None:
        save_training_result(result, experiment, target)
    return result
