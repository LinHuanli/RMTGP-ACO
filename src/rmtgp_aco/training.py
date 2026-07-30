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
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
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
from .baseline import BaselineArchive, backend_semantic_id
from .config import (
    ExecutionBackend,
    ExperimentConfig,
    FitnessMode,
    GPUMode,
    SelectionMode,
)
from .genetic import (
    RMTGPIndividual,
    compile_individual,
    evolve_generation,
    initialise_population,
    is_baseline_individual,
    make_individual,
)
from .program import create_primitive_sets
from .sampling import EvaluationCase

_WORKER_EXPERIMENT: ExperimentConfig | None = None
_CHECKPOINT_SCHEMA_VERSION = 5
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
    standard_error_by_scale: dict[int, float] = field(default_factory=dict)
    nonzero_fraction_by_scale: dict[int, float] = field(default_factory=dict)
    wins_by_scale: dict[int, int] = field(default_factory=dict)
    ties_by_scale: dict[int, int] = field(default_factory=dict)
    losses_by_scale: dict[int, int] = field(default_factory=dict)
    mean_basin_gap_by_scale: dict[int, float] = field(default_factory=dict)
    baseline_basin_gap_by_scale: dict[int, float] = field(default_factory=dict)
    mean_basin_delta_by_scale: dict[int, float] = field(default_factory=dict)
    mean_anytime_gap_by_scale: dict[int, float] = field(default_factory=dict)
    baseline_anytime_gap_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    mean_anytime_delta_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    fitness_delta_by_scale: dict[int, float] = field(default_factory=dict)


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
    racing_screen_evaluated_unique: int = 0
    racing_high_evaluated_unique: int = 0
    racing_screen_iterations: int = 0
    racing_high_iterations: int = 0
    racing_screen_instances: int = 0
    racing_high_instances: int = 0
    racing_finalist_hashes: tuple[str, ...] = ()
    racing_screen_fitness_by_hash: dict[str, float] = field(
        default_factory=dict
    )
    racing_high_fitness_by_hash: dict[str, float] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class BaselineMeasurement:
    """一个 case 的原始 ACO final、basin 与 anytime 统计（均在 CPU）。"""

    best_length: torch.Tensor
    basin_mean_length: torch.Tensor | None = None
    anytime_mean_length: torch.Tensor | None = None


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
    validation_monitor_candidate_gap_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    validation_monitor_baseline_gap_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    validation_monitor_delta_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    validation_monitor_wall_time: float = 0.0
    best_standard_error_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    best_nonzero_fraction_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    best_wins_by_scale: dict[int, int] = field(default_factory=dict)
    best_ties_by_scale: dict[int, int] = field(default_factory=dict)
    best_losses_by_scale: dict[int, int] = field(default_factory=dict)
    baseline_anchor_count: int = 0
    training_aco_iterations: int = 0
    best_mean_basin_gap_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    baseline_mean_basin_gap_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    best_mean_basin_delta_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    best_mean_anytime_gap_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    baseline_mean_anytime_gap_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    best_mean_anytime_delta_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    best_fitness_delta_by_scale: dict[int, float] = field(
        default_factory=dict
    )
    racing_enabled: bool = False
    racing_screen_evaluated_unique: int = 0
    racing_high_evaluated_unique: int = 0
    racing_screen_iterations: int = 0
    racing_high_iterations: int = 0
    racing_screen_instances: int = 0
    racing_high_instances: int = 0
    racing_finalist_hashes: tuple[str, ...] = ()
    racing_screen_fitness_by_hash: dict[str, float] = field(
        default_factory=dict
    )
    racing_high_fitness_by_hash: dict[str, float] = field(
        default_factory=dict
    )


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
    candidate_mean_gap_percent: float = float("nan")
    baseline_mean_gap_percent: float = float("nan")
    relative_improvement: float = float("nan")
    bootstrap_upper_bound: float = float("nan")


@dataclass(slots=True)
class ValidationSelection:
    """validation 模型选择、门控和候选审计信息。"""

    champion: RMTGPIndividual
    selected_candidate: RMTGPIndividual
    backend: ExecutionBackend
    passed_noninferiority: bool
    selected_candidate_hash: str
    selected_macro_gap_percent: float
    selected_macro_delta_pp: float
    screened_candidates: int
    finalist_candidates: int
    unique_candidates: int
    wall_time_sec: float
    scales: list[ValidationScaleSummary]
    selection_score: float = float("nan")
    selection_mode: str = SelectionMode.LEGACY_NONINFERIORITY.value


@dataclass(slots=True)
class TrainingResult:
    """一次独立 GP run 的返回值。"""

    champion: RMTGPIndividual
    history: list[GenerationRecord]
    checkpoints: list[RMTGPIndividual]
    passed_noninferiority: bool
    validation: ValidationSelection
    cpu_fp64_audit: ValidationSelection | None = None
    output_directory: Path | None = None


class BaselineCache:
    """按配置、后端、实例 IDs 和 seed 缓存原始 ACO 结果。"""

    def __init__(self, archive: BaselineArchive | None = None) -> None:
        self._values: dict[tuple[object, ...], BaselineMeasurement] = {}
        self.archive = archive

    @staticmethod
    def key(
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> tuple[object, ...]:
        return (
            experiment.aco.baseline_behavior_hash,
            backend_semantic_id(
                experiment.runtime.aco_backend,
                experiment.runtime,
            ),
            tuple(case.batch.instance_ids),
            case.seed,
            (
                experiment.gp.basin_top_q
                if experiment.gp.fitness_mode.uses_basin
                else 0
            ),
            experiment.gp.fitness_mode.uses_anytime,
        )

    def get_measurement_cpu(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> BaselineMeasurement | None:
        cached = self._values.get(self.key(case, experiment))
        if cached is not None:
            return cached
        if (
            self.archive is not None
            and self.archive.config.baseline_behavior_hash
            == experiment.aco.baseline_behavior_hash
        ):
            archived = self.archive.lookup(case)
            if archived is not None:
                basin = None
                if experiment.gp.fitness_mode.uses_basin:
                    basin = self.archive.lookup_basin_mean_length(
                        case,
                        top_q=experiment.gp.basin_top_q,
                    )
                    if basin is None:
                        return None
                anytime = None
                if experiment.gp.fitness_mode.uses_anytime:
                    anytime = self.archive.lookup_anytime_mean_length(case)
                    if anytime is None:
                        return None
                self._values[self.key(case, experiment)] = BaselineMeasurement(
                    best_length=archived.detach().cpu(),
                    basin_mean_length=(
                        None if basin is None else basin.detach().cpu()
                    ),
                    anytime_mean_length=(
                        None if anytime is None else anytime.detach().cpu()
                    ),
                )
                return self._values[self.key(case, experiment)]
        return None

    def get_cpu(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> torch.Tensor | None:
        """兼容旧调用方，只返回 final best length。"""

        measurement = self.get_measurement_cpu(case, experiment)
        return None if measurement is None else measurement.best_length

    def put(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
        value: torch.Tensor,
        basin_mean_length: torch.Tensor | None = None,
        anytime_mean_length: torch.Tensor | None = None,
    ) -> None:
        self._values[self.key(case, experiment)] = BaselineMeasurement(
            best_length=value.detach().cpu(),
            basin_mean_length=(
                None
                if basin_mean_length is None
                else basin_mean_length.detach().cpu()
            ),
            anytime_mean_length=(
                None
                if anytime_mean_length is None
                else anytime_mean_length.detach().cpu()
            ),
        )

    def measurement(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> BaselineMeasurement:
        value = self.get_measurement_cpu(case, experiment)
        if value is None:
            if (
                experiment.gp.fitness_mode.uses_basin
                or experiment.gp.fitness_mode.uses_anytime
            ):
                programs = [(None, None)]
                if experiment.runtime.aco_backend in {
                    ExecutionBackend.CUDA_FUSED_FP32,
                    ExecutionBackend.CUDA_TILED_V2,
                }:
                    from .aco_cuda import solve_population_cuda

                    result = solve_population_cuda(
                        case.batch,
                        experiment.aco,
                        programs,
                        seed=case.seed,
                        runtime=experiment.runtime,
                        basin_top_q=(
                            experiment.gp.basin_top_q
                            if experiment.gp.fitness_mode.uses_basin
                            else 0
                        ),
                    )
                elif (
                    experiment.runtime.aco_backend
                    is ExecutionBackend.NUMBA_BATCH
                ):
                    from .aco_numba import solve_population_numba

                    result = solve_population_numba(
                        case.batch,
                        experiment.aco,
                        programs,
                        seed=case.seed,
                        threads=experiment.runtime.cpu_threads,
                        basin_top_q=(
                            experiment.gp.basin_top_q
                            if experiment.gp.fitness_mode.uses_basin
                            else 0
                        ),
                    )
                else:
                    raise ValueError(
                        "basin/anytime fitness 要求 numba_batch 或 "
                        "CUDA population 后端"
                    )
                if (
                    experiment.gp.fitness_mode.uses_basin
                    and result.basin_mean_length is None
                ):
                    raise RuntimeError("population 后端未返回 basin_mean_length")
                if (
                    experiment.gp.fitness_mode.uses_anytime
                    and result.anytime_mean_length is None
                ):
                    raise RuntimeError(
                        "population 后端未返回 anytime_mean_length"
                    )
                self.put(
                    case,
                    experiment,
                    result.best_length[0],
                    (
                        result.basin_mean_length[0]
                        if result.basin_mean_length is not None
                        else None
                    ),
                    (
                        result.anytime_mean_length[0]
                        if result.anytime_mean_length is not None
                        else None
                    ),
                )
            else:
                result = solve(
                    case.batch,
                    experiment.aco,
                    seed=case.seed,
                    backend=experiment.runtime.aco_backend,
                    runtime=experiment.runtime,
                )
                self.put(case, experiment, result.best_length)
            value = self.get_measurement_cpu(case, experiment)
            assert value is not None
        return value

    def best_length(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> torch.Tensor:
        return self.measurement(
            case,
            experiment,
        ).best_length.to(case.batch.device)


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
    standard_error: dict[int, float] = {}
    nonzero_fraction: dict[int, float] = {}
    wins: dict[int, int] = {}
    ties: dict[int, int] = {}
    losses: dict[int, int] = {}
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
        standard_error[scale] = (
            float(delta.std(unbiased=True).item() / np.sqrt(delta.numel()))
            if delta.numel() > 1
            else 0.0
        )
        wins[scale] = int(torch.sum(delta < -1e-12).item())
        ties[scale] = int(torch.sum(torch.abs(delta) <= 1e-12).item())
        losses[scale] = int(torch.sum(delta > 1e-12).item())
        nonzero_fraction[scale] = (
            float((wins[scale] + losses[scale]) / delta.numel())
        )
        terms.append(mean_gap[scale])
    return FitnessBreakdown(
        fitness=fmean(terms),
        mean_gap_by_scale=mean_gap,
        median_gap_by_scale=median_gap,
        baseline_gap_by_scale=baseline_gap,
        mean_delta_by_scale=mean_delta,
        degradation_by_scale=degradation,
        standard_error_by_scale=standard_error,
        nonzero_fraction_by_scale=nonzero_fraction,
        wins_by_scale=wins,
        ties_by_scale=ties,
        losses_by_scale=losses,
    )


def paired_ucb_fitness(
    candidate_lengths: dict[int, list[torch.Tensor]],
    baseline_lengths: dict[int, list[torch.Tensor]],
    references: dict[int, list[torch.Tensor]],
    *,
    z: float = 1.0,
) -> FitnessBreakdown:
    """以配对 gap 差值的均值加标准误上界作为训练 fitness。"""

    if z < 0.0:
        raise ValueError("paired UCB 的 z 不得为负")
    current = reference_gap_fitness(
        candidate_lengths,
        baseline_lengths,
        references,
    )
    terms = [
        current.mean_delta_by_scale[scale]
        + z * current.standard_error_by_scale[scale]
        for scale in sorted(current.mean_delta_by_scale)
    ]
    return FitnessBreakdown(
        fitness=fmean(terms),
        mean_gap_by_scale=current.mean_gap_by_scale,
        median_gap_by_scale=current.median_gap_by_scale,
        baseline_gap_by_scale=current.baseline_gap_by_scale,
        mean_delta_by_scale=current.mean_delta_by_scale,
        degradation_by_scale=current.degradation_by_scale,
        standard_error_by_scale=current.standard_error_by_scale,
        nonzero_fraction_by_scale=current.nonzero_fraction_by_scale,
        wins_by_scale=current.wins_by_scale,
        ties_by_scale=current.ties_by_scale,
        losses_by_scale=current.losses_by_scale,
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
        standard_error_by_scale=current.standard_error_by_scale,
        nonzero_fraction_by_scale=current.nonzero_fraction_by_scale,
        wins_by_scale=current.wins_by_scale,
        ties_by_scale=current.ties_by_scale,
        losses_by_scale=current.losses_by_scale,
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
            runtime=experiment.runtime,
        )
        candidate.setdefault(case.scale, []).append(result.best_length)
        baseline.setdefault(case.scale, []).append(
            baseline_length.to(case.batch.device)
        )
        references.setdefault(case.scale, []).append(case.batch.reference_length)
    if (
        experiment.gp.fitness_mode.uses_basin
        or experiment.gp.fitness_mode.uses_anytime
    ):
        raise ValueError(
            "basin/anytime fitness 要求 population-batched evaluator"
        )
    if experiment.gp.fitness_mode.is_paired:
        return paired_ucb_fitness(
            candidate,
            baseline,
            references,
            z=experiment.gp.fitness_ucb_z,
        )
    return reference_gap_fitness(candidate, baseline, references)


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
            runtime=experiment.runtime,
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
    baseline_values: Sequence[BaselineMeasurement],
) -> tuple[list[FitnessBreakdown], int]:
    """用一个 population×instance native 边界计算全部训练 fitness。"""

    if experiment.runtime.aco_backend in {
        ExecutionBackend.CUDA_FUSED_FP32,
        ExecutionBackend.CUDA_TILED_V2,
    }:
        from .aco_cuda import solve_population_cuda

        def solve_population(case, programs):
            return solve_population_cuda(
                case.batch,
                experiment.aco,
                programs,
                seed=case.seed,
                runtime=experiment.runtime,
                basin_top_q=(
                    experiment.gp.basin_top_q
                    if experiment.gp.fitness_mode.uses_basin
                    else 0
                ),
            )
    else:
        from .aco_numba import solve_population_numba

        def solve_population(case, programs):
            return solve_population_numba(
                case.batch,
                experiment.aco,
                programs,
                seed=case.seed,
                threads=experiment.runtime.cpu_threads,
                basin_top_q=(
                    experiment.gp.basin_top_q
                    if experiment.gp.fitness_mode.uses_basin
                    else 0
                ),
            )

    programs = [compile_individual(individual) for individual in individuals]
    candidate: dict[int, list[torch.Tensor]] = {}
    baseline: dict[int, list[torch.Tensor]] = {}
    candidate_basin: dict[int, list[torch.Tensor]] = {}
    baseline_basin: dict[int, list[torch.Tensor]] = {}
    candidate_anytime: dict[int, list[torch.Tensor]] = {}
    baseline_anytime: dict[int, list[torch.Tensor]] = {}
    references: dict[int, list[torch.Tensor]] = {}
    constructed_tours = 0
    for case, baseline_value in zip(cases, baseline_values, strict=True):
        result = solve_population(case, programs)
        candidate.setdefault(case.scale, []).append(result.best_length)
        baseline.setdefault(case.scale, []).append(
            baseline_value.best_length.detach().cpu()
        )
        if experiment.gp.fitness_mode.uses_basin:
            if (
                result.basin_mean_length is None
                or baseline_value.basin_mean_length is None
            ):
                raise RuntimeError("basin fitness 缺少 candidate/baseline basin 统计")
            candidate_basin.setdefault(case.scale, []).append(
                result.basin_mean_length
            )
            baseline_basin.setdefault(case.scale, []).append(
                baseline_value.basin_mean_length.detach().cpu()
            )
        if experiment.gp.fitness_mode.uses_anytime:
            if (
                result.anytime_mean_length is None
                or baseline_value.anytime_mean_length is None
            ):
                raise RuntimeError(
                    "anytime fitness 缺少 candidate/baseline anytime 统计"
                )
            candidate_anytime.setdefault(case.scale, []).append(
                result.anytime_mean_length
            )
            baseline_anytime.setdefault(case.scale, []).append(
                baseline_value.anytime_mean_length.detach().cpu()
            )
        references.setdefault(case.scale, []).append(
            case.batch.reference_length.detach().cpu()
        )
        constructed_tours += result.constructed_tours

    scale_values: dict[int, dict[str, torch.Tensor]] = {}
    for scale in sorted(candidate):
        candidate_length = torch.cat(candidate[scale], dim=1)
        baseline_length = torch.cat(baseline[scale]).unsqueeze(0)
        reference = torch.cat(references[scale]).unsqueeze(0)
        candidate_gap = 100.0 * (candidate_length - reference) / reference
        baseline_gap = 100.0 * (baseline_length - reference) / reference
        final_delta = candidate_gap - baseline_gap
        basin_gap: torch.Tensor | None = None
        baseline_basin_gap: torch.Tensor | None = None
        basin_delta: torch.Tensor | None = None
        if experiment.gp.fitness_mode.uses_basin:
            basin_length = torch.cat(candidate_basin[scale], dim=1)
            baseline_basin_length = torch.cat(
                baseline_basin[scale]
            ).unsqueeze(0)
            basin_gap = 100.0 * (basin_length - reference) / reference
            baseline_basin_gap = (
                100.0 * (baseline_basin_length - reference) / reference
            )
            basin_delta = basin_gap - baseline_basin_gap
        anytime_gap: torch.Tensor | None = None
        baseline_anytime_gap: torch.Tensor | None = None
        anytime_delta: torch.Tensor | None = None
        if experiment.gp.fitness_mode.uses_anytime:
            anytime_length = torch.cat(candidate_anytime[scale], dim=1)
            baseline_anytime_length = torch.cat(
                baseline_anytime[scale]
            ).unsqueeze(0)
            anytime_gap = 100.0 * (
                anytime_length - reference
            ) / reference
            baseline_anytime_gap = 100.0 * (
                baseline_anytime_length - reference
            ) / reference
            anytime_delta = anytime_gap - baseline_anytime_gap

        mode = experiment.gp.fitness_mode
        if mode in {
            FitnessMode.PAIRED_UCB,
            FitnessMode.PAIRED_FINAL_UCB,
        }:
            fitness_observation = final_delta
        elif mode is FitnessMode.PAIRED_BASIN_UCB:
            assert basin_delta is not None
            fitness_observation = basin_delta
        elif mode is FitnessMode.PAIRED_COMBINED_UCB:
            assert basin_delta is not None
            weight = experiment.gp.basin_weight
            fitness_observation = (
                weight * basin_delta + (1.0 - weight) * final_delta
            )
        elif mode is FitnessMode.PAIRED_ANYTIME_UCB:
            assert anytime_delta is not None
            fitness_observation = anytime_delta
        elif mode is FitnessMode.PAIRED_FINAL_ANYTIME_UCB:
            assert anytime_delta is not None
            weight = experiment.gp.anytime_weight
            fitness_observation = (
                weight * anytime_delta + (1.0 - weight) * final_delta
            )
        else:
            fitness_observation = candidate_gap
        audit_observation = (
            fitness_observation
            if mode.is_paired
            else final_delta
        )
        standard_error = (
            audit_observation.std(dim=1, unbiased=True)
            / np.sqrt(audit_observation.shape[1])
            if audit_observation.shape[1] > 1
            else audit_observation.new_zeros(audit_observation.shape[0])
        )
        scale_values[scale] = {
            "candidate_mean": candidate_gap.mean(dim=1),
            "candidate_median": candidate_gap.median(dim=1).values,
            "baseline_mean": baseline_gap.mean(dim=1).expand(len(individuals)),
            "final_delta_mean": final_delta.mean(dim=1),
            "fitness_delta_mean": audit_observation.mean(dim=1),
            "degradation": torch.clamp_min(
                audit_observation,
                0.0,
            ).mean(dim=1),
            "standard_error": standard_error,
            "nonzero": (
                (torch.abs(audit_observation) > 1e-12)
                .to(torch.float64)
                .mean(dim=1)
            ),
            "wins": torch.sum(audit_observation < -1e-12, dim=1),
            "ties": torch.sum(
                torch.abs(audit_observation) <= 1e-12,
                dim=1,
            ),
            "losses": torch.sum(audit_observation > 1e-12, dim=1),
        }
        if basin_gap is not None:
            assert baseline_basin_gap is not None and basin_delta is not None
            scale_values[scale].update(
                {
                    "basin_mean": basin_gap.mean(dim=1),
                    "baseline_basin_mean": baseline_basin_gap.mean(
                        dim=1
                    ).expand(len(individuals)),
                    "basin_delta_mean": basin_delta.mean(dim=1),
                }
            )
        if anytime_gap is not None:
            assert (
                baseline_anytime_gap is not None
                and anytime_delta is not None
            )
            scale_values[scale].update(
                {
                    "anytime_mean": anytime_gap.mean(dim=1),
                    "baseline_anytime_mean": baseline_anytime_gap.mean(
                        dim=1
                    ).expand(len(individuals)),
                    "anytime_delta_mean": anytime_delta.mean(dim=1),
                }
            )

    breakdowns: list[FitnessBreakdown] = []
    for index in range(len(individuals)):
        mean_gap = {
            scale: float(values["candidate_mean"][index].item())
            for scale, values in scale_values.items()
        }
        mean_delta = {
            scale: float(values["final_delta_mean"][index].item())
            for scale, values in scale_values.items()
        }
        fitness_delta = {
            scale: float(values["fitness_delta_mean"][index].item())
            for scale, values in scale_values.items()
        }
        standard_error = {
            scale: float(values["standard_error"][index].item())
            for scale, values in scale_values.items()
        }
        if experiment.gp.fitness_mode.is_paired:
            fitness_terms = [
                fitness_delta[scale]
                + experiment.gp.fitness_ucb_z * standard_error[scale]
                for scale in sorted(fitness_delta)
            ]
        else:
            fitness_terms = list(mean_gap.values())
        breakdowns.append(
            FitnessBreakdown(
                fitness=fmean(fitness_terms),
                mean_gap_by_scale=mean_gap,
                median_gap_by_scale={
                    scale: float(values["candidate_median"][index].item())
                    for scale, values in scale_values.items()
                },
                baseline_gap_by_scale={
                    scale: float(values["baseline_mean"][index].item())
                    for scale, values in scale_values.items()
                },
                mean_delta_by_scale=mean_delta,
                degradation_by_scale={
                    scale: float(values["degradation"][index].item())
                    for scale, values in scale_values.items()
                },
                standard_error_by_scale=standard_error,
                nonzero_fraction_by_scale={
                    scale: float(values["nonzero"][index].item())
                    for scale, values in scale_values.items()
                },
                wins_by_scale={
                    scale: int(values["wins"][index].item())
                    for scale, values in scale_values.items()
                },
                ties_by_scale={
                    scale: int(values["ties"][index].item())
                    for scale, values in scale_values.items()
                },
                losses_by_scale={
                    scale: int(values["losses"][index].item())
                    for scale, values in scale_values.items()
                },
                mean_basin_gap_by_scale={
                    scale: float(values["basin_mean"][index].item())
                    for scale, values in scale_values.items()
                    if "basin_mean" in values
                },
                baseline_basin_gap_by_scale={
                    scale: float(
                        values["baseline_basin_mean"][index].item()
                    )
                    for scale, values in scale_values.items()
                    if "baseline_basin_mean" in values
                },
                mean_basin_delta_by_scale={
                    scale: float(values["basin_delta_mean"][index].item())
                    for scale, values in scale_values.items()
                    if "basin_delta_mean" in values
                },
                mean_anytime_gap_by_scale={
                    scale: float(values["anytime_mean"][index].item())
                    for scale, values in scale_values.items()
                    if "anytime_mean" in values
                },
                baseline_anytime_gap_by_scale={
                    scale: float(
                        values["baseline_anytime_mean"][index].item()
                    )
                    for scale, values in scale_values.items()
                    if "baseline_anytime_mean" in values
                },
                mean_anytime_delta_by_scale={
                    scale: float(
                        values["anytime_delta_mean"][index].item()
                    )
                    for scale, values in scale_values.items()
                    if "anytime_delta_mean" in values
                },
                fitness_delta_by_scale=fitness_delta,
            )
        )
    return breakdowns, constructed_tours


def _batched_validation_data(
    individuals: Sequence[RMTGPIndividual],
    experiment: ExperimentConfig,
    cases: Sequence[EvaluationCase],
    baseline_values: Sequence[BaselineMeasurement],
) -> dict[str, ValidationData]:
    """批量计算 validation absolute/baseline/delta gaps。"""

    if experiment.runtime.aco_backend in {
        ExecutionBackend.CUDA_FUSED_FP32,
        ExecutionBackend.CUDA_TILED_V2,
    }:
        from .aco_cuda import solve_population_cuda

        def solve_population(case, programs):
            return solve_population_cuda(
                case.batch,
                experiment.aco,
                programs,
                seed=case.seed,
                runtime=experiment.runtime,
            )
    else:
        from .aco_numba import solve_population_numba

        def solve_population(case, programs):
            return solve_population_numba(
                case.batch,
                experiment.aco,
                programs,
                seed=case.seed,
                threads=experiment.runtime.cpu_threads,
            )

    programs = [compile_individual(individual) for individual in individuals]
    candidates: dict[int, list[torch.Tensor]] = {}
    baselines: dict[int, list[torch.Tensor]] = {}
    references: dict[int, list[torch.Tensor]] = {}
    ids: dict[int, list[np.ndarray]] = {}
    for case, baseline_value in zip(cases, baseline_values, strict=True):
        result = solve_population(case, programs)
        candidates.setdefault(case.scale, []).append(result.best_length)
        baselines.setdefault(case.scale, []).append(
            baseline_value.best_length.detach().cpu()
        )
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
            runtime=_WORKER_EXPERIMENT.runtime,
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
        runtime=_WORKER_EXPERIMENT.runtime,
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

    def set_experiment(self, experiment: ExperimentConfig) -> None:
        """切换当前 generation 的 ACO horizon。

        持久多进程 worker 会持有初始化时的配置，因此只允许单进程后端
        动态切换。CUDA 正式训练本身即为单进程。
        """

        if self.executor is not None and experiment != self.experiment:
            raise ValueError(
                "多进程 EvaluationPool 不支持动态 training horizon"
            )
        self.experiment = experiment

    def warm(self, case: EvaluationCase) -> set[int]:
        """在正式计时前加载每个 worker 的已编译 Numba cache。"""

        if (
            self.experiment.runtime.aco_backend
            in {
                ExecutionBackend.NUMBA_BATCH,
                ExecutionBackend.CUDA_FUSED_FP32,
                ExecutionBackend.CUDA_TILED_V2,
            }
        ):
            if (
                self.experiment.runtime.aco_backend
                in {
                    ExecutionBackend.CUDA_FUSED_FP32,
                    ExecutionBackend.CUDA_TILED_V2,
                }
            ):
                from .aco_cuda import solve_population_cuda

                solve_population_cuda(
                    case.batch,
                    self.experiment.aco,
                    [(None, None)],
                    seed=case.seed,
                    runtime=self.experiment.runtime,
                )
            else:
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
    ) -> tuple[BaselineMeasurement, ...]:
        missing = [
            case
            for case in cases
            if cache.get_measurement_cpu(case, self.experiment) is None
        ]
        if missing:
            if (
                self.experiment.gp.fitness_mode.uses_basin
                or self.experiment.gp.fitness_mode.uses_anytime
            ):
                # 每个 case 已是一个较大的 instance batch。baseline 只占一个
                # program task，直接复用 population kernel 可避免 Python 循环。
                for case in missing:
                    cache.measurement(case, self.experiment)
            else:
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
                            runtime=self.experiment.runtime,
                        ).best_length.detach().cpu()
                        for case in missing
                    ]
                for case, value in zip(missing, values, strict=True):
                    cache.put(case, self.experiment, value)
        result: list[BaselineMeasurement] = []
        for case in cases:
            value = cache.get_measurement_cpu(case, self.experiment)
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
            in {
                ExecutionBackend.NUMBA_BATCH,
                ExecutionBackend.CUDA_FUSED_FP32,
                ExecutionBackend.CUDA_TILED_V2,
            }
        ):
            breakdowns, constructed_tours = _batched_population_breakdowns(
                individuals,
                self.experiment,
                cases,
                baseline_values,
            )
        elif self.executor is None:
            explicit_baselines = tuple(
                value.best_length for value in baseline_values
            )
            breakdowns = [
                _score_with_explicit_baselines(
                    individual,
                    self.experiment,
                    cases,
                    explicit_baselines,
                )
                for individual in individuals
            ]
        else:
            explicit_baselines = tuple(
                value.best_length for value in baseline_values
            )
            individual_chunks = _chunked(
                individuals,
                self.experiment.runtime.processes,
            )
            payloads = [
                (chunk, tuple(cases), explicit_baselines)
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
            in {
                ExecutionBackend.NUMBA_BATCH,
                ExecutionBackend.CUDA_FUSED_FP32,
                ExecutionBackend.CUDA_TILED_V2,
            }
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
                    tuple(
                        value.best_length for value in baseline_values
                    ),
                )
                for individual in candidates
            }
        candidate_chunks = _chunked(
            candidates,
            self.experiment.runtime.processes * 2,
        )
        payloads = [
            (
                chunk,
                tuple(cases),
                tuple(value.best_length for value in baseline_values),
            )
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


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    confidence: float,
    replicates: int,
) -> tuple[float, float, float]:
    """返回双侧区间以及同置信度的单侧上界。"""

    if values.size < 1:
        raise ValueError("bootstrap values 不得为空")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        values.size,
        size=(replicates, values.size),
    )
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    upper = np.quantile(means, confidence)
    return float(low), float(high), float(upper)


def _relative_gap_improvement(
    candidate_gap: np.ndarray,
    baseline_gap: np.ndarray,
) -> float:
    """按两个 mean gap 计算相对提升；零 baseline 不可宣称正提升。"""

    candidate_mean = float(candidate_gap.mean())
    baseline_mean = float(baseline_gap.mean())
    if baseline_mean <= 1e-15:
        return 0.0 if candidate_mean <= baseline_mean + 1e-15 else float("-inf")
    return (baseline_mean - candidate_mean) / baseline_mean


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


def _validation_quality_score(
    data: ValidationData,
    experiment: ExperimentConfig,
    *,
    stage_code: int,
) -> float:
    """返回 checkpoint 排序分数；严格协议使用配对 bootstrap 上界。"""

    if experiment.selection_mode is SelectionMode.LEGACY_NONINFERIORITY:
        return fmean(
            float(values.mean())
            for values in data.candidate_gap_by_scale.values()
        )
    upper_bounds: list[float] = []
    for scale, raw_values in sorted(data.delta_by_scale.items()):
        values = _aggregate_by_instance(
            raw_values,
            data.instance_ids_by_scale[scale],
        )
        _, _, upper = _bootstrap_mean_interval(
            values,
            seed=int(
                np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            experiment.root_seed,
                            scale,
                            stage_code,
                            0x554342,
                        ]
                    )
                ).integers(0, 2**63 - 1)
            ),
            confidence=experiment.selection_confidence,
            replicates=experiment.selection_bootstrap_replicates,
        )
        upper_bounds.append(upper)
    return fmean(upper_bounds)


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
    """两阶段选 champion，并按配置执行非劣或严格优越门控。"""

    unique = {
        individual.structural_hash: individual
        for individual in candidates
    }
    if experiment.selection_mode is SelectionMode.STRICT_SUPERIORITY:
        unique = {
            key: individual
            for key, individual in unique.items()
            if not is_baseline_individual(individual)
        }
    if not unique:
        raise ValueError("validation candidates 不含可学习的非 baseline 个体")
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
        score = _validation_quality_score(
            data,
            experiment,
            stage_code=0x534352,
        )
        screening_scored.append((score, individual.total_nodes, individual))
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
        score = _validation_quality_score(
            data,
            experiment,
            stage_code=0x53454C,
        )
        scored.append((score, individual.total_nodes, individual, data))
    scored.sort(key=lambda item: (item[0], item[1]))
    best_score = scored[0][0]
    tie_tolerance = (
        1e-12
        if experiment.selection_mode is SelectionMode.STRICT_SUPERIORITY
        else experiment.quality_tie_tolerance
    )
    near_ties = [
        item
        for item in scored
        if item[0] <= best_score + tie_tolerance
    ]
    selection_score, _, selected, selected_data = min(
        near_ties,
        key=lambda item: (item[1], item[0]),
    )
    macro_gap = fmean(
        float(values.mean())
        for values in selected_data.candidate_gap_by_scale.values()
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

    scale_summaries: list[ValidationScaleSummary] = []
    scale_passes: list[bool] = []
    for scale, values in sorted(instance_deltas.items()):
        candidate_instance_gap = _aggregate_by_instance(
            gate_data.candidate_gap_by_scale[scale],
            gate_data.instance_ids_by_scale[scale],
        )
        baseline_instance_gap = _aggregate_by_instance(
            gate_data.baseline_gap_by_scale[scale],
            gate_data.instance_ids_by_scale[scale],
        )
        ci_low, ci_high, bootstrap_upper = _bootstrap_mean_interval(
            values,
            seed=int(
                np.random.default_rng(
                    np.random.SeedSequence(
                        [experiment.root_seed, scale, 0x424F4F54]
                    )
                ).integers(0, 2**63 - 1)
            ),
            confidence=experiment.selection_confidence,
            replicates=experiment.selection_bootstrap_replicates,
        )
        relative_improvement = _relative_gap_improvement(
            candidate_instance_gap,
            baseline_instance_gap,
        )
        if experiment.selection_mode is SelectionMode.STRICT_SUPERIORITY:
            scale_passes.append(
                bootstrap_upper < 0.0
                and relative_improvement
                >= experiment.superiority_min_relative_improvement
            )
        else:
            scale_passes.append(
                _normal_upper_bound(values)
                <= experiment.noninferiority_tolerance
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
                candidate_mean_gap_percent=float(
                    candidate_instance_gap.mean()
                ),
                baseline_mean_gap_percent=float(
                    baseline_instance_gap.mean()
                ),
                relative_improvement=relative_improvement,
                bootstrap_upper_bound=bootstrap_upper,
            )
        )
    passed = all(scale_passes)

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
        selected_candidate=selected.clone(),
        backend=experiment.runtime.aco_backend,
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
        selection_score=float(selection_score),
        selection_mode=experiment.selection_mode.value,
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
    exclude_baseline: bool = False,
    training_aco_iterations: int = 0,
) -> GenerationRecord:
    ranked_population = (
        [
            item
            for item in population
            if not is_baseline_individual(item)
        ]
        if exclude_baseline
        else list(population)
    )
    if evaluation.racing_high_evaluated_unique:
        ranked_population = [
            item
            for item in ranked_population
            if (
                not is_baseline_individual(item)
                and int(
                    item.metadata.get("racing_fidelity_tier", 1)
                ) == 0
            )
        ]
    if not ranked_population:
        raise ValueError("generation record 不含可统计的 GP 个体")
    values = np.asarray(
        [item.fitness.values[0] for item in ranked_population],
        dtype=float,
    )
    best = min(
        ranked_population,
        key=(
            _racing_selection_key
            if evaluation.racing_high_evaluated_unique
            else lambda item: (
                item.fitness.values[0],
                item.total_nodes,
            )
        ),
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
        best_standard_error_by_scale=(
            {} if breakdown is None else dict(breakdown.standard_error_by_scale)
        ),
        best_nonzero_fraction_by_scale=(
            {} if breakdown is None else dict(breakdown.nonzero_fraction_by_scale)
        ),
        best_wins_by_scale=(
            {} if breakdown is None else dict(breakdown.wins_by_scale)
        ),
        best_ties_by_scale=(
            {} if breakdown is None else dict(breakdown.ties_by_scale)
        ),
        best_losses_by_scale=(
            {} if breakdown is None else dict(breakdown.losses_by_scale)
        ),
        baseline_anchor_count=sum(
            is_baseline_individual(item) for item in population
        ),
        training_aco_iterations=training_aco_iterations,
        best_mean_basin_gap_by_scale=(
            {}
            if breakdown is None
            else dict(breakdown.mean_basin_gap_by_scale)
        ),
        baseline_mean_basin_gap_by_scale=(
            {}
            if breakdown is None
            else dict(breakdown.baseline_basin_gap_by_scale)
        ),
        best_mean_basin_delta_by_scale=(
            {}
            if breakdown is None
            else dict(breakdown.mean_basin_delta_by_scale)
        ),
        best_mean_anytime_gap_by_scale=(
            {}
            if breakdown is None
            else dict(breakdown.mean_anytime_gap_by_scale)
        ),
        baseline_mean_anytime_gap_by_scale=(
            {}
            if breakdown is None
            else dict(breakdown.baseline_anytime_gap_by_scale)
        ),
        best_mean_anytime_delta_by_scale=(
            {}
            if breakdown is None
            else dict(breakdown.mean_anytime_delta_by_scale)
        ),
        best_fitness_delta_by_scale=(
            {}
            if breakdown is None
            else dict(breakdown.fitness_delta_by_scale)
        ),
        racing_enabled=bool(evaluation.racing_high_evaluated_unique),
        racing_screen_evaluated_unique=(
            evaluation.racing_screen_evaluated_unique
        ),
        racing_high_evaluated_unique=(
            evaluation.racing_high_evaluated_unique
        ),
        racing_screen_iterations=evaluation.racing_screen_iterations,
        racing_high_iterations=evaluation.racing_high_iterations,
        racing_screen_instances=evaluation.racing_screen_instances,
        racing_high_instances=evaluation.racing_high_instances,
        racing_finalist_hashes=evaluation.racing_finalist_hashes,
        racing_screen_fitness_by_hash=dict(
            evaluation.racing_screen_fitness_by_hash
        ),
        racing_high_fitness_by_hash=dict(
            evaluation.racing_high_fitness_by_hash
        ),
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
    cupy_module: object | None = None
    try:
        import cupy as cupy_module

        cupy_version = cupy_module.__version__
        cupy_devices = []
        for index in range(cupy_module.cuda.runtime.getDeviceCount()):
            name = cupy_module.cuda.runtime.getDeviceProperties(index)["name"]
            cupy_devices.append(
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
            )
        cupy_error = None
    except (ImportError, ModuleNotFoundError):
        cupy_version = "unavailable"
        cupy_devices = []
        cupy_error = None
    except Exception as exc:
        # CPU-only 节点可能装有 CuPy 但没有可用驱动；artifact 记录错误而不
        # 阻断正式 CPU run。
        cupy_version = getattr(cupy_module, "__version__", "unknown")
        cupy_devices = []
        cupy_error = f"{type(exc).__name__}: {exc}"
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "deap": deap_version,
        "numba": numba_version,
        "llvmlite": llvmlite_version,
        "cupy": cupy_version,
        "cupy_devices": cupy_devices,
        "cupy_runtime_error": cupy_error,
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
    _write_training_validation_curve(history, target)


def _write_training_validation_curve(
    history: Sequence[GenerationRecord],
    target: Path,
) -> None:
    """写出可直接绘制 train/validation 曲线的逐代长表。"""

    lines = [
        (
            "generation,scale,train_candidate_gap_percent,"
            "train_baseline_gap_percent,train_delta_pp,"
            "train_delta_standard_error_pp,train_nonzero_fraction,"
            "train_wins,train_ties,train_losses,aco_iterations,"
            "train_anytime_candidate_gap_percent,"
            "train_anytime_baseline_gap_percent,"
            "train_anytime_delta_pp,train_fitness_delta_pp,"
            "racing_enabled,racing_screen_evaluated_unique,"
            "racing_high_evaluated_unique,racing_screen_iterations,"
            "racing_high_iterations,racing_screen_instances,"
            "racing_high_instances,"
            "validation_candidate_gap_percent,"
            "validation_baseline_gap_percent,validation_delta_pp,"
            "generation_wall_time_sec,validation_monitor_wall_time_sec\n"
        )
    ]
    for record in history:
        scales = sorted(
            set(record.best_mean_gap_by_scale)
            | set(record.validation_monitor_candidate_gap_by_scale)
        )
        for scale in scales:
            values = (
                record.best_mean_gap_by_scale.get(scale, float("nan")),
                record.baseline_mean_gap_by_scale.get(scale, float("nan")),
                record.best_mean_delta_by_scale.get(scale, float("nan")),
                record.best_standard_error_by_scale.get(scale, float("nan")),
                record.best_nonzero_fraction_by_scale.get(
                    scale,
                    float("nan"),
                ),
                record.best_wins_by_scale.get(scale, 0),
                record.best_ties_by_scale.get(scale, 0),
                record.best_losses_by_scale.get(scale, 0),
                record.training_aco_iterations,
                record.best_mean_anytime_gap_by_scale.get(
                    scale,
                    float("nan"),
                ),
                record.baseline_mean_anytime_gap_by_scale.get(
                    scale,
                    float("nan"),
                ),
                record.best_mean_anytime_delta_by_scale.get(
                    scale,
                    float("nan"),
                ),
                record.best_fitness_delta_by_scale.get(
                    scale,
                    float("nan"),
                ),
                int(record.racing_enabled),
                record.racing_screen_evaluated_unique,
                record.racing_high_evaluated_unique,
                record.racing_screen_iterations,
                record.racing_high_iterations,
                record.racing_screen_instances,
                record.racing_high_instances,
                record.validation_monitor_candidate_gap_by_scale.get(
                    scale,
                    float("nan"),
                ),
                record.validation_monitor_baseline_gap_by_scale.get(
                    scale,
                    float("nan"),
                ),
                record.validation_monitor_delta_by_scale.get(
                    scale,
                    float("nan"),
                ),
            )
            rendered = ",".join(f"{value:.17g}" for value in values)
            lines.append(
                f"{record.generation},{scale},{rendered},"
                f"{record.generation_wall_time:.17g},"
                f"{record.validation_monitor_wall_time:.17g}\n"
            )
    _atomic_write_text(
        target / "training_validation_curve.csv",
        "".join(lines),
    )


def _experiment_hash(experiment: ExperimentConfig) -> str:
    payload = json.dumps(
        experiment.stable_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _pre_anytime_experiment_hash(experiment: ExperimentConfig) -> str | None:
    """返回新增 Anytime/Racing 字段前的兼容哈希。

    该迁移只适用于未启用新机制的旧实验。删除的两个字段在旧版本中不存在，
    且当前值必须等于默认值；任何实际算法配置变化仍会被拒绝。
    """

    if (
        experiment.racing.enabled
        or experiment.gp.fitness_mode.uses_anytime
        or experiment.gp.anytime_weight != 0.5
    ):
        return None
    payload = experiment.stable_dict()
    payload.pop("racing", None)
    gp = payload.get("gp")
    if not isinstance(gp, dict):
        return None
    gp.pop("anytime_weight", None)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _hydrate_generation_record(record: GenerationRecord) -> None:
    """为旧 pickle 中后来新增、且有默认值的审计字段补默认值。"""

    for descriptor in fields(GenerationRecord):
        if hasattr(record, descriptor.name):
            continue
        if descriptor.default is not MISSING:
            value = deepcopy(descriptor.default)
        elif descriptor.default_factory is not MISSING:
            value = descriptor.default_factory()
        else:
            raise ValueError(
                "旧 checkpoint 缺少无默认值字段 "
                f"GenerationRecord.{descriptor.name}"
            )
        setattr(record, descriptor.name, value)


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
    expected_hashes = {_experiment_hash(experiment)}
    legacy_hash = _pre_anytime_experiment_hash(experiment)
    if legacy_hash is not None:
        expected_hashes.add(legacy_hash)
    if payload.get("experiment_hash") not in expected_hashes:
        raise ValueError("resume 配置与 checkpoint 不一致")
    for record in payload.get("history", ()):
        if not isinstance(record, GenerationRecord):
            raise TypeError("training checkpoint history 类型错误")
        _hydrate_generation_record(record)
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
            f"- Selection backend：{result.validation.backend.value}",
            (
                "- 选中候选 macro reference gap："
                f"{result.validation.selected_macro_gap_percent:.6f}%"
            ),
            f"- 选中候选 macro Δ：{result.validation.selected_macro_delta_pp:+.6f} pp",
            (
                "- Selection non-inferiority："
                + (
                    "通过"
                    if result.validation.passed_noninferiority
                    else "失败"
                )
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
    if result.cpu_fp64_audit is not None:
        audit = result.cpu_fp64_audit
        lines.extend(
            [
                "",
                "## CPU/FP64 final audit",
                "",
                f"- Backend：{audit.backend.value}",
                (
                    "- 选中候选 macro reference gap："
                    f"{audit.selected_macro_gap_percent:.6f}%"
                ),
                f"- 选中候选 macro Δ：{audit.selected_macro_delta_pp:+.6f} pp",
                (
                    "- CPU/FP64 non-inferiority："
                    + ("通过" if audit.passed_noninferiority else "失败")
                ),
            ]
        )
        for summary in audit.scales:
            lines.append(
                f"- TSP{summary.scale}: mean={summary.mean_delta_pp:+.6f} pp, "
                f"95% CI=[{summary.bootstrap_ci_low:+.6f}, "
                f"{summary.bootstrap_ci_high:+.6f}], "
                f"W/T/L={summary.wins}/{summary.ties}/{summary.losses}"
            )
    lines.extend(
        [
            "",
            "## Deployment decision",
            "",
            (
                "- Final non-inferiority："
                + (
                    "通过"
                    if result.passed_noninferiority
                    else "失败，部署 baseline fallback"
                )
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _validation_csv(
    selection: ValidationSelection,
    *,
    deployed: bool,
) -> str:
    """把一个 selection/audit 阶段写成带后端标识的长表。"""

    lines = [
        (
            "backend,scale,observations,instances,mean_delta_pp,"
            "median_delta_pp,bootstrap_ci_low,bootstrap_ci_high,"
            "bootstrap_upper_bound,candidate_mean_gap_percent,"
            "baseline_mean_gap_percent,relative_improvement,"
            "normal_upper_bound_95,wins,ties,losses,"
            "stage_passed_noninferiority,deployed\n"
        )
    ]
    for summary in selection.scales:
        lines.append(
            f"{selection.backend.value},{summary.scale},"
            f"{summary.observations},{summary.instances},"
            f"{summary.mean_delta_pp:.17g},{summary.median_delta_pp:.17g},"
            f"{summary.bootstrap_ci_low:.17g},{summary.bootstrap_ci_high:.17g},"
            f"{summary.bootstrap_upper_bound:.17g},"
            f"{summary.candidate_mean_gap_percent:.17g},"
            f"{summary.baseline_mean_gap_percent:.17g},"
            f"{summary.relative_improvement:.17g},"
            f"{summary.normal_upper_bound_95:.17g},{summary.wins},"
            f"{summary.ties},{summary.losses},"
            f"{str(selection.passed_noninferiority).lower()},"
            f"{str(deployed).lower()}\n"
        )
    return "".join(lines)


def save_training_result(
    result: TrainingResult,
    experiment: ExperimentConfig,
    output_directory: str | Path,
) -> Path:
    """保存可复现 artifact、validation 统计与全部候选。"""

    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    audit_passed = (
        None
        if result.cpu_fp64_audit is None
        else result.cpu_fp64_audit.passed_noninferiority
    )
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
            f"selection_backend: {result.validation.backend.value}\n"
            f"selection_passed_noninferiority: "
            f"{result.validation.passed_noninferiority}\n"
            f"cpu_fp64_audit_passed: {audit_passed}\n"
            f"selected_candidate_hash: "
            f"{result.validation.selected_candidate_hash}\n"
        ),
    )
    with (target / "champion.pkl").open("wb") as handle:
        pickle.dump(result.champion, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_text(
        target / "selected_candidate_expression.txt",
        (
            f"transition: {result.validation.selected_candidate.transition_tree}\n"
            f"pheromone: {result.validation.selected_candidate.pheromone_tree}\n"
            f"selected_candidate_hash: "
            f"{result.validation.selected_candidate_hash}\n"
            f"selection_backend: {result.validation.backend.value}\n"
            f"selection_passed_noninferiority: "
            f"{result.validation.passed_noninferiority}\n"
            f"final_deployed: {result.passed_noninferiority}\n"
        ),
    )
    with (target / "selected_candidate.pkl").open("wb") as handle:
        pickle.dump(
            result.validation.selected_candidate,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    _atomic_write_text(
        target / "deployment_decision.json",
        json.dumps(
            {
                "schema_version": 1,
                "selected_candidate_hash": (
                    result.validation.selected_candidate_hash
                ),
                "deployed_champion_hash": result.champion.structural_hash,
                "selection_backend": result.validation.backend.value,
                "selection_passed_noninferiority": (
                    result.validation.passed_noninferiority
                ),
                "cpu_fp64_audit_passed": audit_passed,
                "final_passed_noninferiority": result.passed_noninferiority,
                "deployed_method": (
                    "rmtgp-selected"
                    if result.passed_noninferiority
                    else "baseline-fallback"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    _atomic_write_text(
        target / "validation_summary.csv",
        _validation_csv(
            result.validation,
            deployed=result.passed_noninferiority,
        ),
    )
    if result.cpu_fp64_audit is not None:
        _atomic_write_text(
            target / "cpu_fp64_audit_summary.csv",
            _validation_csv(
                result.cpu_fp64_audit,
                deployed=result.passed_noninferiority,
            ),
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


def _cpu_fp64_final_audit(
    candidate: RMTGPIndividual,
    experiment: ExperimentConfig,
    gate_cases: Sequence[EvaluationCase],
) -> ValidationSelection:
    """用独立 CPU float64 semantic domain 复评 CUDA 选出的候选。

    GPU baseline archive 属于另一 kernel semantic domain，不能用于此处；
    CPU baseline 与候选均按相同 instance/seed 现场批量计算并在内存中缓存。
    """

    cpu_experiment = replace(
        experiment,
        experiment_id=f"{experiment.experiment_id}-cpu-fp64-audit",
        runtime=replace(
            experiment.runtime,
            aco_backend=ExecutionBackend.NUMBA_BATCH,
            processes=1,
            gpu_devices=(),
            gpu_mode=GPUMode.CPU,
            gpu_block_threads=0,
            gpu_task_chunk_size=0,
        ),
    )
    return validate_candidates(
        [candidate],
        cpu_experiment,
        gate_cases,
        BaselineCache(),
    )


def _training_experiment_for_generation(
    experiment: ExperimentConfig,
    generation: int,
) -> ExperimentConfig:
    iterations = experiment.iterations_for_generation(generation)
    if iterations == experiment.aco.iterations:
        return experiment
    return replace(
        experiment,
        aco=replace(experiment.aco, iterations=iterations),
    )


def _racing_screen_experiment_for_generation(
    experiment: ExperimentConfig,
    generation: int,
) -> ExperimentConfig:
    """构造本代 Stage-1 配置；其 baseline cache 域与高保真域严格分离。"""

    high_iterations = experiment.iterations_for_generation(generation)
    screen_iterations = experiment.racing.iterations_for_generation(
        generation,
        generations=experiment.gp.generations,
        fallback=high_iterations,
    )
    return replace(
        experiment,
        aco=replace(experiment.aco, iterations=screen_iterations),
    )


def _racing_screen_cases(
    cases: Sequence[EvaluationCase],
    instances_per_scale: int,
) -> tuple[EvaluationCase, ...]:
    """从已冻结的本代 batch 取前缀，保证两阶段严格共享实例与 seed。"""

    selected: list[EvaluationCase] = []
    for case in cases:
        if case.batch.batch_size < instances_per_scale:
            raise ValueError(
                f"TSP{case.scale} 本代仅有 {case.batch.batch_size} 个实例，"
                f"少于 racing screen 所需 {instances_per_scale}"
            )
        selected.append(
            EvaluationCase(
                scale=case.scale,
                batch=case.batch.take(tuple(range(instances_per_scale))),
                seed=case.seed,
            )
        )
    return tuple(selected)


def _racing_selection_key(
    individual: RMTGPIndividual,
) -> tuple[object, ...]:
    """用于繁殖的 ``(保真度层级, UCB, 节点数, hash)`` 全序。"""

    tier = int(individual.metadata.get("racing_fidelity_tier", 1))
    breakdown = individual.metadata.get(
        "racing_high_breakdown"
        if tier == 0
        else "racing_screen_breakdown"
    )
    score = (
        float(breakdown.fitness)
        if isinstance(breakdown, FitnessBreakdown)
        else float(individual.fitness.values[0])
    )
    return tier, score, individual.total_nodes, individual.structural_hash


def _racing_checkpoint_candidates(
    population: Sequence[RMTGPIndividual],
    count: int,
) -> list[RMTGPIndividual]:
    """只从本代经过高保真复评的学习个体中选择 checkpoint。"""

    high = [
        item
        for item in _learned_candidates(population)
        if int(item.metadata.get("racing_fidelity_tier", 1)) == 0
    ]
    unique: dict[str, RMTGPIndividual] = {}
    for item in sorted(high, key=_racing_selection_key):
        unique.setdefault(item.structural_hash, item)
    return list(unique.values())[:count]


def _evaluate_population_with_racing(
    *,
    population: Sequence[RMTGPIndividual],
    cases: Sequence[EvaluationCase],
    generation: int,
    experiment: ExperimentConfig,
    evaluator: EvaluationPool,
    baseline_cache: BaselineCache,
) -> tuple[PopulationEvaluationResult, ExperimentConfig]:
    """执行 Stage-1 全量筛选与 Stage-2 finalists 复评。

    ``preserved_elites`` 使用上一代真实高保真分数。探索配额使用独立、
    由 root seed 与 generation 派生的 RNG，不消耗 GP 的全局随机流，
    因而中断恢复后会得到完全相同的 finalists。
    """

    learned = _learned_candidates(population)
    previous_high: dict[str, tuple[float, int]] = {}
    for item in learned:
        score = item.metadata.get("racing_high_score")
        if isinstance(score, (float, int)):
            previous = previous_high.get(item.structural_hash)
            candidate = (float(score), item.total_nodes)
            if previous is None or candidate < previous:
                previous_high[item.structural_hash] = candidate

    screen_experiment = _racing_screen_experiment_for_generation(
        experiment,
        generation,
    )
    high_experiment = _training_experiment_for_generation(
        experiment,
        generation,
    )
    screen_cases = _racing_screen_cases(
        cases,
        experiment.racing.screen_instances_per_scale,
    )
    high_cases = (
        tuple(cases)
        if experiment.racing.high_instances_per_scale is None
        else _racing_screen_cases(
            cases,
            experiment.racing.high_instances_per_scale,
        )
    )

    evaluator.set_experiment(screen_experiment)
    screen_evaluation = evaluator.evaluate_population(
        population,
        screen_cases,
        baseline_cache,
    )
    screen_by_hash: dict[str, RMTGPIndividual] = {}
    for item in learned:
        breakdown = item.metadata.get("fitness_breakdown")
        if not isinstance(breakdown, FitnessBreakdown):
            raise RuntimeError("racing Stage 1 缺少 fitness breakdown")
        item.metadata["racing_screen_breakdown"] = breakdown
        item.metadata["racing_screen_score"] = float(breakdown.fitness)
        item.metadata["racing_fidelity_tier"] = 1
        item.metadata.pop("racing_high_breakdown", None)
        item.metadata.pop("racing_high_score", None)
        screen_by_hash.setdefault(item.structural_hash, item)

    exploitation_count = (
        experiment.racing.finalists
        - experiment.racing.exploration_finalists
    )
    selected_hashes: list[str] = []
    for structural_hash, _ in sorted(
        previous_high.items(),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    ):
        if structural_hash in screen_by_hash:
            selected_hashes.append(structural_hash)
            if len(selected_hashes) >= experiment.racing.preserved_elites:
                break
    screen_ranked = sorted(
        screen_by_hash.values(),
        key=lambda item: (
            float(item.metadata["racing_screen_score"]),
            item.total_nodes,
            item.structural_hash,
        ),
    )
    for item in screen_ranked:
        if item.structural_hash not in selected_hashes:
            selected_hashes.append(item.structural_hash)
        if len(selected_hashes) >= exploitation_count:
            break

    remaining = sorted(
        set(screen_by_hash) - set(selected_hashes)
    )
    exploration_count = min(
        experiment.racing.exploration_finalists,
        len(remaining),
    )
    if exploration_count:
        seed_payload = (
            f"{experiment.root_seed}\0{generation}\0racing-exploration"
        )
        exploration_seed = int.from_bytes(
            sha256(seed_payload.encode("utf-8")).digest()[:8],
            byteorder="little",
        )
        exploration_rng = random.Random(exploration_seed)
        selected_hashes.extend(
            exploration_rng.sample(remaining, exploration_count)
        )
    if len(selected_hashes) < min(
        experiment.racing.finalists,
        len(screen_by_hash),
    ):
        for item in screen_ranked:
            if item.structural_hash not in selected_hashes:
                selected_hashes.append(item.structural_hash)
            if len(selected_hashes) >= min(
                experiment.racing.finalists,
                len(screen_by_hash),
            ):
                break

    selected_set = set(selected_hashes)
    high_population = [
        item
        for item in population
        if is_baseline_individual(item)
        or item.structural_hash in selected_set
    ]
    for item in high_population:
        if item.fitness.valid:
            del item.fitness.values

    evaluator.set_experiment(high_experiment)
    high_evaluation = evaluator.evaluate_population(
        high_population,
        high_cases,
        baseline_cache,
    )
    for item in learned:
        if item.structural_hash not in selected_set:
            continue
        breakdown = item.metadata.get("fitness_breakdown")
        if not isinstance(breakdown, FitnessBreakdown):
            raise RuntimeError("racing Stage 2 缺少 fitness breakdown")
        item.metadata["racing_high_breakdown"] = breakdown
        item.metadata["racing_high_score"] = float(breakdown.fitness)
        item.metadata["racing_fidelity_tier"] = 0

    combined_breakdowns = dict(screen_evaluation.breakdowns)
    combined_breakdowns.update(high_evaluation.breakdowns)
    combined = PopulationEvaluationResult(
        evaluated_unique=(
            screen_evaluation.evaluated_unique
            + high_evaluation.evaluated_unique
        ),
        breakdowns=combined_breakdowns,
        baseline_wall_time=(
            screen_evaluation.baseline_wall_time
            + high_evaluation.baseline_wall_time
        ),
        evaluation_wall_time=(
            screen_evaluation.evaluation_wall_time
            + high_evaluation.evaluation_wall_time
        ),
        constructed_tours=(
            screen_evaluation.constructed_tours
            + high_evaluation.constructed_tours
        ),
        racing_screen_evaluated_unique=(
            screen_evaluation.evaluated_unique
        ),
        racing_high_evaluated_unique=high_evaluation.evaluated_unique,
        racing_screen_iterations=screen_experiment.aco.iterations,
        racing_high_iterations=high_experiment.aco.iterations,
        racing_screen_instances=sum(
            case.batch.batch_size for case in screen_cases
        ),
        racing_high_instances=sum(
            case.batch.batch_size for case in high_cases
        ),
        racing_finalist_hashes=tuple(selected_hashes),
        racing_screen_fitness_by_hash={
            structural_hash: float(breakdown.fitness)
            for structural_hash, breakdown
            in screen_evaluation.breakdowns.items()
            if structural_hash in screen_by_hash
        },
        racing_high_fitness_by_hash={
            structural_hash: float(breakdown.fitness)
            for structural_hash, breakdown
            in high_evaluation.breakdowns.items()
            if structural_hash in selected_set
        },
    )
    return combined, high_experiment


def _learned_candidates(
    population: Sequence[RMTGPIndividual],
) -> list[RMTGPIndividual]:
    """排除 baseline 哨兵，返回真正由 GP 表达式定义的个体。"""

    return [
        individual
        for individual in population
        if not is_baseline_individual(individual)
    ]


def train(
    experiment: ExperimentConfig,
    training_cases_for_generation: Callable[[int], Sequence[EvaluationCase]],
    validation_cases: Sequence[EvaluationCase],
    *,
    validation_screening_cases: Sequence[EvaluationCase] | None = None,
    validation_gate_cases: Sequence[EvaluationCase] | None = None,
    validation_monitor_cases: Sequence[EvaluationCase] | None = None,
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
        warm_experiment = _training_experiment_for_generation(
            experiment,
            first_generation,
        )
    elif validation_cases:
        warm_case = validation_cases[0]
        warm_experiment = experiment
    else:
        raise ValueError("training 与 validation cases 不能同时为空")

    # 主进程先生成 Numba disk cache；随后 worker 只需加载，不计入每代时间。
    if experiment.runtime.aco_backend in {
        ExecutionBackend.NUMBA,
        ExecutionBackend.NUMBA_BATCH,
    }:
        baseline_cache.best_length(warm_case, warm_experiment)

    cumulative = history[-1].cumulative_wall_time if history else 0.0
    with EvaluationPool(warm_experiment) as evaluator:
        evaluator.warm(warm_case)
        active_iterations = warm_experiment.aco.iterations
        for generation in range(
            first_generation,
            experiment.gp.generations + 1,
        ):
            generation_started = perf_counter()
            generation_experiment = _training_experiment_for_generation(
                experiment,
                generation,
            )
            evaluator.set_experiment(generation_experiment)
            if generation_experiment.aco.iterations != active_iterations:
                evaluator.warm(warm_case)
                active_iterations = generation_experiment.aco.iterations
            if generation == first_generation:
                assert pending_cases is not None
                cases = pending_cases
            else:
                cases = training_cases_for_generation(generation)

            for individual in population:
                if individual.fitness.valid:
                    del individual.fitness.values
            if experiment.racing.enabled:
                evaluation, generation_experiment = (
                    _evaluate_population_with_racing(
                        population=population,
                        cases=cases,
                        generation=generation,
                        experiment=experiment,
                        evaluator=evaluator,
                        baseline_cache=baseline_cache,
                    )
                )
                active_iterations = generation_experiment.aco.iterations
            else:
                evaluation = evaluator.evaluate_population(
                    population,
                    cases,
                    baseline_cache,
                )

            if generation % experiment.gp.checkpoint_interval == 0:
                if experiment.racing.enabled:
                    selected = _racing_checkpoint_candidates(
                        population,
                        experiment.gp.checkpoint_top_k,
                    )
                else:
                    checkpoint_pool = _learned_candidates(population)
                    selected = tools.selBest(
                        checkpoint_pool,
                        min(
                            experiment.gp.checkpoint_top_k,
                            len(checkpoint_pool),
                        ),
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
                exclude_baseline=experiment.gp.baseline_anchor,
                training_aco_iterations=(
                    generation_experiment.aco.iterations
                ),
            )

            if (
                validation_monitor_cases
                and generation % experiment.validation_monitor_interval == 0
            ):
                monitor_started = perf_counter()
                evaluator.set_experiment(experiment)
                if active_iterations != experiment.aco.iterations:
                    evaluator.warm(validation_monitor_cases[0])
                    active_iterations = experiment.aco.iterations
                monitor_pool = (
                    _learned_candidates(population)
                    if experiment.gp.baseline_anchor
                    else list(population)
                )
                monitor_candidate = min(
                    monitor_pool,
                    key=(
                        _racing_selection_key
                        if experiment.racing.enabled
                        else lambda item: (
                            item.fitness.values[0],
                            item.total_nodes,
                        )
                    ),
                )
                monitor_data = evaluator.validation_data(
                    [monitor_candidate],
                    validation_monitor_cases,
                    baseline_cache,
                )[monitor_candidate.structural_hash]
                record.validation_monitor_candidate_gap_by_scale = {
                    scale: float(values.mean())
                    for scale, values in (
                        monitor_data.candidate_gap_by_scale.items()
                    )
                }
                record.validation_monitor_baseline_gap_by_scale = {
                    scale: float(values.mean())
                    for scale, values in (
                        monitor_data.baseline_gap_by_scale.items()
                    )
                }
                record.validation_monitor_delta_by_scale = {
                    scale: float(values.mean())
                    for scale, values in monitor_data.delta_by_scale.items()
                }
                record.validation_monitor_wall_time = (
                    perf_counter() - monitor_started
                )
                evaluator.set_experiment(generation_experiment)
                if active_iterations != generation_experiment.aco.iterations:
                    evaluator.warm(warm_case)
                    active_iterations = generation_experiment.aco.iterations

            breeding_started = perf_counter()
            if generation < experiment.gp.generations:
                next_population = evolve_generation(
                    population,
                    transition_pset,
                    pheromone_pset,
                    experiment.gp,
                    selection_key=(
                        _racing_selection_key
                        if experiment.racing.enabled
                        else None
                    ),
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

        if experiment.racing.enabled:
            final_selected = _racing_checkpoint_candidates(
                population,
                experiment.gp.checkpoint_top_k,
            )
        else:
            final_pool = _learned_candidates(population)
            final_selected = tools.selBest(
                final_pool,
                min(experiment.gp.checkpoint_top_k, len(final_pool)),
            )
        checkpoints.extend(deepcopy(final_selected))
        evaluator.set_experiment(experiment)
        if active_iterations != experiment.aco.iterations:
            evaluator.warm(validation_cases[0])
        validation = validate_candidates(
            checkpoints,
            experiment,
            validation_cases,
            baseline_cache,
            screening_cases=validation_screening_cases,
            gate_cases=validation_gate_cases,
            evaluator_pool=evaluator,
        )

    cpu_fp64_audit: ValidationSelection | None = None
    passed_noninferiority = validation.passed_noninferiority
    champion = validation.champion
    if (
        experiment.cpu_fp64_final_audit
        and experiment.runtime.aco_backend
        in {
            ExecutionBackend.CUDA_FUSED_FP32,
            ExecutionBackend.CUDA_TILED_V2,
        }
    ):
        selected_candidate = next(
            candidate
            for candidate in checkpoints
            if candidate.structural_hash
            == validation.selected_candidate_hash
        )
        audit_cases = (
            validation_cases
            if validation_gate_cases is None
            else validation_gate_cases
        )
        cpu_fp64_audit = _cpu_fp64_final_audit(
            selected_candidate,
            experiment,
            audit_cases,
        )
        passed_noninferiority = (
            validation.passed_noninferiority
            and cpu_fp64_audit.passed_noninferiority
        )
        if passed_noninferiority:
            champion = cpu_fp64_audit.champion
        elif validation.passed_noninferiority:
            # CPU audit 已在失败时构造了确定的 baseline fallback。
            champion = cpu_fp64_audit.champion

    result = TrainingResult(
        champion=champion,
        history=history,
        checkpoints=checkpoints,
        passed_noninferiority=passed_noninferiority,
        validation=validation,
        cpu_fp64_audit=cpu_fp64_audit,
    )
    if target is not None:
        save_training_result(result, experiment, target)
    return result
