"""RMTGP 的 fitness、训练循环、validation selection 与 artifacts。"""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import platform
import random
from statistics import fmean
from typing import Callable, Iterable, Sequence

from deap import tools
import numpy as np
import torch
import yaml

from .aco import solve
from .config import ExperimentConfig
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
_WORKER_CASES: tuple[EvaluationCase, ...] = ()
_WORKER_BASELINES: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True, slots=True)
class FitnessBreakdown:
    """一个个体在当前 mini-batch 上的可审计 fitness 组成。"""

    fitness: float
    mean_delta_by_scale: dict[int, float]
    degradation_by_scale: dict[int, float]


@dataclass(slots=True)
class GenerationRecord:
    """一代 GP 的汇总统计。"""

    generation: int
    evaluated_unique: int
    minimum: float
    mean: float
    standard_deviation: float
    best_nodes: int
    best_hash: str


@dataclass(slots=True)
class TrainingResult:
    """一次独立 GP run 的返回值。"""

    champion: RMTGPIndividual
    history: list[GenerationRecord]
    checkpoints: list[RMTGPIndividual]
    passed_noninferiority: bool
    output_directory: Path | None = None


class BaselineCache:
    """按配置、实例 IDs 和 seed 缓存原始 ACO 结果。"""

    def __init__(self) -> None:
        self._values: dict[tuple[object, ...], torch.Tensor] = {}

    def best_length(
        self,
        case: EvaluationCase,
        experiment: ExperimentConfig,
    ) -> torch.Tensor:
        key = (
            experiment.aco.config_hash,
            tuple(case.batch.instance_ids),
            case.seed,
        )
        if key not in self._values:
            result = solve(case.batch, experiment.aco, seed=case.seed)
            self._values[key] = result.best_length.detach().cpu()
        return self._values[key].to(case.batch.device)


def baseline_relative_fitness(
    candidate_lengths: dict[int, list[torch.Tensor]],
    baseline_lengths: dict[int, list[torch.Tensor]],
    references: dict[int, list[torch.Tensor]],
    *,
    degradation_penalty: float,
) -> FitnessBreakdown:
    """实现文档定义的 scale-balanced、退化惩罚 fitness。"""

    mean_delta: dict[int, float] = {}
    degradation: dict[int, float] = {}
    terms: list[float] = []

    for scale in sorted(candidate_lengths):
        candidate = torch.cat(candidate_lengths[scale])
        baseline = torch.cat(baseline_lengths[scale])
        reference = torch.cat(references[scale])
        delta = 100.0 * (candidate - baseline) / reference
        scale_mean = float(delta.mean().item())
        scale_degradation = float(torch.clamp_min(delta, 0.0).mean().item())
        mean_delta[scale] = scale_mean
        degradation[scale] = scale_degradation
        terms.append(scale_mean + degradation_penalty * scale_degradation)

    return FitnessBreakdown(
        fitness=fmean(terms),
        mean_delta_by_scale=mean_delta,
        degradation_by_scale=degradation,
    )


class IndividualEvaluator:
    """把双树编译、嵌入 ACO 并计算 paired fitness。"""

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
        transition, pheromone = compile_individual(individual)
        candidate: dict[int, list[torch.Tensor]] = {}
        baseline: dict[int, list[torch.Tensor]] = {}
        references: dict[int, list[torch.Tensor]] = {}

        for case in self.cases:
            result = solve(
                case.batch,
                self.experiment.aco,
                transition_program=transition,
                pheromone_program=pheromone,
                seed=case.seed,
            )
            candidate.setdefault(case.scale, []).append(result.best_length)
            baseline.setdefault(case.scale, []).append(
                self.baseline_cache.best_length(case, self.experiment)
            )
            references.setdefault(case.scale, []).append(case.batch.reference_length)

        breakdown = baseline_relative_fitness(
            candidate,
            baseline,
            references,
            degradation_penalty=self.experiment.gp.degradation_penalty,
        )
        self.last_breakdown[individual.structural_hash] = breakdown
        return breakdown.fitness


def _score_with_explicit_baselines(
    individual: RMTGPIndividual,
    experiment: ExperimentConfig,
    cases: Sequence[EvaluationCase],
    baseline_values: Sequence[torch.Tensor],
) -> float:
    """在 worker 内评估个体；baseline 已由主进程按 paired seed 计算。"""

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
        )
        candidate.setdefault(case.scale, []).append(result.best_length)
        baseline.setdefault(case.scale, []).append(
            baseline_length.to(case.batch.device)
        )
        references.setdefault(case.scale, []).append(case.batch.reference_length)
    return baseline_relative_fitness(
        candidate,
        baseline,
        references,
        degradation_penalty=experiment.gp.degradation_penalty,
    ).fitness


def _initialise_evaluation_worker(
    experiment: ExperimentConfig,
    cases: tuple[EvaluationCase, ...],
    baseline_values: tuple[torch.Tensor, ...],
) -> None:
    """初始化一个 GP 个体评估进程，控制内部 PyTorch 线程数。"""

    global _WORKER_EXPERIMENT, _WORKER_CASES, _WORKER_BASELINES
    _WORKER_EXPERIMENT = experiment
    _WORKER_CASES = cases
    _WORKER_BASELINES = baseline_values
    torch.set_num_threads(experiment.runtime.torch_threads)
    torch.use_deterministic_algorithms(
        experiment.runtime.deterministic_algorithms
    )


def _score_in_worker(individual: RMTGPIndividual) -> float:
    """ProcessPoolExecutor 的顶层可 pickle worker 函数。"""

    if _WORKER_EXPERIMENT is None:
        raise RuntimeError("评估 worker 尚未初始化")
    return _score_with_explicit_baselines(
        individual,
        _WORKER_EXPERIMENT,
        _WORKER_CASES,
        _WORKER_BASELINES,
    )


def evaluate_invalid_population(
    population: Sequence[RMTGPIndividual],
    experiment: ExperimentConfig,
    cases: Sequence[EvaluationCase],
    baseline_cache: BaselineCache,
) -> int:
    """按 structural hash 去重，并可在 CPU 上跨进程评估 GP 个体。"""

    representatives: dict[str, RMTGPIndividual] = {}
    waiting: dict[str, list[RMTGPIndividual]] = {}
    for individual in population:
        if individual.fitness.valid:
            continue
        key = individual.structural_hash
        representatives.setdefault(key, individual)
        waiting.setdefault(key, []).append(individual)
    if not representatives:
        return 0

    ordered_keys = list(representatives)
    individuals = [representatives[key] for key in ordered_keys]
    baseline_values = tuple(
        baseline_cache.best_length(case, experiment).detach().cpu()
        for case in cases
    )

    if experiment.runtime.processes == 1:
        scores = [
            _score_with_explicit_baselines(
                individual,
                experiment,
                cases,
                baseline_values,
            )
            for individual in individuals
        ]
    else:
        if experiment.aco.device != "cpu":
            raise ValueError(
                "CUDA 模式下 processes 必须为 1；请依靠 tensor batch 并行"
            )
        context = mp.get_context(
            experiment.runtime.multiprocessing_start_method
        )
        with ProcessPoolExecutor(
            max_workers=experiment.runtime.processes,
            mp_context=context,
            initializer=_initialise_evaluation_worker,
            initargs=(experiment, tuple(cases), baseline_values),
        ) as executor:
            scores = list(executor.map(_score_in_worker, individuals, chunksize=1))

    for key, score in zip(ordered_keys, scores, strict=True):
        for individual in waiting[key]:
            individual.fitness.values = (float(score),)
    return len(ordered_keys)


def _normal_upper_bound(values: np.ndarray) -> float:
    """单侧 95% 正态近似上界；单样本时保守返回该值。"""

    if values.size <= 1:
        return float(values.mean())
    standard_error = values.std(ddof=1) / np.sqrt(values.size)
    return float(values.mean() + 1.645 * standard_error)


def validate_candidates(
    candidates: Iterable[RMTGPIndividual],
    experiment: ExperimentConfig,
    validation_cases: Sequence[EvaluationCase],
    baseline_cache: BaselineCache,
) -> tuple[RMTGPIndividual, bool]:
    """在 validation 上选择 champion 并执行逐规模 non-inferiority gate。"""

    unique = {
        individual.structural_hash: individual
        for individual in candidates
    }
    scored: list[tuple[float, int, RMTGPIndividual, dict[int, np.ndarray]]] = []

    for individual in unique.values():
        transition, pheromone = compile_individual(individual)
        deltas: dict[int, list[np.ndarray]] = {}
        for case in validation_cases:
            result = solve(
                case.batch,
                experiment.aco,
                transition_program=transition,
                pheromone_program=pheromone,
                seed=case.seed,
            )
            baseline = baseline_cache.best_length(case, experiment)
            delta = (
                100.0
                * (result.best_length - baseline)
                / case.batch.reference_length
            )
            deltas.setdefault(case.scale, []).append(
                delta.detach().cpu().numpy()
            )
        arrays = {
            scale: np.concatenate(values)
            for scale, values in deltas.items()
        }
        macro = fmean(float(values.mean()) for values in arrays.values())
        scored.append((macro, individual.total_nodes, individual, arrays))

    scored.sort(key=lambda item: (item[0], item[1]))
    best_macro = scored[0][0]
    near_ties = [
        item
        for item in scored
        if item[0] <= best_macro + 0.01
    ]
    _, _, champion, arrays = min(
        near_ties,
        key=lambda item: (item[1], item[0]),
    )
    passed = all(
        _normal_upper_bound(values) <= experiment.noninferiority_tolerance
        for values in arrays.values()
    )
    if passed:
        return champion.clone(), True

    transition_pset, pheromone_pset = create_primitive_sets(
        transition_profile=experiment.gp.transition_profile,
        function_profile=experiment.gp.function_profile,
        transition_terminals=experiment.gp.transition_terminals,
        pheromone_terminals=experiment.gp.pheromone_terminals,
    )
    baseline = make_individual(
        transition_pset,
        pheromone_pset,
        experiment.gp,
        mode="baseline",
    )
    baseline.fitness.values = (0.0,)
    return baseline, False


def _generation_record(
    generation: int,
    population: Sequence[RMTGPIndividual],
    evaluated_unique: int,
) -> GenerationRecord:
    values = np.asarray([item.fitness.values[0] for item in population], dtype=float)
    best = min(population, key=lambda item: (item.fitness.values[0], item.total_nodes))
    return GenerationRecord(
        generation=generation,
        evaluated_unique=evaluated_unique,
        minimum=float(values.min()),
        mean=float(values.mean()),
        standard_deviation=float(values.std()),
        best_nodes=best.total_nodes,
        best_hash=best.structural_hash,
    )


def _environment_payload() -> dict[str, object]:
    """记录足以解释后端差异的基础环境。"""

    try:
        import deap
        deap_version = deap.__version__
    except AttributeError:
        deap_version = "unknown"
    try:
        import numba
        numba_version = numba.__version__
    except (ImportError, ModuleNotFoundError):
        numba_version = "unavailable"
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "deap": deap_version,
        "numba": numba_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def save_training_result(
    result: TrainingResult,
    experiment: ExperimentConfig,
    output_directory: str | Path,
) -> Path:
    """保存小型可复现 artifact；大型 tensor checkpoints 由上层策略控制。"""

    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.yaml").write_text(
        yaml.safe_dump(
            experiment.stable_dict(),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (target / "environment.json").write_text(
        json.dumps(_environment_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_payload = [asdict(record) for record in result.history]
    (target / "training_metrics.json").write_text(
        json.dumps(history_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "training_metrics.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in history_payload
        ),
        encoding="utf-8",
    )
    (target / "champion_expression.txt").write_text(
        (
            f"transition: {result.champion.transition_tree}\n"
            f"pheromone: {result.champion.pheromone_tree}\n"
            f"passed_noninferiority: {result.passed_noninferiority}\n"
        ),
        encoding="utf-8",
    )
    with (target / "champion.pkl").open("wb") as handle:
        pickle.dump(result.champion, handle)
    (target / "validation_summary.csv").write_text(
        (
            "champion_hash,total_nodes,passed_noninferiority\n"
            f"{result.champion.structural_hash},{result.champion.total_nodes},"
            f"{str(result.passed_noninferiority).lower()}\n"
        ),
        encoding="utf-8",
    )
    checkpoint_directory = target / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    for index, checkpoint in enumerate(result.checkpoints):
        with (checkpoint_directory / f"candidate_{index:04d}.pkl").open("wb") as handle:
            pickle.dump(checkpoint, handle)
    result.output_directory = target
    return target


def train(
    experiment: ExperimentConfig,
    training_cases_for_generation: Callable[[int], Sequence[EvaluationCase]],
    validation_cases: Sequence[EvaluationCase],
    *,
    output_directory: str | Path | None = None,
    progress_callback: Callable[[GenerationRecord], None] | None = None,
) -> TrainingResult:
    """执行一次独立、可复现的 Strongly Typed Multi-Tree GP run。"""

    random.seed(experiment.root_seed)
    np.random.seed(experiment.root_seed % (2**32))
    torch.manual_seed(experiment.root_seed)

    population, transition_pset, pheromone_pset = initialise_population(experiment.gp)
    baseline_cache = BaselineCache()
    history: list[GenerationRecord] = []
    checkpoints: list[RMTGPIndividual] = []

    for generation in range(1, experiment.gp.generations + 1):
        cases = training_cases_for_generation(generation)
        # 每代使用不同的 mini-batch。即使是 elite 或 reproduction clone，
        # 其上一代 fitness 也不再可比，因此必须让整代个体共享当前评估环境。
        for individual in population:
            if individual.fitness.valid:
                del individual.fitness.values
        evaluated_unique = evaluate_invalid_population(
            population,
            experiment,
            cases,
            baseline_cache,
        )
        record = _generation_record(generation, population, evaluated_unique)
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

        if generation % experiment.gp.checkpoint_interval == 0:
            selected = tools.selBest(
                population,
                experiment.gp.checkpoint_top_k,
            )
            checkpoints.extend(deepcopy(selected))

        if generation < experiment.gp.generations:
            population = evolve_generation(
                population,
                transition_pset,
                pheromone_pset,
                experiment.gp,
            )

    # 最后一代即使不落在 checkpoint interval，也必须进入候选集合。
    checkpoints.extend(
        deepcopy(tools.selBest(population, experiment.gp.checkpoint_top_k))
    )
    champion, passed = validate_candidates(
        checkpoints,
        experiment,
        validation_cases,
        baseline_cache,
    )
    result = TrainingResult(
        champion=champion,
        history=history,
        checkpoints=checkpoints,
        passed_noninferiority=passed,
    )
    if output_directory is not None:
        save_training_result(result, experiment, output_directory)
    return result
