"""锁定 champion 后的 paired ACO 评测与长表导出。"""

from __future__ import annotations

import csv
import pickle
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from .aco import solve
from .config import ACOConfig, ExecutionBackend, RuntimeConfig
from .genetic import RMTGPIndividual, compile_individual
from .model import ProblemBatch, RunResult
from .program import TensorProgram


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """一条 instance × seed × champion 的可配对结果。"""

    method: str
    variant: str
    champion_id: str
    partition: str
    distribution: str
    scale: int
    instance_id: str
    seed: int
    best_length: float
    reference_length: float
    gap_percent: float
    baseline_length: float
    baseline_gap_percent: float
    delta_pp: float
    outcome: str
    best_iteration: int
    anytime_gap_auc: float
    wall_time_sec: float
    baseline_wall_time_sec: float
    inference_overhead_percent: float
    constructed_tours: int
    tours_per_second: float
    candidate_fallback_count: int
    uniform_fallback_count: int
    bound_clip_count: int
    gp_run_id: str = ""
    gp_root_seed: int = 0
    baseline_best_iteration: int = -1
    baseline_anytime_gap_auc: float = float("nan")
    baseline_tours_per_second: float = float("nan")


def load_champion(path: str | Path) -> RMTGPIndividual:
    """读取本仓库产生的本地 pickle artifact。

    Pickle 不具备不可信输入安全性；调用方只能加载自己生成或已审计的文件。
    """

    with Path(path).open("rb") as handle:
        champion = pickle.load(handle)
    if not isinstance(champion, RMTGPIndividual):
        raise TypeError("champion artifact 不是 RMTGPIndividual")
    return champion


def compile_champion(
    champion: RMTGPIndividual | None,
) -> tuple[TensorProgram | None, TensorProgram | None]:
    """将 champion 的两棵树编译一次，供全部测试 batch 复用。"""

    if champion is None:
        return None, None
    return compile_individual(champion)


def _batch_seed(root_seed: int, batch_number: int, replicate: int) -> int:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [root_seed, batch_number, replicate, 0x54455354]
        )
    )
    return int(rng.integers(0, 2**63 - 1))


def study_test_seed(
    root_seed: int,
    partition: str,
    batch_number: int,
    replicate: int,
) -> int:
    """生成与 GP root seed 解耦、跨 champion 共享的测试随机种子。"""

    partition_code = int.from_bytes(
        sha256(partition.encode("utf-8")).digest()[:4],
        byteorder="little",
        signed=False,
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                root_seed,
                partition_code,
                batch_number,
                replicate,
                0x54455354,
            ]
        )
    )
    return int(rng.integers(0, 2**63 - 1))


def _record_batch(
    *,
    method: str,
    champion_id: str,
    partition: str,
    distribution: str,
    batch: ProblemBatch,
    seed: int,
    candidate: RunResult,
    baseline: RunResult,
    config: ACOConfig,
    tie_tolerance: float,
    gp_run_id: str,
    gp_root_seed: int,
) -> Iterator[EvaluationRecord]:
    reference = batch.reference_length
    candidate_gap = 100.0 * (candidate.best_length - reference) / reference
    baseline_gap = 100.0 * (baseline.best_length - reference) / reference
    delta = candidate_gap - baseline_gap
    anytime_gap = 100.0 * (
        candidate.anytime_best - reference[:, None]
    ) / reference[:, None]
    anytime_auc = anytime_gap.mean(dim=1)
    baseline_anytime_gap = 100.0 * (
        baseline.anytime_best - reference[:, None]
    ) / reference[:, None]
    baseline_anytime_auc = baseline_anytime_gap.mean(dim=1)
    throughput = candidate.constructed_tours / max(candidate.wall_time_sec, 1e-12)
    baseline_throughput = baseline.constructed_tours / max(
        baseline.wall_time_sec,
        1e-12,
    )
    inference_overhead = 100.0 * (
        candidate.wall_time_sec - baseline.wall_time_sec
    ) / max(baseline.wall_time_sec, 1e-12)

    for index, instance_id in enumerate(batch.instance_ids):
        difference = float(delta[index].item())
        outcome = (
            "win"
            if difference < -tie_tolerance
            else "loss"
            if difference > tie_tolerance
            else "tie"
        )
        yield EvaluationRecord(
            method=method,
            variant=config.variant.value,
            champion_id=champion_id,
            partition=partition,
            distribution=distribution,
            scale=batch.n,
            instance_id=instance_id,
            seed=seed,
            best_length=float(candidate.best_length[index].item()),
            reference_length=float(reference[index].item()),
            gap_percent=float(candidate_gap[index].item()),
            baseline_length=float(baseline.best_length[index].item()),
            baseline_gap_percent=float(baseline_gap[index].item()),
            delta_pp=difference,
            outcome=outcome,
            best_iteration=int(candidate.best_iteration[index].item()),
            anytime_gap_auc=float(anytime_auc[index].item()),
            wall_time_sec=float(candidate.wall_time_sec),
            baseline_wall_time_sec=float(baseline.wall_time_sec),
            inference_overhead_percent=float(inference_overhead),
            constructed_tours=candidate.constructed_tours,
            tours_per_second=float(throughput),
            candidate_fallback_count=candidate.diagnostics.candidate_fallback_count,
            uniform_fallback_count=candidate.diagnostics.uniform_fallback_count,
            bound_clip_count=candidate.diagnostics.bound_clip_count,
            gp_run_id=gp_run_id,
            gp_root_seed=gp_root_seed,
            baseline_best_iteration=int(
                baseline.best_iteration[index].item()
            ),
            baseline_anytime_gap_auc=float(
                baseline_anytime_auc[index].item()
            ),
            baseline_tours_per_second=float(baseline_throughput),
        )


def records_from_paired_results(
    *,
    method: str,
    champion_id: str,
    partition: str,
    distribution: str,
    batch: ProblemBatch,
    seed: int,
    candidate: RunResult,
    baseline: RunResult,
    config: ACOConfig,
    tie_tolerance: float = 1e-12,
    gp_run_id: str | None = None,
    gp_root_seed: int = 0,
) -> list[EvaluationRecord]:
    """把已计算的 paired 结果展开为长表。

    该入口允许正式 study 在多个 GP champion 之间复用不可变 baseline，
    避免每次候选测试都重新执行同一组原始 ACO。
    """

    return list(
        _record_batch(
            method=method,
            champion_id=champion_id,
            partition=partition,
            distribution=distribution,
            batch=batch,
            seed=seed,
            candidate=candidate,
            baseline=baseline,
            config=config,
            tie_tolerance=tie_tolerance,
            gp_run_id=gp_run_id or champion_id,
            gp_root_seed=gp_root_seed,
        )
    )


def evaluate_batches(
    batches: Iterable[ProblemBatch],
    config: ACOConfig,
    *,
    method: str,
    champion_id: str,
    partition: str,
    distribution: str,
    root_seed: int,
    seeds_per_batch: int,
    transition_program: TensorProgram | None = None,
    pheromone_program: TensorProgram | None = None,
    backend: ExecutionBackend | str = ExecutionBackend.TORCH,
    runtime: RuntimeConfig | None = None,
    tie_tolerance: float = 1e-12,
    gp_run_id: str | None = None,
) -> list[EvaluationRecord]:
    """以完全相同 seed 成对运行 candidate 与原始 ACO。"""

    if seeds_per_batch < 1:
        raise ValueError("seeds_per_batch 必须为正整数")
    records: list[EvaluationRecord] = []
    is_baseline = transition_program is None and pheromone_program is None
    for batch_number, batch in enumerate(batches):
        for replicate in range(seeds_per_batch):
            seed = _batch_seed(root_seed, batch_number, replicate)
            baseline = solve(
                batch,
                config,
                seed=seed,
                backend=backend,
                runtime=runtime,
            )
            candidate = (
                baseline
                if is_baseline
                else solve(
                    batch,
                    config,
                    transition_program=transition_program,
                    pheromone_program=pheromone_program,
                    seed=seed,
                    backend=backend,
                    runtime=runtime,
                )
            )
            records.extend(
                _record_batch(
                    method=method,
                    champion_id=champion_id,
                    partition=partition,
                    distribution=distribution,
                    batch=batch,
                    seed=seed,
                    candidate=candidate,
                    baseline=baseline,
                    config=config,
                    tie_tolerance=tie_tolerance,
                    gp_run_id=gp_run_id or champion_id,
                    gp_root_seed=root_seed,
                )
            )
    return records


def write_records(
    records: Sequence[EvaluationRecord],
    path: str | Path,
) -> Path:
    """把评测结果写为标准库即可读取的 UTF-8 CSV 长表。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in EvaluationRecord.__dataclass_fields__.values()]
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    temporary.replace(target)
    return target


def read_records(paths: Iterable[str | Path]) -> list[EvaluationRecord]:
    """读取一个或多个长表，并恢复数值类型。"""

    integer_fields = {
        "scale",
        "seed",
        "best_iteration",
        "constructed_tours",
        "candidate_fallback_count",
        "uniform_fallback_count",
        "bound_clip_count",
        "gp_root_seed",
        "baseline_best_iteration",
    }
    float_fields = {
        "best_length",
        "reference_length",
        "gap_percent",
        "baseline_length",
        "baseline_gap_percent",
        "delta_pp",
        "anytime_gap_auc",
        "wall_time_sec",
        "baseline_wall_time_sec",
        "inference_overhead_percent",
        "tours_per_second",
        "baseline_anytime_gap_auc",
        "baseline_tours_per_second",
    }
    records: list[EvaluationRecord] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                values: dict[str, object] = dict(row)
                # 兼容 v0.2 长表；新结果显式保留 GP run 的配对标识。
                values.setdefault("gp_run_id", str(values["champion_id"]))
                values.setdefault("gp_root_seed", "0")
                values.setdefault("baseline_best_iteration", "-1")
                values.setdefault("baseline_anytime_gap_auc", "nan")
                values.setdefault("baseline_tours_per_second", "nan")
                for name in integer_fields:
                    values[name] = int(values[name])
                for name in float_fields:
                    values[name] = float(values[name])
                records.append(EvaluationRecord(**values))
    return records
