"""论文级汇总、非参数检验与层次 bootstrap。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from math import ceil
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from .evaluation import EvaluationRecord


@dataclass(frozen=True, slots=True)
class QualitySummary:
    """先聚合 ACO seeds 后得到的方法级质量与效率摘要。"""

    method: str
    variant: str
    partition: str
    distribution: str
    scale: int
    champions: int
    instances: int
    blocks: int
    mean_gap_percent: float
    median_gap_percent: float
    standard_deviation: float
    q1_gap_percent: float
    q3_gap_percent: float
    mean_delta_pp: float
    median_delta_pp: float
    win_rate: float
    tie_rate: float
    loss_rate: float
    worse_than_baseline_rate: float
    cvar_worst_10_percent: float
    reference_hit_rate: float
    mean_anytime_gap_auc: float
    mean_wall_time_sec: float
    mean_tours_per_second: float
    mean_inference_overhead_percent: float


@dataclass(frozen=True, slots=True)
class PairwiseTest:
    """Holm 校正后的 paired Wilcoxon 结果。"""

    reference_method: str
    compared_method: str
    blocks: int
    mean_difference_pp: float
    wilcoxon_statistic: float
    raw_p_value: float
    holm_p_value: float
    rank_biserial: float


@dataclass(frozen=True, slots=True)
class FriedmanResult:
    """共同 instance blocks 上的 Friedman omnibus test。"""

    methods: tuple[str, ...]
    blocks: int
    statistic: float
    p_value: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """层次重采样得到的 mean paired delta 置信区间。"""

    method: str
    estimate: float
    lower_95: float
    upper_95: float
    replicates: int


def _group_key(record: EvaluationRecord) -> tuple[str, str, str, str, int]:
    return (
        record.method,
        record.variant,
        record.partition,
        record.distribution,
        record.scale,
    )


def summarize_quality(
    records: Iterable[EvaluationRecord],
    *,
    tie_tolerance: float = 1e-12,
    reference_tolerance: float = 1e-9,
) -> list[QualitySummary]:
    """按 champion × instance 先平均 seeds，再汇总方法分布。"""

    groups: dict[
        tuple[str, str, str, str, int],
        dict[tuple[str, str], list[EvaluationRecord]],
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups[_group_key(record)][
            (record.champion_id, record.instance_id)
        ].append(record)

    summaries: list[QualitySummary] = []
    for key, blocks in sorted(groups.items()):
        block_rows = list(blocks.values())
        gaps = np.asarray(
            [fmean(row.gap_percent for row in values) for values in block_rows]
        )
        deltas = np.asarray(
            [fmean(row.delta_pp for row in values) for values in block_rows]
        )
        auc = np.asarray(
            [fmean(row.anytime_gap_auc for row in values) for values in block_rows]
        )
        wall = np.asarray(
            [fmean(row.wall_time_sec for row in values) for values in block_rows]
        )
        throughput = np.asarray(
            [fmean(row.tours_per_second for row in values) for values in block_rows]
        )
        overhead = np.asarray(
            [
                fmean(row.inference_overhead_percent for row in values)
                for values in block_rows
            ]
        )
        tail_count = max(1, ceil(0.10 * gaps.size))
        worst_tail = np.sort(gaps)[-tail_count:]
        method, variant, partition, distribution, scale = key
        summaries.append(
            QualitySummary(
                method=method,
                variant=variant,
                partition=partition,
                distribution=distribution,
                scale=scale,
                champions=len({champion for champion, _ in blocks}),
                instances=len({instance for _, instance in blocks}),
                blocks=len(blocks),
                mean_gap_percent=float(gaps.mean()),
                median_gap_percent=float(np.median(gaps)),
                standard_deviation=float(gaps.std(ddof=1) if gaps.size > 1 else 0.0),
                q1_gap_percent=float(np.quantile(gaps, 0.25)),
                q3_gap_percent=float(np.quantile(gaps, 0.75)),
                mean_delta_pp=float(deltas.mean()),
                median_delta_pp=float(np.median(deltas)),
                win_rate=float(np.mean(deltas < -tie_tolerance)),
                tie_rate=float(np.mean(np.abs(deltas) <= tie_tolerance)),
                loss_rate=float(np.mean(deltas > tie_tolerance)),
                worse_than_baseline_rate=float(np.mean(deltas > tie_tolerance)),
                cvar_worst_10_percent=float(worst_tail.mean()),
                reference_hit_rate=float(np.mean(gaps <= reference_tolerance)),
                mean_anytime_gap_auc=float(auc.mean()),
                mean_wall_time_sec=float(wall.mean()),
                mean_tours_per_second=float(throughput.mean()),
                mean_inference_overhead_percent=float(overhead.mean()),
            )
        )
    return summaries


def _instance_method_matrix(
    records: Sequence[EvaluationRecord],
) -> tuple[list[str], list[str], np.ndarray]:
    """跨 champion 与 seed 聚合，构造 common-instance paired matrix。"""

    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        values[record.method][record.instance_id].append(record.gap_percent)
    methods = sorted(values)
    if len(methods) < 2:
        raise ValueError("至少需要两个方法")
    common = set(values[methods[0]])
    for method in methods[1:]:
        common &= set(values[method])
    instances = sorted(common)
    if not instances:
        raise ValueError("方法之间没有共同 instance blocks")
    matrix = np.asarray(
        [
            [fmean(values[method][instance]) for instance in instances]
            for method in methods
        ],
        dtype=np.float64,
    )
    return methods, instances, matrix


def friedman_test(records: Sequence[EvaluationRecord]) -> FriedmanResult:
    """对三个及以上方法执行 Friedman omnibus test。"""

    methods, instances, matrix = _instance_method_matrix(records)
    if len(methods) < 3:
        raise ValueError("Friedman test 至少需要三个方法")
    result = stats.friedmanchisquare(*(matrix[index] for index in range(len(methods))))
    return FriedmanResult(
        methods=tuple(methods),
        blocks=len(instances),
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
    )


def _rank_biserial(differences: np.ndarray) -> float:
    """paired differences 的 signed-rank rank-biserial effect size。"""

    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / ranks.sum())


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def paired_wilcoxon_holm(
    records: Sequence[EvaluationRecord],
    *,
    reference_method: str,
    compared_methods: Sequence[str] | None = None,
) -> list[PairwiseTest]:
    """以 instance 为 paired block，执行 Wilcoxon 并作 Holm 校正。

    ``mean_difference_pp`` 与 ``rank_biserial`` 使用
    ``reference - compared``；负值表示 reference 的 gap 更小。
    """

    methods, instances, matrix = _instance_method_matrix(records)
    try:
        reference_index = methods.index(reference_method)
    except ValueError as exc:
        raise ValueError(f"未找到 reference method: {reference_method}") from exc
    requested = (
        [method for method in methods if method != reference_method]
        if compared_methods is None
        else list(compared_methods)
    )
    raw: list[tuple[str, np.ndarray, float, float]] = []
    for method in requested:
        if method == reference_method:
            continue
        try:
            compared_index = methods.index(method)
        except ValueError as exc:
            raise ValueError(f"未找到 compared method: {method}") from exc
        differences = matrix[reference_index] - matrix[compared_index]
        if np.all(differences == 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = stats.wilcoxon(
                differences,
                zero_method="pratt",
                alternative="two-sided",
                method="auto",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
        raw.append((method, differences, statistic, p_value))
    adjusted = _holm_adjust([item[3] for item in raw])
    return [
        PairwiseTest(
            reference_method=reference_method,
            compared_method=method,
            blocks=len(instances),
            mean_difference_pp=float(differences.mean()),
            wilcoxon_statistic=statistic,
            raw_p_value=p_value,
            holm_p_value=corrected,
            rank_biserial=_rank_biserial(differences),
        )
        for (method, differences, statistic, p_value), corrected in zip(
            raw,
            adjusted,
            strict=True,
        )
    ]


def hierarchical_bootstrap_delta(
    records: Sequence[EvaluationRecord],
    *,
    replicates: int = 10_000,
    seed: int = 0,
) -> list[BootstrapInterval]:
    """依次重采样 champion、instance 与 ACO seed，估计 mean paired delta。"""

    if replicates < 100:
        raise ValueError("正式 bootstrap replicates 至少为 100")
    methods: dict[
        str,
        dict[str, dict[str, list[float]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        methods[record.method][record.champion_id][record.instance_id].append(
            record.delta_pp
        )

    rng = np.random.default_rng(seed)
    intervals: list[BootstrapInterval] = []
    for method, champion_map in sorted(methods.items()):
        champion_ids = list(champion_map)
        if not champion_ids:
            continue
        observed = fmean(
            fmean(seed_values)
            for instance_map in champion_map.values()
            for seed_values in instance_map.values()
        )
        estimates = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            sampled_champions = rng.choice(
                champion_ids,
                size=len(champion_ids),
                replace=True,
            )
            champion_means: list[float] = []
            for champion_id in sampled_champions:
                instance_map = champion_map[str(champion_id)]
                instance_ids = list(instance_map)
                sampled_instances = rng.choice(
                    instance_ids,
                    size=len(instance_ids),
                    replace=True,
                )
                instance_means: list[float] = []
                for instance_id in sampled_instances:
                    seed_values = np.asarray(
                        instance_map[str(instance_id)],
                        dtype=np.float64,
                    )
                    sampled_seeds = rng.choice(
                        seed_values,
                        size=seed_values.size,
                        replace=True,
                    )
                    instance_means.append(float(sampled_seeds.mean()))
                champion_means.append(fmean(instance_means))
            estimates[replicate] = fmean(champion_means)
        lower, upper = np.quantile(estimates, [0.025, 0.975])
        intervals.append(
            BootstrapInterval(
                method=method,
                estimate=float(observed),
                lower_95=float(lower),
                upper_95=float(upper),
                replicates=replicates,
            )
        )
    return intervals


def write_statistical_report(
    path: str | Path,
    *,
    summaries: Sequence[QualitySummary],
    friedman: FriedmanResult | None,
    pairwise: Sequence[PairwiseTest],
    bootstrap: Sequence[BootstrapInterval],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """保存机器可读、可直接生成论文表格的 JSON。"""

    payload = {
        "metadata": dict(metadata or {}),
        "quality_summary": [asdict(item) for item in summaries],
        "friedman": asdict(friedman) if friedman is not None else None,
        "pairwise_wilcoxon_holm": [asdict(item) for item in pairwise],
        "hierarchical_bootstrap": [asdict(item) for item in bootstrap],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
