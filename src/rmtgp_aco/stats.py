"""论文级汇总、非参数检验与层次 bootstrap。"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from statistics import fmean

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


@dataclass(frozen=True, slots=True)
class FactorialContrast:
    """Core/Full × F0/F1 的 paired factorial contrast。"""

    contrast: str
    estimate_pp: float
    lower_95: float
    upper_95: float
    runs: int
    run_instance_blocks: int
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
            (_paired_run_id(record), record.instance_id)
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
    """依次重采样 GP run、instance 与 ACO seed，估计 mean paired delta。"""

    if replicates < 100:
        raise ValueError("正式 bootstrap replicates 至少为 100")
    methods: dict[
        str,
        dict[str, dict[str, list[float]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in records:
        methods[record.method][_paired_run_id(record)][record.instance_id].append(
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
        estimates = _hierarchical_bootstrap_estimates(
            champion_map,
            replicates=replicates,
            rng=rng,
        )
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


def _hierarchical_bootstrap_estimates(
    champion_map: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成三层 bootstrap estimates，并对平衡设计使用分块向量化。

    正式 study 的每个 GP run 都有相同数量的 instances，且每个 instance
    有相同数量的 ACO seeds。此时一次数组索引可同时完成 GP run、instance
    和 seed 三层有放回采样。非平衡输入仍走通用路径，统计定义不变。
    """

    champion_ids = list(champion_map)
    arrays: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    balanced = True
    for champion_id in champion_ids:
        instance_map = champion_map[champion_id]
        try:
            values = np.asarray(
                [instance_map[key] for key in instance_map],
                dtype=np.float64,
            )
        except ValueError:
            balanced = False
            break
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            balanced = False
            break
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            balanced = False
            break
        arrays.append(values)
    if balanced and shape is not None:
        cube = np.stack(arrays, axis=0)
        champions, instances, seeds = cube.shape
        # 将每个 chunk 的主数据索引控制在约 200 万项，避免用内存换速度时
        # 破坏共享节点上的资源边界。
        chunk_size = max(
            1,
            min(
                replicates,
                2_000_000 // max(champions * instances * seeds, 1),
            ),
        )
        estimates = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, chunk_size):
            stop = min(start + chunk_size, replicates)
            count = stop - start
            sampled_champions = rng.integers(
                0,
                champions,
                size=(count, champions),
            )
            sampled_instances = rng.integers(
                0,
                instances,
                size=(count, champions, instances),
            )
            sampled_seeds = rng.integers(
                0,
                seeds,
                size=(count, champions, instances, seeds),
            )
            sampled = cube[
                sampled_champions[:, :, None, None],
                sampled_instances[:, :, :, None],
                sampled_seeds,
            ]
            estimates[start:stop] = sampled.mean(axis=(1, 2, 3))
        return estimates

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
    return estimates


def _paired_run_id(record: EvaluationRecord) -> str:
    """优先用 root seed 对齐不同方法的同一 GP replicate。"""

    if record.gp_root_seed:
        return f"seed:{record.gp_root_seed}"
    return record.gp_run_id or record.champion_id


def _factorial_value(
    values: Mapping[str, float],
    *,
    name: str,
    core_f0: str,
    core_f1: str,
    full_f0: str,
    full_f1: str,
) -> float:
    if name == "terminal_full_minus_core":
        return 0.5 * (
            values[full_f0]
            - values[core_f0]
            + values[full_f1]
            - values[core_f1]
        )
    if name == "function_f1_minus_f0":
        return 0.5 * (
            values[core_f1]
            - values[core_f0]
            + values[full_f1]
            - values[full_f0]
        )
    if name == "terminal_function_interaction":
        return (
            values[full_f1]
            - values[full_f0]
            - values[core_f1]
            + values[core_f0]
        )
    raise ValueError(f"未知 factorial contrast: {name}")


def factorial_contrasts(
    records: Sequence[EvaluationRecord],
    *,
    core_f0: str,
    core_f1: str,
    full_f0: str,
    full_f1: str,
    replicates: int = 10_000,
    seed: int = 0,
) -> list[FactorialContrast]:
    """按 run→instance→paired ACO seed 重采样 2×2 factorial effects。

    contrast 小于零表示 Full terminals 或 F1 functions 降低 reference gap。
    """

    if replicates < 100:
        raise ValueError("正式 factorial bootstrap replicates 至少为 100")
    selected_methods = (core_f0, core_f1, full_f0, full_f1)
    nested: dict[
        str,
        dict[str, dict[str, dict[int, list[float]]]],
    ] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
    )
    for record in records:
        if record.method in selected_methods:
            nested[record.method][_paired_run_id(record)][record.instance_id][
                record.seed
            ].append(record.gap_percent)
    missing = [method for method in selected_methods if method not in nested]
    if missing:
        raise ValueError(f"factorial methods 缺少记录: {missing}")

    common_runs = set(nested[core_f0])
    for method in selected_methods[1:]:
        common_runs &= set(nested[method])
    if not common_runs:
        raise ValueError("四个 factorial methods 没有共同 GP run")
    run_instances: dict[str, tuple[str, ...]] = {}
    for run_id in sorted(common_runs):
        common_instances = set(nested[core_f0][run_id])
        for method in selected_methods[1:]:
            common_instances &= set(nested[method][run_id])
        if common_instances:
            run_instances[run_id] = tuple(sorted(common_instances))
    if not run_instances:
        raise ValueError("四个 factorial methods 没有共同 run×instance blocks")

    def block_values(
        run_id: str,
        instance_id: str,
        sampled_seeds: np.ndarray | None = None,
    ) -> dict[str, float]:
        common_seeds = set(nested[core_f0][run_id][instance_id])
        for method in selected_methods[1:]:
            common_seeds &= set(nested[method][run_id][instance_id])
        if not common_seeds:
            raise ValueError(
                f"run={run_id}, instance={instance_id} 没有共同 ACO seeds"
            )
        ordered_seeds = np.asarray(sorted(common_seeds), dtype=np.int64)
        chosen = ordered_seeds if sampled_seeds is None else sampled_seeds
        return {
            method: fmean(
                fmean(nested[method][run_id][instance_id][int(seed_value)])
                for seed_value in chosen
            )
            for method in selected_methods
        }

    names = (
        "terminal_full_minus_core",
        "function_f1_minus_f0",
        "terminal_function_interaction",
    )
    observed: dict[str, float] = {}
    for name in names:
        run_means = []
        for run_id, instance_ids in run_instances.items():
            run_means.append(
                fmean(
                    _factorial_value(
                        block_values(run_id, instance_id),
                        name=name,
                        core_f0=core_f0,
                        core_f1=core_f1,
                        full_f0=full_f0,
                        full_f1=full_f1,
                    )
                    for instance_id in instance_ids
                )
            )
        observed[name] = fmean(run_means)

    rng = np.random.default_rng(seed)
    run_ids = np.asarray(sorted(run_instances), dtype=str)
    estimates = {
        name: np.empty(replicates, dtype=np.float64)
        for name in names
    }
    for replicate in range(replicates):
        sampled_runs = rng.choice(run_ids, size=run_ids.size, replace=True)
        replicate_values: dict[str, list[float]] = {
            name: [] for name in names
        }
        for sampled_run in sampled_runs:
            run_id = str(sampled_run)
            instance_ids = np.asarray(run_instances[run_id], dtype=str)
            sampled_instances = rng.choice(
                instance_ids,
                size=instance_ids.size,
                replace=True,
            )
            run_values: dict[str, list[float]] = {
                name: [] for name in names
            }
            for sampled_instance in sampled_instances:
                instance_id = str(sampled_instance)
                common_seeds = set(nested[core_f0][run_id][instance_id])
                for method in selected_methods[1:]:
                    common_seeds &= set(nested[method][run_id][instance_id])
                ordered_seeds = np.asarray(sorted(common_seeds), dtype=np.int64)
                sampled_seeds = rng.choice(
                    ordered_seeds,
                    size=ordered_seeds.size,
                    replace=True,
                )
                values = block_values(run_id, instance_id, sampled_seeds)
                for name in names:
                    run_values[name].append(
                        _factorial_value(
                            values,
                            name=name,
                            core_f0=core_f0,
                            core_f1=core_f1,
                            full_f0=full_f0,
                            full_f1=full_f1,
                        )
                    )
            for name in names:
                replicate_values[name].append(fmean(run_values[name]))
        for name in names:
            estimates[name][replicate] = fmean(replicate_values[name])

    block_count = sum(len(instances) for instances in run_instances.values())
    return [
        FactorialContrast(
            contrast=name,
            estimate_pp=float(observed[name]),
            lower_95=float(np.quantile(estimates[name], 0.025)),
            upper_95=float(np.quantile(estimates[name], 0.975)),
            runs=len(run_instances),
            run_instance_blocks=block_count,
            replicates=replicates,
        )
        for name in names
    ]


def write_statistical_report(
    path: str | Path,
    *,
    summaries: Sequence[QualitySummary],
    friedman: FriedmanResult | None,
    pairwise: Sequence[PairwiseTest],
    bootstrap: Sequence[BootstrapInterval],
    factorial: Sequence[FactorialContrast] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """保存机器可读、可直接生成论文表格的 JSON。"""

    payload = {
        "metadata": dict(metadata or {}),
        "quality_summary": [asdict(item) for item in summaries],
        "friedman": asdict(friedman) if friedman is not None else None,
        "pairwise_wilcoxon_holm": [asdict(item) for item in pairwise],
        "hierarchical_bootstrap": [asdict(item) for item in bootstrap],
        "factorial_contrasts": [
            asdict(item) for item in (factorial or ())
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
