"""2-opt 对 GP 学习信号影响的可复用统计量。

本模块只处理已经生成的数值数组和 tour，不调用 ACO。这样可以分别验证
搜索内核和统计定义，并使 CPU/CUDA 审计使用完全相同的后处理。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _numpy(value: ArrayLike) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tour_edges(tour: np.ndarray) -> set[tuple[int, int]]:
    """把闭合 tour 转为无向边集合。"""

    left = tour[:-1]
    right = tour[1:]
    return {
        (min(int(first), int(second)), max(int(first), int(second)))
        for first, second in zip(left, right, strict=True)
    }


def edge_retention(
    pre_tours: ArrayLike,
    post_tours: ArrayLike,
) -> np.ndarray:
    """计算每条 tour 经局部搜索后保留的 construction edge 比例。

    输入 shape 必须相同，最后一维是闭合 tour 的 ``n+1`` 个城市。返回值
    保留全部前导维度。
    """

    pre = _numpy(pre_tours)
    post = _numpy(post_tours)
    if pre.shape != post.shape or pre.ndim < 2:
        raise ValueError("pre/post tours 必须具有相同的 [...,n+1] shape")
    n = pre.shape[-1] - 1
    if n < 2:
        raise ValueError("tour 至少需要两个城市")
    flat_pre = pre.reshape(-1, n + 1)
    flat_post = post.reshape(-1, n + 1)
    values = np.empty(flat_pre.shape[0], dtype=np.float64)
    for index, (before, after) in enumerate(
        zip(flat_pre, flat_post, strict=True)
    ):
        values[index] = len(
            _tour_edges(before) & _tour_edges(after)
        ) / float(n)
    return values.reshape(pre.shape[:-1])


@dataclass(frozen=True, slots=True)
class EdgeDifferenceResult:
    """residual 与 baseline 的边差异在 2-opt 后的存活情况。"""

    survival: np.ndarray
    pre_difference_edges: np.ndarray
    post_difference_edges: np.ndarray
    survived_difference_edges: np.ndarray


def edge_difference_survival(
    candidate_pre_tours: ArrayLike,
    candidate_post_tours: ArrayLike,
    baseline_pre_tours: ArrayLike,
    baseline_post_tours: ArrayLike,
) -> EdgeDifferenceResult:
    """计算 paired residual-vs-baseline edge difference survival。

    candidate shape 为 ``[P,...,n+1]``；baseline shape 为
    ``[...,n+1]``。当 construction 阶段没有差异时，分母为零且 survival
    记为 ``NaN``，从而不会把“没有产生行为差异”误记为全部被 2-opt 擦除。
    """

    candidate_pre = _numpy(candidate_pre_tours)
    candidate_post = _numpy(candidate_post_tours)
    baseline_pre = _numpy(baseline_pre_tours)
    baseline_post = _numpy(baseline_post_tours)
    if candidate_pre.shape != candidate_post.shape:
        raise ValueError("candidate pre/post tour shape 不一致")
    if baseline_pre.shape != baseline_post.shape:
        raise ValueError("baseline pre/post tour shape 不一致")
    if candidate_pre.shape[1:] != baseline_pre.shape:
        raise ValueError(
            "candidate 必须为 [P,...,n+1]，baseline 必须为 [...,n+1]"
        )
    n = candidate_pre.shape[-1] - 1
    candidate_count = candidate_pre.shape[0]
    contexts = int(np.prod(candidate_pre.shape[1:-1], dtype=np.int64))
    flat_candidate_pre = candidate_pre.reshape(candidate_count, contexts, n + 1)
    flat_candidate_post = candidate_post.reshape(
        candidate_count,
        contexts,
        n + 1,
    )
    flat_baseline_pre = baseline_pre.reshape(contexts, n + 1)
    flat_baseline_post = baseline_post.reshape(contexts, n + 1)
    pre_counts = np.empty((candidate_count, contexts), dtype=np.int32)
    post_counts = np.empty_like(pre_counts)
    survived_counts = np.empty_like(pre_counts)
    survival = np.full((candidate_count, contexts), np.nan, dtype=np.float64)

    for context in range(contexts):
        baseline_pre_edges = _tour_edges(flat_baseline_pre[context])
        baseline_post_edges = _tour_edges(flat_baseline_post[context])
        for program in range(candidate_count):
            pre_difference = (
                _tour_edges(flat_candidate_pre[program, context])
                ^ baseline_pre_edges
            )
            post_difference = (
                _tour_edges(flat_candidate_post[program, context])
                ^ baseline_post_edges
            )
            survived = pre_difference & post_difference
            pre_counts[program, context] = len(pre_difference)
            post_counts[program, context] = len(post_difference)
            survived_counts[program, context] = len(survived)
            if pre_difference:
                survival[program, context] = (
                    len(survived) / len(pre_difference)
                )

    output_shape = candidate_pre.shape[:-1]
    return EdgeDifferenceResult(
        survival=survival.reshape(output_shape),
        pre_difference_edges=pre_counts.reshape(output_shape),
        post_difference_edges=post_counts.reshape(output_shape),
        survived_difference_edges=survived_counts.reshape(output_shape),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """返回从 1 开始的 tie-aware average ranks。"""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        # 对应 1-based ranks position+1 ... end。
        average = 0.5 * ((position + 1) + end)
        ranks[order[position:end]] = average
        position = end
    return ranks


def spearman_by_context(
    pre_values: ArrayLike,
    post_values: ArrayLike,
) -> np.ndarray:
    """按 program 维计算 pre/post Spearman 相关。

    输入 shape 为 ``[P,...]``，返回 ``[...]``。某一 context 的任一向量为
    常数时，相关系数未定义并记为 ``NaN``。
    """

    pre = np.asarray(_numpy(pre_values), dtype=np.float64)
    post = np.asarray(_numpy(post_values), dtype=np.float64)
    if pre.shape != post.shape or pre.ndim < 2:
        raise ValueError("pre/post values 必须具有相同的 [P,...] shape")
    programs = pre.shape[0]
    contexts = int(np.prod(pre.shape[1:], dtype=np.int64))
    flat_pre = pre.reshape(programs, contexts)
    flat_post = post.reshape(programs, contexts)
    correlations = np.full(contexts, np.nan, dtype=np.float64)
    for context in range(contexts):
        first = _average_ranks(flat_pre[:, context])
        second = _average_ranks(flat_post[:, context])
        first -= first.mean()
        second -= second.mean()
        denominator = float(
            np.sqrt(np.dot(first, first) * np.dot(second, second))
        )
        if denominator > 0.0:
            correlations[context] = float(
                np.dot(first, second) / denominator
            )
    return correlations.reshape(pre.shape[1:])


@dataclass(frozen=True, slots=True)
class CompressionStatistics:
    """跨 GP programs 的 2-opt 方差压缩统计。"""

    ratio_of_mean_variances: float
    ratios_by_context: np.ndarray
    pre_variance_by_context: np.ndarray
    post_variance_by_context: np.ndarray


def compression_statistics(
    pre_values: ArrayLike,
    post_values: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> CompressionStatistics:
    """计算 ``Var_program(post)/(Var_program(pre)+epsilon)``。"""

    pre = np.asarray(_numpy(pre_values), dtype=np.float64)
    post = np.asarray(_numpy(post_values), dtype=np.float64)
    if pre.shape != post.shape or pre.ndim < 2 or pre.shape[0] < 2:
        raise ValueError("compression 要求相同 [P,...] shape 且 P>=2")
    pre_variance = np.var(pre, axis=0, ddof=1)
    post_variance = np.var(post, axis=0, ddof=1)
    ratios = post_variance / (pre_variance + epsilon)
    return CompressionStatistics(
        ratio_of_mean_variances=float(
            np.mean(post_variance) / (np.mean(pre_variance) + epsilon)
        ),
        ratios_by_context=ratios,
        pre_variance_by_context=pre_variance,
        post_variance_by_context=post_variance,
    )


@dataclass(frozen=True, slots=True)
class SignalToNoiseStatistics:
    """program 间信号与同 program 多 seed 噪声之比。"""

    ratio_of_mean_variances: float
    ratios_by_instance: np.ndarray
    between_program_variance: np.ndarray
    within_program_variance: np.ndarray


def signal_to_noise_statistics(
    values: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> SignalToNoiseStatistics:
    """估计 paired metric 的 SNR。

    输入 shape 为 ``[P,S,I]``。``P`` 是 residual program，``S`` 是 ACO
    seed，``I`` 是 instance。调用方应先减去同 seed、同 instance baseline，
    以消除实例难度和公共随机波动。
    """

    array = np.asarray(_numpy(values), dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("SNR 要求 [P,S,I] shape，且 P>=2、S>=2")
    program_means = np.mean(array, axis=1)
    between = np.var(program_means, axis=0, ddof=1)
    within = np.mean(np.var(array, axis=1, ddof=1), axis=0)
    ratios = between / (within + epsilon)
    return SignalToNoiseStatistics(
        ratio_of_mean_variances=float(
            np.mean(between) / (np.mean(within) + epsilon)
        ),
        ratios_by_instance=ratios,
        between_program_variance=between,
        within_program_variance=within,
    )


def finite_summary(values: ArrayLike) -> dict[str, float | int]:
    """输出不会被 NaN 污染的简洁描述统计。"""

    flat = np.asarray(_numpy(values), dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {
            "count": int(flat.size),
            "finite_count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "standard_deviation": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }
    return {
        "count": int(flat.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "standard_deviation": float(np.std(finite, ddof=1))
        if finite.size > 1
        else 0.0,
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }
