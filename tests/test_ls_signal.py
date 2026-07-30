"""2-opt 学习信号统计量的确定性单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from rmtgp_aco.ls_signal import (
    compression_statistics,
    edge_difference_survival,
    edge_retention,
    signal_to_noise_statistics,
    spearman_by_context,
)


def test_edge_retention_and_difference_survival() -> None:
    baseline_pre = np.asarray([[0, 1, 2, 3, 0]])
    baseline_post = np.asarray([[0, 1, 3, 2, 0]])
    candidate_pre = np.asarray(
        [
            [[0, 2, 1, 3, 0]],
            [[0, 1, 2, 3, 0]],
        ]
    )
    candidate_post = np.asarray(
        [
            [[0, 2, 1, 3, 0]],
            [[0, 1, 3, 2, 0]],
        ]
    )
    retention = edge_retention(candidate_pre, candidate_post)
    np.testing.assert_allclose(retention, [[1.0], [0.5]])

    difference = edge_difference_survival(
        candidate_pre,
        candidate_post,
        baseline_pre,
        baseline_post,
    )
    # 第一个候选 pre/post 均有 4 条相对 baseline 的差异边，其中两条存活。
    assert difference.pre_difference_edges[0, 0] == 4
    assert difference.post_difference_edges[0, 0] == 4
    assert difference.survived_difference_edges[0, 0] == 2
    assert difference.survival[0, 0] == pytest.approx(0.5)
    # 第二个候选 construction 与 baseline 完全相同，survival 不定义。
    assert np.isnan(difference.survival[1, 0])


def test_compression_and_spearman_are_computed_across_programs() -> None:
    pre = np.asarray(
        [
            [[0.0, 0.0]],
            [[1.0, 2.0]],
            [[2.0, 4.0]],
        ]
    )
    post = 0.5 * pre + 3.0
    compression = compression_statistics(pre, post)
    assert compression.ratio_of_mean_variances == pytest.approx(0.25)
    np.testing.assert_allclose(compression.ratios_by_context, 0.25)
    np.testing.assert_allclose(spearman_by_context(pre, post), 1.0)


def test_signal_to_noise_uses_seed_variance() -> None:
    # program effect 的方差为 1；每个 program 的两个 seed 围绕均值 ±1，
    # sample variance 为 2，所以 SNR=0.5。
    values = np.asarray(
        [
            [[-1.0, -1.0], [1.0, 1.0]],
            [[0.0, 0.0], [2.0, 2.0]],
            [[1.0, 1.0], [3.0, 3.0]],
        ]
    )
    statistics = signal_to_noise_statistics(values)
    assert statistics.ratio_of_mean_variances == pytest.approx(0.5)
    np.testing.assert_allclose(statistics.ratios_by_instance, 0.5)
