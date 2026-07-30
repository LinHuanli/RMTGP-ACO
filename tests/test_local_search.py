"""候选表局部搜索的合法性、单调性与确定性测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rmtgp_aco.aco import solve
from rmtgp_aco.config import ACOConfig, LocalSearch
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.local_search import three_opt_first, tour_length, two_opt_first


@pytest.mark.parametrize("search", [two_opt_first, three_opt_first])
def test_reference_local_search_is_valid_monotone_and_repeatable(search) -> None:
    rng = np.random.default_rng(831)
    coords = rng.random((40, 2))
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    nearest = np.argsort(distances, axis=1)[:, 1:21]
    permutation = rng.permutation(40)
    tour = np.concatenate((permutation, permutation[:1]))

    first, first_stats = search(
        tour,
        distances,
        nearest,
        seed=17,
        instance_key=29,
        iteration=3,
        ant=4,
        candidate_size=20,
    )
    repeated, repeated_stats = search(
        tour,
        distances,
        nearest,
        seed=17,
        instance_key=29,
        iteration=3,
        ant=4,
        candidate_size=20,
    )
    assert np.array_equal(first, repeated)
    assert first_stats == repeated_stats
    assert first[0] == first[-1]
    assert np.array_equal(np.sort(first[:-1]), np.arange(40))
    assert tour_length(first, distances) <= tour_length(tour, distances) + 1e-12
    assert -1.0 <= 2.0 * first_stats.normalized_gain - 1.0 <= 1.0


def test_numba_two_opt_runs_after_construction(small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "acs",
            local_search=LocalSearch.TWO_OPT,
            iterations=2,
            ants=4,
        ),
        candidate_size=2,
        local_search_candidate_size=2,
    )
    result = solve(batch, config, seed=73, backend="numba")
    assert result.diagnostics.local_search_improved_tour_count > 0
    assert result.diagnostics.local_search_move_count > 0
    assert result.diagnostics.local_search_candidate_check_count > 0
