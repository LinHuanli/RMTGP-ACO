"""机制可视化的无 GPU 单元测试。"""
import numpy as np

from control_experiments.mmas_ls.mechanism_visualization import (
    BEHAVIORS,
    REGIMES,
    SAMPLE_ITERATIONS,
)
from control_experiments.mmas_ls.visualization_analysis import (
    _aggregate_instances,
    _candidate_metrics,
    _directed_mask,
    _edge_codes,
    _summarize_instances,
)


def test_replay_design_has_complete_factorial_and_frames():
    assert len(REGIMES) == 4
    assert len(BEHAVIORS) == 2
    assert len(SAMPLE_ITERATIONS) == 201
    assert SAMPLE_ITERATIONS[0] == 1 and SAMPLE_ITERATIONS[-1] == 5000
    assert REGIMES["native_mmas"][1] == "mmas_native"


def test_undirected_source_support_is_rotation_and_direction_invariant():
    first = np.array([0, 1, 2, 3, 0])
    rotated = np.array([2, 3, 0, 1, 2])
    reversed_tour = np.array([0, 3, 2, 1, 0])
    np.testing.assert_array_equal(np.sort(_edge_codes(first, 4)), np.sort(_edge_codes(rotated, 4)))
    np.testing.assert_array_equal(
        np.sort(_edge_codes(first, 4)), np.sort(_edge_codes(reversed_tour, 4))
    )
    mask = _directed_mask(set(_edge_codes(first, 4)), 4)
    assert mask.sum() == 8
    np.testing.assert_array_equal(mask, mask.T)


def test_candidate_landscape_metrics_for_uniform_pheromone():
    tau = np.ones((5, 5), dtype=np.float32)
    nearest = np.array([[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]])
    result = _candidate_metrics(tau, nearest, 1.0)
    assert np.isclose(result["candidate_effective_arcs"], 10)
    assert abs(result["candidate_gini"]) < 1e-12
    assert result["floor_fraction"] == 1


def test_seed_and_expression_rows_are_averaged_before_instance_interval():
    rows = []
    for instance, base in ((0, 1.0), (1, 3.0)):
        for replicate in range(3):
            for expression in range(2):
                rows.append({
                    "instance": instance,
                    "condition": "固定条件",
                    "replicate": replicate,
                    "expression": expression,
                    "effect": base + replicate + expression,
                })
    per_instance = _aggregate_instances(rows, ("condition",), ("effect",))
    assert len(per_instance) == 2
    np.testing.assert_allclose([row["effect"] for row in per_instance], [2.5, 4.5])
    summary = _summarize_instances(per_instance, ("condition",), ("effect",))
    assert len(summary) == 1 and summary[0]["instances"] == 2
    assert summary[0]["mean"] == 3.5
