"""组件归因报告与新队列的 CPU 验收，不占用 GPU。"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from control_experiments.mmas_ls.explanation_labels import benefit_interval, component_columns
from control_experiments.mmas_ls.explanation_analysis import canonical_edges, colony_structure, fixed_context_probabilities, window_metrics, WINDOW_FIELDS
from control_experiments.mmas_ls.explanation_campaign import tasks, confirmation_contrasts, conditions, CONFIRMATION_ORDER
from control_experiments.mmas_ls.explanation_forks import structure_difference, branch_specs
from rmtgp_aco.mechanisms import MechanismConfig


def test_benefit_direction_and_full_labels():
    assert benefit_interval({"mean_pp": -.3, "ci_low_pp": -.5, "ci_high_pp": -.1}) == (.3, .1, .5)
    assert component_columns("C110") == {"重启": "启用", "信息素下界保护": "启用", "历史路径强化": "关闭"}
    with pytest.raises(ValueError):
        component_columns("invalid")


def test_freeze_sizes_and_no_retraining():
    all_tasks = tasks()
    assert len({t["id"] for t in all_tasks}) == len(all_tasks)
    dev = [t for t in all_tasks if t["kind"] == "explanation_development"]
    confirm = [t for t in all_tasks if t["kind"] == "explanation_confirmation"]
    heavy = [t for t in all_tasks if t["kind"] == "explanation_heavy"]
    forks = [t for t in all_tasks if t["kind"] == "explanation_fork"]
    assert len(dev) == 35 and sum(len(t["indices"])*7 for t in dev) == 7840
    assert len(confirm) == 280 and sum(len(t["indices"])*4 for t in confirm) == 35840
    assert len(heavy) == 6 and len(forks) == 216
    assert {t["condition"] for t in confirm} == set(CONFIRMATION_ORDER)
    assert all(t["training_variants"] == ["as", "mmas"] for t in dev)
    assert all("training_variants" not in t for t in confirm)
    assert all(t["iterations"] == 5000 for t in (*dev, *confirm, *heavy))
    assert max(t["snapshot_iteration"]+t["continuation_iterations"] for t in forks) == 3000


def test_independent_confirmation_contrasts_have_expected_signs():
    names, matrix = confirmation_contrasts()
    assert matrix.shape == (14, 7) and len(set(names)) == 14
    gains = np.array([0., .4, .01, .02, 1., .8, .2])
    result = matrix @ gains
    np.testing.assert_array_equal(result[:7], gains)
    np.testing.assert_allclose(result[7:], [.4, .01, .02, -.2, -.6, .39, .38])


def test_source_controls_do_not_remove_other_components():
    c = conditions()
    for variant in ("mmas", "as"):
        assert c[variant+"_current_best"].source_policy == "iteration_best"
        assert c[variant+"_history_calendar"].source_policy == "global_calendar"
        assert c[variant+"_current_best"].restart_policy == "native"
        assert c[variant+"_current_best"].floor_scale == 1
    assert c["mmas_all_current"].source_count == 32


def test_cross_framework_entries_preserve_original_defaults(monkeypatch):
    from control_experiments.mmas_ls import evaluate
    def models(variant):
        return [{"id": f"{variant}-{s}", "seed": s, "structural_hash": str(s), "file_hash": "f", "program": (None, None)}
                for s in (81001, 81002, 81003)]
    monkeypatch.setattr(evaluate, "models", models)
    old = evaluate.program_entries("mmas")
    new = evaluate.program_entries("as", training_variants=("as", "mmas"))
    assert len(old) == 4 and len(new) == 7
    assert all("execution_variant" not in e for e in old)
    assert new[-1]["training_variant"] == "mmas" and new[-1]["execution_variant"] == "as"
    assert sum(e["id"] == "baseline" for e in new) == 1
    with pytest.raises(ValueError):
        evaluate.program_entries("as", training_variants=("as", "as"))


def test_canonical_edge_sets_and_diversity():
    tours = np.array([[0, 1, 2, 3, 0], [2, 3, 0, 1, 2], [0, 3, 2, 1, 0]])
    codes = canonical_edges(tours, 4)
    np.testing.assert_array_equal(codes, np.broadcast_to(codes[0], codes.shape))
    _, count, disagreement, effective = colony_structure(tours, 4)
    assert count == 1 and disagreement == 0 and effective == 4


def test_same_context_probability_counterfactual():
    tr = np.full((1, 1, 3, 22), np.nan)
    tr[0, 0, :, 17] = [1, 1, 2]; tr[0, 0, :, 19] = [.125, .125, .75]
    context = np.zeros((1, 1, 10), dtype=int); context[..., 7] = 3
    distance, changed, entropy = fixed_context_probabilities(tr, context)
    assert distance[0] == .25 and changed[0] == 0 and 0 < entropy[0] < 1
    context[..., 8] = 1; tr[0, 0, :, 19] = [0, 0, 1]
    assert fixed_context_probabilities(tr, context)[0][0] == 0


def test_local_search_difference_zero_denominator_is_undefined():
    tour = np.array([[[0, 1, 2, 3, 0]]])
    a = {"pre_tour_workspace": tour, "tour_workspace": tour}
    before, after, ratio = structure_difference(a, a)
    assert not before.any() and not after.any() and np.isnan(ratio).all()


def test_branch_design_includes_source_by_gp_interaction():
    specs = branch_specs(MechanismConfig(), (None, None))
    assert len(specs) == 6
    assert specs["native_without_gp"][1].source_policy == "native_schedule"
    assert specs["current_without_gp"][1].source_policy == "iteration_best"
    assert specs["current_both_trees"][1].source_policy == "iteration_best"


def test_spatial_variance_is_not_temporal_variance():
    info = np.zeros((1, 25, 1, 6)); info[..., 2] = 10; info[..., 5] = 1
    info[0, :, 0, 3] = np.arange(1, 26)
    moments = np.zeros((1, 25, 1, 6)); moments[..., 0] = 500
    x = np.linspace(-1, 1, 25)[None, :, None]
    moments[..., 4] = x*500; moments[..., 5] = x*x*500
    trace = np.zeros((1, 25, 26)); trace[..., [16, 17, 18, 19, 24, 25]] = 10
    ls = np.zeros((1, 25, 32, 3)); ls[..., :2] = 10
    data = {"source_info": info, "ph_moments": moments, "trace": trace, "ls": ls,
            "anytime": np.full((1, 25), 10.), "start": 1}
    result, _ = window_metrics(data, np.array([10.]), np.array([10.]))
    assert result[0, 0, WINDOW_FIELDS.index("within_source_tanh_std")] < 1e-8
