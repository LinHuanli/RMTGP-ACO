"""三种 ACO 语义、残差不变量与可复现性测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rmtgp_aco.aco import (
    _apply_acs_local_update,
    _global_update,
    _initial_search_state,
    _residual_deposit,
    solve,
)
from rmtgp_aco.config import (
    ACOConfig,
    ACOVariant,
    PheromoneIntegration,
)
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.model import DepositEventBatch, RunDiagnostics
from rmtgp_aco.program import compile_tree, constant_zero_tree, create_primitive_sets


def _zero_programs():
    transition_pset, pheromone_pset = create_primitive_sets()
    return (
        compile_tree(
            constant_zero_tree(transition_pset, "ZERO_TR"),
            role="transition",
        ),
        compile_tree(
            constant_zero_tree(pheromone_pset, "ZERO_PH"),
            role="pheromone",
        ),
    )


def _assert_valid_tours(tours: torch.Tensor) -> None:
    n = tours.shape[1] - 1
    assert torch.equal(tours[:, 0], tours[:, -1])
    expected = torch.arange(n, device=tours.device).expand(tours.shape[0], n)
    assert torch.equal(torch.sort(tours[:, :-1], dim=1).values, expected)


@pytest.mark.parametrize("variant", list(ACOVariant))
def test_zero_residual_exactly_recovers_baseline(variant, small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default(variant, iterations=3),
        ants=4,
        candidate_size=2,
    )
    transition, pheromone = _zero_programs()
    baseline = solve(batch, config, seed=2026)
    residual = solve(
        batch,
        config,
        transition_program=transition,
        pheromone_program=pheromone,
        seed=2026,
    )
    assert torch.equal(baseline.best_tour, residual.best_tour)
    assert torch.equal(baseline.best_length, residual.best_length)
    assert torch.equal(baseline.anytime_best, residual.anytime_best)
    _assert_valid_tours(residual.best_tour)


def test_same_seed_is_deterministic(small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("mmas", iterations=3),
        ants=4,
        candidate_size=2,
    )
    first = solve(batch, config, seed=31)
    second = solve(batch, config, seed=31)
    assert torch.equal(first.best_tour, second.best_tour)
    assert torch.equal(first.anytime_best, second.anytime_best)


def test_torch_mmas_full_restart_is_audited(small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("mmas", iterations=6),
        ants=4,
        candidate_size=2,
        mmas_branch_check_period=2,
        mmas_restart_stagnation=0,
        mmas_branch_threshold=100.0,
    )
    result = solve(batch, config, seed=31)
    assert result.diagnostics.mmas_restart_count > 0


def test_candidate_list_exhaustion_uses_full_fallback(small_instances) -> None:
    batch = make_problem_batch(small_instances[:1], candidate_size=1)
    config = replace(
        ACOConfig.acotsp_default("as", iterations=2),
        ants=3,
        candidate_size=1,
    )
    result = solve(batch, config, seed=8)
    assert result.diagnostics.candidate_fallback_count > 0
    _assert_valid_tours(result.best_tour)


def test_acs_synchronous_local_update_respects_edge_multiplicity() -> None:
    pheromone = torch.full((1, 4, 4), 2.0, dtype=torch.float64)
    tau0 = torch.tensor([1.0], dtype=torch.float64)
    u = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    v = torch.tensor([[1, 0, 2]], dtype=torch.int64)
    _apply_acs_local_update(pheromone, u, v, tau0, xi=0.1)
    # (0,1) 被两只蚂蚁同时经过，闭式结果为 (1-xi)^2*tau + (1-(1-xi)^2)*tau0。
    assert pheromone[0, 0, 1].item() == pytest.approx(1.81)
    assert pheromone[0, 1, 0].item() == pytest.approx(1.81)
    assert pheromone[0, 0, 2].item() == pytest.approx(1.9)


def test_pheromone_residual_preserves_each_source_budget() -> None:
    shape = (2, 3, 5)
    base = torch.rand(shape, dtype=torch.float64) + 0.1
    budget = torch.tensor(
        [[2.0, 3.0, 4.0], [1.5, 2.5, 3.5]],
        dtype=torch.float64,
    )
    zeros = torch.zeros(shape, dtype=torch.int64)
    terminals = {
        name: torch.randn(shape, dtype=torch.float64)
        for name in (
            "EdgeEta",
            "EdgeTau",
            "NNRank",
            "ColonyFreq",
            "SourceQuality",
            "ACOProg",
            "Stagnation",
        )
    }
    events = DepositEventBatch(
        edge_u=zeros,
        edge_v=zeros,
        edge_id=zeros,
        source_length=torch.ones((2, 3), dtype=torch.float64),
        base_deposit=base,
        base_budget=budget,
        terminals=terminals,
    )
    _, pheromone_pset = create_primitive_sets()
    nonconstant_tree = pheromone_pset.mapping["EdgeEta"]
    program = compile_tree(
        type(constant_zero_tree(pheromone_pset, "ZERO_PH"))([nonconstant_tree]),
        role="pheromone",
    )
    config = ACOConfig.acotsp_default("as")
    deposit = _residual_deposit(events, program, config)
    torch.testing.assert_close(deposit.sum(dim=-1), budget)


@pytest.mark.parametrize(
    "mode",
    [
        PheromoneIntegration.UNNORMALIZED_MULTIPLICATIVE,
        PheromoneIntegration.ADDITIVE,
        PheromoneIntegration.REPLACEMENT,
    ],
)
def test_pheromone_ablation_modes_remain_positive(mode) -> None:
    shape = (1, 1, 4)
    base = torch.full(shape, 0.25, dtype=torch.float64)
    zeros = torch.zeros(shape, dtype=torch.int64)
    terminals = {
        name: torch.linspace(-1.0, 1.0, 4, dtype=torch.float64).reshape(shape)
        for name in (
            "EdgeEta",
            "EdgeTau",
            "NNRank",
            "ColonyFreq",
            "SourceQuality",
            "ACOProg",
            "Stagnation",
        )
    }
    events = DepositEventBatch(
        edge_u=zeros,
        edge_v=zeros,
        edge_id=zeros,
        source_length=torch.ones((1, 1), dtype=torch.float64),
        base_deposit=base,
        base_budget=torch.ones((1, 1), dtype=torch.float64),
        terminals=terminals,
    )
    _, pset = create_primitive_sets()
    tree_type = type(constant_zero_tree(pset, "ZERO_PH"))
    program = compile_tree(tree_type([pset.mapping["EdgeEta"]]), role="pheromone")
    config = replace(
        ACOConfig.acotsp_default("as"),
        pheromone_integration=mode,
    )
    deposit = _residual_deposit(events, program, config)
    assert torch.isfinite(deposit).all()
    assert torch.all(deposit > 0)


@pytest.mark.parametrize("variant", list(ACOVariant))
def test_variant_global_update_is_symmetric_and_semantic(
    variant,
    small_instances,
) -> None:
    problem = make_problem_batch(small_instances[:1], candidate_size=2)
    n = problem.n
    pheromone = torch.full((1, n, n), 2.0, dtype=torch.float64)
    diagonal = torch.arange(n)
    pheromone[:, diagonal, diagonal] = 0.0
    edge_u = torch.arange(n, dtype=torch.int64).reshape(1, 1, n)
    edge_v = torch.roll(edge_u, shifts=-1, dims=-1)
    first = torch.minimum(edge_u, edge_v)
    second = torch.maximum(edge_u, edge_v)
    events = DepositEventBatch(
        edge_u=edge_u,
        edge_v=edge_v,
        edge_id=first * n + second,
        source_length=torch.tensor([[4.0]], dtype=torch.float64),
        base_deposit=torch.full((1, 1, n), 0.25, dtype=torch.float64),
        base_budget=torch.tensor([[n / 4.0]], dtype=torch.float64),
        terminals={
            name: torch.zeros((1, 1, n), dtype=torch.float64)
            for name in (
                "EdgeEta",
                "EdgeTau",
                "NNRank",
                "ColonyFreq",
                "SourceQuality",
                "ACOProg",
                "Stagnation",
            )
        },
    )
    config = replace(
        ACOConfig.acotsp_default(variant),
        ants=3,
        candidate_size=2,
    )
    state = _initial_search_state(
        problem,
        tau_min=torch.full((1,), 0.5, dtype=torch.float64),
        tau_max=torch.full((1,), 1.0, dtype=torch.float64),
    )
    diagnostics = RunDiagnostics()
    updated = _global_update(
        pheromone,
        events,
        state,
        config,
        None,
        diagnostics,
    )
    torch.testing.assert_close(updated, updated.transpose(1, 2))
    assert torch.all(torch.diagonal(updated, dim1=1, dim2=2) == 0)
    if variant is ACOVariant.AS:
        assert updated[0, 0, 2].item() == pytest.approx(1.0)
        assert updated[0, 0, 1].item() == pytest.approx(1.25)
    elif variant is ACOVariant.ACS:
        assert updated[0, 0, 2].item() == pytest.approx(2.0)
        assert updated[0, 0, 1].item() == pytest.approx(1.825)
    else:
        assert torch.all(updated[0][~torch.eye(n, dtype=torch.bool)] <= 1.0)
        assert diagnostics.bound_clip_count == n * (n - 1)
