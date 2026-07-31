"""Numba 正式后端的 GP 数值语义、确定性与 batch 不变性测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from deap import gp

from rmtgp_aco.aco import solve
from rmtgp_aco.aco_numba import (
    _TRANSITION_TERMINAL_INDEX,
    _encode_program,
    _evaluate_program,
    solve_population_numba,
)
from rmtgp_aco.config import (
    ACOConfig,
    ACOVariant,
    LocalSearch,
    LSGainSemantics,
)
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.program import (
    ORIGIN_PHEROMONE_TERMINALS,
    compile_tree,
    constant_zero_tree,
    create_primitive_sets,
)


def _assert_valid_tours(tours: torch.Tensor) -> None:
    n = tours.shape[1] - 1
    assert torch.equal(tours[:, 0], tours[:, -1])
    expected = torch.arange(n).expand(tours.shape[0], n)
    assert torch.equal(torch.sort(tours[:, :-1], dim=1).values, expected)


@pytest.mark.parametrize("variant", list(ACOVariant))
def test_numba_zero_residual_exactly_recovers_baseline(
    variant,
    small_instances,
) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default(variant, iterations=3),
        ants=4,
        candidate_size=2,
    )
    transition_pset, pheromone_pset = create_primitive_sets()
    transition = compile_tree(
        constant_zero_tree(transition_pset, "ZERO_TR"),
        role="transition",
    )
    pheromone = compile_tree(
        constant_zero_tree(pheromone_pset, "ZERO_PH"),
        role="pheromone",
    )
    baseline = solve(batch, config, seed=909, backend="numba")
    residual = solve(
        batch,
        config,
        transition_program=transition,
        pheromone_program=pheromone,
        seed=909,
        backend="numba",
    )
    assert torch.equal(baseline.best_tour, residual.best_tour)
    assert torch.equal(baseline.best_length, residual.best_length)
    assert torch.equal(baseline.anytime_best, residual.anytime_best)
    _assert_valid_tours(residual.best_tour)


def test_numba_result_is_batch_partition_invariant(small_instances) -> None:
    config = replace(
        ACOConfig.acotsp_default("mmas", iterations=4),
        ants=4,
        candidate_size=2,
    )
    joint = make_problem_batch(small_instances, candidate_size=2)
    together = solve(joint, config, seed=77, backend="numba")
    separate = [
        solve(
            make_problem_batch([instance], candidate_size=2),
            config,
            seed=77,
            backend="numba",
        )
        for instance in small_instances
    ]
    assert torch.equal(
        together.best_tour,
        torch.cat([result.best_tour for result in separate]),
    )
    assert torch.equal(
        together.best_length,
        torch.cat([result.best_length for result in separate]),
    )


def test_numba_postfix_matches_tensor_interpreter() -> None:
    transition_pset, _ = create_primitive_sets()
    tree = gp.PrimitiveTree.from_string(
        "MAX(PDIV(ADD(RTau, REta), SUB(BaseConf, DistRank)), NEG(Entropy))",
        transition_pset,
    )
    program = compile_tree(tree, role="transition")
    encoded = _encode_program(program, role="transition")
    rng = np.random.default_rng(31)
    terminal_values = rng.normal(size=(14, 11))
    context = {
        name: torch.from_numpy(terminal_values[index])
        for name, index in _TRANSITION_TERMINAL_INDEX.items()
        if name in program.required_terminals
    }
    expected = program.evaluate(context).numpy()
    stack = np.empty(len(program.instructions), dtype=np.float64)
    actual = np.asarray(
        [
            _evaluate_program(
                encoded.opcodes,
                encoded.float_arguments,
                encoded.integer_arguments,
                terminal_values,
                column,
                stack,
            )
            for column in range(terminal_values.shape[1])
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_population_backend_merges_residual_scalar_introns(
    small_instances,
) -> None:
    """候选/边内常数 residual 经归一化后应合并为一个 baseline 行为。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("acs", iterations=3),
        ants=4,
        candidate_size=2,
    )
    transition_pset, pheromone_pset = create_primitive_sets()
    scalar_transition = compile_tree(
        gp.PrimitiveTree.from_string("ACOProg", transition_pset),
        role="transition",
    )
    scalar_pheromone = compile_tree(
        gp.PrimitiveTree.from_string("Stagnation", pheromone_pset),
        role="pheromone",
    )
    result = solve_population_numba(
        batch,
        config,
        [
            (None, None),
            (scalar_transition, None),
            (None, scalar_pheromone),
        ],
        seed=123,
        threads=2,
    )
    assert result.best_length.shape == (3, batch.batch_size)
    assert result.best_tour.shape == (3, batch.batch_size, batch.n + 1)
    assert torch.equal(result.best_length[0], result.best_length[1])
    assert torch.equal(result.best_length[0], result.best_length[2])
    assert torch.equal(result.best_iteration[0], result.best_iteration[1])
    assert torch.equal(result.best_iteration[0], result.best_iteration[2])
    assert result.constructed_tours == (
        batch.batch_size * config.resolve_ants(batch.n) * config.iterations
    )


def test_population_anytime_mean_matches_full_curve(small_instances) -> None:
    """训练标量必须等于逐轮 global-best-so-far 曲线的算术均值。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "as",
            local_search=LocalSearch.TWO_OPT,
            iterations=4,
        ),
        ants=4,
        candidate_size=2,
        local_search_candidate_size=2,
    )
    quality = solve_population_numba(
        batch,
        config,
        [(None, None)],
        seed=991,
        threads=2,
    )
    full = solve(batch, config, seed=991, backend="numba")
    assert quality.anytime_mean_length is not None
    torch.testing.assert_close(
        quality.anytime_mean_length[0],
        full.anytime_best.mean(dim=1),
        rtol=0.0,
        atol=0.0,
    )


def test_numba_mmas_full_restart_is_audited(small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("mmas", iterations=6),
        ants=4,
        candidate_size=2,
        mmas_branch_check_period=2,
        mmas_restart_stagnation=0,
        mmas_branch_threshold=100.0,
    )
    result = solve(batch, config, seed=17, backend="numba")
    assert result.diagnostics.mmas_restart_count > 0


def test_numba_two_opt_basin_audit_is_monotone_and_noninvasive(
    small_instances,
) -> None:
    """审计只读；post-2opt top-q 不能比同轮 pre-2opt 更差。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "acs",
            local_search=LocalSearch.TWO_OPT,
            iterations=3,
        ),
        ants=4,
        candidate_size=2,
        local_search_candidate_size=2,
    )
    audited = solve_population_numba(
        batch,
        config,
        [(None, None)],
        seed=2027,
        threads=2,
        basin_top_q=2,
        audit_local_search=True,
    )
    normal = solve_population_numba(
        batch,
        config,
        [(None, None)],
        seed=2027,
        threads=2,
        basin_top_q=2,
        audit_local_search=False,
    )

    assert audited.basin_mean_length is not None
    assert audited.pre_basin_mean_length is not None
    assert audited.edge_retention is not None
    assert audited.final_colony_tour is not None
    assert audited.final_pre_colony_tour is not None
    assert audited.final_colony_tour.shape == (
        1,
        batch.batch_size,
        config.resolve_ants(batch.n),
        batch.n + 1,
    )
    assert torch.all(
        audited.basin_mean_length
        <= audited.pre_basin_mean_length + 1e-12
    )
    assert torch.all(audited.edge_retention >= 0.0)
    assert torch.all(audited.edge_retention <= 1.0)
    assert torch.equal(audited.best_tour, normal.best_tour)
    assert torch.equal(audited.best_length, normal.best_length)
    assert torch.equal(audited.best_iteration, normal.best_iteration)
    assert torch.equal(audited.basin_mean_length, normal.basin_mean_length)


def test_numba_origin_pheromone_terminal_is_repeatable(
    small_instances,
) -> None:
    """Origin 的 ±1 edge provenance 可进入强类型 pheromone tree。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "as",
            local_search=LocalSearch.TWO_OPT,
            iterations=2,
        ),
        ants=4,
        candidate_size=2,
        local_search_candidate_size=2,
    )
    _, pheromone_pset = create_primitive_sets(
        pheromone_terminals=ORIGIN_PHEROMONE_TERMINALS,
    )
    origin = compile_tree(
        gp.PrimitiveTree.from_string("Origin", pheromone_pset),
        role="pheromone",
    )
    first = solve_population_numba(
        batch,
        config,
        [(None, origin)],
        seed=2029,
        threads=2,
        basin_top_q=2,
    )
    repeated = solve_population_numba(
        batch,
        config,
        [(None, origin)],
        seed=2029,
        threads=2,
        basin_top_q=2,
    )
    assert torch.equal(first.best_tour, repeated.best_tour)
    assert torch.equal(first.best_length, repeated.best_length)
    assert torch.equal(first.basin_mean_length, repeated.basin_mean_length)
    assert torch.isfinite(first.best_length).all()


def test_numba_ls_aware_terminals_are_repeatable(
    small_instances,
) -> None:
    """CPU oracle 应覆盖 CUDA v2 的全部 LS-aware terminal 数据流。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "as",
            local_search=LocalSearch.TWO_OPT,
            iterations=2,
        ),
        ants=4,
        candidate_size=2,
        local_search_candidate_size=2,
        ls_gain_semantics=LSGainSemantics.EDGE_LAST_MOVE,
    )
    transition_pset, pheromone_pset = create_primitive_sets(
        pheromone_terminals=(
            "LSGain",
            "TauHeadroom",
            "PreFreq",
            "PostFreq",
            "Origin",
        ),
    )
    transition = compile_tree(
        gp.PrimitiveTree.from_string(
            "ADD(MutualRank, TurnCos)",
            transition_pset,
        ),
        role="transition",
    )
    pheromone = compile_tree(
        gp.PrimitiveTree.from_string(
            "ADD(ADD(LSGain, TauHeadroom), ADD(PreFreq, PostFreq))",
            pheromone_pset,
        ),
        role="pheromone",
    )
    programs = [(transition, pheromone)]
    first = solve_population_numba(
        batch,
        config,
        programs,
        seed=2081,
        threads=2,
    )
    repeated = solve_population_numba(
        batch,
        config,
        programs,
        seed=2081,
        threads=2,
    )
    assert torch.equal(first.best_tour, repeated.best_tour)
    assert torch.equal(first.best_length, repeated.best_length)
    assert torch.equal(first.diagnostics, repeated.diagnostics)
    assert torch.isfinite(first.best_length).all()
