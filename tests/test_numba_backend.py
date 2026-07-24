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
)
from rmtgp_aco.config import ACOConfig, ACOVariant
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.program import (
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
