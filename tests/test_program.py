"""Strongly Typed 双树表示与 tensor interpreter 测试。"""

from __future__ import annotations

import random

import torch
from deap import gp

from rmtgp_aco.config import GPConfig
from rmtgp_aco.genetic import (
    initialise_population,
    mate_role_preserving,
    mutate_role_preserving,
    valid_size,
)
from rmtgp_aco.program import (
    LEGACY_LS_PHEROMONE_TERMINAL,
    ORIGIN_PHEROMONE_TERMINALS,
    PHEROMONE_TERMINALS,
    TRANSITION_TERMINALS,
    compile_tree,
    constant_zero_tree,
    create_primitive_sets,
)


def test_zero_program_is_exact_and_shape_preserving() -> None:
    transition_pset, _ = create_primitive_sets()
    tree = constant_zero_tree(transition_pset, "ZERO_TR")
    program = compile_tree(tree, role="transition")
    context = {
        name: torch.randn(2, 3, 4, dtype=torch.float64)
        for name in TRANSITION_TERMINALS
    }
    value = program.evaluate(context)
    assert program.is_exact_zero
    assert value.shape == (2, 3, 4)
    assert torch.count_nonzero(value) == 0


def test_random_typed_population_compiles_and_evaluates() -> None:
    random.seed(7)
    config = GPConfig(
        population_size=10,
        generations=1,
        elite_size=2,
        initial_min_depth=1,
        initial_max_depth=2,
        max_depth=3,
        max_nodes_per_tree=31,
        checkpoint_interval=1,
        checkpoint_top_k=2,
    )
    population, _, _ = initialise_population(config)
    transition_context = {
        name: torch.randn(1, 2, 5, dtype=torch.float64)
        for name in TRANSITION_TERMINALS
    }
    pheromone_context = {
        name: torch.randn(1, 2, 5, dtype=torch.float64)
        for name in PHEROMONE_TERMINALS
    }
    for individual in population:
        transition = compile_tree(individual.transition_tree, role="transition")
        pheromone = compile_tree(individual.pheromone_tree, role="pheromone")
        assert torch.isfinite(transition.evaluate(transition_context)).all()
        assert torch.isfinite(pheromone.evaluate(pheromone_context)).all()
        assert valid_size(individual, config)


def test_genetic_operators_preserve_roles_and_limits() -> None:
    random.seed(19)
    config = GPConfig(
        population_size=10,
        generations=1,
        elite_size=2,
        initial_min_depth=1,
        initial_max_depth=2,
        max_depth=3,
        max_nodes_per_tree=31,
        checkpoint_interval=1,
        checkpoint_top_k=2,
    )
    population, transition_pset, pheromone_pset = initialise_population(config)
    first, second = mate_role_preserving(
        population[0].clone(),
        population[1].clone(),
        config,
    )
    (mutant,) = mutate_role_preserving(
        population[2].clone(),
        transition_pset,
        pheromone_pset,
        config,
    )
    assert valid_size(first, config)
    assert valid_size(second, config)
    assert valid_size(mutant, config)
    compile_tree(mutant.transition_tree, role="transition")
    compile_tree(mutant.pheromone_tree, role="pheromone")


def test_origin_is_gated_and_lsgain_is_not_in_default_search_space() -> None:
    """Origin 必须显式启用；不成熟的 LSGain 不得被随机 GP 使用。"""

    _, default_pheromone = create_primitive_sets()
    assert "Origin" not in default_pheromone.mapping
    assert LEGACY_LS_PHEROMONE_TERMINAL not in default_pheromone.mapping
    assert "Origin" not in PHEROMONE_TERMINALS

    _, origin_pheromone = create_primitive_sets(
        pheromone_terminals=ORIGIN_PHEROMONE_TERMINALS,
    )
    tree = gp.PrimitiveTree.from_string("Origin", origin_pheromone)
    program = compile_tree(tree, role="pheromone")
    context = {
        name: torch.full((2, 3), 0.25, dtype=torch.float64)
        for name in ORIGIN_PHEROMONE_TERMINALS
    }
    context["Origin"] = torch.tensor(
        [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]],
        dtype=torch.float64,
    )
    assert program.required_terminals == frozenset({"Origin"})
    assert torch.equal(program.evaluate(context), context["Origin"])
