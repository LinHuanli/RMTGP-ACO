"""Full-replacement ablation 与跨进程个体评估测试。"""

from __future__ import annotations

from dataclasses import replace

import torch
from conftest import make_instance

from rmtgp_aco.aco import solve
from rmtgp_aco.config import (
    ACOConfig,
    ExperimentConfig,
    GPConfig,
    RuntimeConfig,
    TransitionIntegration,
)
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.genetic import (
    compile_individual,
    initialise_population,
    make_individual,
)
from rmtgp_aco.sampling import in_memory_cases
from rmtgp_aco.training import BaselineCache, evaluate_invalid_population


def test_legacy_typed_tree_runs_as_positive_replacement(small_instances) -> None:
    gp = GPConfig(
        population_size=6,
        generations=1,
        elite_size=1,
        initial_min_depth=1,
        initial_max_depth=2,
        max_depth=3,
        checkpoint_interval=1,
        checkpoint_top_k=1,
        train_transition=True,
        train_pheromone=False,
        transition_profile="legacy",
    )
    population, _, _ = initialise_population(gp)
    individual = next(
        item
        for item in population
        if not item.metadata.get("baseline_passthrough", False)
    )
    transition, pheromone = compile_individual(individual)
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("as", iterations=2),
        ants=3,
        candidate_size=2,
        transition_integration=TransitionIntegration.REPLACEMENT,
    )
    result = solve(
        batch,
        config,
        transition_program=transition,
        pheromone_program=pheromone,
        seed=5,
    )
    assert torch.isfinite(result.best_length).all()


def test_baseline_sentinel_bypasses_replacement(small_instances) -> None:
    gp = GPConfig(
        population_size=6,
        generations=1,
        elite_size=1,
        train_transition=True,
        train_pheromone=False,
        transition_profile="legacy",
    )
    population, transition_pset, pheromone_pset = initialise_population(gp)
    baseline = make_individual(
        transition_pset,
        pheromone_pset,
        gp,
        mode="baseline",
    )
    assert compile_individual(baseline) == (None, None)
    assert (
        baseline.structural_hash != population[0].structural_hash
        or baseline is not population[0]
    )


def test_cpu_process_pool_evaluates_unique_individuals() -> None:
    cases = in_memory_cases(
        {5: [make_instance(5, 1)]},
        seed=80,
        candidate_size=2,
    )
    gp = GPConfig(
        population_size=4,
        generations=1,
        elite_size=1,
        tournament_size=2,
        initial_min_depth=1,
        initial_max_depth=1,
        max_depth=2,
        checkpoint_interval=1,
        checkpoint_top_k=1,
    )
    aco = replace(
        ACOConfig.acotsp_default("as", iterations=1),
        ants=2,
        candidate_size=2,
    )
    experiment = ExperimentConfig(
        experiment_id="parallel-test",
        root_seed=2,
        aco=aco,
        gp=gp,
        runtime=RuntimeConfig(
            processes=2,
            torch_threads=1,
            torch_interop_threads=1,
            multiprocessing_start_method="spawn",
        ),
        train_scales=(5,),
        validation_scales=(5,),
        test_scales=(5,),
    )
    population, _, _ = initialise_population(gp)
    count = evaluate_invalid_population(
        population,
        experiment,
        cases,
        BaselineCache(),
    )
    assert 1 <= count <= len(population)
    assert all(item.fitness.valid for item in population)
