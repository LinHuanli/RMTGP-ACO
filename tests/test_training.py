"""Fitness、训练循环和 non-inferiority fallback 测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import yaml
from conftest import make_instance

from rmtgp_aco.config import (
    ACOConfig,
    ExecutionBackend,
    ExperimentConfig,
    FitnessMode,
    GPConfig,
    LocalSearch,
    RacingConfig,
    RuntimeConfig,
    SelectionMode,
)
from rmtgp_aco.genetic import (
    evolve_generation,
    initialise_population,
    is_baseline_individual,
)
from rmtgp_aco.sampling import in_memory_cases
from rmtgp_aco.training import (
    BaselineCache,
    EvaluationPool,
    _experiment_hash,
    _pre_anytime_experiment_hash,
    baseline_relative_fitness,
    paired_ucb_fitness,
    train,
)


def test_scale_balanced_baseline_relative_fitness() -> None:
    candidate = {
        50: [torch.tensor([11.0, 9.0])],
        100: [torch.tensor([22.0])],
    }
    baseline = {
        50: [torch.tensor([10.0, 10.0])],
        100: [torch.tensor([20.0])],
    }
    reference = {
        50: [torch.tensor([10.0, 10.0])],
        100: [torch.tensor([20.0])],
    }
    result = baseline_relative_fitness(
        candidate,
        baseline,
        reference,
        degradation_penalty=1.0,
    )
    assert result.mean_delta_by_scale[50] == 0.0
    assert result.degradation_by_scale[50] == 5.0
    assert result.mean_delta_by_scale[100] == 10.0
    assert result.fitness == 12.5


def test_paired_ucb_fitness_has_exact_zero_baseline() -> None:
    baseline = {100: [torch.tensor([10.0, 10.0, 10.0])]}
    reference = {100: [torch.tensor([10.0, 10.0, 10.0])]}
    exact = paired_ucb_fitness(
        baseline,
        baseline,
        reference,
        z=1.0,
    )
    assert exact.fitness == 0.0
    assert exact.standard_error_by_scale[100] == 0.0
    assert exact.nonzero_fraction_by_scale[100] == 0.0

    mixed = paired_ucb_fitness(
        {100: [torch.tensor([9.0, 10.0, 11.0])]},
        baseline,
        reference,
        z=1.0,
    )
    assert mixed.mean_delta_by_scale[100] == pytest.approx(0.0)
    assert mixed.fitness > 0.0
    assert mixed.wins_by_scale[100] == 1
    assert mixed.ties_by_scale[100] == 1
    assert mixed.losses_by_scale[100] == 1


def test_baseline_anchor_is_unique_and_never_enters_breeding_pool() -> None:
    gp = GPConfig(
        population_size=12,
        generations=2,
        elite_size=2,
        tournament_size=2,
        initial_min_depth=1,
        initial_max_depth=2,
        max_depth=3,
        max_nodes_per_tree=31,
        max_total_nodes=31,
        checkpoint_interval=1,
        checkpoint_top_k=2,
        baseline_anchor=True,
        fitness_mode="paired_ucb",
    )
    population, transition_pset, pheromone_pset = initialise_population(gp)
    assert sum(is_baseline_individual(item) for item in population) == 1
    for index, individual in enumerate(population):
        individual.fitness.values = (0.0 if is_baseline_individual(individual) else index + 1.0,)
    evolved = evolve_generation(
        population,
        transition_pset,
        pheromone_pset,
        gp,
    )
    assert len(evolved) == gp.population_size
    assert sum(is_baseline_individual(item) for item in evolved) == 1
    assert all(
        not is_baseline_individual(item)
        for item in evolved[:-1]
    )


@pytest.mark.parametrize(
    "fitness_mode",
    [
        FitnessMode.PAIRED_FINAL_UCB,
        FitnessMode.PAIRED_BASIN_UCB,
        FitnessMode.PAIRED_COMBINED_UCB,
        FitnessMode.PAIRED_ANYTIME_UCB,
        FitnessMode.PAIRED_FINAL_ANYTIME_UCB,
    ],
)
def test_population_fitness_modes_keep_exact_zero_baseline_anchor(
    fitness_mode,
) -> None:
    """三种 paired fitness 均以同 seed 原始 ACO 为精确零点。"""

    cases = in_memory_cases(
        {5: [make_instance(5, 41), make_instance(5, 42)]},
        seed=1221,
        candidate_size=2,
    )
    experiment = ExperimentConfig(
        experiment_id=f"fitness-{fitness_mode.value}",
        root_seed=91,
        aco=replace(
            ACOConfig.acotsp_local_search_default(
                "acs",
                local_search=LocalSearch.TWO_OPT,
                iterations=2,
            ),
            ants=4,
            candidate_size=2,
            local_search_candidate_size=2,
        ),
        gp=GPConfig(
            population_size=6,
            generations=1,
            elite_size=1,
            tournament_size=2,
            initial_min_depth=1,
            initial_max_depth=2,
            max_depth=3,
            baseline_anchor=True,
            fitness_mode=fitness_mode,
            basin_top_q=2,
            basin_weight=0.8,
        ),
        runtime=RuntimeConfig(
            aco_backend=ExecutionBackend.NUMBA_BATCH,
            cpu_threads=2,
        ),
        train_scales=(5,),
        validation_scales=(5,),
        test_scales=(5,),
        validation_seeds=1,
    )
    population, _, _ = initialise_population(experiment.gp)
    anchor = next(
        individual
        for individual in population
        if is_baseline_individual(individual)
    )
    with EvaluationPool(experiment) as evaluator:
        result = evaluator.evaluate_population(
            population,
            cases,
            BaselineCache(),
        )
    breakdown = result.breakdowns[anchor.structural_hash]
    assert anchor.fitness.values == (0.0,)
    assert breakdown.fitness == 0.0
    assert breakdown.fitness_delta_by_scale[5] == 0.0
    assert breakdown.standard_error_by_scale[5] == 0.0
    if fitness_mode.uses_basin:
        assert breakdown.mean_basin_delta_by_scale[5] == 0.0
        assert breakdown.mean_basin_gap_by_scale
        assert breakdown.baseline_basin_gap_by_scale
    else:
        assert breakdown.mean_basin_gap_by_scale == {}
    if fitness_mode.uses_anytime:
        assert breakdown.mean_anytime_delta_by_scale[5] == 0.0
        assert breakdown.mean_anytime_gap_by_scale
        assert breakdown.baseline_anytime_gap_by_scale
    else:
        assert breakdown.mean_anytime_gap_by_scale == {}


def test_multifidelity_racing_records_both_stages(tmp_path) -> None:
    cases = in_memory_cases(
        {
            5: [
                make_instance(5, 51),
                make_instance(5, 52),
                make_instance(5, 53),
                make_instance(5, 54),
            ]
        },
        seed=1901,
        candidate_size=2,
    )
    experiment = ExperimentConfig(
        experiment_id="racing-tiny",
        root_seed=88,
        aco=replace(
            ACOConfig.acotsp_local_search_default(
                "as",
                local_search=LocalSearch.TWO_OPT,
                iterations=3,
            ),
            ants=3,
            candidate_size=2,
            local_search_candidate_size=2,
        ),
        gp=GPConfig(
            population_size=8,
            generations=2,
            elite_size=1,
            tournament_size=2,
            initial_min_depth=1,
            initial_max_depth=2,
            max_depth=3,
            checkpoint_interval=1,
            checkpoint_top_k=2,
            baseline_anchor=True,
            fitness_mode=FitnessMode.PAIRED_FINAL_ANYTIME_UCB,
        ),
        runtime=RuntimeConfig(
            aco_backend=ExecutionBackend.NUMBA_BATCH,
            cpu_threads=2,
        ),
        racing=RacingConfig(
            enabled=True,
            screen_instances_per_scale=2,
            finalists=3,
            exploration_finalists=1,
            preserved_elites=1,
            screen_horizon_schedule=((1, 1), (2, 2)),
        ),
        train_scales=(5,),
        validation_scales=(5,),
        test_scales=(5,),
        validation_seeds=1,
        training_horizon_schedule=((1, 2), (2, 3)),
        cpu_fp64_final_audit=False,
    )
    result = train(
        experiment,
        lambda _generation: cases,
        cases,
        output_directory=tmp_path / "racing",
    )
    assert len(result.history) == 2
    assert [
        record.racing_screen_iterations for record in result.history
    ] == [1, 2]
    assert [
        record.racing_high_iterations for record in result.history
    ] == [2, 3]
    assert all(record.racing_screen_instances == 2 for record in result.history)
    assert all(record.racing_high_instances == 4 for record in result.history)
    assert all(
        len(record.racing_finalist_hashes) == 3
        for record in result.history
    )
    assert all(
        len(record.racing_screen_fitness_by_hash) >= 3
        and len(record.racing_high_fitness_by_hash) == 3
        for record in result.history
    )
    assert all(record.best_mean_anytime_delta_by_scale for record in result.history)
    assert result.checkpoints

    interrupted = tmp_path / "racing-resume"

    def stop_after_first(record) -> None:
        if record.generation == 1:
            raise RuntimeError("stop racing")

    with pytest.raises(RuntimeError, match="stop racing"):
        train(
            experiment,
            lambda _generation: cases,
            cases,
            output_directory=interrupted,
            progress_callback=stop_after_first,
        )
    resumed = train(
        experiment,
        lambda _generation: cases,
        cases,
        output_directory=interrupted,
        resume_from=interrupted,
    )
    assert [
        (record.best_hash, record.racing_finalist_hashes)
        for record in resumed.history
    ] == [
        (record.best_hash, record.racing_finalist_hashes)
        for record in result.history
    ]


def test_training_horizon_schedule_is_recorded(tmp_path) -> None:
    cases = in_memory_cases(
        {5: [make_instance(5, 11), make_instance(5, 12)]},
        seed=910,
        candidate_size=2,
    )
    experiment = ExperimentConfig(
        experiment_id="horizon-tiny",
        root_seed=77,
        aco=replace(
            ACOConfig.acotsp_default("as", iterations=3),
            ants=3,
            candidate_size=2,
        ),
        gp=GPConfig(
            population_size=6,
            generations=3,
            elite_size=1,
            tournament_size=2,
            initial_min_depth=1,
            initial_max_depth=2,
            max_depth=3,
            checkpoint_interval=1,
            checkpoint_top_k=2,
            baseline_anchor=True,
            fitness_mode="paired_ucb",
        ),
        train_scales=(5,),
        validation_scales=(5,),
        test_scales=(5,),
        selection_mode=SelectionMode.LEGACY_NONINFERIORITY,
        training_horizon_schedule=((1, 1), (2, 2), (3, 3)),
        validation_monitor_interval=2,
    )
    result = train(
        experiment,
        lambda _generation: cases,
        cases,
        validation_monitor_cases=cases,
        output_directory=tmp_path / "horizon",
    )
    assert [
        record.training_aco_iterations
        for record in result.history
    ] == [1, 2, 3]
    assert result.history[0].validation_monitor_delta_by_scale == {}
    assert result.history[1].validation_monitor_delta_by_scale
    assert all(record.baseline_anchor_count == 1 for record in result.history)


def test_pre_anytime_checkpoint_hash_migration_is_strict() -> None:
    experiment = ExperimentConfig(
        experiment_id="legacy-hash",
        root_seed=1,
        aco=ACOConfig.acotsp_default("as", iterations=2),
        train_scales=(5,),
        validation_scales=(5,),
        test_scales=(5,),
    )
    legacy = _pre_anytime_experiment_hash(experiment)
    assert legacy is not None
    assert legacy != _experiment_hash(experiment)
    anytime = replace(
        experiment,
        aco=replace(
            ACOConfig.acotsp_local_search_default(
                "as",
                local_search=LocalSearch.TWO_OPT,
                iterations=2,
            )
        ),
        gp=replace(
            experiment.gp,
            fitness_mode=FitnessMode.PAIRED_FINAL_ANYTIME_UCB,
        ),
    )
    assert _pre_anytime_experiment_hash(anytime) is None


def test_tiny_training_run_is_reproducible(tmp_path) -> None:
    cases = in_memory_cases(
        {5: [make_instance(5, 1), make_instance(5, 2)]},
        seed=100,
        candidate_size=2,
    )
    gp = GPConfig(
        population_size=6,
        generations=2,
        elite_size=1,
        tournament_size=2,
        initial_min_depth=1,
        initial_max_depth=2,
        max_depth=3,
        max_nodes_per_tree=31,
        checkpoint_interval=1,
        checkpoint_top_k=2,
    )
    aco = replace(
        ACOConfig.acotsp_default("as", iterations=2),
        ants=3,
        candidate_size=2,
    )
    experiment = ExperimentConfig(
        experiment_id="tiny",
        root_seed=42,
        aco=aco,
        gp=gp,
        train_scales=(5,),
        validation_scales=(5,),
        test_scales=(5,),
    )
    output = tmp_path / "run"
    result = train(
        experiment,
        lambda _generation: cases,
        cases,
        validation_monitor_cases=cases,
        output_directory=output,
    )
    assert len(result.history) == 2
    assert result.champion.total_nodes <= gp.max_total_nodes
    assert result.validation.backend.value == "torch"
    assert result.cpu_fp64_audit is None
    assert (output / "config.yaml").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "champion.pkl").is_file()
    assert (output / "selected_candidate.pkl").is_file()
    assert (output / "deployment_decision.json").is_file()
    assert (output / "training_metrics.jsonl").is_file()
    assert (output / "training_validation_curve.csv").is_file()
    assert (output / "validation_summary.csv").is_file()
    assert (output / "checkpoints").is_dir()
    assert all(
        record.validation_monitor_candidate_gap_by_scale
        for record in result.history
    )
    loaded_config = yaml.safe_load(
        (output / "config.yaml").read_text(encoding="utf-8")
    )
    assert loaded_config["experiment_id"] == "tiny"


def test_training_resume_matches_uninterrupted_evolution(tmp_path) -> None:
    cases = in_memory_cases(
        {5: [make_instance(5, 7), make_instance(5, 8)]},
        seed=300,
        candidate_size=2,
    )
    gp = GPConfig(
        population_size=6,
        generations=3,
        elite_size=1,
        tournament_size=2,
        initial_min_depth=1,
        initial_max_depth=2,
        max_depth=3,
        checkpoint_interval=1,
        checkpoint_top_k=2,
    )
    experiment = ExperimentConfig(
        experiment_id="resume-tiny",
        root_seed=123,
        aco=replace(
            ACOConfig.acotsp_default("as", iterations=2),
            ants=3,
            candidate_size=2,
        ),
        gp=gp,
        train_scales=(5,),
        validation_scales=(5,),
        test_scales=(5,),
    )
    uninterrupted = train(
        experiment,
        lambda _generation: cases,
        cases,
    )

    run_directory = tmp_path / "interrupted"

    def interrupt_after_first(record) -> None:
        if record.generation == 1:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(
            experiment,
            lambda _generation: cases,
            cases,
            output_directory=run_directory,
            progress_callback=interrupt_after_first,
        )
    resumed = train(
        experiment,
        lambda _generation: cases,
        cases,
        output_directory=run_directory,
        resume_from=run_directory,
    )
    assert [
        (record.minimum, record.mean, record.best_hash)
        for record in resumed.history
    ] == [
        (record.minimum, record.mean, record.best_hash)
        for record in uninterrupted.history
    ]
    assert resumed.champion.structural_hash == uninterrupted.champion.structural_hash
