"""Protocol A v0.4 的 batching、schedule、cache 与 fitness 回归测试。"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from conftest import make_instance
from deap import gp

from rmtgp_aco.aco import solve
from rmtgp_aco.aco_numba import solve_population_numba
from rmtgp_aco.baseline import (
    BaselineArchive,
    precompute_baseline_cases,
    write_baseline_shard,
)
from rmtgp_aco.config import ACOConfig, ACOVariant, GPConfig
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.experiment_plan import build_protocol_a_v04_pilot_plan
from rmtgp_aco.genetic import initialise_population, make_individual
from rmtgp_aco.program import compile_tree, create_primitive_sets
from rmtgp_aco.sampling import in_memory_cases, pools_from_paths
from rmtgp_aco.schedule import (
    ScheduledTrainingSampler,
    build_protocol_schedule,
    load_schedule,
    validate_schedule_contract,
    write_schedule,
)
from rmtgp_aco.training import reference_gap_fitness


@pytest.mark.parametrize("variant", list(ACOVariant))
def test_population_batch_matches_scalar_numba(variant, small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default(variant, iterations=2),
        ants=3,
        candidate_size=2,
    )
    transition_pset, _ = create_primitive_sets()
    transition = compile_tree(
        gp.PrimitiveTree.from_string("ADD(RTau, DistRank)", transition_pset),
        role="transition",
    )
    programs = [(None, None), (transition, None)]
    batched = solve_population_numba(
        batch,
        config,
        programs,
        seed=731,
        threads=2,
    )
    for index, (transition_program, pheromone_program) in enumerate(programs):
        scalar = solve(
            batch,
            config,
            transition_program=transition_program,
            pheromone_program=pheromone_program,
            seed=731,
            backend="numba",
        )
        assert torch.equal(batched.best_length[index], scalar.best_length)
        assert torch.equal(batched.best_iteration[index], scalar.best_iteration)

    one_thread = solve_population_numba(
        batch,
        config,
        programs,
        seed=731,
        threads=1,
    )
    assert torch.equal(one_thread.best_length, batched.best_length)


def test_compact_schedule_roundtrip_and_disjoint_validation(tmp_path) -> None:
    line = "0 0 1 0 1 1 0 1 output 1 2 3 4 1"
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text("\n".join([line] * 20) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join([line] * 12) + "\n", encoding="utf-8")
    training_pools = pools_from_paths({4: [train_path]})
    validation_pools = pools_from_paths({4: [validation_path]})
    schedule = build_protocol_schedule(
        training_pools,
        validation_pools,
        protocol_id="test-v0.3",
        phase="development",
        root_seed=17,
        replicate_id=2,
        generations=3,
        train_instances_per_scale=2,
        validation_selection_instances_per_scale=3,
        validation_gate_instances_per_scale=3,
        validation_seeds=2,
    )
    train_indices = [
        index
        for record in schedule.records
        if record.split == "train"
        for index in record.logical_indices
    ]
    assert len(train_indices) == len(set(train_indices)) == 6
    selection = next(record for record in schedule.records if record.role == "selection")
    gate = next(record for record in schedule.records if record.role == "gate")
    assert set(selection.logical_indices).isdisjoint(gate.logical_indices)

    path = write_schedule(schedule, tmp_path / "schedule.json")
    restored = load_schedule(path)
    assert restored.manifest_hash == schedule.manifest_hash
    sampler = ScheduledTrainingSampler(
        restored,
        training_pools,
        replicate_id=2,
        candidate_size=2,
    )
    sampler.cases_for_generation(1)
    assert len(json.dumps(sampler.state_dict())) < 500


def test_pilot_and_formal_schedules_use_disjoint_index_domains(tmp_path) -> None:
    lines = [
        (
            f"0 0 1 0 1 {1 + index / 1000:.6f} 0 1 "
            "output 1 2 3 4 1"
        )
        for index in range(40)
    ]
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    training_pools = pools_from_paths({4: [train_path]})
    validation_pools = pools_from_paths({4: [validation_path]})

    def build(phase: str):
        return build_protocol_schedule(
            training_pools,
            validation_pools,
            protocol_id="protocol-a-v0.4",
            phase=phase,
            root_seed=17,
            replicate_id=2,
            generations=2,
            train_instances_per_scale=3,
            validation_selection_instances_per_scale=4,
            validation_gate_instances_per_scale=4,
            validation_seeds=2,
        )

    pilot = build("pilot")
    formal = build("formal")
    pilot_indices = {
        (record.split, record.scale, index)
        for record in pilot.records
        for index in record.logical_indices
    }
    formal_indices = {
        (record.split, record.scale, index)
        for record in formal.records
        for index in record.logical_indices
    }
    assert pilot_indices.isdisjoint(formal_indices)

    validate_schedule_contract(
        formal,
        protocol_id="protocol-a-v0.4",
        phase="formal",
        root_seed=17,
        replicate_id=2,
        generations=2,
        train_scales=(4,),
        validation_scales=(4,),
        train_instances_per_scale=3,
        validation_selection_instances_per_scale=4,
        validation_gate_instances_per_scale=4,
        validation_seeds=2,
    )
    with pytest.raises(ValueError, match="phase"):
        validate_schedule_contract(
            formal,
            protocol_id="protocol-a-v0.4",
            phase="pilot",
            root_seed=17,
            replicate_id=2,
            generations=2,
            train_scales=(4,),
            validation_scales=(4,),
            train_instances_per_scale=3,
            validation_selection_instances_per_scale=4,
            validation_gate_instances_per_scale=4,
            validation_seeds=2,
        )


def test_pilot_plan_contains_paired_78_run_matrix() -> None:
    plan = build_protocol_a_v04_pilot_plan(
        runs_root="runs/test",
        python="python",
    )
    assert plan["setup_task_count"] == 30
    assert plan["training_task_count"] == 78
    tasks = plan["training_tasks"]
    main = [
        task
        for task in tasks
        if task["protocol"] in {"as", "acs", "mmas"}
    ]
    specialists = [
        task
        for task in tasks
        if task["protocol"] in {"acs-tsp50-only", "acs-tsp100-only"}
    ]
    assert len(main) == 72
    assert len(specialists) == 6
    schedules_by_protocol_seed: dict[tuple[str, int], set[str]] = {}
    for task in main:
        command = task["command"]
        schedule = command[command.index("--schedule") + 1]
        key = (task["protocol"], task["root_seed"])
        schedules_by_protocol_seed.setdefault(key, set()).add(schedule)
    assert all(len(values) == 1 for values in schedules_by_protocol_seed.values())


def test_immutable_baseline_archive_roundtrip(tmp_path) -> None:
    cases = in_memory_cases(
        {5: [make_instance(5, 1), make_instance(5, 2)]},
        seed=100,
        candidate_size=2,
    )
    config = replace(
        ACOConfig.acotsp_default("as", iterations=1),
        ants=3,
        candidate_size=2,
    )
    records = precompute_baseline_cases(
        cases,
        config,
        "numba_batch",
        threads=2,
    )
    root = tmp_path / "archive"
    write_baseline_shard(records, root / "train.npz")
    write_baseline_shard(records, root / "validation-duplicate.npz")
    archive = BaselineArchive(root, config, "numba_batch", require=True)
    observed = archive.lookup(cases[0])
    expected = solve(cases[0].batch, config, seed=cases[0].seed, backend="numba")
    assert observed is not None
    assert torch.equal(observed, expected.best_length)

    incompatible = BaselineArchive(
        root,
        replace(config, iterations=2),
        "numba_batch",
        require=True,
    )
    with pytest.raises(KeyError, match="cache miss"):
        incompatible.lookup(cases[0])


def test_reference_gap_is_primary_fitness() -> None:
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
    result = reference_gap_fitness(candidate, baseline, reference)
    assert result.mean_gap_by_scale == {50: 0.0, 100: 10.0}
    assert result.baseline_gap_by_scale == {50: 0.0, 100: 0.0}
    assert result.fitness == 5.0


def test_initial_population_obeys_global_node_budget() -> None:
    config = GPConfig(
        population_size=20,
        generations=1,
        elite_size=2,
        initial_min_depth=1,
        initial_max_depth=2,
        max_depth=3,
        max_nodes_per_tree=15,
        max_total_nodes=15,
        checkpoint_interval=1,
        checkpoint_top_k=1,
    )
    population, transition_pset, pheromone_pset = initialise_population(config)
    assert all(individual.total_nodes <= 15 for individual in population)
    baseline = make_individual(
        transition_pset,
        pheromone_pset,
        config,
        mode="baseline",
    )
    assert baseline.total_nodes == 0
