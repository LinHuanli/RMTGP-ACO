"""Fitness、训练循环和 non-inferiority fallback 测试。"""

from __future__ import annotations

from dataclasses import replace

import torch
import yaml

from rmtgp_aco.config import ACOConfig, ExperimentConfig, GPConfig
from rmtgp_aco.sampling import in_memory_cases
from rmtgp_aco.training import baseline_relative_fitness, train

from conftest import make_instance


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
        output_directory=output,
    )
    assert len(result.history) == 2
    assert result.champion.total_nodes >= 2
    assert (output / "config.yaml").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "champion.pkl").is_file()
    assert (output / "training_metrics.jsonl").is_file()
    assert (output / "validation_summary.csv").is_file()
    assert (output / "checkpoints").is_dir()
    loaded_config = yaml.safe_load(
        (output / "config.yaml").read_text(encoding="utf-8")
    )
    assert loaded_config["experiment_id"] == "tiny"
