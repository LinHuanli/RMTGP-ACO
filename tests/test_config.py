"""配置默认值与约束测试。"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import pytest
import yaml

from rmtgp_aco.config import (
    ACOConfig,
    ACOVariant,
    ExecutionBackend,
    ExperimentConfig,
    FitnessMode,
    GPUMode,
    LocalSearch,
    RacingConfig,
    RuntimeConfig,
)


@pytest.mark.parametrize(
    ("variant", "ants", "rho", "q0"),
    [
        (ACOVariant.AS, None, 0.5, 0.0),
        (ACOVariant.ACS, 10, 0.1, 0.9),
        (ACOVariant.MMAS, None, 0.02, 0.0),
    ],
)
def test_acotsp_no_local_search_defaults(
    variant: ACOVariant,
    ants: int | None,
    rho: float,
    q0: float,
) -> None:
    config = ACOConfig.acotsp_default(variant)
    assert config.ants == ants
    assert config.alpha == 1.0
    assert config.beta == 2.0
    assert config.rho == rho
    assert config.q0 == q0
    assert config.candidate_size == 20


def test_dynamic_ant_and_candidate_resolution() -> None:
    config = ACOConfig.acotsp_default("as")
    assert config.resolve_ants(50) == 50
    assert config.resolve_candidate_size(8) == 7


@pytest.mark.parametrize(
    ("variant", "rho", "q0"),
    [
        (ACOVariant.AS, 0.5, 0.0),
        (ACOVariant.ACS, 0.1, 0.98),
        (ACOVariant.MMAS, 0.2, 0.0),
    ],
)
def test_acotsp_local_search_profile_keeps_comparable_ant_budget(
    variant: ACOVariant,
    rho: float,
    q0: float,
) -> None:
    config = ACOConfig.acotsp_local_search_default(variant)
    assert config.ants == 32
    assert config.local_search is LocalSearch.TWO_OPT
    assert config.local_search_candidate_size == 20
    assert config.local_search_dlb
    assert config.rho == rho
    assert config.q0 == q0


def test_invalid_residual_bound_is_rejected() -> None:
    config = ACOConfig.acotsp_default("mmas")
    with pytest.raises(ValueError, match="gamma_transition"):
        replace(config, gamma_transition=1.0)


def test_cuda_runtime_rejects_unsafe_block_and_cpu_mode() -> None:
    with pytest.raises(ValueError, match="gpu_block_threads"):
        RuntimeConfig(gpu_block_threads=128)
    with pytest.raises(ValueError, match="cuda_three_opt_block_threads"):
        RuntimeConfig(cuda_three_opt_block_threads=64)
    with pytest.raises(ValueError, match="至少保留 20%"):
        RuntimeConfig(gpu_memory_fraction=0.81)
    with pytest.raises(ValueError, match="gpu_mode=cpu"):
        RuntimeConfig(
            aco_backend=ExecutionBackend.CUDA_FUSED_FP32,
            gpu_mode=GPUMode.CPU,
        )


def test_experiment_stable_dict_is_yaml_safe_with_racing_fitness_mode() -> None:
    """正式 racing 配置不得把 StrEnum 泄漏给 PyYAML。"""

    experiment = ExperimentConfig(
        experiment_id="yaml-racing",
        root_seed=17,
        aco=ACOConfig.acotsp_local_search_default("as", iterations=2),
        racing=RacingConfig(
            enabled=True,
            screen_fitness_mode=FitnessMode.PAIRED_BASIN_MEAN,
        ),
    )
    payload = experiment.stable_dict()

    def enum_paths(value: object, path: str = "root") -> list[str]:
        if isinstance(value, Enum):
            return [path]
        if isinstance(value, dict):
            return [
                item
                for key, child in value.items()
                for item in enum_paths(child, f"{path}.{key}")
            ]
        if isinstance(value, (list, tuple)):
            return [
                item
                for index, child in enumerate(value)
                for item in enum_paths(child, f"{path}[{index}]")
            ]
        return []

    assert enum_paths(payload) == []
    assert payload["racing"]["screen_fitness_mode"] == "paired_basin_mean"
    assert yaml.safe_load(yaml.safe_dump(payload))["racing"][
        "screen_fitness_mode"
    ] == "paired_basin_mean"
