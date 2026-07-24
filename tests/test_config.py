"""配置默认值与约束测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rmtgp_aco.config import (
    ACOConfig,
    ACOVariant,
    ExecutionBackend,
    GPUMode,
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


def test_invalid_residual_bound_is_rejected() -> None:
    config = ACOConfig.acotsp_default("mmas")
    with pytest.raises(ValueError, match="gamma_transition"):
        replace(config, gamma_transition=1.0)


def test_cuda_runtime_rejects_unsafe_block_and_cpu_mode() -> None:
    with pytest.raises(ValueError, match="gpu_block_threads"):
        RuntimeConfig(gpu_block_threads=128)
    with pytest.raises(ValueError, match="至少保留 20%"):
        RuntimeConfig(gpu_memory_fraction=0.81)
    with pytest.raises(ValueError, match="gpu_mode=cpu"):
        RuntimeConfig(
            aco_backend=ExecutionBackend.CUDA_FUSED_FP32,
            gpu_mode=GPUMode.CPU,
        )
