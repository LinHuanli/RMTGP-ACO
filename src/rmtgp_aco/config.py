"""实验配置及其一致性检查。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any

import torch


class ACOVariant(StrEnum):
    """本研究支持的 ACO 变体。"""

    AS = "as"
    ACS = "acs"
    MMAS = "mmas"


class ExecutionBackend(StrEnum):
    """ACO 数值内核后端。

    ``torch`` 保留为易审计的参考实现；``numba`` 是标量/进程级参考；
    ``numba_batch`` 在一个进程内以 population×instance 任务矩阵并行。
    """

    TORCH = "torch"
    NUMBA = "numba"
    NUMBA_BATCH = "numba_batch"
    CUDA_FUSED_FP32 = "cuda_fused_fp32"


class GPUMode(StrEnum):
    """CUDA 设备的使用方式。"""

    CPU = "cpu"
    SINGLE = "single"
    DUAL = "dual"
    CAMPAIGN = "campaign"
    AUTO = "auto"


class TransitionIntegration(StrEnum):
    """GP transition tree 与 ACO desirability 的结合方式。"""

    RESIDUAL = "residual"
    REPLACEMENT = "replacement"


class PheromoneIntegration(StrEnum):
    """GP pheromone tree 的强化边集成方式。"""

    BUDGET_RESIDUAL = "budget_residual"
    UNNORMALIZED_MULTIPLICATIVE = "unnormalized_multiplicative"
    ADDITIVE = "additive"
    REPLACEMENT = "replacement"


@dataclass(frozen=True, slots=True)
class ACOConfig:
    """ACO 运行配置。

    `ants=None` 表示蚂蚁数随实例规模取 `n`，对应 ACOTSP 中的 `-1`。
    `rho` 始终表示蒸发比例，即全局蒸发后的保留比例为 `1-rho`。
    """

    variant: ACOVariant
    ants: int | None
    alpha: float
    beta: float
    rho: float
    candidate_size: int = 20
    iterations: int = 100
    q0: float = 0.0
    xi: float = 0.1
    gamma_transition: float = 1.0 / 3.0
    gamma_pheromone: float = 1.0 / 3.0
    transition_integration: TransitionIntegration = TransitionIntegration.RESIDUAL
    pheromone_integration: PheromoneIntegration = (
        PheromoneIntegration.BUDGET_RESIDUAL
    )
    acs_synchronous: bool = True
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    epsilon_distance: float = 1e-12
    epsilon_numeric: float = 1e-12
    mmas_update_period: int = 25
    mmas_p_best: float = 0.05
    mmas_branch_check_period: int = 100
    mmas_branch_lambda: float = 0.05
    mmas_branch_threshold: float = 1.00001
    mmas_restart_stagnation: int = 250

    @classmethod
    def acotsp_default(
        cls,
        variant: ACOVariant | str,
        *,
        iterations: int = 100,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        acs_synchronous: bool = True,
    ) -> ACOConfig:
        """建立主实验使用的 ACOTSP 无局部搜索默认配置。"""

        selected = ACOVariant(variant)
        common: dict[str, Any] = {
            "variant": selected,
            "candidate_size": 20,
            "iterations": iterations,
            "device": device,
            "dtype": dtype,
            "acs_synchronous": acs_synchronous,
        }
        if selected is ACOVariant.AS:
            return cls(ants=None, alpha=1.0, beta=2.0, rho=0.5, **common)
        if selected is ACOVariant.ACS:
            return cls(
                ants=10,
                alpha=1.0,
                beta=2.0,
                rho=0.1,
                q0=0.9,
                xi=0.1,
                **common,
            )
        return cls(ants=None, alpha=1.0, beta=2.0, rho=0.02, **common)

    def __post_init__(self) -> None:
        """尽早拒绝会破坏概率或残差边界的配置。"""

        object.__setattr__(self, "variant", ACOVariant(self.variant))
        object.__setattr__(
            self,
            "transition_integration",
            TransitionIntegration(self.transition_integration),
        )
        object.__setattr__(
            self,
            "pheromone_integration",
            PheromoneIntegration(self.pheromone_integration),
        )
        if self.ants is not None and self.ants < 1:
            raise ValueError("ants 必须为正整数或 None")
        if self.candidate_size < 1:
            raise ValueError("candidate_size 必须为正整数")
        if self.iterations < 1:
            raise ValueError("iterations 必须为正整数")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha 和 beta 不得为负")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho 必须位于 (0, 1]")
        if not 0.0 <= self.q0 <= 1.0:
            raise ValueError("q0 必须位于 [0, 1]")
        if not 0.0 < self.xi <= 1.0:
            raise ValueError("xi 必须位于 (0, 1]")
        if not 0.0 <= self.gamma_transition < 1.0:
            raise ValueError("gamma_transition 必须位于 [0, 1)")
        if not 0.0 <= self.gamma_pheromone < 1.0:
            raise ValueError("gamma_pheromone 必须位于 [0, 1)")
        if self.mmas_update_period < 1:
            raise ValueError("mmas_update_period 必须为正整数")
        if self.mmas_branch_check_period < 1:
            raise ValueError("mmas_branch_check_period 必须为正整数")
        if not 0.0 <= self.mmas_branch_lambda <= 1.0:
            raise ValueError("mmas_branch_lambda 必须位于 [0, 1]")
        if self.mmas_branch_threshold < 0.0:
            raise ValueError("mmas_branch_threshold 不得为负")
        if self.mmas_restart_stagnation < 0:
            raise ValueError("mmas_restart_stagnation 不得为负")

    def resolve_ants(self, n: int) -> int:
        """把 ACOTSP 的 `ants=n` 约定解析为实际蚂蚁数。"""

        if n < 2:
            raise ValueError("TSP 至少需要两个城市")
        return n if self.ants is None else self.ants

    def resolve_candidate_size(self, n: int) -> int:
        """候选列表不能包含当前城市，因此至多为 `n-1`。"""

        return min(self.candidate_size, n - 1)

    def stable_dict(self) -> dict[str, Any]:
        """返回可序列化、可哈希的配置。"""

        values = asdict(self)
        values["variant"] = self.variant.value
        values["transition_integration"] = self.transition_integration.value
        values["pheromone_integration"] = self.pheromone_integration.value
        values["dtype"] = str(self.dtype).removeprefix("torch.")
        return values

    @property
    def config_hash(self) -> str:
        """用于 cache 和 artifact 的稳定短哈希。"""

        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class GPConfig:
    """Strongly Typed Multi-Tree GP 的默认参数。"""

    population_size: int = 100
    generations: int = 50
    crossover_probability: float = 0.80
    mutation_probability: float = 0.15
    reproduction_probability: float = 0.05
    elite_size: int = 10
    tournament_size: int = 4
    initial_min_depth: int = 2
    initial_max_depth: int = 4
    max_depth: int = 5
    max_nodes_per_tree: int = 31
    max_total_nodes: int = 31
    checkpoint_interval: int = 5
    checkpoint_top_k: int = 5
    # v0.3 的主 fitness 不再使用 baseline-relative penalty。保留字段只为
    # 读取 v0.2 checkpoint/config，非零值不会进入新的 fitness。
    degradation_penalty: float = 0.0
    train_transition: bool = True
    train_pheromone: bool = True
    transition_profile: str = "main"
    function_profile: str = "f1"
    transition_terminals: tuple[str, ...] | None = None
    pheromone_terminals: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        total = (
            self.crossover_probability
            + self.mutation_probability
            + self.reproduction_probability
        )
        if abs(total - 1.0) > 1e-12:
            raise ValueError("crossover、mutation 和 reproduction 概率之和必须为 1")
        if not 0 < self.elite_size < self.population_size:
            raise ValueError("elite_size 必须位于 (0, population_size)")
        if self.initial_max_depth > self.max_depth:
            raise ValueError("初始最大深度不能超过 max_depth")
        if self.max_nodes_per_tree < 1 or self.max_total_nodes < 1:
            raise ValueError("GP 节点上限必须为正整数")
        if self.max_total_nodes > 2 * self.max_nodes_per_tree:
            raise ValueError("max_total_nodes 不得超过两棵树节点上限之和")
        if not self.train_transition and not self.train_pheromone:
            raise ValueError("至少必须训练 transition 或 pheromone 中的一棵树")
        if self.population_size < 2:
            raise ValueError("population_size 至少为 2")
        if self.generations < 1:
            raise ValueError("generations 必须为正整数")
        if self.checkpoint_interval < 1 or self.checkpoint_top_k < 1:
            raise ValueError("checkpoint_interval 和 checkpoint_top_k 必须为正整数")
        if self.transition_profile not in {"main", "legacy"}:
            raise ValueError("transition_profile 仅支持 main 或 legacy")
        if self.function_profile not in {"f0", "f1"}:
            raise ValueError("function_profile 仅支持 f0 或 f1")
        if self.transition_profile == "legacy" and self.train_pheromone:
            raise ValueError("Legacy-GP 是单 transition tree，必须关闭 pheromone 训练")
        if self.transition_terminals is not None:
            object.__setattr__(
                self,
                "transition_terminals",
                tuple(self.transition_terminals),
            )
            if not self.transition_terminals:
                raise ValueError("transition_terminals 不得为空")
        if self.pheromone_terminals is not None:
            object.__setattr__(
                self,
                "pheromone_terminals",
                tuple(self.pheromone_terminals),
            )
            if not self.pheromone_terminals:
                raise ValueError("pheromone_terminals 不得为空")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """计算资源与确定性设置。

    CPU 上可在 GP 个体层使用多个进程，并让每个进程内部使用受控数量的
    PyTorch 线程。CUDA 主实验通常令 ``processes=1``，依靠 batch tensor
    并行，避免多个进程争用同一设备。
    """

    processes: int = 1
    cpu_threads: int = 16
    torch_threads: int = 1
    torch_interop_threads: int = 1
    multiprocessing_start_method: str = "spawn"
    deterministic_algorithms: bool = True
    aco_backend: ExecutionBackend = ExecutionBackend.TORCH
    gpu_devices: tuple[int, ...] = (0,)
    gpu_mode: GPUMode = GPUMode.AUTO
    gpu_block_threads: int = 0
    gpu_memory_fraction: float = 0.80
    gpu_task_chunk_size: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aco_backend",
            ExecutionBackend(self.aco_backend),
        )
        object.__setattr__(self, "gpu_mode", GPUMode(self.gpu_mode))
        object.__setattr__(
            self,
            "gpu_devices",
            tuple(int(device) for device in self.gpu_devices),
        )
        if self.processes < 1:
            raise ValueError("processes 必须为正整数")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads 必须为正整数")
        if self.torch_threads < 1 or self.torch_interop_threads < 1:
            raise ValueError("PyTorch thread 数必须为正整数")
        if self.multiprocessing_start_method not in {"spawn", "forkserver", "fork"}:
            raise ValueError("multiprocessing_start_method 必须为 spawn/forkserver/fork")
        if self.aco_backend is ExecutionBackend.NUMBA_BATCH and self.processes != 1:
            raise ValueError("numba_batch 使用单进程内部线程，processes 必须为 1")
        if len(set(self.gpu_devices)) != len(self.gpu_devices):
            raise ValueError("gpu_devices 不得包含重复设备")
        if any(device < 0 for device in self.gpu_devices):
            raise ValueError("gpu_devices 必须是非负整数")
        if self.gpu_block_threads not in {0, 32, 64}:
            raise ValueError("gpu_block_threads 必须为 0/32/64")
        if not 0.0 < self.gpu_memory_fraction <= 0.8:
            raise ValueError("gpu_memory_fraction 必须位于 (0, 0.8]，至少保留 20%")
        if self.gpu_task_chunk_size < 0:
            raise ValueError("gpu_task_chunk_size 不得为负")
        if self.aco_backend is ExecutionBackend.CUDA_FUSED_FP32:
            if self.processes != 1:
                raise ValueError("CUDA fused 后端使用单进程，processes 必须为 1")
            if self.gpu_mode is GPUMode.CPU:
                raise ValueError("CUDA fused 后端不能使用 gpu_mode=cpu")
            if not self.gpu_devices:
                raise ValueError("CUDA fused 后端至少需要一个 gpu_devices")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """一次训练或测试实验的可复现配置。"""

    experiment_id: str
    root_seed: int
    aco: ACOConfig
    gp: GPConfig = field(default_factory=GPConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    train_scales: tuple[int, ...] = (50, 100)
    validation_scales: tuple[int, ...] = (50, 100)
    test_scales: tuple[int, ...] = (500,)
    validation_seeds: int = 3
    validation_screening_seeds: int = 1
    validation_top_k: int = 5
    noninferiority_tolerance: float = 0.1

    def __post_init__(self) -> None:
        for name in ("train_scales", "validation_scales", "test_scales"):
            values = tuple(int(value) for value in getattr(self, name))
            if not values:
                raise ValueError(f"{name} 不得为空")
            object.__setattr__(self, name, values)
        if self.validation_seeds < 1 or self.validation_screening_seeds < 1:
            raise ValueError("validation seed 数必须为正整数")
        if self.validation_screening_seeds > self.validation_seeds:
            raise ValueError("screening seeds 不得多于完整 validation seeds")
        if self.validation_top_k < 1:
            raise ValueError("validation_top_k 必须为正整数")

    def stable_dict(self) -> dict[str, Any]:
        """递归转换为 YAML/JSON 友好的字典。"""

        gp_values = asdict(self.gp)
        for name in ("transition_terminals", "pheromone_terminals"):
            if gp_values[name] is not None:
                gp_values[name] = list(gp_values[name])
        runtime_values = asdict(self.runtime)
        runtime_values["aco_backend"] = self.runtime.aco_backend.value
        runtime_values["gpu_mode"] = self.runtime.gpu_mode.value
        runtime_values["gpu_devices"] = list(self.runtime.gpu_devices)
        return {
            "experiment_id": self.experiment_id,
            "root_seed": self.root_seed,
            "aco": self.aco.stable_dict(),
            "gp": gp_values,
            "runtime": runtime_values,
            "train_scales": list(self.train_scales),
            "validation_scales": list(self.validation_scales),
            "test_scales": list(self.test_scales),
            "validation_seeds": self.validation_seeds,
            "validation_screening_seeds": self.validation_screening_seeds,
            "validation_top_k": self.validation_top_k,
            "noninferiority_tolerance": self.noninferiority_tolerance,
        }
