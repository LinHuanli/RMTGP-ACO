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


class LocalSearch(StrEnum):
    """在每轮构造后作用于全部蚂蚁 tour 的局部搜索。"""

    NONE = "none"
    TWO_OPT = "two_opt"
    THREE_OPT = "three_opt"


class LocalSearchProfile(StrEnum):
    """局部搜索及其配套 ACO 动力学的语义 profile。"""

    ACOTSP = "acotsp"


class ExecutionBackend(StrEnum):
    """ACO 数值内核后端。

    ``torch`` 保留为易审计的参考实现；``numba`` 是标量/进程级参考；
    ``numba_batch`` 在一个进程内以 population×instance 任务矩阵并行。
    """

    TORCH = "torch"
    NUMBA = "numba"
    NUMBA_BATCH = "numba_batch"
    CUDA_FUSED_FP32 = "cuda_fused_fp32"
    CUDA_TILED_V2 = "cuda_tiled_v2"


class CudaProvider(StrEnum):
    """CUDA v2 kernel 的实现方式。"""

    RAW_CUDA = "raw_cuda"
    CUTILE = "cutile"
    AUTO = "auto"


class CudaPrecision(StrEnum):
    """CUDA v2 搜索过程的数值 profile。

    所有 profile 返回的 tour 均由 CPU float64 距离矩阵重新计分。
    ``fp16_mixed`` 和 ``bf16_mixed`` 只压缩静态只读表，动态信息素、
    GP primitive 和概率归约仍使用 float32。
    """

    FP64 = "fp64"
    FP32 = "fp32"
    FP32_FAST = "fp32_fast"
    FP16_MIXED = "fp16_mixed"
    BF16_MIXED = "bf16_mixed"
    FP16_SEARCH = "fp16_search"
    FP8_E4M3 = "fp8_e4m3"
    NVFP4 = "nvfp4"
    AUTO = "auto"


class CudaTaskOrder(StrEnum):
    """population×instance task 在网格中的线性顺序。"""

    PROGRAM_MAJOR = "program_major"
    INSTANCE_MAJOR = "instance_major"
    AUTO = "auto"


class GPUMode(StrEnum):
    """CUDA 设备的使用方式。"""

    CPU = "cpu"
    SINGLE = "single"
    DUAL = "dual"
    CAMPAIGN = "campaign"
    AUTO = "auto"


class FitnessMode(StrEnum):
    """GP 个体在一个训练 mini-batch 上的评分方式。"""

    ABSOLUTE_GAP = "absolute_gap"
    # ``paired_ucb`` 是早期 checkpoint/config 的兼容名称。新实验应显式
    # 使用 paired_final_ucb、paired_basin_ucb 或 paired_combined_ucb。
    PAIRED_UCB = "paired_ucb"
    PAIRED_FINAL_UCB = "paired_final_ucb"
    PAIRED_BASIN_UCB = "paired_basin_ucb"
    PAIRED_COMBINED_UCB = "paired_combined_ucb"

    @property
    def is_paired(self) -> bool:
        return self is not FitnessMode.ABSOLUTE_GAP

    @property
    def uses_basin(self) -> bool:
        return self in {
            FitnessMode.PAIRED_BASIN_UCB,
            FitnessMode.PAIRED_COMBINED_UCB,
        }


class SelectionMode(StrEnum):
    """validation checkpoint 的选择和门控协议。"""

    LEGACY_NONINFERIORITY = "legacy_noninferiority"
    STRICT_SUPERIORITY = "strict_superiority"


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
    local_search: LocalSearch = LocalSearch.NONE
    local_search_profile: LocalSearchProfile = LocalSearchProfile.ACOTSP
    local_search_candidate_size: int = 20
    local_search_dlb: bool = True

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

    @classmethod
    def acotsp_local_search_default(
        cls,
        variant: ACOVariant | str,
        *,
        local_search: LocalSearch | str = LocalSearch.TWO_OPT,
        iterations: int = 100,
        ants: int = 32,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        acs_synchronous: bool = True,
    ) -> ACOConfig:
        """建立采用 ACOTSP 局部搜索语义、但统一 32 只蚂蚁的配置。

        原始 ACOTSP 的局部搜索默认参数随算法改变蚂蚁数。本文为了让
        AS/ACS/MMAS 的计算预算可比，固定蚂蚁数，保留其 rho、q0、候选表、
        DLB、MMAS 边界及动态强化周期等其余语义。
        """

        selected = ACOVariant(variant)
        search = LocalSearch(local_search)
        if search is LocalSearch.NONE:
            raise ValueError("局部搜索默认配置要求 two_opt 或 three_opt")
        common: dict[str, Any] = {
            "variant": selected,
            "ants": ants,
            "alpha": 1.0,
            "beta": 2.0,
            "candidate_size": 20,
            "iterations": iterations,
            "device": device,
            "dtype": dtype,
            "acs_synchronous": acs_synchronous,
            "local_search": search,
            "local_search_profile": LocalSearchProfile.ACOTSP,
            "local_search_candidate_size": 20,
            "local_search_dlb": True,
        }
        if selected is ACOVariant.AS:
            return cls(rho=0.5, q0=0.0, **common)
        if selected is ACOVariant.ACS:
            return cls(rho=0.1, q0=0.98, xi=0.1, **common)
        return cls(rho=0.2, q0=0.0, **common)

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
        object.__setattr__(self, "local_search", LocalSearch(self.local_search))
        object.__setattr__(
            self,
            "local_search_profile",
            LocalSearchProfile(self.local_search_profile),
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
        if self.local_search_candidate_size < 1:
            raise ValueError("local_search_candidate_size 必须为正整数")

    def resolve_ants(self, n: int) -> int:
        """把 ACOTSP 的 `ants=n` 约定解析为实际蚂蚁数。"""

        if n < 2:
            raise ValueError("TSP 至少需要两个城市")
        return n if self.ants is None else self.ants

    def resolve_candidate_size(self, n: int) -> int:
        """候选列表不能包含当前城市，因此至多为 `n-1`。"""

        return min(self.candidate_size, n - 1)

    def resolve_local_search_candidate_size(self, n: int) -> int:
        """局部搜索候选表深度，至多包含其余 ``n-1`` 个城市。"""

        if n < 2:
            raise ValueError("TSP 至少需要两个城市")
        return min(self.local_search_candidate_size, n - 1)

    @property
    def uses_local_search(self) -> bool:
        """是否在每轮构造后对全部蚂蚁执行局部搜索。"""

        return self.local_search is not LocalSearch.NONE

    def stable_dict(self) -> dict[str, Any]:
        """返回可序列化、可哈希的配置。"""

        values = asdict(self)
        values["variant"] = self.variant.value
        values["transition_integration"] = self.transition_integration.value
        values["pheromone_integration"] = self.pheromone_integration.value
        values["local_search"] = self.local_search.value
        values["local_search_profile"] = self.local_search_profile.value
        values["dtype"] = str(self.dtype).removeprefix("torch.")
        return values

    def baseline_stable_dict(self) -> dict[str, Any]:
        """返回只描述无 GP program 时原始 ACO 行为的稳定配置。

        residual 半径和两种 integration mode 只有在对应 GP program 存在时
        才会进入求解路径。把它们放进原始 ACO cache key 会使
        residual/replacement 消融无法共享数学上完全相同的 baseline。
        """

        values = self.stable_dict()
        for name in (
            "gamma_transition",
            "gamma_pheromone",
            "transition_integration",
            "pheromone_integration",
        ):
            values.pop(name)
        return values

    @property
    def config_hash(self) -> str:
        """用于 cache 和 artifact 的稳定短哈希。"""

        payload = json.dumps(self.stable_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def baseline_behavior_hash(self) -> str:
        """无 GP program 的原始 ACO 行为哈希。"""

        payload = json.dumps(
            self.baseline_stable_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
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
    fitness_mode: FitnessMode = FitnessMode.ABSOLUTE_GAP
    fitness_ucb_z: float = 1.0
    baseline_anchor: bool = False
    basin_top_q: int = 7
    basin_weight: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "fitness_mode", FitnessMode(self.fitness_mode))
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
        if self.fitness_ucb_z < 0.0:
            raise ValueError("fitness_ucb_z 不得为负")
        if self.basin_top_q < 1:
            raise ValueError("basin_top_q 必须为正整数")
        if not 0.0 <= self.basin_weight <= 1.0:
            raise ValueError("basin_weight 必须位于 [0, 1]")
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
    cuda_provider: CudaProvider = CudaProvider.RAW_CUDA
    cuda_precision: CudaPrecision = CudaPrecision.FP32_FAST
    cuda_candidate_lanes: int = 8
    cuda_register_cap: int = 0
    cuda_task_order: CudaTaskOrder = CudaTaskOrder.INSTANCE_MAJOR
    cuda_generated_gp: bool = True
    cuda_graph_replay: bool = False
    cuda_tuning_manifest: str | None = None
    cuda_ls_warps_per_block: int = 8
    cuda_three_opt_block_threads: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "aco_backend",
            ExecutionBackend(self.aco_backend),
        )
        object.__setattr__(self, "gpu_mode", GPUMode(self.gpu_mode))
        object.__setattr__(
            self,
            "cuda_provider",
            CudaProvider(self.cuda_provider),
        )
        object.__setattr__(
            self,
            "cuda_precision",
            CudaPrecision(self.cuda_precision),
        )
        object.__setattr__(
            self,
            "cuda_task_order",
            CudaTaskOrder(self.cuda_task_order),
        )
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
        if self.cuda_candidate_lanes not in {0, 1, 4, 8, 16, 32}:
            raise ValueError("cuda_candidate_lanes 必须为 0/1/4/8/16/32")
        if self.cuda_register_cap not in {0, 64, 80, 96, 112, 128}:
            raise ValueError("cuda_register_cap 必须为 0/64/80/96/112/128")
        if self.cuda_ls_warps_per_block not in {4, 8}:
            raise ValueError("cuda_ls_warps_per_block 必须为 4 或 8")
        if self.cuda_three_opt_block_threads not in {128, 256, 512}:
            raise ValueError(
                "cuda_three_opt_block_threads 必须为 128/256/512"
            )
        if self.cuda_provider is CudaProvider.CUTILE and (
            self.cuda_precision
            not in {
                CudaPrecision.FP32,
                CudaPrecision.FP32_FAST,
                CudaPrecision.AUTO,
            }
        ):
            raise ValueError("cuTile prototype 当前只支持 fp32/fp32_fast/auto")
        if self.aco_backend in {
            ExecutionBackend.CUDA_FUSED_FP32,
            ExecutionBackend.CUDA_TILED_V2,
        }:
            if self.processes != 1:
                raise ValueError("CUDA 后端使用单进程，processes 必须为 1")
            if self.gpu_mode is GPUMode.CPU:
                raise ValueError("CUDA 后端不能使用 gpu_mode=cpu")
            if not self.gpu_devices:
                raise ValueError("CUDA 后端至少需要一个 gpu_devices")


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
    selection_mode: SelectionMode = SelectionMode.LEGACY_NONINFERIORITY
    superiority_min_relative_improvement: float = 0.10
    selection_confidence: float = 0.95
    selection_bootstrap_replicates: int = 10_000
    quality_tie_tolerance: float = 0.01
    training_horizon_schedule: tuple[tuple[int, int], ...] | None = None
    validation_monitor_interval: int = 1
    cpu_fp64_final_audit: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_mode",
            SelectionMode(self.selection_mode),
        )
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
        if not 0.0 <= self.superiority_min_relative_improvement < 1.0:
            raise ValueError(
                "superiority_min_relative_improvement 必须位于 [0, 1)"
            )
        if not 0.5 < self.selection_confidence < 1.0:
            raise ValueError("selection_confidence 必须位于 (0.5, 1)")
        if self.selection_bootstrap_replicates < 100:
            raise ValueError("selection_bootstrap_replicates 至少为 100")
        if self.quality_tie_tolerance < 0.0:
            raise ValueError("quality_tie_tolerance 不得为负")
        if self.validation_monitor_interval < 1:
            raise ValueError("validation_monitor_interval 必须为正整数")
        if self.gp.fitness_mode.uses_basin:
            for scale in self.train_scales:
                if self.gp.basin_top_q > self.aco.resolve_ants(scale):
                    raise ValueError(
                        "basin_top_q 不得超过训练规模对应的蚂蚁数"
                    )
            if not self.aco.uses_local_search:
                raise ValueError(
                    "basin fitness 只用于 local-search 训练；"
                    "无局部搜索实验应使用 final fitness"
                )
        if self.training_horizon_schedule is not None:
            schedule = tuple(
                (int(end_generation), int(iterations))
                for end_generation, iterations in self.training_horizon_schedule
            )
            if not schedule:
                raise ValueError("training_horizon_schedule 不得为空")
            previous = 0
            for end_generation, iterations in schedule:
                if end_generation <= previous:
                    raise ValueError(
                        "training_horizon_schedule 的代数边界必须严格递增"
                    )
                if iterations < 1:
                    raise ValueError("training horizon 必须为正整数")
                previous = end_generation
            if schedule[-1][0] != self.gp.generations:
                raise ValueError(
                    "training_horizon_schedule 最后边界必须等于 gp.generations"
                )
            object.__setattr__(self, "training_horizon_schedule", schedule)

    def iterations_for_generation(self, generation: int) -> int:
        """返回指定 GP generation 使用的 ACO 迭代数。"""

        if generation < 1 or generation > self.gp.generations:
            raise ValueError(
                f"generation 必须位于 [1, {self.gp.generations}]"
            )
        if self.training_horizon_schedule is None:
            return self.aco.iterations
        for end_generation, iterations in self.training_horizon_schedule:
            if generation <= end_generation:
                return iterations
        raise RuntimeError("training_horizon_schedule 未覆盖当前 generation")

    def stable_dict(self) -> dict[str, Any]:
        """递归转换为 YAML/JSON 友好的字典。"""

        gp_values = asdict(self.gp)
        gp_values["fitness_mode"] = self.gp.fitness_mode.value
        for name in ("transition_terminals", "pheromone_terminals"):
            if gp_values[name] is not None:
                gp_values[name] = list(gp_values[name])
        runtime_values = asdict(self.runtime)
        runtime_values["aco_backend"] = self.runtime.aco_backend.value
        runtime_values["gpu_mode"] = self.runtime.gpu_mode.value
        runtime_values["gpu_devices"] = list(self.runtime.gpu_devices)
        runtime_values["cuda_provider"] = self.runtime.cuda_provider.value
        runtime_values["cuda_precision"] = self.runtime.cuda_precision.value
        runtime_values["cuda_task_order"] = self.runtime.cuda_task_order.value
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
            "selection_mode": self.selection_mode.value,
            "superiority_min_relative_improvement": (
                self.superiority_min_relative_improvement
            ),
            "selection_confidence": self.selection_confidence,
            "selection_bootstrap_replicates": (
                self.selection_bootstrap_replicates
            ),
            "quality_tie_tolerance": self.quality_tie_tolerance,
            "training_horizon_schedule": (
                None
                if self.training_horizon_schedule is None
                else [
                    [end_generation, iterations]
                    for end_generation, iterations
                    in self.training_horizon_schedule
                ]
            ),
            "validation_monitor_interval": self.validation_monitor_interval,
            "cpu_fp64_final_audit": self.cpu_fp64_final_audit,
        }
