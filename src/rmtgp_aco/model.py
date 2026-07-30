"""算法层共享的数据结构。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class ProblemBatch:
    """同一城市规模的一批 TSP 实例。"""

    coords: torch.Tensor
    distances: torch.Tensor
    heuristic: torch.Tensor
    nn_indices: torch.Tensor
    full_nn_rank: torch.Tensor
    reference_tour: torch.Tensor
    reference_length: torch.Tensor
    instance_ids: tuple[str, ...]
    coordinate_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.coords.ndim != 3 or self.coords.shape[-1] != 2:
            raise ValueError("coords 必须具有 [B,n,2] shape")
        batch, n, _ = self.coords.shape
        if self.distances.shape != (batch, n, n):
            raise ValueError("distances 必须具有 [B,n,n] shape")
        if self.heuristic.shape != (batch, n, n):
            raise ValueError("heuristic 必须具有 [B,n,n] shape")
        if self.reference_tour.shape != (batch, n + 1):
            raise ValueError("reference_tour 必须具有 [B,n+1] shape")
        if self.reference_length.shape != (batch,):
            raise ValueError("reference_length 必须具有 [B] shape")
        if len(self.instance_ids) != batch:
            raise ValueError("instance_ids 数量必须等于 batch size")
        if len(self.coordinate_hashes) != batch:
            raise ValueError("coordinate_hashes 数量必须等于 batch size")

    @property
    def batch_size(self) -> int:
        return self.coords.shape[0]

    @property
    def n(self) -> int:
        return self.coords.shape[1]

    @property
    def device(self) -> torch.device:
        return self.coords.device

    def to(self, device: str | torch.device) -> ProblemBatch:
        """把全部 tensor 移到同一 device。"""

        return ProblemBatch(
            coords=self.coords.to(device),
            distances=self.distances.to(device),
            heuristic=self.heuristic.to(device),
            nn_indices=self.nn_indices.to(device),
            full_nn_rank=self.full_nn_rank.to(device),
            reference_tour=self.reference_tour.to(device),
            reference_length=self.reference_length.to(device),
            instance_ids=self.instance_ids,
            coordinate_hashes=self.coordinate_hashes,
        )


@dataclass(slots=True)
class TransitionContext:
    """一次候选选择所需的动态张量。"""

    current_city: torch.Tensor
    candidates: torch.Tensor
    feasible_mask: torch.Tensor
    base_score: torch.Tensor
    base_probability: torch.Tensor
    terminals: Mapping[str, torch.Tensor]


@dataclass(slots=True)
class DepositEventBatch:
    """一轮全局强化中的来源 tour 与边。"""

    edge_u: torch.Tensor
    edge_v: torch.Tensor
    edge_id: torch.Tensor
    source_length: torch.Tensor
    base_deposit: torch.Tensor
    base_budget: torch.Tensor
    terminals: Mapping[str, torch.Tensor]


@dataclass(slots=True)
class RunDiagnostics:
    """不影响算法结果、但用于审计数值和控制流的计数器。"""

    uniform_fallback_count: int = 0
    nan_sanitized_count: int = 0
    candidate_fallback_count: int = 0
    bound_clip_count: int = 0
    mmas_restart_count: int = 0
    local_search_move_count: int = 0
    local_search_candidate_check_count: int = 0
    local_search_improved_tour_count: int = 0
    local_search_pass_count: int = 0


@dataclass(slots=True)
class RunResult:
    """一个 batch 的 ACO 运行结果。"""

    best_tour: torch.Tensor
    best_length: torch.Tensor
    best_iteration: torch.Tensor
    anytime_best: torch.Tensor
    wall_time_sec: float
    constructed_tours: int
    diagnostics: RunDiagnostics = field(default_factory=RunDiagnostics)
    backend_metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class PopulationQualityResult:
    """population-batched 训练内核的轻量输出。

    第一维为唯一 GP genotype，第二维为 instance。后端返回每个任务的最优
    tour 以支持 CPU float64 精确计分，但不返回完整 colony 或 anytime curve，
    从而控制 Python object 数量和跨设备传输量。
    """

    best_tour: torch.Tensor
    best_length: torch.Tensor
    best_iteration: torch.Tensor
    diagnostics: torch.Tensor
    wall_time_sec: float
    constructed_tours: int
    # ``basin_mean_length[p,b]`` 是每轮 post-LS 最好 q 只蚂蚁平均
    # tour length 的跨轮均值。它不是 global-best anytime AUC，因而能保留
    # 整个 colony 进入优质局部搜索盆地的密集学习信号。
    basin_mean_length: torch.Tensor | None = None
    # 以下三个字段只在显式 local-search signal audit 中返回。正常训练不
    # 分配相应轨迹或把 colony 搬回主机。
    pre_basin_mean_length: torch.Tensor | None = None
    edge_retention: torch.Tensor | None = None
    final_colony_tour: torch.Tensor | None = None
    final_pre_colony_tour: torch.Tensor | None = None
    backend_metrics: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.best_length.ndim != 2:
            raise ValueError("population best_length 必须具有 [P,B] shape")
        if (
            self.best_tour.ndim != 3
            or self.best_tour.shape[:2] != self.best_length.shape
        ):
            raise ValueError("population best_tour 必须具有 [P,B,n+1] shape")
        if self.best_iteration.shape != self.best_length.shape:
            raise ValueError("population best_iteration shape 不一致")
        if (
            self.diagnostics.ndim != 2
            or self.diagnostics.shape[0] != self.best_length.shape[0]
            or self.diagnostics.shape[1] < 4
        ):
            raise ValueError("population diagnostics 必须具有 [P,D] shape，D>=4")
        for name in (
            "basin_mean_length",
            "pre_basin_mean_length",
            "edge_retention",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != self.best_length.shape:
                raise ValueError(f"population {name} 必须具有 [P,B] shape")
        for name in ("final_colony_tour", "final_pre_colony_tour"):
            value = getattr(self, name)
            if value is not None and (
                value.ndim != 4
                or value.shape[:2] != self.best_length.shape
                or value.shape[-1] != self.best_tour.shape[-1]
            ):
                raise ValueError(
                    f"population {name} 必须具有 [P,B,A,n+1] shape"
                )


@dataclass(slots=True)
class PopulationRunResult:
    """锁定 programs 的批量测试结果，包含完整 anytime 曲线。

    与训练用 ``PopulationQualityResult`` 分离，避免训练阶段为每个
    program×instance 保存 ``iterations`` 长度的轨迹。
    """

    best_tour: torch.Tensor
    best_length: torch.Tensor
    best_iteration: torch.Tensor
    anytime_best: torch.Tensor
    diagnostics: torch.Tensor
    wall_time_sec: float
    constructed_tours: int
    backend_metrics: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.best_length.ndim != 2:
            raise ValueError("population run best_length 必须具有 [P,B] shape")
        if self.best_tour.shape[:2] != self.best_length.shape:
            raise ValueError("population run best_tour shape 不一致")
        if self.best_iteration.shape != self.best_length.shape:
            raise ValueError("population run best_iteration shape 不一致")
        if self.anytime_best.shape[:2] != self.best_length.shape:
            raise ValueError("population run anytime_best shape 不一致")
        if (
            self.diagnostics.ndim != 2
            or self.diagnostics.shape[0] != self.best_length.shape[0]
            or self.diagnostics.shape[1] < 4
        ):
            raise ValueError("population run diagnostics 必须具有 [P,D] shape，D>=4")
