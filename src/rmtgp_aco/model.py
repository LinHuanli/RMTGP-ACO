"""算法层共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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

    @property
    def batch_size(self) -> int:
        return self.coords.shape[0]

    @property
    def n(self) -> int:
        return self.coords.shape[1]

    @property
    def device(self) -> torch.device:
        return self.coords.device

    def to(self, device: str | torch.device) -> "ProblemBatch":
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
