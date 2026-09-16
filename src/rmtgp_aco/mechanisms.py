"""受控机制实验接口；未传入控制项时不改变原求解器路径。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Callable


SOURCE_POLICIES = (
    "native_schedule", "iteration_best", "restart_best_only", "global_best_only",
    "native_slots_use_restart_best", "native_slots_use_global_best", "top_k",
    "history_probability", "global_calendar",
)

# trace 为每个逻辑求解、每轮一行，计数不在 instance 维度提前合并。
TRACE_FIELDS = (
    "source_kind", "native_source_kind", "resolved_period", "source_count",
    "target_budget", "actual_budget", "floor_clips", "upper_clips",
    "restart_executed", "restart_would_trigger", "branch_factor",
    "tau_min", "tau_max", "source_length", "source_age", "stagnation",
    "pre_best", "post_best", "pre_mean", "post_mean", "budget_error",
    "source_is_ib", "source_is_gb", "deposit_cv", "pre_top7", "post_top7",
)


@dataclass(frozen=True)
class MechanismConfig:
    """科学配置；来源长度与预算参考长度始终是两个独立量。"""

    restart_policy: str = "native"
    reset_pheromone: bool = True
    reset_epoch_memory: bool = True
    floor_scale: float = 1.0
    hard_upper_clip: bool = False
    source_policy: str = "native_schedule"
    source_count: int = 1
    budget_policy: str = "native_shadow_total"
    p_history: float = 1.0
    initial_tau_scale: float = 1.0
    restart_tau_scale: float = 1.0
    terminal_normalization_horizon: int | None = None
    terminal_clip: bool = False
    tau_headroom_override: float | None = None

    def __post_init__(self) -> None:
        import math
        if self.restart_policy not in ("native", "off", "replay"):
            raise ValueError("未知 restart_policy")
        if self.source_policy not in SOURCE_POLICIES:
            raise ValueError("未知 source_policy")
        if self.budget_policy not in ("native_shadow_total", "actual_source_total"):
            raise ValueError("未知 budget_policy")
        for name in ("floor_scale", "initial_tau_scale", "restart_tau_scale", "p_history"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} 必须有限")
        if self.floor_scale < 0 or min(self.initial_tau_scale, self.restart_tau_scale) <= 0:
            raise ValueError("floor 倍率非负，初始化和重置倍率为正")
        if not 0 <= self.p_history <= 1 or not 1 <= self.source_count <= 32:
            raise ValueError("history 概率或来源数越界")
        if self.source_policy != "top_k" and self.source_count != 1:
            raise ValueError("source_count 仅用于 top_k")
        if self.terminal_normalization_horizon is not None and self.terminal_normalization_horizon < 1:
            raise ValueError("terminal horizon 必须为正")
        if self.tau_headroom_override is not None and not 0 <= self.tau_headroom_override <= 1:
            raise ValueError("TauHeadroom 替代值须位于 [0,1]")
        if self.restart_policy != "replay" and not (self.reset_pheromone and self.reset_epoch_memory):
            raise ValueError("部分状态重置只用于 replay")

    @property
    def digest(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def cuda_prefix(self, record: bool) -> str:
        """开关成为编译源码的一部分，因此自动隔离 RawModule 缓存。"""
        values = {
            "CONTROL": 1, "RECORD": int(record),
            "RESTART": ("off", "native", "replay").index(self.restart_policy),
            "RESET_PHEROMONE": int(self.reset_pheromone),
            "RESET_EPOCH": int(self.reset_epoch_memory),
            "FLOOR_SCALE": f"{self.floor_scale:.17e}f",
            "UPPER": int(self.hard_upper_clip),
            "SOURCE": SOURCE_POLICIES.index(self.source_policy),
            "SOURCE_COUNT": self.source_count,
            "SHADOW_BUDGET": int(self.budget_policy == "native_shadow_total"),
            "P_HISTORY": f"{self.p_history:.17e}f",
            "RESTART_SCALE": f"{self.restart_tau_scale:.17e}f",
            "TERMINAL_H": self.terminal_normalization_horizon or 0,
            "TERMINAL_CLIP": int(self.terminal_clip),
            "HEADROOM": f"{self.tau_headroom_override if self.tau_headroom_override is not None else -1:.17e}f",
            "TRACE_WIDTH": len(TRACE_FIELDS),
        }
        return "\n".join(f"#define RMTGP_MECH_{k} {v}" for k, v in values.items()) + "\n"


@dataclass(frozen=True)
class InstrumentationConfig:
    """记录粒度；heavy 保证将来分叉需要的历史元数据已经维护。"""

    level: str = "light"
    aggregate_every: int = 25
    snapshot_iterations: tuple[int, ...] = ()
    # 独立版本，不能把历史 light 文件解释为完整机制诊断。
    profile: str = "legacy"
    schema_version: int = 1
    commit_every: int = 100
    checkpoint_every: int = 500
    probe_ants: tuple[int, ...] = (0, 8, 16, 24)

    def __post_init__(self) -> None:
        object.__setattr__(self,"probe_ants",tuple(self.probe_ants))
        object.__setattr__(self,"snapshot_iterations",tuple(self.snapshot_iterations))
        if self.level not in ("off", "light", "heavy") or self.aggregate_every < 1:
            raise ValueError("无效审计配置")
        if self.profile not in ("legacy", "mechanism_v3"):
            raise ValueError("未知诊断 profile")
        if self.profile == "mechanism_v3":
            if self.level == "off" or self.schema_version != 3:
                raise ValueError("mechanism_v3 必须启用记录并使用 schema 3")
            if self.commit_every != 100 or self.checkpoint_every != 500:
                raise ValueError("正式诊断固定每 100 轮提交、每 500 轮恢复检查点")
            if self.aggregate_every != 25 or self.probe_ants != (0, 8, 16, 24):
                raise ValueError("正式采样规则固定为 25 轮、蚂蚁 0/8/16/24")

    @property
    def detailed(self) -> bool:
        return self.profile == "mechanism_v3"

    def cuda_prefix(self) -> str:
        return (f"#define RMTGP_DIAG_V3 {int(self.detailed)}\n"
                f"#define RMTGP_DIAG_EVERY {self.aggregate_every}\n"
                f"#define RMTGP_DIAG_RING {self.commit_every}\n")


@dataclass
class SolverControl:
    """一次调用的科学配置和运行钩子。observer 的工作不计为无审计部署时间。"""

    mechanism: MechanismConfig = MechanismConfig()
    instrumentation: InstrumentationConfig = InstrumentationConfig()
    # replay 数组为 [B,H]，所有模型共享；source slots 为 0/1/2。
    replay_restarts: Any = None
    source_slots: Any = None
    observer: Callable[[str, int, dict[str, Any]], None] | None = None
    resume: Any = None
    stop_iteration: int | None = None
    collected: list[dict[str, Any]] | None = None
    kernel_hashes: set[str] = field(default_factory=set)
    stage_timings: list[dict[str, Any]] = field(default_factory=list)


def factorial_conditions() -> dict[str, MechanismConfig]:
    return {
        f"C{r}{f}{h}": MechanismConfig(
            restart_policy="native" if r else "off",
            floor_scale=float(f),
            source_policy="native_schedule" if h else "iteration_best",
        )
        for r in (1, 0) for f in (1, 0) for h in (1, 0)
    }


def native_schedule(iteration: int, restart_iteration: int, restart_found_best: int) -> tuple[int, int]:
    """独立的整数日程验收函数，返回 (周期, IB/RB/GB 编号)。"""
    age = max(iteration - restart_iteration - 1, 0)
    period = 25 if age < 25 else 5 if age < 75 else 3 if age < 125 else 2 if age < 250 else 1
    source = 0 if iteration % period else (2 if period == 1 and iteration - restart_found_best > 50 else 1)
    return period, source
