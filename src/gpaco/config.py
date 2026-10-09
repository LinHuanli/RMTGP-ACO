"""将科学预算与硬件执行设置分离，避免调优时悄悄减少工作量。"""

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256


class InfeasiblePlan(ValueError):
    """合法科学工作负载在某个编译资源布局下无法启动，不代表数值失败。"""


@dataclass(frozen=True)
class SearchConfig:
    variant: str = "as"
    ants: int = 32
    iterations: int = 500
    candidate_size: int = 20
    gamma: float = 1 / 3
    epsilon: float = 1e-12
    rho: float | None = None
    q0: float = 0.9
    xi: float = 0.1
    mmas_period: int = 25
    mmas_p_best: float = 0.05
    branch_period: int = 100
    branch_lambda: float = 0.05
    branch_threshold: float = 1.00001
    restart_stagnation: int = 250

    def __post_init__(self):
        if self.variant not in ("as", "acs", "mmas"):
            raise ValueError("variant 必须为 as/acs/mmas")
        if not 1 <= self.ants <= 32 or self.iterations < 1 or self.candidate_size < 1:
            raise ValueError("要求 1<=ants<=32，iterations 和 candidate_size 为正")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma 必须在 [0,1) 内")
        if self.rho is None:
            object.__setattr__(self, "rho", {"as": 0.5, "acs": 0.1, "mmas": 0.02}[self.variant])
        if not 0 < self.rho <= 1 or not 0 <= self.q0 <= 1 or not 0 < self.xi <= 1:
            raise ValueError("无效的 ACO 概率参数")
        if self.mmas_period < 1 or self.branch_period < 1 or self.restart_stagnation < 0:
            raise ValueError("MMAS 周期必须为正，停滞阈值不能为负")
        if not 0 < self.mmas_p_best < 1 or not 0 <= self.branch_lambda <= 1:
            raise ValueError("无效的 MMAS 参数")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("数值保护 epsilon 必须是有限正数")
        if not math.isfinite(self.branch_threshold) or self.branch_threshold < 0:
            raise ValueError("分支阈值必须是有限非负数")
        if self.iterations >= 2**31:
            raise ValueError("迭代编号超出有符号 32 位宿主索引")

    @property
    def variant_id(self):
        return ("as", "acs", "mmas").index(self.variant)


@dataclass(frozen=True)
class ExecutionPlan:
    backend: str = "cuda_existing"
    candidate_lanes: int = 8
    active_tasks: int = 3200
    cpu_threads: int = 1
    generated: bool = False
    profile_stages: bool = False

    def __post_init__(self):
        if self.backend not in ("cpu_existing", "cuda_existing"):
            raise ValueError("未实现的执行后端")
        if self.candidate_lanes not in (4, 8, 16, 32):
            raise ValueError("candidate_lanes 必须是 4/8/16/32")
        if min(self.active_tasks, self.cpu_threads) < 1:
            raise ValueError("任务数与线程数必须为正")


def config_hash(value) -> str:
    """规范哈希用于比较组可比性检查，不依赖字典插入顺序。"""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
