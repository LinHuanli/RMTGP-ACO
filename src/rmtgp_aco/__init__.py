"""RMTGP-ACO 研究实现。

该包提供数据解析、AS/ACS/MMAS、Strongly Typed 双树 GP、训练与实验接口。
"""

from .config import (
    ACOConfig,
    ACOVariant,
    ExecutionBackend,
    ExperimentConfig,
    GPConfig,
    PheromoneIntegration,
    RuntimeConfig,
    TransitionIntegration,
)

__all__ = [
    "ACOConfig",
    "ACOVariant",
    "ExecutionBackend",
    "ExperimentConfig",
    "GPConfig",
    "PheromoneIntegration",
    "RuntimeConfig",
    "TransitionIntegration",
]

__version__ = "0.3.0"
