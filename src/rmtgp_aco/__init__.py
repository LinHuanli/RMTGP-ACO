"""RMTGP-ACO 研究实现。

该包提供数据解析、AS/ACS/MMAS、Strongly Typed 双树 GP、训练与实验接口。
"""

from .config import (
    ACOConfig,
    ACOVariant,
    CudaPrecision,
    CudaProvider,
    CudaTaskOrder,
    ExecutionBackend,
    ExperimentConfig,
    FitnessMode,
    GPConfig,
    GPUMode,
    LocalSearch,
    LocalSearchProfile,
    LSGainSemantics,
    PheromoneIntegration,
    RuntimeConfig,
    SelectionMode,
    TransitionIntegration,
)

__all__ = [
    "ACOConfig",
    "ACOVariant",
    "CudaPrecision",
    "CudaProvider",
    "CudaTaskOrder",
    "ExecutionBackend",
    "ExperimentConfig",
    "FitnessMode",
    "GPUMode",
    "GPConfig",
    "LSGainSemantics",
    "LocalSearch",
    "LocalSearchProfile",
    "PheromoneIntegration",
    "RuntimeConfig",
    "SelectionMode",
    "TransitionIntegration",
]

__version__ = "0.5.0"
