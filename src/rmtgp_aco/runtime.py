"""CPU/GPU 线程设置与可重复运行控制。"""

from __future__ import annotations

import os

import torch

from .config import RuntimeConfig


def configure_runtime(config: RuntimeConfig) -> dict[str, int | bool | str]:
    """应用一次运行的线程与确定性配置并返回实际状态。

    ``set_num_interop_threads`` 在 PyTorch 开始并行工作后不可再次修改；
    因此仅在当前值不同时尝试设置，并给出清晰错误，而不是静默偏离配置。
    """

    os.environ["OMP_NUM_THREADS"] = str(config.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(config.torch_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(config.cpu_threads)
    if config.aco_backend.value == "numba_batch":
        from numba import set_num_threads

        set_num_threads(config.cpu_threads)
    torch.set_num_threads(config.torch_threads)
    current_interop = torch.get_num_interop_threads()
    if current_interop != config.torch_interop_threads:
        try:
            torch.set_num_interop_threads(config.torch_interop_threads)
        except RuntimeError as exc:
            raise RuntimeError(
                "PyTorch inter-op 线程池已经启动；请在任何 tensor 计算前调用 "
                "configure_runtime，或保持当前 torch_interop_threads="
                f"{current_interop}"
            ) from exc
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    return runtime_state(config)


def runtime_state(config: RuntimeConfig) -> dict[str, int | bool | str]:
    """返回 artifact 应记录的实际并行状态。"""

    return {
        "processes": config.processes,
        "cpu_threads": config.cpu_threads,
        "aco_backend": config.aco_backend.value,
        "gpu_mode": config.gpu_mode.value,
        "gpu_devices": ",".join(str(device) for device in config.gpu_devices),
        "gpu_block_threads": config.gpu_block_threads,
        "gpu_task_chunk_size": config.gpu_task_chunk_size,
        "cuda_provider": config.cuda_provider.value,
        "cuda_precision": config.cuda_precision.value,
        "cuda_candidate_lanes": config.cuda_candidate_lanes,
        "cuda_register_cap": config.cuda_register_cap,
        "cuda_task_order": config.cuda_task_order.value,
        "cuda_generated_gp": config.cuda_generated_gp,
        "cuda_graph_replay": config.cuda_graph_replay,
        "cuda_tuning_manifest": config.cuda_tuning_manifest or "",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "multiprocessing_start_method": config.multiprocessing_start_method,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
