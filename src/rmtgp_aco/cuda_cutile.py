"""cuTile candidate-score prototype。

该模块只回答 tile 编程是否适合规则的 ``task×ant×candidate`` 子问题。
完整 ACO 仍包含 visited set、roulette、ACS local update 和 GP 控制流，不能
根据这个规则子核的结果直接声称完整 solver 已获得同样加速。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from statistics import median

import numpy as np
import torch

try:
    import cuda.tile as ct
except ImportError:  # pragma: no cover - 允许 CPU-only 安装导入本包
    ct = None


if ct is not None:

    @ct.kernel(occupancy=4)
    def _candidate_score_cutile(
        pheromone,
        heuristic,
        output,
        ants: ct.Constant[int],
        candidates: ct.Constant[int],
    ):
        task = ct.bid(0)
        tau = ct.load(
            pheromone,
            (task, 0, 0),
            shape=(1, ants, candidates),
        )
        eta = ct.load(
            heuristic,
            (task, 0, 0),
            shape=(1, ants, candidates),
        )
        ct.store(
            output,
            (task, 0, 0),
            tau * eta * eta,
        )


_RAW_SOURCE = r"""
extern "C" __global__ void candidate_score_raw(
    const float* pheromone,
    const float* heuristic,
    float* output,
    int tasks,
    int ants,
    int candidates
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = tasks * ants * candidates;
    if (index >= count) {
        return;
    }
    const float tau = pheromone[index];
    const float eta = heuristic[index];
    output[index] = tau * eta * eta;
}
"""


def _torch_timings(launch, repeats: int) -> list[float]:
    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def benchmark_cutile_candidate_scores(
    *,
    device: int = 0,
    tasks: int = 2528,
    ants: int = 32,
    candidates: int = 20,
    repeats: int = 20,
    seed: int = 20260730,
) -> dict[str, object]:
    """配对比较 cuTile 和简单 raw CUDA 子核，单位为毫秒。"""

    if ct is None:
        raise RuntimeError("cuda-tile 未安装")
    if min(tasks, ants, candidates, repeats) < 1:
        raise ValueError("tasks/ants/candidates/repeats 必须为正整数")
    tile_candidates = 1 << (candidates - 1).bit_length()

    import cupy as cp

    # 非交互 batch shell 可能不读取 ~/.zshrc。优先使用当前环境；只有在
    # tileiras 完全不可见时才回退到集群的系统 CUDA 13 toolkit。
    if shutil.which("tileiras") is None:
        system_cuda = Path("/opt/cuda")
        compiler = system_cuda / "bin" / "tileiras"
        if compiler.is_file():
            os.environ["CUDA_HOME"] = str(system_cuda)
            os.environ["PATH"] = (
                f"{system_cuda / 'bin'}:{os.environ.get('PATH', '')}"
            )

    torch.manual_seed(seed)
    with torch.cuda.device(device):
        pheromone = torch.rand(
            (tasks, ants, tile_candidates),
            dtype=torch.float32,
            device=f"cuda:{device}",
        )
        heuristic = torch.rand_like(pheromone)
        cutile_output = torch.empty_like(pheromone)

        def launch_cutile() -> None:
            ct.launch(
                torch.cuda.current_stream(),
                (tasks,),
                _candidate_score_cutile,
                (
                    pheromone,
                    heuristic,
                    cutile_output,
                    ants,
                    tile_candidates,
                ),
            )

        launch_cutile()
        torch.cuda.synchronize(device)
        cutile_ms = _torch_timings(launch_cutile, repeats)

        # DLPack 建立零拷贝 CuPy views，保证两条路径读取完全相同的数据。
        with cp.cuda.Device(device):
            tau_cp = cp.from_dlpack(pheromone)
            eta_cp = cp.from_dlpack(heuristic)
            raw_output_cp = cp.empty_like(tau_cp)
            module = cp.RawModule(
                code=_RAW_SOURCE,
                options=("--std=c++17",),
                backend="nvrtc",
            )
            kernel = module.get_function("candidate_score_raw")

            def launch_raw() -> None:
                raw_threads = 256
                raw_blocks = (
                    tasks * ants * tile_candidates + raw_threads - 1
                ) // raw_threads
                kernel(
                    (raw_blocks,),
                    (raw_threads,),
                    (
                        tau_cp,
                        eta_cp,
                        raw_output_cp,
                        np.int32(tasks),
                        np.int32(ants),
                        np.int32(tile_candidates),
                    ),
                )

            launch_raw()
            cp.cuda.get_current_stream().synchronize()
            raw_ms: list[float] = []
            for _ in range(repeats):
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()
                launch_raw()
                end.record()
                end.synchronize()
                raw_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
            raw_output = torch.from_dlpack(raw_output_cp)

        torch.testing.assert_close(cutile_output, raw_output, rtol=0.0, atol=0.0)
        raw_median = median(raw_ms)
        cutile_median = median(cutile_ms)
        device_name = torch.cuda.get_device_name(device)

    return {
        "schema_version": 1,
        "scope": "candidate-score-subkernel-only",
        "device": device,
        "device_name": device_name,
        "tasks": tasks,
        "ants": ants,
        "candidates": candidates,
        "padded_candidates": tile_candidates,
        "values": tasks * ants * tile_candidates,
        "repeats": repeats,
        "raw_cuda_ms": raw_ms,
        "raw_cuda_median_ms": raw_median,
        "cutile_ms": cutile_ms,
        "cutile_median_ms": cutile_median,
        "cutile_over_raw_speedup": raw_median / max(cutile_median, 1.0e-12),
        "outputs_bitwise_equal": True,
    }
