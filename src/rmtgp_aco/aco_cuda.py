"""FP32 搜索、FP64 计分的融合 CUDA ACO 后端。

CUDA 只负责组合搜索。最优 tour 回到主机后，fitness 一律用
``ProblemBatch.distances`` 的原始 float64 矩阵重算，避免设备精度差异直接
进入 GP 选择。模块按需导入 CuPy，因此纯 CPU 环境仍可安装和运行本项目。
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .aco_numba import (
    _PHEROMONE_VECTOR_TERMINAL_MASK,
    _TRANSITION_VECTOR_TERMINAL_MASK,
    _instance_key,
    _pack_programs,
    _semantic_representatives,
)
from .config import (
    ACOConfig,
    ACOVariant,
    GPUMode,
    PheromoneIntegration,
    RuntimeConfig,
    TransitionIntegration,
)
from .model import (
    PopulationQualityResult,
    ProblemBatch,
    RunDiagnostics,
    RunResult,
)
from .program import TensorProgram

_CUDA_SOURCE = Path(__file__).with_name("cuda") / "aco_fused.cu"
_MODULES: dict[tuple[int, str], Any] = {}
_MODULE_LOCK = threading.Lock()
_RESIDENT_LOCK = threading.Lock()
_RESIDENT: OrderedDict[tuple[int, tuple[object, ...]], _ResidentProblem] = (
    OrderedDict()
)


@dataclass(slots=True)
class _ResidentProblem:
    """一个设备上的只读问题数据。"""

    distances: Any
    heuristic: Any
    log_heuristic: Any
    nearest: Any
    full_nn_rank: Any
    node_log_eta_mean: Any
    instance_keys: Any
    nbytes: int
    transfer_seconds: float


@dataclass(slots=True)
class _DeviceResult:
    flat_indices: np.ndarray
    best_tours: np.ndarray
    gpu_best_lengths: np.ndarray
    best_iterations: np.ndarray
    anytime: np.ndarray | None
    diagnostics: np.ndarray
    kernel_seconds: float
    compile_seconds: float
    h2d_seconds: float
    d2h_seconds: float
    chunks: int
    block_threads: int
    device_name: str


def cuda_available() -> bool:
    """仅做轻量探测，不在 import 阶段建立 CUDA context。"""

    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def cuda_device_count() -> int:
    """返回可见 CUDA 设备数；驱动不可用时返回 0。"""

    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def clear_cuda_problem_cache() -> None:
    """释放驻留问题数据；主要供 benchmark 隔离 cold/warm 路径。"""

    with _RESIDENT_LOCK:
        _RESIDENT.clear()
    try:
        import cupy as cp

        for device in range(int(cp.cuda.runtime.getDeviceCount())):
            with cp.cuda.Device(device):
                cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        return


def clear_cuda_kernel_cache() -> None:
    """释放进程内 RawKernel 对象；磁盘 source-hash cache 保持不变。"""

    with _MODULE_LOCK:
        _MODULES.clear()


def _problem_key(
    problem: ProblemBatch,
    config: ACOConfig,
) -> tuple[object, ...]:
    return (
        problem.coordinate_hashes,
        problem.instance_ids,
        problem.n,
        int(problem.nn_indices.shape[-1]),
        str(problem.distances.dtype),
        float(config.epsilon_distance),
        float(config.epsilon_numeric),
    )


def _pinned_copy(cp: Any, value: np.ndarray, stream: Any) -> tuple[Any, Any]:
    """通过 page-locked staging buffer 发起异步 H2D copy。"""

    source = np.ascontiguousarray(value)
    pinned = cp.cuda.alloc_pinned_memory(source.nbytes)
    staging = np.frombuffer(
        pinned,
        dtype=source.dtype,
        count=source.size,
    ).reshape(source.shape)
    np.copyto(staging, source)
    target = cp.empty(source.shape, dtype=source.dtype)
    target.set(staging, stream=stream)
    return target, pinned


def _resident_problem(
    problem: ProblemBatch,
    config: ACOConfig,
    device: int,
    memory_fraction: float,
) -> _ResidentProblem:
    """取得或建立当前 batch 的设备驻留表示。"""

    import cupy as cp

    key = (device, _problem_key(problem, config))
    with _RESIDENT_LOCK:
        cached = _RESIDENT.get(key)
        if cached is not None:
            _RESIDENT.move_to_end(key)
            cached.transfer_seconds = 0.0
            return cached

    if problem.distances.dtype != torch.float64 or problem.device.type != "cpu":
        raise ValueError(
            "CUDA fused 后端要求 CPU float64 ProblemBatch，"
            "以便返回 tour 后执行精确计分"
        )
    distances64 = np.ascontiguousarray(problem.distances.detach().numpy())
    distances = distances64.astype(np.float32)
    heuristic = np.ascontiguousarray(
        problem.heuristic.detach().numpy().astype(np.float32)
    )
    log_heuristic = np.log(
        np.maximum(heuristic, np.float32(config.epsilon_numeric))
    ).astype(np.float32)
    nearest = np.ascontiguousarray(
        problem.nn_indices.detach().numpy().astype(np.uint16)
    )
    ranks = np.ascontiguousarray(
        problem.full_nn_rank.detach().numpy().astype(np.uint16)
    )
    selected_log_eta = np.take_along_axis(log_heuristic, nearest, axis=2)
    node_mean = np.ascontiguousarray(selected_log_eta.mean(axis=2))
    instance_keys = np.asarray(
        [_instance_key(instance_id) for instance_id in problem.instance_ids],
        dtype=np.uint64,
    )
    host_values = (
        distances,
        heuristic,
        log_heuristic,
        nearest,
        ranks,
        node_mean,
        instance_keys,
    )

    with cp.cuda.Device(device):
        stream = cp.cuda.Stream(non_blocking=True)
        started = perf_counter()
        pinned_buffers: list[Any] = []
        device_values: list[Any] = []
        with stream:
            for value in host_values:
                target, pinned = _pinned_copy(cp, value, stream)
                device_values.append(target)
                pinned_buffers.append(pinned)
        stream.synchronize()
        transfer_seconds = perf_counter() - started
        del pinned_buffers
        resident = _ResidentProblem(
            distances=device_values[0],
            heuristic=device_values[1],
            log_heuristic=device_values[2],
            nearest=device_values[3],
            full_nn_rank=device_values[4],
            node_log_eta_mean=device_values[5],
            instance_keys=device_values[6],
            nbytes=sum(value.nbytes for value in device_values),
            transfer_seconds=transfer_seconds,
        )

        total_memory = int(cp.cuda.runtime.memGetInfo()[1])
        cache_limit = int(total_memory * memory_fraction)
        with _RESIDENT_LOCK:
            resident_bytes = sum(
                item.nbytes
                for (cached_device, _), item in _RESIDENT.items()
                if cached_device == device
            )
            while resident_bytes + resident.nbytes > cache_limit:
                victim_key = next(
                    (
                        cached_key
                        for cached_key in _RESIDENT
                        if cached_key[0] == device
                    ),
                    None,
                )
                if victim_key is None:
                    break
                victim = _RESIDENT.pop(victim_key)
                resident_bytes -= victim.nbytes
            _RESIDENT[key] = resident
    return resident


def _load_kernel(device: int) -> tuple[Any, float]:
    """用 NVRTC 编译 source-hash 缓存的融合 kernel。

    当前节点的 GCC 16 超出 CUDA 12.6 的 NVCC host-compiler 支持范围；
    NVRTC 直接生成 PTX，不依赖主机 C++ 编译器，同时仍使用 CuPy 的磁盘
    source cache。
    """

    import cupy as cp

    source = _CUDA_SOURCE.read_text(encoding="utf-8")
    digest = sha256(source.encode("utf-8")).hexdigest()[:16]
    key = (device, digest)
    with _MODULE_LOCK:
        cached = _MODULES.get(key)
        if cached is not None:
            return cached, 0.0
        with cp.cuda.Device(device):
            started = perf_counter()
            module = cp.RawModule(
                code=source,
                options=("--std=c++14",),
                backend="nvrtc",
            )
            kernel = module.get_function("fused_aco")
            compile_seconds = perf_counter() - started
            _MODULES[key] = kernel
            return kernel, compile_seconds


def _mix64_python(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _counter_uniform_fp32(
    seed: int,
    instance_key: int,
    iteration: int,
    ant: int,
    step: int,
    stream_kind: int,
) -> float:
    mask = (1 << 64) - 1
    value = (seed ^ instance_key) & mask
    value ^= ((iteration + 1) * 0xD2B74407B1CE6E93) & mask
    value ^= ((ant + 1) * 0xCA5A826395121157) & mask
    value ^= ((step + 1) * 0x9E3779B185EBCA87) & mask
    value ^= ((stream_kind + 1) * 0x94D049BB133111EB) & mask
    return float(np.float32((_mix64_python(value) >> 40) / 16777216.0))


def _initial_pheromone_parameters(
    problem: ProblemBatch,
    config: ACOConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 FP64 几何上计算 NN 初始化，再显式量化为 FP32。"""

    distances = np.ascontiguousarray(problem.distances.detach().numpy())
    batch, n, _ = distances.shape
    tau0 = np.empty(batch, dtype=np.float32)
    tau_min = np.empty(batch, dtype=np.float32)
    tau_max = np.empty(batch, dtype=np.float32)
    seed64 = int(seed) % (2**64)
    for batch_index, instance_id in enumerate(problem.instance_ids):
        key = int(_instance_key(instance_id))
        start = min(
            int(
                _counter_uniform_fp32(seed64, key, 0, 0, 0, 0)
                * n
            ),
            n - 1,
        )
        visited = np.zeros(n, dtype=np.bool_)
        visited[start] = True
        current = start
        nn_length = 0.0
        for _ in range(1, n):
            row = np.where(visited, np.inf, distances[batch_index, current])
            chosen = int(np.argmin(row))
            nn_length += float(distances[batch_index, current, chosen])
            visited[chosen] = True
            current = chosen
        nn_length += float(distances[batch_index, current, start])
        if config.variant is ACOVariant.AS:
            tau0[batch_index] = 1.0 / (config.rho * nn_length)
            tau_min[batch_index] = 0.0
            tau_max[batch_index] = np.inf
        elif config.variant is ACOVariant.ACS:
            tau0[batch_index] = 1.0 / (n * nn_length)
            tau_min[batch_index] = 0.0
            tau_max[batch_index] = np.inf
        else:
            maximum = 1.0 / (config.rho * nn_length)
            tau0[batch_index] = maximum
            tau_max[batch_index] = maximum
            tau_min[batch_index] = maximum / (2.0 * n)
    return tau0, tau_min, tau_max


def _exact_tour_lengths(
    problem: ProblemBatch,
    tours: torch.Tensor,
) -> torch.Tensor:
    """用原始 FP64 distance matrix 精确重算 `[P,B,n+1]` tours。"""

    if tours.ndim != 3:
        raise ValueError("tours 必须具有 [P,B,n+1] shape")
    distance = np.ascontiguousarray(problem.distances.detach().cpu().numpy())
    route = np.ascontiguousarray(tours.detach().cpu().numpy())
    population, batch, n_plus_one = route.shape
    if batch != problem.batch_size or n_plus_one != problem.n + 1:
        raise ValueError("tour shape 与 ProblemBatch 不一致")
    flat = route.reshape(population * batch, n_plus_one)
    if (
        np.any(flat < 0)
        or np.any(flat >= problem.n)
        or not np.array_equal(flat[:, 0], flat[:, -1])
    ):
        raise RuntimeError("CUDA 返回了越界或未闭合 tour")
    expected = np.arange(problem.n, dtype=flat.dtype)[None, :]
    if not np.array_equal(
        np.sort(flat[:, :-1], axis=1),
        np.broadcast_to(expected, (flat.shape[0], problem.n)),
    ):
        raise RuntimeError("CUDA 返回的 tour 不是 Hamiltonian cycle")
    lengths = np.empty((population, batch), dtype=np.float64)
    for batch_index in range(batch):
        current = route[:, batch_index, :-1]
        following = route[:, batch_index, 1:]
        lengths[:, batch_index] = distance[
            batch_index,
            current,
            following,
        ].sum(axis=1, dtype=np.float64)
    return torch.from_numpy(lengths)


def _resolved_devices(runtime: RuntimeConfig) -> tuple[int, ...]:
    import cupy as cp

    available = int(cp.cuda.runtime.getDeviceCount())
    requested = tuple(runtime.gpu_devices)
    if any(device >= available for device in requested):
        raise ValueError(
            f"请求 GPU {requested}，但当前仅发现 {available} 个 CUDA 设备"
        )
    if runtime.gpu_mode is GPUMode.SINGLE:
        return requested[:1]
    if runtime.gpu_mode is GPUMode.DUAL:
        if len(requested) < 2:
            raise ValueError("gpu_mode=dual 至少需要两个 gpu_devices")
        return requested[:2]
    if runtime.gpu_mode is GPUMode.CAMPAIGN:
        explicit = os.environ.get("RMTGP_ACO_GPU_DEVICE")
        if explicit is not None:
            selected = int(explicit)
            if selected not in requested:
                raise ValueError(
                    "RMTGP_ACO_GPU_DEVICE 不在 runtime.gpu_devices 中"
                )
            return (selected,)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return (requested[local_rank % len(requested)],)
    if runtime.gpu_mode is GPUMode.AUTO:
        return requested[:2] if len(requested) >= 2 else requested[:1]
    raise ValueError("CUDA fused 后端不支持 gpu_mode=cpu")


def _active_and_representative_programs(
    programs: list[tuple[TensorProgram | None, TensorProgram | None]],
    config: ACOConfig,
) -> tuple[
    Any,
    Any,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    transition = _pack_programs(
        [pair[0] for pair in programs],
        role="transition",
    )
    pheromone = _pack_programs(
        [pair[1] for pair in programs],
        role="pheromone",
    )
    transition_mode = (
        1
        if config.transition_integration is TransitionIntegration.REPLACEMENT
        else 0
    )
    pheromone_mode = {
        PheromoneIntegration.BUDGET_RESIDUAL: 0,
        PheromoneIntegration.UNNORMALIZED_MULTIPLICATIVE: 1,
        PheromoneIntegration.ADDITIVE: 2,
        PheromoneIntegration.REPLACEMENT: 3,
    }[config.pheromone_integration]
    transition_active = transition.active.copy()
    if transition_mode == 0:
        transition_active[
            (transition.exact_zero.astype(bool))
            | (config.gamma_transition == 0.0)
            | (
                (transition.required_masks & _TRANSITION_VECTOR_TERMINAL_MASK)
                == 0
            )
        ] = 0
    pheromone_active = pheromone.active.copy()
    if pheromone_mode != 3:
        pheromone_active[
            (pheromone.exact_zero.astype(bool))
            | (config.gamma_pheromone == 0.0)
            | (
                (pheromone_mode == 0)
                & (
                    (
                        pheromone.required_masks
                        & _PHEROMONE_VECTOR_TERMINAL_MASK
                    )
                    == 0
                )
            )
        ] = 0
    representatives, inverse = _semantic_representatives(
        transition,
        pheromone,
        transition_active,
        pheromone_active,
    )
    return (
        transition,
        pheromone,
        transition_active,
        pheromone_active,
        representatives,
        inverse,
    )


def _task_bytes(n: int, ants: int, iterations: int, record_anytime: bool) -> int:
    words = (n + 63) // 64
    return (
        4 * n * n  # pheromone
        + n * n  # edge frequency
        + 2 * ants * (n + 1)  # current tours
        + 8 * ants * words  # visited bitsets
        + 4 * ants * n  # per-source deposits
        + 2 * (n + 1)  # restart tour
        + 2 * (n + 1)  # output tour
        + 4  # output length
        + 4  # output iteration
        + 8 * 4  # diagnostics
        + (4 * iterations if record_anytime else 0)
    )


def _cost_balanced_shards(
    *,
    representative_count: int,
    batch_size: int,
    device_count: int,
    n: int,
    candidate_size: int,
    ants: int,
    iterations: int,
    variant: ACOVariant,
    transition_lengths: np.ndarray,
    pheromone_lengths: np.ndarray,
    transition_active: np.ndarray,
    pheromone_active: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """按估算指令量做确定性 LPT 分片，而非按 program 编号对半切。"""

    flat = np.arange(representative_count * batch_size, dtype=np.int64)
    if device_count == 1:
        return (flat,)
    source_count = ants if variant is ACOVariant.AS else 1
    program_cost = np.empty(representative_count, dtype=np.float64)
    for program in range(representative_count):
        transition_nodes = (
            int(transition_lengths[program])
            if transition_active[program]
            else 0
        )
        pheromone_nodes = (
            int(pheromone_lengths[program])
            if pheromone_active[program]
            else 0
        )
        construction = (
            ants
            * n
            * candidate_size
            * (4.0 + transition_nodes)
        )
        update = n * n + source_count * n * (2.0 + pheromone_nodes)
        program_cost[program] = iterations * (construction + update)

    order = sorted(
        flat.tolist(),
        key=lambda task: (
            -program_cost[task // batch_size],
            task,
        ),
    )
    shards: list[list[int]] = [[] for _ in range(device_count)]
    loads = np.zeros(device_count, dtype=np.float64)
    for task in order:
        target = int(np.argmin(loads))
        shards[target].append(task)
        loads[target] += program_cost[task // batch_size]
    return tuple(
        np.asarray(sorted(shard), dtype=np.int64)
        for shard in shards
        if shard
    )


def _run_device(
    *,
    device: int,
    flat_indices: np.ndarray,
    problem: ProblemBatch,
    config: ACOConfig,
    runtime: RuntimeConfig,
    seed: int,
    transition: Any,
    pheromone_programs: Any,
    transition_active: np.ndarray,
    pheromone_active: np.ndarray,
    representatives: np.ndarray,
    initial_tau: tuple[np.ndarray, np.ndarray, np.ndarray],
    record_anytime: bool,
) -> _DeviceResult:
    import cupy as cp

    with cp.cuda.Device(device):
        resident = _resident_problem(
            problem,
            config,
            device,
            runtime.gpu_memory_fraction,
        )
        kernel, compile_seconds = _load_kernel(device)
        h2d_started = perf_counter()
        tr_ops = cp.asarray(
            np.ascontiguousarray(transition.opcodes[representatives])
        )
        tr_fargs = cp.asarray(
            np.ascontiguousarray(
                transition.float_arguments[representatives].astype(np.float32)
            )
        )
        tr_iargs = cp.asarray(
            np.ascontiguousarray(transition.integer_arguments[representatives])
        )
        tr_lengths = cp.asarray(
            np.ascontiguousarray(transition.lengths[representatives])
        )
        tr_masks = cp.asarray(
            np.ascontiguousarray(transition.required_masks[representatives])
        )
        tr_is_active = cp.asarray(
            np.ascontiguousarray(transition_active[representatives])
        )
        ph_ops = cp.asarray(
            np.ascontiguousarray(pheromone_programs.opcodes[representatives])
        )
        ph_fargs = cp.asarray(
            np.ascontiguousarray(
                pheromone_programs.float_arguments[representatives].astype(
                    np.float32
                )
            )
        )
        ph_iargs = cp.asarray(
            np.ascontiguousarray(
                pheromone_programs.integer_arguments[representatives]
            )
        )
        ph_lengths = cp.asarray(
            np.ascontiguousarray(pheromone_programs.lengths[representatives])
        )
        ph_masks = cp.asarray(
            np.ascontiguousarray(
                pheromone_programs.required_masks[representatives]
            )
        )
        ph_is_active = cp.asarray(
            np.ascontiguousarray(pheromone_active[representatives])
        )
        tau0 = cp.asarray(initial_tau[0])
        tau_min = cp.asarray(initial_tau[1])
        tau_max = cp.asarray(initial_tau[2])
        cp.cuda.get_current_stream().synchronize()
        program_h2d = perf_counter() - h2d_started

        n = problem.n
        ants = config.resolve_ants(n)
        batch = problem.batch_size
        block_threads = runtime.gpu_block_threads or 32
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()
        reserve = int(total_memory * (1.0 - runtime.gpu_memory_fraction))
        usable = max(0, int(free_memory) - reserve)
        bytes_per_task = _task_bytes(
            n,
            ants,
            config.iterations,
            record_anytime,
        )
        max_tasks = usable // max(bytes_per_task, 1)
        if runtime.gpu_task_chunk_size:
            max_tasks = min(max_tasks, runtime.gpu_task_chunk_size)
        if max_tasks < 1:
            raise MemoryError(
                f"GPU {device} 无法容纳单个 n={n} ACO task；"
                f"估算需 {bytes_per_task / 2**20:.1f} MiB"
            )

        tour_parts: list[np.ndarray] = []
        length_parts: list[np.ndarray] = []
        iteration_parts: list[np.ndarray] = []
        anytime_parts: list[np.ndarray] = []
        diagnostic_parts: list[np.ndarray] = []
        kernel_seconds = 0.0
        d2h_seconds = 0.0
        chunk_count = 0
        for start in range(0, flat_indices.size, max_tasks):
            selected = flat_indices[start : start + max_tasks]
            count = int(selected.size)
            task_program = cp.asarray(
                np.ascontiguousarray((selected // batch).astype(np.int32))
            )
            task_instance = cp.asarray(
                np.ascontiguousarray((selected % batch).astype(np.int32))
            )
            words = (n + 63) // 64
            pheromone_workspace = cp.empty((count, n, n), dtype=cp.float32)
            tour_workspace = cp.empty(
                (count, ants, n + 1),
                dtype=cp.uint16,
            )
            visited_workspace = cp.empty(
                (count, ants, words),
                dtype=cp.uint64,
            )
            deposit_workspace = cp.empty(
                (count, ants, n),
                dtype=cp.float32,
            )
            edge_frequency = cp.empty((count, n, n), dtype=cp.uint8)
            restart_tour = cp.empty((count, n + 1), dtype=cp.uint16)
            best_tours = cp.empty((count, n + 1), dtype=cp.uint16)
            best_lengths = cp.empty(count, dtype=cp.float32)
            best_iterations = cp.empty(count, dtype=cp.int32)
            anytime = (
                cp.empty((count, config.iterations), dtype=cp.float32)
                if record_anytime
                else cp.empty(1, dtype=cp.float32)
            )
            diagnostics = cp.empty((count, 4), dtype=cp.uint64)

            transition_mode = (
                1
                if config.transition_integration
                is TransitionIntegration.REPLACEMENT
                else 0
            )
            pheromone_mode = {
                PheromoneIntegration.BUDGET_RESIDUAL: 0,
                PheromoneIntegration.UNNORMALIZED_MULTIPLICATIVE: 1,
                PheromoneIntegration.ADDITIVE: 2,
                PheromoneIntegration.REPLACEMENT: 3,
            }[config.pheromone_integration]
            variant = {
                ACOVariant.AS: 0,
                ACOVariant.ACS: 1,
                ACOVariant.MMAS: 2,
            }[config.variant]
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()
            kernel(
                (count,),
                (block_threads,),
                (
                    resident.distances,
                    resident.heuristic,
                    resident.log_heuristic,
                    resident.nearest,
                    resident.full_nn_rank,
                    resident.node_log_eta_mean,
                    tau0,
                    tau_min,
                    tau_max,
                    tr_ops,
                    tr_fargs,
                    tr_iargs,
                    tr_lengths,
                    tr_masks,
                    tr_is_active,
                    np.int32(tr_ops.shape[1]),
                    ph_ops,
                    ph_fargs,
                    ph_iargs,
                    ph_lengths,
                    ph_masks,
                    ph_is_active,
                    np.int32(ph_ops.shape[1]),
                    task_program,
                    task_instance,
                    np.int32(count),
                    np.int32(n),
                    np.int32(problem.nn_indices.shape[-1]),
                    np.int32(ants),
                    np.int32(config.iterations),
                    np.int32(variant),
                    np.float32(config.alpha),
                    np.float32(config.beta),
                    np.float32(config.rho),
                    np.float32(config.q0),
                    np.float32(config.xi),
                    np.float32(config.gamma_transition),
                    np.float32(config.gamma_pheromone),
                    np.int32(transition_mode),
                    np.int32(pheromone_mode),
                    np.float32(config.epsilon_numeric),
                    np.int32(config.mmas_update_period),
                    np.float32(config.mmas_p_best),
                    np.int32(config.mmas_branch_check_period),
                    np.float32(config.mmas_branch_lambda),
                    np.float32(config.mmas_branch_threshold),
                    np.int32(config.mmas_restart_stagnation),
                    np.uint64(int(seed) % (2**64)),
                    resident.instance_keys,
                    pheromone_workspace,
                    tour_workspace,
                    visited_workspace,
                    deposit_workspace,
                    edge_frequency,
                    restart_tour,
                    best_tours,
                    best_lengths,
                    best_iterations,
                    anytime,
                    np.int32(record_anytime),
                    diagnostics,
                ),
            )
            end_event.record()
            end_event.synchronize()
            kernel_seconds += float(cp.cuda.get_elapsed_time(
                start_event,
                end_event,
            )) / 1000.0

            d2h_started = perf_counter()
            tour_parts.append(cp.asnumpy(best_tours))
            length_parts.append(cp.asnumpy(best_lengths))
            iteration_parts.append(cp.asnumpy(best_iterations))
            diagnostic_parts.append(cp.asnumpy(diagnostics))
            if record_anytime:
                anytime_parts.append(cp.asnumpy(anytime))
            d2h_seconds += perf_counter() - d2h_started
            chunk_count += 1

        properties = cp.cuda.runtime.getDeviceProperties(device)
        raw_name = properties["name"]
        device_name = (
            raw_name.decode("utf-8")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        return _DeviceResult(
            flat_indices=flat_indices,
            best_tours=np.concatenate(tour_parts),
            gpu_best_lengths=np.concatenate(length_parts),
            best_iterations=np.concatenate(iteration_parts),
            anytime=(
                np.concatenate(anytime_parts)
                if record_anytime
                else None
            ),
            diagnostics=np.concatenate(diagnostic_parts),
            kernel_seconds=kernel_seconds,
            compile_seconds=compile_seconds,
            h2d_seconds=resident.transfer_seconds + program_h2d,
            d2h_seconds=d2h_seconds,
            chunks=chunk_count,
            block_threads=block_threads,
            device_name=device_name,
        )


def _solve_population_impl(
    problem: ProblemBatch,
    config: ACOConfig,
    programs: list[tuple[TensorProgram | None, TensorProgram | None]],
    *,
    seed: int,
    runtime: RuntimeConfig,
    record_anytime: bool,
) -> tuple[PopulationQualityResult, torch.Tensor | None]:
    if not programs:
        raise ValueError("program population 不得为空")
    if config.resolve_ants(problem.n) > 32:
        raise ValueError("CUDA fused v1 最多支持 32 只蚂蚁")
    if problem.n > np.iinfo(np.uint16).max - 1:
        raise ValueError("CUDA fused v1 的城市数必须小于 65535")
    if config.variant is ACOVariant.ACS and not config.acs_synchronous:
        raise ValueError("CUDA fused v1 只支持同步 ACS local update")
    (
        transition,
        pheromone_programs,
        transition_active,
        pheromone_active,
        representatives,
        inverse,
    ) = _active_and_representative_programs(programs, config)
    if max(transition.stack_size, pheromone_programs.stack_size) > 32:
        raise ValueError("CUDA fused v1 的 GP postfix stack 深度不得超过 32")

    devices = _resolved_devices(runtime)
    representative_count = int(representatives.size)
    task_count = representative_count * problem.batch_size
    representative_transition_active = transition_active[representatives]
    representative_pheromone_active = pheromone_active[representatives]
    shards = _cost_balanced_shards(
        representative_count=representative_count,
        batch_size=problem.batch_size,
        device_count=len(devices),
        n=problem.n,
        candidate_size=config.resolve_candidate_size(problem.n),
        ants=config.resolve_ants(problem.n),
        iterations=config.iterations,
        variant=config.variant,
        transition_lengths=transition.lengths[representatives],
        pheromone_lengths=pheromone_programs.lengths[representatives],
        transition_active=representative_transition_active,
        pheromone_active=representative_pheromone_active,
    )
    used_devices = devices[: len(shards)]
    initial_tau = _initial_pheromone_parameters(problem, config, seed)
    started = perf_counter()
    if len(shards) == 1:
        device_results = [
            _run_device(
                device=devices[0],
                flat_indices=shards[0],
                problem=problem,
                config=config,
                runtime=runtime,
                seed=seed,
                transition=transition,
                pheromone_programs=pheromone_programs,
                transition_active=transition_active,
                pheromone_active=pheromone_active,
                representatives=representatives,
                initial_tau=initial_tau,
                record_anytime=record_anytime,
            )
        ]
    else:
        with ThreadPoolExecutor(max_workers=len(shards)) as executor:
            futures = [
                executor.submit(
                    _run_device,
                    device=device,
                    flat_indices=shard,
                    problem=problem,
                    config=config,
                    runtime=runtime,
                    seed=seed,
                    transition=transition,
                    pheromone_programs=pheromone_programs,
                    transition_active=transition_active,
                    pheromone_active=pheromone_active,
                    representatives=representatives,
                    initial_tau=initial_tau,
                    record_anytime=record_anytime,
                )
                for device, shard in zip(used_devices, shards, strict=True)
            ]
            device_results = [future.result() for future in futures]

    best_tour_flat = np.empty(
        (task_count, problem.n + 1),
        dtype=np.uint16,
    )
    gpu_length_flat = np.empty(task_count, dtype=np.float32)
    best_iteration_flat = np.empty(task_count, dtype=np.int32)
    diagnostics_flat = np.empty((task_count, 4), dtype=np.uint64)
    anytime_flat = (
        np.empty((task_count, config.iterations), dtype=np.float32)
        if record_anytime
        else None
    )
    for result in device_results:
        best_tour_flat[result.flat_indices] = result.best_tours
        gpu_length_flat[result.flat_indices] = result.gpu_best_lengths
        best_iteration_flat[result.flat_indices] = result.best_iterations
        diagnostics_flat[result.flat_indices] = result.diagnostics
        if record_anytime:
            assert anytime_flat is not None and result.anytime is not None
            anytime_flat[result.flat_indices] = result.anytime

    representative_tours = torch.from_numpy(
        best_tour_flat.reshape(
            representative_count,
            problem.batch_size,
            problem.n + 1,
        ).astype(np.int64)
    )
    exact_started = perf_counter()
    representative_lengths = _exact_tour_lengths(
        problem,
        representative_tours,
    )
    exact_seconds = perf_counter() - exact_started
    representative_iterations = torch.from_numpy(
        best_iteration_flat.reshape(
            representative_count,
            problem.batch_size,
        ).astype(np.int64)
    )
    representative_diagnostics = torch.from_numpy(
        diagnostics_flat.reshape(
            representative_count,
            problem.batch_size,
            4,
        ).sum(axis=1, dtype=np.uint64).astype(np.int64)
    )

    best_tours = representative_tours
    best_lengths = representative_lengths
    best_iterations = representative_iterations
    diagnostics = representative_diagnostics
    anytime_tensor: torch.Tensor | None = None
    if anytime_flat is not None:
        anytime_tensor = torch.from_numpy(
            anytime_flat.reshape(
                representative_count,
                problem.batch_size,
                config.iterations,
            ).astype(np.float64)
        )
    if representatives.size != len(programs):
        best_tours = best_tours[inverse]
        best_lengths = best_lengths[inverse]
        best_iterations = best_iterations[inverse]
        diagnostics = diagnostics[inverse]
        if anytime_tensor is not None:
            anytime_tensor = anytime_tensor[inverse]

    elapsed = perf_counter() - started
    kernel_values = [item.kernel_seconds for item in device_results]
    metrics: dict[str, float | int | str] = {
        "devices": ",".join(str(device) for device in used_devices),
        "device_names": " | ".join(
            item.device_name for item in device_results
        ),
        "device_count": len(used_devices),
        "kernel_seconds_critical": max(kernel_values),
        "kernel_seconds_sum": sum(kernel_values),
        "compile_seconds_sum": sum(
            item.compile_seconds for item in device_results
        ),
        "h2d_seconds_sum": sum(item.h2d_seconds for item in device_results),
        "d2h_seconds_sum": sum(item.d2h_seconds for item in device_results),
        "exact_fp64_scoring_seconds": exact_seconds,
        "chunks": sum(item.chunks for item in device_results),
        "block_threads": device_results[0].block_threads,
        "gpu_fp32_length_checksum": float(gpu_length_flat.sum(dtype=np.float64)),
    }
    quality = PopulationQualityResult(
        best_tour=best_tours,
        best_length=best_lengths,
        best_iteration=best_iterations,
        diagnostics=diagnostics,
        wall_time_sec=elapsed,
        constructed_tours=(
            representative_count
            * problem.batch_size
            * config.resolve_ants(problem.n)
            * config.iterations
        ),
        backend_metrics=metrics,
    )
    return quality, anytime_tensor


def solve_population_cuda(
    problem: ProblemBatch,
    config: ACOConfig,
    programs: list[tuple[TensorProgram | None, TensorProgram | None]],
    *,
    seed: int,
    runtime: RuntimeConfig,
) -> PopulationQualityResult:
    """融合评估 population，并返回 CPU FP64 精确长度。"""

    result, _ = _solve_population_impl(
        problem,
        config,
        programs,
        seed=seed,
        runtime=runtime,
        record_anytime=False,
    )
    return result


def solve_cuda(
    problem: ProblemBatch,
    config: ACOConfig,
    *,
    transition_program: TensorProgram | None = None,
    pheromone_program: TensorProgram | None = None,
    seed: int = 0,
    runtime: RuntimeConfig,
) -> RunResult:
    """运行一个或多个实例，并保留 baseline 所需 anytime curve。"""

    quality, anytime = _solve_population_impl(
        problem,
        config,
        [(transition_program, pheromone_program)],
        seed=seed,
        runtime=runtime,
        record_anytime=True,
    )
    assert anytime is not None
    diagnostic = quality.diagnostics[0]
    return RunResult(
        best_tour=quality.best_tour[0],
        best_length=quality.best_length[0],
        best_iteration=quality.best_iteration[0],
        anytime_best=anytime[0],
        wall_time_sec=quality.wall_time_sec,
        constructed_tours=quality.constructed_tours,
        diagnostics=RunDiagnostics(
            candidate_fallback_count=int(diagnostic[0].item()),
            uniform_fallback_count=int(diagnostic[1].item()),
            bound_clip_count=int(diagnostic[2].item()),
            mmas_restart_count=int(diagnostic[3].item()),
        ),
        backend_metrics=quality.backend_metrics,
    )
