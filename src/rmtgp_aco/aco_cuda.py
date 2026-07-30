"""FP32 搜索、FP64 计分的融合 CUDA ACO 后端。

CUDA 只负责组合搜索。最优 tour 回到主机后，fitness 一律用
``ProblemBatch.distances`` 的原始 float64 矩阵重算，避免设备精度差异直接
进入 GP 选择。模块按需导入 CuPy，因此纯 CPU 环境仍可安装和运行本项目。
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
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
    CudaPrecision,
    CudaProvider,
    CudaTaskOrder,
    ExecutionBackend,
    GPUMode,
    LocalSearch,
    PheromoneIntegration,
    RuntimeConfig,
    TransitionIntegration,
)
from .local_search import two_opt_first
from .model import (
    PopulationQualityResult,
    PopulationRunResult,
    ProblemBatch,
    RunDiagnostics,
    RunResult,
)
from .program import TensorProgram

_CUDA_SOURCE = Path(__file__).with_name("cuda") / "aco_fused.cu"
_CUDA_V2_SOURCE = Path(__file__).with_name("cuda") / "aco_tiled_v2.cu"
_CUDA_LS_SOURCE = Path(__file__).with_name("cuda") / "aco_local_search.cu"
_MODULES: dict[tuple[int, str], Any] = {}
_V2_MODULES: dict[tuple[int, str], tuple[Any, ...]] = {}
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
    provider: str = "raw_cuda"
    precision: str = "fp32"
    candidate_lanes: int = 1
    register_cap: int = 0


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
        _V2_MODULES.clear()


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


def _v2_precision_macros(
    precision: CudaPrecision,
) -> tuple[int, int]:
    """返回静态表存储类型和 score 量化开关。"""

    if precision in {CudaPrecision.FP32, CudaPrecision.FP32_FAST}:
        return 0, 0
    if precision is CudaPrecision.FP16_MIXED:
        return 1, 0
    if precision is CudaPrecision.BF16_MIXED:
        return 2, 0
    if precision is CudaPrecision.FP16_SEARCH:
        return 1, 1
    if precision is CudaPrecision.FP64:
        raise NotImplementedError(
            "CUDA v2 的 fp64 profile 仅作为独立 oracle benchmark，"
            "不能伪装成 FP32 动态状态"
        )
    if precision in {CudaPrecision.FP8_E4M3, CudaPrecision.NVFP4}:
        raise NotImplementedError(
            f"{precision.value} 只在低精度 storage probe 中评估；"
            "未通过 probe 前不允许进入完整 ACO solver"
        )
    raise ValueError("正式运行必须把 cuda_precision=auto 解析为具体 profile")


def _generated_program_body(
    packed: Any,
    index: int,
) -> list[str]:
    """把一个 postfix program 翻译为无解释器分支的 CUDA 标量语句。"""

    stack: list[str] = []
    statements: list[str] = []
    length = int(packed.lengths[index])
    for instruction in range(length):
        opcode = int(packed.opcodes[index, instruction])
        name = f"value_{instruction}"
        if opcode == 0:
            value = np.float32(packed.float_arguments[index, instruction])
            bits = int(value.view(np.uint32))
            expression = f"__uint_as_float(0x{bits:08x}U)"
            stack.append(name)
        elif opcode == 1:
            terminal = int(packed.integer_arguments[index, instruction])
            expression = f"terminals[{terminal}]"
            stack.append(name)
        elif opcode in {9, 10}:
            if not stack:
                raise ValueError("非法 postfix GP：一元 primitive 缺少操作数")
            operand = stack.pop()
            expression = (
                f"sanitize(fabsf({operand}))"
                if opcode == 9
                else f"sanitize(-({operand}))"
            )
            stack.append(name)
        else:
            if len(stack) < 2:
                raise ValueError("非法 postfix GP：二元 primitive 缺少操作数")
            right = stack.pop()
            left = stack.pop()
            if opcode == 2:
                raw = f"({left}) + ({right})"
            elif opcode == 3:
                raw = f"({left}) - ({right})"
            elif opcode == 4:
                raw = f"({left}) * ({right})"
            elif opcode == 5:
                raw = (
                    f"({left}) * ({right}) / "
                    f"(({right}) * ({right}) + 1.0e-6f)"
                )
            elif opcode == 6:
                raw = (
                    f"fabsf({right}) > 1.0e-6f "
                    f"? ({left}) / ({right}) : 1.0f"
                )
            elif opcode == 7:
                raw = f"fminf({left}, {right})"
            elif opcode == 8:
                raw = f"fmaxf({left}, {right})"
            else:
                raise ValueError(f"CUDA generated GP 不支持 opcode={opcode}")
            expression = f"sanitize({raw})"
            stack.append(name)
        statements.append(f"        const float {name} = {expression};")
    if len(stack) != 1:
        if length == 0:
            return ["        return 0.0f;"]
        raise ValueError("非法 postfix GP：计算结束后 stack 大小不为 1")
    statements.append(f"        return sanitize({stack[0]});")
    return statements


def _generated_switch(
    function_name: str,
    packed: Any,
) -> str:
    """生成每个 semantic program 一个 case 的 CUDA device function。"""

    lines = [
        f"__device__ float {function_name}(",
        "    int program,",
        "    const float* terminals",
        ") {",
        "    switch (program) {",
    ]
    for index in range(int(packed.lengths.size)):
        lines.extend((f"    case {index}: {{", *_generated_program_body(packed, index)))
        lines.append("    }")
    lines.extend(
        (
            "    default:",
            "        return 0.0f;",
            "    }",
            "}",
        )
    )
    return "\n".join(lines)


def _generated_gp_source(
    transition: Any,
    pheromone: Any,
    representatives: np.ndarray,
) -> str:
    """生成与当前一代 semantic population 精确配对的 CUDA 源码。"""

    tr = type(transition)(
        opcodes=np.ascontiguousarray(transition.opcodes[representatives]),
        float_arguments=np.ascontiguousarray(
            transition.float_arguments[representatives]
        ),
        integer_arguments=np.ascontiguousarray(
            transition.integer_arguments[representatives]
        ),
        lengths=np.ascontiguousarray(transition.lengths[representatives]),
        required_masks=np.ascontiguousarray(
            transition.required_masks[representatives]
        ),
        stack_size=transition.stack_size,
        active=np.ascontiguousarray(transition.active[representatives]),
        exact_zero=np.ascontiguousarray(transition.exact_zero[representatives]),
    )
    ph = type(pheromone)(
        opcodes=np.ascontiguousarray(pheromone.opcodes[representatives]),
        float_arguments=np.ascontiguousarray(
            pheromone.float_arguments[representatives]
        ),
        integer_arguments=np.ascontiguousarray(
            pheromone.integer_arguments[representatives]
        ),
        lengths=np.ascontiguousarray(pheromone.lengths[representatives]),
        required_masks=np.ascontiguousarray(
            pheromone.required_masks[representatives]
        ),
        stack_size=pheromone.stack_size,
        active=np.ascontiguousarray(pheromone.active[representatives]),
        exact_zero=np.ascontiguousarray(pheromone.exact_zero[representatives]),
    )
    return "\n".join(
        (
            "namespace {",
            _generated_switch("evaluate_pheromone_generated", ph),
            "}  // namespace",
            "namespace rmtgp_v2 {",
            _generated_switch("evaluate_transition_generated", tr),
            "}  // namespace rmtgp_v2",
            "",
        )
    )


def _runtime_from_tuning_manifest(
    runtime: RuntimeConfig,
    devices: tuple[int, ...],
) -> tuple[RuntimeConfig, str]:
    """加载硬件绑定的调优选择，并拒绝在不匹配 GPU 上静默复用。"""

    if runtime.cuda_tuning_manifest is None:
        return runtime, ""
    import cupy as cp

    path = Path(runtime.cuda_tuning_manifest).resolve()
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("CUDA tuning manifest schema_version 必须为 1")
    if payload.get("backend") != ExecutionBackend.CUDA_TILED_V2.value:
        raise ValueError("CUDA tuning manifest 不是 cuda_tiled_v2")
    expected_name = str(payload["hardware"]["device_name"])
    expected_capability = str(payload["hardware"]["compute_capability"])
    for device in devices:
        properties = cp.cuda.runtime.getDeviceProperties(device)
        raw_name = properties["name"]
        name = (
            raw_name.decode("utf-8")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        capability = f"{int(properties['major'])}.{int(properties['minor'])}"
        if name != expected_name or capability != expected_capability:
            raise ValueError(
                "CUDA tuning manifest 与当前 GPU 不匹配："
                f"manifest={expected_name}/sm{expected_capability}，"
                f"device{device}={name}/sm{capability}"
            )
    selected = payload["selected"]
    resolved = replace(
        runtime,
        cuda_provider=CudaProvider(selected["provider"]),
        cuda_precision=CudaPrecision(selected["precision"]),
        cuda_candidate_lanes=int(selected["candidate_lanes"]),
        cuda_register_cap=int(selected["register_cap"]),
        cuda_task_order=CudaTaskOrder(selected["task_order"]),
        cuda_generated_gp=bool(selected["generated_gp"]),
        cuda_graph_replay=bool(selected["graph_replay"]),
    )
    if resolved.cuda_graph_replay:
        raise NotImplementedError(
            "当前每代 GP 源码和指针均变化，CUDA graph capture 不能安全复用"
        )
    return resolved, sha256(payload_bytes).hexdigest()


def _load_v2_kernels(
    *,
    device: int,
    variant: ACOVariant,
    candidate_size: int,
    candidate_lanes: int,
    stack_depth: int,
    precision: CudaPrecision,
    register_cap: int,
    generated_gp_source: str = "",
) -> tuple[tuple[Any, ...], float]:
    """编译按 variant/shape/precision 专门化的 CUDA v2 kernel。"""

    import cupy as cp

    if candidate_lanes not in {1, 4, 8, 16, 32}:
        raise ValueError("CUDA v2 candidate lanes 必须为 1/4/8/16/32")
    if stack_depth < 1 or stack_depth > 32:
        raise ValueError("CUDA v2 GP postfix stack 深度必须位于 1--32")
    candidate_pad = 1 << max(0, candidate_size - 1).bit_length()
    if candidate_pad > 64:
        raise ValueError("CUDA v2 当前要求 candidate_size <= 64")
    static_precision, score_quantize = _v2_precision_macros(precision)
    variant_code = {
        ACOVariant.AS: 0,
        ACOVariant.ACS: 1,
        ACOVariant.MMAS: 2,
    }[variant]
    prefix = "\n".join(
        (
            f"#define RMTGP_CANDIDATE_LANES {candidate_lanes}",
            f"#define RMTGP_CANDIDATE_PAD {candidate_pad}",
            f"#define RMTGP_MAX_STACK_V2 {stack_depth}",
            f"#define RMTGP_VARIANT {variant_code}",
            f"#define RMTGP_STATIC_PRECISION {static_precision}",
            f"#define RMTGP_SCORE_QUANTIZE_FP16 {score_quantize}",
            f"#define RMTGP_GENERATED_GP {int(bool(generated_gp_source))}",
            "",
        )
    )
    source = (
        prefix
        + _CUDA_SOURCE.read_text(encoding="utf-8")
        + "\n"
        + generated_gp_source
        + "\n"
        + _CUDA_V2_SOURCE.read_text(encoding="utf-8")
        + "\n"
        + _CUDA_LS_SOURCE.read_text(encoding="utf-8")
    )
    properties = cp.cuda.runtime.getDeviceProperties(device)
    major = int(properties["major"])
    minor = int(properties["minor"])
    options = [
        "--std=c++17",
        f"--gpu-architecture=compute_{major}{minor}",
    ]
    if register_cap:
        options.append(f"--maxrregcount={register_cap}")
    if precision is CudaPrecision.FP32_FAST:
        options.append("--use_fast_math")
    digest_payload = "\0".join((source, *options))
    digest = sha256(digest_payload.encode("utf-8")).hexdigest()[:20]
    key = (device, digest)
    with _MODULE_LOCK:
        cached = _V2_MODULES.get(key)
        if cached is not None:
            return cached, 0.0
        with cp.cuda.Device(device):
            started = perf_counter()
            module = cp.RawModule(
                code=source,
                options=tuple(options),
                backend="nvrtc",
            )
            kernels = (
                module.get_function("v2_init"),
                module.get_function("v2_construct"),
                module.get_function("v2_update"),
                module.get_function("v2_two_opt"),
                module.get_function("v2_three_opt"),
            )
            compile_seconds = perf_counter() - started
            _V2_MODULES[key] = kernels
            return kernels, compile_seconds


def _bfloat16_storage(cp: Any, values: Any) -> Any:
    """以 round-to-nearest-even 生成可由 kernel 直接解码的 BF16 bits。"""

    source = values.astype(cp.float32, copy=False)
    bits = source.view(cp.uint32)
    rounded = bits + cp.uint32(0x7FFF) + ((bits >> 16) & cp.uint32(1))
    return (rounded >> 16).astype(cp.uint16)


def _v2_static_problem_arrays(
    cp: Any,
    resident: _ResidentProblem,
    precision: CudaPrecision,
) -> tuple[Any, Any, Any]:
    """为 construction kernel 建立指定精度的只读表。"""

    if precision in {CudaPrecision.FP32, CudaPrecision.FP32_FAST}:
        return resident.distances, resident.heuristic, resident.log_heuristic
    if precision in {CudaPrecision.FP16_MIXED, CudaPrecision.FP16_SEARCH}:
        return (
            resident.distances.astype(cp.float16),
            resident.heuristic.astype(cp.float16),
            resident.log_heuristic.astype(cp.float16),
        )
    if precision is CudaPrecision.BF16_MIXED:
        return (
            _bfloat16_storage(cp, resident.distances),
            _bfloat16_storage(cp, resident.heuristic),
            _bfloat16_storage(cp, resident.log_heuristic),
        )
    _v2_precision_macros(precision)
    raise AssertionError("unreachable")


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
    nearest = np.ascontiguousarray(problem.nn_indices.detach().numpy())
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
        nn_tour = np.empty(n + 1, dtype=np.int64)
        nn_tour[0] = start
        for phase in range(1, n):
            row = np.where(visited, np.inf, distances[batch_index, current])
            chosen = int(np.argmin(row))
            visited[chosen] = True
            current = chosen
            nn_tour[phase] = chosen
        nn_tour[n] = start
        if config.uses_local_search:
            nn_tour, _ = two_opt_first(
                nn_tour,
                distances[batch_index],
                nearest[batch_index],
                seed=seed64,
                instance_key=key,
                iteration=0,
                ant=0,
                candidate_size=config.resolve_local_search_candidate_size(n),
                use_dlb=config.local_search_dlb,
            )
        nn_length = float(
            distances[
                batch_index,
                nn_tour[:-1],
                nn_tour[1:],
            ].sum(dtype=np.float64)
        )
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


def _task_bytes_v2(
    n: int,
    ants: int,
    iterations: int,
    record_anytime: bool,
    local_search: LocalSearch = LocalSearch.NONE,
) -> int:
    """CUDA v2 staged state 的保守每 task 显存估算。"""

    base = (
        _task_bytes(n, ants, iterations, record_anytime)
        + 4 * ants  # incremental colony lengths
        + 4 * ants * n  # fallback/cached transition scores
        + 4 * 4  # best/restart length and tau bounds
        + 4 * 3  # best iteration/stagnation/restart marker
        + 8 * 4  # v2 扩展 diagnostics
    )
    if local_search is LocalSearch.NONE:
        return base + 4 * ants  # LSGain=-1 workspace
    local = (
        2 * ants * n  # city -> position
        + 2 * ants * n  # 随机城市顺序
        + ants * n  # DLB
        + 4 * ants  # construction 前长度
        + 4 * ants  # LSGain
        + 4 * 2  # global/restart best LSGain
        + 4  # restart iteration
    )
    if local_search is LocalSearch.THREE_OPT:
        local += 2 * ants * (n + 1)
    return base + local


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


def _run_device_v2(
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
    """在一个设备上运行分阶段、candidate-tiled CUDA v2。"""

    import cupy as cp

    if runtime.cuda_provider not in {
        CudaProvider.RAW_CUDA,
        CudaProvider.AUTO,
    }:
        raise NotImplementedError(
            "cuTile 仅作为独立 construction prototype；"
            "完整求解器应先使用 cuda_provider=raw_cuda"
        )
    if runtime.cuda_precision is CudaPrecision.AUTO:
        raise ValueError(
            "正式 CUDA v2 运行必须由 tuning manifest 将 precision=auto "
            "解析为具体 profile"
        )
    precision = runtime.cuda_precision
    candidate_lanes = runtime.cuda_candidate_lanes or 8
    candidate_size = config.resolve_candidate_size(problem.n)
    nearest_stride = int(problem.nn_indices.shape[-1])
    local_candidate_size = config.resolve_local_search_candidate_size(problem.n)
    if config.uses_local_search and local_candidate_size > nearest_stride:
        raise ValueError(
            "local_search_candidate_size 超过 ProblemBatch 中预计算的 "
            "nearest-neighbour 表宽度"
        )
    if local_candidate_size > 32:
        raise ValueError("CUDA 局部搜索当前要求 candidate_size <= 32")
    stack_depth = max(
        1,
        int(transition.stack_size),
        int(pheromone_programs.stack_size),
    )

    with cp.cuda.Device(device):
        resident = _resident_problem(
            problem,
            config,
            device,
            runtime.gpu_memory_fraction,
        )
        kernels, compile_seconds = _load_v2_kernels(
            device=device,
            variant=config.variant,
            candidate_size=candidate_size,
            candidate_lanes=candidate_lanes,
            stack_depth=stack_depth,
            precision=precision,
            register_cap=runtime.cuda_register_cap,
            generated_gp_source=(
                _generated_gp_source(
                    transition,
                    pheromone_programs,
                    representatives,
                )
                if runtime.cuda_generated_gp
                else ""
            ),
        )
        (
            init_kernel,
            construct_kernel,
            update_kernel,
            two_opt_kernel,
            three_opt_kernel,
        ) = kernels
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
        static_distances, static_heuristic, static_log_heuristic = (
            _v2_static_problem_arrays(cp, resident, precision)
        )
        cp.cuda.get_current_stream().synchronize()
        program_h2d = perf_counter() - h2d_started

        n = problem.n
        ants = config.resolve_ants(n)
        batch = problem.batch_size
        construct_threads = ants * candidate_lanes
        if construct_threads > 1024:
            raise ValueError(
                f"ants×candidate_lanes={construct_threads} 超过 CUDA block 上限"
            )
        ls_threads = 32 * runtime.cuda_ls_warps_per_block
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()
        reserve = int(total_memory * (1.0 - runtime.gpu_memory_fraction))
        usable = max(0, int(free_memory) - reserve)
        bytes_per_task = _task_bytes_v2(
            n,
            ants,
            config.iterations,
            record_anytime,
            config.local_search,
        )
        max_tasks = usable // max(bytes_per_task, 1)
        if runtime.gpu_task_chunk_size:
            max_tasks = min(max_tasks, runtime.gpu_task_chunk_size)
        if max_tasks < 1:
            raise MemoryError(
                f"GPU {device} 无法容纳单个 CUDA v2 n={n} task；"
                f"估算需 {bytes_per_task / 2**20:.1f} MiB"
            )

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

        ordered_indices = np.asarray(flat_indices, dtype=np.int64)
        task_order = runtime.cuda_task_order
        if task_order in {
            CudaTaskOrder.INSTANCE_MAJOR,
            CudaTaskOrder.AUTO,
        }:
            ordered_indices = ordered_indices[
                np.lexsort(
                    (
                        ordered_indices // batch,
                        ordered_indices % batch,
                    )
                )
            ]

        index_parts: list[np.ndarray] = []
        tour_parts: list[np.ndarray] = []
        length_parts: list[np.ndarray] = []
        iteration_parts: list[np.ndarray] = []
        anytime_parts: list[np.ndarray] = []
        diagnostic_parts: list[np.ndarray] = []
        kernel_seconds = 0.0
        d2h_seconds = 0.0
        chunk_count = 0
        for start in range(0, ordered_indices.size, max_tasks):
            selected = ordered_indices[start : start + max_tasks]
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
            length_workspace = cp.empty((count, ants), dtype=cp.float32)
            score_workspace = cp.empty(
                (count, ants, n),
                dtype=cp.float32,
            )
            deposit_workspace = cp.empty(
                (count, ants, n),
                dtype=cp.float32,
            )
            edge_frequency = cp.empty((count, n, n), dtype=cp.uint8)
            restart_tour = cp.empty((count, n + 1), dtype=cp.uint16)
            best_tours = cp.empty((count, n + 1), dtype=cp.uint16)
            global_best_lengths = cp.empty(count, dtype=cp.float32)
            restart_best_lengths = cp.empty(count, dtype=cp.float32)
            task_tau_min = cp.empty(count, dtype=cp.float32)
            task_tau_max = cp.empty(count, dtype=cp.float32)
            best_iterations = cp.empty(count, dtype=cp.int32)
            stagnation = cp.empty(count, dtype=cp.int32)
            restart_found_best = cp.empty(count, dtype=cp.int32)
            restart_iteration = cp.empty(count, dtype=cp.int32)
            global_best_ls_gain = cp.empty(count, dtype=cp.float32)
            restart_best_ls_gain = cp.empty(count, dtype=cp.float32)
            ls_gain_workspace = cp.full(
                (count, ants),
                -1.0,
                dtype=cp.float32,
            )
            length_before_workspace = cp.empty(
                (count, ants),
                dtype=cp.float32,
            )
            if config.uses_local_search:
                position_workspace = cp.empty(
                    (count, ants, n),
                    dtype=cp.uint16,
                )
                order_workspace = cp.empty(
                    (count, ants, n),
                    dtype=cp.uint16,
                )
                dlb_workspace = cp.empty(
                    (count, ants, n),
                    dtype=cp.uint8,
                )
            else:
                position_workspace = cp.empty(1, dtype=cp.uint16)
                order_workspace = cp.empty(1, dtype=cp.uint16)
                dlb_workspace = cp.empty(1, dtype=cp.uint8)
            scratch_tour_workspace = (
                cp.empty(
                    (count, ants, n + 1),
                    dtype=cp.uint16,
                )
                if config.local_search is LocalSearch.THREE_OPT
                else cp.empty(1, dtype=cp.uint16)
            )
            anytime = (
                cp.empty((count, config.iterations), dtype=cp.float32)
                if record_anytime
                else cp.empty(1, dtype=cp.float32)
            )
            diagnostics = cp.empty((count, 8), dtype=cp.uint64)

            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()
            start_event.record()
            init_kernel(
                (count,),
                (256,),
                (
                    tau0,
                    tau_min,
                    tau_max,
                    task_instance,
                    np.int32(count),
                    np.int32(n),
                    pheromone_workspace,
                    edge_frequency,
                    global_best_lengths,
                    restart_best_lengths,
                    task_tau_min,
                    task_tau_max,
                    best_iterations,
                    stagnation,
                    restart_found_best,
                    restart_iteration,
                    global_best_ls_gain,
                    restart_best_ls_gain,
                    diagnostics,
                ),
            )
            for iteration in range(1, config.iterations + 1):
                construct_kernel(
                    (count,),
                    (construct_threads,),
                    (
                        static_distances,
                        static_heuristic,
                        static_log_heuristic,
                        resident.nearest,
                        tr_ops,
                        tr_fargs,
                        tr_iargs,
                        tr_lengths,
                        tr_masks,
                        tr_is_active,
                        np.int32(tr_ops.shape[1]),
                        task_program,
                        task_instance,
                        np.int32(count),
                        np.int32(n),
                        np.int32(candidate_size),
                        np.int32(ants),
                        np.int32(config.iterations),
                        np.int32(iteration),
                        np.float32(config.alpha),
                        np.float32(config.beta),
                        np.float32(config.q0),
                        np.float32(config.xi),
                        np.float32(config.gamma_transition),
                        np.int32(transition_mode),
                        np.float32(config.epsilon_numeric),
                        np.uint64(int(seed) % (2**64)),
                        resident.instance_keys,
                        tau0,
                        pheromone_workspace,
                        tour_workspace,
                        visited_workspace,
                        length_workspace,
                        score_workspace,
                        stagnation,
                        diagnostics,
                    ),
                )
                if config.uses_local_search:
                    total_tours = count * ants
                    ls_blocks = (
                        total_tours
                        + runtime.cuda_ls_warps_per_block
                        - 1
                    ) // runtime.cuda_ls_warps_per_block
                    two_opt_kernel(
                        (ls_blocks,),
                        (ls_threads,),
                        (
                            resident.distances,
                            resident.nearest,
                            task_instance,
                            np.int32(count),
                            np.int32(n),
                            np.int32(nearest_stride),
                            np.int32(local_candidate_size),
                            np.int32(ants),
                            np.int32(iteration),
                            np.int32(config.local_search_dlb),
                            np.int32(
                                config.local_search is LocalSearch.TWO_OPT
                            ),
                            np.uint64(int(seed) % (2**64)),
                            resident.instance_keys,
                            tour_workspace,
                            length_workspace,
                            length_before_workspace,
                            ls_gain_workspace,
                            position_workspace,
                            order_workspace,
                            dlb_workspace,
                            diagnostics,
                        ),
                    )
                    if config.local_search is LocalSearch.THREE_OPT:
                        three_opt_kernel(
                            (total_tours,),
                            (runtime.cuda_three_opt_block_threads,),
                            (
                                resident.distances,
                                resident.nearest,
                                task_instance,
                                np.int32(count),
                                np.int32(n),
                                np.int32(nearest_stride),
                                np.int32(local_candidate_size),
                                np.int32(ants),
                                np.int32(config.local_search_dlb),
                                tour_workspace,
                                length_workspace,
                                length_before_workspace,
                                ls_gain_workspace,
                                position_workspace,
                                order_workspace,
                                dlb_workspace,
                                scratch_tour_workspace,
                                diagnostics,
                            ),
                        )
                update_kernel(
                    (count,),
                    (256,),
                    (
                        resident.log_heuristic,
                        resident.nearest,
                        resident.full_nn_rank,
                        resident.node_log_eta_mean,
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
                        np.int32(candidate_size),
                        np.int32(ants),
                        np.int32(config.iterations),
                        np.int32(iteration),
                        np.float32(config.rho),
                        np.float32(config.gamma_pheromone),
                        np.int32(pheromone_mode),
                        np.float32(config.epsilon_numeric),
                        np.int32(config.mmas_update_period),
                        np.float32(config.mmas_p_best),
                        np.int32(config.mmas_branch_check_period),
                        np.float32(config.mmas_branch_lambda),
                        np.float32(config.mmas_branch_threshold),
                        np.int32(config.mmas_restart_stagnation),
                        np.int32(config.uses_local_search),
                        pheromone_workspace,
                        tour_workspace,
                        length_workspace,
                        ls_gain_workspace,
                        deposit_workspace,
                        edge_frequency,
                        restart_tour,
                        best_tours,
                        global_best_lengths,
                        restart_best_lengths,
                        task_tau_min,
                        task_tau_max,
                        best_iterations,
                        stagnation,
                        restart_found_best,
                        restart_iteration,
                        global_best_ls_gain,
                        restart_best_ls_gain,
                        anytime,
                        np.int32(record_anytime),
                        diagnostics,
                    ),
                )
            end_event.record()
            end_event.synchronize()
            kernel_seconds += float(
                cp.cuda.get_elapsed_time(start_event, end_event)
            ) / 1000.0

            d2h_started = perf_counter()
            index_parts.append(selected)
            tour_parts.append(cp.asnumpy(best_tours))
            length_parts.append(cp.asnumpy(global_best_lengths))
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
            flat_indices=np.concatenate(index_parts),
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
            block_threads=construct_threads,
            device_name=device_name,
            provider=CudaProvider.RAW_CUDA.value,
            precision=precision.value,
            candidate_lanes=candidate_lanes,
            register_cap=runtime.cuda_register_cap,
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
    use_v2 = runtime.aco_backend is ExecutionBackend.CUDA_TILED_V2
    if config.uses_local_search and not use_v2:
        raise ValueError("CUDA 局部搜索只由 cuda_tiled_v2 后端实现")
    if config.resolve_ants(problem.n) > 32:
        raise ValueError("CUDA 后端最多支持 32 只蚂蚁")
    if use_v2 and config.resolve_ants(problem.n) != 32:
        raise ValueError(
            "CUDA v2 的 tiled construction 当前固定 32 只蚂蚁，"
            "以保证完整 warp/subgroup 语义"
        )
    if problem.n > np.iinfo(np.uint16).max - 1:
        raise ValueError("CUDA 后端的城市数必须小于 65535")
    if config.variant is ACOVariant.ACS and not config.acs_synchronous:
        raise ValueError("CUDA 后端只支持同步 ACS local update")
    (
        transition,
        pheromone_programs,
        transition_active,
        pheromone_active,
        representatives,
        inverse,
    ) = _active_and_representative_programs(programs, config)
    if max(transition.stack_size, pheromone_programs.stack_size) > 32:
        raise ValueError("CUDA GP postfix stack 深度不得超过 32")

    devices = _resolved_devices(runtime)
    tuning_manifest_hash = ""
    if use_v2:
        runtime, tuning_manifest_hash = _runtime_from_tuning_manifest(
            runtime,
            devices,
        )
        if runtime.cuda_graph_replay:
            raise NotImplementedError(
                "CUDA graph replay 尚未满足跨 generation 指针更新契约"
            )
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
    run_device = _run_device_v2 if use_v2 else _run_device
    started = perf_counter()
    if len(shards) == 1:
        device_results = [
            run_device(
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
                    run_device,
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
    diagnostic_width = 8 if use_v2 else 4
    diagnostics_flat = np.empty(
        (task_count, diagnostic_width),
        dtype=np.uint64,
    )
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
            diagnostic_width,
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
        "provider": device_results[0].provider,
        "precision": device_results[0].precision,
        "candidate_lanes": device_results[0].candidate_lanes,
        "register_cap": device_results[0].register_cap,
        "gpu_fp32_length_checksum": float(gpu_length_flat.sum(dtype=np.float64)),
        "local_search": config.local_search.value,
        "local_search_candidate_size": (
            config.resolve_local_search_candidate_size(problem.n)
        ),
        "local_search_warps_per_block": runtime.cuda_ls_warps_per_block,
        "three_opt_block_threads": runtime.cuda_three_opt_block_threads,
    }
    if tuning_manifest_hash:
        metrics["tuning_manifest_sha256"] = tuning_manifest_hash
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


def solve_population_cuda_anytime(
    problem: ProblemBatch,
    config: ACOConfig,
    programs: list[tuple[TensorProgram | None, TensorProgram | None]],
    *,
    seed: int,
    runtime: RuntimeConfig,
) -> PopulationRunResult:
    """融合评估多个锁定 program，并保留各自完整 anytime 曲线。"""

    quality, anytime = _solve_population_impl(
        problem,
        config,
        programs,
        seed=seed,
        runtime=runtime,
        record_anytime=True,
    )
    assert anytime is not None
    return PopulationRunResult(
        best_tour=quality.best_tour,
        best_length=quality.best_length,
        best_iteration=quality.best_iteration,
        anytime_best=anytime,
        diagnostics=quality.diagnostics,
        wall_time_sec=quality.wall_time_sec,
        constructed_tours=quality.constructed_tours,
        backend_metrics=quality.backend_metrics,
    )


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
    local_values = [
        int(diagnostic[index].item())
        if diagnostic.numel() > index
        else 0
        for index in range(4, 8)
    ]
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
            local_search_move_count=local_values[0],
            local_search_candidate_check_count=local_values[1],
            local_search_improved_tour_count=local_values[2],
            local_search_pass_count=local_values[3],
        ),
        backend_metrics=quality.backend_metrics,
    )
