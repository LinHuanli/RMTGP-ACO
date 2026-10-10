"""CUDA 局部搜索模块与显存工作区；无逐迭代主机传输。"""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

_MODULES = {}


def kernel(executor):
    import cupy as cp

    started = perf_counter()
    root = Path(__file__).with_name("cuda")
    code = (root / "common.cuh").read_text() + "\n" + (root / "local_search.cu").read_text()
    lanes = 1 if executor == "scalar" else 32
    options = (
        "--std=c++17",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        f"-DLS_LANES={lanes}",
    )
    key = (cp.cuda.Device().id, sha256(code.encode()).hexdigest(), options)
    hit = key in _MODULES
    if not hit:
        module = cp.RawModule(code=code, options=options, backend="nvrtc")
        _MODULES[key] = (
            module,
            {name: module.get_function(name) for name in ("improve_tours", "make_orders")},
        )
    return _MODULES[key][1], perf_counter() - started, hit


def workspace(tasks, ants, n, batch=None):
    import cupy as cp

    shape = (tasks * ants, n)
    return (
        cp.empty(shape, cp.int32),
        cp.empty(((tasks if batch is None else batch) * ants, n), cp.int32),
        cp.empty(shape, cp.uint8),
        cp.empty(shape, cp.uint16),
        cp.zeros((tasks * ants, 4), cp.uint64),
    )


def launch(
    function,
    executor,
    distance,
    nearest,
    task_instance,
    keys,
    tasks,
    n,
    ants,
    iteration,
    seed,
    mode,
    limit,
    tours,
    lengths,
    buffers,
):
    if nearest.dtype != np.dtype("uint16") or distance.dtype != np.dtype("float32"):
        raise ValueError("CUDA局部搜索要求 uint16 候选表和 FP32 距离")
    stride = nearest.shape[2]
    lanes = 1 if executor == "scalar" else 32
    batch = distance.shape[0]
    if buffers[1].shape[0] < batch * ants:
        raise ValueError("共享随机排列工作区不足")
    function["make_orders"](
        ((batch * ants + 127) // 128,),
        (128,),
        (buffers[1], keys, *map(np.int32, (batch, ants, n, iteration)), np.uint64(seed)),
    )
    function["improve_tours"](
        ((tasks * ants * lanes + 127) // 128,),
        (128,),
        (
            distance,
            nearest,
            task_instance,
            keys,
            *map(np.int32, (tasks, n, stride, min(limit, stride), ants, iteration)),
            np.uint64(seed),
            np.int32(mode),
            tours,
            lengths,
            *buffers,
        ),
    )
