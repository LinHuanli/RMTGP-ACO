"""单卡 population×instance tiled 执行器，保留完整搜索在 GPU 上。"""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from ..config import InfeasiblePlan
from ..language import pack_programs
from .cpu import EvaluationResult, problem_initialization

_MODULES = {}
SOURCE_ROOT = Path(__file__).with_name("cuda")


def generated_source(programs):
    """基线已有的树专门化；每个 primitive 的保护边界必须保留。"""
    lines = [
        "__device__ __forceinline__ float evaluate_transition_generated(int p,const float* t) { switch(p) {"
    ]
    for index, program in enumerate(programs):
        lines.append(f"case {index}: {{")
        stack = []
        for instruction, (op, value, terminal) in enumerate(program.instructions):
            name = f"v{instruction}"
            if op == 0:
                bits = np.float32(value).view(np.uint32).item()
                expression = f"__uint_as_float({bits}U)"
            elif op == 1:
                expression = f"t[{terminal}]"
            else:
                b = stack.pop()
                if op == 9:
                    expression = f"fabsf({b})"
                elif op == 10:
                    expression = f"-{b}"
                else:
                    a = stack.pop()
                    if op == 2:
                        expression = f"{a}+{b}"
                    elif op == 3:
                        expression = f"{a}-{b}"
                    elif op == 4:
                        expression = f"{a}*{b}"
                    elif op == 5:
                        expression = f"({a}*{b})/({b}*{b}+1.0e-6f)"
                    elif op == 7:
                        expression = f"gp_min({a},{b})"
                    else:
                        expression = f"gp_max({a},{b})"
                expression = f"sanitize({expression})"
            lines.append(f"float {name}={expression};")
            stack.append(name)
        lines.append(f"return sanitize({stack[0]}); }}")
    lines.append("default: return 0.0f; } }")
    return "\n".join(lines)


def kernels(config, plan, programs):
    import cupy as cp

    begin = perf_counter()
    common = (SOURCE_ROOT / "common.cuh").read_text()
    generated = generated_source(programs) if plan.generated else ""
    code = (
        common
        + "\n"
        + generated
        + "\n"
        + (SOURCE_ROOT / "construct.cu").read_text()
        + "\n"
        + (SOURCE_ROOT / "update.cu").read_text()
    )
    options = (
        "--std=c++17",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        f"-DRMTGP_VARIANT={config.variant_id}",
        f"-DRMTGP_CANDIDATE_LANES={plan.candidate_lanes}",
        f"-DRMTGP_CANDIDATE_PAD={max(32, config.candidate_size)}",
        f"-DRMTGP_GENERATED_GP={int(plan.generated)}",
    )
    key = (cp.cuda.Device().id, sha256(code.encode()).hexdigest(), options)
    hit = key in _MODULES
    if not hit:
        module = cp.RawModule(code=code, options=options, backend="nvrtc")
        functions = {
            name: module.get_function(name)
            for name in ("initialize", "v2_construct", "update", "probe_rng", "probe_program")
        }
        _MODULES[key] = (module, functions)
        # 保留近期编译模块，防止长期训练无限占用驱动内存。
        if len(_MODULES) > 8:
            _MODULES.pop(next(iter(_MODULES)))
    return _MODULES[key][1], perf_counter() - begin, hit


def evaluate(programs, problem, config, seed, plan, *, capture_state=False):
    import cupy as cp

    cp.cuda.get_current_stream().synchronize()
    begin = perf_counter()
    functions, compile_s, hit = kernels(config, plan, programs)
    p, b, n = len(programs), problem.size, problem.n
    a, k = config.ants, problem.nearest.shape[2]
    maximum_threads = functions["v2_construct"].attributes["max_threads_per_block"]
    if a * plan.candidate_lanes > maximum_threads:
        raise InfeasiblePlan(
            f"编译后的寄存器资源只支持 {maximum_threads} threads/block，"
            f"当前 ants×lanes={a * plan.candidate_lanes}；此执行计划不可行"
        )
    packed = pack_programs(programs)
    device_programs = tuple(cp.asarray(v) for v in packed)
    geometry = tuple(
        cp.asarray(v)
        for v in (
            problem.coords,
            problem.distances,
            problem.heuristic,
            problem.log_heuristic,
            problem.nearest,
            problem.full_nn_rank,
        )
    )
    keys = cp.asarray(problem.instance_keys)
    initial = tuple(
        cp.asarray(v)
        for v in problem_initialization(problem, config, seed)
    )
    total = p * b
    free, _ = cp.cuda.runtime.memGetInfo()
    # 两个密集矩阵、蚂蚁状态、路径与分数；容量控制只能分波次，不能减少任务。
    task_bytes = 8 * n * n + a * (8 * ((n + 63) // 64) + 6 * n + 8) + 8 * n + 512
    active = min(total, plan.active_tasks, max(1, int(free * 0.75) // task_bytes))
    lengths_out = np.empty(total, np.float32)
    tours_out = np.empty((total, n + 1), np.int32)
    diagnostics_out = np.empty((total, 8), np.uint64)
    state_capture = []
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    device_ms = 0.0
    stage_events = []
    for offset in range(0, total, active):
        tasks = min(active, total - offset)
        logical = np.arange(offset, offset + tasks)
        task_program = cp.asarray((logical // b).astype(np.int32))
        task_instance = cp.asarray((logical % b).astype(np.int32))
        tau = cp.empty((tasks, n, n), cp.float32)
        deposits = cp.empty_like(tau)
        state = cp.empty((tasks, 4), cp.float32)
        counts = cp.empty(tasks * 2, cp.int32)
        tours = cp.empty((tasks, a, n + 1), cp.uint16)
        visited = cp.empty((tasks, a, (n + 63) // 64), cp.uint64)
        colony_lengths = cp.empty((tasks, a), cp.float32)
        scores = cp.empty((tasks, a, n), cp.float32)
        best = cp.empty((tasks, n + 1), cp.uint16)
        restart = cp.empty_like(best)
        diagnostics = cp.empty((tasks, 8), cp.uint64)
        integer = np.int32
        floating = np.float32
        functions["initialize"](
            (tasks,),
            (256,),
            (*initial, task_instance, integer(tasks), integer(n), tau, state, counts, diagnostics),
        )
        start.record()
        for iteration in range(1, config.iterations + 1):
            if plan.profile_stages:
                # 诊断专用：事件插桩开销不可混入正式无插桩性能样本。
                stage_start, stage_middle, stage_end = (cp.cuda.Event() for _ in range(3))
                stage_start.record()
            functions["v2_construct"](
                (tasks,),
                (a * plan.candidate_lanes,),
                (
                    *geometry,
                    *device_programs,
                    integer(packed[0].shape[1]),
                    task_program,
                    task_instance,
                    integer(tasks),
                    integer(n),
                    integer(k),
                    integer(a),
                    integer(config.iterations),
                    integer(iteration),
                    floating(1),
                    floating(2),
                    floating(config.q0),
                    floating(config.xi),
                    floating(config.gamma),
                    integer(0),
                    floating(config.epsilon),
                    np.uint64(seed),
                    keys,
                    initial[0],
                    tau,
                    tours,
                    visited,
                    colony_lengths,
                    scores,
                    counts,
                    diagnostics,
                ),
            )
            if plan.profile_stages:
                stage_middle.record()
            functions["update"](
                (tasks,),
                (256,),
                (
                    geometry[4],
                    task_instance,
                    integer(tasks),
                    integer(n),
                    integer(k),
                    integer(a),
                    integer(iteration),
                    floating(config.rho),
                    integer(config.mmas_period),
                    floating(config.mmas_p_best),
                    integer(config.branch_period),
                    floating(config.branch_lambda),
                    floating(config.branch_threshold),
                    integer(config.restart_stagnation),
                    tau,
                    tours,
                    colony_lengths,
                    deposits,
                    best,
                    restart,
                    state,
                    counts,
                    diagnostics,
                ),
            )
            if plan.profile_stages:
                stage_end.record()
                stage_events.append((stage_start, stage_middle, stage_end))
        stop.record()
        stop.synchronize()
        device_ms += cp.cuda.get_elapsed_time(start, stop)
        lengths_out[offset : offset + tasks] = cp.asnumpy(state[:, 0])
        tours_out[offset : offset + tasks] = cp.asnumpy(best)
        diagnostics_out[offset : offset + tasks] = cp.asnumpy(diagnostics)
        if capture_state:
            state_capture.append(
                {
                    "tau": cp.asnumpy(tau),
                    "colony_tours": cp.asnumpy(tours),
                    "colony_lengths": cp.asnumpy(colony_lengths),
                    "state": cp.asnumpy(state),
                }
            )
        del (
            tau,
            deposits,
            state,
            counts,
            tours,
            visited,
            colony_lengths,
            scores,
            best,
            restart,
            diagnostics,
        )
    cp.cuda.get_current_stream().synchronize()
    timing = {
        "backend": "cuda_existing",
        "eval_wall_s": perf_counter() - begin,
        "compile_s": compile_s,
        "code_cache_hit": hit,
        "code_cache_hit_scope": "in_process_module; cupy_disk_cache_may_also_hit",
        "device_search_s": device_ms / 1000,
        "executed_tasks": total,
        "active_tasks": active,
        "waves": (total + active - 1) // active,
        "candidate_lanes": plan.candidate_lanes,
        "generated": plan.generated,
        "gpu_pool_reserved_bytes": cp.get_default_memory_pool().total_bytes(),
        "instrumented": plan.profile_stages,
        "kernel_resources": {
            name: functions[name].attributes for name in ("v2_construct", "update")
        },
    }
    if plan.profile_stages:
        timing["construct_device_s"] = (
            sum(cp.cuda.get_elapsed_time(a, b) for a, b, _ in stage_events) / 1000
        )
        timing["update_device_s"] = (
            sum(cp.cuda.get_elapsed_time(b, c) for _, b, c in stage_events) / 1000
        )
    result = EvaluationResult(
        lengths_out.reshape(p, b),
        tours_out.reshape(p, b, n + 1),
        diagnostics_out.reshape(p, b, 8),
        timing,
    )
    if capture_state:
        result.state_capture = state_capture
    return result
