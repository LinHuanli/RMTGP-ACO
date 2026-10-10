"""无 JIT 的 Python/NumPy 基线；实例多进程、候选向量化，所有搜索浮点量为 FP32。

不能用 solve_kernel.py_func 冒充这个基线：其下层调用仍可能是 Numba dispatcher。
本模块的数值执行不调用任何 njit 函数；与 Numba 共享的仅是结果数据类型。
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from ..data import ROOT, ProblemSpec
from ..language import evaluate_reference
from .cpu import EvaluationResult
from .local_search import improve

F = np.float32


def uniform(seed, key, iteration, ant, step, purpose):
    """向量化 Philox4x32-10；整数溢出显式截断，不依赖 Python 的隐式精度。"""
    shape = np.broadcast_shapes(np.shape(ant), np.shape(step))
    mask = np.uint64(0xFFFFFFFF)
    combined = int(seed) ^ int(key)
    k0, k1 = np.uint64(combined & 0xFFFFFFFF), np.uint64(combined >> 32)
    c0 = np.full(shape, iteration, np.uint64)
    c1 = np.broadcast_to(np.asarray(ant, np.uint64), shape)
    c2 = np.broadcast_to(np.asarray(step, np.uint64), shape)
    c3 = np.full(shape, purpose, np.uint64)
    for _ in range(10):
        p0, p1 = c0 * np.uint64(0xD2511F53), c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(32)) ^ c1 ^ k0,
            p1 & mask,
            (p0 >> np.uint64(32)) ^ c3 ^ k1,
            p0 & mask,
        )
        k0, k1 = (k0 + np.uint64(0x9E3779B9)) & mask, (k1 + np.uint64(0xBB67AE85)) & mask
    return (c0 >> np.uint64(8)).astype(np.float32) * F(1 / 16777216)


def ordered_sum(values):
    """使用累积加法保持顺序 FP32 归约，不使用默认的分组求和。"""
    return np.cumsum(values, dtype=np.float32)[-1]


def stdrel(values):
    centered = values - values[0]
    centered = centered - ordered_sum(centered) / F(len(values))
    denominator = np.sqrt(ordered_sum(centered * centered) / F(len(values))) + F(1e-8)
    return np.tanh(centered / denominator)


def fields(
    program,
    geometry,
    tau_row,
    current,
    previous,
    candidates,
    fallback,
    step,
    iteration,
    stagnation,
    config,
):
    """同状态重放也使用此接口；返回终端字典、未修正分数及保护标记。"""
    coords, distances, eta, log_eta, nearest, ranks = geometry
    n, m = len(coords), len(candidates)
    epsilon = F(config.epsilon)
    score = (tau_row[candidates] * eta[current, candidates]) * eta[current, candidates]
    total = ordered_sum(score)
    uniform_base = total <= epsilon
    context = {}
    mask = program.required_mask
    if mask & ((1 << 2) | (1 << 4)):
        probability = (
            np.full(m, F(1) / F(m), np.float32) if uniform_base else score * (F(1) / total)
        )
        if mask & (1 << 2):
            context["BaseConf"] = np.tanh(np.log(np.maximum(probability, epsilon)) + np.log(F(m)))
        if mask & (1 << 4):
            value = (
                F(-1)
                if m == 1
                else F(2)
                * ordered_sum(-probability * np.log(np.maximum(probability, epsilon)))
                / max(np.log(F(m)), epsilon)
                - F(1)
            )
            context["Entropy"] = np.full(m, value, np.float32)
    if mask & 1:
        context["RTau"] = stdrel(np.log(np.maximum(tau_row[candidates], epsilon)))
    if mask & 2:
        context["REta"] = stdrel(log_eta[current, candidates])
    if mask & (1 << 3):
        # 保留基线的两两排名定义，不能把尚待研究的排名重写预先混入基线。
        rank = np.arange(m, dtype=np.float32)
        if fallback:
            distance = distances[current, candidates]
            less = distance[None, :] < distance[:, None]
            tie = (distance[None, :] == distance[:, None]) & (rank[None, :] < rank[:, None])
            rank = np.count_nonzero(less | tie, axis=1).astype(np.float32)
        context["DistRank"] = np.zeros(m, np.float32) if m == 1 else F(1) - F(2) * rank / F(m - 1)
    for bit, name, value in (
        (5, "ConstructProg", F(2) * F(step) / F(n - 1) - F(1)),
        (6, "ACOProg", F(2) * F(iteration - 1) / F(max(config.iterations - 1, 1)) - F(1)),
        (7, "Stagnation", F(2) * min(F(stagnation) / F(config.iterations), F(1)) - F(1)),
    ):
        if mask & (1 << bit):
            context[name] = np.full(m, value, np.float32)
    if mask & (1 << 14):
        forward = ranks[current, candidates].astype(np.float32) - F(1)
        reverse = ranks[candidates, current].astype(np.float32) - F(1)
        context["MutualRank"] = np.clip(
            F(1) - (forward + reverse) / (F(2) * F(max(n - 2, 1))), F(0), F(1)
        )
    if mask & (1 << 15):
        value = np.zeros(m, np.float32)
        if previous >= 0:
            incoming, outgoing = (
                coords[current] - coords[previous],
                coords[candidates] - coords[current],
            )
            ni = np.sqrt(incoming[0] * incoming[0] + incoming[1] * incoming[1])
            no = np.sqrt(outgoing[:, 0] ** 2 + outgoing[:, 1] ** 2)
            value = (incoming[0] * outgoing[:, 0] + incoming[1] * outgoing[:, 1]) / (
                ni * no + epsilon
            )
        context["TurnCos"] = np.clip(value, F(-1), F(1))
    return context, score, uniform_base


def initialization(problem, config, seed):
    frozen = problem.initialization
    if frozen is not None:
        if (frozen.seed, frozen.variant, frozen.rho, frozen.instance_ids) != (
            seed,
            config.variant,
            config.rho,
            problem.instance_ids,
        ) or any(v.dtype != np.float32 or v.shape != (problem.size,) for v in frozen.values):
            raise ValueError("冻结初始化身份、形状或类型错误")
        return frozen.values
    values = np.empty((3, problem.size), np.float32)
    for b in range(problem.size):
        n, distance = problem.n, problem.distances[b]
        current = start = min(
            int(uniform(seed, problem.instance_keys[b], 0, 0, 0, 0) * F(n)), n - 1
        )
        visited, length = np.zeros(n, bool), F(0)
        visited[current] = True
        for _ in range(n - 1):
            candidates = np.flatnonzero(~visited)
            chosen = candidates[np.argmin(distance[current, candidates])]
            length += distance[current, chosen]
            current, visited[chosen] = chosen, True
        length += distance[current, start]
        high = (
            F(1) / (F(n) * length) if config.variant == "acs" else F(1) / (F(config.rho) * length)
        )
        values[:, b] = (
            high,
            high / (F(2) * F(n)) if config.variant == "mmas" else F(0),
            high if config.variant == "mmas" else F(np.inf),
        )
    return tuple(values)


def local_update(tau, u, v, tau0, factors):
    """同步 ACS：相同无向边合并 multiplicity，全部蚂蚁选择之后才可见。"""
    n = len(tau)
    edges, counts = np.unique(np.minimum(u, v) * n + np.maximum(u, v), return_counts=True)
    first, second = edges // n, edges % n
    factor = factors[counts]
    value = factor * tau[first, second] + (F(1) - factor) * tau0
    tau[first, second], tau[second, first] = value, value


def solve_instance(programs, geometry, key, initial, config, seed):
    """一个实例内顺序评估程序，与现有 Numba 的任务粒度一致。"""
    n, a = len(geometry[0]), config.ants
    distance, nearest = geometry[1], geometry[4]
    rho, gamma, epsilon = F(config.rho), F(config.gamma), F(config.epsilon)
    factors = (F(1) - F(config.xi)) ** np.arange(a + 1, dtype=np.float32)
    output, routes, diagnostics = (
        np.empty(len(programs), np.float32),
        np.empty((len(programs), n + 1), np.int32),
        np.zeros((len(programs), 8), np.uint64),
    )
    tau, dense = np.empty((n, n), np.float32), np.empty((n, n), np.float32)
    off_diagonal = ~np.eye(n, dtype=bool)
    tours, visited = np.empty((a, n + 1), np.int32), np.empty((a, n), bool)
    ant_indices = np.arange(a)
    ls_mode = ("none", "two_opt", "three_opt").index(config.local_search)
    ls_stats = np.zeros((len(programs), 4), np.uint64)
    ls_pos, ls_scratch = np.empty(n, np.int32), np.empty(n, np.int32)
    ls_dlb = np.empty(n, np.uint8)
    for p, program in enumerate(programs):
        tau.fill(initial[0])
        np.fill_diagonal(tau, F(0))
        best = restart_best = F(np.inf)
        low, high = initial[1:]
        stagnation = found = 0
        for iteration in range(1, config.iterations + 1):
            # 一次生成该迭代的逻辑随机输入；仍在计时区间内，不复用适应度。
            starts = np.minimum(
                (uniform(seed, key, iteration, ant_indices, 0, 1) * F(n)).astype(int), n - 1
            )
            greedy = (
                uniform(seed, key, iteration, ant_indices[:, None], np.arange(1, n)[None, :], 2)
                if config.variant == "acs"
                else None
            )
            roulette = uniform(
                seed, key, iteration, ant_indices[:, None], np.arange(1, n)[None, :], 3
            )
            tours[:, 0] = starts
            visited.fill(False)
            visited[ant_indices, starts] = True
            for step in range(1, n):
                for ant in range(a):
                    current = tours[ant, step - 1]
                    previous = tours[ant, step - 2] if step > 1 else -1
                    candidates = nearest[current][~visited[ant, nearest[current]]].astype(np.int32)
                    fallback = len(candidates) == 0
                    if fallback:
                        candidates = np.flatnonzero(~visited[ant])
                        diagnostics[p, :2] += 1
                    context, score, base_uniform = fields(
                        program,
                        geometry,
                        tau[current],
                        current,
                        previous,
                        candidates,
                        fallback,
                        step,
                        iteration,
                        stagnation,
                        config,
                    )
                    score *= F(1) + gamma * np.tanh(
                        evaluate_reference(program, context, (len(candidates),))
                    )
                    if fallback or (greedy is not None and greedy[ant, step - 1] <= F(config.q0)):
                        selected = int(np.argmax(score))
                        if base_uniform and not fallback:
                            diagnostics[p, 1] += 1
                    else:
                        cumulative = np.cumsum(score, dtype=np.float32)
                        total, random = cumulative[-1], roulette[ant, step - 1]
                        if base_uniform or total <= epsilon:
                            diagnostics[p, 1] += 1
                        selected = (
                            min(int(random * F(len(candidates))), len(candidates) - 1)
                            if total <= epsilon
                            else min(
                                int(np.searchsorted(cumulative, random * total, side="left")),
                                len(candidates) - 1,
                            )
                        )
                    tours[ant, step] = candidates[selected]
                if config.variant == "acs":
                    local_update(tau, tours[:, step - 1], tours[:, step], initial[0], factors)
                visited[ant_indices, tours[:, step]] = True
            tours[:, n] = tours[:, 0]
            if config.variant == "acs":
                local_update(tau, tours[:, n - 1], tours[:, 0], initial[0], factors)
            if ls_mode:
                for ant in range(a):
                    order = np.arange(n, dtype=np.int32)
                    draws = uniform(seed, key, iteration, ant, np.arange(n - 1), 4)
                    for index in range(n - 1):
                        other = index + min(int(draws[index] * F(n - index)), n - index - 1)
                        order[index], order[other] = order[other], order[index]
                    improve(
                        tours[ant],
                        distance,
                        nearest,
                        order,
                        ls_mode,
                        min(config.ls_candidate_size, nearest.shape[1]),
                        ls_pos,
                        ls_dlb,
                        ls_scratch,
                        ls_stats[p],
                    )
            lengths = np.cumsum(distance[tours[:, :-1], tours[:, 1:]], axis=1, dtype=np.float32)[
                :, -1
            ]
            winner = int(np.argmin(lengths))
            if lengths[winner] < best:
                best, best_tour, stagnation = lengths[winner], tours[winner].copy(), 0
                if config.variant == "mmas":
                    high = F(1) / (rho * best)
                    px = np.exp(np.log(F(config.mmas_p_best)) / F(n))
                    low = high * (F(1) - px) / (px * F((nearest.shape[1] + 1) // 2))
            else:
                stagnation += 1
            if lengths[winner] < restart_best:
                restart_best, restart_tour, found = lengths[winner], tours[winner].copy(), iteration
            if config.variant == "acs":
                u, v = best_tour[:-1], best_tour[1:]
                value = (F(1) - rho) * tau[u, v] + rho * (F(1) / best)
                tau[u, v], tau[v, u] = value, value
            else:
                dense.fill(F(0))
                sources = (
                    zip(tours, lengths, strict=True)
                    if config.variant == "as"
                    else [
                        (restart_tour, restart_best)
                        if iteration % config.mmas_period == 0
                        else (tours[winner], lengths[winner])
                    ]
                )
                for tour, length in sources:
                    deposit = F(1) / length
                    np.add.at(dense, (tour[:-1], tour[1:]), deposit)
                    np.add.at(dense, (tour[1:], tour[:-1]), deposit)
                raw = (F(1) - rho) * tau + dense
                if config.variant == "mmas":
                    tau[:] = np.clip(raw, low, high)
                    diagnostics[p, 2] += np.uint64(np.count_nonzero((tau != raw) & off_diagonal))
                else:
                    tau[:] = raw
                np.fill_diagonal(tau, F(0))
            if (
                config.variant == "mmas"
                and iteration % config.branch_period == 0
                and iteration - found > config.restart_stagnation
            ):
                edges = np.take_along_axis(tau, nearest.astype(int), axis=1)
                cutoff = edges.min(axis=1) + F(config.branch_lambda) * (
                    edges.max(axis=1) - edges.min(axis=1)
                )
                if F(np.count_nonzero(edges > cutoff[:, None])) / (F(2) * F(n)) < F(
                    config.branch_threshold
                ):
                    tau.fill(high)
                    np.fill_diagonal(tau, F(0))
                    restart_best, found = F(np.inf), iteration
                    diagnostics[p, 3] += 1
        output[p], routes[p] = best, best_tour
    return output, routes, diagnostics, ls_stats


_WORKER = None


def _init_worker(problem, programs, config, seed, initial):
    global _WORKER
    if isinstance(problem, dict):
        problem = ProblemSpec(
            **{
                k: np.load(v, mmap_mode="r", allow_pickle=False)
                if k != "instance_ids"
                else tuple(v)
                for k, v in problem.items()
            }
        )
    _WORKER = (problem, programs, config, seed, initial)


def _instance(b):
    problem, programs, config, seed, initial = _WORKER
    geometry = tuple(
        getattr(problem, name)[b]
        for name in ("coords", "distances", "heuristic", "log_heuristic", "nearest", "full_nn_rank")
    )
    return solve_instance(
        programs, geometry, problem.instance_keys[b], tuple(v[b] for v in initial), config, seed
    )


def evaluate(programs, problem, config, seed, plan):
    begin = perf_counter()
    initial = initialization(problem, config, seed)
    init_s = perf_counter() - begin
    # spawn 不继承已初始化的 OpenMP/Numba runtime；计时包含池的启动和回收。
    if plan.cpu_threads == 1:
        geometry_names = (
            "coords",
            "distances",
            "heuristic",
            "log_heuristic",
            "nearest",
            "full_nn_rank",
        )
        rows = [
            solve_instance(
                programs,
                tuple(getattr(problem, f)[b] for f in geometry_names),
                problem.instance_keys[b],
                tuple(v[b] for v in initial),
                config,
                seed,
            )
            for b in range(problem.size)
        ]
    else:
        # 大几何不通过 pickle 给每个进程复制。文件映射落在项目内，内核共享只读页。
        parent = ROOT / ".cache/cpu-python"
        parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="geometry-", dir=parent) as directory:
            descriptor = {"instance_ids": problem.instance_ids}
            for name in (
                "coords",
                "distances",
                "heuristic",
                "log_heuristic",
                "nearest",
                "full_nn_rank",
                "reference",
                "instance_keys",
            ):
                path = Path(directory) / f"{name}.npy"
                np.save(path, getattr(problem, name), allow_pickle=False)
                descriptor[name] = str(path)
            with ProcessPoolExecutor(
                max_workers=plan.cpu_threads,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(descriptor, programs, config, seed, initial),
            ) as pool:
                rows = list(pool.map(_instance, range(problem.size)))
    lengths, tours, diagnostics = (np.stack([row[i] for row in rows], axis=1) for i in range(3))
    return EvaluationResult(
        lengths,
        tours,
        diagnostics,
        {
            "backend": "cpu_python",
            "eval_wall_s": perf_counter() - begin,
            "initialization_s": init_s,
            "executed_tasks": len(programs) * problem.size,
            "cpu_workers": plan.cpu_threads,
            "parallelism": "processes",
            "pool_lifecycle_included": True,
            "numba_numeric_execution": False,
        },
        np.stack([row[3] for row in rows], axis=1) if config.local_search != "none" else None,
    )
