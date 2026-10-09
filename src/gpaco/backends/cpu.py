"""Numba 单树 FP32 强基线；按实例并行，复用线程私有构造工作区。"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit, prange, set_num_threads

from ..language import pack_programs
from .cpu_existing import _construct_tours, _nearest_neighbour_length, _tour_lengths
from .numeric import counter_uniform


@dataclass
class EvaluationResult:
    lengths: np.ndarray
    tours: np.ndarray
    diagnostics: np.ndarray
    timings: dict


@njit(cache=True)
def initial_parameters(distances, keys, seed, variant, rho):
    batch, n, _ = distances.shape
    tau0 = np.empty(batch, np.float32)
    low = np.empty(batch, np.float32)
    high = np.empty(batch, np.float32)
    for b in range(batch):
        start = min(int(counter_uniform(seed, keys[b], 0, 0, 0, 0) * np.float32(n)), n - 1)
        length = np.float32(_nearest_neighbour_length(distances[b], start))
        if variant == 1:
            tau0[b] = np.float32(1) / (np.float32(n) * length)
            low[b], high[b] = np.float32(0), np.float32(np.inf)
        else:
            tau0[b] = np.float32(1) / (rho * length)
            high[b] = tau0[b] if variant == 2 else np.float32(np.inf)
            low[b] = tau0[b] / (np.float32(2) * np.float32(n)) if variant == 2 else np.float32(0)
    return tau0, low, high


@njit(cache=True, nogil=True, parallel=True)
def solve_kernel(
    coords,
    distances,
    eta,
    log_eta,
    nearest,
    ranks,
    keys,
    seed,
    opcodes,
    fargs,
    iargs,
    code_lengths,
    masks,
    ants,
    iterations,
    variant,
    rho,
    q0,
    xi,
    gamma,
    epsilon,
    period,
    p_best,
    branch_period,
    branch_lambda,
    branch_threshold,
    restart_stagnation,
    initial_tau,
    initial_low,
    initial_high,
):
    programs, batch, n = opcodes.shape[0], coords.shape[0], coords.shape[1]
    output_lengths = np.empty((programs, batch), np.float32)
    output_tours = np.empty((programs, batch, n + 1), np.int32)
    output_diagnostics = np.zeros((programs, batch, 8), np.uint64)
    one, zero = np.float32(1), np.float32(0)
    local_factors = np.empty(ants + 1, np.float32)
    for i in range(ants + 1):
        local_factors[i] = (one - xi) ** np.float32(i)
    for b in prange(batch):
        # 同一实例的程序依次运行，共享只读几何并复用工作区容量。
        tau = np.empty((n, n), np.float32)
        dense = np.zeros((n, n), np.float32)
        tours = np.empty((ants, n + 1), np.int32)
        visited = np.empty((ants, n), np.uint8)
        lengths = np.empty(ants, np.float32)
        best_tour, restart_tour = np.empty(n + 1, np.int32), np.empty(n + 1, np.int32)
        workspace = (
            np.empty(n, np.int32),
            np.zeros((16, n), np.float32),
            np.empty(n, np.float32),
            np.empty(n, np.float32),
            np.empty((31, n), np.float32),
            np.empty(ants, np.int32),
            np.empty(ants, np.int32),
            np.zeros(n * n, np.int32),
            np.empty(ants, np.int32),
            np.zeros(8, np.uint64),
        )
        geometry = (coords[b], distances[b], eta[b], log_eta[b], nearest[b], ranks[b])
        for p in range(programs):
            tau[:, :] = initial_tau[b]
            for i in range(n):
                tau[i, i] = zero
            dense[:, :] = zero
            workspace[-1][:] = 0
            best, restart_best = np.float32(np.inf), np.float32(np.inf)
            low, high = initial_low[b], initial_high[b]
            stagnation, found = 0, 0
            length = code_lengths[p]
            program = (True, opcodes[p, :length], fargs[p, :length], iargs[p, :length], masks[p])
            for iteration in range(1, iterations + 1):
                state = (
                    variant,
                    True,
                    one,
                    np.float32(2),
                    q0,
                    local_factors,
                    initial_tau[b],
                    epsilon,
                    0,
                    gamma,
                )
                random_state = (seed, keys[b], iteration, stagnation, iterations)
                _construct_tours(
                    geometry, tau, tours, visited, state, program, random_state, workspace
                )
                _tour_lengths(distances[b], tours, lengths)
                winner = 0
                for a in range(1, ants):
                    if lengths[a] < lengths[winner]:
                        winner = a
                if lengths[winner] < best:
                    best = lengths[winner]
                    best_tour[:] = tours[winner]
                    stagnation = 0
                    if variant == 2:
                        high = one / (rho * best)
                        px = np.exp(np.log(p_best) / np.float32(n))
                        low = high * (one - px) / (px * np.float32((nearest.shape[2] + 1) // 2))
                else:
                    stagnation += 1
                if lengths[winner] < restart_best:
                    restart_best = lengths[winner]
                    restart_tour[:] = tours[winner]
                    found = iteration
                if variant == 1:
                    deposit = one / best
                    for i in range(n):
                        u, v = best_tour[i], best_tour[i + 1]
                        value = (one - rho) * tau[u, v] + rho * deposit
                        tau[u, v], tau[v, u] = value, value
                else:
                    sources = ants if variant == 0 else 1
                    for a in range(sources):
                        if variant == 0:
                            route, score = tours[a], lengths[a]
                        elif iteration % period == 0:
                            route, score = restart_tour, restart_best
                        else:
                            route, score = tours[winner], lengths[winner]
                        deposit = one / score
                        for i in range(n):
                            u, v = route[i], route[i + 1]
                            dense[u, v] += deposit
                            dense[v, u] += deposit
                    for u in range(n):
                        for v in range(u + 1, n):
                            raw = (one - rho) * tau[u, v] + dense[u, v]
                            value = min(max(raw, low), high) if variant == 2 else raw
                            tau[u, v], tau[v, u] = value, value
                            dense[u, v], dense[v, u] = zero, zero
                            if value != raw:
                                workspace[-1][2] += 2
                if (
                    variant == 2
                    and iteration % branch_period == 0
                    and iteration - found > restart_stagnation
                ):
                    branches = 0
                    for u in range(n):
                        smallest, largest = np.float32(np.inf), np.float32(-np.inf)
                        for j in range(nearest.shape[2]):
                            value = tau[u, nearest[b, u, j]]
                            smallest, largest = min(smallest, value), max(largest, value)
                        cutoff = smallest + branch_lambda * (largest - smallest)
                        for j in range(nearest.shape[2]):
                            if tau[u, nearest[b, u, j]] > cutoff:
                                branches += 1
                    if np.float32(branches) / (np.float32(2) * np.float32(n)) < branch_threshold:
                        tau[:, :] = high
                        for u in range(n):
                            tau[u, u] = zero
                        restart_best = np.float32(np.inf)
                        found = iteration
                        workspace[-1][3] += 1
            output_lengths[p, b] = best
            output_tours[p, b] = best_tour
            output_diagnostics[p, b] = workspace[-1]
    return output_lengths, output_tours, output_diagnostics


def evaluate(programs, problem, config, seed, plan):
    begin = perf_counter()
    set_num_threads(plan.cpu_threads)
    packed = pack_programs(programs)
    initial = initial_parameters(
        problem.distances,
        problem.instance_keys,
        np.uint64(seed),
        config.variant_id,
        np.float32(config.rho),
    )
    setup = perf_counter() - begin
    values = solve_kernel(
        problem.coords,
        problem.distances,
        problem.heuristic,
        problem.log_heuristic,
        problem.nearest,
        problem.full_nn_rank,
        problem.instance_keys,
        np.uint64(seed),
        *packed[:5],
        config.ants,
        config.iterations,
        config.variant_id,
        *map(np.float32, (config.rho, config.q0, config.xi, config.gamma, config.epsilon)),
        config.mmas_period,
        np.float32(config.mmas_p_best),
        config.branch_period,
        np.float32(config.branch_lambda),
        np.float32(config.branch_threshold),
        config.restart_stagnation,
        *initial,
    )
    return EvaluationResult(
        *values,
        {
            "eval_wall_s": perf_counter() - begin,
            "initialization_s": setup,
            "backend": "cpu_existing",
            "executed_tasks": len(programs) * problem.size,
        },
    )
