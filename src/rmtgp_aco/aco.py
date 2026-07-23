"""AS、ACS 与 MMAS 的 PyTorch 向量化实现。"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from time import perf_counter

import torch
import torch.nn.functional as functional

from .config import (
    ACOConfig,
    ACOVariant,
    PheromoneIntegration,
    TransitionIntegration,
)
from .model import (
    DepositEventBatch,
    ProblemBatch,
    RunDiagnostics,
    RunResult,
    TransitionContext,
)
from .program import TensorProgram


@dataclass(slots=True)
class _SearchState:
    """一次 batch run 中跨 iteration 保留的搜索状态。"""

    global_best_tour: torch.Tensor
    global_best_length: torch.Tensor
    global_best_iteration: torch.Tensor
    restart_best_tour: torch.Tensor
    restart_best_length: torch.Tensor
    stagnation: torch.Tensor
    tau_min: torch.Tensor
    tau_max: torch.Tensor


def _batch_indices(batch: int, *tail: int, device: torch.device) -> torch.Tensor:
    """构造可广播到给定尾部 shape 的 batch index。"""

    shape = (batch,) + (1,) * len(tail)
    return torch.arange(batch, device=device).reshape(shape).expand((batch,) + tail)


def _gather_edges(
    matrix: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
) -> torch.Tensor:
    """从 `[B,n,n]` 矩阵按同 shape 的 rows/columns 读取边值。"""

    batch = matrix.shape[0]
    batch_index = _batch_indices(
        batch,
        *rows.shape[1:],
        device=matrix.device,
    )
    return matrix[batch_index, rows, columns]


def _masked_stdrel(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """在最后一维做 population z-score，再经 tanh 映射到 [-1,1]。"""

    weights = mask.to(values.dtype)
    count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * weights).sum(dim=-1, keepdim=True) / count
    centered = (values - mean) * weights
    variance = (centered * centered).sum(dim=-1, keepdim=True) / count
    result = torch.tanh(centered / (torch.sqrt(variance) + epsilon))
    return torch.where(mask, result, torch.zeros_like(result))


def _normalized_distance_rank(
    distances: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """计算 active feasible set 内从近到远的归一化 rank。"""

    masked = torch.where(mask, distances, torch.full_like(distances, torch.inf))
    order = torch.argsort(masked, dim=-1, stable=True)
    positions = torch.arange(
        distances.shape[-1],
        dtype=torch.int64,
        device=distances.device,
    )
    positions = positions.reshape((1,) * (distances.ndim - 1) + (-1,)).expand_as(order)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, positions)
    count = mask.sum(dim=-1, keepdim=True)
    denominator = (count - 1).clamp_min(1)
    normalized = 1.0 - 2.0 * ranks.to(distances.dtype) / denominator.to(distances.dtype)
    normalized = torch.where(count > 1, normalized, torch.zeros_like(normalized))
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def _base_probability(
    score: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """归一化 score；数值退化时返回 feasible uniform。"""

    masked_score = torch.where(mask, score, torch.zeros_like(score))
    total = masked_score.sum(dim=-1, keepdim=True)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    uniform = mask.to(score.dtype) / count.to(score.dtype)
    valid = total > epsilon
    probability = torch.where(valid, masked_score / total.clamp_min(epsilon), uniform)
    return probability, ~valid.squeeze(-1)


def _build_transition_context(
    problem: ProblemBatch,
    pheromone: torch.Tensor,
    current_city: torch.Tensor,
    candidates: torch.Tensor,
    feasible_mask: torch.Tensor,
    config: ACOConfig,
    *,
    construction_step: int,
    iteration: int,
    stagnation: torch.Tensor,
) -> tuple[TransitionContext, torch.Tensor]:
    """构造 transition tree 的全部主 terminals。"""

    tau = _gather_edges(
        pheromone,
        current_city.unsqueeze(-1).expand_as(candidates),
        candidates,
    )
    eta = _gather_edges(
        problem.heuristic,
        current_city.unsqueeze(-1).expand_as(candidates),
        candidates,
    )
    distance = _gather_edges(
        problem.distances,
        current_city.unsqueeze(-1).expand_as(candidates),
        candidates,
    )
    base_score = torch.pow(tau, config.alpha) * torch.pow(eta, config.beta)
    base_score = torch.where(feasible_mask, base_score, torch.zeros_like(base_score))
    base_probability, uniform_fallback = _base_probability(
        base_score,
        feasible_mask,
        config.epsilon_numeric,
    )

    log_tau = torch.log(tau.clamp_min(config.epsilon_numeric))
    log_eta = torch.log(eta.clamp_min(config.epsilon_numeric))
    rtau = _masked_stdrel(log_tau, feasible_mask)
    reta = _masked_stdrel(log_eta, feasible_mask)
    feasible_count = feasible_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    base_conf = torch.tanh(
        torch.log(base_probability.clamp_min(config.epsilon_numeric))
        + torch.log(feasible_count.to(base_probability.dtype))
    )
    base_conf = torch.where(feasible_mask, base_conf, torch.zeros_like(base_conf))
    dist_rank = _normalized_distance_rank(distance, feasible_mask)

    entropy_value = -(
        base_probability
        * torch.log(base_probability.clamp_min(config.epsilon_numeric))
    ).sum(dim=-1, keepdim=True)
    entropy_denominator = torch.log(feasible_count.to(base_probability.dtype))
    entropy = torch.where(
        feasible_count > 1,
        2.0 * entropy_value / entropy_denominator.clamp_min(config.epsilon_numeric) - 1.0,
        torch.full_like(entropy_value, -1.0),
    ).expand_as(base_score)

    construct_progress = torch.full_like(
        base_score,
        2.0 * construction_step / max(problem.n - 1, 1) - 1.0,
    )
    aco_progress = torch.full_like(
        base_score,
        2.0 * (iteration - 1) / max(config.iterations - 1, 1) - 1.0,
    )
    stagnation_value = (
        2.0
        * torch.clamp(stagnation.to(base_score.dtype) / config.iterations, max=1.0)
        - 1.0
    )
    stagnation_field = stagnation_value[:, None, None].expand_as(base_score)
    feasible_weights = feasible_mask.to(base_score.dtype)
    feasible_denominator = feasible_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_tau = (
        (tau * feasible_weights).sum(dim=-1, keepdim=True)
        / feasible_denominator
    ).expand_as(base_score)
    mean_distance = (
        (distance * feasible_weights).sum(dim=-1, keepdim=True)
        / feasible_denominator
    ).expand_as(base_score)

    context = TransitionContext(
        current_city=current_city,
        candidates=candidates,
        feasible_mask=feasible_mask,
        base_score=base_score,
        base_probability=base_probability,
        terminals={
            "RTau": rtau,
            "REta": reta,
            "BaseConf": base_conf,
            "DistRank": dist_rank,
            "Entropy": entropy,
            "ConstructProg": construct_progress,
            "ACOProg": aco_progress,
            "Stagnation": stagnation_field,
            # Legacy-GP terminals：保留原始量纲以复现 full-replacement 对照。
            "Tau": torch.where(feasible_mask, tau, torch.zeros_like(tau)),
            "Distance": torch.where(
                feasible_mask,
                distance,
                torch.zeros_like(distance),
            ),
            "MeanTau": mean_tau,
            "MeanDistance": mean_distance,
            "Size": torch.full_like(base_score, float(problem.n)),
            "FeasibleCount": feasible_count.to(base_score.dtype).expand_as(base_score),
        },
    )
    return context, uniform_fallback


def _residual_score(
    context: TransitionContext,
    program: TensorProgram | None,
    config: ACOConfig,
) -> torch.Tensor:
    """应用 residual 或 full-replacement；`None` 始终表示原始 ACO。"""

    if program is None:
        return context.base_score
    if config.transition_integration is TransitionIntegration.REPLACEMENT:
        raw = program.evaluate(context.terminals)
        score = (
            functional.softplus(torch.clamp(raw, -20.0, 20.0))
            + config.epsilon_numeric
        )
        return torch.where(
            context.feasible_mask,
            score,
            torch.zeros_like(score),
        )
    if program.is_exact_zero or config.gamma_transition == 0.0:
        return context.base_score
    raw = program.evaluate(context.terminals)
    multiplier = 1.0 + config.gamma_transition * torch.tanh(raw)
    score = context.base_score * multiplier
    return torch.where(context.feasible_mask, score, torch.zeros_like(score))


def _roulette_indices(
    score: torch.Tensor,
    mask: torch.Tensor,
    uniform: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用共享 uniform 执行向量化 inverse-CDF sampling。"""

    probability, used_uniform = _base_probability(score, mask, epsilon)
    cumulative = torch.cumsum(probability, dim=-1)
    # 最后一个有效累积概率在浮点误差下可能略小于 1，比较计数法会自然 clamp。
    indices = (cumulative < uniform.unsqueeze(-1)).sum(dim=-1)
    indices = indices.clamp_max(score.shape[-1] - 1)
    return indices, used_uniform


def _choose_next(
    problem: ProblemBatch,
    pheromone: torch.Tensor,
    current_city: torch.Tensor,
    visited: torch.Tensor,
    config: ACOConfig,
    transition_program: TensorProgram | None,
    *,
    construction_step: int,
    iteration: int,
    stagnation: torch.Tensor,
    greedy_uniform: torch.Tensor,
    roulette_uniform: torch.Tensor,
    diagnostics: RunDiagnostics,
) -> torch.Tensor:
    """为一个或多个蚂蚁选择下一城市。"""

    batch, ants = current_city.shape
    batch_index = _batch_indices(batch, ants, device=problem.device)
    candidates = problem.nn_indices[batch_index, current_city]
    candidate_visited = torch.gather(visited, 2, candidates)
    feasible_mask = ~candidate_visited
    has_candidate = feasible_mask.any(dim=-1)

    context, uniform_fallback = _build_transition_context(
        problem,
        pheromone,
        current_city,
        candidates,
        feasible_mask,
        config,
        construction_step=construction_step,
        iteration=iteration,
        stagnation=stagnation,
    )
    score = _residual_score(context, transition_program, config)
    roulette_index, residual_uniform = _roulette_indices(
        score,
        feasible_mask,
        roulette_uniform,
        config.epsilon_numeric,
    )
    diagnostics.uniform_fallback_count += int(
        (uniform_fallback | residual_uniform).sum().item()
    )
    greedy_score = torch.where(
        feasible_mask,
        score,
        torch.full_like(score, -torch.inf),
    )
    greedy_index = torch.argmax(greedy_score, dim=-1)

    if config.variant is ACOVariant.ACS:
        choose_greedy = greedy_uniform <= config.q0
        selected_index = torch.where(choose_greedy, greedy_index, roulette_index)
    else:
        selected_index = roulette_index
    selected = torch.gather(candidates, 2, selected_index.unsqueeze(-1)).squeeze(-1)

    if not bool(has_candidate.all()):
        diagnostics.candidate_fallback_count += int((~has_candidate).sum().item())
        all_candidates = torch.arange(
            problem.n,
            dtype=torch.int64,
            device=problem.device,
        ).reshape(1, 1, problem.n)
        all_candidates = all_candidates.expand(batch, ants, problem.n)
        full_mask = ~visited
        full_context, _ = _build_transition_context(
            problem,
            pheromone,
            current_city,
            all_candidates,
            full_mask,
            config,
            construction_step=construction_step,
            iteration=iteration,
            stagnation=stagnation,
        )
        full_score = _residual_score(
            full_context,
            transition_program,
            config,
        )
        full_score = torch.where(
            full_mask,
            full_score,
            torch.full_like(full_score, -torch.inf),
        )
        fallback_city = torch.argmax(full_score, dim=-1)
        selected = torch.where(has_candidate, selected, fallback_city)
    return selected


def _apply_acs_local_update(
    pheromone: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    tau0: torch.Tensor,
    xi: float,
) -> None:
    """按无向边 multiplicity 原位执行同步 ACS local update。"""

    batch, n, _ = pheromone.shape
    first = torch.minimum(edge_u, edge_v)
    second = torch.maximum(edge_u, edge_v)
    edge_id = first * n + second
    counts = torch.zeros(
        (batch, n * n),
        dtype=pheromone.dtype,
        device=pheromone.device,
    )
    counts.scatter_add_(1, edge_id, torch.ones_like(edge_id, dtype=pheromone.dtype))

    for batch_index in range(batch):
        active = torch.nonzero(counts[batch_index] > 0, as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            continue
        u = torch.div(active, n, rounding_mode="floor")
        v = active % n
        multiplicity = counts[batch_index, active]
        factor = torch.pow(1.0 - xi, multiplicity)
        updated = (
            factor * pheromone[batch_index, u, v]
            + (1.0 - factor) * tau0[batch_index]
        )
        pheromone[batch_index, u, v] = updated
        pheromone[batch_index, v, u] = updated


def _nearest_neighbour_length(
    problem: ProblemBatch,
    starts: torch.Tensor,
) -> torch.Tensor:
    """从给定起点并行构造 nearest-neighbour tour。"""

    batch, n = problem.batch_size, problem.n
    visited = torch.zeros((batch, n), dtype=torch.bool, device=problem.device)
    tour = torch.empty((batch, n + 1), dtype=torch.int64, device=problem.device)
    current = starts
    tour[:, 0] = current
    visited.scatter_(1, current[:, None], True)
    batch_index = torch.arange(batch, device=problem.device)

    for step in range(1, n):
        row = problem.distances[batch_index, current]
        row = torch.where(visited, torch.full_like(row, torch.inf), row)
        current = torch.argmin(row, dim=-1)
        tour[:, step] = current
        visited.scatter_(1, current[:, None], True)
    tour[:, n] = tour[:, 0]
    u = tour[:, :-1]
    v = tour[:, 1:]
    return problem.distances[
        batch_index[:, None],
        u,
        v,
    ].sum(dim=-1)


def _initial_pheromone(
    problem: ProblemBatch,
    config: ACOConfig,
    nn_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据 ACO 变体计算 `tau0/tau_min/tau_max`。"""

    batch, n = problem.batch_size, problem.n
    if config.variant is ACOVariant.AS:
        tau0 = 1.0 / (config.rho * nn_length)
        tau_max = torch.full_like(tau0, torch.inf)
        tau_min = torch.zeros_like(tau0)
    elif config.variant is ACOVariant.ACS:
        tau0 = 1.0 / (n * nn_length)
        tau_max = torch.full_like(tau0, torch.inf)
        tau_min = torch.zeros_like(tau0)
    else:
        tau_max = 1.0 / (config.rho * nn_length)
        tau_min = tau_max / (2.0 * n)
        tau0 = tau_max

    pheromone = tau0[:, None, None].expand(batch, n, n).clone()
    diagonal = torch.arange(n, device=problem.device)
    pheromone[:, diagonal, diagonal] = 0.0
    return pheromone, tau0, tau_min, tau_max


def _tour_lengths(problem: ProblemBatch, tours: torch.Tensor) -> torch.Tensor:
    """计算 `[B,M,n+1]` tours 的连续长度。"""

    batch, ants, _ = tours.shape
    u = tours[:, :, :-1]
    v = tours[:, :, 1:]
    batch_index = _batch_indices(
        batch,
        ants,
        problem.n,
        device=problem.device,
    )
    return problem.distances[batch_index, u, v].sum(dim=-1)


def _construct_solutions(
    problem: ProblemBatch,
    pheromone: torch.Tensor,
    tau0: torch.Tensor,
    config: ACOConfig,
    transition_program: TensorProgram | None,
    *,
    iteration: int,
    stagnation: torch.Tensor,
    generator: torch.Generator,
    diagnostics: RunDiagnostics,
) -> torch.Tensor:
    """构造一轮 colony tours，支持同步或顺序 ACS。"""

    batch, n = problem.batch_size, problem.n
    ants = config.resolve_ants(n)
    tours = torch.empty(
        (batch, ants, n + 1),
        dtype=torch.int64,
        device=problem.device,
    )
    visited = torch.zeros(
        (batch, ants, n),
        dtype=torch.bool,
        device=problem.device,
    )
    current = torch.randint(
        0,
        n,
        (batch, ants),
        generator=generator,
        device=problem.device,
    )
    tours[:, :, 0] = current
    visited.scatter_(2, current.unsqueeze(-1), True)

    for step in range(1, n):
        greedy_uniform = torch.rand(
            (batch, ants),
            generator=generator,
            dtype=problem.coords.dtype,
            device=problem.device,
        )
        roulette_uniform = torch.rand(
            (batch, ants),
            generator=generator,
            dtype=problem.coords.dtype,
            device=problem.device,
        )

        if config.variant is ACOVariant.ACS and not config.acs_synchronous:
            next_city = torch.empty_like(current)
            for ant_index in range(ants):
                chosen = _choose_next(
                    problem,
                    pheromone,
                    current[:, ant_index : ant_index + 1],
                    visited[:, ant_index : ant_index + 1],
                    config,
                    transition_program,
                    construction_step=step,
                    iteration=iteration,
                    stagnation=stagnation,
                    greedy_uniform=greedy_uniform[:, ant_index : ant_index + 1],
                    roulette_uniform=roulette_uniform[:, ant_index : ant_index + 1],
                    diagnostics=diagnostics,
                )
                next_city[:, ant_index] = chosen[:, 0]
                _apply_acs_local_update(
                    pheromone,
                    current[:, ant_index : ant_index + 1],
                    chosen,
                    tau0,
                    config.xi,
                )
        else:
            next_city = _choose_next(
                problem,
                pheromone,
                current,
                visited,
                config,
                transition_program,
                construction_step=step,
                iteration=iteration,
                stagnation=stagnation,
                greedy_uniform=greedy_uniform,
                roulette_uniform=roulette_uniform,
                diagnostics=diagnostics,
            )
            if config.variant is ACOVariant.ACS:
                _apply_acs_local_update(
                    pheromone,
                    current,
                    next_city,
                    tau0,
                    config.xi,
                )

        tours[:, :, step] = next_city
        visited.scatter_(2, next_city.unsqueeze(-1), True)
        current = next_city

    tours[:, :, n] = tours[:, :, 0]
    if config.variant is ACOVariant.ACS:
        _apply_acs_local_update(
            pheromone,
            tours[:, :, n - 1],
            tours[:, :, 0],
            tau0,
            config.xi,
        )
    return tours


def _update_search_state(
    problem: ProblemBatch,
    tours: torch.Tensor,
    lengths: torch.Tensor,
    state: _SearchState,
    config: ACOConfig,
    *,
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """更新 iteration/restart/global best，并返回 iteration-best。"""

    batch = problem.batch_size
    iteration_length, iteration_index = torch.min(lengths, dim=1)
    batch_index = torch.arange(batch, device=problem.device)
    iteration_tour = tours[batch_index, iteration_index]

    improved_global = iteration_length < state.global_best_length
    state.global_best_length = torch.where(
        improved_global,
        iteration_length,
        state.global_best_length,
    )
    state.global_best_tour = torch.where(
        improved_global[:, None],
        iteration_tour,
        state.global_best_tour,
    )
    state.global_best_iteration = torch.where(
        improved_global,
        torch.full_like(state.global_best_iteration, iteration),
        state.global_best_iteration,
    )
    state.stagnation = torch.where(
        improved_global,
        torch.zeros_like(state.stagnation),
        state.stagnation + 1,
    )

    improved_restart = iteration_length < state.restart_best_length
    state.restart_best_length = torch.where(
        improved_restart,
        iteration_length,
        state.restart_best_length,
    )
    state.restart_best_tour = torch.where(
        improved_restart[:, None],
        iteration_tour,
        state.restart_best_tour,
    )

    if config.variant is ACOVariant.MMAS and bool(improved_global.any()):
        p_x = float(torch.exp(torch.tensor(log(config.mmas_p_best) / problem.n)))
        denominator = p_x * ((config.resolve_candidate_size(problem.n) + 1) // 2)
        factor = (1.0 - p_x) / denominator
        new_tau_max = 1.0 / (config.rho * state.global_best_length)
        new_tau_min = new_tau_max * factor
        state.tau_max = torch.where(improved_global, new_tau_max, state.tau_max)
        state.tau_min = torch.where(improved_global, new_tau_min, state.tau_min)

    return iteration_tour, iteration_length


def _source_tours(
    config: ACOConfig,
    tours: torch.Tensor,
    lengths: torch.Tensor,
    state: _SearchState,
    iteration_tour: torch.Tensor,
    iteration_length: torch.Tensor,
    *,
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 ACO 变体选择全局强化来源。"""

    if config.variant is ACOVariant.AS:
        return tours, lengths
    if config.variant is ACOVariant.ACS:
        return (
            state.global_best_tour[:, None, :],
            state.global_best_length[:, None],
        )
    if iteration % config.mmas_update_period:
        return iteration_tour[:, None, :], iteration_length[:, None]
    return (
        state.restart_best_tour[:, None, :],
        state.restart_best_length[:, None],
    )


def _colony_edge_frequency(
    tours: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """统计本轮 colony 中每条无向边出现于多少条 tour。"""

    batch, ants, _ = tours.shape
    u = tours[:, :, :-1]
    v = tours[:, :, 1:]
    edge_id = torch.minimum(u, v) * n + torch.maximum(u, v)
    counts = torch.zeros(
        (batch, n * n),
        dtype=torch.float64,
        device=tours.device,
    )
    counts.scatter_add_(
        1,
        edge_id.reshape(batch, -1),
        torch.ones(
            (batch, ants * n),
            dtype=counts.dtype,
            device=tours.device,
        ),
    )
    return counts


def _build_deposit_events(
    problem: ProblemBatch,
    pheromone: torch.Tensor,
    source_tours: torch.Tensor,
    source_lengths: torch.Tensor,
    colony_tours: torch.Tensor,
    colony_lengths: torch.Tensor,
    state: _SearchState,
    config: ACOConfig,
    *,
    iteration: int,
    node_log_eta_mean: torch.Tensor,
) -> DepositEventBatch:
    """构造统一的强化 event 和 pheromone terminals。"""

    batch, sources, _ = source_tours.shape
    n = problem.n
    u = source_tours[:, :, :-1]
    v = source_tours[:, :, 1:]
    first = torch.minimum(u, v)
    second = torch.maximum(u, v)
    edge_id = first * n + second
    tau = _gather_edges(pheromone, u, v)
    eta = _gather_edges(problem.heuristic, u, v)
    base_deposit = source_lengths.reciprocal().unsqueeze(-1).expand(batch, sources, n)
    base_budget = n / source_lengths

    edge_tau = _masked_stdrel(
        torch.log(tau.clamp_min(config.epsilon_numeric)),
        torch.ones_like(tau, dtype=torch.bool),
    )
    endpoint_mean_u = torch.gather(
        node_log_eta_mean,
        1,
        u.reshape(batch, -1),
    ).reshape_as(u)
    endpoint_mean_v = torch.gather(
        node_log_eta_mean,
        1,
        v.reshape(batch, -1),
    ).reshape_as(v)
    eta_relative = (
        torch.log(eta.clamp_min(config.epsilon_numeric))
        - 0.5 * (endpoint_mean_u + endpoint_mean_v)
    )
    edge_eta = _masked_stdrel(
        eta_relative,
        torch.ones_like(eta_relative, dtype=torch.bool),
    )

    rank_uv = _gather_edges(problem.full_nn_rank, u, v).to(pheromone.dtype)
    rank_vu = _gather_edges(problem.full_nn_rank, v, u).to(pheromone.dtype)
    rank_denominator = max(n - 2, 1)
    normalized_uv = 1.0 - 2.0 * (rank_uv - 1.0) / rank_denominator
    normalized_vu = 1.0 - 2.0 * (rank_vu - 1.0) / rank_denominator
    nn_rank = 0.5 * (normalized_uv + normalized_vu)

    frequency_flat = _colony_edge_frequency(colony_tours, n).to(pheromone.dtype)
    colony_frequency = torch.gather(
        frequency_flat,
        1,
        edge_id.reshape(batch, -1),
    ).reshape_as(edge_id)
    colony_frequency = 2.0 * colony_frequency / colony_tours.shape[1] - 1.0

    mean_length = colony_lengths.mean(dim=1, keepdim=True)
    std_length = colony_lengths.std(dim=1, keepdim=True, unbiased=False)
    source_quality = torch.tanh(
        (mean_length - source_lengths)
        / (std_length + config.epsilon_numeric)
    ).unsqueeze(-1).expand(batch, sources, n)
    aco_progress = torch.full_like(
        base_deposit,
        2.0 * (iteration - 1) / max(config.iterations - 1, 1) - 1.0,
    )
    stagnation_value = (
        2.0
        * torch.clamp(
            state.stagnation.to(pheromone.dtype) / config.iterations,
            max=1.0,
        )
        - 1.0
    )
    stagnation_field = stagnation_value[:, None, None].expand_as(base_deposit)

    return DepositEventBatch(
        edge_u=u,
        edge_v=v,
        edge_id=edge_id,
        source_length=source_lengths,
        base_deposit=base_deposit,
        base_budget=base_budget,
        terminals={
            "EdgeEta": edge_eta,
            "EdgeTau": edge_tau,
            "NNRank": nn_rank,
            "ColonyFreq": colony_frequency,
            "SourceQuality": source_quality,
            "ACOProg": aco_progress,
            "Stagnation": stagnation_field,
        },
    )


def _residual_deposit(
    events: DepositEventBatch,
    program: TensorProgram | None,
    config: ACOConfig,
) -> torch.Tensor:
    """按预注册消融模式集成 pheromone tree 输出。"""

    if program is None:
        return events.base_deposit
    raw = program.evaluate(events.terminals)
    gamma = config.gamma_pheromone
    mode = config.pheromone_integration
    if mode is PheromoneIntegration.REPLACEMENT:
        unit = events.base_budget.unsqueeze(-1) / events.base_deposit.shape[-1]
        return unit * (
            functional.softplus(torch.clamp(raw, -20.0, 20.0))
            + config.epsilon_numeric
        )
    if program.is_exact_zero or gamma == 0.0:
        return events.base_deposit
    if mode is PheromoneIntegration.ADDITIVE:
        unit = events.base_budget.unsqueeze(-1) / events.base_deposit.shape[-1]
        return torch.clamp_min(
            events.base_deposit + gamma * unit * torch.tanh(raw),
            config.epsilon_numeric,
        )
    multiplier = 1.0 + gamma * torch.tanh(raw)
    unnormalized = events.base_deposit * multiplier
    if mode is PheromoneIntegration.UNNORMALIZED_MULTIPLICATIVE:
        return unnormalized
    denominator = unnormalized.sum(dim=-1, keepdim=True).clamp_min(
        config.epsilon_numeric
    )
    return events.base_budget.unsqueeze(-1) * unnormalized / denominator


def _scatter_undirected_deposit(
    events: DepositEventBatch,
    deposit: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """把 event deposits 聚合为对称 dense matrix。"""

    batch = events.edge_id.shape[0]
    flat = torch.zeros(
        (batch, n * n),
        dtype=deposit.dtype,
        device=deposit.device,
    )
    flat.scatter_add_(
        1,
        events.edge_id.reshape(batch, -1),
        deposit.reshape(batch, -1),
    )
    upper = flat.reshape(batch, n, n)
    return upper + upper.transpose(1, 2)


def _global_update(
    pheromone: torch.Tensor,
    events: DepositEventBatch,
    state: _SearchState,
    config: ACOConfig,
    pheromone_program: TensorProgram | None,
    diagnostics: RunDiagnostics,
) -> torch.Tensor:
    """执行三种 ACO 的全局信息素更新。"""

    n = pheromone.shape[-1]
    deposit = _residual_deposit(
        events,
        pheromone_program,
        config,
    )
    dense = _scatter_undirected_deposit(events, deposit, n)

    if config.variant is ACOVariant.AS:
        updated = (1.0 - config.rho) * pheromone + dense
    elif config.variant is ACOVariant.ACS:
        support = dense > 0
        updated = torch.where(
            support,
            (1.0 - config.rho) * pheromone + config.rho * dense,
            pheromone,
        )
    else:
        raw = (1.0 - config.rho) * pheromone + dense
        lower = state.tau_min[:, None, None]
        upper = state.tau_max[:, None, None]
        clipped = torch.minimum(torch.maximum(raw, lower), upper)
        off_diagonal = ~torch.eye(
            n,
            dtype=torch.bool,
            device=pheromone.device,
        ).unsqueeze(0)
        diagnostics.bound_clip_count += int(
            ((clipped != raw) & off_diagonal).sum().item()
        )
        updated = clipped

    diagonal = torch.arange(n, device=pheromone.device)
    updated[:, diagonal, diagonal] = 0.0
    # 所有 deposit 都按无向 edge ID 聚合；此赋值同时抵御后端舍入不对称。
    updated = 0.5 * (updated + updated.transpose(1, 2))
    return updated


def _initial_search_state(
    problem: ProblemBatch,
    tau_min: torch.Tensor,
    tau_max: torch.Tensor,
) -> _SearchState:
    """建立尚未发现可行解的搜索状态。"""

    batch, n = problem.batch_size, problem.n
    empty_tour = torch.zeros(
        (batch, n + 1),
        dtype=torch.int64,
        device=problem.device,
    )
    infinity = torch.full(
        (batch,),
        torch.inf,
        dtype=problem.coords.dtype,
        device=problem.device,
    )
    return _SearchState(
        global_best_tour=empty_tour.clone(),
        global_best_length=infinity.clone(),
        global_best_iteration=torch.zeros(
            batch,
            dtype=torch.int64,
            device=problem.device,
        ),
        restart_best_tour=empty_tour.clone(),
        restart_best_length=infinity.clone(),
        stagnation=torch.zeros(batch, dtype=torch.int64, device=problem.device),
        tau_min=tau_min,
        tau_max=tau_max,
    )


def solve(
    problem: ProblemBatch,
    config: ACOConfig,
    *,
    transition_program: TensorProgram | None = None,
    pheromone_program: TensorProgram | None = None,
    seed: int = 0,
) -> RunResult:
    """运行一个 batch 的 AS、ACS 或 MMAS。

    传入 `None` program 即为原始 ACO；常数零 program 在数学上恢复同一
    baseline。所有随机数均来自本次调用的独立 generator。
    """

    if problem.device != torch.device(config.device):
        problem = problem.to(config.device)
    if problem.coords.dtype != config.dtype:
        raise ValueError(
            f"ProblemBatch dtype={problem.coords.dtype}，配置要求 {config.dtype}"
        )

    device = problem.device
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    diagnostics = RunDiagnostics()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()

    with torch.inference_mode():
        nn_starts = torch.randint(
            0,
            problem.n,
            (problem.batch_size,),
            generator=generator,
            device=device,
        )
        nn_length = _nearest_neighbour_length(problem, nn_starts)
        pheromone, tau0, tau_min, tau_max = _initial_pheromone(
            problem,
            config,
            nn_length,
        )
        state = _initial_search_state(problem, tau_min, tau_max)

        k = problem.nn_indices.shape[-1]
        batch_index = _batch_indices(
            problem.batch_size,
            problem.n,
            k,
            device=device,
        )
        node_index = (
            torch.arange(problem.n, device=device)
            .reshape(1, problem.n, 1)
            .expand(problem.batch_size, problem.n, k)
        )
        nn_eta = problem.heuristic[batch_index, node_index, problem.nn_indices]
        node_log_eta_mean = torch.log(
            nn_eta.clamp_min(config.epsilon_numeric)
        ).mean(dim=-1)

        anytime: list[torch.Tensor] = []
        for iteration in range(1, config.iterations + 1):
            tours = _construct_solutions(
                problem,
                pheromone,
                tau0,
                config,
                transition_program,
                iteration=iteration,
                stagnation=state.stagnation,
                generator=generator,
                diagnostics=diagnostics,
            )
            lengths = _tour_lengths(problem, tours)
            iteration_tour, iteration_length = _update_search_state(
                problem,
                tours,
                lengths,
                state,
                config,
                iteration=iteration,
            )
            source_tours, source_lengths = _source_tours(
                config,
                tours,
                lengths,
                state,
                iteration_tour,
                iteration_length,
                iteration=iteration,
            )
            events = _build_deposit_events(
                problem,
                pheromone,
                source_tours,
                source_lengths,
                tours,
                lengths,
                state,
                config,
                iteration=iteration,
                node_log_eta_mean=node_log_eta_mean,
            )
            pheromone = _global_update(
                pheromone,
                events,
                state,
                config,
                pheromone_program,
                diagnostics,
            )
            anytime.append(state.global_best_length.clone())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    return RunResult(
        best_tour=state.global_best_tour.clone(),
        best_length=state.global_best_length.clone(),
        best_iteration=state.global_best_iteration.clone(),
        anytime_best=torch.stack(anytime, dim=1),
        wall_time_sec=elapsed,
        constructed_tours=(
            problem.batch_size
            * config.resolve_ants(problem.n)
            * config.iterations
        ),
        diagnostics=diagnostics,
    )
