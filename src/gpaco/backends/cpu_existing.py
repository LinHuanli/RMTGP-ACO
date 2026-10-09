"""从 2b847698 移植的单树 CPU 构造器；不含信息素树或局部搜索。

保留实例内部紧凑候选、标量广播和向量列解释器作为强基线。
浮点常量和整数比例显式 FP32；随机流和稳定方差由共享规范替换。
"""

import numpy as np
from numba import njit

from .numeric import counter_uniform as _counter_uniform
from .numeric import stdrel as _masked_stdrel_values

_CONST, _TERMINAL, _ADD, _SUB, _MUL, _PDIV, _PDIV1, _MIN, _MAX, _ABS, _NEG = range(11)
_TRANSITION_SCALAR_TERMINAL_MASK = np.uint64(sum(1 << x for x in (4, 5, 6, 7, 10, 11, 12, 13)))


@njit(cache=True, inline="always")
def _mask_has(mask: np.uint64, position: int) -> bool:
    return bool(mask & np.uint64(1) << np.uint64(position))


@njit(cache=True, inline="always")
def _sanitize(value: float) -> float:
    """与 PyTorch interpreter 相同的 finite/clip 保护。"""
    if np.isnan(value):
        return np.float32(0.0)
    if value > np.float32(10.0):
        return np.float32(10.0)
    if value < -np.float32(10.0):
        return -np.float32(10.0)
    return value


@njit(cache=True, inline="always")
def _softplus_clipped(value: float) -> float:
    value = min(np.float32(20.0), max(-np.float32(20.0), value))
    return np.log1p(np.exp(value))


@njit(cache=True)
def _evaluate_program(
    opcodes: np.ndarray,
    float_arguments: np.ndarray,
    integer_arguments: np.ndarray,
    terminals: np.ndarray,
    column: int,
    stack: np.ndarray,
) -> float:
    """对一个 candidate/event 标量执行 postfix program。"""
    top = 0
    for instruction_index in range(opcodes.shape[0]):
        opcode = opcodes[instruction_index]
        if opcode == _CONST:
            stack[top] = float_arguments[instruction_index]
            top += 1
            continue
        if opcode == _TERMINAL:
            stack[top] = terminals[integer_arguments[instruction_index], column]
            top += 1
            continue
        if opcode == _ABS or opcode == _NEG:
            value = stack[top - 1]
            if opcode == _ABS:
                stack[top - 1] = _sanitize(abs(value))
            else:
                stack[top - 1] = _sanitize(-value)
            continue
        right = stack[top - 1]
        left = stack[top - 2]
        top -= 1
        if opcode == _ADD:
            result = left + right
        elif opcode == _SUB:
            result = left - right
        elif opcode == _MUL:
            result = left * right
        elif opcode == _PDIV:
            result = left * right / (right * right + np.float32(1e-06))
        elif opcode == _PDIV1:
            result = left / right if abs(right) > np.float32(1e-06) else np.float32(1.0)
        elif opcode == _MIN:
            result = np.minimum(left, right)
        else:
            result = np.maximum(left, right)
        stack[top - 1] = _sanitize(result)
    return _sanitize(stack[0])


@njit(cache=True)
def _evaluate_program_columns(
    opcodes: np.ndarray,
    float_arguments: np.ndarray,
    integer_arguments: np.ndarray,
    terminals: np.ndarray,
    scalar_terminal_mask: np.uint64,
    count: int,
    stack: np.ndarray,
    output: np.ndarray,
) -> None:
    """按列解释 program；标量 terminal 只存一次再广播。"""
    top = 0
    for instruction_index in range(opcodes.shape[0]):
        opcode = opcodes[instruction_index]
        if opcode == _CONST:
            value = float_arguments[instruction_index]
            for column in range(count):
                stack[top, column] = value
            top += 1
            continue
        if opcode == _TERMINAL:
            terminal = integer_arguments[instruction_index]
            if _mask_has(scalar_terminal_mask, terminal):
                value = terminals[terminal, 0]
                for column in range(count):
                    stack[top, column] = value
            else:
                for column in range(count):
                    stack[top, column] = terminals[terminal, column]
            top += 1
            continue
        if opcode == _ABS or opcode == _NEG:
            for column in range(count):
                value = stack[top - 1, column]
                if opcode == _ABS:
                    stack[top - 1, column] = _sanitize(abs(value))
                else:
                    stack[top - 1, column] = _sanitize(-value)
            continue
        if opcode == _ADD:
            for column in range(count):
                result = stack[top - 2, column] + stack[top - 1, column]
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _SUB:
            for column in range(count):
                result = stack[top - 2, column] - stack[top - 1, column]
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _MUL:
            for column in range(count):
                result = stack[top - 2, column] * stack[top - 1, column]
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _PDIV:
            for column in range(count):
                left = stack[top - 2, column]
                right = stack[top - 1, column]
                result = left * right / (right * right + np.float32(1e-06))
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _PDIV1:
            for column in range(count):
                left = stack[top - 2, column]
                right = stack[top - 1, column]
                result = left / right if abs(right) > np.float32(1e-06) else np.float32(1.0)
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _MIN:
            for column in range(count):
                left = stack[top - 2, column]
                right = stack[top - 1, column]
                result = np.minimum(left, right)
                stack[top - 2, column] = _sanitize(result)
        else:
            for column in range(count):
                left = stack[top - 2, column]
                right = stack[top - 1, column]
                result = np.maximum(left, right)
                stack[top - 2, column] = _sanitize(result)
        top -= 1
    for column in range(count):
        output[column] = _sanitize(stack[0, column])


@njit(cache=True)
def _nearest_neighbour_length(distances: np.ndarray, start: int) -> float:
    n = distances.shape[0]
    visited = np.zeros(n, dtype=np.uint8)
    visited[start] = 1
    current = start
    length = np.float32(0.0)
    for _ in range(1, n):
        best_city = -1
        best_distance = np.float32(np.inf)
        for city in range(n):
            if visited[city] == 0 and distances[current, city] < best_distance:
                best_city = city
                best_distance = distances[current, city]
        visited[best_city] = 1
        current = best_city
        length += best_distance
    length += distances[current, start]
    return length


@njit(cache=True)
def _prepare_transition_scores(
    coords: np.ndarray,
    distances: np.ndarray,
    heuristic: np.ndarray,
    log_heuristic: np.ndarray,
    pheromone: np.ndarray,
    current_city: int,
    previous_city: int,
    full_nn_rank: np.ndarray,
    candidates: np.ndarray,
    count: int,
    candidate_fallback: bool,
    alpha: float,
    beta: float,
    epsilon_numeric: float,
    transition_mode: int,
    gamma_transition: float,
    program_active: bool,
    opcodes: np.ndarray,
    float_arguments: np.ndarray,
    integer_arguments: np.ndarray,
    required_mask: np.uint64,
    construction_step: int,
    iteration: int,
    stagnation: int,
    total_iterations: int,
    terminals: np.ndarray,
    scores: np.ndarray,
    scratch: np.ndarray,
    stack: np.ndarray,
) -> bool:
    """构造所需 terminals 和 residual score，返回 baseline uniform 标志。"""
    base_total = np.float32(0.0)
    if alpha == np.float32(1.0) and beta == np.float32(2.0):
        for index in range(count):
            city = candidates[index]
            tau = pheromone[current_city, city]
            eta = heuristic[current_city, city]
            score = (tau * eta) * eta
            scores[index] = score
            base_total += score
    else:
        for index in range(count):
            city = candidates[index]
            tau = pheromone[current_city, city]
            eta = heuristic[current_city, city]
            tau_component = tau if alpha == np.float32(1.0) else tau**alpha
            if beta == np.float32(2.0):
                eta_component = eta * eta
            elif beta == np.float32(1.0):
                eta_component = eta
            else:
                eta_component = eta**beta
            score = tau_component * eta_component
            scores[index] = score
            base_total += score
    uniform_fallback = base_total <= epsilon_numeric
    if not program_active:
        return uniform_fallback
    if _mask_has(required_mask, 2) or _mask_has(required_mask, 4):
        if uniform_fallback:
            probability = np.float32(1.0) / np.float32(count)
            for index in range(count):
                scratch[index] = probability
        else:
            inverse = np.float32(1.0) / base_total
            for index in range(count):
                scratch[index] = scores[index] * inverse
        if _mask_has(required_mask, 2):
            log_count = np.log(np.float32(count))
            for index in range(count):
                terminals[2, index] = np.tanh(
                    np.log(max(scratch[index], epsilon_numeric)) + log_count
                )
        if _mask_has(required_mask, 4):
            if count == 1:
                entropy = -np.float32(1.0)
            else:
                entropy_value = np.float32(0.0)
                for index in range(count):
                    probability = scratch[index]
                    entropy_value -= probability * np.log(max(probability, epsilon_numeric))
                entropy = np.float32(2.0) * entropy_value / max(
                    np.log(np.float32(count)), epsilon_numeric
                ) - np.float32(1.0)
            terminals[4, 0] = entropy
    if _mask_has(required_mask, 0):
        for index in range(count):
            scratch[index] = np.log(
                max(pheromone[current_city, candidates[index]], epsilon_numeric)
            )
        _masked_stdrel_values(scratch, count, terminals[0])
    if _mask_has(required_mask, 1):
        for index in range(count):
            scratch[index] = log_heuristic[current_city, candidates[index]]
        _masked_stdrel_values(scratch, count, terminals[1])
    if _mask_has(required_mask, 3):
        if count == 1:
            terminals[3, 0] = np.float32(0.0)
        elif not candidate_fallback:
            denominator = np.float32(count - 1)
            for index in range(count):
                terminals[3, index] = (
                    np.float32(1.0) - np.float32(2.0) * np.float32(index) / denominator
                )
        else:
            denominator = np.float32(count - 1)
            for index in range(count):
                rank = 0
                value = distances[current_city, candidates[index]]
                for other in range(count):
                    if distances[current_city, candidates[other]] < value or (
                        distances[current_city, candidates[other]] == value and other < index
                    ):
                        rank += 1
                terminals[3, index] = (
                    np.float32(1.0) - np.float32(2.0) * np.float32(rank) / denominator
                )
    if _mask_has(required_mask, 5):
        value = np.float32(2.0) * np.float32(construction_step) / np.float32(
            max(distances.shape[0] - 1, 1)
        ) - np.float32(1.0)
        terminals[5, 0] = value
    if _mask_has(required_mask, 6):
        value = np.float32(2.0) * np.float32(iteration - 1) / np.float32(
            max(total_iterations - 1, 1)
        ) - np.float32(1.0)
        terminals[6, 0] = value
    if _mask_has(required_mask, 7):
        value = np.float32(2.0) * min(
            np.float32(stagnation) / np.float32(total_iterations), np.float32(1.0)
        ) - np.float32(1.0)
        terminals[7, 0] = value
    if _mask_has(required_mask, 8):
        for index in range(count):
            terminals[8, index] = pheromone[current_city, candidates[index]]
    if _mask_has(required_mask, 9):
        for index in range(count):
            terminals[9, index] = distances[current_city, candidates[index]]
    if _mask_has(required_mask, 10):
        mean_tau = np.float32(0.0)
        for index in range(count):
            mean_tau += pheromone[current_city, candidates[index]]
        mean_tau /= np.float32(count)
        terminals[10, 0] = mean_tau
    if _mask_has(required_mask, 11):
        mean_distance = np.float32(0.0)
        for index in range(count):
            mean_distance += distances[current_city, candidates[index]]
        mean_distance /= np.float32(count)
        terminals[11, 0] = mean_distance
    if _mask_has(required_mask, 12):
        terminals[12, 0] = np.float32(distances.shape[0])
    if _mask_has(required_mask, 13):
        terminals[13, 0] = np.float32(count)
    if _mask_has(required_mask, 14):
        denominator = np.float32(max(distances.shape[0] - 2, 1))
        for index in range(count):
            candidate = candidates[index]
            forward = np.float32(full_nn_rank[current_city, candidate] - 1)
            reverse = np.float32(full_nn_rank[candidate, current_city] - 1)
            terminals[14, index] = min(
                max(
                    np.float32(1.0) - (forward + reverse) / (np.float32(2.0) * denominator),
                    np.float32(0.0),
                ),
                np.float32(1.0),
            )
    if _mask_has(required_mask, 15):
        if previous_city < 0:
            for index in range(count):
                terminals[15, index] = np.float32(0.0)
        else:
            incoming_x = coords[current_city, 0] - coords[previous_city, 0]
            incoming_y = coords[current_city, 1] - coords[previous_city, 1]
            incoming_norm = np.sqrt(incoming_x * incoming_x + incoming_y * incoming_y)
            for index in range(count):
                candidate = candidates[index]
                outgoing_x = coords[candidate, 0] - coords[current_city, 0]
                outgoing_y = coords[candidate, 1] - coords[current_city, 1]
                outgoing_norm = np.sqrt(outgoing_x * outgoing_x + outgoing_y * outgoing_y)
                value = (incoming_x * outgoing_x + incoming_y * outgoing_y) / (
                    incoming_norm * outgoing_norm + epsilon_numeric
                )
                terminals[15, index] = min(max(value, -np.float32(1.0)), np.float32(1.0))
    _evaluate_program_columns(
        opcodes,
        float_arguments,
        integer_arguments,
        terminals,
        _TRANSITION_SCALAR_TERMINAL_MASK,
        count,
        stack,
        scratch,
    )
    for index in range(count):
        raw = scratch[index]
        if transition_mode == 1:
            scores[index] = _softplus_clipped(raw) + epsilon_numeric
        else:
            scores[index] *= np.float32(1.0) + gamma_transition * np.tanh(raw)
    return uniform_fallback


@njit(cache=True)
def _choose_city(
    geometry: tuple,
    pheromone: np.ndarray,
    visited: np.ndarray,
    candidate_state: tuple,
    transition_state: tuple,
    random_state: tuple,
    workspace: tuple,
) -> int:
    coords, distances, heuristic, log_heuristic, nearest, full_nn_rank = geometry
    ant, current_city, previous_city = candidate_state
    (
        variant,
        alpha,
        beta,
        q0,
        epsilon_numeric,
        transition_mode,
        gamma_transition,
        program_active,
        opcodes,
        float_arguments,
        integer_arguments,
        required_mask,
    ) = transition_state
    seed, instance_key, construction_step, iteration, stagnation, total_iterations = random_state
    candidates, terminals, scores, scratch, stack, diagnostics = workspace
    count = 0
    for candidate_index in range(nearest.shape[1]):
        city = nearest[current_city, candidate_index]
        if visited[ant, city] == 0:
            candidates[count] = city
            count += 1
    candidate_fallback = count == 0
    if candidate_fallback:
        diagnostics[0] += 1
        diagnostics[1] += 1
        for city in range(distances.shape[0]):
            if visited[ant, city] == 0:
                candidates[count] = city
                count += 1
    base_uniform = _prepare_transition_scores(
        coords,
        distances,
        heuristic,
        log_heuristic,
        pheromone,
        current_city,
        previous_city,
        full_nn_rank,
        candidates,
        count,
        candidate_fallback,
        alpha,
        beta,
        epsilon_numeric,
        transition_mode,
        gamma_transition,
        program_active,
        opcodes,
        float_arguments,
        integer_arguments,
        required_mask,
        construction_step,
        iteration,
        stagnation,
        total_iterations,
        terminals,
        scores,
        scratch,
        stack,
    )
    greedy_index = 0
    greedy_score = scores[0]
    for index in range(1, count):
        if scores[index] > greedy_score:
            greedy_score = scores[index]
            greedy_index = index
    if candidate_fallback:
        return candidates[greedy_index]
    if variant == 1:
        greedy_uniform = _counter_uniform(seed, instance_key, iteration, ant, construction_step, 2)
        if greedy_uniform <= q0:
            if base_uniform:
                diagnostics[1] += 1
            return candidates[greedy_index]
    total = np.float32(0.0)
    for index in range(count):
        total += scores[index]
    residual_uniform = total <= epsilon_numeric
    if base_uniform or residual_uniform:
        diagnostics[1] += 1
    roulette_uniform = _counter_uniform(seed, instance_key, iteration, ant, construction_step, 3)
    roulette_index = count - 1
    if residual_uniform:
        roulette_index = min(int(roulette_uniform * np.float32(count)), count - 1)
    else:
        threshold = roulette_uniform * total
        cumulative = np.float32(0.0)
        for index in range(count):
            cumulative += scores[index]
            if cumulative >= threshold:
                roulette_index = index
                break
    return candidates[roulette_index]


@njit(cache=True)
def _apply_acs_edges(
    pheromone: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    count: int,
    tau0: float,
    local_factors: np.ndarray,
    edge_counts: np.ndarray,
    active_edges: np.ndarray,
) -> None:
    """按 edge multiplicity 同步执行 ACS local update。"""
    n = pheromone.shape[0]
    active_count = 0
    for index in range(count):
        u = edge_u[index]
        v = edge_v[index]
        first = min(u, v)
        second = max(u, v)
        edge_id = first * n + second
        if edge_counts[edge_id] == 0:
            active_edges[active_count] = edge_id
            active_count += 1
        edge_counts[edge_id] += 1
    for index in range(active_count):
        edge_id = active_edges[index]
        u = edge_id // n
        v = edge_id % n
        multiplicity = edge_counts[edge_id]
        factor = local_factors[multiplicity]
        updated = factor * pheromone[u, v] + (np.float32(1.0) - factor) * tau0
        pheromone[u, v] = updated
        pheromone[v, u] = updated
        edge_counts[edge_id] = 0


@njit(cache=True)
def _construct_tours(
    geometry: tuple,
    pheromone: np.ndarray,
    tours: np.ndarray,
    visited: np.ndarray,
    aco_state: tuple,
    transition_program_state: tuple,
    random_state: tuple,
    construction_workspace: tuple,
) -> None:
    coords, distances, heuristic, log_heuristic, nearest, full_nn_rank = geometry
    (
        variant,
        synchronous_acs,
        alpha,
        beta,
        q0,
        local_factors,
        tau0,
        epsilon_numeric,
        transition_mode,
        gamma_transition,
    ) = aco_state
    transition_active, tr_opcodes, tr_float_arguments, tr_integer_arguments, tr_required_mask = (
        transition_program_state
    )
    seed, instance_key, iteration, stagnation, total_iterations = random_state
    (
        candidates,
        tr_terminals,
        scores,
        scratch,
        stack,
        edge_u,
        edge_v,
        edge_counts,
        active_edges,
        diagnostics,
    ) = construction_workspace
    ants, n_plus_one = tours.shape
    n = n_plus_one - 1
    choice_geometry = (coords, distances, heuristic, log_heuristic, nearest, full_nn_rank)
    choice_transition_state = (
        variant,
        alpha,
        beta,
        q0,
        epsilon_numeric,
        transition_mode,
        gamma_transition,
        transition_active,
        tr_opcodes,
        tr_float_arguments,
        tr_integer_arguments,
        tr_required_mask,
    )
    choice_workspace = (candidates, tr_terminals, scores, scratch, stack, diagnostics)
    for ant in range(ants):
        for city in range(n):
            visited[ant, city] = 0
        start_uniform = _counter_uniform(seed, instance_key, iteration, ant, 0, 1)
        start = min(int(start_uniform * np.float32(n)), n - 1)
        tours[ant, 0] = start
        visited[ant, start] = 1
    for step in range(1, n):
        choice_random_state = (seed, instance_key, step, iteration, stagnation, total_iterations)
        for ant in range(ants):
            current = tours[ant, step - 1]
            previous = -1 if step < 2 else tours[ant, step - 2]
            candidate_state = (ant, current, previous)
            chosen = _choose_city(
                choice_geometry,
                pheromone,
                visited,
                candidate_state,
                choice_transition_state,
                choice_random_state,
                choice_workspace,
            )
            tours[ant, step] = chosen
            edge_u[ant] = current
            edge_v[ant] = chosen
            if variant == 1 and (not synchronous_acs):
                _apply_acs_edges(
                    pheromone,
                    edge_u[ant : ant + 1],
                    edge_v[ant : ant + 1],
                    1,
                    tau0,
                    local_factors,
                    edge_counts,
                    active_edges,
                )
        if variant == 1 and synchronous_acs:
            _apply_acs_edges(
                pheromone, edge_u, edge_v, ants, tau0, local_factors, edge_counts, active_edges
            )
        for ant in range(ants):
            visited[ant, tours[ant, step]] = 1
    for ant in range(ants):
        tours[ant, n] = tours[ant, 0]
        edge_u[ant] = tours[ant, n - 1]
        edge_v[ant] = tours[ant, 0]
    if variant == 1:
        _apply_acs_edges(
            pheromone, edge_u, edge_v, ants, tau0, local_factors, edge_counts, active_edges
        )


@njit(cache=True)
def _tour_lengths(distances: np.ndarray, tours: np.ndarray, lengths: np.ndarray) -> None:
    for ant in range(tours.shape[0]):
        length = np.float32(0.0)
        for edge in range(tours.shape[1] - 1):
            length += distances[tours[ant, edge], tours[ant, edge + 1]]
        lengths[ant] = length
