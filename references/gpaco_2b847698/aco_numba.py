"""AS、ACS 与 MMAS 的确定性 Numba CPU 内核。

该模块把 tour construction、GP postfix 解释、信息素更新和 counter-based
随机数生成放在同一个 ``njit`` 边界内。正式训练以 instance 为外层并行单元，
每个线程连续评估同一实例上的行为唯一 GP programs，以复用几何数据和工作区，
同时避免小张量 PyTorch kernel 的调度开销。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
import torch
from numba import njit, prange, set_num_threads

from .config import (
    ACOConfig,
    ACOVariant,
    LocalSearch,
    LSGainSemantics,
    PheromoneIntegration,
    TransitionIntegration,
)
from .local_search import _counter_uniform as _ls_counter_uniform
from .local_search import two_opt_first
from .model import (
    PopulationQualityResult,
    ProblemBatch,
    RunDiagnostics,
    RunResult,
)
from .program import TensorProgram

# Postfix opcode。整数编码既减少 pickle 体积，也让 Numba 避免字符串分支。
_CONST = np.int8(0)
_TERMINAL = np.int8(1)
_ADD = np.int8(2)
_SUB = np.int8(3)
_MUL = np.int8(4)
_PDIV = np.int8(5)
_PDIV1 = np.int8(6)
_MIN = np.int8(7)
_MAX = np.int8(8)
_ABS = np.int8(9)
_NEG = np.int8(10)

_OPCODE_BY_NAME = {
    "ADD": _ADD,
    "SUB": _SUB,
    "MUL": _MUL,
    "PDIV": _PDIV,
    "PDIV1": _PDIV1,
    "MIN": _MIN,
    "MAX": _MAX,
    "ABS": _ABS,
    "NEG": _NEG,
}

_TRANSITION_TERMINAL_INDEX = {
    "RTau": 0,
    "REta": 1,
    "BaseConf": 2,
    "DistRank": 3,
    "Entropy": 4,
    "ConstructProg": 5,
    "ACOProg": 6,
    "Stagnation": 7,
    "Tau": 8,
    "Distance": 9,
    "MeanTau": 10,
    "MeanDistance": 11,
    "Size": 12,
    "FeasibleCount": 13,
    "MutualRank": 14,
    "TurnCos": 15,
}

_PHEROMONE_TERMINAL_INDEX = {
    "EdgeEta": 0,
    "EdgeTau": 1,
    "NNRank": 2,
    "ColonyFreq": 3,
    "SourceQuality": 4,
    "ACOProg": 5,
    "Stagnation": 6,
    "LSGain": 7,
    "Origin": 8,
    "TauHeadroom": 9,
    "PreFreq": 10,
    "PostFreq": 11,
}

# 这些 terminal 对同一次 candidate/edge vector 的全部列取值相同。只存一次，
# interpreter 再广播到工作栈，以内存换掉热循环中的重复写入。
_TRANSITION_SCALAR_TERMINAL_MASK = np.uint64(
    (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 10) | (1 << 11) | (1 << 12) | (1 << 13)
)
_TRANSITION_VECTOR_TERMINAL_MASK = np.uint64(
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 8) | (1 << 9) | (1 << 14) | (1 << 15)
)
_PHEROMONE_SCALAR_TERMINAL_MASK = np.uint64((1 << 4) | (1 << 5) | (1 << 6))
_PHEROMONE_VECTOR_TERMINAL_MASK = np.uint64(
    (1 << 0)
    | (1 << 1)
    | (1 << 2)
    | (1 << 3)
    | (1 << 7)
    | (1 << 8)
    | (1 << 9)
    | (1 << 10)
    | (1 << 11)
)


@dataclass(frozen=True, slots=True)
class EncodedProgram:
    """Numba 可直接接收的紧凑 GP program。"""

    opcodes: np.ndarray
    float_arguments: np.ndarray
    integer_arguments: np.ndarray
    required_mask: np.uint64
    stack_size: int
    active: bool
    exact_zero: bool


@dataclass(frozen=True, slots=True)
class PackedPrograms:
    """一组变长 postfix programs 的定长二维传输表示。"""

    opcodes: np.ndarray
    float_arguments: np.ndarray
    integer_arguments: np.ndarray
    lengths: np.ndarray
    required_masks: np.ndarray
    stack_size: int
    active: np.ndarray
    exact_zero: np.ndarray


def _encode_program(
    program: TensorProgram | None,
    *,
    role: str,
) -> EncodedProgram:
    """把通用 ``TensorProgram`` 编码为固定 dtype 数组。"""

    if program is None:
        return EncodedProgram(
            opcodes=np.empty(0, dtype=np.int8),
            float_arguments=np.empty(0, dtype=np.float64),
            integer_arguments=np.empty(0, dtype=np.int16),
            required_mask=np.uint64(0),
            stack_size=1,
            active=False,
            exact_zero=False,
        )
    terminal_index = (
        _TRANSITION_TERMINAL_INDEX
        if role == "transition"
        else _PHEROMONE_TERMINAL_INDEX
    )
    count = len(program.instructions)
    opcodes = np.empty(count, dtype=np.int8)
    float_arguments = np.zeros(count, dtype=np.float64)
    integer_arguments = np.full(count, -1, dtype=np.int16)
    required_mask = 0
    stack_top = 0
    stack_size = 1
    for index, instruction in enumerate(program.instructions):
        if instruction.opcode == "CONST":
            opcodes[index] = _CONST
            float_arguments[index] = float(instruction.argument)
            stack_top += 1
        elif instruction.opcode == "TERMINAL":
            name = str(instruction.argument)
            try:
                terminal = terminal_index[name]
            except KeyError as exc:
                raise ValueError(
                    f"Numba {role} 后端不支持 terminal {name!r}"
                ) from exc
            opcodes[index] = _TERMINAL
            integer_arguments[index] = terminal
            required_mask |= 1 << terminal
            stack_top += 1
        else:
            try:
                opcodes[index] = _OPCODE_BY_NAME[instruction.opcode]
            except KeyError as exc:
                raise ValueError(
                    f"Numba 后端不支持 GP opcode {instruction.opcode!r}"
                ) from exc
            if instruction.opcode not in {"ABS", "NEG"}:
                stack_top -= 1
        stack_size = max(stack_size, stack_top)
    return EncodedProgram(
        opcodes=opcodes,
        float_arguments=float_arguments,
        integer_arguments=integer_arguments,
        required_mask=np.uint64(required_mask),
        stack_size=stack_size,
        active=True,
        exact_zero=program.is_exact_zero,
    )


def _pack_programs(
    programs: list[TensorProgram | None],
    *,
    role: str,
) -> PackedPrograms:
    """把 population programs 打包为 Numba 可并行索引的二维数组。"""

    if not programs:
        raise ValueError("population programs 不得为空")
    encoded = [_encode_program(program, role=role) for program in programs]
    width = max(1, *(item.opcodes.size for item in encoded))
    count = len(encoded)
    opcodes = np.zeros((count, width), dtype=np.int8)
    floats = np.zeros((count, width), dtype=np.float64)
    integers = np.full((count, width), -1, dtype=np.int16)
    lengths = np.zeros(count, dtype=np.int16)
    masks = np.zeros(count, dtype=np.uint64)
    active = np.zeros(count, dtype=np.uint8)
    exact_zero = np.zeros(count, dtype=np.uint8)
    for index, item in enumerate(encoded):
        length = item.opcodes.size
        lengths[index] = length
        if length:
            opcodes[index, :length] = item.opcodes
            floats[index, :length] = item.float_arguments
            integers[index, :length] = item.integer_arguments
        masks[index] = item.required_mask
        active[index] = item.active
        exact_zero[index] = item.exact_zero
    return PackedPrograms(
        opcodes=opcodes,
        float_arguments=floats,
        integer_arguments=integers,
        lengths=lengths,
        required_masks=masks,
        stack_size=max(item.stack_size for item in encoded),
        active=active,
        exact_zero=exact_zero,
    )


def _semantic_representatives(
    transition: PackedPrograms,
    pheromone: PackedPrograms,
    transition_active: np.ndarray,
    pheromone_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """合并只相差无效/intron 子树的等价 program pairs。"""

    representatives: list[int] = []
    inverse = np.empty(transition.lengths.size, dtype=np.int64)
    key_to_position: dict[tuple[object, object], int] = {}

    def program_key(
        packed: PackedPrograms,
        index: int,
        active: np.ndarray,
    ) -> object:
        if not active[index]:
            return None
        length = int(packed.lengths[index])
        return (
            packed.opcodes[index, :length].tobytes(),
            packed.float_arguments[index, :length].tobytes(),
            packed.integer_arguments[index, :length].tobytes(),
        )

    for index in range(transition.lengths.size):
        key = (
            program_key(transition, index, transition_active),
            program_key(pheromone, index, pheromone_active),
        )
        position = key_to_position.get(key)
        if position is None:
            position = len(representatives)
            key_to_position[key] = position
            representatives.append(index)
        inverse[index] = position
    return np.asarray(representatives, dtype=np.int64), inverse


@njit(cache=True, inline="always")
def _mask_has(mask: np.uint64, position: int) -> bool:
    return bool(mask & (np.uint64(1) << np.uint64(position)))


@njit(cache=True, inline="always")
def _sanitize(value: float) -> float:
    """与 PyTorch interpreter 相同的 finite/clip 保护。"""

    if np.isnan(value):
        return 0.0
    if value > 10.0:
        return 10.0
    if value < -10.0:
        return -10.0
    return value


@njit(cache=True, inline="always")
def _softplus_clipped(value: float) -> float:
    value = min(20.0, max(-20.0, value))
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
            result = left * right / (right * right + 1e-6)
        elif opcode == _PDIV1:
            result = left / right if abs(right) > 1e-6 else 1.0
        elif opcode == _MIN:
            result = min(left, right)
        else:
            result = max(left, right)
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

        # opcode 分支放在列循环外，避免对每个 candidate 重复 dispatch。
        if opcode == _ADD:
            for column in range(count):
                result = (
                    stack[top - 2, column] + stack[top - 1, column]
                )
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _SUB:
            for column in range(count):
                result = (
                    stack[top - 2, column] - stack[top - 1, column]
                )
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _MUL:
            for column in range(count):
                result = (
                    stack[top - 2, column] * stack[top - 1, column]
                )
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _PDIV:
            for column in range(count):
                left = stack[top - 2, column]
                right = stack[top - 1, column]
                result = left * right / (right * right + 1e-6)
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _PDIV1:
            for column in range(count):
                left = stack[top - 2, column]
                right = stack[top - 1, column]
                result = left / right if abs(right) > 1e-6 else 1.0
                stack[top - 2, column] = _sanitize(result)
        elif opcode == _MIN:
            for column in range(count):
                left = stack[top - 2, column]
                right = stack[top - 1, column]
                result = min(left, right)
                stack[top - 2, column] = _sanitize(result)
        else:
            for column in range(count):
                left = stack[top - 2, column]
                right = stack[top - 1, column]
                result = max(left, right)
                stack[top - 2, column] = _sanitize(result)
        top -= 1
    for column in range(count):
        output[column] = _sanitize(stack[0, column])


@njit(cache=True, inline="always")
def _mix64(value: np.uint64) -> np.uint64:
    """SplitMix64 finalizer；所有溢出均为定义良好的 uint64 回绕。"""

    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    value = (value ^ (value >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return value ^ (value >> np.uint64(31))


@njit(cache=True, inline="always")
def _counter_uniform(
    seed: np.uint64,
    instance_key: np.uint64,
    iteration: int,
    ant: int,
    step: int,
    stream_kind: int,
) -> float:
    r"""与执行顺序无关的 \([0,1)\) 随机数。"""

    value = seed ^ instance_key
    value ^= np.uint64(iteration + 1) * np.uint64(0xD2B74407B1CE6E93)
    value ^= np.uint64(ant + 1) * np.uint64(0xCA5A826395121157)
    value ^= np.uint64(step + 1) * np.uint64(0x9E3779B185EBCA87)
    value ^= np.uint64(stream_kind + 1) * np.uint64(0x94D049BB133111EB)
    mixed = _mix64(value)
    return float(mixed >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _nearest_neighbour_length(
    distances: np.ndarray,
    start: int,
) -> float:
    n = distances.shape[0]
    visited = np.zeros(n, dtype=np.uint8)
    visited[start] = 1
    current = start
    length = 0.0
    for _ in range(1, n):
        best_city = -1
        best_distance = np.inf
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
def _prepare_static_geometry(
    heuristic: np.ndarray,
    nearest: np.ndarray,
    epsilon_numeric: float,
) -> tuple[np.ndarray, np.ndarray]:
    """每个 instance 只计算一次静态 log-eta 与节点局部均值。"""

    batch, n, _ = heuristic.shape
    log_heuristic = np.empty_like(heuristic)
    node_log_eta_mean = np.empty((batch, n), dtype=np.float64)
    for batch_index in range(batch):
        for first in range(n):
            for second in range(n):
                log_heuristic[batch_index, first, second] = np.log(
                    max(
                        heuristic[batch_index, first, second],
                        epsilon_numeric,
                    )
                )
            value = 0.0
            for candidate_index in range(nearest.shape[2]):
                candidate = nearest[batch_index, first, candidate_index]
                value += log_heuristic[batch_index, first, candidate]
            node_log_eta_mean[batch_index, first] = (
                value / nearest.shape[2]
            )
    return log_heuristic, node_log_eta_mean


@njit(cache=True)
def _prepare_initial_pheromone_parameters(
    distances: np.ndarray,
    seeds: np.ndarray,
    instance_keys: np.ndarray,
    variant: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把与 GP genotype 无关的 nearest-neighbour 初始化移出 task matrix。"""

    batch = distances.shape[0]
    tau0 = np.empty(batch, dtype=np.float64)
    tau_min = np.empty(batch, dtype=np.float64)
    tau_max = np.empty(batch, dtype=np.float64)
    for batch_index in range(batch):
        n = distances.shape[1]
        nn_uniform = _counter_uniform(
            seeds[batch_index],
            instance_keys[batch_index],
            0,
            0,
            0,
            0,
        )
        nn_start = min(int(nn_uniform * n), n - 1)
        nn_length = _nearest_neighbour_length(
            distances[batch_index],
            nn_start,
        )
        if variant == 0:
            tau0[batch_index] = 1.0 / (rho * nn_length)
            tau_min[batch_index] = 0.0
            tau_max[batch_index] = np.inf
        elif variant == 1:
            tau0[batch_index] = 1.0 / (n * nn_length)
            tau_min[batch_index] = 0.0
            tau_max[batch_index] = np.inf
        else:
            tau_max[batch_index] = 1.0 / (rho * nn_length)
            tau_min[batch_index] = tau_max[batch_index] / (2.0 * n)
            tau0[batch_index] = tau_max[batch_index]
    return tau0, tau_min, tau_max


def _apply_local_search_initial_pheromone(
    distances: np.ndarray,
    nearest: np.ndarray,
    seeds: np.ndarray,
    instance_keys: np.ndarray,
    variant: int,
    rho: float,
    candidate_size: int,
    use_dlb: bool,
    tau0: np.ndarray,
    tau_min: np.ndarray,
    tau_max: np.ndarray,
) -> None:
    """按 ACOTSP 语义以 2-opt 改进 NN 初始化 tour。"""

    batch, n, _ = distances.shape
    for batch_index in range(batch):
        seed = int(seeds[batch_index])
        instance_key = int(instance_keys[batch_index])
        start = min(
            int(
                _ls_counter_uniform(
                    seed,
                    instance_key,
                    0,
                    0,
                    0,
                    0,
                )
                * n
            ),
            n - 1,
        )
        tour = np.empty(n + 1, dtype=np.int64)
        visited = np.zeros(n, dtype=np.bool_)
        current = start
        tour[0] = start
        visited[start] = True
        for phase in range(1, n):
            row = np.where(visited, np.inf, distances[batch_index, current])
            current = int(np.argmin(row))
            tour[phase] = current
            visited[current] = True
        tour[n] = start
        improved, stats = two_opt_first(
            tour,
            distances[batch_index],
            nearest[batch_index],
            seed=seed,
            instance_key=instance_key,
            iteration=0,
            ant=0,
            candidate_size=candidate_size,
            use_dlb=use_dlb,
            tolerance=1e-12,
        )
        del improved
        nn_length = stats.length_after
        if variant == 0:
            tau0[batch_index] = 1.0 / (rho * nn_length)
            tau_min[batch_index] = 0.0
            tau_max[batch_index] = np.inf
        elif variant == 1:
            tau0[batch_index] = 1.0 / (n * nn_length)
            tau_min[batch_index] = 0.0
            tau_max[batch_index] = np.inf
        else:
            maximum = 1.0 / (rho * nn_length)
            tau0[batch_index] = maximum
            tau_min[batch_index] = maximum / (2.0 * n)
            tau_max[batch_index] = maximum


@njit(cache=True)
def _masked_stdrel_values(
    raw_values: np.ndarray,
    count: int,
    output: np.ndarray,
) -> None:
    mean = 0.0
    for index in range(count):
        mean += raw_values[index]
    mean /= count
    variance = 0.0
    for index in range(count):
        centered = raw_values[index] - mean
        variance += centered * centered
    variance /= count
    denominator = np.sqrt(variance) + 1e-8
    for index in range(count):
        output[index] = np.tanh((raw_values[index] - mean) / denominator)


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

    base_total = 0.0
    if alpha == 1.0 and beta == 2.0:
        # ACOTSP 三个正式变体的共同默认值；把参数分支移出最热循环。
        for index in range(count):
            city = candidates[index]
            tau = pheromone[current_city, city]
            eta = heuristic[current_city, city]
            score = tau * (eta * eta)
            scores[index] = score
            base_total += score
    else:
        for index in range(count):
            city = candidates[index]
            tau = pheromone[current_city, city]
            eta = heuristic[current_city, city]
            tau_component = tau if alpha == 1.0 else tau**alpha
            if beta == 2.0:
                eta_component = eta * eta
            elif beta == 1.0:
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
            probability = 1.0 / count
            for index in range(count):
                scratch[index] = probability
        else:
            inverse = 1.0 / base_total
            for index in range(count):
                scratch[index] = scores[index] * inverse
        if _mask_has(required_mask, 2):
            log_count = np.log(float(count))
            for index in range(count):
                terminals[2, index] = np.tanh(
                    np.log(max(scratch[index], epsilon_numeric)) + log_count
                )
        if _mask_has(required_mask, 4):
            if count == 1:
                entropy = -1.0
            else:
                entropy_value = 0.0
                for index in range(count):
                    probability = scratch[index]
                    entropy_value -= probability * np.log(
                        max(probability, epsilon_numeric)
                    )
                entropy = (
                    2.0
                    * entropy_value
                    / max(np.log(float(count)), epsilon_numeric)
                    - 1.0
                )
            terminals[4, 0] = entropy

    if _mask_has(required_mask, 0):
        for index in range(count):
            scratch[index] = np.log(
                max(
                    pheromone[current_city, candidates[index]],
                    epsilon_numeric,
                )
            )
        _masked_stdrel_values(scratch, count, terminals[0])
    if _mask_has(required_mask, 1):
        for index in range(count):
            scratch[index] = log_heuristic[
                current_city,
                candidates[index],
            ]
        _masked_stdrel_values(scratch, count, terminals[1])
    if _mask_has(required_mask, 3):
        if count == 1:
            terminals[3, 0] = 0.0
        elif not candidate_fallback:
            # candidate list 本身已按 (distance, city-id) 稳定排序；过滤
            # visited 后相对顺序不变，因此位置就是精确的 feasible rank。
            denominator = float(count - 1)
            for index in range(count):
                terminals[3, index] = 1.0 - 2.0 * index / denominator
        else:
            denominator = float(count - 1)
            for index in range(count):
                rank = 0
                value = distances[current_city, candidates[index]]
                for other in range(count):
                    if (
                        distances[current_city, candidates[other]] < value
                        or (
                            distances[current_city, candidates[other]] == value
                            and other < index
                        )
                    ):
                        rank += 1
                terminals[3, index] = 1.0 - 2.0 * rank / denominator
    if _mask_has(required_mask, 5):
        value = 2.0 * construction_step / max(distances.shape[0] - 1, 1) - 1.0
        terminals[5, 0] = value
    if _mask_has(required_mask, 6):
        value = 2.0 * (iteration - 1) / max(total_iterations - 1, 1) - 1.0
        terminals[6, 0] = value
    if _mask_has(required_mask, 7):
        value = 2.0 * min(stagnation / total_iterations, 1.0) - 1.0
        terminals[7, 0] = value
    if _mask_has(required_mask, 8):
        for index in range(count):
            terminals[8, index] = pheromone[
                current_city,
                candidates[index],
            ]
    if _mask_has(required_mask, 9):
        for index in range(count):
            terminals[9, index] = distances[
                current_city,
                candidates[index],
            ]
    if _mask_has(required_mask, 10):
        mean_tau = 0.0
        for index in range(count):
            mean_tau += pheromone[current_city, candidates[index]]
        mean_tau /= count
        terminals[10, 0] = mean_tau
    if _mask_has(required_mask, 11):
        mean_distance = 0.0
        for index in range(count):
            mean_distance += distances[current_city, candidates[index]]
        mean_distance /= count
        terminals[11, 0] = mean_distance
    if _mask_has(required_mask, 12):
        terminals[12, 0] = float(distances.shape[0])
    if _mask_has(required_mask, 13):
        terminals[13, 0] = float(count)
    if _mask_has(required_mask, 14):
        denominator = float(max(distances.shape[0] - 2, 1))
        for index in range(count):
            candidate = candidates[index]
            forward = float(full_nn_rank[current_city, candidate] - 1)
            reverse = float(full_nn_rank[candidate, current_city] - 1)
            terminals[14, index] = min(
                max(1.0 - (forward + reverse) / (2.0 * denominator), 0.0),
                1.0,
            )
    if _mask_has(required_mask, 15):
        if previous_city < 0:
            for index in range(count):
                terminals[15, index] = 0.0
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
                terminals[15, index] = min(max(value, -1.0), 1.0)

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
            scores[index] *= (
                1.0 + gamma_transition * np.tanh(raw)
            )
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
    (
        coords,
        distances,
        heuristic,
        log_heuristic,
        nearest,
        full_nn_rank,
    ) = geometry
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
    (
        seed,
        instance_key,
        construction_step,
        iteration,
        stagnation,
        total_iterations,
    ) = random_state
    (
        candidates,
        terminals,
        scores,
        scratch,
        stack,
        diagnostics,
    ) = workspace
    count = 0
    for candidate_index in range(nearest.shape[1]):
        city = nearest[current_city, candidate_index]
        if visited[ant, city] == 0:
            candidates[count] = city
            count += 1

    candidate_fallback = count == 0
    if candidate_fallback:
        diagnostics[0] += 1
        # 与参考实现一致：空 candidate mask 会记一次 uniform fallback。
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

    # candidate-list fallback 在参考实现中固定使用 argmax。
    if candidate_fallback:
        return candidates[greedy_index]

    # Counter-based RNG 没有可变状态，因此 ACS 可先判断 q0 exploitation。
    # 90% 的默认 greedy steps 不再无谓计算 roulette 累积和。
    if variant == 1:
        greedy_uniform = _counter_uniform(
            seed,
            instance_key,
            iteration,
            ant,
            construction_step,
            2,
        )
        if greedy_uniform <= q0:
            if base_uniform:
                diagnostics[1] += 1
            return candidates[greedy_index]

    total = 0.0
    for index in range(count):
        total += scores[index]
    residual_uniform = total <= epsilon_numeric
    if base_uniform or residual_uniform:
        diagnostics[1] += 1

    roulette_uniform = _counter_uniform(
        seed,
        instance_key,
        iteration,
        ant,
        construction_step,
        3,
    )
    roulette_index = count - 1
    if residual_uniform:
        roulette_index = min(int(roulette_uniform * count), count - 1)
    else:
        threshold = roulette_uniform * total
        cumulative = 0.0
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
        updated = factor * pheromone[u, v] + (1.0 - factor) * tau0
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
    (
        coords,
        distances,
        heuristic,
        log_heuristic,
        nearest,
        full_nn_rank,
    ) = geometry
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
    (
        transition_active,
        tr_opcodes,
        tr_float_arguments,
        tr_integer_arguments,
        tr_required_mask,
    ) = transition_program_state
    (
        seed,
        instance_key,
        iteration,
        stagnation,
        total_iterations,
    ) = random_state
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
    choice_geometry = (
        coords,
        distances,
        heuristic,
        log_heuristic,
        nearest,
        full_nn_rank,
    )
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
    choice_workspace = (
        candidates,
        tr_terminals,
        scores,
        scratch,
        stack,
        diagnostics,
    )
    for ant in range(ants):
        for city in range(n):
            visited[ant, city] = 0
        start_uniform = _counter_uniform(
            seed,
            instance_key,
            iteration,
            ant,
            0,
            1,
        )
        start = min(int(start_uniform * n), n - 1)
        tours[ant, 0] = start
        visited[ant, start] = 1

    for step in range(1, n):
        choice_random_state = (
            seed,
            instance_key,
            step,
            iteration,
            stagnation,
            total_iterations,
        )
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
            if variant == 1 and not synchronous_acs:
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
                pheromone,
                edge_u,
                edge_v,
                ants,
                tau0,
                local_factors,
                edge_counts,
                active_edges,
            )
        for ant in range(ants):
            visited[ant, tours[ant, step]] = 1

    for ant in range(ants):
        tours[ant, n] = tours[ant, 0]
        edge_u[ant] = tours[ant, n - 1]
        edge_v[ant] = tours[ant, 0]
    if variant == 1:
        _apply_acs_edges(
            pheromone,
            edge_u,
            edge_v,
            ants,
            tau0,
            local_factors,
            edge_counts,
            active_edges,
        )


@njit(cache=True)
def _tour_lengths(
    distances: np.ndarray,
    tours: np.ndarray,
    lengths: np.ndarray,
) -> None:
    for ant in range(tours.shape[0]):
        length = 0.0
        for edge in range(tours.shape[1] - 1):
            length += distances[tours[ant, edge], tours[ant, edge + 1]]
        lengths[ant] = length


@njit(cache=True, inline="always")
def _reverse_between_edge_starts(
    tour: np.ndarray,
    position: np.ndarray,
    edge_gain: np.ndarray,
    first_start: int,
    second_start: int,
    accepted_gain: float,
    track_edge_gain: bool,
) -> None:
    """以连续片段翻转实现一个对称 TSP 2-opt move。"""

    left = position[first_start]
    right = position[second_start]
    if left > right:
        help = left
        left = right
        right = help
    left += 1
    segment_left = left
    segment_right = right
    while left < right:
        first_city = tour[left]
        second_city = tour[right]
        tour[left] = second_city
        tour[right] = first_city
        position[first_city] = right
        position[second_city] = left
        left += 1
        right -= 1
    if track_edge_gain:
        gain_left = segment_left
        gain_right = segment_right - 1
        while gain_left < gain_right:
            value = edge_gain[gain_left]
            edge_gain[gain_left] = edge_gain[gain_right]
            edge_gain[gain_right] = value
            gain_left += 1
            gain_right -= 1
        edge_gain[segment_left - 1] = accepted_gain
        edge_gain[segment_right] = accepted_gain


@njit(cache=True)
def _two_opt_first_inplace(
    distances: np.ndarray,
    nearest: np.ndarray,
    tour: np.ndarray,
    length_before: float,
    seed: np.uint64,
    instance_key: np.uint64,
    iteration: int,
    ant: int,
    candidate_size: int,
    use_dlb: bool,
    position: np.ndarray,
    order: np.ndarray,
    dlb: np.ndarray,
    edge_gain: np.ndarray,
    track_edge_gain: bool,
) -> tuple[float, float, int, int, int]:
    """ACOTSP 风格的候选表、DLB、first-improvement 2-opt。"""

    n = distances.shape[0]
    for index in range(n):
        city = tour[index]
        position[city] = index
        order[index] = index
        dlb[index] = 0
        if track_edge_gain:
            edge_gain[index] = 0.0
    for index in range(n - 1):
        remaining = n - index
        offset = min(
            int(
                _counter_uniform(
                    seed,
                    instance_key,
                    iteration,
                    ant,
                    index,
                    17,
                )
                * remaining
            ),
            remaining - 1,
        )
        other = index + offset
        help = order[index]
        order[index] = order[other]
        order[other] = help

    moves = 0
    checks = 0
    passes = 0
    improvement = True
    while improvement and moves < n * 100:
        passes += 1
        improvement = False
        for ordinal in range(n):
            city = order[ordinal]
            if use_dlb and dlb[city] != 0:
                continue
            city_position = position[city]
            successor = tour[(city_position + 1) % n]
            successor_radius = distances[city, successor]
            moved = False
            for candidate_index in range(candidate_size):
                candidate = nearest[city, candidate_index]
                checks += 1
                candidate_position = position[candidate]
                candidate_successor = tour[(candidate_position + 1) % n]
                if (
                    candidate == city
                    or candidate == successor
                    or candidate_successor == city
                    or distances[city, candidate] >= successor_radius
                ):
                    continue
                delta = (
                    distances[city, candidate]
                    + distances[successor, candidate_successor]
                    - successor_radius
                    - distances[candidate, candidate_successor]
                )
                if delta < -1e-12:
                    accepted_gain = min(
                        max(
                            -delta / (length_before / n + 1e-300),
                            0.0,
                        ),
                        1.0,
                    )
                    _reverse_between_edge_starts(
                        tour,
                        position,
                        edge_gain,
                        city,
                        candidate,
                        accepted_gain,
                        track_edge_gain,
                    )
                    dlb[city] = 0
                    dlb[successor] = 0
                    dlb[candidate] = 0
                    dlb[candidate_successor] = 0
                    moves += 1
                    moved = True
                    improvement = True
                    break
            if moved:
                continue

            predecessor = tour[(city_position + n - 1) % n]
            predecessor_radius = distances[predecessor, city]
            for candidate_index in range(candidate_size):
                candidate = nearest[city, candidate_index]
                checks += 1
                candidate_position = position[candidate]
                candidate_predecessor = tour[(candidate_position + n - 1) % n]
                if (
                    candidate == city
                    or candidate_predecessor == city
                    or predecessor == candidate
                    or distances[city, candidate] >= predecessor_radius
                ):
                    continue
                delta = (
                    distances[city, candidate]
                    + distances[predecessor, candidate_predecessor]
                    - predecessor_radius
                    - distances[candidate_predecessor, candidate]
                )
                if delta < -1e-12:
                    accepted_gain = min(
                        max(
                            -delta / (length_before / n + 1e-300),
                            0.0,
                        ),
                        1.0,
                    )
                    _reverse_between_edge_starts(
                        tour,
                        position,
                        edge_gain,
                        predecessor,
                        candidate_predecessor,
                        accepted_gain,
                        track_edge_gain,
                    )
                    dlb[predecessor] = 0
                    dlb[city] = 0
                    dlb[candidate_predecessor] = 0
                    dlb[candidate] = 0
                    moves += 1
                    moved = True
                    improvement = True
                    break
            if not moved:
                dlb[city] = 1

    tour[n] = tour[0]
    length_after = 0.0
    for edge in range(n):
        length_after += distances[tour[edge], tour[edge + 1]]
    gain = min(
        max((length_before - length_after) / max(length_before, 1e-300), 0.0),
        1.0,
    )
    return length_after, 2.0 * gain - 1.0, moves, checks, passes


@njit(cache=True)
def _prepare_pheromone_terminals(
    heuristic: np.ndarray,
    log_heuristic: np.ndarray,
    pheromone: np.ndarray,
    full_nn_rank: np.ndarray,
    node_log_eta_mean: np.ndarray,
    source_tour: np.ndarray,
    source_length: float,
    colony_lengths: np.ndarray,
    pre_edge_frequency: np.ndarray,
    post_edge_frequency: np.ndarray,
    required_mask: np.uint64,
    epsilon_numeric: float,
    iteration: int,
    stagnation: int,
    total_iterations: int,
    source_ls_gain: float,
    source_edge_gain: np.ndarray,
    source_origin: np.ndarray,
    ls_gain_semantics: int,
    tau_min: float,
    tau_max: float,
    rho: float,
    terminals: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    scratch: np.ndarray,
) -> None:
    n = source_tour.shape[0] - 1
    for edge in range(n):
        edge_u[edge] = source_tour[edge]
        edge_v[edge] = source_tour[edge + 1]

    if _mask_has(required_mask, 0):
        for edge in range(n):
            u = edge_u[edge]
            v = edge_v[edge]
            scratch[edge] = (
                log_heuristic[u, v]
                - 0.5 * (node_log_eta_mean[u] + node_log_eta_mean[v])
            )
        _masked_stdrel_values(scratch, n, terminals[0])
    if _mask_has(required_mask, 1):
        for edge in range(n):
            scratch[edge] = np.log(
                max(
                    pheromone[edge_u[edge], edge_v[edge]],
                    epsilon_numeric,
                )
            )
        _masked_stdrel_values(scratch, n, terminals[1])
    if _mask_has(required_mask, 2):
        denominator = max(n - 2, 1)
        for edge in range(n):
            u = edge_u[edge]
            v = edge_v[edge]
            rank_uv = full_nn_rank[u, v]
            rank_vu = full_nn_rank[v, u]
            normalized_uv = 1.0 - 2.0 * (rank_uv - 1.0) / denominator
            normalized_vu = 1.0 - 2.0 * (rank_vu - 1.0) / denominator
            terminals[2, edge] = 0.5 * (normalized_uv + normalized_vu)
    if _mask_has(required_mask, 3):
        ants = colony_lengths.shape[0]
        for edge in range(n):
            u = min(edge_u[edge], edge_v[edge])
            v = max(edge_u[edge], edge_v[edge])
            terminals[3, edge] = 2.0 * post_edge_frequency[u * n + v] / ants - 1.0
    if _mask_has(required_mask, 4):
        mean = 0.0
        for ant in range(colony_lengths.shape[0]):
            mean += colony_lengths[ant]
        mean /= colony_lengths.shape[0]
        variance = 0.0
        for ant in range(colony_lengths.shape[0]):
            centered = colony_lengths[ant] - mean
            variance += centered * centered
        variance /= colony_lengths.shape[0]
        quality = np.tanh(
            (mean - source_length)
            / (np.sqrt(variance) + epsilon_numeric)
        )
        terminals[4, 0] = quality
    if _mask_has(required_mask, 5):
        progress = 2.0 * (iteration - 1) / max(total_iterations - 1, 1) - 1.0
        terminals[5, 0] = progress
    if _mask_has(required_mask, 6):
        value = 2.0 * min(stagnation / total_iterations, 1.0) - 1.0
        terminals[6, 0] = value
    if _mask_has(required_mask, 7):
        if ls_gain_semantics == 1:
            for edge in range(n):
                terminals[7, edge] = source_edge_gain[edge]
        else:
            for edge in range(n):
                terminals[7, edge] = source_ls_gain
    if _mask_has(required_mask, 8):
        for edge in range(n):
            terminals[8, edge] = source_origin[edge]
    if _mask_has(required_mask, 9):
        denominator = tau_max - tau_min + epsilon_numeric
        for edge in range(n):
            value = (tau_max - (1.0 - rho) * pheromone[edge_u[edge], edge_v[edge]]) / denominator
            terminals[9, edge] = min(max(value, 0.0), 1.0)
    if _mask_has(required_mask, 10):
        ants = colony_lengths.shape[0]
        for edge in range(n):
            u = min(edge_u[edge], edge_v[edge])
            v = max(edge_u[edge], edge_v[edge])
            terminals[10, edge] = 2.0 * pre_edge_frequency[u * n + v] / ants - 1.0
    if _mask_has(required_mask, 11):
        ants = colony_lengths.shape[0]
        for edge in range(n):
            u = min(edge_u[edge], edge_v[edge])
            v = max(edge_u[edge], edge_v[edge])
            terminals[11, edge] = 2.0 * post_edge_frequency[u * n + v] / ants - 1.0


@njit(cache=True)
def _global_pheromone_update(
    geometry: tuple,
    pheromone: np.ndarray,
    tours: np.ndarray,
    lengths: np.ndarray,
    best_state: tuple,
    local_search_state: tuple,
    aco_state: tuple,
    pheromone_program_state: tuple,
    iteration: int,
    stagnation: int,
    total_iterations: int,
    update_workspace: tuple,
) -> None:
    (
        heuristic,
        log_heuristic,
        nearest,
        full_nn_rank,
        node_log_eta_mean,
    ) = geometry
    (
        iteration_best_index,
        global_best_tour,
        global_best_length,
        restart_best_tour,
        restart_best_length,
    ) = best_state
    (
        ls_gains,
        global_best_ls_gain,
        restart_best_ls_gain,
        edge_gains,
        global_best_edge_gain,
        restart_best_edge_gain,
        origins,
        global_best_origin,
        restart_best_origin,
        pre_tours,
        ls_gain_semantics,
    ) = local_search_state
    (
        tau_min,
        tau_max,
        variant,
        rho,
        mmas_update_period,
        local_search_active,
        restart_iteration,
        restart_found_best,
        epsilon_numeric,
        pheromone_mode,
        gamma_pheromone,
    ) = aco_state
    (
        pheromone_active,
        ph_opcodes,
        ph_float_arguments,
        ph_integer_arguments,
        ph_required_mask,
    ) = pheromone_program_state
    (
        ph_terminals,
        dense,
        edge_frequency,
        frequency_active_edges,
        pre_edge_frequency,
        pre_frequency_active_edges,
        source_edge_u,
        source_edge_v,
        scratch,
        deposits,
        stack,
        diagnostics,
    ) = update_workspace
    n = pheromone.shape[0]

    frequency_active_count = 0
    if _mask_has(ph_required_mask, 3) or _mask_has(ph_required_mask, 11):
        for ant in range(tours.shape[0]):
            for edge in range(n):
                u = min(tours[ant, edge], tours[ant, edge + 1])
                v = max(tours[ant, edge], tours[ant, edge + 1])
                edge_id = u * n + v
                if edge_frequency[edge_id] == 0:
                    frequency_active_edges[frequency_active_count] = edge_id
                    frequency_active_count += 1
                edge_frequency[edge_id] += 1
    pre_frequency_active_count = 0
    if _mask_has(ph_required_mask, 10):
        for ant in range(pre_tours.shape[0]):
            for edge in range(n):
                u = min(pre_tours[ant, edge], pre_tours[ant, edge + 1])
                v = max(pre_tours[ant, edge], pre_tours[ant, edge + 1])
                edge_id = u * n + v
                if pre_edge_frequency[edge_id] == 0:
                    pre_frequency_active_edges[pre_frequency_active_count] = edge_id
                    pre_frequency_active_count += 1
                pre_edge_frequency[edge_id] += 1

    source_count = tours.shape[0] if variant == 0 else 1
    resolved_period = mmas_update_period
    source_kind = 0
    if variant == 2:
        if local_search_active:
            # ACOTSP 在本轮 update 后才为下一轮更新 u_gb。这里按上一轮
            # 已知的 restart age 解析当前周期，避免在 25/75/125/250
            # 边界提前一轮切换。
            restart_age = max(
                iteration - restart_iteration - 1,
                0,
            )
            if restart_age < 25:
                resolved_period = 25
            elif restart_age < 75:
                resolved_period = 5
            elif restart_age < 125:
                resolved_period = 3
            elif restart_age < 250:
                resolved_period = 2
            else:
                resolved_period = 1
        if iteration % resolved_period == 0:
            if (
                local_search_active
                and resolved_period == 1
                and iteration - restart_found_best > 50
            ):
                source_kind = 2
            else:
                source_kind = 1
    for source in range(source_count):
        if variant == 0:
            source_tour = tours[source]
            source_length = lengths[source]
            source_ls_gain = ls_gains[source]
            source_edge_gain = edge_gains[source]
            source_origin = origins[source]
        elif variant == 1:
            source_tour = global_best_tour
            source_length = global_best_length
            source_ls_gain = global_best_ls_gain
            source_edge_gain = global_best_edge_gain
            source_origin = global_best_origin
        elif source_kind == 0:
            source_tour = tours[iteration_best_index]
            source_length = lengths[iteration_best_index]
            source_ls_gain = ls_gains[iteration_best_index]
            source_edge_gain = edge_gains[iteration_best_index]
            source_origin = origins[iteration_best_index]
        elif source_kind == 1:
            source_tour = restart_best_tour
            source_length = restart_best_length
            source_ls_gain = restart_best_ls_gain
            source_edge_gain = restart_best_edge_gain
            source_origin = restart_best_origin
        else:
            source_tour = global_best_tour
            source_length = global_best_length
            source_ls_gain = global_best_ls_gain
            source_edge_gain = global_best_edge_gain
            source_origin = global_best_origin

        _prepare_pheromone_terminals(
            heuristic,
            log_heuristic,
            pheromone,
            full_nn_rank,
            node_log_eta_mean,
            source_tour,
            source_length,
            lengths,
            pre_edge_frequency,
            edge_frequency,
            ph_required_mask,
            epsilon_numeric,
            iteration,
            stagnation,
            total_iterations,
            source_ls_gain,
            source_edge_gain,
            source_origin,
            ls_gain_semantics,
            tau_min,
            tau_max,
            rho,
            ph_terminals,
            source_edge_u,
            source_edge_v,
            scratch,
        )

        base_deposit = 1.0 / source_length
        budget = n / source_length
        if not pheromone_active:
            for edge in range(n):
                deposits[edge] = base_deposit
        else:
            _evaluate_program_columns(
                ph_opcodes,
                ph_float_arguments,
                ph_integer_arguments,
                ph_terminals,
                _PHEROMONE_SCALAR_TERMINAL_MASK,
                n,
                stack,
                scratch,
            )
            for edge in range(n):
                raw = scratch[edge]
                if pheromone_mode == 3:
                    deposits[edge] = base_deposit * (
                        _softplus_clipped(raw) + epsilon_numeric
                    )
                elif pheromone_mode == 2:
                    deposits[edge] = max(
                        base_deposit
                        + gamma_pheromone * base_deposit * np.tanh(raw),
                        epsilon_numeric,
                    )
                else:
                    deposits[edge] = base_deposit * (
                        1.0 + gamma_pheromone * np.tanh(raw)
                    )
            if pheromone_mode == 0:
                total = 0.0
                for edge in range(n):
                    total += deposits[edge]
                scale = budget / max(total, epsilon_numeric)
                for edge in range(n):
                    deposits[edge] *= scale

        if variant == 1:
            # ACS 只蒸发并强化 global-best tour 上的 n 条边，无需构造和扫描
            # n×n dense deposit 矩阵。各 tour edge 相互独立，数值语义不变。
            for edge in range(n):
                u = source_edge_u[edge]
                v = source_edge_v[edge]
                updated = (
                    (1.0 - rho) * pheromone[u, v]
                    + rho * deposits[edge]
                )
                pheromone[u, v] = updated
                pheromone[v, u] = updated
        else:
            for edge in range(n):
                u = source_edge_u[edge]
                v = source_edge_v[edge]
                dense[u, v] += deposits[edge]
                dense[v, u] += deposits[edge]

    if variant == 1:
        for index in range(frequency_active_count):
            edge_frequency[frequency_active_edges[index]] = 0
        for index in range(pre_frequency_active_count):
            pre_edge_frequency[pre_frequency_active_edges[index]] = 0
        return

    if local_search_active:
        # ACOTSP-LS 只蒸发有向 construction candidate-list arcs。deposit
        # 仍作用于来源 tour 的全部边；MMAS 仅在蒸发时检查下界。
        for i in range(n):
            for candidate_index in range(nearest.shape[1]):
                j = nearest[i, candidate_index]
                raw = (1.0 - rho) * pheromone[i, j]
                if variant == 2:
                    updated = max(raw, tau_min)
                    if updated != raw:
                        diagnostics[2] += 1
                    pheromone[i, j] = updated
                else:
                    pheromone[i, j] = raw
        for i in range(n):
            pheromone[i, i] = 0.0
            for j in range(n):
                if i != j:
                    pheromone[i, j] += dense[i, j]
                dense[i, j] = 0.0
    else:
        for i in range(n):
            pheromone[i, i] = 0.0
            for j in range(i + 1, n):
                if variant == 0:
                    updated = (1.0 - rho) * pheromone[i, j] + dense[i, j]
                else:
                    raw = (1.0 - rho) * pheromone[i, j] + dense[i, j]
                    updated = min(max(raw, tau_min), tau_max)
                    if updated != raw:
                        # PyTorch 参考计数器统计两个有向位置。
                        diagnostics[2] += 2
                pheromone[i, j] = updated
                pheromone[j, i] = updated
                dense[i, j] = 0.0
                dense[j, i] = 0.0
    for index in range(frequency_active_count):
        edge_frequency[frequency_active_edges[index]] = 0
    for index in range(pre_frequency_active_count):
        pre_edge_frequency[pre_frequency_active_edges[index]] = 0


@njit(cache=True)
def _allocate_solver_workspace(
    n: int,
    ants: int,
    iterations: int,
    stack_size: int,
    xi: float,
) -> tuple:
    """为一个并行 instance 分配可跨全部 GP programs 复用的工作区。"""

    edge_capacity = max(ants, n)
    local_factors = np.empty(ants + 1, dtype=np.float64)
    for multiplicity in range(ants + 1):
        local_factors[multiplicity] = (1.0 - xi) ** multiplicity
    return (
        np.empty((n, n), dtype=np.float64),  # pheromone
        np.empty(n + 1, dtype=np.int64),  # global best tour
        np.empty(n + 1, dtype=np.int64),  # restart best tour
        np.empty(iterations, dtype=np.float64),  # anytime
        np.empty(8, dtype=np.int64),  # diagnostics（含四个 LS 计数器）
        np.empty((ants, n + 1), dtype=np.int64),  # tours
        np.empty((ants, n), dtype=np.uint8),  # visited
        np.empty(ants, dtype=np.float64),  # lengths
        np.empty(n, dtype=np.int64),  # candidates
        np.empty((16, n), dtype=np.float64),  # transition terminals
        np.empty(n, dtype=np.float64),  # scores
        np.empty(n, dtype=np.float64),  # scratch
        np.empty((stack_size, n), dtype=np.float64),  # GP stack
        np.empty(edge_capacity, dtype=np.int64),  # edge u
        np.empty(edge_capacity, dtype=np.int64),  # edge v
        np.zeros(n * n, dtype=np.int64),  # ACS edge counts
        np.empty(edge_capacity, dtype=np.int64),  # active edges
        local_factors,
        np.empty((12, n), dtype=np.float64),  # pheromone terminals
        np.zeros((n, n), dtype=np.float64),  # dense deposits
        np.zeros(n * n, dtype=np.int64),  # edge frequency
        np.empty(n * n, dtype=np.int64),  # active frequency edges
        np.zeros(n * n, dtype=np.int64),  # construction 前 edge frequency
        np.empty(n * n, dtype=np.int64),  # active pre-frequency edges
        np.empty(n, dtype=np.float64),  # deposits
        np.empty(ants, dtype=np.float64),  # LSGain
        np.empty((ants, n), dtype=np.float64),  # 逐边 last-move LSGain
        np.empty(n, dtype=np.float64),  # global-best 逐边 LSGain
        np.empty(n, dtype=np.float64),  # restart-best 逐边 LSGain
        np.empty(n, dtype=np.int64),  # LS city -> position
        np.empty(n, dtype=np.int64),  # LS random order
        np.empty(n, dtype=np.uint8),  # LS DLB
        np.empty((ants, n + 1), dtype=np.int64),  # construction 前 tours
        np.empty((ants, n), dtype=np.float64),  # post edge provenance
        np.empty(n, dtype=np.float64),  # global-best provenance
        np.empty(n, dtype=np.float64),  # restart-best provenance
        np.empty(ants, dtype=np.float64),  # top-q insertion buffer
    )


@njit(cache=True)
def _top_q_mean(
    values: np.ndarray,
    q: int,
    buffer: np.ndarray,
) -> float:
    """不修改 colony lengths，确定性计算最小 q 个值的平均数。"""

    for index in range(q):
        buffer[index] = np.inf
    for index in range(values.shape[0]):
        value = values[index]
        if value >= buffer[q - 1]:
            continue
        position = q - 1
        while position > 0 and value < buffer[position - 1]:
            buffer[position] = buffer[position - 1]
            position -= 1
        buffer[position] = value
    total = 0.0
    for index in range(q):
        total += buffer[index]
    return total / q


@njit(cache=True, nogil=True)
def _solve_instance_inplace(
    geometry: tuple,
    initial_pheromone: tuple,
    solver_parameters: tuple,
    transition_program: tuple,
    pheromone_program: tuple,
    task_parameters: tuple,
    workspace: tuple,
) -> tuple:
    """在调用方提供的可复用工作区内求解一个 TSP 实例。"""

    (
        coords,
        distances,
        heuristic,
        log_heuristic,
        nearest,
        full_nn_rank,
        node_log_eta_mean,
    ) = geometry
    tau0, tau_min, tau_max = initial_pheromone
    (
        variant,
        ants,
        iterations,
        alpha,
        beta,
        rho,
        q0,
        xi,
        gamma_transition,
        gamma_pheromone,
        transition_mode,
        pheromone_mode,
        synchronous_acs,
        epsilon_numeric,
        mmas_update_period,
        mmas_p_best,
        mmas_branch_check_period,
        mmas_branch_lambda,
        mmas_branch_threshold,
        mmas_restart_stagnation,
        local_search_mode,
        local_search_candidate_size,
        local_search_dlb,
        ls_gain_semantics,
    ) = solver_parameters
    (
        tr_active,
        tr_opcodes,
        tr_float_arguments,
        tr_integer_arguments,
        tr_required_mask,
    ) = transition_program
    (
        ph_active,
        ph_opcodes,
        ph_float_arguments,
        ph_integer_arguments,
        ph_required_mask,
    ) = pheromone_program
    seed, instance_key, basin_top_q, audit_local_search = task_parameters
    n = distances.shape[0]
    (
        pheromone,
        global_best_tour,
        restart_best_tour,
        anytime,
        diagnostics,
        tours,
        visited,
        lengths,
        candidates,
        tr_terminals,
        scores,
        scratch,
        stack,
        edge_u,
        edge_v,
        edge_counts,
        active_edges,
        local_factors,
        ph_terminals,
        dense,
        edge_frequency,
        frequency_active_edges,
        pre_edge_frequency,
        pre_frequency_active_edges,
        deposits,
        ls_gains,
        edge_gains,
        global_best_edge_gain,
        restart_best_edge_gain,
        ls_position,
        ls_order,
        ls_dlb,
        pre_tours,
        origins,
        global_best_origin,
        restart_best_origin,
        top_q_buffer,
    ) = workspace
    for first in range(n):
        for second in range(n):
            pheromone[first, second] = tau0
        pheromone[first, first] = 0.0
    for index in range(8):
        diagnostics[index] = 0

    global_best_length = np.inf
    restart_best_length = np.inf
    global_best_iteration = 0
    stagnation = 0
    restart_found_best = 0
    restart_iteration = 1
    global_best_ls_gain = -1.0
    restart_best_ls_gain = -1.0
    basin_sum = 0.0
    pre_basin_sum = 0.0
    retained_edges = 0
    track_origin = audit_local_search or _mask_has(ph_required_mask, 8)
    track_pre_tour = track_origin or _mask_has(ph_required_mask, 10)
    track_edge_gain = ls_gain_semantics == 1 and _mask_has(ph_required_mask, 7)
    construction_geometry = (
        coords,
        distances,
        heuristic,
        log_heuristic,
        nearest,
        full_nn_rank,
    )
    construction_aco_state = (
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
    )
    transition_program_state = (
        tr_active,
        tr_opcodes,
        tr_float_arguments,
        tr_integer_arguments,
        tr_required_mask,
    )
    construction_workspace = (
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
    )

    for iteration in range(1, iterations + 1):
        construction_random_state = (
            seed,
            instance_key,
            iteration,
            stagnation,
            iterations,
        )
        _construct_tours(
            construction_geometry,
            pheromone,
            tours,
            visited,
            construction_aco_state,
            transition_program_state,
            construction_random_state,
            construction_workspace,
        )
        _tour_lengths(distances, tours, lengths)
        if basin_top_q > 0:
            pre_basin_sum += _top_q_mean(lengths, basin_top_q, top_q_buffer)
        if track_pre_tour:
            for ant in range(ants):
                for city in range(n + 1):
                    pre_tours[ant, city] = tours[ant, city]
        for ant in range(ants):
            ls_gains[ant] = -1.0
        if local_search_mode != 0:
            for ant in range(ants):
                before = lengths[ant]
                (
                    lengths[ant],
                    ls_gains[ant],
                    moves,
                    checks,
                    passes,
                ) = _two_opt_first_inplace(
                    distances,
                    nearest,
                    tours[ant],
                    before,
                    seed,
                    instance_key,
                    iteration,
                    ant,
                    local_search_candidate_size,
                    local_search_dlb,
                    ls_position,
                    ls_order,
                    ls_dlb,
                    edge_gains[ant],
                    track_edge_gain,
                )
                diagnostics[4] += moves
                diagnostics[5] += checks
                diagnostics[7] += passes
                if lengths[ant] < before - 1e-12:
                    diagnostics[6] += 1
        if basin_top_q > 0:
            basin_sum += _top_q_mean(lengths, basin_top_q, top_q_buffer)
        if track_origin:
            for ant in range(ants):
                # 2-opt 已结束，复用 order workspace 建立 construction
                # city->position 映射。这样每条 post edge 只需检查原 tour
                # 中该城市的两个邻居，避免 O(n^2) provenance scan。
                for pre_position in range(n):
                    ls_order[pre_tours[ant, pre_position]] = pre_position
                for edge in range(n):
                    first = tours[ant, edge]
                    second = tours[ant, edge + 1]
                    pre_position = ls_order[first]
                    retained = (
                        pre_tours[
                            ant,
                            (pre_position + 1) % n,
                        ]
                        == second
                        or pre_tours[
                            ant,
                            (pre_position + n - 1) % n,
                        ]
                        == second
                    )
                    origins[ant, edge] = 1.0 if retained else -1.0
                    retained_edges += int(retained)
        else:
            # Origin 未被请求时仍给 source 视图一个确定值，防止未初始化内存
            # 在后续代码重构中被意外读取。
            for ant in range(ants):
                for edge in range(n):
                    origins[ant, edge] = 1.0
        iteration_best_index = 0
        iteration_best_length = lengths[0]
        for ant in range(1, ants):
            if lengths[ant] < iteration_best_length:
                iteration_best_index = ant
                iteration_best_length = lengths[ant]

        improved_global = iteration_best_length < global_best_length
        if improved_global:
            global_best_length = iteration_best_length
            global_best_ls_gain = ls_gains[iteration_best_index]
            global_best_iteration = iteration
            for city in range(n + 1):
                global_best_tour[city] = tours[iteration_best_index, city]
            for edge in range(n):
                global_best_origin[edge] = origins[iteration_best_index, edge]
                if track_edge_gain:
                    global_best_edge_gain[edge] = edge_gains[
                        iteration_best_index,
                        edge,
                    ]
            stagnation = 0
        else:
            stagnation += 1

        if iteration_best_length < restart_best_length:
            restart_best_length = iteration_best_length
            restart_best_ls_gain = ls_gains[iteration_best_index]
            restart_found_best = iteration
            for city in range(n + 1):
                restart_best_tour[city] = tours[iteration_best_index, city]
            for edge in range(n):
                restart_best_origin[edge] = origins[iteration_best_index, edge]
                if track_edge_gain:
                    restart_best_edge_gain[edge] = edge_gains[
                        iteration_best_index,
                        edge,
                    ]

        if variant == 2 and improved_global:
            tau_max = 1.0 / (rho * global_best_length)
            if local_search_mode != 0:
                tau_min = tau_max / (2.0 * n)
            else:
                p_x = np.exp(np.log(mmas_p_best) / n)
                denominator = p_x * ((nearest.shape[1] + 1) // 2)
                factor = (1.0 - p_x) / denominator
                tau_min = tau_max * factor

        resolved_ls_gain_semantics = ls_gain_semantics if track_edge_gain else 0
        geometry_state = (
            heuristic,
            log_heuristic,
            nearest,
            full_nn_rank,
            node_log_eta_mean,
        )
        best_state = (
            iteration_best_index,
            global_best_tour,
            global_best_length,
            restart_best_tour,
            restart_best_length,
        )
        local_search_state = (
            ls_gains,
            global_best_ls_gain,
            restart_best_ls_gain,
            edge_gains,
            global_best_edge_gain,
            restart_best_edge_gain,
            origins,
            global_best_origin,
            restart_best_origin,
            pre_tours,
            resolved_ls_gain_semantics,
        )
        local_search_active = local_search_mode != 0
        aco_state = (
            tau_min,
            tau_max,
            variant,
            rho,
            mmas_update_period,
            local_search_active,
            restart_iteration,
            restart_found_best,
            epsilon_numeric,
            pheromone_mode,
            gamma_pheromone,
        )
        pheromone_program_state = (
            ph_active,
            ph_opcodes,
            ph_float_arguments,
            ph_integer_arguments,
            ph_required_mask,
        )
        update_workspace = (
            ph_terminals,
            dense,
            edge_frequency,
            frequency_active_edges,
            pre_edge_frequency,
            pre_frequency_active_edges,
            edge_u,
            edge_v,
            scratch,
            deposits,
            stack,
            diagnostics,
        )
        _global_pheromone_update(
            geometry_state,
            pheromone,
            tours,
            lengths,
            best_state,
            local_search_state,
            aco_state,
            pheromone_program_state,
            iteration,
            stagnation,
            iterations,
            update_workspace,
        )
        if (
            variant == 2
            and iteration % mmas_branch_check_period == 0
            and iteration - restart_found_best > mmas_restart_stagnation
        ):
            branch_sum = 0.0
            for city in range(n):
                minimum = np.inf
                maximum = -np.inf
                for candidate_index in range(nearest.shape[1]):
                    candidate = nearest[city, candidate_index]
                    value = pheromone[city, candidate]
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                cutoff = minimum + mmas_branch_lambda * (
                    maximum - minimum
                )
                branches = 0
                for candidate_index in range(nearest.shape[1]):
                    candidate = nearest[city, candidate_index]
                    if pheromone[city, candidate] > cutoff:
                        branches += 1
                branch_sum += branches
            branching_factor = branch_sum / (2.0 * n)
            if branching_factor < mmas_branch_threshold:
                for first in range(n):
                    for second in range(n):
                        pheromone[first, second] = tau_max
                    pheromone[first, first] = 0.0
                restart_best_length = np.inf
                restart_found_best = iteration
                restart_iteration = iteration
                restart_best_ls_gain = -1.0
                diagnostics[3] += 1
        anytime[iteration - 1] = global_best_length

    return (
        global_best_tour,
        global_best_length,
        global_best_iteration,
        anytime,
        diagnostics,
        basin_sum / iterations if basin_top_q > 0 else np.nan,
        pre_basin_sum / iterations if basin_top_q > 0 else np.nan,
        (
            retained_edges / (iterations * ants * n)
            if audit_local_search
            else np.nan
        ),
        tours,
        pre_tours,
    )


def _instance_key(instance_id: str) -> np.uint64:
    digest = sha256(instance_id.encode("utf-8")).digest()
    return np.uint64(int.from_bytes(digest[:8], byteorder="little", signed=False))


@njit(cache=True, nogil=True, parallel=True)
def _solve_population_quality_kernel(
    coords: np.ndarray,
    distances: np.ndarray,
    heuristic: np.ndarray,
    log_heuristic: np.ndarray,
    nearest: np.ndarray,
    full_nn_rank: np.ndarray,
    node_log_eta_mean: np.ndarray,
    initial_tau0: np.ndarray,
    initial_tau_min: np.ndarray,
    initial_tau_max: np.ndarray,
    variant: int,
    ants: int,
    iterations: int,
    alpha: float,
    beta: float,
    rho: float,
    q0: float,
    xi: float,
    gamma_transition: float,
    gamma_pheromone: float,
    transition_mode: int,
    pheromone_mode: int,
    synchronous_acs: bool,
    epsilon_numeric: float,
    mmas_update_period: int,
    mmas_p_best: float,
    mmas_branch_check_period: int,
    mmas_branch_lambda: float,
    mmas_branch_threshold: float,
    mmas_restart_stagnation: int,
    local_search_mode: int,
    local_search_candidate_size: int,
    local_search_dlb: bool,
    ls_gain_semantics: int,
    tr_opcodes: np.ndarray,
    tr_float_arguments: np.ndarray,
    tr_integer_arguments: np.ndarray,
    tr_lengths: np.ndarray,
    tr_required_masks: np.ndarray,
    tr_active: np.ndarray,
    ph_opcodes: np.ndarray,
    ph_float_arguments: np.ndarray,
    ph_integer_arguments: np.ndarray,
    ph_lengths: np.ndarray,
    ph_required_masks: np.ndarray,
    ph_active: np.ndarray,
    seeds: np.ndarray,
    instance_keys: np.ndarray,
    stack_size: int,
    basin_top_q: int,
    audit_local_search: bool,
) -> tuple:
    """在一个 native 边界内并行全部 genotype×instance 任务。"""

    population = tr_opcodes.shape[0]
    batch = distances.shape[0]
    best_lengths = np.empty((population, batch), dtype=np.float64)
    best_iterations = np.empty((population, batch), dtype=np.int64)
    best_tours = np.empty(
        (population, batch, distances.shape[1] + 1),
        dtype=np.int64,
    )
    diagnostics = np.empty((population, batch, 8), dtype=np.int64)
    basin_mean_lengths = np.empty((population, batch), dtype=np.float64)
    anytime_mean_lengths = np.empty((population, batch), dtype=np.float64)
    pre_basin_mean_lengths = np.empty((population, batch), dtype=np.float64)
    edge_retention = np.empty((population, batch), dtype=np.float64)
    final_colony_tours = np.empty(
        (population, batch, ants, distances.shape[1] + 1),
        dtype=np.int64,
    )
    final_pre_colony_tours = np.empty_like(final_colony_tours)
    solver_parameters = (
        variant,
        ants,
        iterations,
        alpha,
        beta,
        rho,
        q0,
        xi,
        gamma_transition,
        gamma_pheromone,
        transition_mode,
        pheromone_mode,
        synchronous_acs,
        epsilon_numeric,
        mmas_update_period,
        mmas_p_best,
        mmas_branch_check_period,
        mmas_branch_lambda,
        mmas_branch_threshold,
        mmas_restart_stagnation,
        local_search_mode,
        local_search_candidate_size,
        local_search_dlb,
        ls_gain_semantics,
    )
    for batch_index in prange(batch):
        # 一个线程连续求解同一 instance 的整个人口，复用大工作区与几何 cache。
        workspace = _allocate_solver_workspace(
            distances.shape[1],
            ants,
            iterations,
            stack_size,
            xi,
        )
        for individual in range(population):
            tr_length = int(tr_lengths[individual])
            ph_length = int(ph_lengths[individual])
            geometry = (
                coords[batch_index],
                distances[batch_index],
                heuristic[batch_index],
                log_heuristic[batch_index],
                nearest[batch_index],
                full_nn_rank[batch_index],
                node_log_eta_mean[batch_index],
            )
            initial_pheromone = (
                initial_tau0[batch_index],
                initial_tau_min[batch_index],
                initial_tau_max[batch_index],
            )
            transition_program = (
                bool(tr_active[individual]),
                tr_opcodes[individual, :tr_length],
                tr_float_arguments[individual, :tr_length],
                tr_integer_arguments[individual, :tr_length],
                tr_required_masks[individual],
            )
            pheromone_program = (
                bool(ph_active[individual]),
                ph_opcodes[individual, :ph_length],
                ph_float_arguments[individual, :ph_length],
                ph_integer_arguments[individual, :ph_length],
                ph_required_masks[individual],
            )
            task_parameters = (
                seeds[batch_index],
                instance_keys[batch_index],
                basin_top_q,
                audit_local_search,
            )
            (
                best_tour,
                best_length,
                best_iteration,
                task_anytime,
                task_diagnostics,
                basin_mean_length,
                pre_basin_mean_length,
                task_edge_retention,
                final_colony,
                final_pre_colony,
            ) = _solve_instance_inplace(
                geometry,
                initial_pheromone,
                solver_parameters,
                transition_program,
                pheromone_program,
                task_parameters,
                workspace,
            )
            best_lengths[individual, batch_index] = best_length
            best_iterations[individual, batch_index] = best_iteration
            best_tours[individual, batch_index] = best_tour
            diagnostics[individual, batch_index] = task_diagnostics
            basin_mean_lengths[individual, batch_index] = basin_mean_length
            anytime_mean_lengths[individual, batch_index] = np.mean(
                task_anytime
            )
            pre_basin_mean_lengths[individual, batch_index] = (
                pre_basin_mean_length
            )
            edge_retention[individual, batch_index] = task_edge_retention
            if audit_local_search:
                final_colony_tours[individual, batch_index] = final_colony
                final_pre_colony_tours[individual, batch_index] = (
                    final_pre_colony
                )
    return (
        best_tours,
        best_lengths,
        best_iterations,
        diagnostics,
        basin_mean_lengths,
        anytime_mean_lengths,
        pre_basin_mean_lengths,
        edge_retention,
        final_colony_tours,
        final_pre_colony_tours,
    )


def solve_population_numba(
    problem: ProblemBatch,
    config: ACOConfig,
    programs: list[tuple[TensorProgram | None, TensorProgram | None]],
    *,
    seed: int,
    threads: int = 16,
    basin_top_q: int = 0,
    audit_local_search: bool = False,
) -> PopulationQualityResult:
    """以 16-thread task matrix 评估一组唯一 GP genotypes。"""

    if config.device != "cpu" or problem.device.type != "cpu":
        raise ValueError("Numba population 后端仅支持 CPU ProblemBatch")
    if config.dtype != torch.float64 or problem.coords.dtype != torch.float64:
        raise ValueError("正式 Numba population 后端仅支持 float64")
    if threads < 1:
        raise ValueError("threads 必须为正整数")
    if not programs:
        raise ValueError("program population 不得为空")
    ants = config.resolve_ants(problem.n)
    if basin_top_q < 0 or basin_top_q > ants:
        raise ValueError("basin_top_q 必须位于 [0, ants]")
    if audit_local_search and not config.uses_local_search:
        raise ValueError("local-search signal audit 要求启用局部搜索")
    if config.local_search is LocalSearch.THREE_OPT:
        raise NotImplementedError(
            "Numba 审计后端当前实现 two_opt；three_opt 正式路径使用 CUDA"
        )
    local_search_mode = int(config.local_search is LocalSearch.TWO_OPT)
    local_candidate_size = config.resolve_local_search_candidate_size(problem.n)
    if (
        config.uses_local_search
        and local_candidate_size > problem.nn_indices.shape[-1]
    ):
        raise ValueError("局部搜索候选深度超过 ProblemBatch nearest 表宽度")
    local_candidate_size = min(
        local_candidate_size,
        int(problem.nn_indices.shape[-1]),
    )

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
    variant = {
        ACOVariant.AS: 0,
        ACOVariant.ACS: 1,
        ACOVariant.MMAS: 2,
    }[config.variant]

    tr_active = transition.active.copy()
    if transition_mode == 0:
        tr_active[
            (transition.exact_zero.astype(bool))
            | (config.gamma_transition == 0.0)
            | (
                (transition.required_masks & _TRANSITION_VECTOR_TERMINAL_MASK)
                == 0
            )
        ] = 0
    ph_active = pheromone.active.copy()
    if pheromone_mode != 3:
        ph_active[
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
        tr_active,
        ph_active,
    )
    selected_ph_masks = pheromone.required_masks[representatives]
    needs_two_opt_provenance = bool(
        np.any((selected_ph_masks & np.uint64((1 << 8) | (1 << 10))) != 0)
    )
    needs_edge_gain = config.ls_gain_semantics is LSGainSemantics.EDGE_LAST_MOVE and bool(
        np.any((selected_ph_masks & np.uint64(1 << 7)) != 0)
    )
    if (
        needs_two_opt_provenance or needs_edge_gain
    ) and config.local_search is not LocalSearch.TWO_OPT:
        raise ValueError("Origin/PreFreq/逐边 LSGain terminal 要求 local_search=two_opt")

    coords = np.ascontiguousarray(problem.coords.detach().numpy())
    distances = np.ascontiguousarray(problem.distances.detach().numpy())
    heuristic = np.ascontiguousarray(problem.heuristic.detach().numpy())
    nearest = np.ascontiguousarray(problem.nn_indices.detach().numpy())
    ranks = np.ascontiguousarray(problem.full_nn_rank.detach().numpy())
    batch = problem.batch_size
    seeds = np.full(batch, np.uint64(int(seed) % (2**64)), dtype=np.uint64)
    instance_keys = np.asarray(
        [_instance_key(instance_id) for instance_id in problem.instance_ids],
        dtype=np.uint64,
    )
    log_heuristic, node_log_eta_mean = _prepare_static_geometry(
        heuristic,
        nearest,
        config.epsilon_numeric,
    )
    initial_tau0, initial_tau_min, initial_tau_max = (
        _prepare_initial_pheromone_parameters(
            distances,
            seeds,
            instance_keys,
            variant,
            config.rho,
        )
    )
    if config.uses_local_search:
        _apply_local_search_initial_pheromone(
            distances,
            nearest,
            seeds,
            instance_keys,
            variant,
            config.rho,
            local_candidate_size,
            config.local_search_dlb,
            initial_tau0,
            initial_tau_min,
            initial_tau_max,
        )

    set_num_threads(threads)
    started = perf_counter()
    (
        best_tours,
        best_lengths,
        best_iterations,
        diagnostics,
        basin_mean_lengths,
        anytime_mean_lengths,
        pre_basin_mean_lengths,
        edge_retention,
        final_colony_tours,
        final_pre_colony_tours,
    ) = _solve_population_quality_kernel(
        coords,
        distances,
        heuristic,
        log_heuristic,
        nearest,
        ranks,
        node_log_eta_mean,
        initial_tau0,
        initial_tau_min,
        initial_tau_max,
        variant,
        config.resolve_ants(problem.n),
        config.iterations,
        config.alpha,
        config.beta,
        config.rho,
        config.q0,
        config.xi,
        config.gamma_transition,
        config.gamma_pheromone,
        transition_mode,
        pheromone_mode,
        config.acs_synchronous,
        config.epsilon_numeric,
        config.mmas_update_period,
        config.mmas_p_best,
        config.mmas_branch_check_period,
        config.mmas_branch_lambda,
        config.mmas_branch_threshold,
        config.mmas_restart_stagnation,
        local_search_mode,
        local_candidate_size,
        config.local_search_dlb,
        int(config.ls_gain_semantics is LSGainSemantics.EDGE_LAST_MOVE),
        np.ascontiguousarray(transition.opcodes[representatives]),
        np.ascontiguousarray(transition.float_arguments[representatives]),
        np.ascontiguousarray(transition.integer_arguments[representatives]),
        np.ascontiguousarray(transition.lengths[representatives]),
        np.ascontiguousarray(transition.required_masks[representatives]),
        np.ascontiguousarray(tr_active[representatives]),
        np.ascontiguousarray(pheromone.opcodes[representatives]),
        np.ascontiguousarray(pheromone.float_arguments[representatives]),
        np.ascontiguousarray(pheromone.integer_arguments[representatives]),
        np.ascontiguousarray(pheromone.lengths[representatives]),
        np.ascontiguousarray(pheromone.required_masks[representatives]),
        np.ascontiguousarray(ph_active[representatives]),
        seeds,
        instance_keys,
        max(transition.stack_size, pheromone.stack_size),
        basin_top_q,
        audit_local_search,
    )
    if representatives.size != len(programs):
        best_tours = best_tours[inverse]
        best_lengths = best_lengths[inverse]
        best_iterations = best_iterations[inverse]
        diagnostics = diagnostics[inverse]
        basin_mean_lengths = basin_mean_lengths[inverse]
        anytime_mean_lengths = anytime_mean_lengths[inverse]
        pre_basin_mean_lengths = pre_basin_mean_lengths[inverse]
        edge_retention = edge_retention[inverse]
        final_colony_tours = final_colony_tours[inverse]
        final_pre_colony_tours = final_pre_colony_tours[inverse]
    elapsed = perf_counter() - started
    return PopulationQualityResult(
        best_tour=torch.from_numpy(best_tours),
        best_length=torch.from_numpy(best_lengths),
        best_iteration=torch.from_numpy(best_iterations),
        diagnostics=torch.from_numpy(diagnostics.sum(axis=1)),
        wall_time_sec=elapsed,
        constructed_tours=(
            representatives.size
            * batch
            * config.resolve_ants(problem.n)
            * config.iterations
        ),
        basin_mean_length=(
            torch.from_numpy(basin_mean_lengths)
            if basin_top_q > 0
            else None
        ),
        anytime_mean_length=torch.from_numpy(anytime_mean_lengths),
        pre_basin_mean_length=(
            torch.from_numpy(pre_basin_mean_lengths)
            if audit_local_search
            else None
        ),
        edge_retention=(
            torch.from_numpy(edge_retention)
            if audit_local_search
            else None
        ),
        final_colony_tour=(
            torch.from_numpy(final_colony_tours)
            if audit_local_search
            else None
        ),
        final_pre_colony_tour=(
            torch.from_numpy(final_pre_colony_tours)
            if audit_local_search
            else None
        ),
        backend_metrics={
            "basin_top_q": basin_top_q,
            "audit_local_search": int(audit_local_search),
        },
    )


def solve_numba(
    problem: ProblemBatch,
    config: ACOConfig,
    *,
    transition_program: TensorProgram | None = None,
    pheromone_program: TensorProgram | None = None,
    seed: int = 0,
) -> RunResult:
    """使用确定性 Numba CPU 内核求解一个同规模 batch。"""

    if config.device != "cpu" or problem.device.type != "cpu":
        raise ValueError("Numba ACO 后端仅支持 CPU ProblemBatch")
    if config.dtype != torch.float64 or problem.coords.dtype != torch.float64:
        raise ValueError("正式 Numba ACO 后端仅支持 float64")
    if config.local_search is LocalSearch.THREE_OPT:
        raise NotImplementedError(
            "Numba 审计后端当前实现 two_opt；three_opt 正式路径使用 CUDA"
        )
    local_search_mode = int(config.local_search is LocalSearch.TWO_OPT)
    local_candidate_size = config.resolve_local_search_candidate_size(problem.n)
    if (
        config.uses_local_search
        and local_candidate_size > problem.nn_indices.shape[-1]
    ):
        raise ValueError("局部搜索候选深度超过 ProblemBatch nearest 表宽度")
    local_candidate_size = min(
        local_candidate_size,
        int(problem.nn_indices.shape[-1]),
    )

    transition = _encode_program(transition_program, role="transition")
    pheromone = _encode_program(pheromone_program, role="pheromone")
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
    variant = {
        ACOVariant.AS: 0,
        ACOVariant.ACS: 1,
        ACOVariant.MMAS: 2,
    }[config.variant]

    transition_active = transition.active
    if (
        transition_mode == 0
        and (
            transition.exact_zero
            or config.gamma_transition == 0.0
            or (
                transition.required_mask
                & _TRANSITION_VECTOR_TERMINAL_MASK
            )
            == 0
        )
    ):
        transition_active = False
    pheromone_active = pheromone.active
    if (
        pheromone_mode != 3
        and (
            pheromone.exact_zero
            or config.gamma_pheromone == 0.0
            or (
                pheromone_mode == 0
                and (
                    pheromone.required_mask
                    & _PHEROMONE_VECTOR_TERMINAL_MASK
                )
                == 0
            )
        )
    ):
        pheromone_active = False
    needs_two_opt_provenance = bool(pheromone.required_mask & np.uint64((1 << 8) | (1 << 10)))
    needs_edge_gain = bool(
        config.ls_gain_semantics is LSGainSemantics.EDGE_LAST_MOVE
        and pheromone.required_mask & np.uint64(1 << 7)
    )
    if (
        needs_two_opt_provenance or needs_edge_gain
    ) and config.local_search is not LocalSearch.TWO_OPT:
        raise ValueError("Origin/PreFreq/逐边 LSGain terminal 要求 local_search=two_opt")

    coords = np.ascontiguousarray(problem.coords.detach().numpy())
    distances = np.ascontiguousarray(problem.distances.detach().numpy())
    heuristic = np.ascontiguousarray(problem.heuristic.detach().numpy())
    nearest = np.ascontiguousarray(problem.nn_indices.detach().numpy())
    ranks = np.ascontiguousarray(problem.full_nn_rank.detach().numpy())
    batch = problem.batch_size
    n = problem.n
    best_tours = np.empty((batch, n + 1), dtype=np.int64)
    best_lengths = np.empty(batch, dtype=np.float64)
    best_iterations = np.empty(batch, dtype=np.int64)
    anytime = np.empty((batch, config.iterations), dtype=np.float64)
    diagnostics = np.zeros(8, dtype=np.int64)
    seed_value = np.uint64(int(seed) % (2**64))
    seeds = np.full(batch, seed_value, dtype=np.uint64)
    instance_keys = np.asarray(
        [_instance_key(instance_id) for instance_id in problem.instance_ids],
        dtype=np.uint64,
    )
    log_heuristic, node_log_eta_mean = _prepare_static_geometry(
        heuristic,
        nearest,
        config.epsilon_numeric,
    )
    initial_tau0, initial_tau_min, initial_tau_max = (
        _prepare_initial_pheromone_parameters(
            distances,
            seeds,
            instance_keys,
            variant,
            config.rho,
        )
    )
    if config.uses_local_search:
        _apply_local_search_initial_pheromone(
            distances,
            nearest,
            seeds,
            instance_keys,
            variant,
            config.rho,
            local_candidate_size,
            config.local_search_dlb,
            initial_tau0,
            initial_tau_min,
            initial_tau_max,
        )
    workspace = _allocate_solver_workspace(
        n,
        config.resolve_ants(n),
        config.iterations,
        max(transition.stack_size, pheromone.stack_size),
        config.xi,
    )
    solver_parameters = (
        variant,
        config.resolve_ants(n),
        config.iterations,
        config.alpha,
        config.beta,
        config.rho,
        config.q0,
        config.xi,
        config.gamma_transition,
        config.gamma_pheromone,
        transition_mode,
        pheromone_mode,
        config.acs_synchronous,
        config.epsilon_numeric,
        config.mmas_update_period,
        config.mmas_p_best,
        config.mmas_branch_check_period,
        config.mmas_branch_lambda,
        config.mmas_branch_threshold,
        config.mmas_restart_stagnation,
        local_search_mode,
        local_candidate_size,
        config.local_search_dlb,
        int(config.ls_gain_semantics is LSGainSemantics.EDGE_LAST_MOVE),
    )
    transition_program_state = (
        transition_active,
        transition.opcodes,
        transition.float_arguments,
        transition.integer_arguments,
        transition.required_mask,
    )
    pheromone_program_state = (
        pheromone_active,
        pheromone.opcodes,
        pheromone.float_arguments,
        pheromone.integer_arguments,
        pheromone.required_mask,
    )

    started = perf_counter()
    for batch_index in range(batch):
        geometry = (
            coords[batch_index],
            distances[batch_index],
            heuristic[batch_index],
            log_heuristic[batch_index],
            nearest[batch_index],
            ranks[batch_index],
            node_log_eta_mean[batch_index],
        )
        initial_pheromone = (
            initial_tau0[batch_index],
            initial_tau_min[batch_index],
            initial_tau_max[batch_index],
        )
        task_parameters = (
            seed_value,
            instance_keys[batch_index],
            0,
            False,
        )
        (
            best_tour,
            best_length,
            best_iteration,
            instance_anytime,
            instance_diagnostics,
            _,
            _,
            _,
            _,
            _,
        ) = _solve_instance_inplace(
            geometry,
            initial_pheromone,
            solver_parameters,
            transition_program_state,
            pheromone_program_state,
            task_parameters,
            workspace,
        )
        best_tours[batch_index] = best_tour
        best_lengths[batch_index] = best_length
        best_iterations[batch_index] = best_iteration
        anytime[batch_index] = instance_anytime
        diagnostics += instance_diagnostics
    elapsed = perf_counter() - started

    return RunResult(
        best_tour=torch.from_numpy(best_tours),
        best_length=torch.from_numpy(best_lengths),
        best_iteration=torch.from_numpy(best_iterations),
        anytime_best=torch.from_numpy(anytime),
        wall_time_sec=elapsed,
        constructed_tours=(
            batch * config.resolve_ants(n) * config.iterations
        ),
        diagnostics=RunDiagnostics(
            candidate_fallback_count=int(diagnostics[0]),
            uniform_fallback_count=int(diagnostics[1]),
            bound_clip_count=int(diagnostics[2]),
            mmas_restart_count=int(diagnostics[3]),
            local_search_move_count=int(diagnostics[4]),
            local_search_candidate_check_count=int(diagnostics[5]),
            local_search_improved_tour_count=int(diagnostics[6]),
            local_search_pass_count=int(diagnostics[7]),
        ),
    )
