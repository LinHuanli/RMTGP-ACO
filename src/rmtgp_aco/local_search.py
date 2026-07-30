"""ACOTSP 风格候选表局部搜索的可审计 CPU 参考实现。

正式大规模计算由 CUDA kernel 完成。本模块刻意保持简单，用于：

* 初始化信息素时改进 nearest-neighbour tour；
* 单元测试固定 tour 的 2-opt/3-opt 合法性与单调性；
* 为 CUDA 搜索结果提供小规模 CPU 语义审计。

实现采用连续欧氏距离，而不是 ACOTSP 原程序的整数 TSPLIB 距离。城市扫描
顺序由 counter-based RNG 生成，因此不依赖线程调度，也不复用全局随机状态。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MASK64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class LocalSearchStats:
    """一条 tour 的局部搜索审计计数。"""

    moves: int
    candidate_checks: int
    passes: int
    length_before: float
    length_after: float

    @property
    def improved(self) -> bool:
        return self.length_after < self.length_before

    @property
    def normalized_gain(self) -> float:
        denominator = max(self.length_before, np.finfo(np.float32).tiny)
        return float(
            np.clip(
                (self.length_before - self.length_after) / denominator,
                0.0,
                1.0,
            )
        )


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _counter_uniform(
    seed: int,
    instance_key: int,
    iteration: int,
    ant: int,
    step: int,
    stream_kind: int,
) -> float:
    value = (seed ^ instance_key) & _MASK64
    value ^= ((iteration + 1) * 0xD2B74407B1CE6E93) & _MASK64
    value ^= ((ant + 1) * 0xCA5A826395121157) & _MASK64
    value ^= ((step + 1) * 0x9E3779B185EBCA87) & _MASK64
    value ^= ((stream_kind + 1) * 0x94D049BB133111EB) & _MASK64
    return float(np.float32((_mix64(value) >> 40) / 16777216.0))


def _random_permutation(
    n: int,
    *,
    seed: int,
    instance_key: int,
    iteration: int,
    ant: int,
) -> np.ndarray:
    """Fisher--Yates 排列；随机数坐标与 CUDA 实现一致。"""

    order = np.arange(n, dtype=np.int64)
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
        order[index], order[other] = order[other], order[index]
    return order


def tour_length(tour: np.ndarray, distances: np.ndarray) -> float:
    """以显式闭环或隐式闭环表示计算 tour 长度。"""

    cities = np.asarray(tour, dtype=np.int64)
    if cities.size == distances.shape[0] + 1:
        cities = cities[:-1]
    following = np.roll(cities, -1)
    return float(distances[cities, following].sum(dtype=np.float64))


def _reverse_between_edge_starts(
    tour: np.ndarray,
    position: np.ndarray,
    first_start: int,
    second_start: int,
) -> None:
    """交换两条 tour 边并翻转两切点之间的连续片段。"""

    left = int(position[first_start])
    right = int(position[second_start])
    if left > right:
        left, right = right, left
    left += 1
    tour[left : right + 1] = tour[left : right + 1][::-1]
    for index in range(left, right + 1):
        position[int(tour[index])] = index


def two_opt_first(
    tour: np.ndarray,
    distances: np.ndarray,
    nearest: np.ndarray,
    *,
    seed: int = 0,
    instance_key: int = 0,
    iteration: int = 0,
    ant: int = 0,
    candidate_size: int | None = None,
    use_dlb: bool = True,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, LocalSearchStats]:
    """候选表、DLB、first-improvement 2-opt。

    与 ACOTSP 一样，先尝试以当前城市的后继边为第一切边；失败后再尝试
    前驱边。实际翻转采用等价的连续片段表示，避免原 C 程序针对“较短内外
    片段”的复杂分支。
    """

    n = distances.shape[0]
    result = np.asarray(tour, dtype=np.int64).copy()
    if result.size == n + 1:
        result = result[:-1].copy()
    if result.shape != (n,):
        raise ValueError("tour 必须含 n 个城市，或含 n+1 个城市且末端闭环")
    if not np.array_equal(np.sort(result), np.arange(n)):
        raise ValueError("tour 不是 Hamiltonian permutation")
    limit = min(
        nearest.shape[1],
        n - 1,
        nearest.shape[1] if candidate_size is None else candidate_size,
    )
    position = np.empty(n, dtype=np.int64)
    position[result] = np.arange(n, dtype=np.int64)
    dlb = np.zeros(n, dtype=np.bool_)
    order = _random_permutation(
        n,
        seed=seed,
        instance_key=instance_key,
        iteration=iteration,
        ant=ant,
    )
    before = tour_length(result, distances)
    moves = 0
    checks = 0
    passes = 0
    improved_pass = True
    while improved_pass:
        passes += 1
        improved_pass = False
        for city_value in order:
            city = int(city_value)
            if use_dlb and dlb[city]:
                continue
            city_position = int(position[city])
            successor = int(result[(city_position + 1) % n])
            successor_radius = float(distances[city, successor])
            moved = False
            for candidate_value in nearest[city, :limit]:
                candidate = int(candidate_value)
                checks += 1
                candidate_position = int(position[candidate])
                candidate_successor = int(result[(candidate_position + 1) % n])
                if (
                    candidate == city
                    or candidate == successor
                    or candidate_successor == city
                    or float(distances[city, candidate]) >= successor_radius
                ):
                    continue
                delta = (
                    float(distances[city, candidate])
                    + float(distances[successor, candidate_successor])
                    - successor_radius
                    - float(distances[candidate, candidate_successor])
                )
                if delta < -tolerance:
                    _reverse_between_edge_starts(
                        result,
                        position,
                        city,
                        candidate,
                    )
                    for endpoint in (
                        city,
                        successor,
                        candidate,
                        candidate_successor,
                    ):
                        dlb[endpoint] = False
                    moves += 1
                    moved = True
                    improved_pass = True
                    break
            if moved:
                continue

            predecessor = int(result[(city_position - 1) % n])
            predecessor_radius = float(distances[predecessor, city])
            for candidate_value in nearest[city, :limit]:
                candidate = int(candidate_value)
                checks += 1
                candidate_position = int(position[candidate])
                candidate_predecessor = int(result[(candidate_position - 1) % n])
                if (
                    candidate == city
                    or candidate_predecessor == city
                    or predecessor == candidate
                    or float(distances[city, candidate]) >= predecessor_radius
                ):
                    continue
                delta = (
                    float(distances[city, candidate])
                    + float(distances[predecessor, candidate_predecessor])
                    - predecessor_radius
                    - float(distances[candidate_predecessor, candidate])
                )
                if delta < -tolerance:
                    _reverse_between_edge_starts(
                        result,
                        position,
                        predecessor,
                        candidate_predecessor,
                    )
                    for endpoint in (
                        predecessor,
                        city,
                        candidate_predecessor,
                        candidate,
                    ):
                        dlb[endpoint] = False
                    moves += 1
                    moved = True
                    improved_pass = True
                    break
            if not moved:
                dlb[city] = True

    after = tour_length(result, distances)
    closed = np.concatenate((result, result[:1]))
    return closed, LocalSearchStats(
        moves=moves,
        candidate_checks=checks,
        passes=passes,
        length_before=before,
        length_after=after,
    )


def _three_opt_edges(
    tour: np.ndarray,
    distances: np.ndarray,
    first: int,
    second: int,
    third: int,
) -> tuple[int, float]:
    """返回四种真 3-opt 重连中最先改善的 pattern 和增量。"""

    n = tour.size
    a, b = int(tour[first]), int(tour[(first + 1) % n])
    c, d = int(tour[second]), int(tour[(second + 1) % n])
    e, f = int(tour[third]), int(tour[(third + 1) % n])
    removed = float(distances[a, b] + distances[c, d] + distances[e, f])
    additions = (
        distances[a, c] + distances[b, e] + distances[d, f],
        distances[a, d] + distances[e, b] + distances[c, f],
        distances[a, e] + distances[d, b] + distances[c, f],
        distances[a, d] + distances[e, c] + distances[b, f],
    )
    for pattern, added in enumerate(additions, start=3):
        delta = float(added) - removed
        if delta < -1e-7:
            return pattern, delta
    return 0, 0.0


def _apply_three_opt(
    tour: np.ndarray,
    first: int,
    second: int,
    third: int,
    pattern: int,
) -> np.ndarray:
    """按两个中间片段的方向和次序应用真 3-opt 重连。"""

    prefix = tour[: first + 1]
    segment_one = tour[first + 1 : second + 1]
    segment_two = tour[second + 1 : third + 1]
    suffix = tour[third + 1 :]
    if pattern == 3:
        middle = np.concatenate((segment_one[::-1], segment_two[::-1]))
    elif pattern == 4:
        middle = np.concatenate((segment_two, segment_one))
    elif pattern == 5:
        middle = np.concatenate((segment_two[::-1], segment_one))
    elif pattern == 6:
        middle = np.concatenate((segment_two, segment_one[::-1]))
    else:
        raise ValueError(f"未知 3-opt pattern: {pattern}")
    return np.concatenate((prefix, middle, suffix))


def three_opt_first(
    tour: np.ndarray,
    distances: np.ndarray,
    nearest: np.ndarray,
    **kwargs: int | bool | float | None,
) -> tuple[np.ndarray, LocalSearchStats]:
    """先达到 2-opt 局部最优，再做候选受限的真 3-opt first improvement。"""

    two_tour, two_stats = two_opt_first(
        tour,
        distances,
        nearest,
        **kwargs,
    )
    n = distances.shape[0]
    result = two_tour[:-1].copy()
    limit_value = kwargs.get("candidate_size")
    limit = min(
        nearest.shape[1],
        n - 1,
        nearest.shape[1] if limit_value is None else int(limit_value),
    )
    seed = int(kwargs.get("seed", 0))
    instance_key = int(kwargs.get("instance_key", 0))
    iteration = int(kwargs.get("iteration", 0))
    ant = int(kwargs.get("ant", 0))
    order = _random_permutation(
        n,
        seed=seed,
        instance_key=instance_key,
        iteration=iteration,
        ant=ant,
    )
    checks = two_stats.candidate_checks
    moves = two_stats.moves
    passes = two_stats.passes
    improved_pass = True
    while improved_pass:
        passes += 1
        improved_pass = False
        position = np.empty(n, dtype=np.int64)
        position[result] = np.arange(n, dtype=np.int64)
        for city_value in order:
            city = int(city_value)
            first_position = int(position[city])
            successor = int(result[(first_position + 1) % n])
            for candidate_one in nearest[city, :limit]:
                second_position = int(position[int(candidate_one)])
                for candidate_two in nearest[successor, :limit]:
                    third_position = int(position[int(candidate_two)])
                    checks += 1
                    cuts = sorted(
                        (first_position, second_position, third_position)
                    )
                    if (
                        len(set(cuts)) < 3
                        or cuts[1] == cuts[0] + 1
                        or cuts[2] == cuts[1] + 1
                        or (cuts[0] == 0 and cuts[2] == n - 1)
                    ):
                        continue
                    pattern, _ = _three_opt_edges(
                        result,
                        distances,
                        cuts[0],
                        cuts[1],
                        cuts[2],
                    )
                    if pattern:
                        result = _apply_three_opt(
                            result,
                            cuts[0],
                            cuts[1],
                            cuts[2],
                            pattern,
                        )
                        moves += 1
                        improved_pass = True
                        break
                if improved_pass:
                    break
            if improved_pass:
                break

    after = tour_length(result, distances)
    return np.concatenate((result, result[:1])), LocalSearchStats(
        moves=moves,
        candidate_checks=checks,
        passes=passes,
        length_before=two_stats.length_before,
        length_after=after,
    )
