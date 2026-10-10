"""连续距离 FP32 候选局部搜索：Python 与 Numba 共用可审计的移动顺序。

遵循 ACOTSP 的候选表、first-improvement 和 2-opt DLB 思路，但不冒称整数
ACOTSP 的逐位移植。所有改变路径的判定只有 FP32；没有在线 FP64 回退。
register_jitable 保留普通 Python 函数，不让 CPU-Python 基线偷用 JIT。
"""

import numpy as np
from numba import njit
from numba.extending import register_jitable

from .numeric import counter_uniform


@register_jitable
def accepts(removed, added):
    scale = np.float32(removed + added)
    guard = np.float32(np.float32(1e-7) + np.float32(9.5367431640625e-7) * scale)
    return np.float32(added - removed) < -guard


@register_jitable
def two_candidate(tour, pos, distance, nearest, city, ordinal, limit):
    """先后继边、后前驱边；返回两个切边起点的有序位置。"""
    n = len(pos)
    candidate = int(nearest[city, ordinal % limit])
    i, j = int(pos[city]), int(pos[candidate])
    if ordinal < limit:
        other1, other2 = int(tour[(i + 1) % n]), int(tour[(j + 1) % n])
    else:
        i, j = (i - 1) % n, (j - 1) % n
        other1, other2 = int(tour[i]), int(tour[j])
    if candidate == city or candidate == other1 or other2 == city:
        return -1, -1
    if distance[city, candidate] >= distance[city, other1]:
        return -1, -1
    removed = np.float32(distance[city, other1] + distance[candidate, other2])
    added = np.float32(distance[city, candidate] + distance[other1, other2])
    return (min(i, j), max(i, j)) if accepts(removed, added) else (-1, -1)


@register_jitable
def three_candidate(tour, pos, distance, nearest, city, ordinal, limit):
    """两层候选及四种真正三边重连；序号固定为 (j, k, pattern)。"""
    n = len(pos)
    pattern = ordinal % 4
    outer, inner = ordinal // (4 * limit), (ordinal // 4) % limit
    i = int(pos[city])
    j = int(pos[int(nearest[city, outer])])
    successor = int(tour[(i + 1) % n])
    k = int(pos[int(nearest[successor, inner])])
    if i > j:
        i, j = j, i
    if j > k:
        j, k = k, j
    if i > j:
        i, j = j, i
    if j - i < 2 or k - j < 2 or (i == 0 and k == n - 1):
        return -1, -1, -1, -1
    a, b, c, d, e, f = (
        int(tour[i]),
        int(tour[i + 1]),
        int(tour[j]),
        int(tour[j + 1]),
        int(tour[k]),
        int(tour[(k + 1) % n]),
    )
    removed = np.float32(np.float32(distance[a, b] + distance[c, d]) + distance[e, f])
    if pattern == 0:
        x, y, z = distance[a, c], distance[b, e], distance[d, f]
    elif pattern == 1:
        x, y, z = distance[a, d], distance[e, b], distance[c, f]
    elif pattern == 2:
        x, y, z = distance[a, e], distance[d, b], distance[c, f]
    else:
        x, y, z = distance[a, d], distance[e, c], distance[b, f]
    added = np.float32(np.float32(x + y) + z)
    return (i, j, k, pattern) if accepts(removed, added) else (-1, -1, -1, -1)


@register_jitable
def source_index(index, i, j, k, pattern):
    """重连后位置到旧路径位置的映射；并行写临时缓冲，不原地竞争。"""
    if index <= i or index > k:
        return index
    if pattern == -1:  # 2-opt，k=j。
        return i + 1 + j - index
    offset = index - i - 1
    size1, size2 = j - i, k - j
    if pattern == 0:
        return j - offset if offset < size1 else k - (offset - size1)
    if offset < size2:
        return k - offset if pattern == 2 else j + 1 + offset
    offset -= size2
    return j - offset if pattern == 3 else i + 1 + offset


@register_jitable
def apply_move(tour, pos, dlb, scratch, i, j, k, pattern):
    n = len(pos)
    for edge in (i, j, k):
        dlb[int(tour[edge])] = 0
        dlb[int(tour[(edge + 1) % n])] = 0
    for index in range(i + 1, k + 1):
        scratch[index] = tour[source_index(index, i, j, k, pattern)]
    for index in range(i + 1, k + 1):
        tour[index] = scratch[index]
        pos[int(tour[index])] = index
    tour[n] = tour[0]


@register_jitable
def improve(tour, distance, nearest, order, mode, limit, pos, dlb, scratch, stats):
    """就地下降至候选邻域无改进；没有隐藏的移动数/遍数截断。

    stats: 2-opt移动、3-opt移动、逻辑候选检查数、2-opt扫描遍数。
    3-opt接受后重置全部2-opt DLB，再从2-opt开始；3-opt扫描不使用DLB。
    """
    n = len(pos)
    for index in range(n):
        pos[int(tour[index])] = index
        dlb[index] = 0
    while True:
        changed = True
        while changed:
            stats[3] += 1
            changed = False
            for index in range(n):
                city = int(order[index])
                if dlb[city]:
                    continue
                moved = False
                for ordinal in range(2 * limit):
                    stats[2] += 1
                    i, j = two_candidate(tour, pos, distance, nearest, city, ordinal, limit)
                    if i >= 0:
                        apply_move(tour, pos, dlb, scratch, i, j, j, -1)
                        stats[0] += 1
                        changed = moved = True
                        break
                if not moved:
                    dlb[city] = 1
        if mode == 1:
            break
        moved = False
        for index in range(n):
            city = int(order[index])
            for ordinal in range(4 * limit * limit):
                stats[2] += 1
                i, j, k, pattern = three_candidate(
                    tour, pos, distance, nearest, city, ordinal, limit
                )
                if i >= 0:
                    apply_move(tour, pos, dlb, scratch, i, j, k, pattern)
                    stats[1] += 1
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
        dlb[:] = 0


@njit(cache=True)
def permutation(order, seed, key, iteration, ant):
    """用途4的 Fisher--Yates，不影响构造/初始化随机流。"""
    n = len(order)
    for i in range(n):
        order[i] = i
    for i in range(n - 1):
        offset = min(
            int(counter_uniform(seed, key, iteration, ant, i, 4) * np.float32(n - i)), n - i - 1
        )
        other = i + offset
        order[i], order[other] = order[other], order[i]


compiled_improve = njit(cache=True, nogil=True)(improve)
