// 候选表局部搜索：每条 tour 一个 warp，每个 block 同时处理多个 tour。
//
// 算法语义参考 ACOTSP 1.03 的随机城市顺序、候选表、DLB 和
// first-improvement 设计。这里使用连续 FP32 欧氏距离和 counter RNG，
// 并以等价的连续片段重排实现 move，以适应 GPU SIMT 执行。

#include <cuda_runtime.h>

namespace rmtgp_ls {

constexpr unsigned int FULL_WARP = 0xffffffffu;
constexpr float LS_TOLERANCE = 1.0e-7f;

__device__ __forceinline__ float edge_distance(
    const float* distances,
    int n,
    int first,
    int second
) {
    return distances[static_cast<size_t>(first) * n + second];
}

__device__ __forceinline__ void sort_three(
    int& first,
    int& second,
    int& third
) {
    if (first > second) {
        const int value = first;
        first = second;
        second = value;
    }
    if (second > third) {
        const int value = second;
        second = third;
        third = value;
    }
    if (first > second) {
        const int value = first;
        first = second;
        second = value;
    }
}

__device__ __forceinline__ void reverse_between_edges(
    uint16_t* tour,
    uint16_t* position,
    int n,
    int first_start,
    int second_start,
    int lane
) {
    int left = static_cast<int>(position[first_start]);
    int right = static_cast<int>(position[second_start]);
    if (left > right) {
        const int value = left;
        left = right;
        right = value;
    }
    ++left;
    const int pairs = (right - left + 1) / 2;
    for (int offset = lane; offset < pairs; offset += 32) {
        const int first_index = left + offset;
        const int second_index = right - offset;
        const uint16_t first_city = tour[first_index];
        const uint16_t second_city = tour[second_index];
        tour[first_index] = second_city;
        tour[second_index] = first_city;
        position[first_city] = static_cast<uint16_t>(second_index);
        position[second_city] = static_cast<uint16_t>(first_index);
    }
    // 奇数长度片段的中心城市位置不变，无需更新 position。
    __syncwarp();
}

__device__ __forceinline__ float exact_length(
    const uint16_t* tour,
    const float* distances,
    int n,
    int lane
) {
    float partial = 0.0f;
    for (int edge = lane; edge < n; edge += 32) {
        partial += edge_distance(
            distances,
            n,
            static_cast<int>(tour[edge]),
            static_cast<int>(tour[(edge + 1) % n])
        );
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(FULL_WARP, partial, offset);
    }
    return __shfl_sync(FULL_WARP, partial, 0);
}

__device__ __forceinline__ int three_opt_pattern(
    const uint16_t* tour,
    const float* distances,
    int n,
    int first,
    int second,
    int third
) {
    const int a = static_cast<int>(tour[first]);
    const int b = static_cast<int>(tour[(first + 1) % n]);
    const int c = static_cast<int>(tour[second]);
    const int d = static_cast<int>(tour[(second + 1) % n]);
    const int e = static_cast<int>(tour[third]);
    const int f = static_cast<int>(tour[(third + 1) % n]);
    const float removed = edge_distance(distances, n, a, b)
        + edge_distance(distances, n, c, d)
        + edge_distance(distances, n, e, f);
    const float additions[4] = {
        edge_distance(distances, n, a, c)
            + edge_distance(distances, n, b, e)
            + edge_distance(distances, n, d, f),
        edge_distance(distances, n, a, d)
            + edge_distance(distances, n, e, b)
            + edge_distance(distances, n, c, f),
        edge_distance(distances, n, a, e)
            + edge_distance(distances, n, d, b)
            + edge_distance(distances, n, c, f),
        edge_distance(distances, n, a, d)
            + edge_distance(distances, n, e, c)
            + edge_distance(distances, n, b, f),
    };
#pragma unroll
    for (int pattern = 0; pattern < 4; ++pattern) {
        if (additions[pattern] - removed < -LS_TOLERANCE) {
            return pattern + 3;
        }
    }
    return 0;
}

__device__ __forceinline__ int three_opt_source_index(
    int output,
    int first,
    int second,
    int third,
    int pattern
) {
    if (output <= first || output > third) {
        return output;
    }
    const int length_one = second - first;
    const int length_two = third - second;
    const int offset = output - first - 1;
    if (pattern == 3) {
        if (offset < length_one) {
            return second - offset;
        }
        return third - (offset - length_one);
    }
    if (pattern == 4) {
        if (offset < length_two) {
            return second + 1 + offset;
        }
        return first + 1 + (offset - length_two);
    }
    if (pattern == 5) {
        if (offset < length_two) {
            return third - offset;
        }
        return first + 1 + (offset - length_two);
    }
    // pattern == 6
    if (offset < length_two) {
        return second + 1 + offset;
    }
    return second - (offset - length_two);
}

}  // namespace rmtgp_ls

extern "C" __global__ void v2_two_opt(
    const float* distances_all,
    const uint16_t* nearest_all,
    const int32_t* task_instance,
    int task_count,
    int n,
    int nearest_stride,
    int ls_candidate_size,
    int ants,
    int iteration,
    int use_dlb,
    int final_stage,
    uint64_t seed,
    const uint64_t* instance_keys,
    uint16_t* tour_workspace,
    float* length_workspace,
    float* length_before_workspace,
    float* ls_gain_workspace,
    uint16_t* position_workspace,
    uint16_t* order_workspace,
    uint8_t* dlb_workspace,
    uint64_t* diagnostics
) {
    using namespace rmtgp_ls;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int flat_tour = blockIdx.x * warps_per_block + warp_in_block;
    const int total_tours = task_count * ants;
    if (flat_tour >= total_tours) {
        return;
    }
    const int task = flat_tour / ants;
    const int ant = flat_tour - task * ants;
    const int instance = task_instance[task];
    const float* distances = distances_all
        + static_cast<size_t>(instance) * n * n;
    const uint16_t* nearest = nearest_all
        + static_cast<size_t>(instance) * n * nearest_stride;
    uint16_t* tour = tour_workspace
        + static_cast<size_t>(flat_tour) * (n + 1);
    uint16_t* position = position_workspace
        + static_cast<size_t>(flat_tour) * n;
    uint16_t* order = order_workspace
        + static_cast<size_t>(flat_tour) * n;
    uint8_t* dlb = dlb_workspace + static_cast<size_t>(flat_tour) * n;

    __shared__ int move_first[32];
    __shared__ int move_second[32];
    __shared__ int move_endpoints[32][4];
    __shared__ int pass_improved[32];
    __shared__ unsigned long long move_counts[32];
    __shared__ unsigned long long check_counts[32];
    __shared__ unsigned long long pass_counts[32];

    for (int index = lane; index < n; index += 32) {
        const int city = static_cast<int>(tour[index]);
        position[city] = static_cast<uint16_t>(index);
        order[index] = static_cast<uint16_t>(index);
        dlb[index] = 0;
    }
    __syncwarp();
    if (lane == 0) {
        // 与 CPU oracle 一致的 Fisher--Yates 城市排列。
        for (int index = 0; index < n - 1; ++index) {
            const int remaining = n - index;
            const int offset = min(
                static_cast<int>(
                    counter_uniform(
                        seed,
                        instance_keys[instance],
                        iteration,
                        ant,
                        index,
                        17
                    ) * static_cast<float>(remaining)
                ),
                remaining - 1
            );
            const int other = index + offset;
            const uint16_t value = order[index];
            order[index] = order[other];
            order[other] = value;
        }
        move_counts[warp_in_block] = 0;
        check_counts[warp_in_block] = 0;
        pass_counts[warp_in_block] = 0;
        const float before = length_workspace[flat_tour];
        length_before_workspace[flat_tour] = before;
        ls_gain_workspace[flat_tour] = -1.0f;
    }
    __syncwarp();

    int continue_search = 1;
    while (continue_search != 0) {
        if (lane == 0) {
            pass_improved[warp_in_block] = 0;
            ++pass_counts[warp_in_block];
        }
        __syncwarp();
        for (int ordinal = 0; ordinal < n; ++ordinal) {
            const int city = static_cast<int>(order[ordinal]);
            if (use_dlb != 0 && dlb[city] != 0) {
                continue;
            }
            const int city_position = static_cast<int>(position[city]);
            const int successor = static_cast<int>(
                tour[(city_position + 1) % n]
            );
            const float successor_radius = edge_distance(
                distances,
                n,
                city,
                successor
            );
            int candidate = -1;
            float delta = CUDART_INF_F;
            if (lane < ls_candidate_size) {
                candidate = static_cast<int>(
                    nearest[city * nearest_stride + lane]
                );
                const int candidate_position = static_cast<int>(
                    position[candidate]
                );
                const int candidate_successor = static_cast<int>(
                    tour[(candidate_position + 1) % n]
                );
                if (
                    candidate != city
                    && candidate != successor
                    && candidate_successor != city
                    && edge_distance(distances, n, city, candidate)
                        < successor_radius
                ) {
                    delta = edge_distance(distances, n, city, candidate)
                        + edge_distance(
                            distances,
                            n,
                            successor,
                            candidate_successor
                        )
                        - successor_radius
                        - edge_distance(
                            distances,
                            n,
                            candidate,
                            candidate_successor
                        );
                }
            }
            unsigned int improving = __ballot_sync(
                FULL_WARP,
                delta < -LS_TOLERANCE
            );
            if (lane == 0) {
                check_counts[warp_in_block] += ls_candidate_size;
            }
            if (improving != 0) {
                const int chosen_lane = __ffs(improving) - 1;
                const int chosen = __shfl_sync(
                    FULL_WARP,
                    candidate,
                    chosen_lane
                );
                if (lane == 0) {
                    const int chosen_position = static_cast<int>(
                        position[chosen]
                    );
                    move_first[warp_in_block] = city;
                    move_second[warp_in_block] = chosen;
                    move_endpoints[warp_in_block][0] = city;
                    move_endpoints[warp_in_block][1] = successor;
                    move_endpoints[warp_in_block][2] = chosen;
                    move_endpoints[warp_in_block][3] = static_cast<int>(
                        tour[(chosen_position + 1) % n]
                    );
                }
                __syncwarp();
                reverse_between_edges(
                    tour,
                    position,
                    n,
                    move_first[warp_in_block],
                    move_second[warp_in_block],
                    lane
                );
                if (lane == 0) {
                    for (int endpoint = 0; endpoint < 4; ++endpoint) {
                        dlb[move_endpoints[warp_in_block][endpoint]] = 0;
                    }
                    ++move_counts[warp_in_block];
                    pass_improved[warp_in_block] = 1;
                }
                __syncwarp();
                continue;
            }

            const int predecessor = static_cast<int>(
                tour[(city_position + n - 1) % n]
            );
            const float predecessor_radius = edge_distance(
                distances,
                n,
                predecessor,
                city
            );
            candidate = -1;
            delta = CUDART_INF_F;
            int candidate_predecessor = -1;
            if (lane < ls_candidate_size) {
                candidate = static_cast<int>(
                    nearest[city * nearest_stride + lane]
                );
                const int candidate_position = static_cast<int>(
                    position[candidate]
                );
                candidate_predecessor = static_cast<int>(
                    tour[(candidate_position + n - 1) % n]
                );
                if (
                    candidate != city
                    && candidate_predecessor != city
                    && predecessor != candidate
                    && edge_distance(distances, n, city, candidate)
                        < predecessor_radius
                ) {
                    delta = edge_distance(distances, n, city, candidate)
                        + edge_distance(
                            distances,
                            n,
                            predecessor,
                            candidate_predecessor
                        )
                        - predecessor_radius
                        - edge_distance(
                            distances,
                            n,
                            candidate_predecessor,
                            candidate
                        );
                }
            }
            improving = __ballot_sync(
                FULL_WARP,
                delta < -LS_TOLERANCE
            );
            if (lane == 0) {
                check_counts[warp_in_block] += ls_candidate_size;
            }
            if (improving != 0) {
                const int chosen_lane = __ffs(improving) - 1;
                const int chosen = __shfl_sync(
                    FULL_WARP,
                    candidate,
                    chosen_lane
                );
                const int chosen_predecessor = __shfl_sync(
                    FULL_WARP,
                    candidate_predecessor,
                    chosen_lane
                );
                if (lane == 0) {
                    move_first[warp_in_block] = predecessor;
                    move_second[warp_in_block] = chosen_predecessor;
                    move_endpoints[warp_in_block][0] = predecessor;
                    move_endpoints[warp_in_block][1] = city;
                    move_endpoints[warp_in_block][2] = chosen_predecessor;
                    move_endpoints[warp_in_block][3] = chosen;
                }
                __syncwarp();
                reverse_between_edges(
                    tour,
                    position,
                    n,
                    move_first[warp_in_block],
                    move_second[warp_in_block],
                    lane
                );
                if (lane == 0) {
                    for (int endpoint = 0; endpoint < 4; ++endpoint) {
                        dlb[move_endpoints[warp_in_block][endpoint]] = 0;
                    }
                    ++move_counts[warp_in_block];
                    pass_improved[warp_in_block] = 1;
                }
                __syncwarp();
            } else if (lane == 0) {
                dlb[city] = 1;
            }
            __syncwarp();
        }
        if (lane == 0) {
            continue_search = (
                pass_improved[warp_in_block] != 0
                && move_counts[warp_in_block]
                    < static_cast<unsigned long long>(n) * 100ULL
            );
        }
        continue_search = __shfl_sync(FULL_WARP, continue_search, 0);
    }

    const float after = exact_length(tour, distances, n, lane);
    if (lane == 0) {
        const float before = length_before_workspace[flat_tour];
        tour[n] = tour[0];
        length_workspace[flat_tour] = after;
        const float gain = fminf(
            1.0f,
            fmaxf(0.0f, (before - after) / fmaxf(before, 1.0e-20f))
        );
        ls_gain_workspace[flat_tour] = 2.0f * gain - 1.0f;
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 8 + 4
            ),
            move_counts[warp_in_block]
        );
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 8 + 5
            ),
            check_counts[warp_in_block]
        );
        if (final_stage != 0 && after < before - LS_TOLERANCE) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(
                    diagnostics + static_cast<size_t>(task) * 8 + 6
                ),
                1ULL
            );
        }
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 8 + 7
            ),
            pass_counts[warp_in_block]
        );
    }
}

extern "C" __global__ void v2_three_opt(
    const float* distances_all,
    const uint16_t* nearest_all,
    const int32_t* task_instance,
    int task_count,
    int n,
    int nearest_stride,
    int ls_candidate_size,
    int ants,
    int use_dlb,
    uint16_t* tour_workspace,
    float* length_workspace,
    const float* length_before_workspace,
    float* ls_gain_workspace,
    uint16_t* position_workspace,
    uint16_t* order_workspace,
    uint8_t* dlb_workspace,
    uint16_t* scratch_tour_workspace,
    uint64_t* diagnostics
) {
    using namespace rmtgp_ls;
    // 真 3-opt 的内层有 ls_candidate_size^2 个候选对。与 2-opt 的
    // 一条 tour 一个 warp 不同，这里让一个 256-thread block 处理一条
    // tour。线程并行检查同一连续候选区间，再用 pair index 的 atomicMin
    // 恢复确定性的 first-improvement 顺序。
    const int thread = threadIdx.x;
    const int flat_tour = blockIdx.x;
    const int total_tours = task_count * ants;
    if (flat_tour >= total_tours) {
        return;
    }
    const int task = flat_tour / ants;
    const int instance = task_instance[task];
    const float* distances = distances_all
        + static_cast<size_t>(instance) * n * n;
    const uint16_t* nearest = nearest_all
        + static_cast<size_t>(instance) * n * nearest_stride;
    uint16_t* tour = tour_workspace
        + static_cast<size_t>(flat_tour) * (n + 1);
    uint16_t* position = position_workspace
        + static_cast<size_t>(flat_tour) * n;
    uint16_t* order = order_workspace
        + static_cast<size_t>(flat_tour) * n;
    uint8_t* dlb = dlb_workspace + static_cast<size_t>(flat_tour) * n;
    uint16_t* scratch = scratch_tour_workspace
        + static_cast<size_t>(flat_tour) * (n + 1);

    __shared__ int selected_pair;
    __shared__ int selected_first;
    __shared__ int selected_second;
    __shared__ int selected_third;
    __shared__ int selected_pattern;
    __shared__ int pass_improved;
    __shared__ int continue_search;
    __shared__ unsigned long long move_count;
    __shared__ unsigned long long check_count;
    __shared__ unsigned long long pass_count;
    __shared__ float length_reduction[512];

    for (int city = thread; city < n; city += blockDim.x) {
        dlb[city] = 0;
        position[static_cast<int>(tour[city])] = static_cast<uint16_t>(city);
    }
    if (thread == 0) {
        move_count = 0;
        check_count = 0;
        pass_count = 0;
        continue_search = 1;
    }
    __syncthreads();

    while (continue_search != 0) {
        if (thread == 0) {
            pass_improved = 0;
            ++pass_count;
        }
        __syncthreads();
        for (int ordinal = 0; ordinal < n; ++ordinal) {
            const int anchor = static_cast<int>(order[ordinal]);
            if (use_dlb != 0 && dlb[anchor] != 0) {
                continue;
            }
            const int first_position = static_cast<int>(position[anchor]);
            const int successor = static_cast<int>(
                tour[(first_position + 1) % n]
            );
            int moved = 0;
            const int pair_count = ls_candidate_size * ls_candidate_size;
            for (int base = 0; base < pair_count; base += blockDim.x) {
                if (thread == 0) {
                    selected_pair = 0x7fffffff;
                }
                __syncthreads();
                const int pair = base + thread;
                if (pair < pair_count) {
                    const int first_candidate = pair / ls_candidate_size;
                    const int second_candidate = pair % ls_candidate_size;
                    const int city_two = static_cast<int>(
                        nearest[
                            anchor * nearest_stride + first_candidate
                        ]
                    );
                    const int city_three = static_cast<int>(
                        nearest[
                            successor * nearest_stride + second_candidate
                        ]
                    );
                    int first = first_position;
                    int second = static_cast<int>(position[city_two]);
                    int third = static_cast<int>(position[city_three]);
                    sort_three(first, second, third);
                    const bool valid = first != second
                        && second != third
                        && second != first + 1
                        && third != second + 1
                        && !(first == 0 && third == n - 1);
                    if (valid) {
                        const int pattern = three_opt_pattern(
                            tour,
                            distances,
                            n,
                            first,
                            second,
                            third
                        );
                        if (pattern != 0) {
                            atomicMin(&selected_pair, pair);
                        }
                    }
                }
                __syncthreads();
                if (thread == 0) {
                    check_count += min(
                        static_cast<int>(blockDim.x),
                        pair_count - base
                    );
                    if (selected_pair != 0x7fffffff) {
                        const int first_candidate =
                            selected_pair / ls_candidate_size;
                        const int second_candidate =
                            selected_pair % ls_candidate_size;
                        const int city_two = static_cast<int>(
                            nearest[
                                anchor * nearest_stride + first_candidate
                            ]
                        );
                        const int city_three = static_cast<int>(
                            nearest[
                                successor * nearest_stride + second_candidate
                            ]
                        );
                        selected_first = first_position;
                        selected_second = static_cast<int>(
                            position[city_two]
                        );
                        selected_third = static_cast<int>(
                            position[city_three]
                        );
                        sort_three(
                            selected_first,
                            selected_second,
                            selected_third
                        );
                        selected_pattern = three_opt_pattern(
                            tour,
                            distances,
                            n,
                            selected_first,
                            selected_second,
                            selected_third
                        );
                    }
                }
                __syncthreads();
                if (selected_pair != 0x7fffffff) {
                    for (
                        int output = thread;
                        output < n;
                        output += blockDim.x
                    ) {
                        const int source = three_opt_source_index(
                            output,
                            selected_first,
                            selected_second,
                            selected_third,
                            selected_pattern
                        );
                        scratch[output] = tour[source];
                    }
                    __syncthreads();
                    for (
                        int output = thread;
                        output < n;
                        output += blockDim.x
                    ) {
                        const uint16_t city = scratch[output];
                        tour[output] = city;
                        position[static_cast<int>(city)]
                            = static_cast<uint16_t>(output);
                    }
                    __syncthreads();
                    for (
                        int city = thread;
                        city < n;
                        city += blockDim.x
                    ) {
                        // move 后重新开放全部城市，保证 DLB 不遗漏改善。
                        dlb[city] = 0;
                    }
                    if (thread == 0) {
                        ++move_count;
                        pass_improved = 1;
                    }
                    __syncthreads();
                    moved = 1;
                    break;
                }
            }
            if (moved) {
                break;
            }
            if (thread == 0) {
                dlb[anchor] = 1;
            }
            __syncthreads();
        }
        if (thread == 0) {
            continue_search = (
                pass_improved != 0
                && move_count
                    < static_cast<unsigned long long>(n) * 100ULL
            );
        }
        __syncthreads();
    }

    float partial = 0.0f;
    for (int edge = thread; edge < n; edge += blockDim.x) {
        partial += edge_distance(
            distances,
            n,
            static_cast<int>(tour[edge]),
            static_cast<int>(tour[(edge + 1) % n])
        );
    }
    length_reduction[thread] = partial;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (thread < offset) {
            length_reduction[thread] += length_reduction[thread + offset];
        }
        __syncthreads();
    }
    if (thread == 0) {
        const float after = length_reduction[0];
        const float before = length_before_workspace[flat_tour];
        tour[n] = tour[0];
        length_workspace[flat_tour] = after;
        const float gain = fminf(
            1.0f,
            fmaxf(0.0f, (before - after) / fmaxf(before, 1.0e-20f))
        );
        ls_gain_workspace[flat_tour] = 2.0f * gain - 1.0f;
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 8 + 4
            ),
            move_count
        );
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 8 + 5
            ),
            check_count
        );
        if (after < before - LS_TOLERANCE) {
            atomicAdd(
                reinterpret_cast<unsigned long long*>(
                    diagnostics + static_cast<size_t>(task) * 8 + 6
                ),
                1ULL
            );
        }
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 8 + 7
            ),
            pass_count
        );
    }
}
