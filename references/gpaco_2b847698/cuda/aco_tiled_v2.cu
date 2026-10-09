// RMTGP–ACO CUDA v2：population×instance×ant×candidate 分层并行。
//
// 本文件与 aco_fused.cu 拼接后由 NVRTC 编译。旧文件提供稳定的 opcode、
// counter RNG 和 pheromone residual 辅助函数；这里新增分阶段 kernel。

#include <cuda_fp16.h>

#ifndef RMTGP_CANDIDATE_LANES
#define RMTGP_CANDIDATE_LANES 8
#endif
#ifndef RMTGP_CANDIDATE_PAD
#define RMTGP_CANDIDATE_PAD 32
#endif
#ifndef RMTGP_MAX_STACK_V2
#define RMTGP_MAX_STACK_V2 8
#endif
#ifndef RMTGP_VARIANT
#define RMTGP_VARIANT 1
#endif
#ifndef RMTGP_STATIC_PRECISION
// 0: FP32；1: FP16 storage；2: BF16 storage。
#define RMTGP_STATIC_PRECISION 0
#endif
#ifndef RMTGP_SCORE_QUANTIZE_FP16
#define RMTGP_SCORE_QUANTIZE_FP16 0
#endif

namespace rmtgp_v2 {

constexpr int V2_MAX_ANTS = 32;
constexpr int V2_UPDATE_THREADS = 256;

__device__ __forceinline__ void atomic_increment_u8(uint8_t* address) {
    // CUDA 没有原生 byte atomicAdd。一个 32-bit CAS 同时保留相邻三个
    // counter；蚂蚁数固定为 32，因此单 byte 不会溢出。
    const size_t raw = reinterpret_cast<size_t>(address);
    unsigned int* word = reinterpret_cast<unsigned int*>(raw & ~size_t(3));
    const unsigned int shift = static_cast<unsigned int>((raw & 3U) * 8U);
    const unsigned int mask = 0xffU << shift;
    unsigned int observed = *word;
    while (true) {
        const unsigned int count = (observed & mask) >> shift;
        const unsigned int updated = (
            (observed & ~mask)
            | (((count + 1U) & 0xffU) << shift)
        );
        const unsigned int previous = atomicCAS(word, observed, updated);
        if (previous == observed) {
            return;
        }
        observed = previous;
    }
}

struct Stats {
    int count;
    float base_total;
    float log_tau_mean;
    float log_tau_std;
    float log_eta_mean;
    float log_eta_std;
    float tau_mean;
    float distance_mean;
    float entropy;
};

__device__ __forceinline__ float load_static(
    const void* values,
    size_t index
) {
#if RMTGP_STATIC_PRECISION == 1
    return __half2float(reinterpret_cast<const __half*>(values)[index]);
#elif RMTGP_STATIC_PRECISION == 2
    const uint16_t bits = reinterpret_cast<const uint16_t*>(values)[index];
    return __uint_as_float(static_cast<unsigned int>(bits) << 16);
#else
    return reinterpret_cast<const float*>(values)[index];
#endif
}

__device__ __forceinline__ float score_storage_round(float value) {
#if RMTGP_SCORE_QUANTIZE_FP16
    return __half2float(__float2half_rn(value));
#else
    return value;
#endif
}

template <typename T>
__device__ __forceinline__ T group_sum(T value) {
    const int warp_lane = threadIdx.x & 31;
    const int group_start = (
        warp_lane / RMTGP_CANDIDATE_LANES
    ) * RMTGP_CANDIDATE_LANES;
    const unsigned int group_bits = (
        0xffffffffu >> (32 - RMTGP_CANDIDATE_LANES)
    ) << group_start;
#pragma unroll
    for (int offset = RMTGP_CANDIDATE_LANES / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(
            group_bits,
            value,
            offset,
            RMTGP_CANDIDATE_LANES
        );
    }
    return __shfl_sync(
        group_bits,
        value,
        0,
        RMTGP_CANDIDATE_LANES
    );
}

__device__ __forceinline__ float base_score_tiled(
    const float* pheromone,
    const void* heuristic_all,
    size_t instance_base,
    int n,
    int current,
    int city,
    float alpha,
    float beta
) {
    const int edge = current * n + city;
    const float tau = pheromone[edge];
    const float eta = load_static(heuristic_all, instance_base + edge);
    if (alpha == 1.0f && beta == 2.0f) {
        return tau * eta * eta;
    }
    const float tau_component = alpha == 1.0f ? tau : powf(tau, alpha);
    float eta_component;
    if (beta == 2.0f) {
        eta_component = eta * eta;
    } else if (beta == 1.0f) {
        eta_component = eta;
    } else {
        eta_component = powf(eta, beta);
    }
    return tau_component * eta_component;
}

__device__ __forceinline__ float evaluate_program_small(
    const int8_t* opcodes,
    const float* float_arguments,
    const int16_t* integer_arguments,
    int length,
    const float* terminals
) {
    float stack[RMTGP_MAX_STACK_V2];
    int top = 0;
#pragma unroll 1
    for (int instruction = 0; instruction < length; ++instruction) {
        const int8_t opcode = opcodes[instruction];
        if (opcode == OP_CONST) {
            stack[top++] = float_arguments[instruction];
            continue;
        }
        if (opcode == OP_TERMINAL) {
            stack[top++] = terminals[integer_arguments[instruction]];
            continue;
        }
        if (opcode == OP_ABS || opcode == OP_NEG) {
            const float value = stack[top - 1];
            stack[top - 1] = sanitize(
                opcode == OP_ABS ? fabsf(value) : -value
            );
            continue;
        }
        const float right = stack[top - 1];
        const float left = stack[top - 2];
        --top;
        float result;
        if (opcode == OP_ADD) {
            result = left + right;
        } else if (opcode == OP_SUB) {
            result = left - right;
        } else if (opcode == OP_MUL) {
            result = left * right;
        } else if (opcode == OP_PDIV) {
            result = left * right / (right * right + 1.0e-6f);
        } else if (opcode == OP_PDIV1) {
            result = fabsf(right) > 1.0e-6f ? left / right : 1.0f;
        } else if (opcode == OP_MIN) {
            result = fminf(left, right);
        } else {
            result = fmaxf(left, right);
        }
        stack[top - 1] = sanitize(result);
    }
    return sanitize(stack[0]);
}

__device__ __forceinline__ int candidate_ordinal(
    const uint16_t* nearest,
    const uint64_t* visited,
    int position
) {
    int result = 0;
    for (int prior = 0; prior < position; ++prior) {
        result += !is_visited(visited, static_cast<int>(nearest[prior]));
    }
    return result;
}

__device__ __forceinline__ float transition_score_tiled(
    const float* coords_all,
    const void* distances_all,
    const void* heuristic_all,
    const void* log_heuristic_all,
    const uint16_t* full_nn_rank_all,
    size_t instance_base,
    size_t coordinate_base,
    const float* pheromone,
    const uint16_t* nearest,
    const uint64_t* visited,
    int n,
    int current,
    int previous,
    int city,
    int position,
    bool fallback,
    float alpha,
    float beta,
    float epsilon_numeric,
    int transition_mode,
    float gamma_transition,
    bool program_active,
    int program_index,
    const int8_t* opcodes,
    const float* float_arguments,
    const int16_t* integer_arguments,
    int program_length,
    uint64_t required_mask,
    int construction_step,
    int iteration,
    int stagnation,
    int total_iterations,
    const Stats& stats
) {
    const int edge = current * n + city;
    const float baseline = base_score_tiled(
        pheromone,
        heuristic_all,
        instance_base,
        n,
        current,
        city,
        alpha,
        beta
    );
    if (!program_active) {
        return baseline;
    }

    float terminals[16];
    if ((required_mask & (UINT64_C(1) << 0)) != 0) {
        terminals[0] = tanhf(
            (
                logf(fmaxf(pheromone[edge], epsilon_numeric))
                - stats.log_tau_mean
            ) / (stats.log_tau_std + 1.0e-8f)
        );
    }
    if ((required_mask & (UINT64_C(1) << 1)) != 0) {
        terminals[1] = tanhf(
            (
                load_static(log_heuristic_all, instance_base + edge)
                - stats.log_eta_mean
            ) / (stats.log_eta_std + 1.0e-8f)
        );
    }
    if ((required_mask & (UINT64_C(1) << 2)) != 0) {
        const float probability = stats.base_total > epsilon_numeric
            ? baseline / stats.base_total
            : 1.0f / static_cast<float>(stats.count);
        terminals[2] = tanhf(
            logf(fmaxf(probability, epsilon_numeric))
            + logf(static_cast<float>(stats.count))
        );
    }
    if ((required_mask & (UINT64_C(1) << 3)) != 0) {
        int rank;
        if (fallback) {
            rank = 0;
            const float value = load_static(
                distances_all,
                instance_base + edge
            );
            for (int other = 0; other < n; ++other) {
                if (is_visited(visited, other)) {
                    continue;
                }
                const float other_value = load_static(
                    distances_all,
                    instance_base + current * n + other
                );
                if (
                    other_value < value
                    || (other_value == value && other < city)
                ) {
                    ++rank;
                }
            }
        } else {
            rank = candidate_ordinal(nearest, visited, position);
        }
        terminals[3] = stats.count == 1
            ? 0.0f
            : 1.0f - 2.0f * static_cast<float>(rank)
                / static_cast<float>(stats.count - 1);
    }
    terminals[4] = stats.entropy;
    terminals[5] = 2.0f * static_cast<float>(construction_step)
        / static_cast<float>(n > 1 ? n - 1 : 1) - 1.0f;
    terminals[6] = 2.0f * static_cast<float>(iteration - 1)
        / static_cast<float>(total_iterations > 1 ? total_iterations - 1 : 1)
        - 1.0f;
    terminals[7] = 2.0f * fminf(
        static_cast<float>(stagnation) / static_cast<float>(total_iterations),
        1.0f
    ) - 1.0f;
    terminals[8] = pheromone[edge];
    terminals[9] = load_static(distances_all, instance_base + edge);
    terminals[10] = stats.tau_mean;
    terminals[11] = stats.distance_mean;
    terminals[12] = static_cast<float>(n);
    terminals[13] = static_cast<float>(stats.count);
    if ((required_mask & (UINT64_C(1) << 14)) != 0) {
        const float denominator = static_cast<float>(max(n - 2, 1));
        const float forward = static_cast<float>(
            full_nn_rank_all[instance_base + current * n + city]
        ) - 1.0f;
        const float reverse = static_cast<float>(
            full_nn_rank_all[instance_base + city * n + current]
        ) - 1.0f;
        terminals[14] = fminf(
            1.0f,
            fmaxf(
                0.0f,
                1.0f - (forward + reverse) / (2.0f * denominator)
            )
        );
    }
    if ((required_mask & (UINT64_C(1) << 15)) != 0) {
        if (previous < 0) {
            terminals[15] = 0.0f;
        } else {
            const float incoming_x = coords_all[
                coordinate_base + static_cast<size_t>(current) * 2
            ] - coords_all[
                coordinate_base + static_cast<size_t>(previous) * 2
            ];
            const float incoming_y = coords_all[
                coordinate_base + static_cast<size_t>(current) * 2 + 1
            ] - coords_all[
                coordinate_base + static_cast<size_t>(previous) * 2 + 1
            ];
            const float outgoing_x = coords_all[
                coordinate_base + static_cast<size_t>(city) * 2
            ] - coords_all[
                coordinate_base + static_cast<size_t>(current) * 2
            ];
            const float outgoing_y = coords_all[
                coordinate_base + static_cast<size_t>(city) * 2 + 1
            ] - coords_all[
                coordinate_base + static_cast<size_t>(current) * 2 + 1
            ];
            const float numerator = incoming_x * outgoing_x
                + incoming_y * outgoing_y;
            const float denominator = sqrtf(
                incoming_x * incoming_x + incoming_y * incoming_y
            ) * sqrtf(
                outgoing_x * outgoing_x + outgoing_y * outgoing_y
            ) + epsilon_numeric;
            terminals[15] = fminf(
                1.0f,
                fmaxf(-1.0f, numerator / denominator)
            );
        }
    }

#if RMTGP_GENERATED_GP
    const float raw = evaluate_transition_generated(
        program_index,
        terminals
    );
#else
    const float raw = evaluate_program_small(
        opcodes,
        float_arguments,
        integer_arguments,
        program_length,
        terminals
    );
#endif
    if (transition_mode == 1) {
        return softplus_clipped(raw) + epsilon_numeric;
    }
    return baseline * (1.0f + gamma_transition * tanhf(raw));
}

}  // namespace rmtgp_v2

extern "C" __global__ void v2_init(
    const float* initial_tau0,
    const float* initial_tau_min,
    const float* initial_tau_max,
    const int32_t* task_instance,
    int task_count,
    int n,
    float* pheromone_workspace,
    uint8_t* edge_frequency_workspace,
    float* global_best_lengths,
    float* restart_best_lengths,
    float* task_tau_min,
    float* task_tau_max,
    int32_t* global_best_iterations,
    int32_t* stagnation,
    int32_t* restart_found_best,
    int32_t* restart_iteration,
    float* global_best_ls_gain,
    float* restart_best_ls_gain,
    uint64_t* diagnostics
) {
    const int task = blockIdx.x;
    if (task >= task_count) {
        return;
    }
    const int tid = threadIdx.x;
    const int instance = task_instance[task];
    float* pheromone = pheromone_workspace
        + static_cast<size_t>(task) * n * n;
    const size_t frequency_stride = (
        static_cast<size_t>(n) * n + 3U
    ) & ~static_cast<size_t>(3U);
    uint8_t* edge_frequency = edge_frequency_workspace
        + static_cast<size_t>(task) * frequency_stride;
    const float tau0 = initial_tau0[instance];
    for (int edge = tid; edge < n * n; edge += blockDim.x) {
        pheromone[edge] = edge / n == edge % n ? 0.0f : tau0;
        edge_frequency[edge] = 0;
    }
    if (tid < 8) {
        diagnostics[static_cast<size_t>(task) * 8 + tid] = 0;
    }
    if (tid == 0) {
        global_best_lengths[task] = CUDART_INF_F;
        restart_best_lengths[task] = CUDART_INF_F;
        task_tau_min[task] = initial_tau_min[instance];
        task_tau_max[task] = initial_tau_max[instance];
        global_best_iterations[task] = 0;
        stagnation[task] = 0;
        restart_found_best[task] = 0;
        restart_iteration[task] = 1;
        global_best_ls_gain[task] = -1.0f;
        restart_best_ls_gain[task] = -1.0f;
    }
}

extern "C" __global__ void v2_construct(
    const float* coords_all,
    const void* distances_all,
    const void* heuristic_all,
    const void* log_heuristic_all,
    const uint16_t* nearest_all,
    const uint16_t* full_nn_rank_all,
    const int8_t* tr_opcodes,
    const float* tr_float_arguments,
    const int16_t* tr_integer_arguments,
    const int16_t* tr_lengths,
    const uint64_t* tr_required_masks,
    const uint8_t* tr_active,
    int tr_width,
    const int32_t* task_program,
    const int32_t* task_instance,
    int task_count,
    int n,
    int candidate_size,
    int ants,
    int total_iterations,
    int iteration,
    float alpha,
    float beta,
    float q0,
    float xi,
    float gamma_transition,
    int transition_mode,
    float epsilon_numeric,
    uint64_t seed,
    const uint64_t* instance_keys,
    const float* initial_tau0,
    float* pheromone_workspace,
    uint16_t* tour_workspace,
    uint64_t* visited_workspace,
    float* length_workspace,
    float* score_workspace,
    const int32_t* stagnation_all,
    uint64_t* diagnostics
) {
    using namespace rmtgp_v2;
    const int task = blockIdx.x;
    if (task >= task_count || ants > V2_MAX_ANTS) {
        return;
    }
    const int tid = threadIdx.x;
    const int ant = tid / RMTGP_CANDIDATE_LANES;
    const int lane = tid & (RMTGP_CANDIDATE_LANES - 1);
    if (ant >= ants) {
        return;
    }
    const int program = task_program[task];
    const int instance = task_instance[task];
    const int words = (n + 63) / 64;
    const size_t instance_base = static_cast<size_t>(instance) * n * n;
    const size_t coordinate_base = static_cast<size_t>(instance) * n * 2;
    const uint16_t* nearest = nearest_all
        + static_cast<size_t>(instance) * n * candidate_size;
    const uint64_t instance_key = instance_keys[instance];
    float* pheromone = pheromone_workspace
        + static_cast<size_t>(task) * n * n;
    uint16_t* ant_tour = tour_workspace
        + (static_cast<size_t>(task) * ants + ant) * (n + 1);
    uint64_t* ant_visited = visited_workspace
        + (static_cast<size_t>(task) * ants + ant) * words;
    float* ant_scores = score_workspace
        + (static_cast<size_t>(task) * ants + ant) * n;
    const int8_t* tr_ops = tr_opcodes
        + static_cast<size_t>(program) * tr_width;
    const float* tr_fargs = tr_float_arguments
        + static_cast<size_t>(program) * tr_width;
    const int16_t* tr_iargs = tr_integer_arguments
        + static_cast<size_t>(program) * tr_width;
    const bool program_active = tr_active[program] != 0;
    const uint64_t required_mask = program_active
        ? tr_required_masks[program]
        : UINT64_C(0);

    __shared__ int edge_u[V2_MAX_ANTS];
    __shared__ int edge_v[V2_MAX_ANTS];
    __shared__ float candidate_scores[
        V2_MAX_ANTS * RMTGP_CANDIDATE_PAD
    ];
    __shared__ uint64_t candidate_counts[V2_MAX_ANTS];
    __shared__ uint64_t uniform_counts[V2_MAX_ANTS];

    uint64_t local_candidate_fallbacks = 0;
    uint64_t local_uniform_fallbacks = 0;
    if (lane == 0) {
        for (int word = 0; word < words; ++word) {
            ant_visited[word] = 0;
        }
        const int start = min(
            static_cast<int>(
                counter_uniform(
                    seed,
                    instance_key,
                    iteration,
                    ant,
                    0,
                    1
                ) * static_cast<float>(n)
            ),
            n - 1
        );
        ant_tour[0] = static_cast<uint16_t>(start);
        mark_visited(ant_visited, start);
        length_workspace[static_cast<size_t>(task) * ants + ant] = 0.0f;
    }
    __syncthreads();

    for (int step = 1; step < n; ++step) {
        const int current = static_cast<int>(ant_tour[step - 1]);
        int local_count = 0;
        for (
            int position = lane;
            position < candidate_size;
            position += RMTGP_CANDIDATE_LANES
        ) {
            local_count += !is_visited(
                ant_visited,
                static_cast<int>(nearest[current * candidate_size + position])
            );
        }
        const int nearest_count = group_sum(local_count);
        const bool fallback = nearest_count == 0;
        if (fallback && lane == 0) {
            ++local_candidate_fallbacks;
            ++local_uniform_fallbacks;
        }
        const int limit = fallback ? n : candidate_size;
        const uint16_t* current_nearest = nearest
            + current * candidate_size;

        float log_tau_sum = 0.0f;
        float log_tau_sq = 0.0f;
        float log_eta_sum = 0.0f;
        float log_eta_sq = 0.0f;
        float tau_sum = 0.0f;
        float distance_sum = 0.0f;
        float base_total = 0.0f;
        local_count = 0;
        const bool need_log_tau = (
            required_mask & (UINT64_C(1) << 0)
        ) != 0;
        const bool need_log_eta = (
            required_mask & (UINT64_C(1) << 1)
        ) != 0;
        const bool need_tau_mean = (
            required_mask & (UINT64_C(1) << 10)
        ) != 0;
        const bool need_distance_mean = (
            required_mask & (UINT64_C(1) << 11)
        ) != 0;
        for (
            int position = lane;
            position < limit;
            position += RMTGP_CANDIDATE_LANES
        ) {
            const int city = fallback
                ? position
                : static_cast<int>(current_nearest[position]);
            if (is_visited(ant_visited, city)) {
                continue;
            }
            const int edge = current * n + city;
            if (need_log_tau) {
                const float value = logf(fmaxf(
                    pheromone[edge],
                    epsilon_numeric
                ));
                log_tau_sum += value;
                log_tau_sq += value * value;
            }
            if (need_log_eta) {
                const float value = load_static(
                    log_heuristic_all,
                    instance_base + edge
                );
                log_eta_sum += value;
                log_eta_sq += value * value;
            }
            if (need_tau_mean) {
                tau_sum += pheromone[edge];
            }
            if (need_distance_mean) {
                distance_sum += load_static(
                    distances_all,
                    instance_base + edge
                );
            }
            base_total += base_score_tiled(
                pheromone,
                heuristic_all,
                instance_base,
                n,
                current,
                city,
                alpha,
                beta
            );
            ++local_count;
        }

        Stats stats{};
        stats.count = group_sum(local_count);
        stats.base_total = group_sum(base_total);
        log_tau_sum = group_sum(log_tau_sum);
        log_tau_sq = group_sum(log_tau_sq);
        log_eta_sum = group_sum(log_eta_sum);
        log_eta_sq = group_sum(log_eta_sq);
        tau_sum = group_sum(tau_sum);
        distance_sum = group_sum(distance_sum);
        const float inverse = 1.0f / static_cast<float>(stats.count);
        if (need_log_tau) {
            stats.log_tau_mean = log_tau_sum * inverse;
            stats.log_tau_std = sqrtf(fmaxf(
                0.0f,
                log_tau_sq * inverse
                    - stats.log_tau_mean * stats.log_tau_mean
            ));
        }
        if (need_log_eta) {
            stats.log_eta_mean = log_eta_sum * inverse;
            stats.log_eta_std = sqrtf(fmaxf(
                0.0f,
                log_eta_sq * inverse
                    - stats.log_eta_mean * stats.log_eta_mean
            ));
        }
        if (need_tau_mean) {
            stats.tau_mean = tau_sum * inverse;
        }
        if (need_distance_mean) {
            stats.distance_mean = distance_sum * inverse;
        }

        if ((required_mask & (UINT64_C(1) << 4)) != 0) {
            if (stats.count == 1) {
                stats.entropy = -1.0f;
            } else if (stats.base_total <= epsilon_numeric) {
                stats.entropy = 1.0f;
            } else {
                float entropy = 0.0f;
                for (
                    int position = lane;
                    position < limit;
                    position += RMTGP_CANDIDATE_LANES
                ) {
                    const int city = fallback
                        ? position
                        : static_cast<int>(current_nearest[position]);
                    if (is_visited(ant_visited, city)) {
                        continue;
                    }
                    const float probability = base_score_tiled(
                        pheromone,
                        heuristic_all,
                        instance_base,
                        n,
                        current,
                        city,
                        alpha,
                        beta
                    ) / stats.base_total;
                    entropy -= probability * logf(fmaxf(
                        probability,
                        epsilon_numeric
                    ));
                }
                entropy = group_sum(entropy);
                stats.entropy = 2.0f * entropy
                    / logf(static_cast<float>(stats.count)) - 1.0f;
            }
        }

        float local_score_total = 0.0f;
        for (
            int position = lane;
            position < limit;
            position += RMTGP_CANDIDATE_LANES
        ) {
            const int city = fallback
                ? position
                : static_cast<int>(current_nearest[position]);
            float score = -CUDART_INF_F;
            if (!is_visited(ant_visited, city)) {
                score = transition_score_tiled(
                    coords_all,
                    distances_all,
                    heuristic_all,
                    log_heuristic_all,
                    full_nn_rank_all,
                    instance_base,
                    coordinate_base,
                    pheromone,
                    current_nearest,
                    ant_visited,
                    n,
                    current,
                    step < 2
                        ? -1
                        : static_cast<int>(ant_tour[step - 2]),
                    city,
                    position,
                    fallback,
                    alpha,
                    beta,
                    epsilon_numeric,
                    transition_mode,
                    gamma_transition,
                    program_active,
                    program,
                    tr_ops,
                    tr_fargs,
                    tr_iargs,
                    tr_lengths[program],
                    required_mask,
                    step,
                    iteration,
                    stagnation_all[task],
                    total_iterations,
                    stats
                );
                score = score_storage_round(score);
                local_score_total += score;
            }
            if (fallback) {
                ant_scores[position] = score;
            } else {
                candidate_scores[
                    ant * RMTGP_CANDIDATE_PAD + position
                ] = score;
            }
        }
        const float score_total = group_sum(local_score_total);
        __syncwarp();

        if (lane == 0) {
            int greedy_city = -1;
            float greedy_score = -CUDART_INF_F;
            for (int position = 0; position < limit; ++position) {
                const int city = fallback
                    ? position
                    : static_cast<int>(current_nearest[position]);
                if (is_visited(ant_visited, city)) {
                    continue;
                }
                const float score = fallback
                    ? ant_scores[position]
                    : candidate_scores[
                        ant * RMTGP_CANDIDATE_PAD + position
                    ];
                if (score > greedy_score) {
                    greedy_score = score;
                    greedy_city = city;
                }
            }

            int chosen = greedy_city;
            if (!fallback) {
                const bool base_uniform = stats.base_total <= epsilon_numeric;
                const bool greedy = RMTGP_VARIANT == 1 && counter_uniform(
                    seed,
                    instance_key,
                    iteration,
                    ant,
                    step,
                    2
                ) <= q0;
                if (greedy) {
                    if (base_uniform) {
                        ++local_uniform_fallbacks;
                    }
                } else {
                    const bool residual_uniform = score_total
                        <= epsilon_numeric;
                    if (base_uniform || residual_uniform) {
                        ++local_uniform_fallbacks;
                    }
                    const float uniform = counter_uniform(
                        seed,
                        instance_key,
                        iteration,
                        ant,
                        step,
                        3
                    );
                    if (residual_uniform) {
                        const int selected = min(
                            static_cast<int>(
                                uniform * static_cast<float>(stats.count)
                            ),
                            stats.count - 1
                        );
                        int ordinal = 0;
                        for (
                            int position = 0;
                            position < limit;
                            ++position
                        ) {
                            const int city = static_cast<int>(
                                current_nearest[position]
                            );
                            if (!is_visited(ant_visited, city)) {
                                if (ordinal == selected) {
                                    chosen = city;
                                    break;
                                }
                                ++ordinal;
                            }
                        }
                    } else {
                        const float threshold = uniform * score_total;
                        float cumulative = 0.0f;
                        for (
                            int position = 0;
                            position < limit;
                            ++position
                        ) {
                            const int city = static_cast<int>(
                                current_nearest[position]
                            );
                            if (is_visited(ant_visited, city)) {
                                continue;
                            }
                            chosen = city;
                            cumulative += candidate_scores[
                                ant * RMTGP_CANDIDATE_PAD + position
                            ];
                            if (cumulative >= threshold) {
                                break;
                            }
                        }
                    }
                }
            }
            ant_tour[step] = static_cast<uint16_t>(chosen);
            edge_u[ant] = current;
            edge_v[ant] = chosen;
            length_workspace[
                static_cast<size_t>(task) * ants + ant
            ] += load_static(
                distances_all,
                instance_base + current * n + chosen
            );
        }
        __syncthreads();

#if RMTGP_VARIANT == 1
        if (lane == 0) {
            const int first = min(edge_u[ant], edge_v[ant]);
            const int second = max(edge_u[ant], edge_v[ant]);
            bool first_occurrence = true;
            int multiplicity = 0;
            for (int other = 0; other < ants; ++other) {
                const int other_first = min(edge_u[other], edge_v[other]);
                const int other_second = max(edge_u[other], edge_v[other]);
                if (other_first == first && other_second == second) {
                    if (other < ant) {
                        first_occurrence = false;
                        break;
                    }
                    ++multiplicity;
                }
            }
            if (first_occurrence) {
                const float factor = powf(
                    1.0f - xi,
                    static_cast<float>(multiplicity)
                );
                const int forward = first * n + second;
                const float tau0 = initial_tau0[instance];
                const float updated = factor * pheromone[forward]
                    + (1.0f - factor) * tau0;
                pheromone[forward] = updated;
                pheromone[second * n + first] = updated;
            }
        }
        __syncthreads();
#endif
        if (lane == 0) {
            mark_visited(
                ant_visited,
                static_cast<int>(ant_tour[step])
            );
        }
        __syncthreads();
    }

    if (lane == 0) {
        const int last = static_cast<int>(ant_tour[n - 1]);
        const int first = static_cast<int>(ant_tour[0]);
        ant_tour[n] = ant_tour[0];
        edge_u[ant] = last;
        edge_v[ant] = first;
        length_workspace[
            static_cast<size_t>(task) * ants + ant
        ] += load_static(
            distances_all,
            instance_base + last * n + first
        );
    }
    __syncthreads();
#if RMTGP_VARIANT == 1
    if (lane == 0) {
        const int first = min(edge_u[ant], edge_v[ant]);
        const int second = max(edge_u[ant], edge_v[ant]);
        bool first_occurrence = true;
        int multiplicity = 0;
        for (int other = 0; other < ants; ++other) {
            const int other_first = min(edge_u[other], edge_v[other]);
            const int other_second = max(edge_u[other], edge_v[other]);
            if (other_first == first && other_second == second) {
                if (other < ant) {
                    first_occurrence = false;
                    break;
                }
                ++multiplicity;
            }
        }
        if (first_occurrence) {
            const float factor = powf(
                1.0f - xi,
                static_cast<float>(multiplicity)
            );
            const int forward = first * n + second;
            const float tau0 = initial_tau0[instance];
            const float updated = factor * pheromone[forward]
                + (1.0f - factor) * tau0;
            pheromone[forward] = updated;
            pheromone[second * n + first] = updated;
        }
    }
    __syncthreads();
#endif

    if (lane == 0) {
        candidate_counts[ant] = local_candidate_fallbacks;
        uniform_counts[ant] = local_uniform_fallbacks;
    }
    __syncthreads();
    if (tid == 0) {
        uint64_t candidate_total = 0;
        uint64_t uniform_total = 0;
        for (int other = 0; other < ants; ++other) {
            candidate_total += candidate_counts[other];
            uniform_total += uniform_counts[other];
        }
        diagnostics[static_cast<size_t>(task) * 8] += candidate_total;
        diagnostics[static_cast<size_t>(task) * 8 + 1] += uniform_total;
    }
}

extern "C" __global__ void v2_update(
    const float* log_heuristic_all,
    const uint16_t* nearest_all,
    const uint16_t* full_nn_rank_all,
    const float* node_log_eta_mean_all,
    const int8_t* ph_opcodes,
    const float* ph_float_arguments,
    const int16_t* ph_integer_arguments,
    const int16_t* ph_lengths,
    const uint64_t* ph_required_masks,
    const uint8_t* ph_active,
    int ph_width,
    const int32_t* task_program,
    const int32_t* task_instance,
    int task_count,
    int n,
    int candidate_size,
    int ants,
    int total_iterations,
    int iteration,
    float rho,
    float gamma_pheromone,
    int pheromone_mode,
    float epsilon_numeric,
    int mmas_update_period,
    float mmas_p_best,
    int mmas_branch_check_period,
    float mmas_branch_lambda,
    float mmas_branch_threshold,
    int mmas_restart_stagnation,
    int local_search_active,
    int ls_gain_semantics,
    float* pheromone_workspace,
    uint16_t* tour_workspace,
    const uint16_t* pre_tour_workspace,
    float* length_workspace,
    const float* length_before_workspace,
    const float* ls_gain_workspace,
    const float* edge_gain_workspace,
    float* deposit_workspace,
    uint8_t* edge_frequency_workspace,
    uint8_t* pre_edge_frequency_workspace,
    uint16_t* restart_tour_workspace,
    uint16_t* best_tours,
    float* global_best_lengths,
    float* restart_best_lengths,
    float* task_tau_min,
    float* task_tau_max,
    int32_t* global_best_iterations,
    int32_t* stagnation_all,
    int32_t* restart_found_best,
    int32_t* restart_iteration,
    float* global_best_ls_gain,
    float* restart_best_ls_gain,
    float* global_best_edge_gain_workspace,
    float* restart_best_edge_gain_workspace,
    const int8_t* origin_workspace,
    int8_t* global_best_origin_workspace,
    int8_t* restart_best_origin_workspace,
    float* basin_sum,
    float* pre_basin_sum,
    unsigned long long* retained_edge_sum,
    int basin_top_q,
    int audit_local_search,
    float* anytime_sum,
    float* anytime,
    int record_anytime,
    uint64_t* diagnostics
) {
    using namespace rmtgp_v2;
    const int task = blockIdx.x;
    if (task >= task_count) {
        return;
    }
    const int tid = threadIdx.x;
    const int program = task_program[task];
    const int instance = task_instance[task];
    const float* log_heuristic = log_heuristic_all
        + static_cast<size_t>(instance) * n * n;
    const uint16_t* nearest = nearest_all
        + static_cast<size_t>(instance) * n * candidate_size;
    const uint16_t* full_nn_rank = full_nn_rank_all
        + static_cast<size_t>(instance) * n * n;
    const float* node_log_eta_mean = node_log_eta_mean_all
        + static_cast<size_t>(instance) * n;
    float* pheromone = pheromone_workspace
        + static_cast<size_t>(task) * n * n;
    uint16_t* tours = tour_workspace
        + static_cast<size_t>(task) * ants * (n + 1);
    const uint16_t* pre_tours = pre_tour_workspace
        + static_cast<size_t>(task) * ants * (n + 1);
    float* colony_lengths = length_workspace
        + static_cast<size_t>(task) * ants;
    float* deposits = deposit_workspace
        + static_cast<size_t>(task) * ants * n;
    const size_t frequency_stride = (
        static_cast<size_t>(n) * n + 3U
    ) & ~static_cast<size_t>(3U);
    uint8_t* edge_frequency = edge_frequency_workspace
        + static_cast<size_t>(task) * frequency_stride;
    uint8_t* pre_edge_frequency = pre_edge_frequency_workspace
        + static_cast<size_t>(task) * frequency_stride;
    uint16_t* restart_tour = restart_tour_workspace
        + static_cast<size_t>(task) * (n + 1);
    uint16_t* global_best_tour = best_tours
        + static_cast<size_t>(task) * (n + 1);
    const int8_t* origins = origin_workspace
        + static_cast<size_t>(task) * ants * n;
    int8_t* global_best_origin = global_best_origin_workspace
        + static_cast<size_t>(task) * n;
    int8_t* restart_best_origin = restart_best_origin_workspace
        + static_cast<size_t>(task) * n;
    const float* edge_gains = edge_gain_workspace
        + static_cast<size_t>(task) * ants * n;
    float* global_best_edge_gain = global_best_edge_gain_workspace
        + static_cast<size_t>(task) * n;
    float* restart_best_edge_gain = restart_best_edge_gain_workspace
        + static_cast<size_t>(task) * n;
    const int8_t* ph_ops = ph_opcodes
        + static_cast<size_t>(program) * ph_width;
    const float* ph_fargs = ph_float_arguments
        + static_cast<size_t>(program) * ph_width;
    const int16_t* ph_iargs = ph_integer_arguments
        + static_cast<size_t>(program) * ph_width;

    __shared__ int iteration_best_index;
    __shared__ float iteration_best_length;
    __shared__ int copy_global_best;
    __shared__ int copy_restart_best;
    __shared__ int restart_now;
    __shared__ int mmas_source_kind;
    __shared__ int mmas_resolved_period;
    __shared__ unsigned int bound_counts[V2_UPDATE_THREADS];
    __shared__ float basin_post_best[V2_MAX_ANTS];
    __shared__ float basin_pre_best[V2_MAX_ANTS];

    if (tid == 0) {
        if (basin_top_q > 0) {
            for (int rank = 0; rank < basin_top_q; ++rank) {
                basin_post_best[rank] = CUDART_INF_F;
                if (audit_local_search != 0) {
                    basin_pre_best[rank] = CUDART_INF_F;
                }
            }
            for (int ant = 0; ant < ants; ++ant) {
                const float post_value = colony_lengths[ant];
                const float pre_value = local_search_active != 0
                    ? length_before_workspace[
                        static_cast<size_t>(task) * ants + ant
                    ]
                    : post_value;
                int post_position = basin_top_q - 1;
                if (post_value < basin_post_best[post_position]) {
                    while (
                        post_position > 0
                        && post_value < basin_post_best[post_position - 1]
                    ) {
                        basin_post_best[post_position]
                            = basin_post_best[post_position - 1];
                        --post_position;
                    }
                    basin_post_best[post_position] = post_value;
                }
                if (audit_local_search != 0) {
                    int pre_position = basin_top_q - 1;
                    if (pre_value < basin_pre_best[pre_position]) {
                        while (
                            pre_position > 0
                            && pre_value < basin_pre_best[pre_position - 1]
                        ) {
                            basin_pre_best[pre_position]
                                = basin_pre_best[pre_position - 1];
                            --pre_position;
                        }
                        basin_pre_best[pre_position] = pre_value;
                    }
                }
            }
            float post_total = 0.0f;
            float pre_total = 0.0f;
            for (int rank = 0; rank < basin_top_q; ++rank) {
                post_total += basin_post_best[rank];
                if (audit_local_search != 0) {
                    pre_total += basin_pre_best[rank];
                }
            }
            basin_sum[task] += post_total / static_cast<float>(basin_top_q);
            if (audit_local_search != 0) {
                pre_basin_sum[task] += pre_total
                    / static_cast<float>(basin_top_q);
            }
        }
        if (audit_local_search != 0) {
            unsigned long long retained = 0;
            for (int ant = 0; ant < ants; ++ant) {
                for (int edge = 0; edge < n; ++edge) {
                    retained += origins[
                        static_cast<size_t>(ant) * n + edge
                    ] > 0;
                }
            }
            retained_edge_sum[task] += retained;
        }
        iteration_best_index = 0;
        iteration_best_length = colony_lengths[0];
        for (int ant = 1; ant < ants; ++ant) {
            if (colony_lengths[ant] < iteration_best_length) {
                iteration_best_length = colony_lengths[ant];
                iteration_best_index = ant;
            }
        }
        copy_global_best = iteration_best_length < global_best_lengths[task];
        copy_restart_best = iteration_best_length < restart_best_lengths[task];
        restart_now = 0;
        if (copy_global_best) {
            global_best_lengths[task] = iteration_best_length;
            global_best_ls_gain[task] = ls_gain_workspace[
                static_cast<size_t>(task) * ants + iteration_best_index
            ];
            global_best_iterations[task] = iteration;
            stagnation_all[task] = 0;
#if RMTGP_VARIANT == 2
            task_tau_max[task] = 1.0f
                / (rho * global_best_lengths[task]);
            if (local_search_active != 0) {
                task_tau_min[task] = task_tau_max[task]
                    / (2.0f * static_cast<float>(n));
            } else {
                const float p_x = expf(
                    logf(mmas_p_best) / static_cast<float>(n)
                );
                const float denominator = p_x
                    * static_cast<float>((candidate_size + 1) / 2);
                task_tau_min[task] = task_tau_max[task]
                    * (1.0f - p_x) / denominator;
            }
#endif
        } else {
            ++stagnation_all[task];
        }
        if (copy_restart_best) {
            restart_best_lengths[task] = iteration_best_length;
            restart_found_best[task] = iteration;
            restart_best_ls_gain[task] = ls_gain_workspace[
                static_cast<size_t>(task) * ants + iteration_best_index
            ];
        }
        mmas_resolved_period = mmas_update_period;
        mmas_source_kind = 0;
#if RMTGP_VARIANT == 2
    if (local_search_active != 0) {
        // ACOTSP 在当前 pheromone update 后才为下一轮更新 u_gb。
        // 使用上一轮的 restart age，避免在分段边界提前一轮切换。
        const int restart_age = max(
            iteration - restart_iteration[task] - 1,
            0
        );
            mmas_resolved_period = restart_age < 25
                ? 25
                : (restart_age < 75
                    ? 5
                    : (restart_age < 125
                        ? 3
                        : (restart_age < 250 ? 2 : 1)));
        }
        if (iteration % mmas_resolved_period == 0) {
            mmas_source_kind = (
                local_search_active != 0
                && mmas_resolved_period == 1
                && iteration - restart_found_best[task] > 50
            ) ? 2 : 1;
        }
#endif
    }
    __syncthreads();

    const uint16_t* iteration_best_tour = tours
        + static_cast<size_t>(iteration_best_index) * (n + 1);
    if (copy_global_best) {
        for (int city = tid; city <= n; city += blockDim.x) {
            global_best_tour[city] = iteration_best_tour[city];
        }
        for (int edge = tid; edge < n; edge += blockDim.x) {
            global_best_origin[edge] = origins[
                static_cast<size_t>(iteration_best_index) * n + edge
            ];
            if (ls_gain_semantics == 1) {
                global_best_edge_gain[edge] = edge_gains[
                    static_cast<size_t>(iteration_best_index) * n + edge
                ];
            }
        }
    }
    if (copy_restart_best) {
        for (int city = tid; city <= n; city += blockDim.x) {
            restart_tour[city] = iteration_best_tour[city];
        }
        for (int edge = tid; edge < n; edge += blockDim.x) {
            restart_best_origin[edge] = origins[
                static_cast<size_t>(iteration_best_index) * n + edge
            ];
            if (ls_gain_semantics == 1) {
                restart_best_edge_gain[edge] = edge_gains[
                    static_cast<size_t>(iteration_best_index) * n + edge
                ];
            }
        }
    }
    __syncthreads();

    if (
        ph_active[program] != 0
        && (
            ph_required_masks[program]
            & (
                (UINT64_C(1) << 3)
                | (UINT64_C(1) << 11)
            )
        ) != 0
    ) {
        for (int edge = tid; edge < n * n; edge += blockDim.x) {
            edge_frequency[edge] = 0;
        }
        __syncthreads();
        for (
            int flat_edge = tid;
            flat_edge < ants * n;
            flat_edge += blockDim.x
        ) {
            const int ant = flat_edge / n;
            const int edge = flat_edge - ant * n;
            const uint16_t* ant_tour = tours
                + static_cast<size_t>(ant) * (n + 1);
            const int first = min(
                static_cast<int>(ant_tour[edge]),
                static_cast<int>(ant_tour[edge + 1])
            );
            const int second = max(
                static_cast<int>(ant_tour[edge]),
                static_cast<int>(ant_tour[edge + 1])
            );
            atomic_increment_u8(
                edge_frequency + first * n + second
            );
        }
        __syncthreads();
    }
    if (
        ph_active[program] != 0
        && (ph_required_masks[program] & (UINT64_C(1) << 10)) != 0
    ) {
        for (int edge = tid; edge < n * n; edge += blockDim.x) {
            pre_edge_frequency[edge] = 0;
        }
        __syncthreads();
        for (
            int flat_edge = tid;
            flat_edge < ants * n;
            flat_edge += blockDim.x
        ) {
            const int ant = flat_edge / n;
            const int edge = flat_edge - ant * n;
            const uint16_t* ant_tour = pre_tours
                + static_cast<size_t>(ant) * (n + 1);
            const int first = min(
                static_cast<int>(ant_tour[edge]),
                static_cast<int>(ant_tour[edge + 1])
            );
            const int second = max(
                static_cast<int>(ant_tour[edge]),
                static_cast<int>(ant_tour[edge + 1])
            );
            atomic_increment_u8(
                pre_edge_frequency + first * n + second
            );
        }
        __syncthreads();
    }

    const int source_count = RMTGP_VARIANT == 0 ? ants : 1;
    if (tid < source_count) {
        const uint16_t* source_tour;
        float source_length;
        float source_ls_gain;
        const float* source_edge_gain;
        const int8_t* source_origin;
#if RMTGP_VARIANT == 0
        source_tour = tours + static_cast<size_t>(tid) * (n + 1);
        source_length = colony_lengths[tid];
        source_ls_gain = ls_gain_workspace[
            static_cast<size_t>(task) * ants + tid
        ];
        source_edge_gain = edge_gains + static_cast<size_t>(tid) * n;
        source_origin = origins + static_cast<size_t>(tid) * n;
#elif RMTGP_VARIANT == 1
        source_tour = global_best_tour;
        source_length = global_best_lengths[task];
        source_ls_gain = global_best_ls_gain[task];
        source_edge_gain = global_best_edge_gain;
        source_origin = global_best_origin;
#else
        if (mmas_source_kind == 0) {
            source_tour = iteration_best_tour;
            source_length = iteration_best_length;
            source_ls_gain = ls_gain_workspace[
                static_cast<size_t>(task) * ants + iteration_best_index
            ];
            source_edge_gain = edge_gains
                + static_cast<size_t>(iteration_best_index) * n;
            source_origin = origins
                + static_cast<size_t>(iteration_best_index) * n;
        } else if (mmas_source_kind == 1) {
            source_tour = restart_tour;
            source_length = restart_best_lengths[task];
            source_ls_gain = restart_best_ls_gain[task];
            source_edge_gain = restart_best_edge_gain;
            source_origin = restart_best_origin;
        } else {
            source_tour = global_best_tour;
            source_length = global_best_lengths[task];
            source_ls_gain = global_best_ls_gain[task];
            source_edge_gain = global_best_edge_gain;
            source_origin = global_best_origin;
        }
#endif
        prepare_source_deposits(
            source_tour,
            source_length,
            tid,
            n,
            ants,
            iteration,
            total_iterations,
            stagnation_all[task],
            log_heuristic,
            pheromone,
            full_nn_rank,
            node_log_eta_mean,
            pre_edge_frequency,
            edge_frequency,
            colony_lengths,
            epsilon_numeric,
            pheromone_mode,
            gamma_pheromone,
            program,
            ph_active[program] != 0,
            ph_ops,
            ph_fargs,
            ph_iargs,
            ph_lengths[program],
            ph_required_masks[program],
            source_ls_gain,
            source_edge_gain,
            source_origin,
            ls_gain_semantics,
            task_tau_min[task],
            task_tau_max[task],
            rho,
            deposits
        );
    }
    __syncthreads();

#if RMTGP_VARIANT == 1
    if (tid == 0) {
        for (int edge = 0; edge < n; ++edge) {
            const int u = global_best_tour[edge];
            const int v = global_best_tour[edge + 1];
            const float updated = (1.0f - rho) * pheromone[u * n + v]
                + rho * deposits[edge];
            pheromone[u * n + v] = updated;
            pheromone[v * n + u] = updated;
        }
    }
#else
    if (local_search_active != 0) {
        // ACOTSP 的 LS profile 只蒸发 construction candidate-list arcs。
        for (
            int entry = tid;
            entry < n * candidate_size;
            entry += blockDim.x
        ) {
            const int u = entry / candidate_size;
            const int position = entry % candidate_size;
            const int v = nearest[u * candidate_size + position];
            const int edge = u * n + v;
            const float raw = (1.0f - rho) * pheromone[edge];
#if RMTGP_VARIANT == 2
            const float bounded = fmaxf(task_tau_min[task], raw);
            pheromone[edge] = bounded;
            if (bounded != raw) {
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(
                        diagnostics
                        + static_cast<size_t>(task) * 8 + 2
                    ),
                    1ULL
                );
            }
#else
            pheromone[edge] = raw;
#endif
        }
    } else {
        for (int edge = tid; edge < n * n; edge += blockDim.x) {
            const int u = edge / n;
            const int v = edge % n;
            if (u != v) {
                pheromone[edge] *= 1.0f - rho;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int source = 0; source < source_count; ++source) {
            const uint16_t* source_tour;
#if RMTGP_VARIANT == 0
            source_tour = tours + static_cast<size_t>(source) * (n + 1);
#else
            source_tour = mmas_source_kind == 0
                ? iteration_best_tour
                : (mmas_source_kind == 1
                    ? restart_tour
                    : global_best_tour);
#endif
            for (int edge = 0; edge < n; ++edge) {
                const int u = source_tour[edge];
                const int v = source_tour[edge + 1];
                const float value = deposits[source * n + edge];
                pheromone[u * n + v] += value;
                pheromone[v * n + u] += value;
            }
        }
    }
    __syncthreads();
#if RMTGP_VARIANT == 2
    if (local_search_active == 0) {
        unsigned int local_bound_clips = 0;
        for (int edge = tid; edge < n * n; edge += blockDim.x) {
            const int u = edge / n;
            const int v = edge % n;
            if (u == v) {
                pheromone[edge] = 0.0f;
                continue;
            }
            const float raw = pheromone[edge];
            const float clipped = fminf(
                task_tau_max[task],
                fmaxf(task_tau_min[task], raw)
            );
            pheromone[edge] = clipped;
            local_bound_clips += clipped != raw;
        }
        bound_counts[tid] = local_bound_clips;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                bound_counts[tid] += bound_counts[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            diagnostics[static_cast<size_t>(task) * 8 + 2]
                += static_cast<uint64_t>(bound_counts[0]);
        }
    }
#endif
#endif
    __syncthreads();

#if RMTGP_VARIANT == 2
    if (
        iteration % mmas_branch_check_period == 0
        && tid == 0
        && iteration - restart_found_best[task] > mmas_restart_stagnation
    ) {
        float branch_sum = 0.0f;
        for (int city = 0; city < n; ++city) {
            float minimum = CUDART_INF_F;
            float maximum = -CUDART_INF_F;
            for (int position = 0; position < candidate_size; ++position) {
                const int candidate = nearest[
                    city * candidate_size + position
                ];
                const float value = pheromone[city * n + candidate];
                minimum = fminf(minimum, value);
                maximum = fmaxf(maximum, value);
            }
            const float cutoff = minimum
                + mmas_branch_lambda * (maximum - minimum);
            int branches = 0;
            for (int position = 0; position < candidate_size; ++position) {
                const int candidate = nearest[
                    city * candidate_size + position
                ];
                branches += pheromone[city * n + candidate] > cutoff;
            }
            branch_sum += static_cast<float>(branches);
        }
        const float branching_factor = branch_sum
            / (2.0f * static_cast<float>(n));
        if (branching_factor < mmas_branch_threshold) {
            restart_now = 1;
            restart_best_lengths[task] = CUDART_INF_F;
            restart_found_best[task] = iteration;
            restart_iteration[task] = iteration;
            restart_best_ls_gain[task] = -1.0f;
            ++diagnostics[static_cast<size_t>(task) * 8 + 3];
        }
    }
    __syncthreads();
    if (restart_now) {
        for (int edge = tid; edge < n * n; edge += blockDim.x) {
            pheromone[edge] = edge / n == edge % n
                ? 0.0f
                : task_tau_max[task];
        }
    }
#endif
    if (tid == 0) {
        // 训练只需要 best-so-far 曲线的均值。直接在设备端累加一个标量，
        // 避免为 population×instance×iteration 分配和回传完整曲线。
        anytime_sum[task] += global_best_lengths[task];
        if (record_anytime != 0) {
            anytime[
                static_cast<size_t>(task) * total_iterations + iteration - 1
            ] = global_best_lengths[task];
        }
    }
}
