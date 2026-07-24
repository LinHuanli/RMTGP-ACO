// RMTGP–ACO 融合 CUDA 内核。
//
// 一个 block 独占一个 (GP program, TSP instance) 任务；ACO 的完整迭代
// 留在设备端。32 只蚂蚁分别由一个线程负责，额外线程只参与矩阵初始化和
// 蒸发。该布局避免 Python 循环、小 kernel 和逐步 host synchronization。

#include <cuda_runtime.h>
#include <math_constants.h>

// NVRTC 在部分集群环境中没有可见的系统 stdint.h；固定宽度类型在这里
// 显式定义，ABI 与 NumPy/CuPy 传入 dtype 对齐。
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
#define UINT64_C(value) value##ULL

namespace {

constexpr int MAX_ANTS = 32;
constexpr int MAX_STACK = 32;

enum Opcode : int8_t {
    OP_CONST = 0,
    OP_TERMINAL = 1,
    OP_ADD = 2,
    OP_SUB = 3,
    OP_MUL = 4,
    OP_PDIV = 5,
    OP_PDIV1 = 6,
    OP_MIN = 7,
    OP_MAX = 8,
    OP_ABS = 9,
    OP_NEG = 10,
};

struct CandidateStats {
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

__device__ __forceinline__ float sanitize(float value) {
    if (isnan(value)) {
        return 0.0f;
    }
    return fminf(10.0f, fmaxf(-10.0f, value));
}

__device__ __forceinline__ float softplus_clipped(float value) {
    value = fminf(20.0f, fmaxf(-20.0f, value));
    return log1pf(expf(value));
}

__device__ __forceinline__ uint64_t mix64(uint64_t value) {
    value += UINT64_C(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

__device__ __forceinline__ float counter_uniform(
    uint64_t seed,
    uint64_t instance_key,
    int iteration,
    int ant,
    int step,
    int stream_kind
) {
    uint64_t value = seed ^ instance_key;
    value ^= static_cast<uint64_t>(iteration + 1)
        * UINT64_C(0xD2B74407B1CE6E93);
    value ^= static_cast<uint64_t>(ant + 1)
        * UINT64_C(0xCA5A826395121157);
    value ^= static_cast<uint64_t>(step + 1)
        * UINT64_C(0x9E3779B185EBCA87);
    value ^= static_cast<uint64_t>(stream_kind + 1)
        * UINT64_C(0x94D049BB133111EB);
    // 搜索状态明确采用 FP32；取高 24 bit 可保证跨设备可复现。
    return static_cast<float>(mix64(value) >> 40) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ bool is_visited(
    const uint64_t* visited,
    int city
) {
    return (visited[city >> 6] & (UINT64_C(1) << (city & 63))) != 0;
}

__device__ __forceinline__ void mark_visited(
    uint64_t* visited,
    int city
) {
    visited[city >> 6] |= UINT64_C(1) << (city & 63);
}

__device__ __forceinline__ float base_score(
    const float* pheromone,
    const float* heuristic,
    int n,
    int current,
    int city,
    float alpha,
    float beta
) {
    const int edge = current * n + city;
    const float tau = pheromone[edge];
    const float eta = heuristic[edge];
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

__device__ float evaluate_program(
    const int8_t* opcodes,
    const float* float_arguments,
    const int16_t* integer_arguments,
    int length,
    const float* terminals
) {
    float stack[MAX_STACK];
    int top = 0;
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

__device__ CandidateStats candidate_stats(
    const float* distances,
    const float* heuristic,
    const float* log_heuristic,
    const float* pheromone,
    const uint16_t* nearest,
    const uint64_t* visited,
    int n,
    int candidate_size,
    int current,
    bool fallback,
    float alpha,
    float beta,
    float epsilon_numeric,
    uint64_t required_mask
) {
    CandidateStats stats{};
    float log_tau_sum = 0.0f;
    float log_tau_sq = 0.0f;
    float log_eta_sum = 0.0f;
    float log_eta_sq = 0.0f;
    float tau_sum = 0.0f;
    float distance_sum = 0.0f;
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

    const int limit = fallback ? n : candidate_size;
    for (int position = 0; position < limit; ++position) {
        const int city = fallback ? position : nearest[position];
        if (is_visited(visited, city)) {
            continue;
        }
        const int edge = current * n + city;
        if (need_log_tau) {
            const float log_tau = logf(fmaxf(
                pheromone[edge],
                epsilon_numeric
            ));
            log_tau_sum += log_tau;
            log_tau_sq += log_tau * log_tau;
        }
        if (need_log_eta) {
            const float log_eta = log_heuristic[edge];
            log_eta_sum += log_eta;
            log_eta_sq += log_eta * log_eta;
        }
        if (need_tau_mean) {
            tau_sum += pheromone[edge];
        }
        if (need_distance_mean) {
            distance_sum += distances[edge];
        }
        stats.base_total += base_score(
            pheromone,
            heuristic,
            n,
            current,
            city,
            alpha,
            beta
        );
        ++stats.count;
    }
    const float inverse = 1.0f / static_cast<float>(stats.count);
    if (need_log_tau) {
        stats.log_tau_mean = log_tau_sum * inverse;
        stats.log_tau_std = sqrtf(fmaxf(
            0.0f,
            log_tau_sq * inverse - stats.log_tau_mean * stats.log_tau_mean
        ));
    }
    if (need_log_eta) {
        stats.log_eta_mean = log_eta_sum * inverse;
        stats.log_eta_std = sqrtf(fmaxf(
            0.0f,
            log_eta_sq * inverse - stats.log_eta_mean * stats.log_eta_mean
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
            for (int position = 0; position < limit; ++position) {
                const int city = fallback ? position : nearest[position];
                if (is_visited(visited, city)) {
                    continue;
                }
                const float probability = base_score(
                    pheromone,
                    heuristic,
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
            stats.entropy = 2.0f * entropy / logf(
                static_cast<float>(stats.count)
            ) - 1.0f;
        }
    }
    return stats;
}

__device__ float transition_score(
    const float* distances,
    const float* heuristic,
    const float* log_heuristic,
    const float* pheromone,
    const uint64_t* visited,
    int n,
    int current,
    int city,
    int ordinal,
    bool fallback,
    float alpha,
    float beta,
    float epsilon_numeric,
    int transition_mode,
    float gamma_transition,
    bool program_active,
    const int8_t* opcodes,
    const float* float_arguments,
    const int16_t* integer_arguments,
    int program_length,
    uint64_t required_mask,
    int construction_step,
    int iteration,
    int stagnation,
    int total_iterations,
    const CandidateStats& stats
) {
    const int edge = current * n + city;
    const float baseline = base_score(
        pheromone,
        heuristic,
        n,
        current,
        city,
        alpha,
        beta
    );
    if (!program_active) {
        return baseline;
    }

    float terminals[14];
    if ((required_mask & (UINT64_C(1) << 0)) != 0) {
        terminals[0] = tanhf(
            (logf(fmaxf(pheromone[edge], epsilon_numeric))
                - stats.log_tau_mean)
            / (stats.log_tau_std + 1.0e-8f)
        );
    }
    if ((required_mask & (UINT64_C(1) << 1)) != 0) {
        terminals[1] = tanhf(
            (log_heuristic[edge] - stats.log_eta_mean)
            / (stats.log_eta_std + 1.0e-8f)
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
        int rank = ordinal;
        if (fallback) {
            rank = 0;
            const float value = distances[edge];
            for (int other = 0; other < n; ++other) {
                if (is_visited(visited, other)) {
                    continue;
                }
                const float other_value = distances[current * n + other];
                if (other_value < value || (
                    other_value == value && other < city
                )) {
                    ++rank;
                }
            }
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
    terminals[9] = distances[edge];
    terminals[10] = stats.tau_mean;
    terminals[11] = stats.distance_mean;
    terminals[12] = static_cast<float>(n);
    terminals[13] = static_cast<float>(stats.count);

    const float raw = evaluate_program(
        opcodes,
        float_arguments,
        integer_arguments,
        program_length,
        terminals
    );
    if (transition_mode == 1) {
        return softplus_clipped(raw) + epsilon_numeric;
    }
    return baseline * (1.0f + gamma_transition * tanhf(raw));
}

__device__ int choose_city(
    const float* distances,
    const float* heuristic,
    const float* log_heuristic,
    const uint16_t* nearest,
    const float* pheromone,
    const uint64_t* visited,
    int n,
    int candidate_size,
    int current,
    int variant,
    float alpha,
    float beta,
    float q0,
    float epsilon_numeric,
    int transition_mode,
    float gamma_transition,
    bool program_active,
    const int8_t* opcodes,
    const float* float_arguments,
    const int16_t* integer_arguments,
    int program_length,
    uint64_t required_mask,
    uint64_t seed,
    uint64_t instance_key,
    int ant,
    int construction_step,
    int iteration,
    int stagnation,
    int total_iterations,
    uint64_t& candidate_fallback_count,
    uint64_t& uniform_fallback_count
) {
    int count = 0;
    for (int position = 0; position < candidate_size; ++position) {
        if (!is_visited(visited, nearest[position])) {
            ++count;
        }
    }
    const bool fallback = count == 0;
    if (fallback) {
        ++candidate_fallback_count;
        ++uniform_fallback_count;
    }
    const CandidateStats stats = candidate_stats(
        distances,
        heuristic,
        log_heuristic,
        pheromone,
        nearest,
        visited,
        n,
        candidate_size,
        current,
        fallback,
        alpha,
        beta,
        epsilon_numeric,
        program_active ? required_mask : UINT64_C(0)
    );

    const int limit = fallback ? n : candidate_size;
    int greedy_city = -1;
    float greedy_score = -CUDART_INF_F;
    float total = 0.0f;
    int ordinal = 0;
    for (int position = 0; position < limit; ++position) {
        const int city = fallback ? position : nearest[position];
        if (is_visited(visited, city)) {
            continue;
        }
        const float score = transition_score(
            distances,
            heuristic,
            log_heuristic,
            pheromone,
            visited,
            n,
            current,
            city,
            ordinal,
            fallback,
            alpha,
            beta,
            epsilon_numeric,
            transition_mode,
            gamma_transition,
            program_active,
            opcodes,
            float_arguments,
            integer_arguments,
            program_length,
            required_mask,
            construction_step,
            iteration,
            stagnation,
            total_iterations,
            stats
        );
        if (score > greedy_score) {
            greedy_score = score;
            greedy_city = city;
        }
        total += score;
        ++ordinal;
    }
    if (fallback) {
        return greedy_city;
    }

    const bool base_uniform = stats.base_total <= epsilon_numeric;
    if (variant == 1 && counter_uniform(
        seed,
        instance_key,
        iteration,
        ant,
        construction_step,
        2
    ) <= q0) {
        if (base_uniform) {
            ++uniform_fallback_count;
        }
        return greedy_city;
    }

    const bool residual_uniform = total <= epsilon_numeric;
    if (base_uniform || residual_uniform) {
        ++uniform_fallback_count;
    }
    const float uniform = counter_uniform(
        seed,
        instance_key,
        iteration,
        ant,
        construction_step,
        3
    );
    if (residual_uniform) {
        const int selected = min(
            static_cast<int>(uniform * static_cast<float>(stats.count)),
            stats.count - 1
        );
        ordinal = 0;
        for (int position = 0; position < limit; ++position) {
            const int city = fallback ? position : nearest[position];
            if (!is_visited(visited, city)) {
                if (ordinal == selected) {
                    return city;
                }
                ++ordinal;
            }
        }
    }

    const float threshold = uniform * total;
    float cumulative = 0.0f;
    int last_city = greedy_city;
    ordinal = 0;
    for (int position = 0; position < limit; ++position) {
        const int city = fallback ? position : nearest[position];
        if (is_visited(visited, city)) {
            continue;
        }
        last_city = city;
        cumulative += transition_score(
            distances,
            heuristic,
            log_heuristic,
            pheromone,
            visited,
            n,
            current,
            city,
            ordinal,
            fallback,
            alpha,
            beta,
            epsilon_numeric,
            transition_mode,
            gamma_transition,
            program_active,
            opcodes,
            float_arguments,
            integer_arguments,
            program_length,
            required_mask,
            construction_step,
            iteration,
            stagnation,
            total_iterations,
            stats
        );
        if (cumulative >= threshold) {
            return city;
        }
        ++ordinal;
    }
    return last_city;
}

__device__ void apply_acs_local(
    float* pheromone,
    const int* edge_u,
    const int* edge_v,
    int ants,
    int n,
    float tau0,
    float xi
) {
    for (int ant = 0; ant < ants; ++ant) {
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
        if (!first_occurrence) {
            continue;
        }
        const float factor = powf(1.0f - xi, static_cast<float>(multiplicity));
        const int forward = first * n + second;
        const float updated = factor * pheromone[forward]
            + (1.0f - factor) * tau0;
        pheromone[forward] = updated;
        pheromone[second * n + first] = updated;
    }
}

__device__ float pheromone_terminal(
    int terminal,
    int edge,
    int u,
    int v,
    int n,
    int ants,
    int iteration,
    int total_iterations,
    int stagnation,
    const float* log_heuristic,
    const float* pheromone,
    const uint16_t* full_nn_rank,
    const float* node_log_eta_mean,
    const uint8_t* edge_frequency,
    float edge_eta_mean,
    float edge_eta_std,
    float edge_tau_mean,
    float edge_tau_std,
    float source_quality,
    float epsilon_numeric
) {
    if (terminal == 0) {
        const float raw = log_heuristic[u * n + v]
            - 0.5f * (node_log_eta_mean[u] + node_log_eta_mean[v]);
        return tanhf((raw - edge_eta_mean) / (edge_eta_std + 1.0e-8f));
    }
    if (terminal == 1) {
        const float raw = logf(fmaxf(pheromone[u * n + v], epsilon_numeric));
        return tanhf((raw - edge_tau_mean) / (edge_tau_std + 1.0e-8f));
    }
    if (terminal == 2) {
        const float denominator = static_cast<float>(max(n - 2, 1));
        const float rank_uv = static_cast<float>(full_nn_rank[u * n + v]);
        const float rank_vu = static_cast<float>(full_nn_rank[v * n + u]);
        const float normalized_uv = 1.0f - 2.0f * (rank_uv - 1.0f)
            / denominator;
        const float normalized_vu = 1.0f - 2.0f * (rank_vu - 1.0f)
            / denominator;
        return 0.5f * (normalized_uv + normalized_vu);
    }
    if (terminal == 3) {
        const int first = min(u, v);
        const int second = max(u, v);
        return 2.0f * static_cast<float>(
            edge_frequency[first * n + second]
        ) / static_cast<float>(ants) - 1.0f;
    }
    if (terminal == 4) {
        return source_quality;
    }
    if (terminal == 5) {
        return 2.0f * static_cast<float>(iteration - 1)
            / static_cast<float>(max(total_iterations - 1, 1)) - 1.0f;
    }
    return 2.0f * fminf(
        static_cast<float>(stagnation) / static_cast<float>(total_iterations),
        1.0f
    ) - 1.0f;
}

__device__ void prepare_source_deposits(
    const uint16_t* source_tour,
    float source_length,
    int source_index,
    int n,
    int ants,
    int iteration,
    int total_iterations,
    int stagnation,
    const float* log_heuristic,
    const float* pheromone,
    const uint16_t* full_nn_rank,
    const float* node_log_eta_mean,
    const uint8_t* edge_frequency,
    const float* colony_lengths,
    float epsilon_numeric,
    int pheromone_mode,
    float gamma_pheromone,
    bool program_active,
    const int8_t* opcodes,
    const float* float_arguments,
    const int16_t* integer_arguments,
    int program_length,
    uint64_t required_mask,
    float* deposits
) {
    const float base_deposit = 1.0f / source_length;
    if (!program_active) {
        for (int edge = 0; edge < n; ++edge) {
            deposits[source_index * n + edge] = base_deposit;
        }
        return;
    }

    float edge_eta_sum = 0.0f;
    float edge_eta_sq = 0.0f;
    float edge_tau_sum = 0.0f;
    float edge_tau_sq = 0.0f;
    const bool need_edge_eta = (
        required_mask & (UINT64_C(1) << 0)
    ) != 0;
    const bool need_edge_tau = (
        required_mask & (UINT64_C(1) << 1)
    ) != 0;
    for (int edge = 0; edge < n; ++edge) {
        const int u = source_tour[edge];
        const int v = source_tour[edge + 1];
        if (need_edge_eta) {
            const float edge_eta = log_heuristic[u * n + v]
                - 0.5f * (node_log_eta_mean[u] + node_log_eta_mean[v]);
            edge_eta_sum += edge_eta;
            edge_eta_sq += edge_eta * edge_eta;
        }
        if (need_edge_tau) {
            const float edge_tau = logf(fmaxf(
                pheromone[u * n + v],
                epsilon_numeric
            ));
            edge_tau_sum += edge_tau;
            edge_tau_sq += edge_tau * edge_tau;
        }
    }
    const float inverse_n = 1.0f / static_cast<float>(n);
    const float edge_eta_mean = need_edge_eta
        ? edge_eta_sum * inverse_n
        : 0.0f;
    const float edge_tau_mean = need_edge_tau
        ? edge_tau_sum * inverse_n
        : 0.0f;
    const float edge_eta_std = need_edge_eta
        ? sqrtf(fmaxf(
            0.0f,
            edge_eta_sq * inverse_n - edge_eta_mean * edge_eta_mean
        ))
        : 0.0f;
    const float edge_tau_std = need_edge_tau
        ? sqrtf(fmaxf(
            0.0f,
            edge_tau_sq * inverse_n - edge_tau_mean * edge_tau_mean
        ))
        : 0.0f;

    float colony_mean = 0.0f;
    const bool need_source_quality = (
        required_mask & (UINT64_C(1) << 4)
    ) != 0;
    if (need_source_quality) {
        for (int ant = 0; ant < ants; ++ant) {
            colony_mean += colony_lengths[ant];
        }
        colony_mean /= static_cast<float>(ants);
    }
    float colony_variance = 0.0f;
    if (need_source_quality) {
        for (int ant = 0; ant < ants; ++ant) {
            const float centered = colony_lengths[ant] - colony_mean;
            colony_variance += centered * centered;
        }
        colony_variance /= static_cast<float>(ants);
    }
    const float source_quality = need_source_quality
        ? tanhf(
            (colony_mean - source_length)
            / (sqrtf(colony_variance) + epsilon_numeric)
        )
        : 0.0f;

    const float budget = static_cast<float>(n) / source_length;
    float total = 0.0f;
    for (int edge = 0; edge < n; ++edge) {
        float deposit = base_deposit;
        const int u = source_tour[edge];
        const int v = source_tour[edge + 1];
        float terminals[7];
        for (int terminal = 0; terminal < 7; ++terminal) {
            if ((required_mask & (UINT64_C(1) << terminal)) != 0) {
                terminals[terminal] = pheromone_terminal(
                    terminal,
                    edge,
                    u,
                    v,
                    n,
                    ants,
                    iteration,
                    total_iterations,
                    stagnation,
                    log_heuristic,
                    pheromone,
                    full_nn_rank,
                    node_log_eta_mean,
                    edge_frequency,
                    edge_eta_mean,
                    edge_eta_std,
                    edge_tau_mean,
                    edge_tau_std,
                    source_quality,
                    epsilon_numeric
                );
            }
        }
        const float raw = evaluate_program(
            opcodes,
            float_arguments,
            integer_arguments,
            program_length,
            terminals
        );
        if (pheromone_mode == 3) {
            deposit = base_deposit * (
                softplus_clipped(raw) + epsilon_numeric
            );
        } else if (pheromone_mode == 2) {
            deposit = fmaxf(
                base_deposit
                    + gamma_pheromone * base_deposit * tanhf(raw),
                epsilon_numeric
            );
        } else {
            deposit = base_deposit * (
                1.0f + gamma_pheromone * tanhf(raw)
            );
        }
        deposits[source_index * n + edge] = deposit;
        total += deposit;
    }
    if (pheromone_mode == 0) {
        const float scale = budget / fmaxf(total, epsilon_numeric);
        for (int edge = 0; edge < n; ++edge) {
            deposits[source_index * n + edge] *= scale;
        }
    }
}

}  // namespace

extern "C" __global__ void fused_aco(
    const float* distances_all,
    const float* heuristic_all,
    const float* log_heuristic_all,
    const uint16_t* nearest_all,
    const uint16_t* full_nn_rank_all,
    const float* node_log_eta_mean_all,
    const float* initial_tau0,
    const float* initial_tau_min,
    const float* initial_tau_max,
    const int8_t* tr_opcodes,
    const float* tr_float_arguments,
    const int16_t* tr_integer_arguments,
    const int16_t* tr_lengths,
    const uint64_t* tr_required_masks,
    const uint8_t* tr_active,
    int tr_width,
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
    int iterations,
    int variant,
    float alpha,
    float beta,
    float rho,
    float q0,
    float xi,
    float gamma_transition,
    float gamma_pheromone,
    int transition_mode,
    int pheromone_mode,
    float epsilon_numeric,
    int mmas_update_period,
    float mmas_p_best,
    int mmas_branch_check_period,
    float mmas_branch_lambda,
    float mmas_branch_threshold,
    int mmas_restart_stagnation,
    uint64_t seed,
    const uint64_t* instance_keys,
    float* pheromone_workspace,
    uint16_t* tour_workspace,
    uint64_t* visited_workspace,
    float* deposit_workspace,
    uint8_t* edge_frequency_workspace,
    uint16_t* restart_tour_workspace,
    uint16_t* best_tours,
    float* best_lengths,
    int32_t* best_iterations,
    float* anytime,
    int record_anytime,
    uint64_t* diagnostics
) {
    const int task = blockIdx.x;
    if (task >= task_count || ants > MAX_ANTS) {
        return;
    }
    const int tid = threadIdx.x;
    const int program = task_program[task];
    const int instance = task_instance[task];
    const int words = (n + 63) / 64;

    const float* distances = distances_all
        + static_cast<size_t>(instance) * n * n;
    const float* heuristic = heuristic_all
        + static_cast<size_t>(instance) * n * n;
    const float* log_heuristic = log_heuristic_all
        + static_cast<size_t>(instance) * n * n;
    const uint16_t* nearest = nearest_all
        + static_cast<size_t>(instance) * n * candidate_size;
    const uint16_t* full_nn_rank = full_nn_rank_all
        + static_cast<size_t>(instance) * n * n;
    const float* node_log_eta_mean = node_log_eta_mean_all
        + static_cast<size_t>(instance) * n;
    const uint64_t instance_key = instance_keys[instance];

    float* pheromone = pheromone_workspace
        + static_cast<size_t>(task) * n * n;
    uint16_t* tours = tour_workspace
        + static_cast<size_t>(task) * ants * (n + 1);
    uint64_t* visited = visited_workspace
        + static_cast<size_t>(task) * ants * words;
    float* deposits = deposit_workspace
        + static_cast<size_t>(task) * ants * n;
    uint8_t* edge_frequency = edge_frequency_workspace
        + static_cast<size_t>(task) * n * n;
    uint16_t* restart_tour = restart_tour_workspace
        + static_cast<size_t>(task) * (n + 1);
    uint16_t* global_best_tour = best_tours
        + static_cast<size_t>(task) * (n + 1);

    const int8_t* tr_ops = tr_opcodes
        + static_cast<size_t>(program) * tr_width;
    const float* tr_fargs = tr_float_arguments
        + static_cast<size_t>(program) * tr_width;
    const int16_t* tr_iargs = tr_integer_arguments
        + static_cast<size_t>(program) * tr_width;
    const int8_t* ph_ops = ph_opcodes
        + static_cast<size_t>(program) * ph_width;
    const float* ph_fargs = ph_float_arguments
        + static_cast<size_t>(program) * ph_width;
    const int16_t* ph_iargs = ph_integer_arguments
        + static_cast<size_t>(program) * ph_width;

    __shared__ int edge_u[MAX_ANTS];
    __shared__ int edge_v[MAX_ANTS];
    __shared__ float colony_lengths[MAX_ANTS];
    __shared__ float global_best_length;
    __shared__ float restart_best_length;
    __shared__ float tau_min;
    __shared__ float tau_max;
    __shared__ int global_best_iteration;
    __shared__ int iteration_best_index;
    __shared__ int stagnation;
    __shared__ int restart_found_best;

    const float tau0 = initial_tau0[instance];
    for (int edge = tid; edge < n * n; edge += blockDim.x) {
        pheromone[edge] = edge / n == edge % n ? 0.0f : tau0;
        edge_frequency[edge] = 0;
    }
    for (int index = tid; index < 4; index += blockDim.x) {
        diagnostics[static_cast<size_t>(task) * 4 + index] = 0;
    }
    if (tid == 0) {
        global_best_length = CUDART_INF_F;
        restart_best_length = CUDART_INF_F;
        tau_min = initial_tau_min[instance];
        tau_max = initial_tau_max[instance];
        global_best_iteration = 0;
        stagnation = 0;
        restart_found_best = 0;
    }
    __syncthreads();

    uint64_t local_candidate_fallbacks = 0;
    uint64_t local_uniform_fallbacks = 0;
    uint64_t local_bound_clips = 0;

    for (int iteration = 1; iteration <= iterations; ++iteration) {
        for (int index = tid; index < ants * words; index += blockDim.x) {
            visited[index] = 0;
        }
        if (tid < ants) {
            uint64_t* ant_visited = visited + tid * words;
            const int start = min(
                static_cast<int>(
                    counter_uniform(
                        seed,
                        instance_key,
                        iteration,
                        tid,
                        0,
                        1
                    ) * static_cast<float>(n)
                ),
                n - 1
            );
            tours[static_cast<size_t>(tid) * (n + 1)] = start;
            mark_visited(ant_visited, start);
        }
        __syncthreads();

        for (int step = 1; step < n; ++step) {
            if (tid < ants) {
                uint16_t* ant_tour = tours
                    + static_cast<size_t>(tid) * (n + 1);
                uint64_t* ant_visited = visited + tid * words;
                const int current = ant_tour[step - 1];
                const int chosen = choose_city(
                    distances,
                    heuristic,
                    log_heuristic,
                    nearest + current * candidate_size,
                    pheromone,
                    ant_visited,
                    n,
                    candidate_size,
                    current,
                    variant,
                    alpha,
                    beta,
                    q0,
                    epsilon_numeric,
                    transition_mode,
                    gamma_transition,
                    tr_active[program] != 0,
                    tr_ops,
                    tr_fargs,
                    tr_iargs,
                    tr_lengths[program],
                    tr_required_masks[program],
                    seed,
                    instance_key,
                    tid,
                    step,
                    iteration,
                    stagnation,
                    iterations,
                    local_candidate_fallbacks,
                    local_uniform_fallbacks
                );
                ant_tour[step] = static_cast<uint16_t>(chosen);
                edge_u[tid] = current;
                edge_v[tid] = chosen;
            }
            __syncthreads();
            if (variant == 1 && tid == 0) {
                apply_acs_local(
                    pheromone,
                    edge_u,
                    edge_v,
                    ants,
                    n,
                    tau0,
                    xi
                );
            }
            __syncthreads();
            if (tid < ants) {
                mark_visited(
                    visited + tid * words,
                    tours[static_cast<size_t>(tid) * (n + 1) + step]
                );
            }
            __syncthreads();
        }

        if (tid < ants) {
            uint16_t* ant_tour = tours
                + static_cast<size_t>(tid) * (n + 1);
            ant_tour[n] = ant_tour[0];
            edge_u[tid] = ant_tour[n - 1];
            edge_v[tid] = ant_tour[0];
        }
        __syncthreads();
        if (variant == 1 && tid == 0) {
            apply_acs_local(
                pheromone,
                edge_u,
                edge_v,
                ants,
                n,
                tau0,
                xi
            );
        }
        __syncthreads();

        if (tid < ants) {
            const uint16_t* ant_tour = tours
                + static_cast<size_t>(tid) * (n + 1);
            float length = 0.0f;
            for (int edge = 0; edge < n; ++edge) {
                length += distances[
                    static_cast<int>(ant_tour[edge]) * n
                    + static_cast<int>(ant_tour[edge + 1])
                ];
            }
            colony_lengths[tid] = length;
        }
        __syncthreads();

        if (tid == 0) {
            iteration_best_index = 0;
            float iteration_best_length = colony_lengths[0];
            for (int ant = 1; ant < ants; ++ant) {
                if (colony_lengths[ant] < iteration_best_length) {
                    iteration_best_length = colony_lengths[ant];
                    iteration_best_index = ant;
                }
            }
            const uint16_t* iteration_best_tour = tours
                + static_cast<size_t>(iteration_best_index) * (n + 1);
            if (iteration_best_length < global_best_length) {
                global_best_length = iteration_best_length;
                global_best_iteration = iteration;
                stagnation = 0;
                for (int city = 0; city <= n; ++city) {
                    global_best_tour[city] = iteration_best_tour[city];
                }
                if (variant == 2) {
                    const float p_x = expf(logf(mmas_p_best)
                        / static_cast<float>(n));
                    const float denominator = p_x
                        * static_cast<float>((candidate_size + 1) / 2);
                    tau_max = 1.0f / (rho * global_best_length);
                    tau_min = tau_max * (1.0f - p_x) / denominator;
                }
            } else {
                ++stagnation;
            }
            if (iteration_best_length < restart_best_length) {
                restart_best_length = iteration_best_length;
                restart_found_best = iteration;
                for (int city = 0; city <= n; ++city) {
                    restart_tour[city] = iteration_best_tour[city];
                }
            }

            if (
                ph_active[program] != 0
                && (ph_required_masks[program] & (UINT64_C(1) << 3)) != 0
            ) {
                for (int edge = 0; edge < n * n; ++edge) {
                    edge_frequency[edge] = 0;
                }
                for (int ant = 0; ant < ants; ++ant) {
                    const uint16_t* ant_tour = tours
                        + static_cast<size_t>(ant) * (n + 1);
                    for (int edge = 0; edge < n; ++edge) {
                        const int first = min(
                            static_cast<int>(ant_tour[edge]),
                            static_cast<int>(ant_tour[edge + 1])
                        );
                        const int second = max(
                            static_cast<int>(ant_tour[edge]),
                            static_cast<int>(ant_tour[edge + 1])
                        );
                        ++edge_frequency[first * n + second];
                    }
                }
            }

            const int source_count = variant == 0 ? ants : 1;
            for (int source = 0; source < source_count; ++source) {
                const uint16_t* source_tour;
                float source_length;
                if (variant == 0) {
                    source_tour = tours
                        + static_cast<size_t>(source) * (n + 1);
                    source_length = colony_lengths[source];
                } else if (variant == 1) {
                    source_tour = global_best_tour;
                    source_length = global_best_length;
                } else if (iteration % mmas_update_period != 0) {
                    source_tour = tours
                        + static_cast<size_t>(iteration_best_index) * (n + 1);
                    source_length = colony_lengths[iteration_best_index];
                } else {
                    source_tour = restart_tour;
                    source_length = restart_best_length;
                }
                prepare_source_deposits(
                    source_tour,
                    source_length,
                    source,
                    n,
                    ants,
                    iteration,
                    iterations,
                    stagnation,
                    log_heuristic,
                    pheromone,
                    full_nn_rank,
                    node_log_eta_mean,
                    edge_frequency,
                    colony_lengths,
                    epsilon_numeric,
                    pheromone_mode,
                    gamma_pheromone,
                    ph_active[program] != 0,
                    ph_ops,
                    ph_fargs,
                    ph_iargs,
                    ph_lengths[program],
                    ph_required_masks[program],
                    deposits
                );
            }

            if (variant == 1) {
                for (int edge = 0; edge < n; ++edge) {
                    const int u = global_best_tour[edge];
                    const int v = global_best_tour[edge + 1];
                    const float updated = (1.0f - rho) * pheromone[u * n + v]
                        + rho * deposits[edge];
                    pheromone[u * n + v] = updated;
                    pheromone[v * n + u] = updated;
                }
            }
        }
        __syncthreads();

        if (variant != 1) {
            for (int edge = tid; edge < n * n; edge += blockDim.x) {
                const int u = edge / n;
                const int v = edge % n;
                if (u != v) {
                    pheromone[edge] *= 1.0f - rho;
                }
            }
            __syncthreads();
            if (tid == 0) {
                const int source_count = variant == 0 ? ants : 1;
                for (int source = 0; source < source_count; ++source) {
                    const uint16_t* source_tour;
                    if (variant == 0) {
                        source_tour = tours
                            + static_cast<size_t>(source) * (n + 1);
                    } else if (iteration % mmas_update_period != 0) {
                        source_tour = tours
                            + static_cast<size_t>(iteration_best_index) * (n + 1);
                    } else {
                        source_tour = restart_tour;
                    }
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
            if (variant == 2) {
                for (int edge = tid; edge < n * n; edge += blockDim.x) {
                    const int u = edge / n;
                    const int v = edge % n;
                    if (u == v) {
                        pheromone[edge] = 0.0f;
                        continue;
                    }
                    const float raw = pheromone[edge];
                    const float clipped = fminf(tau_max, fmaxf(tau_min, raw));
                    pheromone[edge] = clipped;
                    if (clipped != raw) {
                        ++local_bound_clips;
                    }
                }
            }
            __syncthreads();
        }

        if (
            variant == 2
            && iteration % mmas_branch_check_period == 0
            && tid == 0
            && iteration - restart_found_best > mmas_restart_stagnation
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
                const float cutoff = minimum + mmas_branch_lambda
                    * (maximum - minimum);
                int branches = 0;
                for (int position = 0; position < candidate_size; ++position) {
                    const int candidate = nearest[
                        city * candidate_size + position
                    ];
                    if (pheromone[city * n + candidate] > cutoff) {
                        ++branches;
                    }
                }
                branch_sum += static_cast<float>(branches);
            }
            const float branching_factor = branch_sum
                / (2.0f * static_cast<float>(n));
            if (branching_factor < mmas_branch_threshold) {
                for (int edge = 0; edge < n * n; ++edge) {
                    pheromone[edge] = edge / n == edge % n ? 0.0f : tau_max;
                }
                restart_best_length = CUDART_INF_F;
                restart_found_best = iteration;
                ++diagnostics[static_cast<size_t>(task) * 4 + 3];
            }
        }
        __syncthreads();

        if (tid == 0 && record_anytime != 0) {
            anytime[static_cast<size_t>(task) * iterations + iteration - 1]
                = global_best_length;
        }
        __syncthreads();
    }

    if (tid < ants) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 4
            ),
            static_cast<unsigned long long>(local_candidate_fallbacks)
        );
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 4 + 1
            ),
            static_cast<unsigned long long>(local_uniform_fallbacks)
        );
    }
    if (local_bound_clips != 0) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                diagnostics + static_cast<size_t>(task) * 4 + 2
            ),
            static_cast<unsigned long long>(local_bound_clips)
        );
    }
    if (tid == 0) {
        best_lengths[task] = global_best_length;
        best_iterations[task] = global_best_iteration;
    }
}
