// Blackwell 数值存储探针。该文件不参与正式 ACO 求解，只测量代表性的
// 随机读取、反量化和标量累加成本，并给出各格式的往返量化误差。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

typedef unsigned char uint8_t;

extern "C" __global__ void encode_fp64(
    const float* input,
    double* output,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = static_cast<double>(input[index]);
    }
}

extern "C" __global__ void encode_fp16(
    const float* input,
    __half* output,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2half_rn(input[index]);
    }
}

extern "C" __global__ void encode_bf16(
    const float* input,
    __nv_bfloat16* output,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2bfloat16_rn(input[index]);
    }
}

extern "C" __global__ void encode_fp8_e4m3(
    const float* input,
    uint8_t* output,
    int count,
    float inverse_scale
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __nv_fp8_e4m3(
            input[index] * inverse_scale
        ).__x;
    }
}

extern "C" __global__ void encode_nvfp4(
    const float* input,
    uint8_t* output,
    int count,
    const float* block_scales
) {
    const int packed_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int first = packed_index * 2;
    if (first < count) {
        const float inverse_scale = 1.0f
            / fmaxf(block_scales[first >> 4], 1.0e-30f);
        const uint8_t low = __nv_fp4_e2m1(
            input[first] * inverse_scale
        ).__x & 0x0fU;
        uint8_t high = 0;
        if (first + 1 < count) {
            high = __nv_fp4_e2m1(
                input[first + 1] * inverse_scale
            ).__x & 0x0fU;
        }
        output[packed_index] = low | static_cast<uint8_t>(high << 4);
    }
}

extern "C" __global__ void decode_fp64(
    const double* input,
    float* output,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = static_cast<float>(input[index]);
    }
}

extern "C" __global__ void decode_fp16(
    const __half* input,
    float* output,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __half2float(input[index]);
    }
}

extern "C" __global__ void decode_bf16(
    const __nv_bfloat16* input,
    float* output,
    int count
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __bfloat162float(input[index]);
    }
}

extern "C" __global__ void decode_fp8_e4m3(
    const uint8_t* input,
    float* output,
    int count,
    float scale
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        __nv_fp8_e4m3 value;
        value.__x = input[index];
        output[index] = static_cast<float>(value) * scale;
    }
}

extern "C" __global__ void decode_nvfp4(
    const uint8_t* input,
    float* output,
    int count,
    const float* block_scales
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const uint8_t packed = input[index >> 1];
        __nv_fp4_e2m1 value;
        value.__x = (index & 1) == 0
            ? packed & 0x0fU
            : (packed >> 4) & 0x0fU;
        output[index] = static_cast<float>(value)
            * block_scales[index >> 4];
    }
}

#define PROBE_INDEX(index, repeat, mask) \
    (((index) + (repeat) * 7919U) & (mask))

extern "C" __global__ void probe_fp64(
    const double* input,
    float* output,
    int count,
    int repetitions
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const unsigned int mask = static_cast<unsigned int>(count - 1);
    double accumulator = 0.0;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        const double value = input[PROBE_INDEX(index, repeat, mask)];
        accumulator = fma(value, 1.00000011920928955078125, accumulator);
    }
    output[index] = static_cast<float>(accumulator);
}

extern "C" __global__ void probe_fp32(
    const float* input,
    float* output,
    int count,
    int repetitions
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const unsigned int mask = static_cast<unsigned int>(count - 1);
    float accumulator = 0.0f;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        const float value = input[PROBE_INDEX(index, repeat, mask)];
        accumulator = fmaf(value, 1.00000011920928955078125f, accumulator);
    }
    output[index] = accumulator;
}

extern "C" __global__ void probe_fp16(
    const __half* input,
    float* output,
    int count,
    int repetitions
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const unsigned int mask = static_cast<unsigned int>(count - 1);
    float accumulator = 0.0f;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        const float value = __half2float(
            input[PROBE_INDEX(index, repeat, mask)]
        );
        accumulator = fmaf(value, 1.00000011920928955078125f, accumulator);
    }
    output[index] = accumulator;
}

extern "C" __global__ void probe_bf16(
    const __nv_bfloat16* input,
    float* output,
    int count,
    int repetitions
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const unsigned int mask = static_cast<unsigned int>(count - 1);
    float accumulator = 0.0f;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        const float value = __bfloat162float(
            input[PROBE_INDEX(index, repeat, mask)]
        );
        accumulator = fmaf(value, 1.00000011920928955078125f, accumulator);
    }
    output[index] = accumulator;
}

extern "C" __global__ void probe_fp8_e4m3(
    const uint8_t* input,
    float* output,
    int count,
    int repetitions,
    float scale
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const unsigned int mask = static_cast<unsigned int>(count - 1);
    float accumulator = 0.0f;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        __nv_fp8_e4m3 packed;
        packed.__x = input[PROBE_INDEX(index, repeat, mask)];
        const float value = static_cast<float>(packed) * scale;
        accumulator = fmaf(value, 1.00000011920928955078125f, accumulator);
    }
    output[index] = accumulator;
}

extern "C" __global__ void probe_nvfp4(
    const uint8_t* input,
    float* output,
    int count,
    int repetitions,
    const float* block_scales
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned int>(count)) {
        return;
    }
    const unsigned int mask = static_cast<unsigned int>(count - 1);
    float accumulator = 0.0f;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        const unsigned int selected = PROBE_INDEX(index, repeat, mask);
        const uint8_t byte = input[selected >> 1];
        __nv_fp4_e2m1 packed;
        packed.__x = (selected & 1U) == 0U
            ? byte & 0x0fU
            : (byte >> 4) & 0x0fU;
        const float value = static_cast<float>(packed)
            * block_scales[selected >> 4];
        accumulator = fmaf(value, 1.00000011920928955078125f, accumulator);
    }
    output[index] = accumulator;
}
