// 单树 FP32 CUDA 公共合同：明确舍入、保护规则和逻辑随机流。
#include <cuda_runtime.h>
#include <math_constants.h>
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
#define UINT64_C(x) x##ULL
#ifndef RMTGP_GENERATED_GP
#define RMTGP_GENERATED_GP 0
#endif
enum Opcode : int8_t { OP_CONST=0, OP_TERMINAL=1, OP_ADD=2, OP_SUB=3,
    OP_MUL=4, OP_PDIV=5, OP_PDIV1=6, OP_MIN=7, OP_MAX=8, OP_ABS=9, OP_NEG=10 };

__device__ __forceinline__ float sanitize(float x) {
    return isnan(x) ? 0.0f : fminf(10.0f, fmaxf(-10.0f, x));
}
// IEEE NaN 先传播到算子输出，再由统一 sanitize 映射为 0。
__device__ __forceinline__ float gp_min(float a, float b) {
    return (isnan(a) || isnan(b)) ? CUDART_NAN_F : fminf(a,b);
}
__device__ __forceinline__ float gp_max(float a, float b) {
    return (isnan(a) || isnan(b)) ? CUDART_NAN_F : fmaxf(a,b);
}
__device__ __forceinline__ float softplus_clipped(float x) {
    return log1pf(expf(fminf(20.0f, fmaxf(-20.0f, x))));
}
__device__ __forceinline__ bool is_visited(const uint64_t* v, int city) {
    return (v[city >> 6] & (1ULL << (city & 63))) != 0;
}
__device__ __forceinline__ void mark_visited(uint64_t* v, int city) {
    v[city >> 6] |= 1ULL << (city & 63);
}
__device__ __forceinline__ uint4 philox(uint64_t seed, uint64_t instance,
                                      unsigned iteration, unsigned ant,
                                      unsigned step, unsigned purpose) {
    const uint64_t key = seed ^ instance;
    unsigned k0 = (unsigned)key, k1 = (unsigned)(key >> 32);
    uint4 c = make_uint4(iteration, ant, step, purpose);
    #pragma unroll
    for (int round=0; round<10; ++round) {
        uint64_t p0 = (uint64_t)c.x * 0xD2511F53U;
        uint64_t p1 = (uint64_t)c.z * 0xCD9E8D57U;
        c = make_uint4((unsigned)(p1 >> 32)^c.y^k0, (unsigned)p1,
                      (unsigned)(p0 >> 32)^c.w^k1, (unsigned)p0);
        k0 += 0x9E3779B9U; k1 += 0xBB67AE85U;
    }
    return c;
}
__device__ __forceinline__ float counter_uniform(uint64_t seed, uint64_t instance,
                                                int iteration, int ant, int step, int purpose) {
    return (float)(philox(seed,instance,iteration,ant,step,purpose).x >> 8) * (1.0f/16777216.0f);
}
extern "C" __global__ void probe_rng(uint64_t seed, uint64_t instance,
                                    const int* coordinates, unsigned* output, int count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= count) return;
    uint4 bits=philox(seed,instance,coordinates[index*4],coordinates[index*4+1],
                     coordinates[index*4+2],coordinates[index*4+3]);
    output[index*4]=bits.x; output[index*4+1]=bits.y;
    output[index*4+2]=bits.z; output[index*4+3]=bits.w;
}
