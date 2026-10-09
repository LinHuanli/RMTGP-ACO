"""共享 FP32 数值与 Philox4x32-10 随机合同。"""

import numpy as np
from numba import njit


@njit(cache=True)
def philox(seed, instance_key, iteration, ant, step, purpose):
    key = np.uint64(seed) ^ np.uint64(instance_key)
    k0, k1 = np.uint32(key), np.uint32(key >> np.uint64(32))
    c0, c1, c2, c3 = np.uint32(iteration), np.uint32(ant), np.uint32(step), np.uint32(purpose)
    for _ in range(10):
        p0 = np.uint64(c0) * np.uint64(0xD2511F53)
        p1 = np.uint64(c2) * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            np.uint32(p1 >> np.uint64(32)) ^ c1 ^ k0,
            np.uint32(p1),
            np.uint32(p0 >> np.uint64(32)) ^ c3 ^ k1,
            np.uint32(p0),
        )
        k0 = np.uint32(k0 + np.uint32(0x9E3779B9))
        k1 = np.uint32(k1 + np.uint32(0xBB67AE85))
    return c0, c1, c2, c3


@njit(cache=True, inline="always")
def counter_uniform(seed, instance_key, iteration, ant, step, purpose):
    bits = philox(seed, instance_key, iteration, ant, step, purpose)[0]
    return np.float32(bits >> np.uint32(8)) * np.float32(1 / 16777216)


@njit(cache=True)
def stdrel(values, count, output):
    """以首项为原点计算均值和方差，保证完全相等的输入严格输出零。"""
    origin = values[0]
    mean = np.float32(0)
    for i in range(count):
        mean += values[i] - origin
    mean /= np.float32(count)
    variance = np.float32(0)
    for i in range(count):
        centered = (values[i] - origin) - mean
        variance += centered * centered
    variance /= np.float32(count)
    denominator = np.sqrt(variance) + np.float32(1e-8)
    for i in range(count):
        output[i] = np.tanh(((values[i] - origin) - mean) / denominator)
