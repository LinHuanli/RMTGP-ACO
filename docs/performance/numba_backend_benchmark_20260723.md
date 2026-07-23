# Numba 正式后端基准（2026-07-23）

## 环境

- CPU：2 × Intel Xeon Gold 5122，共 8 个物理核、16 个逻辑 CPU；
- Python 3.12.13；
- NumPy 2.2.6、Numba 0.61.2、llvmlite 0.44.0；
- float64，candidate list \(K=20\)，ACO iterations \(=100\)；
- Numba `fastmath=False`，每个 worker 单线程；
- 首次 JIT 编译与 worker cache warm-up 不计入下述时间。

## 单个 GP 个体

每次同时求解一个 TSP50 和一个 TSP100 训练实例。PyTorch 与 Numba 使用
各自冻结的确定性随机流，因此此表只比较计算时间，不比较 tour 的逐位结果。

| ACO | Numba (s) | 优化后 PyTorch (s) | 加速比 |
|---|---:|---:|---:|
| ACS | 0.227 | 7.097 | 31.2× |
| AS | 0.969 | 8.977 | 9.3× |
| MMAS | 0.920 | 8.585 | 9.3× |

## 完整一代

使用正式 population 100；初始种群按 structural hash 去重后为 91 个唯一
genotypes。外层使用 8 个持久 worker。

| ACO | baseline (s) | population evaluation (s) | 整代测量 (s) |
|---|---:|---:|---:|
| ACS | 0.078 | 3.529 | 3.609 |
| AS | 0.816 | 30.615 | 31.435 |
| MMAS | 0.771 | 27.648 | 28.425 |

结果超过正式训练的 5× 加速门槛。实际各代时间会随唯一 genotype 数、树所需
terminals 和操作系统负载变化，因此训练 artifact 仍逐代记录真实墙钟时间。
