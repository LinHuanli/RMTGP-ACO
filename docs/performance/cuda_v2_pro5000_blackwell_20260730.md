# RTX PRO 5000 Blackwell 上的 CUDA v2 加速与质量门

## 1. 结论

本轮优化得到一个可用于后续正式实验的 GPU profile：

```text
backend          = cuda_tiled_v2
provider         = raw_cuda
precision        = fp32_fast
candidate_lanes  = 8
register_cap     = 0
task_order       = instance_major
generated_gp     = true
graph_replay     = false
```

硬件绑定清单位于
`configs/cuda_tuning/rtx_pro5000_blackwell_sm120_v1.json`。清单会核对 GPU
名称和 compute capability。硬件不匹配时程序会终止。程序不会静默复用
其他 GPU 上的调优结果。

在 ACS、TSP100、100 requested individuals、79 个语义 programs、32 个
instances、32 ants、500 ACO iterations 的固定工作负载上，旧 CUDA
内核单卡墙钟为 15.574 s。最终 v2 单卡为 5.221 s，双卡为 2.997 s。
因此，新单卡相对旧单卡加速 2.98 倍。新双卡相对旧单卡加速 5.20 倍。

这些数字是工程性能结果。它们不构成算法效果结论。

## 2. 软件环境

| 组件 | 版本 |
|---|---:|
| Python | 3.12.13 |
| PyTorch | 2.12.1+cu132 |
| PyTorch CUDA runtime | 13.2 |
| CuPy | 14.1.1 |
| 系统 CUDA toolkit | 13.3 |
| cuda-tile | 1.5.0 |
| NumPy | 2.2.6 |
| Numba | 0.61.2 |

项目环境为 `.venv`。旧 CUDA 12.6 环境已归档为
`.venv-cu126-archive-20260730`，并已从 Git 排除。登录 shell 使用
`/opt/cuda`。NumPy 固定为 2.2.6，因为 Numba 0.61 不支持 NumPy 2.3
及以上版本。

## 3. CUDA v2 的并行结构

一个 CUDA task 对应一个“语义 program × TSP instance”组合。一个
construction block 包含

\[
32\ \text{ants}\times 8\ \text{candidate lanes}=256\ \text{threads}.
\]

同一只蚂蚁的 8 个 lanes 并行完成 candidate list 扫描、terminal 计算、
GP 表达式求值和 score 归约。lane 0 执行 roulette 或 ACS greedy 选择。
32 只蚂蚁仍共享同一个信息素快照。ACS local update 对重复无向边进行同步
合并。因此，新结构没有改变同步 ACS 的算法定义。

每个 ACO iteration 分为两个 kernel：

1. `v2_construct` 构造 32 条 tour；
2. `v2_update` 选择强化来源并更新信息素。

问题几何常驻显存。Program×instance tasks 由确定性 LPT 规则分配给两张
GPU。代价估计同时使用 TSP 规模、transition tree 长度、pheromone tree
长度和 ACO 变体。最优 tour 返回 CPU 后，仍使用原始 FP64 distance matrix
重新计分。

## 4. 生成式 GP

旧路径在每次 candidate 和每条强化边上解释 postfix bytecode。v2 可把当代
语义 programs 生成为 CUDA 标量语句。生成代码保留每个 primitive 后的
finite check 和 \([-10,10]\) clipping。它也使用 FP32 bit pattern 表示常数。

79 个真实语义 programs 的 20-iteration 配对测试结果如下。

| GP 执行方式 | kernel time (s) | 相对解释器 |
|---|---:|---:|
| postfix 解释器 | 0.3527 | 1.00× |
| 生成式表达式 | 0.2266 | 1.56× |

AS、ACS 和 MMAS 的回归测试均逐位比较 tour、FP64 length、best iteration
和 diagnostics。全部相同。真实 79-program 的 TSP50/TSP100 测试也逐位
相同。生成代码的编译成本约在每代发生一次。500 iterations 足以摊薄该成本。

## 5. 结构调优

短任务只用于筛选候选。最终选择必须在 500 iterations 上复核。

### 5.1 主要阶段

| 实现 | 单卡 kernel time (s) | 相对旧内核 |
|---|---:|---:|
| 旧 `cuda_fused_fp32` | 15.5405 | 1.00× |
| tiled v2，GP 解释器 | 10.0425 | 1.55× |
| tiled v2，生成式 GP，FP32，4 lanes | 6.7344 | 2.31× |
| tiled v2，生成式 GP，FP32-fast，8 lanes | 5.2048 | 2.99× |

最终复核中，FP32-fast 的 4 lanes、4 lanes/96-register cap 和 8 lanes
分别为 5.6904、5.4439 和 5.2096 s。8 lanes 且不限制寄存器最快。
`instance_major` 在重复测量中优于 `program_major`。因此最终选择为
8 lanes、无 register cap、instance-major。

### 5.2 单卡与双卡

| 模式 | wall time (s) | kernel time (s) | tours/s | 输出签名 |
|---|---:|---:|---:|---|
| 单卡 | 5.2210 | 5.2048 | 7.75 M | 相同 |
| 双卡 | 2.9971 | 2.9722 | 13.50 M | 相同 |

双卡扩展为 1.742 倍。并行效率为 87.1%。它通过 1.7 倍的预设扩展门。
采样期间没有外部 GPU 进程。单卡 GPU0 的平均利用率为 94.7%，峰值为
100%。双卡运行时 GPU0 和 GPU1 的平均利用率分别为 100.0% 和 88.7%。
两卡峰值均为 100%。每卡显存约 458 MiB。因此，该工作负载受 kernel
计算限制，不受显存容量限制。

## 6. 数值精度选择

### 6.1 完整求解器

下表使用同一个 500-iteration 初始 population。所有结果都进行 CPU FP64
tour 复算。

| Profile | kernel time (s) | 结论 |
|---|---:|---|
| FP32 | 6.7344 | 精确参考 profile |
| FP32 fast-math | 5.6876 | 候选；约快 18.4% |
| FP16 mixed storage | 6.6807 | 无速度收益 |
| BF16 mixed storage | 6.7639 | 无速度收益 |
| FP16 storage + FP16 score rounding | 6.6769 | 无速度收益 |

FP16/BF16 只减少静态距离表的存储。动态信息素、GP primitives、score
归约和 roulette 仍需要 FP32。ACO 热点是分支密集的标量控制流。它不是
矩阵乘法。因此，Tensor Core 低精度吞吐不能直接转化为完整 solver 加速。

### 6.2 FP8 与 NVFP4 探针

独立探针实际调用 CUDA 13 的 E4M3 和 E2M1 转换。它测量随机读取、
反量化和 FP32 累加。误差使用距离、启发式和 log-启发式的混合动态范围。

| Storage | probe time (ms) | mean relative error |
|---|---:|---:|
| FP64 | 0.0917 | 0 |
| FP32 | 0.0260 | 0 |
| FP16 | 0.0195 | 0.0178% |
| BF16 | 0.0179 | 0.1421% |
| FP8 E4M3 | 0.0199 | 2.3090% |
| NVFP4 E2M1 block-16 proxy | 0.0445 | 73.9733% |

NVFP4 行使用每 16 个值一个 FP32 scale。它是对真实 NVFP4 的乐观代理：
真实格式还会量化 block scale。该代理仍要执行随机 scale 读取和 E2M1
反量化。实测中，它比 FP32 慢约 71%，平均相对误差约 74%。FP8 也没有优于
FP16 的读取速度，且误差更大。因此，二者均被淘汰。

## 7. cuTile 评估

cuTile 1.5.0 已在真实 Blackwell 工具链上运行。规则子核使用 2,528 tasks、
32 ants 和 20→32 padded candidates。cuTile 与 raw CUDA 输出逐位相同。

| 实现 | median time (ms) |
|---|---:|
| 优化 raw CUDA | 0.01250 |
| cuTile | 0.01331 |

cuTile 相对 raw CUDA 的 speedup 为 0.939，即慢约 6.5%。该子核已经是最有利
于 tile 化的规则部分。完整 ACO 还有 visited set、fallback、roulette、
同步 ACS local update 和动态 GP 控制流。把这些阶段拆出当前融合内核会增加
显存流量和 kernel launch。因此，正式 provider 选择 raw CUDA。

## 8. 正式数值质量门

最终 profile 为 FP32-fast、8 lanes。质量门使用 ACS、32 ants、500
iterations、TSP50/TSP100 各 128 个 validation instances 和 3 个 paired
ACO seeds。统计时先在 instance 内聚合 seeds。非劣阈值为 0.10 percentage
points。定义

\[
\Delta=\mathrm{gap}_{GPU}-\mathrm{gap}_{CPU}.
\]

| 范围 | 独立单位 | mean \(\Delta\) (pp) | 单侧 95% 上界 (pp) | 结果 |
|---|---:|---:|---:|---|
| TSP50 | 128 | \(3.05\times10^{-14}\) | \(3.24\times10^{-14}\) | pass |
| TSP100 | 128 | -0.005704 | 0.002897 | pass |
| pooled | 256 | -0.002852 | 0.001450 | pass |

三个上界均低于 0.10 pp。最终 profile 通过质量门。该结论只说明搜索精度
非劣。正式训练锁定的 candidate 仍需要独立 CPU FP64 holdout audit。

## 9. 三算法正式规模短跑

短跑均为纯 TSP100。每代使用 32 个不同 instances、population 100、
32 ants 和 500 iterations。它们只用于工程验证。它们不执行 validation，
也不产生论文 champion。

| Variant | generation | generation time (s) | evaluation time (s) | best minus baseline (pp) |
|---|---:|---:|---:|---:|
| ACS | 1 | 3.495 | 3.003 | -0.373 |
| ACS | 2 | 4.687 | 4.173 | -0.602 |
| ACS | 3 | 5.308 | 4.800 | -0.265 |
| AS | 1 | 6.142 | 5.564 | -1.319 |
| AS | 2 | 5.099 | 4.489 | -1.030 |
| AS | 3 | 5.267 | 4.666 | -1.598 |
| MMAS | 1 | 5.263 | 4.783 | -1.464 |
| MMAS | 2 | 4.241 | 3.739 | -1.707 |
| MMAS | 3 | 4.112 | 3.613 | -1.852 |

三次短跑总墙钟分别为 ACS 14.64 s、AS 18.45 s 和 MMAS 15.47 s。每代
最优个体都优于同一 instance×seed 的原始 ACO baseline。这说明 fitness
方向和选择压力工作正常。

不同代使用不同训练批。原始 gap 和 baseline gap 会随批次变化。因此，
不能把三行直接解释为单调学习曲线。是否优于 baseline 必须由冻结
validation 和最终 paired test 回答。

## 10. 可复现实验入口

精度和 cuTile 探针：

```bash
.venv/bin/python -m rmtgp_aco probe-cuda-precision
.venv/bin/python -m rmtgp_aco benchmark-cutile
```

使用硬件绑定 profile 做 3 代短跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m rmtgp_aco benchmark-training \
  --config configs/acs_tsp100_gpu0.yaml \
  --backend cuda_tiled_v2 --gpu-devices 0 1 --gpu-mode dual \
  --cuda-tuning-manifest \
    configs/cuda_tuning/rtx_pro5000_blackwell_sm120_v1.json \
  --baseline-policy compute --phase development --generations 3
```

正式运行仍必须使用冻结 schedule 和预计算的同 kernel-semantic baseline
archive。开发短跑中的 `baseline-policy=compute` 不能替代正式 cache。
CUDA v2 的 baseline semantic ID 包含 provider、搜索精度和 candidate lanes。
因此，FP32、FP32-fast、混合精度或不同 lane 数不会误用同一个 baseline
archive。旧融合 CUDA 内核也位于独立的 cache 域。
