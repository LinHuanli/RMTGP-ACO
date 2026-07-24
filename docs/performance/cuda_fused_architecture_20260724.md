# 融合 CUDA 后端：实现、数值质量与性能审计

> 初稿日期：2026-07-24
> 独占性能复核：2026-07-25
> 状态：工程实现、正式数值质量门、单卡速度门和双卡 campaign
> 吞吐门通过；单 run 双卡扩展门未通过

## 实现范围

`cuda_fused_fp32` 已覆盖 AS、同步 ACS、MMAS、transition/pheromone 两棵
postfix GP 树、counter RNG、candidate fallback、ACS edge multiplicity、
AS budget residual、MMAS bounds 和完整 restart。一个 CUDA block 独占一个
`semantic program × instance` task，并在设备内完成全部 ACO iterations。

GPU 返回 `best_tour`，训练层使用原始 CPU float64 distance matrix 重算
`best_length`。单卡、双卡、chunk partition 和相同 seed 重复运行必须返回
相同 tour/length/iteration/diagnostics。

当前节点的 GCC 16 不受 CUDA 12.6 NVCC 支持，因此 CuPy 使用 NVRTC 生成
PTX；source hash 进入 CuPy 磁盘 cache。该选择避免引入不受支持的 host
compiler，不启用 fast math。

`benchmark-accelerators` 在每个 GPU mode 前清空进程内 resident 与
RawKernel cache；`cold_seconds` 包含磁盘 PTX cache 装载/必要编译及静态
H2D，但不包含已建立的 CUDA context。warm median 另行记录 kernel、H2D、
D2H、CPU FP64 scoring 和 host overhead。监控线程在测量全程采样外部
compute PID，而不只检查开始/结束两个时点。

## Block size 探索

内核采用 ant-per-thread 结构，32 ants 对应一个完整 warp。TSP100、
16 instances、首代 79 个语义唯一 programs、50 ACO iterations 的初步结果：

| block threads | kernel time |
|---:|---:|
| 32 | 2.63 s |
| 64 | 4.48 s |

128 threads 以上会在当前每线程 GP 栈/调用栈布局下触发 illegal-address，
因此配置层只接受 0（自动取 32）、32 和 64。正式默认值为 32。该限制应在改为
warp-per-ant 或消除解释器调用栈之后再重新评估。

## 50-iteration 历史同代矩阵（受污染，只作功能检查）

命令：

```bash
python -m rmtgp_aco benchmark-accelerators \
  --config configs/acs_protocol_a.yaml \
  --phase development --iterations 50 \
  --max-individuals 100 --repeats 1 \
  --modes cpu16 gpu0 gpu1 dual campaign \
  --gpu-devices 0 1 --gpu-block-threads 32
```

同代 workload 为 TSP50/TSP100 各 16 instances、100 requested individuals、
79 semantic programs、32 ants，共构造 4,044,800 tours：

| 模式 | warm end-to-end | tours/s |
|---|---:|---:|
| CPU16 | 28.17 s | 143,587 |
| GPU0 | 4.00 s | 1,010,923 |
| GPU1 | 4.30 s | 941,107 |
| dual | 3.18 s | 1,272,076 |
| campaign（两个 run） | 4.63 s | 1,747,510 |

在这一次观测中，单 GPU/CPU16 为 7.04×，dual/CPU16 为 8.86×，但
single-run dual scaling 仅 1.26×，campaign throughput scaling 仅 1.73×。
所有 GPU0/GPU1/dual 输出 signature 相同。

这些数值不能用于门控：测量结束时两张 GPU 上同时存在另一用户的
Ultralytics DDP 进程，各占约 7 GiB、GPU utilization 约 76%--88%、功耗约
203 W。后续必须在两张卡均无外部 compute process 时，按 500 iterations、
3 次 warm repeats 重新测量。正式报告保留：

\[
S_1=T_{\mathrm{CPU16}}/T_{\mathrm{GPU1}},\quad
S_2=T_{\mathrm{CPU16}}/T_{\mathrm{dual}},\quad
G_2=T_{\mathrm{GPU1}}/T_{\mathrm{dual}}.
\]

若独占重测仍有 \(G_2<1.7\)，调度采用每张 GPU 一个独立 GP run，而不是
强制一个 run 跨两张 GPU。

## RTX 4000 Ada、500-iteration 正式性能门

2026-07-25 在 GPU0、GPU1 两张 20 GiB NVIDIA RTX 4000 Ada Generation
上完成独占复核。GPU2 上存在与本实验无关的作业；监控按 GPU UUID 映射到
目标设备索引，只将 GPU0/1 上的外部 compute process 计为争用。复核期间
GPU0/1 的外部进程集合为空。

固定 workload 为 ACS、100 requested individuals（行为去重后 79 个语义
programs）、TSP50/TSP100 各 16 instances、32 ants、500 iterations。每个
单 run 构造 \(40{,}448{,}000\) 条 tour。每种模式执行一个 cold run 和三个
warm runs；下表为 warm median：

```bash
python -m rmtgp_aco benchmark-accelerators \
  --config configs/development_acs_cuda.yaml \
  --phase development --generation 1 \
  --iterations 500 --max-individuals 100 --repeats 3 \
  --modes cpu8 cpu16 gpu0 gpu1 dual campaign \
  --gpu-devices 0 1 --gpu-block-threads 32
```

| 模式 | 完成的 workload 数 | warm median (s) | tours/s | 相对 CPU16 |
|---|---:|---:|---:|---:|
| CPU8 | 1 | 397.422 | 101,776 | 0.766× |
| CPU16 | 1 | 304.325 | 132,911 | 1.000× |
| GPU0 | 1 | 20.807 | 1,943,957 | 14.626× |
| GPU1 | 1 | 20.838 | 1,941,110 | 14.605× |
| dual shard | 1 | 12.776 | 3,165,928 | 23.820× |
| campaign | 2 | 21.101 | 3,833,840 | 约 28.9× aggregate |

其中 GPU0/GPU1/dual 的 signature 完全一致，故设备选择和 LPT 分片不改变
同一 CUDA 随机实验。CPU8/CPU16 的 signature 也相同；CPU float64 与 GPU
FP32 的 signature 不要求逐位相同，其非劣性由下一节的独立质量门判断。

正式门控量为

\[
\begin{aligned}
S_1 &=
\frac{T_{\mathrm{CPU16}}}{\min(T_{\mathrm{GPU0}},T_{\mathrm{GPU1}})}
=14.626 \ge 3,\\
S_2 &= \frac{T_{\mathrm{CPU16}}}{T_{\mathrm{dual}}}
=23.820 \ge 5,\\
G_2 &=
\frac{\min(T_{\mathrm{GPU0}},T_{\mathrm{GPU1}})}
{T_{\mathrm{dual}}}
=1.629 < 1.7,\\
E_2 &= G_2/2=0.814,\\
G_{\mathrm{campaign}} &=
\frac{2\min(T_{\mathrm{GPU0}},T_{\mathrm{GPU1}})}
{T_{\mathrm{campaign}}}
=1.972 \ge 1.8.
\end{aligned}
\]

因此单卡速度门、dual 相对 CPU16 速度门和 campaign 吞吐门通过，但单 run
双卡扩展门未通过。默认正式调度结论如下：

1. 最大化完整实验吞吐时，每张 GPU 启动一个独立 run，按
   variant/method/replicate 分配，即双卡 campaign；
2. 单个 run 的一代必须尽快完成时，可以使用 dual shard；它把一代由
   20.81 s 降到 12.78 s，但不能声称达到预注册的 1.7× 双卡扩展门；
3. CPU16 保留为 float64 oracle、最终 champion holdout audit 和 GPU
   不可用时的可复现 fallback，不再作为大规模候选 population 的首选后端。

单卡 GPU0 的平均利用率约 99.0%，GPU1 约 99.0%；campaign 中两卡约为
99.8% 和 98.3%。峰值显存仅约 286 MiB，warm 时间中约 99.8% 位于 CUDA
kernel，H2D、D2H 与 CPU FP64 tour 复算合计远低于 0.1 s。因此当前正式
workload 是 kernel-compute-bound，而不是显存容量或 PCIe 传输受限。增加
显存缓存不会带来主要收益；后续优化应集中于 ant construction、候选选择和
双卡任务粒度。

## 500-iteration 正式数值质量门

使用 ACS、32 ants、500 iterations、TSP50/TSP100 各 128 个 validation
instances 和 3 个 paired ACO seeds。统计时先在同一 instance 内聚合 seeds，
因此置信区间的样本数为 256，而不是把 768 个随机重复误作独立实例。

```bash
python -m rmtgp_aco validate-cuda-quality \
  --config configs/development_acs_cuda.yaml \
  --instances-per-scale 128 --seeds 3 --individuals 1 \
  --batch-size 16 --cpu-threads 16 \
  --gpu-devices 0 1 --gpu-mode dual \
  --tolerance-pp 0.10
```

定义 \(\Delta=g_{\mathrm{GPU}}-g_{\mathrm{CPU}}\)，结果为：

| 范围 | instance 单位 | raw seed observations | mean Δ (pp) | 单侧 95% 上界 (pp) | 结论 |
|---|---:|---:|---:|---:|---|
| TSP50 | 128 | 384 | \(3.05\times10^{-14}\) | \(3.24\times10^{-14}\) | pass |
| TSP100 | 128 | 384 | -0.006889 | 0.001724 | pass |
| pooled | 256 | 768 | -0.003445 | 0.000868 | pass |

三项均低于预注册阈值 0.10 pp，故 FP32-search/FP64-score 数值契约通过。
测量期间两张 GPU 上存在外部 Ultralytics DDP 进程，因此 GPU 232.76 s 与
CPU16 39.22 s 仅说明共享资源竞争，禁止用于 speedup 或调度结论。
