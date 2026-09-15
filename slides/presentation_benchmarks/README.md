# A5000 presentation benchmarks

本目录用于 36 分钟正文、4 分钟讨论的 GPU 加速 GP 报告。
英文图表与标题，中文讲解稿。本轮固定同步 ACS、无 local search。

## 工作负载

- Population 100；5 代；每代 32 个 TSP100；32 ants；500 ACO iterations；K=20。
- 双树合计 31 个有效节点，最大深度 5；GP root seed 2001。
- CPU 使用 numba_batch，FP64，分别为 1/8 个物理核；GPU 搜索使用普通 FP32。
- GPU-v1：cuda_fused_fp32；GPU-v2：cuda_tiled_v2、generated GP、8 lanes。
- 关闭 validation、final test、训练 checkpoint。输入、baseline 均提前准备。
- 最优标签用于 fitness；报告分析计时与计算结构，不分析算法收敛或优劣。

## 运行入口

在项目根目录执行：

```bash
.venv/bin/python scripts/run_presentation_benchmarks.py prepare
.venv/bin/python scripts/run_presentation_benchmarks.py launch
.venv/bin/python scripts/run_presentation_benchmarks.py status
.venv/bin/python scripts/run_presentation_benchmarks.py report
```

`launch` 启动 nohup 调度器；远端每个完整比较组同样由 nohup worker 执行。
每次任务使用独立进程和编译缓存。worker 同机互斥，记录 host、GPU UUID、PID、心跳。
同一张卡只启动一个本项目仿真任务。独立比较组可运行于不同服务器。

初始优先主机为 cuda02:0、cuda04:0、cuda08:2。实际启动前由 gpu-free 重新查询。
主对照整体固定同机；所有规模扫描整体固定同机。记录实际主机，不以型号相同代替同机配对。

E0 功能检查、baseline、trace 为依赖任务。其余实验：

| 编号 | 内容 |
|---|---|
| E1 | CPU-8 / GPU-v2，5 代真实短跑 |
| E2 | CPU-1 / CPU-8 / GPU-v1 / GPU-v2，完整相同 trace 回放 |
| E3 | 8 棵真实树、5 档调用次数，Python / individual JIT / postfix Numba |
| E4 | population 1/8/32/100/256；TSP50/100/500，7 个不同扫描点 |
| E5 | 两个真实快照上的 fused / v2 解释器 / v2 生成式 / lanes 对照 |
| E6 | 代表性 evaluation 的 profiling；工具不兼容或权限受限时记录原因 |

E1/E2/E4/E5 默认各 3 次计时重复。组内某后端的极差超过中位数的 10% 时整组补至 5 次。
CPU/GPU 精度差异可能改变进化轨迹，因此 E1 的总时间不能充当严格速度比。
E2 固定 trace 的每次输入、顺序及预算，并检查各后端实际任务数。

## 计时定义

| 字段 | 含义 |
|---|---|
| startup_wall_s | Python 启动、数据/基线载入、通用 JIT/CUDA 预热、环境采集；由 worker 提供进程开始时间 |
| generation_wall_s | 真实短跑的一代总墙钟；输入数组已在 startup 载入 |
| evaluation_request_wall_s | 结构去重、baseline 读取、程序编码、求解、fitness 聚合和写回 |
| evaluation_wall_s | 结构去重、编码、求解及必要 CPU 计分/fitness 聚合；不含 baseline 读取和 fitness 写回 |
| compile_load_s | CuPy 模块编译或装载；NVRTC 实际调用数另外记录 |
| gpu_span_s | CUDA events 设备区间，包含 kernel 间隙，不是 kernel 时间之和 |
| kernel_sum_s | 仅 profiling 可提供，普通测量保持 null |
| cpu_compute_s | 当前 Numba 后端原生计时边界，包括原生求解及输出封装 |
| sampled_peak_device_memory_bytes | 2 秒 GPU 遥测的峰值，是实际峰值的下界；不冒充连续采样峰值 |

评价墙钟与 GPU 区间、编译、传输是嵌套/可能重叠的指标，不能全部相加。
阶段堆叠图只叠加 evaluation request、evolve、logging、other 四个互斥部分。
源码导出使用独立 trace 运行；正式重复不保存训练 checkpoint。

## 数据与版本

- `source.json` / `snapshots/`：commit、完整源码快照和内容哈希。快照包含新增未提交文件。
- `workload_manifest.json` / `inputs/`：冻结实例 ID、坐标、最优 tour、种子及内容哈希。
- `trace/`：5 代全部 evaluation 输入；树以带类型的节点保存，支持常数与 baseline 哨兵。
- `baselines/`：按数值语义域提前计算的结果。运行时缺少或不匹配则失败。
- `jobs/`：每个任务的命令、缓存目录、日志、原始结果和 GPU 遥测。
- `status.json`：全部比较组进度。`RESULTS.md`：当前可整理的有效结果。
- `generations.csv` / `evaluations.csv`：真实进化与匹配回放分开保存。
- `figures/`：可导出的 SVG/PDF/PNG。`presentation_outline_zh.md`：21 页提纲与中文讲解。

三台已核对 A5000 均为 sm86、64 SM、6 MiB L2、24 GB 显存。
环境为 Python 3.12.13、NumPy 2.2.6、Numba 0.61.2、llvmlite 0.44.0、
PyTorch 2.12.1+cu132、CuPy 14.1.1；CUDA toolkit 明确使用 /opt/cuda 13.3。
硬件属性以每个任务的 environment 记录为准。

当前主机的 RmProfilingAdminOnly=1；Nsight Compute 硬件计数器不可由普通用户读取。
旧 Nsight Systems 的 CUDA 13 兼容性由诊断任务验证，失败记录不计入主性能测量。

已完成任务可以恢复跳过；同卡外部进程干扰的测量不进入有效汇总。
显式 timeout/failed/unsupported 与 completed 分开记录，不补造数值。
