# CPU 基线与 GPU 瓶颈证据补充协议

本协议落实六组 CPU 基线、历史数据复用和独立 GPU 诊断。先测冻结真实种群的一代评价，不要求所有 CPU 后端训练50代。不涉及 NeSI 的环境部署或任务提交。

## 1. 两类基线

普通 AS、同步 ACS、MMAS 是**求解质量基线**。用 `ZERO` 转移树关闭残差影响，ACO预算仍为32只蚂蚁、500次迭代、无局部搜索。gap以统一FP32标签路径长度为分母。相同随机流下重复的ZERO个体只算一份实例×seed观察，不能制造额外重复。

CPU-Python、CPU-Numba、GPU解释器和GPU生成式是**执行性能基线**。比较时必须保持程序、实例、初始化、随机流、科学预算、精度和计时边界一致。普通ACO单个程序时间不是GP种群评价的速度分母。

已有AS缓存覆盖两规模的训练和验证。三宿主性能实验中的ZERO结果可作为其32实例holdout对照。训练、validation、性能holdout和最终test必须分表。最终标准测试仍未开启；TSP100的1280个和TSP500的128个测试实例不变。

## 2. CPU实现及运行接口

| 后端 | 数值执行 | 1／8／16核方式 | 对照角色 |
|---|---|---|---|
| `cpu_python` | Python流程、NumPy候选向量及postfix解释，无Numba数值调用 | 单进程／8进程／16进程 | 基础实现 |
| `cpu_existing` | Numba编译完整仿真循环和通用GP解释器 | 1／8／16线程，实例级prange | 现有强基线 |

不能称 `cpu_existing` 为逐树Numba JIT，也不能用最外层 `.py_func` 包住仍被JIT编译的下层函数来冒充CPU-Python。基础版保留同一排名定义、逐节点保护和ACO更新时序。Python使用只读文件映射共享几何，映射文件和缓存均在项目内。每次评价包括进程池启动、共享映射准备和回收；该实现未宣称持久进程池已预热。

物理核由Linux拓扑和允许的affinity共同确定，每个核仅取一个逻辑CPU。NumPy/BLAS隐式线程限制为1，Numba线程数显式设置。CPU型号、核绑定、NUMA拓扑、主机负载和内存记录必须保留。主机锁仅避免本项目基线互相争用，不是对整台共享服务器的独占保证。

CPU-only依赖见 `configs/environment.cpu.requirements.txt`。已有项目环境可直接使用；独立安装时在项目内建立Python 3.12环境，再安装锁定依赖及本包，不需要CUDA、CuPy或Torch。

```bash
source scripts/env_cpu.sh
# 自行激活项目内CPU环境，或使用当前已安装的环境：
.envs/main/bin/python scripts/benchmark_cpu.py probe

# 导入原始GPU冻结输入，不重算几何或随机初始化。
.envs/main/bin/python scripts/benchmark_cpu.py prepare \
  --output artifacts/inputs/frozen-population/p01

# 单个完整科学cell：P100、B32、A32、I500，数据来自输入包。
.envs/main/bin/python scripts/benchmark_cpu.py run \
  --bundle artifacts/inputs/frozen-population/p01/tsp100/as \
  --backend cpu_existing --cores 16 --generation 1 --block 0 \
  --output artifacts/runs/pilot/E01/p01/cpu-baselines/tsp100-as-g001-numba-c16-b000-a01

# 默认只打印六组×3block的计划；--execute才实际计算。
.envs/main/bin/python scripts/run_cpu_matrix.py \
  --bundle artifacts/inputs/frozen-population/p01/tsp100/as \
  --output artifacts/runs/pilot/E01/p01/cpu-baselines/tsp100-as-g001-matrix-a01 --blocks 3
```

`run_cpu_matrix.py`在同一主机按固定随机顺序串行测六配置。每cell新进程和独立缓存。无可用16物理核时直接拒绝，不用SMT替代。默认单cell墙钟限制24小时，包含启动/预热；超时不是已完成evaluation，也不能当作evaluation的下界。只终止自身创建的进程组，保留日志和超时记录。失败或超时复测使用新输出目录。

### 计时边界

- `setup_s`：输入校验、加载和只读映射。与搜索分开。
- `warmup_s`：同Numba签名的小预算运行；不是科学测量。
- `eval_wall_s`：兼容旧GPU数据的后端返回边界，含仍需的准备、搜索、回传；CPU-Python含池生命周期。
- `fitness_available_wall_s`：后端评价及FP32 gap/fitness归约。不能与旧GPU仅后端时间直接相除。
- `--cache-state cold`：不进行预热；每cell缓存独立，首次编译包括在评价内。它不表示操作系统文件缓存已清空。
- 工作量为全部 `P×B` 个实际执行任务；不去重。一次评价完成的tour数为 `PBAI`，有效构造转移数为 `PBAI(n-1)`，不把闭合边或padding算成额外选择。

RSS采样覆盖主进程及子进程。共享页可能在RSS相加中重复计数，同时报告系统可提供的PSS。采样峰值不保证捕获瞬时最高值。CPU硬件cycles、IPC、cache miss未测时保持空值。

## 3. 冻结输入与历史资料

新的 `BenchmarkInputs` 显式从原GPU输入库导入。保留原manifest哈希、源码身份、配置、实例顺序和初始化字节。文件路径相对输入包；逐文件核对SHA256，并要求语义合同相同。不会放宽或修改现有 `FrozenStore` 的源码校验。缺少第25／50代的真实cohort时标记待依赖；后续导出新版本输入包，禁止补造或复制第1代。

旧slides实验在 `experiment/presentation-review-v1` 的本地工作目录中找到。源码工作量包括双树、CPU FP64、GPU FP32后主机FP64计分、去重及结果复用。因此旧CPU1／CPU8数据仅进入历史表和图。表达式Python/逐树JIT微基准不能冒充完整GP-ACO。导入保留当前文件实际哈希，包括尚未提交的导出更新，不假装全都来自分支HEAD。

## 4. 诊断指标的含义

现有原始数据能够整理：后端时间、设备区间、实际任务/tour/转移吞吐、构造/全局更新事件时间、fallback事件、树结构、内存池、NVML显存/能耗和编译资源。

新增 `diagnostic_work=True` 编译独立插桩内核。正式路径默认关闭。记录正常候选检查位置、fallback初始扫描位置、有效候选数、fallback排名的visited检查及距离对数、源码逻辑GP节点×候选数、统计扫描及原语逻辑调用。它们是**源码逻辑工作量**，不是编译后硬件指令数量；编译器可能删除无用中间计算。

诊断另外记录四段lane-0周期累加：候选集合、公共统计、逐候选特征＋GP分数、选择＋同步＋ACS局部更新。周期包含插桩和等待影响，跨任务累加不等于GPU墙钟，也不是跨SM统一时间戳。只用它定位后续调查方向，不直接把比例写成生产内核的耗时比例。

CUDA事件保存每波次各ACO迭代的构造/全局更新区间，时间原点是该波次开始。事件区间可能包含主机提交间隙，不冒充Nsight kernel时间。插桩模式提供NVTX构造/更新范围，可供兼容的Nsight Systems采集。

### 快照与同状态重放

每次最多采样8个均匀分散的逻辑任务、前2只蚂蚁、早/中/末3个迭代，以及构造早/中/末和首次真实fallback四类状态。理论上限192条，未出现的状态保持缺失。保存当前行信息素、visited、前后城市、任务状态及实际CUDA候选分数，不转存每个状态的完整信息素矩阵。

CPU NumPy在相同快照上重新计算候选、终端与修正分数，按 `atol=1e-5, rtol=1e-4` 与捕获CUDA分数比较。然后以逐位相同的终端数组比较GPU解释树和生成树的局部输出。树微基准只表示独立树kernel，不等于融合构造器内的GP时间。快照是诊断样本，不按其出现次数推断全体fallback分布。

```bash
source scripts/env.sh
# UUID须先经gpu-free确认空闲；入口还会复查并取得项目锁。
CUDA_VISIBLE_DEVICES=GPU-实际UUID python scripts/diagnose_gpu.py \
  --bundle artifacts/inputs/frozen-population/p01/tsp100/as \
  --output artifacts/runs/diagnostic/E01/p01/work-diagnostics/tsp100-as-g001-b000-a01
```

同一命令可重复指定 `--bundle`，例如依次追加 `tsp100/acs` 和 `tsp100/mmas`。此时输出目录内分任务保存，并在整批期间持有项目 GPU 锁。同一张卡顺序执行各宿主，不以并发争用提高吞吐。后台运行使用已提交源码的不可变快照，`nohup` 日志、请求、心跳、源码身份和结果均保留在项目内。

`--smoke-iterations 3`只做缩小迭代数的检查，输出明确标记，自动报告排除这些样本。完整诊断使用输入包中的500次迭代。只读记录工具路径及计数器权限，不更改驱动权限。occupancy、DRAM带宽、stall和spill流量未实际采集时保持缺失。

## 5. 报告与研究结论

```bash
source scripts/env.sh
python scripts/report_baseline_diagnostics.py \
  --historical-root /vol/grid-solar/sgeusers/linbocheng/MTGP_ACO_presentation_review/slides/presentation_benchmarks
```

报告分开输出至 `docs/results/pilot/E01/p01`、`docs/results/diagnostic/E01/p01` 和 `docs/results/historical/E02/p01`。目录内固定为README、tables、figures和provenance。CPU/GPU配对必须通过原输入manifest、程序、seed、配置和任务数校验。跨主机比较明确记录两个硬件环境，不宣称纯粹由某个GPU硬件特性造成。输入p01已经导出时不重复prepare；新输入版本须先登记，禁止覆盖。

CPU数据缺失时不生成数值加速比。新GPU计数到齐后增加工作量及插桩周期图。所有图同时输出PDF/SVG/PNG。未测硬件指标不填零。描述性关联只能提出假设；“排名、访存或寄存器是瓶颈”的因果结论仍需要等义重写、布局或资源干预，并验证无插桩端到端收益。

目前这不是E00–E13全部完成声明。CPU六组完整计时、真实冷/暖GPU对照、硬件计数器、M1/M2/M3消融及最终质量测试仍按独立实验推进。
