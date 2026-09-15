# GPU 加速 Genetic Programming：Slides 与实验计划

## 1. 范围

- 报告时长：36 分钟正文，4 分钟讨论，21 页正文。
- 只讨论加速：Python 开销、Numba 编译、GPU 并行、数据布局、编译与执行成本、规模扩展。
- GP-ACO 只作为 simulation workload。固定一种 ACO 配置，不讲算法创新，不比较 2024 与当前算法的效果。
- 连续运行实验默认 5 代；需要观察更多新程序编译和缓存行为时，再补到 10 代。
- 不跑 50 代，不做收敛曲线、最终解质量排名、多算法效果比较或大规模非劣性实验。
- 保留小规模功能检查，确认没有因为漏算任务、缩减预算或返回缓存结果而得到虚假的加速。

本文中的实验编号、参数扫描、计时字段和图表安排都是本次报告的新计划，不是仓库已有实验结果。

## 2. 现有材料和需要补充的部分

本计划按仓库快照 `128c789b508705274ef27fa2d887f395a947aa77` 核对实现。实际实验必须另行记录运行时的 commit 和修改补丁。

| 内容 | 已确认的情况 | 本次处理 |
|---|---|---|
| 2024 slides | 已讲 individual 级 CPU 并行、simulation 内循环矩阵化和 mask。[S1] | 压缩成 1 页方法回顾，不搬用历史加速倍率。 |
| 当前 Numba 后端 | 将 GP postfix 解释、tour construction、信息素更新和 RNG 放在同一个 `njit` 边界内。[R1] | 作为 CPU 主基线。 |
| 逐 individual Numba 编译 | 与当前通用 Numba 解释器不是同一种实现；本次核对未确认有可直接切换的完整后端。 | 单独补一个表达式微基准，不要求为报告重写完整求解器。 |
| 旧 GPU 后端 | `cuda_fused_fp32`。[R2] | 与新 GPU 后端做同任务工程对照。 |
| 新 GPU 后端 | `cuda_tiled_v2`；可生成 GP CUDA 表达式，展开 program × instance × ant × candidate。[R2] | 报告主体。 |
| 当前短跑入口 | `benchmark-training` 的参数检查只允许 1–3 代。[R3] | 扩展到 5–10 代，保留不运行 validation/checkpoint 的短跑行为。 |
| 当前示例配置 | `acs_tsp100_gpu0.yaml` 中仍是 50 代和 `cuda_fused_fp32`。[R4] | 新建报告专用配置，不直接使用原配置宣称测了 v2。 |
| GPU 调优配置 | 已有 v2 性能文档和调优清单针对 RTX PRO 5000 Blackwell，并非 A5000。[R2] | A5000 用自己的配置；不关闭硬件检查来强行套用 Blackwell 清单。 |

## 3. Slides 逐页安排

数据图没有结果时只留占位，不提前写“快了多少倍”或“哪个版本最好”。E0–E7 的实验定义见第 7 节。

| 页 | 标题 | 时间 | 要画或展示的内容 | 数据来源 | 本页回答的问题 |
|---:|---|---:|---|---|---|
| 1 | GPU Acceleration for Simulation-based GP | 0.5 分钟 | 一条流程：population → simulations → fitness。 | 示意图 | 本报告加速什么？ |
| 2 | 一次 evaluation 中有多少重复计算？ | 2 分钟 | D1：population、instance、iteration、ant、construction step、candidate、GP expression 的嵌套结构。 | 代码结构；实际任务计数 | 并行机会在哪里，哪些步骤仍有依赖？ |
| 3 | 一代的时间花在哪里？ | 1.5 分钟 | C1：当前 CPU 基线的一代时间分解。 | E1；必要时 E6 | evaluation 是否确实占主导？ |
| 4 | 以前的加速：多核、向量化、mask | 1.5 分钟 | CPU worker 分配图；循环变数组和 mask 的小示意。 | 2024 slides，仅复用思路 | 这些方法消除了什么开销，还留下什么？ |
| 5 | DEAP 管 population，Numba 做数值计算 | 2 分钟 | D2：DEAP → 树/程序表示 → 编译后的评估 → fitness。 | R1；E3 的实现片段 | Python 管理层和数值热点如何分离？ |
| 6 | Individual JIT：执行收益与编译成本 | 2 分钟 | C2：Python 与 individual JIT 的累计时间—调用次数曲线；旁边区分通用 JIT 解释器。 | E3 | 重复调用多少次，编译才值得？ |
| 7 | CPU 与 GPU：从少量任务到批量吞吐 | 1.5 分钟 | CPU workers 与 GPU 多 block 的任务队列示意，不使用虚构的时间比例。 | 示意图 | 为什么要提供足够多的独立任务？ |
| 8 | RTX A5000 的硬件结构 | 2 分钟 | D3：主机、PCIe、显存、GPU 内的 L2 和多个 SM；放大一个 SM。 | H1–H3；设备查询 | 计算单元和数据存储在哪里？ |
| 9 | 显存、缓存、shared memory、registers | 2 分钟 | D4：存储层次和程序数据的对应关系。 | H3–H4；代码；E6 | 数据复用和访问成本如何影响速度？ |
| 10 | Grid、block、warp、thread | 2 分钟 | D5：任务到 block，再到 warp/thread 的映射。 | H4；R2 | 一个大任务如何交给 GPU 执行？ |
| 11 | 为什么代码放到 GPU 上不一定快？ | 1.5 分钟 | 三个小示意：任务不足、分支分歧、寄存器/访存压力。 | H3；E6 的对应指标 | 需要检查哪些限制，而不是只看 GPU 利用率？ |
| 12 | 我们的 CPU–GPU 工作流程 | 2 分钟 | D6：CPU 演化与编码、设备执行、结果返回、CPU 计分。 | R1–R2；E1 时间字段 | 哪些工作留在 CPU，哪些进入 GPU？ |
| 13 | 四层并行：program × instance × ant × candidate | 2 分钟 | 放大 D5：一个 program–instance task 内的 ant 和 candidate lanes。 | R2；本次实际配置 | 怎样把仿真中的并行机会映射到线程？ |
| 14 | GP 解释器与生成式 CUDA 表达式 | 1.5 分钟 | 同一棵真实树：postfix 指令循环 vs 生成后的 CUDA 标量语句。 | R2；E5 | 少解释指令能省多少时间，编译增加多少？ |
| 15 | 数据驻留与 kernel 组织 | 1.5 分钟 | D7：静态数据上传、循环中的构造/更新、结果返回；旧/新执行时间线。 | R2；E6；可选 E5 | 减少传输、改变任务组织分别影响什么？ |
| 16 | 同一批评估任务：各后端时间对比 | 2.5 分钟 | C3：CPU 单线程、CPU 多线程、旧 GPU、新 GPU 的 5 代 trace 累计 evaluation 时间。 | E2 | 在相同工作量下，实际加速多少？ |
| 17 | 每代时间：编译、评估与 CPU 管理 | 2 分钟 | C4：5 代真实短跑的阶段时间；C5：5 代匹配回放的 evaluation 时间与工作量。 | E1、E2 | 首代为什么不同，后续各代为什么波动？ |
| 18 | Population 增大后，GPU 的收益如何变化？ | 2 分钟 | C6：population—时间和 population—吞吐曲线；instances 扫描可作备份。 | E4 | 小任务的额外开销何时被摊薄？ |
| 19 | 问题规模与 GP 程序规模 | 1.5 分钟 | C7：城市数—evaluation 时间；有数据再加入树规模分组。 | E4 | 时间增长来自更大仿真，还是更复杂的程序？ |
| 20 | 哪些优化真正有效？ | 1.5 分钟 | C8：旧内核、v2 解释器、v2 生成式、不同 lanes 的消融图。 | E5 | 每次修改的代价与收益是什么？ |
| 21 | 结论与使用边界 | 1 分钟 | 三项实测结论：匹配任务加速、5 代开销结构、适合 GPU 的工作量范围。 | E1–E5 | 什么时候用 Numba，什么时候用 GPU？ |

正文合计 36 分钟。第 18–20 页的补充曲线、多 GPU、详细 profiling 放备份页，不额外挤进正文。

## 4. 示意图绘制要求

### D1：计算层次与依赖

展示顺序：

```text
GP generation
  └─ program × instance
       └─ ACO iteration
            ├─ ants
            │    └─ construction step
            │         └─ candidate scores → GP expression
            └─ state update
```

`program × instance` 是独立仿真任务来源；ants 和候选评分提供任务内部并行。用依赖箭头保留路径构造的先后步骤，以及 iteration 之间的状态更新。不要把所有层都画成可同时执行。

这只是解释计算结构，不展开 GP-ACO 的算法公式。

### D2：DEAP 与 Numba 的边界

画两条路径，并明确命名：

```text
DEAP tree → 数值函数 → Numba 编译 → 在编译后的热点循环中调用
DEAP tree → postfix 数组 → Numba 编译的通用解释器/仿真器
```

第一条用于介绍 individual JIT，第二条对应当前仓库 CPU 实现。[R1] DEAP 的 `gp.compile` 返回函数或表达式结果，不等于已经生成 Numba 机器码。[N1–N2]

不画成“直接给整个 DEAP population 加一个 @njit”。

### D3：RTX A5000 硬件图

画三个层级：

1. CPU + 系统 RAM，经 PCIe 连接 GPU。
2. GPU 封装外侧的 GDDR6 显存；GPU 内部的 L2 和多个 SM。
3. 一个 SM 内的 warp scheduler、执行单元、register file、L1/shared memory。

可直接标注的 A5000 规格：24 GB GDDR6 ECC、8,192 CUDA cores、最高 768 GB/s 显存带宽、PCIe 4.0 ×16。[H1–H2]

SM 数、L2 容量、compute capability 和实际可用显存随设备查询结果一起保存；可使用 CUDA `deviceQuery` 或 CuPy 的设备属性查询。图上不要把“CUDA cores 数”当作 CPU 线程数，也不要把 CUDA threads 画成固定占用一颗 core。

Tensor Cores 和 RT Cores 可淡化，不作为本工作负载加速的主要解释。当前实现文档描述的是分支密集的标量计算，而不是矩阵乘法热点。[R2]

### D4：存储层次与代码数据

| 图中位置 | 解释 | GP-ACO 中可以标注的对象 |
|---|---|---|
| 系统 RAM | CPU 管理和主机端数据 | DEAP population、配置、原始 FP64 距离矩阵 |
| GPU global memory / 显存 | 容量较大，放批量数据和持久工作区 | 问题几何、候选列表、各 task 的动态状态和输出缓冲 |
| L2 / L1 cache | 硬件管理的缓存 | 对已有 global-memory 数据的访问复用，不画成另一份显式软件数组 |
| Shared memory | block 内协作使用的存储 | 只标实际 kernel 中明确分配的协作数据 |
| Registers | 线程执行中的局部标量 | 索引、局部累加值、部分表达式中间结果；实际占用由编译器和 profiling 确认 |

Ampere compute capability 8.6 的 combined L1/shared 资源是 128 KB，shared-memory 部分上限为每 SM 100 KB；每 SM register file 为 64K 个 32-bit registers。[H3] 不画成独立的“128 KB L1 + 128 KB shared”，也不要把 A100 的 L2 容量搬到 A5000。

这张图要区分“硬件原理”与“代码已经采用的分配”。不能因为某个数组是线程局部数组，就断言它全部位于寄存器。

### D5：任务到线程的映射

先画二维网格 `unique programs × instances`。再放大一个 task：

```text
1 task = 1 program × 1 instance
1 construction block = 32 ants × L candidate lanes
L = 8 时：256 threads = 8 warps
```

8 lanes 是同一只 ant 的协作组，不是一个完整 warp。每个 warp 为 32 threads；当 L=8 时，一个 warp 包含 4 个 ant groups。[H4；R2]

lanes 并行评分和归约，选择下一城市的部分仍有顺序。candidate list 长度可以大于 lane 数，不能把“8 lanes”标成“只计算 8 个候选”。

A5000 图中的 L 应与实验配置相同；8 lanes 不是预先认定的最优值。

### D6：完整执行流程

CPU：选择/变异/交叉 → 请求评估 → 去重与编码/生成源码。

CPU / CUDA runtime：编译与 module 装载 → 提交必要的数据传输和 kernels。

GPU：执行仿真 kernels → 回传结果。

CPU：对返回 tour 做 FP64 计分 → 写回 fitness → 下一代。[R2]

编译发生在主机工具链/运行时侧，不画成由 GPU CUDA cores 执行。

代码生成、编译、上传、执行、回传、计分使用与日志完全一致的名称。CPU FP64 计分是计时项目，不延伸成算法质量讨论。

### D7：数据与 kernel 时间线

旧融合路径与 v2 分开画。v2 的每个 ACO iteration 包含 `v2_construct` 和 `v2_update`，不是“所有 iterations 都融合在一个 kernel 里”。[R2]

图中显示：静态问题数据上传、重复使用、必要的动态数据传输、construct/update、输出回传。换了一批新实例时，可能需要新上传，不能画成整个训练永远只传一次。

kernel 数量从实际 trace 读取。不能预设“kernel 越少越快”；v2 与旧路径的比较需要同时考虑并行粒度和执行效率。

## 5. 数据图的定义

所有时间图标明硬件、后端、精度、线程/lanes、工作量和 cache 状态。默认展示原始秒数；倍率标注必须写明分母。

| 图 | 数据与坐标 | 比较与分析 | 注意事项 |
|---|---|---|---|
| C1：单代阶段时间 | 横轴为计时阶段，纵轴为秒；选择一个注明编号的代表代。 | CPU 管理、输入处理、evaluation 各占多少。 | CPU/GPU 各自的典型代可在第 17 页复用；不能只展示百分比。 |
| C2：JIT 收益起点 | 横轴为表达式调用次数，纵轴为累计秒数；Python、JIT 含编译、JIT 已编译。 | 计算减少是否足以摊薄编译成本。 | 标注为表达式微基准，不代表完整 simulation 的倍率。 |
| C3：同任务后端比较 | 横轴为后端，纵轴为同一 5 代 trace 的 evaluation 总秒数。 | CPU-1、CPU-T、GPU-v1、GPU-v2；以 CPU-T 为主 speedup 分母。 | 使用 E2 匹配回放，不用不同演化轨迹的训练总时间计算严格加速比。 |
| C4：逐代阶段开销 | 横轴为 generation 1–5，纵轴为每代秒数；CPU-T 和 GPU-v2 各一张图。 | 编译、评估、管理开销及其波动。 | 各段必须互斥；存在异步重叠时改用时间线，不强行堆叠。 |
| C5：逐代匹配回放 | 横轴为 trace generation，纵轴为 evaluation 秒数，各后端一条线。 | 同任务加速是否稳定；哪一代编译或计算更重。 | 下方单独给 requested/unique/executed programs、新源码数和节点数，不用混单位双轴。 |
| C6：并行规模 | 横轴为 population size；分别画 evaluation 秒数与实际 tasks/s。 | GPU 的收益起点、吞吐增长和趋稳区间。 | 主图包含该任务实际所需编译；纯 warm 执行单独标记。 |
| C7：仿真/程序规模 | 横轴为城市数，纵轴为 evaluation 秒数；树规模另作分组图。 | 工作量增长、内存变化、程序复杂度影响。 | 不把不同城市数的 tours/s 直接视为同等工作吞吐。 |
| C8：优化消融 | 横轴为实现配置，分别展示 evaluation 时间、kernel 时间和编译时间。 | 结构重排、生成式 GP、lanes 调整分别产生什么变化。 | 每个相邻对照明确改变了什么；精度变化单独比较。 |
| C9：GPU 执行时间线 | CPU 提交、H2D、kernels、D2H 的时间线。 | 空隙来自主机提交、同步、传输还是设备执行。 | 使用 E6 的诊断运行，不用 profiler 包裹下的墙钟做主速度结论。 |

默认每个性能点 3 次独立计时重复，保留全部原始值。画中位数和最小–最大范围，并在图注写 `n=3`。这不是算法统计置信区间。波动明显的点补到 5 次。

## 6. 统一工作负载和计时口径

### 6.1 主配置

主配置沿用仓库已有工作负载规模，但只运行短代数。[R4]

| 参数 | 本次默认值 |
|---|---|
| Workload | 固定 ACS，不展开算法差异 |
| GP generations | 5；10 代为补充 |
| Requested population | 100 |
| Instances / generation | 32 个 TSP100；冻结每代的实例与随机输入 |
| Ants | 32 |
| ACO iterations | 500 |
| Candidate list size | 20 |
| GP primitive set、树预算、遗传参数 | 固定使用报告配置；不为速度对比换成更简单的树 |
| 主 GPU | 单卡；原理图使用 A5000，性能图按实际测量卡标注 |
| CPU 线程 | 单线程，以及不超过物理核心数的合理多线程 T；记录实际 T |
| GPU-v2 起始配置 | 生成式 GP、普通 FP32、8 candidate lanes；不声称已对 A5000 最优 |
| Fast-math | 单独消融；不与同精度优化收益混算 |
| Validation / final test / checkpoint | 关闭 |
| baseline | 预先准备，运行时不得临时补算；载入/准备成本单列 |
| 日志和 snapshot 导出 | 简单计时日志保留；大文件导出不混入性能测量 |

减少的是 GP 代数，不是偷偷减少 simulation 的 ants、iterations 或实际评估的 programs。若某项 CPU 测量超时，记录 `timeout`；另开较小 workload，让所有被比较后端一起重测。

### 6.2 两类实验不能混用

**真实短跑（E1）**：正常进行选择、交叉、变异、评估，运行 5 代。用于展示实际每代组成、编译开销和总耗时。

**匹配回放（E2）**：保存一条真实 5 代轨迹的完整评估输入，各后端按同样顺序回放。用于计算严格的 evaluation 加速比。这里没有重新演化，不把 replay 时间叫作完整 generation 时间。

同一个 GP seed 不足以保证不同数值后端在后续代产生完全相同的 population。因此 E1 的不同后端曲线主要用于开销观察；严格对照使用 E2。

trace 至少保存每次评估调用的程序、实例 ID、随机流、实际预算、去重规则和调用顺序。若一代有多次评估调用，全部保存，不能只保存最后一个 population。

### 6.3 计时边界

| 层级 | 定义 |
|---|---|
| `startup_wall_s` | 启动、数据/归档载入、设备 context 初始化等一次性准备；单独报告。 |
| `generation_wall_s` | 一代中实际的输入准备、评估、fitness 写回、遗传操作和普通日志总墙钟。validation/checkpoint 已关闭。 |
| `evaluation_wall_s` | 从一批评估请求进入后端，到全部返回结果完成必要 CPU 计分；包括去重/编码、源码生成、必要编译/装载、传输和主机调度。 |
| `cpu_compute_s` | CPU 数值内核计算时间；是 evaluation 的子项。 |
| `gpu_span_s` | 使用 CUDA events 测量的设备执行区间；可能包含 kernel 间隙。 |
| `kernel_sum_s` | 实际 kernels 的执行时间和；来自诊断计时或 profiler，不等同于 `gpu_span_s` 或总墙钟。 |

CPU 墙钟用单调高精度时钟。GPU 工作是异步提交的，最终计时边界必须等待结果完成；单独的设备区间用 CUDA events。[H5] 不只测 kernel launch 的 Python 调用时间，也不在每个 candidate 操作后插入同步。

异步阶段可能重叠。`evaluation_wall_s`、`gpu_span_s`、`kernel_sum_s` 是嵌套/交叠指标，不能相加。堆叠图只使用互斥墙钟阶段；不能可靠拆开的部分保留为 `backend_wall` 或 `other_wall`。

### 6.4 Cache 状态

必须区分以下状态：

- `trace_first_use`：context 等通用准备单列，但未提前编译目标 trace 的新程序；按代顺序运行，保留正常程序/问题数据缓存。每代需要的编译计入 evaluation。
- `warm_same_workload`：同一批程序和输入已经运行过，再次评估；仍执行仿真，而不是返回已缓存 fitness。
- `process_cold`：新进程的启动总成本。若磁盘缓存已存在，就标明存在，不称为“完全冷启动”。

主比较用 `trace_first_use`。不同重复使用隔离的 benchmark 编译缓存或记录明确的缓存初始状态，不能让第一次重复为冷、后面为热，再直接取中位数隐藏差异。

预热 GPU 或 Numba 通用基础设施时不要提前编译目标 population。E2/E4/E5 的重复测量不能由 fitness 结果缓存直接返回；代码缓存、静态数据驻留和 fitness 结果缓存是三件不同的事。

## 7. 要跑的实验

### E0：准备、日志与功能检查

**用途**：保证后续每一张速度图有可解释的数据。

**需要补的工程工作**：

1. 复制一份报告配置，固定主工作负载，关闭 validation/checkpoint，明确选择后端。
2. 扩展 `benchmark-training` 到 5–10 代，并检查相关参数验证和 schedule 长度；不要只绕过一个报错。[R3]
3. 增加逐代计时、实际工作量和编译计数；字段见第 8 节。
4. 增加完整 evaluation trace 导出和匹配回放。这是需要补充的能力，本文不假定仓库已有同名 CLI。
5. 在目标 GPU 做编译和小规模运行检查；A5000 不复用 Blackwell 硬件绑定 profile。

**小规模检查**：选 8 个真实 programs、4 个 TSP100 instances、32 ants、50 iterations，给 CPU 和 GPU 相同输入。确认任务计数、预算、合法 tour、有限长度和无异常。相同语义优化路径运行已有回归测试；涉及 FP32/fast-math 的差异只记录，不声称得到统计上的等价或非劣结论。

**输出**：`environment.json`、`config.yaml`、`sanity.csv`、计时字段和 replay 能力。

### E1：5 代真实运行与时间分解

**后端**：CPU-T、GPU-v2。

**运行量**：每后端 5 代 × 3 次进程级重复；固定同一个 GP root seed。这里的重复用于计时，不是多个算法独立种子的效果实验。各后端固定所有非计时设置；连续运行中保留正常的代码缓存和问题数据缓存。

另采集一条 GPU-v2 的真实 5 代 evaluation trace。采集/导出开销不混入正式速度结果。将这条 trace 固定给 E2 使用。

**每代记录**：generation 总时间、evaluation 总时间、遗传操作、去重/编码、源码生成、编译/装载、CPU 计算或 GPU 执行、传输、CPU 计分、普通日志及未归类开销；同时记录实际执行程序数、tasks、树节点数和新源码数。

**分析**：首代额外成本来自哪里；第 2–5 代是否仍发生编译；某代变慢是否伴随更多独特程序、更长树、更多任务或更多编译。

**Slides**：3、12、17、21。**图**：C1、C4。

5 代足够完成正文。只有确实需要观察缓存变化或更多新代码时才补 10 代；不根据 5 代外推 50 代的确定总时间。

### E2：同一 5 代 trace 的后端对照

**后端**：

| 图中名称 | 实际含义 |
|---|---|
| CPU-1 | 当前 Numba 仿真器/解释器，单线程 |
| CPU-T | 相同 Numba 后端，多线程 T |
| GPU-v1 | 旧 `cuda_fused_fp32`，单 GPU |
| GPU-v2 | 新 `cuda_tiled_v2`，单 GPU，配置完整记录 |

**运行量**：4 个后端 × 同一条 5 代 trace × 3 次重复。trace 内按原顺序执行，不把每一代单独预热成已缓存任务。

GPU v1/v2 的主对比使用匹配的普通 FP32 设置。CPU FP64 与 GPU FP32 的工程比较在图注中明确标精度；不将其描述为仅更换硬件的实验。

**主要结果**：5 代 trace 累计 evaluation 时间、每代 evaluation 时间、编译时间、实际 tasks/s，以及对 CPU-T 的 speedup。

同一重复中：

```text
trace_speedup = sum(CPU-T evaluation_wall_s) / sum(GPU evaluation_wall_s)
```

不要先算每代倍率再无权平均。单独的 warm 执行结果可补测第 1、5 代，但不得替代包含新程序编译的主结果。

完整 Python/多进程/张量版本只有在能运行相同当前 workload 时才加入本表；没有现成后端就不为这场报告重写。Python 与 Numba 的基础效果由 E3 展示。

**Slides**：16、17、21。**图**：C3、C5。

### E3：DEAP individual 的 Numba 编译微基准

**目的**：支撑“DEAP 管理 population，Numba 加速 individual 求值”的讲解。

从真实 population 取 8 棵具有不同节点数的 GP 树，冻结 terminal 输入数组。比较相同 primitive 和数值保护规则下的 Python 求值函数与 Numba 编译函数；可额外加入当前 postfix Numba 解释器。

| 维度 | 设置 |
|---|---|
| 重复调用次数 | `10^2, 10^3, 10^4, 10^5, 10^6` |
| 每个点 | 3 次计时重复 |
| 编译状态 | 首次编译 + 执行；已编译重复执行 |
| 新个体成本 | 再用 trace 第 1–5 代的唯一树集合测新增编译数量和总时间 |

输入随索引变化，保留输出校验和，避免编译器消除无用的重复运算。不要把每次 Python 调用 Numba dispatcher 的开销误当作函数体的执行成本：至少测一版“热点循环也位于编译边界内”的路径，并在图注写清楚。

当前通用 CPU 后端不等于逐 individual JIT 后端。微基准只回答表达式求值和编译摊销，不能外推为完整 simulation 的相同倍数。[R1；N2]

**输出**：`jit_microbench.csv`。**Slides**：5–6。**图**：C2。

### E4：规模扩展

**方式**：从 trace 第 3 代取程序结构作为来源，构造并冻结扫描 workload。每个配置只做一次完整 evaluation，再重复 3 次；不用为每个配置跑 5 代。

**后端**：CPU-T、GPU-v2。

| 优先级 | 扫描项 | 取值 | 其他参数 |
|---|---|---|---|
| 必做 | Population size | 1、8、32、100、256 | TSP100、32 instances、32 ants、500 iterations |
| 必做 | 城市数 | 50、100、500 | population 100、32 instances、32 ants、500 iterations |
| 补充 | Instances | 1、8、32、64 | population 100、TSP100 |
| 补充 | 程序规模 | 总节点数按短/中/长三组，例如 5–11、12–21、22–31；先检查合法性和样本数 | population、instances、城市数均相同 |
| 补充 | Simulation 长度 | 50、100、500 ACO iterations | population 100、TSP100、32 instances |

不做参数笛卡尔积。公共主配置结果可以复用，不必重复启动同一点。

扩展 population 时，生成/选取语义不同的 programs，并尽量匹配节点数分布；不能复制同一棵树后被去重成一次评估。同步保存 requested、unique 和实际执行数量。

城市数扫描使用冻结数据；若需要生成额外规模，仅作为性能数据固定坐标与 seed，不展开泛化或解质量结论。TSP500 发生超时或内存不足时记录状态；如需降低该规模的 workload，CPU/GPU 必须同时降低并单独标注。

主图包含当前新程序需要的编译成本；warm-only 曲线可作补充。每个点记录峰值显存、chunk 数和实际 tasks 数。

**Slides**：18–19。**图**：C6、C7。

### E5：GPU 优化消融

**方式**：使用 trace 第 3、5 代的固定输入。每配置 × 2 个快照 × 3 次重复，不重新演化。

**必做路径**：

| 配置 | 变化 | 要回答的问题 |
|---|---|---|
| A | 旧 fused，GP 解释器，普通 FP32 | 旧工程基准 |
| B | v2，GP 解释器，4 lanes，普通 FP32 | 更换任务/内核组织后的整体收益 |
| C | v2，生成式 GP，4 lanes，普通 FP32 | 只改 GP 执行方式的收益与编译代价 |
| D | v2，生成式 GP，8 lanes，普通 FP32 | 只改候选协作粒度的收益 |

A→B 是组合的结构变化，不能全归因于单个指令优化。B→C 固定其它参数，C→D 固定精度和 GP 方式。检查目标硬件和实现是否支持配置；不支持的点记录为 `unsupported`，不能伪造等价实现。

**补充消融**：同一 v2 配置的 FP32/FP32-fast；同一输入下静态数据驻留开启/强制重新上传；instance-major/program-major。每次只改变一项。

驻留消融只改变静态数据复用，不同时清空编译缓存或 fitness 缓存。若当前没有独立开关，需要先加 benchmark 专用开关；不用不同进程启动成本冒充数据驻留收益。

分别报告 evaluation、编译/装载、kernel 时间；不能只展示生成代码之后的 kernel 变快而隐藏生成/编译成本。

**Slides**：14–15、20。**图**：C8。

### E6：解释 GPU 时间的 profiling

对固定的第 3 代 evaluation 做诊断，不跑完整 5 代。

- Nsight Systems：GPU-v1 与 GPU-v2 各采集一次代表性时间线，观察 kernel、传输、同步、主机提交间隙。
- Nsight Compute：对主要 construct/update kernels 的代表调用采集寄存器、shared memory、occupancy、SM/DRAM 吞吐、cache 指标和主要 stall 原因。
- CPU 只在需要时补粗粒度 profile。Python profiler 不能自动细分整个 `njit` 内核；拿不到的子项留空，不编造分解。

使用与 E2 相同的输入和配置。profile 单独运行，其墙钟不计入主速度表。

GPU 利用率高只能说明设备忙；不能仅凭利用率高、显存占用低，就断定是 arithmetic-compute-bound。是否主要受算术、访存、分支、同步或延迟限制，要结合 kernel 指标。

**输出**：原始 `.nsys-rep`、`.ncu-rep` 或相应报告；`kernel_metrics.csv`。**Slides**：9、11、15。**图**：C9。

### E7：可选，不影响正文完成

| 项目 | 运行方式 | 展示范围 |
|---|---|---|
| 10 代时间轨迹 | E1 的同配置延长到 10 代 | 后续编译、缓存、工作量变化，不画算法收敛 |
| CPU 线程扩展 | 固定一个 snapshot，1/4/8/16 线程；不超过物理核心限制 | 备份页；主基线 T 的选择依据 |
| 单任务双 GPU | 同一 snapshot，1 GPU vs 2 GPU，3 次重复 | 延迟下降与双卡效率 |
| 每 GPU 独立任务 | 两张卡各跑一个相同规模任务 | 吞吐增长，不能与单任务延迟混算 |

## 8. 必须保存的数据

时间统一用秒，显存统一用 bytes，数量用整数。未测到的字段写空值或 `null`，不要填 0。每条数据应能通过 workload ID 找到对应程序、实例和参数。

### 8.1 `environment.json`

保存 CPU 型号、物理/逻辑核心数、RAM、操作系统、GPU 型号/UUID、compute capability、SM 数、L2 大小、总显存、驱动、CUDA runtime/toolkit、Python/Numba/llvmlite/CuPy/PyTorch 版本、Git commit、dirty 状态和 patch 文件名。

另保存线程设置、dtype、fast-math、candidate lanes、block threads、task order、设备列表、缓存路径/状态、GPU 功耗限制，以及目标 GPU 是否有外部进程。没有测量的指标保持缺失。

### 8.2 `generations.csv`

每行：一个真实运行中的一代。

| 字段组 | 最低字段 |
|---|---|
| 标识 | `experiment_id, run_id, repeat_id, backend, gp_seed, generation, config_id` |
| 工作量 | `population_requested, programs_unique, programs_executed, instances, tasks_executed, aco_iterations, ants` |
| 程序变化 | `nodes_mean, nodes_max, depth_mean, new_programs, generated_source_hash, modules_compiled, compile_cache_hits` |
| 总时间 | `startup_wall_s`（单列记录）、`generation_wall_s, evaluation_wall_s` |
| 主机阶段 | `evolve_s, input_prepare_s, encode_codegen_s, fitness_assign_s, logging_s, other_wall_s` |
| 后端子项 | `compile_load_s, cpu_compute_s, h2d_s, gpu_span_s, d2h_s, fp64_score_s` |
| 状态 | `peak_device_memory_bytes, status, timing_notes` |

各时间字段的包含关系写进说明文件；不保证表中所有列都可相加。若源码按一整批 programs 编译，`modules_compiled` 不能被称为“编译 individual 数”。

### 8.3 `evaluations.csv`

每行：一次匹配回放、规模扫描或消融中的 evaluation。

```text
experiment_id, run_id, repeat_id, backend, config_id,
trace_generation, evaluation_call_id, workload_id, cache_state,
cpu_threads, gpu_name, precision, fast_math, generated_gp,
candidate_lanes, task_order, resident_data,
population_requested, programs_unique, programs_executed,
instances, cities, ants, aco_iterations, candidate_size,
nodes_mean, nodes_max, tasks_executed, tours_executed, chunks,
evaluation_wall_s, encode_codegen_s, compile_load_s,
cpu_compute_s, h2d_s, gpu_span_s, kernel_sum_s, d2h_s, fp64_score_s,
peak_device_memory_bytes, output_signature, status, notes
```

`tasks_executed` 按实际 program–instance–seed 仿真任务计数。`tours_executed` 按实际构造量计数；若使用理论数量，注明没有 early exit、漏算或结果缓存。不同城市数或树复杂度的 tasks 并不代表相同计算量。

### 8.4 `jit_microbench.csv`

```text
program_hash, node_count, input_hash, dtype, method,
call_count, repeat_id, cache_state,
codegen_s, compile_s, execute_s, total_s,
compiled_signatures, checksum, status
```

### 8.5 `trace/` 与辅助文件

`trace/` 保存第 1–5 代全部 evaluation 输入、程序内容、实例引用、随机流和调用顺序。`workload_manifest.json` 保存哈希和参数。`sanity.csv` 保存小规模功能检查。`profiles/` 保存 profiling 原始报告和关键截图。

建议交付目录：

```text
presentation_benchmarks/
  README.md
  environment.json
  config.yaml
  code.patch
  workload_manifest.json
  sanity.csv
  generations.csv
  evaluations.csv
  jit_microbench.csv
  trace/
  profiles/
    kernel_metrics.csv
```

只交截图或平均值不够；保留每次原始测量。编译/传输/CPU 计分拿不到独立时间时，保留可信的总 evaluation 时间，并明确缺失部分。

## 9. 实验清单与执行顺序

| 顺序 | 实验 | 必做范围 | 产出 |
|---:|---|---|---|
| 1 | E0 | 报告配置；5–10 代入口；计时日志；trace/replay；小规模功能检查 | 可重复运行的基准与环境记录 |
| 2 | E1 | CPU-T、GPU-v2，各 5 代 × 3 次；另固定一条 5 代 trace | 每代时间和真实工作量 |
| 3 | E2 | 4 后端 × 同一 5 代 trace × 3 次 | 主要速度对比与逐代匹配曲线 |
| 4 | E3 | 8 棵树，5 档调用次数；新增树的编译成本 | DEAP + individual JIT 的数据 |
| 5 | E4 | Population 5 档；城市数 3 档；CPU-T/GPU-v2，每点 3 次 | 并行规模与问题规模曲线 |
| 6 | E5 | A/B/C/D 配置，2 个 snapshot，每点 3 次 | 结构、生成式 GP、lanes 的消融 |
| 7 | E6 | 旧/新 GPU 代表性时间线；关键 kernels 指标 | 对加速来源的解释 |
| 8 | E7 | 按需选择 | 备份页 |

只做 5 代真实短跑；规模、消融和 profiling 都使用固定快照，不分别运行完整进化。第一次收集的数据应优先覆盖 E0、E1、E2，随后补 E3–E6。

## 10. 资料依据

以下材料仅支撑现有实现和硬件/计时事实；本文提出的新实验设置不属于已验证结论。引用仓库时使用固定 commit，避免默认分支后续变更。

- [S1] 用户提供的 `Bocheng_Lin_2024.6.5.pptx`：第 11–12 页为 CPU 并行评估，第 25 页为 Loops to Matrix，第 28–29 页为 unvisited cities 与 mask。仅用于回顾方法。
- [R1] [当前 Numba 后端 `aco_numba.py`](https://github.com/LinHuanli/RMTGP-ACO/blob/128c789b508705274ef27fa2d887f395a947aa77/src/rmtgp_aco/aco_numba.py)。
- [R2] [CUDA v2 实现与性能记录](https://github.com/LinHuanli/RMTGP-ACO/blob/128c789b508705274ef27fa2d887f395a947aa77/docs/performance/cuda_v2_pro5000_blackwell_20260730.md)。本文没有将其中 Blackwell 的秒数当作 A5000 实验结果。
- [R3] [CLI 实现 `cli.py`](https://github.com/LinHuanli/RMTGP-ACO/blob/128c789b508705274ef27fa2d887f395a947aa77/src/rmtgp_aco/cli.py)，`_command_benchmark_training` 的 1–3 代参数限制，以及已有时间字段。
- [R4] [当前 TSP100 ACS 配置](https://github.com/LinHuanli/RMTGP-ACO/blob/128c789b508705274ef27fa2d887f395a947aa77/configs/acs_tsp100_gpu0.yaml)。
- [N1] [DEAP GP API：`gp.compile`](https://deap.readthedocs.io/en/master/api/gp.html)。
- [N2] [Numba 5-minute guide](https://numba.readthedocs.io/en/stable/user/5minguide.html) 与 [Performance Tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)。
- [H1] [NVIDIA RTX A5000 产品资料](https://www.nvidia.com/en-us/products/workstations/rtx-a5000/)。
- [H2] [Leadtek RTX A5000 硬件规格](https://www.leadtek.com/eng/products/workstation_graphics%282%29/NVIDIA_RTX_A5000%2840914%29/detail)。
- [H3] [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)，注意区分 compute capability 8.6 与 A100 的 8.0。
- [H4] [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)。
- [H5] [CUDA asynchronous execution 与 events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) 及 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。
