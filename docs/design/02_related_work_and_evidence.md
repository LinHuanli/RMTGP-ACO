# GP-ACO 加速：技术调研与证据表

版本：v1.0；核查日期：2026-10-02。  
用途：支持[主研究计划](01_research_plan.md)的方法选择和贡献边界；不将尚未运行的假设写成实验事实。

导航：[相关工作](#prior-art) · [技术选项](#technologies) · [代码证据](#code-evidence) · [假设表](#hypotheses) · [来源](#sources) · [文档索引](README.md)

## 1. 证据等级与调研范围

| 标记 | 含义 | 使用规则 |
|---|---|---|
| CODE | 在固定 commit 的源码或配置中可定位 | 可证明实现结构，不能证明它是运行热点 |
| PAPER | 原始论文、作者预印本、出版社或作者机构记录 | 区分全文、摘要、出版状态和作者报告的性能 |
| DOC | 硬件厂商或项目作者的官方文档 | 用于工具和硬件机制，不据此预言 GP-ACO 的收益 |
| USER | 团队在讨论中报告的实验观察 | 原始日志、代码和计时边界尚待复核 |
| HYP | 本研究提出的解释或方法假设 | 必须通过指定实验检验 |
| RESULT | 本研究按冻结协议得到的结果 | 当前尚无此类新结果 |

本轮是围绕 GP-ACO 研究问题的定向调研，不是声称穷尽全部 GPU 优化论文的系统综述。覆盖 TEVC 的进化框架和种群 GPU-GP、JPDC 的 GPU-ACO、GPEM 的硬件树求值、CPU-GP 优化、近期 AAD/GP 调度工作，以及 GPU 系统实践。

检索主题包括：`GPU evolutionary computation tensorization TEVC`、`population-level tree genetic programming GPU`、`automatic algorithm design batched GPU evaluation`、`high-throughput ant colony optimization GPU`、`multi-core SIMD genetic programming`、`GP hyper-heuristics scheduling acceleration`、`DeepGEMM DeepJIT`、`MiMo full-pipeline inference`、`Nsight profiling`。

未成功取得全文的条目仅用于其摘要或元数据所支持的判断，不推断其未展示的实验细节。后续新增来源必须记录版本和访问范围。

---

<a id="prior-art"></a>
## 2. 直接相关工作的对照

| 工作 | 已验证的研究对象和机制 | 可以借鉴什么 | 不能据此声称什么 |
|---|---|---|---|
| EvoX，TEVC [R05] | 面向分布式 GPU 的进化计算框架和编程/计算模型 | 状态与执行组织、多算法与多硬件验证 | “首次 GPU 进化计算” |
| EMO tensorization，TEVC [R06] | 将数据和控制转成张量，对 NSGA-III、MOEA/D、HypE 展开研究 | 方法、代表性实现、外层/问题规模、应用的证据链 | “批量张量化本身尚无人研究” |
| EvoGP，TEVC Early Access [R07] | 树的规则化表示、种群级执行、自适应个体内/个体间并行 | 异构程序表示、强 GPU-GP 组合基线 | “首次种群级 GPU-GP”或“GPU-GP 只能做静态回归” |
| AutoPSO，作者论文与 TEVC 记录 [R08] | 双层 PSO 自动设计，GPU 批量评估内层候选 | 外层算法设计与内层批量执行的连接 | “首次 GPU 双层 AAD” |
| 高吞吐 GPU-ACO，JPDC [R09] | warp 级向量化、Scan–Stencil Roulette、原子更新和融合 | 对具体阶段的硬件级归因和强求解器设计 | “候选并行、轮盘并行或融合是新思想” |
| TensorACO，GECCO Companion [R10] | ACO 张量化、矩阵预计算、更新映射和 AdaIR | 避免重复计算及批量更新 | 把改变选择分布后的结果作为同语义实现速度 |
| CPU 与 GPU 的 GP 比较，Soft Computing [R11] | CPU 缓存、SIMD、多线程组织 | 认真优化 CPU，而不是依赖解释器对照 | 将 GPU 相对弱 CPU 的差距视为硬件必然优势 |
| FPGA 树求值，GPEM [R12] | 展开、流水化的树求值，与软件实现及能效对照 | 强 CPU 基线及吞吐/能效分开评价 | 用一个慢 CPU 对照证明专用硬件优越性 |
| GP 调度仿真加速，2026 预印本 [R13] | 摘要描述 profiling、重构、正确性检查和 HPC 工作负载 | AAD 相邻任务的端到端加速过程 | 声称 GP 调度加速是空白，或将其当成已验证的 GPU 方法 |

表中“不能声称”的内容是本研究的贡献约束，不是对原论文质量的评价。

---

## 3. 期刊论文的叙事与实验组织

### 3.1 从计算问题出发，而不是从硬件峰值出发

EvoX 的研究单位是进化计算执行框架；EvoGP 的研究单位是变长树种群的高效执行；EMO tensorization 将问题明确为算法表示和操作与 GPU 执行之间的适配。这些工作都提供了具体的技术抽象，而不只是更换运行设备。[R05] [R06] [R07]

本研究应对应为：特征—程序联合计算、CPU 局部性映射、GPU 状态与资源感知执行。不能仅将“GP-ACO 很慢”当作充分动机。

### 3.2 扩展维度要分开

EMO tensorization 的实验分别改变种群规模与问题维度，并比较 CPU/GPU 和不同实现。[R06] 我们应分别改变 P/B/R 与 n，而不是把它们合成一个“更大问题”。

本研究额外增加程序终端依赖、编译新颖率和活跃状态数。这些是针对 GP-ACO 的实验设计，不是该论文已验证的结论。

### 3.3 底层机制必须对应整体收益

JPDC 的 GPU-ACO 工作将向量宽度、warp 通信、轮盘实现和信息素更新联系起来。[R09] 我们相应研究有效候选、扫描、寄存器、访存、同步及整代完成时间。

工具指标只支持机制解释。某项缓存命中率上升、寄存器下降或 occupancy 上升，不自动证明端到端改善。

### 3.4 强 CPU 和失败结果属于证据

Chitty 的工作直接研究 CPU 缓存、SIMD 和多线程；GPEM 2025 论文摘要明确报告 FPGA 在其设置中仍慢于 Operon CPU，但具有不同的能效表现。[R11] [R12]

因此，CPU 优化缩小 GPU 相对加速比并不是需要修饰的结果。它可能说明一部分原差距来自软件组织，也可能确定 CPU/GPU 各自适合的负载区间。

### 3.5 质量、执行速度和算法设计价值分开验证

最终路径质量用于判断本研究不同实现的数值路径是否产生有意义的质量差异。等时间完整训练用于判断加速释放的预算是否改善候选算法。二者与固定工作量速度不是同一个问题。

本研究采用明确的非劣容忍界和置信区间；不把“差异不显著”直接当成“质量相同”。具体协议由本研究预设，见 E10/E11。

---

<a id="technologies"></a>
## 4. 技术选项与采用条件

| 技术 | 在 GP-ACO 中的用途 | 已有依据 | 采用条件 | 风险/反证 |
|---|---|---|---|---|
| 依赖闭包与公共统计 | 只生成必要终端，避免重复归约 | 当前 required masks 是起点 [R02] [R03] | 相对掩码强基线仍减少工作或字节流量 | 当前实现已足够，联合化收益小 |
| 动态排名扫描 | 替代逐候选重复排名 | 源码可定位重复扫描 [R02] | 排名完全一致且整体收益存在 | K 小时同步/预处理成本抵消 |
| 联合生成代码 | 融合特征、树和分数变换 | 现有生成式树；专门化系统实践 [R02] [R14] | 计入编译后回本 | spill、代码膨胀、新程序太多 |
| ERC 结构缓存 | 相同结构复用代码 | 本研究提出的代码缓存设计 | 编译节省超过参数化损失 | 常数折叠损失大 |
| SIMD 与候选分块 | CPU 原生批量求值 | CPU-GP 与向量化文档 [R11] [R19] | 汇编和计数器证实真实向量化 | 数学库或 gather 成为热点 |
| 程序—实例二维调度 | 权衡代码局部性、数据局部性与负载 | 现有实例复用是对照 [R03] | 多种 P/B/n 下比固定策略好 | 调度开销或缓存竞争增加 |
| 程序资源分组 | 控制组内资源和运行时间差异 | GPU-GP 异构性问题 [R07] | 编译资源或尾部变化与速度一致 | 增加 kernel/JIT 数，分组无净收益 |
| 活跃状态分块 | 限制任务工作集，保留足够并行 | CUDA 内存原则；状态生命周期启示 [R15] [R18] | 区间化测量找到可迁移收益 | 少量任务不足以填满设备 |
| CUDA Graph | 降低重复提交成本 | 官方工具与执行文档 [R17] [R18] | 时间线显示启动/主机间隙显著 | 长 kernel 主导或结构频繁变化 |
| 长驻留/深融合 kernel | 降低调度与状态搬运 | GPU-ACO 有相关先例 [R09] | 收益超过资源占用和调度损失 | 低并发、同步限制、长尾 |
| 计数器随机数 | 任务重排时保持逻辑随机流 | Random123 [R20] | 键空间和浮点转换定义一致 | 逻辑坐标打包碰撞，随机用途混用 |
| FP8/FP16/Tensor Core | 暂不进入核心 | DeepGEMM 服务矩阵负载 [R14] | 未来发现匹配的核心矩阵运算再评估 | 峰值吞吐与当前标量/索引负载不对应 |

### 4.1 DeepSeek：采用的启示

DeepGEMM 官方文档介绍运行时编译及多类专门化/融合 kernel。[R14] 本研究借鉴“按真实计算结构专门化、显式计入代码生成与资源成本”，不直接采用其 GEMM kernel，也不外推其矩阵性能数字。

### 4.2 MiMo：采用的启示

MiMo 的全流程报告研究缓存生命周期、预取和缓存亲和路由与负载均衡。[R15] 本研究借鉴“状态在哪个范围有效、复用是否减少真实工作、局部性与调度如何权衡”。TSP 静态几何、任务信息素和蚂蚁 visited 的共享边界与 KVCache 不同，必须重新建模。

### 4.3 不优先做的方向

不优先开发多节点通信、自定义低精度、任意程序编译、全套自动 kernel 生成或 FPGA 原型。没有 profiling 证据时，不把硬件术语直接变成方法模块。

---

<a id="code-evidence"></a>
## 5. 现有仓库的可继承组件和新增空间

固定代码基点：`2b847698225e1d0fff36a25c1015a080350a7f00`，main 分支提交时间为 2026-09-16。该基点用于源码观察，不代表所有历史实验使用此版本。

| 组件 | 已有内容 | 新研究如何处理 | 不确定性 |
|---|---|---|---|
| `program.py` | F1 primitives、十终端、常数、程序描述 | 保留单树语言并显式落实 FP32 数值合同 | 新后端编译可能改变浮点路径，需 E00 |
| `aco_numba.py` | 编译边界、实例局部性、紧凑程序、标量终端掩码 | 作为 CPU-Existing-FP32 的基础 | 真实 SIMD 和当前热点需测 |
| `aco_tiled_v2.cu` | 候选子组、特征生成、生成式树、状态更新 | 作为 GPU-Existing-FP32 的基础 | 某条源码路径占多少时间未知 |
| `candidate_ordinal` | 对前序候选逐次统计 | 与一次掩码扫描比较 | K=20 时净收益不保证 |
| fallback DistRank | 对每个候选扫描全城市 | 与等义排名重写比较 | 历史 TSP500 现象是否由它造成未知 |
| 当前配置 | A=32、I=500、K=20、P=100、G=50、gamma=1/3 | 首轮科学预算默认值；去掉双树与 FP64 审计 | 性能敏感性必须单独改变因素 |

移除信息素树所独有的计算，以及取消在线 FP64 审计，是**新单树 FP32 基线的构建步骤**。它们不能算入 M1/M2/M3 相对该基线的收益。

---

<a id="hypotheses"></a>
## 6. 证据到假设的登记表

| 编号 | 当前证据 | 假设 | 验证实验 | 何时不能继续声称 |
|---|---|---|---|---|
| H1 | 终端存在动态依赖，代码有重复排名 | 联合计算可降低评价成本 | E03/E04 | 只改善孤立树吞吐，不改善代表性评价 |
| H2 | CPU 已优化但仍有解释器/调度选择 | 原生联合求值与二维调度有额外收益 | E05 | 相对现有 Numba FP32 无稳定净收益 |
| H3 | 程序异构、独立状态量可变 | 资源与状态感知计划优于固定映射 | E07/E08 | 选择开销后不优于最佳固定配置 |
| H4 | USER：TSP500 相对加速较弱 | 可由工作量、状态、资源或分母变化解释 | E02/E06 | 不复现，或只有相关性没有干预证据 |
| H5 | 新树持续出现 | 结构缓存与编译选择改善摊销 | E04/E09 | 只在预热/重复树场景有效 |
| H6 | 理论上可释放计算预算 | 等时间能改善至少一种 AAD 预算配置 | E11 | 独立测试无可信收益；保留速度贡献 |
| H7 | 接口分离了外层和内层任务 | 部分执行机制可迁移 | E12/E13 | 第二任务需要重写全部核心实现 |

H1—H7 当前均未获得本计划的新实验支持。

---

<a id="sources"></a>
## 7. 来源目录

以下条目是可追溯的原始来源。论文性能数字不作为我们计划的目标或预测；正文通常只使用机制和研究设计信息。

<a id="r01"></a>
### R01 — 当前单树语言的来源

类型：CODE。文件：`src/rmtgp_aco/program.py`。检查内容：完整 F1 函数集、状态转移终端、常数及程序表示；数值公式亦可由 R02 交叉核对。

[固定版本源码](https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/src/rmtgp_aco/program.py)

局限：源码定义不能证明终端使用频率或运行热点。

<a id="r02"></a>
### R02 — 当前 GPU 特征与候选求值

类型：CODE。文件：`src/rmtgp_aco/cuda/aco_tiled_v2.cu`。重点：`candidate_ordinal`、`transition_score_tiled`、终端 required masks、生成式求值和残差分数。此次复核包括约 190—410 行。

[固定版本源码](https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/src/rmtgp_aco/cuda/aco_tiled_v2.cu)

局限：尚未运行 profiler。重复排名的复杂度是源码推导，热点占比仍未知。

<a id="r03"></a>
### R03 — 当前优化 CPU 后端

类型：CODE。文件：`src/rmtgp_aco/aco_numba.py`。重点：Numba 编译、实例局部性、标量/向量终端掩码、程序打包与列求值路径。

[固定版本源码](https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/src/rmtgp_aco/aco_numba.py)

局限：该实现含 FP64 数据路径；不能直接用其旧时间当成新 FP32 CPU 基线。

<a id="r04"></a>
### R04 — 当前纯 TSP100 配置

类型：CODE。文件：`configs/as_tsp100_gpu0.yaml`。核对预算、GP 参数和 gamma。该配置还训练两棵树并含旧数值/审计设定，因此只继承明确列出的科学预算。

[固定版本配置](https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/configs/as_tsp100_gpu0.yaml)

<a id="r05"></a>
### R05 — EvoX

B. Huang 等，**EvoX: A Distributed GPU-Accelerated Framework for Scalable Evolutionary Computation**。IEEE Transactions on Evolutionary Computation，29(5):1649–1662，2025；在线发表 2024-04-15。DOI：`10.1109/TEVC.2024.3388550`。

[出版社记录](https://ieeexplore.ieee.org/document/10499977)；[作者预印本](https://arxiv.org/abs/2301.12457)。

核查范围：出版社摘要/元数据、作者预印本信息。使用依据：GPU/分布式执行的进化计算模型及多算法验证。不是本研究 GP-ACO 的直接性能测量。

<a id="r06"></a>
### R06 — EMO 的 GPU 张量化

Z. Liang 等，**Bridging Evolutionary Multiobjective Optimization and GPU Acceleration via Tensorization**。TEVC，30(1):420–434，2026；在线发表 2025-03-28。DOI：`10.1109/TEVC.2025.3555605`。

[出版社记录](https://ieeexplore.ieee.org/abstract/document/10944658/media)；[作者全文 v5](https://arxiv.org/html/2503.20286v5)。

核查范围：方法和实验章节的 HTML 全文。分别研究种群规模与问题维度，并比较张量/非张量、CPU/GPU、解质量与机器人任务。本研究借鉴其分层证据组织，不外推其最大加速比，也不假定本研究的最强 CPU 与其 CPU 实现等价。

<a id="r07"></a>
### R07 — EvoGP

Z. Wu、L. Wang、K. Sun、Z. Li、R. Cheng，**Enabling Population-Level Parallelism in Tree-Based Genetic Programming for GPU Acceleration**。TEVC Early Access，2026-02-10；DOI：`10.1109/TEVC.2026.3663396`。

[作者全文 v7](https://arxiv.org/html/2501.17168v7)；[arXiv 版本/接收信息](https://arxiv.org/abs/2501.17168)；[作者机构出版记录](https://ira.lib.polyu.edu.hk/handle/10397/119907)。

核查范围：v7 HTML、版本记录和机构元数据。树张量化、种群级操作、自适应个体内/个体间求值及 Python 环境集成均属于直接相关先例。更正早期讨论：截至本次核查，不能再仅称其为未发表预印本。

<a id="r08"></a>
### R08 — AutoPSO

X. Yu 等，**AutoPSO: A Metaframework for Automated Particle Swarm Optimization**。2026；作者仓库给出的 TEVC DOI：`10.1109/TEVC.2026.3718908`。arXiv：`2608.07539`。

[作者预印本索引](https://arxiv.org/abs/2608.07539)；[作者代码与论文信息](https://github.com/EMI-Group/autopso)。

核查范围：检索返回的作者摘要、作者仓库 README；本轮出版社页和 arXiv 全文打开未成功。可确认外层候选 PSO 设计与内层批量评估的结构。未根据不可访问的全文补写实验细节。

<a id="r09"></a>
### R09 — 高吞吐 GPU-ACO

J. M. Cecilia 等，**High-throughput Ant Colony Optimization on graphics processing units**。Journal of Parallel and Distributed Computing，113:261–274，2018。DOI：`10.1016/j.jpdc.2017.12.002`。

[出版社页面](https://www.sciencedirect.com/science/article/pii/S0743731517303337)。

核查范围：出版社摘要、highlights 和可访问的方法说明。研究 32/64 宽向量映射、warp 通信、SS-Roulette、原子更新和融合。我们使用其机制作为先例，不将这些优化重新命名为新方法。

<a id="r10"></a>
### R10 — TensorACO

L. Yang、T. Jiang、R. Cheng，**Tensorized Ant Colony Optimization for GPU Acceleration**。GECCO 2024 Companion，755–758。DOI：`10.1145/3638530.3654394`。

[作者全文](https://arxiv.org/html/2404.04895v1)；[作者机构出版记录](https://research.polyu.edu.hk/en/publications/tensorized-ant-colony-optimization-for-gpu-acceleration/)。

核查范围：HTML 全文与出版元数据。矩阵预计算、路径更新映射和 AdaIR 有直接先例。AdaIR 调整选择机制，不能无说明地作为本研究标准轮盘的同语义实现基线。

<a id="r11"></a>
### R11 — CPU/GPU GP 的强基线问题

D. M. Chitty，**Fast parallel genetic programming: multi-core CPU versus many-core GPU**。Soft Computing，16:1795–1814，2012。DOI：`10.1007/s00500-012-0862-0`。

[出版社页面](https://link.springer.com/article/10.1007/s00500-012-0862-0)。

核查范围：出版社摘要和元数据。研究缓存、SIMD 与多线程组织，说明 CPU 优化本身是实质性的研究内容。旧硬件结果不用于预测当前 CPU/GPU 的速度关系。

<a id="r12"></a>
### R12 — FPGA 树求值与优化 CPU 对照

C. Crary 等，**Using FPGA devices to accelerate the evaluation phase of tree-based genetic programming: an extended analysis**。Genetic Programming and Evolvable Machines，26，文章 8，2025-01-07。DOI：`10.1007/s10710-024-09505-2`。

[出版社页面](https://link.springer.com/article/10.1007/s10710-024-09505-2)。

核查范围：摘要、元数据及公开注释，非付费正文全量复核。摘要报告展开/流水树架构在其设置中仍慢于 Operon CPU，而能效有不同结果。用于支持强基线和性能/能效分开评价的必要性，不推演本研究速度。

<a id="r13"></a>
### R13 — GP 调度仿真的近期加速研究

H. T. Li、A. Pletzer、Y. Tian、Y. Mei、M. Zhang，**Accelerated Genetic Programming Hyper-Heuristics for Simulation-Based Scheduling via Agentic AI**。2026 预印本，arXiv：`2608.19487`。

[作者预印本索引](https://arxiv.org/abs/2608.19487)。

核查范围：检索返回的原始摘要；页面和 HTML 全文获取失败。摘要涉及离散事件调度、profiling、重构和正确性检查。只作为扩展任务的直接相关线索，尚不足以评价其具体 GPU 方法或实验控制。

<a id="r14"></a>
### R14 — DeepGEMM / DeepJIT

DeepSeek 官方项目，**DeepGEMM**。版本依据：2026-10-02 可访问 README，其中包含 2026-09-10 更新记录。

[官方 README](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md)。

核查范围：运行时编译、专门化与融合 kernel 的官方说明。借鉴编译、融合和工作负载专门化的工程原则。此链接随 main 变化，正式复现实验应另锁 commit；当前计划不依赖其库进行 GP-ACO 执行。

<a id="r15"></a>
### R15 — MiMo 全流程推理优化

Xiaomi MiMo Team，**Full-Pipeline Inference Optimization for MiMo-V2.5 Series: Pushing Hybrid SWA Efficiency to the Limit**。2026 技术报告，arXiv：`2607.13095`。

[作者全文 v1](https://arxiv.org/html/2607.13095v1)；[官方博客](https://mimo.xiaomi.com/blog/mimo-v2-5-inference)。

核查范围：HTML 报告和官方博客。采用的启示为缓存有效期、分层数据管理和亲和性/负载均衡；不将 LLM 的 KVCache 和 GP-ACO 信息素视为同一数据结构。

<a id="r16"></a>
### R16 — Nsight Compute

NVIDIA，**Nsight Compute Profiling Guide**。

[官方文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

核查范围：性能区段、资源/访存/调度分析以及重放等说明。实验中按实际工具版本查询 section 和 metric，不直接照抄其他架构的计数器名称。官方文档支持工具含义，不支持未经测量的瓶颈判断。

<a id="r17"></a>
### R17 — Nsight Systems

NVIDIA，**Nsight Systems User Guide**。

[官方文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

用途：CPU/GPU 时间线、CUDA API、NVTX 区间和关键路径。工具采集运行不作为正式速度样本。

<a id="r18"></a>
### R18 — CUDA 执行和内存优化

NVIDIA，**CUDA C++ Best Practices Guide**。

[官方文档](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

用途：数据传输、内存访问、资源占用和执行组织的原则。具体配置仍需在本研究 GPU 和负载上标定。

<a id="r19"></a>
### R19 — CPU 向量化与内存诊断

Intel，**VTune Profiler: HPC Performance Characterization View**。

[官方文档](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-1/hpc-performance-characterization-view.html)。

用途：解释向量化比例与向量宽度、内存访问和计算特征。该链接是具体版本的官方说明；硬件支持需按实际 CPU 检查。非 Intel 平台使用相应可用工具，不把缺失的指标写成零。

<a id="r20"></a>
### R20 — 计数器随机数

J. K. Salmon、M. A. Moraes、R. O. Dror、D. E. Shaw，**Parallel Random Numbers: As Easy as 1, 2, 3**，SC11；Random123 官方资料。

[作者项目页](https://random123.com/)；[接口文档](https://random123.com/releases/docs/)。

用途：用逻辑 key/counter 定义随机数，不依赖线程执行顺序。随机数生成器提供这种能力，并不自动保证本系统的坐标编码无碰撞；编码和用途隔离仍需测试。

---

## 8. 后续调研任务与完成条件

| 优先级 | 任务 | 完成产物 | 未完成时的处理 |
|---|---|---|---|
| 必须 | EvoGP 适配当前 PDIV、逐算子裁剪和十终端的成本 | 接口验证、受支持语义表、组合基线 | 不能以不支持为由直接排除，先完成接入验证 |
| 必须 | 外部张量基线的编译/批量/内存配置 | 调优预算与最终配置记录 | 不报告默认慢配置作为唯一外部结果 |
| 必须 | 历史 TSP500 运行 commit 与日志 | E02 复核表 | 只保留 USER 观察，不写为新实验事实 |
| 条件触发 | 第二类 AAD 的直接先行工作，包括 R13 全文 | 方法和工作负载差异表 | 不声称跨 AAD 的通用性 |
| 条件触发 | 更大 n 的稀疏信息素或延迟更新 | 语义证明与独立质量实验 | 不加入核心同语义速度比较 |
| 提交前 | 更新 TEVC/相关预印本及项目版本 | 文献与实现版本冻结记录 | 保留检索截止日期，不用“最新最优”概括 |

后续论文比较不要求每个框架都成为完整训练基线，但必须明确它是语言执行器、求解器、完整系统还是方法先例，避免比较不同层次的时间。

[R01]: #r01
[R02]: #r02
[R03]: #r03
[R04]: #r04
[R05]: #r05
[R06]: #r06
[R07]: #r07
[R08]: #r08
[R09]: #r09
[R10]: #r10
[R11]: #r11
[R12]: #r12
[R13]: #r13
[R14]: #r14
[R15]: #r15
[R16]: #r16
[R17]: #r17
[R18]: #r18
[R19]: #r19
[R20]: #r20
