# RMTGP-ACO：面向 TSP50、TSP100 与 TSP500 的 Multi-Tree GP–ACO 研究设计

> **文档状态**：Design v1.4 / Protocol A v0.5（2026-07-27 pilot 测试口径修订）
>
> **研究对象**：对称二维 Euclidean TSP；Ant System（AS）、Ant Colony System（ACS）和 MAX–MIN Ant System（MMAS）
>
> **核心方法**：使用两棵 Strongly Typed GP 树分别学习状态转移残差和全局信息素强化残差
>
> **实现技术**：DEAP、PyTorch、NumPy、Numba、CuPy Raw CUDA；CPU/GPU 分层并行
>
> **范围约束**：本文档定义研究、算法、接口、伪代码与实验协议；实现位于
> `src/rmtgp_aco`

---

## v1.4：selected-champion 测试与消融作用域

本节优先于后文可能产生歧义的 pilot 测试描述，但不改变 ACO、GP、训练数据、
fitness、primitive set 或搜索预算。修订只消除两个不必要的计算扩张：
测试 GP population，以及把全部消融方法扩展到全部 OOD partitions。

### 模型选择单位

每个独立 GP run \(r\) 在完成 50 代搜索后，仅根据冻结的 selection
validation 选择一个 champion：

\[
c_r^\star
=\arg\min_{c\in\mathcal C_r}
\widehat g_{\mathrm{selection}}(c).
\]

该 candidate 随后接受 holdout gate 与 CPU/FP64 audit，并以
`selected_candidate.pkl` 锁定。正式测试集合为

\[
\{c_1^\star,c_2^\star,c_3^\star\}
\]

（确认性实验则为 30 个 runs），而不是每代 population 中的所有 individuals。
三个 GP runs 均作为模型学习方差来源保留；禁止根据 test performance 再从
三个 seeds 中挑选一个“最好 run”。因此“测试最好的 individual”严格表示
“测试每个 run 内由 validation 预先选择的一个 champion”，不表示用测试集
二次选模。

### 最终测试与消融测试的分工

主方法 RMTGP-Full-F1 的三个 locked champions 在下列全部分区做最终评测：

\[
\mathcal P_{\mathrm{final}}
=\{
\mathrm{TSP50\mbox{-}U},
\mathrm{TSP100\mbox{-}U},
\mathrm{TSP500\mbox{-}U},
\mathrm{TSP1000\mbox{-}U},
\mathrm{TSP500\mbox{-}C},
\mathrm{TSP500\mbox{-}G},
\mathrm{TSPLIB}_{\le500}
\}.
\]

方法、表示、primitive 与机制消融只在

\[
\mathcal P_{\mathrm{abl}}
=\{\mathrm{TSP100\mbox{-}U},\mathrm{TSP500\mbox{-}U}\}
\]

计算。前者给出训练分布内证据，后者给出规模外推证据；这已经足以回答
residual vs replacement、双树 vs 单树、Core vs Full、F0 vs F1 以及
drop/shuffle 机制问题。把所有消融扩到 cluster、Gaussian、TSPLIB、
TSP50 和 TSP1000 会显著增加计算量，却不改变这些预注册 estimands。
已经生成的历史全 OOD 消融 artifact 保留供 provenance 审计，但不得进入
v1.4 的描述汇总、置信区间、Wilcoxon 或 Holm family。

### 统计口径

每个作用域内的配对键仍为

\[
(\text{GP root seed},\text{instance},\text{ACO seed}).
\]

三个 ACO seeds 先在 `GP run×instance` block 内聚合；层次 bootstrap 保留
`GP run→instance→ACO seed` 三层变异。消融 contrasts 只跨
\(\mathcal P_{\mathrm{abl}}\) 校正。非核心 OOD partitions 只报告 Full-F1
相对 paired original ACO 的质量与鲁棒性描述，不能用于声称某个消融因素在
该分布上的因果效应。

## v1.3 / Protocol A v0.5：融合 CUDA 与完整 MMAS 重启

本节优先于 v1.2 及后文的旧执行建议。ACO、GP、数据量、fitness 与消融矩阵
保持 v1.2 不变；v0.5 只改变执行后端契约和 MMAS 搜索控制语义。由于 kernel
semantic version、ACO config hash、baseline schema 与 checkpoint schema 均已
升级，v0.4 的 baseline archive 和 checkpoint 不得用于 v0.5。

### 数值契约

GPU 后端采用“FP32 搜索、FP64 计分”：

1. distance、heuristic、pheromone、GP terminal 和 ACO 控制状态在 GPU 上
   使用 FP32；
2. GPU 只返回最优 Hamiltonian tour
   \(\pi\in\{0,\ldots,n-1\}^{n+1}\)；
3. 主机用由原始坐标生成的 FP64 distance matrix 重算

\[
L_{64}(\pi)=\sum_{j=0}^{n-1}d_{64}(\pi_j,\pi_{j+1});
\]

4. fitness、reference gap、baseline delta 和最终论文表格只使用
   \(L_{64}\)，不使用 GPU 累加的 FP32 length；
5. TF32、FP16、BF16 和 `--use_fast_math` 均关闭。Tensor Core 不属于该
   不规则组合搜索的计算路径。

FP32 仍可能通过 roulette、argmax 和 best-tour 比较改变搜索轨迹。因此 GPU
不是 CPU 的逐位替代，而是独立的 kernel semantic domain。进入正式训练前，
在固定 128 个 TSP50、128 个 TSP100、3 个 ACO seeds 上计算

\[
D_i=g_i^{\mathrm{GPU}}-g_i^{\mathrm{CPU}},
\qquad
U_{0.95}=\overline D+1.645\frac{s_D}{\sqrt N}.
\]

统计单位为 `program × instance`：先在同一 instance 内聚合 3 个 ACO
seeds，再计算置信上界，避免把随机重复误当成独立 TSP 样本。仅当 TSP50、
TSP100 及 pooled 的 \(U_{0.95}\le0.10\) percentage points 时，GPU 数值契约
通过质量门控。最终 champion 还必须在 CPU float64 后端复评；GPU selection
gate 或 CPU/FP64 gate 任一失败均部署原始 ACO，两个阶段分别写入
`validation_summary.csv` 与 `cpu_fp64_audit_summary.csv`。

### 分层并行架构

并行层级固定为：

```text
independent GP runs               -> campaign：GPU0 / GPU1 各一个 run
one GP generation                 -> 严格顺序，完成 fitness 后才能 breeding
semantic program × instance       -> 一个 CUDA block 对应一个独立 ACO task
ACO iteration / construction step -> block 内严格顺序
32 ants                           -> 32 个 CUDA threads 同步构造
matrix init / evaporation         -> block 中全部 threads 分片处理
```

CPU 是 control plane，负责 schedule、数据解析、GP 结构/语义去重、postfix
打包、FP64 精确计分、DEAP breeding、统计和 checkpoint。GPU 是 data plane，
在一个 kernel launch 内完成一个 task 的全部 500 次 ACO iterations。禁止在
同一代中把部分个体交给 CPU、部分个体交给 GPU，因为这会把数值后端变成
fitness 的混杂因素。

### 驻留数据和内存策略

每张 GPU 缓存当前用到的只读问题数据：

- FP32 distance、heuristic、log-heuristic 和节点局部均值；
- FP64 distance 排序后得到的 candidate list 与完整 nearest-neighbour rank；
- instance-keyed counter RNG key。

H2D 使用 page-locked staging buffer 和异步 copy。两张 GPU 各自保存一份
静态数据，不依赖 P2P 或 NVLink。tour 使用 `uint16`，visited set 使用
64-bit bitset。对每个 task，主要工作区近似为

\[
4n^2+n^2+2M(n+1)+8M\left\lceil\frac n{64}\right\rceil+4Mn
\quad\text{bytes},
\]

分别对应 pheromone、edge frequency、tour、visited 和 deposit。运行时至少
保留 20% 显存；若任务矩阵超过剩余预算，按 task chunk 分批，不得静默回退
CPU。禁止物化 \([P,B,M,n,K]\) terminal tensor；candidate terminal 在寄存器
和局部栈中流式求值。

### 融合 kernel 与 GP 解释器

GP 树被编译成定长二维 packed postfix buffers：

\[
(\text{opcode},\text{float argument},\text{integer argument},
\text{length},\text{terminal mask}).
\]

每个 thread 使用实际最大栈深度不超过 32 的局部栈解释 `ADD, SUB, MUL,
PDIV, PDIV1, MIN, MAX, ABS, NEG`。只计算 required-terminal mask 引用的
动态量。候选列表为空时扫描完整未访问城市集，并保持“fallback 固定
argmax”的参考语义。

ACS 在一个 construction step 中先让全部 ants 基于同一 pheromone snapshot
选择边，再由 thread 0 按无向 edge multiplicity 执行同步局部更新：

\[
\tau'_{uv}
=(1-\xi)^c\tau_{uv}
+\left[1-(1-\xi)^c\right]\tau_0.
\]

全局强化、GP pheromone residual、best state 和 anytime state 均留在同一
kernel 内，主机不参与 500 次迭代。当前 ant-per-thread 实现只允许
32/64-thread block；早期 A5000 探索和后续 RTX 4000 Ada 正式复核均采用
32 threads，64 threads 在相同短跑中更慢。128 threads 以上会因每线程 GP
栈和调用栈压力失效，因而配置层直接拒绝该不安全区域。

集群 CUDA 12.6 所见 GCC 16 超过 NVCC 支持范围，所以当前实现通过 CuPy
NVRTC 生成 PTX，并以 CUDA source hash 使用磁盘编译缓存；这不改变 kernel
代码或数值契约。

### 单 GPU、双 GPU 与 campaign 调度

同一 scale 内对 task \(q=(p,b)\) 估算成本：

\[
C_q \propto T\left[
M n K(4+L_{\mathrm{tr},p})
+n^2
+S n(2+L_{\mathrm{ph},p})
\right],
\]

其中 AS 的 \(S=M\)，ACS/MMAS 的 \(S=1\)。双 GPU 使用 longest-processing-
time greedy shard，而不是按 program 编号简单对半切；counter RNG 只依赖
`seed, instance, iteration, ant, step, stream`，所以 task 移到另一张 GPU
不会改变输出。

运行模式为：

- `single`：一个 run 使用指定的一张 GPU；
- `dual`：一个 run 的 task matrix 分给两张 GPU；
- `campaign`：两张 GPU 各运行一个独立 GP replicate；
- `auto`：单个训练进程有两张可见卡时使用 dual。跨 run 的 ready queue
  由集群 orchestrator 管理：有至少两个 ready runs 时，根据实测门控显式
  启动两个 `campaign` 进程，并分别设置 `RMTGP_ACO_GPU_DEVICE` 或
  `LOCAL_RANK`；训练进程不猜测其他作业的全局状态。

由于 TSP100 约为 TSP50 的两倍成本，不得采用“一张卡固定 TSP50、另一张卡
固定 TSP100”的静态 scale split。

### 性能门控和记录

在同一冻结 generation、同一 programs、instances 和 seeds 上，先 cold run，
再 warm-up，最后至少 3 次完整重复并报告 median。矩阵包括 CPU8、CPU16、
GPU0、GPU1、dual 和两个 concurrent single-GPU runs。定义

\[
S_1=\frac{T_{\mathrm{CPU16}}}{T_{\mathrm{GPU1}}},
\quad
S_2=\frac{T_{\mathrm{CPU16}}}{T_{\mathrm{GPU2}}},
\quad
G_2=\frac{T_{\mathrm{GPU1}}}{T_{\mathrm{GPU2}}},
\quad
E_2=\frac{G_2}{2}.
\]

GPU 正式启用门槛为 \(S_1\ge3\)、\(S_2\ge5\)；单 run 双卡要求
\(G_2\ge1.7\)，campaign 总吞吐要求至少为单卡的 1.8 倍。若质量门控通过
而双卡 scaling 不通过，正式调度改为每张 GPU 一个独立 run，不把“有两张
卡”等同于“单个 run 必须双卡”。

每次 benchmark 至少记录 cold/end-to-end/kernel/H2D/D2H/FP64-scoring
时间、constructed tours、tours/s、设备名、block size、chunk 数和输出
signature。还应通过 `nvidia-smi`/Nsight 记录利用率、带宽、occupancy、
功耗和温度；参与 benchmark 的目标 GPU 上存在其他进程时，结果标记为受
污染，不用于门控。未参与运行的其他设备不属于该次测量的争用域。

2026-07-25 在两张独占 RTX 4000 Ada 上完成正式门控。固定 100 requested
individuals（79 个语义 program）、TSP50/TSP100 各 16 instances、32 ants
和 500 iterations，CPU16、单卡和 dual 的 warm median 分别为 304.325 s、
20.807 s 和 12.776 s。故 \(S_1=14.626\)、\(S_2=23.820\)，但
\(G_2=1.629<1.7\)。两张卡并发两个独立 workload 的吞吐扩展为
1.972 倍，通过 1.8 倍门槛。因此确认性实验按每卡一个独立 run 调度；
dual 只用于降低单 run 代时，不作为默认吞吐策略。完整审计见
`docs/performance/cuda_fused_architecture_20260724.md`。

### 完整 MMAS restart

v0.5 补齐 ACOTSP-1.03 的无局部搜索重启控制。每 100 iterations 计算
\(\lambda=0.05\) 的平均 node branching factor：

\[
b_\lambda
=\frac{1}{2n}\sum_{i=1}^{n}
\left|
\left\{j\in\mathcal N_i:
\tau_{ij}>
\tau_i^{\min}+\lambda(\tau_i^{\max}-\tau_i^{\min})
\right\}
\right|.
\]

若

\[
b_\lambda<1.00001
\quad\land\quad
t-t_{\mathrm{restart\ best}}>250,
\]

则把非对角 pheromone 重置为当前 \(\tau_{\max}\)，清空 restart-best，并从
本次 iteration 重新计时。CPU PyTorch、CPU Numba 和 CUDA 使用相同控制
参数，并把 restart 次数写入 diagnostics。该语义改变要求新的 MMAS
baseline archive。

### 后端正确性门

实现必须通过：

- tour 首尾相同且恰好访问每个城市一次；
- 同 GPU 重复逐位相同；
- GPU0、GPU1、single、dual、chunk partition 输出相同；
- 精确零 residual 恢复该 GPU semantic domain 内的原始 ACO；
- ACS edge multiplicity、AS budget、MMAS bounds/restart 回归测试；
- GPU best tour 的 FP64 重算长度与训练使用长度完全相同；
- CPU/GPU 质量非劣门和最终 champion CPU float64 复评。

## v1.2 协议修订（历史；仅在不与 v1.3 冲突时适用）

本节记录历史 Protocol A v0.4。下文保留的早期候选参数仅用于解释设计演化；如与
本节冲突，以本节为准。

相对于 v0.3，三种 ACO 统一采用：

\[
M=32,\qquad T_{\mathrm{ACO}}=500.
\]

这里 \(T_{\mathrm{ACO}}\) 是每次 ACO simulation 的迭代数；GP 仍采用
population 100、50 generations。v0.3 的 baseline archive、训练 checkpoint
和性能数字均不得用于 v0.4。

### 数据量与模型单位

- AS、ACS、MMAS 分别训练一套 GP；每套 GP 同时学习 TSP50 与 TSP100；
- 每代每规模无放回抽取 16 个实例，50 代共使用每规模 800 个不同实例；
- 所有 GP 个体在同一代共享相同的 instance/ACO-seed；
- 同一 replicate 的全部消融方法共享 schedule，pilot 与 formal schedule
  互不重叠；
- validation 每规模 64 个实例，固定拆为 32 个 selection 实例和 32 个
  holdout gate 实例。全部 checkpoint 先在 selection 上以 1 seed 筛选，
  前 5 名以 3 seeds 重评；最终只把选中的一名送入 3-seed gate；
- ACS 追加一个等计算预算的数据协议对照：TSP50-only、TSP100-only 每代
  各用 32 个本规模实例，mixed 使用 \(16+16\)。

训练 schedule 必须预先生成紧凑 manifest，内容至少包括 protocol、phase、
replicate、generation、scale、源文件、行号、coordinate hash 与 ACO seed。
禁止再把 128 万长度的完整随机排列写入 checkpoint。

### 统一质量指标

数据中的合法 reference tour 未全部附带全局最优性证明，因此统一使用
`reference gap`：

\[
g_i(\theta)
=
100
\frac{L_i(\theta)-L_i^{\mathrm{ref}}}
{L_i^{\mathrm{ref}}}.
\]

训练采用单目标、分规模宏平均 fitness：

\[
F(\theta)
=
\frac12
\left(
\overline g_{50}(\theta)+
\overline g_{100}(\theta)
\right),
\qquad \min F.
\]

原始 ACO 不进入训练 fitness，而作为预计算的 paired baseline。每代同时记录：

\[
\Delta_i
=
g_i(\theta)-g_i^{\mathrm{ACO}},
\]

其中 \(\Delta_i<0\) 表示学习规则优于对应原始 ACO，单位为 percentage
points。原先的 baseline-relative degradation penalty 不再属于主协议。

### 公平的 GP 容量与核心消融

所有单树、双树统一满足：

\[
N_{\mathrm{tr}}+N_{\mathrm{ph}}\le 31.
\]

其中未启用角色的精确零残差哨兵按 0 个有效节点计数，避免单树方法因实现用
的占位节点而平白少一个节点预算。

主方法为 `RMTGP-Full-F1`。确认性消融固定为：

| 方法 | 状态转移 | 信息素 | 表示 |
|---|---|---|---|
| Legacy-GP | replacement | 无 | 上一研究的 terminals/functions |
| Matched-Replace | replacement | 无 | Full/F1 |
| TR-RGP | residual | 无 | Full/F1 |
| PH-RGP | 无 | residual | Full/F1 |
| RMTGP-Core-F0 | residual | residual | Core/F0 |
| RMTGP-Core-F1 | residual | residual | Core/F1 |
| RMTGP-Full-F0 | residual | residual | Full/F0 |
| RMTGP-Full-F1 | residual | residual | Full/F1 |

Transition Core 为 `RTau, REta, BaseConf, DistRank`；Full 再加入
`Entropy, ConstructProg, ACOProg, Stagnation`。Pheromone Core 为
`EdgeEta, EdgeTau, NNRank, SourceQuality`；Full 再加入
`ColonyFreq, ACOProg, Stagnation`。F0 为
`ADD, SUB, MUL, PDIV, NEG`；F1 再加入 `MIN, MAX, ABS`。

Residual 优势由 `TR-RGP - Matched-Replace` 识别；双树优势要求主方法同时
优于 TR-RGP 与 PH-RGP；terminal/function 贡献使用 Core/Full × F0/F1
的 \(2\times2\) factorial contrasts。

### CPU 批量执行与 baseline archive

正式后端优先使用单进程、16 个 Numba threads，把独立任务组织为：

\[
\text{individual}\times\text{instance}\times\text{ACO seed}.
\]

ACO iteration 与 tour construction step 的因果顺序不做伪并行。GP programs
使用连续 packed opcode/argument buffers；训练和 validation 只返回质量与
诊断量。PyTorch 保留为语义参考。只有端到端 float64 GPU 吞吐达到最佳 CPU
后端的 5 倍且质量无漂移，GPU 才可进入正式路径。

原始 ACO 必须在训练前生成不可变 baseline archive。cache key 至少包含
coordinate hash、完整 ACO config hash、seed、dtype 与 kernel semantic
version。正式训练遇到 cache miss 或 hash 不一致必须失败，不允许静默重算。

#### CPU population-batch 的实际并行层级

逻辑任务矩阵为 `GP program × instance`，但当前物理并行轴是 instance，而
不是把整个矩阵扁平化：

```text
for scale in [TSP50, TSP100]:                 # 两个尺度顺序执行
    parallel_for instance in batch[16]:       # Numba prange，16 个线程
        分配并复用该 instance 的工作区
        for semantic_unique_GP_program:        # 在线程内顺序执行
            for ACO_iteration in 1..500:
                构造 32 只蚂蚁的 tours
                更新 best state 与 pheromone
```

同一线程连续处理一个 instance 上的全部程序，使 distance、candidate list、
静态 terminals 和 solver workspace 保持缓存热，并避免每个
program--instance 重复分配大数组。GP population 在进入内核前先按结构哈希
去重，再合并 residual 位置上的语义 intron；结果通过 inverse mapping 展开回
原 population。DEAP 的 selection、crossover 和 mutation 在代间串行执行，
其用时相对 ACO simulation 很小。ACO iteration、tour construction step 和
单个 solver 内的 ants 不启用嵌套线程，以保持确定性并避免过度订阅。

每代实际构造 tour 数为：

\[
N_{\mathrm{tour}}
=
P_{\mathrm{sem}}
\sum_{s\in\{50,100\}}
B_s M T_{\mathrm{ACO}}.
\]

以首代 \(P_{\mathrm{sem}}=79\)、\(B_{50}=B_{100}=16\)、\(M=32\)、
\(T_{\mathrm{ACO}}=500\) 为例：

\[
N_{\mathrm{tour}}
=79\times32\times32\times500
=40{,}448{,}000.
\]

### 推断边界

当前 pilot 对每个 ACO × trainable method 使用 3 个 GP seeds；它只用于验证
实现、耗时和方差，不产生显著性结论。正式论文实验在配置冻结后使用 30 个
独立 GP runs，并以 run→instance→ACO-seed 的层次 bootstrap、paired test
和 Holm 校正进行推断。

因此完整 pilot 为 \(3\times8\times3=72\) 个主消融 runs，另加
ACS 的 \(2\times3=6\) 个单尺度数据协议 runs，总计 78 个训练 runs。

## 0. 执行摘要

### 0.1 核心研究思想

本研究不让 GP 从零重写 ACO，而是在经典 ACO 的两个关键决策位置上学习有界修正：

\[
I^{(v)}
=
\left(
T_{\mathrm{tr}}^{(v)},
T_{\mathrm{ph}}^{(v)}
\right),
\qquad
v\in\{\mathrm{AS},\mathrm{ACS},\mathrm{MMAS}\}.
\]

- \(T_{\mathrm{tr}}\) 修改候选城市的相对优先级，但不改变可行集合；
- \(T_{\mathrm{ph}}\) 在原更新 tour 内重新分配强化预算，但不增加新的更新边；
- AS、ACS、MMAS 分别独立进化双树，避免把三个不同搜索动力学强行压缩到同一模型；
- 当两棵树恒为 0 时，算法严格退化为对应的原始 ACO。

状态转移残差定义为：

\[
\widetilde s_{ij}
=
s^0_{ij}
\left[
1+\gamma_{\mathrm{tr}}
\tanh\left(T_{\mathrm{tr}}(x_{ij})\right)
\right],
\]

其中：

\[
s^0_{ij}
=
\tau_{ij}^{\alpha}\eta_{ij}^{\beta},
\qquad
\eta_{ij}=\frac{1}{d_{ij}+\varepsilon_d}.
\]

全局信息素强化残差定义为：

\[
\widetilde D_{r,e}
=
B_r
\frac{
D^0_{r,e}
\left[
1+\gamma_{\mathrm{ph}}
\tanh\left(T_{\mathrm{ph}}(z_{r,e})\right)
\right]
}{
\sum_{f\in E(r)}
D^0_{r,f}
\left[
1+\gamma_{\mathrm{ph}}
\tanh\left(T_{\mathrm{ph}}(z_{r,f})\right)
\right]
},
\]

其中 \(r\) 表示一个强化来源 tour，\(B_r=\sum_{e\in E(r)}D^0_{r,e}\)。

主配置固定：

\[
\gamma_{\mathrm{tr}}
=
\gamma_{\mathrm{ph}}
=
\frac13.
\]

### 0.2 研究规模与主协议

本研究仅使用：

\[
\mathcal N=\{50,100,500\}.
\]

TSP200、TSP1000 和 TSP10000 不进入本研究的训练、模型选择或主测试。

主泛化协议为：

- 训练与验证：uniform TSP50 + TSP100；
- 测试：TSP500-uniform；
- 分布外测试：TSP500-cluster、TSP500-Gaussian；
- 额外真实分布测试：TSPLIB 中 \(n\le 500\) 的 42 个实例。

### 0.3 ACO 主配置

主实验关闭局部搜索；除统一的蚂蚁数和迭代预算外，其余参数采用
ACOTSP-1.03 中三个算法各自的无局部搜索默认值：

| 变体 | 蚂蚁数 \(M\) | \(\alpha\) | \(\beta\) | 蒸发率 \(\rho\) | 其他参数 |
|---|---:|---:|---:|---:|---|
| AS | 32 | 1 | 2 | 0.50 | 所有蚂蚁全局强化 |
| ACS | 32 | 1 | 2 | 0.10 | \(q_0=0.90,\ \xi=0.10\) |
| MMAS | 32 | 1 | 2 | 0.02 | 动态 \(\tau_{\min},\tau_{\max}\) |

统一设置：

- nearest-neighbour candidate list：\(K=\min(20,n-1)\)；
- 主质量预算：500 个 ACO iterations；
- 次级实用预算：10 秒等时运行；
- 主实验无 2-opt/3-opt；
- 距离使用连续 float64 Euclidean 距离，不采用 ACOTSP 的 TSPLIB 整数舍入。

### 0.4 主要比较方法

对每个 ACO 变体分别比较：

1. 原始 ACO；
2. `Legacy-GP`：上一篇研究的单树 transition 完全替换表示；
3. `Matched-Replace-GP`：使用本研究 terminals/functions、但完全替换 transition；
4. `TR-RGP`：仅使用 transition residual；
5. `PH-RGP`：仅使用 pheromone residual；
6. `RMTGP-ACO`：联合使用两棵 residual trees。

---

# 1. 研究背景与参考实现审计

## 1.1 上一篇研究的定位

Lin、Mei 和 Zhang（2025）研究了用 GP 自动设计 ACO 状态转移规则，其核心做法是：

1. 每个 GP 个体表示一条状态转移规则；
2. 将该规则嵌入 AS、ACS 或 MMAS；
3. 完整运行 ACO；
4. 用最终 tour length 作为 GP fitness；
5. 训练后将最优规则迁移到测试实例。

该研究的重要经验包括：

- GP 规则可以跨同分布实例泛化；
- AS、ACS、MMAS 的动力学会改变规则形态，但不消除 GP 学习能力；
- 额外上下文信息通常优于仅使用 raw pheromone 和 distance；
- 2-opt 可能削弱 MMAS 中 transition rule 的学习信号；
- PyTorch 矩阵化是嵌套 GP–ACO 可行性的关键。

本研究将其作为**方法学基线**，但不要求复现旧实现的程序轨迹。所有旧方法都在本研究的数据、ACO 参数、随机流和计算预算下重新训练。

## 1.2 ACOTSP-1.03 的可复用语义

`references/ACOTSP-1.03` 提供 AS、ACS、MMAS 的统一 C 实现。本研究保留以下设计：

- \(K=20\) 的 nearest-neighbour candidate list；
- candidate list 耗尽后的全未访问城市 fallback；
- AS 的 all-ant deposit；
- ACS 的局部更新与 global-best 全局更新；
- MMAS 的 iteration-best / restart-best 调度；
- MMAS 的动态信息素上下界；
- 无向 TSP 信息素矩阵的对称更新；
- nearest-neighbour tour 驱动的信息素初始化。

## 1.3 有意改变的语义

### 连续距离

ACOTSP 面向 TSPLIB，默认把二维 Euclidean 距离舍入为整数。本研究数据坐标多位于 \([0,1]^2\)，若直接舍入会产生大量 0 或 1 距离，因此必须使用：

\[
d_{ij}
=
\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}.
\]

所有 reference length 也按同一连续距离重新计算。

### 固定迭代预算

ACOTSP 可按 wall-clock time 或 constructed tours 终止。本研究主质量实验固定
500 iterations，避免 GP 树的推理开销改变搜索步数。10 秒等时结果单独报告，
回答实际部署效率问题。

### ACS 同步步进

ACOTSP 按蚂蚁顺序选边，后续蚂蚁能看到先前蚂蚁刚执行的 local update。本研究为支持蚂蚁维度向量化，采用：

1. 同一步的全部蚂蚁读取同一个 \(\tau\) 快照；
2. 全部蚂蚁完成本步选择；
3. 按被使用次数聚合 local update；
4. 进入下一 construction step。

这不是逐行复现 ACOTSP，而是预注册的矩阵化 ACS 语义。顺序版与同步版的差异必须作为实验报告。

---

# 2. 研究目标、问题与假设

## 2.1 总体目标

研究目标是：

> 在保留经典 ACO 算法结构、候选机制和强化支持集的前提下，通过两棵可解释 GP 树自动学习状态转移信用与信息素强化信用的有界重分配，并检验其跨规模、跨分布和跨 ACO 变体的泛化能力。

## 2.2 研究问题

### RQ1：单一 residual 是否有效？

- transition residual 是否优于原始 ACO？
- pheromone residual 是否优于原始 ACO？

### RQ2：双树是否产生增量价值？

\[
(T_{\mathrm{tr}},T_{\mathrm{ph}})
\]

是否显著优于：

\[
(T_{\mathrm{tr}},0)
\quad\text{和}\quad
(0,T_{\mathrm{ph}})?
\]

### RQ3：residual 表示是否优于完全替换？

相对于 single-tree full replacement，双树 residual 是否具有：

- 更低的测试退化率；
- 更好的 TSP500 外推；
- 更小的性能方差；
- 更稳定的数值行为；
- 更清晰的组件归因？

### RQ4：不同 ACO 变体是否学习不同规则？

AS、ACS、MMAS 的最优 trees 在结构、terminal 使用、输出分布和迁移能力上是否显著不同？

### RQ5：能否从 TSP50/100 外推到 TSP500？

训练期间从未出现 TSP500 时，规则能否改善：

- TSP500-uniform；
- TSP500-cluster；
- TSP500-Gaussian？

### RQ6：两棵树是否 co-adapt？

随机打乱不同 GP runs 的 transition/pheromone tree 配对后，性能是否下降？

### RQ7：哪些 terminals 和 functions 真正重要？

需要区分：

- 局部 pheromone/geometry；
- candidate context；
- colony/search-state context；
- 显式规模信息；
- 变体特有信息。

### RQ8：矩阵化是否改变结论？

- ACS 同步更新相对顺序更新的差异多大？
- PyTorch/Numba 加速是否保持统计结论？
- GP inference overhead 是否可接受？

## 2.3 主要假设

- **H1**：至少一棵 residual tree 能显著改善对应 ACO baseline；
- **H2**：联合双树优于两个单树模型；
- **H3**：residual 比 full replacement 具有更低的 OOD 退化率；
- **H4**：相对化、无量纲 terminals 支持 TSP50/100 到 TSP500 的外推；
- **H5**：AS、ACS、MMAS 会形成可区分的规则结构；
- **H6**：原始配对优于 shuffled pairing，说明存在 coadaptation；
- **H7**：向量化后，训练耗时相对候选级 Python 实现显著下降。

---

# 3. 问题定义、符号与数值约定

## 3.1 TSP 定义

给定完全无向图：

\[
G=(V,E),\qquad |V|=n,
\]

城市 \(i\) 的坐标为：

\[
\mathbf c_i=(x_i,y_i)\in\mathbb R^2.
\]

边长为：

\[
d_{ij}=\|\mathbf c_i-\mathbf c_j\|_2.
\]

一个合法 tour：

\[
\pi=(\pi_0,\pi_1,\ldots,\pi_{n-1},\pi_0)
\]

必须恰好访问每个城市一次，其长度：

\[
L(\pi)
=
\sum_{k=0}^{n-1}
d_{\pi_k,\pi_{k+1}}.
\]

## 3.2 启发式信息

\[
\eta_{ij}
=
\frac{1}{d_{ij}+\varepsilon_d}.
\]

默认：

\[
\varepsilon_d
=
10^{-12}
\max\left(1,\operatorname{median}_{i<j}d_{ij}\right).
\]

对角线位置不可选，令其 desirability 为 0。

## 3.3 数值精度

正式实验规定：

- coordinates：float64；
- distance/reference length：float64；
- pheromone/desirability：float64；
- city/tour/edge ID：int64；
- masks：bool；
- GP opcode：int16 或 int32；
- 最终输出禁止 NaN 和 \(\pm\infty\)。

float32 只用于开发期速度评估，不产生论文主结果。

## 3.4 主要符号

| 符号 | 含义 |
|---|---|
| \(n\) | 城市数 |
| \(M\) | 蚂蚁数 |
| \(K\) | candidate list 长度 |
| \(t\) | ACO iteration |
| \(q\) | tour construction step |
| \(F_{a,q}\) | 蚂蚁 \(a\) 当前可行候选集合 |
| \(\tau_{ij}\) | 边 \((i,j)\) 的信息素 |
| \(\eta_{ij}\) | heuristic information |
| \(s^0_{ij}\) | baseline desirability |
| \(p^0_{ij}\) | baseline roulette probability |
| \(T_{\mathrm{tr}}\) | transition GP tree |
| \(T_{\mathrm{ph}}\) | pheromone GP tree |
| \(D^0_{r,e}\) | 来源 tour \(r\) 对边 \(e\) 的 baseline deposit |
| \(B_r\) | 来源 tour \(r\) 的总 deposit budget |
| \(L^\star_i\) | 实例 \(i\) 的 reference length |

---

# 4. 统一 ACO 外壳

## 4.1 Candidate list

对每个城市 \(i\)，按连续距离升序预计算：

\[
C_i=(c_{i,1},\ldots,c_{i,K}),
\qquad
K=\min(20,n-1).
\]

在 construction step \(q\)，蚂蚁 \(a\) 位于城市 \(i\)，首先使用：

\[
F_{a,q}
=
\{j\in C_i:j\notin V_{a,q}^{\mathrm{visited}}\}.
\]

若 \(F_{a,q}=\varnothing\)，则 fallback 为：

\[
F_{a,q}
=
V\setminus V_{a,q}^{\mathrm{visited}}.
\]

按照 ACOTSP 行为，fallback 使用全可行集合上的最大 desirability 选择，而不是重新执行 roulette。

GP residual 在 candidate-list 和 fallback 两种状态中都生效。

## 4.2 初始城市

每只蚂蚁独立、均匀地选择初始城市：

\[
\Pr(\pi^a_0=i)=\frac1n.
\]

同一个实例、同一个 seed 下，baseline 和所有 GP 个体使用相同初始城市张量。

## 4.3 Nearest-neighbour 初始化 tour

按照 ACOTSP 思路，从一个由随机流确定的起点构造 nearest-neighbour tour，长度记为：

\[
L_{\mathrm{nn}}.
\]

该随机起点属于 ACO 随机流的一部分，并被 baseline 与 GP 方法共享。

## 4.4 基础 desirability

\[
s^0_{aij}
=
\tau_{ij}^{\alpha}\eta_{ij}^{\beta}.
\]

对 roulette 分支：

\[
p^0_{aij}
=
\frac{s^0_{aij}}
{\sum_{\ell\in F_{a,q}}s^0_{ai\ell}}.
\]

若所有有效 score 因数值异常变为 0，则使用可行集合上的均匀分布，并记录 diagnostic counter。

---

# 5. Ant System

## 5.1 参数与初始化

\[
M=32,\quad
\alpha=1,\quad
\beta=2,\quad
\rho=0.5,\quad
q_0=0.
\]

初始信息素：

\[
\tau_0
=
\frac{1}{\rho L_{\mathrm{nn}}}.
\]

所有非对角边初始化为 \(\tau_0\)。

## 5.2 Solution construction

AS 在 candidate list 未耗尽时始终按 roulette 选择：

\[
j\sim \operatorname{Categorical}(p_{aij}).
\]

fallback 时选择：

\[
j
=
\arg\max_{\ell\in F_{a,q}}\widetilde s_{ai\ell}.
\]

## 5.3 Baseline 信息素更新

每轮先蒸发：

\[
\tau_e
\leftarrow
(1-\rho)\tau_e.
\]

第 \(a\) 只蚂蚁的 tour 为 \(\pi^a\)，长度为 \(L_a\)，其 baseline deposit：

\[
D^0_{a,e}
=
\begin{cases}
\dfrac{1}{L_a}, & e\in E(\pi^a),\\
0, & \text{otherwise}.
\end{cases}
\]

最终：

\[
\tau_e
\leftarrow
\tau_e+\sum_{a=1}^{M}D^0_{a,e}.
\]

## 5.4 AS 中的 pheromone residual event

每只蚂蚁构成一个独立 event：

\[
r=a,\qquad
B_a
=
\sum_{e\in E(\pi^a)}D^0_{a,e}
=
\frac{n}{L_a}.
\]

先在每条 tour 内重分配预算，再对重复边执行 `scatter_add`。

---

# 6. Ant Colony System

## 6.1 参数与初始化

\[
M=32,\quad
\alpha=1,\quad
\beta=2,\quad
\rho=0.1,\quad
q_0=0.9,\quad
\xi=0.1.
\]

初始信息素：

\[
\tau_0
=
\frac{1}{nL_{\mathrm{nn}}}.
\]

## 6.2 Pseudorandom proportional rule

对每只蚂蚁采样：

\[
q\sim U(0,1).
\]

若 \(q\le q_0\)：

\[
j
=
\arg\max_{\ell\in F_{a,q}}\widetilde s_{ai\ell}.
\]

否则：

\[
j\sim
\operatorname{Categorical}(p_{aij}).
\]

同一 residual score 同时服务 greedy 和 roulette 分支。

## 6.3 同步局部更新

一个 construction step 完成后，令 \(c_e\) 为该步选择边 \(e\) 的蚂蚁数。

单次 ACS local update：

\[
\mathcal L(\tau_e)
=(1-\xi)\tau_e+\xi\tau_0.
\]

连续应用 \(c_e\) 次的闭式形式：

\[
\boxed{
\tau_e
\leftarrow
(1-\xi)^{c_e}\tau_e
+
\left[1-(1-\xi)^{c_e}\right]\tau_0
}.
\]

该式避免对同一边逐蚂蚁循环，并保持重复 local update 的代数结果。

## 6.4 Baseline 全局更新

ACS 仅使用 global-best tour \(\pi^{\mathrm{gb}}\)。

\[
D^0_e
=
\frac{1}{L_{\mathrm{gb}}},
\qquad
e\in E(\pi^{\mathrm{gb}}).
\]

只有 global-best edges 执行：

\[
\tau_e
\leftarrow
(1-\rho)\tau_e+\rho D^0_e.
\]

非 global-best 边不执行该全局更新。

## 6.5 ACS 中的 pheromone residual

ACS 每轮只有一个全局强化 event：

\[
r=\pi^{\mathrm{gb}},
\qquad
B_r=\frac{n}{L_{\mathrm{gb}}}.
\]

\(T_{\mathrm{ph}}\) 只重分配该 event，不读取或修改 local update action。

---

# 7. MAX–MIN Ant System

## 7.1 参数与初始化

\[
M=32,\quad
\alpha=1,\quad
\beta=2,\quad
\rho=0.02,\quad
q_0=0.
\]

初始上下界：

\[
\tau_{\max}
=
\frac{1}{\rho L_{\mathrm{nn}}},
\qquad
\tau_{\min}
=
\frac{\tau_{\max}}{2n}.
\]

所有非对角边初始化为 \(\tau_{\max}\)。

## 7.2 Solution construction

candidate list 有可行城市时使用 roulette；fallback 使用全可行集合最大 residual desirability。

## 7.3 更新 tour 调度

无局部搜索时，ACOTSP 将 \(u_{\mathrm{gb}}\) 固定为 25：

- 当 \(t\bmod 25\ne0\)：iteration-best；
- 当 \(t\bmod 25=0\)：restart-best。

主实验运行 500 iterations，因此必须记录并审计 ACOTSP 中超过 250 次停滞
时可能触发的 restart 语义；当前实现保留 restart-best 状态，正式 MMAS
运行前需用专项回归确认 pheromone restart 行为。

## 7.4 动态信息素界

当发现新的 global best 时：

\[
p_x
=
\exp\left(\frac{\log 0.05}{n}\right),
\]

\[
\tau_{\max}
=
\frac{1}{\rho L_{\mathrm{gb}}},
\]

\[
\tau_{\min}
=
\tau_{\max}
\frac{1-p_x}
{p_x\lfloor(K+1)/2\rfloor}.
\]

## 7.5 信息素更新

先蒸发：

\[
\tau_e
\leftarrow
(1-\rho)\tau_e.
\]

被选中的 update tour \(r\) 提供：

\[
D^0_{r,e}
=
\frac{1}{L_r},
\qquad e\in E(r).
\]

加入 residual deposit 后执行：

\[
\tau_e
\leftarrow
\operatorname{clip}
\left(
\tau_e+\widetilde D_{r,e},
\tau_{\min},
\tau_{\max}
\right).
\]

---

# 8. Transition residual tree

## 8.1 定义

对 active feasible set 中每个候选 \(j\)：

\[
u_{aij}
=
T_{\mathrm{tr}}(x_{aij}),
\]

\[
r_{aij}
=
\tanh(u_{aij}),
\]

\[
m_{aij}
=
1+\gamma_{\mathrm{tr}}r_{aij}.
\]

最终 score：

\[
\boxed{
\widetilde s_{aij}
=
s^0_{aij}m_{aij}
}.
\]

roulette 概率：

\[
\boxed{
p_{aij}
=
\frac{\widetilde s_{aij}}
{\sum_{\ell\in F_{a,q}}\widetilde s_{ai\ell}}
}.
\]

## 8.2 性质

### Baseline recoverability

当 \(T_{\mathrm{tr}}\equiv0\)：

\[
\widetilde s=s^0,\qquad p=p^0.
\]

### Support preservation

\[
p^0_{aij}=0
\Longrightarrow
p_{aij}=0.
\]

### Positivity

当 \(0<\gamma_{\mathrm{tr}}<1\)：

\[
m_{aij}\in
[1-\gamma_{\mathrm{tr}},1+\gamma_{\mathrm{tr}}].
\]

### Probability ratio bound

\[
\frac{1-\gamma_{\mathrm{tr}}}
{1+\gamma_{\mathrm{tr}}}
\le
\frac{p_{aij}}{p^0_{aij}}
\le
\frac{1+\gamma_{\mathrm{tr}}}
{1-\gamma_{\mathrm{tr}}}.
\]

当 \(\gamma_{\mathrm{tr}}=1/3\)：

\[
\frac12
\le
\frac{p_{aij}}{p^0_{aij}}
\le
2.
\]

---

# 9. Pheromone residual tree

## 9.1 Event 表示

一个全局强化来源表示为：

\[
r=(\pi_r,L_r,w_r),
\]

其中：

- \(\pi_r\)：来源 tour；
- \(L_r\)：tour length；
- \(w_r\)：baseline 权重。

对本研究三个变体：

| 变体 | 每轮 event 数 \(R\) | \(w_r\) |
|---|---:|---:|
| AS | \(M\) | 1 |
| ACS | 1 | ACS global coefficient 在外层使用 |
| MMAS | 1 | 1 |

baseline event deposit：

\[
D^0_{r,e}
=
\frac{w_r}{L_r},
\qquad e\in E(\pi_r).
\]

总预算：

\[
B_r
=
\frac{nw_r}{L_r}.
\]

## 9.2 有界预算重分配

\[
v_{r,e}
=
T_{\mathrm{ph}}(z_{r,e}),
\]

\[
h_{r,e}
=
1+\gamma_{\mathrm{ph}}\tanh(v_{r,e}),
\]

\[
\boxed{
\widetilde D_{r,e}
=
B_r
\frac{D^0_{r,e}h_{r,e}}
{\sum_{f\in E(\pi_r)}D^0_{r,f}h_{r,f}}
}.
\]

## 9.3 性质

1. \(T_{\mathrm{ph}}\equiv0\) 时恢复 baseline deposit；
2. baseline 未强化的边不会被新增；
3. 每个来源 tour 独立守恒：

   \[
   \sum_{e\in E(\pi_r)}
   \widetilde D_{r,e}
   =
   B_r;
   \]

4. AS 不会把较差蚂蚁的总预算转移给较好蚂蚁；
5. MMAS 的 clipping 与 update-tour 调度不变；
6. ACS 的 local update 不变。

---

# 10. Strongly Typed Multi-Tree GP

## 10.1 个体结构

\[
I=(T_{\mathrm{tr}},T_{\mathrm{ph}}).
\]

两棵树共享 fitness，但具有不同的：

- marker type；
- terminal set；
- 输入 shape；
- 输出语义；
- opcode vocabulary。

## 10.2 类型定义

| 类型 | 语义 | 运行时 shape |
|---|---|---|
| `TrField` | candidate-level 数值场 | `[B,M,K_active]` |
| `PhField` | event-edge-level 数值场 | `[B,R,n]` |
| `TrProgram` | transition postfix opcode 序列 | `[N_tr_ops]` |
| `PhProgram` | pheromone postfix opcode 序列 | `[N_ph_ops]` |
| `FitnessMin` | 单目标最小化 fitness | scalar |

DEAP 使用两个独立的 `PrimitiveSetTyped`：

```text
pset_tr: () -> TrField
pset_ph: () -> PhField
```

terminal 是带名称的符号引用。自定义 interpreter 在运行时把符号映射到当前 context tensor。

ERC 分别产生 typed symbolic constant：

```text
ConstTr(c): TrField
ConstPh(c): PhField
```

interpreter 将其广播到相应 shape。

## 10.3 强类型 primitive 签名

对 `TrField`：

```text
add_tr(TrField, TrField) -> TrField
sub_tr(TrField, TrField) -> TrField
mul_tr(TrField, TrField) -> TrField
pdiv_tr(TrField, TrField) -> TrField
min_tr(TrField, TrField) -> TrField
max_tr(TrField, TrField) -> TrField
abs_tr(TrField) -> TrField
neg_tr(TrField) -> TrField
```

对 `PhField` 定义同构但类型不同的 primitives。

任何 `TrField` subtree 都不能替换 `PhField` subtree。

## 10.4 Individual 接口

逻辑接口：

```text
RMTGPIndividual:
    transition_tree: PrimitiveTree[TrField]
    pheromone_tree: PrimitiveTree[PhField]
    fitness: FitnessMin
    structural_hash: str
    metadata:
        generation_created
        parent_ids
        operator
```

---

# 11. Transition terminal set

记 active feasible set 大小：

\[
k_f=|F_{a,q}|.
\]

定义稳定相对标准化：

\[
\operatorname{stdrel}(y_j)
=
\tanh
\left(
\frac{y_j-\operatorname{mean}_{\ell\in F}y_\ell}
{\operatorname{std}_{\ell\in F}y_\ell+\varepsilon_z}
\right),
\]

其中 \(\varepsilon_z=10^{-8}\)。

## 11.1 主 terminal set

\[
\boxed{
\mathcal T_{\mathrm{tr}}
=
\{
RTau,
REta,
BaseConf,
DistRank,
Entropy,
ConstructProg,
ACOProg,
Stagnation,
ERC
\}
}
\]

### `RTau`

\[
RTau_{ij}
=
\operatorname{stdrel}
\left(\log(\tau_{ij}+\varepsilon_\tau)\right).
\]

范围：\([-1,1]\)。

### `REta`

\[
REta_{ij}
=
\operatorname{stdrel}
\left(\log(\eta_{ij}+\varepsilon_\eta)\right).
\]

### `BaseConf`

\[
BaseConf_{ij}
=
\tanh
\left(
\log(p^0_{ij}+\varepsilon_p)
+
\log k_f
\right).
\]

当 baseline 为均匀分布时接近 0。

### `DistRank`

候选按距离从近到远的 rank 为 \(r_d\in\{1,\ldots,k_f\}\)：

\[
DistRank_{ij}
=
\begin{cases}
1-\dfrac{2(r_d-1)}{k_f-1}, & k_f>1,\\
0, & k_f=1.
\end{cases}
\]

### `Entropy`

\[
H(p^0_i)
=
-\sum_{j\in F}
p^0_{ij}\log(p^0_{ij}+\varepsilon_p),
\]

\[
Entropy_i
=
\begin{cases}
\dfrac{2H(p^0_i)}{\log k_f}-1, & k_f>1,\\
-1, & k_f=1.
\end{cases}
\]

该 scalar broadcast 到全部候选。

### `ConstructProg`

\[
ConstructProg
=
\frac{2q}{n-1}-1.
\]

### `ACOProg`

\[
ACOProg
=
\frac{2(t-1)}{T-1}-1.
\]

### `Stagnation`

令 \(s_t\) 为距上次 global-best 改进的 iterations：

\[
Stagnation
=
2\min\left(\frac{s_t}{T},1\right)-1.
\]

### `ERC`

\[
ERC\sim U[-1,1].
\]

## 11.2 Transition ablation terminals

| Terminal | 定义与目的 |
|---|---|
| `RawTau` | 原始 \(\tau_{ij}\)，用于旧论文表示 |
| `RawDist` | 原始 \(d_{ij}\)，用于旧论文表示 |
| `MeanTau` | 当前候选平均 pheromone |
| `MeanDist` | 当前候选平均 distance |
| `NumCities` | raw \(n\) |
| `NumFeasible` | raw \(k_f\) |
| `LogSize` | 映射后的 \(\log n\)，检验显式规模条件化 |
| `TauRank` | pheromone rank，检验对异常值的稳健性 |

主模型不使用 raw scale terminals。

---

# 12. Pheromone terminal set

## 12.1 主 terminal set

\[
\boxed{
\mathcal T_{\mathrm{ph}}
=
\{
EdgeEta,
EdgeTau,
NNRank,
ColonyFreq,
SourceQuality,
ACOProg,
Stagnation,
ERC
\}
}
\]

## 12.2 `EdgeEta`

对无向边 \(e=(u,v)\)，定义两个端点 candidate neighbourhood 的平均 log heuristic：

\[
\mu_u^\eta
=
\frac{1}{|C_u|}
\sum_{j\in C_u}
\log(\eta_{uj}+\varepsilon_\eta).
\]

\[
g_e^\eta
=
\log(\eta_{uv}+\varepsilon_\eta)
-
\frac12
\left(
\mu_u^\eta+\mu_v^\eta
\right).
\]

在每个 event 内标准化：

\[
EdgeEta_e
=
\operatorname{stdrel}_{f\in E(\pi_r)}
(g_f^\eta).
\]

## 12.3 `EdgeTau`

\[
EdgeTau_e
=
\operatorname{stdrel}_{f\in E(\pi_r)}
\left(
\log(\tau_f+\varepsilon_\tau)
\right).
\]

## 12.4 `NNRank`

预计算每个端点到其余 \(n-1\) 个城市的完整距离 rank。令归一化 rank：

\[
\widehat r_u(v)
=
1-
\frac{2(r_u(v)-1)}{n-2}.
\]

无向边取对称平均：

\[
NNRank_{(u,v)}
=
\frac{
\widehat r_u(v)+\widehat r_v(u)
}{2}.
\]

## 12.5 `ColonyFreq`

令 \(c_e\) 为本轮 \(M\) 条 constructed tours 中包含边 \(e\) 的数量：

\[
ColonyFreq_e
=
\frac{2c_e}{M}-1.
\]

## 12.6 `SourceQuality`

令本轮 colony tour lengths 的均值和标准差为 \(\mu_L,\sigma_L\)。来源 tour \(r\)：

\[
SourceQuality_r
=
\tanh
\left(
\frac{\mu_L-L_r}
{\sigma_L+\varepsilon_L}
\right).
\]

该值 broadcast 到 event 中全部边。

在 AS 中，它允许同一树根据 ant quality 使用不同的边重分配模式；在 ACS/MMAS 中，它表示被选中 tour 相对本轮 colony 的质量。

## 12.7 Pheromone ablation terminals

| Terminal | 使用范围 | 作用 |
|---|---|---|
| `BoundPosition` | MMAS | \(\tau_e\) 在 \([\tau_{\min},\tau_{\max}]\) 中的位置 |
| `UpdateKind` | MMAS | iteration-best / restart-best / global-best |
| `EdgeAge` | 全部 | 距边上次被强化的 iterations |
| `BestImprove` | 全部 | 当前 global-best 的相对改进 |
| `ColonyDiversity` | 全部 | colony edge diversity |

这些 terminals 不进入第一版主配置。

---

# 13. Function set 与数值保护

## 13.1 主 function set

\[
\boxed{
\mathcal F
=
\{
+,-,\times,\operatorname{pdiv},
\min,\max,\operatorname{abs},\operatorname{neg}
\}
}
\]

连续 protected division：

\[
\operatorname{pdiv}(x,y)
=
\frac{xy}{y^2+\varepsilon_{\mathrm{div}}},
\qquad
\varepsilon_{\mathrm{div}}=10^{-6}.
\]

## 13.2 节点后处理

每个 primitive 输出执行：

\[
z
\leftarrow
\operatorname{nan\_to\_num}
(z,0,10,-10),
\]

\[
z
\leftarrow
\operatorname{clip}(z,-10,10).
\]

树根输出再经过外部 `tanh`。

## 13.3 Constants

精确常数：

\[
\{-1,-0.5,0,0.5,1\}.
\]

ERC：

\[
c\sim U[-1,1].
\]

ERC mutation：

\[
c'
=
\operatorname{clip}
(c+\epsilon,-1,1),
\qquad
\epsilon\sim\mathcal N(0,0.1^2).
\]

## 13.4 不进入主 function set 的运算

- `exp`、`log`：只在 terminal preprocessing 或外部 wrapper 中使用；
- 任意幂：数值不稳定；
- `sin`、`cos`：缺少明确归纳偏置；
- reduction：所有 mean/std/rank/entropy 在树外计算；
- `if-then-else`：搜索空间与不连续性过大；
- 直接修改 pheromone matrix 的 action primitive：破坏 residual 边界。

---

# 14. GP 初始化与遗传算子

## 14.1 种群组成

| 初始类型 | 比例 |
|---|---:|
| \((0,0)\) | 10% |
| \((T_{\mathrm{tr}},0)\) | 20% |
| \((0,T_{\mathrm{ph}})\) | 20% |
| \((T_{\mathrm{tr}},T_{\mathrm{ph}})\) | 50% |

随机树使用 ramped half-and-half，初始深度 2–4。

至少一个 \((0,0)\) 个体永久保存在 archive。

## 14.2 Crossover

Role-preserving subtree crossover：

1. 以 0.5 概率选择 transition 或 pheromone role；
2. 只交换两个 parent 的同角色、同类型 subtree；
3. 另一棵树原样保留；
4. 若超过深度或节点上限，则拒绝本次操作并复制 parent。

## 14.3 Mutation

先以 0.5 概率选择一棵树，再选择：

| Mutation | 条件概率 |
|---|---:|
| typed subtree mutation | 0.50 |
| typed point mutation | 0.30 |
| ERC perturbation | 0.20 |

## 14.4 推荐 GP 参数

| 参数 | 值 |
|---|---:|
| Population | 100 |
| Generations | 50 |
| Crossover | 0.80 |
| Mutation | 0.15 |
| Reproduction | 0.05 |
| Elites | 10 |
| Tournament size | 4 |
| Initial depth | 2–4 |
| Max depth per tree | 5 |
| Max nodes per tree | 31 |
| Max effective nodes per individual | 31 |
| Development GP runs | 10 |
| Final GP runs | 30 |

树复杂度不直接混入第一版 fitness；validation 性能近似相同时，以总节点数作为 tie-breaker。

主实验采用

\[
N_{\mathrm{tr}}+N_{\mathrm{ph}}\le31
\]

而不是令两棵树分别都可使用 31 个节点。其目的不是声称这是 Multi-Tree GP
的通用约定，而是在单树与双树对照中固定个体总符号容量：若双树可使用
62 个节点而单树只能使用 31 个节点，则性能差异同时混入角色分解与表达
容量翻倍，无法单独归因于 Multi-Tree 结构。零残差哨兵不计入有效节点。

节点数只是表达容量的可解释代理，不等同于精确运行成本，因为 transition
与 pheromone 两棵树的调用频率和 tensor 形状不同。因此另行报告逐 champion
的 warm inference timing，并执行第 22.9 节的 31/62 节点容量敏感性实验。

---

# 15. 上一篇研究与表示消融

## 15.1 `Legacy-GP`

单棵树完全替代 baseline desirability：

\[
y_{ij}=T_{\mathrm{legacy}}(x_{ij}).
\]

为保证任意 signed GP 输出可用于 ACO，统一使用：

\[
s^{\mathrm{legacy}}_{ij}
=
\operatorname{softplus}
\left(
\operatorname{clip}(y_{ij},-20,20)
\right)
+
\varepsilon_s.
\]

该映射单调、严格为正，并使 ACS greedy 和 roulette 共用同一 score。

### 基础 terminal set

\[
\{\tau_{ij},d_{ij},ERC\}.
\]

### 扩展 terminal set

\[
\{
\tau_{ij},
d_{ij},
\overline\tau_i,
\overline d_i,
n,
k_f,
ERC
\}.
\]

### Legacy function set

\[
\{+,-,\times,\operatorname{pdiv}_1,\operatorname{neg}\},
\]

其中：

\[
\operatorname{pdiv}_1(x,y)
=
\begin{cases}
x/y,&|y|>\varepsilon,\\
1,&\text{otherwise}.
\end{cases}
\]

## 15.2 `Matched-Replace-GP`

使用本研究的 transition terminals、主 function set 和 GP 参数，但仍完全替换 \(s^0\)。

该对照隔离：

\[
\text{residual representation}
\quad\text{vs}\quad
\text{full replacement representation}.
\]

## 15.3 公平性要求

所有 GP 方法必须共享：

- ACO 外壳；
- 训练实例；
- ACO random streams；
- GP population/generations；
- validation selection；
- test instances/seeds；
- 计算预算。

旧论文已发表数值不进入当前统计表。

---

# 16. 数据设计

## 16.1 文件格式

每一行：

```text
x_1 y_1 ... x_n y_n output v_0 v_1 ... v_n
```

其中：

- 前 \(2n\) 个 token 为 coordinates；
- `output` 是分隔符；
- 后 \(n+1\) 个 token 为一基 city IDs；
- \(v_0=v_n\)；
- \(v_0,\ldots,v_{n-1}\) 是 \(1,\ldots,n\) 的排列。

内部统一转换为零基 city IDs。

## 16.2 数据清单

### Training

| Scale | 文件 | 命名标称实例数 |
|---|---:|---:|
| TSP50 | 10 × `128k` | 1,280,000 |
| TSP100 | 10 × `128k` | 1,280,000 |
| TSP500 | 4 × `16k` | 64,000 |

### Validation

| Scale | 分布 | 实例数 |
|---|---|---:|
| TSP50 | uniform | 1,280 |
| TSP100 | uniform | 1,280 |
| TSP500 | uniform | 128 |

### Test

| 数据 | 分布 | 实例数 |
|---|---|---:|
| TSP50 | uniform / Concorde-labelled | 1,280 |
| TSP100 | uniform / Concorde-labelled | 1,280 |
| TSP500 | uniform | 128 |
| TSP500 | cluster | 128 |
| TSP500 | Gaussian | 128 |
| TSPLIB | heterogeneous, \(n\le500\) | 42 |
| TSPLIB extra | heterogeneous, \(n>500\) | 7 |

`tsp100_concorde_7.756 copy.txt` 与原文件校验值相同，必须排除。

## 16.3 Reference length

数据文件不单独存储长度，必须由给定 tour 重新计算：

\[
L_i^\star
=
\sum_{k=0}^{n-1}
d_{v_k,v_{k+1}}.
\]

除非能证明 tour 是 optimum，否则统一称为 `reference tour` / `reference length`，不称为 optimal。

## 16.4 数据验证

每行必须通过：

1. token 数与 \(n\) 一致；
2. coordinates 全部 finite；
3. 存在且仅存在一个 `output`；
4. tour 长度为 \(n+1\)；
5. tour 闭合；
6. 前 \(n\) 个 city IDs 构成完整排列；
7. reference length finite 且 \(>0\)；
8. coordinate hash 与 split manifest 一致。

## 16.5 防泄漏

- 不重新混合现有 train/validation/test；
- test 不用于 terminal/function/\(\gamma\) 选择；
- TSP500-cluster/Gaussian 永远只用于最终测试；
- reference tour edges 不进入 terminals；
- train/validation/test 使用不同 root seeds；
- 同一个 coordinate hash 不得出现在多个 split；
- TSP200/1K/10K 不得被误采样。

---

# 17. 数据协议

## 17.1 Protocol A：Scale extrapolation（主协议）

训练：

\[
\{TSP50,TSP100\}_{\mathrm{train}}.
\]

模型选择：

\[
\{TSP50,TSP100\}_{\mathrm{validation}}.
\]

锁定后测试：

\[
TSP500_{\mathrm{uniform}},
TSP500_{\mathrm{cluster}},
TSP500_{\mathrm{Gaussian}}.
\]

回答规模与分布联合外推问题。

## 17.2 Protocol B：Mixed-scale

训练与验证：

\[
\{TSP50,TSP100,TSP500\}.
\]

测试三个规模的 uniform test。

回答一套规则覆盖完整范围时的 in-range 性能。

## 17.3 Protocol C：Single-scale transfer

独立训练：

\[
Train\text{-}50,\quad
Train\text{-}100,\quad
Train\text{-}500.
\]

在 50、100、500 上交叉测试，得到 \(3\times3\) transfer matrix。

## 17.4 Protocol D：Cross-ACO transfer

将某个 ACO 变体进化的两棵树直接放入其他变体的相同 typed interface：

\[
TrainVariant
\times
TestVariant.
\]

不进行再训练，检验规则是否依赖特定 ACO 动力学。

---

# 18. Fitness 与训练协议

## 18.1 Reference gap

方法 \(A\) 在实例 \(i\)、seed \(r\) 上：

\[
g_{A,i,r}
=
100
\frac{L_{A,i,r}-L_i^\star}
{L_i^\star}.
\]

## 18.2 Baseline-relative paired difference

对同一实例和随机流：

\[
\delta_{I,i,r}
=
100
\frac{L_{I,i,r}-L_{0,i,r}}
{L_i^\star}.
\]

- \(\delta<0\)：优于 baseline；
- \(\delta=0\)：相同；
- \(\delta>0\)：退化。

## 18.3 Scale-balanced fitness

对 active scales \(S\)：

\[
\overline\delta_s
=
\operatorname{mean}_{i\in B_s,r}
\delta_{I,i,r}.
\]

训练 fitness：

\[
\boxed{
F(I)
=
\frac{1}{|S|}
\sum_{s\in S}
\left[
\overline\delta_s
+
\lambda_{\mathrm{deg}}
\operatorname{mean}
\max(0,\delta_{I,i,r})
\right]
},
\]

默认：

\[
\lambda_{\mathrm{deg}}=1.
\]

## 18.4 Mini-batch

每个 generation：

- Protocol A：抽 1 个 TSP50 + 1 个 TSP100；
- Protocol B：抽 1 个 TSP50 + 1 个 TSP100 + 1 个 TSP500；
- 单规模协议：抽 2 个该规模实例。

所有个体共享同一个 mini-batch 和随机流。

每个 GP run 内按无放回打乱的索引流取样；索引用尽后重新洗牌。

## 18.5 Common random numbers

对同一 generation、instance、ACO variant：

- initial cities 相同；
- ACS \(q_0\) uniforms 相同；
- roulette uniforms 相同；
- nearest-neighbour initialization start 相同；
- baseline 和所有 GP 个体相同。

随机数由层次 seed 唯一确定：

```text
root_seed
  / split
  / aco_variant
  / gp_run
  / generation
  / instance_id
  / aco_replication
```

## 18.6 Checkpoint 与 champion selection

1. 每 5 generations 保存 top-5；
2. 合并全部 checkpoints，按 structural hash 去重；
3. 在固定 validation subset 上用 5 个独立 seeds 重评；
4. 按 scale-balanced \(\overline\delta\) 排序；
5. 差异小于 0.01 percentage point 时选更小树；
6. 执行 non-inferiority gate；
7. 未通过则返回 \((0,0)\)。

## 18.7 Non-inferiority gate

对每个 validation scale：

\[
\operatorname{UCB}_{95\%}
\left(\overline\delta_s\right)
\le
0.1.
\]

任一 scale 不满足，则该 GP run 的 deployed champion 为 baseline。

---

# 19. 张量与接口设计

## 19.1 `ProblemBatch`

```text
ProblemBatch:
    coords:          float64 [B,n,2]
    distances:       float64 [B,n,n]
    heuristic:       float64 [B,n,n]
    nn_indices:      int64   [B,n,K]
    full_nn_rank:    int64   [B,n,n]
    reference_tour: int64   [B,n+1]
    reference_length: float64 [B]
    instance_ids:    str[B]
```

只把同一 \(n\) 的实例组成 batch。

## 19.2 `TransitionContext`

```text
TransitionContext:
    current_city:    int64   [B,M]
    candidates:      int64   [B,M,K_active]
    feasible_mask:   bool    [B,M,K_active]
    base_score:      float64 [B,M,K_active]
    base_prob:       float64 [B,M,K_active]
    terminals:       dict[str, float64[B,M,K_active]]
```

fallback ants 单独形成 ragged/padded micro-batch。

## 19.3 `DepositEventBatch`

```text
DepositEventBatch:
    edge_u:          int64   [B,R,n]
    edge_v:          int64   [B,R,n]
    edge_id:         int64   [B,R,n]
    source_length:   float64 [B,R]
    base_deposit:    float64 [B,R,n]
    base_budget:     float64 [B,R]
    terminals:       dict[str, float64[B,R,n]]
```

## 19.4 `RunResult`

```text
RunResult:
    best_tour:       int64   [B,n+1]
    best_length:     float64 [B]
    best_iteration:  int64   [B]
    anytime_best:    float64 [B,T]
    wall_time_sec:   float64
    constructed_tours: int64
    diagnostics:
        uniform_fallback_count
        nan_sanitized_count
        bound_clip_count
        candidate_fallback_count
```

---

# 20. 向量化与加速

## 20.1 技术职责

| 技术 | 职责 |
|---|---|
| DEAP | Strongly Typed GP、选择、遗传算子、Hall of Fame |
| NumPy | 数据索引、memmap、SeedSequence、静态缓存 |
| Numba | 解析、distance、rank、tour length、2-opt/3-opt 热点 |
| PyTorch | ACO 动态状态、candidate scoring、tree tensor execution、scatter updates |

## 20.2 Tree compilation

每棵树编译为 postfix opcode：

```text
RTau REta MUL BaseConf ADD ERC MAX
```

运行时使用预分配 stack：

1. terminal opcode：压入 context tensor view；
2. constant opcode：压入 broadcast scalar；
3. unary opcode：原位或复用 buffer；
4. binary opcode：取两个 operands 并写入复用 buffer；
5. root tensor 作为 raw tree output。

禁止在候选城市循环内调用 Python function。

## 20.3 Transition tensorization

普通 candidate-list 分支：

\[
X_{\mathrm{tr}}
\in
\mathbb R^{B\times M\times K\times D_{\mathrm{tr}}}.
\]

tree output：

\[
U_{\mathrm{tr}}
\in
\mathbb R^{B\times M\times K}.
\]

roulette 使用向量化 inverse-CDF，而不是改变随机变量映射的 Gumbel-max。

## 20.4 Pheromone tensorization

\[
X_{\mathrm{ph}}
\in
\mathbb R^{B\times R\times n\times D_{\mathrm{ph}}}.
\]

AS：

\[
R=M.
\]

ACS/MMAS：

\[
R=1.
\]

无向 edge ID：

\[
id(u,v)
=
\min(u,v)n+\max(u,v).
\]

用 `scatter_add` 聚合重复 edges，再同时更新 \((u,v)\) 和 \((v,u)\)。

## 20.5 Population 并行

### CPU

- 个体或 genotype micro-batch 作为进程级任务；
- 每个 worker 内 `torch.set_num_threads(1)`；
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；
- Numba parallelism 与进程数二选一，禁止嵌套 oversubscription；
- static arrays 使用只读 memory map 或共享内存。

### GPU

- 单主进程管理 device；
- 按 \(n\)、ACO variant、tree length 分桶；
- 多个 GP individuals 组成 micro-batch；
- 根据显存动态选择 micro-batch size；
- 使用 `torch.inference_mode()`；
- 不保存 autograd graph。

## 20.6 缓存

缓存键至少包括：

```text
instance_hash
aco_variant
aco_config_hash
seed
budget
backend
precision
```

可缓存：

- baseline RunResult；
- distance/heuristic/NN ranks；
- static terminals；
- 同 generation 内重复 genotype fitness；
- compiled opcode。

不得跨不同 mini-batch 或 seed 复用 stochastic fitness。

## 20.7 Profiling 指标

- preprocessing time；
- candidate scoring time；
- tree evaluation time；
- tour construction time；
- pheromone update time；
- total wall-clock；
- peak RSS；
- peak GPU memory；
- constructed tours/s；
- candidate scores/s。

---

# 21. 核心伪代码

## 21.1 GP training

```text
Algorithm TrainRMTGP(variant, protocol, root_seed)
    读取并验证训练与验证 manifests
    构建 variant 对应的 ACOConfig
    初始化 typed transition/pheromone primitive sets
    初始化包含 baseline seeds 的 GP population
    archive <- {(0, 0)}

    for generation = 1 ... 50:
        batch <- 按 protocol 分层抽取训练实例
        random_stream <- 从层次 seed 构造 CRN
        baseline_result <- 读取或计算 baseline cache

        并行评估所有 invalid individuals:
            result <- EvaluateIndividual(
                individual, variant, batch, random_stream
            )
            fitness <- BaselineRelativeFitness(
                result, baseline_result, reference_length
            )

        保存 generation top-5
        elites <- 当前 top-10
        parents <- tournament selection
        offspring <- typed crossover / mutation / reproduction
        应用 depth/node static limits
        population <- elites + offspring
        更新 archive

    candidates <- 合并 checkpoint elites 并去重
    champion <- ValidationSelection(candidates)
    if champion 未通过 non-inferiority gate:
        champion <- (0, 0)
    保存 champion、配置、seed、Git commit 和训练统计
    return champion
```

## 21.2 Evaluate individual

```text
Algorithm EvaluateIndividual(I, variant, batch, random_stream)
    初始化 pheromone、ants、best states

    for t = 1 ... T:
        tours <- ConstructSolutions(
            T_tr = I.transition_tree,
            variant,
            pheromone,
            random_stream[t]
        )
        计算所有 tour lengths
        更新 iteration-best / restart-best / global-best
        events <- BuildDepositEvents(variant, tours, search_state)
        pheromone <- GlobalUpdate(
            T_ph = I.pheromone_tree,
            variant,
            pheromone,
            events
        )
        记录 anytime best

    return RunResult
```

## 21.3 Transition residual

```text
Algorithm ResidualTransition(context, tree, gamma)
    terminals <- BuildTransitionTerminals(context)
    raw <- EvaluatePostfix(tree, terminals)
    raw <- FiniteClip(raw, -10, 10)
    multiplier <- 1 + gamma * tanh(raw)
    score <- context.base_score * multiplier
    score <- ApplyFeasibleMask(score)

    if roulette branch:
        probability <- score / Sum(score)
        若 Sum(score) <= epsilon:
            probability <- feasible uniform
        return InverseCDFSample(probability, shared_uniform)
    else:
        return StableArgmax(score)
```

## 21.4 ACS synchronous construction step

```text
Algorithm ACSSynchronousStep(all_ants, pheromone_snapshot)
    对全部蚂蚁并行构建 TransitionContext
    对全部蚂蚁并行执行 q0 greedy / roulette
    chosen_edges <- 当前城市到下一城市
    counts <- CountUndirectedEdges(chosen_edges)

    for each unique edge e in parallel:
        tau[e] <- (1-xi)^counts[e] * tau[e]
                  + (1-(1-xi)^counts[e]) * tau0
    对称写回 tau
    return updated ants, tau
```

## 21.5 Pheromone residual

```text
Algorithm ResidualDeposit(events, tree, gamma)
    terminals <- BuildPheromoneTerminals(events)
    raw <- EvaluatePostfix(tree, terminals)
    raw <- FiniteClip(raw, -10, 10)
    multiplier <- 1 + gamma * tanh(raw)
    unnormalized <- events.base_deposit * multiplier
    normalizer <- SumOverEdges(unnormalized)
    deposit <- events.base_budget * unnormalized / normalizer
    Assert SumOverEdges(deposit) == events.base_budget
    return deposit
```

## 21.6 Variant-specific update

```text
Algorithm GlobalUpdate(variant, tau, events, T_ph)
    deposit <- ResidualDeposit(events, T_ph)

    if variant == AS:
        tau <- (1-rho) * tau
        tau <- tau + ScatterSumAllAntEvents(deposit)

    else if variant == ACS:
        对 global-best edges:
            tau[e] <- (1-rho) * tau[e] + rho * deposit[e]

    else if variant == MMAS:
        tau <- (1-rho) * tau
        tau <- tau + ScatterSelectedTour(deposit)
        tau <- Clip(tau, tau_min, tau_max)

    Symmetrize(tau)
    return tau
```

---

# 22. 实验设计

## 22.1 E0：数据与 baseline correctness

目的：

- 验证数据解析；
- 验证连续距离；
- 验证三个 ACO 的公式与参数；
- 验证 source-inspired 行为。

检查：

1. 所有 reference tours 合法；
2. reference length 可重复；
3. duplicate TSP100 copy 被排除；
4. candidate lists 排序正确；
5. AS deposit 等于所有 ant deposits 之和；
6. ACS local/global update 正确；
7. MMAS 调度和 bounds 正确。

## 22.2 E1：主组件实验

每个 ACO 变体分别比较：

| 方法 | Transition | Pheromone |
|---|---|---|
| ACO | baseline | baseline |
| Legacy-GP | full replacement | baseline |
| Matched-Replace-GP | matched full replacement | baseline |
| TR-RGP | residual | baseline |
| PH-RGP | baseline | residual |
| RMTGP-Core-F0 | residual | residual |
| RMTGP-Core-F1 | residual | residual |
| RMTGP-Full-F0 | residual | residual |
| RMTGP-Full-F1 | residual | residual |

前五个 trainable 对照回答 residual 与双树增益；四个 RMTGP 组合给出
Core/Full × F0/F1 的预注册 \(2\times2\) factorial contrasts。主方法是
RMTGP-Full-F1。

## 22.3 E2：规模外推

训练：

\[
TSP50+TSP100.
\]

测试：

\[
TSP500\text{-uniform}.
\]

比较：

- mean/median gap；
- paired improvement；
- worse-than-baseline rate；
- convergence；
- tree size；
- inference overhead。

## 22.4 E3：分布外泛化

不重新训练，直接把 E2 champions 测试于：

- TSP500-cluster；
- TSP500-Gaussian；
- TSPLIB \(n\le500\)。

## 22.5 E4：Mixed-scale 与 single-scale transfer

- 仅在 ACS 上执行，避免把次要数据协议扩成三变体的巨大笛卡尔积；
- mixed train：每代 TSP50 16 个 + TSP100 16 个；
- Train-50：每代 TSP50 32 个；
- Train-100：每代 TSP100 32 个；
- 三者每代均为 32 个 instance evaluations，并使用相同 population、
  generations 与 ACO 预算；
- 分别测试 TSP50、TSP100 与 TSP500。

形成：

\[
3\times3
\]

的 train protocol × test scale heatmap。

不设置 Train-500，因为它既破坏“从 50/100 外推到 500”的核心问题，又会
显著增加训练成本。

## 22.6 E5：Terminal ablation

确认性分析只使用 Core/Full × F0/F1 四组合。下列更细阶梯仅在该 factorial
结果显示 terminal 主效应后作为探索性 follow-up，不参与主假设的多重检验。

### Transition

| 名称 | Terminals |
|---|---|
| TR-Raw | raw \(\tau,d\), ERC |
| TR-Relative | RTau, REta, BaseConf, DistRank, ERC |
| TR-Context | TR-Relative + Entropy, ConstructProg, ACOProg, Stagnation |
| TR-Size | TR-Context + LogSize |

### Pheromone

| 名称 | Terminals |
|---|---|
| PH-Local | EdgeEta, EdgeTau, ERC |
| PH-Credit | PH-Local + NNRank, ColonyFreq, SourceQuality |
| PH-Context | PH-Credit + ACOProg, Stagnation |
| PH-Variant | PH-Context + 变体特有 terminal |

不执行任意完整笛卡尔积；按预注册阶梯逐步增加信息。

## 22.7 E6：Function 与 residual radius

Function sets：

- F0：\(+,-,\times,pdiv,neg\)；
- F1：F0 + \(\min,\max,abs\)。

Residual radius：

\[
\gamma\in\{0.1,1/3,0.5\}.
\]

F0/F1 属于确认性 factorial；\(\gamma\) 扫描属于后续敏感性分析，不与主
78-run pilot 同时展开。

## 22.8 E7：Budget preservation

比较：

1. per-source budget-preserving residual；
2. unnormalized multiplicative deposit；
3. additive deposit；
4. full replacement pheromone rule。

观察：

- 信息素爆炸/消失；
- MMAS clipping rate；
- validation–test degradation；
- OOD worse rate。

## 22.9 E8：结构 × 节点容量敏感性

主实验的 31 节点约束用于容量匹配，但仍需检验它是否过度限制双树。固定
其余协议，定义两级总预算：

\[
B\in\{31,62\}.
\]

对三个结构分别重新训练：

| 结构 | \(B=31\) | \(B=62\) |
|---|---:|---:|
| TR 单树 | 活动树 \(\le31\) | 活动树 \(\le62\) |
| PH 单树 | 活动树 \(\le31\) | 活动树 \(\le62\) |
| Full-F1 双树 | 两树合计 \(\le31\) | 每树 \(\le31\)，合计 \(\le62\) |

因此新增 62 节点训练量为

\[
3\ \mathrm{ACO variants}
\times3\ \mathrm{structures}
\times3\ \mathrm{GP seeds}
=27\ \mathrm{runs}.
\]

31 节点结果从冻结主消融只读复用。所有条件共享逐代 instance schedule、
baseline archive、GP root seed、ACO test seed 与数据划分。

记结构 \(a\in\{\mathrm{TR},\mathrm{PH},\mathrm{MT}\}\) 在预算 \(B\) 下的
reference gap 为 \(g_{a,B}\)。预注册三个 estimand：

1. 同结构的容量效应

   \[
   C_a=g_{a,62}-g_{a,31};
   \]

2. 固定容量下的双树效应

   \[
   A_{\mathrm{TR},B}=g_{\mathrm{MT},B}-g_{\mathrm{TR},B},
   \qquad
   A_{\mathrm{PH},B}=g_{\mathrm{MT},B}-g_{\mathrm{PH},B};
   \]

3. 结构与容量的 difference-in-differences

   \[
   I_{\mathrm{TR}}
   =
   (g_{\mathrm{MT},62}-g_{\mathrm{TR},62})
   -
   (g_{\mathrm{MT},31}-g_{\mathrm{TR},31}),
   \]

   \[
   I_{\mathrm{PH}}
   =
   (g_{\mathrm{MT},62}-g_{\mathrm{PH},62})
   -
   (g_{\mathrm{MT},31}-g_{\mathrm{PH},31}).
   \]

所有定义中负值表示公式前侧具有更低 gap。若 \(C_{\mathrm{MT}}\) 接近零，
则 31 节点并非双树的主要瓶颈；若双树在两级容量下均优于 TR 与 PH，
且 \(I_{\mathrm{TR}},I_{\mathrm{PH}}\) 接近零，则证据更符合角色分解本身；
若优势只在 \(B=62\) 出现，则主实验可能受到容量约束。

同时统计实际 transition/pheromone/total nodes、预算利用率、撞上上限的
run 数、单代时间和同条件孤立推理开销。正式合同见
`experiments/tsp100_capacity_sensitivity_3seed/study.yaml`。

## 22.10 E9：Coadaptation

### Shuffled pairing

从独立 runs 获得：

\[
\{(T_{\mathrm{tr}}^i,T_{\mathrm{ph}}^i)\}_{i=1}^{R}.
\]

随机排列：

\[
(T_{\mathrm{tr}}^i,T_{\mathrm{ph}}^{\pi(i)}).
\]

### Compatibility matrix

\[
M_{ij}
=
\operatorname{Gap}
(T_{\mathrm{tr}}^i,T_{\mathrm{ph}}^j).
\]

## 22.11 E10：Local-search robustness

该实验检验局部搜索是否只在测试时提供独立收益，以及 GP 是否能在训练时
适应局部搜索后的搜索动力学。正式实现采用以下四个条件：

1. 原始 ACO + 2-opt；
2. 原始 ACO + 3-opt；
3. 不含局部搜索训练的 RMTGP-ACO，测试时接 2-opt；
4. 含 2-opt 联合训练的 RMTGP-ACO，测试时接 2-opt。

后两个条件都在相同的 ACO+2-opt validation 环境中从各 run 的 checkpoints
重新选择最终个体。因此，比较不会混入不同模型选择环境的影响。每个条件
分别对 AS、ACS 和 MMAS 运行三个 GP seeds。局部搜索训练仅使用纯 TSP100。
每代使用 64 个实例。GP 种群为 100，进化 50 代。ACO 固定为 32 只蚂蚁和
500 次迭代。

为允许两棵树在局部搜索压缩解空间后表示更细的条件规则，本实验把初始深度
设为 2--5，最大深度设为 7。两棵树共享 62 个节点的总预算。任一单树也不能
超过 62 个节点。这里增加的是总容量，不把每棵树分别赋予 62 个节点。

局部搜索参考 ACOTSP-1.03。它对每轮构造出的全部蚂蚁路线执行搜索。2-opt
使用 20-nearest-neighbour candidate list、don't-look bits、随机城市扫描
和 first improvement。3-opt 先达到相同的 2-opt 局部最优，再检查候选受限
的真 3-opt 重连。本文使用连续欧氏距离和 counter-based RNG。因此，语义是
ACOTSP-style，而不是原 C 程序的逐位复现。

局部搜索条件增加一个可选的信息素终端
\(\mathrm{LSGain}\)。设来源路线在局部搜索前后的长度分别为
\(L^{\mathrm{pre}}_r\) 和 \(L^{\mathrm{post}}_r\)，则

\[
g^{\mathrm{LS}}_r
=
2\,\operatorname{clip}
\left(
\frac{L^{\mathrm{pre}}_r-L^{\mathrm{post}}_r}
{\max(L^{\mathrm{pre}}_r,\epsilon)},
0,1
\right)-1.
\]

它是强类型 \(\texttt{PhField}\)。一个来源路线先得到一个 scalar
\(g^{\mathrm{LS}}_r\)，再沿该路线的 \(n\) 条增强边广播。因此，其逻辑
shape 为 \([B,R,n]\)。输入仅包括当前构造路线、局部搜索前长度和局部搜索
后长度。它不读取最优标签。AS 的 \(R=M\)，因为全部蚂蚁都增强信息素。
ACS 和 MMAS 的 \(R=1\)，因为每轮只选择一个增强来源。全局最优或
restart-best 路线在被发现时同时保存其 LSGain。关闭局部搜索时该终端恒为
\(-1\)。这保持两个训练条件的 grammar 一致。

最终测试只使用锁定后的最终个体，不测试整个 population。TSP100 使用前
128 个独立测试实例。TSP500 使用前 32 个独立测试实例。每个实例使用三个
共同 ACO seeds。所有方法均运行 5000 次 ACO 迭代。主要指标是相对最优标签
的 gap%。RMTGP+2-opt 与 ACO+3-opt 的直接差异定义为

\[
\Delta_{3\mathrm{opt}}
=
\operatorname{Gap}(\mathrm{RMTGP+2opt})
-
\operatorname{Gap}(\mathrm{ACO+3opt}).
\]

对该差异执行 GP run、instance 和 ACO seed 三层 bootstrap。若 95% 区间
上界小于 0，则记为优于 ACO+3-opt；若上界不超过预先固定的 0.10
percentage-point margin，则只记为非劣。在线时间用单个 program 独立运行
测量。CUDA 编译时间单独报告，不计入在线求解时间。

GPU 实现把 individual×instance 平铺为任务。每条 2-opt 路线由一个 warp
处理，一个 block 同时处理 8 条路线。每条 3-opt 路线由一个 512-thread
block 处理，并行检查 \(20^2\) 个候选对。候选对用最小 pair index 归约，
从而保持确定性的 first-improvement 顺序。RTX PRO 5000 Blackwell 的短
基准中，8 warps/block 比 4 warps/block 快约 4.5%。3-opt 的 block 化相对
旧的一 warp/tour 实现，在 TSP100 和 TSP500 小工作负载上分别把局部搜索
内核时间降低约 1.9 倍和 3.5 倍。正式结论仍以完整运行 artifact 为准。

## 22.12 E11：ACS 语义审计

比较：

- ACOTSP sequential local update；
- 本研究 synchronous step update。

在相同 initial cities 和 random streams 下报告：

- 首次 tour divergence step；
- final gap difference；
- convergence difference；
- runtime speedup；
- 结论排序是否改变。

## 22.13 E12：效率

比较：

- candidate-level Python scalar；
- NumPy/Numba；
- PyTorch CPU；
- PyTorch GPU；
- 不同 process/thread 配置。

---

# 23. Test protocol

## 23.1 独立训练

正式结果：

- 每个 trainable method 30 个独立 GP runs；
- 不同 GP runs 使用不同 GP/random streams；
- 每个 run 单独选择 champion；
- 不以 30 runs 中最好的一个代表方法。

## 23.2 测试预算平衡

为使不同规模总 ACO runs 接近：

| Test set | Instances | Seeds per champion | Runs per champion |
|---|---:|---:|---:|
| TSP50 | 1,280 | 3 | 3,840 |
| TSP100 | 1,280 | 3 | 3,840 |
| 每个 TSP500 分布 | 128 | 30 | 3,840 |
| TSPLIB \(n\le500\) | 42 | 30 | 1,260 |

baseline 使用完全相同 seeds。

## 23.3 统计单位

ACO seeds 不是独立问题实例。推荐层次：

1. 每个 GP champion、每个 instance 先聚合 ACO seeds；
2. 保留 30 个 GP runs 的模型学习方差；
3. 以 test instance 为 paired block；
4. 同时使用层次 bootstrap 重采样 GP run、instance 和 seed。

---

# 24. 评价指标与统计

## 24.1 质量指标

- mean gap%；
- median gap%；
- standard deviation / IQR；
- paired improvement percentage points；
- win/tie/loss；
- worse-than-baseline instance rate；
- worst-scale mean gap；
- worst 10% CVaR；
- reference hit rate；
- anytime area under curve。

## 24.2 效率指标

- wall-clock；
- constructed tours/s；
- candidate scores/s；
- inference overhead；
- peak memory；
- training core-hours / GPU-hours；
- tree nodes 与 runtime 相关性。

## 24.3 统计检验

多方法比较：

1. Friedman omnibus test；
2. Holm-corrected paired Wilcoxon signed-rank；
3. paired rank-biserial effect size；
4. hierarchical bootstrap 95% confidence interval。

主比较预注册为：

\[
RMTGP
\quad\text{vs}\quad
ACO,\ TR\text{-}RGP,\ PH\text{-}RGP,\ Legacy\text{-}GP.
\]

每个 ACO variant 分别分析，不把三个变体的 raw gaps 当成可交换样本。

---

# 25. 正确性测试与验收条件

## 25.1 数据测试

- 格式解析；
- one-based 到 zero-based；
- tour permutation；
- 闭合边；
- reference length；
- duplicate file；
- split hash；
- scale allowlist。

## 25.2 ACO 单元测试

### AS

- 蒸发覆盖全部边；
- 每条 ant tour 的每条边增加 \(1/L_a\)；
- 无向对称；
- \(M=32\)。

### ACS

- \(q_0\) greedy/roulette 分支；
- 同步 local update 闭式公式；
- closing edge local update；
- global-best-only update；
- 非 global-best edge 不执行 global update。

### MMAS

- initialization bounds；
- dynamic bounds；
- iteration/restart-best schedule；
- clipping；
- 500 iterations 下 restart 检测与状态重置。

## 25.3 Residual 性质测试

对随机生成 tensor 做 property tests：

1. probability 非负；
2. feasible probability 和为 1；
3. mask support 不改变；
4. \(\gamma<1\) 时 multiplier 正；
5. probability ratio 满足理论界；
6. 每个 deposit event 预算守恒；
7. \((0,0)\) 与 baseline 在容差内一致；
8. 所有输出 finite。

正式 float64 容差：

\[
\operatorname{atol}=10^{-12},
\qquad
\operatorname{rtol}=10^{-10}.
\]

## 25.4 Strong typing 测试

- transition/phero subtree 不能交叉；
- crossover 后 return type 正确；
- mutation 后 tree 可编译；
- max depth/nodes 生效；
- ERC 类型与广播正确；
- opcode interpreter 与递归 oracle 一致。

## 25.5 可重复性测试

同一：

```text
Git commit
config hash
data manifest
root seed
backend
precision
```

必须产生相同结果。

---

# 26. Artifact 与版本记录

每个 run 保存：

```text
runs/<experiment_id>/<run_id>/
    config.yaml
    manifest.json
    seeds.json
    environment.json
    training_metrics.jsonl
    checkpoints/
    champion.pkl
    champion_expression.txt
    validation_summary.csv
    test_summary.csv
```

`manifest.json` 至少包含：

- Git commit SHA；
- project version；
- data file hashes；
- ACO/GP config hashes；
- Python/NumPy/Numba/PyTorch/DEAP versions；
- CPU/GPU 信息；
- thread/process 配置；
- start/end timestamps；
- run status。

大型 artifacts 不提交 Git；论文汇总表、绘图脚本和生成汇总所需的小型配置可提交。

---

# 27. 风险与控制

## 27.1 统一 32 只蚂蚁与 500 iterations 导致训练昂贵

三种 ACO 均使用 \(M=32\)，每次 simulation 固定运行 500 iterations。
相对 v0.3 的 ACS（10 ants、100 iterations），单个
program--instance 的构造 tour 数提高 16 倍。

控制：

- 主协议只在 TSP50/100 训练；
- TSP500 首先作为外推测试；
- mixed-scale training 独立核算成本；
- candidate list 固定为 20；
- population/ants/candidates 向量化；
- baseline 与 static terminal 缓存。

## 27.2 Residual 太弱

表现：

- multiplier 接近 1；
- 与 baseline 无差异。

控制：

- \(\gamma\) 消融；
- 观察 raw output 与 saturation；
- 先扩展 context terminals，不先扩展危险 functions。

## 27.3 Residual 太强

表现：

- training 大幅改善但 OOD 退化；
- multiplier 长期触边；
- MMAS clipping 频繁。

控制：

- 减小 \(\gamma\)；
- 保留 budget normalization；
- non-inferiority gate；
- 在独立 holdout gate 报告 worse-than-baseline rate。

## 27.4 Pheromone tree 信号不足

控制：

- PH-only 实验；
- `ColonyFreq`、`SourceQuality`；
- per-edge multiplier 分布；
- shuffled-pair；
- budget-preservation 消融。

## 27.5 显式规模过拟合

控制：

- 主 terminals 无量纲；
- `LogSize` 仅消融；
- Protocol A 完全不查看 TSP500 validation；
- 分布外测试只执行一次正式分析。

## 27.6 同步 ACS 改变算法

控制：

- 名称和论文中明确标注；
- 保留 sequential oracle；
- E10 报告差异；
- 不把同步版称为 ACOTSP bitwise reproduction。

## 27.7 多线程 oversubscription

控制：

- 进程和 intra-op threads 联合配置；
- 每个 run 记录环境变量；
- 提供单进程 deterministic mode；
- profiling 后锁定正式配置。

---

# 28. 实施阶段

## Phase 0：版本与设计冻结

1. 初始化 Git；
2. 建立 `.gitignore`；
3. 提交本文档；
4. 标记 `design-v1.0`；
5. 文档定版后才开始代码。

## Phase 1：数据与 scalar oracle

1. 数据 parser 与 manifest；
2. 连续 distance/reference length；
3. candidate list；
4. AS/ACS/MMAS scalar oracle；
5. E0 correctness。

## Phase 2：向量化 ACO

1. PyTorch state tensors；
2. vectorized roulette；
3. AS/MMAS scatter update；
4. ACS synchronous local update；
5. scalar/vector regression。

## Phase 3：Strongly Typed GP

1. typed primitive sets；
2. two-tree individual；
3. opcode compiler/interpreter；
4. genetic operators；
5. zero-residual recovery。

## Phase 4：训练系统

1. mini-batch sampler；
2. CRN；
3. baseline cache；
4. parallel evaluation；
5. validation selection；
6. checkpoint/artifact。

## Phase 5：科学验证

1. TSP100 development experiment；
2. 三变体 core component experiment；
3. Protocol A；
4. 锁定 terminal/function/\(\gamma\)。

## Phase 6：正式实验

1. 30 GP runs；
2. TSP500 三分布；
3. TSPLIB；
4. ablations；
5. coadaptation；
6. local search；
7. efficiency。

---

# 29. 预注册清单

在查看正式 TSP500 test 结果前锁定：

- 三个 ACO 的全部参数；
- 100-iteration 与 10-second budgets；
- ACS synchronous semantics；
- candidate fallback；
- distance precision；
- train/validation/test manifests；
- root seeds；
- transition/phero terminals；
- function sets；
- \(\gamma\) 候选；
- GP 参数；
- fitness；
- checkpoint frequency；
- champion selection；
- non-inferiority threshold；
- 主比较和多重检验；
- test seed 分配；
- 图表与汇总指标；
- failure handling；
- 正式代码 Git tag。

---

# 30. 预期贡献

若实验支持假设，预期贡献为：

1. 一种统一适用于 AS、ACS、MMAS 的 baseline-recoverable 双 residual 表示；
2. 一种 support-preserving 的状态转移信用重分配方法；
3. 一种 per-source budget-preserving 的信息素边信用重分配方法；
4. 一个显式 Strongly Typed、可向量化的 Multi-Tree GP 定义；
5. 一项 TSP50/100 到 TSP500 的规模外推研究；
6. 一项 uniform 到 cluster/Gaussian/TSPLIB 的分布外泛化研究；
7. 一套区分组件增益、表示增益和 coadaptation 的实验协议；
8. 一套可复现、可审计的 NumPy–Numba–PyTorch–DEAP 实现框架。

---

# 31. 纯 TSP100 三种子 GPU pilot 执行附录

本节冻结 2026-07-25 启动的工程与科学 pilot。它不替代上文的 30-run
确认性设计，也不改变消融实验矩阵。

## 31.1 训练与 validation

AS、ACS、MMAS 分别执行三个独立 GP runs：

\[
\mathcal S_{\mathrm{AS}}=\{1001,1002,1003\},\quad
\mathcal S_{\mathrm{ACS}}=\{2001,2002,2003\},\quad
\mathcal S_{\mathrm{MMAS}}=\{3001,3002,3003\}.
\]

所有 run 仅使用 TSP100。每一代无放回选择 32 个不同训练 instances；GP
population 为 100、运行 50 代，ACO 为 32 ants、500 iterations。每个
run 因而在逐代训练中暴露 \(32\times50=1600\) 个 TSP100 instance
occurrences。Selection 与 holdout gate 各使用 32 个独立 TSP100
instances；screening 使用 1 个 ACO seed，完整 selection/gate 使用 3 个。

主 fitness 仍为相对 reference label 的绝对 gap：

\[
g_{i,s}=100\frac{L_{i,s}-L_i^\star}{L_i^\star},\qquad
F(x)=\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}
\frac{1}{|\mathcal S|}\sum_{s\in\mathcal S}g_{i,s}(x).
\]

相对原始 ACO 的
\(\Delta_{i,s}=g_{i,s}^{\mathrm{candidate}}-g_{i,s}^{\mathrm{baseline}}\)
只用于逐代解释、validation gate 和最终 paired 分析，不替代 fitness。
每代对固定 selection-screening 子集评估当代 train-best，写出 train 与
validation 的 candidate gap、baseline gap、\(\Delta\)、墙钟时间和 ETA。
该监控结果不参与 breeding，避免把 validation 反馈泄漏进进化搜索。

## 31.2 单 GPU0 执行与恢复

正式进程必须通过 `CUDA_VISIBLE_DEVICES=<physical-index>` 只暴露一张
RTX 4000 Ada；配置中的设备 0 始终表示进程内的逻辑 GPU0。因此恢复任务时
可迁移到另一张同型号空闲物理卡，而不改变配置 hash、随机流或 CUDA 数值
语义。训练的 `program population × instance batch` 在一个融合 CUDA 调用
中并行；九个独立 run 则串行执行，以避免同卡并发导致显存争用和不可解释的
代时波动。每代原子保存 checkpoint、曲线和 provenance。

队列顺序采用 replicate-major：

\[
(\mathrm{AS}_1,\mathrm{ACS}_1,\mathrm{MMAS}_1),
(\mathrm{AS}_2,\mathrm{ACS}_2,\mathrm{MMAS}_2),
(\mathrm{AS}_3,\mathrm{ACS}_3,\mathrm{MMAS}_3).
\]

每个 run 依次生成冻结 schedule、预计算不可变 baseline archive、训练。
任务完成由 artifact 内容而不是文件存在性判断；中断后从最近完整 generation
继续。整个队列使用文件锁、PID、原子 state JSON 和逐任务日志，重复启动会被
拒绝。

## 31.3 锁定后的 selected-champion 测试

每个 ACO 变体的三个 GP runs 各自产生一个 validation-selected candidate；
九个主方法 champions 在
\(\{\mathrm{TSP50},\mathrm{TSP100},\mathrm{TSP500},
\mathrm{TSP1000}\}\) uniform、TSP500 cluster、TSP500 Gaussian 和
TSPLIB \(n\le500\) 上测试。每个 instance 使用 3 个 ACO seeds。测试 root
seed 固定为 9001，并由

\[
s_{p,b,r}=H(9001,p,b,r,\texttt{TEST})
\]

生成，故与 GP root seed 解耦，同一 ACO 变体的三个 champions 使用完全
相同的随机流。原始 ACO baseline 按
`variant×partition×batch×ACO-seed` 只计算一次并缓存，随后供三个
champions 共享；candidate 使用原子 batch shards，从而支持精确续跑。

TSP1000 永远不参与训练、validation、候选选择或 gate，仅作为补充尺度外推。
它不能被表述为主协议的模型选择结果。

消融方法也遵守“一 run 一 champion”，但只在 TSP100-U 与 TSP500-U
执行。Residual campaign 包含对应条件的 selected champions，replacement
campaign 单独执行；campaign 中 program 维度绝不包含 GP population 的
100 个 individuals。TSP50-U、TSP1000-U 和三个 OOD 分区只测试主方法
Full-F1。

## 31.4 pilot 统计

每个 variant×partition 分别报告 selected champions、baseline ACO 与 gate
后 deployed 行为，至少包括 mean/median gap、standard deviation、四分位数、
win/tie/loss、worst-10% CVaR、reference-hit rate、anytime gap AUC、best
iteration、墙钟时间和 throughput。Wilcoxon 以 GP-run×instance 为 paired
block，先在 ACO seeds 内平均。消融 contrasts 仅在 TSP100-U 与 TSP500-U
组成的预注册 family 内作 Holm 校正；效应量为 paired rank-biserial。
置信区间使用 GP run、instance、ACO seed 三层 bootstrap，重复 10,000 次。
其余分区只给出 Full-F1 相对原始 ACO 的锁定测试汇总。

由于每个 ACO 变体仅有三个 GP runs，本阶段的 \(p\) 值与区间均为探索性描述。
Residual vs replacement、双树 vs 单树以及 terminal/function set 的因果
结论必须由上文预注册消融回答，不能从这九个主方法 runs 单独推断。

## 31.5 节点容量敏感性的测试边界

31/62 节点 × 单/双树容量实验同样只使用 TSP100-U 与 TSP500-U。每个
`variant×method×capacity×GP root seed` 只测试一个 validation-selected
champion；三个 roots 全部保留。该实验回答容量主效应、固定容量的结构效应
与结构 × 容量交互，不重复主方法的完整 OOD 最终测试。

---

## 32. Blackwell CUDA v2 执行合同

后续 GPU 训练使用 `cuda_tiled_v2`。一个 task 对应一个语义
program×instance。每个 construction block 使用 32 ants×8 candidate
lanes，即 256 个 threads。GP postfix programs 在每代生成为 CUDA 标量
表达式。单卡和双卡必须产生相同的 tour、best iteration 和诊断计数。

RTX PRO 5000 Blackwell 的已门控 profile 为 raw CUDA、FP32-fast、
8 candidate lanes、无 register cap、instance-major 和生成式 GP。搜索结束
后，fitness 仍由 CPU 使用原始 FP64 distance matrix 计算。最终 profile 已在
TSP50/TSP100 各 128 个 instances、3 个 ACO seeds 上通过 0.10 pp 的单侧
非劣质量门。FP16、BF16、FP8、NVFP4 和 cuTile 不进入正式 solver。原因不是
硬件不支持，而是它们没有同时提供完整 solver 加速和质量保证。

硬件绑定参数见
`configs/cuda_tuning/rtx_pro5000_blackwell_sm120_v1.json`。完整的结构扫描、
精度门、双卡扩展和三算法 3 代短跑见
`docs/performance/cuda_v2_pro5000_blackwell_20260730.md`。该执行合同只改变
工程实现。它不改变 ACO 参数、GP fitness、数据 schedule 或实验统计单位。
baseline archive 的 kernel semantic ID 同时编码 CUDA provider、搜索精度
和 candidate lanes。任何可能改变归约或选择轨迹的 profile 变化都必须重新
预计算 baseline。

---

# 参考文献

[1] M. Dorigo, V. Maniezzo, and A. Colorni, “Ant system: Optimization by a colony of cooperating agents,” *IEEE Transactions on Systems, Man, and Cybernetics, Part B*, vol. 26, no. 1, pp. 29–41, 1996.

[2] M. Dorigo and L. M. Gambardella, “Ant colony system: A cooperative learning approach to the traveling salesman problem,” *IEEE Transactions on Evolutionary Computation*, vol. 1, no. 1, pp. 53–66, 1997.

[3] T. Stützle and H. H. Hoos, “MAX–MIN Ant System,” *Future Generation Computer Systems*, vol. 16, no. 8, pp. 889–914, 2000.

[4] M. Dorigo and T. Stützle, *Ant Colony Optimization*. MIT Press, 2004.

[5] T. Stützle, “ACOTSP, Version 1.0,” 2004. 本研究参考本地 `references/ACOTSP-1.03`。

[6] B.-C. Lin, Y. Mei, and M. Zhang, “Automated design of state transition rules in ant colony optimization by genetic programming: a comprehensive investigation,” *Memetic Computing*, vol. 17, article 2, 2025.

[7] D. J. Montana, “Strongly typed genetic programming,” *Evolutionary Computation*, vol. 3, no. 2, pp. 199–230, 1995.

[8] J. R. Koza, *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press, 1992.

[9] F.-A. Fortin, F.-M. De Rainville, M.-A. Gardner, M. Parizeau, and C. Gagné, “DEAP: Evolutionary algorithms made easy,” *Journal of Machine Learning Research*, vol. 13, pp. 2171–2175, 2012.

[10] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in *NeurIPS*, 2019.

[11] C. R. Harris et al., “Array programming with NumPy,” *Nature*, vol. 585, pp. 357–362, 2020.

[12] S. K. Lam, A. Pitrou, and S. Seibert, “Numba: A LLVM-based Python JIT compiler,” in *LLVM Compiler Infrastructure in HPC*, 2015.

[13] J. Tavares and F. B. Pereira, “Designing pheromone update strategies with strongly typed genetic programming,” in *EuroGP*, 2011.
