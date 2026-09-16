# MMAS + local search 下 RMTGP 增益不明显：GPU 受控消融实验计划

版本：v1.0  
日期：2026-09-16  
执行环境：研究者自己的服务器，复用 RMTGP-ACO 原始 CUDA 后端、原始模型和数据。  
执行状态：**实验计划；本文不包含本轮服务器实验结果。**

## 0. 研究目标与结论边界

本实验只回答一个问题：

> RMTGP-ACO + local search 在 AS 上有收益、在 MMAS 上收益不明显，究竟是 MMAS 的哪一项设计造成的？是重启、信息素边界、强化来源，还是这些设计之间的交互？

**不预先认定历史精英强化是主因。** 前期本地诊断只作为提出假设的线索，不作为服务器实验的确认性证据，也不用于设定预期效应大小。最终结论由原始 GPU 算法上的配对干预决定。

需要分别回答三个层次的问题，不能混用：

| 层次 | 要回答的问题 | 需要的实验 |
|---|---|---|
| 已有模型的行为归因 | 为什么当前三个已学冠军在原始 MMAS+LS 上增益小？ | 冻结 checkpoint，对 MMAS 组件做配对消融 |
| 学习方法的归因 | 这种现象是否只与当前冠军有关，还是会在独立训练后重复出现？ | 各消融环境下重新训练，使用独立 GP seeds |
| 算法改进 | 修改某个机制后，是否真的超过原始完整 MMAS+LS？ | 比较绝对质量、相同计算预算和相同时间预算 |

**“删除组件后 GP 相对优势扩大”不等于“修改后的算法更好”。** 可能只是无 GP 基线退化更多。所有报告必须同时给出两者的绝对变化。

---

## 1. 固定实现和实验对象

### 1.1 代码版本

编写计划时读取到的仓库 HEAD 为：

```text
repository: LinHuanli/RMTGP-ACO
inspected_head: 2b847698225e1d0fff36a25c1015a080350a7f00
previous_analysis_commit: 128c789b508705274ef27fa2d887f395a947aa77
```

已核对的 CUDA v2 更新文件在两个版本中的 blob 均为：

```text
src/rmtgp_aco/cuda/aco_tiled_v2.cu
blob: 3c35da1568def204f1ea27471bcfffeaadd15435
```

这不代表所有依赖文件均未变化。正式实验以服务器上的**历史运行 manifest 所记的完整提交、环境及 kernel 配置**为复现基准；新增消融在单独分支实现。保存基准提交、消融提交、完整 diff、工作区状态与生成 kernel 哈希。[S1–S3]

不得用另写的简化 ACO 替代正式 CUDA 求解器，不得仅从论文表达式重新构造模型作为主实验模型。

### 1.2 原始配置

从每个正式 run 的实际配置和 manifest 读取，不只读取公共模板。模板的默认 variant 是 ACS；campaign 会覆盖 MMAS 的蒸发率等字段。[S4][S5]

| 项目 | 本研究主实验设置 |
|---|---|
| 问题 | TSP500，连续二维 Euclidean 距离 |
| 变体 | MMAS；AS 为现象复现与反向验证对照 |
| 蚂蚁数量 | 32 |
| 构造候选表 / LS 候选表 | 20 / 20 |
| `alpha`, `beta` | 1.0, 2.0 |
| MMAS `rho` | 0.2，表示蒸发比例 |
| AS `rho` | 0.5，先保持原始设置，不偷偷与 MMAS 对齐 |
| 局部搜索 | 原始 `two_opt`、ACOTSP profile、DLB；每轮全部蚂蚁 |
| `gamma_transition`, `gamma_pheromone` | 1/3, 1/3 |
| PH 集成 | `budget_residual`，每个来源内部守恒预算 |
| 正式评估 horizon | 5000 iterations |
| 搜索后端 | 原始 `cuda_tiled_v2` 与对应生成式 GP 路径 |
| 数值设置 | 先复现原运行设置，再做独立精度审计 |

表中值必须与服务器原运行配置核对；不一致时以原运行配置为准，并记录差异。不要把早期 TSP100 的 `gamma=1/6` 混进本轮 TSP500 实验。[S4][S5]

### 1.3 模型对象

主分析使用 MMAS 的三个正式 `selected_candidate.pkl`，对应 GP seeds 81001、81002、81003；AS 使用对应三个正式冠军。核对路径、文件 SHA-256、结构哈希、两棵树、训练提交、validation 和 deployment 状态。[S6]

**主分析使用原始入选候选，不用 gate 后退回 baseline 的部署策略替代它。** 部署策略另表报告。不能把多个 checkpoint 挑到同一测试集上重新选择，也不能把整个 population 当成独立 GP runs。

---

## 2. 必须先核实的实现语义

以下是代码核对得到的实验设计依据，不是性能归因结论。

### 2.1 将“restart”与“restart-best 强化”分开

`restart` 是一次状态重初始化操作；`restart-best` 是从最近一次重启起保存的最好路径。关闭重初始化，不等于关闭历史路径强化。[S2][S3]

实际重启会涉及：信息素重置、restart-best 有效性、重启时钟和停滞计数相关状态。重启时钟又会影响后续强化来源。因此：

> 普通的 restart on/off 是“整套重启策略”的总效应，不能自动解释为“清空信息素”这一单独动作的效应。

本计划先测整套策略，再用固定事件时刻实验分解状态重置与时钟变化。

### 2.2 LS 分支没有硬上界裁剪

当前 CUDA v2 的 LS 分支执行候选弧蒸发时的下界保护，然后加入 deposit；完整的上下界裁剪位于 `local_search_active == 0` 分支。[S3]

因此要分开三个变量：

| 变量 | 当前 MMAS+LS 中的角色 | 实验处理 |
|---|---|---|
| 硬 `tau_max` clipping | 核对版本的 LS 路径未执行 | 做运行期计数核实；不是主消融开关 |
| 蒸发下界保护 | 对构造候选弧执行 | 纳入主全因子实验 |
| 标称 `tau_max` | 初始化、下界尺度、重置值和部分 terminal 的输入 | 保留；需要时独立做尺度实验 |

如果服务器代码实际执行上界裁剪，必须修改本协议：将上界单独纳入因子，而不是沿用“没有上界”的判断。**新增上界的实验只能评价新增规则，不能解释原本不存在的裁剪。**

### 2.3 历史来源日程必须按真实代码实现

当前 LS 分支按重启年龄使用 `25 → 5 → 3 → 2 → 1` 的强化周期；在相应条件下从 iteration-best 转向 restart-best 或 global-best。年龄计算含 `-1`，来源选择还使用迭代号取模。[S2]

不得用“第 250 轮后永远 global-best”替代真实规则。边界单元测试应覆盖重启年龄 25、75、125、250 的前后，以及 `iteration - restart_found_best = 50/51`。

**只修改配置里的 `mmas_update_period` 可能并没有改变 LS 分支的实际日程；`mmas_p_best` 也不是当前 LS 下界公式的控制参数。** 必须记录 kernel 实际使用的周期、下界和来源。

### 2.4 不得顺手改变其他语义

LS 蒸发作用于有向 construction candidate arcs，deposit 按路径边写入两个方向。不要在消融补丁中把矩阵强行对称化、改成全图蒸发，或改候选表耗尽时的选择规则。[S3]

`Origin`、逐边 `LSGain` 必须绑定到实际强化来源；历史路径使用该路径保存的元数据，当前路径使用本轮元数据。`SourceQuality` 使用实际来源长度，不能误用预算匹配的参考长度。`PostFreq` 等种群频率按原实现的当前种群语义计算。[S2][S7]

---

## 3. 假设及可证伪条件

| 编号 | 待检验因素 | 干预 | 什么结果才支持它 |
|---|---|---|---|
| H-R | 重启策略压缩 GP 收益 | 禁止原始重启；进一步固定重启事件时刻 | GP 净收益发生有实际意义的变化，且不是未触发开关 |
| H-F | 信息素下界保护压缩 GP 收益 | 只移除蒸发阶段的下界保护 | 配对净收益变化；并观察到下界机制实际作用 |
| H-H | 历史精英强化压缩 GP 收益 | 原来源日程改为 iteration-best，匹配预算 | 在仍为单路径强化时，GP 净收益发生变化 |
| H-K | 强化来源数量限制 GP 收益 | 同一预算下，iteration-best 改为 top-k 来源 | 在均不使用历史来源的条件下，来源数有独立影响 |
| H-I | 组件交互，而非单组件，造成现象 | R×F×H 全因子 | 某因素效应依赖其他因素状态，交互有实际意义 |
| H-T | 短 horizon 训练与长 horizon 评估失配 | 分离实际运行长度与时间 terminal 归一化；匹配训练预算 | 训练/评估错配被修正后现象减弱 |
| H-L | LS 消除构造差异或 PH 输出退化 | 相同状态下的 TR/PH 与 pre/post-LS 探针 | 直接观测差异在哪个环节消失，并用干预确认 |
| H-P | 蒸发率、初值或候选弧规则等其他因素 | 后续独立敏感性实验 | 主因子解释不足时，其他控制变量产生可重复效应 |

假设不按“预期谁会赢”排序。R、F、H 是第一批，理由是它们直接对应问题、成本可控，并能做完整交叉设计。经典 MMAS 文献区分了精英反馈、边界和初始化；这里只用它组织假设，不用文献替代本研究的因果证据。[S8]

---

## 4. 实验分阶段执行

| 阶段 | 内容 | 完成后能回答什么 |
|---|---|---|
| P0 | 代码审计、原结果复现、补丁与计数器校验 | 测的是否仍是原算法；各开关是否有效 |
| P1 | 独立开发集上的 R×F×H 全因子与轻量轨迹 | 估计方差、检查统计口径与计算成本 |
| P2 | 锁定后的正式 R×F×H 全因子 | 对当前冠军作推理期组件归因 |
| P3 | 来源细分、重启分解、TR/PH、同状态行为探针 | 说明效应通过哪个环节发生 |
| P4 | 独立数据、分布、长预算与 AS 反向验证 | 结果是否可重复，能否解释 AS/MMAS 差异 |
| P5 | 消融条件下重新训练及交叉部署 | 结论能否从当前冠军推广到学习方法 |

P0–P2 是最小完整的组件定位实验。论文层面解释“为什么”至少还需要 P3 和独立重复。对学习方法作广泛结论需要 P5。

P3 的扩展条件优先依据 P1 开发集选择，并在 P2 结果可见前冻结。若看到 P2 后才新增解释或对照，在 P2 数据上的分析只能标为探索性；确认它必须另建未使用的确认集。P5 的新模型另有最终测试集，不把已经参与方法判断的 P2 集重新称为未见测试集。

---

## 5. 数据、随机种子与规模

### 5.1 数据分区

建议建立下面的独立 manifest；这些是本计划的目标规模，不意味着仓库已提供足够的未使用实例。

| 数据集 | 建议规模 | 用途与限制 |
|---|---:|---|
| `reproduction_old` | 原论文 32 个实例/分布，原 ACO seeds | 只复现历史结果；已经看过，不作为新确认集 |
| `diagnosis_dev` | 32 个新 Uniform TSP500 | 工程调试、功效与预算估计、选择 P3 的扩展范围 |
| `confirm_uniform` | 默认 128；必要时预先扩为 256/512 | 主要确认集，开发结束前不得查看结果 |
| `confirm_cluster` | 128 | 固定协议后的分布外验证 |
| `confirm_gaussian` | 128 | 固定协议后的分布外验证 |
| `retrain_train/selection/gate` | 与上述集合不重叠，按训练预算冻结 | 仅用于 P5；不使用确认集调参或选 checkpoint |
| `retrain_confirm` | 默认 128 个新 Uniform 实例；另加锁定 OOD 集 | P5 的最终测试，不参与模型、条件或训练协议选择 |

仓库原最终脚本使用各分布前 32 个实例，但不能因此认定其余实例从未被使用；必须查训练、验证、测试和前期试验记录。[S6] 不足时使用新生成且提前冻结的实例，并准确记录生成分布。

保存坐标哈希、来源文件哈希、行号、生成器版本、生成 seed、距离定义、reference tour 和 reference length。哈希排重需要覆盖历史训练、validation、test 和前期诊断数据，不只检查文件名。

### 5.2 Reference 的要求

沿用原研究 reference tour，并以相同连续欧氏距离在 CPU FP64 重算其长度。新数据需要独立 reference；如果仅有可行参考解而无最优性证明，指标名称必须为 `reference_gap`，不能称为“最优差距”。

不得用本次被比较算法中的最好结果事后制造最优参考；不得把整数距离求解的最优性直接移植到连续距离问题。负 reference gap 应原样保留，不能裁为零。

没有可靠 reference 的数据可以报告配对路径长度差，但应使用独立命名，不与 gap 百分点混合。

### 5.3 随机流

正式评估默认每实例 **10 个 ACO seeds**；开发集用 5 个。较大 seeds 数用于降低运行噪声，不能替代独立实例和独立 GP 训练。

RNG 键由 `evaluation_seed + instance_identity + iteration + ant + step + stream_kind` 等冻结坐标决定。**算法条件、checkpoint、GPU 编号、batch 顺序不得改变共同随机流。** 新增的历史来源抽样与审计抽样使用独立 stream_kind。

同 seed 只保证随机输入配对。策略改变后，状态、可行候选集和路径自然会分叉，不要求两个算法生成相同路径。

### 5.4 功效与预算冻结

先在 `diagnosis_dev` 上计算实例级配对效应的标准差，再用预先指定的实际意义阈值估算确认集大小。不得因为开发集某因素“看起来效应大”就给它更少 seeds，或跑到显著才停。

默认实际意义阈值建议为 **ε = 0.01 reference-gap 百分点**；这是待研究团队在确认实验前签字冻结的研究标准，不是从历史诊断结果估计的常数。可根据研究用途调整，但不能在查看确认结果后改变。

粗略规划可用：

\[
N\approx\frac{(z_{1-\alpha_*/2}+z_{1-\beta})^2\sigma_d^2}{\epsilon^2},
\]

其中 `d` 是先在实例内平均种子与固定冠军后的配对效应，`α*` 考虑计划比较数。正式样本量还应用开发数据的重采样评估，而不是把公式当作保证。目标功效建议 0.8–0.9。

如果估算规模超过资源预算，预先限定样本量并允许结论为“精度不足”。不使用未校正的反复查看与追加停止规则。

---

## 6. P0：复现与实现验收

### 6.1 原始现象复现

使用原 checkpoint、原数据、原 seeds 和原后端，分别运行：

```text
AS+2-opt baseline / 三个 AS-RMTGP+2-opt
MMAS+2-opt baseline / 三个 MMAS-RMTGP+2-opt
```

对照原保存的逐实例结果，而不只对齐汇总均值。核对最终路径、CPU FP64 长度、gate/fallback 标识、horizon、实际 precision、搜索后端及运行配置。

如不能复现 AS 有收益、MMAS 增益小这一现象，先调查实现与运行差异，不继续给原现象作机制归因。跨 GPU/编译器不能要求必然逐位一致，但不能忽略系统性的质量偏移。

### 6.2 新补丁必须保持原始分支

给组件增加显式配置开关，默认配置等价于原始算法。先在同一硬件、同一编译环境验证 `native`：

| 检查 | 验收要求 |
|---|---|
| 未修改版 vs 修改版 native | 小样本中路径、best-so-far、来源、重启时刻一致；任何差异先定位 |
| instrumentation off vs on | 不改变 RNG、决策和路径；计时开销另外记录 |
| GP 两树都关闭 | 与同配置无 GP 基线一致 |
| 预算 | 每轮实际 deposit 总量与规定预算一致；采用预注册数值容差 |
| R=0 | 不执行任何原重启状态写入；仍可记录“本应触发”的事件 |
| F=0 | 物理下界保护次数为零；数值 fallback 单独统计 |
| H=0 | 实际强化来源只为 iteration-best；global-best 输出记忆仍保留 |
| 来源元数据 | tour、length、Origin、LSGain 对应同一实际来源 |
| 路径合法性 | 每条最终 tour 合法；2-opt 不增加同一距离定义下的长度 |
| 默认硬上界计数 | 核对 LS 分支是否为零；不能只看统一 `bound_clips` 计数 |

预算验收可用 `abs(actual/target-1) ≤ 1e-5` 作为 FP32 初始门槛、`1e-10` 作为 FP64 初始门槛；必须通过无结果导向的数值测试冻结，不得掩盖系统性质量误差。

### 6.3 数值精度审计

CUDA 搜索的 FP32 与 CPU 对返回 tour 的 FP64 重算不是同一件事。重算长度不能消除搜索阶段的舍入影响。

对原始条件和主要消融条件，在固定独立小集合上运行对应 CPU/FP64 原后端，或真实支持的 GPU 高精度路径。**不能在 YAML 中写一个未实现的 FP64 GPU 模式就称为完成审计。**

重点比较“组件效应”的差，而不只是两个后端各自均值。若精度引起的系统性差接近 ε，先解决数值不确定性。不同精度 RNG 或约简顺序导致的路径差异，应通过同状态算子测试与分布级比较区分，不强求整条随机轨迹完全相同。

---

## 7. P1/P2：R × F × H 主全因子实验

### 7.1 三个开关的严格定义

| 因子 | 1：保留 | 0：移除 | 必须保持不变的部分 |
|---|---|---|---|
| R：原始重启策略 | 原始触发、信息素重置、restart 状态与时钟更新 | 禁止这套重启写操作 | global-best 输出记忆、当前/历史最优正常维护、蒸发与 LS |
| F：蒸发下界保护 | 候选弧使用 `max((1-rho)*tau, tau_min)` | 候选弧仅使用 `(1-rho)*tau` | 初始信息素、标称上下界计算、TauHeadroom 公式、R/H 策略 |
| H：历史来源策略 | 原始 iteration/restart/global-best 日程 | 强化路径始终选本轮 iteration-best | 仍只强化一条路径；仍记录历史最好解；同状态预算函数匹配 |

R=0 后不再更新重启时钟，因此会改变 H=1 的日程经历；这属于该策略的闭环总效应。不能隐瞒这一点，P3 会进一步分解。

F=0 不能通过把 `tau_min` 参数一概改为零来实现，否则会连带改变初始化、terminal 或其他机制。只禁用物理保护操作，标称量仍正常计算。记录接近零、下溢与 uniform fallback；它们可能是移除下界的真实后果，但不能误称为“GP 被裁剪”。

### 7.2 完整的 8 个条件

| condition_id | R | F | H | 含义 |
|---|---:|---:|---:|---|
| C111 | 1 | 1 | 1 | 原始 MMAS+LS |
| C011 | 0 | 1 | 1 | 只移除重启 |
| C101 | 1 | 0 | 1 | 只移除下界保护 |
| C110 | 1 | 1 | 0 | 只移除历史强化 |
| C001 | 0 | 0 | 1 | 无重启、无下界保护 |
| C010 | 0 | 1 | 0 | 无重启、无历史强化 |
| C100 | 1 | 0 | 0 | 无下界保护、无历史强化 |
| C000 | 0 | 0 | 0 | 三项均移除 |

**每个条件都运行自己的无 GP baseline，加上同样的三个固定冠军。** 不允许只修改 GP 版本，或把消融 GP 与原始 C111 baseline 直接比较后称为净收益。

### 7.3 来源改变必须匹配预算

在每条运行自己的当前状态 `x_t` 中，先用原来源规则计算一个“影子原生来源” `T_ref(x_t)`，但不要求实际强化它。设其长度为 `L_refsrc`，定义本轮预算：

\[
B(x_t)=\frac{n}{L_{refsrc}(x_t)}.
\]

实际选中来源 `T_k` 后，其边权为：

\[
w_{k,e}=1+\gamma_P\tanh(f_P(z_{k,e})),\qquad
q_{k,e}=\frac{w_{k,e}}{\sum_{a\in T_k}w_{k,a}}.
\]

对单路径来源：

\[
D_e=B(x_t)q_{k,e}\mathbf 1[e\in T_k],\qquad \sum_e D_e=B(x_t).
\]

无 GP 时 `q=1/n`。原始 H=1 下上述规则与原预算一致；H=0 只换实际来源的路径与其元数据，不因来源长度变化连带改变总预算规则。

以上是数学上的预算约束，不要求重写 native 的浮点计算顺序。原始分支应继续使用现有 deposit 计算路径；新增匹配逻辑只应用于干预分支，并通过 native 等价验收。

预算定义在**无向来源边**上；写入两个方向的 dense matrix 总和为两倍，验收时不能混淆。

这是**同一状态下的预算匹配**。不同条件的轨迹会分叉，各自影子来源长度也会变化；不声称跨条件每轮实际预算数值完全相同。P3 的同状态干预用于进一步检查直接作用。

**H=0 禁止把历史 tour 的边集作为实际强化来源，不是删除算法中全部历史信息。** 标称上下界、最终输出记忆以及上述预算函数仍可依赖历史最好长度。报告应称为“历史路径来源消融”，而不是“完全无记忆 MMAS”。

另设 `H=iteration_best_native_budget` 作为次级对照，使用实际来源的 `n/L_ib`。如果它与预算匹配版结果不同，应把“来源”和“强化量”分开解释。主表不得把两者混成同一个消融。

---

## 8. 主要效应：比较 GP 净收益及其变化

### 8.1 统一指标

设实例 `i` 的独立 reference 长度为 `L_i^ref`，条件为 `c`，ACO seed 为 `s`，固定冠军为 `p`：

\[
g_{c,i,s,p}=100\left(\frac{L_{c,i,s,p}}{L_i^{ref}}-1\right).
\]

定义同条件 GP 净差：

\[
\Delta_{c,i,s,p}=g^{GP}_{c,i,s,p}-g^{base}_{c,i,s}.
\]

`Δ<0` 表示 GP 优于同配置 baseline，单位为 **reference-gap 百分点（pp）**。

在原始配置附近，三个单组件移除效应为：

\[
E_R=\overline\Delta_{011}-\overline\Delta_{111},\quad
E_F=\overline\Delta_{101}-\overline\Delta_{111},\quad
E_H=\overline\Delta_{110}-\overline\Delta_{111}.
\]

`E<0` 表示移除该组件后 GP 的相对收益扩大。这是主要归因量；不是只看某个版本自身有没有变差。

### 8.2 绝对变化分解

对每个干预同时报告：

\[
A_{base}(c)=\overline g^{base}_c-\overline g^{base}_{111},
\]
\[
A_{GP}(c)=\overline g^{GP}_c-\overline g^{GP}_{111}.
\]

于是：

\[
E(c)=A_{GP}(c)-A_{base}(c).
\]

| 结果模式 | 可以说什么 | 不可以说什么 |
|---|---|---|
| GP 和 baseline 都改善，GP 改善更多 | 干预提高质量，并扩大 GP 净收益 | 尚不能推断训练方法普遍有效 |
| baseline 退化很多，GP 基本不变 | 原组件给 baseline 的收益更多；相对增益被压缩 | “删除组件让 GP 求解更好” |
| 两者都退化，baseline 退化更多 | 相对差扩大，但绝对方案变差 | “消融后算法更优” |
| 两者变化接近 | 组件对两者类似；不是明显的 GP 交互因素 | “组件对算法没用” |
| Δ 改善但仍大于零 | 负面作用减轻 | “GP 优势已恢复” |

“GP 优势恢复”需要消融后的 `Δ` 的同时置信区间上界低于 `−ε`，而不只是 `E<0`。若只有 `E` 的区间上界低于 `−ε`，只能说“相对差有实际改善”。

### 8.3 交互不能省略

例如保留 R 时，历史来源在两种下界状态下的效应差为：

\[
J_{FH|R=1}
=(\overline\Delta_{100}-\overline\Delta_{101})
 -(\overline\Delta_{110}-\overline\Delta_{111}).
\]

同时计算 `R×H`、`R×F` 和三阶交互，报告各背景下的条件效应。为避免各脚本符号不一致，固定以下定义（所有量均为同一实例配对后再取均值）：

\[
J_{RF|H=1}=\overline\Delta_{001}-\overline\Delta_{011}-\overline\Delta_{101}+\overline\Delta_{111},
\]
\[
J_{RH|F=1}=\overline\Delta_{010}-\overline\Delta_{011}-\overline\Delta_{110}+\overline\Delta_{111},
\]
\[
J_{RFH}=\overline\Delta_{000}-\overline\Delta_{001}-\overline\Delta_{010}-\overline\Delta_{100}
+\overline\Delta_{011}+\overline\Delta_{101}+\overline\Delta_{110}-\overline\Delta_{111}.
\]

三阶项表示移除 H 后 `R×F` 交互的变化。直接大小比较固定为 `E_R-E_F`、`E_R-E_H`、`E_F-E_H`。完整 8 格表是必要结果，不只画三个主效应柱状图。

如果一个因素的作用在其他开关变化后反向，应给出条件化结论，例如“在保留下界时，历史来源策略抑制了现有 GP 的增量收益”，不能删去条件后写成普遍主因。

---

## 9. P3：把“信息素更新”进一步拆开

### 9.1 历史来源、具体历史类型与日程

如果 H 有效应，至少增加下面的对照。仍固定 R、F、预算规则和 GP：

| source_policy | 用途 |
|---|---|
| `native_schedule` | 原始对照 |
| `iteration_best` | 无历史来源，但仍单路径强化 |
| `restart_best_only` | 区分 restart-best 的持续强化 |
| `global_best_only` | 区分 global-best 的持续强化 |
| `native_slots_use_restart_best` | 保留原历史来源时隙，但历史时隙只使用 restart-best |
| `native_slots_use_global_best` | 保留原历史来源时隙，但历史时隙只使用 global-best |

后两项用于减少“来源类型”和“历史强化频率”的混淆。固定时隙需要明确：可以来自冻结的日历，或来自原始 baseline 的外部回放；如果仍由各条件自身重启状态决定，就属于闭环策略比较，不是相同时隙对照。

进一步做历史使用概率的剂量实验 `p_history ∈ {0, 0.25, 0.5, 0.75, 1}`。固定历史类型与外部时间定义，用独立 counter stream 决定是否采用历史来源，保持每轮仅一条来源、匹配预算。观察效应是否随剂量变化，而不是只比较两个极端。

必须同时记录来源**标签**与实际路径哈希。历史来源可能与 iteration-best 是同一条路径；只看到标签为 global-best，不说明实际强化了不同结构。

### 9.2 来源数量 K

固定不使用历史来源，将 actual source 改为当前 post-LS 种群的 `top-k`，取 `k ∈ {1, 4, 8, 32}`。路径按长度、再按 ant index 稳定排序。

保持总预算 `B(x_t)` 不变，按预先固定的规则分配来源权重，例如：

\[
a_k=\frac{1/L_k}{\sum_{j\in\mathcal K}1/L_j},\qquad
D_e=B(x_t)\sum_{k\in\mathcal K}a_k q_{k,e}\mathbf 1[e\in T_k].
\]

每个来源内部仍按自己的边权归一化，来源预算之和为 `B`。并列或重复路径不凭主观去重；保留原种群多重性，另报实际独特来源数。

这组实验回答“单来源支持集限制”是否有独立作用。`top-32 MMAS` 仍不是完整 AS，因为蒸发率、初值、下界与重启可能不同。

### 9.3 边界与尺度

F 的主消融只关物理下界。补充下面的实验时一次只改一项：

| 干预 | 保持不变 | 解释对象 |
|---|---|---|
| 物理 floor 倍率 `a∈{0,0.25,1,4}` | 标称 tau、初始化、原 terminal 公式 | 探索概率下限的强弱 |
| 只改变初始 τ 倍率 `a∈{0.25,1,4}` | 后续边界与重启值保持原值 | 初始化尺度 |
| 只改变 restart 写入的 τ 倍率 | 初始状态与日程保持原值 | 重置强度 |
| 在 LS 中新增硬上界 | 其他规则完全显式冻结 | 新算法的敏感性，不是原算法的“上界消融” |
| 单独冻结/替换 `TauHeadroom` 输入 | 不改真实信息素机制 | terminal 输入作用，不是物理边界作用 |

若某模型未使用 `TauHeadroom`，最后一项应为精确零干预。对于标称范围为无穷的变体，先核实原 terminal 的 finite/sanitize 语义，不在实验中私自修正。

---

## 10. P3：重启的分解与回放

### 10.1 为什么单纯 no-restart 不够

移除重启不仅保存了信息素，也让 restart-best 存活更久、改变重启年龄及历史日程。主全因子的 R 测的是这组操作的总效应。

为了区分这些路径，先在**无 GP 原始 baseline** 的固定运行中记录重启事件序列。将事件时刻冻结，作为外部输入，供配对算法使用。事件序列不能从“效果最好”的运行中挑选。

### 10.2 外部事件时刻下的 2×2 分解

在同一组预定事件时刻 `t_1,t_2,...`，执行：

| 条件 | 信息素重置 | restart-best 有效性与日程时钟重置 |
|---|---:|---:|
| Y00 | 否 | 否 |
| Y10 | 是 | 否 |
| Y01 | 否 | 是 |
| Y11 | 是 | 是 |

事件时刻之外不允许算法自发重启。global-best 最终输出记忆始终保留。`Y01/Y11` 中的元数据失效标记必须与原逻辑匹配，避免引用无效 restart-best。

这四个条件是人工受控的算法，不声称都对应标准 MMAS。它们用于分离信息素抹除与记忆/时钟重置的作用。再将 Y11 与 native adaptive restart 比较，评估**自适应事件时刻**这一部分。

如果只有 Y10 有效，支持信息素重置路径；只有 Y01 有效，支持记忆/时钟路径；二者共同才有效，则是交互。

### 10.3 固定来源日程的进一步控制

为减少时钟带来的混淆，可再回放原 baseline 的 source-kind 时隙序列，但路径仍取当前运行相应的 best tour。不要回放 baseline 的具体最优路径作为主实验来源，否则引入了另一条运行的解信息，研究问题已改变。

外部序列回放给出受控机制证据，但不是未经额外假设的“自然中介效应”。不能把 native 与回放的差自动分解成严格可加的因果百分比。

---

## 11. P3：TR 与 PH 的贡献

对 C111、主要候选消融条件及一个交互背景，运行每个固定冠军的四种模式：

| 模式 | TR | PH |
|---|---|---|
| `baseline` | 关闭 | 关闭 |
| `tr_only` | 原树 | 关闭 |
| `ph_only` | 关闭 | 原树 |
| `full` | 原树 | 原树 |

关闭通过使对应残差严格为零实现，不改变另一棵树、RNG 或搜索预算。结构或行为完全重复的模式可以缓存复用，但不能作为额外独立样本。

需要回答：组件效应主要作用于 TR、PH，还是两树协同？如果 PH-only 没有对应效应，而 full 有效，不能直接写成“PH 与 MMAS 功能重叠”。

这是训练后干预，只解释当前两树的行为依赖；“重新训练单树是否足够”必须由 P5 的训练对照回答。

---

## 12. P3：同状态分叉与行为探针

最终质量差给出干预效应；轨迹用于定位过程。避免只画相关性图就宣布机制成立。

### 12.1 保存共同起点，分叉运行

在预先固定的迭代点保存**完整 solver state**。同一状态分别继续运行 baseline/GP、原始组件/移除组件，使用相同未来 counter RNG 坐标。

起点同时来自无 GP 轨迹和 GP 轨迹，按固定数量抽取，不能只使用一方更有利的状态。快照应包含信息素、当前/历史路径与长度、时间与停滞状态、来源元数据、参数、program 哈希以及恢复所需工作区信息。

建议分叉继续长度为 `1、25、100、500` 轮，仍使用原定总 horizon 的时间 terminal 归一化，不把分叉长度当成 `ACOProg/Stagnation` 的新分母。

### 12.2 单步 PH 传递

在完全相同的 post-LS 状态、来源、预算与边界值下，分别计算零 PH 和真实 PH 的更新。记录：

\[
A_D=\frac{\|D_{GP}-D_0\|_1}{B},\qquad
A_\tau=\frac{\|\tau'_{GP}-\tau'_0\|_1}{\|\tau'_0\|_1+\varepsilon_{num}}.
\]

再对相同城市、相同 visited mask 的下一步候选分布计算 TV 距离、argmax 是否改变、历史路径边的概率质量变化。

把传递链分成：

```text
GP 原始输出 → tanh 后边权 → 预算归一化后的 deposit
→ 蒸发/边界/重置后的 pheromone → 候选概率 → LS 后路径
```

如原始树输出变化很大而归一化后 deposit 几乎不变，应调查饱和和常数抵消；如 deposit 明显改变但后续概率与 LS 终点变化很小，不能归因于“树没有输出”。

### 12.3 TR 与 LS 的差异存活

在同一初始状态上，用相同随机坐标分别构造 baseline 和 GP 路径，再执行相同 LS 规则和扫描随机流。计算 pre-LS 与 post-LS 无向边集差异，并报告 pre-LS 差为零的比例；不能对零分母定义存活率。

post-LS 的相同路径哈希只表示相同终点结构。因为候选表、DLB 和随机扫描次序存在，不能把“相同或不同终点”直接当作数学意义上同一个或不同局部搜索盆地的完整判定。

### 12.4 历史元数据与当前反馈的错位

记录历史来源的发现时间、距当前迭代的年龄、Origin/LSGain 生成时间，以及当前 PostFreq/SourceQuality。它们可能来自不同时刻。[S2][S7]

如果怀疑陈旧元数据导致问题，做独立 terminal 干预，并明确新定义；不要在来源消融时偷偷重算历史路径的 Origin/LSGain。原始历史来源没有保存对应构造路径时，不能凭空恢复其当前“来源”。

---

## 13. 需要记录的详细数据

### 13.1 全量运行记录：每次求解一行

建议 `run_metrics.parquet`，唯一键为：

```text
condition_id × model_id × model_mode × instance_hash × aco_seed × horizon
× protocol_hash × code/kernel_hash × precision_profile
```

baseline 的 `model_id` 为空，每个条件只保存一份；与三个冠军比较时引用同一份，不复制成三次独立运行。

| 字段组 | 必须包含的字段 |
|---|---|
| 复现标识 | dataset/split/distribution、instance/ref hashes、config/code/kernel/checkpoint hashes |
| 运行标识 | condition、R/F/H、source/budget mode、model/mode、ACO seed、GP train seed、H、terminal H |
| 质量 | final CPU-FP64 length、reference gap、best-found iteration、anytime AUC、指定时间点 best |
| 计算量 | constructed tours、LS calls/moves/checks/passes、实际迭代数 |
| 耗时 | warmup/compile/search/LS/update/audit/transfer/FP64-score 分项，支持的分项必须实测 |
| 机制 | IB/RB/GB 实际次数、重启次数、floor/upper/numeric 各自计数、fallback 次数 |
| 完整性 | status、failure reason、重试次数、是否复用缓存、设备/驱动/编译信息 |

### 13.2 轻量时间序列：所有运行

不要默认对所有运行保存每轮全部蚂蚁与完整信息素矩阵。建议在 GPU 上按 **25 轮窗口**聚合；best-so-far 用改进事件稀疏保存。

每个窗口至少记录：

| 过程 | 统计量 |
|---|---|
| 质量 | pre/post-LS best、mean、top-7 mean；窗口内改进次数与幅度 |
| 来源 | IB/RB/GB 占比、来源切换次数、连续相同来源长度、来源年龄分位数 |
| 结构 | 来源与 IB/GB 的边重叠、实际路径相同率、post-LS 独特路径数 |
| PH | raw output / tanh multiplier / 归一化边权的均值、标准差、分位数、饱和比例 |
| 信息素 | 候选弧 tau 分位数、tau/标称 tau_max、floor 触发率、超过标称上界的比例 |
| 搜索 | 候选 fallback、numeric fallback、构造概率熵的固定抽样统计 |
| 重启 | branch factor、stagnation、重启事件及事件前后统计 |

`ColonyFreq` 与 `PostFreq` 是否为同义实现需核对并记录，不能当成两个独立观测来源。原始输出方差也不能替代有效 deposit 方差。[S7]

### 13.3 重启事件表

每个事件一行，保存：

```text
iteration, executed/would_trigger, branch_factor, branch_cutoff,
restart_age, global_stagnation, restart_best_age,
best_length_before, tau_summary_before/after,
source_kind_before/after, restart_memory_changes,
budget_before, reset_scale, event_schedule_id
```

另外计算事件后 25/100/500 轮的改进量。事件发生率与质量的相关性仅作描述；固定事件回放才是干预证据。

最后一轮更新后的重启单独标记：它不影响后续构造，不能算成解释最终质量的有效重启。

### 13.4 重型审计子集

预先固定 8 个开发实例 × 3 个 ACO seeds × 全部 8 条件 × baseline/三个冠军。选择不依赖效果。

在该子集保存 post-LS tours、来源元数据、抽样候选状态和完整快照。基础快照点建议为 `t={1,100,250,500,1000,2500}`，另在少量预定事件边界保存前后状态。

TSP500 单个 FP32 信息素矩阵约 1 MB；这组 768 次运行保存 6 个矩阵时，仅信息素原始数组约 4.6 GB，尚不含其他状态。依据实际存储预算调整快照数，但须先冻结。

### 13.5 结构与熵的定义

对无向 TSP 路径，先消除旋转与反向等价再生成哈希，并抽查哈希匹配的实际边集。不要因路径起点不同把同一 tour 计为不同终点。

来源重复度、post-LS 独特路径数、边频率熵和候选选择熵分别定义、分别报告。多样性更高不自动意味着质量更好，熵更低也不自动证明存在有害停滞。

下界触发率分母使用实际被检查的有向候选弧数；来源边、全图边和构造中实际可行边的占比不能混用。

---

## 14. P4：horizon、分布与 AS 反向验证

### 14.1 不把 5000 轮曲线前缀当成独立 500 轮运行

现有时间 terminal 使用总迭代预算归一化，短预算运行和长预算前缀的输入未必相同。[S7]

必须区分：

| 实验 | 实际运行 | 时间 terminal 分母 | 用途 |
|---|---|---|---|
| 长预算前缀 | 一次 5000 轮 | 固定 5000 | 观察同一条轨迹中效应出现的时间 |
| 独立短预算 | 分别运行 100/200/500/1000/5000 | 各自的总预算 | 复现真实 budget shift |
| 终端归一化控制 | 固定实际运行长度 | 比较原分母与预定固定分母，明确裁剪规则 | 分离迭代长度与时间输入变化 |

在未使用时间 terminal 的模型上，归一化控制应为零干预。终端控制是新的策略定义，不能混入原配置复现。

### 14.2 分布外验证

在不重新选择模型、不调整因子取值和 ε 的前提下，运行 `confirm_cluster` 与 `confirm_gaussian`。完整版使用全部 8 格；资源有限时可只验证预先冻结的主要对照，但不能据此声称所有交互都跨分布成立。

各分布分别报告，跨分布汇总采用预先指定的等权均值，不能用不同实例数暗中改变权重。

### 14.3 AS 反向验证

要解释“为什么 AS 有收益、MMAS 没有”，只在 MMAS 内删组件还不够。对 AS 增加以下匹配预算的来源干预，保持 AS 原蒸发率、初值、无 MMAS floor/restart 的原语义：

| AS 条件 | 来源 | 本轮预算 |
|---|---|---|
| AS-native | 全部本轮 post-LS ants | 原 `sum_k n/L_k` |
| AS-IB | 本轮 iteration-best 单路径 | 匹配同状态 AS-native 总预算 |
| AS-history | 按明确日程的历史单路径 | 同上 |

`AS-IB → AS-history` 主要检验历史来源，`AS-native → AS-IB` 主要检验来源数量/选择。AS 没有原生 restart epoch，因此 AS-history 使用明确的全局日历与 global-best，或外部冻结时隙；不要声称它是未修改 MMAS 日程的直接复制。

只有反向加入某机制后 GP 净收益也被压缩，才更有依据解释变体差异。即使反向结果不成立，也如实报告，不能只保留 MMAS 一侧的有利证据。

### 14.4 其他混淆因素

主实验无意同时改变蒸发率。若需要解释变体差异中的剩余部分，再做 `rho ∈ {0.1,0.2,0.5}` 与来源政策的交叉对照，所有其他规则保持显式固定。

候选表尺度、全图/候选弧蒸发、残差半径和 LS 强度可以继续做独立敏感性实验。涉及 `none/2-opt/3-opt` 时，Origin/LSGain 的可用语义必须处理；不得让 LS-aware 冠军读取未定义 terminal，也不能把改变整个 LS profile 称为只开关一个算子。

---

## 15. P5：消融环境下重新训练

### 15.1 哪些结果需要重新训练才能回答

固定模型消融回答的是部署环境与已有策略的交互，不能证明消融环境更容易学习，也不能说明已有冠军代表全部可能策略。

对原始 C111、一个候选单组件干预、一个必要的交互条件重新训练。条件选择依据开发集与预先规定的规则完成；不能在查看最终确认集后用同一集反复选择新模型。

### 15.2 建议训练规模

默认 **3 个条件 × 10 个独立 GP seeds = 30 次 MMAS 训练**。资源充足时增加训练种子数，并根据训练级不确定性决定是否需要 20/30 个；三个 seeds 只适合训练 pilot。

各条件使用相同训练实例 schedule、ACO 随机流设计、初始 GP 分布、种群规模、代数、节点预算、遗传操作概率、fitness、selection/gate 规则与高保真计算预算。每个条件必须重新生成自己的 baseline archive，缓存键包含完整机制配置。

训练复现默认使用 population 100、50 generations、两树合计节点预算 62；筛选 fitness 为 paired basin mean，高保真 fitness 为 final/anytime 各 0.5。高保真评价日程为：[S4]

| GP 代数 | Screen：实例×ACO seeds×ACO iterations | High：实例×ACO seeds×ACO iterations |
|---|---|---|
| 1–15 | 16×1×50 | 32×1×100 |
| 16–35 | 32×1×100 | 64×2×200 |
| 36–50 | 64×1×200 | 128×3×500 |

原三阶段训练、no-LS TR 注入和训练 horizon 先保持一致，以回答“只换机制会怎样”。同一个训练 replicate 在各条件使用相同 donor TR；如果 donor 只有三个，明确其复用结构，不把 donor 变化误当成 GP seed 的独立贡献。[S4][S5]

原 selection/gate 的相对改善门槛与本计划的绝对 ε 不是同一指标：前者用于候选部署，后者用于组件效应判定。保留 raw selected candidate 的测试结果，不因 gate 失败而让它从机制分析中消失。

另外做一个从零开始的训练敏感性或匹配长 horizon 训练，可排查旧 no-LS TR 与短预算适应度的影响；这些属于新训练协议，单独命名、单独比较预算。

### 15.3 交叉部署矩阵

每个训练条件得到的候选都部署到原始 C111 和对应消融条件：

| 训练环境 \ 评估环境 | 原始 MMAS | 主要消融 MMAS | 交互条件 |
|---|---:|---:|---:|
| 原始 MMAS 训练 | 测试 | 测试 | 测试 |
| 主要消融环境训练 | 测试 | 测试 | 测试 |
| 交互环境训练 | 测试 | 测试 | 测试 |

每一列有自己的无 GP baseline。原始冠军的跨条件部署与重新训练后最优候选的表现分表报告。

如果只在消融环境重新训练后有改善，结论更接近“训练适应/搜索信号改变”；如果固定模型和重新训练都重复出现同一组件交互，归因更稳固。

---

## 16. 统计分析协议

### 16.1 独立单位与配对结构

阶段 P2 的主推断条件化于三个固定冠军：先在同一实例内平均 ACO seeds 与冠军，再对独立实例重采样。另给每个冠军单独的效应与区间。

不能把“3 冠军 × 128 实例 × 10 seeds”当成 3840 个独立实例；蚂蚁、迭代和 checkpoint 更不能作为独立重复。

P5 推断需要同时包含训练 seed 与实例两个交叉维度：同一次重采样对所有条件使用相同训练 replicate 权重、相同实例权重；实例内种子索引也同步用于所有冠军和条件。**不能给每个 GP 模型单独抽实例，破坏共享 baseline 和配对结构。**

### 16.2 主要与次要指标

主指标：`Δ@5000` 及组件移除效应 `E@5000`，单位 pp。

次指标：anytime gap AUC、实例 win/tie/loss、worst-10% tail、达到预先指定 gap 阈值的时间/构造数、重启与来源行为。目标阈值在看确认结果前冻结；未达到者保留为右删失，不能删掉再算平均时间。

AUC 的定义固定为 `1/H * sum_t g_best(t)`。离散采样近似与完整改进事件积分分别标明；不要比较定义不同的 AUC。

### 16.3 置信区间与多重比较

默认 30,000 次配对重采样，保存 bootstrap seed。正式比较族包括：三个原配置附近的移除效应、三个直接效应大小比较、三项二阶和一项三阶交互，共十项对比；再加入 8 格各自的净差 Δ，合计 **18 个预定统计量**。后八项用于判断“GP 优势是否恢复”，避免只校正 E 却对多个 Δ 任意作显著性判断。

建议实现基于实例块重采样的同时置信区间（例如 bootstrap max-t），保留每次重采样中所有条件的相关性。实现须通过模拟覆盖率与零效应单元测试。采用其他统计方法时，需在正式结果可见前冻结，并对这组比较控制整体错误率。

行为指标与每个时间点的相关分析属于次级探索，单独标明；不能从几百个轨迹指标中挑最显著的一项当作预注册主证据。

### 16.4 “没有作用”的判据

`p>0.05` 或区间跨零只表示证据不足。

要称某因素“在本设定下影响可忽略”，其调整后的区间须落在预定 `[-ε,+ε]` 内，并且该机制在原条件中确实执行。若从未触发，只能说“在这批运行中未实际作用”。

要称 X 比 Y 更主要，直接检验 `E_X-E_Y`，不能用“X 显著、Y 不显著”代替二者差异。存在强交互时优先报告条件效应，不强制给所有因素一个普遍排名。

### 16.5 失败、缓存与缺失

预先定义有限次数的工程重试；不得丢弃不利结果后换 seed。缺失运行先补齐相同实验单元；仍失败时报告条件、原因与数量，并做完整配对集和缺失敏感性分析。

NaN、非法路径、数值崩溃是结果完整性问题，不能静默删行。缓存复用不增加独立样本数量。

---

## 17. GPU 实现与工程安排

### 17.1 建议修改位置

下表是实现任务，不表示这些新接口已经存在。

| 文件/模块 | 新增内容 |
|---|---|
| `config.py` / `spec.py` | 显式机制枚举、预算模式、审计配置、严格验证 |
| `aco_cuda.py` | 开关传递、条件分组、审计缓冲、快照/恢复、缓存键 |
| `cuda/aco_tiled_v2.cu` | source/budget/floor/restart 开关，分离计数器，保留 native 分支 |
| `aco_numba.py` | 对应控制变量语义与 FP64 oracle，不另写简化求解器 |
| `evaluation.py` 或独立 evaluation 脚本 | 逐条件 baseline 配对、固定冠军、轨迹与 schema |
| `stats.py` 或独立统计模块 | Δ/E/绝对变化/交互/同时区间，保护配对结构 |
| `tests/` | native 等价、预算、元数据、日程边界、重放恢复、缓存污染测试 |

### 17.2 必须分开的配置字段

建议显式命名：

```yaml
# 协议草案。以下新字段需实现，不可直接交给现有 CLI 运行。
mechanism:
  restart_policy: native       # native | off | replay
  reset_pheromone: true        # replay 分解专用
  reset_epoch_memory: true     # replay 分解专用
  floor_enforcement: native    # native | off | scaled
  floor_scale: 1.0
  hard_upper_clip: native      # native 在当前 LS 路径表示不执行硬上界
  source_policy: native_schedule
  source_count: 1
  budget_policy: native_shadow_total
  metadata_policy: actual_source_original_semantics
  evaporation_scope: native_candidate_arcs

inference:
  model_selection: frozen_selected_candidate
  model_modes: [baseline, full]
  aco_iterations: 5000
  terminal_normalization_horizon: 5000
  aco_seeds_per_instance: 10

analysis:
  primary_metric: paired_reference_gap_pp
  materiality_epsilon_pp: 0.01
  bootstrap_replicates: 30000
  inference_target: fixed_champions

instrumentation:
  level: light
  aggregate_every_iterations: 25
  best_curve: improvement_events
  heavy_subset_manifest: manifests/heavy_subset.json
  algorithm_rng_unchanged: true
```

禁止复用一个 `mmas_enabled` 开关一次关闭来源、边界、重启和初始化。所有默认路径保持原语义。

### 17.3 任务调度

每张 GPU 同时运行一个主要实验 worker，按条件和模型语义分组形成 task matrix，避免不同进程抢同一设备。尽量把配对条件放在同一 GPU/同一运行批次；跨 GPU 时做硬件分层和校准，并随机化条件执行顺序。

生成式 GP 编译缓存与结果缓存都必须包含机制开关。重启/来源政策不同的任务不能误用同一个特化 kernel。可按预编译开关生成少量 kernel 版本，避免在构造热循环中引入不必要分支。

质量主实验统一迭代数和 LS 调用规则；相同 wall-clock 的效率实验另跑。不能因某消融更快就额外给它迭代，混入机制归因主表。

### 17.4 审计成本

所有运行保留轻量统计；完整 τ、全部 tours 与候选上下文仅在重型子集保存。采用设备端窗口约简、稀疏改进事件与分块写盘，避免每步同步 CPU。

在开发集测量 instrumentation off/light/heavy 的耗时和路径一致性。轻量开销目标可设为 5% 以内；达不到时降低审计频率，不能为了速度删掉主指标或改变算法路径。heavy 耗时不能直接拿来比较部署速度。

### 17.5 缓存、恢复与状态快照

运行 manifest 的科学配置哈希至少覆盖：代码、生成 kernel、完整参数、机制、checkpoint、数据坐标与 reference、seed、H 与 terminal H、precision 和审计 schema。

恢复 solver 时不能只保存 τ 与 global-best，还需保存 restart-best、其元数据、时间与停滞状态、相关数组有效性及下一步迭代位置。恢复后 native 继续轨迹应与不中断运行一致。

---

## 18. 推荐运行清单与规模估算

### 18.1 先完成的批次

| 批次 | 条件与规模 | 用途 |
|---|---|---|
| B0 | P0 原始 AS/MMAS，原 32 实例×3 ACO seeds，baseline+3 冠军 | 原现象复现 |
| B1 | R×F×H 8 格，开发集 32×5 seeds，baseline+3 冠军，5000 轮 | 方差、成本、工程检查 |
| B2 | 相同 8 格，独立 Uniform 128×10 seeds，baseline+3 冠军，5000 轮 | 主确认实验 |
| B3 | P3 来源细分、K、重启回放、两树与同状态干预 | 机制解释；先开发、后锁定重复 |
| B4 | Cluster/Gaussian 与 AS 反向验证 | 分布与变体外部验证 |
| B5 | 选定条件的配对重新训练和交叉部署 | 方法级归因 |

B2 的逻辑求解数：

\[
8\times128\times10\times(1+3)=40{,}960.
\]

每次 5000 轮、32 蚂蚁，共约 **65.54 亿条构造路径**，不含初始化和 LS 内部操作。三个分布全矩阵共 122,880 次逻辑求解、约 196.61 亿条构造路径。

这里的“逻辑求解”不是 kernel launch 数，也不是独立统计样本数。实际运行时间必须用服务器 B1 的实测吞吐估算，不从其他 GPU 型号或无 LS 基准外推。

### 18.2 资源不足时的减量顺序

优先减少 P3 的敏感性取值和 heavy 快照频率，其次减少分布外验证的非主要条件。不要先删掉主 8 格、同条件 baseline、独立确认集或基本统计配对。

若 B2 仍超预算，可以在看确认结果前把规模冻结为 64 实例×5 seeds，作为较小确认实验，并明确功效限制。增加相同实例的 seeds 不等价于增加独立实例。

### 18.3 推荐输出目录

```text
runs/mmas-ls-mechanism-v1/
  protocol/
    protocol.md
    preregistered_contrasts.json
    frozen_config.yaml
    code_diff.patch
    source_audit.json
  manifests/
    reproduction.json
    diagnosis_dev.json
    confirm_uniform.json
    confirm_cluster.json
    confirm_gaussian.json
    heavy_subset.json
    retrain_confirm.json
    checkpoints.json
    rng_schedule.json
  validation/
    native_equivalence.json
    switch_and_budget_tests.json
    snapshot_replay_tests.json
    precision_audit.json
  results/
    run_metrics.parquet
    window_metrics.parquet
    best_improvement_events.parquet
    restart_events.parquet
    source_events.parquet
    state_snapshots/
    final_tours/
  analysis/
    factorial_cells.csv
    planned_contrasts.csv
    absolute_change_decomposition.csv
    per_champion_effects.csv
    bootstrap_manifest.json
    failures_and_coverage.json
  report/
    report.md
    figures/
```

### 18.4 执行入口的约定

建议实现 `audit / evaluate / replay / summarize / train-ablation` 五类任务入口。它们是待实现功能，不是当前仓库已提供的命令。

实验工单必须先写实现和验收，再提交正式 GPU 队列。第一张工单应是 P0 加 native/开关测试，第二张是 B1 全因子，第三张才是冻结后的 B2。

---

## 19. 最终必须产出的表和图

| 输出 | 必须表达的信息 |
|---|---|
| 全因子 8 格表 | baseline gap、GP gap、Δ、相对 C111 的 E、区间、每冠军方向 |
| 绝对变化分解表 | A_base、A_GP；区分 baseline 退化与 GP 改善 |
| 组件效应森林图 | 三个移除效应与直接效应大小比较，使用同时区间 |
| 条件效应图 | R/F/H 的背景依赖与交互，不只展示平均主效应 |
| 来源与质量轨迹 | 实际 IB/RB/GB、来源重复/年龄、best-so-far、有效 PH 方差 |
| 重启分解图 | native/no-restart/replay 与 Y00/Y10/Y01/Y11 |
| PH 传递图 | raw→weight→deposit→tau→probability 的变化幅度 |
| LS 差异表 | pre/post-LS 终点质量与结构差异、零分母情况 |
| 稳健性表 | 独立分布、horizon、精度与 AS 反向验证 |
| 重新训练矩阵 | 训练环境×部署环境，含训练 seed 不确定性 |

不得只提交“关 H 后更好”一张表。无法区分 baseline 被削弱还是 GP 得到提升的报告，不足以回答本研究问题。

---

## 20. 结论判定规则

### 20.1 允许的结论类型

| 观察到的证据 | 最终结论写法 |
|---|---|
| 只有一个组件有超过 ε 的稳健净效应，直接比较也支持其更大，独立重复成立 | “在该配置与模型集合中，X 是主要的 GP 收益压缩因素” |
| 多个组件都有实际效应 | “X 与 Y 共同作用”；给出大小与条件，不强行唯一归因 |
| 单组件无稳定效应，联合条件有效或效应反向 | “主要是 X×Y 交互，而非 X 单独作用” |
| Δ 扩大完全来自 baseline 明显退化 | “X 为 baseline 提供了更多收益，因此压缩 GP 的边际优势”；不称为改进算法 |
| 固定冠军有效，但重新训练不重复 | “当前策略与部署机制不匹配”；不能泛化为学习方法普遍限制 |
| GPU 原始实验不重复前期诊断 | 否定或收缩前期假设，以正式实验为准 |
| 区间宽、机制少触发或数值误差相近 | “证据不足”，列出缺口；不能把不显著写成排除 |

### 20.2 “功能重叠”需要什么证据

相对效应和绝对分解可以支持“该组件为 baseline 提供更多收益”。要进一步写成“与 GP 功能重叠”，至少还需要 TR/PH 分解、匹配状态干预或反向加回机制的证据。

如果没有这些证据，最终结论停在具体可观测的干预效应，不把“搜索集中到同一盆地”“抹除了 GP 信息”“替代了 GP 学到的结构识别”当作已经证明的机制。

### 20.3 最终答案模板

> 在固定原始实现、原始冠军和配对随机流的 GPU 实验中，移除 **[组件 X]** 后，GP 相对同配置 baseline 的净差改变 **[E，pp]**，同时区间为 **[区间]**。该效应在 **[交互背景/独立数据/重新训练范围]** 中 **[成立或不成立]**。baseline 的绝对变化为 **[A_base]**，GP 的绝对变化为 **[A_GP]**，因此现象主要属于 **[baseline 已被组件增强 / GP 受到限制 / 二者并存]**。对 restart、下界和来源数量的结论分别为 **[证据]**。结论仅适用于 **[明确范围]**。

所有方括号均待实验填入；本文不提供预填的性能结论。

---

## 21. 开跑前签字检查

正式 GPU 队列提交前，研究团队应确认：

- 原运行版本、模型、数据、precision 已复现；native 补丁和审计不会改变行为。
- R/F/H 开关、来源预算、元数据及重启时钟语义已经通过验收。
- 开发集与确认集不重叠；实例清单、seeds、horizon、ε、比较族和停止规则已冻结。
- 各条件自己的 baseline、共享随机流和缓存键已正确设置。
- 轻量轨迹、重型子集、失败与缺失处理、最终表格 schema 已就绪。
- 所有新入口明确实现状态；不将本协议 YAML 误认为现有可执行配置。

---

## 22. 来源与代码定位

[S1] 仓库提交与当前读取版本：

```text
https://github.com/LinHuanli/RMTGP-ACO/commit/2b847698225e1d0fff36a25c1015a080350a7f00
```

[S2] CUDA v2 最优状态、上下界标称量和来源日程；重点为 `v2_update` 中约 1190–1320 行：

```text
https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/src/rmtgp_aco/cuda/aco_tiled_v2.cu#L1190-L1320
```

[S3] CUDA v2 蒸发、deposit、无 LS 上界分支与重启；重点约 1450–1710 行，具体上界以文件实际行数为准：

```text
https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/src/rmtgp_aco/cuda/aco_tiled_v2.cu#L1450
```

[S4] TSP500 LS-aware v2 模板：

```text
https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/experiments/tsp500_2opt_ls_v2/config.yaml
```

[S5] 实际 variant 参数覆盖与三个正式 GP seeds：

```text
https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/scripts/run_tsp500_ls_v2_campaign.py
```

[S6] 原最终评估的模型、实例数量、seeds、horizon 与输出结构：

```text
https://github.com/LinHuanli/RMTGP-ACO/blob/2b847698225e1d0fff36a25c1015a080350a7f00/scripts/evaluate_tsp500_ls_v2_final.py
```

[S7] Numba 对照实现的来源、terminal、预算与时间输入语义。前期核对版本如下；服务器需与正式冻结版本再核对：

```text
https://github.com/LinHuanli/RMTGP-ACO/blob/128c789b508705274ef27fa2d887f395a947aa77/src/rmtgp_aco/aco_numba.py
```

[S8] Stützle, T.; Hoos, H. H. *MAX–MIN Ant System*. Future Generation Computer Systems, 16 (2000), 889–914。以下为作者论文预印本；第 4 节讨论来源、边界与初始化。文献中的蒸发符号习惯须与本仓库区分，本文统一使用仓库的 `rho=蒸发比例`：

```text
https://lia.disi.unibo.it/Courses/SistInt/articoli/max-min-ant.pdf
```

以上来源支持实现事实与背景。所有样本规模、阶段设计、ε、审计方案和判定规则均为本计划提出的实验选择，须在正式运行前冻结。
