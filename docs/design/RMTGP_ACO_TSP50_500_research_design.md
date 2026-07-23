# RMTGP-ACO：面向 TSP50、TSP100 与 TSP500 的 Multi-Tree GP–ACO 研究设计

> **文档状态**：Design v1.0
>
> **研究对象**：对称二维 Euclidean TSP；Ant System（AS）、Ant Colony System（ACS）和 MAX–MIN Ant System（MMAS）
>
> **核心方法**：使用两棵 Strongly Typed GP 树分别学习状态转移残差和全局信息素强化残差
>
> **实现技术**：DEAP、PyTorch、NumPy、Numba；CPU 多进程与 GPU micro-batching
>
> **范围约束**：本文档只定义研究、算法、接口、伪代码与实验协议，不包含实现代码

---

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

主实验关闭局部搜索，采用 ACOTSP-1.03 中三个算法各自的无局部搜索默认参数：

| 变体 | 蚂蚁数 \(M\) | \(\alpha\) | \(\beta\) | 蒸发率 \(\rho\) | 其他参数 |
|---|---:|---:|---:|---:|---|
| AS | \(n\) | 1 | 2 | 0.50 | 所有蚂蚁全局强化 |
| ACS | 10 | 1 | 2 | 0.10 | \(q_0=0.90,\ \xi=0.10\) |
| MMAS | \(n\) | 1 | 2 | 0.02 | 动态 \(\tau_{\min},\tau_{\max}\) |

统一设置：

- nearest-neighbour candidate list：\(K=\min(20,n-1)\)；
- 主质量预算：100 个 ACO iterations；
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

ACOTSP 可按 wall-clock time 或 constructed tours 终止。本研究主质量实验固定 100 iterations，避免 GP 树的推理开销改变搜索步数。10 秒等时结果单独报告，回答实际部署效率问题。

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
M=n,\quad
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
M=10,\quad
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
M=n,\quad
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

主实验仅运行 100 iterations，ACOTSP 中需要超过 250 次停滞的 restart 条件通常不会触发，但实现仍保留完整状态。

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
| Development GP runs | 10 |
| Final GP runs | 30 |

树复杂度不直接混入第一版 fitness；validation 性能近似相同时，以总节点数作为 tie-breaker。

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
| RMTGP-ACO | residual | residual |

主回答 RQ1–RQ4。

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

- mixed train：50+100+500；
- Train-50；
- Train-100；
- Train-500。

形成：

\[
4\times3
\]

的 train protocol × test scale heatmap。

## 22.6 E5：Terminal ablation

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

## 22.9 E8：Coadaptation

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

## 22.10 E9：Local-search robustness

主模型训练完成后，分别增加：

- ACOTSP-style 2-opt；
- ACOTSP-style 3-opt。

先做 test-time plug-in，不重新训练；若存在稳定收益，再追加 trained-with-LS 实验。

## 22.11 E10：ACS 语义审计

比较：

- ACOTSP sequential local update；
- 本研究 synchronous step update。

在相同 initial cities 和 random streams 下报告：

- 首次 tour divergence step；
- final gap difference；
- convergence difference；
- runtime speedup；
- 结论排序是否改变。

## 22.12 E11：效率

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
- \(M=n\)。

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
- 100 iterations 下 restart 状态。

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

## 27.1 AS/MMAS 默认蚂蚁数导致训练昂贵

AS/MMAS 使用 \(M=n\)，TSP500 每轮构造 500 条 tours。

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
- degradation penalty。

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
