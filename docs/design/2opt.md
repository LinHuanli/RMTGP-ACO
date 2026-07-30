# 这个结果非常关键：问题大概率不在“GP 表达能力”，而在 2-opt 把学习信号压平了

你现在观察到的现象并不意外，而且几乎复现了参考论文在 **MMAS + 2-opt** 上的核心结论：加入 2-opt 后，MMAS 对状态转移规则的依赖降低，GP-MMAS+2-opt 的提升明显变小，在混合规模数据上甚至可能退化。论文进一步发现，扩大 terminal set 在无局部搜索时有效，但在 MMAS+2-opt 下反而可能因为搜索空间更大、选择压力更弱而更难学习。 

你的残差设计比直接替换规则更保守，因此这种现象可能更明显：

[
\text{small residual change}
;\xrightarrow{\text{2-opt}};
\text{same local optimum}.
]

所以现在最不应该做的事情，是直接继续增加 terminals、functions 或树深度。那会扩大搜索空间，却不会恢复被 2-opt 消除的 fitness signal。

---

# 一、2-opt 为什么会让 RMTGP 学不到

设残差规则构造出的 tour 为：

[
s_{\theta}^{\mathrm{pre}}
=========================

C_\theta(x,\omega),
]

2-opt 是一个映射：

[
H:s^{\mathrm{pre}}\rightarrow s^{\mathrm{post}},
]

最终用于评价和信息素更新的是：

[
s_{\theta}^{\mathrm{post}}
==========================

H(C_\theta(x,\omega)).
]

于是 GP 实际优化的是：

[
F(\theta)
=========

L\left(H(C_\theta(x,\omega))\right).
]

问题在于 (H) 是一个典型的 **many-to-one basin projection**。很多不同的构造 tour 会被 2-opt 映射到同一个或质量非常相近的局部最优解：

[
C_{\theta_1}\neq C_{\theta_2},
\qquad
H(C_{\theta_1})=H(C_{\theta_2}).
]

因此，残差树虽然确实改变了构造过程，但这些改变在最终 fitness 中不可见。

## 1. 有界残差进一步加剧了 basin masking

当前残差最多把概率或 deposit 调整到 baseline 的有限倍数。这在无 local search 时是优点，因为搜索稳定；但有 2-opt 时，小幅度变化通常只是在同一个局部搜索盆地内改变起点：

[
\Delta p \text{ 较小}
\Rightarrow
C_\theta \text{ 改变}
\Rightarrow
H(C_\theta) \text{ 不改变}.
]

fitness landscape 因而变成：

* 大量平台：不同树得到相同结果；
* 少量悬崖：偶尔跨越 basin 后突然发生较大变化；
* 高噪声：偶然 basin switch 容易被错误当成有效规则。

---

## 2. 最终 best-tour fitness 本身过于稀疏

假设每轮有 (m) 只蚂蚁、运行 (T) 轮，GP 最终只看到：

[
\min_{t,a}L(s_{a,t}^{\mathrm{post}}).
]

这相当于从 (mT) 个经过 2-opt 的解中只保留一个极值。即使一个残差规则改善了大部分蚂蚁的 basin quality，只要最终最优解没有变化，它的 fitness 就完全没有改善。

2-opt 越强，这种 floor effect 越严重：

* 多个个体都达到最优解或近似最优解；
* 最终 gap 差异极小；
* GP 选择主要受到随机种子影响。

参考论文的 Table 2 中，加入 2-opt 后很多设置的 normalized mean 都已经非常接近 1，标准差也很小，说明最终结果确实接近饱和。

---

## 3. 信息素更新存在 credit mismatch

这里需要区分你的具体执行顺序。

### 情况 A：2-opt 后的 tour 用于信息素更新

这是较常见的设置。此时信息素更新看到的边包括：

* 构造阶段产生并被 2-opt 保留的边；
* 2-opt 新加入的边；
* 构造阶段产生但已被 2-opt 删除的边则完全消失。

如果信息素树不知道 edge provenance，它无法区分：

> 这条边是状态转移规则找到的，还是 2-opt 修补出来的。

于是 pheromone residual 只能对最终 tour 上的边进行盲目重分配。

### 情况 B：2-opt 前的 tour 用于信息素更新

这种情况下，信息素树直接奖励构造行为，但 fitness 评价的是 2-opt 后质量：

[
\text{update target}=s^{pre},
\qquad
\text{fitness target}=s^{post}.
]

这同样存在目标错位。

因此，无论哪种顺序，如果没有记录 pre/post-LS 的关系，信息素树都缺少正确的信用信息。

---

## 4. MMAS 的信息素上下界可能让 residual 失效

如果 2-opt 经常产生相似的优质 tour，其边会快速接近：

[
\tau_{\max}.
]

即使信息素树把更多 deposit 分配给某条边，最终也可能被 clipping 截断：

[
\tau_e^{t+1}
============

\min
\left(
\tau_{\max},
(1-\rho)\tau_e^t+D_e
\right).
]

这意味着“数学上重新分配了 deposit”，但“实际生效的 deposit”接近零。

建议记录：

[
D_e^{\mathrm{effective}}
========================

\min
\left(
D_e,,
\tau_{\max}-(1-\rho)\tau_e
\right)
]

以及：

[
\text{clipping-loss ratio}
==========================

1-
\frac{\sum_eD_e^{\mathrm{effective}}}
{\sum_eD_e}.
]

如果这个比例很高，当前 pheromone residual 实际上没有多少控制空间。

---

# 二、第一步不要修改算法，先做一次“学习信号审计”

在重新设计前，建议先用现有代码完成一个小型诊断。它比再运行几十次完整 GP 更重要。

可以随机抽取：

* 50–100 个随机 residual 个体；
* 10–20 个已有 GP 个体；
* 10–20 个 baseline 附近的小变异个体；
* 每个个体运行相同的 10 个实例和 5–10 个随机种子。

记录以下数据。

## 1. Headroom

分别统计 baseline ACO+2opt 在 TSP50、100、200、500 上的：

* mean optimality gap；
* median gap；
* optimum hit rate；
* 不同种子间标准差。

如果在 TSP50/100 上：

[
\Pr(L=L^*)>70%
]

或者 mean gap 已小于约 (0.1%)，那么这些规模不适合作为 full-2opt GP 的主要训练集，因为可学习空间已经过小。

这种情况下，TSP50/100 可以保留为泛化测试，但训练应更多使用 TSP200/500 或更小的 ACO 预算。

---

## 2. Local-search compression ratio

对每个实例和相同随机流，计算不同 residual rules 产生的 pre-LS 与 post-LS gap 方差：

[
R_{\mathrm{compression}}
========================

\frac{
\operatorname{Var}*{\theta}
\left[
g^{\mathrm{post}}*\theta
\right]
}{
\operatorname{Var}*{\theta}
\left[
g^{\mathrm{pre}}*\theta
\right]+\epsilon
}.
]

如果：

[
R_{\mathrm{compression}}\ll 1,
]

说明 residual 在构造阶段产生的差异大部分被 2-opt 消除了。

同时计算不同个体 pre/post 质量的 Spearman correlation：

[
\rho_{\mathrm{pre,post}}
========================

\operatorname{Spearman}
\left(
g^{\mathrm{pre}},
g^{\mathrm{post}}
\right).
]

解释如下：

* compression 很小、correlation 很高：2-opt 保留排序，但缩小差异；
* compression 很小、correlation 也低：2-opt 几乎彻底改变了评价景观；
* compression 尚可：主要问题可能是噪声或当前 fitness 太稀疏。

---

## 3. Edge survival

对每只蚂蚁记录：

[
E^{pre},\qquad E^{post}.
]

计算：

[
R_{\mathrm{retained}}
=====================

\frac{|E^{pre}\cap E^{post}|}{n}.
]

还可以比较 residual 与 baseline 的差异是否在 2-opt 后仍然存在：

[
R_{\mathrm{difference-survival}}
================================

\frac{
|
(E_R^{pre}\triangle E_B^{pre})
\cap
(E_R^{post}\triangle E_B^{post})
|
}{
|E_R^{pre}\triangle E_B^{pre}|+\epsilon
}.
]

如果 residual 明显改变 pre-tour，但 post-tour difference survival 接近零，就可以直接证明：

> 不是 GP 没有产生行为差异，而是 2-opt 擦除了这些差异。

---

## 4. Fitness signal-to-noise ratio

估计：

[
\mathrm{SNR}
============

\frac{
\operatorname{Var}*{\theta}
\left[
\mathbb E*{\omega}F(\theta,\omega)
\right]
}{
\mathbb E_{\theta}
\left[
\operatorname{Var}_{\omega}
F(\theta,\omega)
\right]
}.
]

其中：

* 分子是不同规则之间的真实差异；
* 分母是同一个规则在不同随机种子下的波动。

如果：

[
\mathrm{SNR}<1,
]

那么用一次 ACO run 评价 GP 个体时，selection 大部分是在选择随机噪声，而不是选择更好的规则。

参考论文为了加速训练，每个个体只执行一次随机 ACO run；在无 local search、差异较大时尚可，但在 2-opt 把差异压缩后，这种评价很可能不够稳定。

---

## 5. Residual 是否真的产生非平凡输出

分别记录：

[
\operatorname{Std}*{j\in C_i}
r*{ij}^{tr}
]

和：

[
\operatorname{Std}_{e\in U_t}
r_e^{ph}.
]

还要记录：

* `tanh` 输出位于 ([-0.05,0.05]) 的比例；
* 位于饱和区 (|r|>0.95) 的比例；
* 实际 (p/p^0) 分布；
* 实际 (D/D^0) 分布。

因为候选集合中相同的常数 residual 会在归一化后完全抵消。如果 2-opt 环境下选择压力太弱，GP 很可能漂移到语义上接近常数的树。

---

# 三、最小改进不应先改表示，而应先改 fitness

当前最值得尝试的，是把 fitness 从“最终最好 tour”改为“经过 2-opt 后的整群 basin quality”。

## 1. 使用 post-2opt top-(q) quality

设每一轮经过 2-opt 后的蚂蚁 tour 为：

[
L_{1,t}^{post},\ldots,L_{m,t}^{post}.
]

取最好的 (q) 只蚂蚁，例如：

[
q=\max(2,\lceil0.2m\rceil).
]

定义：

[
G_{q,t}^{post}
==============

\frac{1}{q}
\sum_{a\in \operatorname{Top}*q(t)}
\frac{
L*{a,t}^{post}-L^*
}{
L^*
}.
]

再计算整个 ACO 过程的 AUC：

[
F_{\mathrm{basin}}
==================

\frac1T
\sum_{t=1}^{T}
G_{q,t}^{post}.
]

这个指标仍然完全评价 **2-opt 后的质量**，但比最终 best-tour 提供更密集的信号：

* 好规则让更多蚂蚁进入优质 basin；
* 好规则可能更早进入优质 basin；
* 不必等到最终 global best 改变才获得 fitness 改进。

## 2. 推荐训练 fitness

可以先使用：

[
F_{\mathrm{train}}
==================

0.8\Delta F_{\mathrm{basin}}
+
0.2\Delta F_{\mathrm{final}},
]

其中：

[
\Delta F
========

F_{\mathrm{RMTGP}}-F_{\mathrm{baseline}}
]

是在相同实例、相同随机流下相对于 baseline 的配对差异。

最终 validation 和 test 仍然报告：

* final best gap；
* anytime AUC；
* time-to-target；
* win/tie/loss rate。

暂时不要把 pre-LS tour quality 赋予太高权重，因为“构造得短”未必等于“进入更好的 2-opt basin”。pre-LS gap 和 2-opt move 数可以先作为 tie-breaker 或辅助分析指标。

---

## 3. 使用 common random numbers

同一代所有个体应使用相同的：

* 实例；
* 起始城市；
* ant-level random streams；
* categorical/Gumbel noise；
* 2-opt 扫描顺序。

每个 residual 个体还应与 baseline 使用相同随机流，fitness 使用配对差异。

由于状态一旦分叉，后续随机选择不可能完全对应，但按“ant–iteration–construction-step”对齐随机流仍然能显著降低评价方差。

## 4. 训练时使用 racing

不需要对所有个体都运行很多种子：

1. 所有个体先运行 1 个公共 seed；
2. 前 20% 个体增加 2 个 seeds；
3. elite/archive 候选增加到 5 个 seeds；
4. validation champion 再做完整重复。

这样比所有个体单次随机评价可靠得多，但成本不会增加到五倍。

---

# 四、最值得增加的是 2-opt-aware pheromone terminals

现在不建议先扩展 transition terminal set。参考论文已经显示，在 local search 削弱选择压力后，直接增加更多状态转移 terminals 可能使搜索更困难。

最直接的改进是让信息素树知道：

> 最终 tour 上的这条边到底是构造规则发现的，还是 2-opt 修复出来的？

假设 baseline 仍然使用 post-2opt tour 更新信息素，不改变其更新支持和总预算。

## 推荐的 pheromone terminal set

[
\mathcal T_{\mathrm{ph}}^{LS}
=============================

{
EdgeEta,
EdgeTau,
TauHeadroom,
PreFreq,
PostFreq,
Origin,
LSGain,
ACOProg,
Stagnation,
ERC
}.
]

### `Origin`

对 post-2opt update tour 上的边 (e)：

[
Origin_e=
\begin{cases}
+1,& e\in E^{pre}\cap E^{post},\
-1,& e\in E^{post}\setminus E^{pre}.
\end{cases}
]

它区分：

* constructed-and-retained；
* introduced-by-2opt。

这一个 terminal 就可以回答一个非常有意义的科学问题：

> ACO 应该主要记忆构造规则成功找到的边，还是记忆局部搜索修复得到的边？

### `LSGain`

每次 accepted 2-opt move：

[
(a,b),(c,d)
\rightarrow
(a,c),(b,d)
]

对应 gain：

[
g=
d(a,b)+d(c,d)-d(a,c)-d(b,d).
]

对该 move 引入的新边赋予归一化 gain：

[
LSGain_e
========

\operatorname{clip}
\left(
\frac{g}{L^{pre}/n+\epsilon},
0,1
\right).
]

构造并保留的边设为 0。

如果一条边被多次引入和删除，记录它最后一次进入当前 tour 时对应的 gain。

### `PreFreq` 和 `PostFreq`

对于本轮所有蚂蚁：

[
PreFreq_e
=========

2\frac{#{a:e\in E^{pre}_a}}{m}-1,
]

[
PostFreq_e
==========

2\frac{#{a:e\in E^{post}_a}}{m}-1.
]

它们能区分：

* 构造阶段已有较高 consensus 的边；
* 经过 2-opt 后才形成 consensus 的边；
* 只在某一条偶然 best tour 中出现的边。

### `TauHeadroom`

相比原来的 `TauPosition`，更直接的量是蒸发后剩余的有效强化空间：

[
TauHeadroom_e
=============

\operatorname{clip}
\left(
\frac{
\tau_{\max}-(1-\rho)\tau_e
}{
\tau_{\max}-\tau_{\min}+\epsilon
},
0,1
\right).
]

它帮助更新树避免把 deposit 预算继续分配给即将被 clipping 的边。

---

## 信息素残差公式不需要改

仍然使用你现在的 budget-preserving 形式：

[
D_{e,t}
=======

B_t
\frac{
D^0_{e,t}
\exp
\left[
\lambda_{ph}
\tanh(T_{ph}(z_{e,t}))
\right]
}{
\sum_{e'}
D^0_{e',t}
\exp
\left[
\lambda_{ph}
\tanh(T_{ph}(z_{e',t}))
\right]
}.
]

因此：

* baseline update support 不变；
* 总 deposit 预算不变；
* 2-opt 不变；
* MMAS 不变；
* 只是给 residual tree 增加它原本缺失的信用信息。

这是当前最小、最符合原研究思想的改进。

---

# 五、transition terminal 暂时只增加两个 2-opt 稳定性特征

原 transition terminals 可以保留：

[
{
RTau,
REta,
BaseConf,
DistRank,
Entropy,
ConstructProg,
ACOProg,
Stagnation,
ERC
}.
]

不建议一次增加很多几何量。可以先加入两个成本低、容易向量化的特征。

## 1. `MutualRank`

设 (r_i(j)) 是 (j) 在城市 (i) 候选列表中的归一化排名，则：

[
MutualRank_{ij}
===============

1-
\frac{
r_i(j)+r_j(i)
}{
2
}.
]

相互都是近邻的边通常比单向近邻更稳定。该特征可以预计算，不增加在线复杂度。

## 2. `TurnCos`

若 partial tour 的前两个城市为 (h\rightarrow i)，候选为 (j)：

[
TurnCos_{hij}
=============

\frac{
(x_i-x_h)^\top(x_j-x_i)
}{
|x_i-x_h|
|x_j-x_i|+\epsilon
}.
]

它位于 ([-1,1])，可以直接向量化。它不能完整预测 2-opt，但能提供局部几何兼容性。

更复杂的 crossing risk 或 one-step 2-opt gain proxy，可以放到后续实验，不应立即加入 core set。

---

# 六、function set 先不要扩展

继续使用：

[
\boxed{
\mathcal F=
{
+,-,\times,\operatorname{pdiv},
\min,\max,\operatorname{abs}
}
}
]

并保留精确的常数：

[
0,\quad 1,\quad -1.
]

现在的瓶颈不是树无法表达更复杂公式，而是复杂公式没有稳定的 fitness signal。

特别不建议立即增加：

* `exp`；
* `log`；
* `sin/cos`；
* 大量条件分支；
* reduction operators；
* 更深的树。

`Origin` 这类二值 terminal 不需要 `if-then-else`。GP 可以通过：

[
Origin\times f(x)
]

或：

[
\frac{1+Origin}{2}f_1(x)
+
\frac{1-Origin}{2}f_2(x)
]

形成门控逻辑。

---

# 七、训练方式建议改为分阶段，而不是直接 joint + full 2-opt

直接同时进化两棵树，并在每次评价中运行完整 2-opt，搜索空间大而信号弱。

更合理的顺序是：

## Phase A：确认 PH residual 能否在 full 2-opt 下学习

固定：

[
T_{tr}=0,
]

只进化带 2-opt provenance terminals 的：

[
T_{ph}.
]

比较：

* MMAS+2opt；
* PH-GP+2opt。

如果 PH-GP 仍完全学不到，那么问题主要在：

* update residual 控制范围太小；
* 信息素 clipping；
* fitness 过度饱和；
* 评价噪声。

此时没有必要立即恢复 joint evolution。

## Phase B：使用无 local search 的 transition champion

取已经在无 local search 环境中学到的：

[
T_{tr}^{noLS},
]

在 full 2-opt 环境中固定它，只进化：

[
T_{ph}^{LS}.
]

这一步检验：

> 信息素树能否学会配合一个已经有意义的构造 residual。

## Phase C：联合微调

最后再同时进化两棵树，但将操作概率设为：

[
P(\text{modify }T_{ph})=0.7,
\qquad
P(\text{modify }T_{tr})=0.3.
]

因为在 MMAS+2opt 下，信息素路径通常比状态转移路径拥有更强的有效控制力。

这三个阶段都只改变训练过程，最终测试算法仍然是完全相同的 MMAS+2opt。

---

# 八、可以加入 local-search curriculum，但要保持 full-2opt validation

另一种方法是逐步增加 2-opt 强度。

例如：

| GP generations |                          2-opt 强度 |
| -------------- | --------------------------------: |
| 1–15           | (n/16) 次 accepted/attempted steps |
| 16–30          |                             (n/8) |
| 31–45          |                             (n/4) |
| 46–60          |                            当前完整设置 |

更稳妥的是混合评价：

[
F_g
===

(1-\lambda_g)F_{\mathrm{weakLS}}
+
\lambda_gF_{\mathrm{fullLS}},
]

其中：

[
\lambda_g
=========

\min\left(1,\frac{g}{G_c}\right).
]

这样 GP 早期能从较平滑、差异较大的环境中获得方向，后期逐步适应真实的 full-2opt 目标。

但是：

* champion selection 必须始终依据 full-2opt validation；
* test 只运行 full-2opt；
* curriculum 只是训练技术，不能用弱 local search 的结果替代最终结果。

---

# 九、残差半径需要实验，但不应直接无限放大

当前保守 residual 可能不足以跨越 2-opt basin。建议比较三个半径，而不是直接取消约束。

例如对于 log-residual：

[
r=\lambda\tanh(T(x)),
]

测试：

[
\lambda\in
\left{
\frac{\log 2}{2},
\frac{\log 4}{2},
\frac{\log 8}{2}
\right}.
]

它们对应越来越大的局部概率比值范围。

需要同时报告：

* post-2opt basin switch rate；
* final gap；
* worse-than-baseline rate；
* residual multiplier 分布；
* GP-run variance。

预期可能出现：

[
\text{small radius}
\Rightarrow
\text{安全但不能跨 basin},
]

[
\text{large radius}
\Rightarrow
\text{能跨 basin 但噪声和退化增加}.
]

这个 improvement–risk trade-off 本身就是很好的实验结果。

---

# 十、建议下一轮优先完成的实验矩阵

## E1：2-opt 强度与 learnability

固定现有 terminals 和 functions，比较：

[
{0,n/16,n/8,n/4,\text{full}}
]

五种 local-search 强度。

方法：

* baseline；
* TR-only；
* PH-only；
* joint RMTGP。

报告：

* final gap；
* top-(q) post-LS AUC；
* compression ratio；
* SNR；
* edge retention；
* basin diversity；
* clipping-loss ratio。

这个实验会明确告诉我们学习信号到底在哪个强度开始消失。

---

## E2：fitness ablation

比较：

1. final best only；
2. post-LS top-(q) AUC；
3. (0.8) AUC (+0.2) final best；
4. 方法 3 + paired common random numbers；
5. 方法 4 + racing。

如果方法 2–5 恢复了 GP 提升，说明主要瓶颈是评价，而不是 representation。

---

## E3：LS-aware pheromone terminals

比较：

1. 原有 pheromone terminals；
2. `+ Origin`；
3. `+ Origin + LSGain`；
4. `+ Origin + LSGain + PreFreq + PostFreq`；
5. 完整 set，再加 `TauHeadroom`。

这一消融应先在 PH-only 中进行，避免 joint search space 混淆结论。

---

## E4：训练策略

比较：

* direct joint + full 2-opt；
* no-LS pretraining → full-LS fine-tuning；
* transition freeze → pheromone learning → joint tuning；
* local-search curriculum。

---

## E5：规模和饱和度

如果 TSP50/100 的 baseline+2opt 经常达到 optimum，主训练应转向：

* TSP200；
* TSP500；
* 或更少的 ACO iterations。

TSP50/100 仍保留用于：

* baseline recovery；
* correctness；
* scale-transfer；
* optimum-hit analysis。

---

# 十一、如何利用你已有的标签

## 如果只有最优 tour length

最适合用于：

* 每只蚂蚁 post-LS gap；
* top-(q) AUC；
* instance difficulty；
* hard-instance weighting；
* time-to-target。

训练集中的实例可以根据 baseline+2opt gap 分层采样：

[
w_x
\propto
\operatorname{clip}
\left(
Gap_{\mathrm{baseline}}(x),
w_{\min},w_{\max}
\right).
]

这样不会让大量已经被 baseline 轻易求解的实例淹没困难实例的信号。

## 如果还有完整最优 tour

可以额外计算：

[
EdgeRecall(s,s^*)
=================

\frac{|E(s)\cap E(s^*)|}{n}.
]

但它更适合作为：

* 解释指标；
* tie-breaker；
* pre/post-LS edge analysis；

不建议把它作为主要 fitness，因为：

* 一个实例可能有多个等价或近似等价 tour；
* edge overlap 较低不一定意味着 tour length 差；
* 它可能把 GP 推向某一个特定 optimum 的边结构。

---

# 十二、推荐的 RMTGP-ACO+2opt 第二版

## Transition terminals

先使用：

[
\boxed{
{
RTau,
REta,
BaseConf,
DistRank,
Entropy,
ConstructProg,
ACOProg,
Stagnation,
MutualRank,
TurnCos,
ERC
}
}
]

## Pheromone terminals

使用：

[
\boxed{
{
EdgeEta,
EdgeTau,
TauHeadroom,
PreFreq,
PostFreq,
Origin,
LSGain,
ACOProg,
Stagnation,
ERC
}
}
]

## Function set

保持：

[
\boxed{
{
+,-,\times,\operatorname{pdiv},
\min,\max,\operatorname{abs}
}
}
]

## Training fitness

[
\boxed{
F_{\mathrm{train}}
==================

0.8\Delta
\left[
\frac1T\sum_tG^{post}*{q,t}
\right]
+
0.2\Delta g^{post}*{best}
}
]

其中所有差异均相对于相同随机流下的 baseline。

## 训练顺序

[
\boxed{
\text{PH-only full-LS}
\rightarrow
\text{freeze no-LS transition champion}
\rightarrow
\text{joint full-LS fine-tuning}
}
]

---

# 十三、这个结果可以把研究变得更有意思

现在的研究问题已经不只是：

> residual GP 能不能提高 ACO？

而可以变成：

> **Local search 在多大程度上消除了自动设计规则的行为差异？怎样通过 basin-level fitness 和 local-search-aware pheromone credit 恢复可学习性？**

这样论文可以形成一条很清晰的机制链：

1. RMTGP 在无 local search 下显著改善 ACO；
2. 2-opt 将不同构造规则映射到相同局部最优 basin；
3. fitness variance 和 component credit 因而显著下降；
4. 单纯增加 terminals 无法解决，甚至扩大无效搜索空间；
5. post-LS population fitness、edge provenance 和 staged evolution 恢复学习信号；
6. 最终 joint residual 在 full-2opt 环境下改善 basin quality、收敛速度或最终 gap。

因此，你现在的结果不是简单的负结果。它准确暴露了下一步真正值得解决的问题。最优先的动作是先完成 **compression/SNR/edge-survival audit**，随后只改两件事：**fitness 改成 post-LS top-(q) AUC，pheromone tree 加入 edge provenance**。这两个变化最小，但最直接针对当前失败机制。
