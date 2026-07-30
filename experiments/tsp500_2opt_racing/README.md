# TSP500 Anytime+Final 多保真训练

本实验把 TSP100 full-2opt 正式训练保留为对照，并在 TSP500 上恢复被
2-opt 压缩的选择信号。算法结构、terminal/function set 与残差半径不变。
唯一新增的学习机制是可审计的 Anytime+Final fitness 和两阶段 racing。

每代先加载 16 个新实例。Stage 1 在其中 8 个实例上评价全部学习个体。
Stage 2 在 16 个实例上复评 32 个 finalists。32 个名额包括最多 4 个上一代
高保真 elite、24 个 screen 最优个体和 4 个确定性探索个体。baseline anchor
不占这些名额，也不进入繁殖池。繁殖使用
`(fidelity tier, paired UCB, total nodes, hash)` 全序。

主 fitness 为

\[
F=\overline{\tfrac12\Delta g_{\mathrm{anytime}}
 +\tfrac12\Delta g_{\mathrm{final}}}
 +z\,SE,\qquad z=1.
\]

这里两个 gap 都相对同一个最优标签计算；\(\Delta\) 是候选减去同 seed
原始 ACO+2opt。负值表示优于 baseline。Anytime 项是逐轮
global-best-so-far gap 的均值。

正式训练前必须运行 `scripts/audit_tsp500_racing.py`。审计复用冻结的 64
个 programs，并使用 3 个 ACO seeds、100/200/500/5000 horizons。门控要求
final 与 anytime 的非零比例均不低于 0.5，组合信噪比不低于 1，Top-32
召回率不低于 0.8，Spearman 相关不低于 0.7。审计不通过的 variant 不启动
昂贵训练。

三个 ACO variants 各运行 3 个 GP seeds。AS 使用残差半径 \(1/3\)；
ACS 与 MMAS 使用 \(1/6\)。最终 checkpoint selection 使用 validation 的前 16 个
实例；独立 gate 使用其余 48 个实例。最终测试运行 5000 轮。
