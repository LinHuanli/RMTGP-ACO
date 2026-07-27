# 纯 TSP100：31/62 节点结构 × 容量敏感性实验

> 合同：`experiments/tsp100_capacity_sensitivity_3seed/study.yaml`
>
> 输出：`runs/tsp100-capacity-sensitivity-3seed`（不进入 Git）
>
> 依赖：`runs/tsp100-ablation-gpu1-3seed` 必须按 v2 合同完整完成 112/112

## 1. 研究问题

主消融对所有个体施加

\[
N_{\mathrm{tr}}+N_{\mathrm{ph}}\le31.
\]

该约束使单树和双树具有相同总符号容量，适合隔离 Multi-Tree 的角色分解
效应，但仍存在一个经验问题：31 个节点是否成为双树的性能瓶颈？

本实验用完整的结构 × 容量对照回答：

1. 对同一结构，节点预算由 31 增至 62 是否降低 reference gap；
2. 在总预算固定为 31 或 62 时，双树是否同时优于 TR、PH 两种单树；
3. 双树相对单树的优势是否会随容量变化，即是否存在结构 × 容量交互；
4. 更大表达式是否显著增加训练时间和锁定模型的推理开销。

仅增加双树容量会把“结构”和“表达容量”再次混杂。因此 TR 单树、PH 单树
和 Full-F1 双树都训练 62 节点版本。

## 2. 冻结实验矩阵

| Method | \(B=31\) | \(B=62\) | 角色 |
|---|---|---|---|
| TR-RGP | 单活动树最多 31 | 单活动树最多 62 | transition residual |
| PH-RGP | 单活动树最多 31 | 单活动树最多 62 | pheromone residual |
| RMTGP-Full-F1 | 两树合计最多 31 | 每树最多 31、合计最多 62 | 双 residual |

两级容量均保持：

\[
\text{max depth}=5.
\]

62 节点单树配置将 `max_nodes_per_tree=max_total_nodes=62`；62 节点双树配置
将 `max_nodes_per_tree=31, max_total_nodes=62`。非活动零残差树的哨兵节点
不计入有效节点。

31 节点结果从主消融逐文件哈希复用。新增训练量为

\[
3\ \text{ACO variants}\times
3\ \text{methods}\times
3\ \text{GP seeds}=27\ \text{runs}.
\]

每个 run 继续使用：

\[
\text{population}=100,\quad
\text{generations}=50,\quad
\text{ants}=32,\quad
\text{ACO iterations}=500.
\]

训练为纯 TSP100，每代 32 个实例。三个容量方法与对应 31 节点 run 共享
同一个 root seed、逐代无放回 instance schedule、原始 ACO baseline archive、
selection/holdout validation 和 CPU/FP64 audit。

与主消融相同，每个独立 run 仅用 validation 选择一个 locked champion；
三个 GP root seeds 的三个 champions 全部测试，不测试 GP population，也
不从三个 seeds 中依据 test 结果挑选最好者。

## 3. 测试数据与批量执行

容量与结构消融只在主消融预注册的两个核心分区执行：

\[
\mathcal P_{\mathrm{capacity}}
=\{\mathrm{TSP100\mbox{-}U},\mathrm{TSP500\mbox{-}U}\}.
\]

这两个分区分别测量同训练分布和规模外推；容量实验不重复 TSP50、
TSP1000、cluster、Gaussian 或 TSPLIB 的全部结构笛卡尔积。主方法的完整
OOD 泛化由主消融 v2 的 Full-F1 final test 回答。

每个 instance 使用相同的三个 paired ACO seeds：

\[
s=H(9001,\mathrm{partition},\mathrm{batch},\mathrm{replicate}).
\]

它们不依赖 GP 方法、容量或 root seed。62 节点的 9 个 programs
（3 methods × 3 GP seeds）组成一个 packed postfix program matrix，在同一
CUDA campaign 中并行执行；31 节点质量长表直接复用，因为测试数据、随机流、
ACO 行为哈希和 kernel semantic 均相同。

原始 ACO baseline 不重新计算：

- TSP100-U 与 TSP500-U 读取已完成主 study 的 test cache。

## 4. 预注册指标与统计

基本质量指标为

\[
g_{a,B,r,i,s}
=100\frac{L_{a,B,r,i,s}-L_i^{\mathrm{ref}}}
{L_i^{\mathrm{ref}}},
\]

其中 \(a\) 是结构，\(B\) 是节点预算，\(r\) 是 GP root seed，\(i\) 是
instance，\(s\) 是 ACO seed。

### 4.1 容量主效应

\[
C_{\mathrm{TR}}=g_{\mathrm{TR},62}-g_{\mathrm{TR},31},
\]

\[
C_{\mathrm{PH}}=g_{\mathrm{PH},62}-g_{\mathrm{PH},31},
\]

\[
C_{\mathrm{MT}}=g_{\mathrm{MT},62}-g_{\mathrm{MT},31}.
\]

负值表示同一结构从额外容量中获益。

### 4.2 固定容量的结构效应

\[
A_{\mathrm{TR},B}=g_{\mathrm{MT},B}-g_{\mathrm{TR},B},
\]

\[
A_{\mathrm{PH},B}=g_{\mathrm{MT},B}-g_{\mathrm{PH},B},
\qquad B\in\{31,62\}.
\]

只有同一 \(B\) 下两项均稳定为负，才支持双树优于两个单树对照。

### 4.3 结构 × 容量交互

\[
I_{\mathrm{TR}}
=A_{\mathrm{TR},62}-A_{\mathrm{TR},31},
\]

\[
I_{\mathrm{PH}}
=A_{\mathrm{PH},62}-A_{\mathrm{PH},31}.
\]

\(I<0\) 表示增加预算后双树相对相应单树更有利；\(I>0\) 表示相应单树
从额外容量中获益更多。

### 4.4 推断单位

Wilcoxon signed-rank 先在每个 `GP root×instance` block 内平均三个 ACO
seeds，因此 block 数必须为

\[
3\times N_{\mathrm{instances}},
\]

而不是 `GP root×ACO seed`。95% CI 使用 10,000 次层次 bootstrap：

\[
\text{GP root}\rightarrow\text{instance}\rightarrow\text{ACO seed}.
\]

Holm 校正在每个 `ACO variant×question family` 内跨两个核心 partition 和同族
contrast 执行。本实验为 3-seed pilot，不自动升级为确认性因果结论。

## 5. 复杂度与效率指标

每个 selected champion 保存并汇总：

- transition nodes；
- pheromone nodes；
- total effective nodes；
- `total_nodes / node_budget`；
- 是否恰好撞上预算；
- 每代 wall time；
- train/validation reference gap 与相对 ACO 的 \(\Delta\)；
- non-inferiority gate 与 CPU/FP64 audit。

效率不复用旧 timing。在同一进程中对 31/62 的全部 champions 使用相同
warm-up、batch 和 ACO seeds，逐 program 孤立运行，报告：

- median wall time；
- 相对原始 ACO 的 overhead%；
- tours/s。

质量 campaign 的 packed wall time不分摊给单个 program。

## 6. 任务图与恢复语义

冻结队列共有 46 项：

1. 9 个 `variant×method` 单代预检；
2. 27 个 62 节点正式训练；
3. 6 个 `variant×core-partition` packed tests；
4. 3 个同条件效率 benchmark；
5. 1 个最终统计报告。

阶段间存在硬 barrier：

\[
\text{source complete}
\rightarrow\text{preflight}
\rightarrow\text{train}
\rightarrow\text{test}
\rightarrow\text{efficiency}
\rightarrow\text{report}.
\]

双 GPU worker 对原子任务动态负载均衡。训练从完整
`training_state.pkl` 恢复；测试按 `ACO replicate×batch` 原子 shard 恢复；
每项完成后验证 manifest 和预期行数。源消融在任何时候失败都会使本队列
停止，而不是误用部分结果。

## 7. 后台启动与监控

若源消融仍在运行，可以立即挂起等待队列：

```bash
nohup setsid -f env -u CUDA_VISIBLE_DEVICES PYTHONPATH=src \
  .venv/bin/python -m rmtgp_aco run-capacity-study \
  --study-config experiments/tsp100_capacity_sensitivity_3seed/study.yaml \
  --physical-gpus 0 1 \
  --wait \
  --poll-seconds 60 \
  >> runs/tsp100-capacity-sensitivity-3seed/nohup.log 2>&1
```

runner 等待源状态达到 `completed` 且两张 GPU 均没有外部计算进程后才初始化
CUDA。等待状态、依赖原因、worker、当前任务和训练代数写入原子状态文件。

监控：

```bash
PYTHONPATH=src .venv/bin/python -m rmtgp_aco capacity-status \
  --study-config experiments/tsp100_capacity_sensitivity_3seed/study.yaml
```

## 8. 最终 artifacts

报告目录为 `runs/tsp100-capacity-sensitivity-3seed/report`，至少包含：

- `training_runs.csv`；
- `training_summary.csv`；
- `training_validation_curves_all.csv`；
- `training_validation_curves_aggregate.csv`；
- `quality_summary.csv`；
- `capacity_contrasts.csv`；
- `efficiency_summary.csv`；
- `evidence_audit.csv`；
- `capacity_summary.json`；
- `capacity_report.md`；
- 三种 ACO 的训练曲线 PNG/SVG。

建议论文表述：

- 若 \(C_{\mathrm{MT}}\) 没有稳定负效应，31 节点是有证据支持的简约主配置；
- 若双树在两级容量下均同时优于 TR 与 PH，且交互接近零，优势更符合角色
  分解而非额外节点；
- 若双树优势只在 62 节点下出现，应把主实验的 31 节点限制列为边界条件；
- 若 TR/PH 的 \(C_a\) 比 \(C_{\mathrm{MT}}\) 更负，说明单树更能利用额外容量，
  不能把原始双树差异解释为容量不足。
