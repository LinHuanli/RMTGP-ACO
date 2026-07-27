# 纯 TSP100 三种子消融与 OOD pilot

> 状态：v2 selected-champion 测试合同已冻结；可从现有 artifact 断点续跑
>
> 合同：`experiments/tsp100_ablation_gpu1_3seed/study.yaml`
>
> 输出：`runs/tsp100-ablation-gpu1-3seed`（不进入 Git）

## 1. 目的与推断边界

本 study 用严格配对实验回答四类问题：

1. transition residual 与 pheromone residual 是否分别优于原始 ACO；
2. 双树 Full-F1 是否同时优于两个独立训练的单树 TR-RGP、PH-RGP；
3. residual transition 是否优于使用相同 primitives 与容量的完整替换；
4. Full terminals、F1 functions 以及两棵树同 run 配对是否产生增量价值。

本阶段每个 `ACO variant×method` 只有 3 个 GP root seeds，是工程与方差
pilot，不是确认性实验。任何“更好”的陈述均限于 observed contexts；冻结
协议后仍需扩展到 30 个独立 GP runs。

测试单位不是 GP population，也不是从三个 seeds 中事后挑出的“最好一次”。
每个独立 GP run 先仅依据 validation 选择一个
`selected_candidate.pkl`，然后三个 runs 的三个 locked champions 全部进入
测试和层次统计：

\[
\text{method performance}
=\{c_{r}^{\mathrm{val}}:r=1,2,3\}.
\]

因此既不会测试每代的 100 个 individuals，也不会用 test 表现重新选择
individual 或 GP seed。

## 2. 冻结计算预算

三种 ACO 均使用：

\[
M=32,\qquad T_{\mathrm{ACO}}=500,\qquad K=20.
\]

GP 使用：

\[
P=100,\qquad G=50,\qquad
N_{\mathrm{tr}}+N_{\mathrm{ph}}\le31.
\]

训练为纯 TSP100，每代 32 个实例，50 代由冻结 schedule 无放回抽取
1,600 个训练实例。Validation 使用 32 个 selection instances 与另 32 个
holdout gate instances；最终 candidate 还必须通过 CPU/FP64 audit。

新训练 7 个方法：

| 方法 | Transition | Pheromone | Primitives |
|---|---|---|---|
| Legacy-GP | replacement | 无 | 上一研究 |
| Matched-Replace | replacement | 无 | Full/F1 |
| TR-RGP | residual | 无 | Full/F1 |
| PH-RGP | 无 | residual | Full/F1 |
| RMTGP-Core-F0 | residual | residual | Core/F0 |
| RMTGP-Core-F1 | residual | residual | Core/F1 |
| RMTGP-Full-F0 | residual | residual | Full/F0 |

主方法 RMTGP-Full-F1 的 9 个 runs 从已完成的
`runs/tsp100-gpu0-3seed` 只读复用。每个相同 ACO variant 与 GP replicate
的全部方法共享 schedule 和原始 ACO baseline archive。因此新训练量为

\[
3\ \text{variants}\times 7\ \text{methods}\times
3\ \text{seeds}=63\ \text{runs}.
\]

## 3. Baseline 复用合同

原始 ACO 在没有 GP program 时不读取
`gamma_transition`、`gamma_pheromone`、`transition_integration` 或
`pheromone_integration`。因此 baseline cache 使用行为哈希

\[
h_{\mathrm{ACO}}
=H(\text{ACO fields that affect the no-program execution}),
\]

而完整候选执行仍使用完整 ACO config hash。Schema v3 显式保存两个哈希；
已完成主实验的 schema v2 archive 只有完整 residual config hash，读取时
仅允许迁移冻结的 residual、\(\gamma=1/3\) 配置，随后在内存中重建行为 key。
这使 Matched-Replace/Legacy 与 residual 方法共享数学上相同的 paired
baseline，同时仍拒绝 ants、iterations、candidate size 或 kernel semantic
不一致的 cache。

## 4. Selected-champion 测试架构

主方法 RMTGP-Full-F1 的锁定最终测试 partitions 为：

| Partition | 实例数 | 角色 |
|---|---:|---|
| TSP50 uniform | 1,280 | 小尺度迁移 |
| TSP100 uniform | 1,280 | 同分布 |
| TSP500 uniform | 128 | 尺度外推 |
| TSP1000 uniform | 128 | 补充尺度外推 |
| TSP500 cluster | 128 | 分布外 |
| TSP500 Gaussian | 128 | 分布外 |
| TSPLIB \(n\le500\) | manifest 决定 | 真实 benchmark |

方法、表示和机制消融的预注册作用域仅为：

\[
\mathcal P_{\mathrm{abl}}
=\{\mathrm{TSP100\mbox{-}U},\mathrm{TSP500\mbox{-}U}\}.
\]

TSP100-U 衡量训练分布内效应，TSP500-U 衡量规模外推效应。Core/Full、
F0/F1、residual/replacement、单树/双树以及 drop/shuffle contrasts 只在
这两个分区进行推断。TSP50-U、TSP1000-U、cluster、Gaussian 和 TSPLIB
只报告主方法 Full-F1；先前已经产生的“全方法 × OOD”历史 artifact 保留
以便审计，但显式排除于 v2 汇总和显著性检验。

每个 instance 使用由 root seed 9001 派生的 3 个 ACO seeds；种子只依赖
`partition×batch×replicate`，不依赖 GP run 或方法。原始 ACO baseline 按
`variant×partition×batch×ACO seed` 只运行一次。

同一种 integration 的多个 selected champions 被编译成 packed postfix matrix，
一次 CUDA 调用同时计算

\[
\text{selected champion}\times\text{instance}
\]

的全部 500 iterations，并返回 `[program, instance, iteration]` anytime
轨迹。Residual 与 replacement 分成两个 campaign，因为它们的 integration
语义不同。在每个核心分区中，residual campaign 含
\(9\text{ conditions}\times3\text{ GP runs}=27\) 个 champions，
replacement campaign 含 \(2\times3=6\) 个 champions；非核心分区的 final
campaign 仅含 Full-F1 的 3 个 champions。以上数量均与 GP population
size=100 无关。Quality campaign 的墙钟不能无偏分摊到单个并行 program，故
最终长表把该字段标为缺失；效率另在 TSP50/100/500/1000 固定小 batch 上
逐 champion 预热后独立运行 3 次，报告中位数、相对 ACO 开销和 tours/s。

## 5. 预注册 estimands

所有长度先由原始坐标的 FP64 distance matrix 精确重算。基本指标为
reference gap：

\[
g_{r,i,s}^{(m)}
=100\frac{L_{r,i,s}^{(m)}-L_i^{\mathrm{ref}}}
{L_i^{\mathrm{ref}}}.
\]

所有 contrast 使用：

\[
D=g^{(\mathrm{first})}-g^{(\mathrm{second})};
\qquad D<0 \Longleftrightarrow \mathrm{first\ better}.
\]

以下主要比较只在
\(\mathcal P_{\mathrm{abl}}\) 上计算：

\[
\begin{aligned}
&\mathrm{TR\!-\!RGP}-\mathrm{ACO},\quad
\mathrm{PH\!-\!RGP}-\mathrm{ACO},\quad
\mathrm{Full\!-\!F1}-\mathrm{ACO};\\
&\mathrm{Full\!-\!F1}-\mathrm{TR\!-\!RGP},\quad
\mathrm{Full\!-\!F1}-\mathrm{PH\!-\!RGP};\\
&\mathrm{TR\!-\!RGP}-\mathrm{Matched\!-\!Replace};\\
&\mathrm{Full\!-\!F1}-\mathrm{Legacy\!-\!GP}.
\end{aligned}
\]

Residual 的隔离比较必须使用 TR-RGP 与 Matched-Replace，因为二者都是
单 transition tree、Full/F1、相同总节点预算，区别仅为 residual 与
replacement。Full-F1 对 Legacy 的比较用于说明相对上一篇研究的整体差异，
不能单独归因于 residual。

Core/Full × F0/F1 的 factorial contrasts 为：

\[
C_T=\tfrac12[(F_{10}-F_{00})+(F_{11}-F_{01})],
\]

\[
C_F=\tfrac12[(F_{01}-F_{00})+(F_{11}-F_{10})],
\]

\[
C_{TF}=F_{11}-F_{10}-F_{01}+F_{00},
\]

其中第一下标表示 Core/Full，第二下标表示 F0/F1；负主效应表示相应扩展
降低 reference gap。

Post-hoc 机制测试包括：

- Full-F1 中删除 transition tree；
- Full-F1 中删除 pheromone tree；
- 保留 transition run，分别循环配对另外两个 root seed 的 pheromone tree。

删除测试衡量已进化 pair 内组件的必要性，不等于重新训练的单树模型；
shuffled pairing 只探索 coadaptation。

## 6. 统计与 artifact

描述统计先在每个 `GP run×instance` 内平均三个 ACO seeds，再报告
mean/median gap、标准差、IQR、W/T/L、worst-10% CVaR、reference hit rate、
anytime gap AUC 与 best iteration。主方法的 TSPLIB 同时报总体与
\(n\le100\)、\(101\le n\le200\)、\(201\le n\le500\) 三个规模带。

Wilcoxon signed-rank 以 `GP run×instance` 为 block；Holm 校正在每个
`ACO variant×RQ family` 内跨两个核心 partitions 与同族 contrasts 执行。
主方法在其余分区只给出预注册描述统计和相对原始 ACO 的 paired 描述，不将
缺少消融对照的 OOD 结果纳入消融检验。95% CI
联合重采样：

\[
\text{GP run}\rightarrow\text{instance}\rightarrow\text{ACO seed},
\]

共 10,000 次。最终报告至少生成：

- `training_runs.csv` 与 `training_summary.csv`；
- `quality_summary.csv` 与 `tsplib_size_band_summary.csv`；
- `primary_contrasts.csv`；
- `factorial_contrasts.csv`；
- `mechanism_contrasts.csv`；
- `efficiency_summary.csv`；
- `ablation_summary.json`；
- `ablation_report.md` 与训练曲线 PNG/SVG。

## 7. 可恢复队列

任务顺序固定为：

1. 24 个 `variant×method` 单代预检；
2. 63 个新 50-generation runs（replicate-major）；
3. 12 个核心 `variant×partition×{residual,replacement}` 测试；
4. 9 个 OOD `variant×partition×final` 主方法测试；
5. 3 个孤立效率 benchmark；
6. 1 个最终报告。

共 112 个任务。其中主 study 已完成的 TSP50/100/500/1000 Full-F1 记录
直接逐文件复用，故无需生成重复任务。每个任务先验证 artifact 内容；训练若存在合法
`training_state.pkl` 会自动 `--resume`，测试按原子 batch shard 恢复。
主 runner 使用文件锁、PID、原子 `study_state.json` 和逐任务 log。

正式入口必须使用兼容的项目环境：

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import numpy, numba, torch
print(numpy.__version__, numba.__version__, torch.__version__)
PY
```

当前锁定组合为 NumPy 2.2.6、Numba 0.61.2、PyTorch 2.10.0+cu126。
系统 Python 中的 NumPy 2.4 不得用于本 study。

单卡后台启动：

```bash
mkdir -p runs/tsp100-ablation-gpu1-3seed
CUDA_VISIBLE_DEVICES=1 nohup env PYTHONPATH=src \
  .venv/bin/python -m rmtgp_aco run-ablation-study \
  --study-config experiments/tsp100_ablation_gpu1_3seed/study.yaml \
  > runs/tsp100-ablation-gpu1-3seed/nohup.log 2>&1 &
```

迁移到具有两张空闲 GPU 的机器后，可忽略 GPU 型号差异并从完整 artifact
边界双卡续跑：

```bash
nohup setsid -f env -u CUDA_VISIBLE_DEVICES PYTHONPATH=src \
  .venv/bin/python -m rmtgp_aco run-ablation-study \
  --study-config experiments/tsp100_ablation_gpu1_3seed/study.yaml \
  --physical-gpus 0 1 \
  >> runs/tsp100-ablation-gpu1-3seed/nohup-parallel.log 2>&1
```

双卡 runner 对训练任务动态负载均衡。同一核心
`variant × partition` 的 residual 与 replacement 测试固定在同一张卡上
顺序执行，避免两个进程竞争写入共享 baseline cache；final-only 任务单独
调度。报告仅在全部作用域内训练、测试和效率 artifact 验证通过后生成。

监控：

```bash
PYTHONPATH=src .venv/bin/python -m rmtgp_aco ablation-status \
  --study-config experiments/tsp100_ablation_gpu1_3seed/study.yaml
```
