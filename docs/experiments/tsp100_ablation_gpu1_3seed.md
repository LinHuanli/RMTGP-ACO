# 纯 TSP100 三种子 GPU1 消融与 OOD pilot

> 状态：实现完成，待后台队列产生实验结果
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

## 4. 批量测试架构

最终测试 partitions 为：

| Partition | 实例数 | 角色 |
|---|---:|---|
| TSP50 uniform | 1,280 | 小尺度迁移 |
| TSP100 uniform | 1,280 | 同分布 |
| TSP500 uniform | 128 | 尺度外推 |
| TSP1000 uniform | 128 | 补充尺度外推 |
| TSP500 cluster | 128 | 分布外 |
| TSP500 Gaussian | 128 | 分布外 |
| TSPLIB \(n\le500\) | manifest 决定 | 真实 benchmark |

每个 instance 使用由 root seed 9001 派生的 3 个 ACO seeds；种子只依赖
`partition×batch×replicate`，不依赖 GP run 或方法。原始 ACO baseline 按
`variant×partition×batch×ACO seed` 只运行一次。

同一种 integration 的多个锁定 programs 被编译成 packed postfix matrix，
一次 CUDA 调用同时计算

\[
\text{program}\times\text{instance}
\]

的全部 500 iterations，并返回 `[program, instance, iteration]` anytime
轨迹。Residual 与 replacement 分成两个 campaign，因为它们的 integration
语义不同。Quality campaign 的墙钟不能无偏分摊到单个并行 program，故
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

主要比较：

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
anytime gap AUC 与 best iteration。TSPLIB 同时报总体与
\(n\le100\)、\(101\le n\le200\)、\(201\le n\le500\) 三个规模带。

Wilcoxon signed-rank 以 `GP run×instance` 为 block；Holm 校正在每个
`ACO variant×RQ family` 内跨 partitions 与同族 contrasts 执行。95% CI
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
3. 42 个 `variant×partition×{residual,replacement}` 测试；
4. 3 个孤立效率 benchmark；
5. 1 个最终报告。

共 133 个任务。每个任务先验证 artifact 内容；训练若存在合法
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

后台启动：

```bash
mkdir -p runs/tsp100-ablation-gpu1-3seed
CUDA_VISIBLE_DEVICES=1 nohup env PYTHONPATH=src \
  .venv/bin/python -m rmtgp_aco run-ablation-study \
  --study-config experiments/tsp100_ablation_gpu1_3seed/study.yaml \
  > runs/tsp100-ablation-gpu1-3seed/nohup.log 2>&1 &
```

监控：

```bash
PYTHONPATH=src .venv/bin/python -m rmtgp_aco ablation-status \
  --study-config experiments/tsp100_ablation_gpu1_3seed/study.yaml
```
