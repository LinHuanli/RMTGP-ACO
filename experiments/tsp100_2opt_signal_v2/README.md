# TSP100 full-2opt 学习信号实验

本实验先诊断 2-opt 是否压缩 GP 的行为差异，再决定是否开放 `Origin`
terminal。训练和模型选择只使用 TSP100。测试也先锁定为 TSP100。

已经完成的审计、三代 pilot 数值与正式配置冻结理由见
[`decision_report.md`](decision_report.md)。

审计对每个 AS、ACS、MMAS 分别使用 64 个 residual programs：

- 32 个新随机个体；
- 16 个已有 checkpoint；
- 16 个已有 checkpoint 的单次小变异。

原始 ACO 是独立的第 0 个 program。每个 program 在 8 个 easy 和 8 个 hard
validation instances、5 个公共随机 seeds、500/2000/5000 三个 horizon 上
运行。记录 final gap、pre/post-2opt top-7 AUC、edge retention、
difference survival、SNR 和 MMAS clipping 计数。

`Origin` 的开放门槛在审计脚本中冻结为：

1. 平均 introduced-edge fraction 至少为 0.05；
2. residual 在 construction 阶段产生边差异的比例至少为 0.20；
3. 平均 difference survival 小于 0.80。

默认训练 fitness 为相同实例和 seed 下的 paired UCB：

\[
F=0.8\Delta F_{\mathrm{basin}}+0.2\Delta F_{\mathrm{final}}.
\]

其中 \(F_{\mathrm{basin}}\) 是每轮 post-2opt 最好 7 只蚂蚁的平均 gap，
再对全部 ACO iterations 取平均。最终 validation 和 test 仍以 final
best gap 为主，并使用独立 holdout gate。

每代训练使用 128 个 TSP100 instances。模型选择使用 128 个固定
validation instances，与每代训练批量等大。最终 gate 使用另外 512 个
互不重叠的 validation instances。这样正好使用验证文件中的 640 个实例，
并将频繁使用的模型选择集与一次性的确认 gate 严格分离。

单 variant 审计命令：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/audit_2opt_signal.py \
  --variant acs
```

每个 horizon×seed 都保存为独立 NPZ shard。相同命令可以安全续跑。

两张物理 GPU 的完整审计可由一个可恢复队列启动：

```bash
nohup .venv/bin/python scripts/run_2opt_signal_campaign.py \
  --phase audit --physical-gpus 0 1 \
  > runs/tsp100-2opt-signal-v2/audit-nohup.log 2>&1 &
```

审计完成后，三种 fitness、两个额外 residual radius 和通过门控的
`+Origin` 使用相同初始 population 做三代 pilot：

```bash
nohup .venv/bin/python scripts/run_2opt_signal_campaign.py \
  --phase pilots --physical-gpus 0 1 --pilot-iterations 500 \
  > runs/tsp100-2opt-signal-v2/pilot-nohup.log 2>&1 &
```

确认性训练固定 3 个独立 GP seeds。训练 horizon 固定为 500 次 ACO
迭代。审计显示，继续增加到 5000 次会显著压缩 final-gap 信号；MMAS
baseline 在该 horizon 已全部命中最优。最终测试仍独立使用 5000 次迭代。

三个变体均使用 combined fitness，且不在主配置中加入 `Origin`。AS 使用
\(\gamma=1/3\)。ACS 和 MMAS 使用更保守的 \(\gamma=1/6\)。这些冻结决定
记录在 `formal_decisions.json` 中：

```bash
nohup .venv/bin/python scripts/run_2opt_signal_formal.py \
  --physical-gpus 0 1 --iterations 500 \
  --decisions experiments/tsp100_2opt_signal_v2/formal_decisions.json \
  --origin-mode decisions \
  > runs/tsp100-2opt-signal-v2/formal-nohup.log 2>&1 &
```

9 个 run 全部完成后，只加载每个 run 最终选定的个体。最终测试使用 128 个
TSP100 instances、3 个 ACO seeds 和 5000 iterations：

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python \
  scripts/evaluate_2opt_signal_final.py \
  > runs/tsp100-2opt-signal-v2/test-nohup.log 2>&1 &
```

主成功标准同时要求：

1. mean final gap 的相对降低不少于 10%；
2. paired 层次 bootstrap 的单侧 95% 上界小于 0；
3. 三个独立 GP runs 中至少两个改善。
