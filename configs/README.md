# 实验配置

`as_protocol_a.yaml`、`acs_protocol_a.yaml` 与 `mmas_protocol_a.yaml` 是主协议
的冻结模板。三种 ACO 分别训练，均从 TSP50 与 TSP100 学习，在 validation
上选择 champion，再锁定后测试 TSP500 的 uniform、cluster 与 Gaussian
分布。每代每规模使用 16 个不同实例，50 代共使用每规模 800 个实例。
Validation 固定拆成每规模 32 个 selection 与 32 个 holdout gate。
Protocol A v0.5 将三种 ACO 的蚂蚁数统一为 32，并将每次 ACO simulation
统一为 500 个 iterations；GP 仍为 population 100、50 generations。
v0.5 还补齐 MMAS 每 100 iterations 的 branching-factor 检查与
ACOTSP-style pheromone restart，因此 v0.4 baseline/checkpoint 不兼容。

`acs_protocol_a_tsp50_only.yaml` 与
`acs_protocol_a_tsp100_only.yaml` 是等计算预算的数据协议对照：单尺度每代
使用 32 个实例，只训练主方法 `rmtgp-full-f1`。

全部确认性方法由 `train --method-profile` 明确选择：

- `legacy`
- `matched-replace`
- `tr-rgp`
- `ph-rgp`
- `rmtgp-core-f0`
- `rmtgp-core-f1`
- `rmtgp-full-f0`
- `rmtgp-full-f1`（主方法；`rmtgp` 是其兼容别名）

正式 30 个独立 run 必须使用不同 `experiment.root_seed` 与 `experiment_id`，
并保留每个 run 的 `manifest.json`、`seeds.json` 和 champion，不得只挑最好
的一次报告。

正式运行顺序为：

```bash
rmtgp-aco prepare-schedules \
  --config configs/acs_protocol_a.yaml \
  --phase pilot --replicate-id 0 \
  --output runs/protocol-a-v0.5/schedules/acs-seed-2001.json

rmtgp-aco precompute-baselines \
  --config configs/acs_protocol_a.yaml \
  --schedule runs/protocol-a-v0.5/schedules/acs-seed-2001.json \
  --output runs/protocol-a-v0.5/baselines/acs/seed-2001.npz

rmtgp-aco train \
  --config configs/acs_protocol_a.yaml \
  --schedule runs/protocol-a-v0.5/schedules/acs-seed-2001.json \
  --method-profile rmtgp-full-f1
```

Baseline archive 必须先完整生成；正式 YAML 使用
`baseline_policy: require`，cache miss、配置 hash 或内核语义变化都会使训练
立即失败。不同消融方法在相同 replicate 上必须复用同一个 schedule 和
baseline archive。

全部方法使用 `max_total_nodes: 31`；双树不再拥有两倍于单树的节点容量。
当前 pilot 每个 ACO×方法运行 3 个 GP seeds；正式论文实验冻结后扩展为 30。
主矩阵为 \(3\times8\times3=72\) runs，另有 6 个 ACS 单尺度 runs，
合计 78。可一次生成带依赖关系的任务图：

```bash
rmtgp-aco prepare-pilot-plan \
  --output runs/protocol-a-v0.5/pilot-plan.json
```

JSON 适合集群任务调度；同目录 `.sh` 是严格顺序执行的可复现版本。

`smoke_as.yaml` 只运行 1 generation、1 iteration 和极小蚁群，用来验证
安装、数据索引、CLI 与 artifact 闭环；其参数不属于任何科学实验。

`development_acs_tsp50.yaml` 是较完整但仍属探索性的 CPU pilot，用于确认
GP 是否存在可学习信号；它只训练 TSP50，不能替代 Protocol A 的正式结果。

`development_acs_cuda.yaml` 使用与正式 ACS 相同的 100-individual、
32-instance、32-ant、500-iteration 单代负载，只把 generations 限为 3，并
启用两张 A5000 的融合 CUDA 后端。它用于 1--3 代工程/质量门控，不产生论文
确认性结果。

三个 `*_protocol_a.yaml` 正式配置固定使用 `numba_batch` float64 CPU
后端：一个进程、16 个 Numba threads，按 genotype×instance 批量调度。
PyTorch 与标量 Numba 保留为语义参考。可先运行 `benchmark-backends`
生成旧 8-process 与新 16-thread 后端的数值一致性和吞吐报告。
`benchmark-accelerators` 进一步比较 CPU16、GPU0、GPU1、dual 和 campaign；
CUDA 搜索为 FP32，但所有 fitness 长度由 CPU 使用原始 FP64 distance matrix
重算。CUDA 训练产生 champion 前还会强制用 `numba_batch` float64 在独立
holdout gate 上复评；该阶段不读取 CUDA baseline archive，避免跨 kernel
semantic domain 混用缓存。

性能方案探索时不要修改正式 YAML 中的 `generations: 50`，而应使用
`benchmark-training --generations 1|2|3`。该命令读取同一冻结 schedule 和
baseline archive，只跳过 validation/checkpoint，因而既保持正式单代计算量，
又不会误把短跑 artifact 当作确认性实验。已有 ACS 短代报告属于旧的
v0.3（10 ants、100 iterations）历史基准，v0.5 必须重新测量：
`docs/performance/acs_short_generation_acceleration_20260724.md`。
