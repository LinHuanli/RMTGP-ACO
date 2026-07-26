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

`as_tsp100_gpu0.yaml`、`acs_tsp100_gpu0.yaml` 和
`mmas_tsp100_gpu0.yaml` 是当前纯 TSP100 三种子完整预算 pilot 的单卡模板。
三者均固定每代 32 个 TSP100 instances、population 100、50 generations、
32 ants、500 iterations，并只允许逻辑 GPU0。统一编排文件为
`experiments/tsp100_gpu0_3seed/study.yaml`；它使用 GP seeds
1001–1003、2001–2003、3001–3003，最终 paired test 独立使用 root seed
9001 和 3 个 ACO seeds。TSP1000 只在该 pilot 中作为 test-only 补充外推，
不能进入 schedule 或 validation。

`as_tsp100_ablation_gpu1.yaml`、`acs_tsp100_ablation_gpu1.yaml` 与
`mmas_tsp100_ablation_gpu1.yaml` 是其严格配对的消融/OOD 模板。统一合同为
`experiments/tsp100_ablation_gpu1_3seed/study.yaml`。该 study 复用主实验
已经锁定的 9 个 `rmtgp-full-f1` runs、schedule、baseline archive、四个
uniform test baseline cache 与 records；新训练其余 7 个方法，共
\(3\times7\times3=63\) 个 50-generation runs。

锁定测试覆盖：

- TSP50、TSP100、TSP500、TSP1000 uniform；
- TSP500 cluster 与 Gaussian；
- TSPLIB \(n\le500\)。

核心方法为 Legacy-GP、Matched-Replace、TR-RGP、PH-RGP 与
Core/Full × F0/F1。另对 Full-F1 做 drop-transition、drop-pheromone 和
两次 shuffled pairing；这些 post-hoc 结果用于机制解释，不计作独立训练
方法。质量评测把 programs 拼成 CUDA task matrix；单方法推理时间另用
warm、固定 batch、逐 champion 的孤立 benchmark 测量，禁止把整个 campaign
墙钟平均分摊给并行 programs。

本机系统 Python 可能解析到 NumPy 2.4，而 Numba 0.61 要求 NumPy <2.3。
正式任务必须显式使用项目锁定环境（当前为 NumPy 2.2.6、Numba 0.61.2）：

```bash
CUDA_VISIBLE_DEVICES=1 nohup env PYTHONPATH=src \
  .venv/bin/python -m rmtgp_aco run-ablation-study \
  --study-config experiments/tsp100_ablation_gpu1_3seed/study.yaml \
  > runs/tsp100-ablation-gpu1-3seed/nohup.log 2>&1 &
```

查看可恢复队列状态：

```bash
PYTHONPATH=src .venv/bin/python -m rmtgp_aco ablation-status \
  --study-config experiments/tsp100_ablation_gpu1_3seed/study.yaml
```

队列共 133 个可验证任务：24 个单代 method-profile 预检、63 个新训练、
42 个 `variant×partition×integration-group` 批量测试、3 个孤立效率测试
和 1 个最终报告。它要求所选物理 GPU 独占运行；启动时要求 clean Git、
至少 50 GiB 可用磁盘，并逐文件记录主 study 复用 artifact 的 SHA-256。
