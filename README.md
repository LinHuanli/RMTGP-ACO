# RMTGP-ACO

本仓库研究 Strongly Typed Multi-Tree Genetic Programming 与 Ant Colony
Optimization 的结合，目标问题为对称二维 Euclidean TSP。

当前支持的 ACO 变体：

- Ant System（AS）
- Ant Colony System（ACS）
- MAX–MIN Ant System（MMAS）

每个 GP 个体包含两棵角色不同的树：

- transition tree：对候选城市 desirability 做有界残差重加权；
- pheromone tree：在每个强化来源 tour 内守恒总预算并重分配边强化。

详细研究协议见
[`docs/design/RMTGP_ACO_TSP50_500_research_design.md`](docs/design/RMTGP_ACO_TSP50_500_research_design.md)。

## 开发约定

- 源码标识符使用英文，公共接口和关键算法使用中文注释与 docstring；
- 正式结果使用 float64；
- 原始数据、模型 checkpoint 和大规模实验输出不提交 Git；
- 所有正式运行必须记录配置、数据 manifest、root seed 和 Git commit。

## 快速检查

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m rmtgp_aco --help
```

集群正式环境使用 `constraints-py312.txt` 固定 NumPy/Numba/llvmlite
兼容组合。Protocol A v0.5 的科学预算固定为 32 只蚂蚁、500 个 ACO
iterations。默认正式配置仍使用单进程、16-thread Numba float64 CPU
后端；另提供融合 CUDA 候选后端，在 GPU 上 FP32 搜索、返回 tour 后由 CPU
FP64 精确计分。RTX 4000 Ada 上的正式质量门、单卡速度门和双卡 campaign
吞吐门均已通过；单 run 双卡扩展为 1.629×，未达到预注册的 1.7×。因此
大规模实验默认采用“一张 GPU 一个独立 run”的 campaign 调度，dual shard
仅用于确实需要降低单代延迟的运行。CPU 配置继续作为 float64 oracle 和
fallback，不因加速后端加入而改写科学协议。

仓库中的 `references/ACOTSP-1.03` 是算法语义参考，保留其原始许可证。

## 主要入口

数据快照已经用完整 SHA-256 固定在 `Datasets/manifest.json`。训练前可做
轻量预检：

```bash
python -m rmtgp_aco verify-data \
  --manifest Datasets/manifest.json \
  --root Datasets/TSP \
  --skip-hashes
```

三种 ACO 必须分别训练。正式训练先冻结 schedule，再一次性预计算原始 ACO
baseline；训练过程中 cache miss 会直接失败。以 AS 的一个 pilot replicate
为例：

```bash
python -m rmtgp_aco prepare-schedules \
  --config configs/as_protocol_a.yaml \
  --phase pilot \
  --root-seed 1001 \
  --output runs/protocol-a-v0.5/schedules/as-seed-1001.json

python -m rmtgp_aco precompute-baselines \
  --config configs/as_protocol_a.yaml \
  --root-seed 1001 \
  --schedule runs/protocol-a-v0.5/schedules/as-seed-1001.json \
  --output runs/protocol-a-v0.5/baselines/as/seed-1001.npz

python -m rmtgp_aco train \
  --config configs/as_protocol_a.yaml \
  --phase pilot \
  --method-profile rmtgp-full-f1 \
  --root-seed 1001 \
  --schedule runs/protocol-a-v0.5/schedules/as-seed-1001.json \
  --baseline-archive runs/protocol-a-v0.5/baselines/as
```

每代使用 TSP50 与 TSP100 各 16 个实例，即 32 个实例；50 代累计使用每规模
800 个不同实例。Validation 每规模另有 32 个 selection 和 32 个独立 gate
实例。

训练在每代后原子写入 `training_state.pkl`。中断后使用完全相同配置恢复：

```bash
python -m rmtgp_aco train \
  --config configs/as_protocol_a.yaml \
  --resume runs/protocol-a-v05-as-rmtgp/seed-1001
```

`--method-profile` 支持 `legacy`、`matched-replace`、`tr-rgp`、
`ph-rgp` 以及 Core/Full × F0/F1 四种双树组合；同一 replicate 必须共享
schedule、baseline archive 和总节点预算。完整 78-run pilot 任务图可用
`prepare-pilot-plan` 生成。

锁定 champion 后在 TSP500-uniform 上做 paired test：

```bash
python -m rmtgp_aco evaluate \
  --config configs/as_protocol_a.yaml \
  --partition tsp500_uniform \
  --champion runs/protocol-a-v05-as-rmtgp/seed-1001/champion.pkl \
  --method RMTGP-ACO \
  --champion-id as-run-01 \
  --seeds 30 \
  --output results/as_run01_tsp500_uniform.csv
```

原始 ACO 使用相同命令但省略 `--champion`。每条输出保留 candidate 与
baseline 的同 seed 配对结果。合并同一
`variant/partition/distribution` 的多个方法长表后：

```bash
python -m rmtgp_aco summarize \
  --inputs results/as_*_tsp500_uniform.csv \
  --reference-method RMTGP-ACO \
  --factorial-methods Core-F0 Core-F1 Full-F0 Full-F1 \
  --bootstrap-replicates 10000 \
  --output results/as_tsp500_uniform_statistics.json
```

统计报告包括 mean/median gap、win/tie/loss、worst-10% CVaR、Friedman、
Holm-Wilcoxon、paired rank-biserial 和 champion–instance–seed 三层
bootstrap 置信区间。

## 纯 TSP100 单卡三种子 study

`experiments/tsp100_gpu0_3seed/study.yaml` 冻结了当前完整预算 pilot：
AS、ACS、MMAS 各 3 个 GP seeds，每代 32 个不同 TSP100 instances，
population 100、50 generations、32 ants、500 ACO iterations。TSP50、
TSP100、TSP500、TSP1000 只在候选锁定后测试，其中 TSP1000 明确为补充
外推，不能用于训练、validation、候选选择或 gate。

后台队列只接受一张物理 GPU 可见，并逐个执行九个训练 run，避免多个进程
争用同一卡。下面以物理 GPU0 为例；恢复时也可将其改为另一张空闲卡：

```bash
mkdir -p runs/tsp100-gpu0-3seed
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  .venv/bin/python -m rmtgp_aco run-study \
  --study-config experiments/tsp100_gpu0_3seed/study.yaml \
  > runs/tsp100-gpu0-3seed/nohup.log 2>&1 &
```

查看当前任务、训练代数、每代时间与 ETA：

```bash
.venv/bin/python -m rmtgp_aco study-status \
  --study-config experiments/tsp100_gpu0_3seed/study.yaml
```

队列为每个 run 冻结 schedule、预计算 baseline、原子 checkpoint，并在失败
后从最近完整 generation 恢复。最终测试的随机流由独立 root seed 9001
生成；三个 GP champions 共享相同的 instance×ACO-seed baseline cache。
报告位于 `runs/tsp100-gpu0-3seed/report/`，同时区分原始 selected candidate
与 validation gate 后的 deployed/fallback 行为。3 个 GP seeds 只构成
pilot，不能作为 30-run 确认性结论。

## 1--3 代加速短跑

在恢复 50 代正式训练前，使用 `benchmark-training` 按完整单代负载执行
1--3 代。该命令复用冻结 schedule 与 baseline archive，但不运行 validation、
不保存 checkpoint，也不产生可作为论文结果的 champion：

```bash
python -m rmtgp_aco benchmark-training \
  --config configs/acs_protocol_a.yaml \
  --schedule runs/protocol-a-v0.5/schedules/acs-seed-2001.json \
  --baseline-archive runs/protocol-a-v0.5/baselines/acs \
  --method-profile rmtgp-full-f1 \
  --generations 3 --cpu-threads 16 \
  --output runs/protocol-a-v0.5/benchmarks/acs-3gen.json
```

Protocol A v0.3（10 ants、100 iterations）的历史 ACS 三代端到端合计由
97.09 s 降至 40.52 s，fitness 与优化前逐代一致；该时间不能外推为 v0.5
的正式代时。历史优化过程、golden 等价检查和逐代记录见
[`docs/performance/acs_short_generation_acceleration_20260724.md`](docs/performance/acs_short_generation_acceleration_20260724.md)。

融合 CUDA 的工程短跑使用相同的正式单代规模，但 baseline 可在开发阶段按
需计算：

```bash
python -m rmtgp_aco benchmark-training \
  --config configs/development_acs_cuda.yaml \
  --phase development --generations 1 \
  --output /tmp/acs-cuda-generation.json
```

同一 population 的 CPU16、两张单卡、双卡和双 run campaign 对照：

```bash
python -m rmtgp_aco benchmark-accelerators \
  --config configs/acs_protocol_a.yaml \
  --phase development --repeats 3 \
  --gpu-devices 0 1 \
  --modes cpu16 gpu0 gpu1 dual campaign \
  --output /tmp/acs-accelerator-matrix.json
```

共享目标 GPU 上若有其他进程，报告只能用于功能检查，不能用于正式速度
门控。监控按目标设备过滤，不会把未参与 benchmark 的其他 GPU 作业误算为
争用。RTX 4000 Ada 的 500-iteration 复核中，CPU16、单卡、dual 分别为
304.32 s、20.81 s、12.78 s；双卡 campaign 在 21.10 s 内完成两个独立
workload，吞吐扩展 1.972×。实现、内存布局、历史受污染测量和正式门控见
[`docs/performance/cuda_fused_architecture_20260724.md`](docs/performance/cuda_fused_architecture_20260724.md)。

GPU 进入正式训练前还必须通过固定的 paired 质量门：

```bash
python -m rmtgp_aco validate-cuda-quality \
  --config configs/acs_protocol_a.yaml \
  --instances-per-scale 128 --seeds 3 \
  --gpu-devices 0 1 --gpu-mode dual \
  --tolerance-pp 0.10 \
  --output /tmp/acs-cuda-quality.json
```

质量门以 `program × instance` 为统计单位，先聚合同一 instance 的 3 个 ACO
seeds，再分别要求 TSP50、TSP100 和 pooled 的单侧 95% 上界不超过
0.10 pp。即使全局 CUDA 质量门已通过，每次 CUDA 训练选出的最终候选仍会在
独立 Numba float64 后端对 holdout gate 复评；GPU gate 或 CPU/FP64 gate
任一失败都保存 baseline fallback。两阶段结果分别写入
`validation_summary.csv` 和 `cpu_fp64_audit_summary.csv`。

## 实现结构

- `data.py` / `sampling.py` / `schedule.py`：严格数据解析、连续距离、
  candidate list、offset cache、phase 隔离的冻结 schedule；
- `aco.py`：AS、同步 ACS、MMAS 的 PyTorch batch 实现；
- `aco_numba.py`：确定性标量 oracle 与 population×instance 并行内核；
- `aco_cuda.py` / `cuda/aco_fused.cu`：问题驻留、FP32 融合搜索、单/双 GPU
  cost-balanced shard 和 CPU FP64 tour 计分；
- `program.py` / `genetic.py`：DEAP Strongly Typed 双树、postfix tensor
  interpreter 和角色保持遗传算子；
- `baseline.py` / `training.py`：不可变 baseline archive、absolute reference
  gap fitness、staged validation 与 non-inferiority fallback；
- `evaluation.py` / `stats.py`：锁定模型后的 paired test 和论文统计；
- `study.py` / `study_report.py`：单 GPU 可恢复队列、跨 champion baseline
  test cache、训练/验证曲线与三层统计报告；
- `manifest.py` / `artifacts.py`：数据哈希、Git/环境/seed provenance。

正式主方法使用 support-preserving transition residual 与 per-source
budget-preserving pheromone residual。配置也支持：

- 只训练 transition（TR-RGP）或 pheromone（PH-RGP）；
- matched full-replacement transition；
- 按上一篇研究概念重实现的 Legacy-GP typed profile；
- F0/F1 function set、terminal 子集和四种 pheromone 集成方式。

主实验始终关闭局部搜索；2-opt/3-opt 只属于锁定模型后的鲁棒性扩展。
