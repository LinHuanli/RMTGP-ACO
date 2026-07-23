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

三种 ACO 必须分别训练。例如运行 AS 的一个独立 GP run：

```bash
python -m rmtgp_aco train \
  --config configs/as_protocol_a.yaml \
  --method-profile rmtgp \
  --root-seed 1001 \
  --processes 4
```

`--method-profile` 也可取 `tr-rgp`、`ph-rgp`、`matched-replace` 或
`legacy`；它们共享同一 ACO 外壳、GP 预算、数据和模型选择协议。

锁定 champion 后在 TSP500-uniform 上做 paired test：

```bash
python -m rmtgp_aco evaluate \
  --config configs/as_protocol_a.yaml \
  --partition tsp500_uniform \
  --champion runs/protocol-a-as-rmtgp/seed-1001/champion.pkl \
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
  --bootstrap-replicates 10000 \
  --output results/as_tsp500_uniform_statistics.json
```

统计报告包括 mean/median gap、win/tie/loss、worst-10% CVaR、Friedman、
Holm-Wilcoxon、paired rank-biserial 和 champion–instance–seed 三层
bootstrap 置信区间。

## 实现结构

- `data.py` / `sampling.py`：严格数据解析、连续距离、candidate list、
  offset cache 与跨代无放回采样；
- `aco.py`：AS、同步 ACS、MMAS 的 PyTorch batch 实现；
- `program.py` / `genetic.py`：DEAP Strongly Typed 双树、postfix tensor
  interpreter 和角色保持遗传算子；
- `training.py`：common random numbers、baseline cache、CPU 多进程个体
  评估、validation champion selection 与 non-inferiority fallback；
- `evaluation.py` / `stats.py`：锁定模型后的 paired test 和论文统计；
- `manifest.py` / `artifacts.py`：数据哈希、Git/环境/seed provenance。

正式主方法使用 support-preserving transition residual 与 per-source
budget-preserving pheromone residual。配置也支持：

- 只训练 transition（TR-RGP）或 pheromone（PH-RGP）；
- matched full-replacement transition；
- 按上一篇研究概念重实现的 Legacy-GP typed profile；
- F0/F1 function set、terminal 子集和四种 pheromone 集成方式。

主实验始终关闭局部搜索；2-opt/3-opt 只属于锁定模型后的鲁棒性扩展。
