# 实验配置

`as_protocol_a.yaml`、`acs_protocol_a.yaml` 与 `mmas_protocol_a.yaml` 是主协议
的冻结模板。三种 ACO 分别训练，均从 TSP50 与 TSP100 学习，在 validation
上选择 champion，再锁定后测试 TSP500 的 uniform、cluster 与 Gaussian
分布。

单树消融不需要改变 ACO：复制对应模板后，仅修改：

```yaml
gp:
  train_transition: true
  train_pheromone: false  # TR-RGP
```

或：

```yaml
gp:
  train_transition: false
  train_pheromone: true   # PH-RGP
```

正式 30 个独立 run 必须使用不同 `experiment.root_seed` 与 `experiment_id`，
并保留每个 run 的 `manifest.json`、`seeds.json` 和 champion，不得只挑最好
的一次报告。

Matched full-replacement 对照在 ACO 段设置：

```yaml
aco:
  transition_integration: replacement
gp:
  transition_profile: main
  train_transition: true
  train_pheromone: false
```

概念性 Legacy-GP 对照使用同一个 ACO 外壳和实验预算：

```yaml
aco:
  transition_integration: replacement
gp:
  transition_profile: legacy
  train_transition: true
  train_pheromone: false
```

F0/F1 与 terminal ablation 可分别通过 `gp.function_profile`、
`gp.transition_terminals`、`gp.pheromone_terminals` 固定。E7 的强化集成
方式为 `budget_residual`（主方法）、`unnormalized_multiplicative`、
`additive` 或 `replacement`。

`smoke_as.yaml` 只运行 1 generation、1 iteration 和极小蚁群，用来验证
安装、数据索引、CLI 与 artifact 闭环；其参数不属于任何科学实验。

`development_acs_tsp50.yaml` 是较完整但仍属探索性的 CPU pilot，用于确认
GP 是否存在可学习信号；它只训练 TSP50，不能替代 Protocol A 的正式结果。

三个 `*_protocol_a.yaml` 正式配置固定使用 Numba float64 CPU 后端、8 个
单线程 worker。PyTorch 后端保留为语义参考，可用 CLI `--backend torch`
显式覆盖，但覆盖后的结果必须作为不同执行后端单独记录。
