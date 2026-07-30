# TSP100 2-opt 学习信号审计与正式配置冻结报告

## 1. 报告范围

本报告记录正式训练开始前已经冻结的证据与决策。学习信号审计使用
64 个 residual programs、16 个 TSP100 validation instances 和 5 个
公共 ACO seeds。三代 pilot 使用 100 个 GP individuals，并在每代
128 个 TSP100 training instances 上评价。

下文中的 \(\Delta\) 均表示

\[
\Delta=\text{RMTGP-ACO}-\text{baseline ACO}.
\]

因此，负值表示 RMTGP-ACO 更好。单位 `pp` 表示 gap 的百分点差。
Pilot 只有一个 GP seed。它只用于冻结机制，不作为确认性结果。

## 2. 学习信号审计

表 1 报告 500 次 ACO 迭代时的主要审计结果。

| ACO | baseline final gap (%) | 最优命中率 | 压缩比 | pre/post Spearman | final SNR | basin SNR | 2-opt 引入边比例 | 差异存活率 | Origin 门控 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| AS | 0.2519 | 0.3625 | 0.0347 | 0.9399 | 0.1002 | 35.8115 | 0.2110 | 0.3350 | 通过 |
| ACS | 0.1103 | 0.7000 | 0.0108 | 0.6602 | 0.1135 | 1.2395 | 0.0938 | 0.3544 | 通过 |
| MMAS | 0.0270 | 0.8625 | 0.0146 | 0.8106 | 0.1463 | 0.3012 | 0.0946 | 0.3425 | 通过 |

压缩比远小于 1。这说明 2-opt 显著压缩了不同构造规则在最终解上的方差。
三种算法的 final SNR 均远小于 1。AS 的 basin SNR 为 35.81，明显高于
final SNR。ACS 的 basin SNR 也更高，但信号较弱。MMAS 在 TSP100 上已经
接近饱和。

当 horizon 从 500 增加到 5000 时，baseline 的 final gap 和最优命中率
变化如下。

| ACO | gap@500 (%) | gap@5000 (%) | hit@500 | hit@5000 |
|---|---:|---:|---:|---:|
| AS | 0.2519 | 0.0848 | 0.3625 | 0.6375 |
| ACS | 0.1103 | 0.0196 | 0.7000 | 0.9250 |
| MMAS | 0.0270 | 约 0 | 0.8625 | 1.0000 |

因此，正式训练使用 500 次迭代。最终测试仍使用 5000 次迭代。该设置减少
训练饱和，也保留了长 horizon 泛化检验。

## 3. 三代 pilot

表 2 报告第 3 代最优个体在该代训练批次上的配对差。`—` 表示
final-only fitness 没有请求 basin 统计。

| ACO | 配置 | final \(\Delta\) (pp) | basin \(\Delta\) (pp) | 正式采用 |
|---|---|---:|---:|:---:|
| AS | final-only, \(\gamma=1/3\) | -0.0274 | — | 否 |
| AS | basin-only, \(\gamma=1/3\) | -0.0114 | -0.8133 | 否 |
| AS | combined, \(\gamma=1/3\) | -0.0238 | -0.8795 | 是 |
| AS | combined, \(\gamma=1/6\) | -0.0072 | -0.4667 | 否 |
| AS | combined, \(\gamma=2/3\) | +0.0574 | -1.6900 | 否 |
| AS | combined, \(\gamma=1/3\), `+Origin` | -0.0050 | -1.0855 | 否 |
| ACS | final-only, \(\gamma=1/3\) | -0.0305 | — | 否 |
| ACS | basin-only, \(\gamma=1/3\) | -0.0005 | -0.0032 | 否 |
| ACS | combined, \(\gamma=1/3\) | -0.0001 | -0.0015 | 否 |
| ACS | combined, \(\gamma=1/6\) | -0.0135 | -0.0115 | 是 |
| ACS | combined, \(\gamma=2/3\) | -0.0018 | -0.0027 | 否 |
| ACS | combined, \(\gamma=1/3\), `+Origin` | -0.0123 | -0.0186 | 否 |
| MMAS | final-only, \(\gamma=1/3\) | -0.0079 | — | 否 |
| MMAS | basin-only, \(\gamma=1/3\) | +0.0053 | -0.0602 | 否 |
| MMAS | combined, \(\gamma=1/3\) | +0.0053 | -0.0602 | 否 |
| MMAS | combined, \(\gamma=1/6\) | -0.0053 | -0.0432 | 是 |
| MMAS | combined, \(\gamma=2/3\) | -0.0013 | -0.1172 | 否 |
| MMAS | combined, \(\gamma=1/3\), `+Origin` | +0.0053 | -0.0602 | 否 |

## 4. 冻结规则

正式配置遵循同一规则：

1. 使用 combined fitness，保留 dense basin 信号和 final 目标。
2. 在不使 final gap 退化的候选中，选择能够改善 basin 的较小残差半径。
3. 不根据单 seed pilot 单独切换到 final-only fitness。
4. `Origin` 只保留为后续消融，不进入主配置。

最终冻结结果为：

| ACO | fitness | \(\gamma_T=\gamma_P\) | `Origin` | 训练迭代 |
|---|---|---:|:---:|---:|
| AS | paired combined UCB | \(1/3\) | 否 | 500 |
| ACS | paired combined UCB | \(1/6\) | 否 | 500 |
| MMAS | paired combined UCB | \(1/6\) | 否 | 500 |

`Origin` 通过了可用性门控。这说明 2-opt 确实引入了足够多的新边，且构造
差异不会全部穿过 2-opt。但是，三代 pilot 没有显示稳定的 final-gap
收益。另外，加入新 terminal 会改变随机初始化得到的表达式。因此，该
pilot 不能单独证明 `Origin` 的因果效果。正式的 terminal 消融必须使用
多个 GP seeds，并报告独立 validation 或 test 结果。

## 5. 结果边界

上述结果可以支持以下陈述：

- 2-opt 会显著压缩 GP 构造规则的可观察差异。
- final-only 指标在当前设置下具有较低信噪比。
- basin-level 指标能够恢复更密集的学习信号，尤其是在 AS 上。
- 过大的 residual 半径可能改善 basin 指标，同时损害 final gap。

上述结果尚不能支持“RMTGP-ACO+2-opt 显著优于 ACO+2-opt”的结论。该结论
必须由 3 个独立 GP runs、独立 validation gate 和 5000 次迭代最终测试
共同决定。
