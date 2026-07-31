# TSP500 full-2opt LS-aware v2

本实验不覆盖历史 `tsp500_2opt_racing` 合同。它使用新的 run root
`runs/tsp500-2opt-ls-v2`。

训练只使用 TSP500。每代冻结 128 个实例和 3 个 ACO seeds。多保真评价
使用以下前缀：

| 代数 | Screen | High |
|---|---:|---:|
| 1--15 | 16 instances × 1 seed × 50 iterations | 32 × 1 × 100 |
| 16--35 | 32 × 1 × 100 | 64 × 2 × 200 |
| 36--50 | 64 × 1 × 200 | 128 × 3 × 500 |

Screen fitness 是 paired basin mean。High fitness 是 final 与 anytime
各占 0.5 的 paired mean。育种不加入标准误惩罚。风险控制只用于独立
validation selection/gate。

GP 使用三个阶段：

1. 第 1--10 代只进化 pheromone tree，transition residual 为零；
2. 第 11--30 代注入并冻结对应 ACO variant 的 no-LS transition
   champion；
3. 第 31--50 代联合微调，修改 pheromone/transition tree 的概率为
   0.7/0.3。

两棵树合计最多 62 个有效节点。同一双树结构最多保留 2 个副本。
新增 transition terminals 为 `MutualRank`、`TurnCos`。新增 pheromone
terminals 为 `Origin`、逐边 `LSGain`、`PreFreq`、`PostFreq` 和
`TauHeadroom`。

CUDA 训练把 individual × instance 铺平成 task 矩阵。2-opt 的逐边 Gain
使用 `[task, ant, city, 2]` packed 邻接矩阵。矩阵驻留共享内存。每个
accepted move 只更新四个端点。`PreFreq/PostFreq` 使用
`[task, n, n]` byte matrix 和并行 scatter。普通程序与需要 LS 状态的程序
按语义分片，避免整个 population 承担重路径开销。

Gain 不使用稠密 `[task, ant, n, n]` 矩阵。Gain 只定义在 tour 的
\(n\) 条边上。若在上述基准规模中使用 FP32 稠密矩阵，仅 Gain 状态就
需要约 62 GiB。紧凑邻接矩阵把空间复杂度从 \(O(n^2)\) 降为 \(O(n)\)。
TSP500、每 block 8 条 tour 时，共享内存开销为 32 KiB。2-opt 的 20 个
候选边由一个 warp 并行计算。3-opt 的 \(20^2\) 个候选对由一个
512-thread block 并行计算；tour、position、order、DLB 与 scratch 均驻留
共享内存。

在 RTX PRO 5000 Blackwell 上的隔离基准使用 65 个语义不同的程序、
32 个 TSP500 实例和 10 次 ACO iteration。各程序的残差系数设为
\(10^{-9}\)，因此四组运行具有完全相同的 2-opt move 数。普通
`EdgeEta` 路径用时 1.602 s。矩阵化 `PreFreq` 和 `PostFreq` 分别用时
1.606 s 和 1.613 s，额外成本为 0.26% 和 0.72%。矩阵化逐边 `LSGain`
用时 2.217 s。它比旧的逐边循环实现快 25.64 倍。`LSGain` 仍比轻路径
多 38.44% 的状态维护成本，因此训练时继续采用语义分片，而不让不使用
该 terminal 的个体承担此成本。

训练完成后，最终测试只加载每个正式 run 的 `selected_candidate.pkl`：

```bash
.venv/bin/python scripts/evaluate_tsp500_ls_v2_final.py
```

该命令在三个 TSP500 test partition 上分别测试前 32 个实例。每个
variant 使用 3 个 ACO seeds 和 5000 次 iteration。对照方法为
ACO+2-opt 与 ACO+3-opt。最终个体不会以 population 形式重复计入统计。
