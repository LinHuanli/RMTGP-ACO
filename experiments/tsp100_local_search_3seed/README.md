# TSP100 局部搜索与深树实验

本实验固定 `32 ants × 500 ACO iterations`、GP population 100 和 50 代。
每代使用 64 个纯 TSP100 训练实例。两棵树共享 62 个节点预算；单棵树最多
62 个节点，最大深度 7，初始深度 2--5。

正式任务为 `3 ACO variants × 2 training environments × 3 GP seeds`：

- `none`：训练阶段不做局部搜索；
- `two_opt`：构造后对全部蚂蚁执行候选表 2-opt；
- 两组 checkpoint 最终都在同一 `two_opt / 500 iterations` validation
  环境中重新选择，避免把 checkpoint 选择规则混入“训练时是否见过 LS”的
  对比。

AS、ACS 和 MMAS 均固定 32 只蚂蚁。其余 ACOTSP-LS 参数分别为：
AS `rho=0.5`；ACS `rho=0.1, q0=0.98, xi=0.1`；MMAS `rho=0.2`。
局部搜索使用 20-nearest-neighbour candidate list、DLB 和 first improvement。
信息素树增加强类型 `LSGain: PhField[B,R,n]`。它把来源路线经局部搜索得到
的相对长度改善从 `[B,R]` 广播到来源路线的全部边。关闭局部搜索时固定为
`-1`。两组训练使用相同 grammar。

最终测试只使用 TSP100-uniform 128 个实例和 TSP500-uniform 32 个实例。
每个实例运行 3 个 ACO seeds 和 5000 iterations。对照为 ACO+2opt、
ACO+3opt、无 LS 训练后接 2-opt 的深树模型，以及 2-opt 联合训练模型。

`scripts/run_local_search_campaign.py` 负责可恢复任务队列。
`scripts/select_local_search_checkpoints.py` 负责统一 2-opt validation 选择。
`scripts/evaluate_local_search_final.py` 只加载最终选择的个体并执行 5000 代测试。

CUDA 调优固定为 2-opt 每 block 8 个 warps，3-opt 每条路线一个
512-thread block。短基准显示，8 warps 相对 4 warps 的 2-opt 内核快约
4.5%。3-opt block 并行相对旧的一 warp/tour 内核在 TSP100 和 TSP500 上
分别约快 1.9 倍和 3.5 倍。速度数字是实现选择依据，不是最终实验结果。
原始数值和六个一代短跑记录见 `benchmark_summary.json`。
