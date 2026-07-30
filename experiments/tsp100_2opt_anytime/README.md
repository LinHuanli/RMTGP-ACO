# TSP100 Anytime+Final matched control

该矩阵只把 TSP100 v2 正式训练的 fitness 改为
`paired_final_anytime_ucb`。种群、树容量、terminal/function set、ACO
预算、训练实例数、validation 协议和三个 GP seeds 均保持不变。

AS 的残差半径为 1/3。ACS 和 MMAS 为 1/6。每个 variant 运行三个独立
GP seeds。该矩阵用于区分“改进来自 fitness”与“改进来自 TSP500 训练
规模”。运行入口为 `scripts/run_tsp100_anytime_control.py`。
