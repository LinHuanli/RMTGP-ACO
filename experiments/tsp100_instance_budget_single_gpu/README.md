# TSP100 每代训练实例预算实验

本实验比较 ACS 在单卡 `cuda_tiled_v2` 后端下，每代使用 32、64 和 128 个
训练实例时的计算时间与泛化质量。每个配置均使用 population 100、50 GP
generations、32 ants 和 500 ACO iterations。

三个预算共享 GP root seeds 2001、2002、2003。对每个 seed 和 generation，
32-instance 集合是 64-instance 集合的前缀，64-instance 集合是
128-instance 集合的前缀。三者使用相同的初始 GP population、ACO seed、
validation selection、validation gate 和 validation ACO seeds。

每个训练进程只看见一张 GPU。两张物理 GPU 仅用于同时运行两个独立的单卡
replicate，不用于一个 run 内的 task sharding。

结果写入 `runs/tsp100-instance-budget-single-gpu`。训练效果以冻结
validation 的最终候选 gap、相对 baseline 的 delta、非劣门和 CPU FP64
audit 为准。训练 batch 上的每代最优 gap 仅用于描述优化轨迹。

训练完成后，`scripts/evaluate_instance_budget_parallel.py` 只加载九个
`selected_candidate.pkl`。它把一个原始 ACO baseline 与九个最终候选组成
10-program batch，并使用 128-instance batch 和确定性双 GPU LPT 分片。
测试不会重新评测 GP population。三个训练预算共享每个
partition×instance×ACO-seed 的 baseline 和随机流。
