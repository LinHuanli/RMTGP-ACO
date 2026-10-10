# E01／p01：ACO、CPU与GPU基线先导

证据等级：`pilot`；实验：E01；协议：p01。

本报告只包含先导测量。5个性能block不是5个GP训练seed。没有CPU时间时不计算GPU/CPU加速比。

## 普通 ACO 质量基线

| TSP | 宿主 | 数据范围 | 实例×随机种子记录数 | 平均 gap/% |
|---|---|---|---:|---:|
| 100 | acs | performance_holdout_32 | 160 | 1.7939 |
| 100 | as | performance_holdout_32 | 160 | 5.9125 |
| 100 | as | train | 288 | 5.8219 |
| 100 | as | validation | 1152 | 5.7882 |
| 100 | mmas | performance_holdout_32 | 160 | 4.0549 |
| 500 | acs | performance_holdout_32 | 160 | 16.1061 |
| 500 | as | performance_holdout_32 | 160 | 23.0333 |
| 500 | as | train | 288 | 22.9634 |
| 500 | as | validation | 1152 | 22.7936 |
| 500 | mmas | performance_holdout_32 | 128 | 20.4799 |

均为32只蚂蚁、500次迭代、无局部搜索。gap 相对于同一 FP32 最优标签路径长度。ZERO重复个体已合并，不是独立重复。训练场景、验证与性能holdout不混合平均；标准测试未开启。

## CPU 覆盖

| 后端 | 物理核 | 当前协议 |
|---|---:|---|
| cpu_python | 1 | not_measured |
| cpu_python | 8 | not_measured |
| cpu_python | 16 | not_measured |
| cpu_existing | 1 | not_measured |
| cpu_existing | 8 | not_measured |
| cpu_existing | 16 | not_measured |

历史CPU不在本报告；见 `docs/results/historical/E02/p01`。未采集的六组CPU计时不能用历史数据补齐。

## 当前证据边界

- 已汇总 174 条无插桩GPU测量。插桩记录另存诊断报告，不作为速度分母。
- 图中的程序规模、fallback与时间关系是描述性关联，不是因果结论。
- 显存为NVML采样的整卡值；内存池保留量单列。能耗是整卡外层评价区间，不是整机能耗。
- 现有compile字段是预热评价中的compile/load调用成本，不是真冷编译。

## 图表和原始表

- [tsp100_energy](figures/tsp100_energy.svg)
- [tsp100_work_and_resources](figures/tsp100_work_and_resources.svg)
- [tsp500_energy](figures/tsp500_energy.svg)
- [tsp500_work_and_resources](figures/tsp500_work_and_resources.svg)

CSV/JSON在 `tables/`；PDF/SVG/PNG在 `figures/`。来源、筛选规则和文件校验见 `provenance.json`。

正式结果与本报告分开；未采集值不填零。
