# Presentation experiment results

自动汇总完成且未检测到同卡外部进程干扰的任务。

当前有效测量行：5；功能检查结果：6。

## Matched trace results

| Backend | Repeats | Median evaluation (s) | Speedup vs CPU-8 |
|---|---:|---:|---:|

速度比仅使用同主机、同输入哈希、同实际 tasks/tours 的完整回放。
CPU 为 FP64；GPU 搜索为普通 FP32，返回 tour 后 CPU FP64 计分。
图表显示中位数与最小–最大范围；时间重复不是算法效果的独立种子。

## Real 5-generation runs

| Backend | Repeats | Median startup (s) | Median 5-generation wall (s) |
|---|---:|---:|---:|
| GPU-v2 generated / 8 lanes | 1 | 10.307 | 91.552 |

不足 3 次重复时只是初步数据；不据此宣称稳定加速倍率。

## Queue status

更新时间（UTC）：2026-09-15T10:08:45.286459+00:00

| Group | Status | Host | Current task |
|---|---|---|---|
| main | running | cuda04 | E1-cpu8-r0 |
| gpu-setup | completed | cuda08 | — |
| scans-prepare | pending | — | — |
| jit | pending | — | — |
| scaling | pending | — | — |
| ablation-g3 | pending | — | — |
| ablation-g5 | pending | — | — |
| profile-v1 | pending | — | — |
| profile-v2 | pending | — | — |

## Available figures

- [C1_stage_time_v2](figures/C1_stage_time_v2.svg)
- [C4_generation_stages_v2](figures/C4_generation_stages_v2.svg)
- [D1_compute_hierarchy](figures/D1_compute_hierarchy.svg)
- [D2_deap_numba](figures/D2_deap_numba.svg)
- [D3_a5000_hardware](figures/D3_a5000_hardware.svg)
- [D4_sm_memory](figures/D4_sm_memory.svg)
- [D5_task_mapping](figures/D5_task_mapping.svg)
- [D6_generation_pipeline](figures/D6_generation_pipeline.svg)
- [D7_residency_timeline](figures/D7_residency_timeline.svg)
