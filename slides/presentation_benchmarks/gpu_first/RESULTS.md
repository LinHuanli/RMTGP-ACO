# Presentation experiment results

本页是独立的 GPU 先行测量组。沿用原冻结源码、输入和预算，不等待 CPU 对照。
不与另一主机的 CPU 时间拼接计算加速比；原同机对照仍独立运行。

自动汇总完成且未检测到同卡外部进程干扰的任务。

当前有效测量行：66；功能检查结果：3。

## Matched trace results

| Backend | Repeats | Median evaluation (s) | Speedup vs CPU-8 |
|---|---:|---:|---:|
| GPU-v1 (FP32) | 3 | 133.8885 | pending matched reference |
| GPU-v2 generated / 8 lanes | 3 | 101.2323 | pending matched reference |

速度比仅使用同主机、同输入哈希、同实际 tasks/tours 的完整回放。
CPU 为 FP64；GPU 搜索为普通 FP32，返回 tour 后 CPU FP64 计分。
图表显示中位数与最小–最大范围；时间重复不是算法效果的独立种子。

## Real 5-generation runs

| Backend | Repeats | Median startup (s) | Median 5-generation wall (s) |
|---|---:|---:|---:|
| GPU-v2 generated / 8 lanes | 3 | 8.418 | 100.350 |

不足 3 次重复时只是初步数据；不据此宣称稳定加速倍率。

## Queue status

更新时间（UTC）：2026-09-15T12:14:20.572811+00:00

| Group | Status | Host | Current task |
|---|---|---|---|
| gpu-main | completed | cuda12 | — |
| gpu-scaling | completed | cuda12 | — |
| gpu-profile | completed | cuda12 | — |

## Available figures

- [C1_stage_time_v2](figures/C1_stage_time_v2.svg)
- [C3_matched_backends](figures/C3_matched_backends.svg)
- [C4_generation_stages_v2](figures/C4_generation_stages_v2.svg)
- [C5_matched_generations](figures/C5_matched_generations.svg)
- [C6_population](figures/C6_population.svg)
- [C7_cities](figures/C7_cities.svg)
- [C9_timeline_v1](figures/C9_timeline_v1.svg)
- [C9_timeline_v2](figures/C9_timeline_v2.svg)
- [D1_compute_hierarchy](figures/D1_compute_hierarchy.svg)
- [D2_deap_numba](figures/D2_deap_numba.svg)
- [D3_a5000_hardware](figures/D3_a5000_hardware.svg)
- [D4_sm_memory](figures/D4_sm_memory.svg)
- [D5_task_mapping](figures/D5_task_mapping.svg)
- [D6_generation_pipeline](figures/D6_generation_pipeline.svg)
- [D7_residency_timeline](figures/D7_residency_timeline.svg)
