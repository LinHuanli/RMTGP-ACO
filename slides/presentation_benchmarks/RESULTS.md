# Presentation experiment results

自动汇总完成且未检测到同卡外部进程干扰的任务。

当前有效测量行：163；功能检查结果：6。

## Matched trace results

| Backend | Repeats | Median evaluation (s) | Speedup vs CPU-8 |
|---|---:|---:|---:|
| CPU-1 (FP64) | 3 | 14002.8823 | 0.14× |
| CPU-8 (FP64) | 3 | 1939.9309 | 1.00× |
| GPU-v1 (FP32) | 3 | 132.6446 | 14.64× |
| GPU-v2 generated / 8 lanes | 3 | 93.0572 | 20.85× |

速度比仅使用同主机、同输入哈希、同实际 tasks/tours 的完整回放。
CPU 为 FP64；GPU 搜索为普通 FP32，返回 tour 后 CPU FP64 计分。
图表显示中位数与最小–最大范围；时间重复不是算法效果的独立种子。

## Real 5-generation runs

| Backend | Repeats | Median startup (s) | Median 5-generation wall (s) |
|---|---:|---:|---:|
| CPU-8 (FP64) | 3 | 42.218 | 2191.471 |
| GPU-v2 generated / 8 lanes | 3 | 10.307 | 91.552 |

不足 3 次重复时只是初步数据；不据此宣称稳定加速倍率。

## Queue status

更新时间（UTC）：2026-09-16T07:35:10.307233+00:00

| Group | Status | Host | Current task |
|---|---|---|---|
| main | completed | cuda04 | — |
| gpu-setup | completed | cuda08 | — |
| scans-prepare | completed | cuda08 | — |
| jit | completed | cuda08 | — |
| scaling | failed | cuda02 | — |
| ablation-g3 | completed | cuda01 | — |
| ablation-g5 | completed | cuda08 | — |
| profile-v1 | completed | cuda01 | — |
| profile-v2 | completed | cuda08 | — |

## GPU-first queue

[GPU 先行结果与图表](gpu_first/RESULTS.md)独立汇总，不受 CPU 队列进度限制。

| Group | Status | Host / GPU | Current task |
|---|---|---|---|
| gpu-main | completed | cuda12 / 0 | — |
| gpu-scaling | completed | cuda12 / 1 | — |
| gpu-profile | completed | cuda12 / 0 | — |

## Available figures

- [C1_stage_time_cpu8](figures/C1_stage_time_cpu8.svg)
- [C1_stage_time_v2](figures/C1_stage_time_v2.svg)
- [C2_jit_amortization](figures/C2_jit_amortization.svg)
- [C3_matched_backends](figures/C3_matched_backends.svg)
- [C4_generation_stages_cpu8](figures/C4_generation_stages_cpu8.svg)
- [C4_generation_stages_v2](figures/C4_generation_stages_v2.svg)
- [C5_matched_generations](figures/C5_matched_generations.svg)
- [C6_population](figures/C6_population.svg)
- [C7_cities](figures/C7_cities.svg)
- [C8_ablations_g3](figures/C8_ablations_g3.svg)
- [C8_ablations_g5](figures/C8_ablations_g5.svg)
- [D1_compute_hierarchy](figures/D1_compute_hierarchy.svg)
- [D2_deap_numba](figures/D2_deap_numba.svg)
- [D3_a5000_hardware](figures/D3_a5000_hardware.svg)
- [D4_sm_memory](figures/D4_sm_memory.svg)
- [D5_task_mapping](figures/D5_task_mapping.svg)
- [D6_generation_pipeline](figures/D6_generation_pipeline.svg)
- [D7_residency_timeline](figures/D7_residency_timeline.svg)
