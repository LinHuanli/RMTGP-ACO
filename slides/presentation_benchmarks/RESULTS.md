# Presentation experiment results

自动汇总完成且未检测到同卡外部进程干扰的任务。

当前有效测量行：0；功能检查结果：0。

## Matched trace results

| Backend | Repeats | Median evaluation (s) | Speedup vs CPU-8 |
|---|---:|---:|---:|

速度比仅使用同主机、同输入哈希、同实际 tasks/tours 的完整回放。
CPU 为 FP64；GPU 搜索为普通 FP32，返回 tour 后 CPU FP64 计分。
图表显示中位数与最小–最大范围；时间重复不是算法效果的独立种子。

## Available figures

- [D1_compute_hierarchy](figures/D1_compute_hierarchy.svg)
- [D2_deap_numba](figures/D2_deap_numba.svg)
- [D3_a5000_hardware](figures/D3_a5000_hardware.svg)
- [D4_sm_memory](figures/D4_sm_memory.svg)
- [D5_task_mapping](figures/D5_task_mapping.svg)
- [D6_generation_pipeline](figures/D6_generation_pipeline.svg)
- [D7_residency_timeline](figures/D7_residency_timeline.svg)
