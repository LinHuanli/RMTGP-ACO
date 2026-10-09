# E09／p01：GPU-Existing正式基线

预登记20次AS连续训练：TSP100／500各10个根种子2001–2010。协议见 [正式基线前置执行](../../../../design/10_formal_baseline_and_mapping.md)。

这只是完整方法比较中的固定对照。优化CPU／GPU方法未完成，标准测试未开启，不据此宣布端到端加速或最终质量结论。

原始数据：`artifacts/runs/formal/E09/p01/gpu-existing/`。每个attempt包含job、硬件、遥测、输入预生成成本以及training子目录的逐代曲线、checkpoint、冠军和完成标记。运行管理见`artifacts/operations/research-dispatch/p01/`。

所有预登记seed须交代成功、失败、争用和缺失。未完成时不提前计算正式10-seed均值或用先导数据补齐。分类表图由报告脚本从完成标记与校验后的原始数据生成。
