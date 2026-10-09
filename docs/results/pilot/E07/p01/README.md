# E07／p01：固定线程映射与活跃状态先导

18个同卡配对block，每block随机顺序执行12种`lanes × active_tasks`计划，共216个完整预算评价。每种规模及cohort阶段独立分析。协议见 [映射补充](../../../../design/10_formal_baseline_and_mapping.md)。

原始数据：`artifacts/runs/pilot/E07/p01/fixed-mapping/`。不同计划保持P100、B32、A32、I500和K20；不在holdout选择用于正式训练的执行计划。

尚未完成的配对block保持缺失。infeasible、失败与争用单列，不当作零时间或无限加速，不把这组对照称为完整GPU-Opt或自动选择器。
