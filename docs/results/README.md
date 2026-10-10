# 实验结果总索引

本目录只保存分类报告、可复算的小表和图。原始数据、日志及checkpoint在 `artifacts/`。**当前没有完成的正式确认性结果**；完整预算不等于正式重复充分。

| 证据等级 | 报告 | 当前用途 |
|---|---|---|
| 正式 `formal` | [正式结果状态](formal/README.md) | TSP100固定GPU对照10次已齐；TSP500进行中；完整方法对比未完成 |
| 先导 `pilot` | [E01／p01 基线](pilot/E01/p01/README.md) | ACO质量基线、GPU无插桩性能、CPU六组覆盖 |
| 先导 `pilot` | [E07／p01 固定映射](pilot/E07/p01/README.md) | 候选lane与活跃状态上限的同卡配对，不在holdout选择计划 |
| 诊断 `diagnostic` | [E01／p01 瓶颈证据](diagnostic/E01/p01/README.md) | 阶段、fallback、逻辑工作量与资源；不是正式加速比 |
| 历史 `historical` | [E02／p01 历史协议](historical/E02/p01/README.md) | 旧双树FP64实验，只作单独复核 |
| 功能验证 `smoke` | [验证记录边界](smoke/README.md) | 说明验证位置；不合并成科研结果 |
| 管理状态 `status` | [产物盘点和迁移状态](status/storage_inventory.md) | 路径、用途、规模和迁移例外，不等于实验完成状态 |

后台队列、GPU分配和最近一次核查见 [运行状态](status/execution_status.md)。这是带时间戳的管理快照，不是实时仪表盘或科研结果。

## 运行中任务的位置

- E09／p01：AS两规模×3根种子×50代。原始运行仍在 `artifacts/pilot-v1`。
- E09／p01 正式固定对照：AS两规模×10根种子×50代，在 `artifacts/runs/formal/E09/p01/gpu-existing`；仅GPU-Existing控制组，不含未实现的优化方法。
- E07／p01 固定映射先导：18个同卡配对block，在 `artifacts/runs/pilot/E07/p01/fixed-mapping`；与正式训练共用有限优先级调度 `artifacts/operations/research-dispatch/p01`。
- E12／p01：五类GPU的调优、留出与短训练。原始运行仍在 `artifacts/hardware-pilot-v1`。
- E01／p01：A5000冻结cohort基线队列。原始运行已迁至 `artifacts/runs/pilot/E01/p01/gpu-baselines`；`artifacts/a5000-main-v1`只是兼容旧manifest的软链接，不是第二份数据。
- E01／p01：完整预算的工作量诊断队列。原始记录在 `artifacts/runs/diagnostic/E01/p01/work-diagnostics`，调度状态在 `artifacts/operations/diagnostic-dispatch/p01`。矩阵和解释边界见 [诊断队列协议](../design/09_diagnostic_campaign.md)。

这些目录的增量 `summary/`是运行期视图，不是正式发表结果。读取其状态时必须注明核查时间。旧目录仍被不可变worker或下游cohort引用时不迁移；不改动历史manifest。

## 按研究问题阅读

[RQ分析总览](analysis/p01/README.md)将现有证据组织为六个问题。每个问题都有对应表/图和结论边界。

- RQ1瓶颈：[阶段与细粒度诊断](diagnostic/E01/p01/README.md)。
- RQ2已有执行收益：[解释器/JIT配对](pilot/E04/p01/README.md)；CPU1/8/16核完整数据仍缺失。
- RQ3任务映射：[固定lanes×active热图](pilot/E07/p01/README.md)。
- RQ4规模效应：[逐transition成本与回退](pilot/E01/p01/README.md)。
- RQ5训练：[正式固定对照](formal/E09/p01/README.md)、[独立3-seed先导](pilot/E09/p01/README.md)。
- RQ6迁移：[五类GPU留出比较](pilot/E12/p01/README.md)。

2-opt/3-opt新增结果按p02分类；实验清单和门禁见[扩展协议](../design/11_local_search_extension.md)。

## 溯源规则

每份报告固定含 `README.md`、`tables/`、`figures/`、`provenance.json`。先导、插桩、缩小预算、旧协议数据不能混成一组统计。缺失值保持缺失；失败、争用和排除记录保留。正式报告需预先冻结协议并通过发布检查，不能通过重命名生成。

命名和迁移细则见 [产物管理规范](../design/08_artifact_and_result_management.md)。登记表为 `configs/experiments/registry.yaml`。

历史启动记录移至 `status/snapshots/`，只记录当时情况，不作为实时进度。
