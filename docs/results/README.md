# 实验结果总索引

本目录只保存分类报告、可复算的小表和图。原始数据、日志及checkpoint在 `artifacts/`。**当前没有完成的正式确认性结果**；完整预算不等于正式重复充分。

| 证据等级 | 报告 | 当前用途 |
|---|---|---|
| 正式 `formal` | [正式结果状态](formal/README.md) | 尚未开始，不用先导结果填充 |
| 先导 `pilot` | [E01／p01 基线](pilot/E01/p01/README.md) | ACO质量基线、GPU无插桩性能、CPU六组覆盖 |
| 诊断 `diagnostic` | [E01／p01 瓶颈证据](diagnostic/E01/p01/README.md) | 阶段、fallback、逻辑工作量与资源；不是正式加速比 |
| 历史 `historical` | [E02／p01 历史协议](historical/E02/p01/README.md) | 旧双树FP64实验，只作单独复核 |
| 功能验证 `smoke` | [验证记录边界](smoke/README.md) | 说明验证位置；不合并成科研结果 |
| 管理状态 `status` | [产物盘点和迁移状态](status/storage_inventory.md) | 路径、用途、规模和迁移例外，不等于实验完成状态 |

## 仍在旧路径运行的先导任务

- E09／p01：AS两规模×3根种子×50代。原始运行仍在 `artifacts/pilot-v1`。
- E12／p01：五类GPU的调优、留出与短训练。原始运行仍在 `artifacts/hardware-pilot-v1`。
- E01／p01：A5000冻结cohort基线队列。原始运行仍在 `artifacts/a5000-main-v1`。

这些目录的增量 `summary/`是运行期视图，不是正式发表结果。读取其状态时必须注明核查时间。已登记为待迁移；不在运行时改动路径和历史manifest。

## 溯源规则

每份报告固定含 `README.md`、`tables/`、`figures/`、`provenance.json`。先导、插桩、缩小预算、旧协议数据不能混成一组统计。缺失值保持缺失；失败、争用和排除记录保留。正式报告需预先冻结协议并通过发布检查，不能通过重命名生成。

命名和迁移细则见 [产物管理规范](../design/08_artifact_and_result_management.md)。登记表为 `configs/experiments/registry.yaml`。

历史启动记录移至 `status/snapshots/`，只记录当时情况，不作为实时进度。
