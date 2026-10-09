# 实验产物入口

原始产物按证据等级和E00–E13登记。论文图表请从 [结果索引](../docs/results/README.md) 阅读。

- `runs/`：formal、pilot、diagnostic、smoke分开存储。
- `inputs/`：只读冻结输入；不是实验结果。
- `cache/`：共享可复算缓存。
- `provenance/`：环境、源码、历史资料和迁移前备份。
- `operations/`：运行管理和迁移日志。

## 暂存的旧路径

以下地址仍被后台任务或其消费者依赖。它们不是新的命名范例，不移动、不复制锁、不做软链接伪迁移。

- `artifacts/a5000-main-v1` → `artifacts/runs/pilot/E01/p01/gpu-baselines`：控制器正在写入，且等待TSP500的第25和50代cohort；不可拆分输入与绝对路径记录。
- `artifacts/pilot-v1` → `artifacts/runs/pilot/E09/p01/training`：TSP500训练未结束，已完成TSP100的cohort也仍被其他队列引用。
- `artifacts/hardware-pilot-v1` → `artifacts/runs/pilot/E12/p01/cross-gpu`：各卡仍在执行TSP500任务及后续训练和审计。
- `artifacts/baselines` → `artifacts/cache/aco-baselines`：正在运行的不可变源码继续读写旧路径。
- `artifacts/runtime` → `artifacts/provenance/source-snapshots`：存活进程仍从这些目录导入源码。
- `artifacts/launches` → `artifacts/operations/launches`：worker持续写入心跳和日志。
- `artifacts/locks` → `artifacts/operations/locks`：新旧worker必须继续共用同一锁文件；迁移需要所有旧worker退出。

详见 [当前盘点](../docs/results/status/storage_inventory.md) 和 [目录管理规范](../docs/design/08_artifact_and_result_management.md)。
