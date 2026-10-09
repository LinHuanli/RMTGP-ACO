# 产物盘点与迁移状态

盘点时间（UTC）：2026-10-09T17:48:47.957058+00:00。大小是逻辑文件字节，不是磁盘占用；活跃目录仍会变化。

本表的状态是**存储状态**，不是实验完成状态。原始记录不会因目录整理升级为正式结果。

| 登记ID | 类别 | 当前地址 | 文件数 | MiB | 存储状态 |
|---|---|---|---:|---:|---|
| E09-p01-formal-gpu-existing | formal | `artifacts/runs/formal/E09/p01/gpu-existing` | 1840 | 247.14 | canonical |
| shared-formal-training-inputs-p01 | input | `artifacts/inputs/formal-training/p01` | 7029 | 28615.67 | canonical |
| E07-p01-fixed-mapping | pilot | 尚未创建 | 0 | 0.00 | not_created |
| shared-research-dispatch-p01 | operations | `artifacts/operations/research-dispatch/p01` | 31 | 0.10 | canonical |
| E00-p01-initial-validation | smoke | `artifacts/runs/smoke/E00/p01/initial-validation` | 24 | 8.29 | canonical |
| E00-p01-instrumentation-checks | smoke | `artifacts/runs/smoke/E00/p01/instrumentation-checks` | 34 | 13.13 | canonical |
| E01-p01-initial-stage-profile | diagnostic | `artifacts/runs/diagnostic/E01/p01/initial-stage-profile` | 5 | 1.53 | canonical |
| E01-p01-gpu-baselines | pilot | `artifacts/runs/pilot/E01/p01/gpu-baselines` | 1973 | 934.95 | canonical_with_legacy_alias |
| E01-p01-cpu-baselines | pilot | 尚未创建 | 0 | 0.00 | not_created |
| E01-p01-work-diagnostics | diagnostic | `artifacts/runs/diagnostic/E01/p01/work-diagnostics` | 726 | 231.25 | canonical |
| E09-p01-training | pilot | `artifacts/pilot-v1` | 68 | 3.31 | legacy_deferred |
| E12-p01-cross-gpu | pilot | `artifacts/hardware-pilot-v1` | 1397 | 3078.90 | legacy_deferred |
| shared-frozen-population-p01 | input | `artifacts/inputs/frozen-population/p01` | 169 | 336.99 | canonical |
| shared-aco-cache | cache | `artifacts/baselines` | 294 | 12.21 | legacy_deferred |
| shared-source-snapshots | provenance | `artifacts/runtime` | 117 | 0.68 | legacy_deferred |
| shared-managed-source-snapshots | provenance | `artifacts/provenance/source-snapshots/managed` | 147 | 0.93 | canonical |
| shared-diagnostic-dispatch | operations | `artifacts/operations/diagnostic-dispatch/p01` | 188 | 1.67 | canonical |
| shared-diagnostic-cohorts-p01 | input | `artifacts/inputs/diagnostic-cohorts/p01` | 81 | 323.12 | canonical |
| shared-bootstrap-records | provenance | `artifacts/provenance/bootstrap/p01` | 417 | 201.70 | canonical |
| shared-environment-records | provenance | 尚未创建 | 0 | 0.00 | not_created |
| shared-launch-records | operations | `artifacts/launches` | 42 | 1.87 | legacy_deferred |
| shared-device-locks | operations | `artifacts/locks` | 25 | 0.00 | legacy_deferred |
| E02-p01-historical-inputs | historical | `artifacts/provenance/historical/E02/p01` | 9 | 0.88 | canonical |
| shared-pre-layout-report | provenance | `artifacts/provenance/report-layout-migration/p01` | 52 | 4.81 | canonical |

## 暂缓迁移的原因

- `E09-p01-training`：TSP500训练未结束，已完成TSP100的cohort也仍被其他队列引用。
- `E12-p01-cross-gpu`：各卡仍在执行TSP500任务及后续训练和审计。
- `shared-aco-cache`：正在运行的不可变源码继续读写旧路径。
- `shared-source-snapshots`：存活进程仍从这些目录导入源码。
- `shared-launch-records`：worker持续写入心跳和日志。
- `shared-device-locks`：新旧worker必须继续共用同一锁文件；迁移需要所有旧worker退出。

## 已迁移且保留旧地址兼容入口

- `artifacts/a5000-main-v1` 是指向 `artifacts/runs/pilot/E01/p01/gpu-baselines` 的兼容软链接；数据只在新目录中保存一份。

登记源：`configs/experiments/registry.yaml`。迁移逐文件SHA日志在 `artifacts/operations/migrations/`。

复查命令：`python scripts/manage_artifacts.py inventory --write`。新实验启动后应再次生成盘点。
