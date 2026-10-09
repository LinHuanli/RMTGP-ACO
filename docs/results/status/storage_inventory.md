# 产物盘点与迁移状态

盘点时间（UTC）：2026-10-09T14:28:00.786363+00:00。大小是逻辑文件字节，不是磁盘占用；活跃目录仍会变化。

本表的状态是**存储状态**，不是实验完成状态。原始记录不会因目录整理升级为正式结果。

| 登记ID | 类别 | 当前地址 | 文件数 | MiB | 存储状态 |
|---|---|---|---:|---:|---|
| E00-p01-initial-validation | smoke | `artifacts/runs/smoke/E00/p01/initial-validation` | 24 | 8.29 | canonical |
| E00-p01-instrumentation-checks | smoke | `artifacts/runs/smoke/E00/p01/instrumentation-checks` | 34 | 13.13 | canonical |
| E01-p01-initial-stage-profile | diagnostic | `artifacts/runs/diagnostic/E01/p01/initial-stage-profile` | 5 | 1.53 | canonical |
| E01-p01-gpu-baselines | pilot | `artifacts/a5000-main-v1` | 1470 | 714.41 | legacy_deferred |
| E01-p01-cpu-baselines | pilot | 尚未创建 | 0 | 0.00 | not_created |
| E01-p01-work-diagnostics | diagnostic | 尚未创建 | 0 | 0.00 | not_created |
| E09-p01-training | pilot | `artifacts/pilot-v1` | 65 | 2.56 | legacy_deferred |
| E12-p01-cross-gpu | pilot | `artifacts/hardware-pilot-v1` | 1208 | 2762.19 | legacy_deferred |
| shared-frozen-population-p01 | input | `artifacts/inputs/frozen-population/p01` | 169 | 336.99 | canonical |
| shared-aco-cache | cache | `artifacts/baselines` | 227 | 8.07 | legacy_deferred |
| shared-source-snapshots | provenance | `artifacts/runtime` | 117 | 0.68 | legacy_deferred |
| shared-bootstrap-records | provenance | `artifacts/provenance/bootstrap/p01` | 417 | 201.70 | canonical |
| shared-environment-records | provenance | 尚未创建 | 0 | 0.00 | not_created |
| shared-launch-records | operations | `artifacts/launches` | 42 | 1.10 | legacy_deferred |
| shared-device-locks | operations | `artifacts/locks` | 20 | 0.00 | legacy_deferred |
| E02-p01-historical-inputs | historical | `artifacts/provenance/historical/E02/p01` | 9 | 0.88 | canonical |
| shared-pre-layout-report | provenance | `artifacts/provenance/report-layout-migration/p01` | 52 | 4.81 | canonical |

## 暂缓迁移的原因

- `E01-p01-gpu-baselines`：控制器正在写入，且等待TSP500的第25和50代cohort；不可拆分输入与绝对路径记录。
- `E09-p01-training`：TSP500训练未结束，已完成TSP100的cohort也仍被其他队列引用。
- `E12-p01-cross-gpu`：各卡仍在执行TSP500任务及后续训练和审计。
- `shared-aco-cache`：正在运行的不可变源码继续读写旧路径。
- `shared-source-snapshots`：存活进程仍从这些目录导入源码。
- `shared-launch-records`：worker持续写入心跳和日志。
- `shared-device-locks`：新旧worker必须继续共用同一锁文件；迁移需要所有旧worker退出。

登记源：`configs/experiments/registry.yaml`。迁移逐文件SHA日志在 `artifacts/operations/migrations/`。

复查命令：`python scripts/manage_artifacts.py inventory --write`。新实验启动后应再次生成盘点。
