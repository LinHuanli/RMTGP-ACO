# E01完整预算诊断的持续运行协议

登记配置：`configs/workloads/diagnostic_queue.yaml`。证据等级为diagnostic，不是正式性能或训练质量结果。

## 工作矩阵

| 轴 | 设定 |
|---|---|
| TSP规模 | 100、500 |
| 执行宿主 | AS、同步ACS、MMAS |
| 冻结种群阶段 | 第1、25、50代 |
| 随机重复 | 3个ACO随机block，取原输入block 0、1、2 |
| 科学预算 | P100、B32、R1、ants32、ACO迭代500、K20、FP32 |
| 执行配置 | 单A5000、生成式GP、8候选lane；没有多任务共享同一GPU |
| 单次诊断 | 同一输入先无插桩，再插桩；最终路径和长度逐位核验 |

共54项。启动核查时，TSP100三个阶段和TSP500第1代共36项输入就绪。TSP500第25／50代对应18项依赖真实训练产物。三宿主使用的是AS、GP根种子1002产生的同一冻结种群，不是三个宿主分别重新训练。三个block也不是三个独立GP训练seed。

## 指标与边界

保留源码逻辑工作量、fallback扫描／排名工作、候选数、GP节点×候选数、lane-0诊断周期、CUDA事件时间线、快照重放、编译资源和NVML采样。没有权限的硬件计数器保持缺失。单个固定顺序的无插桩／插桩比值用于说明观测扰动，不当作优化加速结论。

缩小到3迭代的旧smoke不进入本队列。代码、预算、输入和原始结果均记录身份。标准测试集不打开，正式10-seed训练不自动解锁。

## 后台运行

```bash
source scripts/env.sh
python scripts/diagnostic_pool.py             # 只读查看数量，不启动
python scripts/diagnostic_pool.py --execute   # 首次创建，要求源码已提交
# 控制器确实退出且协议未变时才用 --execute --resume；不重复启动
```

控制器及每个worker均使用nohup、独立进程会话和不可变源码快照。控制器每60秒扫描空闲A5000。启动前重新确认GPU型号、UUID、进程与显存；共享旧队列的同一UUID锁。锁位于NFS挂载上，系统配置为本机锁；同一物理GPU的所有worker均在所属同一主机取得锁，不能将其解释为通用跨主机分布式锁。

设备启动时已占用只记REJECTED，可以在新attempt重新分配。科学计算后的异常保留FAILED，不自动覆盖或重复采样。检测到争用的完成记录标为excluded_contended。失联但不能确认worker退出时保留租约，不发第二份任务。

后期种群生成后导出到独立的只读输入包，不修改已有冻结输入。输入缺失是依赖等待，不能用第1代种群替代第25／50代。

## 存储位置

- 控制器、清单、扫描与派发事件：`artifacts/operations/diagnostic-dispatch/p01/`。
- 新源码快照：`artifacts/provenance/source-snapshots/managed/<commit>/`。
- 每个独立attempt：`artifacts/runs/diagnostic/E01/p01/work-diagnostics/tsp100-as-g001-b000-a01/`。
- 追加的后期输入：`artifacts/inputs/diagnostic-cohorts/p01/`。
- 分类报告：`docs/results/diagnostic/E01/p01/`，仅纳入完成且未检测到争用的诊断。

原74项完成、36项依赖等待的A5000基线队列仍使用原commit及原协议。它与新诊断是两个有限队列，不重复计算已完成的解释器/JIT配对。目录迁移只暂停了没有测量worker运行的控制器；所有训练和跨卡计算保持连续执行。
