# 实验运行状态

最近队列核查：2026-10-10 04:00 NZDT（2026-10-09 15:00 UTC）。这是管理快照，不是实时状态，也不是正式论文结果。

## 已启动的有限队列

| 队列 | 范围 | 已完成 | 执行中 | 输入就绪、等待空卡 | 等待真实cohort |
|---|---|---:|---:|---:|---:|
| E01／p01 GPU基线 | 110项，包括2项输入准备、90项解释器/JIT配对、18项阶段诊断 | 74 | 0 | 0 | 36 |
| E01／p01 工作量诊断 | 2规模×3宿主×3种群阶段×3个ACO随机block，共54项 | 10 | 11 | 15 | 18 |

上表为控制器写入的队列状态，最多有一个扫描周期的延迟。已完成不自动等于可发表样本；报告还检查结束标记、输入身份、争用及计时边界。新诊断首批9项均完成，未检测到同卡争用，插桩与无插桩的最终路径及长度逐位相等，真实状态重放校验通过。它们是TSP100第1代、AS／同步ACS／MMAS各3个ACO随机block，不是9个独立GP训练seed。最新分类报告已纳入10项完整预算诊断，包含一项后期种群诊断。

新队列已验证完成后自动接续：首批6项结束后，6张卡已获得下一项任务。TSP500第25／50代必须等待AS根种子1002的实际训练产物，不能拿第1代替代。

## GPU及进程

本次新增使用11张空闲RTX A5000，每张卡只执行一个测量worker：

| 主机 | 本次使用的物理GPU序号 |
|---|---|
| cuda01 | 1 |
| cuda02 | 1 |
| cuda04 | 0、1 |
| cuda08 | 0、2 |
| cuda10 | 0、1、2 |
| cuda13 | 0、2 |

每张卡均通过31项基础CUDA测试。实际绑定使用UUID，序号只方便人工核对。进程独立nohup运行，扫描间隔60秒。启动前复核进程、显存、型号及UUID；不终止其他用户进程。最新采样曾显示11张卡均为100% GPU利用率、无其他同卡进程；此值不能解释为SM占用率或最优内核效率。

| 控制器 | PID（cuda-small1） | 不可变源码commit |
|---|---:|---|
| 原GPU基线，迁移后恢复 | 1116956 | `c5b93ac3950f828beab5ca67579f526adba8e2dd` |
| 新工作量诊断 | 1120488 | `93e253c7df4831c7873133535aadb17245eb8546` |

原有8个计算worker未中断，远程进程身份已核对：3个TSP500长训练，另5个为A5000、A40、L40S、L4、PRO5000跨卡先导。03:56 NZDT的历史核查中，TSP100三个根种子均完成50代；TSP500三个根种子分别记录21、18、19代；跨卡TSP500阶段分别为tuning、tuning、holdout、tuning、training。

## 目录与核查入口

- 基线队列：`artifacts/runs/pilot/E01/p01/gpu-baselines/{queue.json,dispatcher_status.json}`。
- 诊断队列：`artifacts/operations/diagnostic-dispatch/p01/{queue.json,dispatcher_status.json,dispatcher.log}`。
- 独立诊断attempt：`artifacts/runs/diagnostic/E01/p01/work-diagnostics/`；每项含job、日志、measurement和结束标记。
- 每卡资格检查：`artifacts/operations/diagnostic-dispatch/p01/qualifications/<GPU-UUID>/`。
- 正文图表入口：[分类结果索引](../README.md)。目录规模和延期理由：[存储盘点](storage_inventory.md)。

基线原目录已物理迁移，SHA核验通过，只保留旧地址兼容入口。训练、跨卡实验、共享缓存、旧源码、启动记录和锁共6个旧目录仍被存活进程引用，暂不移动。没有删除实验数据。

## 本轮没有启动的部分

CPU Python／Numba各1、8、16核代码已准备；完整CPU计时未在这些GPU主机上启动。CPU-Opt、GPU-Tensor、M1／M2／M3消融和正式10-seed训练尚有实现或协议前置条件，不能通过重复基线替代。标准测试集未开启。

本次启动表示“当前输入就绪且实现已验证的GPU基线／诊断已进入后台队列”，不表示E00–E13全部完成。
