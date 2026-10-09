# 实验运行状态

最近队列核查：2026-10-10 06:46 NZDT（2026-10-09 17:46 UTC）。这是管理快照，不是实时状态，也不是正式论文结果。

## 已启动的有限队列

| 队列 | 范围 | 已完成 | 执行中 | 输入就绪、等待空卡 | 等待真实cohort |
|---|---|---:|---:|---:|---:|
| E01／p01 GPU基线 | 110项，包括2项输入准备、90项解释器/JIT配对、18项阶段诊断 | 92 | 0 | 0 | 18 |
| E01／p01 工作量诊断 | 2规模×3宿主×3种群阶段×3个ACO随机block，共54项 | 45 | 0 | 0 | 9 |
| E09／p01 正式GPU-Existing基线 | AS，两规模各10个独立根种子，共20次50代训练 | 0 | 16 | 4 | 0 |
| E07／p01 固定映射先导 | 18个同卡配对block，每block随机顺序测12种计划，共216个评价 | 0 | 0 | 15 | 3 |

上表为控制器写入的队列状态；扫描周期60秒，远程操作和共享文件系统可能增加可见性延迟。已完成不自动等于可发表样本；报告还检查结束标记、输入身份、争用及计时边界。45项诊断表示配对测量完成，不是45次独立GP训练。分类报告此前纳入的10项也不代表原始数据只有10项。

剩余依赖全部为TSP500、AS先导根种子1002的第50代真实种群，不能用第1代或第25代代替。其余独立任务已经派发，不继续空等这个依赖。

## 本轮修复及正式训练核查

已核验并恢复4个被控制器误记为失败的诊断：`tsp500-acs-g001-b000`、`tsp500-mmas-g001-b000`、`tsp500-as-g001-b001`、`tsp500-as-g001-b002`。各自原始结果、SHA、插桩逐位一致性和重放校验通过。原控制器在本机暂未看到完成标记而远端进程已退出时立即判失败，与共享文件系统可见性延迟相符。没有重新运行这4个科学样本，也没有删除原FAILED或日志；每项另有`STATE_RECONCILED.json`审计。

新控制器增加远端结束标记复查、至少180秒且多次观测的宽限期，以及晚到成功标记的内容核验。SSH未知状态不释放设备租约。真正worker失败与完成标记冲突时保持失败待复核，不能静默覆盖。升级只替换调度代码，旧科学worker源码和预算不变。

新增正式任务使用与先导不同的根种子2001–2010，固定P100、B32、A32、I500、K20、G50。16项执行中任务具体为：

- TSP500的10个根种子全部启动，正在预生成53个输入／ZERO基线场景。
- TSP100的6个根种子已完成输入准备并进入GP，最近读到第4–5代；另外4个根种子等待空卡。
- 对6个已进入GP的run，初始种群逐项等于冻结副本，运行等级为formal；所有已记录代均执行3200个任务，基线缓存全部命中。

输入准备与GP训练分别计时，并保留两者之和。16张新增卡各通过31项CUDA资格测试。此处只确认正式固定对照正常启动，不提前计算10-seed统计，也不宣布优化方法加速成立。协议及计时边界见[正式基线与映射补充](../../design/10_formal_baseline_and_mapping.md)。

## GPU及进程

本次新增使用16张空闲RTX A5000，每张卡只执行一个worker：

| 主机 | 本次使用的物理GPU序号 |
|---|---|
| cuda01 | 1 |
| cuda02 | 1 |
| cuda04 | 0、1 |
| cuda07 | 0、1、2 |
| cuda08 | 0、2 |
| cuda10 | 0、1、2 |
| cuda12 | 0、2 |
| cuda13 | 0、2 |

实际绑定使用UUID，序号只方便人工核对。进程独立nohup运行，扫描间隔60秒。启动前复核进程、显存、型号及UUID；不终止其他用户进程。06:46的`gpu-free`扫描显示没有空闲A5000。本项目共占用20张A5000，另有A40与L4各一张执行原先导。显存、编译和存储阶段可能出现瞬时0%利用率，不能据此判断worker已经退出；100% GPU利用率也不能解释为SM占用率或最优效率。

| 控制器 | PID（cuda-small1） | 科学worker源码commit |
|---|---:|---|
| 原GPU基线，保留科学版本 | 1207829 | `c5b93ac3950f828beab5ca67579f526adba8e2dd` |
| 工作量诊断，保留科学版本 | 1207811 | `93e253c7df4831c7873133535aadb17245eb8546` |
| 正式基线与映射补充 | 1207956 | `b963cbffcb1fe3828a83d5524d09be9bd77352d4` |

三个控制器均使用修复后的`b963cbf`不可变快照。原两个campaign的controller升级审计保存在各自`controller-revisions/`中，不能把controller版本误当科学worker版本。

原训练与跨卡worker没有中断。最近读取的TSP100先导三个根种子均完成50代；TSP500根种子1001／1002／1003分别记录39／38／39代。跨卡先导中，L40S、PRO5000已写入完成标记并退出；A5000、A40、L4仍处于TSP500 holdout。非A5000已完成的先导不重复执行来填满显卡，也不将不同型号的计时塞进A5000正式样本。

## 目录与核查入口

- 基线队列：`artifacts/runs/pilot/E01/p01/gpu-baselines/{queue.json,dispatcher_status.json}`。
- 诊断队列：`artifacts/operations/diagnostic-dispatch/p01/{queue.json,dispatcher_status.json,dispatcher.log}`。
- 独立诊断attempt：`artifacts/runs/diagnostic/E01/p01/work-diagnostics/`；每项含job、日志、measurement和结束标记。
- 每卡资格检查：`artifacts/operations/diagnostic-dispatch/p01/qualifications/<GPU-UUID>/`。
- 正式／映射调度：`artifacts/operations/research-dispatch/p01/{queue.json,dispatcher_status.json,dispatcher.log}`。
- 正式训练attempt：`artifacts/runs/formal/E09/p01/gpu-existing/<run>-aNN/`；`training/history.json`是每代数据，输入准备不混进该曲线。
- 正式冻结输入：`artifacts/inputs/formal-training/p01/<run>-aNN/`；READY后只读。
- 映射配对attempt：`artifacts/runs/pilot/E07/p01/fixed-mapping/<block>-aNN/`。
- 正文图表入口：[分类结果索引](../README.md)。目录规模和延期理由：[存储盘点](storage_inventory.md)。

基线原目录已物理迁移，SHA核验通过，只保留旧地址兼容入口。训练、跨卡实验、共享缓存、旧源码、启动记录和锁共6个旧目录仍被存活进程引用，暂不移动。没有删除实验数据。

## 本轮没有启动的部分

CPU Python／Numba各1、8、16核代码已准备；完整CPU计时未在这些GPU主机上启动。CPU-Opt、GPU-Tensor、M1／M2／M3消融仍有实现和验证前置条件，不能通过重复基线替代。正式10-seed固定GPU对照已启动，但不是完整确认性方法对比；标准测试集未开启。

本次后台清单有限：旧队列继续完成依赖任务，新队列20次正式基线与18个映射block自动接续。不能无限扩充重复数来占卡，也不表示E00–E13全部完成。
