# 空闲 A5000 持续队列与主实验执行次序

日期：2026-10-10。目的：按用户授权，持续调用 `gpu-free`，利用所有确认空闲的 RTX A5000 推进当前已具备实现与基础验证条件的主实验。不改动正在运行的训练，不抢占其他用户进程。

## 1. 当前可执行范围

原 TSP100 的3个50代AS训练已经完成。TSP500的3个50代训练继续运行。其科学预算和不可变快照不变。

本次自动队列补齐 E01 的 **GPU-Existing主基线部分**，以及 E04 中已有的解释器/树JIT路径对比和三宿主执行迁移。它不意味着E00–E13全部已经实现。

| 维度 | 冻结设置 |
|---|---|
| GPU | 仅 NVIDIA RTX A5000；不依据可用性混入其他型号 |
| 规模 | TSP100、TSP500 |
| 执行宿主 | AS、同步ACS、MMAS |
| cohort | AS pilot seed1002 的第1、25、50代完整100程序；分别来自对应n的训练 |
| 主工作点 | P100、B32、R1、ants32、I500、K20；FP32、无局部搜索 |
| 实例 | 性能holdout的前32个独立实例；不读取标准测试 |
| 配对 | 同一物理GPU上的现有树JIT与字节码解释器；顺序按预先冻结命名空间随机化 |
| 固定计划 | lanes8、active上限3200；不在本队列用holdout挑配置 |
| 重复 | 每个n×cohort阶段×宿主5个配对block；不同block可分配到不同A5000 |
| 诊断 | 每个cell额外一次独立construct/update事件插桩；与正式计时分开 |

总计2个输入准备任务、90个配对block（180次完整评估）和18个独立诊断任务，共110个调度任务。诊断必须等该cell的第一个配对block完成后再启动。5个性能block不是5个GP训练种子；跨框架重放AS学到的规则，不冒充三种宿主分别训练的结果。

优先完成TSP100就绪cell。TSP500已有的第1代可以独立运行；第25/50代cohort出现后自动解锁。缺少cohort时不复制早期程序填充，也不占用GPU等待。

## 2. 统一输入和计量

每个n的输入准备任务在明确分配的一张A5000主机上执行一次。FP32距离、启发值、log启发值、完整排名、候选表、标签长度和实例key只计算一次，再为三个宿主保存逐位相同的几何数组。不同宿主/随机重复所需初始化参数分别冻结。每个文件登记SHA256；消费者只读mmap。此队列不训练，因此不需要另跑ZERO适应度对照缓存。

同一block的程序、几何、初始化、随机流和科学参数在解释/JIT两端完全一致。所有100×32任务实际执行，结果缓存关闭。编译/I=1预热单列，暖评估仍使用I=500。保存完整最终路径与长度，报告逐位一致性，而非只比较平均gap。

CUDA事件、eval wall、显存pool、NVML整卡采样峰值和能耗分开记录。不同block可能使用不同CPU主机和不同A5000物理卡；配对block作为统计单元，保存具体硬件身份。先导bootstrap区间是当前分配条件下的探索性结果，不把它当作GPU型号总体的硬件抽样置信区间。

## 3. 持续调度与安全条件

调度器每60秒调用一次 `gpu-free` 并保存原始输出。只识别 `IDLE` 且型号准确为RTX A5000的行；之后通过SSH再次查询UUID、GPU利用率、显存和全部compute进程。空闲条件为：无compute进程、利用率≤5%、显存≤1024MiB。设备索引只用于解析候选，实际运行按UUID唯一绑定。

控制器持有项目内单实例锁；启动前先持久化任务租约。worker持有与原训练/跨卡实验共用的GPU锁，并第三次复查空闲。每张卡同时只派一个任务。不同宿主/编译模式不跨GPU拆开同一配对block。

每个源码快照×物理GPU先通过基础CUDA检查。检测到新的空闲卡后自动加入，不限定最初的服务器列表，不更改驱动功耗、ECC或时钟设置。

失败与重试规则：

- 启动前锁竞争或设备被占用：保存 `REJECTED.json`，尚无科学样本，可重新排队；同卡重复拒绝有限次后改用其他空卡。
- 实际计算失败/进程意外退出：保存原始记录与 `FAILED.json`，不静默重跑、不覆盖、不降低预算。
- SSH不可达、启动确认丢失：保留租约，等待进程身份核查；不因心跳暂时缺失启动第二份任务。
- 运行中其他用户加入：记录争用，污染样本不进入干净性能配对；不触碰其他用户进程。
- 所有已登记任务完成或明确失败后停止调度；失败/依赖失败不写成全部成功。

## 4. 操作与文件

```bash
source scripts/env.sh
python scripts/a5000_pool.py                 # 查看冻结任务规模，不启动
python scripts/a5000_pool.py --execute       # 首次创建队列，并nohup启动持续控制器
python scripts/a5000_pool.py --execute --resume  # 仅控制器已退出时显式恢复
python scripts/report_a5000_main.py --campaign artifacts/a5000-main-v1
```

控制器和worker均来自项目内已提交的不可变快照。所有缓存、扫描输出、结果、图表都保存在本项目内。`STOP_DISPATCH`文件存在时仅停止新派发，已开始的任务继续运行；再次恢复需明确撤销该文件。

```text
artifacts/a5000-main-v1/
  campaign.json, queue.json
  dispatcher.log, dispatcher_status.json, dispatcher_events.jsonl
  scans/                             # gpu-free原始快照
  qualifications/<gpu-uuid>/         # 每卡基础E00记录
  inputs/tsp<n>/<variant>/           # 统一几何/初始化/SHA清单
  tasks/<task-id>/attempt-<id>/       # 不覆盖失败或拒绝的attempt
    job.json, STARTED.json, worker.log, hardware.json
    cohort.json, manifest.json, telemetry.jsonl
    measurements/{generated,interpreted}/
  summary/                           # CSV/JSON/Markdown/PNG/PDF
```

## 5. 尚未自动解锁的研究部分

CPU-Opt、GPU-Tensor、特征—树联合优化M1、CPU执行优化M2、完整GPU计划优化M3及其消融仍需实现和对应E00验证。正式10-seed训练、等时间预算和最终标准测试必须等比较方法及协议冻结后再启动；不以现有基线多跑几次代替尚未实现的方法。

因此，本队列的完成应表述为“当前主GPU基线与诊断前置部分完成”，不是“所有主实验完成”。可用GPU只加速已授权的有限任务集，不自动无限增加统计重复或新增研究问题。
