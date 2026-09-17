# MMAS + LS 机制实验：实现和运行记录

## 1. 范围与当前状态

本目录对应 `../MMAS_LS_GPU_ablation_plan.md`。工作分支为
`experiment/mmas-ls-mechanism-v1`。本轮实现针对固定冠军的 P0–P4；不启动 P5 重新训练。

截至 2026-09-17，`artifacts/numerical-v1/` 的 **98/98 个任务已完成**。
本轮是固定冠军的开发集数值对照和历史实现机制分析，不是确认性结论，也不是整个 P0–P5 计划完成。

统计报告入口为 [独立中文报告](reports/numerical-v1/report_zh.md)。
表格、PDF/PNG、完整性校验和分析版本清单放在同一目录。
只有 `report_manifest.json` 标为 complete 时，才表示报告所依赖的核验与生成步骤全部完成。

| 本轮项目 | 规模与状态 |
|---|---|
| N0 数值验收 | 3 种 GPU × AS/MMAS × 3 种统计模式，18/18 完成；稳定模式通过输入 oracle；legacy 保留已知偏差 |
| N1 数值质量对照 | 32 个 dev 实例 × 5 seeds × 2 框架 × baseline/3 冠军 × 3 模式；40/40 配对任务完成 |
| P1 历史 R/F/H 八格 | 32 个 dev 实例 × 5 seeds × 8 条件 × baseline/3 冠军；40/40 完成 |
| 统计单位 | 32 个实例；ACO seeds 和三个固定冠军不作为独立实例 |
| 诊断记录 | schema 3 全量逐轮窗口和固定探针；原始记录不改写 |
| 尚未完成的科学范围 | 独立确认、完整 P3 分叉/回放、P4 反向验证与稳健性、P5 重新训练；本次不追加运行 |

以下为早期实现记录，不代表当前队列仍在等待或运行。

后续完整诊断实现与验收记录见 [DIAGNOSTICS_V3.md](DIAGNOSTICS_V3.md)。
旧 v2 的 42 个 GPU P0 工单现已完成。旧 light 的缺失项不作追溯填补。
新的 `mechanism_v3` 使用独立队列与完整性门禁；本文第 6 节描述的是旧记录，不能代替新版本说明。

| 项目 | 当前状态 |
|---|---|
| R/F/H、来源预算、重启回放、来源/尺度/时间 terminal 开关 | 已实现 CUDA 控制接口 |
| CPU FP64 oracle | 已实现 R/F/H、固定来源、top-k、尺度控制；不支持回放及全部 terminal 控制 |
| 原生数值等价 | 六个真实冠军、两个 baseline、2 个 TSP500、300 轮；同卡旧/新源码路径、长度、anytime、diagnostics 全部一致 |
| 审计开关和分块 RNG | off/light/heavy、实例重排和 task chunk 检查通过 |
| 历史数据审计 | 1,697 份 JSON、971 份 NPZ、132 个有 manifest 的训练运行；冻结的 32/128 Uniform 实例未发现重叠 |
| P0 GPU 精度审计 | 24 个小样本任务已完成，仍需合并 CPU FP64 与分析组件效应差 |
| 历史现象复现 | 18 个任务后台运行；保留旧 instance IDs、seed、raw 冠军和连续距离 |
| CPU FP64 精度审计 | 后台运行，8 个实例 × 3 seeds × 4 条件 × baseline/3 冠军，300 轮 |
| P1 | 40 个任务已冻结，等待 P0 验收 |
| P2 | 128 × 10 × 8 × 4 = 40,960 个逻辑求解；未开放确认集结果 |
| OOD 数据 | 新 Cluster/Gaussian 各 128 个实例的独立 LKH reference 正在生成 |
| P3/P4 | 已提供机制工单生成器；重型快照/分叉入口需完整验收后加入正式队列 |

`artifacts/v2/` 是早期运行目录，当前完整分析批次是 `artifacts/numerical-v1/`。
`artifacts/v1/` 中的队列试运行已作废且保留。
原因是共享盘的 `flock` 未提供跨主机互斥。现在使用原子 `mkdir` 锁；
cuda02 成功领取时，cuda03 的同时领取返回失败。没有复用 v1 的队列结果。
独立完成的原生验收和数据准备未受影响，已保留其记录。

## 2. 固定科学配置

- TSP500，32 ants，5,000 次 ACO iteration，candidate/LS candidate 均为 20。
- 2-opt 使用现有 ACOTSP/DLB CUDA 实现。MMAS `rho=0.2`，AS `rho=0.5`。
- TR/PH 残差系数均为 `1/3`；精度主配置是 `fp32_fast`。
- 每个算法固定三个 `selected_candidate.pkl`，GP seeds 为 81001、81002、81003。
- 使用未回退的 raw 冠军进行归因。部署 gate 和 baseline fallback 另行解释。
- MMAS 的 R 是原重启写操作，F 是物理下界，H 是原历史路径来源日程。
  关闭 F 不改变标称 tau 范围或 TauHeadroom 定义。
- LS 原始分支没有硬上界裁剪。新增 `hard_upper_clip` 是新算法敏感性分析。
- H=0 只取消历史路径作为强化边集；仍保存 global-best，并保留影子来源预算。
- MMAS 同状态目标预算是原生影子来源的 `n/L`；AS 是当前蚂蚁的 `sum(n/L_a)`。
  top-k 按 `1/L` 分配来源预算，再在每个来源内部归一化 GP 边权。
- SourceQuality、Origin、LSGain 始终属于实际来源，不属于预算参考路径。

所有字段见 `protocol.yaml` 和 `src/rmtgp_aco/mechanisms.py`。不修改历史训练 YAML。

## 3. 数据与 reference

Uniform 优先从已有 64k training pool 留出未用实例。先排除历史 train/val/test/诊断中
出现的坐标哈希及 instance ID，再排除全部旧 validation/test 文件。32 个实例供开发；
另 128 个供确认。缺失 TSP500 训练日程或无法读取历史文件时，审计失败，不静默继续。
这里的审计覆盖现存仓库记录，不能证明被删除的记录或仓库外从未使用过这些实例。

历史复现保留旧 ID，否则 counter RNG 不同。新实验以坐标哈希作为稳定身份。
所有新条件、冠军和执行设备共享同一个由 `split + replicate` 确定的 ACO seed。
历史使用过的数据不加入新确认性样本。

旧 OOD 池不能提供各 128 个可证明未用的实例，因此生成完整的新分布：

- Cluster：5 个等权中心，中心服从 `[0.1,0.9]^2` 上的 Uniform，标准差 0.05；越界点重采样。
- Gaussian：均值 `(0.5,0.5)`，协方差 `0.2² I`；越界点重采样。

这两个明确的新分布不冒充旧数据未知的生成器。使用 LKH-3.0.13，10 runs、
每 run 最多 10,000 trials、距离比例 1e6。独立 reference tour 在 CPU FP64 下按
连续欧氏距离重算。它是可行参考解，不声明最优性。负 reference gap 原样保留。
独立求解器来源、源码压缩包 SHA256、可执行文件 SHA256、参数和完整日志均保留在 artifacts。

## 4. 指标和统计

`reference_gap = 100 × (L / L_ref − 1)`。
同条件净差 `Delta = GP_gap − baseline_gap`，负值表示 GP 更好。

每个条件只保存一次 baseline，不把它复制成三个独立样本。先在实例内平均 ACO seeds
与三个固定冠军，再按实例重采样。正式 18 项比较包含 3 个移除效应、3 个效应差、
3 个二阶交互、1 个三阶交互和 8 个净差。使用 30,000 次 paired bootstrap max-t，
固定 bootstrap seed，实际意义阈值为 0.01 pp。

`statistics.py` 生成全八格、绝对变化分解、同时区间、每冠军区间、AUC、win/tie/loss、
尾部指标及 PDF/PNG。正式分析要求冻结的全部结果完整且哈希一致，不按可用子样本出确认结论。
零方差与相关零效应模拟测试已通过；bootstrap 区间仍是近似推断，不是有限样本覆盖率证明。

## 5. 调度、复现与缓存

根据用户后续授权，`campaign.py` 使用 `gpu-free` 当前空闲的 **RTX A5000、RTX 4000 Ada、RTX A4000**。每卡一个 worker，CUDA 只看到
这一张卡的 UUID。远程启动前重新检查进程、显存和型号；运行中检查后来进入的其他进程。
如果发生资源冲突，只停止本 worker 的子进程，不操作其他用户进程。

`watch.py` 是 nohup 后台资源监控，每 60 秒扫描一次，自动为已放行、未领取的任务增加
worker。SSH 或发现失败后继续重试；同一设备启动尝试冷却 300 秒。没有可运行任务时
仍监控，但不占用 GPU，也不跳过验收门禁。停止监控可创建 `artifacts/v2/WATCH_STOP`，
该标记不停止已有 worker；`STOP` 会同时停止监控和本 cohort 的 GPU worker。
通过原子目录锁保证仅一个监控进程；遗留锁需核查进程后处理，不自动抢占。

扩充硬件不改变冻结的模型、随机流或迭代预算。资源授权单独记录于
`protocol/resource_amendment.json`。保留每个任务的型号、UUID、驱动和 kernel 哈希；
不同型号的耗时不直接混合解释为算法加速，数值一致性也不能仅凭相同精度名称假定。

每个任务包含同条件 baseline 和全部固定冠军。GP individual × instance 维度使用原 GPU
task matrix。蚂蚁及 2-opt 使用现有 CUDA 并行。不同任务分配到不同 GPU。
任务级并行不要求 DDP，也不在 GPU 之间同步信息素。

源码按 SHA256 复制到不可变快照，后台任务不读取随后编辑的求解器文件。结果保存源码、
生成 kernel、checkpoint、输入、reference、协议和完整参数哈希。常规求解失败最多重试 3 次，
不自动改变科学参数。资源暂停不消耗数值重试次数。

跨主机任务锁使用原子目录。进程崩溃后可能留下锁；必须核实 owner 和子任务已经退出，
才能移走锁。禁止仅因运行时间长而抢占锁。`STOP` 文件在任务检查点暂停当前 cohort。
缓存或已完成结果若科学配置变化，不应覆盖；改用新的 cohort。

## 6. 历史 light 记录及后续修订

本节保留早期 light 的限制，用于解释旧产物。当前 numerical-v1 的 schema 3 已记录下述
窗口矩、PH 饱和、信息素分布和真实 TR 探针。字段定义见 DIAGNOSTICS_V3.md。
数学正确性、完整性与因果识别是三个不同问题；补齐日志不自动证明机制。

目前 light 保存每轮 26 个 GPU 标量和完整 anytime 曲线，压缩写入 NPZ。它包含来源、
来源年龄、预算误差、下/上界计数、重启、branch factor、pre/post-LS 质量、deposit CV 和
来源与 IB/GB 的路径哈希相同标记。不是每轮保存完整信息素矩阵。
AS native 的 `source_kind=0, source_count=32` 表示当前种群多来源，并非 iteration-best；
路径相同标记只在 `source_count=1` 时有效。不能仅按 source_kind 汇总 AS 为 IB 强化。

**不能把当前记录称为计划中全部轻量诊断已经完成。** 尚需补齐 GPU 窗口约简、PH raw
输出/饱和度、候选 tau 分位数和真实构造概率熵。`aggregate_every` 目前只是记录接口参数，
尚未实施设备端 25 轮聚合。当前 light 耗时仅为小样本验收，不是稳健的部署性能结论。

重型入口 `probes.py` 保存 post-LS 与 update 后的完整可恢复状态，支持 baseline/冠军
作为 owner 和 baseline/TR-only/PH-only/full 分叉。原生快照恢复测试已通过；完整的
8 实例 × 3 seeds × 8 条件重型工单尚未运行。当前 PH 探针的 TV 是基础
`tau^alpha × eta^beta` 核在固定 visited mask 下的局部敏感性，不是完整 GP TR 的概率 TV。

P2 启动前仍须完成：历史现象审阅、FP64 组件效应审阅、强制触发的 R/F 和历史元数据
验收、P3 扩展冻结、完善的轻量诊断及开销评测。精度差若接近 0.01 pp，不自动放行。
已有旧 final-test NPZ 没有 tour，因此不能声称与旧文件逐路径复现；同环境旧源码对照
另有 tour，可以逐路径比较。这两种验收不可混为一谈。

## 7. 当前报告与核验入口

本轮报告只使用开发集，不访问确认集，不启动 GPU 求解。
默认使用 4 个 CPU worker；共享盘校验较慢时可显式调整 `--workers`。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m control_experiments.mmas_ls.research_report --workers 4

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
  control_experiments/mmas_ls/tests/test_research_report.py \
  control_experiments/mmas_ls/tests/test_statistics.py
```

入口默认读取 `artifacts/numerical-v1`，输出到 `reports/numerical-v1`。
`--phase quality` 仅生成质量统计与图，不宣告完整诊断报告完成。
派生缓存绑定输入身份和分析代码；文件大小、mtime 或 ctime 变化时重新核验。
原始冻结快照、队列、已完成结果和旧 gate 保持不变。

### 早期队列操作入口（默认指向 v2）

在仓库根目录使用当前 `.venv`：

```bash
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.campaign status
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.watch status
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.review
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.statistics --split diagnosis_dev
```

启动持续资源监控：

```bash
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.watch start --interval 60
```

创建新 cohort 时先 prepare、数据审计、原生验收，再 freeze/launch。已有 cohort 的
冻结队列不能用修改后的源码静默重写。P0/P1 审阅通过后，分别写入带证据和理由的
`gates/P0.json`、`gates/P1.json`；状态值为 `approved`。此操作是研究验收，不因任务进程
结束自动执行。P2/P3/P4 还要求 `protocol/extensions_frozen.json` 存在。

GPU 和 CPU 测试分别执行；GPU 测试只能指定经过核查的空闲 GPU。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q \
  control_experiments/mmas_ls/tests/test_controls.py \
  control_experiments/mmas_ls/tests/test_numba_controls.py \
  control_experiments/mmas_ls/tests/test_statistics.py \
  control_experiments/mmas_ls/tests/test_artifacts.py
```

大型输入、快照、结果与第三方 reference solver 均位于忽略目录，不提交 Git。
算法改动、实验入口、测试、协议和此说明纳入版本管理。
