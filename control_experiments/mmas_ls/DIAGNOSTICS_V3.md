# MMAS–LS 机制诊断记录：schema 3

## 1. 范围

新增 `mechanism_v3` profile。原 `off/light/heavy` 接口仍然可用，默认保持 `legacy`。
旧 light 只有 26 个逐轮标量，不能当作 schema 3。旧 P0 只支持原有复现和工程问题。
不从旧曲线推造缺失的路径、terminal 或信息素状态。

科学设置不变：TSP500、32 ants、5000 轮、原冠军、原实例和随机种子。
LSGain 只作为现有实现的可观测元数据，不添加新的训练 terminal 或重新训练。
维护逐边元数据与选择 LSGain 的科学定义分开，不因打开审计切换 tour/edge 语义。

## 2. 采样与信息来源

记 T 为本次执行分块的求解数，N 为城市数，S 为来源槽数（最多 32）。
实际执行模型与逻辑冠军通过 `behavior_alias` 对应。别名不是独立样本。

| 信息 | 时点与 shape | 定义与用途 |
|---|---|---|
| 质量、来源、预算、重启与界限 | 每轮，`[T,W,26]` | 原 trace，W 不超过 100 |
| 来源身份 | 每轮，`[T,W,S,2] uint64` | 无向边双校验码；采样路径另用规范边表 SHA256 核对 |
| 来源元数据 | 每轮，`[T,W,S,6]` | kind、蚂蚁编号、来源长度、生成轮次、LSGain、实际来源预算 |
| PH 全边窗口矩 | 每轮，`[T,W,S,6] float64` | 边数、饱和边数、raw 和 tanh 的和及平方和 |
| LS 质量与结构 | 每轮，`[T,W,32,3]` | pre/post 长度、保留的构造边数 |
| LS 工作量 | 每轮，`[T,W,32,4] uint64` | 接受移动数、检查数、按 LS tolerance 改善标记、扫描 passes |
| 重启前后记忆 | 每轮，`[T,W,6] float64`，前后各一组 | RB 长度/生成轮次、epoch、RB gain、stagnation、GB 生成轮次 |
| TR 探针 | 第 1 轮及每 25 轮，`[T,16,N,22]` | 4 ants × 4 steps；仅可行候选有效 |
| TR 上下文 | 同上，`[T,16,10]` 及 visited bitset | 当前/前一城市、选中城市、分支、候选数、完整 visited |
| PH 全来源边 | 同上，`[T,S,N,18]` | 12 terminals、protected raw、tanh、残差倍率、归一化前量、来源归一化系数、基础量 |
| 最终 deposit | 同上，`[T,S,N]` | 已包括干预总预算校正；不能只看未归一化倍率 |
| 来源及 colony 路径 | 同上 | 实际来源、全部 pre/post-LS tours、GB；保留完整路径以核对结构统计 |
| 信息素分布 | 同上，四阶段，每阶段全部非对角有向边 | 更新前、蒸发及 floor 后、沉积后、重启及可选硬上界后 |
| 固定几何输入 | 每分块一次 | coords、distance、log heuristic、nearest、rank、node mean、实例 RNG key |

所有 NPZ 的字段 dtype 和实际 shape 同时写入索引。`schema.json` 提供字段定义、抽样总体和缺失规则。
浮点缺失为 NaN。整数标识缺失为 -1，来源槽由 `source_valid` 明确标记。缺失不是零。

### TR 输入

TR 的强类型是 `TrField`。16 个内部 terminal 槽为：

`RTau, REta, BaseConf, DistRank, Entropy, ConstructProg, ACOProg, Stagnation,
Tau, Distance, MeanTau, MeanDistance, Size, FeasibleCount, MutualRank, TurnCos`。

其中包含兼容旧 GP terminal 的槽，不表示本次冠军使用全部槽。
输入来自当前信息素、几何、可行候选、构造位置、迭代位置及停滞时长。
固定蚂蚁为 0、8、16、24。TSP500 的固定步为 1、125、250、375。
记录候选顺序、baseline score、GP 后实际 score、实际分支概率及 stream 3 随机值。
seed、实例 key、iteration、ant、step、stream 构成可复核的随机流坐标。

### PH 输入

PH 的强类型是 `PhField`。12 个内部 terminal 槽为：

`EdgeEta, EdgeTau, NNRank, ColonyFreq, SourceQuality, ACOProg, Stagnation,
LSGain, Origin, TauHeadroom, PreFreq, PostFreq`。

输入包括实际来源边、来源长度、当前 colony 长度、当前信息素、几何和 pre/post-LS 边频率。
Origin、LSGain 属于来源生成时的路径，不应重新绑定到当前 iteration-best。
SourceQuality 则以该实际来源长度与当前 colony 比较。
ColonyFreq 与 PostFreq 在当前实现中相同，不能按两个独立机制解释。

baseline 同样计算影子 terminal。`required_masks` 和 `program_active` 区分“可观测值”与“程序实际使用”。
raw 是经过现有数值保护的 GP 输出，不是未保护的中间表达式。

## 3. 不能混淆的定义

- 候选集耗尽后，当前实现执行全体未访问城市上的贪心选择，不是均匀抽样。
  新计数分别保存候选耗尽、baseline 退化、GP score 退化、实际均匀抽样。
  不修改旧计数或算法行为来迁就字段名称。
- AS native 的多来源是 32 只当前蚂蚁，不是 32 个 iteration-best。
- MMAS–LS 先对候选有向弧蒸发及施加 floor，再沉积。native LS 没有硬上界裁剪。
- 总沉积预算对 tour 无向边计一次。对称写入稠密矩阵后，矩阵总量是其两倍。
- PH 饱和定义为 `abs(tanh(raw)) >= 0.99`。每 25 轮窗口饱和率来自全部轮次和全部实际来源边。
  末点的 quantile、来源重合率及 TR 探针均明确标记 endpoint/probe，不能称作全窗口总体。
- 相同长度不表示相同路径。相同 post-LS 边集也不能单独证明相同吸引域。

## 4. 保存与恢复

GPU 使用 100 轮环形缓冲。每 100 轮提交日志。传输使用 pinned host memory，压缩和写盘在线程中执行。
最多保留两个待写块，避免无界内存增长。反压发生在提交/采样边界，不在每个构造步骤同步 CPU。

后续 I/O 修订将相同几何矩阵改为每任务保存一份，各执行分块只保存映射和文件引用。
NPZ 使用 DEFLATE level 1 无损压缩；不改变 dtype、数值或有效采样。
原压缩版和 I/O 修订版使用不同不可变快照，不覆盖已完成的计时结果。

每 500 轮保存恢复检查点。同一分块保留最新两个恢复副本。预指定永久快照不删除。
索引包含文件 SHA256、数组内容摘要、shape、dtype 和覆盖区间。先写数据，再原子提交索引。
损坏、缺字段、重复区间、缺轮次或重放不一致均报错。失败的采样放入 quarantine，不计入成功记录。

恢复校验科学配置和执行分块。已有提交块按内容核对，不重复计数。
恢复包括曲线前缀、来源/重启状态、审计环、LS 元数据和绝对迭代位置。
随机流使用原 counter 坐标，不按恢复时间重新抽 seed。

空间不足时停止并报告，不自动降低采样密度。性能报告分列 CUDA 阶段、传输和后台 NPZ 写入。
阶段可能重叠，不直接相加；update 内的审计目前与 update 一起计时。
不把 population 批量 wall time 除以冠军数，冒充无审计单模型部署时间。

## 5. 重型子集与分叉

固定 8 个 dev 实例 × 3 seeds × 8 条件 × baseline/3 冠军，共 768 个逻辑求解。
heavy 在 1、100、250、500、1000、2500 轮保存 post-LS 和 iteration-end 完整状态。
每个重启检查点预留 post-LS 状态，只在实际重启时提交该状态和 iteration-end 状态。
这两个文件准确标注为包围重启的阶段；kernel 内的沉积后 tau 和重启前记忆另有记录。
**目前不把 post-LS 快照称为精确 pre-restart 可恢复快照。**

`diagnostic_forks.py` 从永久状态运行 baseline/TR-only/PH-only/full 和单项 R/F/H 翻转。
每分支只续跑一次 500 轮，读取 1、25、100、500 轮结果，保留原 horizon 分母。
baseline owner 与三冠军分别配对；冠军 owner 只与自身配对。完全相同行为的分支记录别名。

同 post-LS 状态比较 TR-only 与 full，可得到零 PH 与真实 PH 的沉积差：

\[
 AD=\frac{\sum_{e\in E}|D_{\mathrm{PH}}(e)-D_0(e)|}{B}.
\]

E 为规范无向边集合，B 为共同预算。另记录四阶段信息素相对 L1/L2 差异。
固定 visited 上下文下同时计算基础核和完整 GP TR 的 TV。
当前分叉 TV 是 CPU FP64 参考计算，字段明确带 `cpu_fp64`；不是 GPU bitwise 实测概率。
GPU 实际概率由构造时的 TR 日志提供。两者不可混写。

LS 结构比率为 post 差异除以 pre 差异。pre 差异为零时比率未定义，并保留未定义比例。
续跑后的后续轮次已包含历史反馈，不能自动解释为单次 LS 的纯中介作用。

## 6. 报告与验收状态

`diagnostic_analysis.py` 输出：

- 每 25 轮窗口表：PH 矩与饱和率、来源切换/连续重复、LS 改善、退化分支、界限/重启计数。
- 末点表：来源年龄/唯一边集/与 IB、GB 重合，归一化后的沉积分布、同来源预算的 AD、真实 TR 熵及 max-p。
- 重启事件表：执行与 would-trigger、branch factor、停滞时长、25/100/500 轮后的质量变化；越过 horizon 标为缺失。
- 六面板机制趋势 PDF/PNG。正式八格比较及 simultaneous CI 仍由原统计模块生成。

这些描述性图不能单独证明因果关系。机制结论需要控制干预、配对统计和相应验收。

已在 RTX 4000 Ada 上通过小规模原生等价、两种阶段恢复、500 轮磁盘检查点续跑、强制 uniform/floor/replay、历史来源元数据及 v3 分叉入口检查。
TSP500 的 2 实例 × 25 轮试跑中，六个冠军及两个 baseline 的路径和 anytime 与关闭审计一致。
CPU GP 输出重算最大误差：MMAS 约 `5.42e-7`，AS 约 `2.20e-6`。
该短跑包含首次诊断启动成本，wall time 比值约 2.28/2.00，**不能声称达到 5% 开销目标**。
后续验收会预热两个 profile，再测 100 轮及正式 batch。

首次 TSP500 验收因 fast-math tanh 与 CPU 相差约 `7.4e-6` 而失败。
失败记录保留在 `artifacts/v3`，不覆盖。超越函数复核阈值单独设为 `2e-5`，预算相对误差仍为 `1e-5`。
成功开发试跑记录在 `artifacts/v3-debug`。修订的不可变验收队列使用 `artifacts/v3-r2`。

独立 terminal oracle 已进一步实现，见下节。正式 32 实例分块/重排、三种型号、正式 batch 开销与存储测量，以及完整重启和分叉覆盖仍需完成验收。
`validation/diagnostics.json` 在全部项目完成前保持 `complete_acceptance=false`。
P1/P2 不因 pilot 运行结束而自动放行。

## 7. 入口

```bash
# 创建独立队列，不覆盖已有 cohort
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.diagnostic_campaign

# nohup 持续监控；A4000 忙时等待，不使用其他型号替代验收
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.watch start \
  --output control_experiments/mmas_ls/artifacts/v3-r2

# 对一个已完整提交的任务生成描述性机制表图
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.diagnostic_analysis \
  control_experiments/mmas_ls/artifacts/v3-r2/jobs/TASK/diagnostics
```

正式任务仍由不可变代码快照执行。所有同实例块、同随机重复的比较条件固定 GPU 型号。
设备锁跨 cohort 共享。GPU UUID、驱动及型号分别保留；不声称跨型号 bitwise 相等。

## 8. 独立输入核验发现的阻断项

`terminal_oracle.py` 从保存的几何、来源边信息素、colony 长度、路径、时钟及 TR 上下文，
独立重算全部 TR/PH terminal 的数学定义。方差采用 FP64 中心化计算。
它独立于“把保存的 terminal 重新输入 GP”这一程序输出检查。

对 A5000 的 TSP500、2 实例、100 轮验收记录，得到：

| 算法及字段 | 检查元素数 | 超出 `atol=rtol=2e-5` 的元素 | 其中被程序实际使用 | 最大绝对误差 |
|---|---:|---:|---:|---:|
| MMAS PH.EdgeTau | 20,000 | 4,489 | 1,489 | 1.0 |
| MMAS TR.RTau | 9,620 | 3,303 | 0 | 1.0 |
| AS PH.EdgeTau | 640,000 | 128,000 | 0 | 1.0 |
| AS TR.REta | 9,242 | 10 | 2 | 0.02550 |

这些是相关的元素级核验数，不是独立统计样本数。

有一个明确的退化例：第 1 轮的来源路径上，500 条边的信息素都为
`0.2818513810634613`。按定义，EdgeTau 应接近 0。但 GPU 记录为 1。
原实现使用 FP32 顺序求和，并由 `mean(x²) - mean(x)²` 计算方差。
在 CPU 上模拟该 FP32 运算，均值为 `-1.266376256942749`，
log-tau 为 `-1.2663754224777222`，方差约 `-6.3982e-6`。
方差被截为 0 后，均值误差被 `1e-8` 分母放大，最终 tanh 为 1。
这一例的方向和幅度与 GPU 记录一致。

因此，开启记录后轨迹不变，**不能证明原 terminal 的数值定义正确**。
本轮没有修改这个冻结计算，也没有把 oracle 阈值放大到容纳该伪信号。
不能据此断言以往 GP 负收益全部由这个问题导致。需要另行设计稳定计算对照。
AS 的无限名义 tau_max 另按原 fmin/fmax 保护语义核对，其 TauHeadroom 的 0 不表示有限物理上界余量。

修订队列的 A5000 100 轮配对验收已完成。预热后记录/无记录 wall time 比值为
MMAS 约 1.236、AS 约 1.080。均尚不能宣称满足 5% 目标。
500 轮记录检查与 RTX 4000 Ada 100 轮配对检查也已完成；正式 batch 检查仍单独运行。

门禁保持关闭。建议将“原 FP32 方差公式”和“稳定方差实现”作为独立数值控制实验，
先量化 terminal、GP 输出、路径与质量差异，再决定是否更改主实验的冻结版本。

补充回归测试：原机制专项 28 项 CPU 与 10 项 GPU 检查通过；后续另加零方差 oracle 与无损压缩测试。
全仓 CPU 回归为 130 passed、30 skipped、1 failed。失败来自旧机器 `/local/scratch/...`
路径下缺失的历史 checkpoint；未修改旧 manifest 或制造替代文件来掩盖该失败。

32-instance MMAS、100 轮的修订前流水线耗时为 6.259 秒（无审计）和 22.845 秒（审计），
比值约 3.65。后台 NPZ 写入累计 17.31 秒，其中重复几何矩阵占 11.21 秒；传输累计约 0.506 秒。
这些阶段有重叠，不能简单相加。瓶颈证据支持先优化几何重复保存和 CPU 压缩。
I/O 修订的独立配对试跑写入 `artifacts/v3-io`，不把优化前的数字替换为预测值。
