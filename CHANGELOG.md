# Changelog

本项目遵循语义化版本。研究设计的统计结论冻结标签与软件版本标签分开。

## Unreleased — 2026-07-31

- 新增 paired Anytime+Final UCB fitness。CUDA 训练只在设备端累加每个
  program×instance 的 best-so-far 均值，不保存完整曲线；Numba、baseline
  archive schema 6 和逐代 metrics 使用相同定义；
- 新增 TSP500 两阶段多保真 racing：全种群 8-instance screen、32 个
  finalists 的 16-instance high-fidelity 复评、跨代高保真 elite、确定性
  exploration、保真度优先繁殖和逐代 screen/high 全量分数审计；
- 新增 64-program TSP500 学习信号门控、72 小时实测投影与冻结降载规则，
  以及 TSP100 matched control、3×3 正式调度和 TSP100/TSP500 5000-iteration
  跨规模最终测试；
- 新增纯 TSP100 的 AS/ACS/MMAS 单 GPU0 三种子完整预算配置与可恢复后台
  study 队列；九个训练 run 按 replicate-major 顺序独占一张 RTX 4000 Ada；
- 每代记录 train/validation 的 reference gap%、paired baseline delta、代时和
  ETA，并保存独立 selected candidate 与 validation gate deployment 决策；
- 最终测试扩展到 TSP50/100/500/1000；TSP1000 严格限定为 test-only 补充
  外推，测试 seed 与 GP seed 解耦；
- 新增跨三个 champion 共享的不可变 baseline test cache、原子 candidate
  shards、精确续跑、Holm-Wilcoxon、rank-biserial、三层 bootstrap、曲线和
  中文汇总报告。
- 单卡 runner 支持把逻辑 GPU0 映射到任意一张唯一可见的物理 RTX 4000
  Ada，并在任务边界按实际物理 index 检查外部进程；平衡三层 bootstrap
  改为内存受控的分块向量化实现。

## 0.5.0 — 2026-07-24

- 新增 CuPy Raw CUDA 融合后端：一个 block 完成一个
  program×instance 的全部 ACO 迭代，支持 AS、同步 ACS、MMAS 和双树
  postfix GP；
- 实施“GPU FP32 搜索、CPU FP64 tour 精确计分”契约、page-locked H2D、
  problem resident cache、显存保留/chunk、单 GPU、双 GPU LPT shard 与
  campaign 调度；
- population 后端统一返回最优 tour，新增单卡/双卡/chunk 逐位不变性和
  GPU 内部零 residual 回归测试；
- CUDA 训练选出的候选必须再通过独立 Numba float64 gate，GPU gate 与
  CPU/FP64 gate 任一失败均部署原始 ACO fallback，并单独保存审计表；
- CPU PyTorch、CPU Numba 和 CUDA 补齐 ACOTSP-style MMAS branching-factor
  检查与 pheromone restart，并记录 restart diagnostics；
- 新增 `benchmark-accelerators` 和 `development_acs_cuda.yaml`，baseline、
  checkpoint、kernel semantic 与 Protocol A 根目录升级到 v0.5。
- 在两张 RTX 4000 Ada 上完成 32 ants × 500 iterations 正式性能复核：
  单卡相对 CPU16 为 14.63×，dual 为 23.82×，双 run campaign 吞吐扩展为
  1.972×；单 run 双卡扩展 1.629× 未过 1.7× 门槛，故正式多实验调度采用
  每卡一个独立 run；
- GPU benchmark 的外部进程检测改为按 UUID 映射并仅检查目标设备，避免
  未参与测试的其他 GPU 作业误触发争用标记。

## 0.4.0 — 2026-07-24

- 将 Protocol A 的 AS、ACS、MMAS 统一为 32 只蚂蚁和 500 个 ACO
  iterations；GP 仍使用 population 100、50 generations；
- 协议标识与新生成的 schedule/baseline/run 根目录升级为
  `protocol-a-v0.4`，禁止把 v0.3 的 baseline archive 或第 7 代 checkpoint
  用于新配置；
- 保留 ACOTSP 原始无局部搜索默认值工厂，正式 YAML 通过显式覆盖实现统一计算
  预算。

## 0.3.0 — 2026-07-24

- 冻结 Protocol A v0.3 的 train/selection/gate schedule 与 baseline archive；
- fitness 统一为对 reference optimum 的 scale-balanced absolute gap%，同时
  逐代记录相对原始 ACO baseline 的 paired delta；
- 新增 16-thread population-batch 后端、行为等价 GP 去重、instance-major
  工作区复用、ACS 稀疏信息素更新和 candidate-level 内存流量优化；
- 新增 `benchmark-training`，以完整单代负载执行 1--3 代加速短跑而不运行
  validation/checkpoint；
- ACS 同一 seed 的前三代端到端时间由 97.09 s 降为 40.52 s，逐代 fitness
  与优化前记录一致。

## 0.2.0 — 2026-07-23

- 新增 float64 Numba CPU 后端，覆盖 AS、同步/顺序 ACS、MMAS 与双树
  postfix program；
- 使用 instance-keyed counter RNG，保证 baseline/candidate CRN、batch
  划分不变性和多进程确定性；
- 按 GP program 实际引用集合惰性构造 transition/pheromone terminals；
- 训练改为跨代复用 8 个单线程 worker，并并行执行 validation；
- 新增逐代耗时、吞吐率、分规模 fitness、原子 checkpoint 和 `--resume`；
- 固定 NumPy 2.2.6、Numba 0.61.2、llvmlite 0.44.0 正式环境组合。

## 0.1.0 — 2026-07-23

- 冻结 TSP50/100/500 中文研究设计与实验预注册；
- 实现连续 Euclidean AS、同步 ACS 与 MMAS；
- 实现 Strongly Typed transition/pheromone 双树 GP；
- 实现 residual、full-replacement、Legacy-GP 与信息素预算消融接口；
- 实现分层数据采样、CPU 多进程评估、validation non-inferiority gate；
- 实现数据 SHA-256 manifest、运行 artifacts、paired test 与论文统计；
- 加入端到端、确定性、类型和算法不变量测试。
