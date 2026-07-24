# Changelog

本项目遵循语义化版本。研究设计的统计结论冻结标签与软件版本标签分开。

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
