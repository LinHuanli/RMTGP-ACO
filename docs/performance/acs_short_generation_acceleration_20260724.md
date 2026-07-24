# ACS 正式规模短代加速实验（2026-07-24）

> 历史基准：本文使用 Protocol A v0.3 的 10 ants、100 ACO iterations。
> Protocol A v0.5 已改为 32 ants、500 ACO iterations，本文代时不可直接用于
> 估计新协议。

## 1. 目的与比较原则

本阶段不执行 50 代确认性训练，而以 1--3 代短跑寻找等价计算下的最高吞吐。
所有候选实现共享冻结的 schedule、初始种群、ACO seeds 和 float64 参数，并以
旧内核保存的逐个体、逐实例输出作为 golden reference。一次优化只有同时满足
下列条件才可进入正式后端：

1. `best_length` 对全部 \(100\times(16+16)\) 个
   genotype--instance 组合逐元素相同；
2. `best_iteration` 逐元素相同；
3. AS、ACS、MMAS 的语义、batch partition invariance 和确定性测试通过；
4. 墙钟时间有可重复的正收益，或能删除确定无用的内存流量。

因此，本报告中的加速不是通过减少蚂蚁数、ACO 迭代数、训练实例数或降低精度
获得的。

## 2. 环境与正式单代负载

- CPU：Intel Xeon W-2245，8 个物理核、16 个逻辑 CPU；
- GPU：2 × NVIDIA RTX A5000 24 GB（仅用于排除性短测）；
- Python 3.12.13；
- NumPy 2.2.6、Numba 0.61.2、llvmlite 0.44.0；
- ACS，population \(P=100\)，TSP50 与 TSP100 各 16 个实例；
- 每个实例 10 只蚂蚁、100 次 ACO 迭代，candidate list \(K=20\)；
- float64，16 个 Numba threads；
- schedule hash：
  `648102f4a03c362380e3fa7a88c16c8fad835baaf3e459559dffc91d0253de9e`；
- baseline 全部从冻结 archive 查询，不在代内重新运行。

原始 golden 一代内核时间为 39.989 s。该文件还保存了 100 个结构哈希、
TSP50/TSP100 的最优长度与最优出现迭代，供每次改动做精确回归。

## 3. 最终保留的优化

### 3.1 计算布局

- 将任务次序改为 instance-major；一个线程连续计算同一实例上的整个 program
  集合，使距离矩阵、候选表和静态几何量保持热缓存；
- 每个并行 instance 复用 pheromone、tour、visited、GP stack 等工作区；
- TSP50 与 TSP100 仍顺序各使用 16 线程。实测顺序执行为 15.627 s，
  并发 \(8+8\) 线程为 16.405 s，并发 \(6+10\) 线程为 17.639 s，故不采用
  跨尺度并发。

### 3.2 静态量与稀疏更新

- 每个实例只计算一次 \(\log\eta\)、局部均值和 nearest-neighbour
  pheromone 初值；
- ACS global update 只访问 global-best tour 的 \(n\) 条边，不再扫描
  \(n^2\) dense deposit；
- `ColonyFreq` 与 ACS local-update edge counts 只清零本轮实际激活的 edge；
- 预计算同一 construction step 中重复边的 local-update multiplicity
  factors。

### 3.3 GP postfix 解释

- 采用 instruction-major、candidate-column 内循环的解释器；
- opcode dispatch 位于列循环之外；
- 只构造 program 的 required-terminal mask 所需输入；
- 对 candidate/edge 内相同的 scalar terminals 只计算一次；
- transition 不再无条件写入 \(\tau,\eta,d\) 三个临时向量，而是在相应
  terminal 确实出现时直接读取原矩阵；
- `scores`、probability scratch 和 interpreter scratch 复用，减少热循环
  内存流量；
- 对正式参数 \(\alpha=1,\beta=2\) 使用等价专门路径；
- ACS 的 \(q_0\) exploitation 判定先于 roulette 累积和，默认 \(q_0=0.9\)
  时多数 step 不再计算无用累积和。

### 3.4 语义 intron 与行为去重

Residual transition 为

\[
s_{ij}=s^{\mathrm{ACO}}_{ij}
\left(1+\gamma_{\mathrm{tr}}\tanh r_{ij}\right).
\]

当 \(r_{ij}=c\) 对当前候选集合恒定时，它只给所有候选乘以同一正数，不改变
argmax 或归一化后的选择分布。因此，只依赖
`Entropy`、`ConstructProg`、`ACOProg`、`Stagnation` 和常数的 transition
树是该位置的语义 intron。

Budget-residual pheromone 在 edge residual 恒定时也会被预算归一化消去：

\[
\Delta\tau_e
=\frac{n/L}{\sum_{e'}(1/L)(1+\gamma_{\mathrm{ph}}\tanh c)}
  \frac{1}{L}(1+\gamma_{\mathrm{ph}}\tanh c)
=\frac{1}{L}.
\]

后端先将这些树标为 inactive，再按两棵有效 postfix programs 合并行为等价
个体。首代 91 个结构唯一体由此变成 79 个行为唯一体；输出随后按 inverse
mapping 展开，故 DEAP 仍获得每个结构个体的 fitness。该规则有独立回归测试，
golden 首代也保持逐元素完全相同。

## 4. 优化轨迹

下表为同一固定首代的代表性单次测量。共享机器的短时负载会造成小幅波动，
因此中间数字用于定位方向，最终判断以 3 代端到端短跑为准。

| 阶段 | 时间 (s) | 相对 39.989 s |
|---|---:|---:|
| 原始 population-batch golden | 39.989 | 1.00× |
| 列解释器 + ACS 稀疏 update | 33.37 | 1.20× |
| terminal/DistRank/\(q_0\) 热循环优化 | 29.94 | 1.34× |
| instance-major 调度 | 25.40 | 1.57× |
| opcode 外提与 scalar terminal | 24.56 | 1.63× |
| 删除 \(\tau,\eta,d\) 无条件临时向量 | 19.48 | 2.05× |
| 语义 intron + 行为去重 | 16.04 | 2.49× |
| score/scratch 复用后的最终内核 | 14.50 | 2.76× |

以下方案经短测后未保留：

- “标量子树延迟展开”需要在解释器维护动态 scalar flags，实测
  25.09/25.23 s，慢于当时的 24.56 s；
- 两尺度并发造成共享缓存和线程运行时竞争，慢于顺序满 16 线程；
- OpenMP、workqueue 和 TBB 差异很小，保留默认 TBB；
- 单纯缩小 GP stack 行数没有稳定可辨认的代时收益，但因能降低工作集且不改变
  语义，最终实现仍按真实 postfix 最大栈深分配。

## 5. 三代端到端结果

运行命令：

```bash
rmtgp-aco benchmark-training \
  --config configs/acs_protocol_a.yaml \
  --schedule runs/protocol-a-v0.3/schedules/acs-seed-2001.json \
  --baseline-archive runs/protocol-a-v0.3/baselines/acs \
  --method-profile rmtgp-full-f1 \
  --generations 3 --cpu-threads 16 \
  --output runs/protocol-a-v0.3/benchmarks/acs-3gen-optimized.json
```

| 代 | 结构唯一 | 行为唯一 | evaluation (s) | 整代 (s) | 最佳 absolute gap (%) | \(\Delta_{50}\) (pp) | \(\Delta_{100}\) (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 91 | 79 | 14.596 | 14.607 | 3.427308 | -0.309121 | -1.173959 |
| 2 | 77 | 67 | 12.199 | 12.256 | 3.014360 | -0.544552 | -2.055411 |
| 3 | 71 | 69 | 13.595 | 13.655 | 3.266735 | -0.595283 | -0.924608 |

优化前同一 seed 的前三代整代时间为 35.76、31.71、29.62 s；三代合计由
97.09 s 降为 40.52 s，即端到端加速约 2.40×。三代的 minimum fitness 与
两个尺度的 baseline delta 均和优化前记录一致。

第 3 代的 fitness 高于第 2 代不能解释为进化倒退：Protocol A 每代使用不同
的冻结 mini-batch，跨代 raw fitness 不在同一个样本集合上。学习曲线应另用
固定 monitoring/validation panel 比较。

## 6. GPU 与 C++ 决策

现有 PyTorch GPU 路径在 RTX A5000 上对 TSP100、batch 16、10 次 ACS
迭代的短测为 3.405 s，而 Numba CPU 为 0.102 s，GPU 慢约 33 倍。原因是
候选选择、visited mask 和 GP 树解释包含大量小规模、分支化、顺序依赖操作，
无法摊薄 PyTorch kernel launch；A5000 的 float64 吞吐也不适合该负载。
因此当前不采用 GPU。

定制 CUDA block-per-instance 内核可能改变结论，但它需要重新实现
counter-RNG、三种 ACO、typed GP interpreter 和全部数值回归，只有预估达到
数量级加速时才值得开展。当前 Numba 已生成原生 LLVM 代码；没有 profile
证据表明把同一动态解释器机械翻译为 C++ 会获得足以抵消维护成本的收益，
故也暂不新增 C++ 后端。

## 7. 当前结论

正式 50 代运行继续暂停。现阶段推荐冻结 CPU `numba_batch` 路径，先用
`benchmark-training` 对 ACS 重复 1--3 代并做系统负载方差检查，再为 AS 和
MMAS 各建立对应的短代 baseline archive，分别运行 1 代性能审计。只有短代
结果稳定后才恢复完整训练。
