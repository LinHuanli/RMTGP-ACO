# 历史路径强化机制可视化协议

## 目的

本协议回答一个具体问题：完整 MMAS 中哪个设计压缩了固定 GP 规则与 2-opt 结合后的边际收益，以及该设计如何通过信息素影响后续搜索。

分析链固定为：

1. 来源策略选择实际强化路径及其边支持集。
2. GP 信息素更新树在已选来源边内重分配固定预算。
3. 信息素改变下一轮构造选择概率。
4. 构造路径变化进入 2-opt。
5. 2-opt 后的优良路径更新历史记忆，并影响后续来源选择。

最终质量对照、同状态干预、自然轨迹和单实例动画分开报告。动画不作为统计证明。

## 已有证据

- 来源对照：32 个 TSP500 开发实例，5 个 ACO 种子，AS/MMAS 各三个固定表达式。
- 重型完整状态：8 个实例，3 个 ACO 种子，baseline 和三个 MMAS 表达式。
- 同状态干预：216/216 完成。包含 6 个快照轮次、baseline/表达式自身状态、三种表达式和 1、25、100、500 轮续跑。
- 独立确认：280 项仍因共享盘空间不足暂停。本协议不读取其不完整子集。

统计单位是实例。ACO 种子和固定表达式先在实例内平均。

## 定向重放

重放从 baseline 预处理特征中选择代表实例。五个特征是最终 reference gap、来源年龄、历史来源不同于本轮最优的比例、2-opt 后不同路径数和构造选择熵。先选择离分量中位数最近的实例，再选择离该实例五种子均值最近的种子。选择过程不读取 GP 结果或处理效应。

四种来源策略为：

- 强化本轮全部 32 条路径。
- 仅强化本轮最优路径。
- 按固定日程强化历史全局最优路径。
- MMAS 原生日程。

每种来源策略分别运行 baseline 和 MMAS 训练种子 81002 的双树表达式。八个求解使用同一实例、同一 ACO counter seed、32 只蚂蚁、5000 轮、候选表 20 和全蚂蚁 2-opt。每 25 轮保存四个信息素阶段。第 1 轮也保存。重放结果必须与对应开发任务的完整 anytime 曲线和最终路径逐元素一致。

种子 81002 只用于动画。它是三个 MMAS 冠军中唯一构造树和信息素更新树都非零的表达式。静态统计保留三个表达式。三个表达式均未通过原训练的非劣选择门槛，也未被最终部署。

重放目录有 3 GiB 硬上限。它独立位于 `artifacts/mechanism-visualization-v1`。它不解除 `mechanism-explanation-v2` 的存储停止标记。

## 指标

候选弧是每个城市的 20 条 construction candidate arcs。自然轨迹主要在候选弧域统计。全体非对角弧只作敏感性分析。

- 信息素熵和有效候选弧数。
- Gini 系数。
- 最高 500 和 2500 条候选弧的信息素质量占比。
- 下界占比。
- 实际来源、本轮最优、全局最优和参考路径边相对候选弧均值的信息素倍数。
- 最高 500 条候选弧的相邻采样点 Jaccard 持续率。
- 2-opt 前后路径边集差异和差异存活率。
- 同状态来源替换后的来源特有边、共有边和其他候选弧变化。
- 固定构造上下文中，候选集含来源差异边和不含差异边时的下一步概率总变差距离。

同状态来源比较保持更新前完整状态和总预算一致。同来源 GP 更新比较保持来源路径和总预算一致。

## 运行

```bash
PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.mechanism_visualization select
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.mechanism_visualization replay --device 0
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.mechanism_visualization analyze
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.mechanism_visualization render
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m control_experiments.mmas_ls.mechanism_visualization verify
```

输出报告位于 `reports/mechanism-explanation-v2`。旧版 `mechanism-explanation-v1` 保留，不覆盖。
