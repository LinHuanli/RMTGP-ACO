# GPU Acceleration for Simulation-based GP

英文页面与图表；中文讲解稿。正文 36 分钟，讨论 4 分钟。

结果页只引用本目录已完成的测量；缺失数据保持待补。

## 1. GPU Acceleration for Simulation-based GP

时间：0.5 分钟。图：D6。

用 population、simulation、fitness 三个环节定义本次加速对象。报告围绕计算成本，不展开 GP-ACO 的算法创新。

## 2. How much work is inside one evaluation?

时间：2 分钟。图：D1。

从个体、实例、ACO iteration、蚂蚁、路径步骤和候选逐层展开。指出独立任务与状态依赖，结合日志中的实际 tasks 和 tours 解释预算。

## 3. Where does generation time go?

时间：1.5 分钟。图：C1。

展示 CPU 的一个注明编号的代表代。用秒数说明评估、遗传操作和其它开销的比例。评价是否占主导以实测为准。

## 4. Earlier approaches: workers, arrays and masks

时间：1.5 分钟。图：D1。

回顾 2024 年的多进程、循环数组化和 visited mask。说明当时解决的 Python 调度与逐元素循环开销，历史倍率不用于本次比较。

## 5. DEAP manages the population; Numba computes

时间：2 分钟。图：D2。

区分两条路径：将每棵树编译成函数；将树编码为指令交给已编译解释器。当前 CPU solver 采用后者。DEAP 的 gp.compile 本身不生成 Numba 机器码。

## 6. Individual JIT: when does compilation pay off?

时间：2 分钟。图：C2。

用同一数值函数展示 Python、JIT 含编译和已编译执行的累计时间。强调热点循环也在 JIT 边界内，微基准倍率不能直接外推为完整仿真倍率。

## 7. From CPU workers to GPU throughput

时间：1.5 分钟。图：D5。

CPU 用有限数量的核处理任务，GPU 需要足够多的并行工作来隐藏延迟。任务规模小时，准备和提交开销可能占较大比例。

## 8. Inside an RTX A5000

时间：2 分钟。图：D3。

说明 CPU 内存、PCIe、显存与 GPU 芯片的关系。64 个 SM 和 6 MiB L2 来自设备查询，8192 CUDA cores 来自规格表。线程不是固定绑定的一颗 core。

## 9. Memory hierarchy and data reuse

时间：2 分钟。图：D4。

说明寄存器、shared memory、L1、L2 和显存各自保存或缓存什么。L1/shared 是组合资源。线程局部数组可能溢出到 local memory，不能一概当作寄存器。

## 10. Grid, block, warp and thread

时间：2 分钟。图：D5。

用一个 program–instance task 放大到 block。32 ants 乘 8 lanes 是 256 threads，即 8 warps；一个 warp 内有四个 ant groups。8 lanes 不等于只评估8个候选。

## 11. Why GPU code can still be slow

时间：1.5 分钟。图：D4。

解释任务不足、warp 内分支分歧、寄存器压力、随机访问和主机提交间隙。只有实际诊断数据能支持具体瓶颈结论，GPU 利用率并不等于计算单元效率。

## 12. Our CPU–GPU evaluation pipeline

时间：2 分钟。图：D6。

DEAP 负责选择、交叉和变异。CPU 编码或生成代码，工具链编译，GPU 执行仿真，最后 CPU 用 FP64 重算返回 tour。编译成本计入首次使用。

## 13. Programs × instances × ants × candidates

时间：2 分钟。图：D5。

说明跨任务和任务内两层并行。当前 Numba 按 instance 并行、instance 内遍历 programs。GPU 将 task 映射到 block，并在 ant 内使用 lanes 协作。

## 14. Interpreted GP versus generated CUDA

时间：1.5 分钟。图：C8。

同一棵树，一边是 postfix 指令循环，一边是生成后的标量语句。解释减少指令解释开销的原因，同时展示编译和装载的成本。

## 15. Resident data and kernel organization

时间：1.5 分钟。图：D7。

相同问题批次可复用显存数据；更换批次需要新上传。v2 每个 iteration 有 construct/update。kernel 数量增加或减少本身不能决定快慢。

## 16. The same evaluations on four backends

时间：2.5 分钟。图：C3。

用完整冻结 trace 比较 CPU-1、CPU-8、GPU-v1 和 GPU-v2。相同 GP seed 不能保证跨精度演化轨迹相同，因此严格加速比采用回放。

## 17. Generation time and compilation overhead

时间：2 分钟。图：C4 / C5。

分开讲真实进化和匹配回放。观察首代与后续代的编译、实际唯一程序数、树规模和耗时关系。堆叠图只叠加互斥的墙钟阶段。

## 18. Scaling the population

时间：2 分钟。图：C6。

同时看 evaluation 秒数和实际 tasks/s。确认扩大 population 后实际执行程序确实增加，再讨论 GPU 利用和吞吐是否趋于稳定。

## 19. Simulation size and program complexity

时间：1.5 分钟。图：C7。

比较 TSP50、100、500 的计算时间。更大的 tour 带来更多步骤和更大状态；不同规模的 task 不是相同计算量。树复杂度使用节点统计解释。

## 20. Which optimizations matter?

时间：1.5 分钟。图：C8。

按 fused→v2 解释器→生成式 GP→8 lanes 的顺序解释对照。第一个变化包含多项内核组织调整，不能归因于单条优化。其余比较固定其它配置。

## 21. Conclusions and practical limits

时间：1 分钟。图：C3 / C6。

只填写已经完成的匹配测量结论：整体加速、编译摊销和适合 GPU 的工作量范围。说明硬件、精度、共享主机和 profiling 权限的实际限制。

## Sources

- [NVIDIA RTX A5000 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [DEAP GP API](https://deap.readthedocs.io/en/master/api/gp.html)
- [Numba Performance Tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
