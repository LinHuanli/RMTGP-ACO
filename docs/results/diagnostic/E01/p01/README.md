# E01／p01：独立瓶颈诊断

证据等级：`diagnostic`；实验：E01；协议：p01。

已整理 12 条独立阶段诊断和 0 条完整预算细粒度诊断。

- 构造阶段包含终端、GP、候选选择和同步ACS局部更新，不等于纯GP时间。
- 逻辑工作计数不是硬件指令。插桩lane-0周期分布不是整卡墙钟比例。
- occupancy、DRAM带宽、stall和spill流量尚未实测，不用NVML利用率替代。
- 三迭代smoke位于E00功能验证区，不纳入此处的完整预算诊断。
- 配对路径一致性和快照重放用于检查插桩；不能证明最终求解质量提升。

## 图表和原始表

- [tsp100_stages](figures/tsp100_stages.svg)
- [tsp500_stages](figures/tsp500_stages.svg)

CSV/JSON在 `tables/`；PDF/SVG/PNG在 `figures/`。来源、筛选规则和文件校验见 `provenance.json`。

正式结果与本报告分开；未采集值不填零。
