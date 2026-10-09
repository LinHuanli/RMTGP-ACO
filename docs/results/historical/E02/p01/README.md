# E02／p01：旧GP-ACO实验的协议复核

证据等级：`historical`；实验：E02；协议：p01。

这些是旧双树研究的导出。CPU为FP64，GPU为FP32后主机FP64计分，存在去重和结果复用。

不能与本轮单树FP32计时直接配对。TSP100五代累积与TSP500一次评价分别绘图。

历史CPU是Numba，不是纯Python。没有的CPU16/完整Python数据保持缺失。

导出原件的副本放在 `artifacts/provenance/historical/E02/p01`，不混在当前结果表或图目录。

## 图表和原始表

- [historical_timings](figures/historical_timings.svg)

CSV/JSON在 `tables/`；PDF/SVG/PNG在 `figures/`。来源、筛选规则和文件校验见 `provenance.json`。

正式结果与本报告分开；未采集值不填零。
