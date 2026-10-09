# 异构 GPU 先导实施记录

日期：2026-10-10。执行协议见[五类GPU先导方案](../design/05_cross_gpu_pilot.md)。性能、3代质量和50代历史先导分开记录。

## 实现与验证

- 已实现一次生成、SHA256核验和mmap读取的几何/初始化输入库；A5000规范ACO参考只读共享。
- 已实现每卡线程宽度与active任务筛选、独立5-block配对留出、30次3代短训练、统一A5000冠军复评、B128容量探索和独立阶段诊断。
- 已实现NVML能耗/功率/显存采样、争用记录、UUID互斥与nohup不可变快照启动。
- 已实现CSV/JSON、Markdown表、训练/验证/时间曲线、时间—能耗图及容量曲线；增量报告不混用未完成、污染或插桩数据。

| 检查 | 结果 | 说明 |
|---|---|---|
| CPU、进化及跨卡协议单元测试 | 24 passed | 包括只读参考、输入/seed身份、SHA篡改、3%保留规则、配对顺序和能耗单位 |
| A5000 基础E00 | 28 passed | 含冻结初始化与原计算路径逐位一致检查 |
| A40 基础E00 | 28 passed | AS/同步ACS/MMAS、多lane、小型闭环和不变性 |
| L40S 基础E00 | 28 passed | 同上 |
| L4 基础E00 | 28 passed | 同上 |
| RTX PRO 5000 Blackwell 基础E00 | 28 passed | 同上 |
| A5000端到端控制流smoke | 通过 | 冻结输入→调优→留出→训练→冠军复评→出图；缩小预算，不进入科学结果 |
| Ruff及diff空白检查 | 通过 | 不代表已完成所有E00实验卡 |

原始检查日志：`artifacts/bootstrap/hardware-E00-<model>.log`。端到端smoke在`artifacts/bootstrap/hardware-smoke-v2/`。

## 实验范围和结果边界

主工作点为AS、P100、B32、ants32、I500、FP32、无局部搜索；TSP100和TSP500。每型号一张物理GPU。30次短训练仅用于早期进化及时间观察，不能替代正式10-seed完整训练。

实际启动/完成以 `artifacts/hardware-pilot-v1/campaign.json`、各卡的 `launched.json`、`status.json`、`heartbeat.json` 和 `COMPLETE.json` 为准；失败写入 `FAILED.json`。本文件后续只增补已核实的进度和结果。

标准测试不变且未用于本轮实验。原六个A5000 50代训练不改参数或运行快照。
