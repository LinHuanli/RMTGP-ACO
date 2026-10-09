# 功能验证记录

smoke只检查实现和流程，不作为性能或质量统计。即使一次验证使用完整P/B/A/I预算，也不自动成为先导配对或正式结果。

- 初始构造、单代与三代流程：登记 `E00-p01-initial-validation`。
- 插桩逐位一致性、三迭代快照重放：登记 `E00-p01-instrumentation-checks`。
- 旧跨卡流程、调度器流程及E00测试日志：登记 `shared-bootstrap-records`；保留旧原始包。

准确路径见 [存储盘点](../status/storage_inventory.md)。原始数据保存在 `artifacts/runs/smoke` 或已明确标为旧bootstrap的溯源包中，不复制到正式结果区。
