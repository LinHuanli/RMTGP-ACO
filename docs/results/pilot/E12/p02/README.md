# E12／p02：局部搜索GPU执行器性能

状态核查：2026-10-10 16:22 NZDT。证据等级：pilot；性能holdout尚未开始。

本清单共有90个配对block：两规模×三宿主×真实cohort第1/25/50代×5个ACO随机block。每block同卡随机顺序执行none、2-opt scalar、2-opt cooperative、3-opt scalar、3-opt cooperative，共450次评价。P100/B32/A32/I500/K20不变。

scalar/cooperative必须返回相同路径和长度。不同LS模式的时间差是不同工作负载的成本差，不是同算法加速。

独立调优在`artifacts/runs/tuning/E12/p02/local-search/`进行。12种配置各5个配对block，独立程序来源与tuning实例，合计120次评价。**tuning测量不混入本页holdout统计。** 每卡已通过53项功能检查。

已有部分调优测量返回。调优数据保留在独立tuning目录，不在本页发布为holdout性能。在完整调优和短训练门禁通过前，不发布性能汇总或确认性倍数。

科学源码：`50c730bc5698b48a89c1589a3e27e71b82156cee`。控制器为`artifacts/operations/local-search-dispatch/p02/`。规范holdout产物为`artifacts/runs/pilot/E12/p02/local-search/`；测试集未打开。完整协议见[设计11](../../../../design/11_local_search_extension.md)。
