# E09／p02：局部搜索联合训练

状态核查：2026-10-10 16:22 NZDT。证据等级：pilot；尚未发布训练均值。

已提交科学源码`50c730bc5698b48a89c1589a3e27e71b82156cee`。12种配置的调优已在12张A5000上开始，每卡已通过原合同31项及LS扩展22项检查。每配置完成5个调优配对block和3代完整预算短训练后，独立解锁本配置的50代训练。

| 配置维度 | 冻结值 |
|---|---|
| 宿主 | AS、同步ACS、MMAS，分别进化 |
| n | 100、500 |
| 局部搜索 | 2-opt、3-opt，作用于每轮全部蚂蚁 |
| GP根seed | 1001、1002、1003；共36次先导 |
| 科学预算 | P100/B32/A32/I500/K20/G50 |
| 验证 | 128实例×3 ACO重复，每5代 |
| 固定对照 | 每run预生成同宿主、同LS的ZERO对照 |
| 测试 | 未打开 |

当前不能得出GP＋LS的最终质量结论，也不能说36次训练已经开始或完成。失败/争用门禁不能自动放行。

规范原始结果：`artifacts/runs/pilot/E09/p02/local-search/`。短训练单列于`artifacts/runs/smoke/E00/p02/local-search/`，不纳入先导统计。控制器为`artifacts/operations/local-search-dispatch/p02/`；nohup PID记录在`controller_launched.json`，实时状态在`dispatcher_status.json`和`queue.json`。旧p01正式实验不受本队列替换。

协议、算法及失败规则见[设计11](../../../../design/11_local_search_extension.md)。已有无LS结果见[p01先导](../p01/README.md)和[正式固定对照](../../../../results/formal/E09/p01/README.md)。
