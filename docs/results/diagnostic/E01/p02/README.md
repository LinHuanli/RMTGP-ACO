# E01／p02：局部搜索成本分解

状态核查：2026-10-10 16:22 NZDT。证据等级：diagnostic；正在等待各配置的调优及短训练门禁。

预登记12项：3宿主×2规模×2种局部搜索。使用真实第1代cohort、block0，以及tuning选定的执行器。对同输入分别运行无插桩和事件插桩版本，验证完整路径与长度一致。

将分别统计construct、local_search、global_update设备时间；同时保留整次评价墙钟、编译/预热、能耗、显存、执行任务数和LS逻辑计数。共享城市排列生成属于LS时间。逻辑候选检查不是实际GPU指令数，lane采样周期也不是整卡墙钟份额。未获得的性能计数器保持缺失。

插桩结果不得充当无插桩速度分母。尚无数据，不提前填百分比。源码快照为`50c730bc5698b48a89c1589a3e27e71b82156cee`，产物位于`artifacts/runs/diagnostic/E01/p02/local-search/`。协议见[设计11](../../../../design/11_local_search_extension.md)。
