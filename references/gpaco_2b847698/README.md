# 锁定的历史实现快照（只作参考）

来源仓库：`https://github.com/LinHuanli/RMTGP-ACO.git`。

固定 commit：`2b847698225e1d0fff36a25c1015a080350a7f00`。

本目录通过 `git show <commit>:<path>` 提取旧项目相关模块，保留原始内容。原位置为 `src/rmtgp_aco/` 下同名文件，CUDA文件位于 `src/rmtgp_aco/cuda/`。这些模块不能作为当前研究包直接导入，也不是新研究的结果。

当前移植位于 `src/gpaco/`。移植仅保留单棵转移树、固定宿主、FP32搜索；数值/随机协议修正见 `docs/design/04_implementation_protocol.md`。`scripts/port_cpu_baseline.py` 记录CPU提取与FP32机械转换过程；运行版本另含人工核验修正，不能只运行该脚本便视作最终实现。

这是同一研究项目的可追溯源码材料。未为历史材料重新授予许可；使用及分发仍受来源仓库的适用许可约束。
