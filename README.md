# GPU 加速的单树 GP–ACO：可复现研究基线

项目根目录固定为 `/vol/grid-solar/sgeusers/linbocheng/MTGP_ACO_GPU`。代码、环境、缓存、临时文件和实验产物全部保存在此目录内。当前分支为 `research/gpu-acceleration-clean`。

本轮研究改变执行方式，不改变比较组的搜索预算。采用 DEAP 管理单树种群，Numba 实现 CPU 基线，CuPy/NVRTC 编译 CUDA C++ 实现 GPU 基线。PyTorch 已安装，后续用于独立张量化对照。当前不使用多树、局部搜索或低精度计算。

## 文档

- [研究计划](docs/design/README.md)：问题、方法和 E00–E13 实验卡。
- [执行协议 v1.1](docs/design/04_implementation_protocol.md)：本轮决策、准确的数据量、公式、接口和数值约定。
- [结果总索引](docs/results/README.md)：正式、先导、诊断、功能验证、历史资料与运行状态分开记录。
- [产物管理规范](docs/design/08_artifact_and_result_management.md)：登记、命名、证据等级、来源校验与活跃目录迁移规则。
- [五类GPU先导协议](docs/design/05_cross_gpu_pilot.md)：统一冻结输入、硬件调优、留出性能、3-seed短训练及规范冠军审计。
- [六组CPU基线与独立诊断](docs/design/07_cpu_baselines_and_diagnostics.md)：无JIT／Numba的1、8、16物理核入口、历史复用边界、工作量计数和状态重放。
- [已有基线与图表](docs/results/pilot/E01/p01/README.md)：普通ACO、当前CPU覆盖、GPU吞吐／显存／能耗；插桩和历史结果另册。
- [正式基线与映射补充](docs/design/10_formal_baseline_and_mapping.md)：20次独立正式基线训练及18个同卡映射配对block，原始数据分别登记。

## 目录

```text
configs/              科学预算、数值合同、硬件执行参数和环境锁
src/gpaco/            语言、数据、外层进化、实验记录、CPU/CUDA 后端
scripts/              环境安装、数据转换、后台启动、汇总绘图
tests/                E00 单元测试和训练恢复测试
Datasets/             原始数据和派生数组；splits/ 清单进入 Git
references/           外部源码、文献和锁定的历史实现快照
artifacts/            runs/{formal,pilot,diagnostic,smoke}；输入、缓存、溯源与运行管理分开
docs/results/         分类报告：README、tables、figures、provenance；不存大数组和原始日志
.envs/ .tools/ .cache/ 项目内运行环境和缓存（不提交）
```

原始标准测试集保持文件内容和顺序：TSP100 为 **1280** 个实例，TSP500 为 **128** 个实例。不补样、不重划。训练池每规模 32768 个，验证、调参、性能留出各 128 个；均与测试集隔离。

## 环境与检查

以下命令使用 Bash，在项目根目录执行。安装脚本只操作项目内前缀，不替换系统 Python。

```bash
bash scripts/bootstrap_environment.sh
source scripts/env.sh
python scripts/prepare_data.py
python -m ruff check src scripts tests
python -m pytest tests/test_core.py tests/test_evolution.py -q
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_cuda.py -q
```

运行环境：Python 3.12.15、NumPy 2.4.6、Numba 0.68.0、DEAP 1.4.3、CuPy 14.2.0、PyTorch 2.14.0+cu132、CUDA 13.2。完整包版本见 `configs/environment.pip.lock.txt`；初始安装清单现归档于 `artifacts/provenance/bootstrap/p01/conda-explicit.txt`。安装脚本包含网络依赖；换机器须核对驱动、编译和数值测试，不能只依据 GPU 名称判断环境兼容。

## 运行

```bash
source scripts/env.sh
# 一次短训练：科学单代规模仍为 P100、B32、A32、I500。
CUDA_VISIBLE_DEVICES=0 python -m gpaco.cli train --n 100 --generations 3 \
  --output artifacts/runs/smoke/E00/p01/initial-validation/tsp100-as-train-g003-a02

# 检查空闲设备和待启动任务；此命令不启动训练。
python scripts/launch_pilots.py
# 已提交代码且 smoke 通过后，显式启动 2 个规模 × 3 个根种子。
python scripts/launch_pilots.py --execute
python scripts/report_pilots.py --campaign pilot-v1
```

启动器将已提交源码复制到项目内不可变运行快照。每个工作进程用 `nohup` 独立运行，只暴露指定 GPU UUID。启动前复查占用；运行中记录设备状态和争用。空闲快照不是集群资源预留，也不保证之后没有其他用户进入。运行日志位于 `artifacts/launches/<campaign>/<run>/`。

训练默认 50 代。每 5 代对训练前 5 名和历史验证冠军执行验证，验证为 128 实例 × 3 次随机重复。最终只按验证选择一个冠军。每代写入 `history.json`，每代保存可恢复 checkpoint，第 1/25/50 代保存真实程序 cohort。不同代训练实例不同，不能把训练 gap 曲线下降直接解释为同一实例集上的质量提高。

恢复必须使用相同源码快照、参数和输出目录，加 `--resume`。恢复运行记录为非连续计时样本，不能充当正式 uninterrupted 训练时间。`history.json` 是汇总输入；`events.jsonl` 是追加事件日志，中断重试时可能含有重复事件，不直接用于统计。

### 性能诊断

```bash
CUDA_VISIBLE_DEVICES=GPU-实际UUID python scripts/diagnose_gpu.py \
  --bundle artifacts/inputs/frozen-population/p01/tsp100/as \
  --output artifacts/runs/diagnostic/E01/p01/work-diagnostics/tsp100-as-g001-b000-a01
```

诊断入口从已登记的冻结输入读取宿主和科学预算。通用 `gpaco.cli benchmark` 仍支持 `--variant`、`--backend`、`--no-generated`、`--lanes`、`--active-tasks` 等执行参数，但其输出须事先规划所属证据等级。六组CPU测量使用专用 `scripts/benchmark_cpu.py`，不是临时挑线程数。所有逻辑任务都执行，禁止fitness去重；资源不足不自动缩减蚂蚁数。

`--profile-stages` 是单独的事件插桩诊断，不能与无插桩正式性能样本混用。`device_search_s` 不等于端到端时间；`eval_wall_s` 含本次编译、准备、上传、搜索和回传，但不含外层数据读取、适应度归约、验证及保存。`generation_wall_s` 和 `training_wall_s` 是更完整的边界。

当前尚未实现最终测试命令，也不会由训练自动运行测试集。第一阶段的目标是验证强基线并生成真实 cohort；共享优化、CPU/GPU 加速倍数和正式质量结论需后续配对实验。

### 异构 GPU 先导

```bash
source scripts/env.sh
python scripts/launch_hardware_pilot.py            # 检查五张预选卡，不启动
python scripts/launch_hardware_pilot.py --execute  # 仅首次启动；不覆盖已有实验
python scripts/report_hardware_pilot.py           # 从已完成记录增量汇总
```

配置位于 `configs/hardware/cross_gpu_pilot.yaml`。A5000、A40、L40S、L4、RTX PRO 5000 Blackwell各使用一张物理卡；先进行基础E00，再筛选执行计划、完成5个独立随机流的留出配对block，随后进行每规模3个seed、各3代的短训练。每卡每规模的最终选定计划只由tuning数据决定。独立A5000规范参考库和冠军复评用于隔离输入与浮点差异。

产物暂时仍在 `artifacts/hardware-pilot-v1/`，因为任务运行中而延后迁移。`summary/`是运行期增量视图，不是正式结果。历史启动记录见 [异构先导启动快照](docs/results/status/snapshots/2026-10-10-cross-gpu-startup.md)，当前路径见[盘点](docs/results/status/storage_inventory.md)。

### 持续利用空闲 A5000

`scripts/a5000_pool.py --execute` 首次创建项目内nohup调度器，每60秒使用 `gpu-free` 扫描并复查空闲RTX A5000。队列包含两个规模、三阶段真实种群和三ACO宿主的GPU基线配对与独立诊断；缺失cohort自动等待依赖，不占卡空等。已有50代训练及跨卡先导保持原样。详见[持续队列协议](docs/design/06_a5000_continuous_queue.md)，实时进度与图表位于 `artifacts/runs/pilot/E01/p01/gpu-baselines/summary/`。旧 `artifacts/a5000-main-v1` 只作历史地址兼容。

另有 `scripts/diagnostic_pool.py` 管理的54项完整预算细粒度诊断，协议见 [09](docs/design/09_diagnostic_campaign.md)。两个队列均已启动，不要重复创建；原始调度状态和最近核查见 [运行状态](docs/results/status/execution_status.md)。新增科学记录只写登记的 `artifacts/runs/<tier>/<E>/<protocol>/`，不再创建任意顶层实验目录。

启动当时的控制器与GPU清单见[历史启动记录](docs/results/status/snapshots/2026-10-10-a5000-queue-startup.md)。实际进度读取队列及心跳，不能从历史文档推断。
