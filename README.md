# GPU 加速的单树 GP–ACO：可复现研究基线

项目根目录固定为 `/vol/grid-solar/sgeusers/linbocheng/MTGP_ACO_GPU`。代码、环境、缓存、临时文件和实验产物全部保存在此目录内。当前分支为 `research/gpu-acceleration-clean`。

本轮研究改变执行方式，不改变比较组的搜索预算。采用 DEAP 管理单树种群，Numba 实现 CPU 基线，CuPy/NVRTC 编译 CUDA C++ 实现 GPU 基线。PyTorch 已安装，后续用于独立张量化对照。当前不使用多树、局部搜索或低精度计算。

## 文档

- [研究计划](docs/design/README.md)：问题、方法和 E00–E13 实验卡。
- [执行协议 v1.1](docs/design/04_implementation_protocol.md)：本轮决策、准确的数据量、公式、接口和数值约定。
- [第一阶段记录](docs/results/phase1_status.md)：实际完成项、先导测量和未完成项。

## 目录

```text
configs/              科学预算、数值合同、硬件执行参数和环境锁
src/gpaco/            语言、数据、外层进化、实验记录、CPU/CUDA 后端
scripts/              环境安装、数据转换、后台启动、汇总绘图
tests/                E00 单元测试和训练恢复测试
Datasets/             原始数据和派生数组；splits/ 清单进入 Git
references/           外部源码、文献和锁定的历史实现快照
artifacts/            日志、checkpoint、cohort、结果、固定源码快照（不提交）
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

运行环境：Python 3.12.15、NumPy 2.4.6、Numba 0.68.0、DEAP 1.4.3、CuPy 14.2.0、PyTorch 2.14.0+cu132、CUDA 13.2。完整包版本见 `configs/environment.pip.lock.txt`；安装过程中生成的 Conda 显式包清单保存在 `artifacts/bootstrap/conda-explicit.txt`。安装脚本包含网络依赖；换机器须核对驱动、编译和数值测试，不能只依据 GPU 名称判断环境兼容。

## 运行

```bash
source scripts/env.sh
# 一次短训练：科学单代规模仍为 P100、B32、A32、I500。
CUDA_VISIBLE_DEVICES=0 python -m gpaco.cli train --n 100 --generations 3 \
  --output artifacts/my-smoke

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
CUDA_VISIBLE_DEVICES=0 python -m gpaco.cli benchmark --n 100 --blocks 5 \
  --cohort artifacts/pilot-v1/as-tsp100-seed1001/cohorts/generation-001.json \
  --output artifacts/e01/as-tsp100
```

`--variant as/acs/mmas` 切换固定宿主；`--backend cpu_existing --threads 12` 切换 CPU；`--no-generated` 使用 CUDA 字节码解释；`--lanes` 和 `--active-tasks` 控制执行映射。所有逻辑任务都执行，禁止 fitness 去重。寄存器资源不足的映射报为不可行，不自动缩减蚂蚁数。

`--profile-stages` 是单独的事件插桩诊断，不能与无插桩正式性能样本混用。`device_search_s` 不等于端到端时间；`eval_wall_s` 含本次编译、准备、上传、搜索和回传，但不含外层数据读取、适应度归约、验证及保存。`generation_wall_s` 和 `training_wall_s` 是更完整的边界。

当前尚未实现最终测试命令，也不会由训练自动运行测试集。第一阶段的目标是验证强基线并生成真实 cohort；共享优化、CPU/GPU 加速倍数和正式质量结论需后续配对实验。
