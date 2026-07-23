# 数据快照

原始数据约 15 GB，不进入 Git；本目录只版本化本说明与
[`manifest.json`](manifest.json)。manifest 对本研究允许使用的 81 个文件
记录了实例数、字节数和完整 SHA-256。

正式研究只使用 TSP50、TSP100、TSP500：

- training：TSP50（10 × 128,000）、TSP100（10 × 128,000）、
  TSP500（4 × 16,000）；
- validation：TSP50/TSP100 各 1,280，TSP500 为 128；
- test：TSP50/TSP100 各 1,280，TSP500 的 uniform、cluster、Gaussian
  各 128，以及 49 个 TSPLIB 实例。

其中 TSPLIB 有 42 个实例满足 \(n\le500\)，其余 7 个仅作为额外外推，
不进入主统计。下列数据被显式排除：

- `tsp100_concorde_7.756 copy.txt`（与原文件逐字节重复）；
- 所有 TSP200、TSP1K、TSP10K train/test 文件。

快速核验文件状态与每文件首条记录：

```bash
python -m rmtgp_aco verify-data \
  --manifest Datasets/manifest.json \
  --root Datasets/TSP \
  --skip-hashes
```

论文结果冻结前执行完整逐字节复核（会读取约十余 GB）：

```bash
python -m rmtgp_aco verify-data \
  --manifest Datasets/manifest.json \
  --root Datasets/TSP
```

重新生成完整 manifest：

```bash
python -m rmtgp_aco manifest \
  --root Datasets/TSP \
  --output Datasets/manifest.json \
  --full --workers 4
```

数据行格式、reference tour 的定义和防泄漏协议见研究设计第 16–17 节。
