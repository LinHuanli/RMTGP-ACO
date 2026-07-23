"""TSP 文本数据的解析、验证、预计算与批处理。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch

from .model import ProblemBatch

try:
    from numba import njit
except (ImportError, ModuleNotFoundError):
    # 当前环境可能暂时存在 NumPy/Numba 版本冲突；算法保持可运行，
    # 安装 pyproject.toml 中的兼容版本后会自动启用 JIT。
    njit = None


@dataclass(frozen=True, slots=True)
class TSPInstance:
    """一条数据记录及其已验证的 reference tour。"""

    instance_id: str
    coords: np.ndarray
    reference_tour: np.ndarray
    reference_length: float
    coordinate_hash: str

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])


def _tour_length_numpy(coords: np.ndarray, tour: np.ndarray) -> float:
    """用连续 Euclidean 距离计算闭合 tour length。"""

    ordered = coords[tour]
    delta = ordered[1:] - ordered[:-1]
    return float(np.sqrt(np.sum(delta * delta, axis=1)).sum(dtype=np.float64))


if njit is not None:
    _tour_length_numba = njit(cache=True)(_tour_length_numpy)
else:
    _tour_length_numba = _tour_length_numpy


def coordinate_hash(coords: np.ndarray) -> str:
    """对规范化 float64 坐标字节计算稳定哈希。"""

    canonical = np.ascontiguousarray(coords, dtype="<f8")
    return sha256(canonical.tobytes()).hexdigest()


def validate_reference_tour(tour: np.ndarray, n: int) -> None:
    """检查一基输入转为零基后的闭合 Hamiltonian cycle。"""

    if tour.shape != (n + 1,):
        raise ValueError(f"reference tour 应有 {n + 1} 个节点，实际为 {tour.size}")
    if int(tour[0]) != int(tour[-1]):
        raise ValueError("reference tour 未闭合")
    body = tour[:-1]
    if np.any(body < 0) or np.any(body >= n):
        raise ValueError("reference tour 包含越界城市")
    if not np.array_equal(np.sort(body), np.arange(n, dtype=body.dtype)):
        raise ValueError("reference tour 不是完整排列")


def parse_tsp_line(line: str, *, instance_id: str) -> TSPInstance:
    """解析 `coordinates output closed_tour` 格式的一行。

    文件中的城市编号从 1 开始；内部统一转换到从 0 开始。
    """

    tokens = line.strip().split()
    if not tokens:
        raise ValueError(f"{instance_id}: 空数据行")
    try:
        separator = tokens.index("output")
    except ValueError as exc:
        raise ValueError(f"{instance_id}: 缺少 output 分隔符") from exc
    if separator % 2 != 0 or separator < 4:
        raise ValueError(f"{instance_id}: 坐标 token 数必须为大于等于 4 的偶数")
    if tokens.count("output") != 1:
        raise ValueError(f"{instance_id}: output 分隔符必须唯一")

    coords = np.asarray(tokens[:separator], dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(coords).all():
        raise ValueError(f"{instance_id}: coordinates 含 NaN 或无穷值")
    n = int(coords.shape[0])

    try:
        tour = np.asarray(tokens[separator + 1 :], dtype=np.int64) - 1
    except ValueError as exc:
        raise ValueError(f"{instance_id}: tour 必须由整数城市编号组成") from exc
    validate_reference_tour(tour, n)
    length = _tour_length_numba(coords, tour)
    if not np.isfinite(length) or length <= 0:
        raise ValueError(f"{instance_id}: reference length 必须为正有限数")

    return TSPInstance(
        instance_id=instance_id,
        coords=coords,
        reference_tour=tour,
        reference_length=float(length),
        coordinate_hash=coordinate_hash(coords),
    )


def iter_tsp_file(path: str | Path) -> Iterator[TSPInstance]:
    """流式读取大型文本文件，避免把整个 shard 放入内存。"""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            yield parse_tsp_line(
                line,
                instance_id=f"{source.as_posix()}:{line_number}",
            )


def build_line_offsets(path: str | Path) -> np.ndarray:
    """构建可用于随机访问的 byte offset 索引。"""

    offsets: list[int] = []
    with Path(path).open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            offsets.append(offset)
    return np.asarray(offsets, dtype=np.int64)


def load_indexed_instances(
    path: str | Path,
    offsets: np.ndarray,
    indices: Sequence[int],
) -> list[TSPInstance]:
    """按 offset 读取给定行，支持巨大训练 shard 的随机采样。"""

    source = Path(path)
    instances: list[TSPInstance] = []
    with source.open("rb") as handle:
        for index in indices:
            if index < 0 or index >= len(offsets):
                raise IndexError(f"行索引越界: {index}")
            handle.seek(int(offsets[index]))
            line = handle.readline().decode("utf-8")
            instances.append(
                parse_tsp_line(
                    line,
                    instance_id=f"{source.as_posix()}:{index + 1}",
                )
            )
    return instances


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """计算 float64 对称 Euclidean distance matrix。"""

    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    np.fill_diagonal(distances, 0.0)
    return distances


def nearest_neighbour_data(
    distances: np.ndarray,
    candidate_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 candidate list 与每个有向城市对的完整距离 rank。"""

    n = int(distances.shape[0])
    if distances.shape != (n, n):
        raise ValueError("distances 必须为方阵")
    if n < 2:
        raise ValueError("TSP 至少需要两个城市")
    k = min(candidate_size, n - 1)
    work = distances.copy()
    np.fill_diagonal(work, np.inf)
    order = np.argsort(work, axis=1, kind="stable")
    nn_indices = order[:, :k].astype(np.int64, copy=False)

    ranks = np.empty((n, n), dtype=np.int64)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(1, n + 1, dtype=np.int64)
    ranks[np.arange(n), np.arange(n)] = 0
    return nn_indices, ranks


def make_problem_batch(
    instances: Iterable[TSPInstance],
    *,
    candidate_size: int = 20,
    epsilon_distance: float = 1e-12,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> ProblemBatch:
    """把同规模实例转换为算法使用的 tensor batch。"""

    records = list(instances)
    if not records:
        raise ValueError("至少需要一个实例")
    n = records[0].n
    if any(item.n != n for item in records):
        raise ValueError("ProblemBatch 只能包含同一城市规模")

    coords_array = np.stack([item.coords for item in records])
    distance_array = np.stack([pairwise_distances(item.coords) for item in records])
    heuristic_array = np.zeros_like(distance_array)
    positive = distance_array > 0
    heuristic_array[positive] = 1.0 / (
        distance_array[positive] + epsilon_distance
    )

    nn_and_rank = [
        nearest_neighbour_data(distance, candidate_size)
        for distance in distance_array
    ]
    nn_array = np.stack([item[0] for item in nn_and_rank])
    rank_array = np.stack([item[1] for item in nn_and_rank])

    return ProblemBatch(
        coords=torch.as_tensor(coords_array, dtype=dtype, device=device),
        distances=torch.as_tensor(distance_array, dtype=dtype, device=device),
        heuristic=torch.as_tensor(heuristic_array, dtype=dtype, device=device),
        nn_indices=torch.as_tensor(nn_array, dtype=torch.int64, device=device),
        full_nn_rank=torch.as_tensor(rank_array, dtype=torch.int64, device=device),
        reference_tour=torch.as_tensor(
            np.stack([item.reference_tour for item in records]),
            dtype=torch.int64,
            device=device,
        ),
        reference_length=torch.as_tensor(
            [item.reference_length for item in records],
            dtype=dtype,
            device=device,
        ),
        instance_ids=tuple(item.instance_id for item in records),
    )
