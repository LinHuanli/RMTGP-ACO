"""不可变数组数据接口。文本解析、标签检查与运行时加载分离。"""

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("GPACO_ROOT", Path(__file__).resolve().parents[2])).resolve()


def coordinate_hash(coords):
    return sha256(np.asarray(coords, dtype="<f8").tobytes()).hexdigest()


def parse_line(line):
    left, separator, right = line.partition(" output ")
    if not separator:
        raise ValueError("缺少唯一 output 分隔符")
    coords = np.fromstring(left, dtype=np.float64, sep=" ")
    if coords.size % 2 or coords.size < 4 or not np.isfinite(coords).all():
        raise ValueError("坐标不合法")
    coords = coords.reshape(-1, 2)
    tour = np.fromstring(right, dtype=np.int32, sep=" ") - 1
    validate_tours(tour[None], len(coords))
    return coords, tour


def validate_tours(tours, n):
    tours = np.asarray(tours)
    if tours.shape[-1] != n + 1:
        raise ValueError("路径必须显式闭合，长度为 n+1")
    if not np.all(tours[..., 0] == tours[..., -1]):
        raise ValueError("路径未闭合")
    if not np.all(np.sort(tours[..., :-1], axis=-1) == np.arange(n)):
        raise ValueError("路径不是 Hamiltonian cycle")


def tour_length(coords, tour, dtype=np.float64):
    points = np.asarray(coords, dtype=dtype)[tour]
    delta = points[1:] - points[:-1]
    return np.sqrt(np.sum(delta * delta, axis=-1, dtype=dtype)).sum(dtype=dtype)


@dataclass
class ProblemSpec:
    coords: np.ndarray
    distances: np.ndarray
    heuristic: np.ndarray
    log_heuristic: np.ndarray
    nearest: np.ndarray
    full_nn_rank: np.ndarray
    reference: np.ndarray
    instance_keys: np.ndarray
    instance_ids: tuple[str, ...]

    @property
    def n(self):
        return self.coords.shape[1]

    @property
    def size(self):
        return self.coords.shape[0]


def prepare_problem(coords, tours, instance_ids, candidate_size=20):
    """所有搜索几何由同一 FP32 坐标计算，距离平局按城市编号打破。"""
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[-1] != 2 or not np.isfinite(coords).all():
        raise ValueError("坐标必须是有限的 [B,n,2] 数组")
    batch, n, _ = coords.shape
    if batch < 1 or len(instance_ids) != batch or candidate_size < 1:
        raise ValueError("空批次、实例标识数量不匹配或候选表尺寸无效")
    if not 3 <= n < 65536:
        raise ValueError("本实现要求 3<=n<65536")
    difference = coords[:, :, None, :] - coords[:, None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=-1, dtype=np.float32))
    heuristic = np.float32(1) / np.maximum(distance, np.float32(1e-12))
    heuristic[:, np.arange(n), np.arange(n)] = np.float32(0)
    log_eta = np.log(np.maximum(heuristic, np.float32(1e-12)))
    sortable = distance.copy()
    sortable[:, np.arange(n), np.arange(n)] = np.float32(np.inf)
    order = np.argsort(sortable, axis=-1, kind="stable").astype(np.uint16)
    nearest = np.ascontiguousarray(order[:, :, : min(candidate_size, n - 1)])
    ranks = np.empty((batch, n, n), np.uint16)
    np.put_along_axis(
        ranks, order.astype(np.int64), np.arange(1, n + 1, dtype=np.uint16)[None, None, :], axis=2
    )
    reference = np.asarray(
        [tour_length(c, t, np.float32) for c, t in zip(coords, tours, strict=True)], np.float32
    )
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("FP32 标签路径长度必须为有限正数")
    keys = np.asarray([int(s[:16], 16) for s in instance_ids], np.uint64)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("instance_key 重复；不能让不同实例共享随机命名空间")
    return ProblemSpec(
        coords, distance, heuristic, log_eta, nearest, ranks, reference, keys, tuple(instance_ids)
    )


def load_split(n, split, indices=None):
    if not (ROOT / "Datasets/processed/v1" / f"tsp{n}" / "COMPLETE.json").exists():
        raise FileNotFoundError("派生数据尚未完整提交，禁止读取部分划分")
    path = ROOT / "Datasets/processed/v1" / f"tsp{n}" / split
    coords = np.load(path / "coords_fp32.npy", mmap_mode="r")
    tours = np.load(path / "reference_tours.npy", mmap_mode="r")
    ids = np.load(path / "instance_ids.npy", mmap_mode="r")
    if indices is None:
        indices = np.arange(len(coords))
    return (
        np.asarray(coords[indices]),
        np.asarray(tours[indices]),
        tuple(str(x) for x in ids[indices]),
    )


def write_json(path, value):
    """运行产物原子替换；避免后台任务读到半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
