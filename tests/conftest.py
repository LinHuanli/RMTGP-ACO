"""测试使用的小型、确定性 Euclidean TSP 实例。"""

from __future__ import annotations

import numpy as np
import pytest

from rmtgp_aco.data import TSPInstance, coordinate_hash


def make_instance(n: int, seed: int = 0) -> TSPInstance:
    """生成一个带合法闭合参考 tour 的小型实例。"""

    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2), dtype=np.float64)
    body = np.arange(n, dtype=np.int64)
    tour = np.concatenate((body, body[:1]))
    delta = coords[tour[1:]] - coords[tour[:-1]]
    length = float(np.sqrt(np.sum(delta * delta, axis=1)).sum())
    return TSPInstance(
        instance_id=f"synthetic-n{n}-seed{seed}",
        coords=coords,
        reference_tour=tour,
        reference_length=length,
        coordinate_hash=coordinate_hash(coords),
    )


@pytest.fixture
def small_instances() -> list[TSPInstance]:
    """两个同规模实例，用于覆盖 batch 维。"""

    return [make_instance(6, 11), make_instance(6, 29)]
