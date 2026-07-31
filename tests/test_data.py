"""数据解析、随机访问与 batch 预计算测试。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from rmtgp_aco.data import (
    build_line_offsets,
    load_indexed_instances,
    make_problem_batch,
    parse_tsp_line,
)
from rmtgp_aco.sampling import IndexedShard, ScalePool

SQUARE = "0 0 1 0 1 1 0 1 output 1 2 3 4 1"


def test_parse_line_recomputes_continuous_reference_length() -> None:
    instance = parse_tsp_line(SQUARE, instance_id="square")
    assert instance.n == 4
    assert instance.reference_length == pytest.approx(4.0)
    np.testing.assert_array_equal(instance.reference_tour, [0, 1, 2, 3, 0])


@pytest.mark.parametrize(
    "line",
    [
        "0 0 1 0 1 1 0 1 1 2 3 4 1",
        "0 0 1 0 1 1 0 1 output 1 2 2 4 1",
        "0 0 1 0 1 1 0 1 output 1 2 3 4 2",
    ],
)
def test_invalid_records_are_rejected(line: str) -> None:
    with pytest.raises(ValueError):
        parse_tsp_line(line, instance_id="bad")


def test_indexed_shard_and_slots_scale_pool(tmp_path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text(f"{SQUARE}\n{SQUARE}\n", encoding="utf-8")
    offsets = build_line_offsets(path)
    assert offsets.tolist() == [0, len(SQUARE) + 1]
    records = load_indexed_instances(path, offsets, [1, 0])
    assert len(records) == 2

    pool = ScalePool(scale=4, shards=(IndexedShard.open(path),))
    assert len(pool) == 2
    assert pool.get(1).n == 4


def test_indexed_shard_cache_creation_is_concurrency_safe(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "shared.txt"
    path.write_text(f"{SQUARE}\n{SQUARE}\n", encoding="utf-8")
    barrier = threading.Barrier(4)
    original = build_line_offsets

    def synchronized_build(source):
        offsets = original(source)
        barrier.wait(timeout=5.0)
        return offsets

    monkeypatch.setattr(
        "rmtgp_aco.sampling.build_line_offsets",
        synchronized_build,
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        shards = list(
            executor.map(
                lambda _: IndexedShard.open(path),
                range(4),
            )
        )
    assert all(
        shard.offsets.tolist() == [0, len(SQUARE) + 1]
        for shard in shards
    )
    assert IndexedShard.open(path).offsets.tolist() == [
        0,
        len(SQUARE) + 1,
    ]


def test_problem_batch_is_symmetric_and_has_valid_candidates(small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=3)
    assert batch.coords.dtype == torch.float64
    assert batch.nn_indices.shape == (2, 6, 3)
    torch.testing.assert_close(batch.distances, batch.distances.transpose(1, 2))
    assert torch.all(torch.diagonal(batch.distances, dim1=1, dim2=2) == 0)
    rows = torch.arange(6).reshape(1, 6, 1)
    assert not bool(torch.any(batch.nn_indices == rows))
