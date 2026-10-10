"""局部搜索语义、数值保护、Python/Numba/CUDA 同路径检查。"""

from dataclasses import replace

import numpy as np
import pytest

from gpaco.backends import cpu, cpu_python, cuda_backend, cuda_local_search
from gpaco.backends.local_search import (
    accepts,
    compiled_improve,
    improve,
    permutation,
    source_index,
)
from gpaco.config import ExecutionPlan, SearchConfig
from gpaco.data import coordinate_hash, prepare_problem, validate_tours
from gpaco.language import ProgramSpec


def fixture(n, batch=2):
    coords = np.random.default_rng(613).uniform(size=(batch, n, 2))
    tours = np.tile(np.r_[np.arange(n), 0], (batch, 1)).astype(np.int32)
    return prepare_problem(coords, tours, [coordinate_hash(c) for c in coords], min(n - 1, 20))


def reference(tour, data, b, mode, ant=0, interpreted=False):
    n = data.n
    order = np.empty(n, np.int32)
    permutation(order, np.uint64(313), data.instance_keys[b], 1, ant)
    result, stats = tour.copy().astype(np.int32), np.zeros(4, np.uint64)
    (improve if interpreted else compiled_improve)(
        result,
        data.distances[b],
        data.nearest[b],
        order,
        mode,
        data.nearest.shape[2],
        np.empty(n, np.int32),
        np.empty(n, np.uint8),
        np.empty(n, np.int32),
        stats,
    )
    return result, stats


def test_move_segment_mapping():
    # 三切边为(0,1),(2,3),(4,5)，中间两个片段均长2。
    expected = ([0, 2, 1, 4, 3, 5], [0, 3, 4, 1, 2, 5], [0, 4, 3, 1, 2, 5], [0, 3, 4, 2, 1, 5])
    for pattern, route in enumerate(expected):
        assert [source_index(x, 0, 2, 4, pattern) for x in range(6)] == route


def test_guard_rejects_ties_and_sub_ulp_gains():
    assert not accepts(np.float32(2), np.float32(2))
    assert not accepts(np.float32(2), np.nextafter(np.float32(2), np.float32(0)))
    assert accepts(np.float32(2), np.float32(1.99))


@pytest.mark.parametrize("mode", [1, 2])
def test_python_numba_local_moves_exact(mode):
    data = fixture(17)
    tour = np.r_[np.random.default_rng(23).permutation(data.n), 0].astype(np.int32)
    tour[-1] = tour[0]
    python_route, python_stats = reference(tour, data, 0, mode, interpreted=True)
    route, stats = reference(tour, data, 0, mode)
    np.testing.assert_array_equal(route, python_route)
    np.testing.assert_array_equal(stats, python_stats)
    validate_tours(route[None], data.n)
    d = data.distances[0].astype(np.float64)  # 离线检查，不进入搜索。
    assert d[route[:-1], route[1:]].sum() <= d[tour[:-1], tour[1:]].sum()
    again, second = reference(route, data, 0, mode)
    np.testing.assert_array_equal(again, route)
    assert second[:2].sum() == 0


@pytest.mark.parametrize("variant", ["as", "acs", "mmas"])
@pytest.mark.parametrize("mode", ["two_opt", "three_opt"])
def test_python_numba_closed_loop_ls(variant, mode):
    data = fixture(9)
    config = SearchConfig(
        variant=variant,
        local_search=mode,
        ls_candidate_size=8,
        candidate_size=8,
        ants=4,
        iterations=3,
    )
    programs = [ProgramSpec.parse("ZERO"), ProgramSpec.parse("ADD(DistRank, TurnCos)")]
    a = cpu.evaluate(programs, data, config, 121, ExecutionPlan(backend="cpu_existing"))
    b = cpu_python.evaluate(programs, data, config, 121, ExecutionPlan(backend="cpu_python"))
    np.testing.assert_allclose(a.lengths, b.lengths, rtol=1e-5, atol=1e-5)
    assert a.local_search_diagnostics[..., 0].sum() > 0
    validate_tours(a.tours, data.n)
    validate_tours(b.tours, data.n)


@pytest.mark.cuda
@pytest.mark.parametrize("n", [9, 100, 500])
@pytest.mark.parametrize("mode", [1, 2])
def test_cuda_local_moves_exact(n, mode):
    import os

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("资格检查需显式分配空闲GPU")
    cp = pytest.importorskip("cupy")
    data = fixture(n)
    ants = 3
    rng = np.random.default_rng(91)
    tours = np.array(
        [[np.r_[rng.permutation(n), 0] for _ in range(ants)] for _ in range(2)], np.uint16
    )
    tours[:, :, -1] = tours[:, :, 0]
    expected = np.empty_like(tours)
    statistics = np.empty((2, ants, 4), np.uint64)
    for b in range(2):
        for ant in range(ants):
            expected[b, ant], statistics[b, ant] = reference(tours[b, ant], data, b, mode, ant)
    for executor in ("scalar", "cooperative"):
        device_tours, lengths = cp.asarray(tours), cp.empty((2, ants), cp.float32)
        buffers = cuda_local_search.workspace(2, ants, n)
        kernel, _, _ = cuda_local_search.kernel(executor)
        cuda_local_search.launch(
            kernel,
            executor,
            cp.asarray(data.distances),
            cp.asarray(data.nearest),
            cp.arange(2, dtype=cp.int32),
            cp.asarray(data.instance_keys),
            2,
            n,
            ants,
            1,
            313,
            mode,
            20,
            device_tours,
            lengths,
            buffers,
        )
        np.testing.assert_array_equal(cp.asnumpy(device_tours), expected)
        np.testing.assert_array_equal(cp.asnumpy(buffers[-1]).reshape(2, ants, 4), statistics)
        wanted = np.array(
            [
                [
                    np.cumsum(data.distances[b][r[:-1], r[1:]], dtype=np.float32)[-1]
                    for r in expected[b]
                ]
                for b in range(2)
            ],
            np.float32,
        )
        np.testing.assert_array_equal(cp.asnumpy(lengths), wanted)


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["as", "acs", "mmas"])
@pytest.mark.parametrize("mode", ["two_opt", "three_opt"])
def test_cuda_closed_loop_ls(variant, mode):
    import os

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("资格检查需显式分配空闲GPU")
    pytest.importorskip("cupy")
    data = fixture(9)
    config = SearchConfig(
        variant=variant,
        local_search=mode,
        ls_candidate_size=8,
        candidate_size=8,
        ants=4,
        iterations=3,
    )
    programs = [ProgramSpec.parse("ZERO"), ProgramSpec.parse("ADD(DistRank, TurnCos)")]
    plan = ExecutionPlan(active_tasks=2, generated=True, ls_executor="scalar")
    a = cuda_backend.evaluate(programs, data, config, 121, plan, capture_state=True)
    b = cuda_backend.evaluate(
        programs,
        data,
        config,
        121,
        replace(plan, ls_executor="cooperative", profile_stages=True),
        capture_state=True,
    )
    c = cpu.evaluate(programs, data, config, 121, ExecutionPlan(backend="cpu_existing"))
    np.testing.assert_array_equal(a.tours, b.tours)
    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.local_search_diagnostics, b.local_search_diagnostics)
    np.testing.assert_allclose(a.lengths, c.lengths, atol=1e-5, rtol=1e-5)
    assert b.timings["local_search_device_s"] > 0
    for state in b.state_capture:
        validate_tours(state["colony_tours"], data.n)
