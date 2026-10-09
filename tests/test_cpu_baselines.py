"""六组 CPU 基线的功能合同；这些小测试不作为性能实验。"""

import inspect

import numpy as np
import pytest
from test_core import problem

from gpaco.backends import cpu, cpu_python
from gpaco.backends.numeric import counter_uniform
from gpaco.config import ExecutionPlan, SearchConfig
from gpaco.data import validate_tours
from gpaco.language import TERMINALS, ProgramSpec


def test_python_rng_is_exact():
    for seed in (0, 17, 2**64 - 1):
        actual = cpu_python.uniform(seed, 917, 500, np.arange(4)[:, None], np.arange(5), 3)
        expected = [
            [counter_uniform(np.uint64(seed), np.uint64(917), 500, a, s, 3) for s in range(5)]
            for a in range(4)
        ]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float32


@pytest.mark.parametrize("variant", ["as", "acs", "mmas"])
def test_python_closed_loop_and_initialization(variant):
    data = problem(n=9, batch=2)
    config = SearchConfig(
        variant=variant,
        ants=4,
        iterations=8,
        candidate_size=3,
        branch_period=1,
        restart_stagnation=0,
        branch_threshold=100,
    )
    programs = [ProgramSpec.parse(x) for x in ("ZERO", *TERMINALS, "PDIV(RTau, REta)")]
    for a, b in zip(
        cpu_python.initialization(data, config, 17),
        cpu.problem_initialization(data, config, 17),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=1e-6)
    reference = cpu.evaluate(programs, data, config, 17, ExecutionPlan(backend="cpu_existing"))
    got = cpu_python.evaluate(programs, data, config, 17, ExecutionPlan(backend="cpu_python"))
    validate_tours(got.tours, data.n)
    np.testing.assert_allclose(got.lengths, reference.lengths, atol=1e-5, rtol=1e-4)
    assert got.lengths.dtype == np.float32
    assert got.timings["executed_tasks"] == len(programs) * data.size
    if variant == "mmas":
        assert got.diagnostics[..., 3].sum() > 0


def test_python_process_count_and_order_invariant():
    data = problem(n=6, batch=3)
    programs = [ProgramSpec.parse("ZERO"), ProgramSpec.parse("ADD(DistRank, TurnCos)")]
    config = SearchConfig(ants=3, iterations=2, candidate_size=3)
    first = cpu_python.evaluate(programs, data, config, 21, ExecutionPlan(backend="cpu_python"))
    second = cpu_python.evaluate(
        programs[::-1], data, config, 21, ExecutionPlan(backend="cpu_python", cpu_threads=2)
    )
    np.testing.assert_array_equal(first.tours, second.tours[::-1])
    np.testing.assert_array_equal(first.lengths, second.lengths[::-1])
    # 拒绝最外层 Python、内层仍调用 dispatcher 的伪“无 Numba”基线。
    assert ".py_func" not in inspect.getsource(cpu_python.solve_instance)


@pytest.mark.slow
@pytest.mark.parametrize("cores", [1, 8, 16])
def test_requested_physical_core_configurations(cores):
    """小预算功能验证，不作为这些核数的性能测量。"""
    import os

    from gpaco.cpu_benchmark import physical_cpu_ids

    available = physical_cpu_ids()
    if len(available) < cores:
        pytest.skip("物理核不足；禁止以SMT替代")
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, available[:cores])
    try:
        data = problem(n=5, batch=16)
        programs = [ProgramSpec.parse("ZERO")]
        config = SearchConfig(ants=2, iterations=2, candidate_size=3)
        for backend, implementation in [("cpu_python", cpu_python), ("cpu_existing", cpu)]:
            one = implementation.evaluate(
                programs, data, config, 918, ExecutionPlan(backend=backend, cpu_threads=1)
            )
            many = implementation.evaluate(
                programs, data, config, 918, ExecutionPlan(backend=backend, cpu_threads=cores)
            )
            np.testing.assert_array_equal(one.tours, many.tours)
            np.testing.assert_array_equal(one.lengths, many.lengths)
    finally:
        os.sched_setaffinity(0, previous)


def test_numpy_fields_match_numba_local_values():
    from gpaco.backends.cpu_existing import _prepare_transition_scores
    from gpaco.language import TERMINAL_IDS, evaluate_reference, pack_programs

    data = problem(n=9)
    geometry = tuple(
        getattr(data, f)[0]
        for f in ("coords", "distances", "heuristic", "log_heuristic", "nearest", "full_nn_rank")
    )
    tau = np.full((9, 9), 0.17, np.float32)
    np.fill_diagonal(tau, 0)
    config = SearchConfig(ants=4, iterations=7, candidate_size=3)
    for name in TERMINALS:
        p = ProgramSpec.parse(name)
        ops, fargs, iargs, _, masks, _ = pack_programs([p])
        for fallback, candidates in [
            (True, np.array([1, 3, 6, 8], np.int32)),
            (False, data.nearest[0, 0].astype(np.int32)),
        ]:
            m = len(candidates)
            terminals, scores, scratch, stack = (
                np.zeros((16, 9), np.float32),
                np.empty(9, np.float32),
                np.empty(9, np.float32),
                np.empty((31, 9), np.float32),
            )
            _prepare_transition_scores(
                geometry[0],
                geometry[1],
                geometry[2],
                geometry[3],
                tau,
                0,
                2,
                geometry[5],
                candidates,
                m,
                fallback,
                np.float32(1),
                np.float32(2),
                np.float32(1e-12),
                0,
                np.float32(config.gamma),
                True,
                ops[0],
                fargs[0],
                iargs[0],
                masks[0],
                3,
                4,
                2,
                7,
                terminals,
                scores,
                scratch,
                stack,
            )
            context, _, _ = cpu_python.fields(
                p, geometry, tau[0], 0, 2, candidates, fallback, 3, 4, 2, config
            )
            expected = terminals[TERMINAL_IDS[name], :m]
            if name in ("Entropy", "ConstructProg", "ACOProg", "Stagnation"):
                expected = np.full(m, expected[0], np.float32)
            np.testing.assert_allclose(
                evaluate_reference(p, context, (m,)), expected, atol=1e-5, rtol=1e-4
            )
