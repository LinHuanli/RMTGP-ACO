"""在指定的一张可见 GPU 上检查 E00，profiling 与性能实验另行执行。"""

import os

import numpy as np
import pytest
from test_core import problem

from gpaco.backends import cpu, cuda_backend
from gpaco.backends.numeric import philox
from gpaco.config import ExecutionPlan, SearchConfig
from gpaco.data import validate_tours
from gpaco.language import TERMINAL_IDS, ProgramSpec, evaluate_reference, pack_programs

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def cp():
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("必须显式选择已确认空闲的 CUDA_VISIBLE_DEVICES，不自动占用 GPU0")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("无 CUDA GPU")
    return cp


def test_cuda_rng_exact(cp):
    config, plan = SearchConfig(), ExecutionPlan()
    kernels, _, _ = cuda_backend.kernels(config, plan, [ProgramSpec.parse("ZERO")])
    coordinates = np.asarray([[0, 0, 0, 0], [1, 2, 3, 4], [500, 31, 999, 3]], np.int32)
    output = cp.empty((3, 4), cp.uint32)
    kernels["probe_rng"](
        (1,), (32,), (np.uint64(17), np.uint64(22), cp.asarray(coordinates), output, np.int32(3))
    )
    expected = np.asarray(
        [philox(np.uint64(17), np.uint64(22), *map(int, row)) for row in coordinates], np.uint32
    )
    np.testing.assert_array_equal(cp.asnumpy(output), expected)


@pytest.mark.parametrize(
    "expression",
    [
        "ADD(RTau, REta)",
        "SUB(BaseConf, Entropy)",
        "MUL(DistRank, TurnCos)",
        "PDIV(RTau, REta)",
        "MIN(ACOProg, Stagnation)",
        "MAX(MutualRank, ConstructProg)",
        "NEG(ABS(TurnCos))",
    ],
)
def test_cuda_primitives(cp, expression):
    program = ProgramSpec.parse(expression)
    features = np.random.default_rng(9).uniform(-10, 10, (64, 16)).astype(np.float32)
    features[0] = 0
    features[1, ::2] = np.nan
    features[2, ::2] = np.inf
    features[3, 1::2] = -np.inf
    kernels, _, _ = cuda_backend.kernels(SearchConfig(), ExecutionPlan(), [program])
    packed = pack_programs([program])
    output = cp.empty(64, cp.float32)
    kernels["probe_program"](
        (1,),
        (64,),
        (
            *[cp.asarray(v) for v in packed[:3]],
            np.int32(len(program.instructions)),
            cp.asarray(features),
            np.int32(64),
            output,
        ),
    )
    expected = evaluate_reference(
        program, {name: features[:, idx] for name, idx in TERMINAL_IDS.items()}, (64,)
    )
    np.testing.assert_allclose(cp.asnumpy(output), expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("variant", ["as", "acs", "mmas"])
@pytest.mark.parametrize("lanes", [4, 8, 16, 32])
def test_cuda_closed_loop(cp, variant, lanes):
    data = problem(n=9)
    programs = [ProgramSpec.parse(x) for x in ["ZERO", "ADD(DistRank, TurnCos)", "SUB(RTau, REta)"]]
    config = SearchConfig(variant=variant, ants=4, iterations=3, candidate_size=3)
    result = cuda_backend.evaluate(
        programs, data, config, 17, ExecutionPlan(candidate_lanes=lanes, active_tasks=2)
    )
    reference = cpu.evaluate(programs, data, config, 17, ExecutionPlan(backend="cpu_existing"))
    validate_tours(result.tours, data.n)
    # 小型确定性样例选择远离概率阈值；闭环一致不是一般浮点轨迹保证。
    np.testing.assert_allclose(result.lengths, reference.lengths, atol=1e-5, rtol=1e-4)
    assert result.timings["executed_tasks"] == 6
    assert result.timings["waves"] == 3


def test_generated_matches_interpreter(cp):
    data = problem()
    programs = [ProgramSpec.parse(x) for x in ["PDIV(RTau, REta)", "ADD(BaseConf, TurnCos)"]]
    config = SearchConfig(ants=8, iterations=4, candidate_size=3)
    first = cuda_backend.evaluate(programs, data, config, 81, ExecutionPlan(generated=False))
    second = cuda_backend.evaluate(programs, data, config, 81, ExecutionPlan(generated=True))
    np.testing.assert_array_equal(first.tours, second.tours)
    np.testing.assert_array_equal(first.lengths, second.lengths)


@pytest.mark.parametrize("variant", ["as", "acs", "mmas"])
def test_task_mapping_and_pheromone_invariants(cp, variant):
    data = problem()
    programs = [ProgramSpec.parse(name) for name in TERMINAL_IDS]
    config = SearchConfig(
        variant=variant,
        ants=8,
        iterations=12,
        candidate_size=3,
        branch_period=1,
        branch_threshold=100,
        restart_stagnation=0,
    )
    first = cuda_backend.evaluate(
        programs,
        data,
        config,
        417,
        ExecutionPlan(active_tasks=20, generated=True),
        capture_state=True,
    )
    second = cuda_backend.evaluate(
        programs[::-1], data, config, 417, ExecutionPlan(active_tasks=3, generated=True)
    )
    np.testing.assert_array_equal(first.tours, second.tours[::-1])
    np.testing.assert_array_equal(first.lengths, second.lengths[::-1])
    for snapshot in first.state_capture:
        tau = snapshot["tau"]
        np.testing.assert_array_equal(tau, tau.swapaxes(-1, -2))
        np.testing.assert_array_equal(np.diagonal(tau, axis1=-1, axis2=-2), 0)
        assert np.isfinite(tau).all() and (tau >= 0).all()
        if variant == "mmas":
            mask = ~np.eye(data.n, dtype=bool)
            assert (tau[:, mask] >= snapshot["state"][:, 2, None]).all()
            assert (tau[:, mask] <= snapshot["state"][:, 3, None]).all()
    if variant == "mmas":
        assert first.diagnostics[..., 3].sum() > 0


@pytest.mark.parametrize("variant", ["as", "acs", "mmas"])
def test_fixed_update_against_independent_numpy(cp, variant):
    config = SearchConfig(variant=variant, ants=4, iterations=1, candidate_size=3)
    functions, _, _ = cuda_backend.kernels(config, ExecutionPlan(), [ProgramSpec.parse("ZERO")])
    n, ants = 5, 4
    routes = np.asarray(
        [[0, 1, 2, 3, 4, 0], [0, 1, 3, 2, 4, 0], [1, 2, 3, 4, 0, 1], [0, 2, 1, 3, 4, 0]], np.uint16
    )
    lengths = np.asarray([5, 7, 5, 6], np.float32)
    tau = np.full((n, n), 0.3, np.float32)
    np.fill_diagonal(tau, 0)
    gpu_tau = cp.asarray(tau)
    state = cp.asarray([np.inf, np.inf, 0, np.inf], dtype=cp.float32)
    best, restart = cp.empty(n + 1, cp.uint16), cp.empty(n + 1, cp.uint16)
    diagnostics = cp.zeros(8, cp.uint64)
    args = (
        cp.zeros(n * 3, cp.uint16),
        cp.zeros(1, cp.int32),
        *map(np.int32, (1, n, 3, ants, 1)),
        np.float32(config.rho),
        np.int32(25),
        np.float32(0.05),
        np.int32(100),
        np.float32(0.05),
        np.float32(1.00001),
        np.int32(250),
        gpu_tau,
        cp.asarray(routes),
        cp.asarray(lengths),
        cp.empty((n, n), cp.float32),
        best,
        restart,
        state,
        cp.zeros(2, cp.int32),
        diagnostics,
    )
    functions["update"]((1,), (256,), args)
    one, rho = np.float32(1), np.float32(config.rho)
    if variant == "acs":
        for u, v in zip(routes[0, :-1], routes[0, 1:], strict=True):
            tau[u, v] = tau[v, u] = (one - rho) * tau[u, v] + rho * (one / lengths[0])
    else:
        deposits = np.zeros_like(tau)
        for a in range(ants if variant == "as" else 1):
            for u, v in zip(routes[a, :-1], routes[a, 1:], strict=True):
                deposits[u, v] += one / lengths[a]
                deposits[v, u] += one / lengths[a]
        tau = (one - rho) * tau + deposits
        if variant == "mmas":
            high = one / (rho * lengths[0])
            px = np.exp(np.log(np.float32(0.05)) / np.float32(n))
            low = high * (one - px) / (px * np.float32(2))
            tau = np.clip(tau, low, high)
        np.fill_diagonal(tau, 0)
    np.testing.assert_allclose(cp.asnumpy(gpu_tau), tau, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(cp.asnumpy(best), routes[0])
