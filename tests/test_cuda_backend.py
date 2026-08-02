"""融合 CUDA 后端的 tour、确定性、分片和 GP 语义测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from deap import gp

from rmtgp_aco.aco_cuda import (
    clear_cuda_kernel_cache,
    clear_cuda_problem_cache,
    cuda_available,
    cuda_cache_snapshot,
    cuda_device_count,
    solve_population_cuda,
    solve_population_cuda_anytime,
)
from rmtgp_aco.aco_numba import _pack_programs
from rmtgp_aco.config import (
    ACOConfig,
    ACOVariant,
    CudaPrecision,
    CudaTaskOrder,
    ExecutionBackend,
    ExperimentConfig,
    GPConfig,
    GPUMode,
    LocalSearch,
    LSGainSemantics,
    RuntimeConfig,
)
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.genetic import initialise_population
from rmtgp_aco.program import (
    ORIGIN_PHEROMONE_TERMINALS,
    compile_tree,
    constant_zero_tree,
    create_primitive_sets,
)
from rmtgp_aco.sampling import in_memory_cases
from rmtgp_aco.training import BaselineCache, EvaluationPool, train

pytestmark = pytest.mark.skipif(
    not cuda_available(),
    reason="当前节点没有可用 CUDA device",
)


def _runtime(
    mode: GPUMode,
    *,
    devices: tuple[int, ...] = (0, 1),
    chunk_size: int = 0,
    block_threads: int = 0,
) -> RuntimeConfig:
    return RuntimeConfig(
        aco_backend=ExecutionBackend.CUDA_FUSED_FP32,
        gpu_mode=mode,
        gpu_devices=devices,
        gpu_task_chunk_size=chunk_size,
        gpu_block_threads=block_threads,
    )


def _runtime_v2(
    *,
    precision: CudaPrecision = CudaPrecision.FP32_FAST,
    lanes: int = 8,
    chunk_size: int = 0,
    task_order: CudaTaskOrder = CudaTaskOrder.INSTANCE_MAJOR,
    generated_gp: bool = True,
) -> RuntimeConfig:
    return RuntimeConfig(
        aco_backend=ExecutionBackend.CUDA_TILED_V2,
        gpu_mode=GPUMode.SINGLE,
        gpu_devices=(0,),
        gpu_task_chunk_size=chunk_size,
        cuda_precision=precision,
        cuda_candidate_lanes=lanes,
        cuda_register_cap=0,
        cuda_task_order=task_order,
        cuda_generated_gp=generated_gp,
    )


def _assert_valid_population_tours(tours: torch.Tensor) -> None:
    population, batch, n_plus_one = tours.shape
    n = n_plus_one - 1
    flat = tours.reshape(population * batch, n_plus_one)
    assert torch.equal(flat[:, 0], flat[:, -1])
    expected = torch.arange(n).expand(flat.shape[0], n)
    assert torch.equal(torch.sort(flat[:, :-1], dim=1).values, expected)


@pytest.mark.parametrize("variant", list(ACOVariant))
def test_cuda_zero_residual_recovers_internal_baseline(
    variant,
    small_instances,
) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default(variant, iterations=3),
        ants=4,
        candidate_size=2,
    )
    transition_pset, pheromone_pset = create_primitive_sets()
    zero_transition = compile_tree(
        constant_zero_tree(transition_pset, "ZERO_TR"),
        role="transition",
    )
    zero_pheromone = compile_tree(
        constant_zero_tree(pheromone_pset, "ZERO_PH"),
        role="pheromone",
    )
    result = solve_population_cuda(
        batch,
        config,
        [(None, None), (zero_transition, zero_pheromone)],
        seed=909,
        runtime=_runtime(GPUMode.SINGLE, devices=(0,)),
    )
    assert torch.equal(result.best_tour[0], result.best_tour[1])
    assert torch.equal(result.best_length[0], result.best_length[1])
    assert torch.equal(result.best_iteration[0], result.best_iteration[1])
    _assert_valid_population_tours(result.best_tour)


def test_cuda_population_anytime_matches_single_program_path(
    small_instances,
) -> None:
    """批量锁定模型测试必须保留与单 program 路径相同的完整轨迹。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("as", iterations=4),
        ants=4,
        candidate_size=2,
    )
    runtime = _runtime(GPUMode.SINGLE, devices=(0,))
    result = solve_population_cuda_anytime(
        batch,
        config,
        [(None, None), (None, None)],
        seed=211,
        runtime=runtime,
    )
    assert result.anytime_best.shape == (2, 2, 4)
    assert torch.equal(result.best_tour[0], result.best_tour[1])
    assert torch.equal(result.best_length[0], result.best_length[1])
    assert torch.equal(result.anytime_best[0], result.anytime_best[1])


def test_cuda_v2_anytime_scalar_matches_full_curve(
    small_instances,
) -> None:
    """训练期设备端标量应数值等价于完整 best-so-far 曲线均值。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "as",
            local_search=LocalSearch.TWO_OPT,
            iterations=4,
        ),
        ants=32,
        candidate_size=2,
        local_search_candidate_size=2,
    )
    runtime = _runtime_v2()
    quality = solve_population_cuda(
        batch,
        config,
        [(None, None)],
        seed=817,
        runtime=runtime,
    )
    full = solve_population_cuda_anytime(
        batch,
        config,
        [(None, None)],
        seed=817,
        runtime=runtime,
    )
    assert quality.anytime_mean_length is not None
    torch.testing.assert_close(
        quality.anytime_mean_length,
        full.anytime_best.mean(dim=2),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(cuda_device_count() < 2, reason="需要两张可见 CUDA GPU")
def test_cuda_single_dual_device_and_chunk_invariance(small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("acs", iterations=4),
        ants=4,
        candidate_size=2,
    )
    transition_pset, pheromone_pset = create_primitive_sets()
    transition = compile_tree(
        gp.PrimitiveTree.from_string(
            "MAX(PDIV(ADD(RTau, REta), BaseConf), DistRank)",
            transition_pset,
        ),
        role="transition",
    )
    pheromone = compile_tree(
        gp.PrimitiveTree.from_string(
            "MAX(EdgeEta, ColonyFreq)",
            pheromone_pset,
        ),
        role="pheromone",
    )
    programs = [(None, None), (transition, pheromone)]
    single = solve_population_cuda(
        batch,
        config,
        programs,
        seed=731,
        runtime=_runtime(GPUMode.SINGLE),
    )
    repeated = solve_population_cuda(
        batch,
        config,
        programs,
        seed=731,
        runtime=_runtime(GPUMode.SINGLE),
    )
    dual = solve_population_cuda(
        batch,
        config,
        programs,
        seed=731,
        runtime=_runtime(GPUMode.DUAL),
    )
    chunked = solve_population_cuda(
        batch,
        config,
        programs,
        seed=731,
        runtime=_runtime(GPUMode.DUAL, chunk_size=1),
    )
    block64 = solve_population_cuda(
        batch,
        config,
        programs,
        seed=731,
        runtime=_runtime(
            GPUMode.SINGLE,
            devices=(0,),
            block_threads=64,
        ),
    )
    for other in (repeated, dual, chunked, block64):
        assert torch.equal(single.best_tour, other.best_tour)
        assert torch.equal(single.best_length, other.best_length)
        assert torch.equal(single.best_iteration, other.best_iteration)
        assert torch.equal(single.diagnostics, other.diagnostics)


def test_cuda_mmas_full_restart_is_audited(small_instances) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("mmas", iterations=4),
        ants=4,
        candidate_size=2,
        mmas_branch_check_period=2,
        mmas_restart_stagnation=0,
        mmas_branch_threshold=100.0,
    )
    result = solve_population_cuda(
        batch,
        config,
        [(None, None)],
        seed=13,
        runtime=_runtime(GPUMode.SINGLE, devices=(0,)),
    )
    assert int(result.diagnostics[0, 3].item()) > 0


@pytest.mark.parametrize("variant", list(ACOVariant))
def test_cuda_v2_zero_residual_and_repeatability(
    variant,
    small_instances,
) -> None:
    """tiled v2 必须保留 zero residual 语义并可精确重复。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default(variant, iterations=4),
        ants=32,
        candidate_size=2,
    )
    transition_pset, pheromone_pset = create_primitive_sets()
    zero_transition = compile_tree(
        constant_zero_tree(transition_pset, "ZERO_TR"),
        role="transition",
    )
    zero_pheromone = compile_tree(
        constant_zero_tree(pheromone_pset, "ZERO_PH"),
        role="pheromone",
    )
    programs = [(None, None), (zero_transition, zero_pheromone)]
    first = solve_population_cuda(
        batch,
        config,
        programs,
        seed=177,
        runtime=_runtime_v2(),
    )
    repeated = solve_population_cuda(
        batch,
        config,
        programs,
        seed=177,
        runtime=_runtime_v2(chunk_size=1),
    )
    assert torch.equal(first.best_tour[0], first.best_tour[1])
    assert torch.equal(first.best_length[0], first.best_length[1])
    assert torch.equal(first.best_tour, repeated.best_tour)
    assert torch.equal(first.best_length, repeated.best_length)
    assert torch.equal(first.best_iteration, repeated.best_iteration)
    assert torch.equal(first.diagnostics, repeated.diagnostics)
    _assert_valid_population_tours(first.best_tour)


@pytest.mark.parametrize(
    "precision",
    [
        CudaPrecision.FP32_FAST,
        CudaPrecision.FP16_MIXED,
        CudaPrecision.BF16_MIXED,
        CudaPrecision.FP16_SEARCH,
    ],
)
def test_cuda_v2_precision_profiles_return_valid_tours(
    precision,
    small_instances,
) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("acs", iterations=3),
        ants=32,
        candidate_size=2,
    )
    result = solve_population_cuda(
        batch,
        config,
        [(None, None)],
        seed=313,
        runtime=_runtime_v2(precision=precision),
    )
    _assert_valid_population_tours(result.best_tour)
    assert torch.isfinite(result.best_length).all()
    assert result.backend_metrics["precision"] == precision.value


@pytest.mark.parametrize("variant", list(ACOVariant))
def test_cuda_v2_generated_gp_is_bitwise_interpreter_equivalent(
    variant,
    small_instances,
) -> None:
    """生成式 GP 只能改变执行方式，不得改变任一搜索状态。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default(variant, iterations=4),
        ants=32,
        candidate_size=2,
    )
    transition_pset, pheromone_pset = create_primitive_sets()
    transition = compile_tree(
        gp.PrimitiveTree.from_string(
            "MAX(PDIV(ADD(RTau, REta), BaseConf), DistRank)",
            transition_pset,
        ),
        role="transition",
    )
    pheromone = compile_tree(
        gp.PrimitiveTree.from_string(
            "MAX(EdgeEta, ColonyFreq)",
            pheromone_pset,
        ),
        role="pheromone",
    )
    programs = [(None, None), (transition, pheromone)]
    interpreted = solve_population_cuda(
        batch,
        config,
        programs,
        seed=419,
        runtime=_runtime_v2(generated_gp=False),
    )
    generated = solve_population_cuda(
        batch,
        config,
        programs,
        seed=419,
        runtime=_runtime_v2(generated_gp=True),
    )
    assert torch.equal(interpreted.best_tour, generated.best_tour)
    assert torch.equal(interpreted.best_length, generated.best_length)
    assert torch.equal(interpreted.best_iteration, generated.best_iteration)
    assert torch.equal(interpreted.diagnostics, generated.diagnostics)


def test_cuda_v2_generated_module_cache_is_bounded(small_instances) -> None:
    """不同 population 的生成式 module 不得跨代无限驻留显存。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("as", iterations=1),
        ants=32,
        candidate_size=2,
    )
    transition_pset, pheromone_pset = create_primitive_sets()
    clear_cuda_problem_cache()
    clear_cuda_kernel_cache()
    try:
        for index in range(12):
            transition_expression = "RTau"
            pheromone_expression = "EdgeEta"
            for _ in range(index + 1):
                transition_expression = f"ADD({transition_expression}, ZERO_TR)"
                pheromone_expression = f"ADD({pheromone_expression}, ZERO_PH)"
            transition = compile_tree(
                gp.PrimitiveTree.from_string(
                    transition_expression,
                    transition_pset,
                ),
                role="transition",
            )
            pheromone = compile_tree(
                gp.PrimitiveTree.from_string(
                    pheromone_expression,
                    pheromone_pset,
                ),
                role="pheromone",
            )
            result = solve_population_cuda(
                batch,
                config,
                [(transition, pheromone)],
                seed=9000 + index,
                runtime=_runtime_v2(generated_gp=True),
            )
            _assert_valid_population_tours(result.best_tour)
        snapshot = cuda_cache_snapshot(0)
        assert snapshot["kernel_cache_entries"] == 8
        assert snapshot["resident_cache_entries"] == 1
    finally:
        clear_cuda_problem_cache()
        clear_cuda_kernel_cache()


def test_cuda_v2_interpreter_sizes_stack_for_both_trees(
    small_instances,
) -> None:
    """pheromone tree 比 transition tree 深时也不能越界。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("acs", iterations=4),
        ants=32,
        candidate_size=2,
    )
    transition_pset, pheromone_pset = create_primitive_sets()
    transition = compile_tree(
        gp.PrimitiveTree.from_string("RTau", transition_pset),
        role="transition",
    )
    pheromone = compile_tree(
        gp.PrimitiveTree.from_string(
            "ADD(EdgeEta, ADD(EdgeTau, "
            "ADD(NNRank, ADD(ColonyFreq, SourceQuality))))",
            pheromone_pset,
        ),
        role="pheromone",
    )
    transition_depth = _pack_programs(
        [transition],
        role="transition",
    ).stack_size
    pheromone_depth = _pack_programs(
        [pheromone],
        role="pheromone",
    ).stack_size
    assert pheromone_depth > transition_depth
    programs = [(transition, pheromone)]
    interpreted = solve_population_cuda(
        batch,
        config,
        programs,
        seed=421,
        runtime=_runtime_v2(
            precision=CudaPrecision.FP32,
            generated_gp=False,
        ),
    )
    generated = solve_population_cuda(
        batch,
        config,
        programs,
        seed=421,
        runtime=_runtime_v2(
            precision=CudaPrecision.FP32,
            generated_gp=True,
        ),
    )
    assert torch.equal(interpreted.best_tour, generated.best_tour)
    assert torch.equal(interpreted.best_length, generated.best_length)
    assert torch.equal(interpreted.best_iteration, generated.best_iteration)
    assert torch.equal(interpreted.diagnostics, generated.diagnostics)


@pytest.mark.parametrize(
    "local_search",
    [LocalSearch.TWO_OPT, LocalSearch.THREE_OPT],
)
def test_cuda_v2_local_search_is_audited_and_repeatable(
    local_search,
    small_instances,
) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "acs",
            local_search=local_search,
            iterations=2,
        ),
        candidate_size=2,
        local_search_candidate_size=2,
    )
    runtime = replace(
        _runtime_v2(),
        cuda_three_opt_block_threads=512,
    )
    first = solve_population_cuda(
        batch,
        config,
        [(None, None)],
        seed=991,
        runtime=runtime,
    )
    repeated = solve_population_cuda(
        batch,
        config,
        [(None, None)],
        seed=991,
        runtime=replace(runtime, gpu_task_chunk_size=1),
    )
    assert torch.equal(first.best_tour, repeated.best_tour)
    assert torch.equal(first.best_length, repeated.best_length)
    assert torch.equal(first.diagnostics, repeated.diagnostics)
    assert first.diagnostics.shape == (1, 8)
    assert int(first.diagnostics[0, 4]) > 0
    assert int(first.diagnostics[0, 5]) > 0
    assert int(first.diagnostics[0, 6]) > 0


def test_cuda_v2_two_opt_basin_audit_is_monotone_and_noninvasive(
    small_instances,
) -> None:
    """CUDA 审计必须与正常路径逐位同轨，并返回 dense basin 信号。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "acs",
            local_search=LocalSearch.TWO_OPT,
            iterations=3,
        ),
        candidate_size=2,
        local_search_candidate_size=2,
    )
    runtime = _runtime_v2()
    audited = solve_population_cuda(
        batch,
        config,
        [(None, None)],
        seed=2039,
        runtime=runtime,
        basin_top_q=7,
        audit_local_search=True,
    )
    normal = solve_population_cuda(
        batch,
        config,
        [(None, None)],
        seed=2039,
        runtime=runtime,
        basin_top_q=7,
        audit_local_search=False,
    )

    assert audited.basin_mean_length is not None
    assert audited.pre_basin_mean_length is not None
    assert audited.edge_retention is not None
    assert audited.final_colony_tour is not None
    assert audited.final_pre_colony_tour is not None
    assert audited.final_colony_tour.shape == (
        1,
        batch.batch_size,
        config.resolve_ants(batch.n),
        batch.n + 1,
    )
    assert torch.all(
        audited.basin_mean_length
        <= audited.pre_basin_mean_length + 1e-5
    )
    assert torch.all(audited.edge_retention >= 0.0)
    assert torch.all(audited.edge_retention <= 1.0)
    assert torch.equal(audited.best_tour, normal.best_tour)
    assert torch.equal(audited.best_length, normal.best_length)
    assert torch.equal(audited.best_iteration, normal.best_iteration)
    assert torch.equal(audited.basin_mean_length, normal.basin_mean_length)


def test_cuda_v2_origin_pheromone_terminal_is_repeatable(
    small_instances,
) -> None:
    """生成式 CUDA GP 必须能读取 2-opt 新旧边 provenance。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "as",
            local_search=LocalSearch.TWO_OPT,
            iterations=2,
        ),
        candidate_size=2,
        local_search_candidate_size=2,
    )
    _, pheromone_pset = create_primitive_sets(
        pheromone_terminals=ORIGIN_PHEROMONE_TERMINALS,
    )
    origin = compile_tree(
        gp.PrimitiveTree.from_string("Origin", pheromone_pset),
        role="pheromone",
    )
    first = solve_population_cuda(
        batch,
        config,
        [(None, origin)],
        seed=2053,
        runtime=_runtime_v2(),
        basin_top_q=7,
    )
    repeated = solve_population_cuda(
        batch,
        config,
        [(None, origin)],
        seed=2053,
        runtime=_runtime_v2(chunk_size=1),
        basin_top_q=7,
    )
    assert torch.equal(first.best_tour, repeated.best_tour)
    assert torch.equal(first.best_length, repeated.best_length)
    assert torch.equal(first.basin_mean_length, repeated.basin_mean_length)
    assert torch.isfinite(first.best_length).all()


def test_cuda_v2_ls_aware_terminals_are_repeatable(
    small_instances,
) -> None:
    """逐边 LS credit、pre/post consensus 与新几何量必须可批量执行。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "as",
            local_search=LocalSearch.TWO_OPT,
            iterations=3,
        ),
        candidate_size=2,
        local_search_candidate_size=2,
        ls_gain_semantics=LSGainSemantics.EDGE_LAST_MOVE,
    )
    transition_pset, pheromone_pset = create_primitive_sets(
        pheromone_terminals=(
            "LSGain",
            "TauHeadroom",
            "PreFreq",
            "PostFreq",
            "Origin",
        ),
    )
    transition = compile_tree(
        gp.PrimitiveTree.from_string(
            "ADD(MutualRank, TurnCos)",
            transition_pset,
        ),
        role="transition",
    )
    pheromone = compile_tree(
        gp.PrimitiveTree.from_string(
            "ADD(ADD(LSGain, TauHeadroom), ADD(PreFreq, PostFreq))",
            pheromone_pset,
        ),
        role="pheromone",
    )
    programs = [(transition, pheromone)]
    first = solve_population_cuda(
        batch,
        config,
        programs,
        seed=2081,
        runtime=_runtime_v2(),
    )
    repeated = solve_population_cuda(
        batch,
        config,
        programs,
        seed=2081,
        runtime=_runtime_v2(chunk_size=1),
    )
    assert torch.equal(first.best_tour, repeated.best_tour)
    assert torch.equal(first.best_length, repeated.best_length)
    assert torch.equal(first.diagnostics, repeated.diagnostics)
    assert torch.isfinite(first.best_length).all()


def test_cuda_v2_edge_gain_tracking_does_not_change_zero_residual(
    small_instances,
) -> None:
    """读取逐边 Gain 但输出零时，2-opt 轨迹必须与 baseline 完全相同。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_local_search_default(
            "as",
            local_search=LocalSearch.TWO_OPT,
            iterations=3,
        ),
        candidate_size=2,
        local_search_candidate_size=2,
        ls_gain_semantics=LSGainSemantics.EDGE_LAST_MOVE,
    )
    _, pheromone_pset = create_primitive_sets(
        pheromone_terminals=("LSGain",),
    )
    zero_gain = compile_tree(
        gp.PrimitiveTree.from_string(
            "MUL(LSGain, ZERO_PH)",
            pheromone_pset,
        ),
        role="pheromone",
    )
    baseline = solve_population_cuda(
        batch,
        config,
        [(None, None)],
        seed=2087,
        runtime=_runtime_v2(),
    )
    tracked = solve_population_cuda(
        batch,
        config,
        [(None, zero_gain)],
        seed=2087,
        runtime=_runtime_v2(),
    )
    assert torch.equal(tracked.best_tour, baseline.best_tour)
    assert torch.equal(tracked.best_length, baseline.best_length)
    assert torch.equal(tracked.best_iteration, baseline.best_iteration)
    assert torch.equal(tracked.diagnostics, baseline.diagnostics)


@pytest.mark.parametrize("variant", list(ACOVariant))
def test_cuda_v2_local_search_is_monotone_after_one_construction(
    variant,
    small_instances,
) -> None:
    """同一首轮构造上，2-opt 不变差，3-opt 也不弱于 2-opt。"""

    batch = make_problem_batch(small_instances, candidate_size=2)
    two_opt = replace(
        ACOConfig.acotsp_local_search_default(
            variant,
            local_search=LocalSearch.TWO_OPT,
            iterations=1,
        ),
        candidate_size=2,
        local_search_candidate_size=2,
    )
    no_search = replace(two_opt, local_search=LocalSearch.NONE)
    three_opt = replace(two_opt, local_search=LocalSearch.THREE_OPT)
    results = [
        solve_population_cuda(
            batch,
            config,
            [(None, None)],
            seed=1771,
            runtime=replace(
                _runtime_v2(),
                cuda_three_opt_block_threads=512,
            ),
        )
        for config in (no_search, two_opt, three_opt)
    ]
    for result in results:
        _assert_valid_population_tours(result.best_tour)
    assert torch.all(results[1].best_length <= results[0].best_length + 1e-10)
    assert torch.all(results[2].best_length <= results[1].best_length + 1e-10)


def test_cuda_backend_connects_to_population_fitness(small_instances) -> None:
    runtime = _runtime(GPUMode.SINGLE, devices=(0,))
    experiment = ExperimentConfig(
        experiment_id="cuda-test",
        root_seed=19,
        aco=replace(
            ACOConfig.acotsp_default("acs", iterations=2),
            ants=4,
            candidate_size=2,
        ),
        gp=GPConfig(
            population_size=6,
            generations=1,
            elite_size=1,
            tournament_size=2,
            initial_min_depth=1,
            initial_max_depth=2,
            max_depth=3,
        ),
        runtime=runtime,
        train_scales=(50,),
        validation_scales=(50,),
        test_scales=(50,),
        validation_seeds=1,
    )
    cases = in_memory_cases(
        {50: small_instances},
        seed=101,
        candidate_size=2,
    )
    population, _, _ = initialise_population(experiment.gp)
    with EvaluationPool(experiment) as evaluator:
        result = evaluator.evaluate_population(
            population,
            cases,
            BaselineCache(),
        )
    assert result.evaluated_unique > 0
    assert result.constructed_tours > 0
    assert all(individual.fitness.valid for individual in population)


def test_cuda_training_final_candidate_gets_cpu_fp64_audit(
    small_instances,
    tmp_path,
) -> None:
    experiment = ExperimentConfig(
        experiment_id="cuda-final-audit",
        root_seed=29,
        aco=replace(
            ACOConfig.acotsp_default("acs", iterations=2),
            ants=4,
            candidate_size=2,
        ),
        gp=GPConfig(
            population_size=6,
            generations=1,
            elite_size=1,
            tournament_size=2,
            initial_min_depth=1,
            initial_max_depth=2,
            max_depth=3,
            checkpoint_interval=1,
            checkpoint_top_k=2,
        ),
        runtime=_runtime(GPUMode.SINGLE, devices=(0,)),
        train_scales=(50,),
        validation_scales=(50,),
        test_scales=(50,),
        validation_seeds=1,
    )
    cases = in_memory_cases(
        {50: small_instances},
        seed=303,
        candidate_size=2,
    )
    result = train(
        experiment,
        lambda _generation: cases,
        cases,
        output_directory=tmp_path / "cuda-audit",
    )
    assert result.validation.backend is ExecutionBackend.CUDA_FUSED_FP32
    assert result.cpu_fp64_audit is not None
    assert result.cpu_fp64_audit.backend is ExecutionBackend.NUMBA_BATCH
    assert (
        result.cpu_fp64_audit.selected_candidate_hash
        == result.validation.selected_candidate_hash
    )
    assert result.passed_noninferiority == (
        result.validation.passed_noninferiority
        and result.cpu_fp64_audit.passed_noninferiority
    )
    assert (tmp_path / "cuda-audit" / "cpu_fp64_audit_summary.csv").is_file()
