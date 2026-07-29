"""融合 CUDA 后端的 tour、确定性、分片和 GP 语义测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from deap import gp

from rmtgp_aco.aco_cuda import (
    cuda_available,
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
    RuntimeConfig,
)
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.genetic import initialise_population
from rmtgp_aco.program import (
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
