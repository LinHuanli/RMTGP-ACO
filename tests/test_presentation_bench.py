"""报告基准验证：真实树往返、输入防篡改、JIT 语义和计数。"""

import random
from dataclasses import replace

import numpy as np
import pytest
from deap import gp
from numba import njit

from rmtgp_aco.config import ACOConfig, ACOVariant, ExperimentConfig, GPConfig, RuntimeConfig
from rmtgp_aco.data import TSPInstance, coordinate_hash, make_problem_batch
from rmtgp_aco.genetic import initialise_population
from rmtgp_aco.presentation_bench import (
    CompilationAudit,
    TraceWriter,
    evaluate,
    load_trace,
    pack_individual,
    read_json,
    unpack_individual,
)
from rmtgp_aco.presentation_jit import has_variable_output, postfix_sum, scalar_function
from rmtgp_aco.program import compile_tree, create_primitive_sets
from rmtgp_aco.sampling import EvaluationCase
from rmtgp_aco.training import BaselineCache, EvaluationPool


def tiny_case():
    rng = np.random.default_rng(7)
    instances = []
    for i in range(2):
        coords = rng.random((6, 2))
        tour = np.array([0, 1, 2, 3, 4, 5, 0])
        length = np.linalg.norm(np.diff(coords[tour], axis=0), axis=1).sum()
        instances.append(
            TSPInstance(f"example-{i}", coords, tour, float(length), coordinate_hash(coords))
        )
    return EvaluationCase(6, make_problem_batch(instances, candidate_size=5), 71)


def test_tree_round_trip_preserves_constants_types_and_sentinels():
    random.seed(2001)
    population, tr, ph = initialise_population(GPConfig())
    for individual in population:
        restored = unpack_individual(pack_individual(individual), (tr, ph))
        assert restored.structural_hash == individual.structural_hash
        assert [str(t) for t in restored] == [str(t) for t in individual]
        assert pack_individual(restored) == pack_individual(individual)


def test_trace_multiple_calls_duplicate_population_and_hash_validation(tmp_path):
    config = GPConfig()
    population, _, _ = initialise_population(config)
    writer = TraceWriter(tmp_path)
    writer.capture(1, [population[0], population[0]], [tiny_case()])
    writer.capture(1, [population[1]], [tiny_case()])
    calls = load_trace(tmp_path, config)
    assert [row[0]["evaluation_call_id"] for row in calls] == [0, 1]
    assert len(calls[0][1]) == 2
    np.testing.assert_array_equal(calls[0][2][0].batch.coords, tiny_case().batch.coords)
    path = tmp_path / writer.calls[0]["cases"][0]["file"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="哈希"):
        load_trace(tmp_path, config)


@pytest.mark.parametrize("name", ["ADD", "SUB", "MUL", "PDIV", "MIN", "MAX", "ABS", "NEG"])
def test_numeric_codegen_preserves_protected_operations(name):
    tr, _ = create_primitive_sets()
    text = f"{name}(RTau)" if name in {"ABS", "NEG"} else f"{name}(RTau, REta)"
    tree = gp.PrimitiveTree.from_string(text, tr)
    program = compile_tree(tree, role="transition")
    pure, _, encoded = scalar_function(program, "transition")
    inputs = np.array(
        [[0.0, 1.0, -1.0, 10.0, 20.0, np.nan, np.inf], [0.0, 1e-12, -2.0, 0.0, 4.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    compiled = njit(pure)
    for column in range(inputs.shape[1]):
        with np.errstate(all="ignore"):
            expected = pure(inputs, column)
        actual = compiled(inputs, column)
        assert actual == expected
        subset = np.ascontiguousarray(inputs[:, column : column + 1])
        reference = postfix_sum(
            encoded.opcodes,
            encoded.float_arguments,
            encoded.integer_arguments,
            subset,
            1,
            encoded.stack_size,
        )
        assert actual == reference


def test_repeated_evaluation_executes_same_tasks_and_returns_metrics():
    random.seed(7)
    gp_config = GPConfig(population_size=10, elite_size=2)
    pop, _, _ = initialise_population(gp_config)
    population = [p for p in pop if p.total_nodes > 0][:2]
    population.append(population[0].clone())
    runtime = RuntimeConfig(aco_backend="numba_batch", cpu_threads=1)
    exp = ExperimentConfig(
        experiment_id="test-presentation",
        root_seed=7,
        aco=replace(
            ACOConfig.acotsp_default(ACOVariant.ACS), ants=4, iterations=2, candidate_size=5
        ),
        gp=gp_config,
        runtime=runtime,
    )
    case = tiny_case()
    cache = BaselineCache()
    cache.put(case, exp, case.batch.reference_length)
    with EvaluationPool(exp) as pool, CompilationAudit(False) as audit:
        pool.benchmark_metrics_enabled = True
        first = evaluate(exp, "cpu1", population, [case], pool, cache, audit, 1)
        second = evaluate(exp, "cpu1", population, [case], pool, cache, audit, 2)
    assert first["programs_unique"] == 2
    assert first["tours_executed"] == second["tours_executed"] > 0
    assert first["output_signature"] == second["output_signature"]
    assert second["backend_calls"][0]["backend_wall_s"] > 0


@pytest.mark.parametrize("generations", [5, 10])
def test_prepare_freezes_requested_generation_count(tmp_path, monkeypatch, generations):
    # 在真正的短跑循环中验证 5/10 代和 trace 顺序，不读取外部训练数据。
    from rmtgp_aco import presentation_bench as bench

    exp = ExperimentConfig(
        experiment_id="test-presentation",
        root_seed=7,
        aco=replace(
            ACOConfig.acotsp_default(ACOVariant.ACS), ants=4, iterations=2, candidate_size=5
        ),
        gp=GPConfig(population_size=10, elite_size=2, generations=generations),
        runtime=RuntimeConfig(aco_backend="numba_batch", cpu_threads=1),
    )
    case = tiny_case()
    cache = BaselineCache()
    cache.put(case, exp, case.batch.reference_length)
    monkeypatch.setattr(bench, "experiment", lambda *a: exp)
    monkeypatch.setattr(bench, "training_cases", lambda *a: [(None, case)] * generations)
    monkeypatch.setattr(bench, "baseline_cache", lambda *a: cache)
    monkeypatch.setattr(bench, "configure_runtime", lambda *a: None)
    monkeypatch.setattr(bench, "environment", lambda *a: {})
    bench.train(tmp_path, "cpu1", tmp_path / "run", capture=True)
    result = read_json(tmp_path / "run/result.json")
    assert [r["generation"] for r in result["records"]] == list(range(1, generations + 1))
    assert len(load_trace(tmp_path / "trace", exp.gp)) == generations
    assert not (tmp_path / "run/champion.pkl").exists()


def test_speedup_requires_same_host_workload_and_source():
    from rmtgp_aco.presentation_report import matched_summary

    base = {
        "experiment_id": "E2",
        "host": "host-a",
        "repeat_id": 0,
        "source_hash": "revision-a",
        "evaluation_call_id": 0,
        "workload_id": "work-a",
        "tasks_executed": 8,
        "tours_executed": 128,
        "backend": "cpu8",
        "evaluation_wall_s": 10.0,
    }
    gpu = {**base, "backend": "v2", "evaluation_wall_s": 2.0}
    _, ratios = matched_summary([base, gpu])
    assert next(r for r in ratios if r["backend"] == "v2")["speedup_vs_cpu8"] == 5.0
    for changed in (
        {"host": "host-b"},
        {"source_hash": "revision-b"},
        {"workload_id": "work-b"},
        {"tasks_executed": 7},
    ):
        _, ratios = matched_summary([base, {**gpu, **changed}])
        assert not any(r["backend"] == "v2" for r in ratios)


def test_jit_selection_excludes_constant_and_cancelled_expressions():
    tr, _ = create_primitive_sets()
    for expression, expected in (
        ("ABS(NEG_ONE)", False),
        ("SUB(RTau, RTau)", False),
        ("ADD(RTau, REta)", True),
    ):
        tree = gp.PrimitiveTree.from_string(expression, tr)
        assert has_variable_output(compile_tree(tree, role="transition"), "transition") == expected


def test_nvrtc_audit_uses_active_cupy_entry(monkeypatch):
    compiler = pytest.importorskip("cupy.cuda.compiler")
    name = (
        "_compile_using_nvrtc_no_warning"
        if hasattr(compiler, "_compile_using_nvrtc_no_warning")
        else "compile_using_nvrtc"
    )
    monkeypatch.setattr(compiler, name, lambda *args, **kwargs: (b"compiled", {}))
    with CompilationAudit(True) as audit:
        getattr(compiler, name)("source")
        assert audit.compile_count == 1


def test_empty_report_removes_stale_derived_rows(tmp_path):
    from rmtgp_aco.presentation_report import write_csv

    path = tmp_path / "summary.csv"
    write_csv(path, [{"seconds": 3.0}])
    assert "3.0" in path.read_text()
    write_csv(path, [])
    assert path.read_text() == ""
