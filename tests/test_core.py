"""E00：语言保护、随机流、数据隔离及 CPU 固定搜索预算。"""

import json

import numpy as np
import pytest

from gpaco.backends.cpu import evaluate
from gpaco.backends.numeric import counter_uniform, philox, stdrel
from gpaco.config import ExecutionPlan, SearchConfig
from gpaco.data import ROOT, coordinate_hash, parse_line, prepare_problem, validate_tours
from gpaco.language import ProgramSpec, evaluate_reference


def problem(n=9, batch=2):
    coords = np.random.default_rng(12).uniform(size=(batch, n, 2))
    tours = np.tile(np.r_[np.arange(n), 0], (batch, 1)).astype(np.int32)
    return prepare_problem(coords, tours, [coordinate_hash(c) for c in coords], candidate_size=3)


def test_philox_known_vector_and_float32():
    actual = philox(np.uint64(0), np.uint64(0), 0, 0, 0, 0)
    assert tuple(actual) == (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
    values = [counter_uniform(np.uint64(1), np.uint64(2), 3, a, 5, 6) for a in range(32)]
    assert all(0 <= value < 1 for value in values)
    assert len(set(values)) == 32
    assert all(float(np.float32(v)) == v for v in values)


def test_centered_stdrel_handles_constant_and_nearly_equal_inputs():
    for value in [0.0, -25.28351, 8.12834]:
        values = np.full(27, value, np.float32)
        output = np.empty_like(values)
        stdrel(values, len(values), output)
        np.testing.assert_array_equal(output, 0)
    values = np.asarray([1, 1.000001, 1.000002], np.float32)
    output = np.empty_like(values)
    stdrel(values, 3, output)
    centered = values.astype(np.float64) - np.mean(values.astype(np.float64))
    expected = np.tanh(centered / (np.std(values.astype(np.float64)) + 1e-8))
    np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-4)


def test_language_clips_each_node_and_zero_denominator():
    p = ProgramSpec.parse("SUB(ADD(RTau, REta), REta)")
    got = evaluate_reference(
        p, {"RTau": np.asarray([9.0], np.float32), "REta": np.asarray([9.0], np.float32)}
    )
    np.testing.assert_array_equal(got, [1.0])
    p = ProgramSpec.parse("PDIV(POS_ONE, ZERO)")
    assert evaluate_reference(p, {}).item() == 0
    assert p.semantic_hash != ProgramSpec.parse("PDIV(POS_HALF, ZERO)").semantic_hash


def test_parser_rejects_bad_routes():
    parse_line("0 0 1 0 0 1 output 1 2 3 1")
    with pytest.raises(ValueError):
        parse_line("0 0 1 0 0 1 output 1 2 2 1")
    with pytest.raises(ValueError):
        parse_line("0 0 1 0 0 1 output 1 2 3 2")


@pytest.mark.parametrize("operation", ["MIN", "MAX", "PDIV", "ADD", "SUB", "MUL"])
def test_cpu_primitive_nonfinite_contract(operation):
    from gpaco.backends.cpu_existing import _evaluate_program
    from gpaco.language import TERMINAL_IDS, pack_programs

    program = ProgramSpec.parse(f"{operation}(RTau, REta)")
    fields = np.zeros((16, 4), np.float32)
    fields[0] = [np.nan, 1, np.inf, -np.inf]
    fields[1] = [1, np.nan, -np.inf, np.inf]
    packed = pack_programs([program])
    actual = np.asarray(
        [
            _evaluate_program(
                packed[0][0], packed[1][0], packed[2][0], fields, i, np.empty(31, np.float32)
            )
            for i in range(4)
        ]
    )
    expected = evaluate_reference(
        program, {name: fields[index] for name, index in TERMINAL_IDS.items()}
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("variant", ["as", "acs", "mmas"])
def test_cpu_valid_repeatable_and_budget(variant):
    data = problem()
    programs = [ProgramSpec.parse(x) for x in ["ZERO", "ADD(DistRank, TurnCos)", "SUB(RTau, REta)"]]
    config = SearchConfig(variant=variant, ants=4, iterations=3, candidate_size=3)
    plan = ExecutionPlan(backend="cpu_existing")
    first = evaluate(programs, data, config, 17, plan)
    second = evaluate(programs[::-1], data, config, 17, plan)
    assert first.lengths.dtype == np.float32
    validate_tours(first.tours, data.n)
    np.testing.assert_array_equal(first.tours, second.tours[::-1])
    np.testing.assert_array_equal(first.lengths, second.lengths[::-1])
    assert first.timings["executed_tasks"] == 6


def test_processed_splits_and_standard_test_counts():
    for n, expected in [(100, 1280), (500, 128)]:
        directory = ROOT / "Datasets/processed/v1" / f"tsp{n}"
        if not directory.exists():
            pytest.skip("尚未生成实验数据")
        groups = {}
        for split in ("train", "validation", "tuning", "holdout", "test"):
            ids = np.load(directory / split / "instance_ids.npy")
            groups[split] = set(ids.tolist())
            assert len(groups[split]) == len(ids)
        names = list(groups)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                assert not groups[first] & groups[second]
        assert len(groups["test"]) == expected
        complete = json.loads((directory / "COMPLETE.json").read_text())
        assert complete["test_count"] == expected
