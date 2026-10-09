"""外层算法、序列化和断点恢复；临时产物由项目内 TMPDIR 承载。"""

import json
import random

import numpy as np
import pytest

from gpaco import experiment
from gpaco.config import ExecutionPlan, SearchConfig
from gpaco.evolution import initial_population, next_population
from gpaco.language import ProgramSpec, parse_tree


def test_typed_population_and_erc_roundtrip():
    random.seed(1001)
    population = initial_population(100)
    for _ in range(8):
        programs = [ProgramSpec.from_tree(tree) for tree in population]
        for program in programs:
            assert ProgramSpec.from_tree(parse_tree(program.expression)) == program
            assert len(program.instructions) <= 31 and program.depth <= 5
        population = next_population(population, np.arange(100, dtype=np.float32))
        assert len(population) == 100
    a, b = (ProgramSpec.parse(s) for s in ("ADD(RTau, 0.1)", "ADD(RTau, 0.2)"))
    assert a.structural_hash == b.structural_hash
    assert a.semantic_hash != b.semantic_hash
    for value in ("2.0", "float('nan')", "__import__('os')", "ADD(ZERO)"):
        with pytest.raises(ValueError):
            ProgramSpec.parse(value)


def test_resume_replays_same_population_and_history(tmp_path, monkeypatch):
    # 此测试只检查控制流。CPU 闭环和 GPU 闭环由各自 E00 测试独立验证。
    from test_core import problem

    from gpaco.backends.cpu import EvaluationResult

    sample = problem()
    monkeypatch.setattr(
        experiment,
        "load_split",
        lambda *a: (
            sample.coords,
            np.tile(np.r_[np.arange(sample.n), 0], (sample.size, 1)),
            sample.instance_ids,
        ),
    )

    def fake_evaluate(programs, data, search, seed, plan):
        scores = np.asarray([int(p.semantic_hash[:4], 16) / 65536 for p in programs], np.float32)
        return EvaluationResult(
            data.reference[None] + scores[:, None],
            np.zeros((len(programs), data.size, data.n + 1), np.int32),
            np.zeros((len(programs), data.size, 8), np.uint64),
            {"eval_wall_s": 0.0, "executed_tasks": len(programs) * data.size},
        )

    monkeypatch.setattr(experiment, "evaluate", fake_evaluate)
    monkeypatch.setattr(experiment, "baseline", lambda data, *a: (data.reference.copy(), 0.0, True))
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    settings = dict(
        population_size=12, generations=3, batch_size=2, validation_interval=1, validation_repeats=1
    )
    args = (100, 173, SearchConfig(ants=2, iterations=2), ExecutionPlan())
    experiment.train(full, *args, **settings)
    original = experiment.append_record

    def interrupt_after_commit(path, row):
        original(path, row)
        if row["generation"] == 1:
            raise InterruptedError("模拟 checkpoint 提交后的进程中断")

    monkeypatch.setattr(experiment, "append_record", interrupt_after_commit)
    with pytest.raises(InterruptedError):
        experiment.train(resumed, *args, **settings)
    monkeypatch.setattr(experiment, "append_record", original)
    experiment.train(resumed, *args, resume=True, **settings)
    first = json.loads((full / "history.json").read_text())
    second = json.loads((resumed / "history.json").read_text())
    for a, b in zip(first, second, strict=True):
        for key in (
            "best_expression",
            "train_best_gap_percent",
            "train_median_gap_percent",
            "validation_champion_gap_percent",
            "unique_programs",
        ):
            assert a[key] == b[key]
    assert json.loads((resumed / "COMPLETE.json").read_text())["resume_count"] == 1
    with pytest.raises(FileExistsError):
        experiment.train(full, *args, **settings)
    with pytest.raises(FileNotFoundError):
        experiment.train(tmp_path / "missing", *args, resume=True, **settings)
