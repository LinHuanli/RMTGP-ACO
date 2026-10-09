"""输入转换和 CPU-only 元数据检查；破坏性样例只操作 pytest 临时目录。"""

import importlib.metadata
import json
from pathlib import Path

import pytest

from gpaco.artifact_registry import resolve
from gpaco.benchmark_inputs import BenchmarkInputs
from gpaco.cpu_benchmark import physical_cpu_ids
from gpaco.data import ROOT
from gpaco.experiment import metadata


def test_cpu_metadata_without_gpu_packages(monkeypatch):
    original = importlib.metadata.version

    def version(name):
        if name in ("torch", "cupy-cuda13x"):
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    record = metadata()
    assert record["packages"]["torch"] is None
    assert record["packages"]["cupy-cuda13x"] is None


def test_physical_core_selection_excludes_siblings():
    ids = physical_cpu_ids()
    keys = []
    for cpu in ids:
        p = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        keys.append(((p / "physical_package_id").read_text(), (p / "core_id").read_text()))
    assert len(set(keys)) == len(ids)


def test_prepared_bundle_matches_old_gpu_identity():
    directory = resolve("shared-frozen-population-p01") / "tsp100/as"
    if not directory.exists():
        pytest.skip("尚未导入输入包")
    inputs = BenchmarkInputs(directory)
    programs, problem = inputs.load(1, 0)
    assert len(programs) == 100 and problem.size == 32
    original = ROOT / "artifacts/a5000-main-v1/inputs/tsp100/as/READY.json"
    assert (
        problem.initialization.input_manifest_sha256
        == json.loads(original.read_text())["manifest_sha256"]
    )
    assert not problem.coords.flags.writeable
    with pytest.raises(ValueError, match="越界"):
        inputs.path("../manifest.json")


def test_cpu_baseline_cache_miss_does_not_call_cuda(monkeypatch):
    from test_core import problem

    from gpaco import experiment
    from gpaco.config import ExecutionPlan, SearchConfig

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU cache miss 不应触发任何求解")

    monkeypatch.setattr(experiment, "evaluate", forbidden)
    with pytest.raises(FileNotFoundError, match="禁止隐式"):
        experiment.baseline(
            problem(), SearchConfig(candidate_size=3), 981774, ExecutionPlan(backend="cpu_python")
        )
