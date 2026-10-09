"""正式基线的独立种子、完整预算、只读参考与有限映射矩阵。"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import yaml

from gpaco import formal_inputs
from gpaco.data import ROOT
from gpaco.hardware_inputs import TrainingInputs

sys.path.insert(0, str(ROOT / "scripts"))
from research_pool import tasks_for


def test_preregistered_formal_and_mapping_matrix():
    cfg = yaml.safe_load((ROOT / "configs/workloads/research_queue.yaml").read_text())
    tasks = tasks_for(cfg)
    assert len(tasks) == 38 and len({t["id"] for t in tasks}) == 38
    formal = [t for t in tasks if t["kind"] == "formal_train"]
    assert len(formal) == 20 and {t["seed"] for t in formal} == set(range(2001, 2011))
    assert all(t["kind"] == "formal_train" for t in tasks[:20])
    assert cfg["formal"]["generations"] == 50 and cfg["formal"]["search"]["iterations"] == 500
    assert not cfg["formal"]["standard_test_opened"]
    assert (
        len(tasks[20:])
        * len(cfg["mapping"]["active_tasks"])
        * len(cfg["mapping"]["candidate_lanes"])
        == 216
    )
    assert not cfg["mapping"]["select_plan_from_holdout"]


def test_formal_input_preparation_and_readonly_cache(tmp_path, monkeypatch):
    from test_core import problem

    from gpaco.config import SearchConfig

    sample = problem(n=100)
    tours = np.tile(np.r_[np.arange(100), 0], (sample.size, 1)).astype(np.int32)
    monkeypatch.setattr(
        formal_inputs, "load_split", lambda *a: (sample.coords, tours, sample.instance_ids)
    )
    monkeypatch.setattr(formal_inputs, "metadata", lambda: {"fixture": True})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture")
    cuda = SimpleNamespace(
        runtime=SimpleNamespace(getDeviceProperties=lambda _: {"name": b"NVIDIA RTX A5000"})
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=cuda, get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None)
        ),
    )
    monkeypatch.setattr(
        formal_inputs,
        "evaluate",
        lambda programs, p, *args: SimpleNamespace(
            lengths=p.reference[None].copy(), tours=tours[None], timings={"fixture": True}
        ),
    )
    cfg = {
        "search": {"variant": "as", "ants": 2, "iterations": 2},
        "generations": 2,
        "population": 12,
        "batch": sample.size,
        "validation_repeats": 1,
    }
    store = formal_inputs.prepare(tmp_path / "inputs", 100, 2001, cfg, "GPU-fixture")
    assert len(store.manifest["scenarios"]) == 3 and not store.manifest["tests_opened"]
    inputs = TrainingInputs(store, 2001)
    data = inputs.training(1, np.asarray(store.manifest["schedules"]["2001"][0]))
    lengths, _, hit = inputs.baseline(
        data, SearchConfig(**cfg["search"]), data.initialization.seed, None
    )
    assert hit and not data.distances.flags.writeable
    np.testing.assert_array_equal(lengths, data.reference)
    assert len(json.loads((store.directory / "initial_population.json").read_text())) == 12
