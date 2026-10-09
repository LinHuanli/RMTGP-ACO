"""跨卡协议：冻结输入、缓存权限、配对顺序与计量口径。"""

from dataclasses import asdict, replace

import numpy as np
import pytest
from test_core import problem

from gpaco.backends.cpu import initial_parameters, problem_initialization
from gpaco.config import SearchConfig
from gpaco.data import FrozenInitialization, write_json
from gpaco.experiment import source_hash
from gpaco.hardware_campaign import choose_plan, paired_order
from gpaco.hardware_inputs import (
    DEFAULT_PLAN,
    FrozenStore,
    TrainingInputs,
    file_hash,
    save_problem,
    scenario_key,
)
from gpaco.telemetry import summarize_samples


def miniature_store(tmp_path):
    data, config, seed = problem(), SearchConfig(candidate_size=3), 73
    key = scenario_key("train", 1001, 1)
    save_problem(tmp_path / "geometry/train", data)
    values = initial_parameters(
        data.distances, data.instance_keys, np.uint64(seed), 0, np.float32(0.5)
    )
    target = tmp_path / "scenarios" / key
    target.mkdir(parents=True)
    for name, array in zip(("tau0", "low", "high"), values, strict=True):
        np.save(target / f"{name}.npy", array)
    np.savez(target / "baseline.npz", lengths=data.reference, tours=np.zeros((2, 10), np.int32))
    manifest = {
        "source_hash": source_hash(),
        "search": asdict(config),
        "scenarios": {key: {"geometry": "train", "seed": seed, "baseline": True}},
        "geometry": {"train": {"path": "geometry/train", "instances": data.instance_ids}},
        "schedules": {"1001": [[7, 8]]},
        "files": {
            str(p.relative_to(tmp_path)): file_hash(p) for p in tmp_path.rglob("*") if p.is_file()
        },
    }
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "READY.json", {"manifest_sha256": file_hash(tmp_path / "manifest.json")})
    return FrozenStore(tmp_path), data, config, seed, key


def test_frozen_geometry_initialization_and_readonly_reference(tmp_path):
    store, expected, config, seed, key = miniature_store(tmp_path)
    inputs = TrainingInputs(store, 1001)
    actual = inputs.training(1, [7, 8])
    assert actual is store.problem(key)
    np.testing.assert_array_equal(actual.distances, expected.distances)
    assert not actual.distances.flags.writeable
    frozen = problem_initialization(actual, config, seed)
    assert all(not value.flags.writeable for value in frozen)
    with pytest.raises(ValueError, match="身份"):
        problem_initialization(actual, config, seed + 1)
    with pytest.raises(ValueError, match="身份"):
        problem_initialization(actual, SearchConfig(variant="acs"), seed)
    with pytest.raises(ValueError, match="日程"):
        inputs.training(1, [8, 7])
    lengths, elapsed, hit = store.baseline(actual, config, seed, DEFAULT_PLAN)
    np.testing.assert_array_equal(lengths, expected.reference)
    assert elapsed >= 0 and hit
    # 缺失参考不会在本机重新求解，不存在 first writer wins。
    (store.directory / f"scenarios/{key}/baseline.npz").unlink()
    with pytest.raises(FileNotFoundError):
        store.baseline(actual, config, seed, DEFAULT_PLAN)


def test_frozen_file_tamper_is_rejected(tmp_path):
    store, *_ = miniature_store(tmp_path)
    path = store.directory / "geometry/train/reference.npy"
    np.save(path, np.asarray([1, 2], np.float32))
    with pytest.raises(ValueError, match="SHA256"):
        store.problem(scenario_key("train", 1001, 1))


def test_frozen_subset_retains_seed_instance_binding(tmp_path):
    store, _, config, seed, key = miniature_store(tmp_path)
    subset = store.problem(key, 1)
    assert subset.size == 1
    assert all(v.shape == (1,) for v in problem_initialization(subset, config, seed))
    with pytest.raises(ValueError, match="实例"):
        store.problem(key, 3)


def test_initialization_fallback_and_frozen_identical():
    data, config = problem(), SearchConfig()
    expected = problem_initialization(data, config, 12)
    actual = replace(
        data,
        initialization=FrozenInitialization(
            12,
            config.variant,
            config.rho,
            data.instance_ids,
            expected,
            "test",
        ),
    )
    for first, second in zip(expected, problem_initialization(actual, config, 12), strict=True):
        np.testing.assert_array_equal(first, second)


def tuning_row(cell, seconds, lanes=8, contended=False):
    return {
        "cell": cell,
        "eval_wall_s": seconds,
        "plan": asdict(replace(DEFAULT_PLAN, candidate_lanes=lanes)),
        "status": "completed",
        "contended": contended,
        "telemetry_errors": [],
    }


def test_threshold_and_infeasible_selection():
    base, weak, strong = (
        tuning_row("default", 100),
        tuning_row("weak", 98, 4),
        tuning_row("strong", 90, 16),
    )
    assert choose_plan([base, weak], base)["default_retained"]
    choice = choose_plan([base, weak, strong, {"status": "infeasible"}], base)
    assert choice["selected_cell"] == "strong" and not choice["holdout_used_for_selection"]
    contaminated = tuning_row("contended", 1, 32, True)
    assert choose_plan([base, contaminated], base)["default_retained"]
    with pytest.raises(RuntimeError):
        choose_plan([base], contaminated)


def test_order_is_reproducible_and_paired():
    for model in ("a5000", "a40", "l4", "l40s", "pro5000"):
        for n in (100, 500):
            for block in range(5):
                order = paired_order(model, n, block)
                assert sorted(order) == ["default", "selected"]
                assert order == paired_order(model, n, block)


def test_energy_units_and_fallback():
    samples = [
        {"monotonic_s": t, "power_w": 100.0, "gpu_used_bytes": 1000 + t, "other_pids": []}
        for t in (0, 1, 2)
    ]
    result = summarize_samples(samples, 0, 2, 1000.0, 1190.0)
    assert result["energy_j"] == 190.0 and result["energy_method"].startswith("nvml")
    result = summarize_samples(samples, 0, 2)
    assert result["energy_j"] == 200.0 and result["sampled_peak_gpu_used_bytes"] == 1002
    assert result["occupancy_measured"] is None
    samples[1]["other_pids"] = [123]
    assert summarize_samples(samples, 0, 2)["contended"]
    assert summarize_samples([], 0, 2)["energy_j"] is None


def test_manifest_tamper_is_rejected(tmp_path):
    store, *_ = miniature_store(tmp_path)
    write_json(store.directory / "manifest.json", {"tampered": True})
    with pytest.raises(ValueError, match="manifest"):
        FrozenStore(tmp_path)
