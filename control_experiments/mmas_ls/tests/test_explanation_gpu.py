"""新跨框架与完整状态干预入口的 GPU 验收；由空闲设备任务显式执行。"""
from dataclasses import asdict, replace
import numpy as np
import pytest
import torch

from rmtgp_aco.aco_cuda import cuda_available, solve_population_cuda_anytime, _active_and_representative_programs
from rmtgp_aco.mechanisms import MechanismConfig, InstrumentationConfig, SolverControl
from control_experiments.mmas_ls.tests.test_gpu_controls import example, settings

pytestmark = pytest.mark.skipif(not cuda_available(), reason="需要显式空闲 CUDA 设备")


@pytest.mark.parametrize("variant", ("as", "mmas"))
def test_cross_framework_programs_budget_and_no_nan(variant):
    from control_experiments.mmas_ls.evaluate import program_entries
    b = example(); c, r = settings(variant, 30)
    entries = program_entries(variant, training_variants=("as", "mmas"))
    control = SolverControl(MechanismConfig(terminal_statistics="centered_fp32"), collected=[])
    result = solve_population_cuda_anytime(b, c, [e["program"] for e in entries], seed=827, runtime=r, control=control)
    assert result.best_tour.shape[0] == 7
    assert torch.isfinite(result.best_length).all()
    assert all(np.max(s["trace"][..., 20]) < 1e-5 for s in control.collected)


def test_projected_owner_and_complete_explanation_fork(tmp_path, monkeypatch):
    from control_experiments.mmas_ls import explanation_forks as forks
    from control_experiments.mmas_ls.common import atomic_json, atomic_npz, source_manifest, read_json
    from control_experiments.mmas_ls.evaluate import program_entries
    from control_experiments.mmas_ls.diagnostics import DiagnosticRecorder
    b = example(); c, r = settings(iterations=600)
    entries = program_entries("mmas"); programs = [e["program"] for e in entries]
    mechanism = MechanismConfig(terminal_statistics="centered_fp32")
    job = tmp_path/"jobs/owner"
    spec = {"source": source_manifest()["source_hash"], "seed": 829,
            "models": [{k: v for k, v in e.items() if k != "program"} for e in entries],
            "task": {"indices": [0, 1], "variant": "mmas", "split": "diagnosis_dev", "mechanism": asdict(mechanism)}}
    inst = InstrumentationConfig("heavy", profile="mechanism_v3", schema_version=3)
    writer = DiagnosticRecorder(job/"diagnostics", spec, inst)
    try:
        solve_population_cuda_anytime(b, c, programs, seed=829, runtime=r,
            control=SolverControl(mechanism, inst, observer=writer, stop_iteration=100))
        writer.finish(100)
    finally:
        writer.close()
    *_, inverse = _active_and_representative_programs(programs, c)
    atomic_npz(job/"raw.npz", behavior_alias=inverse)
    atomic_json(tmp_path/"manifests/checkpoints.json", [{"id": e["id"], "file_hash": e["file_hash"]} for e in entries[1:]])
    monkeypatch.setattr(forks, "batch", lambda *a, **k: b)
    monkeypatch.setattr(forks, "experiment", lambda *a, **k: (c, r))
    for owner in (0, 2):
        task = {"id": f"fork-owner{owner}", "parent": "owner", "owner": owner, "champion": 81002,
                "snapshot_iteration": 100, "continuation_iterations": 500, "measurement_steps": [1, 25, 100, 500]}
        status = forks.run(task, tmp_path)
        assert status["status"] == "completed"
        manifest = read_json(tmp_path/"jobs"/task["id"]/"manifest.json")
        assert manifest["original_horizon"] == 600
        with np.load(tmp_path/"jobs"/task["id"]/"result.npz") as a:
            assert a["native_without_gp_anytime"].shape == (1, 2, 500)
        rows = read_json(tmp_path/"jobs"/task["id"]/"immediate_comparisons.json")["rows"]
        assert all(0 <= r["cpu_fp64_next_choice_probability_change"] <= 1
                   for r in rows if "cpu_fp64_next_choice_probability_change" in r)
