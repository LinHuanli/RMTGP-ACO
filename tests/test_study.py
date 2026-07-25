"""单 GPU0 三种子 study 的合同、缓存和任务图测试。"""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from rmtgp_aco.evaluation import EvaluationRecord, study_test_seed, write_records
from rmtgp_aco.model import RunDiagnostics, RunResult
from rmtgp_aco.study import (
    _read_test_cache,
    _visible_physical_device,
    _write_test_cache,
    build_study_tasks,
    export_study_contract,
    load_study_spec,
)
from rmtgp_aco.study_report import CURVE_FIELDS, generate_study_report


def test_repository_study_contract_and_replicate_major_queue() -> None:
    study = load_study_spec("experiments/tsp100_gpu0_3seed/study.yaml")
    contract = export_study_contract(study)
    assert contract["test_root_seed"] == 9001
    assert contract["test_seeds"] == 3
    assert contract["partitions"] == [
        "tsp50_uniform",
        "tsp100_uniform",
        "tsp500_uniform",
        "tsp1000_uniform",
    ]
    assert [item["seeds"] for item in contract["variants"]] == [
        [1001, 1002, 1003],
        [2001, 2002, 2003],
        [3001, 3002, 3003],
    ]

    tasks = build_study_tasks(study)
    assert len(tasks) == 40
    assert [task.task_id for task in tasks[:9]] == [
        "schedule-as-1001",
        "baseline-as-1001",
        "train-as-1001",
        "schedule-acs-2001",
        "baseline-acs-2001",
        "train-acs-2001",
        "schedule-mmas-3001",
        "baseline-mmas-3001",
        "train-mmas-3001",
    ]
    train = next(task for task in tasks if task.task_id == "train-as-1001")
    baseline_parent = study.output_root / "baselines" / "as"
    index = train.command.index("--baseline-archive")
    assert train.command[index + 1] == str(baseline_parent)
    assert tasks[-1].task_id == "report"


def test_study_test_seed_is_gp_independent_and_partition_separated() -> None:
    first = study_test_seed(9001, "tsp100_uniform", 3, 2)
    assert first == study_test_seed(9001, "tsp100_uniform", 3, 2)
    assert first != study_test_seed(9001, "tsp500_uniform", 3, 2)
    assert first != study_test_seed(9001, "tsp100_uniform", 4, 2)


def test_visible_physical_device_accepts_exactly_one_index(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert _visible_physical_device() == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="只包含一个"):
        _visible_physical_device()


def test_test_baseline_cache_roundtrip_and_metadata_guard(tmp_path) -> None:
    result = RunResult(
        best_tour=torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
        best_length=torch.tensor([3.5], dtype=torch.float64),
        best_iteration=torch.tensor([7], dtype=torch.int64),
        anytime_best=torch.tensor([[4.0, 3.5]], dtype=torch.float64),
        wall_time_sec=0.25,
        constructed_tours=64,
        diagnostics=RunDiagnostics(
            uniform_fallback_count=1,
            candidate_fallback_count=2,
            bound_clip_count=3,
            mmas_restart_count=4,
        ),
        backend_metrics={"kernel": "test"},
    )
    metadata = {
        "study_id": "tiny",
        "variant": "as",
        "partition": "tiny",
        "seed": 17,
        "instance_ids": ["i0"],
    }
    path = tmp_path / "baseline.npz"
    _write_test_cache(path, result, metadata=metadata)
    restored = _read_test_cache(path, expected_metadata=metadata)
    assert torch.equal(restored.best_tour, result.best_tour)
    assert torch.equal(restored.best_length, result.best_length)
    assert torch.equal(restored.anytime_best, result.anytime_best)
    assert restored.diagnostics.mmas_restart_count == 4
    assert restored.backend_metrics == {"kernel": "test"}

    with np.load(path, allow_pickle=False) as payload:
        stored = json.loads(str(payload["__metadata__"].item()))
    assert stored["schema_version"] == 1


def test_study_report_closes_artifact_loop(tmp_path) -> None:
    base = load_study_spec("experiments/tsp100_gpu0_3seed/study.yaml")
    study = replace(
        base,
        output_root=tmp_path / "study",
        bootstrap_replicates=100,
    )
    for variant in study.variants:
        for root_seed in variant.seeds:
            run = (
                study.output_root
                / "train"
                / variant.name
                / f"seed-{root_seed}"
            )
            run.mkdir(parents=True)
            curve = run / "training_validation_curve.csv"
            with curve.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["generation", "scale", *CURVE_FIELDS],
                )
                writer.writeheader()
                for generation in range(1, 51):
                    writer.writerow(
                        {
                            "generation": generation,
                            "scale": 100,
                            **{
                                field: (
                                    1.0
                                    if "gap_percent" in field
                                    else -0.1
                                    if "delta_pp" in field
                                    else 0.2
                                )
                                for field in CURVE_FIELDS
                            },
                        }
                    )
            (run / "deployment_decision.json").write_text(
                json.dumps(
                    {
                        "selected_candidate_hash": f"h-{root_seed}",
                        "selection_passed_noninferiority": True,
                        "cpu_fp64_audit_passed": True,
                        "final_passed_noninferiority": True,
                        "deployed_method": "rmtgp-selected",
                    }
                ),
                encoding="utf-8",
            )

        for partition_index, partition in enumerate(study.partitions):
            scale = (50, 100, 500, 1000)[partition_index]
            records = [
                EvaluationRecord(
                    method=study.method_profile,
                    variant=variant.name,
                    champion_id=f"h-{root_seed}",
                    partition=partition,
                    distribution="uniform",
                    scale=scale,
                    instance_id=f"{partition}:i{instance}",
                    seed=replicate,
                    best_length=104.9,
                    reference_length=100.0,
                    gap_percent=4.9,
                    baseline_length=105.0,
                    baseline_gap_percent=5.0,
                    delta_pp=-0.1,
                    outcome="win",
                    best_iteration=7,
                    anytime_gap_auc=5.1,
                    wall_time_sec=0.2,
                    baseline_wall_time_sec=0.18,
                    inference_overhead_percent=11.1,
                    constructed_tours=100,
                    tours_per_second=500.0,
                    candidate_fallback_count=0,
                    uniform_fallback_count=0,
                    bound_clip_count=0,
                    gp_run_id=f"{variant.name}-seed-{root_seed}",
                    gp_root_seed=root_seed,
                    baseline_best_iteration=8,
                    baseline_anytime_gap_auc=5.2,
                    baseline_tours_per_second=550.0,
                )
                for root_seed in variant.seeds
                for instance in range(2)
                for replicate in range(3)
            ]
            write_records(
                records,
                study.output_root
                / "test"
                / variant.name
                / partition
                / "records.csv",
            )

    report = generate_study_report(study)
    assert report.is_file()
    assert (report.parent / "study_summary.json").is_file()
    assert (report.parent / "test_summary.csv").is_file()
    assert (report.parent / "train_validation_curve_as.png").is_file()
