"""GPU1 多方法消融合同、baseline 行为哈希与配对统计测试。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rmtgp_aco.ablation import (
    CORE_METHODS,
    MECHANISM_METHODS,
    AblationStudySpec,
    AblationVariant,
    ReuseStudy,
    _expected_batches,
    _expected_instances,
    _parallel_task_units,
    _program_entries,
    build_ablation_tasks,
    load_ablation_spec,
)
from rmtgp_aco.ablation_report import _paired_statistics
from rmtgp_aco.baseline import (
    BaselineArchive,
    BaselineRecord,
    backend_semantic_id,
    baseline_key,
)
from rmtgp_aco.cli import build_parser
from rmtgp_aco.config import (
    ACOConfig,
    CudaPrecision,
    ExecutionBackend,
    RuntimeConfig,
    TransitionIntegration,
)
from rmtgp_aco.evaluation import EvaluationRecord
from rmtgp_aco.manifest import load_manifest
from rmtgp_aco.spec import load_run_spec


def test_baseline_behavior_hash_ignores_gp_only_integration_fields() -> None:
    residual = ACOConfig.acotsp_default("acs", iterations=17)
    replacement = replace(
        residual,
        transition_integration=TransitionIntegration.REPLACEMENT,
        gamma_transition=0.1,
        gamma_pheromone=0.5,
    )
    assert residual.config_hash != replacement.config_hash
    assert residual.baseline_behavior_hash == replacement.baseline_behavior_hash


def test_cuda_v2_has_an_independent_baseline_semantic_domain() -> None:
    """v2 的归约轨迹不同，不得误读 legacy CUDA baseline cache。"""

    legacy = backend_semantic_id(ExecutionBackend.CUDA_FUSED_FP32)
    tiled_v2 = backend_semantic_id(ExecutionBackend.CUDA_TILED_V2)
    assert tiled_v2 != legacy


def test_cuda_v2_baseline_semantic_tracks_precision_and_lanes() -> None:
    """可能改变选择轨迹的 profile 不得共享 baseline archive。"""

    base = RuntimeConfig(
        aco_backend=ExecutionBackend.CUDA_TILED_V2,
        cuda_precision=CudaPrecision.FP32_FAST,
        cuda_candidate_lanes=8,
    )
    standard = replace(base, cuda_precision=CudaPrecision.FP32)
    four_lanes = replace(base, cuda_candidate_lanes=4)
    semantics = {
        backend_semantic_id(ExecutionBackend.CUDA_TILED_V2, runtime)
        for runtime in (base, standard, four_lanes)
    }
    assert len(semantics) == 3


def test_cuda_v2_baseline_semantic_tracks_local_search_launch_shape() -> None:
    base = RuntimeConfig(aco_backend=ExecutionBackend.CUDA_TILED_V2)
    configurations = (
        base,
        replace(base, cuda_ls_warps_per_block=4),
        replace(base, cuda_three_opt_block_threads=512),
    )
    semantics = {
        backend_semantic_id(ExecutionBackend.CUDA_TILED_V2, runtime)
        for runtime in configurations
    }
    assert len(semantics) == len(configurations)


def test_cuda_v2_baseline_semantic_uses_manifest_selection(tmp_path) -> None:
    manifest = tmp_path / "tuning.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": ExecutionBackend.CUDA_TILED_V2.value,
                "selected": {
                    "provider": "raw_cuda",
                    "precision": "fp32_fast",
                    "candidate_lanes": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    runtime = RuntimeConfig(
        aco_backend=ExecutionBackend.CUDA_TILED_V2,
        cuda_precision=CudaPrecision.FP32,
        cuda_candidate_lanes=4,
        cuda_tuning_manifest=str(manifest),
    )
    semantic = backend_semantic_id(
        ExecutionBackend.CUDA_TILED_V2,
        runtime,
    )
    assert semantic.endswith("raw_cuda-fp32_fast-lanes8")


def test_v2_baseline_archive_is_reindexed_by_behavior(tmp_path) -> None:
    """旧 archive 的 residual config 可供等价 replacement baseline 使用。"""

    residual = ACOConfig.acotsp_default("as", iterations=3)
    replacement = replace(
        residual,
        transition_integration=TransitionIntegration.REPLACEMENT,
    )
    semantic = backend_semantic_id("torch")
    record = BaselineRecord(
        key=baseline_key(
            coordinate_hash="coordinate",
            seed=7,
            config_hash=residual.config_hash,
            backend_semantic=semantic,
        ),
        coordinate_hash="coordinate",
        instance_id="i0",
        scale=50,
        seed=7,
        config_hash=residual.config_hash,
        baseline_behavior_hash="",
        backend_semantic=semantic,
        best_length=10.0,
        reference_length=9.0,
        reference_gap_percent=100.0 / 9.0,
        best_iteration=2,
        anytime_gap_auc=12.0,
        candidate_fallback_count=0,
        uniform_fallback_count=0,
        bound_clip_count=0,
        mmas_restart_count=0,
    )
    fields = [
        field
        for field in BaselineRecord.__dataclass_fields__
        if field != "baseline_behavior_hash"
    ]
    payload = {
        field: np.asarray([getattr(record, field)])
        for field in fields
    }
    payload["__metadata__"] = np.asarray(
        json.dumps({"schema_version": 2, "records": 1})
    )
    np.savez_compressed(tmp_path / "old.npz", **payload)

    archive = BaselineArchive(
        tmp_path,
        replacement,
        "torch",
        require=True,
    )
    assert archive.records == 1


@pytest.mark.skipif(
    not Path("runs/tsp100-gpu0-3seed/study_state.json").is_file(),
    reason="仓库 checkout 不含已完成主 study 的外部 artifacts",
)
def test_repository_ablation_contract_and_task_matrix() -> None:
    study = load_ablation_spec(
        "experiments/tsp100_ablation_gpu1_3seed/study.yaml"
    )
    tasks = build_ablation_tasks(study)
    by_kind: dict[str, int] = {}
    for task in tasks:
        by_kind[task.kind] = by_kind.get(task.kind, 0) + 1
    assert tuple(method.name for method in study.methods) == CORE_METHODS
    assert by_kind == {
        "preflight": 24,
        "train": 63,
        "test": 21,
        "efficiency": 3,
        "report": 1,
    }
    assert len(tasks) == 112
    assert tasks[0].task_id == "preflight-as-legacy"
    assert tasks[-1].task_id == "report"
    assert all(
        "--method-profile" in task.command
        for task in tasks
        if task.kind in {"preflight", "train"}
    )
    test_units = _parallel_task_units(tasks, kind="test")
    assert len(test_units) == 15
    assert sum(len(unit) == 2 for unit in test_units) == 6
    assert sum(len(unit) == 1 for unit in test_units) == 9
    assert all(
        len(
            {
                (task.meta()["variant"], task.meta()["partition"])
                for task in unit
            }
        )
        == 1
        and {task.meta()["group"] for task in unit}
        in ({"residual", "replacement"}, {"final"})
        for unit in test_units
    )
    assert study.ablation_partitions == (
        "tsp100_uniform",
        "tsp500_uniform",
    )
    assert len(
        _program_entries(
            study,
            variant_name="as",
            partition="tsp100_uniform",
            group="residual",
        )
    ) == 27
    assert len(
        _program_entries(
            study,
            variant_name="as",
            partition="tsp100_uniform",
            group="replacement",
        )
    ) == 6
    final_entries = _program_entries(
        study,
        variant_name="as",
        partition="tsp500_cluster",
        group="final",
    )
    assert len(final_entries) == 3
    assert {entry.gp_root_seed for entry in final_entries} == {1001, 1002, 1003}
    final_task = next(
        task
        for task in tasks
        if task.task_id == "test-as-tsp500_cluster-final"
    )
    parsed = build_parser().parse_args(final_task.command[3:])
    assert parsed.group == "final"

    spec = load_run_spec(study.variants[0].config)
    expected_tsplib = sum(
        item.instances
        for item in load_manifest(study.manifest).files
        if item.split == "test"
        and item.distribution == "tsplib"
        and item.scale <= 500
    )
    assert _expected_instances(study, spec, "tsplib_le500") == expected_tsplib
    assert _expected_batches(study, spec, "tsplib_le500") <= expected_tsplib


def _record(
    *,
    method: str,
    root_seed: int,
    instance: int,
    aco_seed: int,
    gap: float,
) -> EvaluationRecord:
    baseline_gap = 5.0
    return EvaluationRecord(
        method=method,
        variant="as",
        champion_id=f"{method}-{root_seed}",
        partition="tiny",
        distribution="uniform",
        scale=100,
        instance_id=f"i{instance}",
        seed=aco_seed,
        best_length=100.0 + gap,
        reference_length=100.0,
        gap_percent=gap,
        baseline_length=105.0,
        baseline_gap_percent=baseline_gap,
        delta_pp=gap - baseline_gap,
        outcome="win" if gap < baseline_gap else "loss",
        best_iteration=2,
        anytime_gap_auc=gap + 0.5,
        wall_time_sec=float("nan"),
        baseline_wall_time_sec=1.0,
        inference_overhead_percent=float("nan"),
        constructed_tours=100,
        tours_per_second=float("nan"),
        candidate_fallback_count=0,
        uniform_fallback_count=0,
        bound_clip_count=0,
        gp_run_id=f"as:{method}:seed-{root_seed}",
        gp_root_seed=root_seed,
        baseline_best_iteration=3,
        baseline_anytime_gap_auc=5.5,
        baseline_tours_per_second=100.0,
    )


def test_ablation_pairing_and_factorial_estimands_are_exact(tmp_path) -> None:
    offsets = {
        "legacy": 0.2,
        "matched-replace": 0.0,
        "tr-rgp": -1.0,
        "ph-rgp": -0.5,
        "rmtgp-core-f0": 0.0,
        "rmtgp-core-f1": -1.0,
        "rmtgp-full-f0": -2.0,
        "rmtgp-full-f1": -4.0,
        "rmtgp-full-f1-drop-transition": -0.5,
        "rmtgp-full-f1-drop-pheromone": -1.0,
        "rmtgp-full-f1-shuffle-r1": -2.5,
        "rmtgp-full-f1-shuffle-r2": -2.0,
    }
    records = [
        _record(
            method=method,
            root_seed=root_seed,
            instance=instance,
            aco_seed=aco_seed,
            gap=5.0 + offset + 0.01 * instance + 0.001 * aco_seed,
        )
        for method, offset in offsets.items()
        for root_seed in (11, 12, 13)
        for instance in range(4)
        for aco_seed in range(3)
    ]
    study = AblationStudySpec(
        study_id="tiny",
        output_root=tmp_path,
        phase="pilot",
        test_root_seed=9,
        test_seeds=3,
        partitions=("tiny",),
        ablation_partitions=("tiny",),
        methods=(),
        variants=(
            AblationVariant(
                name="as",
                config=tmp_path / "unused.yaml",
                seeds=(11, 12, 13),
            ),
        ),
        reuse=ReuseStudy(config=tmp_path / "unused", root=tmp_path),
        manifest=tmp_path / "manifest.json",
        bootstrap_replicates=100,
        source_path=tmp_path / "study.yaml",
    )
    primary, mechanism, factorial = _paired_statistics(study, records)
    contrasts = {row["contrast"]: row for row in primary}
    assert contrasts["TR-RGP − Matched-Replace"]["estimate_pp"] == pytest.approx(
        -1.0
    )
    assert contrasts["RMTGP-Full-F1 − TR-RGP"]["estimate_pp"] == pytest.approx(
        -3.0
    )
    posthoc = {row["contrast"]: row for row in mechanism}
    assert posthoc["Full-F1 − shuffle-r1"]["estimate_pp"] == pytest.approx(-1.5)

    factorial_values = {
        row["contrast"]: row["estimate_pp"] for row in factorial
    }
    assert factorial_values["terminal_full_minus_core"] == pytest.approx(-2.5)
    assert factorial_values["function_f1_minus_f0"] == pytest.approx(-1.5)
    assert factorial_values["terminal_function_interaction"] == pytest.approx(
        -1.0
    )
    assert all(row["lower_95"] == pytest.approx(row["estimate_pp"]) for row in factorial)
    assert set(offsets) == set((*CORE_METHODS, *MECHANISM_METHODS))
