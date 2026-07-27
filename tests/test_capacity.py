"""31/62 节点容量敏感性合同、任务矩阵与配对 estimand 测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from rmtgp_aco.capacity import (
    CAPACITY_METHODS,
    SOURCE_CAPACITY,
    TARGET_CAPACITY,
    CapacitySource,
    CapacityStudySpec,
    CapacityVariant,
    build_capacity_tasks,
    capacity_method,
    load_capacity_spec,
)
from rmtgp_aco.capacity_report import (
    CAPACITY_EVALUATED_METHODS,
    _capacity_contrasts,
)
from rmtgp_aco.evaluation import EvaluationRecord
from rmtgp_aco.spec import load_run_spec


@pytest.mark.skipif(
    not Path("runs/tsp100-gpu0-3seed/study_state.json").is_file(),
    reason="checkout 不含已完成主 study 的外部 artifacts",
)
def test_repository_capacity_contract_and_task_matrix() -> None:
    study = load_capacity_spec("experiments/tsp100_capacity_sensitivity_3seed/study.yaml")
    tasks = build_capacity_tasks(study)
    by_kind: dict[str, int] = {}
    for task in tasks:
        by_kind[task.kind] = by_kind.get(task.kind, 0) + 1
    assert by_kind == {
        "preflight": 9,
        "train": 27,
        "test": 6,
        "efficiency": 3,
        "report": 1,
    }
    assert len(tasks) == 46
    assert tasks[0].task_id == "preflight-as-tr-rgp-n62"
    assert tasks[-1].task_id == "report"
    assert study.partitions == ("tsp100_uniform", "tsp500_uniform")

    for variant in study.variants:
        single = load_run_spec(variant.single_config).experiment.gp
        multi = load_run_spec(variant.multi_config).experiment.gp
        assert (
            single.max_nodes_per_tree,
            single.max_total_nodes,
            single.max_depth,
        ) == (62, 62, 5)
        assert (
            multi.max_nodes_per_tree,
            multi.max_total_nodes,
            multi.max_depth,
        ) == (31, 62, 5)

    train = [task for task in tasks if task.kind == "train"]
    assert {(task.meta()["variant"], task.meta()["method"]) for task in train} == {
        (variant.name, method) for variant in study.variants for method in CAPACITY_METHODS
    }
    assert all("--schedule" in task.command for task in train)
    assert all("--baseline-archive" in task.command for task in train)


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


def test_capacity_estimands_and_run_instance_blocks_are_exact(
    tmp_path: Path,
) -> None:
    offsets = {
        capacity_method("tr-rgp", 31): 0.0,
        capacity_method("tr-rgp", 62): -1.0,
        capacity_method("ph-rgp", 31): 0.5,
        capacity_method("ph-rgp", 62): 0.0,
        capacity_method("rmtgp-full-f1", 31): -2.0,
        capacity_method("rmtgp-full-f1", 62): -3.0,
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
    study = CapacityStudySpec(
        study_id="tiny-capacity",
        output_root=tmp_path,
        phase="pilot",
        test_root_seed=9,
        test_seeds=3,
        partitions=("tiny",),
        variants=(
            CapacityVariant(
                name="as",
                single_config=tmp_path / "single.yaml",
                multi_config=tmp_path / "multi.yaml",
                seeds=(11, 12, 13),
            ),
        ),
        source=CapacitySource(
            config=tmp_path / "source.yaml",
            root=tmp_path,
            study=cast(Any, None),
        ),
        manifest=tmp_path / "manifest.json",
        bootstrap_replicates=100,
        source_path=tmp_path / "study.yaml",
    )
    contrasts = _capacity_contrasts(study, records)
    values = {row["formula"]: row for row in contrasts}
    assert values["tr62-tr31"]["estimate_pp"] == pytest.approx(-1.0)
    assert values["ph62-ph31"]["estimate_pp"] == pytest.approx(-0.5)
    assert values["mt62-mt31"]["estimate_pp"] == pytest.approx(-1.0)
    assert values["mt31-tr31"]["estimate_pp"] == pytest.approx(-2.0)
    assert values["mt62-tr62"]["estimate_pp"] == pytest.approx(-2.0)
    assert values["did-mt-vs-tr"]["estimate_pp"] == pytest.approx(0.0)
    assert values["did-mt-vs-ph"]["estimate_pp"] == pytest.approx(-0.5)
    assert all(row["run_instance_blocks"] == 12 for row in contrasts)
    assert set(offsets) == set(CAPACITY_EVALUATED_METHODS)
    assert SOURCE_CAPACITY == 31
    assert TARGET_CAPACITY == 62
