"""评测长表与统计检验测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rmtgp_aco.config import ACOConfig
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.evaluation import (
    EvaluationRecord,
    evaluate_batches,
    read_records,
    write_records,
)
from rmtgp_aco.stats import (
    factorial_contrasts,
    friedman_test,
    hierarchical_bootstrap_delta,
    paired_wilcoxon_holm,
    summarize_quality,
)


def test_paired_evaluation_csv_roundtrip(small_instances, tmp_path) -> None:
    batch = make_problem_batch(small_instances, candidate_size=2)
    config = replace(
        ACOConfig.acotsp_default("acs", iterations=2),
        ants=3,
        candidate_size=2,
    )
    records = evaluate_batches(
        [batch],
        config,
        method="ACO",
        champion_id="baseline",
        partition="tiny",
        distribution="uniform",
        root_seed=17,
        seeds_per_batch=2,
    )
    assert len(records) == 4
    assert all(record.delta_pp == 0.0 for record in records)
    assert all(record.inference_overhead_percent == 0.0 for record in records)
    path = write_records(records, tmp_path / "records.csv")
    assert read_records([path]) == records


def _stat_record(method: str, instance: int, seed: int, gap: float) -> EvaluationRecord:
    baseline_gap = 5.0
    return EvaluationRecord(
        method=method,
        variant="as",
        champion_id=f"{method}-run1",
        partition="test",
        distribution="uniform",
        scale=50,
        instance_id=f"i{instance}",
        seed=seed,
        best_length=100.0 + gap,
        reference_length=100.0,
        gap_percent=gap,
        baseline_length=105.0,
        baseline_gap_percent=baseline_gap,
        delta_pp=gap - baseline_gap,
        outcome="win" if gap < baseline_gap else "loss",
        best_iteration=1,
        anytime_gap_auc=gap,
        wall_time_sec=1.0,
        baseline_wall_time_sec=0.9,
        inference_overhead_percent=100.0 / 9.0,
        constructed_tours=100,
        tours_per_second=100.0,
        candidate_fallback_count=0,
        uniform_fallback_count=0,
        bound_clip_count=0,
    )


def test_nonparametric_statistics_and_hierarchical_bootstrap() -> None:
    records = [
        _stat_record(method, instance, seed, gap)
        for method, offset in (("RMTGP", -1.0), ("TR-RGP", 0.0), ("ACO", 1.0))
        for instance in range(6)
        for seed in range(2)
        for gap in [5.0 + offset + 0.1 * instance + 0.01 * seed]
    ]
    summaries = summarize_quality(records)
    assert len(summaries) == 3
    omnibus = friedman_test(records)
    assert omnibus.blocks == 6
    assert omnibus.p_value < 0.05
    pairwise = paired_wilcoxon_holm(records, reference_method="RMTGP")
    assert {item.compared_method for item in pairwise} == {"ACO", "TR-RGP"}
    assert all(item.mean_difference_pp < 0 for item in pairwise)
    intervals = hierarchical_bootstrap_delta(records, replicates=100, seed=9)
    repeated = hierarchical_bootstrap_delta(records, replicates=100, seed=9)
    assert intervals == repeated
    assert len(intervals) == 3
    rmtgp = next(item for item in intervals if item.method == "RMTGP")
    assert rmtgp.lower_95 <= rmtgp.estimate <= rmtgp.upper_95


def test_friedman_rejects_two_methods() -> None:
    records = [
        _stat_record(method, instance, 0, 5.0)
        for method in ("A", "B")
        for instance in range(3)
    ]
    with pytest.raises(ValueError, match="三个"):
        friedman_test(records)


def test_factorial_contrasts_preserve_run_instance_seed_pairing() -> None:
    offsets = {
        "Core-F0": 0.0,
        "Core-F1": -1.0,
        "Full-F0": -2.0,
        "Full-F1": -4.0,
    }
    records = [
        replace(
            _stat_record(
                method,
                instance,
                aco_seed,
                5.0 + offset + 0.01 * run_seed,
            ),
            gp_run_id=f"run-{run_seed}",
            gp_root_seed=run_seed,
        )
        for method, offset in offsets.items()
        for run_seed in (11, 12)
        for instance in range(4)
        for aco_seed in range(2)
    ]
    contrasts = factorial_contrasts(
        records,
        core_f0="Core-F0",
        core_f1="Core-F1",
        full_f0="Full-F0",
        full_f1="Full-F1",
        replicates=100,
        seed=3,
    )
    observed = {item.contrast: item.estimate_pp for item in contrasts}
    assert observed["terminal_full_minus_core"] == pytest.approx(-2.5)
    assert observed["function_f1_minus_f0"] == pytest.approx(-1.5)
    assert observed["terminal_function_interaction"] == pytest.approx(-1.0)
    assert all(item.runs == 2 for item in contrasts)
