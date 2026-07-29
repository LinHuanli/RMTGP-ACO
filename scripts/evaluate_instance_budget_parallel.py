#!/usr/bin/env python3
"""并行测试实例预算实验的最终候选。

这里只加载每个独立 GP run 的 ``selected_candidate.pkl``。原始 ACO 和九个
最终候选被打包为十个 programs。CUDA v2 再把 program×instance tasks
确定性分配到可见 GPU。任何中间 population individual 都不会进入测试。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any

import numpy as np

from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
from rmtgp_aco.config import GPUMode
from rmtgp_aco.evaluation import (
    EvaluationRecord,
    compile_champion,
    load_champion,
    read_records,
    records_from_paired_results,
    study_test_seed,
    write_records,
)
from rmtgp_aco.model import RunDiagnostics, RunResult
from rmtgp_aco.runtime import configure_runtime
from rmtgp_aco.sampling import iter_problem_batches
from rmtgp_aco.spec import load_run_spec

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "tsp100_instance_budget_single_gpu"
RUN_ROOT = ROOT / "runs" / "tsp100-instance-budget-single-gpu"
CONFIG = EXPERIMENT / "configs" / "acs_n128.yaml"
BUDGETS = (32, 64, 128)
SEEDS = (2001, 2002, 2003)
DEFAULT_PARTITIONS = (
    "tsp50_uniform",
    "tsp100_uniform",
    "tsp500_uniform",
    "tsp1000_uniform",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    budget: int
    gp_seed: int
    champion_id: str
    transition: Any
    pheromone: Any

    @property
    def method(self) -> str:
        return f"rmtgp-n{self.budget}"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _wait_for_training(poll_seconds: int) -> None:
    state_path = RUN_ROOT / "campaign_state.json"
    while True:
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            status = state.get("status")
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError(
                    f"训练 campaign 失败: {state.get('failures', [])}"
                )
            print(
                f"等待训练：active={state.get('active', {})} "
                f"pending={len(state.get('pending', []))}",
                flush=True,
            )
        else:
            print("等待训练：campaign_state.json 尚未生成", flush=True)
        time.sleep(poll_seconds)


def _load_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for budget in BUDGETS:
        for gp_seed in SEEDS:
            run = RUN_ROOT / "train" / f"n{budget}" / f"seed-{gp_seed}"
            manifest_path = run / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"训练 manifest 缺失: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise RuntimeError(f"训练尚未完成: {run}")
            champion = load_champion(run / "selected_candidate.pkl")
            transition, pheromone = compile_champion(champion)
            candidates.append(
                Candidate(
                    budget=budget,
                    gp_seed=gp_seed,
                    champion_id=champion.structural_hash,
                    transition=transition,
                    pheromone=pheromone,
                )
            )
    return candidates


def _member_result(
    population: Any,
    index: int,
    *,
    batch_size: int,
    ants: int,
    iterations: int,
) -> RunResult:
    diagnostic = population.diagnostics[index]
    return RunResult(
        best_tour=population.best_tour[index],
        best_length=population.best_length[index],
        best_iteration=population.best_iteration[index],
        anytime_best=population.anytime_best[index],
        # programs 同时执行，不能把 campaign 墙钟任意分摊到单个候选。
        wall_time_sec=float("nan"),
        constructed_tours=batch_size * ants * iterations,
        diagnostics=RunDiagnostics(
            candidate_fallback_count=int(diagnostic[0].item()),
            uniform_fallback_count=int(diagnostic[1].item()),
            bound_clip_count=int(diagnostic[2].item()),
            mmas_restart_count=int(diagnostic[3].item()),
        ),
        backend_metrics={
            **population.backend_metrics,
            "timing_scope": "packed-final-candidate-test",
        },
    )


def _valid_shard(
    path: Path,
    *,
    candidates: list[Candidate],
    batch: Any,
    partition: str,
    seed: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        records = read_records([path])
    except (OSError, TypeError, ValueError):
        return False
    expected = len(candidates) * batch.batch_size
    allowed = {
        (candidate.method, candidate.gp_seed, candidate.champion_id)
        for candidate in candidates
    }
    keys = {
        (
            record.method,
            record.gp_root_seed,
            record.champion_id,
            record.instance_id,
        )
        for record in records
    }
    return (
        len(records) == expected
        and len(keys) == expected
        and {
            (record.method, record.gp_root_seed, record.champion_id)
            for record in records
        }
        == allowed
        and {record.partition for record in records} == {partition}
        and {record.seed for record in records} == {seed}
        and {record.instance_id for record in records}
        == set(batch.instance_ids)
    )


def _evaluate_partition(
    *,
    partition: str,
    candidates: list[Candidate],
    runtime: Any,
    batch_size: int,
    test_seeds: int,
    test_root_seed: int,
) -> tuple[Path, dict[str, object]]:
    spec = load_run_spec(CONFIG)
    config = spec.experiment.aco
    partition_spec = spec.data.test[partition]
    paths = spec.data.test_paths(partition)
    output = RUN_ROOT / "parallel-test" / partition
    started = perf_counter()
    all_records: list[EvaluationRecord] = []
    campaigns: list[dict[str, object]] = []
    instances = 0
    shard_count = 0

    for batch_number, batch in enumerate(
        iter_problem_batches(
            paths,
            batch_size=batch_size,
            candidate_size=config.candidate_size,
            dtype=config.dtype,
            device=config.device,
            min_scale=partition_spec.min_scale,
            max_scale=partition_spec.max_scale,
        )
    ):
        instances += batch.batch_size
        for replicate in range(test_seeds):
            seed = study_test_seed(
                test_root_seed,
                partition,
                batch_number,
                replicate,
            )
            shard = (
                output
                / f"aco-{replicate:02d}"
                / f"batch-{batch_number:04d}.csv"
            )
            metrics_path = shard.with_suffix(".metrics.json")
            if _valid_shard(
                shard,
                candidates=candidates,
                batch=batch,
                partition=partition,
                seed=seed,
            ) and metrics_path.is_file():
                records = read_records([shard])
                campaign = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
            else:
                programs = [(None, None), *[
                    (candidate.transition, candidate.pheromone)
                    for candidate in candidates
                ]]
                population = solve_population_cuda_anytime(
                    batch,
                    config,
                    programs,
                    seed=seed,
                    runtime=runtime,
                )
                baseline = _member_result(
                    population,
                    0,
                    batch_size=batch.batch_size,
                    ants=config.resolve_ants(batch.n),
                    iterations=config.iterations,
                )
                records = []
                for index, candidate_info in enumerate(candidates, start=1):
                    candidate = _member_result(
                        population,
                        index,
                        batch_size=batch.batch_size,
                        ants=config.resolve_ants(batch.n),
                        iterations=config.iterations,
                    )
                    records.extend(
                        records_from_paired_results(
                            method=candidate_info.method,
                            champion_id=candidate_info.champion_id,
                            partition=partition,
                            distribution=partition_spec.distribution,
                            batch=batch,
                            seed=seed,
                            candidate=candidate,
                            baseline=baseline,
                            config=config,
                            gp_run_id=(
                                f"n{candidate_info.budget}-"
                                f"seed-{candidate_info.gp_seed}"
                            ),
                            gp_root_seed=candidate_info.gp_seed,
                        )
                    )
                write_records(records, shard)
                campaign = {
                    "schema_version": 1,
                    "partition": partition,
                    "batch_number": batch_number,
                    "replicate": replicate,
                    "seed": seed,
                    "instances": batch.batch_size,
                    "final_candidates": len(candidates),
                    "programs_including_baseline": len(programs),
                    "wall_time_sec": population.wall_time_sec,
                    "constructed_tours": population.constructed_tours,
                    "backend_metrics": population.backend_metrics,
                }
                _atomic_json(metrics_path, campaign)
            all_records.extend(records)
            campaigns.append(campaign)
            shard_count += 1
            print(
                f"parallel-test={partition} shard={shard_count} "
                f"batch={batch_number} aco={replicate} "
                f"elapsed={((perf_counter() - started) / 60.0):.2f}min",
                flush=True,
            )

    expected = instances * test_seeds * len(candidates)
    keys = {
        (
            record.method,
            record.gp_root_seed,
            record.instance_id,
            record.seed,
        )
        for record in all_records
    }
    if len(all_records) != expected or len(keys) != expected:
        raise RuntimeError(
            f"{partition}: rows={len(all_records)}, keys={len(keys)}, "
            f"expected={expected}"
        )
    all_records.sort(
        key=lambda record: (
            record.method,
            record.gp_root_seed,
            record.instance_id,
            record.seed,
        )
    )
    merged = write_records(all_records, output / "records.csv")
    total_campaign_seconds = sum(
        float(item["wall_time_sec"]) for item in campaigns
    )
    summary: dict[str, object] = {
        "partition": partition,
        "instances": instances,
        "test_seeds": test_seeds,
        "final_candidates": len(candidates),
        "rows": expected,
        "batch_size": batch_size,
        "wall_time_sec": perf_counter() - started,
        "gpu_campaign_seconds": total_campaign_seconds,
        "device_count": len(runtime.gpu_devices),
        "packed_programs_including_baseline": len(candidates) + 1,
    }
    _atomic_json(
        output / "evaluation_manifest.json",
        {
            "schema_version": 1,
            "status": "completed",
            "selected_candidate_tested": True,
            "population_individuals_tested": False,
            "execution": (
                "baseline + nine final candidates packed; "
                "program×instance LPT multi-GPU sharding"
            ),
            "test_root_seed": test_root_seed,
            "candidates": [
                {
                    "budget": candidate.budget,
                    "gp_seed": candidate.gp_seed,
                    "champion_id": candidate.champion_id,
                }
                for candidate in candidates
            ],
            **summary,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return merged, summary


def _quality_summary(
    paths: list[Path],
    partition_summaries: list[dict[str, object]],
) -> dict[str, object]:
    records = read_records(paths)
    grouped: dict[tuple[str, int, str, int], list[EvaluationRecord]] = {}
    for record in records:
        key = (
            record.partition,
            record.gp_root_seed,
            record.instance_id,
            int(record.method.removeprefix("rmtgp-n")),
        )
        grouped.setdefault(key, []).append(record)
    aggregated = {
        key: {
            "gap": fmean(item.gap_percent for item in values),
            "delta": fmean(item.delta_pp for item in values),
        }
        for key, values in grouped.items()
    }

    by_partition: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for partition in sorted({record.partition for record in records}):
        for budget in BUDGETS:
            values = [
                value
                for (part, _gp_seed, _instance, item_budget), value
                in aggregated.items()
                if part == partition and item_budget == budget
            ]
            by_partition.append(
                {
                    "partition": partition,
                    "budget": budget,
                    "program_instance_units": len(values),
                    "mean_gap_percent": fmean(
                        value["gap"] for value in values
                    ),
                    "median_gap_percent": median(
                        value["gap"] for value in values
                    ),
                    "mean_delta_pp": fmean(
                        value["delta"] for value in values
                    ),
                }
            )
        for larger, smaller in ((64, 32), (128, 32), (128, 64)):
            paired: list[float] = []
            for (
                part,
                gp_seed,
                instance,
                budget,
            ), value in aggregated.items():
                if part != partition or budget != larger:
                    continue
                other = aggregated[
                    (partition, gp_seed, instance, smaller)
                ]
                paired.append(value["gap"] - other["gap"])
            array = np.asarray(paired, dtype=np.float64)
            standard_error = (
                float(array.std(ddof=1) / np.sqrt(array.size))
                if array.size > 1
                else 0.0
            )
            comparisons.append(
                {
                    "partition": partition,
                    "larger_budget": larger,
                    "smaller_budget": smaller,
                    "paired_units": int(array.size),
                    "mean_gap_change_pp": float(array.mean()),
                    "median_gap_change_pp": float(np.median(array)),
                    "normal_ci95_low": float(
                        array.mean() - 1.96 * standard_error
                    ),
                    "normal_ci95_high": float(
                        array.mean() + 1.96 * standard_error
                    ),
                    "larger_budget_better_fraction": float(
                        np.mean(array < 0.0)
                    ),
                }
            )
    return {
        "schema_version": 1,
        "statistical_unit": (
            "GP-seed × instance; three ACO seeds averaged first"
        ),
        "sign_convention": (
            "mean_gap_change_pp < 0 means the larger training budget is better"
        ),
        "partitions": by_partition,
        "paired_budget_comparisons": comparisons,
        "performance": partition_summaries,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--physical-gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--test-root-seed", type=int, default=9001)
    parser.add_argument(
        "--partitions",
        nargs="+",
        default=list(DEFAULT_PARTITIONS),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.poll_seconds, args.batch_size, args.test_seeds) < 1:
        raise ValueError("poll、batch 和 test seeds 必须为正整数")
    if len(set(args.physical_gpus)) != len(args.physical_gpus):
        raise ValueError("physical GPU 不得重复")
    if len(args.physical_gpus) > 2:
        raise ValueError("当前 CUDA v2 单次 LPT shard 最多使用两张 GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and len(visible.split(",")) < len(args.physical_gpus):
        raise ValueError("CUDA_VISIBLE_DEVICES 少于请求的 physical GPUs")
    if args.wait_for_training:
        _wait_for_training(args.poll_seconds)

    candidates = _load_candidates()
    spec = load_run_spec(CONFIG)
    logical_devices = tuple(range(len(args.physical_gpus)))
    runtime = replace(
        spec.experiment.runtime,
        gpu_devices=logical_devices,
        gpu_mode=(
            GPUMode.DUAL
            if len(logical_devices) >= 2
            else GPUMode.SINGLE
        ),
    )
    configure_runtime(runtime)
    record_paths: list[Path] = []
    partition_summaries: list[dict[str, object]] = []
    for partition in args.partitions:
        if partition not in spec.data.test:
            raise KeyError(f"未知 partition: {partition}")
        path, summary = _evaluate_partition(
            partition=partition,
            candidates=candidates,
            runtime=runtime,
            batch_size=args.batch_size,
            test_seeds=args.test_seeds,
            test_root_seed=args.test_root_seed,
        )
        record_paths.append(path)
        partition_summaries.append(summary)
    _atomic_json(
        RUN_ROOT / "parallel-test" / "summary.json",
        _quality_summary(record_paths, partition_summaries),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
