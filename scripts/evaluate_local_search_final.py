#!/usr/bin/env python3
"""局部搜索研究的最终 TSP100/TSP500 评测、统计与作图。

质量评测只加载每个独立 GP run 最终选定的个体。ACO+2-opt 与六个最终
个体在同一 CUDA population 中打包运行。ACO+3-opt 使用相同随机种子单独
运行。在线时间另以单 program 运行测量，编译时间不计入算法在线时间。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

import numpy as np

from rmtgp_aco.aco_cuda import (
    solve_population_cuda,
    solve_population_cuda_anytime,
)
from rmtgp_aco.config import (
    ACOVariant,
    ExecutionBackend,
    GPUMode,
    LocalSearch,
)
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
CONFIG = ROOT / "experiments" / "tsp100_local_search_3seed" / "config.yaml"
RUN_ROOT = ROOT / "runs" / "tsp100-local-search-3seed"
TUNING = ROOT / "configs/cuda_tuning/rtx_pro5000_blackwell_sm120_v1.json"
VARIANTS = tuple(ACOVariant)
ENVIRONMENTS = ("none", "two_opt")
GP_SEEDS = (2001, 2002, 2003)
PARTITIONS = ("tsp100_uniform", "tsp500_uniform")
MAX_INSTANCES = {"tsp100_uniform": 128, "tsp500_uniform": 32}
BATCH_SIZES = {"tsp100_uniform": 128, "tsp500_uniform": 32}
RUNTIME_INSTANCES = {"tsp100_uniform": 32, "tsp500_uniform": 8}
TEST_ROOT_SEED = 94173
TEST_SEEDS = 3
TEST_ITERATIONS = 5000
NONINFERIORITY_TOLERANCE_PP = 0.10


@dataclass(frozen=True, slots=True)
class Candidate:
    """一个最终选定的 GP 个体及其已编译双树。"""

    training_environment: str
    gp_seed: int
    champion_id: str
    transition: Any
    pheromone: Any

    @property
    def method(self) -> str:
        if self.training_environment == "none":
            return "rmtgp-nols-2opt"
        return "rmtgp-ls-2opt"

    @property
    def gp_run_id(self) -> str:
        return f"{self.training_environment}-seed-{self.gp_seed}"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"不能写空 CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _variant_parameters(variant: ACOVariant) -> dict[str, float]:
    return {
        ACOVariant.AS: {"rho": 0.5, "q0": 0.0},
        ACOVariant.ACS: {"rho": 0.1, "q0": 0.98},
        ACOVariant.MMAS: {"rho": 0.2, "q0": 0.0},
    }[variant]


def _experiment(variant: ACOVariant, search: LocalSearch):
    spec = load_run_spec(CONFIG)
    parameters = _variant_parameters(variant)
    aco = replace(
        spec.experiment.aco,
        variant=variant,
        ants=32,
        iterations=TEST_ITERATIONS,
        alpha=1.0,
        beta=2.0,
        rho=parameters["rho"],
        q0=parameters["q0"],
        xi=0.1,
        local_search=search,
        local_search_candidate_size=20,
        local_search_dlb=True,
    )
    runtime = replace(
        spec.experiment.runtime,
        aco_backend=ExecutionBackend.CUDA_TILED_V2,
        gpu_devices=(0,),
        gpu_mode=GPUMode.SINGLE,
        cuda_tuning_manifest=str(TUNING),
    )
    return spec, replace(spec.experiment, aco=aco, runtime=runtime)


def _load_candidates(variant: ACOVariant) -> list[Candidate]:
    candidates: list[Candidate] = []
    for environment in ENVIRONMENTS:
        for gp_seed in GP_SEEDS:
            run = (
                RUN_ROOT
                / "train"
                / variant.value
                / environment
                / f"seed-{gp_seed}"
            )
            manifest_path = run / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"训练 manifest 缺失: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise RuntimeError(f"训练未完成: {run}")
            champion_path = run / "selected_candidate_2opt.pkl"
            champion = load_champion(champion_path)
            transition, pheromone = compile_champion(champion)
            candidates.append(
                Candidate(
                    training_environment=environment,
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
    wall_time_sec: float = float("nan"),
) -> RunResult:
    diagnostic = population.diagnostics[index]
    values = [
        int(diagnostic[item].item()) if diagnostic.numel() > item else 0
        for item in range(8)
    ]
    return RunResult(
        best_tour=population.best_tour[index],
        best_length=population.best_length[index],
        best_iteration=population.best_iteration[index],
        anytime_best=population.anytime_best[index],
        wall_time_sec=wall_time_sec,
        constructed_tours=batch_size * ants * iterations,
        diagnostics=RunDiagnostics(
            candidate_fallback_count=values[0],
            uniform_fallback_count=values[1],
            bound_clip_count=values[2],
            mmas_restart_count=values[3],
            local_search_move_count=values[4],
            local_search_candidate_check_count=values[5],
            local_search_improved_tour_count=values[6],
            local_search_pass_count=values[7],
        ),
        backend_metrics={
            **population.backend_metrics,
            "timing_scope": "packed-quality-test",
        },
    )


def _expected_members(candidates: list[Candidate]) -> set[tuple[str, str]]:
    return {
        ("aco-2opt", "aco-2opt"),
        ("aco-3opt", "aco-3opt"),
        *{
            (candidate.method, candidate.gp_run_id)
            for candidate in candidates
        },
    }


def _valid_shard(
    path: Path,
    *,
    candidates: list[Candidate],
    batch: Any,
    partition: str,
    seed: int,
    variant: ACOVariant,
) -> bool:
    if not path.is_file():
        return False
    try:
        records = read_records([path])
    except (OSError, TypeError, ValueError):
        return False
    expected_members = _expected_members(candidates)
    expected_rows = batch.batch_size * len(expected_members)
    observed_keys = {
        (
            record.method,
            record.gp_run_id,
            record.instance_id,
            record.seed,
        )
        for record in records
    }
    return (
        len(records) == expected_rows
        and len(observed_keys) == expected_rows
        and {
            (record.method, record.gp_run_id) for record in records
        }
        == expected_members
        and {record.instance_id for record in records}
        == set(batch.instance_ids)
        and {record.partition for record in records} == {partition}
        and {record.variant for record in records} == {variant.value}
        and {record.seed for record in records} == {seed}
    )


def _evaluate_partition(
    *,
    variant: ACOVariant,
    partition: str,
    candidates: list[Candidate],
) -> tuple[Path, list[dict[str, object]]]:
    spec, two_opt = _experiment(variant, LocalSearch.TWO_OPT)
    _, three_opt = _experiment(variant, LocalSearch.THREE_OPT)
    configure_runtime(two_opt.runtime)
    partition_spec = spec.data.test[partition]
    paths = spec.data.test_paths(partition)
    output = RUN_ROOT / "final-test" / variant.value / partition
    all_records: list[EvaluationRecord] = []
    campaign_metrics: list[dict[str, object]] = []
    instance_count = 0
    started = perf_counter()

    batches = iter_problem_batches(
        paths,
        batch_size=BATCH_SIZES[partition],
        candidate_size=two_opt.aco.candidate_size,
        dtype=two_opt.aco.dtype,
        device=two_opt.aco.device,
        min_scale=partition_spec.min_scale,
        max_scale=partition_spec.max_scale,
        max_instances=MAX_INSTANCES[partition],
    )
    for batch_number, batch in enumerate(batches):
        instance_count += batch.batch_size
        for replicate in range(TEST_SEEDS):
            seed = study_test_seed(
                TEST_ROOT_SEED,
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
                variant=variant,
            ) and metrics_path.is_file():
                records = read_records([shard])
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                programs = [
                    (None, None),
                    *[
                        (candidate.transition, candidate.pheromone)
                        for candidate in candidates
                    ],
                ]
                packed = solve_population_cuda_anytime(
                    batch,
                    two_opt.aco,
                    programs,
                    seed=seed,
                    runtime=two_opt.runtime,
                )
                baseline_2opt = _member_result(
                    packed,
                    0,
                    batch_size=batch.batch_size,
                    ants=two_opt.aco.resolve_ants(batch.n),
                    iterations=two_opt.aco.iterations,
                )
                separate_3opt = solve_population_cuda_anytime(
                    batch,
                    three_opt.aco,
                    [(None, None)],
                    seed=seed,
                    runtime=three_opt.runtime,
                )
                baseline_3opt = _member_result(
                    separate_3opt,
                    0,
                    batch_size=batch.batch_size,
                    ants=three_opt.aco.resolve_ants(batch.n),
                    iterations=three_opt.aco.iterations,
                    wall_time_sec=separate_3opt.wall_time_sec,
                )
                records = records_from_paired_results(
                    method="aco-2opt",
                    champion_id="aco-2opt",
                    partition=partition,
                    distribution=partition_spec.distribution,
                    batch=batch,
                    seed=seed,
                    candidate=baseline_2opt,
                    baseline=baseline_2opt,
                    config=two_opt.aco,
                    gp_run_id="aco-2opt",
                )
                records.extend(
                    records_from_paired_results(
                        method="aco-3opt",
                        champion_id="aco-3opt",
                        partition=partition,
                        distribution=partition_spec.distribution,
                        batch=batch,
                        seed=seed,
                        candidate=baseline_3opt,
                        baseline=baseline_2opt,
                        config=two_opt.aco,
                        gp_run_id="aco-3opt",
                    )
                )
                for index, candidate_info in enumerate(candidates, start=1):
                    candidate_result = _member_result(
                        packed,
                        index,
                        batch_size=batch.batch_size,
                        ants=two_opt.aco.resolve_ants(batch.n),
                        iterations=two_opt.aco.iterations,
                    )
                    records.extend(
                        records_from_paired_results(
                            method=candidate_info.method,
                            champion_id=candidate_info.champion_id,
                            partition=partition,
                            distribution=partition_spec.distribution,
                            batch=batch,
                            seed=seed,
                            candidate=candidate_result,
                            baseline=baseline_2opt,
                            config=two_opt.aco,
                            gp_run_id=candidate_info.gp_run_id,
                            gp_root_seed=candidate_info.gp_seed,
                        )
                    )
                write_records(records, shard)
                metrics = {
                    "schema_version": 1,
                    "variant": variant.value,
                    "partition": partition,
                    "batch_number": batch_number,
                    "replicate": replicate,
                    "seed": seed,
                    "instances": batch.batch_size,
                    "aco_iterations": TEST_ITERATIONS,
                    "ants": 32,
                    "two_opt_programs": len(programs),
                    "two_opt_packed_wall_time_sec": packed.wall_time_sec,
                    "two_opt_backend_metrics": packed.backend_metrics,
                    "three_opt_wall_time_sec": separate_3opt.wall_time_sec,
                    "three_opt_backend_metrics": separate_3opt.backend_metrics,
                }
                _atomic_json(metrics_path, metrics)
            all_records.extend(records)
            campaign_metrics.append(metrics)
            print(
                f"final-test variant={variant.value} partition={partition} "
                f"batch={batch_number} aco_seed={replicate + 1}/{TEST_SEEDS} "
                f"elapsed={(perf_counter() - started) / 60.0:.2f} min",
                flush=True,
            )

    expected = (
        instance_count
        * TEST_SEEDS
        * len(_expected_members(candidates))
    )
    keys = {
        (
            record.method,
            record.gp_run_id,
            record.instance_id,
            record.seed,
        )
        for record in all_records
    }
    if len(all_records) != expected or len(keys) != expected:
        raise RuntimeError(
            f"{variant.value}/{partition}: rows={len(all_records)}, "
            f"keys={len(keys)}, expected={expected}"
        )
    all_records.sort(
        key=lambda item: (
            item.method,
            item.gp_run_id,
            item.instance_id,
            item.seed,
        )
    )
    merged = write_records(all_records, output / "records.csv")
    _atomic_json(
        output / "evaluation_manifest.json",
        {
            "schema_version": 1,
            "status": "completed",
            "variant": variant.value,
            "partition": partition,
            "instances": instance_count,
            "aco_seeds": TEST_SEEDS,
            "aco_iterations": TEST_ITERATIONS,
            "ants": 32,
            "selected_candidate_tested": True,
            "population_individuals_tested": False,
            "quality_execution": (
                "ACO+2-opt and six final candidates packed; "
                "ACO+3-opt paired separately"
            ),
            "rows": expected,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return merged, campaign_metrics


def _online_seconds(result: Any) -> float:
    compile_seconds = float(result.backend_metrics.get("compile_seconds_sum", 0.0))
    return max(0.0, float(result.wall_time_sec) - compile_seconds)


def _runtime_benchmark(
    *,
    variant: ACOVariant,
    partition: str,
    candidates: list[Candidate],
    repeats: int,
) -> list[dict[str, object]]:
    spec, two_opt = _experiment(variant, LocalSearch.TWO_OPT)
    _, three_opt = _experiment(variant, LocalSearch.THREE_OPT)
    partition_spec = spec.data.test[partition]
    batches = list(
        iter_problem_batches(
            spec.data.test_paths(partition),
            batch_size=RUNTIME_INSTANCES[partition],
            candidate_size=two_opt.aco.candidate_size,
            dtype=two_opt.aco.dtype,
            device=two_opt.aco.device,
            min_scale=partition_spec.min_scale,
            max_scale=partition_spec.max_scale,
            max_instances=RUNTIME_INSTANCES[partition],
        )
    )
    if len(batches) != 1:
        raise RuntimeError(f"{partition} runtime batch 不唯一")
    batch = batches[0]
    entries: list[tuple[str, str, int, str, Any, Any, Any]] = [
        ("aco-2opt", "aco-2opt", 0, "two_opt", None, None, two_opt),
        ("aco-3opt", "aco-3opt", 0, "three_opt", None, None, three_opt),
    ]
    entries.extend(
        (
            candidate.method,
            candidate.gp_run_id,
            candidate.gp_seed,
            "two_opt",
            candidate.transition,
            candidate.pheromone,
            two_opt,
        )
        for candidate in candidates
    )
    rows: list[dict[str, object]] = []
    for repeat_index in range(repeats):
        seed = study_test_seed(
            TEST_ROOT_SEED + 1,
            f"runtime:{partition}",
            0,
            repeat_index,
        )
        for method, run_id, gp_seed, search, transition, pheromone, experiment in entries:
            result = solve_population_cuda(
                batch,
                experiment.aco,
                [(transition, pheromone)],
                seed=seed,
                runtime=experiment.runtime,
            )
            diagnostic = result.diagnostics[0]
            online = _online_seconds(result)
            rows.append(
                {
                    "variant": variant.value,
                    "partition": partition,
                    "scale": batch.n,
                    "method": method,
                    "gp_run_id": run_id,
                    "gp_seed": gp_seed,
                    "local_search": search,
                    "repeat": repeat_index,
                    "seed": seed,
                    "instances": batch.batch_size,
                    "ants": 32,
                    "aco_iterations": TEST_ITERATIONS,
                    "constructed_tours": (
                        batch.batch_size * 32 * TEST_ITERATIONS
                    ),
                    "wall_time_sec": result.wall_time_sec,
                    "compile_time_sec": float(
                        result.backend_metrics.get(
                            "compile_seconds_sum",
                            0.0,
                        )
                    ),
                    "online_batch_time_sec": online,
                    "online_time_per_instance_sec": (
                        online / batch.batch_size
                    ),
                    "tours_per_second": (
                        batch.batch_size
                        * 32
                        * TEST_ITERATIONS
                        / max(online, 1e-12)
                    ),
                    "kernel_time_sec": float(
                        result.backend_metrics.get(
                            "kernel_seconds_critical",
                            float("nan"),
                        )
                    ),
                    "local_search_move_count": int(
                        diagnostic[4].item()
                    ),
                    "local_search_candidate_check_count": int(
                        diagnostic[5].item()
                    ),
                    "local_search_improved_tour_count": int(
                        diagnostic[6].item()
                    ),
                    "local_search_pass_count": int(
                        diagnostic[7].item()
                    ),
                }
            )
            print(
                f"runtime variant={variant.value} partition={partition} "
                f"method={method}/{run_id} repeat={repeat_index + 1}/{repeats} "
                f"online={online:.3f}s",
                flush=True,
            )
    return rows


def _tail_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, int(math.ceil(values.size * fraction)))
    return float(np.sort(values)[-count:].mean())


def _quality_summary(records: list[EvaluationRecord]) -> list[dict[str, object]]:
    groups: dict[
        tuple[str, str, str],
        dict[tuple[str, str], list[EvaluationRecord]],
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups[(record.variant, record.partition, record.method)][
            (record.gp_run_id, record.instance_id)
        ].append(record)

    rows: list[dict[str, object]] = []
    for (variant, partition, method), blocks in sorted(groups.items()):
        first_record = next(iter(next(iter(blocks.values()))))
        gap = np.asarray(
            [fmean(item.gap_percent for item in block) for block in blocks.values()],
            dtype=np.float64,
        )
        delta = np.asarray(
            [fmean(item.delta_pp for item in block) for block in blocks.values()],
            dtype=np.float64,
        )
        auc = np.asarray(
            [
                fmean(item.anytime_gap_auc for item in block)
                for block in blocks.values()
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "variant": variant,
                "partition": partition,
                "scale": first_record.scale,
                "method": method,
                "gp_runs": len({key[0] for key in blocks}),
                "instances": len({key[1] for key in blocks}),
                "run_instance_blocks": len(blocks),
                "mean_gap_percent": float(gap.mean()),
                "median_gap_percent": float(np.median(gap)),
                "standard_deviation": float(
                    gap.std(ddof=1) if gap.size > 1 else 0.0
                ),
                "q1_gap_percent": float(np.quantile(gap, 0.25)),
                "q3_gap_percent": float(np.quantile(gap, 0.75)),
                "cvar_worst_10_percent": _tail_mean(gap),
                "mean_delta_vs_aco2_pp": float(delta.mean()),
                "median_delta_vs_aco2_pp": float(np.median(delta)),
                "win_rate_vs_aco2": float(np.mean(delta < -1e-12)),
                "tie_rate_vs_aco2": float(np.mean(np.abs(delta) <= 1e-12)),
                "loss_rate_vs_aco2": float(np.mean(delta > 1e-12)),
                "mean_anytime_gap_auc": float(auc.mean()),
            }
        )
    return rows


def _hierarchical_bootstrap(
    differences: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """对 GP run→instance→ACO seed 三层进行有放回重采样。"""

    if differences.ndim != 3:
        raise ValueError("differences 必须为 [GP run, instance, ACO seed]")
    runs, instances, seeds = differences.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_runs = rng.integers(0, runs, size=runs)
        total = 0.0
        count = 0
        for selected_run in selected_runs:
            selected_instances = rng.integers(
                0,
                instances,
                size=instances,
            )
            values = differences[selected_run, selected_instances, :]
            selected_seeds = rng.integers(
                0,
                seeds,
                size=(instances, seeds),
            )
            total += float(
                np.take_along_axis(values, selected_seeds, axis=1).sum()
            )
            count += instances * seeds
        estimates[replicate] = total / count
    return (
        float(differences.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        float(np.mean(estimates < 0.0)),
    )


def _paired_vs_aco3(
    records: list[EvaluationRecord],
    *,
    bootstrap_replicates: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in (item.value for item in VARIANTS):
        for partition in PARTITIONS:
            context = [
                item
                for item in records
                if item.variant == variant and item.partition == partition
            ]
            aco3 = {
                (item.instance_id, item.seed): item.gap_percent
                for item in context
                if item.method == "aco-3opt"
            }
            for method in ("rmtgp-nols-2opt", "rmtgp-ls-2opt"):
                selected = [item for item in context if item.method == method]
                run_ids = sorted({item.gp_run_id for item in selected})
                instances = sorted({item.instance_id for item in selected})
                seeds_by_instance = {
                    instance: sorted(
                        {
                            item.seed
                            for item in selected
                            if item.instance_id == instance
                        }
                    )
                    for instance in instances
                }
                seed_counts = {len(values) for values in seeds_by_instance.values()}
                if seed_counts != {TEST_SEEDS}:
                    raise RuntimeError(
                        f"{variant}/{partition}/{method}: ACO seed 层不完整"
                    )
                candidate = {
                    (item.gp_run_id, item.instance_id, item.seed): item.gap_percent
                    for item in selected
                }
                differences = np.empty(
                    (len(run_ids), len(instances), TEST_SEEDS),
                    dtype=np.float64,
                )
                for run_index, run_id in enumerate(run_ids):
                    for instance_index, instance in enumerate(instances):
                        for seed_index, seed in enumerate(
                            seeds_by_instance[instance]
                        ):
                            differences[run_index, instance_index, seed_index] = (
                                candidate[(run_id, instance, seed)]
                                - aco3[(instance, seed)]
                            )
                estimate, lower, upper, probability_better = (
                    _hierarchical_bootstrap(
                        differences,
                        replicates=bootstrap_replicates,
                        seed=(
                            TEST_ROOT_SEED
                            + 1000 * list(item.value for item in VARIANTS).index(
                                variant
                            )
                            + 100 * PARTITIONS.index(partition)
                            + (0 if method == "rmtgp-nols-2opt" else 1)
                        ),
                    )
                )
                blocks = differences.mean(axis=2).reshape(-1)
                if upper < 0.0:
                    conclusion = "superior"
                elif upper <= NONINFERIORITY_TOLERANCE_PP:
                    conclusion = "noninferior"
                else:
                    conclusion = "inconclusive"
                rows.append(
                    {
                        "variant": variant,
                        "partition": partition,
                        "method": method,
                        "reference": "aco-3opt",
                        "gp_runs": len(run_ids),
                        "instances": len(instances),
                        "aco_seeds": TEST_SEEDS,
                        "mean_delta_pp": estimate,
                        "bootstrap_lower_95_pp": lower,
                        "bootstrap_upper_95_pp": upper,
                        "bootstrap_probability_better": probability_better,
                        "win_rate": float(np.mean(blocks < -1e-12)),
                        "tie_rate": float(np.mean(np.abs(blocks) <= 1e-12)),
                        "loss_rate": float(np.mean(blocks > 1e-12)),
                        "noninferiority_margin_pp": (
                            NONINFERIORITY_TOLERANCE_PP
                        ),
                        "quality_conclusion": conclusion,
                        "bootstrap_replicates": bootstrap_replicates,
                    }
                )
    return rows


def _runtime_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["variant"]),
                str(row["partition"]),
                str(row["method"]),
            )
        ].append(row)
    baselines = {
        (variant, partition): fmean(
            float(row["online_batch_time_sec"])
            for row in values
        )
        for (variant, partition, method), values in groups.items()
        if method == "aco-3opt"
    }
    summary: list[dict[str, object]] = []
    for (variant, partition, method), values in sorted(groups.items()):
        online = np.asarray(
            [float(row["online_batch_time_sec"]) for row in values],
            dtype=np.float64,
        )
        kernel = np.asarray(
            [float(row["kernel_time_sec"]) for row in values],
            dtype=np.float64,
        )
        reference = baselines[(variant, partition)]
        summary.append(
            {
                "variant": variant,
                "partition": partition,
                "scale": values[0]["scale"],
                "method": method,
                "measurements": len(values),
                "gp_runs": len({str(row["gp_run_id"]) for row in values}),
                "instances_per_batch": values[0]["instances"],
                "mean_online_batch_time_sec": float(online.mean()),
                "median_online_batch_time_sec": float(np.median(online)),
                "standard_deviation_sec": float(
                    online.std(ddof=1) if online.size > 1 else 0.0
                ),
                "mean_kernel_time_sec": float(kernel.mean()),
                "speedup_vs_aco3": reference / float(online.mean()),
                "time_reduction_vs_aco3_percent": (
                    100.0 * (reference - float(online.mean())) / reference
                ),
            }
        )
    return summary


def _draw_figures(
    quality: list[dict[str, object]],
    paired: list[dict[str, object]],
    runtime: list[dict[str, object]],
    output: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output.mkdir(parents=True, exist_ok=True)
    method_order = (
        "aco-2opt",
        "aco-3opt",
        "rmtgp-nols-2opt",
        "rmtgp-ls-2opt",
    )
    labels = {
        "aco-2opt": "ACO+2-opt",
        "aco-3opt": "ACO+3-opt",
        "rmtgp-nols-2opt": "RMTGP(no-LS)+2-opt",
        "rmtgp-ls-2opt": "RMTGP(2-opt)+2-opt",
    }
    colors = ("#7f8c8d", "#2c3e50", "#3498db", "#e67e22")
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for row_index, variant in enumerate(item.value for item in VARIANTS):
        for column_index, partition in enumerate(PARTITIONS):
            axis = axes[row_index, column_index]
            context = {
                str(item["method"]): item
                for item in quality
                if item["variant"] == variant
                and item["partition"] == partition
            }
            means = [
                float(context[item]["median_gap_percent"])
                for item in method_order
            ]
            lower = [
                means[index] - float(context[item]["q1_gap_percent"])
                for index, item in enumerate(method_order)
            ]
            upper = [
                float(context[item]["q3_gap_percent"]) - means[index]
                for index, item in enumerate(method_order)
            ]
            axis.errorbar(
                range(len(method_order)),
                means,
                yerr=np.asarray([lower, upper]),
                fmt="o",
                capsize=4,
                color="#202020",
            )
            for index, color in enumerate(colors):
                axis.scatter(index, means[index], color=color, s=55, zorder=3)
            axis.set_xticks(
                range(len(method_order)),
                [labels[item] for item in method_order],
                rotation=22,
                ha="right",
            )
            axis.set_title(f"{variant.upper()} / {partition}")
            axis.set_ylabel("Median optimality gap with IQR (%)")
            axis.grid(axis="y", alpha=0.25)
    quality_path = output / "final_quality_gap.png"
    figure.savefig(quality_path, dpi=220)
    figure.savefig(output / "final_quality_gap.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for row_index, variant in enumerate(item.value for item in VARIANTS):
        for column_index, partition in enumerate(PARTITIONS):
            axis = axes[row_index, column_index]
            context = [
                item
                for item in paired
                if item["variant"] == variant
                and item["partition"] == partition
            ]
            for index, item in enumerate(context):
                mean_value = float(item["mean_delta_pp"])
                axis.errorbar(
                    mean_value,
                    index,
                    xerr=[
                        [mean_value - float(item["bootstrap_lower_95_pp"])],
                        [float(item["bootstrap_upper_95_pp"]) - mean_value],
                    ],
                    fmt="o",
                    capsize=4,
                    color=colors[index + 2],
                )
            axis.axvline(0.0, color="black", linewidth=1)
            axis.axvline(
                NONINFERIORITY_TOLERANCE_PP,
                color="#c0392b",
                linestyle="--",
                linewidth=1,
            )
            axis.set_yticks(
                range(len(context)),
                [labels[str(item["method"])] for item in context],
            )
            axis.set_title(f"{variant.upper()} / {partition}")
            axis.set_xlabel("Gap difference vs ACO+3-opt (pp)")
            axis.grid(axis="x", alpha=0.25)
    paired_path = output / "paired_vs_aco3.png"
    figure.savefig(paired_path, dpi=220)
    figure.savefig(output / "paired_vs_aco3.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for row_index, variant in enumerate(item.value for item in VARIANTS):
        for column_index, partition in enumerate(PARTITIONS):
            axis = axes[row_index, column_index]
            quality_map = {
                str(item["method"]): float(item["mean_gap_percent"])
                for item in quality
                if item["variant"] == variant
                and item["partition"] == partition
            }
            runtime_map = {
                str(item["method"]): float(item["mean_online_batch_time_sec"])
                for item in runtime
                if item["variant"] == variant
                and item["partition"] == partition
            }
            for index, method in enumerate(method_order):
                axis.scatter(
                    runtime_map[method],
                    quality_map[method],
                    color=colors[index],
                    s=65,
                    label=labels[method],
                )
            axis.set_title(f"{variant.upper()} / {partition}")
            axis.set_xlabel("Online batch time (s)")
            axis.set_ylabel("Mean optimality gap (%)")
            axis.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
    )
    tradeoff_path = output / "quality_runtime_tradeoff.png"
    figure.savefig(tradeoff_path, dpi=220)
    figure.savefig(output / "quality_runtime_tradeoff.pdf")
    plt.close(figure)
    return [
        str(quality_path),
        str(paired_path),
        str(tradeoff_path),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--test-seeds", type=int, default=TEST_SEEDS)
    parser.add_argument("--iterations", type=int, default=TEST_ITERATIONS)
    parser.add_argument("--runtime-repeats", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--runtime-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.test_seeds != TEST_SEEDS:
        raise ValueError(f"冻结协议要求 test_seeds={TEST_SEEDS}")
    if args.iterations != TEST_ITERATIONS:
        raise ValueError(f"冻结协议要求 iterations={TEST_ITERATIONS}")
    if args.runtime_repeats < 1:
        raise ValueError("runtime_repeats 必须为正整数")
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates 至少为 100")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != str(args.physical_gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES 与 --physical-gpu 不一致："
            f"{visible!r} != {args.physical_gpu}"
        )
    output = RUN_ROOT / "final-test"
    output.mkdir(parents=True, exist_ok=True)
    record_paths: list[Path] = []
    runtime_rows: list[dict[str, object]] = []
    campaign_metrics: list[dict[str, object]] = []
    for variant in VARIANTS:
        candidates = _load_candidates(variant)
        for partition in PARTITIONS:
            path, metrics = _evaluate_partition(
                variant=variant,
                partition=partition,
                candidates=candidates,
            )
            record_paths.append(path)
            campaign_metrics.extend(metrics)
            if args.runtime_benchmark:
                runtime_rows.extend(
                    _runtime_benchmark(
                        variant=variant,
                        partition=partition,
                        candidates=candidates,
                        repeats=args.runtime_repeats,
                    )
                )

    records = read_records(record_paths)
    quality = _quality_summary(records)
    paired = _paired_vs_aco3(
        records,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    _write_dict_csv(output / "quality_summary.csv", quality)
    _write_dict_csv(output / "paired_vs_aco3.csv", paired)
    runtime_summary: list[dict[str, object]] = []
    if runtime_rows:
        _write_dict_csv(output / "runtime_measurements.csv", runtime_rows)
        runtime_summary = _runtime_summary(runtime_rows)
        _write_dict_csv(output / "runtime_summary.csv", runtime_summary)
    figures = (
        _draw_figures(
            quality,
            paired,
            runtime_summary,
            output / "figures",
        )
        if runtime_summary
        else []
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "physical_gpu": args.physical_gpu,
        "visible_cuda_devices": visible,
        "variants": [item.value for item in VARIANTS],
        "partitions": list(PARTITIONS),
        "max_instances": MAX_INSTANCES,
        "test_root_seed": TEST_ROOT_SEED,
        "aco_seeds": TEST_SEEDS,
        "ants": 32,
        "aco_iterations": TEST_ITERATIONS,
        "candidate_list_size": 20,
        "gp_final_candidates_per_condition": len(GP_SEEDS),
        "population_individuals_tested": False,
        "quality_summary": quality,
        "paired_vs_aco3": paired,
        "runtime_summary": runtime_summary,
        "figures": figures,
        "packed_quality_campaigns": len(campaign_metrics),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
