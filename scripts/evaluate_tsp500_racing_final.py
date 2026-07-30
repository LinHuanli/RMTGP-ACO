#!/usr/bin/env python3
"""最终跨规模测试：ACO+2opt/3opt、TSP100-trained 与 TSP500-trained。

TSP100 使用 128 个 uniform test instances，TSP500 使用前 32 个。每个
ACO variant 使用 3 个独立 ACO seeds 和 5000 轮。六个锁定 GP programs
在同一个 CUDA population 调用中并行测试；两个原始 ACO baseline 单独
运行，以便给出可解释的 2-opt/3-opt wall time。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rmtgp_aco.aco_cuda import (
    solve_population_cuda,
    solve_population_cuda_anytime,
)
from rmtgp_aco.config import ACOVariant, ExecutionBackend, GPUMode, LocalSearch
from rmtgp_aco.evaluation import compile_champion, load_champion
from rmtgp_aco.runtime import configure_runtime
from rmtgp_aco.sampling import iter_problem_batches
from rmtgp_aco.spec import load_run_spec

ROOT = Path(__file__).resolve().parents[1]
TSP100_ROOT = ROOT / "runs" / "tsp100-2opt-anytime" / "formal"
TSP500_ROOT = ROOT / "runs" / "tsp500-2opt-racing" / "formal"
OUTPUT = ROOT / "runs" / "tsp500-2opt-racing" / "final-test"
GP_SEEDS_100 = (71001, 71002, 71003)
GP_SEEDS_500 = (81001, 81002, 81003)
VARIANTS = tuple(ACOVariant)
SCHEMA_VERSION = 1
TEST_ROOT_SEED = 94001


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能写空 CSV")
    fields = list(dict.fromkeys(name for row in rows for name in row))
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _completed_run(path: Path) -> None:
    payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"formal run 尚未完成: {path}")


def _load_variant(variant: ACOVariant, iterations: int):
    config = (
        TSP500_ROOT
        / "configs"
        / f"tsp500-2opt-racing-{variant.value}-seed-81001.yaml"
    )
    spec = load_run_spec(config)
    experiment = replace(
        spec.experiment,
        aco=replace(spec.experiment.aco, iterations=iterations),
        runtime=replace(
            spec.experiment.runtime,
            aco_backend=ExecutionBackend.CUDA_TILED_V2,
            gpu_mode=GPUMode.SINGLE,
            gpu_devices=(0,),
        ),
    )
    programs = []
    labels = []
    hashes = []
    for scale, root, seeds in (
        (100, TSP100_ROOT, GP_SEEDS_100),
        (500, TSP500_ROOT, GP_SEEDS_500),
    ):
        for seed in seeds:
            run = root / "train" / variant.value / f"seed-{seed}"
            _completed_run(run)
            candidate = load_champion(run / "selected_candidate.pkl")
            programs.append(compile_champion(candidate))
            labels.append(f"tsp{scale}-trained-seed-{seed}")
            hashes.append(candidate.structural_hash)
    return spec, experiment, programs, labels, hashes


def _test_seed(variant: ACOVariant, replicate: int) -> int:
    code = list(ACOVariant).index(variant) + 1
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [TEST_ROOT_SEED, code, replicate, 0x54455354]
        )
    )
    return int(rng.integers(0, 2**63 - 1))


def _shard_path(
    output: Path,
    scale: int,
    variant: ACOVariant,
    replicate: int,
) -> Path:
    return (
        output
        / "shards"
        / f"tsp{scale}-{variant.value}-seed-{replicate:02d}.npz"
    )


def _gaps(result, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    final = 100.0 * (
        result.best_length.numpy() - reference[None, :]
    ) / reference[None, :]
    anytime = 100.0 * (
        result.anytime_best.numpy() - reference[None, :, None]
    ) / reference[None, :, None]
    return final, anytime.mean(axis=-1)


def _online_wall_time(result) -> float:
    """排除一次性 NVRTC 编译；保留传输、kernel 与 CPU 精确计分。"""

    return max(
        0.0,
        float(result.wall_time_sec)
        - float(result.backend_metrics.get("compile_seconds_sum", 0.0)),
    )


def _run_shard(
    path: Path,
    *,
    batch,
    experiment,
    programs,
    labels: list[str],
    hashes: list[str],
    variant: ACOVariant,
    replicate: int,
    latency_audit: bool,
) -> None:
    seed = _test_seed(variant, replicate)
    baseline_2opt = solve_population_cuda_anytime(
        batch,
        experiment.aco,
        [(None, None)],
        seed=seed,
        runtime=experiment.runtime,
    )
    candidates = solve_population_cuda_anytime(
        batch,
        experiment.aco,
        programs,
        seed=seed,
        runtime=experiment.runtime,
    )
    baseline_3opt = solve_population_cuda_anytime(
        batch,
        replace(experiment.aco, local_search=LocalSearch.THREE_OPT),
        [(None, None)],
        seed=seed,
        runtime=experiment.runtime,
    )
    candidate_latency = np.full(len(programs), np.nan, dtype=np.float64)
    if latency_audit and replicate == 0:
        for index, program in enumerate(programs):
            latency_result = solve_population_cuda(
                batch,
                experiment.aco,
                [program],
                seed=seed,
                runtime=experiment.runtime,
            )
            candidate_latency[index] = _online_wall_time(latency_result)
    reference = batch.reference_length.numpy()
    final_2opt, auc_2opt = _gaps(baseline_2opt, reference)
    final_candidates, auc_candidates = _gaps(candidates, reference)
    final_3opt, auc_3opt = _gaps(baseline_3opt, reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        scale=np.asarray(batch.n, dtype=np.int32),
        variant=np.asarray(variant.value),
        replicate=np.asarray(replicate, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int64),
        candidate_labels=np.asarray(labels),
        candidate_hashes=np.asarray(hashes),
        instance_ids=np.asarray(batch.instance_ids),
        coordinate_hashes=np.asarray(batch.coordinate_hashes),
        reference_length=reference,
        baseline_2opt_final_gap=final_2opt[0],
        baseline_2opt_anytime_gap_auc=auc_2opt[0],
        baseline_3opt_final_gap=final_3opt[0],
        baseline_3opt_anytime_gap_auc=auc_3opt[0],
        candidate_final_gap=final_candidates,
        candidate_anytime_gap_auc=auc_candidates,
        baseline_2opt_wall_time_sec=np.asarray(
            _online_wall_time(baseline_2opt)
        ),
        baseline_3opt_wall_time_sec=np.asarray(
            _online_wall_time(baseline_3opt)
        ),
        candidate_batch_wall_time_sec=np.asarray(
            _online_wall_time(candidates)
        ),
        candidate_individual_wall_time_sec=candidate_latency,
    )
    os.replace(temporary, path)


def _valid_shard(
    path: Path,
    *,
    scale: int,
    variant: ACOVariant,
    replicate: int,
    hashes: list[str],
    instance_hashes: list[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                int(payload["schema_version"]) == SCHEMA_VERSION
                and int(payload["scale"]) == scale
                and str(payload["variant"]) == variant.value
                and int(payload["replicate"]) == replicate
                and payload["candidate_hashes"].tolist() == hashes
                and payload["coordinate_hashes"].tolist() == instance_hashes
            )
    except (OSError, KeyError, ValueError):
        return False


def _hierarchical_bootstrap(
    cube: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """对 GP run、instance、ACO seed 三层有放回采样。"""

    runs, instances, aco_seeds = cube.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        run_indices = rng.integers(0, runs, size=runs)
        sampled = []
        for run in run_indices:
            instance_indices = rng.integers(
                0,
                instances,
                size=instances,
            )
            seed_indices = rng.integers(
                0,
                aco_seeds,
                size=(instances, aco_seeds),
            )
            sampled.append(
                cube[run, instance_indices[:, None], seed_indices]
            )
        estimates[draw] = np.mean(sampled)
    return estimates


def _method_summary(
    *,
    scale: int,
    variant: ACOVariant,
    training_scale: int,
    candidate: np.ndarray,
    baseline_2opt: np.ndarray,
    baseline_3opt: np.ndarray,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    # candidate [S,R,I] -> [R,I,S]
    candidate_cube = np.transpose(candidate, (1, 2, 0))
    baseline_2_cube = np.transpose(baseline_2opt, (1, 0))[None, ...]
    baseline_3_cube = np.transpose(baseline_3opt, (1, 0))[None, ...]
    delta_2 = candidate_cube - baseline_2_cube
    delta_3 = candidate_cube - baseline_3_cube
    bootstrap_2 = _hierarchical_bootstrap(
        delta_2,
        replicates=bootstrap_replicates,
        seed=scale * 100 + list(VARIANTS).index(variant),
    )
    bootstrap_3 = _hierarchical_bootstrap(
        delta_3,
        replicates=bootstrap_replicates,
        seed=scale * 1000 + list(VARIANTS).index(variant),
    )
    baseline_mean = float(baseline_2opt.mean())
    candidate_mean = float(candidate_cube.mean())
    relative = (
        (baseline_mean - candidate_mean) / baseline_mean
        if baseline_mean > 1e-12
        else float("nan")
    )
    run_deltas = delta_2.mean(axis=(1, 2))
    upper_2 = float(np.quantile(bootstrap_2, 0.95))
    upper_3 = float(np.quantile(bootstrap_3, 0.95))
    return {
        "scale": scale,
        "variant": variant.value,
        "method": f"tsp{training_scale}-trained-rmtgp-aco-2opt",
        "gp_runs": int(candidate_cube.shape[0]),
        "instances": int(candidate_cube.shape[1]),
        "aco_seeds": int(candidate_cube.shape[2]),
        "mean_gap_percent": candidate_mean,
        "baseline_2opt_mean_gap_percent": baseline_mean,
        "baseline_3opt_mean_gap_percent": float(baseline_3opt.mean()),
        "mean_delta_vs_2opt_pp": float(delta_2.mean()),
        "mean_delta_vs_3opt_pp": float(delta_3.mean()),
        "relative_reduction_vs_2opt": relative,
        "one_sided_upper_95_vs_2opt": upper_2,
        "one_sided_upper_95_vs_3opt": upper_3,
        "improved_gp_runs_vs_2opt": int(np.sum(run_deltas < 0.0)),
        "success_vs_2opt": bool(
            np.isfinite(relative)
            and relative >= 0.10
            and upper_2 < 0.0
            and np.sum(run_deltas < 0.0) >= 2
        ),
        "better_than_3opt": upper_3 < 0.0,
        "noninferior_to_3opt_margin_0p1pp": upper_3 <= 0.10,
    }


def _summarize(
    output: Path,
    *,
    scales: tuple[int, ...],
    test_seeds: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for scale in scales:
        for variant in VARIANTS:
            shards = []
            for replicate in range(test_seeds):
                with np.load(
                    _shard_path(output, scale, variant, replicate),
                    allow_pickle=False,
                ) as payload:
                    shards.append(
                        {
                            name: np.asarray(payload[name])
                            for name in payload.files
                        }
                    )
            baseline_2 = np.stack(
                [item["baseline_2opt_final_gap"] for item in shards]
            )
            baseline_3 = np.stack(
                [item["baseline_3opt_final_gap"] for item in shards]
            )
            candidates = np.stack(
                [item["candidate_final_gap"] for item in shards]
            )
            for training_scale, indices in ((100, slice(0, 3)), (500, slice(3, 6))):
                rows.append(
                    _method_summary(
                        scale=scale,
                        variant=variant,
                        training_scale=training_scale,
                        candidate=candidates[:, indices, :],
                        baseline_2opt=baseline_2,
                        baseline_3opt=baseline_3,
                        bootstrap_replicates=bootstrap_replicates,
                    )
                )
            runtime_rows.append(
                {
                    "scale": scale,
                    "variant": variant.value,
                    "aco_2opt_wall_time_sec_mean": float(
                        np.mean(
                            [
                                item["baseline_2opt_wall_time_sec"]
                                for item in shards
                            ]
                        )
                    ),
                    "aco_3opt_wall_time_sec_mean": float(
                        np.mean(
                            [
                                item["baseline_3opt_wall_time_sec"]
                                for item in shards
                            ]
                        )
                    ),
                    "aco_3opt_over_2opt_time_ratio": float(
                        np.mean(
                            [
                                item["baseline_3opt_wall_time_sec"]
                                for item in shards
                            ]
                        )
                        / np.mean(
                            [
                                item["baseline_2opt_wall_time_sec"]
                                for item in shards
                            ]
                        )
                    ),
                    "six_gp_program_batch_wall_time_sec_mean": float(
                        np.mean(
                            [
                                item["candidate_batch_wall_time_sec"]
                                for item in shards
                            ]
                        )
                    ),
                    "rmtgp_2opt_individual_wall_time_sec_mean": float(
                        np.nanmean(
                            np.stack(
                                [
                                    item[
                                        "candidate_individual_wall_time_sec"
                                    ]
                                    for item in shards
                                ]
                            )
                        )
                    ),
                    "aco_3opt_over_rmtgp_2opt_time_ratio": float(
                        np.mean(
                            [
                                item["baseline_3opt_wall_time_sec"]
                                for item in shards
                            ]
                        )
                        / np.nanmean(
                            np.stack(
                                [
                                    item[
                                        "candidate_individual_wall_time_sec"
                                    ]
                                    for item in shards
                                ]
                            )
                        )
                    ),
                }
            )
    _write_csv(output / "main_results.csv", rows)
    _write_csv(output / "runtime_results.csv", runtime_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "scales": list(scales),
            "instances": {"100": 128, "500": 32},
            "aco_iterations": 5000,
            "aco_seeds": test_seeds,
            "gp_runs_per_training_scale": 3,
            "bootstrap_replicates": bootstrap_replicates,
        },
        "results": rows,
        "runtime": runtime_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--latency-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在第一个 ACO seed 上逐个运行六个 GP programs 测在线时间",
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.test_seeds < 2:
        parser.error("iterations 必须为正；test seeds 至少为 2")
    if args.bootstrap_replicates < 100:
        parser.error("bootstrap replicates 至少为 100")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
    }
    _atomic_json(args.output / "manifest.json", manifest)
    try:
        for variant in VARIANTS:
            spec, experiment, programs, labels, hashes = _load_variant(
                variant,
                args.iterations,
            )
            configure_runtime(experiment.runtime)
            partitions = (
                (
                    100,
                    ROOT
                    / "Datasets"
                    / "TSP"
                    / "test_dataset"
                    / "tsp"
                    / "tsp100_concorde_7.756.txt",
                    128,
                ),
                (
                    500,
                    spec.data.test_paths("tsp500_uniform")[0],
                    32,
                ),
            )
            for scale, path, instances in partitions:
                batches = list(
                    iter_problem_batches(
                        (path,),
                        batch_size=instances,
                        candidate_size=experiment.aco.candidate_size,
                        dtype=experiment.aco.dtype,
                        device=experiment.aco.device,
                        max_instances=instances,
                    )
                )
                if len(batches) != 1:
                    raise RuntimeError("最终测试必须形成一个确定 batch")
                batch = batches[0]
                for replicate in range(args.test_seeds):
                    shard = _shard_path(
                        args.output,
                        scale,
                        variant,
                        replicate,
                    )
                    if _valid_shard(
                        shard,
                        scale=scale,
                        variant=variant,
                        replicate=replicate,
                        hashes=hashes,
                        instance_hashes=list(batch.coordinate_hashes),
                    ):
                        print(f"reuse {shard.name}", flush=True)
                        continue
                    print(
                        f"TSP{scale} {variant.value} "
                        f"seed={replicate + 1}/{args.test_seeds}",
                        flush=True,
                    )
                    _run_shard(
                        shard,
                        batch=batch,
                        experiment=experiment,
                        programs=programs,
                        labels=labels,
                        hashes=hashes,
                        variant=variant,
                        replicate=replicate,
                        latency_audit=args.latency_audit,
                    )
        payload = _summarize(
            args.output,
            scales=(100, 500),
            test_seeds=args.test_seeds,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "ended_at": datetime.now(UTC).isoformat(),
                "error": repr(error),
            }
        )
        _atomic_json(args.output / "manifest.json", manifest)
        raise
    _atomic_json(args.output / "summary.json", payload)
    manifest.update(
        {
            "status": "completed",
            "ended_at": datetime.now(UTC).isoformat(),
            "summary": "summary.json",
        }
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
