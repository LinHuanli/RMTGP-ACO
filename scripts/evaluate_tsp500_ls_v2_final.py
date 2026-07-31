#!/usr/bin/env python3
"""评测 TSP500 LS-aware v2 最终个体与 ACO+2-opt/3-opt baseline。

每个 ACO variant 只加载三个正式 GP run 的最终 selected candidate。评测
使用 uniform、cluster 和 gaussian 三个 TSP500 partition 的前 32 个
instance、3 个独立 ACO seeds 和 5000 次 iteration。所有方法在同一
partition/variant/seed 下共享 counter-based RNG seed。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts import evaluate_tsp500_racing_final as common
except ModuleNotFoundError:
    import evaluate_tsp500_racing_final as common

from rmtgp_aco.config import ACOVariant, ExecutionBackend, GPUMode
from rmtgp_aco.evaluation import compile_champion, load_champion
from rmtgp_aco.runtime import configure_runtime
from rmtgp_aco.sampling import iter_problem_batches
from rmtgp_aco.spec import load_run_spec

ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "runs" / "tsp500-2opt-ls-v2" / "formal"
OUTPUT = ROOT / "runs" / "tsp500-2opt-ls-v2" / "final-test"
GP_SEEDS = (81001, 81002, 81003)
PARTITIONS = ("tsp500_uniform", "tsp500_cluster", "tsp500_gaussian")


def _completed_run(path: Path) -> None:
    payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"formal run 尚未完成: {path}")


def _load_variant(
    variant: ACOVariant,
    *,
    iterations: int,
    gpu_device: int,
):
    config = (
        FORMAL_ROOT
        / "configs"
        / f"tsp500-2opt-ls-v2-{variant.value}-seed-81001.yaml"
    )
    spec = load_run_spec(config)
    experiment = replace(
        spec.experiment,
        aco=replace(spec.experiment.aco, iterations=iterations),
        runtime=replace(
            spec.experiment.runtime,
            aco_backend=ExecutionBackend.CUDA_TILED_V2,
            gpu_mode=GPUMode.SINGLE,
            gpu_devices=(gpu_device,),
        ),
    )
    programs = []
    labels = []
    hashes = []
    for seed in GP_SEEDS:
        run = FORMAL_ROOT / "train" / variant.value / f"seed-{seed}"
        _completed_run(run)
        candidate = load_champion(run / "selected_candidate.pkl")
        programs.append(compile_champion(candidate))
        labels.append(f"tsp500-ls-v2-seed-{seed}")
        hashes.append(candidate.structural_hash)
    return spec, experiment, programs, labels, hashes


def _shard_path(
    output: Path,
    partition: str,
    variant: ACOVariant,
    replicate: int,
) -> Path:
    return (
        output
        / "shards"
        / f"{partition}-{variant.value}-seed-{replicate:02d}.npz"
    )


def _bootstrap_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    """返回 mean delta、单侧 95% 上界和改善的 GP run 数。"""

    # candidate [ACO seed, GP run, instance] -> [GP run, instance, ACO seed]
    candidate_cube = np.transpose(candidate, (1, 2, 0))
    baseline_cube = np.transpose(baseline, (1, 0))[None, ...]
    delta = candidate_cube - baseline_cube
    draws = common._hierarchical_bootstrap(
        delta,
        replicates=replicates,
        seed=seed,
    )
    return (
        float(delta.mean()),
        float(np.quantile(draws, 0.95)),
        int(np.sum(delta.mean(axis=(1, 2)) < 0.0)),
    )


def _summary_row(
    *,
    partition: str,
    variant: ACOVariant,
    candidates: np.ndarray,
    baseline_2opt: np.ndarray,
    baseline_3opt: np.ndarray,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    partition_code = PARTITIONS.index(partition) + 1
    variant_code = list(ACOVariant).index(variant) + 1
    delta_2, upper_2, improved_runs = _bootstrap_delta(
        candidates,
        baseline_2opt,
        replicates=bootstrap_replicates,
        seed=500_000 + 100 * partition_code + variant_code,
    )
    delta_3, upper_3, _ = _bootstrap_delta(
        candidates,
        baseline_3opt,
        replicates=bootstrap_replicates,
        seed=600_000 + 100 * partition_code + variant_code,
    )
    candidate_mean = float(candidates.mean())
    baseline_2_mean = float(baseline_2opt.mean())
    relative = (
        (baseline_2_mean - candidate_mean) / baseline_2_mean
        if baseline_2_mean > 1.0e-12
        else float("nan")
    )
    return {
        "partition": partition,
        "distribution": partition.removeprefix("tsp500_"),
        "scale": 500,
        "variant": variant.value,
        "method": "tsp500-ls-v2-rmtgp-aco-2opt",
        "gp_runs": int(candidates.shape[1]),
        "instances": int(candidates.shape[2]),
        "aco_seeds": int(candidates.shape[0]),
        "mean_gap_percent": candidate_mean,
        "baseline_2opt_mean_gap_percent": baseline_2_mean,
        "baseline_3opt_mean_gap_percent": float(baseline_3opt.mean()),
        "mean_delta_vs_2opt_pp": delta_2,
        "mean_delta_vs_3opt_pp": delta_3,
        "relative_reduction_vs_2opt": relative,
        "one_sided_upper_95_vs_2opt": upper_2,
        "one_sided_upper_95_vs_3opt": upper_3,
        "improved_gp_runs_vs_2opt": improved_runs,
        "success_vs_2opt": bool(
            np.isfinite(relative)
            and relative >= 0.10
            and upper_2 < 0.0
            and improved_runs >= 2
        ),
        "better_than_3opt": upper_3 < 0.0,
        "noninferior_to_3opt_margin_0p1pp": upper_3 <= 0.10,
    }


def _safe_nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _summarize(
    output: Path,
    *,
    variants: tuple[ACOVariant, ...],
    test_seeds: int,
    bootstrap_replicates: int,
    iterations: int,
) -> dict[str, Any]:
    result_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        for variant in variants:
            shards = []
            for replicate in range(test_seeds):
                with np.load(
                    _shard_path(output, partition, variant, replicate),
                    allow_pickle=False,
                ) as payload:
                    shards.append(
                        {
                            name: np.asarray(payload[name])
                            for name in payload.files
                        }
                    )
            baseline_2opt = np.stack(
                [item["baseline_2opt_final_gap"] for item in shards]
            )
            baseline_3opt = np.stack(
                [item["baseline_3opt_final_gap"] for item in shards]
            )
            candidates = np.stack(
                [item["candidate_final_gap"] for item in shards]
            )
            result_rows.append(
                _summary_row(
                    partition=partition,
                    variant=variant,
                    candidates=candidates,
                    baseline_2opt=baseline_2opt,
                    baseline_3opt=baseline_3opt,
                    bootstrap_replicates=bootstrap_replicates,
                )
            )
            baseline_2_time = np.asarray(
                [item["baseline_2opt_wall_time_sec"] for item in shards],
                dtype=np.float64,
            )
            baseline_3_time = np.asarray(
                [item["baseline_3opt_wall_time_sec"] for item in shards],
                dtype=np.float64,
            )
            individual_time = np.stack(
                [
                    item["candidate_individual_wall_time_sec"]
                    for item in shards
                ]
            )
            individual_mean = _safe_nanmean(individual_time)
            runtime_rows.append(
                {
                    "partition": partition,
                    "variant": variant.value,
                    "aco_2opt_wall_time_sec_mean": float(
                        baseline_2_time.mean()
                    ),
                    "aco_3opt_wall_time_sec_mean": float(
                        baseline_3_time.mean()
                    ),
                    "aco_3opt_over_2opt_time_ratio": float(
                        baseline_3_time.mean() / baseline_2_time.mean()
                    ),
                    "three_gp_program_batch_wall_time_sec_mean": float(
                        np.mean(
                            [
                                item["candidate_batch_wall_time_sec"]
                                for item in shards
                            ]
                        )
                    ),
                    "rmtgp_2opt_individual_wall_time_sec_mean": individual_mean,
                    "aco_3opt_over_rmtgp_2opt_time_ratio": (
                        float(baseline_3_time.mean() / individual_mean)
                        if np.isfinite(individual_mean)
                        else float("nan")
                    ),
                }
            )
    common._write_csv(output / "main_results.csv", result_rows)
    common._write_csv(output / "runtime_results.csv", runtime_rows)
    return {
        "schema_version": 1,
        "protocol": {
            "partitions": list(PARTITIONS),
            "instances_per_partition": 32,
            "aco_iterations": iterations,
            "aco_seeds": test_seeds,
            "gp_runs": len(GP_SEEDS),
            "bootstrap_replicates": bootstrap_replicates,
            "candidate_policy": "selected_candidate_only",
        },
        "results": result_rows,
        "runtime": runtime_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[item.value for item in ACOVariant],
        default=[item.value for item in ACOVariant],
    )
    parser.add_argument(
        "--latency-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="第一个 ACO seed 对三个最终个体分别测在线时间",
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.test_seeds < 2:
        parser.error("iterations 必须为正；test seeds 至少为 2")
    if args.bootstrap_replicates < 100:
        parser.error("bootstrap replicates 至少为 100")
    if args.gpu_device < 0:
        parser.error("gpu device 必须为非负整数")
    variants = tuple(ACOVariant(item) for item in dict.fromkeys(args.variants))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "arguments": {
            **vars(args),
            "output": str(args.output),
            "variants": [item.value for item in variants],
        },
    }
    common._atomic_json(args.output / "manifest.json", manifest)
    try:
        for variant in variants:
            spec, experiment, programs, labels, hashes = _load_variant(
                variant,
                iterations=args.iterations,
                gpu_device=args.gpu_device,
            )
            configure_runtime(experiment.runtime)
            for partition in PARTITIONS:
                path = spec.data.test_paths(partition)[0]
                batches = list(
                    iter_problem_batches(
                        (path,),
                        batch_size=32,
                        candidate_size=experiment.aco.candidate_size,
                        dtype=experiment.aco.dtype,
                        device=experiment.aco.device,
                        max_instances=32,
                    )
                )
                if len(batches) != 1:
                    raise RuntimeError("最终测试必须形成一个 32-instance batch")
                batch = batches[0]
                for replicate in range(args.test_seeds):
                    shard = _shard_path(
                        args.output,
                        partition,
                        variant,
                        replicate,
                    )
                    if common._valid_shard(
                        shard,
                        scale=500,
                        variant=variant,
                        replicate=replicate,
                        hashes=hashes,
                        instance_hashes=list(batch.coordinate_hashes),
                    ):
                        print(f"reuse {shard.name}", flush=True)
                        continue
                    print(
                        f"{partition} {variant.value} "
                        f"seed={replicate + 1}/{args.test_seeds}",
                        flush=True,
                    )
                    common._run_shard(
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
            variants=variants,
            test_seeds=args.test_seeds,
            bootstrap_replicates=args.bootstrap_replicates,
            iterations=args.iterations,
        )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "ended_at": datetime.now(UTC).isoformat(),
                "error": repr(error),
            }
        )
        common._atomic_json(args.output / "manifest.json", manifest)
        raise
    common._atomic_json(args.output / "summary.json", payload)
    manifest.update(
        {
            "status": "completed",
            "ended_at": datetime.now(UTC).isoformat(),
            "summary": "summary.json",
        }
    )
    common._atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
