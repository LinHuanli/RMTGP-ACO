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
import queue
import threading
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


def _run_test_tasks(
    *,
    output: Path,
    variants: tuple[ACOVariant, ...],
    iterations: int,
    test_seeds: int,
    gpu_devices: tuple[int, ...],
    latency_audit: bool,
    manifest: dict[str, Any],
) -> None:
    """用每张物理 GPU 一个线程动态调度独立、可恢复的 shard。"""

    tasks: queue.Queue[tuple[ACOVariant, str, int]] = queue.Queue()
    for variant in variants:
        for partition in PARTITIONS:
            for replicate in range(test_seeds):
                tasks.put((variant, partition, replicate))
    lock = threading.Lock()
    active: dict[int, str] = {}
    completed: list[str] = []
    reused: list[str] = []
    failures: list[dict[str, str]] = []

    def write_progress() -> None:
        manifest["progress"] = {
            "active": {str(key): value for key, value in active.items()},
            "completed": list(completed),
            "reused": list(reused),
            "failures": list(failures),
            "pending_count": tasks.qsize(),
        }
        common._atomic_json(output / "manifest.json", manifest)

    def worker(device: int) -> None:
        loaded: dict[ACOVariant, tuple[Any, ...]] = {}
        batches: dict[tuple[ACOVariant, str], Any] = {}
        while True:
            try:
                variant, partition, replicate = tasks.get_nowait()
            except queue.Empty:
                return
            label = f"{partition}-{variant.value}-seed-{replicate:02d}"
            with lock:
                active[device] = label
                write_progress()
            try:
                if variant not in loaded:
                    loaded[variant] = _load_variant(
                        variant,
                        iterations=iterations,
                        gpu_device=device,
                    )
                spec, experiment, programs, labels, hashes = loaded[variant]
                batch_key = (variant, partition)
                if batch_key not in batches:
                    path = spec.data.test_paths(partition)[0]
                    problem_batches = list(
                        iter_problem_batches(
                            (path,),
                            batch_size=32,
                            candidate_size=experiment.aco.candidate_size,
                            dtype=experiment.aco.dtype,
                            device=experiment.aco.device,
                            max_instances=32,
                        )
                    )
                    if len(problem_batches) != 1:
                        raise RuntimeError(
                            "最终测试必须形成一个 32-instance batch"
                        )
                    batches[batch_key] = problem_batches[0]
                batch = batches[batch_key]
                shard = _shard_path(output, partition, variant, replicate)
                if common._valid_shard(
                    shard,
                    scale=500,
                    variant=variant,
                    replicate=replicate,
                    hashes=hashes,
                    instance_hashes=list(batch.coordinate_hashes),
                ):
                    with lock:
                        reused.append(label)
                    print(f"reuse {shard.name}", flush=True)
                else:
                    print(f"GPU{device} {label}", flush=True)
                    common._run_shard(
                        shard,
                        batch=batch,
                        experiment=experiment,
                        programs=programs,
                        labels=labels,
                        hashes=hashes,
                        variant=variant,
                        replicate=replicate,
                        latency_audit=latency_audit,
                    )
                    with lock:
                        completed.append(label)
            except Exception as error:
                with lock:
                    failures.append(
                        {
                            "task": label,
                            "gpu": str(device),
                            "error": repr(error),
                        }
                    )
            finally:
                with lock:
                    active.pop(device, None)
                    write_progress()
                tasks.task_done()

    with lock:
        write_progress()
    workers = [
        threading.Thread(target=worker, args=(device,), daemon=False)
        for device in gpu_devices
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if failures:
        raise RuntimeError(f"最终测试有 {len(failures)} 个 shard 失败: {failures}")


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
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=None,
        help="兼容旧调用：指定单张 GPU",
    )
    parser.add_argument(
        "--gpu-devices",
        type=int,
        nargs="+",
        default=None,
        help="并行最终测试所用的物理 GPU 列表",
    )
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
    if args.gpu_device is not None and args.gpu_devices is not None:
        parser.error("--gpu-device 与 --gpu-devices 不能同时使用")
    gpu_devices = tuple(
        dict.fromkeys(
            args.gpu_devices
            if args.gpu_devices is not None
            else [0 if args.gpu_device is None else args.gpu_device]
        )
    )
    if not gpu_devices or min(gpu_devices) < 0:
        parser.error("gpu devices 必须是非空非负整数列表")
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
            "gpu_devices": list(gpu_devices),
        },
    }
    common._atomic_json(args.output / "manifest.json", manifest)
    try:
        bootstrap_spec, bootstrap_experiment, *_ = _load_variant(
            variants[0],
            iterations=args.iterations,
            gpu_device=gpu_devices[0],
        )
        del bootstrap_spec
        configure_runtime(bootstrap_experiment.runtime)
        _run_test_tasks(
            output=args.output,
            variants=variants,
            iterations=args.iterations,
            test_seeds=args.test_seeds,
            gpu_devices=gpu_devices,
            latency_audit=args.latency_audit,
            manifest=manifest,
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
