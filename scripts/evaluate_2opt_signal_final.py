#!/usr/bin/env python3
"""锁定 9 个 formal champions 后执行 TSP100、5000-iteration paired test。"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
from rmtgp_aco.config import ACOVariant, ExecutionBackend, GPUMode
from rmtgp_aco.evaluation import compile_champion, load_champion
from rmtgp_aco.runtime import configure_runtime
from rmtgp_aco.sampling import iter_problem_batches
from rmtgp_aco.spec import load_run_spec

ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "runs" / "tsp100-2opt-signal-v2" / "formal"
OUTPUT = ROOT / "runs" / "tsp100-2opt-signal-v2" / "test"
VARIANTS = tuple(ACOVariant)
GP_SEEDS = (71001, 71002, 71003)
TEST_ROOT_SEED = 82001
SCHEMA_VERSION = 1


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
        raise ValueError("不能写空测试 CSV")
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _config_path(variant: ACOVariant, seed: int) -> Path:
    return (
        FORMAL_ROOT
        / "configs"
        / f"tsp100-2opt-v2-{variant.value}-seed-{seed}.yaml"
    )


def _run_path(variant: ACOVariant, seed: int) -> Path:
    return FORMAL_ROOT / "train" / variant.value / f"seed-{seed}"


def _load_variant(variant: ACOVariant, iterations: int):
    config_path = _config_path(variant, GP_SEEDS[0])
    spec = load_run_spec(config_path)
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
    programs = [(None, None)]
    hashes = ["aco-baseline-passthrough"]
    for gp_seed in GP_SEEDS:
        run = _run_path(variant, gp_seed)
        manifest = json.loads(
            (run / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("status") != "completed":
            raise RuntimeError(f"formal run 未完成: {run}")
        champion = load_champion(run / "selected_candidate.pkl")
        programs.append(compile_champion(champion))
        hashes.append(champion.structural_hash)
    return spec, experiment, programs, hashes


def _test_seed(variant: ACOVariant, replicate: int) -> int:
    code = {
        ACOVariant.AS: 1,
        ACOVariant.ACS: 2,
        ACOVariant.MMAS: 3,
    }[variant]
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [TEST_ROOT_SEED, code, replicate, 0x54455354]
        )
    )
    return int(rng.integers(0, 2**63 - 1))


def _shard_path(
    output: Path,
    variant: ACOVariant,
    replicate: int,
) -> Path:
    return output / "shards" / f"{variant.value}-seed-{replicate:02d}.npz"


def _first_target_iteration(
    gap_curve: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """返回每个 program×instance 首次达到 gap threshold 的 1-based 轮数。"""

    reached = gap_curve <= threshold
    first = np.argmax(reached, axis=-1) + 1
    return np.where(np.any(reached, axis=-1), first, -1)


def _run_shard(
    path: Path,
    *,
    variant: ACOVariant,
    replicate: int,
    batch,
    experiment,
    programs,
    hashes: list[str],
) -> None:
    seed = _test_seed(variant, replicate)
    result = solve_population_cuda_anytime(
        batch,
        experiment.aco,
        programs,
        seed=seed,
        runtime=experiment.runtime,
    )
    reference = batch.reference_length.numpy()
    final_gap = (
        100.0
        * (result.best_length.numpy() - reference[None, :])
        / reference[None, :]
    )
    anytime_gap = (
        100.0
        * (
            result.anytime_best.numpy()
            - reference[None, :, None]
        )
        / reference[None, :, None]
    )
    anytime_auc = np.mean(anytime_gap, axis=-1)
    anytime_curve = np.mean(anytime_gap, axis=1)
    target_iteration = _first_target_iteration(anytime_gap, 0.10)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        variant=np.asarray(variant.value),
        replicate=np.asarray(replicate, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int64),
        champion_hashes=np.asarray(hashes),
        instance_ids=np.asarray(batch.instance_ids),
        coordinate_hashes=np.asarray(batch.coordinate_hashes),
        reference_length=reference,
        final_gap_percent=final_gap,
        anytime_gap_auc=anytime_auc,
        anytime_mean_gap_curve=anytime_curve,
        time_to_gap_0p1=target_iteration,
        best_iteration=result.best_iteration.numpy(),
        diagnostics=result.diagnostics.numpy(),
        wall_time_sec=np.asarray(result.wall_time_sec),
    )
    os.replace(temporary, path)


def _valid_shard(
    path: Path,
    *,
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
                and str(payload["variant"]) == variant.value
                and int(payload["replicate"]) == replicate
                and payload["champion_hashes"].tolist() == hashes
                and payload["coordinate_hashes"].tolist() == instance_hashes
            )
    except (OSError, KeyError, ValueError):
        return False


def _hierarchical_bootstrap(
    delta_cube: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """对 GP run→instance→ACO seed 三层进行有放回重采样。"""

    if delta_cube.ndim != 3:
        raise ValueError("delta_cube 必须为 [R,I,S]")
    runs, instances, aco_seeds = delta_cube.shape
    if runs < 2 or instances < 1 or aco_seeds < 2:
        raise ValueError("层次 bootstrap 至少要求 2 runs 和 2 ACO seeds")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    chunk_size = max(
        1,
        min(
            replicates,
            2_000_000 // max(runs * instances * aco_seeds, 1),
        ),
    )
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        count = stop - start
        sampled_runs = rng.integers(0, runs, size=(count, runs))
        sampled_instances = rng.integers(
            0,
            instances,
            size=(count, runs, instances),
        )
        sampled_seeds = rng.integers(
            0,
            aco_seeds,
            size=(count, runs, instances, aco_seeds),
        )
        run_index = sampled_runs[:, :, None, None]
        instance_index = sampled_instances[:, :, :, None]
        sampled = delta_cube[
            run_index,
            instance_index,
            sampled_seeds,
        ]
        estimates[start:stop] = sampled.mean(axis=(1, 2, 3))
    return estimates


def _summarize_variant(
    output: Path,
    variant: ACOVariant,
    *,
    test_seeds: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    shards = []
    for replicate in range(test_seeds):
        with np.load(
            _shard_path(output, variant, replicate),
            allow_pickle=False,
        ) as payload:
            shards.append({name: np.asarray(payload[name]) for name in payload.files})
    # [S,P,I]；P=baseline+3 GP runs。
    final = np.stack([item["final_gap_percent"] for item in shards])
    auc = np.stack([item["anytime_gap_auc"] for item in shards])
    time_to_target = np.stack([item["time_to_gap_0p1"] for item in shards])
    curves = np.stack([item["anytime_mean_gap_curve"] for item in shards])
    baseline = final[:, 0, :]
    candidate = final[:, 1:, :]
    # 转为 [R,I,S]，与层次 bootstrap 定义一致。
    delta = np.transpose(
        candidate - baseline[:, None, :],
        (1, 2, 0),
    )
    candidate_cube = np.transpose(candidate, (1, 2, 0))
    bootstrap = _hierarchical_bootstrap(
        delta,
        replicates=bootstrap_replicates,
        seed=91000 + list(VARIANTS).index(variant),
    )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    one_sided_upper = float(np.quantile(bootstrap, 0.95))
    baseline_mean = float(np.mean(baseline))
    candidate_mean = float(np.mean(candidate_cube))
    relative_reduction = (
        (baseline_mean - candidate_mean) / baseline_mean
        if baseline_mean > 1e-12
        else float("nan")
    )
    gp_run_deltas = np.mean(delta, axis=(1, 2))
    improved_runs = int(np.sum(gp_run_deltas < 0.0))
    summary = {
        "variant": variant.value,
        "gp_runs": len(GP_SEEDS),
        "aco_seeds": test_seeds,
        "instances": int(final.shape[-1]),
        "baseline_mean_gap_percent": baseline_mean,
        "candidate_mean_gap_percent": candidate_mean,
        "mean_delta_pp": float(np.mean(delta)),
        "median_delta_pp": float(np.median(delta)),
        "relative_gap_reduction": relative_reduction,
        "gp_run_mean_delta_pp": gp_run_deltas.tolist(),
        "improved_gp_runs": improved_runs,
        "bootstrap_lower_95": float(lower),
        "bootstrap_upper_95": float(upper),
        "bootstrap_one_sided_upper_95": one_sided_upper,
        "win_rate": float(np.mean(delta < -1e-12)),
        "tie_rate": float(np.mean(np.abs(delta) <= 1e-12)),
        "loss_rate": float(np.mean(delta > 1e-12)),
        "baseline_anytime_gap_auc": float(np.mean(auc[:, 0, :])),
        "candidate_anytime_gap_auc": float(np.mean(auc[:, 1:, :])),
        "baseline_time_to_gap_0p1": float(
            np.mean(time_to_target[:, 0, :][time_to_target[:, 0, :] > 0])
        )
        if np.any(time_to_target[:, 0, :] > 0)
        else float("nan"),
        "candidate_time_to_gap_0p1": float(
            np.mean(time_to_target[:, 1:, :][time_to_target[:, 1:, :] > 0])
        )
        if np.any(time_to_target[:, 1:, :] > 0)
        else float("nan"),
        "success_criteria": {
            "relative_reduction_at_least_10_percent": bool(
                relative_reduction >= 0.10
            )
            if np.isfinite(relative_reduction)
            else False,
            "one_sided_bootstrap_upper_below_zero": one_sided_upper < 0.0,
            "at_least_two_of_three_gp_runs_improve": improved_runs >= 2,
        },
    }
    summary["success"] = all(summary["success_criteria"].values())

    rows: list[dict[str, Any]] = []
    instance_ids = shards[0]["instance_ids"].tolist()
    for aco_seed in range(test_seeds):
        for instance, instance_id in enumerate(instance_ids):
            rows.append(
                {
                    "variant": variant.value,
                    "method": "aco-2opt",
                    "gp_seed": 0,
                    "aco_seed": int(shards[aco_seed]["seed"]),
                    "instance_id": instance_id,
                    "gap_percent": float(baseline[aco_seed, instance]),
                    "delta_pp": 0.0,
                    "anytime_gap_auc": float(auc[aco_seed, 0, instance]),
                    "time_to_gap_0p1": int(
                        time_to_target[aco_seed, 0, instance]
                    ),
                }
            )
            for gp_index, gp_seed in enumerate(GP_SEEDS):
                rows.append(
                    {
                        "variant": variant.value,
                        "method": "rmtgp-aco-2opt",
                        "gp_seed": gp_seed,
                        "aco_seed": int(shards[aco_seed]["seed"]),
                        "instance_id": instance_id,
                        "gap_percent": float(
                            candidate[aco_seed, gp_index, instance]
                        ),
                        "delta_pp": float(
                            delta[gp_index, instance, aco_seed]
                        ),
                        "anytime_gap_auc": float(
                            auc[aco_seed, gp_index + 1, instance]
                        ),
                        "time_to_gap_0p1": int(
                            time_to_target[
                                aco_seed,
                                gp_index + 1,
                                instance,
                            ]
                        ),
                    }
                )
    baseline_curve = np.mean(curves[:, 0, :], axis=0)
    candidate_curve = np.mean(curves[:, 1:, :], axis=(0, 1))
    return summary, rows, baseline_curve, candidate_curve


def _plot(
    output: Path,
    summaries: list[dict[str, Any]],
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    labels = [item["variant"].upper() for item in summaries]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(
        x - width / 2,
        [item["baseline_mean_gap_percent"] for item in summaries],
        width,
        label="ACO+2-opt",
    )
    axis.bar(
        x + width / 2,
        [item["candidate_mean_gap_percent"] for item in summaries],
        width,
        label="RMTGP-ACO+2-opt",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean optimality gap (%)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"final_gap_comparison.{suffix}", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=False)
    for axis, variant in zip(axes, VARIANTS, strict=True):
        baseline, candidate = curves[variant.value]
        iterations = np.arange(1, baseline.size + 1)
        axis.plot(iterations, baseline, label="ACO+2-opt")
        axis.plot(iterations, candidate, label="RMTGP-ACO+2-opt")
        axis.set_title(variant.value.upper())
        axis.set_xlabel("ACO iteration")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean anytime gap (%)")
    axes[-1].legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"anytime_curves.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--test-seeds", type=int, default=3)
    parser.add_argument("--max-instances", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.test_seeds < 2 or args.bootstrap_replicates < 100:
        parser.error("test seeds 至少为 2，bootstrap replicates 至少为 100")
    if args.iterations < 1 or args.max_instances < 1 or args.batch_size < 1:
        parser.error("iterations/instances/batch size 必须为正整数")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "arguments": vars(args) | {"output": str(args.output)},
    }
    _atomic_json(args.output / "manifest.json", manifest)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    curve_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for variant in VARIANTS:
        spec, experiment, programs, hashes = _load_variant(
            variant,
            args.iterations,
        )
        configure_runtime(experiment.runtime)
        batches = list(
            iter_problem_batches(
                spec.data.test_paths("tsp100_uniform"),
                batch_size=args.batch_size,
                candidate_size=experiment.aco.candidate_size,
                dtype=experiment.aco.dtype,
                device=experiment.aco.device,
                max_instances=args.max_instances,
            )
        )
        if len(batches) != 1:
            raise ValueError(
                "当前 paired test 要求 max_instances<=batch_size，"
                "从而每个 seed 只有一个确定 batch"
            )
        batch = batches[0]
        for replicate in range(args.test_seeds):
            shard = _shard_path(args.output, variant, replicate)
            if _valid_shard(
                shard,
                variant=variant,
                replicate=replicate,
                hashes=hashes,
                instance_hashes=list(batch.coordinate_hashes),
            ):
                print(f"reuse {shard.name}", flush=True)
                continue
            print(
                f"test {variant.value} seed={replicate + 1}/{args.test_seeds}",
                flush=True,
            )
            _run_shard(
                shard,
                variant=variant,
                replicate=replicate,
                batch=batch,
                experiment=experiment,
                programs=programs,
                hashes=hashes,
            )
        summary, rows, baseline_curve, candidate_curve = _summarize_variant(
            args.output,
            variant,
            test_seeds=args.test_seeds,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        curve_map[variant.value] = (baseline_curve, candidate_curve)
    _write_csv(args.output / "paired_test_records.csv", all_rows)
    _write_csv(args.output / "main_results.csv", summaries)
    _plot(args.output, summaries, curve_map)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "scale": 100,
            "iterations": args.iterations,
            "ants": 32,
            "gp_runs": 3,
            "aco_seeds": args.test_seeds,
            "instances": args.max_instances,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "results": summaries,
    }
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
