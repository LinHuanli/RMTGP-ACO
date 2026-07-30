#!/usr/bin/env python3
"""审计 TSP500 上 Anytime+Final fitness 与多保真 racing 的可辨识度。

脚本复用 TSP100 信号审计中冻结的 64 个 GP programs。所有 horizon 使用
相同的 instance、ACO seed 和 counter-based RNG。每个 shard 只保存 final
gap 与 best-so-far 曲线均值，不保存完整 colony，因此适合 5000 轮审计。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from rmtgp_aco.aco_cuda import solve_population_cuda
from rmtgp_aco.config import ACOVariant, ExecutionBackend, GPUMode, LocalSearch
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.genetic import compile_individual
from rmtgp_aco.ls_signal import signal_to_noise_statistics, spearman_by_context
from rmtgp_aco.runtime import configure_runtime
from rmtgp_aco.sampling import pools_from_paths
from rmtgp_aco.spec import load_run_spec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "tsp500_2opt_racing" / "config.yaml"
DEFAULT_PROGRAM_ROOT = ROOT / "runs" / "tsp100-2opt-signal-v2" / "audit"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "tsp500-2opt-racing" / "audit"
SCHEMA_VERSION = 1
VARIANT_PARAMETERS = {
    ACOVariant.AS: {"rho": 0.5, "q0": 0.0, "gamma": 1.0 / 3.0},
    ACOVariant.ACS: {"rho": 0.1, "q0": 0.98, "gamma": 1.0 / 6.0},
    ACOVariant.MMAS: {"rho": 0.2, "q0": 0.0, "gamma": 1.0 / 6.0},
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _experiment(config: Path, variant: ACOVariant):
    spec = load_run_spec(config)
    if tuple(spec.experiment.train_scales) != (500,):
        raise ValueError("TSP500 racing audit 要求唯一 train scale 为 500")
    parameters = VARIANT_PARAMETERS[variant]
    experiment = replace(
        spec.experiment,
        aco=replace(
            spec.experiment.aco,
            variant=variant,
            ants=32,
            rho=parameters["rho"],
            q0=parameters["q0"],
            gamma_transition=parameters["gamma"],
            gamma_pheromone=parameters["gamma"],
            local_search=LocalSearch.TWO_OPT,
        ),
        runtime=replace(
            spec.experiment.runtime,
            aco_backend=ExecutionBackend.CUDA_TILED_V2,
            gpu_devices=(0,),
            gpu_mode=GPUMode.SINGLE,
        ),
    )
    return spec, experiment


def _load_programs(
    root: Path,
    variant: ACOVariant,
) -> tuple[list[tuple[Any, Any]], list[str]]:
    source = root / variant.value / "programs.pkl"
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    individuals = list(payload["individuals"])
    if len(individuals) != 64:
        raise ValueError(f"{source}: 预期 64 个 programs")
    programs = [(None, None), *[compile_individual(item) for item in individuals]]
    hashes = [
        "aco-baseline-passthrough",
        *[item.structural_hash for item in individuals],
    ]
    return programs, hashes


def _aco_seed(root_seed: int, variant: ACOVariant, replicate: int) -> int:
    """horizon 不进入 seed，使较长运行严格包含较短运行的 RNG 前缀。"""

    variant_code = list(ACOVariant).index(variant) + 1
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [root_seed, variant_code, replicate, 0x414E5954]
        )
    )
    return int(rng.integers(0, 2**63 - 1))


def _selected_batch(
    *,
    output: Path,
    spec,
    experiment,
    variant: ACOVariant,
    root_seed: int,
    screening_horizon: int,
    per_difficulty: int,
):
    """用独立 baseline run 冻结 easy/hard 各半实例。"""

    artifact = output / "instances.json"
    pool = pools_from_paths(spec.data.validation_paths())[500]
    if artifact.is_file():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        records = [
            pool.get(int(item["logical_index"]))
            for item in payload["selected"]
        ]
        batch = make_problem_batch(
            records,
            candidate_size=experiment.aco.candidate_size,
            dtype=experiment.aco.dtype,
            device=experiment.aco.device,
        )
        expected = [
            str(item["coordinate_hash"]) for item in payload["selected"]
        ]
        if list(batch.coordinate_hashes) != expected:
            raise ValueError("validation 数据与 instances.json 不一致")
        return batch, payload["selected"]

    if 2 * per_difficulty > len(pool):
        raise ValueError("easy/hard 实例需求超过 validation pool")
    records = [pool.get(index) for index in range(len(pool))]
    screening_batch = make_problem_batch(
        records,
        candidate_size=experiment.aco.candidate_size,
        dtype=experiment.aco.dtype,
        device=experiment.aco.device,
    )
    quality = solve_population_cuda(
        screening_batch,
        replace(experiment.aco, iterations=screening_horizon),
        [(None, None)],
        seed=_aco_seed(root_seed, variant, 2**30),
        runtime=experiment.runtime,
    )
    reference = screening_batch.reference_length.numpy()
    gaps = 100.0 * (
        quality.best_length[0].numpy() - reference
    ) / reference
    order = np.argsort(gaps, kind="stable")
    positions = np.concatenate(
        (order[:per_difficulty], order[-per_difficulty:])
    )
    selected: list[dict[str, Any]] = []
    chosen = []
    for rank, position in enumerate(positions):
        record = records[int(position)]
        chosen.append(record)
        selected.append(
            {
                "position": rank,
                "difficulty": (
                    "easy" if rank < per_difficulty else "hard"
                ),
                "logical_index": int(position),
                "instance_id": record.instance_id,
                "coordinate_hash": record.coordinate_hash,
                "screening_final_gap_percent": float(gaps[position]),
            }
        )
    _atomic_json(
        artifact,
        {
            "schema_version": SCHEMA_VERSION,
            "variant": variant.value,
            "screening_horizon": screening_horizon,
            "selected": selected,
        },
    )
    return (
        make_problem_batch(
            chosen,
            candidate_size=experiment.aco.candidate_size,
            dtype=experiment.aco.dtype,
            device=experiment.aco.device,
        ),
        selected,
    )


def _shard_path(output: Path, horizon: int, replicate: int) -> Path:
    return output / "shards" / f"h{horizon:05d}-seed-{replicate:02d}.npz"


def _valid_shard(
    path: Path,
    *,
    horizon: int,
    program_hashes: list[str],
    instance_hashes: list[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            shape = (len(program_hashes), len(instance_hashes))
            return (
                int(payload["schema_version"]) == SCHEMA_VERSION
                and int(payload["horizon"]) == horizon
                and payload["program_hashes"].tolist() == program_hashes
                and payload["instance_hashes"].tolist() == instance_hashes
                and payload["final_gap_percent"].shape == shape
                and payload["anytime_gap_percent"].shape == shape
            )
    except (OSError, KeyError, ValueError):
        return False


def _run_shard(
    path: Path,
    *,
    batch,
    programs,
    program_hashes: list[str],
    experiment,
    variant: ACOVariant,
    root_seed: int,
    horizon: int,
    replicate: int,
) -> None:
    started = perf_counter()
    quality = solve_population_cuda(
        batch,
        replace(experiment.aco, iterations=horizon),
        programs,
        seed=_aco_seed(root_seed, variant, replicate),
        runtime=experiment.runtime,
    )
    if quality.anytime_mean_length is None:
        raise RuntimeError("CUDA backend 未返回 anytime_mean_length")
    reference = batch.reference_length.numpy()[None, :]
    final_gap = 100.0 * (quality.best_length.numpy() - reference) / reference
    anytime_gap = 100.0 * (
        quality.anytime_mean_length.numpy() - reference
    ) / reference
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        variant=np.asarray(variant.value),
        horizon=np.asarray(horizon, dtype=np.int32),
        replicate=np.asarray(replicate, dtype=np.int32),
        seed=np.asarray(
            _aco_seed(root_seed, variant, replicate),
            dtype=np.int64,
        ),
        program_hashes=np.asarray(program_hashes),
        instance_hashes=np.asarray(batch.coordinate_hashes),
        final_gap_percent=final_gap,
        anytime_gap_percent=anytime_gap,
        best_iteration=quality.best_iteration.numpy(),
        diagnostics=quality.diagnostics.numpy(),
        wall_time_sec=np.asarray(perf_counter() - started),
        kernel_time_sec=np.asarray(
            float(quality.backend_metrics["kernel_seconds_critical"])
        ),
    )
    os.replace(temporary, path)


def _balanced_indices(
    selected_instances: list[dict[str, Any]],
    budget: int,
) -> np.ndarray:
    if budget % 2:
        raise ValueError("audit instance budget 必须为偶数")
    half = budget // 2
    easy = [
        index
        for index, item in enumerate(selected_instances)
        if item["difficulty"] == "easy"
    ][:half]
    hard = [
        index
        for index, item in enumerate(selected_instances)
        if item["difficulty"] == "hard"
    ][:half]
    if len(easy) != half or len(hard) != half:
        raise ValueError("easy/hard 实例不足")
    return np.asarray([*easy, *hard], dtype=np.int64)


def _summarize(
    *,
    output: Path,
    variant: ACOVariant,
    horizons: tuple[int, ...],
    seeds: int,
    selected_instances: list[dict[str, Any]],
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    final = []
    anytime = []
    wall = []
    kernel = []
    for horizon in horizons:
        final_by_seed = []
        anytime_by_seed = []
        for replicate in range(seeds):
            with np.load(
                _shard_path(output, horizon, replicate),
                allow_pickle=False,
            ) as payload:
                final_by_seed.append(np.asarray(payload["final_gap_percent"]))
                anytime_by_seed.append(
                    np.asarray(payload["anytime_gap_percent"])
                )
                wall.append(float(payload["wall_time_sec"]))
                kernel.append(float(payload["kernel_time_sec"]))
        final.append(final_by_seed)
        anytime.append(anytime_by_seed)
    # [H,S,P+1,I]
    final_array = np.asarray(final, dtype=np.float64)
    anytime_array = np.asarray(anytime, dtype=np.float64)
    target_index = len(horizons) - 1
    reports: list[dict[str, Any]] = []
    recommended: dict[str, Any] | None = None
    for budget in budgets:
        indices = _balanced_indices(selected_instances, budget)
        target_final = (
            final_array[target_index, :, 1:, :][:, :, indices]
            - final_array[target_index, :, :1, :][:, :, indices]
        )
        target_anytime = (
            anytime_array[target_index, :, 1:, :][:, :, indices]
            - anytime_array[target_index, :, :1, :][:, :, indices]
        )
        target_combined = 0.5 * target_final + 0.5 * target_anytime
        target_scores = target_combined.mean(axis=(0, 2))
        for horizon_index, horizon in enumerate(horizons):
            final_delta = (
                final_array[horizon_index, :, 1:, :][:, :, indices]
                - final_array[horizon_index, :, :1, :][:, :, indices]
            )
            anytime_delta = (
                anytime_array[horizon_index, :, 1:, :][:, :, indices]
                - anytime_array[horizon_index, :, :1, :][:, :, indices]
            )
            combined = 0.5 * final_delta + 0.5 * anytime_delta
            scores = combined.mean(axis=(0, 2))
            top_k = min(32, scores.size)
            screen_top = set(
                np.argsort(scores, kind="stable")[:top_k].tolist()
            )
            target_top = set(
                np.argsort(target_scores, kind="stable")[:top_k].tolist()
            )
            recall = len(screen_top & target_top) / float(top_k)
            correlation = float(
                spearman_by_context(
                    scores[:, None],
                    target_scores[:, None],
                )[0]
            )
            snr = signal_to_noise_statistics(
                np.moveaxis(combined, 1, 0)
            ).ratio_of_mean_variances
            final_nonzero = float(np.mean(np.abs(final_delta) > 1e-12))
            anytime_nonzero = float(
                np.mean(np.abs(anytime_delta) > 1e-12)
            )
            passed = bool(
                final_nonzero >= 0.5
                and anytime_nonzero >= 0.5
                and snr >= 1.0
                and recall >= 0.8
                and correlation >= 0.7
            )
            report = {
                "horizon": horizon,
                "instances": budget,
                "final_nonzero_fraction": final_nonzero,
                "anytime_nonzero_fraction": anytime_nonzero,
                "combined_signal_to_noise": float(snr),
                "top32_recall_at_target": float(recall),
                "spearman_at_target": correlation,
                "mean_final_delta_pp": float(final_delta.mean()),
                "mean_anytime_delta_pp": float(anytime_delta.mean()),
                "mean_combined_delta_pp": float(combined.mean()),
                "gate_passed": passed,
            }
            reports.append(report)
            # 5000 轮只作为最终排序参考，不能成为训练 screen：否则
            # Stage 1 会比 100/200/500 轮的 Stage 2 更昂贵且保真度倒置。
            if (
                recommended is None
                and passed
                and horizon != horizons[-1]
            ):
                recommended = dict(report)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant.value,
        "programs": 64,
        "seeds": seeds,
        "horizons": list(horizons),
        "instance_budgets": list(budgets),
        "gate": {
            "final_nonzero_fraction_min": 0.5,
            "anytime_nonzero_fraction_min": 0.5,
            "combined_signal_to_noise_min": 1.0,
            "top32_recall_min": 0.8,
            "spearman_min": 0.7,
        },
        "reports": reports,
        "recommended_screen": recommended,
        "formal_training_allowed": recommended is not None,
        "runtime": {
            "wall_time_sec_sum": float(np.sum(wall)),
            "kernel_time_sec_sum": float(np.sum(kernel)),
        },
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--variant",
        choices=[item.value for item in ACOVariant],
        required=True,
    )
    parser.add_argument("--program-root", type=Path, default=DEFAULT_PROGRAM_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root-seed", type=int, default=83021)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=(100, 200, 500, 5000),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--screening-horizon", type=int, default=500)
    parser.add_argument("--per-difficulty", type=int, default=8)
    parser.add_argument(
        "--instance-budgets",
        type=int,
        nargs="+",
        default=(8, 16),
    )
    args = parser.parse_args()
    args.horizons = tuple(dict.fromkeys(int(item) for item in args.horizons))
    args.instance_budgets = tuple(
        dict.fromkeys(int(item) for item in args.instance_budgets)
    )
    if args.horizons[-1] != 5000:
        parser.error("最后一个 horizon 必须是目标 5000")
    if args.seeds < 2:
        parser.error("--seeds 至少为 2")
    if max(args.instance_budgets) > 2 * args.per_difficulty:
        parser.error("instance budget 超过已选择的 easy+hard 实例数")

    variant = ACOVariant(args.variant)
    output = args.output or DEFAULT_OUTPUT_ROOT / variant.value
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "variant": variant.value,
        "started_at": datetime.now(UTC).isoformat(),
        "arguments": {
            name: (
                value.as_posix()
                if isinstance(value, Path)
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for name, value in vars(args).items()
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    try:
        spec, experiment = _experiment(args.config, variant)
        configure_runtime(experiment.runtime)
        programs, program_hashes = _load_programs(
            args.program_root,
            variant,
        )
        batch, selected = _selected_batch(
            output=output,
            spec=spec,
            experiment=experiment,
            variant=variant,
            root_seed=args.root_seed,
            screening_horizon=args.screening_horizon,
            per_difficulty=args.per_difficulty,
        )
        total = len(args.horizons) * args.seeds
        completed = 0
        for horizon in args.horizons:
            for replicate in range(args.seeds):
                path = _shard_path(output, horizon, replicate)
                if not _valid_shard(
                    path,
                    horizon=horizon,
                    program_hashes=program_hashes,
                    instance_hashes=list(batch.coordinate_hashes),
                ):
                    _run_shard(
                        path,
                        batch=batch,
                        programs=programs,
                        program_hashes=program_hashes,
                        experiment=experiment,
                        variant=variant,
                        root_seed=args.root_seed,
                        horizon=horizon,
                        replicate=replicate,
                    )
                completed += 1
                print(
                    f"[{variant.value}] {path.name} "
                    f"({completed}/{total})",
                    flush=True,
                )
        summary = _summarize(
            output=output,
            variant=variant,
            horizons=args.horizons,
            seeds=args.seeds,
            selected_instances=selected,
            budgets=args.instance_budgets,
        )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "ended_at": datetime.now(UTC).isoformat(),
                "error": repr(error),
            }
        )
        _atomic_json(output / "manifest.json", manifest)
        raise
    manifest.update(
        {
            "status": "completed",
            "ended_at": datetime.now(UTC).isoformat(),
            "summary": "summary.json",
        }
    )
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
