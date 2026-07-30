#!/usr/bin/env python3
"""审计 full 2-opt 对 RMTGP-ACO 学习信号的压缩与信用错配。

默认每个 ACO variant 使用 32 个随机个体、16 个旧 checkpoint 和 16 个
checkpoint 小变异。原始 ACO 作为第 0 个 program，与全部 residual programs
共享实例、seed 和 ant-level counter RNG。每个 horizon×seed 单独保存 shard，
任务中断后可以直接续跑。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from deap import gp

from rmtgp_aco.aco_cuda import solve_population_cuda
from rmtgp_aco.config import (
    ACOVariant,
    ExecutionBackend,
    GPUMode,
    LocalSearch,
)
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.evaluation import load_champion
from rmtgp_aco.genetic import (
    RMTGPIndividual,
    compile_individual,
    is_baseline_individual,
    make_individual,
    mutate_role_preserving,
)
from rmtgp_aco.ls_signal import (
    compression_statistics,
    edge_difference_survival,
    finite_summary,
    signal_to_noise_statistics,
    spearman_by_context,
)
from rmtgp_aco.program import create_primitive_sets
from rmtgp_aco.runtime import configure_runtime
from rmtgp_aco.sampling import pools_from_paths
from rmtgp_aco.spec import load_run_spec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "experiments" / "tsp100_2opt_signal_v2" / "config.yaml"
)
DEFAULT_CHECKPOINT_ROOT = ROOT / "runs" / "tsp100-local-search-3seed" / "train"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "tsp100-2opt-signal-v2" / "audit"
SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能写空 audit CSV")
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


def _variant_code(variant: ACOVariant) -> int:
    return {
        ACOVariant.AS: 1,
        ACOVariant.ACS: 2,
        ACOVariant.MMAS: 3,
    }[variant]


def _variant_parameters(variant: ACOVariant) -> dict[str, float]:
    return {
        ACOVariant.AS: {"rho": 0.5, "q0": 0.0},
        ACOVariant.ACS: {"rho": 0.1, "q0": 0.98},
        ACOVariant.MMAS: {"rho": 0.2, "q0": 0.0},
    }[variant]


def _audit_experiment(config: Path, variant: ACOVariant):
    spec = load_run_spec(config)
    parameters = _variant_parameters(variant)
    aco = replace(
        spec.experiment.aco,
        variant=variant,
        ants=32,
        alpha=1.0,
        beta=2.0,
        rho=parameters["rho"],
        q0=parameters["q0"],
        xi=0.1,
        local_search=LocalSearch.TWO_OPT,
        local_search_candidate_size=20,
        local_search_dlb=True,
    )
    runtime = replace(
        spec.experiment.runtime,
        aco_backend=ExecutionBackend.CUDA_TILED_V2,
        gpu_devices=(0,),
        gpu_mode=GPUMode.SINGLE,
    )
    return spec, replace(spec.experiment, aco=aco, runtime=runtime)


def _discover_old_candidates(
    root: Path,
    variant: ACOVariant,
) -> list[tuple[Path, RMTGPIndividual]]:
    variant_root = root / variant.value
    patterns = (
        "*/seed-*/selected_candidate_2opt.pkl",
        "*/seed-*/selected_candidate.pkl",
        "*/seed-*/checkpoints/candidate_*.pkl",
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in variant_root.glob(pattern)
        }
    )
    unique: dict[str, tuple[Path, RMTGPIndividual]] = {}
    for path in paths:
        try:
            candidate = load_champion(path)
            if is_baseline_individual(candidate):
                continue
            compile_individual(candidate)
        except (OSError, TypeError, ValueError):
            continue
        unique.setdefault(candidate.structural_hash, (path, candidate))
    return list(unique.values())


def _clean_clone(candidate: RMTGPIndividual) -> RMTGPIndividual:
    """只复制两棵树，丢弃可能使用旧 dataclass schema 的历史 metadata。"""

    return RMTGPIndividual(
        gp.PrimitiveTree(candidate.transition_tree),
        gp.PrimitiveTree(candidate.pheromone_tree),
    )


def _make_program_bundle(
    *,
    experiment,
    variant: ACOVariant,
    checkpoint_root: Path,
    root_seed: int,
    random_count: int,
    old_count: int,
    mutation_count: int,
) -> tuple[list[RMTGPIndividual], list[dict[str, Any]]]:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [root_seed, _variant_code(variant), 0x50524F47]
        )
    )
    python_state = random.getstate()
    random.seed(
        int(
            rng.integers(
                0,
                2**63 - 1,
            )
        )
    )
    try:
        gp_config = replace(
            experiment.gp,
            baseline_anchor=False,
            population_size=max(
                experiment.gp.elite_size + 1,
                random_count + 1,
            ),
        )
        transition_pset, pheromone_pset = create_primitive_sets(
            transition_profile=gp_config.transition_profile,
            function_profile=gp_config.function_profile,
            transition_terminals=gp_config.transition_terminals,
            pheromone_terminals=gp_config.pheromone_terminals,
        )
        individuals: list[RMTGPIndividual] = []
        metadata: list[dict[str, Any]] = []
        hashes: set[str] = set()
        modes = ("transition", "pheromone", "joint", "joint")
        attempts = 0
        while len(individuals) < random_count:
            mode = modes[len(individuals) % len(modes)]
            candidate = make_individual(
                transition_pset,
                pheromone_pset,
                gp_config,
                mode=mode,
            )
            attempts += 1
            if candidate.structural_hash in hashes:
                if attempts > 100_000:
                    raise RuntimeError("无法生成足够多的唯一随机 GP 个体")
                continue
            compile_individual(candidate)
            hashes.add(candidate.structural_hash)
            individuals.append(candidate)
            metadata.append(
                {
                    "category": "random",
                    "source": f"fresh-{mode}",
                }
            )

        old_pool = _discover_old_candidates(checkpoint_root, variant)
        if len(old_pool) < old_count:
            raise FileNotFoundError(
                f"{variant.value} 只有 {len(old_pool)} 个唯一旧个体，"
                f"审计要求 {old_count} 个"
            )
        selection = rng.permutation(len(old_pool))[:old_count]
        selected_old: list[RMTGPIndividual] = []
        for pool_index in selection:
            source, candidate = old_pool[int(pool_index)]
            clone = _clean_clone(candidate)
            if clone.structural_hash in hashes:
                continue
            hashes.add(clone.structural_hash)
            selected_old.append(clone)
            individuals.append(clone)
            metadata.append(
                {
                    "category": "old",
                    "source": source.relative_to(ROOT).as_posix()
                    if source.is_relative_to(ROOT)
                    else source.as_posix(),
                }
            )
        # 极少数 old 与随机树碰撞时，从剩余池补齐。
        if len(selected_old) < old_count:
            for source, candidate in old_pool:
                if len(selected_old) >= old_count:
                    break
                clone = _clean_clone(candidate)
                if clone.structural_hash in hashes:
                    continue
                hashes.add(clone.structural_hash)
                selected_old.append(clone)
                individuals.append(clone)
                metadata.append(
                    {
                        "category": "old",
                        "source": source.as_posix(),
                    }
                )

        mutation_sources = selected_old
        if not mutation_sources and mutation_count:
            raise RuntimeError("没有可用于小变异的旧 GP 个体")
        source_index = 0
        mutation_attempts = 0
        created_mutations = 0
        while created_mutations < mutation_count:
            source = mutation_sources[source_index % len(mutation_sources)]
            source_index += 1
            mutant = _clean_clone(source)
            (mutant,) = mutate_role_preserving(
                mutant,
                transition_pset,
                pheromone_pset,
                gp_config,
            )
            mutation_attempts += 1
            if (
                mutant.structural_hash == source.structural_hash
                or mutant.structural_hash in hashes
            ):
                if mutation_attempts > 100_000:
                    raise RuntimeError("无法生成足够多的唯一 checkpoint 小变异")
                continue
            compile_individual(mutant)
            hashes.add(mutant.structural_hash)
            individuals.append(mutant)
            metadata.append(
                {
                    "category": "mutation",
                    "source": source.structural_hash,
                }
            )
            created_mutations += 1
    finally:
        random.setstate(python_state)

    expected = random_count + old_count + mutation_count
    if len(individuals) != expected:
        raise RuntimeError(
            f"program bundle 数量错误：expected={expected}, actual={len(individuals)}"
        )
    for index, (candidate, item) in enumerate(
        zip(individuals, metadata, strict=True),
        start=1,
    ):
        item.update(
            {
                "program_index": index,
                "structural_hash": candidate.structural_hash,
                "transition_expression": str(candidate.transition_tree),
                "pheromone_expression": str(candidate.pheromone_tree),
                "transition_nodes": candidate.transition_nodes,
                "pheromone_nodes": candidate.pheromone_nodes,
                "total_nodes": candidate.total_nodes,
            }
        )
    return individuals, metadata


def _program_bundle(
    output: Path,
    *,
    experiment,
    variant: ACOVariant,
    checkpoint_root: Path,
    root_seed: int,
    random_count: int,
    old_count: int,
    mutation_count: int,
) -> tuple[list[RMTGPIndividual], list[dict[str, Any]]]:
    artifact = output / "programs.pkl"
    if artifact.is_file():
        with artifact.open("rb") as handle:
            payload = pickle.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("programs.pkl schema 不兼容")
        if (
            int(payload.get("random_count", -1)) != random_count
            or int(payload.get("old_count", -1)) != old_count
            or int(payload.get("mutation_count", -1)) != mutation_count
        ):
            raise ValueError("已有 programs.pkl 与本次 category 数量参数不一致")
        individuals = payload["individuals"]
        metadata = payload["metadata"]
        if len(individuals) != random_count + old_count + mutation_count:
            raise ValueError("已有 programs.pkl 与本次 program 数量参数不一致")
        return individuals, metadata
    individuals, metadata = _make_program_bundle(
        experiment=experiment,
        variant=variant,
        checkpoint_root=checkpoint_root,
        root_seed=root_seed,
        random_count=random_count,
        old_count=old_count,
        mutation_count=mutation_count,
    )
    _atomic_pickle(
        artifact,
        {
            "schema_version": SCHEMA_VERSION,
            "random_count": random_count,
            "old_count": old_count,
            "mutation_count": mutation_count,
            "individuals": individuals,
            "metadata": metadata,
        },
    )
    _atomic_json(
        output / "programs.json",
        {
            "schema_version": SCHEMA_VERSION,
            "baseline": {
                "program_index": 0,
                "category": "baseline",
                "structural_hash": "aco-baseline-passthrough",
            },
            "programs": metadata,
        },
    )
    return individuals, metadata


def _aco_seed(
    root_seed: int,
    variant: ACOVariant,
    horizon: int,
    replicate: int,
) -> int:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                root_seed,
                _variant_code(variant),
                horizon,
                replicate,
                0x41434F,
            ]
        )
    )
    return int(rng.integers(0, 2**63 - 1))


def _select_instances(
    *,
    output: Path,
    spec,
    experiment,
    variant: ACOVariant,
    screening_count: int,
    easy_count: int,
    hard_count: int,
    screening_horizon: int,
    root_seed: int,
    basin_top_q: int,
) -> tuple[Any, list[dict[str, Any]]]:
    artifact = output / "instances.json"
    pool = pools_from_paths(spec.data.validation_paths())[100]
    if artifact.is_file():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        logical_indices = [int(item["logical_index"]) for item in payload["selected"]]
        records = [pool.get(index) for index in logical_indices]
        batch = make_problem_batch(
            records,
            candidate_size=experiment.aco.candidate_size,
            dtype=experiment.aco.dtype,
            device=experiment.aco.device,
        )
        if list(batch.coordinate_hashes) != [
            str(item["coordinate_hash"]) for item in payload["selected"]
        ]:
            raise ValueError("validation 数据与已有 instances.json 不一致")
        return batch, list(payload["selected"])

    if screening_count < easy_count + hard_count:
        raise ValueError("screening_count 不得小于 easy_count+hard_count")
    if screening_count > len(pool):
        raise ValueError("screening_count 超过 validation pool 大小")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [root_seed, _variant_code(variant), 0x494E5354]
        )
    )
    logical_indices = rng.choice(
        len(pool),
        size=screening_count,
        replace=False,
    )
    records = [pool.get(int(index)) for index in logical_indices]
    screening_batch = make_problem_batch(
        records,
        candidate_size=experiment.aco.candidate_size,
        dtype=experiment.aco.dtype,
        device=experiment.aco.device,
    )
    screening_config = replace(
        experiment.aco,
        iterations=screening_horizon,
    )
    screening_seed = _aco_seed(
        root_seed,
        variant,
        screening_horizon,
        2**31 - 1,
    )
    result = solve_population_cuda(
        screening_batch,
        screening_config,
        [(None, None)],
        seed=screening_seed,
        runtime=experiment.runtime,
        basin_top_q=basin_top_q,
    )
    gaps = (
        100.0
        * (
            result.best_length[0].numpy()
            - screening_batch.reference_length.numpy()
        )
        / screening_batch.reference_length.numpy()
    )
    order = np.argsort(gaps, kind="stable")
    chosen_positions = np.concatenate(
        (order[:easy_count], order[-hard_count:])
    )
    selected: list[dict[str, Any]] = []
    selected_records = []
    for rank, position in enumerate(chosen_positions):
        difficulty = "easy" if rank < easy_count else "hard"
        record = records[int(position)]
        selected_records.append(record)
        selected.append(
            {
                "selected_position": rank,
                "difficulty": difficulty,
                "logical_index": int(logical_indices[int(position)]),
                "instance_id": record.instance_id,
                "coordinate_hash": record.coordinate_hash,
                "screening_final_gap_percent": float(gaps[int(position)]),
            }
        )
    _atomic_json(
        artifact,
        {
            "schema_version": SCHEMA_VERSION,
            "variant": variant.value,
            "screening_count": screening_count,
            "screening_horizon": screening_horizon,
            "screening_seed": screening_seed,
            "selected": selected,
        },
    )
    batch = make_problem_batch(
        selected_records,
        candidate_size=experiment.aco.candidate_size,
        dtype=experiment.aco.dtype,
        device=experiment.aco.device,
    )
    return batch, selected


def _shard_path(output: Path, horizon: int, replicate: int) -> Path:
    return output / "shards" / f"h{horizon:05d}-seed-{replicate:02d}.npz"


def _valid_shard(
    path: Path,
    *,
    horizon: int,
    replicate: int,
    program_hashes: list[str],
    instance_hashes: list[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                int(payload["schema_version"]) == SCHEMA_VERSION
                and int(payload["horizon"]) == horizon
                and int(payload["replicate"]) == replicate
                and payload["program_hashes"].tolist() == program_hashes
                and payload["instance_hashes"].tolist() == instance_hashes
                and payload["final_gap_percent"].shape
                == (len(program_hashes), len(instance_hashes))
            )
    except (OSError, KeyError, ValueError):
        return False


def _run_shard(
    path: Path,
    *,
    batch,
    programs,
    program_hashes: list[str],
    instance_hashes: list[str],
    experiment,
    variant: ACOVariant,
    root_seed: int,
    horizon: int,
    replicate: int,
    basin_top_q: int,
) -> None:
    seed = _aco_seed(root_seed, variant, horizon, replicate)
    config = replace(experiment.aco, iterations=horizon)
    started = perf_counter()
    result = solve_population_cuda(
        batch,
        config,
        programs,
        seed=seed,
        runtime=experiment.runtime,
        basin_top_q=basin_top_q,
        audit_local_search=True,
    )
    if (
        result.basin_mean_length is None
        or result.pre_basin_mean_length is None
        or result.edge_retention is None
        or result.final_colony_tour is None
        or result.final_pre_colony_tour is None
    ):
        raise RuntimeError("CUDA audit 未返回完整 pre/post colony 统计")
    reference = batch.reference_length.numpy()[None, :]
    final_gap = 100.0 * (result.best_length.numpy() - reference) / reference
    post_basin_gap = (
        100.0 * (result.basin_mean_length.numpy() - reference) / reference
    )
    pre_basin_gap = (
        100.0
        * (result.pre_basin_mean_length.numpy() - reference)
        / reference
    )
    difference = edge_difference_survival(
        result.final_pre_colony_tour[1:],
        result.final_colony_tour[1:],
        result.final_pre_colony_tour[0],
        result.final_colony_tour[0],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        variant=np.asarray(variant.value),
        horizon=np.asarray(horizon, dtype=np.int32),
        replicate=np.asarray(replicate, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int64),
        program_hashes=np.asarray(program_hashes),
        instance_hashes=np.asarray(instance_hashes),
        final_gap_percent=final_gap,
        pre_basin_gap_percent=pre_basin_gap,
        post_basin_gap_percent=post_basin_gap,
        edge_retention=result.edge_retention.numpy(),
        difference_survival=difference.survival,
        pre_difference_edges=difference.pre_difference_edges,
        post_difference_edges=difference.post_difference_edges,
        survived_difference_edges=difference.survived_difference_edges,
        best_iteration=result.best_iteration.numpy(),
        diagnostics=result.diagnostics.numpy(),
        wall_time_sec=np.asarray(perf_counter() - started),
        kernel_time_sec=np.asarray(
            float(result.backend_metrics["kernel_seconds_critical"])
        ),
        constructed_tours=np.asarray(result.constructed_tours, dtype=np.int64),
    )
    os.replace(temporary, path)


def _load_arrays(
    output: Path,
    horizons: tuple[int, ...],
    seeds: int,
) -> dict[str, np.ndarray]:
    keys = (
        "final_gap_percent",
        "pre_basin_gap_percent",
        "post_basin_gap_percent",
        "edge_retention",
        "difference_survival",
        "pre_difference_edges",
        "post_difference_edges",
        "survived_difference_edges",
        "best_iteration",
        "diagnostics",
        "wall_time_sec",
        "kernel_time_sec",
        "constructed_tours",
    )
    by_key: dict[str, list[list[np.ndarray]]] = {
        key: [] for key in keys
    }
    for horizon in horizons:
        horizon_values = {key: [] for key in keys}
        for replicate in range(seeds):
            with np.load(
                _shard_path(output, horizon, replicate),
                allow_pickle=False,
            ) as payload:
                for key in keys:
                    horizon_values[key].append(np.asarray(payload[key]))
        for key in keys:
            by_key[key].append(horizon_values[key])
    return {
        key: np.asarray(values)
        for key, values in by_key.items()
    }


def _summarize(
    *,
    output: Path,
    variant: ACOVariant,
    horizons: tuple[int, ...],
    arrays: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
    selected_instances: list[dict[str, Any]],
    basin_top_q: int,
    ants: int,
) -> dict[str, Any]:
    final_gap = arrays["final_gap_percent"]
    pre_basin = arrays["pre_basin_gap_percent"]
    post_basin = arrays["post_basin_gap_percent"]
    retention = arrays["edge_retention"]
    difference = arrays["difference_survival"]
    pre_difference = arrays["pre_difference_edges"]
    diagnostics = arrays["diagnostics"]
    categories = sorted({str(item["category"]) for item in metadata})
    program_categories = np.asarray(
        [str(item["category"]) for item in metadata]
    )
    rows: list[dict[str, Any]] = []
    horizon_summaries: list[dict[str, Any]] = []

    for horizon_index, horizon in enumerate(horizons):
        baseline_final = final_gap[horizon_index, :, 0, :]
        candidate_final = final_gap[horizon_index, :, 1:, :]
        baseline_post = post_basin[horizon_index, :, 0, :]
        candidate_post = post_basin[horizon_index, :, 1:, :]
        candidate_pre = pre_basin[horizon_index, :, 1:, :]
        final_delta = candidate_final - baseline_final[:, None, :]
        post_delta = candidate_post - baseline_post[:, None, :]
        compression = compression_statistics(
            np.moveaxis(candidate_pre, 1, 0),
            np.moveaxis(candidate_post, 1, 0),
        )
        spearman = spearman_by_context(
            np.moveaxis(candidate_pre, 1, 0),
            np.moveaxis(candidate_post, 1, 0),
        )
        final_snr = signal_to_noise_statistics(
            np.moveaxis(final_delta, 1, 0)
        )
        basin_snr = signal_to_noise_statistics(
            np.moveaxis(post_delta, 1, 0)
        )
        candidate_retention = retention[horizon_index, :, 1:, :]
        candidate_difference = difference[horizon_index]
        candidate_pre_difference = pre_difference[horizon_index]
        candidate_diagnostics = diagnostics[horizon_index, :, 1:, :]
        baseline_diagnostics = diagnostics[horizon_index, :, 0, :]
        tours_per_program = (
            len(selected_instances) * ants * horizon
        )
        clip_events_per_1000_tours = (
            1000.0
            * candidate_diagnostics[..., 2]
            / float(tours_per_program)
        )
        introduced_fraction = 1.0 - candidate_retention
        behavior_changed = candidate_pre_difference > 0
        origin_gate = bool(
            np.nanmean(introduced_fraction) >= 0.05
            and np.mean(behavior_changed) >= 0.20
            and np.nanmean(candidate_difference) < 0.80
        )

        horizon_payload = {
            "horizon": horizon,
            "baseline_headroom": {
                "final_gap_percent": finite_summary(baseline_final),
                "optimum_hit_rate": float(
                    np.mean(np.abs(baseline_final) <= 1e-8)
                ),
                "mean_seed_standard_deviation": float(
                    np.mean(np.std(baseline_final, axis=0, ddof=1))
                ),
            },
            "compression": {
                "ratio_of_mean_variances": (
                    compression.ratio_of_mean_variances
                ),
                "ratios_by_context": finite_summary(
                    compression.ratios_by_context
                ),
                "spearman_pre_post": finite_summary(spearman),
            },
            "signal_to_noise": {
                "final": {
                    "ratio_of_mean_variances": (
                        final_snr.ratio_of_mean_variances
                    ),
                    "by_instance": finite_summary(
                        final_snr.ratios_by_instance
                    ),
                },
                "post_basin_auc": {
                    "ratio_of_mean_variances": (
                        basin_snr.ratio_of_mean_variances
                    ),
                    "by_instance": finite_summary(
                        basin_snr.ratios_by_instance
                    ),
                },
            },
            "edge_credit": {
                "retention": finite_summary(candidate_retention),
                "introduced_fraction": finite_summary(introduced_fraction),
                "difference_survival": finite_summary(candidate_difference),
                "construction_behavior_changed_fraction": float(
                    np.mean(behavior_changed)
                ),
            },
            "kernel_diagnostics": {
                "baseline_bound_clip_events": finite_summary(
                    baseline_diagnostics[..., 2]
                ),
                "candidate_bound_clip_events_per_1000_tours": finite_summary(
                    clip_events_per_1000_tours
                ),
                "candidate_mmas_restarts": finite_summary(
                    candidate_diagnostics[..., 3]
                ),
                "candidate_local_search_moves": finite_summary(
                    candidate_diagnostics[..., 4]
                ),
            },
            "origin_gate": {
                "enabled": origin_gate,
                "rule": (
                    "mean introduced_fraction>=0.05 AND "
                    "construction_behavior_changed_fraction>=0.20 AND "
                    "mean difference_survival<0.80"
                ),
            },
            "categories": {},
        }
        for category in categories:
            mask = program_categories == category
            category_payload = {
                "program_count": int(np.sum(mask)),
                "final_delta_pp": finite_summary(
                    final_delta[:, mask, :]
                ),
                "post_basin_delta_pp": finite_summary(
                    post_delta[:, mask, :]
                ),
                "pre_basin_gap_percent": finite_summary(
                    candidate_pre[:, mask, :]
                ),
                "post_basin_gap_percent": finite_summary(
                    candidate_post[:, mask, :]
                ),
                "edge_retention": finite_summary(
                    candidate_retention[:, mask, :]
                ),
                "difference_survival": finite_summary(
                    candidate_difference[:, mask, :, :]
                ),
                "bound_clip_events_per_1000_tours": finite_summary(
                    clip_events_per_1000_tours[:, mask]
                ),
                "mmas_restarts": finite_summary(
                    candidate_diagnostics[:, mask, 3]
                ),
            }
            horizon_payload["categories"][category] = category_payload
            rows.append(
                {
                    "variant": variant.value,
                    "horizon": horizon,
                    "category": category,
                    "program_count": category_payload["program_count"],
                    "mean_final_delta_pp": category_payload[
                        "final_delta_pp"
                    ]["mean"],
                    "mean_post_basin_delta_pp": category_payload[
                        "post_basin_delta_pp"
                    ]["mean"],
                    "mean_edge_retention": category_payload[
                        "edge_retention"
                    ]["mean"],
                    "mean_difference_survival": category_payload[
                        "difference_survival"
                    ]["mean"],
                    "compression_ratio": (
                        compression.ratio_of_mean_variances
                    ),
                    "spearman": finite_summary(spearman)["mean"],
                    "final_snr": final_snr.ratio_of_mean_variances,
                    "basin_snr": basin_snr.ratio_of_mean_variances,
                    "origin_gate": origin_gate,
                    "bound_clip_events_per_1000_tours": category_payload[
                        "bound_clip_events_per_1000_tours"
                    ]["mean"],
                }
            )
        horizon_summaries.append(horizon_payload)

    _write_csv(output / "audit_summary.csv", rows)
    aggregate = output / "audit_arrays.npz"
    temporary = aggregate.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        horizons=np.asarray(horizons, dtype=np.int32),
        program_categories=program_categories,
        instance_difficulty=np.asarray(
            [item["difficulty"] for item in selected_instances]
        ),
        **arrays,
    )
    os.replace(temporary, aggregate)
    return {
        "schema_version": SCHEMA_VERSION,
        "variant": variant.value,
        "basin_top_q": basin_top_q,
        "ants": ants,
        "horizons": list(horizons),
        "program_count_excluding_baseline": len(metadata),
        "instances": selected_instances,
        "horizon_summaries": horizon_summaries,
        "artifacts": {
            "raw_arrays": aggregate.name,
            "long_summary": "audit_summary.csv",
            "programs": "programs.json",
            "instances": "instances.json",
        },
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    variant = ACOVariant(args.variant)
    spec, experiment = _audit_experiment(args.config, variant)
    configure_runtime(experiment.runtime)
    args.output.mkdir(parents=True, exist_ok=True)
    individuals, metadata = _program_bundle(
        args.output,
        experiment=experiment,
        variant=variant,
        checkpoint_root=args.checkpoint_root,
        root_seed=args.root_seed,
        random_count=args.random_programs,
        old_count=args.old_programs,
        mutation_count=args.mutation_programs,
    )
    compiled = [(None, None), *[compile_individual(item) for item in individuals]]
    program_hashes = [
        "aco-baseline-passthrough",
        *[item.structural_hash for item in individuals],
    ]
    batch, selected_instances = _select_instances(
        output=args.output,
        spec=spec,
        experiment=experiment,
        variant=variant,
        screening_count=args.screening_instances,
        easy_count=args.easy_instances,
        hard_count=args.hard_instances,
        screening_horizon=args.screening_horizon,
        root_seed=args.root_seed,
        basin_top_q=args.basin_top_q,
    )
    instance_hashes = list(batch.coordinate_hashes)
    total = len(args.horizons) * args.seeds
    completed = 0
    for horizon in args.horizons:
        for replicate in range(args.seeds):
            path = _shard_path(args.output, horizon, replicate)
            if _valid_shard(
                path,
                horizon=horizon,
                replicate=replicate,
                program_hashes=program_hashes,
                instance_hashes=instance_hashes,
            ):
                completed += 1
                print(
                    f"[{variant.value}] reuse {path.name} ({completed}/{total})",
                    flush=True,
                )
                continue
            print(
                f"[{variant.value}] run horizon={horizon} "
                f"seed={replicate + 1}/{args.seeds}",
                flush=True,
            )
            _run_shard(
                path,
                batch=batch,
                programs=compiled,
                program_hashes=program_hashes,
                instance_hashes=instance_hashes,
                experiment=experiment,
                variant=variant,
                root_seed=args.root_seed,
                horizon=horizon,
                replicate=replicate,
                basin_top_q=args.basin_top_q,
            )
            completed += 1
            print(
                f"[{variant.value}] completed {path.name} ({completed}/{total})",
                flush=True,
            )
    arrays = _load_arrays(args.output, args.horizons, args.seeds)
    return _summarize(
        output=args.output,
        variant=variant,
        horizons=args.horizons,
        arrays=arrays,
        metadata=metadata,
        selected_instances=selected_instances,
        basin_top_q=args.basin_top_q,
        ants=experiment.aco.resolve_ants(100),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--variant",
        choices=[item.value for item in ACOVariant],
        required=True,
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root-seed", type=int, default=73021)
    parser.add_argument("--random-programs", type=int, default=32)
    parser.add_argument("--old-programs", type=int, default=16)
    parser.add_argument("--mutation-programs", type=int, default=16)
    parser.add_argument("--screening-instances", type=int, default=64)
    parser.add_argument("--easy-instances", type=int, default=8)
    parser.add_argument("--hard-instances", type=int, default=8)
    parser.add_argument("--screening-horizon", type=int, default=500)
    parser.add_argument("--horizons", type=int, nargs="+", default=(500, 2000, 5000))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--basin-top-q", type=int, default=7)
    args = parser.parse_args()
    args.horizons = tuple(int(value) for value in args.horizons)
    if args.seeds < 2:
        parser.error("--seeds 至少为 2，才能估计 SNR")
    if (
        min(
            args.random_programs,
            args.old_programs,
            args.mutation_programs,
            args.easy_instances,
            args.hard_instances,
            args.screening_horizon,
            *args.horizons,
        )
        < 1
    ):
        parser.error("program/instance/horizon 参数必须为正整数")
    if not 1 <= args.basin_top_q <= 32:
        parser.error("--basin-top-q 必须位于 [1,32]")
    variant = ACOVariant(args.variant)
    args.output = (
        args.output
        if args.output is not None
        else DEFAULT_OUTPUT_ROOT / variant.value
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "variant": variant.value,
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
        "started_at": datetime.now(UTC).isoformat(),
        "ended_at": None,
        "error": None,
    }
    _atomic_json(args.output / "manifest.json", manifest)
    try:
        summary = _execute(args)
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
    _atomic_json(args.output / "summary.json", summary)
    manifest.update(
        {
            "status": "completed",
            "ended_at": datetime.now(UTC).isoformat(),
            "summary": "summary.json",
        }
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
