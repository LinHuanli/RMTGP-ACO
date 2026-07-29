"""GP 表达能力预算（31/62 节点）敏感性实验与可恢复双 GPU 队列。

该模块刻意与主消融输出隔离。31 节点结果只读复用已完成的消融，
62 节点版本对 transition 单树、pheromone 单树和双树分别重新训练，
从而同时估计容量效应、同容量结构效应及二者交互。
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from queue import Empty, Queue
from statistics import median
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any

import yaml

from .ablation import (
    AblationStudySpec,
    AblationTask,
    ProgramEntry,
    _atomic_json,
    _baseline_result,
    _expected_batches,
    _expected_instances,
    _physical_gpu_description,
    _run_result_from_population,
    _test_shard_valid,
    build_ablation_tasks,
    load_ablation_spec,
    method_run_path,
)
from .aco import solve
from .aco_cuda import solve_population_cuda_anytime
from .artifacts import git_state
from .baseline import backend_semantic_id
from .evaluation import (
    EvaluationRecord,
    compile_champion,
    load_champion,
    read_records,
    records_from_paired_results,
    study_test_seed,
    write_records,
)
from .runtime import configure_runtime
from .sampling import iter_problem_batches
from .spec import RunSpec, load_run_spec
from .study import _external_gpu_processes, load_study_spec

CAPACITY_STATE_SCHEMA_VERSION = 1
CAPACITY_TEST_SCHEMA_VERSION = 1
SOURCE_CAPACITY = 31
TARGET_CAPACITY = 62

CAPACITY_METHODS: tuple[str, ...] = (
    "tr-rgp",
    "ph-rgp",
    "rmtgp-full-f1",
)
SINGLE_TREE_METHODS: tuple[str, ...] = ("tr-rgp", "ph-rgp")


def capacity_method(method: str, nodes: int) -> str:
    """返回写入长表的稳定 method id。"""

    if method not in CAPACITY_METHODS:
        raise KeyError(f"容量实验未知 method: {method}")
    if nodes not in {SOURCE_CAPACITY, TARGET_CAPACITY}:
        raise ValueError("容量标签只支持 31 或 62")
    return f"{method}-n{nodes}"


@dataclass(frozen=True, slots=True)
class CapacityVariant:
    """一个 ACO 变体的单树/双树 62 节点配置及三个配对种子。"""

    name: str
    single_config: Path
    multi_config: Path
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CapacitySource:
    """31 节点主消融的只读来源。"""

    config: Path
    root: Path
    study: AblationStudySpec


@dataclass(frozen=True, slots=True)
class CapacityStudySpec:
    """结构 × 节点容量敏感性实验的冻结合同。"""

    study_id: str
    output_root: Path
    phase: str
    test_root_seed: int
    test_seeds: int
    partitions: tuple[str, ...]
    variants: tuple[CapacityVariant, ...]
    source: CapacitySource
    manifest: Path
    bootstrap_replicates: int
    source_path: Path


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _variant(study: CapacityStudySpec, name: str) -> CapacityVariant:
    for variant in study.variants:
        if variant.name == name:
            return variant
    raise KeyError(f"容量 study 中不存在 ACO variant: {name}")


def _source_variant(study: CapacityStudySpec, name: str) -> Any:
    for variant in study.source.study.variants:
        if variant.name == name:
            return variant
    raise KeyError(f"源消融中不存在 ACO variant: {name}")


def _config_for_method(variant: CapacityVariant, method: str) -> Path:
    if method in SINGLE_TREE_METHODS:
        return variant.single_config
    if method == "rmtgp-full-f1":
        return variant.multi_config
    raise KeyError(f"容量实验未知 method: {method}")


def _without_capacity(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("max_nodes_per_tree", None)
    result.pop("max_total_nodes", None)
    return result


def _validate_capacity_config(
    *,
    path: Path,
    source_path: Path,
    architecture: str,
) -> None:
    """核对除节点预算与 experiment id 外的全部训练合同均保持不变。"""

    spec = load_run_spec(path)
    source = load_run_spec(source_path)
    if spec.experiment.aco.stable_dict() != source.experiment.aco.stable_dict():
        raise ValueError(f"{path}: ACO 参数与 31 节点源配置不一致")
    if asdict(spec.experiment.runtime) != asdict(source.experiment.runtime):
        raise ValueError(f"{path}: runtime 与 31 节点源配置不一致")
    if asdict(spec.data) != asdict(source.data):
        raise ValueError(f"{path}: data 合同与 31 节点源配置不一致")

    comparable_experiment_fields = (
        "train_scales",
        "validation_scales",
        "test_scales",
        "validation_seeds",
        "validation_screening_seeds",
        "validation_top_k",
        "noninferiority_tolerance",
    )
    for field in comparable_experiment_fields:
        if getattr(spec.experiment, field) != getattr(source.experiment, field):
            raise ValueError(f"{path}: experiment.{field} 与源配置不一致")
    if _without_capacity(asdict(spec.experiment.gp)) != _without_capacity(
        asdict(source.experiment.gp)
    ):
        raise ValueError(f"{path}: GP 除节点预算外与源配置不一致")

    gp = spec.experiment.gp
    if gp.max_total_nodes != TARGET_CAPACITY or gp.max_depth != 5:
        raise ValueError(f"{path}: 必须 max_total_nodes=62 且 max_depth=5")
    expected_per_tree = TARGET_CAPACITY if architecture == "single" else 31
    if gp.max_nodes_per_tree != expected_per_tree:
        raise ValueError(
            f"{path}: {architecture} 配置 max_nodes_per_tree 必须为 {expected_per_tree}"
        )


def load_capacity_spec(path: str | Path) -> CapacityStudySpec:
    """严格读取容量敏感性 YAML；源实验可尚在运行，但合同必须存在。"""

    source_path = Path(path).resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capacity 配置根节点必须为 mapping")
    allowed = {
        "study_id",
        "output_root",
        "phase",
        "test_root_seed",
        "test_seeds",
        "partitions",
        "variants",
        "source",
        "manifest",
        "bootstrap_replicates",
    }
    unknown = set(payload) - allowed
    missing = allowed - {"phase", "manifest", "bootstrap_replicates"} - set(payload)
    if unknown:
        raise ValueError(f"capacity 配置含未知字段: {sorted(unknown)}")
    if missing:
        raise ValueError(f"capacity 配置缺少字段: {sorted(missing)}")

    source_raw = payload["source"]
    if not isinstance(source_raw, dict) or set(source_raw) != {"config", "root"}:
        raise ValueError("source 仅允许 config 与 root")
    ablation = load_ablation_spec(_project_path(str(source_raw["config"])))
    source_root = _project_path(str(source_raw["root"]))
    if ablation.output_root != source_root:
        raise ValueError("source.root 与源消融 output_root 不一致")
    source = CapacitySource(
        config=_project_path(str(source_raw["config"])),
        root=source_root,
        study=ablation,
    )

    variants_raw = payload["variants"]
    if not isinstance(variants_raw, dict) or set(variants_raw) != {
        item.name for item in ablation.variants
    }:
        raise ValueError("capacity variants 必须与源消融完全一致")
    variants: list[CapacityVariant] = []
    for name, raw in variants_raw.items():
        if not isinstance(raw, dict) or set(raw) != {
            "single_config",
            "multi_config",
            "seeds",
        }:
            raise ValueError(f"variants.{name} 仅允许 single_config、multi_config 与 seeds")
        source_variant = next(item for item in ablation.variants if item.name == name)
        seeds = tuple(int(value) for value in raw["seeds"])
        if seeds != source_variant.seeds:
            raise ValueError(f"{name}: seeds 与源消融不一致")
        variant = CapacityVariant(
            name=str(name),
            single_config=_project_path(str(raw["single_config"])),
            multi_config=_project_path(str(raw["multi_config"])),
            seeds=seeds,
        )
        _validate_capacity_config(
            path=variant.single_config,
            source_path=source_variant.config,
            architecture="single",
        )
        _validate_capacity_config(
            path=variant.multi_config,
            source_path=source_variant.config,
            architecture="multi",
        )
        for config in (variant.single_config, variant.multi_config):
            spec = load_run_spec(config)
            if spec.experiment.aco.variant.value != name:
                raise ValueError(f"{config}: ACO variant 与 key={name} 不一致")
        variants.append(variant)

    partitions = tuple(str(value) for value in payload["partitions"])
    if partitions != ablation.ablation_partitions:
        raise ValueError(
            "capacity partitions 必须与源消融的核心 ablation_partitions "
            "顺序和内容完全一致"
        )
    test_root_seed = int(payload["test_root_seed"])
    test_seeds = int(payload["test_seeds"])
    if test_root_seed != ablation.test_root_seed or test_seeds != ablation.test_seeds:
        raise ValueError("capacity 测试随机合同必须与源消融一致")
    if test_seeds != 3:
        raise ValueError("容量 pilot 固定使用三个 ACO test seeds")
    phase = str(payload.get("phase", "pilot"))
    if phase != "pilot":
        raise ValueError("容量敏感性三种子实验必须为 pilot")
    bootstrap = int(payload.get("bootstrap_replicates", 10_000))
    if bootstrap < 100:
        raise ValueError("bootstrap_replicates 至少为 100")

    return CapacityStudySpec(
        study_id=str(payload["study_id"]),
        output_root=_project_path(str(payload["output_root"])),
        phase=phase,
        test_root_seed=test_root_seed,
        test_seeds=test_seeds,
        partitions=partitions,
        variants=tuple(variants),
        source=source,
        manifest=_project_path(str(payload.get("manifest", "Datasets/manifest.json"))),
        bootstrap_replicates=bootstrap,
        source_path=source_path,
    )


def source_ready(study: CapacityStudySpec) -> tuple[bool, str]:
    """返回 31 节点源消融是否完整可复用及原因。"""

    state_path = study.source.root / "study_state.json"
    if not state_path.is_file():
        return False, f"源状态不存在: {state_path}"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"源状态不可读: {exc}"
    status = str(state.get("status", "unknown"))
    if status == "failed":
        raise RuntimeError(f"源消融失败: {state.get('error')}")
    expected = len(build_ablation_tasks(study.source.study))
    if status != "completed":
        return False, (f"源消融 status={status}, {state.get('completed_count', 0)}/{expected}")
    if int(state.get("completed_count", -1)) != expected:
        return False, "源消融 completed_count 与冻结任务数不一致"
    report = study.source.root / "report" / "ablation_report.md"
    if not report.is_file():
        return False, f"源消融报告缺失: {report}"
    return True, "源消融已完成"


def capacity_run_path(
    study: CapacityStudySpec,
    method: str,
    variant: str,
    root_seed: int,
) -> Path:
    """返回 62 节点训练 run 目录。"""

    if method not in CAPACITY_METHODS:
        raise KeyError(f"容量实验未知 method: {method}")
    return (
        study.output_root
        / "train"
        / capacity_method(method, TARGET_CAPACITY)
        / variant
        / f"seed-{root_seed}"
    )


def source_run_path(
    study: CapacityStudySpec,
    method: str,
    variant: str,
    root_seed: int,
) -> Path:
    """返回源消融中严格配对的 31 节点 run。"""

    return method_run_path(
        study.source.study,
        method,
        variant,
        root_seed,
    )


def _load_capacity_entry(
    study: CapacityStudySpec,
    *,
    method: str,
    variant: str,
    root_seed: int,
) -> ProgramEntry:
    run = capacity_run_path(study, method, variant, root_seed)
    champion = load_champion(run / "selected_candidate.pkl")
    transition, pheromone = compile_champion(champion)
    return ProgramEntry(
        method=capacity_method(method, TARGET_CAPACITY),
        gp_root_seed=root_seed,
        champion_id=champion.structural_hash,
        transition=transition,
        pheromone=pheromone,
    )


def _load_source_entry(
    study: CapacityStudySpec,
    *,
    method: str,
    variant: str,
    root_seed: int,
) -> ProgramEntry:
    """编译一个源消融的 31 节点 champion，并显式重标容量。"""

    run = source_run_path(study, method, variant, root_seed)
    champion = load_champion(run / "selected_candidate.pkl")
    transition, pheromone = compile_champion(champion)
    return ProgramEntry(
        method=capacity_method(method, SOURCE_CAPACITY),
        gp_root_seed=root_seed,
        champion_id=champion.structural_hash,
        transition=transition,
        pheromone=pheromone,
    )


def _capacity_entries(
    study: CapacityStudySpec,
    *,
    variant_name: str,
) -> list[ProgramEntry]:
    variant = _variant(study, variant_name)
    return [
        _load_capacity_entry(
            study,
            method=method,
            variant=variant_name,
            root_seed=root_seed,
        )
        for method in CAPACITY_METHODS
        for root_seed in variant.seeds
    ]


def _expected_instances_capacity(
    study: CapacityStudySpec,
    spec: RunSpec,
    partition: str,
) -> int:
    # ablation helper 只读取 manifest/partitions 字段，容量合同保持相同接口。
    return _expected_instances(study, spec, partition)  # type: ignore[arg-type]


def _expected_batches_capacity(
    study: CapacityStudySpec,
    spec: RunSpec,
    partition: str,
) -> int:
    return _expected_batches(study, spec, partition)  # type: ignore[arg-type]


def evaluate_capacity_partition(
    study: CapacityStudySpec,
    *,
    variant_name: str,
    partition: str,
) -> Path:
    """以 program 矩阵一次评测一个 variant×partition 的全部 62 节点模型。"""

    ready, detail = source_ready(study)
    if not ready:
        raise RuntimeError(f"容量测试要求源消融完成: {detail}")
    variant = _variant(study, variant_name)
    if partition not in study.partitions:
        raise KeyError(f"partition 不在 capacity 合同中: {partition}")
    spec = load_run_spec(variant.multi_config)
    configure_runtime(spec.experiment.runtime)
    entries = _capacity_entries(study, variant_name=variant_name)
    config = spec.experiment.aco
    output = study.output_root / "test" / variant_name / partition
    merged = output / "records.csv"
    expected_instances = _expected_instances_capacity(study, spec, partition)
    expected_rows = expected_instances * study.test_seeds * len(entries)
    manifest_path = output / "evaluation_manifest.json"
    if merged.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = read_records([merged])
        except (OSError, TypeError, ValueError):
            manifest, records = {}, []
        if (
            manifest.get("status") == "completed"
            and manifest.get("rows") == expected_rows
            and len(records) == expected_rows
        ):
            return merged

    paths = spec.data.test_paths(partition)
    partition_spec = spec.data.test[partition]
    all_records: list[EvaluationRecord] = []
    campaigns: list[dict[str, Any]] = []
    completed_shards = 0
    total_shards = _expected_batches_capacity(study, spec, partition) * study.test_seeds
    started = perf_counter()
    for batch_number, batch in enumerate(
        iter_problem_batches(
            paths,
            batch_size=spec.data.evaluation_batch_size,
            candidate_size=config.candidate_size,
            dtype=config.dtype,
            device=config.device,
            min_scale=partition_spec.min_scale,
            max_scale=partition_spec.max_scale,
        )
    ):
        for replicate in range(study.test_seeds):
            seed = study_test_seed(
                study.test_root_seed,
                partition,
                batch_number,
                replicate,
            )
            shard = output / f"aco-{replicate:02d}" / f"batch-{batch_number:04d}.csv"
            metrics_path = shard.with_suffix(".metrics.json")
            if (
                _test_shard_valid(
                    shard,
                    entries=entries,
                    batch=batch,
                    partition=partition,
                    seed=seed,
                )
                and metrics_path.is_file()
            ):
                records = read_records([shard])
                campaign = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                baseline = _baseline_result(
                    study.source.study,
                    spec=spec,
                    variant_name=variant_name,
                    partition=partition,
                    batch_number=batch_number,
                    replicate=replicate,
                    seed=seed,
                    batch=batch,
                )
                population = solve_population_cuda_anytime(
                    batch,
                    config,
                    [(entry.transition, entry.pheromone) for entry in entries],
                    seed=seed,
                    runtime=spec.experiment.runtime,
                )
                records = []
                for index, entry in enumerate(entries):
                    candidate = _run_result_from_population(
                        population,
                        index,
                        config=config,
                        batch_size=batch.batch_size,
                    )
                    records.extend(
                        records_from_paired_results(
                            method=entry.method,
                            champion_id=entry.champion_id,
                            partition=partition,
                            distribution=partition_spec.distribution,
                            batch=batch,
                            seed=seed,
                            candidate=candidate,
                            baseline=baseline,
                            config=config,
                            gp_run_id=(f"{variant_name}:{entry.method}:seed-{entry.gp_root_seed}"),
                            gp_root_seed=entry.gp_root_seed,
                        )
                    )
                write_records(records, shard)
                campaign = {
                    "schema_version": 1,
                    "variant": variant_name,
                    "partition": partition,
                    "batch_number": batch_number,
                    "replicate": replicate,
                    "seed": seed,
                    "programs": len(entries),
                    "instances": batch.batch_size,
                    "scale": batch.n,
                    "campaign_wall_time_sec": population.wall_time_sec,
                    "constructed_tours": population.constructed_tours,
                    "backend_metrics": population.backend_metrics,
                }
                _atomic_json(metrics_path, campaign)
            all_records.extend(records)
            campaigns.append(campaign)
            completed_shards += 1
            elapsed = perf_counter() - started
            rate = completed_shards / max(elapsed, 1e-12)
            eta = (total_shards - completed_shards) / max(rate, 1e-12)
            print(
                f"capacity-test={variant_name}/{partition} "
                f"shards={completed_shards}/{total_shards} "
                f"elapsed={elapsed / 60.0:.1f}min eta={eta / 60.0:.1f}min",
                flush=True,
            )

    if len(all_records) != expected_rows:
        raise RuntimeError(
            f"{variant_name}/{partition}: rows={len(all_records)}，预期 {expected_rows}"
        )
    unique = {
        (
            record.method,
            record.gp_root_seed,
            record.instance_id,
            record.seed,
        )
        for record in all_records
    }
    if len(unique) != expected_rows:
        raise RuntimeError("capacity test records 存在重复配对键")
    all_records.sort(
        key=lambda record: (
            record.method,
            record.gp_root_seed,
            record.instance_id,
            record.seed,
        )
    )
    write_records(all_records, merged)
    _atomic_json(
        manifest_path,
        {
            "schema_version": CAPACITY_TEST_SCHEMA_VERSION,
            "status": "completed",
            "study_id": study.study_id,
            "variant": variant_name,
            "partition": partition,
            "methods": sorted({entry.method for entry in entries}),
            "programs": len(entries),
            "test_root_seed": study.test_root_seed,
            "test_seeds": study.test_seeds,
            "instances": expected_instances,
            "rows": expected_rows,
            "aco_config_hash": config.config_hash,
            "baseline_behavior_hash": config.baseline_behavior_hash,
            "backend_semantic": backend_semantic_id(
                spec.experiment.runtime.aco_backend,
                spec.experiment.runtime,
            ),
            "campaigns": campaigns,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return merged


def benchmark_capacity_efficiency(
    study: CapacityStudySpec,
    *,
    variant_name: str,
    output: str | Path,
) -> Path:
    """在相同预热、batch 与随机种子下孤立测量 31/62 节点 champions。"""

    ready, detail = source_ready(study)
    if not ready:
        raise RuntimeError(f"容量效率测试要求源消融完成: {detail}")
    variant = _variant(study, variant_name)
    spec = load_run_spec(variant.multi_config)
    configure_runtime(spec.experiment.runtime)
    target = Path(output)
    shard_root = target.parent / f"{variant_name}-shards"
    available_batch_sizes = {
        "tsp50_uniform": 32,
        "tsp100_uniform": 32,
        "tsp500_uniform": 8,
        "tsp1000_uniform": 4,
    }
    batch_sizes = {
        partition: available_batch_sizes[partition]
        for partition in study.partitions
    }
    all_rows: list[dict[str, Any]] = []
    for partition, batch_size in batch_sizes.items():
        batch = next(
            iter(
                iter_problem_batches(
                    spec.data.test_paths(partition),
                    batch_size=batch_size,
                    candidate_size=spec.experiment.aco.candidate_size,
                    dtype=spec.experiment.aco.dtype,
                    device=spec.experiment.aco.device,
                    max_instances=batch_size,
                )
            )
        )
        seeds = [
            study_test_seed(
                study.test_root_seed,
                f"capacity-efficiency:{partition}",
                0,
                replicate,
            )
            for replicate in range(3)
        ]
        solve(
            batch,
            spec.experiment.aco,
            seed=seeds[0],
            backend=spec.experiment.runtime.aco_backend,
            runtime=spec.experiment.runtime,
        )
        baseline_times: dict[int, float] = {}
        baseline_metrics: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            baseline = solve(
                batch,
                spec.experiment.aco,
                seed=seed,
                backend=spec.experiment.runtime.aco_backend,
                runtime=spec.experiment.runtime,
            )
            baseline_times[seed] = baseline.wall_time_sec
            baseline_metrics[seed] = dict(baseline.backend_metrics)

        for nodes in (SOURCE_CAPACITY, TARGET_CAPACITY):
            for method in CAPACITY_METHODS:
                label = capacity_method(method, nodes)
                shard = shard_root / label / f"{partition}.json"
                if shard.is_file():
                    try:
                        payload = json.loads(shard.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        payload = {}
                    if (
                        payload.get("status") == "completed"
                        and payload.get("method") == label
                        and payload.get("partition") == partition
                        and len(payload.get("rows", [])) == 9
                    ):
                        all_rows.extend(payload["rows"])
                        continue

                rows: list[dict[str, Any]] = []
                for root_seed in variant.seeds:
                    entry = (
                        _load_source_entry(
                            study,
                            method=method,
                            variant=variant_name,
                            root_seed=root_seed,
                        )
                        if nodes == SOURCE_CAPACITY
                        else _load_capacity_entry(
                            study,
                            method=method,
                            variant=variant_name,
                            root_seed=root_seed,
                        )
                    )
                    solve(
                        batch,
                        spec.experiment.aco,
                        transition_program=entry.transition,
                        pheromone_program=entry.pheromone,
                        seed=seeds[0],
                        backend=spec.experiment.runtime.aco_backend,
                        runtime=spec.experiment.runtime,
                    )
                    for replicate, seed in enumerate(seeds):
                        result = solve(
                            batch,
                            spec.experiment.aco,
                            transition_program=entry.transition,
                            pheromone_program=entry.pheromone,
                            seed=seed,
                            backend=spec.experiment.runtime.aco_backend,
                            runtime=spec.experiment.runtime,
                        )
                        baseline_time = baseline_times[seed]
                        rows.append(
                            {
                                "variant": variant_name,
                                "method": label,
                                "partition": partition,
                                "scale": batch.n,
                                "batch_size": batch.batch_size,
                                "gp_root_seed": root_seed,
                                "aco_replicate": replicate,
                                "aco_seed": seed,
                                "champion_id": entry.champion_id,
                                "wall_time_sec": result.wall_time_sec,
                                "baseline_wall_time_sec": baseline_time,
                                "overhead_percent": 100.0
                                * (result.wall_time_sec - baseline_time)
                                / max(baseline_time, 1e-12),
                                "tours_per_second": result.constructed_tours
                                / max(result.wall_time_sec, 1e-12),
                                "baseline_tours_per_second": (
                                    result.constructed_tours / max(baseline_time, 1e-12)
                                ),
                                "backend_metrics": result.backend_metrics,
                                "baseline_backend_metrics": baseline_metrics[seed],
                            }
                        )
                _atomic_json(
                    shard,
                    {
                        "schema_version": 1,
                        "status": "completed",
                        "variant": variant_name,
                        "method": label,
                        "partition": partition,
                        "rows": rows,
                    },
                )
                all_rows.extend(rows)
                print(
                    f"capacity-efficiency={variant_name}/{partition}/{label}",
                    flush=True,
                )

    summaries: list[dict[str, Any]] = []
    for nodes in (SOURCE_CAPACITY, TARGET_CAPACITY):
        for method in CAPACITY_METHODS:
            label = capacity_method(method, nodes)
            for partition in batch_sizes:
                rows = [
                    row
                    for row in all_rows
                    if row["method"] == label and row["partition"] == partition
                ]
                summaries.append(
                    {
                        "variant": variant_name,
                        "method": label,
                        "partition": partition,
                        "scale": rows[0]["scale"],
                        "observations": len(rows),
                        "median_wall_time_sec": median(row["wall_time_sec"] for row in rows),
                        "median_baseline_wall_time_sec": median(
                            row["baseline_wall_time_sec"] for row in rows
                        ),
                        "median_overhead_percent": median(row["overhead_percent"] for row in rows),
                        "median_tours_per_second": median(row["tours_per_second"] for row in rows),
                    }
                )
    _atomic_json(
        target,
        {
            "schema_version": 1,
            "status": "completed",
            "study_id": study.study_id,
            "variant": variant_name,
            "timing_scope": "isolated-warm-program-median",
            "repeats_per_champion": 3,
            "champions_per_method": 3,
            "batch_sizes": batch_sizes,
            "summaries": summaries,
            "rows": all_rows,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return target


def _training_artifact_valid(
    path: Path,
    *,
    method: str,
    root_seed: int,
) -> bool:
    required = (
        path / "selected_candidate.pkl",
        path / "deployment_decision.json",
        path / "training_validation_curve.csv",
        path / "cpu_fp64_audit_summary.csv",
    )
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or not all(item.is_file() for item in required):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    configuration = manifest.get("configuration", {})
    gp = configuration.get("gp", {})
    expected_per_tree = TARGET_CAPACITY if method in SINGLE_TREE_METHODS else 31
    return (
        manifest.get("status") == "completed"
        and int(configuration.get("root_seed", -1)) == root_seed
        and str(configuration.get("experiment_id", "")).endswith(f"-{method}")
        and int(gp.get("max_total_nodes", -1)) == TARGET_CAPACITY
        and int(gp.get("max_nodes_per_tree", -1)) == expected_per_tree
    )


def build_capacity_tasks(study: CapacityStudySpec) -> list[AblationTask]:
    """构造 9 预检、27 训练、21 测试、3 效率与 1 报告任务。"""

    python = sys.executable
    tasks: list[AblationTask] = []
    for variant in study.variants:
        root_seed = variant.seeds[0]
        source_variant = _source_variant(study, variant.name)
        schedule = (
            study.source.study.reuse.root / "schedules" / f"{variant.name}-seed-{root_seed}.json"
        )
        baseline = study.source.study.reuse.root / "baselines" / variant.name
        for method in CAPACITY_METHODS:
            artifact = (
                study.output_root
                / "preflight"
                / variant.name
                / f"{capacity_method(method, TARGET_CAPACITY)}.json"
            )
            tasks.append(
                AblationTask(
                    task_id=f"preflight-{variant.name}-{method}-n62",
                    command=(
                        python,
                        "-m",
                        "rmtgp_aco",
                        "benchmark-training",
                        "--config",
                        str(_config_for_method(variant, method)),
                        "--phase",
                        study.phase,
                        "--schedule",
                        str(schedule),
                        "--baseline-archive",
                        str(baseline),
                        "--method-profile",
                        method,
                        "--root-seed",
                        str(root_seed),
                        "--replicate-id",
                        "0",
                        "--generations",
                        "1",
                        "--output",
                        str(artifact),
                    ),
                    artifact=artifact,
                    kind="preflight",
                    metadata=(
                        ("variant", variant.name),
                        ("method", method),
                        ("source_config", str(source_variant.config)),
                    ),
                )
            )

    for replicate in range(3):
        for variant in study.variants:
            root_seed = variant.seeds[replicate]
            schedule = (
                study.source.study.reuse.root
                / "schedules"
                / f"{variant.name}-seed-{root_seed}.json"
            )
            baseline = study.source.study.reuse.root / "baselines" / variant.name
            for method in CAPACITY_METHODS:
                run = capacity_run_path(study, method, variant.name, root_seed)
                command = [
                    python,
                    "-m",
                    "rmtgp_aco",
                    "train",
                    "--config",
                    str(_config_for_method(variant, method)),
                    "--phase",
                    study.phase,
                    "--manifest",
                    str(study.manifest),
                    "--schedule",
                    str(schedule),
                    "--baseline-archive",
                    str(baseline),
                    "--method-profile",
                    method,
                    "--root-seed",
                    str(root_seed),
                    "--replicate-id",
                    str(replicate),
                    "--output",
                    str(run),
                    "--traceback",
                ]
                if (run / "training_state.pkl").is_file() and not (
                    _training_artifact_valid(
                        run,
                        method=method,
                        root_seed=root_seed,
                    )
                ):
                    command.extend(["--resume", str(run)])
                tasks.append(
                    AblationTask(
                        task_id=(f"train-{variant.name}-{method}-n62-{root_seed}"),
                        command=tuple(command),
                        artifact=run,
                        kind="train",
                        metadata=(
                            ("variant", variant.name),
                            ("method", method),
                            ("root_seed", str(root_seed)),
                        ),
                    )
                )

    for variant in study.variants:
        for partition in study.partitions:
            artifact = study.output_root / "test" / variant.name / partition / "records.csv"
            tasks.append(
                AblationTask(
                    task_id=f"test-{variant.name}-{partition}",
                    command=(
                        python,
                        "-m",
                        "rmtgp_aco",
                        "evaluate-capacity-study",
                        "--study-config",
                        str(study.source_path),
                        "--variant",
                        variant.name,
                        "--partition",
                        partition,
                    ),
                    artifact=artifact,
                    kind="test",
                    metadata=(
                        ("variant", variant.name),
                        ("partition", partition),
                    ),
                )
            )

    for variant in study.variants:
        artifact = study.output_root / "efficiency" / f"{variant.name}.json"
        tasks.append(
            AblationTask(
                task_id=f"efficiency-{variant.name}",
                command=(
                    python,
                    "-m",
                    "rmtgp_aco",
                    "benchmark-capacity-efficiency",
                    "--study-config",
                    str(study.source_path),
                    "--variant",
                    variant.name,
                    "--output",
                    str(artifact),
                ),
                artifact=artifact,
                kind="efficiency",
                metadata=(("variant", variant.name),),
            )
        )
    tasks.append(
        AblationTask(
            task_id="report",
            command=(
                python,
                "-m",
                "rmtgp_aco",
                "report-capacity-study",
                "--study-config",
                str(study.source_path),
            ),
            artifact=study.output_root / "report" / "capacity_report.md",
            kind="report",
        )
    )
    return tasks


def _task_complete(task: AblationTask, study: CapacityStudySpec) -> bool:
    meta = task.meta()
    if task.kind == "preflight":
        if not task.artifact.is_file():
            return False
        try:
            payload = json.loads(task.artifact.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            payload.get("requested_generations") == 1
            and payload.get("method_profile") == meta["method"]
            and payload.get("variant") == meta["variant"]
        )
    if task.kind == "train":
        return _training_artifact_valid(
            task.artifact,
            method=meta["method"],
            root_seed=int(meta["root_seed"]),
        )
    if task.kind == "test":
        manifest = task.artifact.parent / "evaluation_manifest.json"
        if not task.artifact.is_file() or not manifest.is_file():
            return False
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return payload.get("status") == "completed"
    if task.kind == "efficiency":
        if not task.artifact.is_file():
            return False
        try:
            payload = json.loads(task.artifact.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return payload.get("status") == "completed"
    if task.kind == "report":
        return task.artifact.is_file()
    return False


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reuse_manifest(study: CapacityStudySpec) -> dict[str, Any]:
    """冻结所有 31 节点来源文件哈希。"""

    files: list[Path] = [
        study.source.config,
        study.source.root / "study_state.json",
        study.source.root / "report" / "ablation_summary.json",
    ]
    for variant in study.variants:
        for root_seed in variant.seeds:
            for method in CAPACITY_METHODS:
                run = source_run_path(study, method, variant.name, root_seed)
                files.extend(
                    (
                        run / "selected_candidate.pkl",
                        run / "deployment_decision.json",
                        run / "training_validation_curve.csv",
                    )
                )
        for partition in study.partitions:
            files.append(
                study.source.root / "test" / variant.name / partition / "residual" / "records.csv"
            )
        for partition in load_study_spec(study.source.study.reuse.config).partitions:
            files.append(
                study.source.study.reuse.root / "test" / variant.name / partition / "records.csv"
            )
    unique = sorted(set(files))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("容量复用 artifact 缺失: " + ", ".join(missing))
    return {
        "schema_version": 1,
        "source_study": str(study.source.root),
        "source_capacity": SOURCE_CAPACITY,
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in unique
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _state_payload(
    *,
    study: CapacityStudySpec,
    status: str,
    completed: set[str],
    tasks: list[AblationTask],
    started_at: str,
    phase: str,
    workers: dict[str, dict[str, Any]],
    error: str | None = None,
    detail: str | None = None,
    gpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    order = {task.task_id: index for index, task in enumerate(tasks)}
    active = [
        str(worker["current_task"])
        for worker in workers.values()
        if worker.get("current_task") is not None
    ]
    return {
        "schema_version": CAPACITY_STATE_SCHEMA_VERSION,
        "study_id": study.study_id,
        "status": status,
        "pid": os.getpid(),
        "phase": phase,
        "current_task": active[0] if len(active) == 1 else None,
        "current_tasks": active,
        "completed_tasks": sorted(completed, key=lambda item: order[item]),
        "completed_count": len(completed),
        "total_tasks": len(tasks),
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "workers": workers,
        "gpus": gpus or [],
        "detail": detail,
        "error": error,
        "git": git_state(Path.cwd()),
    }


def _execute_task(
    task: AblationTask,
    study: CapacityStudySpec,
    *,
    physical_device: int | None,
) -> None:
    if physical_device is not None and task.kind in {"preflight", "train", "test", "efficiency"}:
        contention = _external_gpu_processes(physical_device)
        if contention:
            raise RuntimeError(
                f"任务 {task.task_id} 启动前 GPU{physical_device} 出现外部进程: {contention}"
            )
    log = study.output_root / "logs" / f"{task.task_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if physical_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    with log.open("a", encoding="utf-8") as handle:
        gpu_label = f" gpu={physical_device}" if physical_device is not None else ""
        handle.write(
            f"\n[{datetime.now(UTC).isoformat()}] START{gpu_label} " + " ".join(task.command) + "\n"
        )
        handle.flush()
        result = subprocess.run(
            task.command,
            cwd=Path.cwd(),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"[{datetime.now(UTC).isoformat()}] EXIT {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"任务 {task.task_id} 失败（exit={result.returncode}），详见 {log}")
    if not _task_complete(task, study):
        raise RuntimeError(f"任务 {task.task_id} 未产生完整 artifact")


def _wait_for_dependencies(
    study: CapacityStudySpec,
    *,
    physical_devices: tuple[int, ...],
    wait: bool,
    poll_seconds: int,
    write_waiting: Any,
) -> None:
    """等待源消融和 GPU 释放；不把“尚在运行”误报为失败。"""

    while True:
        ready, detail = source_ready(study)
        busy = {device: _external_gpu_processes(device) for device in physical_devices}
        busy = {device: rows for device, rows in busy.items() if rows}
        if ready and not busy:
            return
        reasons = [detail] if not ready else []
        if busy:
            reasons.append(f"GPU 尚忙: {busy}")
        message = "；".join(reasons)
        write_waiting(message)
        if not wait:
            raise RuntimeError(message)
        print(f"capacity queue waiting: {message}", flush=True)
        time.sleep(poll_seconds)


def run_capacity_parallel(
    study: CapacityStudySpec,
    *,
    physical_devices: tuple[int, ...],
    wait: bool = False,
    poll_seconds: int = 60,
) -> None:
    """在多张 GPU 上按阶段运行 46 项容量实验，可等待源队列并断点恢复。"""

    if not physical_devices:
        raise ValueError("并行 capacity study 至少需要一张物理 GPU")
    if len(set(physical_devices)) != len(physical_devices):
        raise ValueError("物理 GPU index 不得重复")
    if poll_seconds < 5 or poll_seconds > 300:
        raise ValueError("poll_seconds 必须位于 [5, 300]")
    repository = git_state(Path.cwd())
    if repository["dirty"]:
        raise RuntimeError("capacity study 启动前 Git worktree 必须 clean")
    if shutil.disk_usage(Path.cwd()).free < 50 * 1024**3:
        raise RuntimeError("共享文件系统剩余空间不足 50 GiB，拒绝启动")

    study.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = study.output_root / "study.lock"
    state_path = study.output_root / "study_state.json"
    pid_path = study.output_root / "runner.pid"
    tasks = build_capacity_tasks(study)
    started_at = datetime.now(UTC).isoformat()
    workers: dict[str, dict[str, Any]] = {
        str(device): {
            "physical_device": device,
            "status": "waiting",
            "current_task": None,
            "last_completed_task": None,
        }
        for device in physical_devices
    }

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有 capacity runner 持有锁") from exc
        pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        completed = {task.task_id for task in tasks if _task_complete(task, study)}
        state_lock = Lock()
        phase = "waiting"
        gpus: list[dict[str, Any]] = []

        def write_state(
            *,
            status: str = "running",
            error: str | None = None,
            detail: str | None = None,
            completed_at: str | None = None,
        ) -> None:
            payload = _state_payload(
                study=study,
                status=status,
                completed=completed,
                tasks=tasks,
                started_at=started_at,
                phase=phase,
                workers=workers,
                error=error,
                detail=detail,
                gpus=gpus,
            )
            if completed_at is not None:
                payload["completed_at"] = completed_at
            _atomic_json(state_path, payload)

        write_state(status="waiting", detail="检查源消融与 GPU")
        try:
            _wait_for_dependencies(
                study,
                physical_devices=physical_devices,
                wait=wait,
                poll_seconds=poll_seconds,
                write_waiting=lambda detail: write_state(status="waiting", detail=detail),
            )
            gpus.extend(_physical_gpu_description(device) for device in physical_devices)
            reuse_path = study.output_root / "reuse_manifest.json"
            if not reuse_path.is_file():
                _atomic_json(reuse_path, _reuse_manifest(study))

            for phase_kind in ("preflight", "train", "test", "efficiency"):
                phase = phase_kind
                units = [
                    (task,)
                    for task in tasks
                    if task.kind == phase_kind
                    and task.task_id not in completed
                    and not _task_complete(task, study)
                ]
                if not units:
                    continue
                work: Queue[tuple[AblationTask, ...]] = Queue()
                for unit in units:
                    work.put(unit)
                stop = Event()
                errors: list[BaseException] = []

                def worker_loop(
                    physical_device: int,
                    *,
                    work_queue: Queue[tuple[AblationTask, ...]] = work,
                    stop_event: Event = stop,
                    phase_errors: list[BaseException] = errors,
                ) -> None:
                    key = str(physical_device)
                    while not stop_event.is_set():
                        try:
                            unit = work_queue.get_nowait()
                        except Empty:
                            return
                        try:
                            for task in unit:
                                if task.task_id in completed or _task_complete(task, study):
                                    with state_lock:
                                        completed.add(task.task_id)
                                    continue
                                with state_lock:
                                    workers[key]["status"] = "running"
                                    workers[key]["current_task"] = task.task_id
                                    write_state()
                                _execute_task(
                                    task,
                                    study,
                                    physical_device=physical_device,
                                )
                                with state_lock:
                                    completed.add(task.task_id)
                                    workers[key]["last_completed_task"] = task.task_id
                                    workers[key]["current_task"] = None
                                    workers[key]["status"] = "idle"
                                    write_state()
                        except BaseException as exc:
                            with state_lock:
                                phase_errors.append(exc)
                                workers[key]["status"] = "failed"
                                workers[key]["error"] = f"{type(exc).__name__}: {exc}"
                                stop_event.set()
                        finally:
                            work_queue.task_done()

                threads = [
                    Thread(
                        target=worker_loop,
                        args=(device,),
                        name=f"capacity-gpu-{device}",
                        daemon=False,
                    )
                    for device in physical_devices
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                if errors:
                    raise RuntimeError(
                        "；".join(f"{type(error).__name__}: {error}" for error in errors)
                    )

            phase = "report"
            report = next(task for task in tasks if task.kind == "report")
            if not _task_complete(report, study):
                key = str(physical_devices[0])
                workers[key]["status"] = "running"
                workers[key]["current_task"] = report.task_id
                write_state()
                _execute_task(report, study, physical_device=None)
            completed.add(report.task_id)
            for worker in workers.values():
                worker["status"] = "completed"
                worker["current_task"] = None
            phase = "completed"
            write_state(
                status="completed",
                detail="容量敏感性实验全部完成",
                completed_at=datetime.now(UTC).isoformat(),
            )
        except BaseException as exc:
            phase = "failed"
            write_state(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def capacity_status(study: CapacityStudySpec) -> dict[str, Any]:
    """返回队列状态、依赖状态及当前训练代数。"""

    state_path = study.output_root / "study_state.json"
    if state_path.is_file():
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "study_id": study.study_id,
            "status": "not-started",
            "completed_count": 0,
            "total_tasks": len(build_capacity_tasks(study)),
        }
    ready, detail = source_ready(study)
    state["source_ready"] = ready
    state["source_detail"] = detail
    tasks = {task.task_id: task for task in build_capacity_tasks(study)}
    progress: dict[str, Any] = {}
    for current in state.get("current_tasks", []):
        if not isinstance(current, str) or not current.startswith("train-"):
            continue
        task = tasks.get(current)
        if task is None:
            continue
        metrics = task.artifact / "training_metrics.jsonl"
        if not metrics.is_file():
            continue
        lines = [line for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        latest = json.loads(lines[-1])
        progress[current] = {
            "generation": latest["generation"],
            "generation_wall_time_sec": latest["generation_wall_time"],
            "eta_seconds": latest["eta_seconds"],
            "train_delta_pp": latest["best_mean_delta_by_scale"],
            "validation_delta_pp": latest.get("validation_monitor_delta_by_scale", {}),
        }
    if progress:
        state["training_progress"] = progress
    pid_path = study.output_root / "runner.pid"
    if pid_path.is_file():
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        state["pid"] = pid
        try:
            os.kill(pid, 0)
            state["process_alive"] = True
        except OSError:
            state["process_alive"] = False
    return state


def export_capacity_contract(study: CapacityStudySpec) -> dict[str, Any]:
    """输出报告使用的纯 JSON 合同。"""

    return {
        "study_id": study.study_id,
        "output_root": str(study.output_root),
        "phase": study.phase,
        "source_capacity": SOURCE_CAPACITY,
        "target_capacity": TARGET_CAPACITY,
        "methods": list(CAPACITY_METHODS),
        "test_root_seed": study.test_root_seed,
        "test_seeds": study.test_seeds,
        "partitions": list(study.partitions),
        "champion_selection": {
            "unit": "one_validation_selected_candidate_per_gp_run",
            "gp_runs_per_method": 3,
            "best_seed_selection": False,
        },
        "variants": [
            {
                "name": variant.name,
                "single_config": str(variant.single_config),
                "multi_config": str(variant.multi_config),
                "seeds": list(variant.seeds),
            }
            for variant in study.variants
        ],
        "source": {
            "config": str(study.source.config),
            "root": str(study.source.root),
        },
        "manifest": str(study.manifest),
        "bootstrap_replicates": study.bootstrap_replicates,
    }
