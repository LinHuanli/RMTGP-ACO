"""纯 TSP100 三算法三种子 study 的编排、缓存与可恢复测试。"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml

from .aco import solve
from .artifacts import git_state
from .baseline import backend_semantic_id, read_baseline_shard
from .evaluation import (
    EvaluationRecord,
    compile_champion,
    load_champion,
    read_records,
    records_from_paired_results,
    study_test_seed,
    write_records,
)
from .manifest import load_manifest, verify_manifest
from .model import ProblemBatch, RunDiagnostics, RunResult
from .runtime import configure_runtime
from .sampling import iter_problem_batches
from .spec import RunSpec, load_run_spec

TEST_CACHE_SCHEMA_VERSION = 1
STUDY_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StudyVariant:
    """一个 ACO 变体及其三个独立 GP root seeds。"""

    name: str
    config: Path
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StudySpec:
    """可提交到后台队列的冻结 study 规范。"""

    study_id: str
    output_root: Path
    phase: str
    method_profile: str
    test_root_seed: int
    test_seeds: int
    partitions: tuple[str, ...]
    variants: tuple[StudyVariant, ...]
    manifest: Path
    bootstrap_replicates: int
    source_path: Path


@dataclass(frozen=True, slots=True)
class StudyTask:
    """后台队列中的一个可恢复原子任务。"""

    task_id: str
    command: tuple[str, ...]
    artifact: Path
    kind: str


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def load_study_spec(path: str | Path) -> StudySpec:
    """严格读取 study YAML，并验证训练/测试合同。"""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study 配置根节点必须为 mapping")
    allowed = {
        "study_id",
        "output_root",
        "phase",
        "method_profile",
        "test_root_seed",
        "test_seeds",
        "partitions",
        "variants",
        "manifest",
        "bootstrap_replicates",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"study 配置含未知字段: {sorted(unknown)}")
    required = {
        "study_id",
        "output_root",
        "method_profile",
        "test_root_seed",
        "test_seeds",
        "partitions",
        "variants",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"study 配置缺少字段: {sorted(missing)}")

    variants_raw = payload["variants"]
    if not isinstance(variants_raw, dict) or not variants_raw:
        raise ValueError("study.variants 必须为非空 mapping")
    variants: list[StudyVariant] = []
    all_seeds: list[int] = []
    for name, raw in variants_raw.items():
        if not isinstance(raw, dict) or set(raw) != {"config", "seeds"}:
            raise ValueError(f"variants.{name} 仅允许 config 与 seeds")
        seeds = tuple(int(value) for value in raw["seeds"])
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"variants.{name}.seeds 必须非空且不重复")
        variant = StudyVariant(
            name=str(name),
            config=_project_path(str(raw["config"])),
            seeds=seeds,
        )
        run_spec = load_run_spec(variant.config)
        if run_spec.experiment.aco.variant.value != variant.name:
            raise ValueError(
                f"{variant.config}: ACO variant 与 study key {variant.name!r} 不一致"
            )
        if (
            run_spec.experiment.train_scales != (100,)
            or run_spec.experiment.validation_scales != (100,)
        ):
            raise ValueError(f"{variant.config}: 正式 study 必须纯 TSP100 train/validation")
        variants.append(variant)
        all_seeds.extend(seeds)
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("不同 ACO 变体的 GP root seeds 也必须唯一")

    partitions = tuple(str(value) for value in payload["partitions"])
    if not partitions or len(set(partitions)) != len(partitions):
        raise ValueError("partitions 必须非空且不重复")
    for variant in variants:
        run_spec = load_run_spec(variant.config)
        absent = set(partitions) - set(run_spec.data.test)
        if absent:
            raise ValueError(f"{variant.config}: 缺少 test partitions {sorted(absent)}")

    phase = str(payload.get("phase", "pilot"))
    if phase not in {"pilot", "formal"}:
        raise ValueError("study.phase 仅支持 pilot 或 formal")
    test_seeds = int(payload["test_seeds"])
    if test_seeds < 1:
        raise ValueError("test_seeds 必须为正整数")
    bootstrap_replicates = int(payload.get("bootstrap_replicates", 10_000))
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates 至少为 100")
    return StudySpec(
        study_id=str(payload["study_id"]),
        output_root=_project_path(str(payload["output_root"])),
        phase=phase,
        method_profile=str(payload["method_profile"]),
        test_root_seed=int(payload["test_root_seed"]),
        test_seeds=test_seeds,
        partitions=partitions,
        variants=tuple(variants),
        manifest=_project_path(str(payload.get("manifest", "Datasets/manifest.json"))),
        bootstrap_replicates=bootstrap_replicates,
        source_path=source,
    )


def _variant(study: StudySpec, name: str) -> StudyVariant:
    for variant in study.variants:
        if variant.name == name:
            return variant
    raise KeyError(f"study 中不存在 ACO variant: {name}")


def _validate_test_manifest(study: StudySpec, spec: RunSpec, partition: str) -> None:
    manifest = load_manifest(study.manifest)
    if manifest.hash_mode != "sha256-full":
        raise ValueError("正式测试要求 sha256-full 数据 manifest")
    paths = spec.data.test_paths(partition)
    declared = {
        (record.path, record.split, record.sha256): record
        for record in manifest.files
    }
    for path in paths:
        relative = path.relative_to(spec.data.root).as_posix()
        candidates = [
            record
            for (name, split, digest), record in declared.items()
            if name == relative and split == "test" and digest is not None
        ]
        if not candidates:
            raise ValueError(f"测试文件不在完整 manifest 中: {relative}")
    errors = verify_manifest(
        manifest,
        root=spec.data.root,
        verify_hashes=False,
        validate_first_record=True,
    )
    if errors:
        raise ValueError("测试数据 manifest 预检失败：" + "；".join(errors[:5]))


def _tensor_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _write_test_cache(
    path: Path,
    result: RunResult,
    *,
    metadata: dict[str, Any],
) -> None:
    """原子写入一个 batch×seed 的完整 baseline 结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    diagnostics = result.diagnostics
    np.savez_compressed(
        temporary,
        __metadata__=np.asarray(
            json.dumps(
                {
                    "schema_version": TEST_CACHE_SCHEMA_VERSION,
                    **metadata,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
        best_tour=_tensor_numpy(result.best_tour),
        best_length=_tensor_numpy(result.best_length),
        best_iteration=_tensor_numpy(result.best_iteration),
        anytime_best=_tensor_numpy(result.anytime_best),
        wall_time_sec=np.asarray(result.wall_time_sec, dtype=np.float64),
        constructed_tours=np.asarray(result.constructed_tours, dtype=np.int64),
        diagnostics=np.asarray(
            [
                diagnostics.uniform_fallback_count,
                diagnostics.nan_sanitized_count,
                diagnostics.candidate_fallback_count,
                diagnostics.bound_clip_count,
                diagnostics.mmas_restart_count,
            ],
            dtype=np.int64,
        ),
        backend_metrics=np.asarray(
            json.dumps(result.backend_metrics, ensure_ascii=False, sort_keys=True)
        ),
    )
    temporary.replace(path)


def _read_test_cache(
    path: Path,
    *,
    expected_metadata: dict[str, Any],
) -> RunResult:
    """读取并严格核对 baseline test cache，拒绝错误复用。"""

    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["__metadata__"].item()))
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"{path}: cache metadata {key}={metadata.get(key)!r}，"
                    f"预期 {expected!r}"
                )
        if metadata.get("schema_version") != TEST_CACHE_SCHEMA_VERSION:
            raise ValueError(f"{path}: test cache schema 不兼容")
        diagnostic = payload["diagnostics"].astype(np.int64).tolist()
        backend_metrics = json.loads(str(payload["backend_metrics"].item()))
        return RunResult(
            best_tour=torch.from_numpy(payload["best_tour"].copy()),
            best_length=torch.from_numpy(payload["best_length"].copy()),
            best_iteration=torch.from_numpy(payload["best_iteration"].copy()),
            anytime_best=torch.from_numpy(payload["anytime_best"].copy()),
            wall_time_sec=float(payload["wall_time_sec"].item()),
            constructed_tours=int(payload["constructed_tours"].item()),
            diagnostics=RunDiagnostics(
                uniform_fallback_count=int(diagnostic[0]),
                nan_sanitized_count=int(diagnostic[1]),
                candidate_fallback_count=int(diagnostic[2]),
                bound_clip_count=int(diagnostic[3]),
                mmas_restart_count=int(diagnostic[4]),
            ),
            backend_metrics=backend_metrics,
        )


def _cache_metadata(
    *,
    study: StudySpec,
    spec: RunSpec,
    partition: str,
    batch_number: int,
    replicate: int,
    seed: int,
    batch: ProblemBatch,
) -> dict[str, Any]:
    return {
        "study_id": study.study_id,
        "variant": spec.experiment.aco.variant.value,
        "partition": partition,
        "batch_number": batch_number,
        "replicate": replicate,
        "seed": seed,
        "aco_config_hash": spec.experiment.aco.config_hash,
        "backend_semantic": backend_semantic_id(
            spec.experiment.runtime.aco_backend
        ),
        "instance_ids": list(batch.instance_ids),
        "coordinate_hashes": list(batch.coordinate_hashes),
        "scale": batch.n,
    }


def _validate_test_shard(
    path: Path,
    *,
    batch: ProblemBatch,
    partition: str,
    seed: int,
    champion_id: str,
    gp_root_seed: int,
    method: str,
) -> list[EvaluationRecord]:
    records = read_records([path])
    if len(records) != batch.batch_size:
        raise ValueError(f"{path}: shard 行数不等于 batch size")
    expected_ids = set(batch.instance_ids)
    if {record.instance_id for record in records} != expected_ids:
        raise ValueError(f"{path}: shard instance IDs 与当前 batch 不一致")
    for record in records:
        if (
            record.partition != partition
            or record.seed != seed
            or record.champion_id != champion_id
            or record.gp_root_seed != gp_root_seed
            or record.method != method
        ):
            raise ValueError(f"{path}: shard provenance 不一致")
    return records


def _merged_test_valid(
    path: Path,
    *,
    expected_instances: int,
    study: StudySpec,
    variant: StudyVariant,
    partition: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        records = read_records([path])
    except (OSError, TypeError, ValueError):
        return False
    expected = expected_instances * study.test_seeds * len(variant.seeds)
    if len(records) != expected:
        return False
    keys = {
        (record.gp_root_seed, record.instance_id, record.seed)
        for record in records
    }
    return (
        len(keys) == expected
        and {record.gp_root_seed for record in records} == set(variant.seeds)
        and {record.partition for record in records} == {partition}
        and {record.variant for record in records} == {variant.name}
        and {record.method for record in records} == {study.method_profile}
    )


def evaluate_study_partition(
    study: StudySpec,
    *,
    variant_name: str,
    partition: str,
    skip_manifest_check: bool = False,
) -> Path:
    """评测一个 ACO variant×partition，并在三个 GP seeds 间共享 baseline。"""

    variant = _variant(study, variant_name)
    spec = load_run_spec(variant.config)
    if partition not in study.partitions:
        raise KeyError(f"partition 不在 study 合同中: {partition}")
    if not skip_manifest_check:
        _validate_test_manifest(study, spec, partition)
    configure_runtime(spec.experiment.runtime)

    selected: list[tuple[int, str, Any, Any]] = []
    decisions: dict[int, dict[str, Any]] = {}
    for root_seed in variant.seeds:
        run = study.output_root / "train" / variant.name / f"seed-{root_seed}"
        champion = load_champion(run / "selected_candidate.pkl")
        transition, pheromone = compile_champion(champion)
        selected.append(
            (root_seed, champion.structural_hash, transition, pheromone)
        )
        decisions[root_seed] = json.loads(
            (run / "deployment_decision.json").read_text(encoding="utf-8")
        )

    partition_spec = spec.data.test[partition]
    paths = spec.data.test_paths(partition)
    output = study.output_root / "test" / variant.name / partition
    merged = output / "records.csv"
    manifest = load_manifest(study.manifest)
    expected_instances = sum(
        record.instances
        for record in manifest.files
        if record.split == "test"
        and (spec.data.root / record.path).resolve() in set(paths)
    )
    if _merged_test_valid(
        merged,
        expected_instances=expected_instances,
        study=study,
        variant=variant,
        partition=partition,
    ):
        print(f"测试已完成，复用 merged artifact: {merged}", flush=True)
        return merged

    started = perf_counter()
    all_records: list[EvaluationRecord] = []
    completed_shards = 0
    total_batches = (expected_instances + spec.data.evaluation_batch_size - 1) // (
        spec.data.evaluation_batch_size
    )
    total_shards = total_batches * study.test_seeds * len(variant.seeds)
    for batch_number, batch in enumerate(
        iter_problem_batches(
            paths,
            batch_size=spec.data.evaluation_batch_size,
            candidate_size=spec.experiment.aco.candidate_size,
            dtype=spec.experiment.aco.dtype,
            device=spec.experiment.aco.device,
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
            metadata = _cache_metadata(
                study=study,
                spec=spec,
                partition=partition,
                batch_number=batch_number,
                replicate=replicate,
                seed=seed,
                batch=batch,
            )
            cache = (
                study.output_root
                / "test-cache"
                / variant.name
                / partition
                / f"aco-{replicate:02d}"
                / f"batch-{batch_number:04d}.npz"
            )
            if cache.is_file():
                baseline = _read_test_cache(cache, expected_metadata=metadata)
            else:
                baseline = solve(
                    batch,
                    spec.experiment.aco,
                    seed=seed,
                    backend=spec.experiment.runtime.aco_backend,
                    runtime=spec.experiment.runtime,
                )
                _write_test_cache(cache, baseline, metadata=metadata)

            for root_seed, champion_id, transition, pheromone in selected:
                shard = (
                    output
                    / f"seed-{root_seed}"
                    / f"aco-{replicate:02d}"
                    / f"batch-{batch_number:04d}.csv"
                )
                if shard.is_file():
                    records = _validate_test_shard(
                        shard,
                        batch=batch,
                        partition=partition,
                        seed=seed,
                        champion_id=champion_id,
                        gp_root_seed=root_seed,
                        method=study.method_profile,
                    )
                else:
                    candidate = solve(
                        batch,
                        spec.experiment.aco,
                        transition_program=transition,
                        pheromone_program=pheromone,
                        seed=seed,
                        backend=spec.experiment.runtime.aco_backend,
                        runtime=spec.experiment.runtime,
                    )
                    records = records_from_paired_results(
                        method=study.method_profile,
                        champion_id=champion_id,
                        partition=partition,
                        distribution=partition_spec.distribution,
                        batch=batch,
                        seed=seed,
                        candidate=candidate,
                        baseline=baseline,
                        config=spec.experiment.aco,
                        gp_run_id=f"{variant.name}-seed-{root_seed}",
                        gp_root_seed=root_seed,
                    )
                    write_records(records, shard)
                all_records.extend(records)
                completed_shards += 1
                elapsed = perf_counter() - started
                rate = completed_shards / max(elapsed, 1e-12)
                eta = (total_shards - completed_shards) / max(rate, 1e-12)
                print(
                    f"test={variant.name}/{partition} "
                    f"shards={completed_shards}/{total_shards} "
                    f"elapsed={elapsed / 60.0:.1f}min eta={eta / 60.0:.1f}min",
                    flush=True,
                )

    expected_rows = expected_instances * study.test_seeds * len(variant.seeds)
    if len(all_records) != expected_rows:
        raise RuntimeError(
            f"{variant.name}/{partition}: 得到 {len(all_records)} 行，"
            f"预期 {expected_rows}"
        )
    unique_keys = {
        (record.gp_root_seed, record.instance_id, record.seed)
        for record in all_records
    }
    if len(unique_keys) != expected_rows:
        raise RuntimeError(f"{variant.name}/{partition}: 测试长表存在重复配对键")
    all_records.sort(
        key=lambda record: (
            record.gp_root_seed,
            record.instance_id,
            record.seed,
        )
    )
    write_records(all_records, merged)
    _atomic_write_json(
        output / "evaluation_manifest.json",
        {
            "schema_version": 1,
            "study_id": study.study_id,
            "variant": variant.name,
            "partition": partition,
            "method": study.method_profile,
            "test_root_seed": study.test_root_seed,
            "aco_seeds": study.test_seeds,
            "gp_root_seeds": list(variant.seeds),
            "instances": expected_instances,
            "rows": expected_rows,
            "aco_config_hash": spec.experiment.aco.config_hash,
            "backend": spec.experiment.runtime.aco_backend.value,
            "backend_semantic": backend_semantic_id(
                spec.experiment.runtime.aco_backend
            ),
            "selected_candidate_tested": True,
            "deployment_decisions": decisions,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return merged


def _schedule_artifact_valid(path: Path, root_seed: int, replicate: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    records = payload.get("records", [])
    return (
        int(payload.get("root_seed", -1)) == root_seed
        and bool(records)
        and {int(record.get("replicate_id", -1)) for record in records}
        == {replicate}
    )


def _baseline_artifact_valid(path: Path, *, variant: str, replicate: int) -> bool:
    if not path.is_file():
        return False
    try:
        records, metadata = read_baseline_shard(path)
    except (OSError, ValueError):
        return False
    return (
        bool(records)
        and metadata.get("variant") == variant
        and int(metadata.get("replicate_id", -1)) == replicate
        and set(metadata.get("splits", [])) == {"train", "selection", "gate"}
    )


def _training_artifact_valid(path: Path, root_seed: int) -> bool:
    manifest = path / "manifest.json"
    required = (
        path / "selected_candidate.pkl",
        path / "deployment_decision.json",
        path / "training_validation_curve.csv",
    )
    if not manifest.is_file() or not all(item.is_file() for item in required):
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("status") == "completed"
        and int(payload["configuration"]["root_seed"]) == root_seed
    )


def build_study_tasks(study: StudySpec) -> list[StudyTask]:
    """按 replicate-major 顺序建立单 GPU 串行队列。"""

    python = sys.executable
    tasks: list[StudyTask] = []
    max_replicates = max(len(variant.seeds) for variant in study.variants)
    for replicate in range(max_replicates):
        for variant in study.variants:
            if replicate >= len(variant.seeds):
                continue
            root_seed = variant.seeds[replicate]
            schedule = (
                study.output_root
                / "schedules"
                / f"{variant.name}-seed-{root_seed}.json"
            )
            baseline = (
                study.output_root
                / "baselines"
                / variant.name
                / f"{variant.name}-seed-{root_seed}.npz"
            )
            run = (
                study.output_root
                / "train"
                / variant.name
                / f"seed-{root_seed}"
            )
            common = (
                "--config",
                str(variant.config),
                "--root-seed",
                str(root_seed),
                "--replicate-id",
                str(replicate),
            )
            tasks.append(
                StudyTask(
                    task_id=f"schedule-{variant.name}-{root_seed}",
                    command=(
                        python,
                        "-m",
                        "rmtgp_aco",
                        "prepare-schedules",
                        *common,
                        "--phase",
                        study.phase,
                        "--manifest",
                        str(study.manifest),
                        "--output",
                        str(schedule),
                    ),
                    artifact=schedule,
                    kind="schedule",
                )
            )
            tasks.append(
                StudyTask(
                    task_id=f"baseline-{variant.name}-{root_seed}",
                    command=(
                        python,
                        "-m",
                        "rmtgp_aco",
                        "precompute-baselines",
                        *common,
                        "--schedule",
                        str(schedule),
                        "--splits",
                        "train",
                        "selection",
                        "gate",
                        "--output",
                        str(baseline),
                    ),
                    artifact=baseline,
                    kind="baseline",
                )
            )
            train_command = [
                python,
                "-m",
                "rmtgp_aco",
                "train",
                *common,
                "--phase",
                study.phase,
                "--manifest",
                str(study.manifest),
                "--schedule",
                str(schedule),
                "--baseline-archive",
                str(baseline.parent),
                "--method-profile",
                study.method_profile,
                "--output",
                str(run),
                "--traceback",
            ]
            if (run / "training_state.pkl").is_file() and not _training_artifact_valid(
                run, root_seed
            ):
                train_command.extend(["--resume", str(run)])
            tasks.append(
                StudyTask(
                    task_id=f"train-{variant.name}-{root_seed}",
                    command=tuple(train_command),
                    artifact=run,
                    kind="train",
                )
            )

    for variant in study.variants:
        for partition in study.partitions:
            artifact = (
                study.output_root
                / "test"
                / variant.name
                / partition
                / "records.csv"
            )
            tasks.append(
                StudyTask(
                    task_id=f"test-{variant.name}-{partition}",
                    command=(
                        python,
                        "-m",
                        "rmtgp_aco",
                        "evaluate-study",
                        "--study-config",
                        str(study.source_path),
                        "--variant",
                        variant.name,
                        "--partition",
                        partition,
                    ),
                    artifact=artifact,
                    kind="test",
                )
            )
    tasks.append(
        StudyTask(
            task_id="report",
            command=(
                python,
                "-m",
                "rmtgp_aco",
                "report-study",
                "--study-config",
                str(study.source_path),
            ),
            artifact=study.output_root / "report" / "study_report.md",
            kind="report",
        )
    )
    return tasks


def _task_complete(task: StudyTask, study: StudySpec) -> bool:
    parts = task.task_id.split("-")
    if task.kind == "schedule":
        return _schedule_artifact_valid(
            task.artifact,
            int(parts[-1]),
            next(
                index
                for variant in study.variants
                if variant.name == parts[-2]
                for index, seed in enumerate(variant.seeds)
                if seed == int(parts[-1])
            ),
        )
    if task.kind == "baseline":
        variant_name = parts[-2]
        root_seed = int(parts[-1])
        replicate = next(
            index
            for variant in study.variants
            if variant.name == variant_name
            for index, seed in enumerate(variant.seeds)
            if seed == root_seed
        )
        return _baseline_artifact_valid(
            task.artifact,
            variant=variant_name,
            replicate=replicate,
        )
    if task.kind == "train":
        return _training_artifact_valid(task.artifact, int(parts[-1]))
    if task.kind == "test":
        variant_name = parts[1]
        partition = "-".join(parts[2:])
        variant = _variant(study, variant_name)
        run_spec = load_run_spec(variant.config)
        manifest = load_manifest(study.manifest)
        paths = set(run_spec.data.test_paths(partition))
        expected_instances = sum(
            record.instances
            for record in manifest.files
            if record.split == "test"
            and (run_spec.data.root / record.path).resolve() in paths
        )
        return _merged_test_valid(
            task.artifact,
            expected_instances=expected_instances,
            study=study,
            variant=variant,
            partition=partition,
        )
    return task.artifact.is_file()


def _assert_single_gpu0() -> dict[str, Any]:
    """拒绝设备映射歧义，确保进程仅能看到物理 GPU0。"""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(
            "正式 study 要求 CUDA_VISIBLE_DEVICES=0（当前为 "
            f"{visible!r}）"
        )
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CUDA study 需要已安装 cupy-cuda12x") from exc
    count = int(cp.cuda.runtime.getDeviceCount())
    if count != 1:
        raise RuntimeError(f"进程应只看到一张 GPU，实际为 {count}")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = properties["name"]
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    if "RTX 4000 Ada" not in name:
        raise RuntimeError(f"逻辑 GPU0 不是预期 RTX 4000 Ada: {name}")
    return {"visible": visible, "logical_devices": count, "device_name": name}


def _external_gpu0_processes() -> list[dict[str, Any]]:
    """查询物理 GPU0 上除当前 runner 外的计算进程。"""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    processes: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        processes.append(
            {
                "pid": pid,
                "process_name": fields[1],
                "used_memory_mib": fields[2],
            }
        )
    return processes


def _state_payload(
    *,
    study: StudySpec,
    status: str,
    current_task: str | None,
    completed: list[str],
    total_tasks: int,
    error: str | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": STUDY_STATE_SCHEMA_VERSION,
        "study_id": study.study_id,
        "status": status,
        "pid": os.getpid(),
        "current_task": current_task,
        "completed_tasks": completed,
        "completed_count": len(completed),
        "total_tasks": total_tasks,
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "error": error,
        "git": git_state(Path.cwd()),
    }


def run_study_queue(study: StudySpec) -> None:
    """在唯一 GPU0 上串行执行并可从最后一个完整 artifact 恢复。"""

    gpu = _assert_single_gpu0()
    contention = _external_gpu0_processes()
    if contention:
        raise RuntimeError(f"物理 GPU0 存在外部计算进程: {contention}")
    repository_state = git_state(Path.cwd())
    if repository_state["dirty"]:
        raise RuntimeError("正式 study 启动前 Git worktree 必须 clean")
    study.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = study.output_root / "study.lock"
    state_path = study.output_root / "study_state.json"
    pid_path = study.output_root / "runner.pid"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有 study runner 持有锁，拒绝重复启动") from exc
        pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        tasks = build_study_tasks(study)
        completed: list[str] = []
        started_at = datetime.now(UTC).isoformat()
        _atomic_write_json(
            state_path,
            {
                **_state_payload(
                    study=study,
                    status="running",
                    current_task=None,
                    completed=completed,
                    total_tasks=len(tasks),
                    started_at=started_at,
                ),
                "gpu": gpu,
            },
        )
        try:
            for task in tasks:
                if _task_complete(task, study):
                    completed.append(task.task_id)
                    continue
                if task.kind in {"baseline", "train", "test"}:
                    contention = _external_gpu0_processes()
                    if contention:
                        raise RuntimeError(
                            f"任务 {task.task_id} 启动前 GPU0 出现外部进程: "
                            f"{contention}"
                        )
                _atomic_write_json(
                    state_path,
                    {
                        **_state_payload(
                            study=study,
                            status="running",
                            current_task=task.task_id,
                            completed=completed,
                            total_tasks=len(tasks),
                            started_at=started_at,
                        ),
                        "gpu": gpu,
                        "command": list(task.command),
                    },
                )
                log = study.output_root / "logs" / f"{task.task_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"\n[{datetime.now(UTC).isoformat()}] START "
                        + " ".join(task.command)
                        + "\n"
                    )
                    handle.flush()
                    result = subprocess.run(
                        task.command,
                        cwd=Path.cwd(),
                        env=os.environ.copy(),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    handle.write(
                        f"[{datetime.now(UTC).isoformat()}] EXIT {result.returncode}\n"
                    )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"任务 {task.task_id} 失败（exit={result.returncode}），"
                        f"详见 {log}"
                    )
                if not _task_complete(task, study):
                    raise RuntimeError(f"任务 {task.task_id} 未产生完整 artifact")
                completed.append(task.task_id)
            _atomic_write_json(
                state_path,
                {
                    **_state_payload(
                        study=study,
                        status="completed",
                        current_task=None,
                        completed=completed,
                        total_tasks=len(tasks),
                        started_at=started_at,
                    ),
                    "gpu": gpu,
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
        except BaseException as exc:
            _atomic_write_json(
                state_path,
                {
                    **_state_payload(
                        study=study,
                        status="failed",
                        current_task=(
                            None if not tasks else next(
                                (
                                    task.task_id
                                    for task in tasks
                                    if task.task_id not in completed
                                ),
                                None,
                            )
                        ),
                        completed=completed,
                        total_tasks=len(tasks),
                        error=f"{type(exc).__name__}: {exc}",
                        started_at=started_at,
                    ),
                    "gpu": gpu,
                },
            )
            raise


def study_status(study: StudySpec) -> dict[str, Any]:
    """汇总 runner 状态及当前训练代数，供非阻塞监控。"""

    state_path = study.output_root / "study_state.json"
    state: dict[str, Any] = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {
            "study_id": study.study_id,
            "status": "not-started",
            "completed_count": 0,
            "total_tasks": len(build_study_tasks(study)),
        }
    )
    current = state.get("current_task")
    if isinstance(current, str) and current.startswith("train-"):
        run = next(
            (
                task.artifact
                for task in build_study_tasks(study)
                if task.task_id == current
            ),
            None,
        )
        if run is not None:
            metrics = run / "training_metrics.jsonl"
            if metrics.is_file():
                lines = [
                    line
                    for line in metrics.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if lines:
                    latest = json.loads(lines[-1])
                    state["training_progress"] = {
                        "generation": latest["generation"],
                        "generation_wall_time_sec": latest[
                            "generation_wall_time"
                        ],
                        "eta_seconds": latest["eta_seconds"],
                        "train_delta_pp": latest["best_mean_delta_by_scale"],
                        "validation_delta_pp": latest.get(
                            "validation_monitor_delta_by_scale",
                            {},
                        ),
                    }
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


def export_study_contract(study: StudySpec) -> dict[str, Any]:
    """返回便于报告和测试审计的纯 JSON study 合同。"""

    return {
        "study_id": study.study_id,
        "output_root": str(study.output_root),
        "phase": study.phase,
        "method_profile": study.method_profile,
        "test_root_seed": study.test_root_seed,
        "test_seeds": study.test_seeds,
        "partitions": list(study.partitions),
        "variants": [
            {
                **asdict(variant),
                "config": str(variant.config),
                "seeds": list(variant.seeds),
            }
            for variant in study.variants
        ],
        "manifest": str(study.manifest),
        "bootstrap_replicates": study.bootstrap_replicates,
    }
