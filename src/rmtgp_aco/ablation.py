"""纯 TSP100 多方法消融实验的合同、批量测试与可恢复 GPU 队列。"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
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

from .aco import solve
from .aco_cuda import solve_population_cuda_anytime
from .artifacts import git_state
from .baseline import backend_semantic_id
from .config import TransitionIntegration
from .evaluation import (
    EvaluationRecord,
    compile_champion,
    load_champion,
    read_records,
    records_from_paired_results,
    study_test_seed,
    write_records,
)
from .manifest import load_manifest
from .model import RunDiagnostics, RunResult
from .runtime import configure_runtime
from .sampling import iter_problem_batches
from .spec import RunSpec, load_run_spec
from .study import (
    _assert_single_gpu,
    _cache_metadata,
    _external_gpu_processes,
    _read_test_cache,
    _write_test_cache,
    load_study_spec,
)

ABLATION_STATE_SCHEMA_VERSION = 2
ABLATION_TEST_SCHEMA_VERSION = 2

CORE_METHODS: tuple[str, ...] = (
    "legacy",
    "matched-replace",
    "tr-rgp",
    "ph-rgp",
    "rmtgp-core-f0",
    "rmtgp-core-f1",
    "rmtgp-full-f0",
    "rmtgp-full-f1",
)
TRAINED_METHODS: tuple[str, ...] = tuple(
    method for method in CORE_METHODS if method != "rmtgp-full-f1"
)
RESIDUAL_METHODS: tuple[str, ...] = (
    "tr-rgp",
    "ph-rgp",
    "rmtgp-core-f0",
    "rmtgp-core-f1",
    "rmtgp-full-f0",
    "rmtgp-full-f1",
)
REPLACEMENT_METHODS: tuple[str, ...] = (
    "legacy",
    "matched-replace",
)
MECHANISM_METHODS: tuple[str, ...] = (
    "rmtgp-full-f1-drop-transition",
    "rmtgp-full-f1-drop-pheromone",
    "rmtgp-full-f1-shuffle-r1",
    "rmtgp-full-f1-shuffle-r2",
)
FINAL_METHOD = "rmtgp-full-f1"
EFFICIENCY_METHODS: tuple[str, ...] = (
    "legacy",
    "matched-replace",
    "tr-rgp",
    "ph-rgp",
    FINAL_METHOD,
)
TEST_GROUPS: tuple[str, ...] = ("residual", "replacement", "final")
ABLATION_TEST_GROUPS: tuple[str, ...] = ("residual", "replacement")


@dataclass(frozen=True, slots=True)
class AblationMethod:
    """一个训练或复用的 method profile。"""

    name: str
    source: str


@dataclass(frozen=True, slots=True)
class AblationVariant:
    """一个 ACO 变体与三个匹配 GP root seeds。"""

    name: str
    config: Path
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReuseStudy:
    """主实验中必须保持只读的 artifact 来源。"""

    config: Path
    root: Path


@dataclass(frozen=True, slots=True)
class AblationStudySpec:
    """GPU1 三种子消融 study 的冻结合同。"""

    study_id: str
    output_root: Path
    phase: str
    test_root_seed: int
    test_seeds: int
    partitions: tuple[str, ...]
    ablation_partitions: tuple[str, ...]
    methods: tuple[AblationMethod, ...]
    variants: tuple[AblationVariant, ...]
    reuse: ReuseStudy
    manifest: Path
    bootstrap_replicates: int
    source_path: Path


@dataclass(frozen=True, slots=True)
class AblationTask:
    """后台队列中的一个可验证原子任务。"""

    task_id: str
    command: tuple[str, ...]
    artifact: Path
    kind: str
    metadata: tuple[tuple[str, str], ...] = ()

    def meta(self) -> dict[str, str]:
        return dict(self.metadata)


@dataclass(frozen=True, slots=True)
class ProgramEntry:
    """批量测试中一个带 provenance 的锁定 program。"""

    method: str
    gp_root_seed: int
    champion_id: str
    transition: Any
    pheromone: Any


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _variant(study: AblationStudySpec, name: str) -> AblationVariant:
    for variant in study.variants:
        if variant.name == name:
            return variant
    raise KeyError(f"消融 study 中不存在 ACO variant: {name}")


def _method(study: AblationStudySpec, name: str) -> AblationMethod:
    for method in study.methods:
        if method.name == name:
            return method
    raise KeyError(f"消融 study 中不存在 method: {name}")


def load_ablation_spec(path: str | Path) -> AblationStudySpec:
    """严格读取多方法消融 YAML，并核对与已完成主实验的配对合同。"""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ablation 配置根节点必须为 mapping")
    allowed = {
        "study_id",
        "output_root",
        "phase",
        "test_root_seed",
        "test_seeds",
        "partitions",
        "ablation_partitions",
        "methods",
        "variants",
        "reuse",
        "manifest",
        "bootstrap_replicates",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"ablation 配置含未知字段: {sorted(unknown)}")
    required = allowed - {"phase", "manifest", "bootstrap_replicates"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ablation 配置缺少字段: {sorted(missing)}")

    methods_raw = payload["methods"]
    if not isinstance(methods_raw, dict):
        raise ValueError("methods 必须为 method -> train/reuse mapping")
    if set(methods_raw) != set(CORE_METHODS):
        raise ValueError(
            "核心消融 methods 必须精确为 "
            f"{list(CORE_METHODS)}"
        )
    methods = tuple(
        AblationMethod(name=name, source=str(methods_raw[name]))
        for name in CORE_METHODS
    )
    for method in methods:
        expected = "reuse" if method.name == "rmtgp-full-f1" else "train"
        if method.source != expected:
            raise ValueError(f"{method.name} 的 source 必须为 {expected}")

    reuse_raw = payload["reuse"]
    if not isinstance(reuse_raw, dict) or set(reuse_raw) != {"config", "root"}:
        raise ValueError("reuse 仅允许 config 与 root")
    reuse = ReuseStudy(
        config=_project_path(str(reuse_raw["config"])),
        root=_project_path(str(reuse_raw["root"])),
    )
    source_study = load_study_spec(reuse.config)
    if source_study.output_root != reuse.root:
        raise ValueError("reuse.root 与主 study output_root 不一致")
    source_state = reuse.root / "study_state.json"
    if not source_state.is_file():
        raise FileNotFoundError(f"主 study 状态不存在: {source_state}")
    state = json.loads(source_state.read_text(encoding="utf-8"))
    if state.get("status") != "completed" or state.get("completed_count") != 40:
        raise ValueError("主 study 尚未完整完成 40/40，拒绝复用")

    variants_raw = payload["variants"]
    if not isinstance(variants_raw, dict) or not variants_raw:
        raise ValueError("variants 必须为非空 mapping")
    source_variants = {item.name: item for item in source_study.variants}
    variants: list[AblationVariant] = []
    all_seeds: list[int] = []
    for name, raw in variants_raw.items():
        if not isinstance(raw, dict) or set(raw) != {"config", "seeds"}:
            raise ValueError(f"variants.{name} 仅允许 config 与 seeds")
        seeds = tuple(int(value) for value in raw["seeds"])
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError(f"variants.{name}.seeds 必须包含三个不同 seed")
        variant = AblationVariant(
            name=str(name),
            config=_project_path(str(raw["config"])),
            seeds=seeds,
        )
        spec = load_run_spec(variant.config)
        if spec.experiment.aco.variant.value != variant.name:
            raise ValueError(f"{variant.config}: ACO variant 与 key 不一致")
        if (
            spec.experiment.train_scales != (100,)
            or spec.experiment.validation_scales != (100,)
        ):
            raise ValueError(f"{variant.config}: 必须纯 TSP100 train/validation")
        if variant.name not in source_variants:
            raise ValueError(f"主 study 不含 variant={variant.name}")
        source_variant = source_variants[variant.name]
        if seeds != source_variant.seeds:
            raise ValueError(f"{variant.name}: seeds 与主 study 不一致")
        source_spec = load_run_spec(source_variant.config)
        if (
            spec.experiment.aco.stable_dict()
            != source_spec.experiment.aco.stable_dict()
        ):
            raise ValueError(f"{variant.name}: ACO 配置与主 study 不一致")
        if asdict(spec.experiment.gp) != asdict(source_spec.experiment.gp):
            raise ValueError(f"{variant.name}: GP 配置与主 study 不一致")
        variants.append(variant)
        all_seeds.extend(seeds)
    if set(source_variants) != {item.name for item in variants}:
        raise ValueError("ablation variants 必须与主 study 完全一致")
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("不同 ACO 变体的 root seeds 必须互异")

    partitions = tuple(str(value) for value in payload["partitions"])
    expected_partitions = {
        "tsp50_uniform",
        "tsp100_uniform",
        "tsp500_uniform",
        "tsp1000_uniform",
        "tsp500_cluster",
        "tsp500_gaussian",
        "tsplib_le500",
    }
    if set(partitions) != expected_partitions:
        raise ValueError(
            "partitions 必须精确覆盖四个 uniform、cluster、Gaussian、"
            "TSPLIB<=500"
        )
    for variant in variants:
        absent = set(partitions) - set(load_run_spec(variant.config).data.test)
        if absent:
            raise ValueError(f"{variant.config}: 缺少 partitions {sorted(absent)}")
    ablation_partitions = tuple(
        str(value) for value in payload["ablation_partitions"]
    )
    if ablation_partitions != ("tsp100_uniform", "tsp500_uniform"):
        raise ValueError(
            "ablation_partitions 必须按顺序精确为 "
            "TSP100 uniform 与 TSP500 uniform"
        )
    if not set(ablation_partitions).issubset(partitions):
        raise ValueError("ablation_partitions 必须是 partitions 的子集")

    phase = str(payload.get("phase", "pilot"))
    if phase != "pilot":
        raise ValueError("本次三种子 ablation phase 必须为 pilot")
    test_seeds = int(payload["test_seeds"])
    if test_seeds != 3:
        raise ValueError("本次 ablation 固定使用三个独立 ACO test seeds")
    bootstrap = int(payload.get("bootstrap_replicates", 10_000))
    if bootstrap < 100:
        raise ValueError("bootstrap_replicates 至少为 100")

    study = AblationStudySpec(
        study_id=str(payload["study_id"]),
        output_root=_project_path(str(payload["output_root"])),
        phase=phase,
        test_root_seed=int(payload["test_root_seed"]),
        test_seeds=test_seeds,
        partitions=partitions,
        ablation_partitions=ablation_partitions,
        methods=methods,
        variants=tuple(variants),
        reuse=reuse,
        manifest=_project_path(str(payload.get("manifest", "Datasets/manifest.json"))),
        bootstrap_replicates=bootstrap,
        source_path=source,
    )
    _validate_reuse_contract(study)
    return study


def _validate_reuse_contract(study: AblationStudySpec) -> None:
    """确认所有要复用的主实验 artifact 完整且未被混用。"""

    source_study = load_study_spec(study.reuse.config)
    if source_study.test_root_seed != study.test_root_seed:
        raise ValueError("ablation 与主 study 的 test_root_seed 不一致")
    if source_study.test_seeds != study.test_seeds:
        raise ValueError("ablation 与主 study 的 test_seeds 不一致")
    for variant in study.variants:
        for root_seed in variant.seeds:
            schedule = (
                study.reuse.root
                / "schedules"
                / f"{variant.name}-seed-{root_seed}.json"
            )
            baseline = (
                study.reuse.root
                / "baselines"
                / variant.name
                / f"{variant.name}-seed-{root_seed}.npz"
            )
            run = (
                study.reuse.root
                / "train"
                / variant.name
                / f"seed-{root_seed}"
            )
            required = (
                schedule,
                baseline,
                run / "selected_candidate.pkl",
                run / "deployment_decision.json",
                run / "training_validation_curve.csv",
                run / "manifest.json",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "主 study 复用 artifact 缺失: " + ", ".join(missing)
                )
        for partition in source_study.partitions:
            records = (
                study.reuse.root
                / "test"
                / variant.name
                / partition
                / "records.csv"
            )
            if not records.is_file():
                raise FileNotFoundError(f"主 study test records 缺失: {records}")


def method_run_path(
    study: AblationStudySpec,
    method: str,
    variant: str,
    root_seed: int,
) -> Path:
    """返回训练或复用方法的 selected-candidate run 目录。"""

    selected = _method(study, method)
    if selected.source == "reuse":
        return study.reuse.root / "train" / variant / f"seed-{root_seed}"
    return (
        study.output_root
        / "train"
        / method
        / variant
        / f"seed-{root_seed}"
    )


def _load_program_entry(
    study: AblationStudySpec,
    *,
    method: str,
    variant: str,
    root_seed: int,
) -> ProgramEntry:
    run = method_run_path(study, method, variant, root_seed)
    champion = load_champion(run / "selected_candidate.pkl")
    transition, pheromone = compile_champion(champion)
    return ProgramEntry(
        method=method,
        gp_root_seed=root_seed,
        champion_id=champion.structural_hash,
        transition=transition,
        pheromone=pheromone,
    )


def _combined_champion_id(
    *,
    method: str,
    root_seed: int,
    transition: Any,
    pheromone: Any,
) -> str:
    payload = "\0".join(
        (
            method,
            str(root_seed),
            "" if transition is None else transition.expression,
            "" if pheromone is None else pheromone.expression,
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def test_methods_for_group(
    study: AblationStudySpec,
    *,
    partition: str,
    group: str,
) -> tuple[str, ...]:
    """返回冻结测试组的方法集合，供任务验证与报告共享。"""

    if group == "final":
        if partition in study.ablation_partitions:
            raise ValueError("final 组不得重复测试核心消融分区")
        return (FINAL_METHOD,)
    if group == "replacement":
        if partition not in study.ablation_partitions:
            raise ValueError("replacement 组仅允许核心消融分区")
        return REPLACEMENT_METHODS
    if group == "residual":
        if partition not in study.ablation_partitions:
            raise ValueError("residual 消融组仅允许核心消融分区")
        source_partitions = set(load_study_spec(study.reuse.config).partitions)
        methods = tuple(
            method
            for method in RESIDUAL_METHODS
            if method != FINAL_METHOD or partition not in source_partitions
        )
        return (*methods, *MECHANISM_METHODS)
    raise ValueError(f"未知 test group: {group}")


def evaluated_methods_for_partition(
    study: AblationStudySpec,
    partition: str,
) -> tuple[str, ...]:
    """返回报告在一个分区中允许出现的方法，防止跨作用域推断。"""

    if partition in study.ablation_partitions:
        return (*CORE_METHODS, *MECHANISM_METHODS)
    return (FINAL_METHOD,)


def _program_entries(
    study: AblationStudySpec,
    *,
    variant_name: str,
    partition: str,
    group: str,
) -> list[ProgramEntry]:
    """构造一个测试组的 selected champions；从不测试 GP population。"""

    variant = _variant(study, variant_name)
    expected_methods = test_methods_for_group(
        study,
        partition=partition,
        group=group,
    )
    if group == "final":
        return [
            _load_program_entry(
                study,
                method=FINAL_METHOD,
                variant=variant_name,
                root_seed=root_seed,
            )
            for root_seed in variant.seeds
        ]
    if group == "replacement":
        return [
            _load_program_entry(
                study,
                method=method,
                variant=variant_name,
                root_seed=root_seed,
            )
            for method in REPLACEMENT_METHODS
            for root_seed in variant.seeds
        ]
    if group != "residual":
        raise AssertionError("test_methods_for_group 已验证 group")

    methods = [
        method
        for method in expected_methods
        if method not in MECHANISM_METHODS
    ]
    entries = [
        _load_program_entry(
            study,
            method=method,
            variant=variant_name,
            root_seed=root_seed,
        )
        for method in methods
        for root_seed in variant.seeds
    ]

    full = {
        root_seed: _load_program_entry(
            study,
            method="rmtgp-full-f1",
            variant=variant_name,
            root_seed=root_seed,
        )
        for root_seed in variant.seeds
    }
    for root_seed in variant.seeds:
        original = full[root_seed]
        for method, transition, pheromone in (
            (
                "rmtgp-full-f1-drop-transition",
                None,
                original.pheromone,
            ),
            (
                "rmtgp-full-f1-drop-pheromone",
                original.transition,
                None,
            ),
        ):
            entries.append(
                ProgramEntry(
                    method=method,
                    gp_root_seed=root_seed,
                    champion_id=_combined_champion_id(
                        method=method,
                        root_seed=root_seed,
                        transition=transition,
                        pheromone=pheromone,
                    ),
                    transition=transition,
                    pheromone=pheromone,
                )
            )
    seeds = variant.seeds
    for rotation in (1, 2):
        method = f"rmtgp-full-f1-shuffle-r{rotation}"
        for index, root_seed in enumerate(seeds):
            transition = full[root_seed].transition
            pheromone = full[seeds[(index + rotation) % len(seeds)]].pheromone
            entries.append(
                ProgramEntry(
                    method=method,
                    gp_root_seed=root_seed,
                    champion_id=_combined_champion_id(
                        method=method,
                        root_seed=root_seed,
                        transition=transition,
                        pheromone=pheromone,
                    ),
                    transition=transition,
                    pheromone=pheromone,
                )
            )
    return entries


def _expected_instances(
    study: AblationStudySpec,
    spec: RunSpec,
    partition: str,
) -> int:
    manifest = load_manifest(study.manifest)
    paths = {path.resolve() for path in spec.data.test_paths(partition)}
    partition_spec = spec.data.test[partition]
    return sum(
        record.instances
        for record in manifest.files
        if record.split == "test"
        and (spec.data.root / record.path).resolve() in paths
        and (
            partition_spec.min_scale is None
            or record.scale >= partition_spec.min_scale
        )
        and (
            partition_spec.max_scale is None
            or record.scale <= partition_spec.max_scale
        )
    )


def _expected_batches(
    study: AblationStudySpec,
    spec: RunSpec,
    partition: str,
) -> int:
    """按城市规模分别计算 batch 数，兼容异构 TSPLIB partition。"""

    manifest = load_manifest(study.manifest)
    paths = {path.resolve() for path in spec.data.test_paths(partition)}
    partition_spec = spec.data.test[partition]
    by_scale: dict[int, int] = defaultdict(int)
    for record in manifest.files:
        if (
            record.split == "test"
            and (spec.data.root / record.path).resolve() in paths
            and (
                partition_spec.min_scale is None
                or record.scale >= partition_spec.min_scale
            )
            and (
                partition_spec.max_scale is None
                or record.scale <= partition_spec.max_scale
            )
        ):
            by_scale[record.scale] += record.instances
    return sum(
        math.ceil(count / spec.data.evaluation_batch_size)
        for count in by_scale.values()
    )


def _baseline_result(
    study: AblationStudySpec,
    *,
    spec: RunSpec,
    variant_name: str,
    partition: str,
    batch_number: int,
    replicate: int,
    seed: int,
    batch: Any,
) -> RunResult:
    """读取主实验 uniform cache，或为新 OOD partition 生成共享 baseline。"""

    source_study = load_study_spec(study.reuse.config)
    if partition in source_study.partitions:
        source_variant = next(
            item for item in source_study.variants if item.name == variant_name
        )
        source_spec = load_run_spec(source_variant.config)
        path = (
            study.reuse.root
            / "test-cache"
            / variant_name
            / partition
            / f"aco-{replicate:02d}"
            / f"batch-{batch_number:04d}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"要求复用的 uniform baseline cache 缺失: {path}")
        expected = _cache_metadata(
            study=source_study,
            spec=source_spec,
            partition=partition,
            batch_number=batch_number,
            replicate=replicate,
            seed=seed,
            batch=batch,
        )
        return _read_test_cache(path, expected_metadata=expected)

    path = (
        study.output_root
        / "test-cache"
        / variant_name
        / partition
        / f"aco-{replicate:02d}"
        / f"batch-{batch_number:04d}.npz"
    )
    expected = _cache_metadata(
        study=study,
        spec=spec,
        partition=partition,
        batch_number=batch_number,
        replicate=replicate,
        seed=seed,
        batch=batch,
    )
    if path.is_file():
        return _read_test_cache(path, expected_metadata=expected)
    result = solve(
        batch,
        spec.experiment.aco,
        seed=seed,
        backend=spec.experiment.runtime.aco_backend,
        runtime=spec.experiment.runtime,
    )
    _write_test_cache(path, result, metadata=expected)
    return result


def _run_result_from_population(
    population: Any,
    index: int,
    *,
    config: Any,
    batch_size: int,
) -> RunResult:
    """把 population 行转换为质量记录入口使用的单 program RunResult。"""

    diagnostic = population.diagnostics[index]
    tours = (
        batch_size
        * config.resolve_ants(population.best_tour.shape[-1] - 1)
        * config.iterations
    )
    return RunResult(
        best_tour=population.best_tour[index],
        best_length=population.best_length[index],
        best_iteration=population.best_iteration[index],
        anytime_best=population.anytime_best[index],
        # campaign wall time 不能诚实地分摊到并行 programs；效率另行孤立测量。
        wall_time_sec=float("nan"),
        constructed_tours=tours,
        diagnostics=RunDiagnostics(
            candidate_fallback_count=int(diagnostic[0].item()),
            uniform_fallback_count=int(diagnostic[1].item()),
            bound_clip_count=int(diagnostic[2].item()),
            mmas_restart_count=int(diagnostic[3].item()),
        ),
        backend_metrics={
            **population.backend_metrics,
            "timing_scope": "batched-quality-campaign",
        },
    )


def _test_shard_valid(
    path: Path,
    *,
    entries: list[ProgramEntry],
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
    expected = len(entries) * batch.batch_size
    allowed = {
        (entry.method, entry.gp_root_seed, entry.champion_id)
        for entry in entries
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
        and {record.partition for record in records} == {partition}
        and {record.seed for record in records} == {seed}
        and {
            (record.method, record.gp_root_seed, record.champion_id)
            for record in records
        }
        == allowed
        and {record.instance_id for record in records} == set(batch.instance_ids)
    )


def _reuse_final_from_legacy_campaign(
    study: AblationStudySpec,
    *,
    variant_name: str,
    partition: str,
    entries: list[ProgramEntry],
    expected_instances: int,
    config: Any,
    output: Path,
) -> bool:
    """从已完成的旧 packed 消融中只读提取 Full-F1，避免重复测试。

    旧 campaign 中每条记录已具有独立 champion、instance 与 ACO seed
    provenance。这里只过滤完整 aggregate；部分 shards 不迁移。
    """

    legacy_output = (
        study.output_root
        / "test"
        / variant_name
        / partition
        / "residual"
    )
    legacy_records_path = legacy_output / "records.csv"
    legacy_manifest_path = legacy_output / "evaluation_manifest.json"
    if not legacy_records_path.is_file() or not legacy_manifest_path.is_file():
        return False
    try:
        legacy_manifest = json.loads(
            legacy_manifest_path.read_text(encoding="utf-8")
        )
        legacy_records = read_records([legacy_records_path])
    except (OSError, TypeError, ValueError):
        return False
    if (
        legacy_manifest.get("status") != "completed"
        or FINAL_METHOD not in set(legacy_manifest.get("methods", []))
        or legacy_manifest.get("test_seeds") != study.test_seeds
        or legacy_manifest.get("instances") != expected_instances
    ):
        return False

    expected_champions = {
        (entry.gp_root_seed, entry.champion_id) for entry in entries
    }
    expected_roots = {root_seed for root_seed, _ in expected_champions}
    selected = [
        record for record in legacy_records if record.method == FINAL_METHOD
    ]
    expected_rows = expected_instances * study.test_seeds * len(entries)
    keys = {
        (
            record.gp_root_seed,
            record.instance_id,
            record.seed,
        )
        for record in selected
    }
    if (
        len(selected) != expected_rows
        or len(keys) != expected_rows
        or {record.gp_root_seed for record in selected} != expected_roots
        or {
            (record.gp_root_seed, record.champion_id)
            for record in selected
        }
        != expected_champions
        or {record.variant for record in selected} != {variant_name}
        or {record.partition for record in selected} != {partition}
    ):
        return False

    selected.sort(
        key=lambda record: (
            record.gp_root_seed,
            record.instance_id,
            record.seed,
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    merged = output / "records.csv"
    write_records(selected, merged)
    _atomic_json(
        output / "evaluation_manifest.json",
        {
            "schema_version": ABLATION_TEST_SCHEMA_VERSION,
            "status": "completed",
            "study_id": study.study_id,
            "variant": variant_name,
            "partition": partition,
            "group": "final",
            "methods": [FINAL_METHOD],
            "programs": len(entries),
            "test_root_seed": study.test_root_seed,
            "test_seeds": study.test_seeds,
            "instances": expected_instances,
            "rows": expected_rows,
            "aco_config_hash": config.config_hash,
            "baseline_behavior_hash": config.baseline_behavior_hash,
            "backend_semantic": backend_semantic_id(
                load_run_spec(_variant(study, variant_name).config)
                .experiment.runtime.aco_backend
            ),
            "campaigns": [],
            "record_provenance": {
                "mode": "filtered-completed-packed-campaign",
                "source_records": str(legacy_records_path),
                "source_records_sha256": _sha256_file(legacy_records_path),
                "source_manifest": str(legacy_manifest_path),
                "source_manifest_sha256": _sha256_file(legacy_manifest_path),
            },
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return True


def evaluate_ablation_group(
    study: AblationStudySpec,
    *,
    variant_name: str,
    partition: str,
    group: str,
) -> Path:
    """批量评测一个 variant×partition×integration group。"""

    variant = _variant(study, variant_name)
    if partition not in study.partitions:
        raise KeyError(f"partition 不在 ablation 合同中: {partition}")
    if group not in TEST_GROUPS:
        raise KeyError(f"未知 integration group: {group}")
    spec = load_run_spec(variant.config)
    configure_runtime(spec.experiment.runtime)
    entries = _program_entries(
        study,
        variant_name=variant_name,
        partition=partition,
        group=group,
    )
    if not entries:
        raise RuntimeError("批量测试 group 不允许为空")
    config = spec.experiment.aco
    if group == "replacement":
        from dataclasses import replace

        config = replace(
            config,
            transition_integration=TransitionIntegration.REPLACEMENT,
        )

    output = study.output_root / "test" / variant_name / partition / group
    merged = output / "records.csv"
    expected_instances = _expected_instances(study, spec, partition)
    expected_rows = expected_instances * study.test_seeds * len(entries)
    expected_methods = sorted({entry.method for entry in entries})
    manifest_path = output / "evaluation_manifest.json"
    if (
        group == "final"
        and not (merged.is_file() and manifest_path.is_file())
        and _reuse_final_from_legacy_campaign(
            study,
            variant_name=variant_name,
            partition=partition,
            entries=entries,
            expected_instances=expected_instances,
            config=config,
            output=output,
        )
    ):
        return merged
    if merged.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = read_records([merged])
        except (OSError, TypeError, ValueError):
            records = []
            manifest = {}
        if (
            manifest.get("status") == "completed"
            and manifest.get("rows") == expected_rows
            and len(records) == expected_rows
            and manifest.get("group") == group
            and manifest.get("methods") == expected_methods
            and manifest.get("programs") == len(entries)
            and manifest.get("test_seeds") == study.test_seeds
        ):
            return merged

    paths = spec.data.test_paths(partition)
    partition_spec = spec.data.test[partition]
    all_records: list[EvaluationRecord] = []
    campaigns: list[dict[str, Any]] = []
    completed_shards = 0
    total_batches = _expected_batches(study, spec, partition)
    total_shards = total_batches * study.test_seeds
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
            shard = (
                output
                / f"aco-{replicate:02d}"
                / f"batch-{batch_number:04d}.csv"
            )
            metrics_path = shard.with_suffix(".metrics.json")
            if _test_shard_valid(
                shard,
                entries=entries,
                batch=batch,
                partition=partition,
                seed=seed,
            ) and metrics_path.is_file():
                records = read_records([shard])
                campaign = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                baseline = _baseline_result(
                    study,
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
                            gp_run_id=(
                                f"{variant_name}:{entry.method}:"
                                f"seed-{entry.gp_root_seed}"
                            ),
                            gp_root_seed=entry.gp_root_seed,
                        )
                    )
                write_records(records, shard)
                campaign = {
                    "schema_version": 1,
                    "variant": variant_name,
                    "partition": partition,
                    "group": group,
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
                f"ablation-test={variant_name}/{partition}/{group} "
                f"shards={completed_shards}/{total_shards} "
                f"elapsed={elapsed / 60.0:.1f}min eta={eta / 60.0:.1f}min",
                flush=True,
            )

    if len(all_records) != expected_rows:
        raise RuntimeError(
            f"{variant_name}/{partition}/{group}: rows={len(all_records)}，"
            f"预期 {expected_rows}"
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
        raise RuntimeError("ablation test records 存在重复配对键")
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
            "schema_version": ABLATION_TEST_SCHEMA_VERSION,
            "status": "completed",
            "study_id": study.study_id,
            "variant": variant_name,
            "partition": partition,
            "group": group,
            "methods": expected_methods,
            "programs": len(entries),
            "test_root_seed": study.test_root_seed,
            "test_seeds": study.test_seeds,
            "instances": expected_instances,
            "rows": expected_rows,
            "aco_config_hash": config.config_hash,
            "baseline_behavior_hash": config.baseline_behavior_hash,
            "backend_semantic": backend_semantic_id(
                spec.experiment.runtime.aco_backend
            ),
            "campaigns": campaigns,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    return merged


def benchmark_ablation_efficiency(
    study: AblationStudySpec,
    *,
    variant_name: str,
    output: str | Path,
) -> Path:
    """在固定小批量上孤立测量各方法推理耗时，质量测试不分摊 campaign 时间。"""

    variant = _variant(study, variant_name)
    spec = load_run_spec(variant.config)
    configure_runtime(spec.experiment.runtime)
    target = Path(output)
    shard_root = target.parent / f"{variant_name}-shards"
    batch_sizes = {
        "tsp50_uniform": 32,
        "tsp100_uniform": 32,
        "tsp500_uniform": 8,
        "tsp1000_uniform": 4,
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
                f"efficiency:{partition}",
                0,
                replicate,
            )
            for replicate in range(3)
        ]
        # 预热 kernel、CUDA context 与 resident problem；预热不进入统计。
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

        for method in EFFICIENCY_METHODS:
            shard = shard_root / method / f"{partition}.json"
            if shard.is_file():
                try:
                    payload = json.loads(shard.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    payload = {}
                if (
                    payload.get("status") == "completed"
                    and payload.get("method") == method
                    and payload.get("partition") == partition
                    and len(payload.get("rows", [])) == 9
                ):
                    all_rows.extend(payload["rows"])
                    continue

            method_rows: list[dict[str, Any]] = []
            config = spec.experiment.aco
            if method in REPLACEMENT_METHODS:
                from dataclasses import replace

                config = replace(
                    config,
                    transition_integration=TransitionIntegration.REPLACEMENT,
                )
            for root_seed in variant.seeds:
                entry = _load_program_entry(
                    study,
                    method=method,
                    variant=variant_name,
                    root_seed=root_seed,
                )
                # 每个 program 先运行一次，排除按 program packing 的首次开销。
                solve(
                    batch,
                    config,
                    transition_program=entry.transition,
                    pheromone_program=entry.pheromone,
                    seed=seeds[0],
                    backend=spec.experiment.runtime.aco_backend,
                    runtime=spec.experiment.runtime,
                )
                for replicate, seed in enumerate(seeds):
                    result = solve(
                        batch,
                        config,
                        transition_program=entry.transition,
                        pheromone_program=entry.pheromone,
                        seed=seed,
                        backend=spec.experiment.runtime.aco_backend,
                        runtime=spec.experiment.runtime,
                    )
                    baseline_time = baseline_times[seed]
                    method_rows.append(
                        {
                            "variant": variant_name,
                            "method": method,
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
                                result.constructed_tours
                                / max(baseline_time, 1e-12)
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
                    "method": method,
                    "partition": partition,
                    "rows": method_rows,
                },
            )
            all_rows.extend(method_rows)
            print(
                f"efficiency={variant_name}/{partition}/{method} completed",
                flush=True,
            )

    summaries: list[dict[str, Any]] = []
    for method in EFFICIENCY_METHODS:
        for partition in batch_sizes:
            rows = [
                row
                for row in all_rows
                if row["method"] == method and row["partition"] == partition
            ]
            summaries.append(
                {
                    "variant": variant_name,
                    "method": method,
                    "partition": partition,
                    "scale": rows[0]["scale"],
                    "observations": len(rows),
                    "median_wall_time_sec": median(
                        row["wall_time_sec"] for row in rows
                    ),
                    "median_baseline_wall_time_sec": median(
                        row["baseline_wall_time_sec"] for row in rows
                    ),
                    "median_overhead_percent": median(
                        row["overhead_percent"] for row in rows
                    ),
                    "median_tours_per_second": median(
                        row["tours_per_second"] for row in rows
                    ),
                }
            )
    _atomic_json(
        target,
        {
            "schema_version": 2,
            "status": "completed",
            "study_id": study.study_id,
            "variant": variant_name,
            "methods": list(EFFICIENCY_METHODS),
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
    manifest_path = path / "manifest.json"
    required = (
        path / "selected_candidate.pkl",
        path / "deployment_decision.json",
        path / "training_validation_curve.csv",
        path / "cpu_fp64_audit_summary.csv",
    )
    if not manifest_path.is_file() or not all(item.is_file() for item in required):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    configuration = manifest.get("configuration", {})
    return (
        manifest.get("status") == "completed"
        and int(configuration.get("root_seed", -1)) == root_seed
        and str(configuration.get("experiment_id", "")).endswith(f"-{method}")
    )


def build_ablation_tasks(study: AblationStudySpec) -> list[AblationTask]:
    """构造训练、核心消融与主方法最终测试的冻结任务矩阵。"""

    python = sys.executable
    tasks: list[AblationTask] = []
    for variant in study.variants:
        root_seed = variant.seeds[0]
        schedule = (
            study.reuse.root
            / "schedules"
            / f"{variant.name}-seed-{root_seed}.json"
        )
        baseline = study.reuse.root / "baselines" / variant.name
        for method in CORE_METHODS:
            artifact = (
                study.output_root
                / "preflight"
                / variant.name
                / f"{method}.json"
            )
            tasks.append(
                AblationTask(
                    task_id=f"preflight-{variant.name}-{method}",
                    command=(
                        python,
                        "-m",
                        "rmtgp_aco",
                        "benchmark-training",
                        "--config",
                        str(variant.config),
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
                    ),
                )
            )

    max_replicates = max(len(variant.seeds) for variant in study.variants)
    for replicate in range(max_replicates):
        for variant in study.variants:
            root_seed = variant.seeds[replicate]
            schedule = (
                study.reuse.root
                / "schedules"
                / f"{variant.name}-seed-{root_seed}.json"
            )
            baseline = study.reuse.root / "baselines" / variant.name
            for method in TRAINED_METHODS:
                run = method_run_path(
                    study,
                    method,
                    variant.name,
                    root_seed,
                )
                command = [
                    python,
                    "-m",
                    "rmtgp_aco",
                    "train",
                    "--config",
                    str(variant.config),
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
                        task_id=(
                            f"train-{variant.name}-{method}-{root_seed}"
                        ),
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

    source_partitions = set(load_study_spec(study.reuse.config).partitions)
    test_scopes = [
        (partition, group)
        for partition in study.ablation_partitions
        for group in ABLATION_TEST_GROUPS
    ]
    test_scopes.extend(
        (partition, "final")
        for partition in study.partitions
        if partition not in source_partitions
    )
    for variant in study.variants:
        for partition, group in test_scopes:
            artifact = (
                study.output_root
                / "test"
                / variant.name
                / partition
                / group
                / "records.csv"
            )
            tasks.append(
                AblationTask(
                    task_id=f"test-{variant.name}-{partition}-{group}",
                    command=(
                        python,
                        "-m",
                        "rmtgp_aco",
                        "evaluate-ablation-study",
                        "--study-config",
                        str(study.source_path),
                        "--variant",
                        variant.name,
                        "--partition",
                        partition,
                        "--group",
                        group,
                    ),
                    artifact=artifact,
                    kind="test",
                    metadata=(
                        ("variant", variant.name),
                        ("partition", partition),
                        ("group", group),
                    ),
                )
            )

    for variant in study.variants:
        artifact = (
            study.output_root
            / "efficiency"
            / f"{variant.name}.json"
        )
        tasks.append(
            AblationTask(
                task_id=f"efficiency-{variant.name}",
                command=(
                    python,
                    "-m",
                    "rmtgp_aco",
                    "benchmark-ablation-efficiency",
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
                "report-ablation-study",
                "--study-config",
                str(study.source_path),
            ),
            artifact=study.output_root / "report" / "ablation_report.md",
            kind="report",
        )
    )
    return tasks


def _task_complete(task: AblationTask, study: AblationStudySpec) -> bool:
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
        expected_methods = sorted(
            test_methods_for_group(
                study,
                partition=meta["partition"],
                group=meta["group"],
            )
        )
        return (
            payload.get("status") == "completed"
            and payload.get("variant") == meta["variant"]
            and payload.get("partition") == meta["partition"]
            and payload.get("group") == meta["group"]
            and payload.get("methods") == expected_methods
            and payload.get("programs")
            == len(expected_methods) * len(_variant(study, meta["variant"]).seeds)
            and payload.get("test_seeds") == study.test_seeds
        )
    if task.kind == "efficiency":
        if not task.artifact.is_file():
            return False
        try:
            payload = json.loads(task.artifact.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            payload.get("status") == "completed"
            and payload.get("variant") == meta["variant"]
            and payload.get("methods") == list(EFFICIENCY_METHODS)
        )
    if task.kind == "report":
        return task.artifact.is_file()
    return False


def _reuse_manifest(study: AblationStudySpec) -> dict[str, Any]:
    """记录实际读取的主实验关键文件哈希，避免只保存路径。"""

    files: list[Path] = [
        study.reuse.config,
        study.reuse.root / "study_state.json",
        study.reuse.root / "report" / "study_summary.json",
    ]
    for variant in study.variants:
        for root_seed in variant.seeds:
            files.extend(
                (
                    study.reuse.root
                    / "schedules"
                    / f"{variant.name}-seed-{root_seed}.json",
                    study.reuse.root
                    / "baselines"
                    / variant.name
                    / f"{variant.name}-seed-{root_seed}.npz",
                    study.reuse.root
                    / "train"
                    / variant.name
                    / f"seed-{root_seed}"
                    / "selected_candidate.pkl",
                    study.reuse.root
                    / "train"
                    / variant.name
                    / f"seed-{root_seed}"
                    / "deployment_decision.json",
                )
            )
        for partition in load_study_spec(study.reuse.config).partitions:
            files.append(
                study.reuse.root
                / "test"
                / variant.name
                / partition
                / "records.csv"
            )
    return {
        "schema_version": 1,
        "source_study": str(study.reuse.root),
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in files
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _state_payload(
    *,
    study: AblationStudySpec,
    status: str,
    current_task: str | None,
    completed: list[str],
    total_tasks: int,
    started_at: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ABLATION_STATE_SCHEMA_VERSION,
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


def _parallel_task_units(
    tasks: list[AblationTask],
    *,
    kind: str,
) -> list[tuple[AblationTask, ...]]:
    """生成双卡单元；核心消融成对，最终主方法任务单独调度。"""

    selected = [task for task in tasks if task.kind == kind]
    if kind != "test":
        return [(task,) for task in selected]

    grouped: dict[tuple[str, str], list[AblationTask]] = {}
    for task in selected:
        meta = task.meta()
        key = (meta["variant"], meta["partition"])
        grouped.setdefault(key, []).append(task)
    units: list[tuple[AblationTask, ...]] = []
    for key, group_tasks in grouped.items():
        by_group = {task.meta()["group"]: task for task in group_tasks}
        groups = set(by_group)
        if groups == set(ABLATION_TEST_GROUPS):
            order = ABLATION_TEST_GROUPS
        elif groups == {"final"}:
            order = ("final",)
        else:
            raise RuntimeError(
                f"测试调度单元 {key} 的 groups 非法: {sorted(groups)}"
            )
        units.append(tuple(by_group[group] for group in order))
    return units


def _physical_gpu_description(physical_device: int) -> dict[str, Any]:
    """读取物理 GPU 身份，并验证单卡 CUDA 映射可用。"""

    if physical_device < 0:
        raise ValueError("物理 GPU index 不得为负数")
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_device),
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",", 3)]
    if len(fields) != 4:
        raise RuntimeError(
            f"无法解析物理 GPU{physical_device} 的 nvidia-smi 输出"
        )
    probe_env = os.environ.copy()
    probe_env["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import cupy as cp;"
                "assert cp.cuda.runtime.getDeviceCount() == 1;"
                "cp.zeros(1).sum().item()"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=probe_env,
        timeout=30,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        raise RuntimeError(
            f"物理 GPU{physical_device} 的单卡 CUDA probe 失败: {detail}"
        )
    return {
        "physical_device": physical_device,
        "logical_device": 0,
        "logical_devices": 1,
        "device_name": fields[1],
        "uuid": fields[2],
        "memory_total_mib": int(fields[3]),
    }


def _execute_ablation_task(
    task: AblationTask,
    study: AblationStudySpec,
    *,
    physical_device: int | None,
) -> None:
    """在指定物理 GPU 上执行并验证一个原子任务。"""

    if (
        physical_device is not None
        and task.kind in {"preflight", "train", "test", "efficiency"}
    ):
        contention = _external_gpu_processes(physical_device)
        if contention:
            raise RuntimeError(
                f"任务 {task.task_id} 启动前 GPU{physical_device} "
                f"出现外部进程: {contention}"
            )
    log = study.output_root / "logs" / f"{task.task_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if physical_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    with log.open("a", encoding="utf-8") as handle:
        gpu_label = (
            f" gpu={physical_device}" if physical_device is not None else ""
        )
        handle.write(
            f"\n[{datetime.now(UTC).isoformat()}] START{gpu_label} "
            + " ".join(task.command)
            + "\n"
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
        handle.write(
            f"[{datetime.now(UTC).isoformat()}] EXIT {result.returncode}\n"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"任务 {task.task_id} 失败（exit={result.returncode}），详见 {log}"
        )
    if not _task_complete(task, study):
        raise RuntimeError(f"任务 {task.task_id} 未产生完整 artifact")


def run_ablation_parallel(
    study: AblationStudySpec,
    *,
    physical_devices: tuple[int, ...],
) -> None:
    """在多张物理 GPU 上按依赖阶段并行执行并恢复完整消融队列。"""

    if not physical_devices:
        raise ValueError("并行 ablation 至少需要一张物理 GPU")
    if len(set(physical_devices)) != len(physical_devices):
        raise ValueError("并行 ablation 的物理 GPU index 不得重复")
    repository = git_state(Path.cwd())
    if repository["dirty"]:
        raise RuntimeError("ablation 启动前 Git worktree 必须 clean")
    free_bytes = shutil.disk_usage(Path.cwd()).free
    if free_bytes < 50 * 1024**3:
        raise RuntimeError("共享文件系统剩余空间不足 50 GiB，拒绝启动")

    gpus: list[dict[str, Any]] = []
    for physical_device in physical_devices:
        contention = _external_gpu_processes(physical_device)
        if contention:
            raise RuntimeError(
                f"物理 GPU{physical_device} 存在外部计算进程: {contention}"
            )
        gpus.append(_physical_gpu_description(physical_device))

    study.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = study.output_root / "study.lock"
    state_path = study.output_root / "study_state.json"
    pid_path = study.output_root / "runner.pid"
    reuse_path = study.output_root / "reuse_manifest.json"
    if not reuse_path.is_file():
        _atomic_json(reuse_path, _reuse_manifest(study))
    tasks = build_ablation_tasks(study)
    task_order = {task.task_id: index for index, task in enumerate(tasks)}
    started_at = datetime.now(UTC).isoformat()

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有 ablation runner 持有锁") from exc
        pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

        completed = {
            task.task_id for task in tasks if _task_complete(task, study)
        }
        workers: dict[str, dict[str, Any]] = {
            str(device): {
                "physical_device": device,
                "status": "idle",
                "current_task": None,
                "last_completed_task": None,
            }
            for device in physical_devices
        }
        state_lock = Lock()
        phase = "initializing"

        def write_state(
            *,
            status: str = "running",
            error: str | None = None,
            completed_at: str | None = None,
        ) -> None:
            active = [
                str(worker["current_task"])
                for worker in workers.values()
                if worker["current_task"] is not None
            ]
            ordered_completed = sorted(
                completed,
                key=lambda task_id: task_order[task_id],
            )
            payload: dict[str, Any] = {
                **_state_payload(
                    study=study,
                    status=status,
                    current_task=active[0] if len(active) == 1 else None,
                    completed=ordered_completed,
                    total_tasks=len(tasks),
                    started_at=started_at,
                    error=error,
                ),
                "mode": "parallel",
                "phase": phase,
                "current_tasks": active,
                "workers": workers,
                "gpus": gpus,
            }
            if completed_at is not None:
                payload["completed_at"] = completed_at
            _atomic_json(state_path, payload)

        write_state()
        try:
            for phase_kind in ("preflight", "train", "test", "efficiency"):
                phase = phase_kind
                units = [
                    unit
                    for unit in _parallel_task_units(tasks, kind=phase_kind)
                    if not all(
                        task.task_id in completed
                        or _task_complete(task, study)
                        for task in unit
                    )
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
                                if task.task_id in completed or _task_complete(
                                    task, study
                                ):
                                    with state_lock:
                                        completed.add(task.task_id)
                                    continue
                                if stop_event.is_set():
                                    return
                                with state_lock:
                                    workers[key]["status"] = "running"
                                    workers[key]["current_task"] = task.task_id
                                    write_state()
                                _execute_ablation_task(
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
                                workers[key]["error"] = (
                                    f"{type(exc).__name__}: {exc}"
                                )
                                stop_event.set()
                        finally:
                            work_queue.task_done()

                threads = [
                    Thread(
                        target=worker_loop,
                        args=(physical_device,),
                        name=f"ablation-gpu-{physical_device}",
                        daemon=False,
                    )
                    for physical_device in physical_devices
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                if errors:
                    raise RuntimeError(
                        "；".join(
                            f"{type(error).__name__}: {error}"
                            for error in errors
                        )
                    )

            phase = "report"
            report = next(task for task in tasks if task.kind == "report")
            if not _task_complete(report, study):
                with state_lock:
                    workers[str(physical_devices[0])]["status"] = "running"
                    workers[str(physical_devices[0])][
                        "current_task"
                    ] = report.task_id
                    write_state()
                _execute_ablation_task(
                    report,
                    study,
                    physical_device=None,
                )
            with state_lock:
                completed.add(report.task_id)
                for worker in workers.values():
                    worker["status"] = "completed"
                    worker["current_task"] = None
                phase = "completed"
                write_state(
                    status="completed",
                    completed_at=datetime.now(UTC).isoformat(),
                )
        except BaseException as exc:
            with state_lock:
                phase = "failed"
                write_state(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            raise


def run_ablation_queue(study: AblationStudySpec) -> None:
    """在唯一可见物理 GPU 上顺序执行、验证并恢复完整消融队列。"""

    gpu = _assert_single_gpu()
    physical_device = int(gpu["physical_device"])
    contention = _external_gpu_processes(physical_device)
    if contention:
        raise RuntimeError(
            f"物理 GPU{physical_device} 存在外部计算进程: {contention}"
        )
    repository = git_state(Path.cwd())
    if repository["dirty"]:
        raise RuntimeError("ablation 启动前 Git worktree 必须 clean")
    free_bytes = shutil.disk_usage(Path.cwd()).free
    if free_bytes < 50 * 1024**3:
        raise RuntimeError("共享文件系统剩余空间不足 50 GiB，拒绝启动")

    study.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = study.output_root / "study.lock"
    state_path = study.output_root / "study_state.json"
    pid_path = study.output_root / "runner.pid"
    reuse_path = study.output_root / "reuse_manifest.json"
    if not reuse_path.is_file():
        _atomic_json(reuse_path, _reuse_manifest(study))
    tasks = build_ablation_tasks(study)
    started_at = datetime.now(UTC).isoformat()
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有 ablation runner 持有锁") from exc
        pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        completed: list[str] = []
        _atomic_json(
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
                if task.kind in {"preflight", "train", "test", "efficiency"}:
                    contention = _external_gpu_processes(physical_device)
                    if contention:
                        raise RuntimeError(
                            f"任务 {task.task_id} 启动前 "
                            f"GPU{physical_device} 出现外部进程: "
                            f"{contention}"
                        )
                _atomic_json(
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
                        f"[{datetime.now(UTC).isoformat()}] "
                        f"EXIT {result.returncode}\n"
                    )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"任务 {task.task_id} 失败（exit={result.returncode}），"
                        f"详见 {log}"
                    )
                if not _task_complete(task, study):
                    raise RuntimeError(f"任务 {task.task_id} 未产生完整 artifact")
                completed.append(task.task_id)
            _atomic_json(
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
            _atomic_json(
                state_path,
                {
                    **_state_payload(
                        study=study,
                        status="failed",
                        current_task=next(
                            (
                                task.task_id
                                for task in tasks
                                if task.task_id not in completed
                            ),
                            None,
                        ),
                        completed=completed,
                        total_tasks=len(tasks),
                        started_at=started_at,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    "gpu": gpu,
                },
            )
            raise


def ablation_status(study: AblationStudySpec) -> dict[str, Any]:
    """返回状态与当前训练代数，供非阻塞监控。"""

    state_path = study.output_root / "study_state.json"
    state: dict[str, Any] = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {
            "study_id": study.study_id,
            "status": "not-started",
            "completed_count": 0,
            "total_tasks": len(build_ablation_tasks(study)),
        }
    )
    current_tasks = state.get("current_tasks")
    if not isinstance(current_tasks, list):
        current = state.get("current_task")
        current_tasks = [current] if isinstance(current, str) else []
    progress: dict[str, Any] = {}
    task_by_id = {
        task.task_id: task for task in build_ablation_tasks(study)
    }
    for current in current_tasks:
        if not isinstance(current, str) or not current.startswith("train-"):
            continue
        task = task_by_id.get(current)
        if task is None:
            continue
        metrics = task.artifact / "training_metrics.jsonl"
        if metrics.is_file():
            lines = [
                line
                for line in metrics.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines:
                latest = json.loads(lines[-1])
                progress[current] = {
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
    if progress:
        state["training_progress"] = progress
    if (study.output_root / "runner.pid").is_file():
        pid = int(
            (study.output_root / "runner.pid")
            .read_text(encoding="utf-8")
            .strip()
        )
        state["pid"] = pid
        try:
            os.kill(pid, 0)
            state["process_alive"] = True
        except OSError:
            state["process_alive"] = False
    return state


def export_ablation_contract(study: AblationStudySpec) -> dict[str, Any]:
    """输出报告可嵌入的纯 JSON 合同。"""

    return {
        "study_id": study.study_id,
        "output_root": str(study.output_root),
        "phase": study.phase,
        "test_root_seed": study.test_root_seed,
        "test_seeds": study.test_seeds,
        "partitions": list(study.partitions),
        "ablation_partitions": list(study.ablation_partitions),
        "champion_selection": {
            "unit": "one_validation_selected_candidate_per_gp_run",
            "gp_runs_per_method": 3,
            "best_seed_selection": False,
        },
        "methods": [asdict(method) for method in study.methods],
        "variants": [
            {
                "name": variant.name,
                "config": str(variant.config),
                "seeds": list(variant.seeds),
            }
            for variant in study.variants
        ],
        "reuse": {
            "config": str(study.reuse.config),
            "root": str(study.reuse.root),
        },
        "manifest": str(study.manifest),
        "bootstrap_replicates": study.bootstrap_replicates,
    }
