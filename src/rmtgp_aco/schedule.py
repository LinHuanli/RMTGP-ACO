"""Protocol A 的紧凑、可审计 instance/seed schedule。

Schedule 只保存实际使用的逻辑索引，不保存百万级 permutation。所有方法可
共享同一 manifest，从而在 GP replicate 之间实施 paired common random
numbers，并使 baseline 能在训练前完整预计算。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import gcd
from pathlib import Path

import numpy as np
import torch

from .data import make_problem_batch
from .sampling import (
    EvaluationCase,
    ScalePool,
)

SCHEDULE_SCHEMA_VERSION = 1
_PHASE_CODE = {
    "pilot": 0x50494C,
    "formal": 0x464F52,
    "development": 0x444556,
    "test": 0x544553,
}


@dataclass(frozen=True, slots=True)
class ScheduleRecord:
    """一个 generation 或 validation role 的确定性数据选择。"""

    split: str
    role: str
    replicate_id: int
    generation: int | None
    scale: int
    logical_indices: tuple[int, ...]
    instance_ids: tuple[str, ...]
    coordinate_hashes: tuple[str, ...]
    aco_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        count = len(self.logical_indices)
        if count < 1:
            raise ValueError("schedule record 不允许空实例集")
        if len(self.instance_ids) != count or len(self.coordinate_hashes) != count:
            raise ValueError("schedule 索引、instance ID 与 coordinate hash 数量不一致")
        if not self.aco_seeds:
            raise ValueError("schedule record 至少需要一个 ACO seed")
        if self.split == "train" and self.generation is None:
            raise ValueError("train schedule record 必须包含 generation")
        if self.split != "train" and self.generation is not None:
            raise ValueError("非 train schedule record 不应包含 generation")


@dataclass(frozen=True, slots=True)
class ScheduleManifest:
    """一个 protocol phase/replicate 的全部 train 与 validation 选择。"""

    schema_version: int
    protocol_id: str
    phase: str
    root_seed: int
    generated_at: str
    records: tuple[ScheduleRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEDULE_SCHEMA_VERSION:
            raise ValueError(
                f"不支持 schedule schema={self.schema_version}，"
                f"期望 {SCHEDULE_SCHEMA_VERSION}"
            )
        if self.phase not in {"pilot", "formal", "development", "test"}:
            raise ValueError(f"未知 schedule phase: {self.phase}")
        if not self.records:
            raise ValueError("schedule manifest 不允许为空")

    def stable_dict(self) -> dict[str, object]:
        """返回排除生成时间的可哈希内容。"""

        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "phase": self.phase,
            "root_seed": self.root_seed,
            "records": [asdict(record) for record in self.records],
        }

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(
            self.stable_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = self.stable_dict()
        payload["generated_at"] = self.generated_at
        payload["manifest_hash"] = self.manifest_hash
        return payload


def _materialise_selection(
    pool: ScalePool,
    indices: np.ndarray,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    instances = [pool.get(int(index)) for index in indices]
    return (
        tuple(instance.instance_id for instance in instances),
        tuple(instance.coordinate_hash for instance in instances),
    )


def _validation_seeds(
    root_seed: int,
    *,
    phase: str,
    replicate_id: int,
    role: str,
    scale: int,
    count: int,
) -> tuple[int, ...]:
    role_code = {
        "selection": 0x53454C,
        "gate": 0x474154,
    }[role]
    return tuple(
        int(
            np.random.default_rng(
                np.random.SeedSequence(
                    [
                        root_seed,
                        replicate_id,
                        _PHASE_CODE[phase],
                        scale,
                        role_code,
                        seed_index,
                        0x41434F,
                    ]
                )
            ).integers(0, 2**63 - 1)
        )
        for seed_index in range(count)
    )


def _phase_position_bounds(pool_size: int, phase: str) -> tuple[int, int]:
    """把 pilot/development 与 formal 放入两个严格不相交的索引域。"""

    if phase not in _PHASE_CODE:
        raise ValueError(f"未知 schedule phase: {phase}")
    midpoint = pool_size // 2
    # development/test 只能查看 pilot 半区，formal 独占另一半。
    return (midpoint, pool_size) if phase == "formal" else (0, midpoint)


def _affine_permutation_parameters(pool_size: int, scale: int) -> tuple[int, int]:
    """构造无需保存全排列的确定性仿射置换。"""

    if pool_size < 2:
        raise ValueError("phase-isolated schedule 要求每个 pool 至少有两个实例")
    multiplier = (0x9E3779B97F4A7C15 ^ (scale * 0x85EBCA6B)) % pool_size
    if multiplier == 0:
        multiplier = 1
    while gcd(multiplier, pool_size) != 1:
        multiplier = (multiplier + 1) % pool_size
        if multiplier == 0:
            multiplier = 1
    offset = (scale * 0xC2B2AE35 + 0x27D4EB2F) % pool_size
    return multiplier, offset


def _permuted_indices(
    positions: np.ndarray,
    *,
    pool_size: int,
    scale: int,
) -> np.ndarray:
    multiplier, offset = _affine_permutation_parameters(pool_size, scale)
    # Python int 中间值避免 int64 乘法溢出；每个正式 run 仅转换约 2,000 项。
    return np.asarray(
        [
            (multiplier * int(position) + offset) % pool_size
            for position in positions
        ],
        dtype=np.int64,
    )


def _draw_unique_positions(
    rng: np.random.Generator,
    *,
    lower: int,
    upper: int,
    count: int,
    used: set[int],
) -> np.ndarray:
    """从 phase 域紧凑无放回抽样，不建立百万长度 permutation。"""

    domain_size = upper - lower
    if count < 1:
        raise ValueError("schedule 每次抽样数必须为正整数")
    if len(used) + count > domain_size:
        raise ValueError(
            f"phase 域仅有 {domain_size} 个实例，累计请求 "
            f"{len(used) + count} 个"
        )
    selected: list[int] = []
    while len(selected) < count:
        value = int(rng.integers(lower, upper))
        if value not in used and value not in selected:
            selected.append(value)
    used.update(selected)
    return np.asarray(selected, dtype=np.int64)


def _train_seed(
    root_seed: int,
    *,
    phase: str,
    replicate_id: int,
    generation: int,
    scale: int,
) -> int:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                root_seed,
                replicate_id,
                _PHASE_CODE[phase],
                generation,
                scale,
                0x41434F,
            ]
        )
    )
    return int(rng.integers(0, 2**63 - 1))


def _training_seeds(
    root_seed: int,
    *,
    phase: str,
    replicate_id: int,
    generation: int,
    scale: int,
    count: int,
) -> tuple[int, ...]:
    """生成训练 seed 前缀；第一个值与 schema v1 历史 schedule 一致。"""

    if count < 1:
        raise ValueError("training seed 数必须为正整数")
    first = _train_seed(
        root_seed,
        phase=phase,
        replicate_id=replicate_id,
        generation=generation,
        scale=scale,
    )
    values = [first]
    for seed_index in range(1, count):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    root_seed,
                    replicate_id,
                    _PHASE_CODE[phase],
                    generation,
                    scale,
                    0x41434F,
                    seed_index,
                ]
            )
        )
        values.append(int(rng.integers(0, 2**63 - 1)))
    return tuple(values)


def build_protocol_schedule(
    training_pools: Mapping[int, ScalePool],
    validation_pools: Mapping[int, ScalePool],
    *,
    protocol_id: str,
    phase: str,
    root_seed: int,
    replicate_id: int,
    generations: int,
    train_instances_per_scale: int,
    validation_selection_instances_per_scale: int,
    validation_gate_instances_per_scale: int,
    validation_seeds: int,
    training_seeds: int = 1,
) -> ScheduleManifest:
    """建立 mixed-scale train + 独立 selection/gate schedule。"""

    if generations < 1 or validation_seeds < 1 or training_seeds < 1:
        raise ValueError("generations、training_seeds 和 validation_seeds 必须为正整数")
    if phase not in _PHASE_CODE:
        raise ValueError(f"未知 schedule phase: {phase}")
    train_rngs = {
        scale: np.random.default_rng(
            np.random.SeedSequence(
                [
                    root_seed,
                    replicate_id,
                    _PHASE_CODE[phase],
                    scale,
                    0x53485546,
                ]
            )
        )
        for scale in training_pools
    }
    train_used = {scale: set() for scale in training_pools}
    records: list[ScheduleRecord] = []
    for generation in range(1, generations + 1):
        for scale, pool in sorted(training_pools.items()):
            lower, upper = _phase_position_bounds(len(pool), phase)
            positions = _draw_unique_positions(
                train_rngs[scale],
                lower=lower,
                upper=upper,
                count=train_instances_per_scale,
                used=train_used[scale],
            )
            indices = _permuted_indices(
                positions,
                pool_size=len(pool),
                scale=scale,
            )
            instance_ids, coordinate_hashes = _materialise_selection(
                training_pools[scale],
                indices,
            )
            records.append(
                ScheduleRecord(
                    split="train",
                    role="generation",
                    replicate_id=replicate_id,
                    generation=generation,
                    scale=scale,
                    logical_indices=tuple(int(index) for index in indices),
                    instance_ids=instance_ids,
                    coordinate_hashes=coordinate_hashes,
                    aco_seeds=_training_seeds(
                        root_seed,
                        phase=phase,
                        replicate_id=replicate_id,
                        generation=generation,
                        scale=scale,
                        count=training_seeds,
                    ),
                )
            )

    for scale, pool in sorted(validation_pools.items()):
        total = (
            validation_selection_instances_per_scale
            + validation_gate_instances_per_scale
        )
        if total > len(pool):
            raise ValueError(
                f"scale={scale} validation 需要 {total} 个实例，pool 仅 {len(pool)}"
            )
        lower, upper = _phase_position_bounds(len(pool), phase)
        if total > upper - lower:
            raise ValueError(
                f"scale={scale} validation phase 域需要 {total} 个实例，"
                f"可用 {upper - lower}"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    root_seed,
                    replicate_id,
                    _PHASE_CODE[phase],
                    scale,
                    0x56414C,
                    0x53504C49,
                ]
            )
        )
        positions = _draw_unique_positions(
            rng,
            lower=lower,
            upper=upper,
            count=total,
            used=set(),
        )
        selected = _permuted_indices(
            positions,
            pool_size=len(pool),
            scale=scale,
        )
        role_slices = {
            "selection": selected[:validation_selection_instances_per_scale],
            "gate": selected[validation_selection_instances_per_scale:],
        }
        for role, indices in role_slices.items():
            indices = np.sort(indices)
            instance_ids, coordinate_hashes = _materialise_selection(pool, indices)
            records.append(
                ScheduleRecord(
                    split="validation",
                    role=role,
                    replicate_id=replicate_id,
                    generation=None,
                    scale=scale,
                    logical_indices=tuple(int(index) for index in indices),
                    instance_ids=instance_ids,
                    coordinate_hashes=coordinate_hashes,
                    aco_seeds=_validation_seeds(
                        root_seed,
                        phase=phase,
                        replicate_id=replicate_id,
                        role=role,
                        scale=scale,
                        count=validation_seeds,
                    ),
                )
            )

    return ScheduleManifest(
        schema_version=SCHEDULE_SCHEMA_VERSION,
        protocol_id=protocol_id,
        phase=phase,
        root_seed=root_seed,
        generated_at=datetime.now(UTC).isoformat(),
        records=tuple(records),
    )


def validate_schedule_contract(
    manifest: ScheduleManifest,
    *,
    protocol_id: str,
    phase: str,
    root_seed: int,
    replicate_id: int,
    generations: int,
    train_scales: tuple[int, ...],
    validation_scales: tuple[int, ...],
    train_instances_per_scale: int,
    validation_selection_instances_per_scale: int,
    validation_gate_instances_per_scale: int,
    validation_seeds: int,
    training_seeds: int = 1,
) -> None:
    """验证冻结 schedule 与本次运行的全部科研合同完全一致。"""

    if manifest.protocol_id != protocol_id:
        raise ValueError(
            f"schedule protocol_id={manifest.protocol_id}，期望 {protocol_id}"
        )
    if manifest.phase != phase:
        raise ValueError(f"schedule phase={manifest.phase}，期望 {phase}")
    if manifest.root_seed != root_seed:
        raise ValueError(
            f"schedule root_seed={manifest.root_seed}，期望 {root_seed}"
        )
    selected = [
        record
        for record in manifest.records
        if record.replicate_id == replicate_id
    ]
    if len(selected) != len(manifest.records):
        raise ValueError("一个 schedule 文件只能包含当前 replicate 的 records")

    train_records = {
        (record.generation, record.scale): record
        for record in selected
        if record.split == "train"
    }
    expected_train_keys = {
        (generation, scale)
        for generation in range(1, generations + 1)
        for scale in train_scales
    }
    if set(train_records) != expected_train_keys:
        raise ValueError("schedule 的 generation×train-scale 网格不完整或含额外项")
    for record in train_records.values():
        if (
            record.role != "generation"
            or len(record.logical_indices) != train_instances_per_scale
            or len(record.aco_seeds) != training_seeds
        ):
            raise ValueError("schedule train record 的 role/count/seed 数不符合配置")

    validation_records = {
        (record.role, record.scale): record
        for record in selected
        if record.split == "validation"
    }
    expected_validation_keys = {
        (role, scale)
        for role in ("selection", "gate")
        for scale in validation_scales
    }
    if set(validation_records) != expected_validation_keys:
        raise ValueError("schedule 的 validation role×scale 网格不完整或含额外项")
    expected_counts = {
        "selection": validation_selection_instances_per_scale,
        "gate": validation_gate_instances_per_scale,
    }
    for record in validation_records.values():
        if (
            len(record.logical_indices) != expected_counts[record.role]
            or len(record.aco_seeds) != validation_seeds
        ):
            raise ValueError("schedule validation record 的实例/seed 数不符合配置")
    for scale in validation_scales:
        selection = validation_records[("selection", scale)]
        gate = validation_records[("gate", scale)]
        if not set(selection.logical_indices).isdisjoint(gate.logical_indices):
            raise ValueError(f"scale={scale} selection 与 gate schedule 泄漏")


def write_schedule(
    manifest: ScheduleManifest,
    path: str | Path,
) -> Path:
    """以临时文件 + rename 原子写入 schedule。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_schedule(path: str | Path) -> ScheduleManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_hash = payload.pop("manifest_hash", None)
    records = tuple(
        ScheduleRecord(
            **{
                **record,
                "logical_indices": tuple(record["logical_indices"]),
                "instance_ids": tuple(record["instance_ids"]),
                "coordinate_hashes": tuple(record["coordinate_hashes"]),
                "aco_seeds": tuple(record["aco_seeds"]),
            }
        )
        for record in payload.pop("records")
    )
    manifest = ScheduleManifest(records=records, **payload)
    if expected_hash is not None and expected_hash != manifest.manifest_hash:
        raise ValueError("schedule manifest hash 校验失败")
    return manifest


def _load_record_instances(
    record: ScheduleRecord,
    pools: Mapping[int, ScalePool],
):
    try:
        pool = pools[record.scale]
    except KeyError as exc:
        raise KeyError(f"schedule scale={record.scale} 不在数据 pools") from exc
    instances = [pool.get(index) for index in record.logical_indices]
    observed_ids = tuple(instance.instance_id for instance in instances)
    observed_hashes = tuple(instance.coordinate_hash for instance in instances)
    if observed_ids != record.instance_ids:
        raise ValueError(f"scale={record.scale} schedule instance ID 已变化")
    if observed_hashes != record.coordinate_hashes:
        raise ValueError(f"scale={record.scale} schedule coordinate hash 已变化")
    return instances


class ScheduledTrainingSampler:
    """由不可变 manifest 驱动、checkpoint 状态为 O(1) 的训练 sampler。"""

    def __init__(
        self,
        manifest: ScheduleManifest,
        pools: Mapping[int, ScalePool],
        *,
        replicate_id: int,
        candidate_size: int = 20,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ) -> None:
        self.manifest = manifest
        self.pools = dict(pools)
        self.replicate_id = replicate_id
        self.candidate_size = candidate_size
        self.dtype = dtype
        self.device = device
        self._next_generation = 1
        self._records = {
            (record.generation, record.scale): record
            for record in manifest.records
            if record.split == "train" and record.replicate_id == replicate_id
        }
        if not self._records:
            raise ValueError(f"schedule 不含 replicate_id={replicate_id} 的训练记录")

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "manifest_hash": self.manifest.manifest_hash,
            "replicate_id": self.replicate_id,
            "next_generation": self._next_generation,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state["manifest_hash"] != self.manifest.manifest_hash:
            raise ValueError("恢复点的 schedule manifest hash 不一致")
        if int(state["replicate_id"]) != self.replicate_id:
            raise ValueError("恢复点的 schedule replicate_id 不一致")
        self._next_generation = int(state["next_generation"])

    def cases_for_generation(self, generation: int) -> list[EvaluationCase]:
        if generation != self._next_generation:
            raise ValueError(
                f"scheduled sampler 期望 generation={self._next_generation}，"
                f"实际为 {generation}"
            )
        self._next_generation += 1
        cases: list[EvaluationCase] = []
        for scale in sorted(self.pools):
            try:
                record = self._records[(generation, scale)]
            except KeyError as exc:
                raise KeyError(
                    f"schedule 缺少 generation={generation}, scale={scale}"
                ) from exc
            instances = _load_record_instances(record, self.pools)
            batch = make_problem_batch(
                instances,
                candidate_size=self.candidate_size,
                dtype=self.dtype,
                device=self.device,
            )
            cases.extend(
                EvaluationCase(
                    scale=scale,
                    batch=batch,
                    seed=seed,
                )
                for seed in record.aco_seeds
            )
        return cases


def validation_cases_from_schedule(
    manifest: ScheduleManifest,
    pools: Mapping[int, ScalePool],
    *,
    role: str,
    replicate_id: int,
    batch_size: int,
    seeds: int | None = None,
    candidate_size: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> list[EvaluationCase]:
    """从 selection 或 gate record 构造 batch×seed cases。"""

    if role not in {"selection", "gate"}:
        raise ValueError("validation role 仅支持 selection 或 gate")
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    records = sorted(
        (
            record
            for record in manifest.records
            if record.split == "validation"
            and record.role == role
            and record.replicate_id == replicate_id
        ),
        key=lambda record: record.scale,
    )
    if not records:
        raise ValueError(f"schedule 不含 validation role={role}")
    cases: list[EvaluationCase] = []
    for record in records:
        instances = _load_record_instances(record, pools)
        selected_seeds = record.aco_seeds[:seeds]
        if seeds is not None and len(selected_seeds) != seeds:
            raise ValueError(
                f"role={role}, scale={record.scale} 仅有 "
                f"{len(record.aco_seeds)} seeds，要求 {seeds}"
            )
        for start in range(0, len(instances), batch_size):
            batch = make_problem_batch(
                instances[start : start + batch_size],
                candidate_size=candidate_size,
                dtype=dtype,
                device=device,
            )
            for seed in selected_seeds:
                cases.append(
                    EvaluationCase(
                        scale=record.scale,
                        batch=batch,
                        seed=seed,
                    )
                )
    return cases
