"""不可变、配置感知的原始 ACO baseline archive。"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .aco import solve
from .config import ACOConfig, ExecutionBackend, RuntimeConfig
from .model import RunResult
from .sampling import EvaluationCase

BASELINE_ARCHIVE_SCHEMA_VERSION = 2
NUMBA_KERNEL_SEMANTIC_VERSION = "counter-rng-numba-v2-mmas-restart"
TORCH_KERNEL_SEMANTIC_VERSION = "torch-generator-v1"
CUDA_KERNEL_SEMANTIC_VERSION = "counter-rng-cuda-fp32-search-v1"


def backend_semantic_id(backend: ExecutionBackend | str) -> str:
    """把调度不同但数值语义相同的 Numba 后端映射到同一 cache 域。"""

    selected = ExecutionBackend(backend)
    if selected in {ExecutionBackend.NUMBA, ExecutionBackend.NUMBA_BATCH}:
        return NUMBA_KERNEL_SEMANTIC_VERSION
    if selected is ExecutionBackend.CUDA_FUSED_FP32:
        return CUDA_KERNEL_SEMANTIC_VERSION
    return TORCH_KERNEL_SEMANTIC_VERSION


def baseline_key(
    *,
    coordinate_hash: str,
    seed: int,
    config_hash: str,
    backend_semantic: str,
) -> str:
    payload = (
        f"{coordinate_hash}\0{int(seed)}\0{config_hash}\0{backend_semantic}"
    )
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BaselineRecord:
    """一个 instance×seed 的完整 baseline 质量记录。"""

    key: str
    coordinate_hash: str
    instance_id: str
    scale: int
    seed: int
    config_hash: str
    backend_semantic: str
    best_length: float
    reference_length: float
    reference_gap_percent: float
    best_iteration: int
    anytime_gap_auc: float
    candidate_fallback_count: int
    uniform_fallback_count: int
    bound_clip_count: int
    mmas_restart_count: int


def records_from_result(
    case: EvaluationCase,
    result: RunResult,
    config: ACOConfig,
    backend: ExecutionBackend | str,
) -> list[BaselineRecord]:
    """把一个 batch RunResult 展开为可随机访问的 archive rows。"""

    semantic = backend_semantic_id(backend)
    reference = case.batch.reference_length.detach().cpu().numpy()
    best = result.best_length.detach().cpu().numpy()
    anytime = result.anytime_best.detach().cpu().numpy()
    gap = 100.0 * (best - reference) / reference
    anytime_gap = 100.0 * (anytime - reference[:, None]) / reference[:, None]
    records: list[BaselineRecord] = []
    for index, coordinate_hash in enumerate(case.batch.coordinate_hashes):
        records.append(
            BaselineRecord(
                key=baseline_key(
                    coordinate_hash=coordinate_hash,
                    seed=case.seed,
                    config_hash=config.config_hash,
                    backend_semantic=semantic,
                ),
                coordinate_hash=coordinate_hash,
                instance_id=case.batch.instance_ids[index],
                scale=case.scale,
                seed=case.seed,
                config_hash=config.config_hash,
                backend_semantic=semantic,
                best_length=float(best[index]),
                reference_length=float(reference[index]),
                reference_gap_percent=float(gap[index]),
                best_iteration=int(result.best_iteration[index].item()),
                anytime_gap_auc=float(anytime_gap[index].mean()),
                candidate_fallback_count=result.diagnostics.candidate_fallback_count,
                uniform_fallback_count=result.diagnostics.uniform_fallback_count,
                bound_clip_count=result.diagnostics.bound_clip_count,
                mmas_restart_count=result.diagnostics.mmas_restart_count,
            )
        )
    return records


def write_baseline_shard(
    records: Sequence[BaselineRecord],
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    """写入不可变压缩 shard；已有目标不会被静默覆盖。"""

    if not records:
        raise ValueError("baseline shard 不允许为空")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"baseline shard 已存在: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    payload = {
        field: np.asarray([getattr(record, field) for record in records])
        for field in BaselineRecord.__dataclass_fields__
    }
    archive_metadata = {
        "schema_version": BASELINE_ARCHIVE_SCHEMA_VERSION,
        "records": len(records),
        **(metadata or {}),
    }
    payload["__metadata__"] = np.asarray(
        json.dumps(archive_metadata, ensure_ascii=False, sort_keys=True)
    )
    np.savez_compressed(temporary, **payload)
    temporary.replace(target)
    return target


def read_baseline_shard(path: str | Path) -> tuple[list[BaselineRecord], dict]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["__metadata__"].item()))
        if metadata.get("schema_version") != BASELINE_ARCHIVE_SCHEMA_VERSION:
            raise ValueError(f"{source}: baseline archive schema 不兼容")
        count = int(metadata["records"])
        records = [
            BaselineRecord(
                key=str(payload["key"][index]),
                coordinate_hash=str(payload["coordinate_hash"][index]),
                instance_id=str(payload["instance_id"][index]),
                scale=int(payload["scale"][index]),
                seed=int(payload["seed"][index]),
                config_hash=str(payload["config_hash"][index]),
                backend_semantic=str(payload["backend_semantic"][index]),
                best_length=float(payload["best_length"][index]),
                reference_length=float(payload["reference_length"][index]),
                reference_gap_percent=float(payload["reference_gap_percent"][index]),
                best_iteration=int(payload["best_iteration"][index]),
                anytime_gap_auc=float(payload["anytime_gap_auc"][index]),
                candidate_fallback_count=int(
                    payload["candidate_fallback_count"][index]
                ),
                uniform_fallback_count=int(payload["uniform_fallback_count"][index]),
                bound_clip_count=int(payload["bound_clip_count"][index]),
                mmas_restart_count=int(payload["mmas_restart_count"][index]),
            )
            for index in range(count)
        ]
    return records, metadata


class BaselineArchive:
    """把若干只读 NPZ shards 暴露为严格 cache lookup。"""

    def __init__(
        self,
        root: str | Path,
        config: ACOConfig,
        backend: ExecutionBackend | str,
        *,
        require: bool,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.backend_semantic = backend_semantic_id(backend)
        self.require = require
        self._records: dict[str, BaselineRecord] = {}
        if self.root.is_dir():
            for path in sorted(self.root.rglob("*.npz")):
                records, _ = read_baseline_shard(path)
                for record in records:
                    if (
                        record.config_hash != config.config_hash
                        or record.backend_semantic != self.backend_semantic
                    ):
                        continue
                    previous = self._records.setdefault(record.key, record)
                    if not _records_equivalent(previous, record):
                        raise ValueError(f"baseline archive 冲突 key={record.key}")
        elif require:
            raise FileNotFoundError(f"baseline archive 不存在: {self.root}")

    def lookup(self, case: EvaluationCase) -> torch.Tensor | None:
        values: list[float] = []
        for coordinate_hash in case.batch.coordinate_hashes:
            key = baseline_key(
                coordinate_hash=coordinate_hash,
                seed=case.seed,
                config_hash=self.config.config_hash,
                backend_semantic=self.backend_semantic,
            )
            record = self._records.get(key)
            if record is None:
                if self.require:
                    raise KeyError(
                        "baseline cache miss: "
                        f"hash={coordinate_hash[:12]}, seed={case.seed}, "
                        f"config={self.config.config_hash}"
                    )
                return None
            values.append(record.best_length)
        return torch.as_tensor(values, dtype=torch.float64)

    @property
    def records(self) -> int:
        return len(self._records)


def _records_equivalent(first: BaselineRecord, second: BaselineRecord) -> bool:
    """把两个 NaN anytime 占位值视为相同，其余字段要求严格一致。"""

    for field_name in BaselineRecord.__dataclass_fields__:
        left = getattr(first, field_name)
        right = getattr(second, field_name)
        if (
            isinstance(left, float)
            and isinstance(right, float)
            and math.isnan(left)
            and math.isnan(right)
        ):
            continue
        if left != right:
            return False
    return True


def precompute_baseline_cases(
    cases: Iterable[EvaluationCase],
    config: ACOConfig,
    backend: ExecutionBackend | str,
    *,
    threads: int = 16,
    runtime: RuntimeConfig | None = None,
) -> list[BaselineRecord]:
    """显式预计算 cases；调用方负责按 split/replicate 写 immutable shard。"""

    records: list[BaselineRecord] = []
    selected = ExecutionBackend(backend)
    for case in cases:
        if selected in {
            ExecutionBackend.NUMBA_BATCH,
            ExecutionBackend.CUDA_FUSED_FP32,
        }:
            if selected is ExecutionBackend.CUDA_FUSED_FP32:
                from .aco_cuda import solve_population_cuda

                if runtime is None:
                    runtime = RuntimeConfig(aco_backend=selected)
                quality = solve_population_cuda(
                    case.batch,
                    config,
                    [(None, None)],
                    seed=case.seed,
                    runtime=runtime,
                )
            else:
                from .aco_numba import solve_population_numba

                quality = solve_population_numba(
                    case.batch,
                    config,
                    [(None, None)],
                    seed=case.seed,
                    threads=threads,
                )
            semantic = backend_semantic_id(selected)
            reference = case.batch.reference_length.detach().cpu().numpy()
            best = quality.best_length[0].detach().cpu().numpy()
            gap = 100.0 * (best - reference) / reference
            diagnostic = quality.diagnostics[0]
            for index, coordinate_hash in enumerate(case.batch.coordinate_hashes):
                records.append(
                    BaselineRecord(
                        key=baseline_key(
                            coordinate_hash=coordinate_hash,
                            seed=case.seed,
                            config_hash=config.config_hash,
                            backend_semantic=semantic,
                        ),
                        coordinate_hash=coordinate_hash,
                        instance_id=case.batch.instance_ids[index],
                        scale=case.scale,
                        seed=case.seed,
                        config_hash=config.config_hash,
                        backend_semantic=semantic,
                        best_length=float(best[index]),
                        reference_length=float(reference[index]),
                        reference_gap_percent=float(gap[index]),
                        best_iteration=int(quality.best_iteration[0, index].item()),
                        anytime_gap_auc=float("nan"),
                        candidate_fallback_count=int(diagnostic[0].item()),
                        uniform_fallback_count=int(diagnostic[1].item()),
                        bound_clip_count=int(diagnostic[2].item()),
                        mmas_restart_count=int(diagnostic[3].item()),
                    )
                )
            continue
        result = solve(
            case.batch,
            config,
            seed=case.seed,
            backend=backend,
            runtime=runtime,
        )
        records.extend(records_from_result(case, result, config, backend))
    return records
