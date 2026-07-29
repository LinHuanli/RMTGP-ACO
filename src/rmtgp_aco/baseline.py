"""不可变、配置感知的原始 ACO baseline archive。"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .aco import solve
from .config import (
    ACOConfig,
    CudaPrecision,
    CudaProvider,
    ExecutionBackend,
    PheromoneIntegration,
    RuntimeConfig,
    TransitionIntegration,
)
from .model import RunResult
from .sampling import EvaluationCase

BASELINE_ARCHIVE_SCHEMA_VERSION = 3
_READABLE_BASELINE_ARCHIVE_SCHEMAS = frozenset({2, 3})
NUMBA_KERNEL_SEMANTIC_VERSION = "counter-rng-numba-v2-mmas-restart"
TORCH_KERNEL_SEMANTIC_VERSION = "torch-generator-v1"
CUDA_KERNEL_SEMANTIC_VERSION = "counter-rng-cuda-fp32-search-v1"
CUDA_V2_KERNEL_SEMANTIC_VERSION = (
    "counter-rng-cuda-tiled-v2-search-v2"
)


@lru_cache(maxsize=16)
def _cuda_v2_manifest_profile(path_text: str) -> tuple[str, str, int]:
    """读取不可变 tuning manifest 的数值 profile。"""

    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("CUDA tuning manifest schema_version 必须为 1")
    if payload.get("backend") != ExecutionBackend.CUDA_TILED_V2.value:
        raise ValueError("CUDA tuning manifest 不是 cuda_tiled_v2")
    selected = payload["selected"]
    provider = CudaProvider(selected["provider"]).value
    precision = CudaPrecision(selected["precision"]).value
    lanes = int(selected["candidate_lanes"])
    if lanes not in {1, 4, 8, 16, 32}:
        raise ValueError("CUDA tuning manifest candidate_lanes 非法")
    return provider, precision, lanes


def _cuda_v2_profile(runtime: RuntimeConfig | None) -> tuple[str, str, int]:
    """解析会改变 CUDA v2 搜索轨迹的 runtime 字段。

    tuning manifest 的 selected 字段优先于 YAML/CLI 默认值。这里只读取
    profile；求解器仍负责核对实际 GPU 名称和 compute capability。
    """

    selected_runtime = runtime or RuntimeConfig(
        aco_backend=ExecutionBackend.CUDA_TILED_V2
    )
    provider = selected_runtime.cuda_provider.value
    precision = selected_runtime.cuda_precision.value
    lanes = selected_runtime.cuda_candidate_lanes or 8
    if selected_runtime.cuda_tuning_manifest is not None:
        path = str(Path(selected_runtime.cuda_tuning_manifest).resolve())
        provider, precision, lanes = _cuda_v2_manifest_profile(path)
    return provider, precision, lanes


def backend_semantic_id(
    backend: ExecutionBackend | str,
    runtime: RuntimeConfig | None = None,
) -> str:
    """返回数值语义 cache 域，而不是仅返回调度后端名称。"""

    selected = ExecutionBackend(backend)
    if selected in {ExecutionBackend.NUMBA, ExecutionBackend.NUMBA_BATCH}:
        return NUMBA_KERNEL_SEMANTIC_VERSION
    if selected is ExecutionBackend.CUDA_FUSED_FP32:
        return CUDA_KERNEL_SEMANTIC_VERSION
    if selected is ExecutionBackend.CUDA_TILED_V2:
        provider, precision, lanes = _cuda_v2_profile(runtime)
        return (
            f"{CUDA_V2_KERNEL_SEMANTIC_VERSION}-"
            f"{provider}-{precision}-lanes{lanes}"
        )
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
    baseline_behavior_hash: str
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
    runtime: RuntimeConfig | None = None,
) -> list[BaselineRecord]:
    """把一个 batch RunResult 展开为可随机访问的 archive rows。"""

    semantic = backend_semantic_id(backend, runtime)
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
                    config_hash=config.baseline_behavior_hash,
                    backend_semantic=semantic,
                ),
                coordinate_hash=coordinate_hash,
                instance_id=case.batch.instance_ids[index],
                scale=case.scale,
                seed=case.seed,
                config_hash=config.config_hash,
                baseline_behavior_hash=config.baseline_behavior_hash,
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
        schema_version = int(metadata.get("schema_version", -1))
        if schema_version not in _READABLE_BASELINE_ARCHIVE_SCHEMAS:
            raise ValueError(f"{source}: baseline archive schema 不兼容")
        count = int(metadata["records"])
        # NPZ 是按列独立压缩的 zip archive。若在 record 循环内反复执行
        # payload["field"]，NumPy 会为每条记录重新解压整列；正式 baseline
        # 有数千行且每个训练进程都要读取，因此必须先把每列解压一次。
        fields = {
            name: payload[name]
            for name in BaselineRecord.__dataclass_fields__
            if name in payload.files
        }
        required = set(BaselineRecord.__dataclass_fields__) - {
            "baseline_behavior_hash"
        }
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{source}: baseline archive 缺少字段 {sorted(missing)}")
        records = [
            BaselineRecord(
                key=str(fields["key"][index]),
                coordinate_hash=str(fields["coordinate_hash"][index]),
                instance_id=str(fields["instance_id"][index]),
                scale=int(fields["scale"][index]),
                seed=int(fields["seed"][index]),
                config_hash=str(fields["config_hash"][index]),
                baseline_behavior_hash=(
                    str(fields["baseline_behavior_hash"][index])
                    if "baseline_behavior_hash" in fields
                    else ""
                ),
                backend_semantic=str(fields["backend_semantic"][index]),
                best_length=float(fields["best_length"][index]),
                reference_length=float(fields["reference_length"][index]),
                reference_gap_percent=float(
                    fields["reference_gap_percent"][index]
                ),
                best_iteration=int(fields["best_iteration"][index]),
                anytime_gap_auc=float(fields["anytime_gap_auc"][index]),
                candidate_fallback_count=int(
                    fields["candidate_fallback_count"][index]
                ),
                uniform_fallback_count=int(
                    fields["uniform_fallback_count"][index]
                ),
                bound_clip_count=int(fields["bound_clip_count"][index]),
                mmas_restart_count=int(fields["mmas_restart_count"][index]),
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
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.backend_semantic = backend_semantic_id(backend, runtime)
        self.require = require
        self._records: dict[str, BaselineRecord] = {}
        # v2 archives 只保存完整 config hash。当前冻结实验的旧 archive
        # 使用 residual/budget-residual 与 gamma=1/3 生成；这些字段在无
        # program baseline 中不生效，因此可严格迁移到行为哈希域。
        legacy_config = ACOConfig(
            **{
                **config.stable_dict(),
                "variant": config.variant,
                "dtype": config.dtype,
                "transition_integration": TransitionIntegration.RESIDUAL,
                "pheromone_integration": PheromoneIntegration.BUDGET_RESIDUAL,
                "gamma_transition": 1.0 / 3.0,
                "gamma_pheromone": 1.0 / 3.0,
            }
        )
        compatible_v2_hashes = {
            config.config_hash,
            legacy_config.config_hash,
        }
        if self.root.is_dir():
            for path in sorted(self.root.rglob("*.npz")):
                records, _ = read_baseline_shard(path)
                for record in records:
                    behavior_matches = (
                        record.baseline_behavior_hash
                        == config.baseline_behavior_hash
                        if record.baseline_behavior_hash
                        else record.config_hash in compatible_v2_hashes
                    )
                    if not behavior_matches or (
                        record.backend_semantic != self.backend_semantic
                    ):
                        continue
                    key = baseline_key(
                        coordinate_hash=record.coordinate_hash,
                        seed=record.seed,
                        config_hash=config.baseline_behavior_hash,
                        backend_semantic=self.backend_semantic,
                    )
                    previous = self._records.setdefault(key, record)
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
                config_hash=self.config.baseline_behavior_hash,
                backend_semantic=self.backend_semantic,
            )
            record = self._records.get(key)
            if record is None:
                if self.require:
                    raise KeyError(
                        "baseline cache miss: "
                        f"hash={coordinate_hash[:12]}, seed={case.seed}, "
                        f"behavior={self.config.baseline_behavior_hash}"
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
            ExecutionBackend.CUDA_TILED_V2,
        }:
            if selected in {
                ExecutionBackend.CUDA_FUSED_FP32,
                ExecutionBackend.CUDA_TILED_V2,
            }:
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
            semantic = backend_semantic_id(selected, runtime)
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
                            config_hash=config.baseline_behavior_hash,
                            backend_semantic=semantic,
                        ),
                        coordinate_hash=coordinate_hash,
                        instance_id=case.batch.instance_ids[index],
                        scale=case.scale,
                        seed=case.seed,
                        config_hash=config.config_hash,
                        baseline_behavior_hash=config.baseline_behavior_hash,
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
        records.extend(
            records_from_result(
                case,
                result,
                config,
                backend,
                runtime,
            )
        )
    return records
