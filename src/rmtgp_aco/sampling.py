"""超大文本 shard 的随机访问与分规模 mini-batch 采样。"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import (
    TSPInstance,
    build_line_offsets,
    load_indexed_instances,
    make_problem_batch,
)
from .model import ProblemBatch


@dataclass(slots=True)
class IndexedShard:
    """一个文本 shard 及其 byte offsets。"""

    path: Path
    offsets: np.ndarray

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        use_cache: bool = True,
    ) -> IndexedShard:
        """建立随机访问索引，并以文件大小和 mtime 校验磁盘缓存。"""

        source = Path(path)
        stat = source.stat()
        cache = source.with_suffix(f"{source.suffix}.offsets.npz")
        offsets: np.ndarray | None = None
        if use_cache and cache.is_file():
            try:
                with np.load(cache, allow_pickle=False) as payload:
                    cached_size = int(payload["size"])
                    cached_mtime_ns = int(payload["mtime_ns"])
                    if (
                        cached_size == stat.st_size
                        and cached_mtime_ns == stat.st_mtime_ns
                    ):
                        offsets = np.asarray(payload["offsets"], dtype=np.int64)
            except (OSError, ValueError, KeyError):
                offsets = None
        if offsets is None:
            offsets = build_line_offsets(source)
            if use_cache:
                temporary = cache.with_suffix(f"{cache.suffix}.tmp.npz")
                np.savez(
                    temporary,
                    offsets=offsets,
                    size=np.asarray(stat.st_size, dtype=np.int64),
                    mtime_ns=np.asarray(stat.st_mtime_ns, dtype=np.int64),
                )
                os.replace(temporary, cache)
        return cls(path=source, offsets=offsets)

    def __len__(self) -> int:
        return int(self.offsets.size)

    def get(self, index: int) -> TSPInstance:
        """读取 shard 中的单个实例。"""

        return load_indexed_instances(self.path, self.offsets, [index])[0]


@dataclass(slots=True)
class ScalePool:
    """一个规模下若干 shard 的逻辑拼接。"""

    scale: int
    shards: tuple[IndexedShard, ...]
    _sizes: np.ndarray = field(init=False, repr=False)
    _cumulative: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.shards:
            raise ValueError("ScalePool 至少需要一个 shard")
        if any(len(shard) == 0 for shard in self.shards):
            raise ValueError("ScalePool 不允许空 shard")
        self._sizes = np.asarray([len(shard) for shard in self.shards], dtype=np.int64)
        self._cumulative = np.cumsum(self._sizes)

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def get(self, logical_index: int) -> TSPInstance:
        """把逻辑索引映射到具体 shard。"""

        if logical_index < 0 or logical_index >= len(self):
            raise IndexError(f"逻辑索引越界: {logical_index}")
        shard_index = int(np.searchsorted(self._cumulative, logical_index, side="right"))
        previous = 0 if shard_index == 0 else int(self._cumulative[shard_index - 1])
        instance = self.shards[shard_index].get(logical_index - previous)
        if instance.n != self.scale:
            raise ValueError(
                f"数据规模错误：pool={self.scale}，实例={instance.n} ({instance.instance_id})"
            )
        return instance


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """一个同规模 batch 及其独立 ACO seed。"""

    scale: int
    batch: ProblemBatch
    seed: int


class ScaleStratifiedSampler:
    """按 active scale 等权采样，避免大规模文件数量主导 fitness。"""

    def __init__(
        self,
        pools: Mapping[int, ScalePool],
        *,
        root_seed: int,
        candidate_size: int = 20,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        instances_per_scale: int = 1,
    ) -> None:
        if not pools:
            raise ValueError("至少需要一个 scale pool")
        self.pools = dict(sorted(pools.items()))
        self.root_seed = int(root_seed)
        self.candidate_size = candidate_size
        self.dtype = dtype
        self.device = device
        self.instances_per_scale = instances_per_scale
        self._rngs = {
            scale: np.random.default_rng(
                np.random.SeedSequence([self.root_seed, scale, 0x53485546])
            )
            for scale in self.pools
        }
        # 旧实现为每个 scale 保存 128 万长度的完整 permutation，导致每个
        # checkpoint 约 20 MB。正式 run 每规模只抽 800 个索引，因此只保存
        # 已使用集合；在池接近耗尽时才构造剩余候选。
        self._used = {scale: set() for scale in self.pools}
        self._cycles = {scale: 0 for scale in self.pools}
        self._legacy_orders: dict[int, np.ndarray] | None = None
        self._legacy_positions: dict[int, int] | None = None
        self._next_generation = 1

    def state_dict(self) -> dict[str, object]:
        """返回可 pickle 的精确采样状态，用于逐代断点恢复。"""

        if self._legacy_orders is not None and self._legacy_positions is not None:
            return {
                "root_seed": self.root_seed,
                "scales": tuple(self.pools),
                "rng_states": {
                    scale: rng.bit_generator.state
                    for scale, rng in self._rngs.items()
                },
                "orders": {
                    scale: order.copy()
                    for scale, order in self._legacy_orders.items()
                },
                "positions": dict(self._legacy_positions),
                "next_generation": self._next_generation,
            }
        return {
            "schema_version": 2,
            "root_seed": self.root_seed,
            "scales": tuple(self.pools),
            "rng_states": {
                scale: rng.bit_generator.state
                for scale, rng in self._rngs.items()
            },
            "used_indices": {
                scale: np.asarray(sorted(values), dtype=np.int64)
                for scale, values in self._used.items()
            },
            "cycles": dict(self._cycles),
            "next_generation": self._next_generation,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """恢复由 :meth:`state_dict` 生成的状态并校验数据池身份。"""

        if int(state["root_seed"]) != self.root_seed:
            raise ValueError("sampler checkpoint 的 root_seed 与当前配置不一致")
        if tuple(state["scales"]) != tuple(self.pools):
            raise ValueError("sampler checkpoint 的 scales 与当前数据池不一致")
        rng_states = state["rng_states"]
        if not isinstance(rng_states, Mapping):
            raise TypeError("sampler rng_states 必须为 mapping")
        for scale in self.pools:
            self._rngs[scale].bit_generator.state = rng_states[scale]
        if "used_indices" in state:
            used = state["used_indices"]
            cycles = state["cycles"]
            if not isinstance(used, Mapping) or not isinstance(cycles, Mapping):
                raise TypeError("sampler used_indices/cycles 必须为 mapping")
            for scale in self.pools:
                restored = np.asarray(used[scale], dtype=np.int64)
                if np.any(restored < 0) or np.any(restored >= len(self.pools[scale])):
                    raise ValueError(f"scale={scale} sampler used index 越界")
                self._used[scale] = set(int(item) for item in restored)
                self._cycles[scale] = int(cycles[scale])
            self._legacy_orders = None
            self._legacy_positions = None
        else:
            # 兼容 v0.2 已存在的恢复点；新 checkpoint 不再产生大 permutation。
            orders = state["orders"]
            positions = state["positions"]
            if not isinstance(orders, Mapping) or not isinstance(positions, Mapping):
                raise TypeError("旧 sampler orders/positions 必须为 mapping")
            self._legacy_orders = {}
            self._legacy_positions = {}
            for scale in self.pools:
                restored = np.asarray(orders[scale], dtype=np.int64)
                if restored.shape != (len(self.pools[scale]),):
                    raise ValueError(f"scale={scale} sampler order shape 不一致")
                self._legacy_orders[scale] = restored.copy()
                self._legacy_positions[scale] = int(positions[scale])
        self._next_generation = int(state["next_generation"])

    def _draw_without_replacement(self, scale: int, count: int) -> np.ndarray:
        """跨 generation 维护无放回索引流；一轮耗尽后再独立洗牌。"""

        pool_size = len(self.pools[scale])
        if count > pool_size:
            raise ValueError(
                f"instances_per_scale={count} 大于 scale={scale} pool={pool_size}"
            )
        if self._legacy_orders is not None and self._legacy_positions is not None:
            position = self._legacy_positions[scale]
            if position + count <= pool_size:
                result = self._legacy_orders[scale][position : position + count].copy()
                self._legacy_positions[scale] += count
                return result
            # 极少数跨旧 permutation 边界的恢复 run 仍保持精确旧语义。
            first = self._legacy_orders[scale][position:].copy()
            remaining = count - first.size
            self._legacy_orders[scale] = self._rngs[scale].permutation(pool_size)
            self._legacy_positions[scale] = remaining
            return np.concatenate((first, self._legacy_orders[scale][:remaining]))

        selected: list[int] = []
        used = self._used[scale]
        rng = self._rngs[scale]
        while len(selected) < count:
            if len(used) == pool_size:
                used.clear()
                self._cycles[scale] += 1
            available = pool_size - len(used)
            needed = count - len(selected)
            take = min(available, needed)
            if len(used) > pool_size // 2:
                remaining = np.fromiter(
                    (index for index in range(pool_size) if index not in used),
                    dtype=np.int64,
                    count=available,
                )
                draw = rng.choice(remaining, size=take, replace=False)
                values = [int(item) for item in np.atleast_1d(draw)]
            else:
                values = []
                while len(values) < take:
                    value = int(rng.integers(0, pool_size))
                    if value not in used and value not in values:
                        values.append(value)
            used.update(values)
            selected.extend(values)
        return np.asarray(selected, dtype=np.int64)

    def selection_for_generation(
        self,
        generation: int,
    ) -> list[tuple[int, np.ndarray, int]]:
        """返回本代逻辑索引和 ACO seed，不触发实例解析或矩阵预计算。"""

        if generation != self._next_generation:
            raise ValueError(
                f"sampler 要求顺序调用 generation={self._next_generation}，"
                f"实际为 {generation}"
            )
        self._next_generation += 1
        selections: list[tuple[int, np.ndarray, int]] = []
        for scale in self.pools:
            indices = self._draw_without_replacement(
                scale,
                self.instances_per_scale,
            )
            aco_seed = int(
                np.random.default_rng(
                    np.random.SeedSequence(
                        [self.root_seed, generation, scale, 0x41434F]
                    )
                ).integers(0, 2**63 - 1)
            )
            selections.append((scale, indices, aco_seed))
        return selections

    def cases_for_generation(self, generation: int) -> list[EvaluationCase]:
        """由 generation 派生确定性、分规模的 mini-batch。"""

        cases: list[EvaluationCase] = []
        for scale, indices, aco_seed in self.selection_for_generation(generation):
            pool = self.pools[scale]
            instances = [pool.get(int(index)) for index in indices]
            batch = make_problem_batch(
                instances,
                candidate_size=self.candidate_size,
                dtype=self.dtype,
                device=self.device,
            )
            cases.append(EvaluationCase(scale=scale, batch=batch, seed=aco_seed))
        return cases


def pools_from_paths(
    paths_by_scale: Mapping[int, Iterable[str | Path]],
) -> dict[int, ScalePool]:
    """从显式路径建立 pools；不依赖文件名猜测规模。"""

    return {
        scale: ScalePool(
            scale=scale,
            shards=tuple(IndexedShard.open(path) for path in paths),
        )
        for scale, paths in paths_by_scale.items()
    }


def in_memory_cases(
    instances_by_scale: Mapping[int, Sequence[TSPInstance]],
    *,
    seed: int,
    candidate_size: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> list[EvaluationCase]:
    """测试和小型开发实验使用的便利构造器。"""

    return [
        EvaluationCase(
            scale=scale,
            batch=make_problem_batch(
                instances,
                candidate_size=candidate_size,
                dtype=dtype,
                device=device,
            ),
            seed=seed + scale,
        )
        for scale, instances in sorted(instances_by_scale.items())
    ]


def fixed_cases_from_pools(
    pools: Mapping[int, ScalePool],
    *,
    root_seed: int,
    instances_per_scale: int,
    aco_seeds: int,
    batch_size: int,
    candidate_size: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> list[EvaluationCase]:
    """确定性抽取 validation 子集，并按 batch 与独立 ACO seed 展开。"""

    if aco_seeds < 1 or batch_size < 1:
        raise ValueError("aco_seeds 和 batch_size 必须为正整数")
    cases: list[EvaluationCase] = []
    for scale, pool in sorted(pools.items()):
        count = min(instances_per_scale, len(pool))
        rng = np.random.default_rng(
            np.random.SeedSequence([root_seed, scale, 0x56414C])
        )
        selected = rng.choice(len(pool), size=count, replace=False)
        selected.sort()
        for batch_number, start in enumerate(range(0, count, batch_size)):
            indices = selected[start : start + batch_size]
            instances = [pool.get(int(index)) for index in indices]
            batch = make_problem_batch(
                instances,
                candidate_size=candidate_size,
                dtype=dtype,
                device=device,
            )
            for seed_index in range(aco_seeds):
                seed_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            root_seed,
                            scale,
                            batch_number,
                            seed_index,
                            0x41434F,
                        ]
                    )
                )
                cases.append(
                    EvaluationCase(
                        scale=scale,
                        batch=batch,
                        seed=int(seed_rng.integers(0, 2**63 - 1)),
                    )
                )
    return cases


def iter_problem_batches(
    paths: Iterable[str | Path],
    *,
    batch_size: int,
    candidate_size: int = 20,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    min_scale: int | None = None,
    max_scale: int | None = None,
    max_instances: int | None = None,
) -> Iterable[ProblemBatch]:
    """流式读取 partition；不同规模自动分开成 batch。

    常规 synthetic 文件规模固定；TSPLIB partition 则可能逐文件异构。
    每个规模的尾 batch 在所有输入读完后发出。
    """

    from .data import iter_tsp_file

    if max_instances is not None and max_instances < 1:
        raise ValueError("max_instances 必须为正整数")
    buffers: dict[int, list[TSPInstance]] = {}
    accepted = 0
    finished = False
    for path in paths:
        for instance in iter_tsp_file(path):
            if min_scale is not None and instance.n < min_scale:
                continue
            if max_scale is not None and instance.n > max_scale:
                continue
            if max_instances is not None and accepted >= max_instances:
                finished = True
                break
            accepted += 1
            buffer = buffers.setdefault(instance.n, [])
            buffer.append(instance)
            if len(buffer) == batch_size:
                yield make_problem_batch(
                    buffer,
                    candidate_size=candidate_size,
                    dtype=dtype,
                    device=device,
                )
                buffers[instance.n] = []
        if finished:
            break
    for scale in sorted(buffers):
        if buffers[scale]:
            yield make_problem_batch(
                buffers[scale],
                candidate_size=candidate_size,
                dtype=dtype,
                device=device,
            )
