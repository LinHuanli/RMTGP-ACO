"""YAML 运行规范：把科研配置与数据路径解析为强类型对象。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from yaml.resolver import BaseResolver

from .config import (
    ACOConfig,
    ACOVariant,
    ExperimentConfig,
    GPConfig,
    RacingConfig,
    RuntimeConfig,
    TransitionIntegration,
)

ALLOWED_TRAIN_VALIDATION_SCALES = frozenset({50, 100, 500})
ALLOWED_TEST_SCALES = frozenset({50, 100, 500, 1000})


class _UniqueKeyLoader(yaml.SafeLoader):
    """拒绝 YAML mapping 中会被 PyYAML 静默覆盖的重复键。"""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"YAML 重复字段 {key!r}（第 {key_node.start_mark.line + 1} 行）"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class TestPartitionSpec:
    """一个不可用于模型选择的最终测试 partition。"""

    scale: int
    distribution: str
    files: tuple[str, ...]
    min_scale: int | None = None
    max_scale: int | None = None


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """数据 root、split patterns 与采样规模。"""

    root: Path
    train: dict[int, tuple[str, ...]]
    validation: dict[int, tuple[str, ...]]
    test: dict[str, TestPartitionSpec]
    train_instances_per_scale: int = 16
    validation_selection_instances_per_scale: int = 32
    validation_gate_instances_per_scale: int = 32
    evaluation_batch_size: int = 32
    schedule_manifest: str | None = None
    baseline_archive: str | None = None
    baseline_policy: str = "compute"

    def __post_init__(self) -> None:
        for label, mapping in (("train", self.train), ("validation", self.validation)):
            invalid = set(mapping) - ALLOWED_TRAIN_VALIDATION_SCALES
            if invalid:
                raise ValueError(f"{label} 含禁止规模 {sorted(invalid)}；仅允许 50/100/500")
        for name, partition in self.test.items():
            if (
                partition.scale not in ALLOWED_TEST_SCALES
                and not name.startswith("tsplib")
            ):
                raise ValueError(
                    f"test partition {name!r} 含禁止规模 {partition.scale}"
                )
            if (
                partition.min_scale is not None
                and partition.max_scale is not None
                and partition.min_scale > partition.max_scale
            ):
                raise ValueError(f"test partition {name!r} 的规模区间为空")
        if self.train_instances_per_scale < 1:
            raise ValueError("train_instances_per_scale 必须为正整数")
        if self.validation_selection_instances_per_scale < 1:
            raise ValueError("validation selection 实例数必须为正整数")
        if self.validation_gate_instances_per_scale < 1:
            raise ValueError("validation gate 实例数必须为正整数")
        if self.evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size 必须为正整数")
        if self.baseline_policy not in {"compute", "require"}:
            raise ValueError("baseline_policy 仅支持 compute 或 require")

    def resolve_patterns(self, patterns: tuple[str, ...]) -> tuple[Path, ...]:
        """相对 root 展开 glob，并显式排除已知重复 copy。"""

        resolved: list[Path] = []
        for pattern in patterns:
            candidates = (
                [Path(pattern)]
                if Path(pattern).is_absolute()
                else sorted(self.root.glob(pattern))
            )
            for path in candidates:
                if path.name == "tsp100_concorde_7.756 copy.txt":
                    continue
                if path.is_file():
                    resolved.append(path.resolve())
        unique = tuple(dict.fromkeys(resolved))
        if not unique:
            raise FileNotFoundError(
                f"数据 pattern 未匹配任何文件: {patterns!r} (root={self.root})"
            )
        return unique

    def training_paths(self) -> dict[int, tuple[Path, ...]]:
        return {
            scale: self.resolve_patterns(patterns)
            for scale, patterns in self.train.items()
        }

    def validation_paths(self) -> dict[int, tuple[Path, ...]]:
        return {
            scale: self.resolve_patterns(patterns)
            for scale, patterns in self.validation.items()
        }

    def test_paths(self, partition: str) -> tuple[Path, ...]:
        try:
            selected = self.test[partition]
        except KeyError as exc:
            raise KeyError(f"未知 test partition: {partition}") from exc
        return self.resolve_patterns(selected.files)

    def resolve_auxiliary_path(self, value: str | None) -> Path | None:
        """按项目工作目录解析 schedule/cache 路径。"""

        if value is None:
            return None
        path = Path(value)
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()

    @property
    def schedule_path(self) -> Path | None:
        return self.resolve_auxiliary_path(self.schedule_manifest)

    @property
    def baseline_path(self) -> Path | None:
        return self.resolve_auxiliary_path(self.baseline_archive)


@dataclass(frozen=True, slots=True)
class RunSpec:
    """训练/评测命令所需的完整规范。"""

    experiment: ExperimentConfig
    data: DatasetSpec
    source_path: Path


def _strict_kwargs(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__} 含未知字段: {sorted(unknown)}")
    return dict(payload)


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.").lower()
    mapping = {"float32": torch.float32, "float64": torch.float64}
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError("dtype 仅支持 float32 或 float64") from exc


def _parse_aco(payload: Mapping[str, Any]) -> ACOConfig:
    values = dict(payload)
    try:
        variant = ACOVariant(values.pop("variant"))
    except KeyError as exc:
        raise ValueError("aco.variant 是必填字段") from exc
    iterations = int(values.pop("iterations", 100))
    device = str(values.pop("device", "cpu"))
    dtype = _torch_dtype(str(values.pop("dtype", "float64")))
    synchronous = bool(values.pop("acs_synchronous", True))
    baseline = ACOConfig.acotsp_default(
        variant,
        iterations=iterations,
        device=device,
        dtype=dtype,
        acs_synchronous=synchronous,
    )
    merged = baseline.stable_dict()
    merged.update(values)
    merged["variant"] = variant
    merged["iterations"] = iterations
    merged["device"] = device
    merged["dtype"] = dtype
    merged["acs_synchronous"] = synchronous
    return ACOConfig(**_strict_kwargs(ACOConfig, merged))


def _parse_scale_patterns(payload: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = {}
    for scale, patterns in payload.items():
        values = [patterns] if isinstance(patterns, str) else list(patterns)
        result[int(scale)] = tuple(str(item) for item in values)
    return result


def _parse_test_partitions(
    payload: Mapping[str, Any],
) -> dict[str, TestPartitionSpec]:
    result: dict[str, TestPartitionSpec] = {}
    for name, raw in payload.items():
        values = dict(raw)
        files = values.get("files")
        if isinstance(files, str):
            values["files"] = (files,)
        else:
            values["files"] = tuple(str(item) for item in files)
        result[str(name)] = TestPartitionSpec(
            **_strict_kwargs(TestPartitionSpec, values)
        )
    return result


def load_run_spec(path: str | Path) -> RunSpec:
    """读取 YAML，并拒绝拼写错误造成的静默默认。"""

    source = Path(path).resolve()
    payload = yaml.load(
        source.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("配置根节点必须是 mapping")
    unknown_sections = set(payload) - {
        "experiment",
        "aco",
        "gp",
        "runtime",
        "racing",
        "data",
    }
    if unknown_sections:
        raise ValueError(f"配置含未知顶层字段: {sorted(unknown_sections)}")

    experiment_raw = dict(payload.get("experiment", {}))
    aco = _parse_aco(dict(payload.get("aco", {})))
    gp = GPConfig(**_strict_kwargs(GPConfig, dict(payload.get("gp", {}))))
    runtime = RuntimeConfig(
        **_strict_kwargs(RuntimeConfig, dict(payload.get("runtime", {})))
    )
    racing = RacingConfig(
        **_strict_kwargs(RacingConfig, dict(payload.get("racing", {})))
    )
    experiment = ExperimentConfig(
        aco=aco,
        gp=gp,
        runtime=runtime,
        racing=racing,
        **_strict_kwargs(ExperimentConfig, experiment_raw),
    )
    if (
        gp.transition_profile == "legacy"
        and aco.transition_integration is not TransitionIntegration.REPLACEMENT
    ):
        raise ValueError(
            "Legacy-GP 必须使用 aco.transition_integration=replacement"
        )

    data_raw = dict(payload.get("data", {}))
    root_value = data_raw.pop("root", "Datasets/TSP")
    root = Path(root_value)
    if not root.is_absolute():
        # 数据路径按项目工作目录解析，而不是按 configs/ 子目录解析。
        root = Path.cwd() / root
    train = _parse_scale_patterns(data_raw.pop("train", {}))
    validation = _parse_scale_patterns(data_raw.pop("validation", {}))
    test = _parse_test_partitions(data_raw.pop("test", {}))
    data = DatasetSpec(
        root=root.resolve(),
        train=train,
        validation=validation,
        test=test,
        **_strict_kwargs(DatasetSpec, data_raw),
    )
    if set(experiment.train_scales) != set(data.train):
        raise ValueError("experiment.train_scales 必须与 data.train 的 keys 完全一致")
    if set(experiment.validation_scales) != set(data.validation):
        raise ValueError(
            "experiment.validation_scales 必须与 data.validation 的 keys 完全一致"
        )
    if (
        experiment.racing.enabled
        and experiment.racing.screen_instances_per_scale
        > data.train_instances_per_scale
    ):
        raise ValueError(
            "racing screen_instances_per_scale 不得超过 "
            "data.train_instances_per_scale"
        )
    if (
        experiment.racing.enabled
        and experiment.racing.high_instances_per_scale is not None
        and experiment.racing.high_instances_per_scale
        > data.train_instances_per_scale
    ):
        raise ValueError(
            "racing high_instances_per_scale 不得超过 "
            "data.train_instances_per_scale"
        )
    return RunSpec(experiment=experiment, data=data, source_path=source)
