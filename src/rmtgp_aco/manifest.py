"""数据 inventory、内容校验和 split 防泄漏审计。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from .data import parse_tsp_line


@dataclass(frozen=True, slots=True)
class ManifestFile:
    """manifest 中的一项数据文件。"""

    path: str
    split: str
    distribution: str
    scale: int
    instances: int
    bytes: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class DuplicateExclusion:
    """已证实逐字节相同、因此只保留一个逻辑副本的文件组。"""

    kept: str
    excluded: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class DataManifest:
    """当前研究允许使用的数据快照。"""

    schema_version: int
    generated_at: str
    root: str
    allowed_scales: tuple[int, ...]
    excluded_files: tuple[str, ...]
    duplicate_exclusions: tuple[DuplicateExclusion, ...]
    hash_mode: str
    files: tuple[ManifestFile, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["files"] = [asdict(item) for item in self.files]
        return payload


def _expected_instances(path: Path, split: str) -> int:
    name = path.name
    if split == "train":
        if "128k" in name:
            return 128_000
        if "16k" in name:
            return 16_000
    if split == "validation":
        if "tsp500_" in name:
            return 128
        return 1_280
    if path.parent.name == "tsplib":
        return 1
    if "tsp500_" in name:
        return 128
    if "tsp50_" in name or "tsp100_" in name:
        return 1_280
    raise ValueError(f"无法从文件名推断实例数: {path}")


def _scale_from_first_line(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    tokens = first.split()
    try:
        separator = tokens.index("output")
    except ValueError as exc:
        raise ValueError(f"{path}: 首行缺少 output") from exc
    if separator % 2:
        raise ValueError(f"{path}: 首行坐标 token 数不是偶数")
    return separator // 2


def _digest_and_count(path: Path) -> tuple[str, int]:
    digest = sha256()
    lines = 0
    last_byte = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            lines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if path.stat().st_size > 0 and last_byte != b"\n":
        lines += 1
    return digest.hexdigest(), lines


def discover_research_files(root: str | Path) -> list[tuple[Path, str, str]]:
    """仅发现 50/100/500 主研究及 TSPLIB 文件，绝不纳入 TSP200。"""

    base = Path(root)
    discovered: list[tuple[Path, str, str]] = []
    for scale, directory in (
        (50, "tsp_50"),
        (100, "tsp_100"),
        (500, "tsp_500"),
    ):
        pattern = base / "train_dataset" / "tsp" / directory
        for path in sorted(pattern.glob("*.txt")):
            discovered.append((path, "train", "uniform"))

        validation = (
            base / "val_dataset" / "tsp" / f"tsp{scale}_uniform_val.txt"
        )
        if validation.is_file():
            discovered.append((validation, "validation", "uniform"))

    test_root = base / "test_dataset" / "tsp"
    synthetic_tests = {
        "tsp50_concorde_5.688.txt": "uniform",
        "tsp100_concorde_7.756.txt": "uniform",
        "tsp500_concorde_16.546.txt": "uniform",
        "tsp500_cluster_10.723.txt": "cluster",
        "tsp500_gaussian_77.521.txt": "gaussian",
    }
    for name, distribution in synthetic_tests.items():
        path = test_root / name
        if path.is_file():
            discovered.append((path, "test", distribution))
    for path in sorted((test_root / "tsplib").glob("*.txt")):
        discovered.append((path, "test", "tsplib"))
    return discovered


def build_manifest(
    root: str | Path,
    *,
    full_hashes: bool,
    workers: int = 4,
) -> DataManifest:
    """生成 quick inventory 或一次读取完成 SHA-256 与行数的完整 manifest。"""

    supplied_root = Path(root)
    base = supplied_root.resolve()
    sources = discover_research_files(base)
    digest_results: dict[Path, tuple[str, int]] = {}
    if full_hashes:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            results = executor.map(_digest_and_count, (item[0] for item in sources))
            digest_results = {
                source[0]: result
                for source, result in zip(sources, results, strict=True)
            }

    records: list[ManifestFile] = []
    for path, split, distribution in sources:
        scale = _scale_from_first_line(path)
        digest: str | None = None
        instances = _expected_instances(path, split)
        if full_hashes:
            digest, observed_lines = digest_results[path]
            if observed_lines != instances:
                raise ValueError(
                    f"{path}: 行数 {observed_lines} 与预期 {instances} 不一致"
                )
        records.append(
            ManifestFile(
                path=path.relative_to(base).as_posix(),
                split=split,
                distribution=distribution,
                scale=scale,
                instances=instances,
                bytes=path.stat().st_size,
                sha256=digest,
            )
        )

    duplicate_exclusions: list[DuplicateExclusion] = []
    kept_duplicate = base / "test_dataset/tsp/tsp100_concorde_7.756.txt"
    excluded_duplicate = (
        base / "test_dataset/tsp/tsp100_concorde_7.756 copy.txt"
    )
    if kept_duplicate.is_file() and excluded_duplicate.is_file():
        duplicate_hash: str | None = None
        if full_hashes:
            kept_hash = next(
                record.sha256
                for record in records
                if record.path == "test_dataset/tsp/tsp100_concorde_7.756.txt"
            )
            excluded_hash, excluded_lines = _digest_and_count(excluded_duplicate)
            if excluded_hash != kept_hash or excluded_lines != 1_280:
                raise ValueError("标记为 duplicate 的 TSP100 copy 与原文件不一致")
            duplicate_hash = excluded_hash
        duplicate_exclusions.append(
            DuplicateExclusion(
                kept="test_dataset/tsp/tsp100_concorde_7.756.txt",
                excluded="test_dataset/tsp/tsp100_concorde_7.756 copy.txt",
                sha256=duplicate_hash,
            )
        )

    return DataManifest(
        schema_version=1,
        generated_at=datetime.now(UTC).isoformat(),
        root=supplied_root.as_posix(),
        allowed_scales=(50, 100, 500),
        excluded_files=(
            "test_dataset/tsp/tsp100_concorde_7.756 copy.txt",
            "train_dataset/tsp/tsp_200/**",
            "train_dataset/tsp/tsp_1k/**",
            "train_dataset/tsp/tsp_10k/**",
            "test_dataset/tsp/tsp200_concorde_10.719.txt",
            "test_dataset/tsp/tsp1000_concorde_23.118.txt",
            "test_dataset/tsp/tsp10000_lkh_500_71.755.txt",
        ),
        duplicate_exclusions=tuple(duplicate_exclusions),
        hash_mode="sha256-full" if full_hashes else "inventory-only",
        files=tuple(records),
    )


def write_manifest(manifest: DataManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_manifest(path: str | Path) -> DataManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    files = tuple(ManifestFile(**item) for item in payload.pop("files"))
    duplicate_exclusions = tuple(
        DuplicateExclusion(**item)
        for item in payload.pop("duplicate_exclusions", [])
    )
    payload["allowed_scales"] = tuple(payload["allowed_scales"])
    payload["excluded_files"] = tuple(payload["excluded_files"])
    return DataManifest(
        files=files,
        duplicate_exclusions=duplicate_exclusions,
        **payload,
    )


def verify_manifest(
    manifest: DataManifest,
    *,
    root: str | Path | None = None,
    verify_hashes: bool = True,
    validate_first_record: bool = True,
) -> list[str]:
    """返回全部错误；空列表表示 inventory 与磁盘一致。"""

    base = Path(root).resolve() if root is not None else Path(manifest.root)
    errors: list[str] = []
    for record in manifest.files:
        path = base / record.path
        if not path.is_file():
            errors.append(f"缺少文件: {record.path}")
            continue
        if path.stat().st_size != record.bytes:
            errors.append(f"文件大小改变: {record.path}")
        if verify_hashes:
            if record.sha256 is None:
                errors.append(f"manifest 缺少 SHA-256: {record.path}")
            else:
                observed, observed_lines = _digest_and_count(path)
                if observed != record.sha256:
                    errors.append(f"SHA-256 不匹配: {record.path}")
                if observed_lines != record.instances:
                    errors.append(f"实例数不匹配: {record.path}")
        if validate_first_record:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    instance = parse_tsp_line(
                        handle.readline(),
                        instance_id=f"{record.path}:1",
                    )
                if instance.n != record.scale:
                    errors.append(f"首行规模不匹配: {record.path}")
            except ValueError as exc:
                errors.append(str(exc))
    for duplicate in manifest.duplicate_exclusions:
        kept = base / duplicate.kept
        excluded = base / duplicate.excluded
        if not kept.is_file() or not excluded.is_file():
            errors.append(f"duplicate exclusion 文件缺失: {duplicate.excluded}")
            continue
        if verify_hashes and duplicate.sha256 is not None:
            kept_hash, _ = _digest_and_count(kept)
            excluded_hash, _ = _digest_and_count(excluded)
            if kept_hash != duplicate.sha256 or excluded_hash != duplicate.sha256:
                errors.append(f"duplicate exclusion 内容改变: {duplicate.excluded}")
    return errors


def sampled_split_leakage(
    manifest: DataManifest,
    *,
    records_per_file: int = 1,
    root: str | Path | None = None,
) -> list[tuple[str, str, str]]:
    """检查每个文件前若干条坐标是否跨 split 重复。

    这是快速审计；正式冻结前仍应在完整数据上构建 coordinate-hash 索引。
    """

    base = Path(root).resolve() if root is not None else Path(manifest.root)
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[tuple[str, str, str]] = []
    for record in manifest.files:
        path = base / record.path
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number > records_per_file:
                    break
                instance = parse_tsp_line(
                    line,
                    instance_id=f"{record.path}:{line_number}",
                )
                previous = seen.get(instance.coordinate_hash)
                if previous is not None and previous[0] != record.split:
                    duplicates.append(
                        (
                            instance.coordinate_hash,
                            f"{previous[1]} ({previous[0]})",
                            f"{record.path}:{line_number} ({record.split})",
                        )
                    )
                else:
                    seen[instance.coordinate_hash] = (
                        record.split,
                        f"{record.path}:{line_number}",
                    )
    return duplicates
