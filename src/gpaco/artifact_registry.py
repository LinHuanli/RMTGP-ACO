"""实验产物身份及路径约束；历史路径只读兼容，不悄悄重命名运行中的任务。"""

from pathlib import Path

import yaml

from .data import ROOT

REGISTRY = ROOT / "configs/experiments/registry.yaml"


def entries():
    rows = yaml.safe_load(REGISTRY.read_text())["entries"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("产物登记ID重复")
    for row in rows:
        for field in ("path", "legacy_path", "report"):
            if field not in row:
                continue
            p = (ROOT / row[field]).resolve()
            if p == ROOT or not p.is_relative_to(ROOT):
                raise ValueError(f"登记路径越界：{row['id']}/{field}")
    return rows


def entry(identifier):
    return next(row for row in entries() if row["id"] == identifier)


def resolve(identifier):
    """目标存在则用目标；迁移前用明确登记的旧地址；两份并存时拒绝猜测。"""
    row = entry(identifier)
    target = ROOT / row["path"]
    legacy = ROOT / row["legacy_path"] if row.get("legacy_path") else None
    if target.exists() and legacy is not None and legacy.exists():
        if legacy.is_symlink() and legacy.resolve() == target.resolve():
            return target
        raise ValueError(f"同一产物出现两份，必须人工核对：{identifier}")
    return legacy if legacy is not None and legacy.exists() else target


def require_output(directory, identifier):
    """新增执行入口只写登记的规范子树，不接受随手命名的顶层目录。"""
    directory = Path(directory).resolve()
    row = entry(identifier)
    base = (ROOT / row["path"]).resolve()
    if directory == base or not directory.is_relative_to(base):
        raise ValueError(f"输出必须是 {row['path']}/ 下的独立运行目录")
    return directory


def identity(identifier):
    row = entry(identifier)
    return {
        "registry_id": identifier,
        "evidence_tier": row["tier"],
        "experiment_ids": row["experiments"],
        "protocol_id": row["protocol"],
        "formal_result": row["tier"] == "formal",
    }
