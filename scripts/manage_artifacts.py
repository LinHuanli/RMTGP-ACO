"""按登记表盘点或迁移；不移动活跃队列，不删除原始数据，不覆盖目标。"""

import argparse
import json
from datetime import datetime, timezone

from gpaco.artifact_registry import entries, resolve
from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import file_hash


def inventory():
    rows = []
    for entry in entries():
        actual = resolve(entry["id"])
        files = sorted(p for p in actual.rglob("*") if p.is_file()) if actual.exists() else []
        rows.append(
            {
                **entry,
                "actual_path": str(actual.relative_to(ROOT)) if actual.exists() else None,
                "storage_status": (
                    "not_created"
                    if not actual.exists()
                    else "canonical_with_legacy_alias"
                    if entry.get("legacy_path")
                    and (ROOT / entry["legacy_path"]).is_symlink()
                    and (ROOT / entry["legacy_path"]).resolve() == actual.resolve()
                    else "canonical"
                    if actual == ROOT / entry["path"]
                    else "legacy_deferred"
                    if entry["migration"] == "defer_live"
                    else "ready_to_migrate"
                ),
                "files": len(files),
                "logical_bytes": sum(p.stat().st_size for p in files),
            }
        )
    return rows


def publish_inventory(rows):
    stamp = datetime.now(timezone.utc).isoformat()
    output = ROOT / "docs/results/status"
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "storage_inventory.json",
        {
            "observed_at_utc": stamp,
            "scope": "非原子只读文件盘点；存活实验仍可新增文件；不是进程存活或实验完成证明",
            "registry_sha256": file_hash(ROOT / "configs/experiments/registry.yaml"),
            "entries": rows,
        },
    )
    lines = [
        "# 产物盘点与迁移状态",
        "",
        f"盘点时间（UTC）：{stamp}。大小是逻辑文件字节，不是磁盘占用；活跃目录仍会变化。",
        "",
        "本表的状态是**存储状态**，不是实验完成状态。原始记录不会因目录整理升级为正式结果。",
        "",
        "| 登记ID | 类别 | 当前地址 | 文件数 | MiB | 存储状态 |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        path = f"`{row['actual_path']}`" if row["actual_path"] else "尚未创建"
        lines.append(
            f"| {row['id']} | {row['tier']} | {path} | {row['files']} | "
            f"{row['logical_bytes'] / 2**20:.2f} | {row['storage_status']} |"
        )
    lines.extend(["", "## 暂缓迁移的原因", ""])
    lines.extend(
        f"- `{row['id']}`：{row['reason']}。"
        for row in rows
        if row["storage_status"] == "legacy_deferred"
    )
    lines.extend(["", "## 已迁移且保留旧地址兼容入口", ""])
    lines.extend(
        f"- `{row['legacy_path']}` 是指向 `{row['path']}` 的兼容软链接；数据只在新目录中保存一份。"
        for row in rows
        if row["storage_status"] == "canonical_with_legacy_alias"
    )
    lines.extend(
        [
            "",
            "登记源：`configs/experiments/registry.yaml`。迁移逐文件SHA日志在 `artifacts/operations/migrations/`。",
            "",
            "复查命令：`python scripts/manage_artifacts.py inventory --write`。新实验启动后应再次生成盘点。",
        ]
    )
    (output / "storage_inventory.md").write_text("\n".join(lines) + "\n")
    artifact_lines = [
        "# 实验产物入口",
        "",
        "原始产物按证据等级和E00–E13登记。论文图表请从 [结果索引](../docs/results/README.md) 阅读。",
        "",
        "- `runs/`：formal、pilot、diagnostic、smoke分开存储。",
        "- `inputs/`：只读冻结输入；不是实验结果。",
        "- `cache/`：共享可复算缓存。",
        "- `provenance/`：环境、源码、历史资料和迁移前备份。",
        "- `operations/`：运行管理和迁移日志。",
        "",
        "## 暂存的旧路径",
        "",
        "以下地址仍被后台任务或其消费者依赖。它们不是新的命名范例，不移动、不复制锁、不做软链接伪迁移。",
        "",
    ]
    artifact_lines.extend(
        f"- `{row['actual_path']}` → `{row['path']}`：{row['reason']}。"
        for row in rows
        if row["storage_status"] == "legacy_deferred"
    )
    artifact_lines.extend(["", "## 兼容入口（不是第二份数据）", ""])
    artifact_lines.extend(
        f"- `{row['legacy_path']}` → `{row['path']}`。已物理迁移，旧入口只服务不可变历史地址。"
        for row in rows
        if row["storage_status"] == "canonical_with_legacy_alias"
    )
    artifact_lines.extend(
        [
            "",
            "详见 [当前盘点](../docs/results/status/storage_inventory.md) 和 [目录管理规范](../docs/design/08_artifact_and_result_management.md)。",
        ]
    )
    (ROOT / "artifacts/README.md").write_text("\n".join(artifact_lines) + "\n")


def tree_hashes(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"迁移目标含软链接，先人工核查：{path}")
        if path.is_file():
            result[str(path.relative_to(directory))] = file_hash(path)
    return result


def migrate(selected, execute, verified_inactive, compatibility_alias=False):
    rows = entries()
    unknown = set(selected) - {row["id"] for row in rows}
    if unknown:
        raise ValueError(f"未登记ID：{unknown}")
    for row in rows:
        if row["id"] not in selected:
            continue
        if row["migration"] not in ("move_inactive", "move_idle_controller"):
            raise ValueError(f"禁止本次迁移：{row['id']} / {row['migration']}")
        if row["migration"] == "move_idle_controller" and not compatibility_alias:
            raise ValueError("旧队列记录含绝对地址；此次必须显式保留兼容入口")
        source, target = ROOT / row["legacy_path"], ROOT / row["path"]
        if not source.exists() or target.exists():
            raise ValueError(f"迁移要求源存在且目标不存在：{source} -> {target}")
        print(f"{row['id']}: {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}", flush=True)
        if not execute:
            continue
        if not verified_inactive:
            raise ValueError("执行前须核查相关写入者和消费者退出，并提供 --verified-inactive")
        # 操作者核查是必要条件；不能从旧status文件或目录名称推断进程已退出。
        before = tree_hashes(source)
        log = ROOT / f"artifacts/operations/migrations/{row['id']}.json"
        if log.exists():
            raise FileExistsError(f"已有迁移日志，不覆盖：{log}")
        record = {
            "registry_id": row["id"],
            "from": row["legacy_path"],
            "to": row["path"],
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "operator_verified_inactive": True,
            "file_sha256": before,
            "status": "prepared",
            "deleted_experiment_data": False,
            "compatibility_alias": compatibility_alias,
        }
        write_json(log, record)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        if compatibility_alias:
            # 只在写入者均已退出后建立；不用于热迁移运行中的NFS目录。
            source.symlink_to(target.relative_to(source.parent), target_is_directory=True)
        if tree_hashes(target) != before:
            write_json(log, {**record, "status": "verification_failed"})
            raise RuntimeError("迁移后校验失败；保留现场，不自动删除或覆盖")
        write_json(log, {**record, "status": "verified"})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("inventory")
    scan.add_argument("--write", action="store_true")
    move = sub.add_parser("migrate")
    move.add_argument("--id", action="append", required=True)
    move.add_argument("--execute", action="store_true")
    move.add_argument("--verified-inactive", action="store_true")
    move.add_argument("--compatibility-alias", action="store_true")
    args = parser.parse_args()
    if args.command == "migrate":
        migrate(args.id, args.execute, args.verified_inactive, args.compatibility_alias)
    else:
        rows = inventory()
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        if args.write:
            publish_inventory(rows)


if __name__ == "__main__":
    main()
