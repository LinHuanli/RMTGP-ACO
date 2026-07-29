#!/usr/bin/env python3
"""从最大预算 schedule 生成逐代严格嵌套的小预算 schedule。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from rmtgp_aco.schedule import (
    ScheduleManifest,
    load_schedule,
    write_schedule,
)


def _parse_output(value: str) -> tuple[int, Path]:
    try:
        budget_text, path_text = value.split("=", 1)
        budget = int(budget_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--output 必须写成 BUDGET=PATH"
        ) from exc
    if budget < 1 or not path_text:
        raise argparse.ArgumentTypeError("budget 和输出路径必须有效")
    return budget, Path(path_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 128-instance schedule 逐代切成嵌套的 64/32 schedule",
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output",
        action="append",
        required=True,
        type=_parse_output,
        metavar="BUDGET=PATH",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = load_schedule(args.source)
    train_counts = {
        len(record.logical_indices)
        for record in source.records
        if record.split == "train"
    }
    if len(train_counts) != 1:
        raise ValueError("源 schedule 的每代训练实例数不一致")
    maximum = train_counts.pop()

    seen: set[int] = set()
    for budget, target in args.output:
        if budget in seen:
            raise ValueError(f"重复 budget={budget}")
        if budget > maximum:
            raise ValueError(f"budget={budget} 超过源预算 {maximum}")
        seen.add(budget)
        records = tuple(
            replace(
                record,
                logical_indices=record.logical_indices[:budget],
                instance_ids=record.instance_ids[:budget],
                coordinate_hashes=record.coordinate_hashes[:budget],
            )
            if record.split == "train"
            else record
            for record in source.records
        )
        manifest = ScheduleManifest(
            schema_version=source.schema_version,
            protocol_id=source.protocol_id,
            phase=source.phase,
            root_seed=source.root_seed,
            generated_at=source.generated_at,
            records=records,
        )
        output = write_schedule(manifest, target)
        print(
            f"budget={budget} records={len(records)} "
            f"hash={manifest.manifest_hash} output={output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
