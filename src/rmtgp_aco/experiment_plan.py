"""Protocol A v0.4 pilot 的可审计命令矩阵。"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

PROTOCOL_ID = "protocol-a-v0.4"

MAIN_METHOD_PROFILES = (
    "legacy",
    "matched-replace",
    "tr-rgp",
    "ph-rgp",
    "rmtgp-core-f0",
    "rmtgp-core-f1",
    "rmtgp-full-f0",
    "rmtgp-full-f1",
)

_MAIN_PROTOCOLS = (
    ("as", "configs/as_protocol_a.yaml", (1001, 1002, 1003)),
    ("acs", "configs/acs_protocol_a.yaml", (2001, 2002, 2003)),
    ("mmas", "configs/mmas_protocol_a.yaml", (3001, 3002, 3003)),
)

_SPECIALIST_PROTOCOLS = (
    (
        "acs-tsp50-only",
        "configs/acs_protocol_a_tsp50_only.yaml",
        (2001, 2002, 2003),
    ),
    (
        "acs-tsp100-only",
        "configs/acs_protocol_a_tsp100_only.yaml",
        (2001, 2002, 2003),
    ),
)


def _base_command(python: str) -> list[str]:
    return [python, "-m", "rmtgp_aco"]


def build_protocol_a_v04_pilot_plan(
    *,
    runs_root: str | Path = "runs/protocol-a-v0.4",
    python: str | None = None,
) -> dict[str, object]:
    """生成 72 个主消融 run 与 6 个 ACS 单尺度 run。"""

    executable = python or sys.executable
    root = Path(runs_root)
    setup_tasks: list[dict[str, object]] = []
    training_tasks: list[dict[str, object]] = []

    protocols = [
        *((name, config, seeds, MAIN_METHOD_PROFILES) for name, config, seeds in _MAIN_PROTOCOLS),
        *(
            (name, config, seeds, ("rmtgp-full-f1",))
            for name, config, seeds in _SPECIALIST_PROTOCOLS
        ),
    ]
    for protocol_name, config, seeds, methods in protocols:
        variant = protocol_name.split("-", maxsplit=1)[0]
        for replicate_id, root_seed in enumerate(seeds):
            schedule = root / "schedules" / f"{protocol_name}-seed-{root_seed}.json"
            baseline = (
                root
                / "baselines"
                / variant
                / f"{protocol_name}-seed-{root_seed}.npz"
            )
            schedule_task_id = f"schedule-{protocol_name}-{root_seed}"
            baseline_task_id = f"baseline-{protocol_name}-{root_seed}"
            setup_tasks.append(
                {
                    "task_id": schedule_task_id,
                    "kind": "schedule",
                    "dependencies": [],
                    "command": [
                        *_base_command(executable),
                        "prepare-schedules",
                        "--config",
                        config,
                        "--phase",
                        "pilot",
                        "--replicate-id",
                        str(replicate_id),
                        "--root-seed",
                        str(root_seed),
                        "--output",
                        str(schedule),
                    ],
                }
            )
            setup_tasks.append(
                {
                    "task_id": baseline_task_id,
                    "kind": "baseline",
                    "dependencies": [schedule_task_id],
                    "command": [
                        *_base_command(executable),
                        "precompute-baselines",
                        "--config",
                        config,
                        "--schedule",
                        str(schedule),
                        "--replicate-id",
                        str(replicate_id),
                        "--root-seed",
                        str(root_seed),
                        "--output",
                        str(baseline),
                    ],
                }
            )
            for method in methods:
                task_id = f"train-{protocol_name}-{method}-{root_seed}"
                output = (
                    root
                    / "pilot"
                    / protocol_name
                    / method
                    / f"seed-{root_seed}"
                )
                training_tasks.append(
                    {
                        "task_id": task_id,
                        "kind": "train",
                        "protocol": protocol_name,
                        "method_profile": method,
                        "root_seed": root_seed,
                        "replicate_id": replicate_id,
                        "dependencies": [baseline_task_id],
                        "command": [
                            *_base_command(executable),
                            "train",
                            "--config",
                            config,
                            "--phase",
                            "pilot",
                            "--schedule",
                            str(schedule),
                            "--baseline-archive",
                            str(baseline.parent),
                            "--method-profile",
                            method,
                            "--replicate-id",
                            str(replicate_id),
                            "--root-seed",
                            str(root_seed),
                            "--output",
                            str(output),
                        ],
                    }
                )

    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "phase": "pilot",
        "cpu_threads_per_run": 16,
        "main_methods": list(MAIN_METHOD_PROFILES),
        "setup_task_count": len(setup_tasks),
        "training_task_count": len(training_tasks),
        "expected_training_task_count": 78,
        "setup_tasks": setup_tasks,
        "training_tasks": training_tasks,
    }


def write_experiment_plan(plan: dict[str, object], path: str | Path) -> tuple[Path, Path]:
    """同时写 JSON 任务图和便于本机顺序复现的 shell 命令。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shell_target = target.with_suffix(".sh")
    tasks = [*plan["setup_tasks"], *plan["training_tasks"]]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# 本文件顺序执行全部任务；集群调度应读取 JSON 中的 dependencies。",
    ]
    for task in tasks:
        lines.append("")
        lines.append(f"# {task['task_id']}")
        lines.append(shlex.join(task["command"]))
    shell_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shell_target.chmod(0o755)
    return target, shell_target
