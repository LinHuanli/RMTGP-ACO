#!/usr/bin/env python3
"""GPU1 上可恢复地执行 18 个深树 RMTGP-ACO 局部搜索训练。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
CONFIG = ROOT / "experiments" / "tsp100_local_search_3seed" / "config.yaml"
RUN_ROOT = ROOT / "runs" / "tsp100-local-search-3seed"
MANIFEST = ROOT / "Datasets" / "manifest.json"
TUNING = ROOT / "configs/cuda_tuning/rtx_pro5000_blackwell_sm120_v1.json"
VARIANTS = ("as", "acs", "mmas")
ENVIRONMENTS = ("none", "two_opt")
SEEDS = (2001, 2002, 2003)


@dataclass(frozen=True, slots=True)
class Task:
    variant: str
    environment: str
    seed: int
    replicate: int

    @property
    def name(self) -> str:
        return f"{self.variant}-{self.environment}-seed-{self.seed}"

    @property
    def schedule(self) -> Path:
        return RUN_ROOT / "schedules" / f"seed-{self.seed}.json"

    @property
    def baseline(self) -> Path:
        return (
            RUN_ROOT
            / "baselines"
            / self.variant
            / self.environment
            / f"seed-{self.seed}"
            / "baseline.npz"
        )

    @property
    def two_opt_baseline_root(self) -> Path:
        return (
            RUN_ROOT
            / "baselines"
            / self.variant
            / "two_opt"
            / f"seed-{self.seed}"
        )

    @property
    def run(self) -> Path:
        return (
            RUN_ROOT
            / "train"
            / self.variant
            / self.environment
            / f"seed-{self.seed}"
        )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, ValueError):
        return False


def _run(command: list[str], *, physical_gpu: int, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
        "PYTHONUNBUFFERED": "1",
    }
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{datetime.now(UTC).isoformat()}] "
            f"CUDA_VISIBLE_DEVICES={physical_gpu} {' '.join(command)}\n"
        )
        stream.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _profile_args(task: Task) -> list[str]:
    return [
        "--config",
        str(CONFIG),
        "--root-seed",
        str(task.seed),
        "--replicate-id",
        str(task.replicate),
        "--aco-variant",
        task.variant,
        "--local-search",
        task.environment,
        "--experiment-id",
        f"tsp100-ls-{task.name}",
        "--backend",
        "cuda_tiled_v2",
        "--gpu-devices",
        "0",
        "--gpu-mode",
        "single",
        "--cuda-tuning-manifest",
        str(TUNING),
    ]


def _prepare_schedule(task: Task, physical_gpu: int) -> None:
    if task.schedule.is_file():
        return
    _run(
        [
            str(PYTHON),
            "-m",
            "rmtgp_aco",
            "prepare-schedules",
            "--config",
            str(CONFIG),
            "--output",
            str(task.schedule),
            "--phase",
            "formal",
            "--root-seed",
            str(task.seed),
            "--replicate-id",
            str(task.replicate),
            "--manifest",
            str(MANIFEST),
        ],
        physical_gpu=physical_gpu,
        log=RUN_ROOT / "logs" / f"schedule-seed-{task.seed}.log",
    )


def _execute(task: Task, physical_gpu: int, *, cpu_audit: bool) -> dict[str, object]:
    started = perf_counter()
    log = RUN_ROOT / "logs" / f"{task.name}.log"
    _prepare_schedule(task, physical_gpu)
    if not task.baseline.is_file():
        _run(
            [
                str(PYTHON),
                "-m",
                "rmtgp_aco",
                "precompute-baselines",
                *_profile_args(task),
                "--schedule",
                str(task.schedule),
                "--splits",
                "train",
                "selection",
                "gate",
                "--output",
                str(task.baseline),
            ],
            physical_gpu=physical_gpu,
            log=log,
        )
    if not _completed(task.run):
        command = [
            str(PYTHON),
            "-m",
            "rmtgp_aco",
            "train",
            *_profile_args(task),
            "--phase",
            "formal",
            "--manifest",
            str(MANIFEST),
            "--schedule",
            str(task.schedule),
            "--baseline-archive",
            str(task.baseline.parent),
            "--method-profile",
            "rmtgp-full-f1",
            "--output",
            str(task.run),
            "--traceback",
        ]
        if (task.run / "training_state.pkl").is_file():
            command.extend(["--resume", str(task.run)])
        _run(command, physical_gpu=physical_gpu, log=log)

    standardized = task.run / "selected_candidate_2opt.pkl"
    if not standardized.is_file():
        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "select_local_search_checkpoints.py"),
            "--config",
            str(CONFIG),
            "--variant",
            task.variant,
            "--seed",
            str(task.seed),
            "--replicate-id",
            str(task.replicate),
            "--schedule",
            str(task.schedule),
            "--baseline-archive",
            str(task.two_opt_baseline_root),
            "--run",
            str(task.run),
            "--output",
            str(standardized),
        ]
        if not cpu_audit:
            command.append("--no-cpu-audit")
        _run(command, physical_gpu=physical_gpu, log=log)
    return {
        **asdict(task),
        "task": task.name,
        "seconds": perf_counter() - started,
        "completed_at": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument(
        "--cpu-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--launch-final-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    # 同一 variant/seed 先训练 two_opt，确保 none run 的统一选择已有 LS baseline。
    tasks = [
        Task(variant, environment, seed, replicate)
        for variant in VARIANTS
        for replicate, seed in enumerate(SEEDS)
        for environment in ("two_opt", "none")
    ]
    completed: list[dict[str, object]] = []
    state = RUN_ROOT / "campaign_state.json"
    for index, task in enumerate(tasks):
        _atomic_json(
            state,
            {
                "schema_version": 1,
                "status": "running",
                "active": task.name,
                "completed": completed,
                "pending": [item.name for item in tasks[index + 1 :]],
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        try:
            completed.append(
                _execute(
                    task,
                    args.physical_gpu,
                    cpu_audit=args.cpu_audit,
                )
            )
        except Exception as error:
            _atomic_json(
                state,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "active": task.name,
                    "completed": completed,
                    "error": repr(error),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            raise

    if args.launch_final_test:
        _atomic_json(
            state,
            {
                "schema_version": 1,
                "status": "final_test_running",
                "active": "final-test-5000-iterations",
                "completed": completed,
                "pending": [],
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        try:
            _run(
                [
                    str(PYTHON),
                    str(ROOT / "scripts" / "evaluate_local_search_final.py"),
                    "--physical-gpu",
                    str(args.physical_gpu),
                ],
                physical_gpu=args.physical_gpu,
                log=RUN_ROOT / "logs" / "final-test.log",
            )
        except Exception as error:
            _atomic_json(
                state,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "active": "final-test-5000-iterations",
                    "completed": completed,
                    "pending": [],
                    "error": repr(error),
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            raise
    _atomic_json(
        state,
        {
            "schema_version": 1,
            "status": "completed",
            "active": None,
            "completed": completed,
            "pending": [],
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
