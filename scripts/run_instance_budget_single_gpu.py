#!/usr/bin/env python3
"""在多张物理 GPU 上并发执行若干“严格单卡”的实例预算训练。

每个 worker 同一时间只运行一个 baseline 或训练进程。子进程只看见一张
物理 GPU，并将其作为逻辑设备 0 使用，因此不会发生双卡 task sharding。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
EXPERIMENT = ROOT / "experiments" / "tsp100_instance_budget_single_gpu"
RUN_ROOT = ROOT / "runs" / "tsp100-instance-budget-single-gpu"
TUNING = ROOT / "configs/cuda_tuning/rtx_pro5000_blackwell_sm120_v1.json"
MANIFEST = ROOT / "Datasets/manifest.json"
SEEDS = (2001, 2002, 2003)
BUDGETS = (32, 64, 128)


@dataclass(frozen=True, slots=True)
class RunTask:
    budget: int
    seed: int
    replicate: int

    @property
    def name(self) -> str:
        return f"n{self.budget}-seed-{self.seed}"

    @property
    def config(self) -> Path:
        return EXPERIMENT / "configs" / f"acs_n{self.budget}.yaml"

    @property
    def schedule(self) -> Path:
        return RUN_ROOT / "schedules" / f"{self.name}.json"

    @property
    def baseline(self) -> Path:
        return (
            RUN_ROOT
            / "baselines"
            / f"n{self.budget}"
            / f"seed-{self.seed}"
            / "baseline.npz"
        )

    @property
    def run(self) -> Path:
        return RUN_ROOT / "train" / f"n{self.budget}" / f"seed-{self.seed}"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed(run: Path) -> bool:
    manifest = run / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(
            manifest.read_text(encoding="utf-8")
        ).get("status") == "completed"
    except (OSError, ValueError):
        return False


def _common(task: RunTask) -> list[str]:
    return [
        "--config",
        str(task.config),
        "--root-seed",
        str(task.seed),
        "--replicate-id",
        str(task.replicate),
        "--backend",
        "cuda_tiled_v2",
        "--gpu-devices",
        "0",
        "--gpu-mode",
        "single",
        "--cuda-tuning-manifest",
        str(TUNING),
    ]


def _run_command(
    command: list[str],
    *,
    physical_gpu: int,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
        "PYTHONUNBUFFERED": "1",
    }
    with log_path.open("a", encoding="utf-8") as stream:
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


def _execute(task: RunTask, physical_gpu: int) -> dict[str, object]:
    started = perf_counter()
    task_log = RUN_ROOT / "logs" / f"{task.name}.log"
    if not task.baseline.is_file():
        _run_command(
            [
                str(PYTHON),
                "-m",
                "rmtgp_aco",
                "precompute-baselines",
                *_common(task),
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
            log_path=task_log,
        )

    if not _completed(task.run):
        command = [
            str(PYTHON),
            "-m",
            "rmtgp_aco",
            "train",
            *_common(task),
            "--phase",
            "pilot",
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
        _run_command(
            command,
            physical_gpu=physical_gpu,
            log_path=task_log,
        )
    return {
        **asdict(task),
        "task": task.name,
        "physical_gpu": physical_gpu,
        "seconds": perf_counter() - started,
        "completed_at": datetime.now(UTC).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physical-gpus",
        nargs="+",
        type=int,
        default=[0, 1],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(set(args.physical_gpus)) != len(args.physical_gpus):
        raise ValueError("physical GPU 不得重复")
    missing = [
        path
        for task in (
            RunTask(budget, seed, replicate)
            for budget in BUDGETS
            for replicate, seed in enumerate(SEEDS)
        )
        for path in (task.config, task.schedule)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"缺少配置或 schedule: {missing[:3]}")

    # 大预算优先。动态队列使两张卡的累计负载更接近。
    tasks = deque(
        RunTask(budget, seed, replicate)
        for budget in reversed(BUDGETS)
        for replicate, seed in enumerate(SEEDS)
    )
    lock = threading.Lock()
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    active: dict[int, str] = {}
    state_path = RUN_ROOT / "campaign_state.json"

    def write_state() -> None:
        status = (
            "failed"
            if failures
            else ("completed" if not tasks and not active else "running")
        )
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": status,
                "active": {
                    str(gpu): name
                    for gpu, name in sorted(active.items())
                },
                "pending": [item.name for item in tasks],
                "completed": records,
                "failures": failures,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def worker(physical_gpu: int) -> None:
        while True:
            with lock:
                if not tasks:
                    return
                task = tasks.popleft()
                active[physical_gpu] = task.name
                write_state()
            try:
                result = _execute(task, physical_gpu)
                with lock:
                    records.append(result)
            except Exception as exc:  # noqa: BLE001 - 必须持久化后台失败
                with lock:
                    failures.append(
                        {
                            **asdict(task),
                            "task": task.name,
                            "physical_gpu": physical_gpu,
                            "error": f"{type(exc).__name__}: {exc}",
                            "failed_at": datetime.now(UTC).isoformat(),
                        }
                    )
            with lock:
                active.pop(physical_gpu, None)
                write_state()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_state()
    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in args.physical_gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
