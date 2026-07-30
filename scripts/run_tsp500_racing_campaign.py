#!/usr/bin/env python3
"""可恢复地调度 TSP500 racing audit 与 3 variants × 3 GP seeds。

每张物理 GPU 同时只运行一个进程。CUDA_VISIBLE_DEVICES 会把分配到的物理
设备映射为进程内 device 0。审计完成后，脚本用实测 kernel throughput
预测完整 run；若超过 72 小时，按冻结顺序把 finalists 32→24，再把高保真
实例 16→8。仍超预算时不启动正式训练。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
TEMPLATE = ROOT / "experiments" / "tsp500_2opt_racing" / "config.yaml"
RUN_ROOT = ROOT / "runs" / "tsp500-2opt-racing"
MANIFEST = ROOT / "Datasets" / "manifest.json"
VARIANTS = ("as", "acs", "mmas")
GP_SEEDS = (81001, 81002, 81003)
PARAMETERS = {
    "as": {"rho": 0.5, "q0": 0.0, "gamma": 1.0 / 3.0},
    "acs": {"rho": 0.1, "q0": 0.98, "gamma": 1.0 / 6.0},
    "mmas": {"rho": 0.2, "q0": 0.0, "gamma": 1.0 / 6.0},
}


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    command: tuple[str, ...]
    log: Path
    result: Path
    physical_gpu: int | None = None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return (
            json.loads(manifest.read_text(encoding="utf-8")).get("status")
            == "completed"
        )
    except (OSError, ValueError):
        return False


def _run_command(
    command: tuple[str, ...] | list[str],
    *,
    physical_gpu: int | None,
    log: Path,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if physical_gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{datetime.now(UTC).isoformat()}] "
            + (
                f"CUDA_VISIBLE_DEVICES={physical_gpu} "
                if physical_gpu is not None
                else ""
            )
            + " ".join(command)
            + "\n"
        )
        stream.flush()
        subprocess.run(
            list(command),
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _dispatch(
    tasks: list[Task],
    gpus: tuple[int, ...],
    *,
    phase: str,
) -> None:
    pending: queue.Queue[Task] = queue.Queue()
    for task in tasks:
        pending.put(task)
    lock = threading.Lock()
    state_path = RUN_ROOT / f"{phase}_state.json"
    active: dict[int, str] = {}
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def write_state(status: str) -> None:
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "phase": phase,
                "status": status,
                "active": dict(active),
                "completed": list(completed),
                "failures": list(failures),
                "pending_count": pending.qsize(),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def worker(gpu: int) -> None:
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                active[gpu] = task.name
                write_state("running")
            try:
                if not _completed(task.result):
                    _run_command(
                        task.command,
                        physical_gpu=gpu,
                        log=task.log,
                    )
                item = {
                    "task": task.name,
                    "physical_gpu": gpu,
                    "result": task.result.as_posix(),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
                with lock:
                    completed.append(item)
            except Exception as error:
                with lock:
                    failures.append(
                        {
                            "task": task.name,
                            "physical_gpu": gpu,
                            "error": repr(error),
                        }
                    )
            finally:
                with lock:
                    active.pop(gpu, None)
                    write_state("running")
                pending.task_done()

    workers = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in gpus
    ]
    write_state("running")
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    write_state("failed" if failures else "completed")
    if failures:
        raise RuntimeError(f"{phase} 有 {len(failures)} 个任务失败")


def _audit_tasks() -> list[Task]:
    return [
        Task(
            name=f"audit-{variant}",
            command=(
                str(PYTHON),
                str(ROOT / "scripts" / "audit_tsp500_racing.py"),
                "--config",
                str(TEMPLATE),
                "--variant",
                variant,
                "--output",
                str(RUN_ROOT / "audit" / variant),
            ),
            log=RUN_ROOT / "logs" / f"audit-{variant}.log",
            result=RUN_ROOT / "audit" / variant,
        )
        for variant in VARIANTS
    ]


def _audit_decisions() -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        path = RUN_ROOT / "audit" / variant / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        recommendation = payload.get("recommended_screen")
        decisions[variant] = {
            "formal_training_allowed": bool(
                payload.get("formal_training_allowed")
            ),
            "recommended_screen": recommendation,
        }
    return decisions


def _kernel_seconds_per_task_iteration(variant: str) -> float:
    """从 500 轮审计 shard 估计 TSP500 program×instance×iteration 成本。"""

    values = []
    for path in sorted(
        (RUN_ROOT / "audit" / variant / "shards").glob(
            "h00500-seed-*.npz"
        )
    ):
        with np.load(path, allow_pickle=False) as payload:
            kernel = float(payload["kernel_time_sec"])
            programs = int(payload["final_gap_percent"].shape[0])
            instances = int(payload["final_gap_percent"].shape[1])
            values.append(kernel / float(programs * instances * 500))
    if not values:
        raise FileNotFoundError(f"{variant} 缺少 500 轮 audit shard")
    return float(np.median(values))


def _project_hours(
    coefficient: float,
    *,
    finalists: int,
    high_instances: int,
    screen_instances: int,
    screen_schedule: tuple[tuple[int, int], ...],
) -> float:
    """估计单个 GP run；包含逐代 baseline 和最终两级 validation。"""

    high_schedule = ((15, 100), (35, 200), (50, 500))

    def scheduled_work(
        programs: int,
        instances: int,
        schedule: tuple[tuple[int, int], ...],
    ) -> int:
        previous = 0
        total = 0
        for end, iterations in schedule:
            total += (end - previous) * programs * instances * iterations
            previous = end
        return total

    learned = 99
    screen_work = scheduled_work(
        learned,
        screen_instances,
        screen_schedule,
    )
    high_work = scheduled_work(
        finalists + 1,
        high_instances,
        high_schedule,
    )
    # 最多 55 个 checkpoint 在 selection16×1 seed 上筛选；Top-5 再在
    # gate48×3 seeds 上运行。二者均使用正式 5000 轮。
    validation_work = (
        55 * 16 * 5000
        + 5 * 48 * 3 * 5000
    )
    # baseline 分别计算，但只占一个 program；计入约 10% 调度/传输余量。
    projected_seconds = (
        coefficient
        * (screen_work + high_work + validation_work)
        * 1.10
    )
    return projected_seconds / 3600.0


def _screen_schedule(recommended_horizon: int) -> tuple[tuple[int, int], ...]:
    defaults = ((15, 50), (35, 100), (50, 200))
    high = ((15, 100), (35, 200), (50, 500))
    return tuple(
        (
            end,
            min(
                high_iterations,
                max(iterations, recommended_horizon),
            ),
        )
        for (end, iterations), (_, high_iterations)
        in zip(defaults, high, strict=True)
    )


def _runtime_decision(
    audit: dict[str, dict[str, Any]],
    budget_hours: float,
) -> dict[str, Any]:
    allowed = [
        variant
        for variant in VARIANTS
        if audit[variant]["formal_training_allowed"]
    ]
    if not allowed:
        return {
            "formal_training_allowed": False,
            "reason": "所有 variants 均未通过学习信号门控",
        }
    coefficients = {
        variant: _kernel_seconds_per_task_iteration(variant)
        for variant in allowed
    }
    worst_coefficient = max(coefficients.values())
    recommended_horizon = max(
        int(audit[variant]["recommended_screen"]["horizon"])
        for variant in allowed
    )
    recommended_instances = max(
        int(audit[variant]["recommended_screen"]["instances"])
        for variant in allowed
    )
    schedule = _screen_schedule(recommended_horizon)
    finalists = 32
    high_instances = 16
    projected = _project_hours(
        worst_coefficient,
        finalists=finalists,
        high_instances=high_instances,
        screen_instances=recommended_instances,
        screen_schedule=schedule,
    )
    reductions: list[str] = []
    if projected > budget_hours:
        finalists = 24
        reductions.append("finalists:32->24")
        projected = _project_hours(
            worst_coefficient,
            finalists=finalists,
            high_instances=high_instances,
            screen_instances=recommended_instances,
            screen_schedule=schedule,
        )
    if projected > budget_hours and recommended_instances <= 8:
        high_instances = 8
        reductions.append("high_instances:16->8")
        projected = _project_hours(
            worst_coefficient,
            finalists=finalists,
            high_instances=high_instances,
            screen_instances=min(recommended_instances, high_instances),
            screen_schedule=schedule,
        )
    permitted = projected <= budget_hours
    return {
        "formal_training_allowed": permitted,
        "reason": (
            "projected runtime within budget"
            if permitted
            else "projected runtime still exceeds budget after frozen reductions"
        ),
        "coefficients_sec_per_task_iteration": coefficients,
        "worst_coefficient": worst_coefficient,
        "projected_hours_per_run": projected,
        "budget_hours": budget_hours,
        "finalists": finalists,
        "high_instances": high_instances,
        "screen_instances": recommended_instances,
        "screen_horizon_schedule": [list(item) for item in schedule],
        "reductions": reductions,
    }


def _write_formal_config(
    *,
    variant: str,
    root_seed: int,
    runtime_decision: dict[str, Any],
) -> Path:
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    parameters = PARAMETERS[variant]
    identifier = f"tsp500-2opt-racing-{variant}-seed-{root_seed}"
    payload["experiment"]["experiment_id"] = identifier
    payload["experiment"]["root_seed"] = root_seed
    payload["aco"].update(
        {
            "variant": variant,
            "rho": parameters["rho"],
            "q0": parameters["q0"],
            "gamma_transition": parameters["gamma"],
            "gamma_pheromone": parameters["gamma"],
        }
    )
    payload["racing"].update(
        {
            "screen_instances_per_scale": runtime_decision[
                "screen_instances"
            ],
            "high_instances_per_scale": runtime_decision["high_instances"],
            "finalists": runtime_decision["finalists"],
            "screen_horizon_schedule": runtime_decision[
                "screen_horizon_schedule"
            ],
        }
    )
    target = (
        RUN_ROOT
        / "formal"
        / "configs"
        / f"{identifier}.yaml"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.yaml")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _prepare_schedules(configs: dict[tuple[str, int], Path]) -> None:
    for replicate, root_seed in enumerate(GP_SEEDS):
        target = (
            RUN_ROOT
            / "formal"
            / "schedules"
            / f"seed-{root_seed}.json"
        )
        if target.is_file():
            continue
        config = configs[("as", root_seed)]
        _run_command(
            [
                str(PYTHON),
                "-m",
                "rmtgp_aco",
                "prepare-schedules",
                "--config",
                str(config),
                "--output",
                str(target),
                "--phase",
                "formal",
                "--replicate-id",
                str(replicate),
                "--manifest",
                str(MANIFEST),
                "--skip-manifest-check",
            ],
            physical_gpu=None,
            log=RUN_ROOT
            / "logs"
            / f"schedule-seed-{root_seed}.log",
        )


def _formal_tasks(
    configs: dict[tuple[str, int], Path],
    allowed_variants: list[str],
) -> list[Task]:
    tasks: list[Task] = []
    for variant in allowed_variants:
        for replicate, root_seed in enumerate(GP_SEEDS):
            run = (
                RUN_ROOT
                / "formal"
                / "train"
                / variant
                / f"seed-{root_seed}"
            )
            command = [
                str(PYTHON),
                "-m",
                "rmtgp_aco",
                "train",
                "--config",
                str(configs[(variant, root_seed)]),
                "--phase",
                "formal",
                "--replicate-id",
                str(replicate),
                "--schedule",
                str(
                    RUN_ROOT
                    / "formal"
                    / "schedules"
                    / f"seed-{root_seed}.json"
                ),
                "--manifest",
                str(MANIFEST),
                "--skip-manifest-check",
                "--method-profile",
                "rmtgp",
                "--output",
                str(run),
                "--backend",
                "cuda_tiled_v2",
                "--gpu-devices",
                "0",
                "--traceback",
            ]
            if (run / "training_state.pkl").is_file():
                command.extend(["--resume", str(run)])
            tasks.append(
                Task(
                    name=f"{variant}-seed-{root_seed}",
                    command=tuple(command),
                    log=(
                        RUN_ROOT
                        / "logs"
                        / f"formal-{variant}-seed-{root_seed}.log"
                    ),
                    result=run,
                )
            )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "audit", "formal"),
        default="all",
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--runtime-budget-hours", type=float, default=72.0)
    args = parser.parse_args()
    gpus = tuple(dict.fromkeys(int(gpu) for gpu in args.gpus))
    if not gpus or min(gpus) < 0:
        parser.error("--gpus 必须是非空非负整数列表")
    if args.runtime_budget_hours <= 0.0:
        parser.error("--runtime-budget-hours 必须为正数")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase in {"all", "audit"}:
        _dispatch(_audit_tasks(), gpus, phase="audit")
        if args.phase == "audit":
            return 0

    audit = _audit_decisions()
    runtime_decision = _runtime_decision(
        audit,
        args.runtime_budget_hours,
    )
    decision_payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "audit": audit,
        "runtime": runtime_decision,
    }
    _atomic_json(RUN_ROOT / "formal_decision.json", decision_payload)
    if not runtime_decision["formal_training_allowed"]:
        raise RuntimeError(runtime_decision["reason"])
    allowed_variants = [
        variant
        for variant in VARIANTS
        if audit[variant]["formal_training_allowed"]
    ]
    configs = {
        (variant, root_seed): _write_formal_config(
            variant=variant,
            root_seed=root_seed,
            runtime_decision=runtime_decision,
        )
        for variant in allowed_variants
        for root_seed in GP_SEEDS
    }
    # prepare-schedules 只需任一 variant；若 AS 未通过，选择第一个通过者。
    if "as" not in allowed_variants:
        for root_seed in GP_SEEDS:
            configs[("as", root_seed)] = configs[
                (allowed_variants[0], root_seed)
            ]
    _prepare_schedules(configs)
    _dispatch(
        _formal_tasks(configs, allowed_variants),
        gpus,
        phase="formal",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
