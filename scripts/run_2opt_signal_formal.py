#!/usr/bin/env python3
"""调度 TSP100 full-2opt 的 3 variants × 3 GP seeds 确认性训练。"""

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

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
TEMPLATE = ROOT / "experiments" / "tsp100_2opt_signal_v2" / "config.yaml"
RUN_ROOT = ROOT / "runs" / "tsp100-2opt-signal-v2" / "formal"
AUDIT_ROOT = ROOT / "runs" / "tsp100-2opt-signal-v2" / "audit"
MANIFEST = ROOT / "Datasets" / "manifest.json"
VARIANTS = ("as", "acs", "mmas")
GP_SEEDS = (71001, 71002, 71003)


@dataclass(frozen=True, slots=True)
class FormalTask:
    variant: str
    root_seed: int
    replicate: int
    config: Path
    schedule: Path
    baseline: Path
    run: Path
    log: Path

    @property
    def name(self) -> str:
        return f"{self.variant}-seed-{self.root_seed}"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _variant_parameters(variant: str) -> dict[str, float]:
    return {
        "as": {"rho": 0.5, "q0": 0.0},
        "acs": {"rho": 0.1, "q0": 0.98},
        "mmas": {"rho": 0.2, "q0": 0.0},
    }[variant]


def _audit_origin_gate(variant: str) -> bool:
    path = AUDIT_ROOT / variant / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {variant} audit summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload["horizon_summaries"][-1]["origin_gate"]["enabled"])


def _load_decisions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {
            variant: {
                "fitness_mode": "paired_combined_ucb",
                "gamma": 1.0 / 3.0,
                "origin": False,
            }
            for variant in VARIANTS
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("variants", payload)
    if set(decisions) != set(VARIANTS):
        raise ValueError("decisions 必须恰好包含 as/acs/mmas")
    result: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        item = dict(decisions[variant])
        mode = str(item.get("fitness_mode", "paired_combined_ucb"))
        if mode not in {
            "paired_final_ucb",
            "paired_basin_ucb",
            "paired_combined_ucb",
        }:
            raise ValueError(f"{variant} fitness_mode 非法: {mode}")
        gamma = float(item.get("gamma", 1.0 / 3.0))
        if gamma <= 0.0:
            raise ValueError(f"{variant} gamma 必须为正")
        result[variant] = {
            "fitness_mode": mode,
            "gamma": gamma,
            "origin": bool(item.get("origin", False)),
        }
    return result


def _write_config(
    *,
    variant: str,
    root_seed: int,
    decision: dict[str, Any],
    iterations: int,
    cpu_audit: bool,
) -> Path:
    payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    parameters = _variant_parameters(variant)
    identifier = f"tsp100-2opt-v2-{variant}-seed-{root_seed}"
    payload["experiment"]["experiment_id"] = identifier
    payload["experiment"]["root_seed"] = root_seed
    payload["experiment"]["cpu_fp64_final_audit"] = cpu_audit
    payload["aco"].update(
        {
            "variant": variant,
            "rho": parameters["rho"],
            "q0": parameters["q0"],
            "iterations": iterations,
            "gamma_transition": decision["gamma"],
            "gamma_pheromone": decision["gamma"],
        }
    )
    payload["gp"]["fitness_mode"] = decision["fitness_mode"]
    terminals = list(payload["gp"]["pheromone_terminals"])
    if decision["origin"] and "Origin" not in terminals:
        terminals.append("Origin")
    payload["gp"]["pheromone_terminals"] = terminals
    payload["data"]["baseline_policy"] = "require"
    target = RUN_ROOT / "configs" / f"{identifier}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _tasks(
    *,
    decisions: dict[str, dict[str, Any]],
    iterations: int,
    cpu_audit: bool,
) -> list[FormalTask]:
    tasks: list[FormalTask] = []
    for variant in VARIANTS:
        for replicate, root_seed in enumerate(GP_SEEDS):
            config = _write_config(
                variant=variant,
                root_seed=root_seed,
                decision=decisions[variant],
                iterations=iterations,
                cpu_audit=cpu_audit,
            )
            tasks.append(
                FormalTask(
                    variant=variant,
                    root_seed=root_seed,
                    replicate=replicate,
                    config=config,
                    schedule=RUN_ROOT
                    / "schedules"
                    / f"seed-{root_seed}.json",
                    baseline=RUN_ROOT
                    / "baselines"
                    / variant
                    / f"seed-{root_seed}"
                    / "baseline.npz",
                    run=RUN_ROOT
                    / "train"
                    / variant
                    / f"seed-{root_seed}",
                    log=RUN_ROOT
                    / "logs"
                    / f"{variant}-seed-{root_seed}.log",
                )
            )
    return tasks


def _run_command(
    command: list[str],
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
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _prepare_schedules(tasks: list[FormalTask]) -> None:
    """三个 ACO variants 对同一 GP seed 共享完全相同的数据 schedule。"""

    by_seed = {task.root_seed: task for task in tasks}
    for task in by_seed.values():
        if task.schedule.is_file():
            continue
        _run_command(
            [
                str(PYTHON),
                "-m",
                "rmtgp_aco",
                "prepare-schedules",
                "--config",
                str(task.config),
                "--output",
                str(task.schedule),
                "--phase",
                "formal",
                "--replicate-id",
                str(task.replicate),
                "--manifest",
                str(MANIFEST),
                "--skip-manifest-check",
            ],
            physical_gpu=None,
            log=RUN_ROOT / "logs" / f"schedule-seed-{task.root_seed}.log",
        )


def _run_completed(path: Path) -> bool:
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


def _execute_task(task: FormalTask, physical_gpu: int) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    if not task.baseline.is_file():
        _run_command(
            [
                str(PYTHON),
                "-m",
                "rmtgp_aco",
                "precompute-baselines",
                "--config",
                str(task.config),
                "--schedule",
                str(task.schedule),
                "--replicate-id",
                str(task.replicate),
                "--splits",
                "train",
                "selection",
                "gate",
                "--output",
                str(task.baseline),
                "--backend",
                "cuda_tiled_v2",
                "--gpu-devices",
                "0",
            ],
            physical_gpu=physical_gpu,
            log=task.log,
        )
    if not _run_completed(task.run):
        command = [
            str(PYTHON),
            "-m",
            "rmtgp_aco",
            "train",
            "--config",
            str(task.config),
            "--phase",
            "formal",
            "--replicate-id",
            str(task.replicate),
            "--schedule",
            str(task.schedule),
            "--baseline-archive",
            str(task.baseline.parent),
            "--manifest",
            str(MANIFEST),
            "--skip-manifest-check",
            "--method-profile",
            "rmtgp",
            "--output",
            str(task.run),
            "--backend",
            "cuda_tiled_v2",
            "--gpu-devices",
            "0",
            "--traceback",
        ]
        if (task.run / "training_state.pkl").is_file():
            command.extend(["--resume", str(task.run)])
        _run_command(
            command,
            physical_gpu=physical_gpu,
            log=task.log,
        )
    return {
        "task": task.name,
        "physical_gpu": physical_gpu,
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "run": str(task.run),
    }


def _dispatch(
    tasks: list[FormalTask],
    physical_gpus: tuple[int, ...],
) -> None:
    pending: queue.Queue[FormalTask] = queue.Queue()
    for task in tasks:
        pending.put(task)
    state_path = RUN_ROOT / "formal_state.json"
    lock = threading.Lock()
    active: dict[int, str] = {}
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def write_state(status: str) -> None:
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
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
                result = _execute_task(task, gpu)
            except Exception as error:
                with lock:
                    failures.append(
                        {
                            "task": task.name,
                            "physical_gpu": gpu,
                            "error": repr(error),
                        }
                    )
                    active.pop(gpu, None)
                    write_state("failed")
            else:
                with lock:
                    completed.append(result)
                    active.pop(gpu, None)
                    write_state("running")
            finally:
                pending.task_done()

    workers = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}")
        for gpu in physical_gpus
    ]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    write_state("failed" if failures else "completed")
    if failures:
        raise RuntimeError(
            f"{len(failures)} 个 formal task 失败；见 {state_path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpus", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument(
        "--origin-mode",
        choices=["decisions", "none", "audit-gate"],
        default="decisions",
    )
    parser.add_argument(
        "--cpu-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CPU FP64 搜索语义复核很慢；默认另行执行，不阻塞 GPU 正式训练",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只冻结 configs 和任务清单，不启动计算",
    )
    args = parser.parse_args()
    physical_gpus = tuple(dict.fromkeys(args.physical_gpus))
    if not physical_gpus or min(physical_gpus) < 0:
        parser.error("--physical-gpus 必须是非负、非空列表")
    if args.iterations < 1:
        parser.error("--iterations 必须为正整数")
    decisions = _load_decisions(args.decisions)
    if args.origin_mode == "none":
        for variant in VARIANTS:
            decisions[variant]["origin"] = False
    elif args.origin_mode == "audit-gate":
        for variant in VARIANTS:
            decisions[variant]["origin"] = _audit_origin_gate(variant)
    tasks = _tasks(
        decisions=decisions,
        iterations=args.iterations,
        cpu_audit=args.cpu_audit,
    )
    plan = {
        "schema_version": 1,
        "purpose": "confirmatory 3 variants × 3 independent GP seeds",
        "iterations": args.iterations,
        "decisions": decisions,
        "cpu_fp64_final_audit": args.cpu_audit,
        "tasks": [
            {
                "name": task.name,
                "variant": task.variant,
                "root_seed": task.root_seed,
                "replicate": task.replicate,
                "config": str(task.config),
                "schedule": str(task.schedule),
                "baseline": str(task.baseline),
                "run": str(task.run),
            }
            for task in tasks
        ],
    }
    _atomic_json(RUN_ROOT / "formal_plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False), flush=True)
        return 0
    _prepare_schedules(tasks)
    _dispatch(tasks, physical_gpus)
    print(
        json.dumps(
            {
                "status": "completed",
                "tasks": len(tasks),
                "run_root": str(RUN_ROOT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
