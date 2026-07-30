#!/usr/bin/env python3
"""在多张物理 GPU 上可恢复地调度 2-opt 信号审计与三代 pilots。"""

from __future__ import annotations

import argparse
import csv
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
CONFIG = ROOT / "experiments" / "tsp100_2opt_signal_v2" / "config.yaml"
RUN_ROOT = ROOT / "runs" / "tsp100-2opt-signal-v2"
VARIANTS = ("as", "acs", "mmas")
FITNESS_MODES = (
    "paired_final_ucb",
    "paired_basin_ucb",
    "paired_combined_ucb",
)
GAMMA_VALUES = (1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0)


@dataclass(frozen=True, slots=True)
class Task:
    """一个只占用一张可见 GPU 的外部进程任务。"""

    name: str
    command: tuple[str, ...]
    log: Path
    result: Path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(payload, dict)
    except (OSError, ValueError):
        return False


def _audit_completed(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return (
            json.loads(manifest.read_text(encoding="utf-8")).get("status")
            == "completed"
            and (path / "summary.json").is_file()
        )
    except (OSError, ValueError):
        return False


def _run_task(task: Task, physical_gpu: int) -> dict[str, Any]:
    if (
        _audit_completed(task.result)
        if task.result.is_dir()
        else _completed_json(task.result)
    ):
        return {
            "task": task.name,
            "physical_gpu": physical_gpu,
            "status": "reused",
            "completed_at": datetime.now(UTC).isoformat(),
        }
    task.log.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
        "PYTHONUNBUFFERED": "1",
    }
    with task.log.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{datetime.now(UTC).isoformat()}] "
            f"CUDA_VISIBLE_DEVICES={physical_gpu} {' '.join(task.command)}\n"
        )
        stream.flush()
        subprocess.run(
            task.command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return {
        "task": task.name,
        "physical_gpu": physical_gpu,
        "status": "completed",
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _dispatch(
    tasks: list[Task],
    physical_gpus: tuple[int, ...],
    *,
    state_path: Path,
    phase: str,
) -> list[dict[str, Any]]:
    """每张 GPU 一个 worker，任务结束后立即领取下一个任务。"""

    pending: queue.Queue[Task] = queue.Queue()
    for task in tasks:
        pending.put(task)
    lock = threading.Lock()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    active: dict[int, str] = {}

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

    def worker(physical_gpu: int) -> None:
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                active[physical_gpu] = task.name
                write_state("running")
            try:
                result = _run_task(task, physical_gpu)
            except Exception as error:
                with lock:
                    failures.append(
                        {
                            "task": task.name,
                            "physical_gpu": physical_gpu,
                            "error": repr(error),
                        }
                    )
                    active.pop(physical_gpu, None)
                    write_state("failed")
                pending.task_done()
                continue
            with lock:
                completed.append(result)
                active.pop(physical_gpu, None)
                write_state("running")
            pending.task_done()

    threads = [
        threading.Thread(
            target=worker,
            args=(gpu,),
            name=f"gpu-{gpu}",
            daemon=False,
        )
        for gpu in physical_gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write_state("failed" if failures else "completed")
    if failures:
        raise RuntimeError(
            f"{phase} 有 {len(failures)} 个任务失败；见 {state_path}"
        )
    return completed


def _audit_tasks(
    *,
    horizons: tuple[int, ...],
    seeds: int,
) -> list[Task]:
    tasks: list[Task] = []
    for variant in VARIANTS:
        output = RUN_ROOT / "audit" / variant
        command = (
            str(PYTHON),
            str(ROOT / "scripts" / "audit_2opt_signal.py"),
            "--config",
            str(CONFIG),
            "--variant",
            variant,
            "--output",
            str(output),
            "--horizons",
            *(str(value) for value in horizons),
            "--seeds",
            str(seeds),
        )
        tasks.append(
            Task(
                name=f"audit-{variant}",
                command=command,
                log=RUN_ROOT / "logs" / f"audit-{variant}.log",
                result=output,
            )
        )
    return tasks


def _variant_parameters(variant: str) -> dict[str, float]:
    return {
        "as": {"rho": 0.5, "q0": 0.0},
        "acs": {"rho": 0.1, "q0": 0.98},
        "mmas": {"rho": 0.2, "q0": 0.0},
    }[variant]


def _slug(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _write_pilot_config(
    *,
    variant: str,
    fitness_mode: str,
    gamma: float,
    origin: bool,
    iterations: int,
    root_seed: int,
    task_name: str,
) -> Path:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parameters = _variant_parameters(variant)
    payload["experiment"]["experiment_id"] = task_name
    payload["experiment"]["root_seed"] = root_seed
    payload["experiment"]["cpu_fp64_final_audit"] = False
    payload["aco"].update(
        {
            "variant": variant,
            "rho": parameters["rho"],
            "q0": parameters["q0"],
            "iterations": iterations,
            "gamma_transition": gamma,
            "gamma_pheromone": gamma,
        }
    )
    payload["gp"]["fitness_mode"] = fitness_mode
    terminals = list(payload["gp"]["pheromone_terminals"])
    if origin and "Origin" not in terminals:
        terminals.append("Origin")
    payload["gp"]["pheromone_terminals"] = terminals
    payload["data"]["baseline_policy"] = "compute"
    target = RUN_ROOT / "pilot" / "configs" / f"{task_name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _origin_gate(variant: str) -> bool:
    summary_path = RUN_ROOT / "audit" / variant / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Origin pilot 需要先完成学习信号审计: {summary_path}"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    horizons = payload["horizon_summaries"]
    if not horizons:
        raise ValueError(f"audit summary 没有 horizon: {summary_path}")
    return bool(horizons[-1]["origin_gate"]["enabled"])


def _pilot_task(
    *,
    variant: str,
    family: str,
    fitness_mode: str,
    gamma: float,
    origin: bool,
    iterations: int,
) -> Task:
    variant_index = VARIANTS.index(variant)
    root_seed = 61001 + 1000 * variant_index
    task_name = (
        f"{variant}-{family}-{fitness_mode.removeprefix('paired_')}"
        f"-g{_slug(gamma)}"
        f"{'-origin' if origin else ''}"
    )
    config = _write_pilot_config(
        variant=variant,
        fitness_mode=fitness_mode,
        gamma=gamma,
        origin=origin,
        iterations=iterations,
        root_seed=root_seed,
        task_name=task_name,
    )
    result = RUN_ROOT / "pilot" / "results" / f"{task_name}.json"
    command = (
        str(PYTHON),
        "-m",
        "rmtgp_aco",
        "benchmark-training",
        "--config",
        str(config),
        "--generations",
        "3",
        "--phase",
        "development",
        "--baseline-policy",
        "compute",
        "--method-profile",
        "rmtgp",
        "--skip-manifest-check",
        "--output",
        str(result),
    )
    return Task(
        name=task_name,
        command=command,
        log=RUN_ROOT / "logs" / f"{task_name}.log",
        result=result,
    )


def _pilot_tasks(iterations: int) -> list[Task]:
    tasks: list[Task] = []
    for variant in VARIANTS:
        for mode in FITNESS_MODES:
            tasks.append(
                _pilot_task(
                    variant=variant,
                    family="fitness",
                    fitness_mode=mode,
                    gamma=1.0 / 3.0,
                    origin=False,
                    iterations=iterations,
                )
            )
        # gamma=1/3 已由 combined fitness task 完全覆盖，不重复计算。
        for gamma in (GAMMA_VALUES[0], GAMMA_VALUES[2]):
            tasks.append(
                _pilot_task(
                    variant=variant,
                    family="radius",
                    fitness_mode="paired_combined_ucb",
                    gamma=gamma,
                    origin=False,
                    iterations=iterations,
                )
            )
        if _origin_gate(variant):
            tasks.append(
                _pilot_task(
                    variant=variant,
                    family="terminal",
                    fitness_mode="paired_combined_ucb",
                    gamma=1.0 / 3.0,
                    origin=True,
                    iterations=iterations,
                )
            )
    return tasks


def _pilot_summary(tasks: list[Task]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        payload = json.loads(task.result.read_text(encoding="utf-8"))
        record = payload["records"][-1]
        task_config = yaml.safe_load(
            Path(payload["config"]).read_text(encoding="utf-8")
        )
        scale = "100"
        rows.append(
            {
                "task": task.name,
                "variant": payload["variant"],
                "iterations": int(task_config["aco"]["iterations"]),
                "generation": record["generation"],
                "fitness_ucb_pp": record["fitness_min_gap_percent"],
                "final_delta_pp": record["best_delta_pp_by_scale"].get(scale),
                "basin_delta_pp": record[
                    "best_basin_delta_pp_by_scale"
                ].get(scale),
                "fitness_delta_pp": record[
                    "fitness_delta_pp_by_scale"
                ].get(scale),
                "evaluation_seconds": record["evaluation_seconds"],
                "tours_per_second": record["tours_per_second"],
                "total_seconds": payload["total_seconds"],
            }
        )
    target = RUN_ROOT / "pilot" / "pilot_summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    payload = {
        "schema_version": 1,
        "purpose": "three-generation mechanism screen; not confirmatory evidence",
        "rows": rows,
    }
    _atomic_json(RUN_ROOT / "pilot" / "pilot_summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["audit", "pilots", "all"],
        default="audit",
    )
    parser.add_argument("--physical-gpus", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--audit-horizons",
        type=int,
        nargs="+",
        default=(500, 2000, 5000),
    )
    parser.add_argument("--audit-seeds", type=int, default=5)
    parser.add_argument("--pilot-iterations", type=int, default=500)
    args = parser.parse_args()
    physical_gpus = tuple(dict.fromkeys(args.physical_gpus))
    if not physical_gpus or min(physical_gpus) < 0:
        parser.error("--physical-gpus 必须是非负、非空设备列表")
    if args.audit_seeds < 2:
        parser.error("--audit-seeds 至少为 2")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase in {"audit", "all"}:
        tasks = _audit_tasks(
            horizons=tuple(args.audit_horizons),
            seeds=args.audit_seeds,
        )
        _dispatch(
            tasks,
            physical_gpus,
            state_path=RUN_ROOT / "audit_state.json",
            phase="audit",
        )
    if args.phase in {"pilots", "all"}:
        tasks = _pilot_tasks(args.pilot_iterations)
        _dispatch(
            tasks,
            physical_gpus,
            state_path=RUN_ROOT / "pilot_state.json",
            phase="pilots",
        )
        _pilot_summary(tasks)
    print(
        json.dumps(
            {
                "status": "completed",
                "phase": args.phase,
                "run_root": str(RUN_ROOT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
