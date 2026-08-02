#!/usr/bin/env python3
"""调度 TSP500 LS-aware v2 的 3 variants × 3 GP seeds 正式训练。"""

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

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
TEMPLATE = ROOT / "experiments" / "tsp500_2opt_ls_v2" / "config.yaml"
RUN_ROOT = ROOT / "runs" / "tsp500-2opt-ls-v2"
MANIFEST = ROOT / "Datasets" / "manifest.json"
SOURCE_RUN = (
    ROOT
    / "runs"
    / "tsp100-ablation-gpu1-3seed.nfs-checkpoint-20260727T1740NZST"
    / "train"
    / "tr-rgp"
)
VARIANTS = ("as", "acs", "mmas")
GP_SEEDS = (81001, 81002, 81003)
TRANSITION_SEEDS = {
    "as": (1001, 1002, 1003),
    "acs": (2001, 2002, 2003),
    "mmas": (3001, 3002, 3003),
}
PARAMETERS = {
    "as": {"rho": 0.5, "q0": 0.0},
    "acs": {"rho": 0.1, "q0": 0.98},
    "mmas": {"rho": 0.2, "q0": 0.0},
}


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    command: tuple[str, ...]
    log: Path
    result: Path


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


def _last_completed_generation(path: Path) -> int:
    """读取增量指标，避免为调度状态反序列化完整 GP checkpoint。"""

    source = path / "training_metrics.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(payload, list) or not payload:
        return 0
    try:
        return int(payload[-1]["generation"])
    except (KeyError, TypeError, ValueError):
        return 0


def _manifest_error(path: Path) -> str:
    try:
        payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(payload.get("error") or "")


def _resume_command(task: Task) -> tuple[str, ...]:
    command = list(task.command)
    if (task.result / "training_state.pkl").is_file() and "--resume" not in command:
        command.extend(["--resume", str(task.result)])
    return tuple(command)


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
            + (f"CUDA_VISIBLE_DEVICES={physical_gpu} " if physical_gpu is not None else "")
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


def _transition_checkpoint(variant: str, replicate: int) -> Path:
    seed = TRANSITION_SEEDS[variant][replicate]
    source = SOURCE_RUN / variant / f"seed-{seed}" / "selected_candidate.pkl"
    if not source.is_file():
        raise FileNotFoundError(f"fixed transition checkpoint 不存在: {source}")
    return source


def _write_configs() -> dict[tuple[str, int], Path]:
    configs: dict[tuple[str, int], Path] = {}
    for variant in VARIANTS:
        for replicate, root_seed in enumerate(GP_SEEDS):
            payload = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
            identifier = f"tsp500-2opt-ls-v2-{variant}-seed-{root_seed}"
            payload["experiment"]["experiment_id"] = identifier
            payload["experiment"]["root_seed"] = root_seed
            payload["aco"].update(
                {
                    "variant": variant,
                    "rho": PARAMETERS[variant]["rho"],
                    "q0": PARAMETERS[variant]["q0"],
                    "gamma_transition": 1.0 / 3.0,
                    "gamma_pheromone": 1.0 / 3.0,
                }
            )
            payload["gp"]["fixed_transition_checkpoint"] = str(
                _transition_checkpoint(variant, replicate).relative_to(ROOT)
            )
            target = RUN_ROOT / "formal" / "configs" / f"{identifier}.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.yaml")
            temporary.write_text(
                yaml.safe_dump(
                    payload,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            temporary.replace(target)
            configs[(variant, root_seed)] = target
    return configs


def _prepare_schedules(configs: dict[tuple[str, int], Path]) -> None:
    for replicate, root_seed in enumerate(GP_SEEDS):
        target = RUN_ROOT / "formal" / "schedules" / f"seed-{root_seed}.json"
        if target.is_file():
            continue
        _run_command(
            [
                str(PYTHON),
                "-m",
                "rmtgp_aco",
                "prepare-schedules",
                "--config",
                str(configs[("as", root_seed)]),
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
            log=RUN_ROOT / "logs" / f"schedule-seed-{root_seed}.log",
        )


def _tasks(configs: dict[tuple[str, int], Path]) -> list[Task]:
    result: list[Task] = []
    for variant in VARIANTS:
        for replicate, root_seed in enumerate(GP_SEEDS):
            run = RUN_ROOT / "formal" / "train" / variant / f"seed-{root_seed}"
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
                str(RUN_ROOT / "formal" / "schedules" / f"seed-{root_seed}.json"),
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
            result.append(
                Task(
                    name=f"{variant}-seed-{root_seed}",
                    command=tuple(command),
                    log=RUN_ROOT / "logs" / f"{variant}-seed-{root_seed}.log",
                    result=run,
                )
            )
    return result


def _dispatch(
    tasks: list[Task],
    gpus: tuple[int, ...],
    *,
    max_retries: int,
) -> None:
    pending: queue.Queue[Task] = queue.Queue()
    for task in tasks:
        if not _completed(task.result):
            pending.put(task)
    lock = threading.Lock()
    active: dict[int, str] = {}
    completed: list[str] = []
    failures: list[dict[str, str]] = []
    attempts: dict[str, int] = {}
    failure_history: list[dict[str, object]] = []
    state_path = RUN_ROOT / "campaign_state.json"

    def write_state(status: str) -> None:
        _atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": status,
                "active": dict(active),
                "completed": list(completed),
                "failures": list(failures),
                "attempts": dict(attempts),
                "failure_history": list(failure_history),
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
                succeeded = False
                final_error = ""
                for attempt in range(max_retries + 1):
                    before_generation = _last_completed_generation(task.result)
                    with lock:
                        attempts[task.name] = attempt + 1
                        write_state("running")
                    try:
                        _run_command(
                            _resume_command(task),
                            physical_gpu=gpu,
                            log=task.log,
                        )
                        succeeded = True
                        break
                    except Exception as exc:
                        after_generation = _last_completed_generation(task.result)
                        manifest_error = _manifest_error(task.result)
                        final_error = repr(exc)
                        transient_memory = any(
                            marker in manifest_error
                            for marker in (
                                "MemoryError",
                                "OutOfMemoryError",
                                "CUDA_ERROR_OUT_OF_MEMORY",
                            )
                        )
                        with lock:
                            failure_history.append(
                                {
                                    "task": task.name,
                                    "attempt": attempt + 1,
                                    "before_generation": before_generation,
                                    "after_generation": after_generation,
                                    "manifest_error": manifest_error,
                                    "exception": final_error,
                                }
                            )
                            write_state("running")
                        can_retry = (
                            attempt < max_retries
                            and (task.result / "training_state.pkl").is_file()
                            and (
                                after_generation > before_generation
                                or transient_memory
                            )
                        )
                        if not can_retry:
                            break
                with lock:
                    if succeeded:
                        completed.append(task.name)
                    else:
                        failures.append(
                            {
                                "task": task.name,
                                "error": final_error,
                                "manifest_error": _manifest_error(task.result),
                            }
                        )
            finally:
                with lock:
                    active.pop(gpu, None)
                    write_state("running")
                pending.task_done()

    write_state("running")
    workers = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    write_state("failed" if failures else "completed")
    if failures:
        raise RuntimeError(f"训练任务失败: {failures}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="每个有 checkpoint 的任务最多额外重试次数",
    )
    args = parser.parse_args()
    gpus = tuple(dict.fromkeys(int(gpu) for gpu in args.gpus))
    if not gpus or min(gpus) < 0:
        parser.error("--gpus 必须是非空非负整数列表")
    if args.max_retries < 0:
        parser.error("--max-retries 不得为负")

    configs = _write_configs()
    _prepare_schedules(configs)
    tasks = _tasks(configs)
    if args.prepare_only:
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "gpus": gpus,
                    "tasks": [{"name": task.name, "command": task.command} for task in tasks],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    _dispatch(tasks, gpus, max_retries=args.max_retries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
