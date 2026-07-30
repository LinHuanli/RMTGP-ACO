#!/usr/bin/env python3
"""可靠地串联旧实验恢复、TSP500 racing、匹配对照与最终测试。

该脚本本身不占用 GPU。它可以先等待一个已存在的正式实验主进程，然后按
依赖关系启动后续任务。每个阶段都写入原子状态文件；意外中断后再次运行
同一命令即可继续，因为下层训练与测试脚本均按 manifest/shard 恢复。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUN_ROOT = ROOT / "runs" / "tsp500-2opt-racing"
STATE_PATH = RUN_ROOT / "master_queue_state.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": 1,
            "status": "initializing",
            "launcher_pid": os.getpid(),
            "created_at": _now(),
            "stages": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {
            "schema_version": 1,
            "status": "recovering_invalid_state",
            "launcher_pid": os.getpid(),
            "created_at": _now(),
            "stages": {},
        }
    payload["launcher_pid"] = os.getpid()
    return payload


def _write_state(
    state: dict[str, Any],
    *,
    status: str,
    path: Path,
) -> None:
    state["status"] = status
    state["launcher_pid"] = os.getpid()
    state["updated_at"] = _now()
    _atomic_json(path, state)


def _process_matches(pid: int, command_fragment: str) -> bool:
    """仅等待原目标进程，避免 PID 被系统复用后无限等待。"""

    process = Path("/proc") / str(pid)
    try:
        fields = (process / "stat").read_text(encoding="utf-8").split()
        if len(fields) >= 3 and fields[2] == "Z":
            return False
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return command_fragment.encode() in command


def _wait_for_process(
    *,
    pid: int | None,
    command_fragment: str,
    poll_seconds: float,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    if pid is None or not _process_matches(pid, command_fragment):
        return
    state["wait_target"] = {
        "pid": pid,
        "command_fragment": command_fragment,
        "started_at": _now(),
    }
    _write_state(state, status="waiting_for_existing_run", path=state_path)
    print(
        f"[{_now()}] 等待现有进程 pid={pid}: {command_fragment}",
        flush=True,
    )
    last_report = time.monotonic()
    while _process_matches(pid, command_fragment):
        time.sleep(poll_seconds)
        if time.monotonic() - last_report >= 600.0:
            state["wait_target"]["last_seen_at"] = _now()
            _write_state(
                state,
                status="waiting_for_existing_run",
                path=state_path,
            )
            print(f"[{_now()}] 仍在等待 pid={pid}", flush=True)
            last_report = time.monotonic()
    state["wait_target"]["finished_at"] = _now()
    print(f"[{_now()}] pid={pid} 已结束，开始后续队列", flush=True)


def _run_stage(
    *,
    name: str,
    command: list[str],
    state: dict[str, Any],
    state_path: Path,
) -> int:
    stages = state.setdefault("stages", {})
    previous = stages.get(name, {})
    if previous.get("status") == "completed":
        print(f"[{_now()}] 跳过已完成阶段: {name}", flush=True)
        return 0
    stages[name] = {
        "status": "running",
        "command": command,
        "started_at": _now(),
    }
    _write_state(state, status=f"running:{name}", path=state_path)
    print(f"[{_now()}] 启动阶段 {name}: {' '.join(command)}", flush=True)
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
        return_code = int(result.returncode)
    except Exception as error:
        stages[name].update(
            {
                "status": "failed",
                "finished_at": _now(),
                "error": repr(error),
            }
        )
        _write_state(state, status=f"failed:{name}", path=state_path)
        print(f"[{_now()}] 阶段 {name} 异常: {error!r}", flush=True)
        return 1
    stages[name].update(
        {
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "finished_at": _now(),
        }
    )
    _write_state(
        state,
        status=(f"completed:{name}" if return_code == 0 else f"failed:{name}"),
        path=state_path,
    )
    print(
        f"[{_now()}] 阶段 {name} 结束，return_code={return_code}",
        flush=True,
    )
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument(
        "--wait-command-fragment",
        default="run_2opt_signal_formal.py",
    )
    parser.add_argument("--wait-poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--skip-legacy-recovery",
        action="store_true",
        help="不重跑旧 TSP100 final-only 正式矩阵",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="训练完成后不启动 5000 轮最终测试",
    )
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    args = parser.parse_args()
    gpus = tuple(dict.fromkeys(int(gpu) for gpu in args.gpus))
    if not gpus or min(gpus) < 0:
        parser.error("--gpus 必须是非空非负整数列表")
    if args.wait_pid is not None and args.wait_pid < 1:
        parser.error("--wait-pid 必须为正整数")
    if args.wait_poll_seconds <= 0.0:
        parser.error("--wait-poll-seconds 必须为正数")

    state = _load_state(args.state)
    state["gpus"] = list(gpus)
    _write_state(state, status="starting", path=args.state)
    _wait_for_process(
        pid=args.wait_pid,
        command_fragment=args.wait_command_fragment,
        poll_seconds=args.wait_poll_seconds,
        state=state,
        state_path=args.state,
    )

    failures: list[str] = []
    gpu_arguments = [str(gpu) for gpu in gpus]
    if not args.skip_legacy_recovery:
        legacy = _run_stage(
            name="recover_tsp100_final_only",
            command=[
                str(PYTHON),
                "scripts/run_2opt_signal_formal.py",
                "--physical-gpus",
                *gpu_arguments,
                "--iterations",
                "500",
                "--decisions",
                "experiments/tsp100_2opt_signal_v2/formal_decisions.json",
                "--origin-mode",
                "decisions",
            ],
            state=state,
            state_path=args.state,
        )
        if legacy != 0:
            failures.append("recover_tsp100_final_only")

    tsp500 = _run_stage(
        name="tsp500_racing",
        command=[
            str(PYTHON),
            "scripts/run_tsp500_racing_campaign.py",
            "--phase",
            "all",
            "--gpus",
            *gpu_arguments,
        ],
        state=state,
        state_path=args.state,
    )
    if tsp500 != 0:
        failures.append("tsp500_racing")

    tsp100 = _run_stage(
        name="tsp100_anytime_control",
        command=[
            str(PYTHON),
            "scripts/run_tsp100_anytime_control.py",
            "--physical-gpus",
            *gpu_arguments,
            "--iterations",
            "500",
            "--origin-mode",
            "decisions",
        ],
        state=state,
        state_path=args.state,
    )
    if tsp100 != 0:
        failures.append("tsp100_anytime_control")

    if not args.skip_final_test:
        if tsp500 == 0 and tsp100 == 0:
            final_test = _run_stage(
                name="final_test_5000",
                command=[
                    str(PYTHON),
                    "scripts/evaluate_tsp500_racing_final.py",
                    "--iterations",
                    "5000",
                    "--test-seeds",
                    "3",
                ],
                state=state,
                state_path=args.state,
            )
            if final_test != 0:
                failures.append("final_test_5000")
        else:
            state.setdefault("stages", {})["final_test_5000"] = {
                "status": "blocked",
                "reason": ("需要 tsp500_racing 与 tsp100_anytime_control 均成功"),
                "updated_at": _now(),
            }

    state["failures"] = failures
    final_status = "completed" if not failures else "completed_with_failures"
    _write_state(state, status=final_status, path=args.state)
    print(
        f"[{_now()}] 主队列结束: {final_status}; failures={failures}",
        flush=True,
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
