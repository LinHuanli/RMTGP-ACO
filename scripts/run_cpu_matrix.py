"""六组 CPU 的串行配对 block；默认仅列出计划，--execute 才实际计算。

不处理 NeSI/Slurm。单个 block 在同一主机执行，固定 shuffle 顺序；超时保留为删失记录。
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from time import perf_counter

import numpy as np

from gpaco.artifact_registry import require_output
from gpaco.cpu_benchmark import physical_cpu_ids
from gpaco.data import ROOT, write_json
from gpaco.experiment import namespace_seed
from gpaco.hardware_inputs import safe_directory


def cells(generation, blocks, backends, cores):
    result = []
    for block in range(blocks):
        combinations = [(backend, core) for backend in backends for core in cores]
        rng = np.random.default_rng(namespace_seed(9401, "cpu-order", generation, block))
        for i in rng.permutation(len(combinations)):
            backend, core = combinations[i]
            result.append(
                {"generation": generation, "block": block, "backend": backend, "cores": core}
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--cores", nargs="+", type=int, choices=[1, 8, 16], default=[1, 8, 16])
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["cpu_python", "cpu_existing"],
        default=["cpu_python", "cpu_existing"],
    )
    parser.add_argument("--wall-limit-per-cell", type=int, default=86400)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    directory = safe_directory(args.output)
    require_output(directory, "E01-p01-cpu-baselines")
    plan = cells(args.generation, args.blocks, args.backends, args.cores)
    if not 1 <= args.blocks <= 5 or args.wall_limit_per_cell < 1:
        parser.error("现有冻结输入含5个block；时间限制必须为正")
    print(
        json.dumps(
            {"cells": plan, "available_physical_cpus": physical_cpu_ids(), "execute": args.execute}
        )
    )
    if not args.execute:
        return
    if max(args.cores) > len(physical_cpu_ids()):
        parser.error("物理核数不足，不开始部分配置或用SMT替代")
    bundle = safe_directory(args.bundle)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "matrix.json",
        {
            "cells": plan,
            "bundle": str(bundle),
            "timeout_s": args.wall_limit_per_cell,
            "cache_state": "warm",
            "formal_exclusive_cpu_allocation": False,
        },
    )
    locks = ROOT / "artifacts/locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"cpu-{os.uname().nodename}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        statuses = []
        for cell in plan:
            name = (
                f"g{cell['generation']:03d}-b{cell['block']:02d}-{cell['backend']}-c{cell['cores']}"
            )
            output = directory / name
            command = [
                sys.executable,
                str(ROOT / "scripts/benchmark_cpu.py"),
                "run",
                "--bundle",
                str(bundle),
                "--output",
                str(output),
                "--generation",
                str(cell["generation"]),
                "--block",
                str(cell["block"]),
                "--backend",
                cell["backend"],
                "--cores",
                str(cell["cores"]),
            ]
            begin = perf_counter()
            with (directory / f"{name}.log").open("w") as log:
                child = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    code = child.wait(timeout=args.wall_limit_per_cell)
                    status = "completed" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    # 仅终止此脚本创建的独立进程组，包括其 Python 工作进程。
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    status = "timeout"
                    write_json(
                        directory / f"{name}-timeout.json",
                        {
                            "status": status,
                            "elapsed_s": perf_counter() - begin,
                            "evaluation_time_s": None,
                            "note": "墙钟上限含启动/预热，不能冒充 evaluation 的下界或完成时间",
                        },
                    )
            statuses.append({**cell, "status": status, "elapsed_s": perf_counter() - begin})
            write_json(directory / "status.json", statuses)


if __name__ == "__main__":
    main()
