"""在显式指定且空闲的单卡上执行独立诊断；复用项目 UUID 锁。"""

import argparse
import fcntl
import os
import traceback

from run_worker import gpu_state

from gpaco.artifact_registry import require_output
from gpaco.benchmark_inputs import BenchmarkInputs
from gpaco.data import ROOT, write_json
from gpaco.diagnostics import run
from gpaco.hardware_inputs import safe_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, action="append", help="可重复；同卡串行诊断")
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--smoke-iterations", type=int)
    args = parser.parse_args()
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid:
        parser.error("必须显式指定单个已确认空闲的 GPU UUID")
    directory = safe_directory(args.output)
    require_output(
        directory,
        "E00-p01-instrumentation-checks" if args.smoke_iterations else "E01-p01-work-diagnostics",
    )
    if directory.exists():
        parser.error("不得覆盖已有诊断记录")
    tasks = []
    for bundle in args.bundle:
        inputs = BenchmarkInputs(bundle)
        programs, problem = inputs.load(args.generation, args.block)
        name = f"tsp{problem.n}-{inputs.search.variant}-g{args.generation}-b{args.block}"
        tasks.append((bundle, name))
    if len({name for _, name in tasks}) != len(tasks):
        parser.error("同一批次不能重复诊断相同规模和宿主")
    batch = len(tasks) > 1
    locks = ROOT / "artifacts/locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{uuid}.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, pids = gpu_state(uuid)
        if pids or int(state.split(",")[2]) > 5 or int(state.split(",")[3]) > 1024:
            raise RuntimeError(f"设备启动前不空闲：{state}; {pids}")
        if batch:
            directory.mkdir(parents=True)
            write_json(directory / "tasks.json", {"gpu_uuid": uuid, "tasks": tasks})
        for bundle, name in tasks:
            # 本项目锁覆盖整个批次；其他用户仍可启动，检测到后不继续下一项。
            _, pids = gpu_state(uuid)
            outsiders = [pid for pid in pids if pid != os.getpid()]
            if outsiders:
                write_json(directory / "BLOCKED.json", {"other_pids": outsiders, "next": name})
                raise RuntimeError(f"检测到其他 GPU 进程，停止后续诊断：{outsiders}")
            target = directory / name if batch else directory
            print(f"开始独立诊断：{name}", flush=True)
            try:
                run(bundle, target, args.generation, args.block, args.smoke_iterations)
            except Exception:
                write_json(target / "FAILED.json", {"traceback": traceback.format_exc()})
                raise
            print(f"完成独立诊断：{name}", flush=True)
        if batch:
            write_json(directory / "COMPLETE.json", {"status": "completed", "tasks": tasks})


if __name__ == "__main__":
    main()
