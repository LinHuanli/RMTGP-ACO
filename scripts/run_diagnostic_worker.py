"""单卡完整预算诊断worker；固定源码、共享UUID锁、原始失败与争用记录。"""

import argparse
import fcntl
import json
import os
import traceback

from run_main_benchmark import DeviceUnavailable, qualify
from run_worker import gpu_state

from gpaco.artifact_registry import resolve
from gpaco.benchmark_inputs import BenchmarkInputs
from gpaco.data import write_json
from gpaco.diagnostics import run
from gpaco.hardware_inputs import file_hash, safe_directory


def execute(job, directory):
    uuid, task = job["gpu_uuid"], job["task"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("必须仅暴露分配给本worker的GPU UUID")
    locks = resolve("shared-device-locks")
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{uuid}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DeviceUnavailable("目标GPU仍被本项目其他worker持有") from error
        state, pids = gpu_state(uuid)
        columns = [v.strip() for v in state.split(",")]
        if pids or int(columns[2]) > 5 or int(columns[3]) > 1024:
            raise DeviceUnavailable(f"启动复查不空闲：{state}; {pids}")
        if columns[1] != job["config"]["gpu_model"]:
            raise DeviceUnavailable("本队列只接受RTX A5000")
        inputs = BenchmarkInputs(job["bundle"])
        programs, problem = inputs.load(task["generation"], task["block"])
        budget = job["config"]["scientific_budget"]
        if (
            len(programs) != budget["population"]
            or problem.size != budget["batch"]
            or problem.n != task["n"]
            or inputs.search.variant != task["variant"]
            or inputs.search.ants != budget["ants"]
            or inputs.search.iterations != budget["iterations"]
            or inputs.search.candidate_size != budget["candidate_size"]
            or inputs.identity != job["bundle_sha256"]
        ):
            raise ValueError("冻结输入与预先登记的完整预算不一致")
        qualify(safe_directory(job["campaign_directory"]), job, directory)
        run(job["bundle"], directory / "measurement", task["generation"], task["block"])
        record = json.loads((directory / "measurement/record.json").read_text())
        pair = json.loads((directory / "measurement/instrumentation_pair.json").read_text())
        clean = not any(
            pair[mode].get("contended") or pair[mode].get("telemetry_errors")
            for mode in ("plain", "instrumented")
        )
        return {
            "status": "completed",
            "clean": clean,
            "task_id": task["id"],
            "record_sha256": file_hash(directory / "measurement/record.json"),
            "snapshots": record["state_replay"]["snapshots"],
            "bitwise_instrumentation_checked": pair["bitwise_equal_tours"]
            and pair["bitwise_equal_lengths"],
            "formal_result": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    path = safe_directory(args.job)
    job = json.loads(path.read_text())
    try:
        result = execute(job, path.parent)
    except DeviceUnavailable as error:
        write_json(
            path.parent / "REJECTED.json", {"reason": str(error), "scientific_work_started": False}
        )
        print(f"未开始科学计算，退回队列：{error}", flush=True)
    except Exception:
        write_json(path.parent / "FAILED.json", {"traceback": traceback.format_exc()})
        raise
    else:
        write_json(path.parent / "COMPLETE.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
