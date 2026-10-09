"""CPU 完整一代基线；核数、首次调用、预热、实际工作量和结果分别登记。"""

import os
import platform
import resource
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifact_registry import identity, require_output
from .benchmark_inputs import BenchmarkInputs
from .config import ExecutionPlan
from .data import validate_tours, write_json
from .experiment import evaluate, gap, metadata
from .hardware_inputs import file_hash, safe_directory


def physical_cpu_ids(allowed=None):
    """每个物理核只选一个允许的逻辑 CPU；优先在同一 socket 取核。"""
    allowed = os.sched_getaffinity(0) if allowed is None else allowed
    cores = {}
    for cpu in sorted(allowed):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (
            int((topology / "physical_package_id").read_text()),
            int((topology / "core_id").read_text()),
        )
        cores.setdefault(key, cpu)
    return [cores[key] for key in sorted(cores)]


class CpuMemory:
    """采样整个子进程树 RSS；共享页会重复计数，因此另报 PSS（系统支持时）。"""

    def __init__(self):
        self.stop = threading.Event()
        self.peak_rss = self.peak_pss = 0
        self.pss_available = False
        self.samples = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        pids, pending = set(), [os.getpid()]
        while pending:
            pid = pending.pop()
            if pid in pids:
                continue
            pids.add(pid)
            try:
                pending.extend(
                    map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
                )
            except (OSError, ValueError):
                pass
        rss = pss = 0
        available = True
        for pid in pids:
            try:
                rss += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf(
                    "SC_PAGE_SIZE"
                )
                data = Path(f"/proc/{pid}/smaps_rollup").read_text()
                pss += (
                    int(
                        next(line for line in data.splitlines() if line.startswith("Pss:")).split()[
                            1
                        ]
                    )
                    * 1024
                )
            except (OSError, ValueError, StopIteration):
                available = False
        self.peak_rss = max(self.peak_rss, rss)
        if available:
            self.pss_available = True
            self.peak_pss = max(self.peak_pss, pss)
        self.samples += 1

    def run(self):
        while not self.stop.wait(1):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()


def run(directory, bundle, generation, block, backend, cores, cache_state="warm"):
    directory = safe_directory(directory)
    require_output(directory, "E01-p01-cpu-baselines")
    cpus = physical_cpu_ids()
    if cores not in (1, 8, 16) or cores > len(cpus):
        raise ValueError(f"请求 {cores} 个物理核，当前仅 {len(cpus)} 个可用；不能用 SMT 冒充")
    if backend not in ("cpu_python", "cpu_existing") or cache_state not in ("warm", "cold"):
        raise ValueError("无效后端或缓存模式")
    directory.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus[:cores])
    try:
        start = perf_counter()
        inputs = BenchmarkInputs(bundle)
        programs, problem = inputs.load(generation, block)
        setup_s = perf_counter() - start
        workload = {
            **inputs.workload(generation, block, programs, problem),
            **identity("E01-p01-cpu-baselines"),
            "run_id": directory.name,
        }
        plan = ExecutionPlan(backend=backend, cpu_threads=cores)
        write_json(
            directory / "request.json",
            {**workload, "backend": backend, "cores": cores, "cache_state": cache_state},
        )
        write_json(
            directory / "hardware.json",
            {
                **metadata(),
                "platform": platform.platform(),
                "lscpu": subprocess.check_output(["lscpu"], text=True),
                "selected_physical_cpus": cpus[:cores],
                "load_average": os.getloadavg(),
                "exclusive_allocation_verified": False,
            },
        )
        warmup_s = None
        if cache_state == "warm":
            # 同签名的小预算只用于编译/运行时预热，不冒充完整科学测量。
            start = perf_counter()
            evaluate(
                programs[:1], problem, replace(inputs.search, iterations=1), workload["seed"], plan
            )
            warmup_s = perf_counter() - start
        before = resource.getrusage(resource.RUSAGE_SELF)
        children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        with CpuMemory() as memory:
            start = perf_counter()
            result = evaluate(programs, problem, inputs.search, workload["seed"], plan)
            backend_wall = perf_counter() - start
            fitness_start = perf_counter()
            gaps = gap(result.lengths, problem.reference[None])
            fitness = np.mean(gaps, axis=1, dtype=np.float32)
            fitness_s = perf_counter() - fitness_start
            eval_s = perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_SELF)
        children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        validate_tours(result.tours, problem.n)
        if result.timings["executed_tasks"] != workload["requested_tasks"]:
            raise RuntimeError("逻辑任务漏算")
        np.savez(
            directory / "result.npz",
            lengths=result.lengths,
            tours=result.tours,
            diagnostics=result.diagnostics,
            fitness=fitness,
        )
        record = {
            **workload,
            **result.timings,
            "schema": "gpaco-measurement-v2",
            "status": "completed",
            "cores": cores,
            "cache_state": cache_state,
            "setup_s": setup_s,
            "warmup_s": warmup_s,
            "backend_wall_s": backend_wall,
            "fitness_s": fitness_s,
            "fitness_available_wall_s": eval_s,
            "timing_boundary": "eval_wall_s preserves old backend-return boundary; fitness_available_wall_s includes FP32 fitness",
            "cpu_user_s": after.ru_utime
            - before.ru_utime
            + children_after.ru_utime
            - children_before.ru_utime,
            "cpu_system_s": after.ru_stime
            - before.ru_stime
            + children_after.ru_stime
            - children_before.ru_stime,
            "sampled_peak_process_tree_rss_bytes": memory.peak_rss,
            "sampled_peak_process_tree_pss_bytes": memory.peak_pss
            if memory.pss_available
            else None,
            "memory_samples": memory.samples,
            "memory_sample_interval_s": 1,
            "cpu_affinity": cpus[:cores],
            "host": platform.node(),
            "tour_valid": True,
            "mean_gap_percent": float(gaps.mean()),
            "selected_fitness": fitness.tolist(),
            "cycles": None,
            "instructions": None,
            "ipc": None,
            "hardware_counter_status": "not_collected",
            "instrumented": False,
            "memory_sampling_enabled": True,
            "result_sha256": file_hash(directory / "result.npz"),
            "complete_training": False,
        }
        write_json(directory / "record.json", record)
        return record
    except Exception as error:
        write_json(directory / "FAILED.json", {"error": repr(error), "status": "failed"})
        raise
    finally:
        os.sched_setaffinity(0, affinity)
