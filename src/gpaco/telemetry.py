"""只读 NVML 采样；能耗、显存与争用记录不冒充 SM occupancy 或核心利用率。"""

import csv
import ctypes as ct
import json
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager

import numpy as np

from .data import write_json


class Memory(ct.Structure):
    _fields_ = [("total", ct.c_ulonglong), ("free", ct.c_ulonglong), ("used", ct.c_ulonglong)]


class Utilization(ct.Structure):
    _fields_ = [("gpu", ct.c_uint), ("memory", ct.c_uint)]


class Nvml:
    """仅依赖系统 NVML 动态库，不修改已有训练使用的 Python 环境。"""

    def __init__(self, uuid):
        self.lib = ct.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("NVML 初始化失败")
        self.handle = ct.c_void_p()
        if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ct.byref(self.handle)) != 0:
            raise RuntimeError("NVML 无法定位 GPU UUID")

    def scalar(self, name, kind=ct.c_uint, *args):
        value = kind()
        function = getattr(self.lib, name, None)
        if function is None or function(self.handle, *args, ct.byref(value)) != 0:
            return None
        return value.value

    def energy(self):
        value = self.scalar("nvmlDeviceGetTotalEnergyConsumption", ct.c_ulonglong)
        return None if value is None else value / 1000.0

    def sample(self):
        memory, utilization = Memory(), Utilization()
        mem_ok = self.lib.nvmlDeviceGetMemoryInfo(self.handle, ct.byref(memory)) == 0
        util_ok = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ct.byref(utilization)) == 0
        power = self.scalar("nvmlDeviceGetPowerUsage")
        return {
            "monotonic_s": time.perf_counter(),
            "unix_s": time.time(),
            "gpu_used_bytes": memory.used if mem_ok else None,
            "utilization_gpu_percent": utilization.gpu if util_ok else None,
            "power_w": None if power is None else power / 1000,
            "energy_counter_j": self.energy(),
            "temperature_c": self.scalar("nvmlDeviceGetTemperature", ct.c_uint, 0),
            "sm_clock_mhz": self.scalar("nvmlDeviceGetClockInfo", ct.c_uint, 1),
            "memory_clock_mhz": self.scalar("nvmlDeviceGetClockInfo", ct.c_uint, 2),
            "throttle_reasons_mask": self.scalar(
                "nvmlDeviceGetCurrentClocksThrottleReasons", ct.c_ulonglong
            ),
        }


def summarize_samples(samples, start_s, end_s, energy_start=None, energy_end=None):
    """计数器差优先；不支持时按实际采样时刻积分，不填造未测量指标。"""
    samples = sorted(
        [row for row in samples if start_s <= row["monotonic_s"] <= end_s],
        key=lambda row: row["monotonic_s"],
    )
    powers = [row for row in samples if row.get("power_w") is not None]
    if energy_start is not None and energy_end is not None and energy_end >= energy_start:
        energy, method = energy_end - energy_start, "nvml_total_energy_counter_delta"
    elif len(powers) >= 2:
        energy = float(
            np.trapezoid(
                [row["power_w"] for row in powers],
                [row["monotonic_s"] for row in powers],
            )
        )
        method = "sampled_power_trapezoid_partial_interval"
    else:
        energy, method = None, "unavailable"
    result = {
        "energy_j": energy,
        "energy_method": method,
        "telemetry_samples": len(samples),
        "energy_scope": "whole_GPU_during_outer_evaluation; includes_setup_not_only_search",
        "power_integration_coverage_s": (
            powers[-1]["monotonic_s"] - powers[0]["monotonic_s"] if powers else 0
        ),
        "sampled_peak_gpu_used_bytes": max(
            (r["gpu_used_bytes"] for r in samples if r.get("gpu_used_bytes") is not None),
            default=None,
        ),
        "other_pids": sorted({p for r in samples for p in r.get("other_pids", [])}),
        "telemetry_errors": sorted(
            {r["telemetry_error"] for r in samples if "telemetry_error" in r}
        ),
        "occupancy_measured": None,
        "dram_bandwidth_measured": None,
    }
    for field in ("power_w", "temperature_c", "sm_clock_mhz", "memory_clock_mhz"):
        values = [r[field] for r in samples if r.get(field) is not None]
        result[f"mean_{field}"] = float(np.mean(values)) if values else None
    result["contended"] = bool(result["other_pids"])
    return result


class Monitor:
    def __init__(self, uuid, directory, interval=1.0):
        self.uuid, self.directory, self.interval = uuid, directory, interval
        self.stop = threading.Event()
        self.samples = []
        self.guard = threading.Lock()
        try:
            self.nvml = Nvml(uuid)
            self.error = None
        except (OSError, RuntimeError, AttributeError) as error:
            self.nvml, self.error = None, str(error)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def read(self):
        if self.nvml is None:
            row = {"monotonic_s": time.perf_counter(), "telemetry_error": self.error}
        else:
            row = self.nvml.sample()
        return row

    def processes(self):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            pids = [
                int(r[1])
                for r in csv.reader(output.splitlines())
                if len(r) >= 2 and r[0].strip() == self.uuid and int(r[1]) != os.getpid()
            ]
            return {"other_pids": pids}
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return {"telemetry_error": str(error), "other_pids": []}

    def run(self):
        # 每秒功耗采样、每十五秒争用检查；GPU counters 无权限时不尝试修改驱动。
        tick, processes = 0, {}
        with (self.directory / "telemetry.jsonl").open("a") as handle:
            while not self.stop.is_set():
                if tick % 15 == 0:
                    processes = self.processes()
                row = {**self.read(), **processes}
                with self.guard:
                    self.samples.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if tick % 15 == 0:
                    write_json(
                        self.directory / "heartbeat.json",
                        {
                            **row,
                            "pid": os.getpid(),
                            "host": platform.node(),
                        },
                    )
                tick += 1
                self.stop.wait(self.interval)

    @contextmanager
    def measure(self):
        # 起止显式采样用于覆盖短测量；查询开销记入外层 wall，不改 CUDA event 区间。
        before = {**self.read(), **self.processes()}
        record = {}
        try:
            yield record
        finally:
            after = {**self.read(), **self.processes()}
            with self.guard:
                rows = [before, *self.samples, after]
            record.update(
                summarize_samples(
                    rows,
                    before["monotonic_s"],
                    after["monotonic_s"],
                    before.get("energy_counter_j"),
                    after.get("energy_counter_j"),
                )
            )

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)
