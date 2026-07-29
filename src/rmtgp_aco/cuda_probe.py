"""独立的 Blackwell 存储精度与标量读取探针。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

_SOURCE = Path(__file__).with_name("cuda") / "precision_probe.cu"


def _elapsed_ms(cp: Any, launch) -> float:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    launch()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end))


def probe_cuda_precisions(
    *,
    device: int = 0,
    elements: int = 1 << 22,
    repetitions: int = 8,
    repeats: int = 5,
    seed: int = 20260730,
) -> dict[str, object]:
    """测量各存储格式；``elements`` 必须为 2 的幂以免热循环使用除法。"""

    if elements < 1024 or elements & (elements - 1):
        raise ValueError("elements 必须是至少 1024 的 2 次幂")
    if repetitions < 1 or repeats < 1:
        raise ValueError("repetitions 和 repeats 必须为正整数")

    import cupy as cp

    source = _SOURCE.read_text(encoding="utf-8")
    digest = sha256(source.encode("utf-8")).hexdigest()
    with cp.cuda.Device(device):
        properties = cp.cuda.runtime.getDeviceProperties(device)
        major = int(properties["major"])
        minor = int(properties["minor"])
        compiled_started = perf_counter()
        module = cp.RawModule(
            code=source,
            options=(
                "--std=c++17",
                f"--gpu-architecture=compute_{major}{minor}",
            ),
            backend="nvrtc",
        )
        compile_seconds = perf_counter() - compiled_started
        raw_name = properties["name"]
        device_name = (
            raw_name.decode("utf-8")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )

        # 混合距离、启发式和 log-启发式的典型动态范围。固定随机输入使
        # 格式间误差与吞吐可直接配对比较。
        rng = cp.random.RandomState(seed)
        selector = rng.randint(0, 3, size=elements, dtype=cp.int32)
        distance = rng.uniform(1.0e-3, 1.5, size=elements).astype(cp.float32)
        heuristic = cp.exp(
            rng.uniform(np.log(0.65), np.log(448.0), size=elements)
        ).astype(cp.float32)
        log_eta = rng.uniform(-6.0, 6.0, size=elements).astype(cp.float32)
        values = cp.where(
            selector == 0,
            distance,
            cp.where(selector == 1, heuristic, log_eta),
        ).astype(cp.float32)
        output = cp.empty(elements, dtype=cp.float32)
        decoded = cp.empty(elements, dtype=cp.float32)
        threads = 256
        blocks = (elements + threads - 1) // threads

        profile_specs: dict[str, dict[str, object]] = {
            "fp64": {
                "dtype": cp.float64,
                "bytes": 8.0,
                "encode": "encode_fp64",
                "decode": "decode_fp64",
                "probe": "probe_fp64",
                "scale": None,
            },
            "fp32": {
                "dtype": cp.float32,
                "bytes": 4.0,
                "encode": None,
                "decode": None,
                "probe": "probe_fp32",
                "scale": None,
            },
            "fp16_mixed": {
                "dtype": cp.float16,
                "bytes": 2.0,
                "encode": "encode_fp16",
                "decode": "decode_fp16",
                "probe": "probe_fp16",
                "scale": None,
            },
            "bf16_mixed": {
                "dtype": cp.uint16,
                "bytes": 2.0,
                "encode": "encode_bf16",
                "decode": "decode_bf16",
                "probe": "probe_bf16",
                "scale": None,
            },
            "fp8_e4m3": {
                "dtype": cp.uint8,
                "bytes": 1.0,
                "encode": "encode_fp8_e4m3",
                "decode": "decode_fp8_e4m3",
                "probe": "probe_fp8_e4m3",
                "scale": float(cp.max(cp.abs(values)).get()) / 448.0,
            },
            "nvfp4_block16_proxy": {
                "dtype": cp.uint8,
                # 4-bit value + optimistic FP32 scale amortized over 16 values.
                "bytes": 0.75,
                "encode": "encode_nvfp4",
                "decode": "decode_nvfp4",
                "probe": "probe_nvfp4",
                "scale": "block16",
            },
        }

        profiles: dict[str, dict[str, float | list[float]]] = {}
        for name, spec in profile_specs.items():
            packed_count = (
                (elements + 1) // 2
                if name == "nvfp4_block16_proxy"
                else elements
            )
            packed = cp.empty(packed_count, dtype=spec["dtype"])
            scale = spec["scale"]
            block_scales = None
            if scale == "block16":
                block_scales = (
                    cp.max(cp.abs(values.reshape(-1, 16)), axis=1) / 6.0
                ).astype(cp.float32)
                block_scales = cp.maximum(
                    block_scales,
                    cp.float32(1.0e-30),
                )
            encode_name = spec["encode"]
            if encode_name is None:
                cp.copyto(packed, values)
                cp.copyto(decoded, values)
            else:
                encode = module.get_function(str(encode_name))
                encode_count = (
                    packed_count
                    if name == "nvfp4_block16_proxy"
                    else elements
                )
                encode_args: tuple[object, ...] = (
                    values,
                    packed,
                    np.int32(elements),
                )
                if scale == "block16":
                    encode_args += (block_scales,)
                elif scale is not None:
                    encode_args += (np.float32(1.0 / max(scale, 1e-30)),)
                encode(
                    ((encode_count + threads - 1) // threads,),
                    (threads,),
                    encode_args,
                )
                decode = module.get_function(str(spec["decode"]))
                decode_args: tuple[object, ...] = (
                    packed,
                    decoded,
                    np.int32(elements),
                )
                if scale == "block16":
                    decode_args += (block_scales,)
                elif scale is not None:
                    decode_args += (np.float32(scale),)
                decode((blocks,), (threads,), decode_args)
            cp.cuda.get_current_stream().synchronize()

            error = decoded - values
            denominator = cp.maximum(cp.abs(values), cp.float32(1.0e-6))
            error_summary = {
                "mean_abs_error": float(cp.mean(cp.abs(error)).get()),
                "max_abs_error": float(cp.max(cp.abs(error)).get()),
                "mean_relative_error": float(
                    cp.mean(cp.abs(error) / denominator).get()
                ),
            }
            probe = module.get_function(str(spec["probe"]))
            probe_args: tuple[object, ...] = (
                packed,
                output,
                np.int32(elements),
                np.int32(repetitions),
            )
            if scale == "block16":
                probe_args += (block_scales,)
            elif scale is not None:
                probe_args += (np.float32(scale),)

            def launch(
                probe_kernel=probe,
                arguments=probe_args,
            ) -> None:
                probe_kernel((blocks,), (threads,), arguments)

            launch()
            cp.cuda.get_current_stream().synchronize()
            timings = [_elapsed_ms(cp, launch) for _ in range(repeats)]
            selected_ms = median(timings)
            scale_summary: dict[str, float | str] = {}
            if scale == "block16":
                assert block_scales is not None
                scale_summary = {
                    "scale_mode": "fp32-per-16-values-proxy",
                    "scale_min": float(cp.min(block_scales).get()),
                    "scale_max": float(cp.max(block_scales).get()),
                }
            else:
                scale_summary = {
                    "scale_mode": "global",
                    "scale": float(scale) if scale is not None else 1.0,
                }
            profiles[name] = {
                "kernel_ms": selected_ms,
                "repeat_ms": timings,
                "giga_values_per_second": (
                    elements * repetitions / (selected_ms * 1.0e6)
                ),
                "effective_storage_gb_per_second": (
                    elements
                    * repetitions
                    * float(spec["bytes"])
                    / (selected_ms * 1.0e6)
                ),
                "storage_bytes_per_value": float(spec["bytes"]),
                "checksum": float(cp.sum(output, dtype=cp.float64).get()),
                **error_summary,
                **scale_summary,
            }

    return {
        "schema_version": 1,
        "source_sha256": digest,
        "device": device,
        "device_name": device_name,
        "compute_capability": f"{major}.{minor}",
        "elements": elements,
        "repetitions": repetitions,
        "repeats": repeats,
        "compile_seconds": compile_seconds,
        "profiles": profiles,
    }
