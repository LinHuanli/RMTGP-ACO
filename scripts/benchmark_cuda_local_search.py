#!/usr/bin/env python3
"""对 CUDA 2-opt/3-opt 做可复现的端到端 kernel 参数基准。

基准固定 instance、ACO seed 和 GP programs，只改变局部搜索 launch
参数。首轮用于 NVRTC/驻留数据预热，统计轮只报告 GPU event 计时。
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from rmtgp_aco.aco_cuda import solve_population_cuda
from rmtgp_aco.config import (
    ACOConfig,
    ACOVariant,
    CudaPrecision,
    ExecutionBackend,
    GPUMode,
    LocalSearch,
    LSGainSemantics,
    RuntimeConfig,
)
from rmtgp_aco.data import make_problem_batch
from rmtgp_aco.genetic import compile_individual
from rmtgp_aco.program import Instruction, TensorProgram
from rmtgp_aco.sampling import IndexedShard

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "Datasets" / "TSP" / "val_dataset" / "tsp" / "tsp500_uniform_val.txt"
DEFAULT_PROGRAM_ROOT = ROOT / "runs" / "tsp100-2opt-signal-v2" / "audit"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--variant", choices=[item.value for item in ACOVariant], default="as")
    parser.add_argument(
        "--local-search",
        choices=[LocalSearch.TWO_OPT.value, LocalSearch.THREE_OPT.value],
        default=LocalSearch.TWO_OPT.value,
    )
    parser.add_argument("--instances", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--programs", type=int, default=65)
    parser.add_argument("--program-root", type=Path, default=DEFAULT_PROGRAM_ROOT)
    parser.add_argument(
        "--program-profile",
        choices=(
            "archive",
            "edge-eta",
            "ls-gain",
            "pre-freq",
            "post-freq",
            "mixed",
        ),
        default="archive",
        help="archive 使用冻结个体；其余 profile 生成可复现的合成 PH programs",
    )
    parser.add_argument(
        "--program-coefficient-scale",
        type=float,
        default=1.0,
        help="合成程序的系数倍率；设为 0 可在相同 ACO 行为下隔离 terminal 开销",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ls-warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--three-opt-threads",
        type=int,
        nargs="+",
        default=[128, 256, 512],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "performance" / "cuda_local_search_v2.json",
    )
    return parser.parse_args()


def _programs(
    root: Path,
    variant: ACOVariant,
    requested: int,
    profile: str,
    coefficient_scale: float,
) -> list[tuple[Any, Any]]:
    if requested < 1:
        raise ValueError("programs 必须为正整数")
    if profile != "archive":
        result: list[tuple[Any, Any]] = []
        terminal_by_profile = {
            "edge-eta": "EdgeEta",
            "ls-gain": "LSGain",
            "pre-freq": "PreFreq",
            "post-freq": "PostFreq",
        }
        for index in range(requested):
            terminal = terminal_by_profile.get(
                profile,
                "LSGain" if index % 2 == 1 else "EdgeEta",
            )
            coefficient = coefficient_scale * float(index + 1) / float(requested + 1)
            program = TensorProgram(
                instructions=(
                    Instruction("TERMINAL", terminal),
                    Instruction("CONST", coefficient),
                    Instruction("MUL"),
                ),
                role="pheromone",
                expression=f"MUL({terminal},{coefficient:.9g})",
                required_terminals=frozenset({terminal}),
            )
            result.append((None, program))
        return result
    source = root / variant.value / "programs.pkl"
    if requested == 1 or not source.is_file():
        return [(None, None)]
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    individuals = list(payload["individuals"])
    compiled = [compile_individual(item) for item in individuals]
    result = [(None, None), *compiled]
    if requested > len(result):
        raise ValueError(f"请求 {requested} 个 programs，但 {source} 只有 {len(result)} 个")
    return result[:requested]


def _batch(path: Path, count: int):
    shard = IndexedShard.open(path)
    if count < 1 or count > len(shard):
        raise ValueError(f"instances 必须位于 [1, {len(shard)}]")
    return make_problem_batch(
        [shard.get(index) for index in range(count)],
        candidate_size=20,
    )


def main() -> None:
    args = _arguments()
    variant = ACOVariant(args.variant)
    local_search = LocalSearch(args.local_search)
    batch = _batch(args.dataset, args.instances)
    programs = _programs(
        args.program_root,
        variant,
        args.programs,
        args.program_profile,
        args.program_coefficient_scale,
    )
    config = replace(
        ACOConfig.acotsp_local_search_default(
            variant,
            local_search=local_search,
            iterations=args.iterations,
            ants=32,
        ),
        candidate_size=20,
        local_search_candidate_size=20,
        ls_gain_semantics=LSGainSemantics.EDGE_LAST_MOVE,
    )
    combinations = [
        (warps, threads)
        for warps in args.ls_warps
        for threads in (args.three_opt_threads if local_search is LocalSearch.THREE_OPT else [256])
    ]
    records: list[dict[str, Any]] = []
    for warps, threads in combinations:
        runtime = RuntimeConfig(
            aco_backend=ExecutionBackend.CUDA_TILED_V2,
            gpu_mode=GPUMode.SINGLE,
            gpu_devices=(args.device,),
            cuda_precision=CudaPrecision.FP32_FAST,
            cuda_candidate_lanes=8,
            cuda_generated_gp=True,
            cuda_ls_warps_per_block=warps,
            cuda_three_opt_block_threads=threads,
        )
        samples: list[float] = []
        result = None
        for repeat in range(args.warmups + args.repeats):
            result = solve_population_cuda(
                batch,
                config,
                programs,
                seed=1771,
                runtime=runtime,
            )
            if repeat >= args.warmups:
                samples.append(float(result.backend_metrics["kernel_seconds_critical"]))
        assert result is not None
        tasks = result.constructed_tours // (config.resolve_ants(batch.n) * args.iterations)
        records.append(
            {
                "requested_ls_warps_per_block": warps,
                "effective_ls_warps_per_block": int(
                    result.backend_metrics[
                        "local_search_warps_per_block"
                    ]
                ),
                "three_opt_block_threads": threads,
                "kernel_seconds": samples,
                "kernel_seconds_median": median(samples),
                "semantic_programs": tasks // args.instances,
                "tasks": tasks,
                "task_iterations_per_second": (tasks * args.iterations / median(samples)),
                "local_search_move_count": int(result.diagnostics[:, 4].sum().item()),
                "local_search_candidate_check_count": int(result.diagnostics[:, 5].sum().item()),
                "device_name": str(result.backend_metrics["device_names"]),
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "variant": variant.value,
        "local_search": local_search.value,
        "instances": args.instances,
        "iterations": args.iterations,
        "requested_programs": args.programs,
        "program_profile": args.program_profile,
        "program_coefficient_scale": args.program_coefficient_scale,
        "records": records,
        "selected": min(records, key=lambda item: item["kernel_seconds_median"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
