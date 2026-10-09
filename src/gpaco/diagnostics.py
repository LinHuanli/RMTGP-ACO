"""独立 GPU 诊断、状态快照和同状态终端／表达式复核；不更改生产算法。"""

import os
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifact_registry import identity, require_output
from .backends import cpu_python, cuda_backend
from .benchmark_inputs import BenchmarkInputs
from .config import ExecutionPlan
from .data import validate_tours, write_json
from .hardware_campaign import hardware_info
from .hardware_inputs import file_hash, safe_directory
from .language import TERMINAL_IDS, evaluate_reference, pack_programs
from .telemetry import Monitor

WORK_FIELDS = (
    "effective_transitions",
    "candidate_list_probe_positions",
    "fallback_initial_scan_positions",
    "valid_candidate_scores",
    "fallback_valid_candidate_scores",
    "fallback_rank_visited_checks",
    "fallback_rank_distance_pairs",
    "logical_gp_node_candidate_pairs",
    "three_stats_scan_positions",
    "stats_log_tau_logical_calls",
    "stats_log_eta_logical_reads",
    "normal_rank_visited_checks",
    "candidate_lane0_cycles_sum",
    "stats_lane0_cycles_sum",
    "feature_gp_score_lane0_cycles_sum",
    "selection_sync_acs_lane0_cycles_sum",
)


def capabilities():
    params = Path("/proc/driver/nvidia/params")
    permission = None
    if params.exists():
        permission = next(
            (
                line
                for line in params.read_text().splitlines()
                if line.startswith("RmProfilingAdminOnly")
            ),
            None,
        )
    return {
        "ncu_path": shutil.which("ncu"),
        "nsys_path": shutil.which("nsys"),
        "perf_path": shutil.which("perf"),
        "driver_counter_permission": permission,
        "counter_status": "not_collected",
        "occupancy": None,
        "dram_bandwidth": None,
        "warp_stall": None,
        "note": "工具存在不等于与当前CUDA兼容或拥有计数器权限；不自动修改权限",
    }


def replay(result, programs, problem, search, plan, directory):
    """NumPy 重放完整终端；GPU 仅微测相同终端输入上的树求值，不冒充构造分解。"""
    import cupy as cp

    rows, terminals, references, program_ids = [], [], [], []
    for slot, meta in enumerate(result.snapshot_meta):
        if meta[0] < 0:
            continue
        p, b, ant, iteration, step, current, previous, fallback, stagnation, kind = map(int, meta)
        geometry = tuple(
            getattr(problem, f)[b]
            for f in (
                "coords",
                "distances",
                "heuristic",
                "log_heuristic",
                "nearest",
                "full_nn_rank",
            )
        )
        visited = result.snapshot_visited[slot].astype(bool)
        candidates = (
            np.flatnonzero(~visited)
            if fallback
            else problem.nearest[b, current][~visited[problem.nearest[b, current]]].astype(np.int32)
        )
        if not len(candidates) or bool(fallback) != (
            not np.any(~visited[problem.nearest[b, current]])
        ):
            raise ValueError("快照候选集合与 fallback 不一致")
        start = perf_counter()
        context, base_scores, _ = cpu_python.fields(
            programs[p],
            geometry,
            result.snapshot_tau[slot],
            current,
            previous,
            candidates,
            bool(fallback),
            step,
            iteration,
            stagnation,
            search,
        )
        feature_s = perf_counter() - start
        start = perf_counter()
        expected = evaluate_reference(programs[p], context, (len(candidates),))
        tree_s = perf_counter() - start
        expected_scores = base_scores * (
            np.float32(1) + np.float32(search.gamma) * np.tanh(expected)
        )
        np.testing.assert_allclose(
            result.snapshot_scores[slot, candidates], expected_scores, atol=1e-5, rtol=1e-4
        )
        field = np.zeros((len(candidates), 16), np.float32)
        for name, value in context.items():
            field[:, TERMINAL_IDS[name]] = value
        rows.append(
            {
                "slot": slot,
                "program": p,
                "instance": b,
                "ant": ant,
                "aco_iteration": iteration,
                "construction_step": step,
                "fallback": bool(fallback),
                "kind": kind,
                "candidates": len(candidates),
                "cpu_numpy_feature_s": feature_s,
                "cpu_numpy_tree_s": tree_s,
                "captured_cuda_scores_allclose": True,
            }
        )
        terminals.append(field)
        references.append(expected)
        program_ids.extend([p] * len(candidates))
    if not rows:
        raise RuntimeError("没有有效状态快照")
    host_fields, expected = np.concatenate(terminals), np.concatenate(references)
    ids = np.asarray(program_ids, np.int32)
    packed = pack_programs(programs)
    # 使用一个独立求值 kernel，解释器和生成式读取逐位相同终端数组。
    source = (
        (cuda_backend.SOURCE_ROOT / "common.cuh").read_text()
        + "\n"
        + cuda_backend.generated_source(programs)
        + """
extern "C" __global__ void replay_tree(const float* fields,const int* ids,const int8_t* ops,const float* args,const int16_t* tids,const int16_t* lengths,int width,int count,int generated,float* output) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    int p=ids[i]; const float* t=fields+i*16;
    if(generated) { output[i]=evaluate_transition_generated(p,t); return; }
    float stack[31]; int top=0;
    for(int j=0;j<lengths[p];++j) {
        int index=p*width+j,op=ops[index];
        if(op==0) stack[top++]=args[index];
        else if(op==1) stack[top++]=t[tids[index]];
        else {
            float b=stack[--top],v;
            if(op==9) v=fabsf(b); else if(op==10) v=-b;
            else { float a=stack[--top]; v=op==2?a+b:op==3?a-b:op==4?a*b:op==5?(a*b)/(b*b+1.0e-6f):op==7?gp_min(a,b):gp_max(a,b); }
            stack[top++]=sanitize(v);
        }
    } output[i]=sanitize(stack[0]);
}
"""
    )
    kernel = cp.RawKernel(
        source,
        "replay_tree",
        options=(
            "--std=c++17",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ),
    )
    gpu_fields, gpu_ids = cp.asarray(host_fields), cp.asarray(ids)
    gpu_packed = tuple(cp.asarray(x) for x in packed[:4])
    output = cp.empty(len(ids), cp.float32)
    measurements = []
    for generated in (False, True):
        args = (
            gpu_fields,
            gpu_ids,
            *gpu_packed,
            np.int32(packed[0].shape[1]),
            np.int32(len(ids)),
            np.int32(generated),
            output,
        )
        kernel(((len(ids) + 127) // 128,), (128,), args)
        cp.cuda.get_current_stream().synchronize()
        np.testing.assert_allclose(cp.asnumpy(output), expected, atol=1e-5, rtol=1e-4)
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        for _ in range(100):
            kernel(((len(ids) + 127) // 128,), (128,), args)
        end.record()
        end.synchronize()
        measurements.append(
            {
                "generated": generated,
                "calls": 100,
                "candidate_fields_per_call": len(ids),
                "device_interval_s": cp.cuda.get_elapsed_time(start, end) / 1000,
                "scope": "isolated tree kernel repeated 100 times; not fused feature/GP stage share",
                "local_values_allclose": True,
            }
        )
    np.savez(
        directory / "replay_inputs.npz",
        terminals=host_fields,
        program_ids=ids,
        expected_fp32=expected,
    )
    write_json(directory / "snapshot_replay.json", rows)
    write_json(directory / "tree_microbenchmark.json", measurements)
    return {
        "snapshots": len(rows),
        "fallback_snapshots": sum(r["fallback"] for r in rows),
        "tree_candidates": len(ids),
        "local_values_allclose": True,
    }


def run(bundle, directory, generation=1, block=0, reduced_iterations=None):
    directory = safe_directory(directory)
    registry_id = (
        "E00-p01-instrumentation-checks"
        if reduced_iterations is not None
        else "E01-p01-work-diagnostics"
    )
    require_output(directory, registry_id)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = BenchmarkInputs(bundle)
    programs, problem = inputs.load(generation, block)
    search = (
        inputs.search
        if reduced_iterations is None
        else replace(inputs.search, iterations=reduced_iterations)
    )
    plan = ExecutionPlan(
        generated=True, candidate_lanes=8, profile_stages=True, diagnostic_work=True
    )
    write_json(directory / "capabilities.json", capabilities())
    write_json(directory / "hardware.json", hardware_info())
    write_json(
        directory / "request.json",
        {
            **identity(registry_id),
            "run_id": directory.name,
            **inputs.workload(generation, block, programs, problem),
            "search": asdict(search),
            "diagnostic_only": True,
            "reduced_budget_smoke": reduced_iterations is not None,
        },
    )
    monitor = Monitor(os.environ["CUDA_VISIBLE_DEVICES"], directory)
    try:
        # 同卡、同输入的无插桩配对，用于量化观测扰动；不是新的正式性能block。
        with monitor.measure() as plain_telemetry:
            plain = cuda_backend.evaluate(
                programs,
                problem,
                search,
                problem.initialization.seed,
                replace(plan, profile_stages=False, diagnostic_work=False),
            )
        with monitor.measure() as telemetry:
            result = cuda_backend.evaluate(
                programs, problem, search, problem.initialization.seed, plan
            )
        validate_tours(result.tours, problem.n)
        np.testing.assert_array_equal(plain.tours, result.tours)
        np.testing.assert_array_equal(plain.lengths, result.lengths)
        write_json(
            directory / "instrumentation_pair.json",
            {
                "plain": {**plain.timings, **plain_telemetry},
                "instrumented": {**result.timings, **telemetry},
                "bitwise_equal_tours": True,
                "bitwise_equal_lengths": True,
                "device_interval_ratio_instrumented_over_plain": result.timings["device_search_s"]
                / plain.timings["device_search_s"],
                "scope": "single fixed-order diagnostic pair, not confirmatory speedup; compile/setup outside device ratio",
            },
        )
        write_json(
            directory / "iteration_timeline.json",
            {
                "scope": "CUDA event intervals relative to each wave start; includes submission gaps, not Nsight kernel timestamps",
                "events": result.iteration_timeline,
            },
        )
        if int(result.work_counts[..., 0].sum()) != len(
            programs
        ) * problem.size * search.ants * search.iterations * (problem.n - 1):
            raise RuntimeError("诊断转移计数不符合固定预算")
        np.savez(
            directory / "result.npz",
            lengths=result.lengths,
            tours=result.tours,
            diagnostics=result.diagnostics,
            work_counts=result.work_counts,
            snapshot_tau=result.snapshot_tau,
            snapshot_visited=result.snapshot_visited,
            snapshot_meta=result.snapshot_meta,
            snapshot_scores=result.snapshot_scores,
            sampled_logical_tasks=result.sampled_logical_tasks,
        )
        replay_report = replay(result, programs, problem, search, plan, directory)
        write_json(
            directory / "record.json",
            {
                **result.timings,
                **telemetry,
                **identity(registry_id),
                "run_id": directory.name,
                "status": "completed",
                "n": problem.n,
                "variant": search.variant,
                "generation": generation,
                "block": block,
                "counts": dict(
                    zip(WORK_FIELDS, map(int, result.work_counts.sum(axis=(0, 1))), strict=True)
                ),
                "counter_scope": "logical source-level work; not hardware instructions; summed lane0 cycles are not wall time or cross-SM timestamps",
                "cycle_scope": "candidate / shared stats / per-candidate feature+GP score / selection+sync+ACS local update; sampled-copy overhead excluded from these cycle intervals",
                "reduced_budget_smoke": reduced_iterations is not None,
                "state_replay": replay_report,
                "result_sha256": file_hash(directory / "result.npz"),
            },
        )
    finally:
        monitor.close()


def profiler_version(command):
    try:
        return subprocess.check_output([command, "--version"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
