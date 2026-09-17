"""新组件对照的逐型号验收：跨框架表达式、预算、来源、记录与恢复均须通过。"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import time
import numpy as np
import torch

from .common import atomic_json, digest, experiment, file_hash, now, read_json, source_manifest, validate_tours


def validate(task, out):
    out = Path(out); target = out/"jobs"/task["id"]
    target.mkdir(parents=True, exist_ok=True)
    queued = read_json(out/"queue"/(task["id"]+".json"))
    if source_manifest()["source_hash"] != queued["source_hash"]:
        raise ValueError("验收不是在冻结源码中执行")
    started = time.perf_counter()
    if task["kind"] == "explanation_kernel_validation":
        import pytest
        tests = Path(__file__).parent/"tests"
        result = pytest.main([str(tests/"test_gpu_controls.py"), str(tests/"test_gpu_diagnostics.py"),
                              str(tests/"test_explanation_gpu.py"), "-q", "-rA",
                              "--basetemp", str(target/"pytest-artifacts"),
                              "--junitxml", str(target/"gpu_checks.xml")])
        if result != 0:
            raise AssertionError(f"GPU 控制与恢复验收失败：{result}")
        # 只把真正执行的测试计为通过，不能让全部 skipped 的任务放行。
        import xml.etree.ElementTree as ET
        root = ET.parse(target/"gpu_checks.xml").getroot()
        cases = root.findall(".//testcase")
        if not cases or any(c.find("skipped") is not None for c in cases):
            raise AssertionError("GPU 验收有跳过项，不能放行正式实验")
        evidence = {"gpu_test_cases": len(cases), "test_report_sha256": file_hash(target/"gpu_checks.xml")}
    else:
        evidence = input_validation(task, out)
    status = {"status": "completed", "validation_status": "passed", "task_hash": digest(task),
              "source_hash": queued["source_hash"], "wall_seconds": time.perf_counter()-started,
              "evidence": evidence, "completed_at": now(),
              "scope": "工程与数学输入验收；不据此判断研究假设成立"}
    atomic_json(target/"status.json", status)
    return status


def input_validation(task, out):
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    from rmtgp_aco.mechanisms import MechanismConfig, SolverControl, InstrumentationConfig
    from .evaluate import program_entries, run_task
    from .prepare import batch
    from .numerical_validation import operator_probe, resume_probe
    from .terminal_oracle import inspect
    from .explanation_campaign import conditions
    target = out/"jobs"/task["id"]; variant = task["variant"]
    regression = None
    if (out/"validation/edge_tau_regression.npz").exists():
        record = read_json(out/"validation/edge_tau_regression.json")
        if file_hash(out/"validation/edge_tau_regression.npz") != record["input_sha256"]:
            raise ValueError("数值回归输入哈希变化")
        with np.load(out/"validation/edge_tau_regression.npz", allow_pickle=False) as a:
            regression = a["values"]
    operator_probe("centered_fp32", target, regression_rows=regression)
    inst = InstrumentationConfig(profile="mechanism_v3", schema_version=3)
    mechanism = MechanismConfig(terminal_statistics="centered_fp32")
    child = {"id": task["id"]+"-recorded", "kind": "explanation_validation_run", "stage": "validation",
        "split": "diagnosis_dev", "indices": list(range(task["instances"])), "replicate": 0, "seed": 57231,
        "variant": variant, "condition": variant+"_native", "iterations": 5000, "stop_iteration": task["steps"],
        "training_variants": ["as", "mmas"], "modes": ["full"], "mechanism": asdict(mechanism),
        "instrumentation": asdict(inst)}
    problem = batch("diagnosis_dev", child["indices"], out)
    aco, runtime = experiment(variant)
    entries = program_entries(variant, training_variants=("as", "mmas"))
    programs = [e["program"] for e in entries]
    # 预热不混入正式求解或审计开销比较。
    solve_population_cuda_anytime(problem, aco, programs, seed=57231, runtime=runtime,
        control=SolverControl(mechanism, InstrumentationConfig("off"), stop_iteration=1))
    record_status = run_task(child, out)
    recorded = out/"jobs"/child["id"]
    oracle = inspect(recorded/"diagnostics")
    if oracle["status"] != "passed":
        raise ArithmeticError("跨框架固定表达式的终端输入核验失败")
    with np.load(recorded/"raw.npz", allow_pickle=False) as a:
        expected_tour = a["tour"]; expected_curve = a["anytime"]
    timings = {}
    for label, order, options in (("without_recording", list(range(problem.batch_size)), runtime),
                                  ("changed_chunks", list(range(problem.batch_size)), replace(runtime, gpu_task_chunk_size=7)),
                                  ("reordered_instances", np.random.default_rng(88241).permutation(problem.batch_size).tolist(), runtime)):
        start = time.perf_counter()
        result = solve_population_cuda_anytime(problem.take(order), aco, programs, seed=57231, runtime=options,
            control=SolverControl(mechanism, InstrumentationConfig("off"), stop_iteration=task["steps"]))
        inverse = np.argsort(order)
        np.testing.assert_array_equal(result.best_tour.numpy()[:, inverse], expected_tour)
        np.testing.assert_array_equal(result.anytime_best.numpy()[:, inverse, :task["steps"]], expected_curve)
        timings[label] = time.perf_counter()-start
    recovery = resume_probe("centered_fp32", variant, problem.take([0, 1]), aco, runtime, programs)
    source_checks = []
    for name, control_config in conditions().items():
        if not name.startswith(variant+"_"):
            continue
        control = SolverControl(control_config, inst, collected=[], stop_iteration=260)
        result = solve_population_cuda_anytime(problem.take([0, 1]), aco, programs, seed=91457, runtime=runtime, control=control)
        validate_tours(result.best_tour.numpy(), problem.n)
        if not torch.isfinite(result.best_length).all():
            raise ArithmeticError("来源干预产生非有限长度")
        for shard in control.collected:
            trace = shard["trace"][:, :260]
            if np.max(trace[..., 20]) > 1e-5:
                raise ArithmeticError("来源干预未保持预算")
            if name.endswith("current_best") and not np.all((trace[..., 0] == 0) & (trace[..., 3] == 1)):
                raise AssertionError("本轮最优来源选择错误")
            if name.endswith("all_current") and not np.all(trace[..., 3] == 32):
                raise AssertionError("全部当前来源数量错误")
            if name.endswith("history_calendar"):
                steps = np.arange(1, 261); age = np.maximum(steps-2, 0)
                period = np.select((age < 25, age < 75, age < 125, age < 250), (25, 5, 3, 2), default=1)
                np.testing.assert_array_equal(trace[..., 0], np.broadcast_to(np.where(steps % period == 0, 2, 0), trace[..., 0].shape))
            if name.endswith("no_restart") and trace[..., 8].any():
                raise AssertionError("关闭重启后仍发生重启")
            if name.endswith("no_floor") and trace[..., 6].any():
                raise AssertionError("关闭下界后仍发生下界保护")
            if trace[..., 7].any():
                raise AssertionError("当前局部搜索分支不应出现硬上界裁剪")
        source_checks.append({"condition": name, "status": "passed", "iterations": 260})
    legacy = None
    if variant == "mmas":
        # 原实现仅用于解释历史行为；仍须与原冻结 CUDA 精确一致。
        from .numerical_validation import validate as legacy_validate
        prior = read_json(ROOT_OLD()/"protocol/queue_freeze.json")
        old_task = {"id": task["id"]+"-legacy-reference", "kind": "numeric_validation", "variant": "mmas",
                    "numeric_mode": "legacy", "instances": 2, "steps": 100, "historical_snapshot": prior["snapshot"]}
        legacy = legacy_validate(old_task, out)
    index = read_json(recorded/"diagnostics/index.json")
    size = sum(r["compressed_bytes"] for r in index["files"].values())
    return {"models": [e["id"] for e in entries], "oracle_status": oracle["status"], "source_checks": source_checks,
        "recovery": recovery, "legacy_reference_status": legacy and legacy["validation_status"],
        "recorded_job": child["id"], "recorded_files": record_status["files"], "wall_seconds": timings,
        "recorded_wall_seconds": record_status["wall_seconds"], "compressed_bytes": size,
        "projected_bytes_per_5000_iteration_logical_solve": size*5000/task["steps"]/(problem.batch_size*len(entries)),
        "scope": "32 实例和六个跨框架固定表达式；260 轮来源边界验收；计时含首次审计准备，不是算法加速比"}


def ROOT_OLD():
    from .common import ROOT
    return ROOT/"control_experiments/mmas_ls/artifacts/numerical-v1"
