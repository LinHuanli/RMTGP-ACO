"""固定完整状态的成对干预；分别保存即时更新和后续 500 轮，不改变归一化时钟。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
import numpy as np

from .common import atomic_json, atomic_npz, digest, experiment, file_hash, now, read_json, source_manifest, validate_tours
from .diagnostic_forks import read_owner_snapshot, transition_distribution
from .evaluate import program_entries
from .explanation_analysis import canonical_edges
from .prepare import batch

BRANCH_NAMES = {
    "native_without_gp": "原生来源，不使用 GP",
    "native_construction_only": "原生来源，只使用构造决策树",
    "native_update_only": "原生来源，只使用信息素更新树",
    "native_both_trees": "原生来源，使用两棵树",
    "current_without_gp": "仅本轮最优来源，不使用 GP",
    "current_both_trees": "仅本轮最优来源，使用两棵树",
}


def branch_specs(mechanism, program):
    tr, ph = program
    current = replace(mechanism, source_policy="iteration_best")
    return {
        "native_without_gp": ((None, None), mechanism),
        "native_construction_only": ((tr, None), mechanism),
        "native_update_only": ((None, ph), mechanism),
        "native_both_trees": ((tr, ph), mechanism),
        "current_without_gp": ((None, None), current),
        "current_both_trees": ((tr, ph), current),
    }


def select_snapshot(job, owner, iteration, phase):
    """查找实际包含 owner 全部实例的分块；不把另一个模型的状态拼接过来。"""
    index = read_json(job/"diagnostics/index.json")
    spec = read_json(job/"diagnostics/specification.json")
    b = len(spec["task"]["indices"])
    with np.load(job/"raw.npz", allow_pickle=False) as raw:
        representative = int(raw["behavior_alias"][owner])
    matches = []
    for name, record in index["files"].items():
        meta = record["metadata"]
        if (meta["kind"] == "permanent" and meta["iteration"] == iteration and meta["phase"] == phase
                and sum(int(v)//b == representative for v in meta["flat_indices"]) == b):
            matches.append(name)
    if len(matches) != 1:
        raise ValueError("永久状态缺失、重复或不能完整覆盖状态生成者")
    return read_owner_snapshot(job, matches[0], owner)


def reduced_state(state, cp, iteration):
    """四个预定后续时刻保存解释所需状态；逐轮质量曲线另存。"""
    names = ("pheromone_workspace", "tour_workspace", "pre_tour_workspace", "best_tours", "global_best_lengths",
             "length_workspace", "length_before_workspace", "audit_tau", "audit_sources", "deposit_workspace", "stagnation")
    result = {k: cp.asnumpy(state[k]) for k in names}
    result["trace"] = cp.asnumpy(state["mechanism_trace"][:, iteration-1])
    result["source_info"] = cp.asnumpy(state["audit_source_info"][:, (iteration-1)%100])
    result["source_hash"] = cp.asnumpy(state["audit_source_hash"][:, (iteration-1)%100])
    return result


def structure_difference(left, right):
    """同一实例、蚂蚁和随机流的差异；分母为零时返回 NaN 而不是假造零比例。"""
    n = left["tour_workspace"].shape[-1]-1
    before = []; after = []
    for a, b, c, d in zip(left["pre_tour_workspace"], right["pre_tour_workspace"],
                         left["tour_workspace"], right["tour_workspace"]):
        pre_a, pre_b = canonical_edges(a, n), canonical_edges(b, n)
        post_a, post_b = canonical_edges(c, n), canonical_edges(d, n)
        before.append([2*(n-np.intersect1d(x, y, assume_unique=True).size) for x, y in zip(pre_a, pre_b)])
        after.append([2*(n-np.intersect1d(x, y, assume_unique=True).size) for x, y in zip(post_a, post_b)])
    before, after = np.asarray(before), np.asarray(after)
    ratio = np.divide(after, before, out=np.full(after.shape, np.nan), where=before > 0)
    return before, after, ratio


def immediate_updates(snapshot, spec, program, problem, aco, runtime, mechanism, target):
    """同一局部搜索结束状态，三种来源 × 开关更新树；总预算应保持一致。"""
    import cupy as cp
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    from rmtgp_aco.mechanisms import SolverControl, InstrumentationConfig
    inst = InstrumentationConfig(profile="mechanism_v3", schema_version=3)
    iteration = snapshot["iteration"]
    states = {}; hashes = {}
    for policy in ("iteration_best", "restart_best_only", "global_best_only"):
        for enabled in (False, True):
            name = policy+("_learned_update" if enabled else "_uniform_update")
            selected_program = (program[0], program[1] if enabled else None)
            actual = replace(mechanism, source_policy=policy)
            captured = {}
            def observe(phase, step, state):
                if phase == "iteration_end":
                    captured.update(reduced_state(state, cp, step))
            solve_population_cuda_anytime(problem, aco, [selected_program], seed=spec["seed"], runtime=runtime,
                control=SolverControl(actual, inst, observer=observe, resume=snapshot, stop_iteration=iteration))
            states[name] = captured
            path = target/("immediate-"+name+".npz")
            atomic_npz(path, **captured); hashes[path.name] = file_hash(path)
    budgets = [s["trace"][:, 4] for s in states.values()]
    for values in budgets[1:]:
        np.testing.assert_array_equal(values, budgets[0])
    distance = problem.distances.numpy(); order = np.argsort(distance, axis=-1, kind="stable")
    ranks = np.empty_like(order); np.put_along_axis(ranks, order, np.arange(problem.n)[None, None, :], axis=-1)
    geometry = {"distances": distance, "coords": problem.coords.numpy(), "nearest": problem.nn_indices.numpy(), "ranks": ranks}
    # 第 1 轮和指定采样轮均有完整上下文；不把构造路径端点当成新的观测样本。
    rows = []; mask = ~np.eye(problem.n, dtype=bool)
    for policy in ("iteration_best", "restart_best_only", "global_best_only"):
        base = states[policy+"_uniform_update"]; full = states[policy+"_learned_update"]
        for i in range(problem.batch_size):
            np.testing.assert_array_equal(base["audit_sources"][i, 0], full["audit_sources"][i, 0])
            budget = float(base["trace"][i, 4])
            deposit_change = np.abs(full["deposit_workspace"][i, 0].astype(float)-base["deposit_workspace"][i, 0]).sum()/budget
            phase_changes = []
            for phase in range(4):
                a = base["audit_tau"][i, phase] if phase < 3 else base["pheromone_workspace"][i]
                b = full["audit_tau"][i, phase] if phase < 3 else full["pheromone_workspace"][i]
                phase_changes.append(float(np.abs(b.astype(float)-a)[mask].sum()/max(np.abs(a.astype(float))[mask].sum(), 1e-30)))
            probabilities = []
            for context, visited in zip(snapshot["arrays"]["audit_context"][i], snapshot["arrays"]["audit_visited"][i]):
                if context[0] < 0:
                    continue
                result = [transition_distribution(s["pheromone_workspace"][i], problem, i, context, visited,
                          program[0], aco, iteration+1, int(s["stagnation"][i]), geometry) for s in (base, full)]
                np.testing.assert_array_equal(result[0][0], result[1][0])
                probabilities.append(float(.5*np.abs(result[1][1]-result[0][1]).sum()))
            rows.append({"source_policy": policy, "instance": i, "source_length": float(full["trace"][i, 13]),
                "same_state_budget": budget, "deposit_change": float(deposit_change), "relative_tau_changes": phase_changes,
                "cpu_fp64_next_choice_probability_change": float(np.mean(probabilities)) if probabilities else None,
                "probe_count": len(probabilities), "restart_executed_uniform": int(base["trace"][i, 8]),
                "restart_executed_learned": int(full["trace"][i, 8])})
    # 同一更新树状态下，不同来源也必须直接比较，不能只有更新树开关对照。
    for mode in ("uniform_update", "learned_update"):
        base = states["iteration_best_"+mode]
        for policy in ("restart_best_only", "global_best_only"):
            full = states[policy+"_"+mode]
            for i in range(problem.batch_size):
                a = canonical_edges(base["audit_sources"][i, 0], problem.n)
                b = canonical_edges(full["audit_sources"][i, 0], problem.n)
                tau_a = base["pheromone_workspace"][i].astype(float); tau_b = full["pheromone_workspace"][i].astype(float)
                probability_changes = []
                for context, visited in zip(snapshot["arrays"]["audit_context"][i], snapshot["arrays"]["audit_visited"][i]):
                    if context[0] < 0:
                        continue
                    candidates_a, probability_a = transition_distribution(tau_a, problem, i, context, visited, program[0],
                        aco, iteration+1, int(base["stagnation"][i]), geometry)
                    candidates_b, probability_b = transition_distribution(tau_b, problem, i, context, visited, program[0],
                        aco, iteration+1, int(full["stagnation"][i]), geometry)
                    np.testing.assert_array_equal(candidates_a, candidates_b)
                    probability_changes.append(float(.5*np.abs(probability_a-probability_b).sum()))
                rows.append({"contrast": "iteration_best_to_"+policy, "update": mode, "instance": i,
                    "same_source_edges": bool(np.array_equal(a, b)), "source_edge_overlap": float(np.intersect1d(a, b).size/problem.n),
                    "source_length_change": float(full["trace"][i, 13]-base["trace"][i, 13]),
                    "relative_tau_change": float(np.abs(tau_b-tau_a)[mask].sum()/max(np.abs(tau_a)[mask].sum(), 1e-30)),
                    "cpu_fp64_next_choice_probability_change": float(np.mean(probability_changes)) if probability_changes else None,
                    "probe_count": len(probability_changes)})
    atomic_json(target/"immediate_comparisons.json", {"rows": rows,
        "scope": "同一局部搜索结束状态、同预算；概率为 CPU FP64 固定上下文参考核，不是 GPU 逐位概率",
        "phases": ["更新前", "蒸发与下界保护后", "沉积后", "重启后"]})
    hashes["immediate_comparisons.json"] = file_hash(target/"immediate_comparisons.json")
    return hashes


def run(task, out):
    import cupy as cp
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    from rmtgp_aco.mechanisms import SolverControl, InstrumentationConfig, MechanismConfig
    out = Path(out); job = out/"jobs"/task["parent"]; target = out/"jobs"/task["id"]
    target.mkdir(parents=True, exist_ok=True)
    snapshot, spec, record = select_snapshot(job, task["owner"], task["snapshot_iteration"], "iteration_end")
    if spec["source"] != source_manifest()["source_hash"]:
        raise ValueError("同状态干预须与状态生成使用同一不可变源码")
    if task["owner"] and spec["models"][task["owner"]]["seed"] != task["champion"]:
        raise ValueError("GP 生成状态只能与自身固定表达式配对")
    entries = program_entries("mmas")
    entry = next(e for e in entries if e["seed"] == task["champion"])
    model = next(m for m in read_json(out/"manifests/checkpoints.json") if m["id"] == entry["id"])
    if entry["file_hash"] != model["file_hash"]:
        raise ValueError("表达式已不匹配冻结模型")
    problem = batch(spec["task"]["split"], spec["task"]["indices"], out)
    aco, runtime = experiment("mmas"); runtime = replace(runtime, gpu_task_chunk_size=0)
    mechanism = MechanismConfig(**spec["task"]["mechanism"])
    inst = InstrumentationConfig(profile="mechanism_v3", schema_version=3)
    stop = snapshot["iteration"]+task["continuation_iterations"]
    if stop > aco.iterations:
        raise ValueError("干预越过原始归一化时域")
    states = {}; arrays = {}; aliases = {}; identities = {}; files = {}
    started = time.perf_counter()
    for name, (program, config) in branch_specs(mechanism, entry["program"]).items():
        identity = digest({"program": [None if p is None or p.is_exact_zero else p.expression for p in program],
                           "mechanism": config.digest})
        if identity in identities:
            other = identities[identity]; aliases[name] = other; states[name] = states[other]
            arrays[name+"_anytime"] = arrays[other+"_anytime"]; arrays[name+"_length"] = arrays[other+"_length"]
            continue
        identities[identity] = name; states[name] = {}; history = {}
        def observe(phase, iteration, state):
            if phase == "initialised":
                if state["count"] != problem.batch_size:
                    raise ValueError("同状态分支必须完整保留实例块")
                count = state["count"]; width = task["continuation_iterations"]
                history["ls"] = cp.empty((count, width, 32, 3), dtype=cp.float32)
                history["ls_counts"] = cp.empty((count, width, 32, 4), dtype=cp.uint64)
                history["sources"] = cp.empty((count, width, 1, 6), dtype=cp.float32)
                history["source_hashes"] = cp.empty((count, width, 1, 2), dtype=cp.uint64)
                history["counters"] = cp.empty((count, width, 4), dtype=cp.uint32)
            elapsed = iteration-snapshot["iteration"]
            if phase == "iteration_end":
                position = elapsed-1; ring = (iteration-1)%100
                history["ls"][:, position, :, 0] = state["length_before_workspace"]
                history["ls"][:, position, :, 1] = state["length_workspace"]
                history["ls"][:, position, :, 2] = (state["origin_workspace"] > 0).sum(-1)
                history["ls_counts"][:, position] = state["audit_ls_counts"][:, ring]
                history["sources"][:, position] = state["audit_source_info"][:, ring, :1]
                history["source_hashes"][:, position] = state["audit_source_hash"][:, ring, :1]
                history["counters"][:, position] = state["audit_counters"][:, ring]
                if elapsed in task["measurement_steps"]:
                    states[name][elapsed] = reduced_state(state, cp, iteration)
                if iteration == stop:
                    history["trace"] = state["mechanism_trace"][:, snapshot["iteration"]:stop].copy()
        result = solve_population_cuda_anytime(problem, aco, [program], seed=spec["seed"], runtime=runtime,
            control=SolverControl(config, inst, observer=observe, resume=snapshot, stop_iteration=stop))
        validate_tours(result.best_tour.numpy(), problem.n)
        arrays[name+"_anytime"] = result.anytime_best.numpy()[..., snapshot["iteration"]:stop]
        arrays[name+"_length"] = result.best_length.numpy()
        path = target/(name+"-iteration-history.npz")
        atomic_npz(path, **{k: cp.asnumpy(v) for k, v in history.items()}); files[path.name] = file_hash(path)
        if set(states[name]) != set(task["measurement_steps"]):
            raise ValueError("分叉没有记录全部预定时刻")
        for elapsed, state in states[name].items():
            path = target/f"{name}-after{elapsed:03d}.npz"
            atomic_npz(path, **state); files[path.name] = file_hash(path)
    structures = {}
    for left, right in (("native_without_gp", "native_construction_only"), ("native_without_gp", "native_both_trees"),
                        ("native_update_only", "native_both_trees"), ("current_without_gp", "current_both_trees")):
        for elapsed in task["measurement_steps"]:
            before, after, ratio = structure_difference(states[left][elapsed], states[right][elapsed])
            prefix = f"{left}__{right}__after{elapsed}"
            structures[prefix+"_before"] = before; structures[prefix+"_after"] = after; structures[prefix+"_ratio"] = ratio
    atomic_npz(target/"result.npz", **arrays, reference=problem.reference_length.numpy())
    atomic_npz(target/"local_search_difference.npz", **structures)
    for name in ("result.npz", "local_search_difference.npz"):
        files[name] = file_hash(target/name)
    post, _, post_record = select_snapshot(job, task["owner"], task["snapshot_iteration"], "post_ls")
    files.update(immediate_updates(post, spec, entry["program"], problem, aco, runtime, mechanism, target))
    atomic_json(target/"manifest.json", {"task": task, "source_hash": spec["source"], "parent": task["parent"],
        "snapshot_hash": record["sha256"], "post_ls_snapshot_hash": post_record["sha256"], "expression_file_hash": entry["file_hash"],
        "branches": BRANCH_NAMES, "behavior_aliases": aliases, "wall_seconds": time.perf_counter()-started,
        "original_horizon": aco.iterations, "completed_at": now(),
        "scope": "固定状态有限时域干预，不是重新训练；局部搜索前差异为零时比例未定义；原实现和稳定实现分别统计"})
    files["manifest.json"] = file_hash(target/"manifest.json")
    status = {"status": "completed", "completed_at": now(), "files": files}
    atomic_json(target/"status.json", status)
    return status
