"""实验记录、不可变运行标识、基线缓存和同步进化训练。"""

import importlib.metadata
import json
import os
import pickle
import platform
import random
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import ExecutionPlan, config_hash
from .data import ROOT, load_split, prepare_problem, validate_tours, write_json
from .evolution import initial_population, next_population
from .language import ProgramSpec, parse_tree


def source_hash():
    directory = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(directory.rglob("*")):
        if path.suffix in (".py", ".cu", ".cuh"):
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def namespace_seed(root_seed, phase, generation=0, repeat=0):
    """命名空间进入 64 位 key；实验内登记和检查不同用途的 key。"""
    return int.from_bytes(
        sha256(f"v1:{root_seed}:{phase}:{generation}:{repeat}".encode()).digest()[:8], "little"
    )


def metadata():
    def command(arguments):
        try:
            return subprocess.check_output(arguments, text=True, timeout=20).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    def version(name):
        # CPU-only 环境不应因没有安装 CUDA / Torch 而无法记录实验。
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_commit": os.environ.get("GPACO_COMMIT")
        or command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "runtime_snapshot": os.environ.get("GPACO_SNAPSHOT"),
        "source_hash": source_hash(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "numba", "llvmlite", "torch", "cupy-cuda13x", "deap")
        },
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit",
                "--format=csv,noheader",
            ]
        ),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "numerical_mode": "fp32",
        "online_fp64_rescore": False,
        "semantic_contract_hash": sha256(
            (
                Path(os.environ.get("GPACO_SNAPSHOT", str(ROOT))) / "configs/semantic_contract.yaml"
            ).read_bytes()
        ).hexdigest(),
    }


def evaluate(programs, problem, search, seed, plan):
    if plan.backend == "cpu_python":
        from .backends.cpu_python import evaluate as implementation
    elif plan.backend == "cpu_existing":
        from .backends.cpu import evaluate as implementation
    elif plan.backend == "cuda_existing":
        from .backends.cuda_backend import evaluate as implementation
    else:
        raise ValueError(f"尚未实现的后端：{plan.backend}")
    result = implementation(programs, problem, search, seed, plan)
    if not np.isfinite(result.lengths).all() or np.any(result.lengths <= 0):
        raise RuntimeError("非法求解结果；不允许写成有效性能样本")
    return result


def baseline(problem, search, seed, plan):
    # 使用同一规范 GPU 后端建立共享参考，不能让 CPU/GPU 各自产生不同对照。
    canonical = ExecutionPlan(
        backend="cuda_existing", candidate_lanes=8, active_tasks=3200, generated=True
    )
    identity = config_hash(
        {
            "source": source_hash(),
            "instances": problem.instance_ids,
            "search": asdict(search),
            "seed": seed,
            "backend": canonical.backend,
        }
    )
    target = ROOT / "artifacts/baselines" / (identity + ".npz")
    if target.exists():
        with np.load(target, allow_pickle=False) as record:
            return record["lengths"].copy(), 0.0, True
    if plan.backend.startswith("cpu_"):
        raise FileNotFoundError("CPU 实验需要预生成的只读 ACO 基线；禁止隐式启动 CUDA")
    begin = perf_counter()
    result = evaluate([ProgramSpec.parse("ZERO")], problem, search, seed, canonical)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + f".{os.getpid()}.tmp.npz")
    np.savez(temporary, lengths=result.lengths[0], tours=result.tours[0])
    temporary.replace(target)
    return result.lengths[0], perf_counter() - begin, False


def gap(lengths, reference):
    return np.float32(100) * ((lengths - reference) / reference)


def append_record(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def train(
    output,
    n,
    root_seed,
    search,
    plan,
    *,
    population_size=100,
    generations=50,
    batch_size=32,
    validation_interval=5,
    validation_repeats=3,
    resume=False,
    inputs=None,
    evidence=None,
):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("实验输出必须位于项目内")
    if min(population_size, generations, batch_size, validation_interval, validation_repeats) < 1:
        raise ValueError("训练预算和验证周期必须为正数")
    if not 0 <= root_seed < 2**64:
        raise ValueError("根种子必须在 uint64 范围内")
    if evidence and evidence.get("evidence_tier") == "formal":
        if inputs is None or resume:
            raise ValueError("正式连续计时需要预冻结输入和ACO参考，且不能断点续跑")
    requested = {
        "n": n,
        "root_seed": root_seed,
        "search": asdict(search),
        "plan": asdict(plan),
        "population": population_size,
        "generations": generations,
        "batch_size": batch_size,
        "validation_interval": validation_interval,
        "validation_repeats": validation_repeats,
        "frozen_inputs": None if inputs is None else inputs.identity,
        "evidence": evidence,
    }
    signature = config_hash(requested)
    checkpoint = output / "checkpoint.pkl"
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("输出目录已有实验；必须显式 resume，不覆盖历史结果")
    if resume and not checkpoint.exists():
        raise FileNotFoundError("没有可恢复的 checkpoint，不能覆盖已有输出")
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    previous_elapsed = 0.0
    schedule_rng = np.random.default_rng(np.random.SeedSequence([root_seed, n, 1801]))
    pool_count = len(
        np.load(ROOT / f"Datasets/processed/v1/tsp{n}/train/instance_ids.npy", mmap_mode="r")
    )
    if generations * batch_size > pool_count:
        raise ValueError("固定代数实验需要无重复训练实例；当前训练池不足")
    schedule = schedule_rng.permutation(pool_count)[: generations * batch_size].reshape(
        generations, batch_size
    )
    random.seed(root_seed)
    population = initial_population(population_size)
    champion, champion_score, start_generation = None, float("inf"), 1
    history = []
    resume_count = 0
    if resume and checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["signature"] != signature or saved["source_hash"] != source_hash():
            raise ValueError("恢复配置或源码不一致；必须启动新实验版本")
        population = [parse_tree(s) for s in saved["population"]]
        random.setstate(saved["random_state"])
        champion, champion_score = saved["champion"], saved["champion_score"]
        start_generation, previous_elapsed = saved["generation"] + 1, saved["elapsed_s"]
        history = saved["history"]
        resume_count = saved.get("resume_count", 0) + 1
        if (output / "COMPLETE.json").exists():
            return
        # checkpoint 是唯一提交边界；中断前未提交的日志不进入结果汇总。
    else:
        np.save(output / "schedule.npy", schedule)
        write_json(
            output / "run_manifest.json",
            {
                **metadata(),
                **requested,
                "signature": signature,
                "stage": (evidence or {}).get("evidence_tier", "pilot"),
                **(evidence or {}),
                "schedule_hash": sha256(schedule.tobytes()).hexdigest(),
                "data_manifest_hashes": {
                    split: sha256(
                        (ROOT / f"Datasets/processed/v1/tsp{n}/{split}/manifest.json").read_bytes()
                    ).hexdigest()
                    for split in ("train", "validation")
                },
            },
        )
    if inputs is None:
        val_coords, val_tours, val_ids = load_split(n, "validation")
        val_problem = prepare_problem(val_coords, val_tours, val_ids, search.candidate_size)
    else:
        val_problem = inputs.validation(0)
    reference_lookup = baseline if inputs is None else inputs.baseline
    for generation in range(start_generation, generations + 1):
        generation_start = perf_counter()
        write_json(
            output / "status.json",
            {
                "status": "running",
                "generation": generation,
                "phase": "training",
                "pid": os.getpid(),
                "hostname": platform.node(),
            },
        )
        setup_start = perf_counter()
        if inputs is None:
            coords, tours, ids = load_split(n, "train", schedule[generation - 1])
            problem = prepare_problem(coords, tours, ids, search.candidate_size)
        else:
            problem = inputs.training(generation, schedule[generation - 1])
        data_setup_s = perf_counter() - setup_start
        programs = [ProgramSpec.from_tree(tree) for tree in population]
        seed = namespace_seed(root_seed, "train", generation)
        base_lengths, baseline_s, baseline_hit = reference_lookup(problem, search, seed, plan)
        result = evaluate(programs, problem, search, seed, plan)
        gaps = gap(result.lengths, problem.reference[None, :])
        fitness = np.mean(gaps, axis=1, dtype=np.float32)
        winner = int(np.argmin(fitness))
        base_gap = float(np.mean(gap(base_lengths, problem.reference), dtype=np.float32))
        validation_s = 0.0
        current_val = None
        val_baseline = None
        val_baseline_s = 0.0
        if generation % validation_interval == 0 or generation == generations:
            validation_start = perf_counter()
            write_json(
                output / "status.json",
                {
                    "status": "running",
                    "generation": generation,
                    "phase": "validation",
                    "pid": os.getpid(),
                    "hostname": platform.node(),
                },
            )
            selected = np.argsort(fitness, kind="stable")[: min(5, population_size)]
            candidates = [programs[int(i)] for i in selected]
            if champion is not None:
                candidates.append(ProgramSpec.parse(champion))
            scores = np.zeros(len(candidates), np.float32)
            base_scores = []
            for repeat in range(validation_repeats):
                val_seed = namespace_seed(root_seed, "validation", repeat=repeat)
                if inputs is not None:
                    val_problem = inputs.validation(repeat)
                val_base, cache_cost, _ = reference_lookup(val_problem, search, val_seed, plan)
                val_baseline_s += cache_cost
                base_scores.append(np.mean(gap(val_base, val_problem.reference), dtype=np.float32))
                validation_result = evaluate(
                    candidates,
                    val_problem,
                    search,
                    val_seed,
                    plan,
                )
                scores += np.mean(
                    gap(validation_result.lengths, val_problem.reference[None, :]),
                    axis=1,
                    dtype=np.float32,
                )
            scores /= np.float32(validation_repeats)
            selected_index = min(
                range(len(candidates)),
                key=lambda i: (
                    float(scores[i]),
                    len(candidates[i].instructions),
                    candidates[i].semantic_hash,
                ),
            )
            champion = candidates[selected_index].expression
            champion_score = float(scores[selected_index])
            current_val = champion_score
            val_baseline = float(np.mean(base_scores, dtype=np.float32))
            write_json(
                output / "champion.json",
                {
                    **candidates[selected_index].record(),
                    "validation_gap_percent": champion_score,
                    "validation_baseline_gap_percent": val_baseline,
                    "validation_delta_pp": champion_score - val_baseline,
                    "generation": generation,
                    "selection": "validation_only",
                },
            )
            validation_s = perf_counter() - validation_start
        if generation in (1, 25, 50) or generation == generations:
            write_json(
                output / f"cohorts/generation-{generation:03d}.json", [p.record() for p in programs]
            )
        record = {
            "generation": generation,
            "train_best_gap_percent": float(fitness[winner]),
            "train_median_gap_percent": float(np.median(fitness)),
            "train_baseline_gap_percent": base_gap,
            "train_delta_pp": float(fitness[winner]) - base_gap,
            "validation_champion_gap_percent": current_val,
            "validation_baseline_gap_percent": val_baseline,
            "validation_delta_pp": None if current_val is None else current_val - val_baseline,
            "data_setup_s": data_setup_s,
            "baseline_setup_s": baseline_s,
            "baseline_cache_hit": baseline_hit,
            "validation_s": validation_s,
            "validation_baseline_setup_s": val_baseline_s,
            **result.timings,
            "fallback_events": int(result.diagnostics[..., 0].sum()),
            "unique_programs": len({p.semantic_hash for p in programs}),
            "unique_structures": len({p.structural_hash for p in programs}),
            "requested_tasks": len(programs) * problem.size,
            "fitness_cache_hits": 0,
            "best_expression": programs[winner].expression,
        }
        if result.local_search_diagnostics is not None:
            record["local_search_counts"] = dict(
                zip(
                    (
                        "two_opt_moves",
                        "three_opt_moves",
                        "logical_candidate_checks",
                        "two_opt_passes",
                    ),
                    map(int, result.local_search_diagnostics.sum(axis=(0, 1))),
                    strict=True,
                )
            )
        if generation < generations:
            population = next_population(population, fitness)
        record["generation_wall_s"] = perf_counter() - generation_start
        record["elapsed_s"] = previous_elapsed + perf_counter() - started
        history.append(record)
        checkpoint_start = perf_counter()
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "signature": signature,
                    "source_hash": source_hash(),
                    "generation": generation,
                    "population": [str(tree) for tree in population],
                    "random_state": random.getstate(),
                    "champion": champion,
                    "champion_score": champion_score,
                    "elapsed_s": record["elapsed_s"],
                    "history": history,
                    "resume_count": resume_count,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(checkpoint)
        record["checkpoint_s"] = perf_counter() - checkpoint_start
        record["generation_wall_s"] = perf_counter() - generation_start
        record["elapsed_s"] = previous_elapsed + perf_counter() - started
        write_json(output / "history.json", history)
        append_record(output / "events.jsonl", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    elapsed = previous_elapsed + perf_counter() - started
    write_json(
        output / "COMPLETE.json",
        {
            "status": "completed",
            "generations": generations,
            "training_wall_s": elapsed,
            "champion": champion,
            "validation_gap_percent": champion_score,
            "tests_run": False,
            "source_hash": source_hash(),
            "resume_count": resume_count,
            "uninterrupted_timing_sample": resume_count == 0,
        },
    )
    write_json(output / "status.json", {"status": "completed", "generation": generations})


def benchmark(
    output,
    n,
    root_seed,
    search,
    plan,
    population_size=100,
    batch_size=32,
    blocks=3,
    cohort=None,
    split="tuning",
):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("输出不在项目内")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("benchmark 输出已存在，必须选择新目录")
    if min(population_size, batch_size, blocks) < 1:
        raise ValueError("性能预算必须为正数")
    if split not in ("tuning", "holdout"):
        raise ValueError("性能实验只能读取 tuning 或 holdout")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(root_seed)
    if cohort:
        records = json.loads(Path(cohort).read_text())
        if len(records) < population_size:
            raise ValueError("cohort 不足，禁止靠复制个体填充真实种群")
        programs = [ProgramSpec.parse(row["expression"]) for row in records[:population_size]]
    else:
        programs = [ProgramSpec.from_tree(tree) for tree in initial_population(population_size)]
    if batch_size > len(load_split(n, split)[0]):
        raise ValueError("请求的批次超过独立实例集合大小")
    write_json(output / "cohort.json", [p.record() for p in programs])
    write_json(
        output / "run_manifest.json",
        {
            **metadata(),
            "n": n,
            "root_seed": root_seed,
            "search": asdict(search),
            "plan": asdict(plan),
            "population": population_size,
            "batch_size": batch_size,
            "blocks": blocks,
            "cohort_source": str(cohort) if cohort else "initial_population_diagnostic",
            "split": split,
        },
    )
    for block in range(blocks):
        start = perf_counter()
        coords, tours, ids = load_split(n, split, np.arange(batch_size))
        problem = prepare_problem(coords, tours, ids, search.candidate_size)
        setup_s = perf_counter() - start
        result = evaluate(
            programs, problem, search, namespace_seed(root_seed, "benchmark", block), plan
        )
        timed_wall = perf_counter() - start
        validate_tours(result.tours, problem.n)
        record = {
            "block": block,
            "data_setup_s": setup_s,
            "wall_with_setup_s": timed_wall,
            "mean_gap_percent": float(
                np.mean(gap(result.lengths, problem.reference[None]), dtype=np.float32)
            ),
            "tour_valid": True,
            **result.timings,
        }
        append_record(output / "performance.jsonl", record)
        np.savez(
            output / f"result-{block:02d}.npz",
            lengths=result.lengths,
            tours=result.tours,
            diagnostics=result.diagnostics,
        )
        print(json.dumps(record), flush=True)
    write_json(output / "COMPLETE.json", {"status": "completed", "blocks": blocks})
