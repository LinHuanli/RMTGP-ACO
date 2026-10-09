"""异构 GPU 实验的不可变输入和只读 A5000 参考库，不接触标准测试集。"""

import json
import os
import random
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .backends.cpu import initial_parameters
from .config import ExecutionPlan, SearchConfig, config_hash
from .data import (
    ROOT,
    FrozenInitialization,
    ProblemSpec,
    load_split,
    prepare_problem,
    validate_tours,
    write_json,
)
from .evolution import initial_population
from .experiment import evaluate, metadata, namespace_seed, source_hash
from .language import ProgramSpec

ARRAYS = (
    "coords",
    "distances",
    "heuristic",
    "log_heuristic",
    "nearest",
    "full_nn_rank",
    "reference",
    "instance_keys",
)
DEFAULT_PLAN = ExecutionPlan(candidate_lanes=8, active_tasks=3200, generated=True)


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_directory(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("实验目录必须位于项目内且不能是项目根目录")
    return path


def training_schedule(n, seed, generations, batch_size):
    pool = np.load(ROOT / f"Datasets/processed/v1/tsp{n}/train/instance_ids.npy", mmap_mode="r")
    if generations * batch_size > len(pool):
        raise ValueError("训练计划超过无重复训练池")
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 1801]))
    return rng.permutation(len(pool))[: generations * batch_size].reshape(generations, batch_size)


def scenario_key(phase, seed=0, index=0):
    return f"{phase}-{seed}-{index}"


def save_problem(directory, problem):
    directory.mkdir(parents=True, exist_ok=False)
    for field in ARRAYS:
        np.save(directory / f"{field}.npy", getattr(problem, field), allow_pickle=False)
    write_json(directory / "ids.json", problem.instance_ids)


def prepare_scale(directory, n, config, writer_uuid):
    """唯一 A5000 写入者生成几何、初始化和基线，READY 是原子提交边界。"""
    import cupy as cp

    directory = safe_directory(directory)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != writer_uuid:
        raise RuntimeError("参考库只能由配置指定的物理 A5000 写入")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode()
    if name != "NVIDIA RTX A5000":
        raise RuntimeError(f"规范参考写入者必须是 RTX A5000，实际为 {name}")
    if (directory / "READY.json").exists():
        store = FrozenStore(directory)
        if store.manifest["config_hash"] != config_hash(config):
            raise ValueError("已有冻结库不属于本实验配置")
        return
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    search = SearchConfig(**config["search"])
    manifest = {
        "n": n,
        "source_hash": source_hash(),
        "config_hash": config_hash(config),
        "search": asdict(search),
        "canonical_plan": asdict(DEFAULT_PLAN),
        "writer_uuid": writer_uuid,
        "writer": metadata(),
        "geometry": {},
        "scenarios": {},
        "schedules": {},
        "cohorts": {},
        "tests_opened": False,
    }

    def geometry(key, split, indices=None):
        problem = prepare_problem(*load_split(n, split, indices), search.candidate_size)
        save_problem(directory / "geometry" / key, problem)
        manifest["geometry"][key] = {
            "path": f"geometry/{key}",
            "instances": problem.instance_ids,
            "source_split": split,
            "source_manifest_hash": file_hash(
                ROOT / f"Datasets/processed/v1/tsp{n}/{split}/manifest.json"
            ),
        }
        return problem

    def scenario(key, geometry_key, problem, seed, reference=False):
        target = directory / "scenarios" / key
        target.mkdir(parents=True)
        initial = initial_parameters(
            problem.distances,
            problem.instance_keys,
            np.uint64(seed),
            search.variant_id,
            np.float32(search.rho),
        )
        for field, value in zip(("tau0", "low", "high"), initial, strict=True):
            np.save(target / f"{field}.npy", value, allow_pickle=False)
        row = {"geometry": geometry_key, "seed": seed, "baseline": reference}
        if reference:
            frozen = FrozenInitialization(
                seed, search.variant, search.rho, problem.instance_ids, initial, key
            )
            result = evaluate(
                [ProgramSpec.parse("ZERO")],
                replace(problem, initialization=frozen),
                search,
                seed,
                DEFAULT_PLAN,
            )
            validate_tours(result.tours, n)
            np.savez(target / "baseline.npz", lengths=result.lengths[0], tours=result.tours[0])
            row["baseline_timings"] = result.timings
        manifest["scenarios"][key] = row
        print(json.dumps({"prepare_n": n, "scenario": key, "reference": reference}), flush=True)

    for phase, cohort_seed in (("tuning", 1001), ("holdout", 1002)):
        path = ROOT / f"artifacts/pilot-v1/as-tsp100-seed{cohort_seed}/cohorts/generation-001.json"
        if not path.exists():
            raise FileNotFoundError(f"必须使用已记录的真实初始种群：{path}")
        rows = json.loads(path.read_text())[: config["population"]]
        if len(rows) != config["population"]:
            raise ValueError("冻结种群不足；禁止复制程序补齐")
        # 验证短训练与所选 cohort 的初始种群确实采用相同的 DEAP 生成协议。
        state = random.getstate()
        random.seed(cohort_seed)
        expected = [ProgramSpec.from_tree(t).semantic_hash for t in initial_population(len(rows))]
        random.setstate(state)
        if expected != [ProgramSpec.parse(r["expression"]).semantic_hash for r in rows]:
            raise ValueError("已有种群与当前初始种群协议不一致")
        write_json(directory / f"cohorts/{phase}.json", rows)
        manifest["cohorts"][phase] = {"source": str(path), "sha256": file_hash(path)}
        data = geometry(phase, phase)
        count = 1 if phase == "tuning" else config["paired_blocks"]
        for block in range(count):
            scenario(
                scenario_key(phase, index=block),
                phase,
                data,
                namespace_seed(9001, f"hardware-{phase}", block),
            )
    data = geometry("validation", "validation")
    for seed in config["seeds"]:
        for repeat in range(config["validation_repeats"]):
            scenario(
                scenario_key("validation", seed, repeat),
                "validation",
                data,
                namespace_seed(seed, "validation", repeat=repeat),
                reference=True,
            )
        schedule = training_schedule(n, seed, config["generations"], config["batch"])
        manifest["schedules"][str(seed)] = schedule.tolist()
        for generation, indices in enumerate(schedule, 1):
            key = scenario_key("train", seed, generation)
            problem = geometry(key, "train", indices)
            scenario(key, key, problem, namespace_seed(seed, "train", generation), reference=True)
    cp.get_default_memory_pool().free_all_blocks()
    manifest["preparation_wall_s"] = perf_counter() - started
    manifest["files"] = {
        str(path.relative_to(directory)): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    write_json(directory / "manifest.json", manifest)
    write_json(
        directory / "READY.json",
        {
            "manifest_sha256": file_hash(directory / "manifest.json"),
            "n": n,
            "source_hash": source_hash(),
            "writer_uuid": writer_uuid,
        },
    )


class FrozenStore:
    """消费者仅有读取接口；每个文件首次打开核验 SHA256，随后 mmap 复用。"""

    def __init__(self, directory):
        self.directory = safe_directory(directory)
        ready = json.loads((self.directory / "READY.json").read_text())
        if file_hash(self.directory / "manifest.json") != ready["manifest_sha256"]:
            raise ValueError("冻结输入 manifest 校验失败")
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["source_hash"] != source_hash():
            raise ValueError("冻结输入属于其他源码版本")
        self.identity = ready["manifest_sha256"]
        self.verified = set()
        self.geometries, self.problems = {}, {}

    def path(self, relative):
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("冻结输入路径越界")
        if relative not in self.verified:
            if file_hash(path) != self.manifest["files"][relative]:
                raise ValueError(f"冻结输入 SHA256 不一致：{relative}")
            self.verified.add(relative)
        return path

    def array(self, relative):
        return np.load(self.path(relative), mmap_mode="r", allow_pickle=False)

    def problem(self, key, count=None):
        cache_key = key, count
        if cache_key in self.problems:
            return self.problems[cache_key]
        row = self.manifest["scenarios"][key]
        geometry = row["geometry"]
        if geometry not in self.geometries:
            prefix = self.manifest["geometry"][geometry]["path"]
            arrays = {field: self.array(f"{prefix}/{field}.npy") for field in ARRAYS}
            ids = tuple(json.loads(self.path(f"{prefix}/ids.json").read_text()))
            self.geometries[geometry] = ProblemSpec(**arrays, instance_ids=ids)
        original = self.geometries[geometry]
        count = original.size if count is None else count
        if not 1 <= count <= original.size:
            raise ValueError("请求超过冻结的独立实例数")
        problem = ProblemSpec(
            **{field: getattr(original, field)[:count] for field in ARRAYS},
            instance_ids=original.instance_ids[:count],
        )
        values = tuple(
            self.array(f"scenarios/{key}/{v}.npy")[:count] for v in ("tau0", "low", "high")
        )
        problem.initialization = FrozenInitialization(
            row["seed"],
            self.manifest["search"]["variant"],
            self.manifest["search"]["rho"],
            problem.instance_ids,
            values,
            key,
            self.identity,
        )
        self.problems[cache_key] = problem
        return problem

    def cohort(self, phase):
        if phase not in ("tuning", "holdout"):
            raise ValueError("性能阶段只能是 tuning/holdout")
        return [
            ProgramSpec.parse(row["expression"])
            for row in json.loads(self.path(f"cohorts/{phase}.json").read_text())
        ]

    def baseline(self, problem, search, seed, plan):
        started = perf_counter()
        frozen = problem.initialization
        if frozen is None or frozen.seed != seed or asdict(search) != self.manifest["search"]:
            raise ValueError("规范参考缓存身份不一致")
        row = self.manifest["scenarios"][frozen.scenario]
        expected_ids = tuple(self.manifest["geometry"][row["geometry"]]["instances"])
        if (
            frozen.input_manifest_sha256 != self.identity
            or frozen.instance_ids != problem.instance_ids
            or problem.instance_ids != expected_ids
            or row["seed"] != seed
        ):
            raise ValueError("参考库或实例顺序身份不一致")
        if not row["baseline"]:
            raise ValueError("此场景没有预生成参考；禁止在消费者 GPU 上补写")
        with np.load(
            self.path(f"scenarios/{frozen.scenario}/baseline.npz"), allow_pickle=False
        ) as data:
            lengths = data["lengths"].copy()
        if lengths.shape != (problem.size,):
            raise ValueError("参考实例数量不一致")
        return lengths, perf_counter() - started, True


class TrainingInputs:
    def __init__(self, store, seed):
        self.store, self.seed = store, seed
        self.identity = {"manifest_sha256": store.identity, "root_seed": seed}

    def training(self, generation, indices):
        expected = self.store.manifest["schedules"][str(self.seed)][generation - 1]
        if not np.array_equal(indices, expected):
            raise ValueError("训练日程与预生成参考不一致")
        return self.store.problem(scenario_key("train", self.seed, generation))

    def validation(self, repeat):
        return self.store.problem(scenario_key("validation", self.seed, repeat))

    def baseline(self, *args):
        return self.store.baseline(*args)
