"""历史路径强化的选择、定向重放、机制分析、可视化和完整性核验入口。"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import os
import socket
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .common import ROOT, atomic_json, atomic_npz, digest, file_hash, now, read_json
from .diagnostics import atomic_diagnostic_npz
from .evaluate import program_entries
from .explanation_analysis import SAMPLE_FIELDS, sample_metrics
from .explanation_campaign import conditions
from .prepare import batch

SOURCE = ROOT / "control_experiments/mmas_ls/artifacts/mechanism-explanation-v2"
OUTPUT = ROOT / "control_experiments/mmas_ls/artifacts/mechanism-visualization-v1"
REPORT = ROOT / "control_experiments/mmas_ls/reports/mechanism-explanation-v2"
CHAMPION = 81002
SAMPLE_ITERATIONS = (1, *range(25, 5001, 25))
STORAGE_CAP_BYTES = 3 * 1024**3
REGIMES = {
    "all_current_paths": ("强化本轮全部 32 条路径", "mmas_all_current"),
    "iteration_best": ("仅强化本轮最优路径", "mmas_current_best"),
    "fixed_history_calendar": ("按固定日程强化历史全局最优路径", "mmas_history_calendar"),
    "native_mmas": ("MMAS 原生日程", "mmas_native"),
}
BEHAVIORS = {
    "baseline": "不使用 GP",
    "gp_mmas_81002": "使用 MMAS 训练的双树 GP 表达式（种子 81002）",
}


def _completed_job(source: Path, task_id: str) -> Path:
    job = source / "jobs" / task_id
    status = read_json(job / "status.json", {})
    if status.get("status") != "completed":
        raise ValueError(f"任务未完整完成：{task_id}")
    return job


def _baseline_sample_features(job: Path, instances: int = 32) -> dict[int, list[np.ndarray]]:
    """仅读取 baseline 逻辑求解；返回值不包含任何 GP 结果字段。"""
    index = read_json(job / "diagnostics/index.json")
    values = {i: [] for i in range(instances)}
    for name, record in index["files"].items():
        meta = record["metadata"]
        if meta.get("kind") != "sample":
            continue
        flat = np.asarray(meta["flat_indices"], dtype=int)
        selected = np.flatnonzero(flat < instances)
        if not len(selected):
            continue
        path = job / "diagnostics" / name
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"诊断文件哈希错误：{path}")
        with np.load(path, allow_pickle=False) as archive:
            data = {
                key: archive[key][selected]
                for key in (
                    "source_valid",
                    "source_tours",
                    "source_info",
                    "post_tours",
                    "pre_tours",
                    "colony_lengths",
                    "best_tours",
                    "deposit",
                    "ph",
                    "tr",
                    "context",
                )
            }
        metrics = sample_metrics(data)
        keep = [
            SAMPLE_FIELDS.index("history_source_is_different_fraction"),
            SAMPLE_FIELDS.index("post_unique_tours"),
            SAMPLE_FIELDS.index("normalized_choice_entropy"),
        ]
        for row, task_index in enumerate(flat[selected]):
            values[int(task_index)].append(metrics[row, keep])
    if any(len(rows) != len(SAMPLE_ITERATIONS) for rows in values.values()):
        raise ValueError("baseline 采样点不完整")
    return values


def select(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    """按预处理 baseline 特征选择中位代表实例和求解种子。"""
    source, output = Path(source), Path(output)
    rows = []
    feature_names = (
        "最终 reference gap（%）",
        "平均来源年龄（轮）",
        "历史来源不同于本轮最优的比例",
        "2-opt 后不同路径数",
        "构造选择归一化熵",
    )
    for replicate in range(5):
        task_id = f"source-development-mmas_native-seed{replicate}"
        job = _completed_job(source, task_id)
        with np.load(job / "raw.npz", allow_pickle=False) as archive:
            final_gap = archive["gap"][0].astype(float)
            source_age = archive["trace"][0, :, :, 14].astype(float).mean(axis=-1)
        samples = _baseline_sample_features(job)
        for instance in range(32):
            process = np.stack(samples[instance]).mean(axis=0)
            rows.append(
                {
                    "instance": instance,
                    "replicate": replicate,
                    "values": np.array([final_gap[instance], source_age[instance], *process]),
                }
            )
    cube = (
        np.stack([row["values"] for row in rows])
        .reshape(5, 32, len(feature_names))
        .transpose(1, 0, 2)
    )
    instance_values = cube.mean(axis=1)
    center = np.median(instance_values, axis=0)
    scale = np.median(np.abs(instance_values - center), axis=0)
    scale = np.where(scale > 1e-12, scale, np.std(instance_values, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    distances = np.linalg.norm((instance_values - center) / scale, axis=1)
    selected_instance = int(np.lexsort((np.arange(32), distances))[0])
    seed_center = cube[selected_instance].mean(axis=0)
    seed_distances = np.linalg.norm((cube[selected_instance] - seed_center) / scale, axis=1)
    selected_replicate = int(np.lexsort((np.arange(5), seed_distances))[0])
    manifest = read_json(source / "manifests/diagnosis_dev.json")
    record = manifest["records"][selected_instance]
    selection = {
        "status": "selected",
        "created_at": now(),
        "source": str(source),
        "source_manifest_sha256": file_hash(source / "manifests/diagnosis_dev.json"),
        "instance": selected_instance,
        "replicate": selected_replicate,
        "aco_seed": read_json(
            source
            / "jobs"
            / f"source-development-mmas_native-seed{selected_replicate}"
            / "manifest.json"
        )["seed"],
        "instance_record": record,
        "selection_features": feature_names,
        "selected_instance_values": instance_values[selected_instance],
        "selected_seed_values": cube[selected_instance, selected_replicate],
        "componentwise_instance_median": center,
        "robust_feature_scale": scale,
        "instance_distance_to_median": float(distances[selected_instance]),
        "seed_distance_to_instance_mean": float(seed_distances[selected_replicate]),
        "all_instance_distances": distances,
        "all_seed_distances_for_selected_instance": seed_distances,
        "selection_rule": (
            "仅使用不使用 GP 的预处理特征；先选离五维分量中位数最近的实例，再选离该实例五种子均值最近的种子；"
            "平局取较小编号。未读取 GP 质量或处理效应。"
        ),
        "forbidden_selection_fields": ["GP gap", "GP 增益", "处理效应", "冠军间差异"],
    }
    atomic_json(output / "selection.json", selection)
    atomic_json(
        output / "protocol.json",
        {
            "cohort": "mechanism-visualization-v1",
            "source_cohort": str(source),
            "confirmation_cohort_resumed": False,
            "champion": CHAMPION,
            "champion_reason": "三个 MMAS 冠军中唯一构造树与信息素更新树均非零；仅用于示例，汇总分析保留三个冠军。",
            "regimes": {key: label for key, (label, _) in REGIMES.items()},
            "behaviors": BEHAVIORS,
            "iterations": 5000,
            "ants": 32,
            "sample_iterations": SAMPLE_ITERATIONS,
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "evidence_scope": "动画是说明性个案；统计结论来自完整开发对照和同状态干预。",
            "created_at": now(),
        },
    )
    return output / "selection.json"


def _replay_size(output: Path) -> int:
    replay = Path(output) / "replay"
    return (
        sum(path.stat().st_size for path in replay.rglob("*") if path.is_file())
        if replay.exists()
        else 0
    )


class ReplayRecorder:
    """每 25 轮保存一个完整单求解帧；记录四阶段信息素，不改变求解状态。"""

    def __init__(self, directory: Path, identity: dict, storage_root: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.storage_root = Path(storage_root)
        self.index_path = self.directory / "index.json"
        self.index = read_json(
            self.index_path,
            {
                "version": 1,
                "identity": digest(identity),
                "files": {},
                "completed": False,
            },
        )
        if self.index["identity"] != digest(identity):
            raise ValueError("重放配置变化，必须新建可视化批次")

    def __call__(self, phase: str, iteration: int, state: dict) -> None:
        if phase != "iteration_end" or iteration not in SAMPLE_ITERATIONS:
            return
        import cupy as cp

        tau = cp.concatenate((state["audit_tau"], state["pheromone_workspace"][:, None]), axis=1)
        slot = (iteration - 1) % 100
        capacity = int(state["audit_source_capacity"])
        arrays = {
            "tau": cp.asnumpy(tau[0]),
            "source_tours": cp.asnumpy(state["audit_sources"][0, :capacity]),
            "source_info": cp.asnumpy(state["audit_source_info"][0, slot, :capacity]),
            "deposit": cp.asnumpy(state["deposit_workspace"][0, :capacity]),
            "ph": cp.asnumpy(state["audit_ph"][0, :capacity]),
            "tr": cp.asnumpy(state["audit_tr"][0]),
            "context": cp.asnumpy(state["audit_context"][0]),
            "pre_tours": cp.asnumpy(state["pre_tour_workspace"][0]),
            "post_tours": cp.asnumpy(state["tour_workspace"][0]),
            "colony_lengths": cp.asnumpy(state["length_workspace"][0]),
            "best_tour": cp.asnumpy(state["best_tours"][0]),
            "best_length": cp.asnumpy(state["global_best_lengths"][0:1]),
            "trace": cp.asnumpy(state["mechanism_trace"][0, iteration - 1]),
            "stagnation": cp.asnumpy(state["stagnation"][0:1]),
            "tau_min": cp.asnumpy(state["task_tau_min"][0:1]),
            "tau_max": cp.asnumpy(state["task_tau_max"][0:1]),
        }
        name = f"sample-{iteration:05d}.npz"
        content = digest(
            {key: [str(value.dtype), list(value.shape)] for key, value in arrays.items()}
        )
        previous = self.index["files"].get(name)
        if previous:
            if file_hash(self.directory / name) != previous["sha256"]:
                raise ValueError(f"已有重放帧损坏：{name}")
            return
        before = _replay_size(self.storage_root)
        raw_size = sum(value.nbytes for value in arrays.values())
        if before + raw_size > STORAGE_CAP_BYTES:
            raise OSError("可视化重放达到 3 GiB 硬上限；不继续写入")
        atomic_diagnostic_npz(self.directory / name, arrays)
        self.index["files"][name] = {
            "iteration": iteration,
            "sha256": file_hash(self.directory / name),
            "content_schema": content,
            "uncompressed_bytes": raw_size,
            "compressed_bytes": (self.directory / name).stat().st_size,
        }
        atomic_json(self.index_path, self.index)

    def finish(self) -> None:
        observed = tuple(row["iteration"] for row in self.index["files"].values())
        if tuple(sorted(observed)) != SAMPLE_ITERATIONS:
            raise ValueError("重放帧不完整")
        self.index.update(completed=True, completed_at=now())
        atomic_json(self.index_path, self.index)


def _expected_development_raw(
    source: Path, regime: str, replicate: int, behavior: str
) -> tuple[np.ndarray, np.ndarray]:
    task_id = f"source-development-{REGIMES[regime][1]}-seed{replicate}"
    job = _completed_job(source, task_id)
    with np.load(job / "raw.npz", allow_pickle=False) as archive:
        model_ids = archive["model_ids"].astype(str)
        model = "baseline" if behavior == "baseline" else f"mmas-{CHAMPION}"
        matches = np.flatnonzero(model_ids == model)
        if len(matches) != 1:
            raise ValueError(f"开发任务中模型标识不唯一：{task_id}/{model}")
        return archive["anytime"][matches[0]].copy(), archive["tour"][matches[0]].copy()


def replay_one(source: Path, output: Path, regime: str, behavior: str) -> Path:
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    from rmtgp_aco.mechanisms import InstrumentationConfig, MechanismConfig, SolverControl

    selection = read_json(output / "selection.json")
    target = output / "replay" / regime / behavior
    status = read_json(target / "status.json", {})
    if status.get("status") == "completed":
        for name, expected in status["files"].items():
            if file_hash(target / name) != expected:
                raise ValueError(f"已完成重放损坏：{target / name}")
        return target
    problem = batch("diagnosis_dev", [selection["instance"]], source)
    from .common import experiment

    aco, runtime = experiment("mmas", iterations=5000, precision="fp32_fast")
    runtime = replace(runtime, gpu_devices=(0,), gpu_task_chunk_size=0)
    mechanism = conditions("centered_fp32")[REGIMES[regime][1]]
    entry = next(item for item in program_entries("mmas") if item["seed"] == CHAMPION)
    program = (None, None) if behavior == "baseline" else entry["program"]
    instrumentation = InstrumentationConfig(
        level="heavy",
        aggregate_every=25,
        snapshot_iterations=(),
        profile="mechanism_v3",
        schema_version=3,
        commit_every=100,
        checkpoint_every=500,
    )
    identity = {
        "regime": regime,
        "behavior": behavior,
        "instance": selection["instance"],
        "coordinate_hash": selection["instance_record"]["coordinate_hash"],
        "replicate": selection["replicate"],
        "seed": selection["aco_seed"],
        "champion": None if behavior == "baseline" else CHAMPION,
        "champion_file_hash": None if behavior == "baseline" else entry["file_hash"],
        "mechanism": asdict(mechanism),
        "aco": {
            "variant": "mmas",
            "ants": 32,
            "iterations": 5000,
            "candidate_size": 20,
            "local_search": "two_opt",
        },
    }
    recorder = ReplayRecorder(target / "samples", identity, output)
    atomic_json(
        target / "status.json",
        {
            "status": "running",
            "started_at": now(),
            "host": socket.gethostname(),
            "identity": identity,
        },
    )
    started = time.perf_counter()
    control = SolverControl(
        mechanism=MechanismConfig(**asdict(mechanism)),
        instrumentation=instrumentation,
        observer=recorder,
        collected=[],
        stop_iteration=5000,
    )
    result = solve_population_cuda_anytime(
        problem,
        aco,
        [program],
        seed=selection["aco_seed"],
        runtime=runtime,
        control=control,
    )
    recorder.finish()
    anytime = result.anytime_best.numpy()[0, 0]
    tour = result.best_tour.numpy()[0, 0]
    atomic_npz(
        target / "result.npz",
        anytime=anytime,
        tour=tour,
        length=result.best_length.numpy()[0, 0:1],
        reference=problem.reference_length.numpy(),
    )
    expected_anytime, expected_tours = _expected_development_raw(
        source,
        regime,
        selection["replicate"],
        behavior,
    )
    np.testing.assert_array_equal(anytime, expected_anytime[selection["instance"]])
    np.testing.assert_array_equal(tour, expected_tours[selection["instance"]])
    manifest = {
        "status": "completed",
        "identity": identity,
        "wall_seconds": time.perf_counter() - started,
        "sample_count": len(SAMPLE_ITERATIONS),
        "same_counter_seed": True,
        "development_path_and_anytime_exact_match": True,
        "completed_at": now(),
        "storage_bytes_after": _replay_size(output),
    }
    atomic_json(target / "manifest.json", manifest)
    files = {
        "result.npz": file_hash(target / "result.npz"),
        "manifest.json": file_hash(target / "manifest.json"),
        "samples/index.json": file_hash(target / "samples/index.json"),
    }
    atomic_json(
        target / "status.json", {"status": "completed", "completed_at": now(), "files": files}
    )
    return target


def replay(source: Path = SOURCE, output: Path = OUTPUT, device: str = "0") -> Path:
    """顺序执行八个单求解，避免同卡并发改变计时或显存压力。"""
    source, output = Path(source), Path(output)
    if not (output / "selection.json").exists():
        select(source, output)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    for regime in REGIMES:
        for behavior in BEHAVIORS:
            print(f"重放 {REGIMES[regime][0]} / {BEHAVIORS[behavior]}", flush=True)
            replay_one(source, output, regime, behavior)
    atomic_json(
        output / "replay/summary.json",
        {
            "status": "completed",
            "runs": len(REGIMES) * len(BEHAVIORS),
            "bytes": _replay_size(output),
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "completed_at": now(),
        },
    )
    return output / "replay/summary.json"


def analyze(source: Path = SOURCE, output: Path = OUTPUT, report: Path = REPORT) -> Path:
    from .visualization_analysis import analyze as run

    return run(Path(source), Path(output), Path(report))


def render(source: Path = SOURCE, output: Path = OUTPUT, report: Path = REPORT) -> Path:
    from .visualization_render import render as run

    return run(Path(source), Path(output), Path(report))


def verify(source: Path = SOURCE, output: Path = OUTPUT, report: Path = REPORT) -> Path:
    from .visualization_analysis import verify as run

    return run(Path(source), Path(output), Path(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("select", "replay", "analyze", "render", "verify", "all")
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--device", default="0", help="物理编号或 UUID；进程内仅暴露这一张卡")
    args = parser.parse_args()
    if args.action in ("select", "all"):
        print(select(args.source, args.output), flush=True)
    if args.action in ("replay", "all"):
        print(replay(args.source, args.output, args.device), flush=True)
    if args.action in ("analyze", "all"):
        print(analyze(args.source, args.output, args.report_dir), flush=True)
    if args.action in ("render", "all"):
        print(render(args.source, args.output, args.report_dir), flush=True)
    if args.action in ("verify", "all"):
        print(verify(args.source, args.output, args.report_dir), flush=True)


if __name__ == "__main__":
    main()
