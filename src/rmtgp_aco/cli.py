"""RMTGP-ACO 可复现训练、评测、数据审计与统计命令行。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

from .artifacts import (
    finalise_run_artifacts,
    initialise_run_artifacts,
    resume_run_artifacts,
)
from .config import (
    ExecutionBackend,
    PheromoneIntegration,
    TransitionIntegration,
)
from .evaluation import (
    compile_champion,
    evaluate_batches,
    load_champion,
    read_records,
    write_records,
)
from .manifest import (
    build_manifest,
    load_manifest,
    sampled_split_leakage,
    verify_manifest,
    write_manifest,
)
from .runtime import configure_runtime
from .sampling import (
    ScaleStratifiedSampler,
    fixed_cases_from_pools,
    iter_problem_batches,
    pools_from_paths,
)
from .spec import load_run_spec
from .stats import (
    friedman_test,
    hierarchical_bootstrap_delta,
    paired_wilcoxon_holm,
    summarize_quality,
    write_statistical_report,
)
from .training import train


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _apply_runtime_overrides(spec, args: argparse.Namespace):
    """为独立 run 覆盖 seed/process 数，不修改冻结的 YAML 模板。"""

    experiment = spec.experiment
    seed = getattr(args, "root_seed", None)
    if seed is not None:
        experiment = replace(experiment, root_seed=seed)
    processes = getattr(args, "processes", None)
    if processes is not None:
        experiment = replace(
            experiment,
            runtime=replace(experiment.runtime, processes=processes),
        )
    backend = getattr(args, "backend", None)
    if backend is not None:
        experiment = replace(
            experiment,
            runtime=replace(
                experiment.runtime,
                aco_backend=ExecutionBackend(backend),
            ),
        )
    method = getattr(args, "method_profile", None)
    if method and method != "rmtgp":
        gp = experiment.gp
        aco = experiment.aco
        if method == "tr-rgp":
            gp = replace(
                gp,
                train_transition=True,
                train_pheromone=False,
                transition_profile="main",
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.RESIDUAL,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        elif method == "ph-rgp":
            gp = replace(
                gp,
                train_transition=False,
                train_pheromone=True,
                transition_profile="main",
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.RESIDUAL,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        elif method == "matched-replace":
            gp = replace(
                gp,
                train_transition=True,
                train_pheromone=False,
                transition_profile="main",
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.REPLACEMENT,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        elif method == "legacy":
            gp = replace(
                gp,
                train_transition=True,
                train_pheromone=False,
                transition_profile="legacy",
                transition_terminals=None,
            )
            aco = replace(
                aco,
                transition_integration=TransitionIntegration.REPLACEMENT,
                pheromone_integration=PheromoneIntegration.BUDGET_RESIDUAL,
            )
        experiment = replace(
            experiment,
            experiment_id=f"{experiment.experiment_id}-{method}",
            gp=gp,
            aco=aco,
        )
    return replace(spec, experiment=experiment)


def _preflight_manifest(
    spec,
    manifest_path: str | Path,
    required_paths: list[tuple[Path, str]],
) -> None:
    """在昂贵运行前确认完整 manifest 与所需 split 文件一致。"""

    manifest = load_manifest(manifest_path)
    if manifest.hash_mode != "sha256-full":
        raise ValueError("正式运行要求 hash_mode=sha256-full 的数据 manifest")
    errors = verify_manifest(
        manifest,
        root=spec.data.root,
        verify_hashes=False,
        validate_first_record=True,
    )
    if errors:
        raise ValueError("数据 manifest 预检失败：" + "；".join(errors[:5]))
    declared = {record.path: record.split for record in manifest.files}
    missing: list[str] = []
    wrong_split: list[str] = []
    for path, expected_split in required_paths:
        try:
            relative = path.resolve().relative_to(spec.data.root).as_posix()
        except ValueError:
            missing.append(path.as_posix())
            continue
        if relative not in declared:
            missing.append(relative)
        elif declared[relative] != expected_split:
            wrong_split.append(
                f"{relative}: manifest={declared[relative]}, config={expected_split}"
            )
    if missing:
        raise ValueError(f"配置引用了 manifest 外的数据文件: {missing[:5]}")
    if wrong_split:
        raise ValueError(f"配置跨 split 使用数据: {wrong_split[:5]}")


def _command_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(
        args.root,
        full_hashes=args.full,
        workers=args.workers,
    )
    target = write_manifest(manifest, args.output)
    print(f"已写入 {target}：{len(manifest.files)} 个文件，模式={manifest.hash_mode}")
    return 0


def _command_verify_data(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    errors = verify_manifest(
        manifest,
        root=args.root,
        verify_hashes=not args.skip_hashes,
        validate_first_record=True,
    )
    duplicates = sampled_split_leakage(
        manifest,
        records_per_file=args.leakage_samples,
        root=args.root,
    )
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    for coordinate_hash, first, second in duplicates:
        print(
            f"ERROR: 跨 split 重复 {coordinate_hash}: {first} <-> {second}",
            file=sys.stderr,
        )
    if errors or duplicates:
        return 1
    print(
        f"数据校验通过：{len(manifest.files)} 个文件；"
        f"每文件抽查 {args.leakage_samples} 条防泄漏记录"
    )
    return 0


def _command_train(args: argparse.Namespace) -> int:
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    training_paths = spec.data.training_paths()
    validation_paths = spec.data.validation_paths()
    if not args.skip_manifest_check:
        _preflight_manifest(
            spec,
            args.manifest,
            [
                *((path, "train") for paths in training_paths.values() for path in paths),
                *(
                    (path, "validation")
                    for paths in validation_paths.values()
                    for path in paths
                ),
            ],
        )
    configure_runtime(spec.experiment.runtime)
    training_pools = pools_from_paths(training_paths)
    validation_pools = pools_from_paths(validation_paths)
    sampler = ScaleStratifiedSampler(
        training_pools,
        root_seed=spec.experiment.root_seed,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
        instances_per_scale=spec.data.train_instances_per_scale,
    )
    validation_cases = fixed_cases_from_pools(
        validation_pools,
        root_seed=spec.experiment.root_seed,
        instances_per_scale=spec.data.validation_instances_per_scale,
        aco_seeds=spec.experiment.validation_seeds,
        batch_size=spec.data.evaluation_batch_size,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
    )
    if args.resume and not args.output:
        resume_path = Path(args.resume)
        output = resume_path if resume_path.is_dir() else resume_path.parent
    else:
        output = (
            Path(args.output)
            if args.output
            else Path("runs")
            / spec.experiment.experiment_id
            / f"seed-{spec.experiment.root_seed}"
        )
    artifact_payload = (
        resume_run_artifacts(output)
        if args.resume
        else initialise_run_artifacts(
            output,
            spec.experiment,
            repository=_repository_root(),
            data_manifest=args.manifest,
        )
    )
    try:
        result = train(
            spec.experiment,
            sampler.cases_for_generation,
            validation_cases,
            output_directory=output,
            resume_from=args.resume,
            progress_callback=lambda record: print(
                (
                    f"generation={record.generation:03d} "
                    f"unique={record.evaluated_unique:03d} "
                    f"min={record.minimum:.6f} "
                    f"median={record.median:.6f} "
                    f"mean={record.mean:.6f} "
                    f"nodes={record.best_nodes} "
                    f"time={record.generation_wall_time:.2f}s "
                    f"eta={record.eta_seconds / 60.0:.1f}min "
                    f"delta={record.best_mean_delta_by_scale}"
                ),
                flush=True,
            ),
        )
    except Exception as exc:
        finalise_run_artifacts(
            output,
            artifact_payload,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        if args.traceback:
            traceback.print_exc()
        else:
            print(f"训练失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finalise_run_artifacts(output, artifact_payload, status="completed")
    print(
        f"训练完成：{output}；champion nodes={result.champion.total_nodes}；"
        f"non-inferiority={'pass' if result.passed_noninferiority else 'fallback'}"
    )
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    spec = _apply_runtime_overrides(load_run_spec(args.config), args)
    paths = spec.data.test_paths(args.partition)
    if not args.skip_manifest_check:
        _preflight_manifest(
            spec,
            args.manifest,
            [(path, "test") for path in paths],
        )
    configure_runtime(spec.experiment.runtime)
    partition_spec = spec.data.test[args.partition]
    batches = iter_problem_batches(
        paths,
        batch_size=args.batch_size or spec.data.evaluation_batch_size,
        candidate_size=spec.experiment.aco.candidate_size,
        dtype=spec.experiment.aco.dtype,
        device=spec.experiment.aco.device,
        min_scale=partition_spec.min_scale,
        max_scale=partition_spec.max_scale,
        max_instances=args.max_instances,
    )
    champion = load_champion(args.champion) if args.champion else None
    transition, pheromone = compile_champion(champion)
    records = evaluate_batches(
        batches,
        spec.experiment.aco,
        method=args.method,
        champion_id=args.champion_id,
        partition=args.partition,
        distribution=partition_spec.distribution,
        root_seed=spec.experiment.root_seed,
        seeds_per_batch=args.seeds,
        transition_program=transition,
        pheromone_program=pheromone,
        backend=spec.experiment.runtime.aco_backend,
    )
    target = write_records(records, args.output)
    print(f"评测完成：{len(records)} 条记录 -> {target}")
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    records = read_records(args.inputs)
    contexts = {
        (record.variant, record.partition, record.distribution)
        for record in records
    }
    if len(contexts) != 1:
        raise ValueError(
            "一次 summarize 只能分析同一 variant/partition/distribution；"
            "请先拆分输入，避免把不可交换的 raw gaps 混合"
        )
    method_counts = Counter(record.method for record in records)
    summaries = summarize_quality(records)
    friedman = friedman_test(records) if len(method_counts) >= 3 else None
    pairwise = (
        paired_wilcoxon_holm(
            records,
            reference_method=args.reference_method,
        )
        if len(method_counts) >= 2
        else []
    )
    bootstrap = hierarchical_bootstrap_delta(
        records,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    context = next(iter(contexts))
    target = write_statistical_report(
        args.output,
        summaries=summaries,
        friedman=friedman,
        pairwise=pairwise,
        bootstrap=bootstrap,
        metadata={
            "variant": context[0],
            "partition": context[1],
            "distribution": context[2],
            "scales": sorted({record.scale for record in records}),
            "input_files": [str(Path(item).resolve()) for item in args.inputs],
        },
    )
    print(
        json.dumps(
            {
                "output": str(target),
                "methods": dict(method_counts),
                "common_context": context,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rmtgp-aco",
        description="Strongly Typed Multi-Tree GP–ACO 研究工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="生成数据 manifest")
    manifest.add_argument("--root", default="Datasets/TSP")
    manifest.add_argument("--output", default="Datasets/manifest.json")
    manifest.add_argument("--full", action="store_true", help="计算完整 SHA-256 与行数")
    manifest.add_argument("--workers", type=int, default=4)
    manifest.set_defaults(handler=_command_manifest)

    verify = subparsers.add_parser("verify-data", help="校验数据 manifest")
    verify.add_argument("--manifest", default="Datasets/manifest.json")
    verify.add_argument("--root")
    verify.add_argument("--skip-hashes", action="store_true")
    verify.add_argument("--leakage-samples", type=int, default=1)
    verify.set_defaults(handler=_command_verify_data)

    train_parser = subparsers.add_parser("train", help="训练一个独立 GP run")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output")
    train_parser.add_argument("--manifest", default="Datasets/manifest.json")
    train_parser.add_argument("--skip-manifest-check", action="store_true")
    train_parser.add_argument("--root-seed", type=int)
    train_parser.add_argument("--processes", type=int)
    train_parser.add_argument(
        "--backend",
        choices=[backend.value for backend in ExecutionBackend],
    )
    train_parser.add_argument(
        "--resume",
        help="run 目录或 training_state.pkl；配置必须与 checkpoint 完全一致",
    )
    train_parser.add_argument(
        "--method-profile",
        choices=["rmtgp", "tr-rgp", "ph-rgp", "matched-replace", "legacy"],
        default="rmtgp",
        help="E1 组件/上一篇研究对照；默认训练双 residual",
    )
    train_parser.add_argument("--traceback", action="store_true")
    train_parser.set_defaults(handler=_command_train)

    evaluate = subparsers.add_parser("evaluate", help="锁定模型后的 paired 测试")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--partition", required=True)
    evaluate.add_argument("--champion", help="省略时评测原始 ACO")
    evaluate.add_argument("--method", required=True)
    evaluate.add_argument("--champion-id", default="baseline")
    evaluate.add_argument("--seeds", type=int, required=True)
    evaluate.add_argument("--max-instances", type=int)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--manifest", default="Datasets/manifest.json")
    evaluate.add_argument("--skip-manifest-check", action="store_true")
    evaluate.add_argument("--root-seed", type=int)
    evaluate.add_argument("--processes", type=int)
    evaluate.add_argument(
        "--backend",
        choices=[backend.value for backend in ExecutionBackend],
    )
    evaluate.set_defaults(handler=_command_evaluate)

    summarize = subparsers.add_parser("summarize", help="统计检验与论文指标汇总")
    summarize.add_argument("--inputs", nargs="+", required=True)
    summarize.add_argument("--reference-method", required=True)
    summarize.add_argument("--bootstrap-replicates", type=int, default=10_000)
    summarize.add_argument("--seed", type=int, default=0)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(handler=_command_summarize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
