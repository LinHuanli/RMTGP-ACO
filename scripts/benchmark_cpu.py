"""与集群调度器无关的 CPU 基线入口；不依赖 GPU，不自动提交远程作业。"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--campaign", default="artifacts/a5000-main-v1")
    prepare.add_argument("--output", required=True)
    probe = sub.add_parser("probe")
    probe.set_defaults(command="probe")
    run = sub.add_parser("run")
    run.add_argument("--bundle", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--backend", choices=["cpu_python", "cpu_existing"], required=True)
    run.add_argument("--cores", type=int, choices=[1, 8, 16], required=True)
    run.add_argument("--generation", type=int, default=1)
    run.add_argument("--block", type=int, default=0)
    run.add_argument("--cache-state", choices=["warm", "cold"], default="warm")
    args = parser.parse_args()
    if args.command == "run":
        from gpaco.artifact_registry import require_output

        require_output(args.output, "E01-p01-cpu-baselines")
        # 必须早于导入 Numba；每个 cell 独立缓存，首次编译可验证。
        root = Path(os.environ.get("GPACO_ROOT", Path(__file__).resolve().parents[1])).resolve()
        output = Path(args.output).resolve()
        if not output.is_relative_to(root) or output == root or output.exists():
            parser.error("输出必须是项目内尚不存在的目录")
        cache = (
            root
            / ".cache/cpu-bench"
            / __import__("hashlib").sha256(str(output).encode()).hexdigest()
        )
        if cache.exists():
            parser.error("该 cell 缓存已存在；使用新的输出目录，不伪造冷启动")
        cache.mkdir(parents=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache)
        os.environ["NUMBA_NUM_THREADS"] = str(args.cores)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
    from gpaco.artifact_registry import resolve
    from gpaco.benchmark_inputs import export_bundle
    from gpaco.cpu_benchmark import physical_cpu_ids
    from gpaco.cpu_benchmark import run as measure
    from gpaco.data import ROOT, write_json
    from gpaco.hardware_inputs import safe_directory

    if args.command == "probe":
        cpus = physical_cpu_ids()
        print(
            json.dumps(
                {
                    "physical_cpu_ids": cpus,
                    "supported_core_counts": [n for n in (1, 8, 16) if n <= len(cpus)],
                    "load_average": os.getloadavg(),
                }
            )
        )
    elif args.command == "run":
        result = measure(
            args.output,
            args.bundle,
            args.generation,
            args.block,
            args.backend,
            args.cores,
            args.cache_state,
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "backend",
                        "cores",
                        "eval_wall_s",
                        "mean_gap_percent",
                        "executed_tasks",
                    )
                }
            )
        )
    else:
        campaign, output = safe_directory(args.campaign), safe_directory(args.output)
        if output != resolve("shared-frozen-population-p01"):
            raise ValueError("输入导出只能写登记路径；新版本先登记，禁止使用inputs-new等临时名称")
        output.mkdir(parents=True, exist_ok=False)
        config = json.loads((campaign / "campaign.json").read_text())["config"]
        rows = []
        for n in config["scales"]:
            paths = {
                g: ROOT
                / f"artifacts/{config['cohort_source_campaign']}/as-tsp{n}-seed{config['cohort_source_seed']}/cohorts/generation-{g:03d}.json"
                for g in config["cohort_generations"]
            }
            available = {g: p for g, p in paths.items() if p.exists()}
            for variant in config["variants"]:
                source = campaign / f"inputs/tsp{n}/{variant}"
                if not (source / "READY.json").exists() or not available:
                    rows.append({"n": n, "variant": variant, "status": "pending_dependency"})
                    continue
                export_bundle(source, available, output / f"tsp{n}/{variant}")
                rows.append(
                    {
                        "n": n,
                        "variant": variant,
                        "status": "ready",
                        "generations": list(available),
                        "pending_generations": sorted(set(paths) - set(available)),
                    }
                )
        write_json(output / "inventory.json", rows)
        print(json.dumps(rows))


if __name__ == "__main__":
    main()
