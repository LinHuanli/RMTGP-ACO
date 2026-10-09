"""所有长任务均显式指定输出、科学预算与执行配置。"""

import argparse
import traceback
from pathlib import Path

from .config import ExecutionPlan, SearchConfig
from .data import ROOT, write_json
from .experiment import benchmark, train


def main():
    parser = argparse.ArgumentParser(prog="gpaco")
    parser.add_argument("command", choices=["train", "benchmark"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, choices=[100, 500], required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--variant", choices=["as", "acs", "mmas"], default="as")
    parser.add_argument(
        "--backend", choices=["cpu_python", "cpu_existing", "cuda_existing"], default="cuda_existing"
    )
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--ants", type=int, default=32)
    parser.add_argument("--lanes", type=int, choices=[4, 8, 16, 32], default=8)
    parser.add_argument("--active-tasks", type=int, default=3200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--generated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--cohort")
    parser.add_argument("--split", choices=["tuning", "holdout"], default="tuning")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output 必须位于项目根目录内")
    search = SearchConfig(variant=args.variant, ants=args.ants, iterations=args.iterations)
    plan = ExecutionPlan(
        backend=args.backend,
        candidate_lanes=args.lanes,
        active_tasks=args.active_tasks,
        cpu_threads=args.threads,
        generated=args.generated,
        profile_stages=args.profile_stages,
    )
    try:
        if args.command == "train":
            train(
                output,
                args.n,
                args.seed,
                search,
                plan,
                population_size=args.population,
                batch_size=args.batch,
                generations=args.generations,
                resume=args.resume,
            )
        else:
            benchmark(
                output,
                args.n,
                args.seed,
                search,
                plan,
                args.population,
                args.batch,
                args.blocks,
                cohort=args.cohort,
                split=args.split,
            )
    except Exception as error:
        if output.exists() and not isinstance(error, (FileExistsError, FileNotFoundError)):
            write_json(
                output / "failure.json",
                {"status": "failed", "error": str(error), "traceback": traceback.format_exc()},
            )
        raise


if __name__ == "__main__":
    main()
