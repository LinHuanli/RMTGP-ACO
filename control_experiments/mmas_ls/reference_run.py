"""独立进程运行未修改源码；此模块不得导入新增机制接口。"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .common import OUT,atomic_npz,experiment,models
from .prepare import batch


def main():
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--iterations",type=int,default=300)
    args=parser.parse_args()
    problem=batch("diagnosis_dev",range(2),args.output)
    for variant in ("as","mmas"):
        aco,runtime=experiment(variant,args.iterations)
        programs=[(None,None)]+[m["program"] for m in models(variant)]
        result=solve_population_cuda_anytime(problem,aco,programs,seed=20260917,runtime=runtime)
        atomic_npz(args.output/f"validation/unmodified-{variant}.npz",tour=result.best_tour.numpy(),
            length=result.best_length.numpy(),anytime=result.anytime_best.numpy(),
            best_iteration=result.best_iteration.numpy(),diagnostics=result.diagnostics.numpy(),
            iterations=np.asarray(args.iterations))
        print(f"unmodified {variant} complete",flush=True)


if __name__=="__main__": main()
