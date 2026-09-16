"""P0 门禁：真实冠军的旧源码等价、审计扰动及固定随机流检查。"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
from .common import OUT,atomic_json,experiment,models,now,validate_tours
from .prepare import batch


def native(out=OUT):
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    from rmtgp_aco.mechanisms import SolverControl,InstrumentationConfig
    out=Path(out); rows=[]
    for variant in ("as","mmas"):
        original=np.load(out/f"validation/unmodified-{variant}.npz",allow_pickle=False)
        aco,runtime=experiment(variant,int(original["iterations"]))
        problem=batch("diagnosis_dev",range(2),out)
        programs=[(None,None)]+[m["program"] for m in models(variant)]
        for level in ("legacy","off","light","heavy"):
            control=None if level=="legacy" else SolverControl(instrumentation=InstrumentationConfig(level),collected=[])
            start=time.perf_counter()
            result=solve_population_cuda_anytime(problem,aco,programs,seed=20260917,runtime=runtime,control=control)
            row={"variant":variant,"instrumentation":level,"seconds":time.perf_counter()-start}
            for field,actual in (("tour",result.best_tour),("length",result.best_length),
                                 ("anytime",result.anytime_best),("best_iteration",result.best_iteration),
                                 ("diagnostics",result.diagnostics)):
                row[field+"_equal"]=bool(np.array_equal(original[field],actual.numpy()))
            validate_tours(result.best_tour.numpy(),problem.n)
            rows.append(row)
            atomic_json(out/"validation/native.json",{"status":"running","rows":rows})
            if not all(v for k,v in row.items() if k.endswith("_equal")):
                raise AssertionError(f"原源码数值等价失败: {row}")
            print(f"native {variant} {level} passed {row['seconds']:.1f}s",flush=True)
        # 分块、实例顺序不能更改 counter RNG；每个模型仍与相同实例配对。
        reordered=batch("diagnosis_dev",[1,0],out)
        result=solve_population_cuda_anytime(reordered,aco,programs,seed=20260917,
            runtime=replace(runtime,gpu_task_chunk_size=2),control=SolverControl())
        assert np.array_equal(original["tour"],result.best_tour.numpy()[:,::-1])
        assert np.array_equal(original["anytime"],result.anytime_best.numpy()[:,::-1])
        rows.append({"variant":variant,"reordered_chunked_equal":True})
    atomic_json(out/"validation/native.json",{"status":"passed","completed_at":now(),"rows":rows,
        "scope":"同卡同环境、六个真实冠军和两个 baseline、2 个 TSP500、300 轮；非历史质量复现"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args(); native(args.output)
