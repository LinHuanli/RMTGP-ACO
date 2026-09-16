"""P0 独立精度审计。Numba 是实际 FP64 搜索，不是 FP32 tour 长度重算。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from .common import OUT,atomic_json,atomic_npz,digest,evaluation_seed,experiment,file_hash,now,read_json,source_manifest
from .evaluate import program_entries
from .prepare import batch


def run(backend,condition,replicate,out=OUT):
    from rmtgp_aco.mechanisms import SolverControl,factorial_conditions
    path=Path(out)/"precision"/f"{backend}-{condition}-s{replicate}"
    specification={"backend":backend,"condition":condition,"replicate":replicate,"indices":list(range(8)),
        "iterations":300,"source":source_manifest()["source_hash"],"seed":evaluation_seed("precision",replicate),
        "manifest":read_json(Path(out)/"manifests/diagnosis_dev.json")["manifest_hash"],
        "models":[{k:v for k,v in p.items() if k!="program"} for p in program_entries("mmas")]}
    existing=read_json(path/"status.json",{})
    if existing.get("status")=="completed":
        if existing.get("scientific_hash")!=digest(specification):raise RuntimeError("精度审计配置变化；使用新 cohort，禁止覆盖已完成结果")
        if file_hash(path/"raw.npz")!=existing["file_hash"]:raise RuntimeError("精度审计缓存损坏")
        return
    problem=batch("diagnosis_dev",range(8),out); entries=program_entries("mmas")
    aco,runtime=experiment("mmas",300,"fp32" if backend=="fp32" else "fp32_fast")
    mechanism=factorial_conditions()[condition];seed=specification["seed"]
    started=time.perf_counter()
    if backend=="numba_fp64":
        from rmtgp_aco.aco_numba import solve_numba
        lengths=[];curves=[];tours=[]
        for entry in entries:
            tr,ph=entry["program"]
            result=solve_numba(problem,aco,transition_program=tr,pheromone_program=ph,seed=seed,
                control=SolverControl(mechanism))
            lengths.append(result.best_length.numpy());curves.append(result.anytime_best.numpy());tours.append(result.best_tour.numpy())
        lengths=np.stack(lengths);curves=np.stack(curves);tours=np.stack(tours)
    else:
        from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
        result=solve_population_cuda_anytime(problem,aco,[x["program"] for x in entries],seed=seed,
            runtime=runtime,control=SolverControl(mechanism))
        lengths=result.best_length.numpy();curves=result.anytime_best.numpy();tours=result.best_tour.numpy()
    reference=problem.reference_length.numpy()
    atomic_npz(path/"raw.npz",length=lengths,gap=100*(lengths/reference[None,:]-1),anytime=curves,tour=tours)
    atomic_json(path/"status.json",{"status":"completed","scientific_hash":digest(specification),
        "specification":specification,"seconds":time.perf_counter()-started,"file_hash":file_hash(path/"raw.npz"),"time":now()})


def report(out=OUT):
    rows=[]
    conditions=("C111","C011","C101","C110"); results={}
    for backend in ("fp32_fast","fp32","numba_fp64"):
        arrays=[]
        for condition in conditions:
            seeds=[]
            for replicate in range(3):
                path=Path(out)/"precision"/f"{backend}-{condition}-s{replicate}"/"raw.npz"
                if not path.exists():return {"status":"pending","missing":str(path)}
                with np.load(path,allow_pickle=False) as data:seeds.append(data["gap"].copy())
            arrays.append(np.stack(seeds))
        cube=np.stack(arrays)  # [condition,seed,model,instance]
        results[backend]=(cube[:,:,1:]-cube[:,:,:1]).mean(axis=(1,2)).T
    for backend in ("fp32_fast","fp32"):
        shift=results[backend]-results["numba_fp64"]
        effects=shift[:,1:]-shift[:,:1]
        rows.append({"backend":backend,"delta_difference_pp":shift.mean(axis=0).tolist(),
            "component_effect_difference_pp":effects.mean(axis=0).tolist(),
            "instance_sd_effect_difference":effects.std(axis=0,ddof=1).tolist()})
    payload={"status":"review_required","instances":8,"seeds":3,"iterations":300,"rows":rows,
        "scope":"独立小样本数值筛查，不能据此认证5000轮无精度差；CPU/GPU约简和LS路径不同，需结合算子测试分析。"}
    atomic_json(Path(out)/"validation/precision.json",payload);return payload


def main():
    p=argparse.ArgumentParser();p.add_argument("--backend",choices=("fp32_fast","fp32","numba_fp64","report"),required=True)
    p.add_argument("--output",type=Path,default=OUT);p.add_argument("--condition");p.add_argument("--replicate",type=int)
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    if a.backend=="report":print(report(a.output));return
    if a.workers>1:
        if a.backend!="numba_fp64":raise ValueError("GPU 一卡只允许单 worker；这里仅为 CPU oracle 并行")
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor,as_completed
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs=[pool.submit(run,a.backend,c,r,a.output) for c in ("C111","C011","C101","C110") for r in range(3)]
            for f in as_completed(jobs):f.result();print("FP64 audit task completed",flush=True)
        return
    for condition in ((a.condition,) if a.condition else ("C111","C011","C101","C110")):
        for rep in ((a.replicate,) if a.replicate is not None else range(3)):
            print(a.backend,condition,rep,flush=True);run(a.backend,condition,rep,a.output)


if __name__=="__main__":main()
