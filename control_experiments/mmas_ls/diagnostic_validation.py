"""机制诊断验收与开销试跑；成功不自动批准 P1/P2 科学门禁。"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import time
import numpy as np
import torch
from .common import OUT,atomic_json,atomic_npz,digest,environment,experiment,now,source_manifest
from .prepare import batch
from .evaluate import program_entries
from .diagnostics import DiagnosticRecorder
from .diagnostic_analysis import summarize,check_program_outputs


def pilot(directory,instances=2,steps=100,variants=("mmas","as"),check_reorder=True):
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime,_active_and_representative_programs
    from rmtgp_aco.mechanisms import SolverControl,InstrumentationConfig
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    specification={"source":source_manifest(),"environment":environment(),"instances":instances,"steps":steps,
        "aco_horizon":5000,"seed":57231,"variants":list(variants),"check_reorder":check_reorder,
        "purpose":"配对轨迹与记录验收；非科学效果比较"}
    atomic_json(directory/"status.json",{"status":"running","started_at":now(),"specification":specification})
    rows=[]; problem=batch("diagnosis_dev",range(instances),OUT)
    try:
        for variant in variants:
            aco,runtime=experiment(variant)
            entries=program_entries(variant);programs=[e["program"] for e in entries]
            # 少量热身避免把第一次 resident/编译全部误计为诊断开销。
            solve_population_cuda_anytime(problem,aco,programs,seed=57231,runtime=runtime,
                control=SolverControl(instrumentation=InstrumentationConfig("off"),stop_iteration=1))
            warm_inst=InstrumentationConfig(profile="mechanism_v3",schema_version=3)
            warm_writer=DiagnosticRecorder(directory/"warmup"/variant,{"pilot":specification,"variant":variant},warm_inst)
            try:
                solve_population_cuda_anytime(problem,aco,programs,seed=57231,runtime=runtime,
                    control=SolverControl(instrumentation=warm_inst,observer=warm_writer,stop_iteration=1))
                warm_writer.finish(1)
            finally:warm_writer.close()
            times={};results={};timings={}
            for profile in ("off","mechanism_v3"):
                inst=(InstrumentationConfig("off") if profile=="off" else
                      InstrumentationConfig(profile="mechanism_v3",schema_version=3))
                writer=(DiagnosticRecorder(directory/variant,{"pilot":specification,"variant":variant},inst)
                        if profile!="off" else None)
                control=SolverControl(instrumentation=inst,stop_iteration=steps,observer=writer,collected=[])
                started=time.perf_counter()
                try:
                    result=solve_population_cuda_anytime(problem,aco,programs,seed=57231,runtime=runtime,control=control)
                    if writer: writer.finish(steps)
                finally:
                    if writer: writer.close()
                times[profile]=time.perf_counter()-started
                results[profile]=result;timings[profile]=control.stage_timings
                atomic_npz(directory/f"{variant}-{profile}.npz",tour=result.best_tour.numpy(),
                    length=result.best_length.numpy(),anytime=result.anytime_best.numpy()[:,:,:steps])
            assert torch.equal(results["off"].best_tour,results["mechanism_v3"].best_tour), "审计改变 tour"
            assert torch.equal(results["off"].anytime_best[:,:,:steps],results["mechanism_v3"].anytime_best[:,:,:steps]), "审计改变轨迹"
            chunk_reorder=None
            if instances>=32 and check_reorder:
                # 单独验收完整的 32-instance 形状；不将这些重复运行视为科学样本。
                changed=solve_population_cuda_anytime(problem,aco,programs,seed=57231,
                    runtime=replace(runtime,gpu_task_chunk_size=7),
                    control=SolverControl(instrumentation=warm_inst,stop_iteration=steps))
                assert torch.equal(changed.best_tour,results["off"].best_tour), "分块改变 tour"
                assert torch.equal(changed.anytime_best[:,:,:steps],results["off"].anytime_best[:,:,:steps]), "分块改变轨迹"
                order=np.random.default_rng(88241).permutation(instances);inverse=np.argsort(order)
                changed=solve_population_cuda_anytime(problem.take(order.tolist()),aco,programs,seed=57231,runtime=runtime,
                    control=SolverControl(instrumentation=warm_inst,stop_iteration=steps))
                assert torch.equal(changed.best_tour[:,inverse],results["off"].best_tour), "重排改变 tour"
                assert torch.equal(changed.anytime_best[:,inverse,:steps],results["off"].anytime_best[:,:,:steps]), "重排改变轨迹"
                chunk_reorder={"status":"passed","instances":instances,"chunk_size":7,"permutation":order.tolist()}
            _,_,_,_,representatives,_=_active_and_representative_programs(programs,aco)
            executed=[programs[i] for i in representatives]
            max_raw_error=0.
            for name,record in writer.journal.index["files"].items():
                meta=record["metadata"]
                if meta["kind"]!="sample": continue
                with np.load(directory/variant/name,allow_pickle=False) as data:
                    max_raw_error=max(max_raw_error,check_program_outputs(dict(data),executed,meta["flat_indices"],instances))
            summarize(directory/variant)
            files=writer.journal.index["files"].values()
            compressed=sum(v["compressed_bytes"] for v in files)
            row={"variant":variant,"tour_exact":True,"anytime_exact":True,"wall_seconds":times,
                "wall_ratio":times["mechanism_v3"]/times["off"],"stage_timings":timings,
                "raw_cpu_max_error":max_raw_error,"compressed_bytes":compressed,
                "chunk_reorder":chunk_reorder,
                "disk_projection_5000_per_logical_solve_bytes":compressed*5000/steps/(instances*4),
                "timing_caveat":"双 profile 预热；包含异步写盘；2 个实例的短跑不能替代正式 batch 的 5% 开销验收"}
            rows.append(row);atomic_json(directory/"results.json",rows)
        report={"status":"passed","completed_at":now(),"rows":rows,"specification":specification,
                "gate":"仅本卡、本样本配对验收；不自动放行正式实验"}
        atomic_json(directory/"status.json",report)
        return report
    except BaseException as error:
        atomic_json(directory/"status.json",{"status":"failed","error":repr(error),"completed_at":now(),"rows":rows})
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path)
    p.add_argument("--instances",type=int,default=2);p.add_argument("--steps",type=int,default=100)
    p.add_argument("--variants",nargs="+",choices=("mmas","as"),default=["mmas","as"])
    p.add_argument("--skip-reorder",action="store_true")
    a=p.parse_args();print(pilot(a.directory,a.instances,a.steps,a.variants,not a.skip_reorder))
