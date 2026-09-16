"""固定冠军的配对求解、缓存校验与逐实例原始结果。"""
from __future__ import annotations
from dataclasses import asdict,replace
import json
from pathlib import Path
import socket
import time
import numpy as np
from rmtgp_aco.mechanisms import MechanismConfig,InstrumentationConfig,SolverControl,TRACE_FIELDS,factorial_conditions
from .common import OUT,atomic_json,atomic_npz,digest,evaluation_seed,experiment,file_hash,models,now,read_json,source_manifest,validate_tours,environment
from .prepare import batch


def program_entries(variant,modes=("full",)):
    entries=[{"id":"baseline","mode":"baseline","seed":None,"hash":"baseline","program":(None,None)}]
    for m in models(variant):
        tr,ph=m["program"]
        for mode in modes:
            if mode not in ("full","tr_only","ph_only"): raise ValueError(mode)
            program=(None if mode=="ph_only" else tr,None if mode=="tr_only" else ph)
            entries.append({"id":m["id"],"mode":mode,"seed":m["seed"],"hash":m["structural_hash"],
                            "file_hash":m["file_hash"],"program":program})
    return entries


def run_task(task,out=OUT):
    """每个任务包含同条件 baseline 和全部固定冠军，按语义分组并行。"""
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime,_active_and_representative_programs
    out=Path(out); split=task["split"]; indices=task["indices"]
    horizon=task.get("iterations",5000); variant=task.get("variant","mmas")
    entries=program_entries(variant,tuple(task.get("modes",["full"])))
    mechanism=MechanismConfig(**task["mechanism"])
    audit_spec=task.get("instrumentation","light")
    instrumentation=(InstrumentationConfig(**audit_spec) if isinstance(audit_spec,dict)
                     else InstrumentationConfig(audit_spec))
    seed=task.get("seed",evaluation_seed(split,task["replicate"]))
    source=source_manifest()
    frozen=read_json(out/"manifests/checkpoints.json")
    frozen_hashes={m["id"]:m["file_hash"] for m in frozen}
    if any(e.get("file_hash")!=frozen_hashes[e["id"]] for e in entries if e["id"]!="baseline"):
        raise ValueError("模型文件与冻结清单不同")
    scientific={"task":task,"seed":seed,"source":source["source_hash"],
        "manifest":read_json(out/f"manifests/{split}.json")["manifest_hash"],
        "models":[{k:v for k,v in e.items() if k!="program"} for e in entries],
        "protocol":file_hash(Path(__file__).with_name("protocol.yaml")),"trace_schema":TRACE_FIELDS,
        "input_sha256":file_hash(out/f"inputs/{split}.npz")}
    key=digest(scientific); target=out/"jobs"/task["id"]
    existing=read_json(target/"status.json",{})
    if existing.get("status")=="completed" and existing.get("scientific_hash")==key:
        for name,expected in existing.get("files",{}).items():
            if file_hash(target/name)!=expected: raise RuntimeError(f"已完成结果损坏: {target/name}")
        if instrumentation.detailed:
            from .diagnostics import validate_completed
            validate_completed(target/"diagnostics")
        return existing
    if existing.get("status")=="completed":
        raise RuntimeError(f"科学配置变化，必须新建 cohort，不能覆盖完成结果: {target}")
    atomic_json(target/"status.json",{"status":"running","scientific_hash":key,"started_at":now(),"host":socket.gethostname()})
    problem=batch(split,indices,out)
    aco,runtime=experiment(variant,horizon,task.get("precision","fp32_fast"))
    runtime=replace(runtime,gpu_task_chunk_size=task.get("task_chunk_size",0))
    programs=[e["program"] for e in entries]
    control=SolverControl(mechanism,instrumentation,collected=[])
    recorder=None
    if instrumentation.detailed:
        from .diagnostics import DiagnosticRecorder
        recorder=DiagnosticRecorder(target/"diagnostics",scientific,instrumentation)
        control.observer=recorder
        control.resume=recorder.journal.latest
    if task.get("replay_from"):
        parent=out/"jobs"/task["replay_from"]
        prior=read_json(parent/"manifest.json")
        if prior["seed"]!=seed or prior["task"]["indices"]!=indices:
            raise ValueError("外部回放的实例和 seed 不匹配")
        with np.load(parent/"raw.npz",allow_pickle=False) as data:
            control.replay_restarts=data["trace"][0,:,:,8].astype(np.int8)
            control.source_slots=data["trace"][0,:,:,0].astype(np.int8)
    started=time.perf_counter()
    try:
        result=solve_population_cuda_anytime(problem,aco,programs,seed=seed,runtime=runtime,control=control)
        if recorder: recorder.finish(horizon)
    finally:
        if recorder: recorder.close()
    elapsed=time.perf_counter()-started
    validate_tours(result.best_tour.numpy(),problem.n)
    _,_,_,_,representatives,inverse=_active_and_representative_programs(programs,aco)
    p=len(representatives); b=problem.batch_size
    trace=np.zeros((p*b,horizon,len(TRACE_FIELDS)),dtype=np.float32)
    diagnostics=np.zeros((p*b,8),dtype=np.uint64)
    for shard in control.collected:
        if shard["trace"] is not None: trace[shard["flat_indices"]]=shard["trace"]
        diagnostics[shard["flat_indices"]]=shard["diagnostics"]
    trace=trace.reshape(p,b,horizon,-1)[inverse]
    diagnostics=diagnostics.reshape(p,b,8)[inverse]
    if instrumentation.level!="off" and np.max(trace[...,20])>1e-5:
        raise ArithmeticError(f"deposit budget error={np.max(trace[...,20])}")
    lengths=result.best_length.numpy(); reference=problem.reference_length.numpy()
    gaps=100*(lengths/reference[None,:]-1)
    curve=result.anytime_best.numpy()
    if not np.isfinite(gaps).all() or not np.isfinite(curve).all(): raise ArithmeticError("非有限质量值")
    atomic_npz(target/"raw.npz",length=lengths,reference=reference,gap=gaps,
        tour=result.best_tour.numpy(),best_iteration=result.best_iteration.numpy(),
        anytime=curve,auc=(100*(curve/reference[None,:,None]-1)).mean(axis=-1),
        trace=trace,diagnostics=diagnostics,behavior_alias=inverse,
        instance_hashes=np.asarray(problem.coordinate_hashes),model_ids=np.asarray([e["id"] for e in entries]),
        modes=np.asarray([e["mode"] for e in entries]))
    manifest={**scientific,"scientific_hash":key,"generated_kernel_hashes":sorted(control.kernel_hashes),
        "aco":asdict(aco),"runtime":asdict(runtime),"host":socket.gethostname(),
        "wall_seconds":elapsed,"backend_metrics":result.backend_metrics,"completed_at":now(),"environment":environment(),
        "trace_length_precision":"GPU FP32 search incumbent; final length CPU FP64",
        "logical_solves":len(entries)*b,"executed_solves":p*b}
    if recorder:
        manifest["diagnostic_index_sha256"]=file_hash(target/"diagnostics/index.json")
        manifest["stage_timings"]=control.stage_timings
        manifest["timing_scope"]="有审计批量运行；不得除以模型数作为单模型部署时间"
    # torch dtype 不可直接 JSON 化；配置哈希另存完整文本表达。
    manifest["aco"]={k:str(v) if k=="dtype" else v for k,v in manifest["aco"].items()}
    atomic_json(target/"manifest.json",manifest)
    status={"status":"completed","scientific_hash":key,"wall_seconds":elapsed,
        "completed_at":now(),"files":{"raw.npz":file_hash(target/"raw.npz"),"manifest.json":file_hash(target/"manifest.json")}}
    if recorder:
        status["files"]["diagnostics/index.json"]=file_hash(target/"diagnostics/index.json")
    atomic_json(target/"status.json",status)
    return status


def factorial_tasks(split="diagnosis_dev",out=OUT):
    count=32 if split=="diagnosis_dev" else 128
    seeds=5 if split=="diagnosis_dev" else 10
    tasks=[]
    for rep in range(seeds):
        for start in range(0,count,32):
            order=list(factorial_conditions().items())
            np.random.default_rng(evaluation_seed("task-order-"+split,rep*100+start)).shuffle(order)
            for condition,mechanism in order:
                tasks.append({"id":f"{split}-mmas-{condition}-s{rep:02d}-b{start:03d}",
                    "stage":"P1" if split=="diagnosis_dev" else "P2" if split=="confirm_uniform" else "P4",
                    "split":split,"indices":list(range(start,min(start+32,count))),"replicate":rep,
                    "condition":condition,"variant":"mmas","mechanism":asdict(mechanism),
                    "iterations":5000,"modes":["full"],"instrumentation":"light"})
    return tasks
