"""P3 重型快照与同状态分叉；探针不混入常规求解耗时。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .common import OUT,atomic_json,atomic_npz,digest,evaluation_seed,experiment,file_hash,now,read_json,source_manifest
from .prepare import batch
from .evaluate import program_entries


def tour_edges(tour):
    return frozenset((min(int(a),int(b)),max(int(a),int(b))) for a,b in zip(tour[:-1],tour[1:]))


def tour_hash(tour):
    """排序无向边编码；对旋转和反向严格不变，不用长度代表路径相同。"""
    import hashlib
    return hashlib.sha256(np.asarray(sorted(tour_edges(tour)),dtype="<i4").tobytes()).hexdigest()


class Recorder:
    def __init__(self,directory,metadata,iterations=(1,100,250,500,1000,2500)):
        self.directory=Path(directory);self.metadata=metadata;self.iterations=set(iterations)

    def __call__(self,phase,iteration,state):
        if iteration not in self.iterations or phase not in ("post_ls","iteration_end"):return
        import cupy as cp
        target=self.directory/f"t{iteration:04d}-{phase}"
        arrays={k:cp.asnumpy(v) for k,v in state.items() if isinstance(v,cp.ndarray)}
        atomic_npz(target.with_suffix(".npz"),**arrays)
        atomic_json(target.with_suffix(".json"),{"metadata":self.metadata,"iteration":iteration,"phase":phase,
            "flat_indices":state["flat_indices"].tolist(),"arrays_sha256":file_hash(target.with_suffix(".npz")),"time":now()})
        # 结构诊断保留所有蚂蚁；不能把同长不同边集视为同一 basin。
        tours=arrays["tour_workspace"];unique=[];retained=[]
        for i,colony in enumerate(tours):
            unique.append(len(set(tour_hash(t) for t in colony)))
            retained.append([len(tour_edges(t)&tour_edges(arrays["pre_tour_workspace"][i,a]))/(len(t)-1) for a,t in enumerate(colony)])
        atomic_json(target.with_name(target.name+"-structure.json"),{"unique_post_ls_tours":unique,"pre_post_edge_retention":retained})


def capture(condition="C111",replicate=0,owner=0,out=OUT):
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    from rmtgp_aco.mechanisms import SolverControl,InstrumentationConfig,factorial_conditions
    problem=batch("diagnosis_dev",range(8),out);aco,runtime=experiment("mmas")
    entry=program_entries("mmas")[owner]
    specification={"split":"diagnosis_dev","indices":list(range(8)),"condition":condition,"replicate":replicate,
        "owner":{k:v for k,v in entry.items() if k!="program"},"iterations":5000,"seed":evaluation_seed("diagnosis_dev",replicate),
        "manifest":read_json(Path(out)/"manifests/diagnosis_dev.json")["manifest_hash"],"source_hash":source_manifest()["source_hash"]}
    directory=Path(out)/"heavy"/f"{condition}-s{replicate}-owner{owner}"
    control=SolverControl(factorial_conditions()[condition],InstrumentationConfig("heavy"),
        observer=Recorder(directory,specification),collected=[])
    result=solve_population_cuda_anytime(problem,aco,[entry["program"]],seed=specification["seed"],runtime=runtime,control=control)
    atomic_npz(directory/"result.npz",tour=result.best_tour.numpy(),length=result.best_length.numpy(),anytime=result.anytime_best.numpy())
    atomic_json(directory/"status.json",{"status":"completed","specification":specification,"time":now()})


def read_snapshot(path):
    path=Path(path);meta=read_json(path.with_suffix(".json"))
    if file_hash(path.with_suffix(".npz"))!=meta["arrays_sha256"]:raise ValueError("快照文件哈希不同")
    with np.load(path.with_suffix(".npz"),allow_pickle=False) as data:
        snapshot={"iteration":meta["iteration"],"phase":meta["phase"],"flat_indices":np.array(meta["flat_indices"]),
                  "arrays":{k:data[k].copy() for k in data.files}}
    return snapshot,meta


def fork(snapshot_path,champion=81001,steps=100,out=OUT):
    """两种 owner 起点都适用；继续使用原绝对 iteration 和原 5000 分母。"""
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    from rmtgp_aco.mechanisms import SolverControl,InstrumentationConfig,factorial_conditions
    snapshot,meta=read_snapshot(snapshot_path);specification=meta["metadata"]
    if specification["source_hash"]!=source_manifest()["source_hash"]:raise ValueError("分叉必须使用保存快照时的不可变源码")
    if specification["manifest"]!=read_json(Path(out)/"manifests/diagnosis_dev.json")["manifest_hash"]:raise ValueError("快照数据不一致")
    problem=batch("diagnosis_dev",specification["indices"],out);aco,runtime=experiment("mmas")
    entries=[e for e in program_entries("mmas",("full","tr_only","ph_only")) if e["seed"] in (None,champion)]
    stop=min(5000,snapshot["iteration"]+steps-(snapshot["phase"]=="post_ls"))
    if steps<1:raise ValueError("分叉长度须为正")
    target=Path(out)/"forks"/Path(snapshot_path).parent.name/Path(snapshot_path).stem/f"champion-{champion}-steps{steps}"
    lengths=[];curves=[];states={}
    for entry in entries:
        def observe(phase,iteration,state):
            if phase=="iteration_end" and iteration==stop:
                import cupy as cp
                states[entry["mode"]]={name:cp.asnumpy(state[name]) for name in
                    ("pheromone_workspace","deposit_workspace","tour_workspace","best_tours","mechanism_trace")}
        control=SolverControl(factorial_conditions()[specification["condition"]],InstrumentationConfig("heavy"),
            resume=snapshot,stop_iteration=stop,observer=observe)
        result=solve_population_cuda_anytime(problem,aco,[entry["program"]],seed=specification["seed"],runtime=runtime,control=control)
        lengths.append(result.best_length.numpy()[0]);curves.append(result.anytime_best.numpy()[0,:,:stop])
    atomic_npz(target/"result.npz",length=np.stack(lengths),anytime=np.stack(curves),modes=np.array([e["mode"] for e in entries]),
        **{mode+"__"+name:array for mode,state in states.items() for name,array in state.items()})
    atomic_json(target/"manifest.json",{"snapshot":str(snapshot_path),"snapshot_hash":meta["arrays_sha256"],
        "owner":specification["owner"],"champion":champion,"steps":steps,"last_iteration":stop,"time":now(),
        "scope":"固定起点的有限时域干预；不能解释为自然中介效应或重新训练收益"})
    if snapshot["phase"]=="post_ls" and steps==1:
        atomic_json(target/"ph_transfer.json",ph_transfer(snapshot,states,problem,aco))


def ph_transfer(snapshot,states,problem,aco):
    """同一 post-LS 状态的一次 PH 更新；候选概率只报告 tau^alpha eta^beta 基础核。"""
    base=states["baseline"]["pheromone_workspace"]
    distance=problem.distances.numpy();n=problem.n;rows=[]
    for mode,state in states.items():
        if mode=="baseline":continue
        tau=state["pheromone_workspace"];difference=tau.astype(np.float64)-base
        tv=[];argmax=[]
        for i in range(problem.batch_size):
            for ant in range(4):
                tour=snapshot["arrays"]["pre_tour_workspace"][i,ant]
                for prefix in (1,n//4,n//2,3*n//4):
                    visited=set(map(int,tour[:prefix]));city=int(tour[prefix-1])
                    candidates=np.argsort(distance[i,city],kind="stable")[1:aco.candidate_size+1]
                    candidates=np.array([v for v in candidates if int(v) not in visited],dtype=int)
                    if not len(candidates):candidates=np.array([v for v in range(n) if v not in visited],dtype=int)
                    probabilities=[]
                    for matrix in (base,tau):
                        scores=np.maximum(matrix[i,city,candidates],aco.epsilon_numeric).astype(float)**aco.alpha/np.maximum(distance[i,city,candidates],aco.epsilon_distance)**aco.beta
                        probabilities.append(scores/scores.sum())
                    tv.append(float(np.abs(probabilities[0]-probabilities[1]).sum()/2));argmax.append(int(np.argmax(probabilities[0])!=np.argmax(probabilities[1])))
        rows.append({"mode":mode,"tau_relative_l1":float(np.abs(difference).sum()/max(float(np.abs(base).sum()),1e-30)),
            "tau_relative_l2":float(np.linalg.norm(difference)/max(float(np.linalg.norm(base)),1e-30)),
            "base_kernel_tv_mean":float(np.mean(tv)),"base_kernel_argmax_change_rate":float(np.mean(argmax)),
            "floor_clips":state["mechanism_trace"][:,snapshot["iteration"]-1,6].tolist()})
    return {"rows":rows,"contexts":"4 ants × 4 fixed pre-LS prefixes × each instance; same visited mask",
        "limitation":"TV 是基础状态转移核的局部敏感性；不包含 GP TR 输出，不能标为完整 GP 构造概率的 TV。"}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("capture","fork"));p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--condition",default="C111");p.add_argument("--replicate",type=int,default=0);p.add_argument("--owner",type=int,default=0)
    p.add_argument("--snapshot",type=Path);p.add_argument("--champion",type=int,default=81001);p.add_argument("--steps",type=int,default=100)
    a=p.parse_args()
    if a.action=="capture":capture(a.condition,a.replicate,a.owner,a.output)
    else:fork(a.snapshot,a.champion,a.steps,a.output)
