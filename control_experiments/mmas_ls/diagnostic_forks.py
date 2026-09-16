"""从 v3 永久快照进行同状态干预。所有分支保留绝对轮次和原 5000 分母。"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .common import OUT,atomic_json,atomic_npz,digest,experiment,file_hash,now,read_json,source_manifest
from .evaluate import program_entries
from .prepare import batch
from .probes import tour_edges


def transition_distribution(tau,problem,instance,context,visited,program,aco,iteration,stagnation):
    """固定可行集合的 CPU FP64 参考核；包含完整 TR，明确区别于 GPU 实际概率日志。"""
    n=problem.n; city=int(context[3]); previous=int(context[4]); step=int(context[2])
    available=np.array([not (int(visited[i//64])>>(i%64)&1) for i in range(n)])
    nearest=problem.nn_indices.numpy()[instance,city,:aco.candidate_size].astype(int)
    candidates=nearest[available[nearest]];fallback=not len(candidates)
    if fallback: candidates=np.flatnonzero(available)
    distance=problem.distances.numpy()[instance];coords=problem.coords.numpy()[instance]
    values=np.asarray(tau[city,candidates],float);length=distance[city,candidates]
    eta=1/np.maximum(length,aco.epsilon_distance)
    baseline=values**aco.alpha*eta**aco.beta
    count=len(candidates);basep=baseline/baseline.sum() if baseline.sum()>aco.epsilon_numeric else np.full(count,1/count)
    logtau=np.log(np.maximum(values,aco.epsilon_numeric));logeta=np.log(eta)
    rank=np.empty(count);rank[np.lexsort((candidates,length))]=np.arange(count)
    entropy=-np.sum(basep*np.log(np.maximum(basep,aco.epsilon_numeric)))
    horizon=aco.iterations
    # 全局 nearest-neighbour rank 按距离、城市编号稳定排序，与数据准备语义对应。
    order=np.argsort(distance,axis=-1,kind="stable"); ranks=np.empty_like(order)
    np.put_along_axis(ranks,order,np.arange(n)[None,:],axis=-1)
    mutual=1-(ranks[city,candidates]+ranks[candidates,city]-2)/(2*max(n-2,1))
    if previous<0:turn=np.zeros(count)
    else:
        incoming=coords[city]-coords[previous];outgoing=coords[candidates]-coords[city]
        turn=(outgoing@incoming)/(np.linalg.norm(incoming)*np.linalg.norm(outgoing,axis=1)+aco.epsilon_numeric)
    ctx={"RTau":np.tanh((logtau-logtau.mean())/(logtau.std()+1e-8)),
        "REta":np.tanh((logeta-logeta.mean())/(logeta.std()+1e-8)),
        "BaseConf":np.tanh(np.log(np.maximum(basep,aco.epsilon_numeric))+np.log(count)),
        "DistRank":np.zeros(count) if count==1 else 1-2*rank/(count-1),
        "Entropy":np.full(count,-1 if count==1 else 2*entropy/np.log(count)-1),
        "ConstructProg":np.full(count,2*step/max(n-1,1)-1),
        "ACOProg":np.full(count,2*(iteration-1)/max(horizon-1,1)-1),
        "Stagnation":np.full(count,2*min(stagnation/horizon,1)-1),
        "Tau":values,"Distance":length,"MeanTau":np.full(count,values.mean()),
        "MeanDistance":np.full(count,length.mean()),"Size":np.full(count,n),"FeasibleCount":np.full(count,count),
        "MutualRank":np.clip(mutual,0,1),"TurnCos":np.clip(turn,-1,1)}
    if program is None or program.is_exact_zero:scores=baseline
    else:
        raw=program.evaluate({k:torch.as_tensor(v,dtype=torch.float64) for k,v in ctx.items()}).numpy()
        scores=baseline*(1+aco.gamma_transition*np.tanh(raw))
    if fallback:
        p=np.zeros(count);p[np.argmax(scores)]=1.
    else:p=scores/scores.sum() if scores.sum()>aco.epsilon_numeric else np.full(count,1/count)
    return candidates,p


def read_owner_snapshot(job,snapshot_name,owner):
    """行为别名映射只用于选择执行状态，不能产生额外独立样本。"""
    job=Path(job);directory=job/"diagnostics";index=read_json(directory/"index.json")
    record=index["files"][snapshot_name];meta=record["metadata"]
    if meta["kind"]!="permanent":raise ValueError("分叉只使用预指定永久快照")
    if file_hash(directory/snapshot_name)!=record["sha256"]:raise ValueError("快照损坏")
    spec=read_json(directory/"specification.json");b=len(spec["task"]["indices"])
    with np.load(job/"raw.npz") as raw:representative=int(raw["behavior_alias"][owner])
    flat=np.asarray(meta["flat_indices"]);select=np.flatnonzero(flat//b==representative)
    if len(select)!=b: raise ValueError("快照不含该 owner 的完整实例块")
    if not np.array_equal(flat[select]%b,np.arange(b)):raise ValueError("实例顺序不同")
    with np.load(directory/snapshot_name,allow_pickle=False) as data:
        arrays={k:(v[select].copy() if v.ndim and v.shape[0]==len(flat) else v.copy()) for k in data.files for v in [data[k]]}
    return {"iteration":meta["iteration"],"phase":meta["phase"],"flat_indices":np.arange(b),"arrays":arrays},spec,record


def deposited_map(state,t):
    count=int(state["mechanism_trace"][t,state["iteration"]-1,3]);values={}
    for tour,deposit in zip(state["audit_sources"][t,:count],state["deposit_workspace"][t,:count]):
        for u,v,w in zip(tour[:-1],tour[1:],deposit):
            key=(min(int(u),int(v)),max(int(u),int(v)));values[key]=values.get(key,0.)+float(w)
    return values


def immediate_transfer(states,snapshot,problem,aco,tr_program):
    """零 PH 与真实 PH 的同来源/预算更新：TR-only 对 full。"""
    base=states["tr_only"][1];full=states["full"][1]; rows=[];n=problem.n
    for t in range(problem.batch_size):
        zero=deposited_map(base,t);real=deposited_map(full,t);budget=sum(zero.values())
        ad=sum(abs(zero.get(k,0)-real.get(k,0)) for k in zero.keys()|real.keys())/budget
        phases=[]
        for phase in range(4):
            a=base["audit_tau"][t,phase] if phase<3 else base["pheromone_workspace"][t]
            b=full["audit_tau"][t,phase] if phase<3 else full["pheromone_workspace"][t]
            mask=~np.eye(n,dtype=bool);difference=(b-a)[mask].astype(float)
            phases.append({"phase":phase,"relative_l1":float(abs(difference).sum()/max(abs(a[mask]).sum(dtype=float),1e-30)),
                "relative_l2":float(np.linalg.norm(difference)/max(np.linalg.norm(a[mask].astype(float)),1e-30))})
        tv={"base":[],"full_gp":[]};changed={"base":[],"full_gp":[]}
        for context,visited in zip(snapshot["arrays"]["audit_context"][t],snapshot["arrays"]["audit_visited"][t]):
            if context[0]<0:continue
            for label,program in (("base",None),("full_gp",tr_program)):
                ca,pa=transition_distribution(base["pheromone_workspace"][t],problem,t,context,visited,program,aco,
                    base["iteration"]+1,int(base["stagnation"][t]))
                cb,pb=transition_distribution(full["pheromone_workspace"][t],problem,t,context,visited,program,aco,
                    full["iteration"]+1,int(full["stagnation"][t]))
                np.testing.assert_array_equal(ca,cb)
                tv[label].append(float(abs(pa-pb).sum()/2));changed[label].append(int(ca[pa.argmax()]!=cb[pb.argmax()]))
        rows.append({"instance":t,"deposit_AD_L1_over_budget":ad,"tau_phases":phases,
            "cpu_fp64_base_tv":float(np.mean(tv["base"])),"cpu_fp64_full_gp_tv":float(np.mean(tv["full_gp"])),
            "cpu_fp64_base_argmax_change":float(np.mean(changed["base"])),
            "cpu_fp64_full_gp_argmax_change":float(np.mean(changed["full_gp"]))})
    return {"rows":rows,"probability_scope":"CPU FP64 固定上下文参考核，包含 TR；不是 GPU bitwise 实测概率",
        "budget_scope":"来源边无向聚合 L1/B；信息素矩阵按全部非对角有向边统计"}


def fork(job,snapshot_name,owner,champion,output=None):
    from rmtgp_aco.aco_cuda import solve_population_cuda_anytime
    from rmtgp_aco.mechanisms import SolverControl,InstrumentationConfig,MechanismConfig
    import cupy as cp
    snapshot,spec,record=read_owner_snapshot(job,snapshot_name,owner)
    if spec["source"]!=source_manifest()["source_hash"]:raise ValueError("必须使用原任务不可变源码执行分叉")
    task=spec["task"]
    if owner and spec["models"][owner]["seed"]!=champion:raise ValueError("冠军 owner 只与其自身配对；baseline owner 才与全部冠军配对")
    out=Path(job).parents[1];problem=batch(task["split"],task["indices"],out)
    aco,runtime=experiment(task["variant"]);mechanism=MechanismConfig(**task["mechanism"])
    entry=next(e for e in program_entries(task["variant"]) if e["seed"]==champion);tr,ph=entry["program"]
    entries={"baseline":((None,None),mechanism),"tr_only":((tr,None),mechanism),
        "ph_only":((None,ph),mechanism),"full":((tr,ph),mechanism),
        "toggle_R":((tr,ph),replace(mechanism,restart_policy="off" if mechanism.restart_policy=="native" else "native")),
        "toggle_F":((tr,ph),replace(mechanism,floor_scale=0. if mechanism.floor_scale else 1.)),
        "toggle_H":((tr,ph),replace(mechanism,source_policy="iteration_best" if mechanism.source_policy=="native_schedule" else "native_schedule"))}
    first=snapshot["iteration"]+(snapshot["phase"]=="iteration_end");stop=first+499
    if stop>aco.iterations:raise ValueError("分叉越过原始 horizon")
    target=Path(output) if output else out/"forks"/Path(job).name/f"owner{owner}-champion{champion}"/Path(snapshot_name).stem
    target.mkdir(parents=True,exist_ok=True)
    states={};arrays={};aliases={};cache={}
    for label,(program,mechanism) in entries.items():
        identity=digest({"program":[None if p is None or p.is_exact_zero else p.expression for p in program],"mechanism":mechanism.digest})
        if identity in cache:
            other=cache[identity];aliases[label]=other;states[label]=states[other]
            arrays[label+"_length"]=arrays[other+"_length"];arrays[label+"_anytime"]=arrays[other+"_anytime"];continue
        cache[identity]=label;states[label]={}
        def observe(phase,iteration,state):
            elapsed=iteration-first+1
            if phase=="iteration_end" and elapsed in (1,25,100,500):
                keys=("pheromone_workspace","audit_tau","audit_sources","deposit_workspace","mechanism_trace",
                      "tour_workspace","pre_tour_workspace","stagnation","global_best_lengths")
                states[label][elapsed]={k:cp.asnumpy(state[k]) for k in keys}
                states[label][elapsed]["iteration"]=iteration
        result=solve_population_cuda_anytime(problem,aco,[program],seed=spec["seed"],runtime=runtime,
            control=SolverControl(mechanism,InstrumentationConfig(profile="mechanism_v3",schema_version=3),
                resume=snapshot,stop_iteration=stop,observer=observe))
        arrays[label+"_length"]=result.best_length.numpy();arrays[label+"_anytime"]=result.anytime_best.numpy()[:,:,first-1:stop]
        for elapsed,state in states[label].items():
            atomic_npz(target/f"{label}-{elapsed:03d}.npz",**{k:v for k,v in state.items() if isinstance(v,np.ndarray)})
    atomic_npz(target/"result.npz",**arrays)
    if snapshot["phase"]=="post_ls":atomic_json(target/"ph_transfer.json",immediate_transfer(states,snapshot,problem,aco,tr))
    survival=[]
    for elapsed in (1,25,100,500):
        for left,right in (("baseline","tr_only"),("ph_only","full"),("baseline","full")):
            base=states[left][elapsed];full=states[right][elapsed]
            for i in range(problem.batch_size):
                for a in range(32):
                    pre_a=tour_edges(base["pre_tour_workspace"][i,a]);pre_b=tour_edges(full["pre_tour_workspace"][i,a])
                    post_a=tour_edges(base["tour_workspace"][i,a]);post_b=tour_edges(full["tour_workspace"][i,a])
                    before=len(pre_a^pre_b);after=len(post_a^post_b)
                    survival.append({"contrast":left+"/"+right,"steps":elapsed,"instance":i,"ant":a,"pre_symmetric_difference":before,
                        "post_symmetric_difference":after,"ratio":after/before if before else None})
    atomic_json(target/"ls_survival.json",{"rows":survival,"zero_pre_difference":"undefined ratio, not zero",
        "scope":"分叉后结构差异比率；相同边集不证明相同吸引域；后续各轮已受历史反馈影响"})
    atomic_json(target/"manifest.json",{"status":"completed","source_snapshot":str(Path(job)/"diagnostics"/snapshot_name),
        "snapshot_hash":record["sha256"],"owner":owner,"champion":champion,"aliases":aliases,
        "steps":[1,25,100,500],"original_horizon":aco.iterations,"completed_at":now(),
        "scope":"固定状态短程干预；不能推断重新训练或自然中介效应"})
    return target


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("job",type=Path);p.add_argument("snapshot")
    p.add_argument("--owner",type=int,required=True);p.add_argument("--champion",type=int,required=True)
    a=p.parse_args();print(fork(a.job,a.snapshot,a.owner,a.champion))
