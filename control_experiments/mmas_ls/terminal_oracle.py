"""从保存的原始输入独立重算 terminal 数学定义，不修改求解器。

采用 FP64 稳定中心化方差。与 CUDA sum(x*x)/n-mean(x)^2 的差异可能
是被诊断对象本身的数值问题。不能通过扩大阈值抹去零方差产生的伪信号。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .common import atomic_json,experiment,file_hash,read_json,now
from .diagnostics import PH_FIELDS,TR_FIELDS,validate_completed


def normalized(values):
    values=np.asarray(values,dtype=np.float64)
    mean=values.mean(axis=-1,keepdims=True)
    std=np.sqrt(((values-mean)**2).mean(axis=-1,keepdims=True))
    return np.tanh((values-mean)/(std+1e-8))


def frequency(tours,n):
    tours=tours.astype(np.int64)
    code=np.minimum(tours[:,:-1],tours[:,1:])*n+np.maximum(tours[:,:-1],tours[:,1:])
    return np.bincount(code.ravel(),minlength=n*n)


def ph_reference(data,geometry,t,iteration,aco,mechanism):
    valid=data["source_valid"][t];tours=data["source_tours"][t,valid].astype(np.int64)
    u=tours[:,:-1];v=tours[:,1:];n=tours.shape[-1]-1
    i=int(geometry["task_instance"][t]);ants=len(data["post_tours"][t])
    eta=geometry["log_heuristic"][i].astype(float);node=geometry["node_log_eta_mean"][i].astype(float)
    tau=data["source_edge_tau_before"][t,valid].astype(float)
    rank=geometry["rank"][i].astype(float)
    lengths=data["colony_lengths"][t].astype(float);source_length=data["source_info"][t,valid,2].astype(float)
    horizon=mechanism.get("terminal_normalization_horizon") or aco.iterations
    progress=2*(iteration-1)/max(horizon-1,1)-1
    if mechanism.get("terminal_clip",False):progress=np.clip(progress,-1,1)
    result=np.empty((*u.shape,12),dtype=float)
    result[...,0]=normalized(eta[u,v]-.5*(node[u]+node[v]))
    result[...,1]=normalized(np.log(np.maximum(tau,aco.epsilon_numeric)))
    result[...,2]=1-(rank[u,v]+rank[v,u]-2)/max(n-2,1)
    codes=np.minimum(u,v)*n+np.maximum(u,v)
    post=frequency(data["post_tours"][t],n);pre=frequency(data["pre_tours"][t],n)
    result[...,3]=2*post[codes]/ants-1
    result[...,4]=np.tanh((lengths.mean()-source_length)/(lengths.std()+aco.epsilon_numeric))[:,None]
    result[...,5]=progress
    result[...,6]=2*min(float(data["trace"][t,15])/horizon,1)-1
    result[...,7]=data["source_gain"][t,valid]
    result[...,8]=data["source_origin"][t,valid]
    low=float(data["trace"][t,11]);high=float(data["trace"][t,12])
    # AS 的名义 tau_max 可为 inf。按原 CUDA fmin/fmax 的 NaN 保护定义核对，
    # 不能把这个 0 解释为一个有限物理上界的余量。
    with np.errstate(invalid="ignore",divide="ignore"):
        headroom=np.fmin(1.,np.fmax(0.,(high-(1-aco.rho)*tau)/(high-low+aco.epsilon_numeric)))
    override=mechanism.get("tau_headroom_override")
    result[...,9]=headroom if override is None else override
    result[...,10]=2*pre[codes]/ants-1;result[...,11]=result[...,3]
    return result


def tr_reference(data,geometry,t,p,iteration,aco,mechanism):
    context=data["context"][t,p];values=data["tr"][t,p]
    valid=np.isfinite(values[:,19]);cities=np.flatnonzero(valid)
    current=int(context[3]);previous=int(context[4]);step=int(context[2]);count=len(cities)
    i=int(geometry["task_instance"][t]);n=geometry["distances"].shape[-1]
    tau=values[valid,8].astype(float)
    distance=geometry["distances"][i,current,cities].astype(float)
    logeta=geometry["log_heuristic"][i,current,cities].astype(float)
    baseline=values[valid,17].astype(float)
    basep=baseline/baseline.sum() if baseline.sum()>aco.epsilon_numeric else np.full(count,1/count)
    # 非 fallback 的序号来自日志中的真实 candidate-list position。
    order=np.lexsort((cities,distance)) if context[6] else np.argsort(values[valid,21])
    rank=np.empty(count);rank[order]=np.arange(count)
    entropy=-np.sum(basep*np.log(np.maximum(basep,aco.epsilon_numeric)))
    horizon=mechanism.get("terminal_normalization_horizon") or aco.iterations
    progress=2*(iteration-1)/max(horizon-1,1)-1
    if mechanism.get("terminal_clip",False):progress=np.clip(progress,-1,1)
    # 构造时的 stagnation 在更新之前；实际 terminal 同一上下文跨候选恒定。
    # 逐轮更新前的停滞值由上一轮 trace 或首次初始化确定，由调用者提供。
    before_stagnation=data["oracle_stagnation_before"][t]
    result=np.empty((count,16),dtype=float)
    result[:,0]=normalized(np.log(np.maximum(tau,aco.epsilon_numeric)))
    result[:,1]=normalized(logeta)
    result[:,2]=np.tanh(np.log(np.maximum(basep,aco.epsilon_numeric))+np.log(count))
    result[:,3]=0 if count==1 else 1-2*rank/(count-1)
    result[:,4]=-1 if count==1 else 2*entropy/np.log(count)-1
    result[:,5]=2*step/max(n-1,1)-1;result[:,6]=progress
    result[:,7]=2*min(before_stagnation/horizon,1)-1
    result[:,8]=tau;result[:,9]=distance;result[:,10]=tau.mean();result[:,11]=distance.mean()
    result[:,12]=n;result[:,13]=count
    full_rank=geometry["rank"][i].astype(float)
    result[:,14]=np.clip(1-(full_rank[current,cities]+full_rank[cities,current]-2)/(2*max(n-2,1)),0,1)
    if previous<0:result[:,15]=0
    else:
        coords=geometry["coords"][i].astype(float)
        incoming=coords[current]-coords[previous];outgoing=coords[cities]-coords[current]
        result[:,15]=np.clip((outgoing@incoming)/(np.linalg.norm(incoming)*np.linalg.norm(outgoing,axis=-1)+aco.epsilon_numeric),-1,1)
    return result,valid


def inspect(directory,atol=2e-5,rtol=2e-5):
    directory=Path(directory);index=validate_completed(directory);spec=read_json(directory/"specification.json")
    variant=spec.get("variant",spec.get("task",{}).get("variant"))
    aco,_=experiment(variant)
    mechanism=spec.get("task",{}).get("mechanism",{})
    summary={};examples=[];geometry={};trace={}
    for name,record in index["files"].items():
        meta=record["metadata"]
        if meta["kind"]=="inputs":
            shared={}
            if meta.get("geometry_file"):
                with np.load(directory/meta["geometry_file"]) as f:shared=dict(f)
            with np.load(directory/name) as f:geometry[meta["shard"]]={**shared,**dict(f)}
        elif meta["kind"]=="iterations":
            with np.load(directory/name) as f:
                for j in range(meta["end"]-meta["start"]+1):
                    trace[meta["shard"],meta["start"]+j]=f["trace"][:,j,15].copy()
    def compare(role,names,reference,actual,mask,active,meta,t,extra):
        for k,term in enumerate(names):
            key=role+"."+term;a=actual[...,k];r=reference[...,k]
            if not np.isfinite(a).all() or not np.isfinite(r).all():
                raise ValueError(f"oracle 非有限输入或结果: {key}")
            difference=abs(a-r);fail=difference>atol+rtol*abs(r)
            used=bool(active and (int(mask)&(1<<k)))
            row=summary.setdefault(key,{"count":0,"failed":0,"used_failed":0,"max_abs_error":0.})
            row["count"]+=int(a.size);row["failed"]+=int(fail.sum());row["used_failed"]+=int(fail.sum()) if used else 0
            row["max_abs_error"]=max(row["max_abs_error"],float(difference.max()))
            if fail.any() and len([e for e in examples if e["terminal"]==key and e["used_by_program"]==used])<3:
                where=np.unravel_index(int(difference.argmax()),difference.shape)
                examples.append({"terminal":key,"iteration":meta["iteration"],"flat_index":meta["flat_indices"][t],
                    "index":list(map(int,where)),"gpu":float(a[where]),"reference":float(r[where]),"used_by_program":used,**extra})
    for name,record in sorted(index["files"].items()):
        meta=record["metadata"]
        if meta["kind"]!="sample":continue
        with np.load(directory/name) as f:data=dict(f)
        t0=meta["iteration"];geo=geometry[meta["shard"]]
        data["oracle_stagnation_before"]=np.zeros(len(data["context"])) if t0==1 else trace[meta["shard"],t0-1]
        for t in range(len(data["context"])):
            valid=data["source_valid"][t]
            ph=ph_reference(data,geo,t,t0,aco,mechanism)
            raw_tau=data["source_edge_tau_before"][t,valid]
            compare("PH",PH_FIELDS[:12],ph,data["ph"][t,valid],data["required_masks"][t,1],data["program_active"][t,1],meta,t,
                {"source_tau_min":float(raw_tau.min()),"source_tau_max":float(raw_tau.max())})
            for p in range(16):
                tr,valid=tr_reference(data,geo,t,p,t0,aco,mechanism)
                compare("TR",TR_FIELDS[:16],tr,data["tr"][t,p,valid],data["required_masks"][t,0],data["program_active"][t,0],meta,t,
                    {"probe":p})
    report={"status":"passed" if not any(r["failed"] for r in summary.values()) else "mismatch",
        "reference":"CPU FP64, centered variance, saved geometric/dynamic inputs",
        "atol":atol,"rtol":rtol,"fields":summary,"examples":examples,"created_at":now(),
        "oracle_source_sha256":file_hash(Path(__file__)),"diagnostic_index_sha256":file_hash(directory/"index.json"),
        "scope":"mismatch 不自动解释为 GP 性能差的全部原因；不更改冻结求解器或阈值以隐藏伪信号"}
    atomic_json(directory/"analysis/terminal_oracle.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);a=p.parse_args()
    report=inspect(a.directory)
    print(report["status"])
    for name,row in report["fields"].items():
        if row["failed"]:print(name,row)
