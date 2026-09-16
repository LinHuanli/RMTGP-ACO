"""按实例块配对推断；三个固定冠军与 ACO seeds 不是独立样本。"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from .common import OUT,atomic_json,read_json,now,file_hash

CONDITIONS=("C111","C110","C101","C100","C011","C010","C001","C000")


def contrasts():
    eye=np.eye(8); e={name:eye[i] for i,name in enumerate(CONDITIONS)}
    r=e["C011"]-e["C111"]; f=e["C101"]-e["C111"]; h=e["C110"]-e["C111"]
    family={"E_R":r,"E_F":f,"E_H":h,"E_R-E_F":r-f,"E_R-E_H":r-h,"E_F-E_H":f-h,
        "J_RF|H=1":e["C001"]-e["C011"]-e["C101"]+e["C111"],
        "J_RH|F=1":e["C010"]-e["C011"]-e["C110"]+e["C111"],
        "J_FH|R=1":e["C100"]-e["C101"]-e["C110"]+e["C111"],
        "J_RFH":e["C000"]-e["C001"]-e["C010"]-e["C100"]+e["C011"]+e["C101"]+e["C110"]-e["C111"]}
    family.update({"Delta_"+c:e[c] for c in CONDITIONS})
    return tuple(family),np.stack(list(family.values()))


def simultaneous_interval(values,replicates=30000,seed=2026091701,confidence=.95):
    """固定原样本 SE 的 bootstrap max-t；整行重采样保留比较间相关性。

    所有重采样共享同一实例索引。零方差列不进入 max-t，区间退化为点，
    只表示该数据集的经验不确定性，不能由此声称总体方差必为零。
    """
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all(): raise ValueError("需要至少两个完整实例")
    mean=x.mean(axis=0); se=x.std(axis=0,ddof=1)/np.sqrt(len(x))
    active=se>np.finfo(float).eps*max(1.,float(np.max(np.abs(x))))
    maxima=np.zeros(replicates); rng=np.random.default_rng(seed)
    for first in range(0,replicates,512):
        stop=min(first+512,replicates)
        idx=rng.integers(len(x),size=(stop-first,len(x)))
        centered=x[idx].mean(axis=1)-mean
        if active.any(): maxima[first:stop]=np.max(np.abs(centered[:,active]/se[active]),axis=1)
    critical=float(np.quantile(maxima,confidence,method="higher"))
    return mean,mean-critical*se,mean+critical*se,se,critical


def classify(lower,upper,epsilon=.01):
    if upper < -epsilon: return "实际负效应"
    if lower > epsilon: return "实际正效应"
    if lower >= -epsilon and upper <= epsilon: return "等效区间内"
    return "精度不足或跨阈值"


def load_factorial(out,split,expected_instances,expected_seeds):
    """缺一项就拒绝正式统计；不以可用结果替代冻结样本。"""
    from .evaluate import factorial_tasks
    data=np.full((8,4,expected_instances,expected_seeds),np.nan)
    auc=np.full_like(data,np.nan); seen=set()
    expected=read_json(Path(out)/f"manifests/{split}.json")["records"]
    for task in factorial_tasks(split,out):
        path=Path(out)/"jobs"/task["id"]
        status=read_json(path/"status.json",{})
        if status.get("status")!="completed": raise ValueError(f"缺少完成结果: {task['id']}")
        for filename,expected_hash in status["files"].items():
            if file_hash(path/filename)!=expected_hash:raise ValueError(f"缓存结果损坏: {task['id']}/{filename}")
        meta=read_json(path/"manifest.json")
        if meta["task"]!=task: raise ValueError("任务规格与冻结全因子不一致")
        with np.load(path/"raw.npz",allow_pickle=False) as arrays:
            ci=CONDITIONS.index(task["condition"]); rep=task["replicate"]
            for j,i in enumerate(task["indices"]):
                if str(arrays["instance_hashes"][j])!=expected[i]["coordinate_hash"]: raise ValueError("配对实例不一致")
                key=(ci,i,rep)
                if key in seen: raise ValueError("重复结果")
                seen.add(key); data[ci,:,i,rep]=arrays["gap"][:,j]; auc[ci,:,i,rep]=arrays["auc"][:,j]
    if not np.isfinite(data).all(): raise ValueError("存在缺失结果；禁止选择性删除")
    return data,auc


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize(split="diagnosis_dev",out=OUT):
    n,s=(32,5) if split=="diagnosis_dev" else (128,10)
    gaps,auc=load_factorial(out,split,n,s)
    delta=(gaps[:,1:]-gaps[:,:1]).mean(axis=(1,3)).T  # [instance,condition]
    names,matrix=contrasts(); values=delta@matrix.T
    mean,low,high,se,critical=simultaneous_interval(values)
    rows=[{"contrast":name,"mean_pp":mean[i],"lower_simultaneous_pp":low[i],
           "upper_simultaneous_pp":high[i],"standard_error":se[i],"interpretation":classify(low[i],high[i])}
          for i,name in enumerate(names)]
    tables=[]
    base=gaps[:,0].mean(axis=(1,2)); gp=gaps[:,1:].mean(axis=(1,2,3))
    for c,condition in enumerate(CONDITIONS):
        inst=delta[:,c]
        tables.append({"condition":condition,"baseline_gap":base[c],"gp_gap":gp[c],"delta_pp":gp[c]-base[c],
          "baseline_change_pp":base[c]-base[0],"gp_change_pp":gp[c]-gp[0],
          "baseline_auc":auc[c,0].mean(),"gp_auc":auc[c,1:].mean(),
          "win_instances":int((inst<-.01).sum()),"tie_instances":int((np.abs(inst)<=.01).sum()),
          "loss_instances":int((inst>.01).sum()),"worst10_delta_pp":np.sort(inst)[-int(np.ceil(n*.1)):].mean()})
    per_model=[]
    for p in range(3):
        v=(gaps[:,p+1]-gaps[:,0]).mean(axis=-1).T@matrix.T
        m,l,h,_,_=simultaneous_interval(v)
        per_model.extend({"champion":81001+p,"contrast":name,"mean_pp":m[i],"lower_pp":l[i],"upper_pp":h[i]} for i,name in enumerate(names))
    target=Path(out)/"reports"/split
    write_csv(target/"factorial.csv",tables); write_csv(target/"contrasts.csv",rows); write_csv(target/"per_champion.csv",per_model)
    atomic_json(target/"summary.json",{"split":split,"role":"development" if split=="diagnosis_dev" else "confirmation",
       "instances":n,"seeds":s,"fixed_champions":3,"epsilon_pp":.01,"family":18,"bootstrap_replicates":30000,
       "bootstrap_seed":2026091701,"max_t_critical":critical,"completed_at":now(),"conditions":tables,"contrasts":rows,
       "scope":"条件化于固定冠军；不能推断消融环境重新训练的收益；每冠军区间为各自单独18项族"})
    plot(target,tables,rows)
    return target


def plot(target,tables,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    pos=np.arange(8)
    axes[0].plot(pos,[x["baseline_gap"] for x in tables],"o-",label="baseline")
    axes[0].plot(pos,[x["gp_gap"] for x in tables],"s-",label="fixed GP champions")
    axes[0].set(xticks=pos,xticklabels=CONDITIONS,ylabel="Reference gap (%)"); axes[0].legend()
    subset=rows[:10]; m=np.array([x["mean_pp"] for x in subset]); l=np.array([x["lower_simultaneous_pp"] for x in subset]); h=np.array([x["upper_simultaneous_pp"] for x in subset])
    axes[1].errorbar(m,np.arange(10),xerr=np.stack([m-l,h-m]),fmt="o")
    axes[1].axvspan(-.01,.01,color="gray",alpha=.15); axes[1].axvline(0,color="black",lw=.7)
    axes[1].set(yticks=np.arange(10),yticklabels=[x["contrast"] for x in subset],xlabel="Paired effect (pp), simultaneous 95% CI")
    fig.savefig(target/"factorial.pdf"); fig.savefig(target/"factorial.png",dpi=180); plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--split",default="diagnosis_dev");p.add_argument("--output",type=Path,default=OUT)
    a=p.parse_args();print(summarize(a.split,a.output))
