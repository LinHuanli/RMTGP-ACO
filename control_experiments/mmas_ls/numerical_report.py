"""持续汇总数值开发集；未配齐实例和 seeds 时不给出完整样本置信区间。"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from .common import atomic_json,read_json,digest,now
from .numerical_campaign import NUMERIC_MODES,DEFAULT_OUT


def summarize(out=DEFAULT_OUT):
    out=Path(out);tasks=[read_json(p)["task"] for p in (out/"queue").glob("*.json")]
    states={t["id"]:read_json(out/"jobs"/t["id"]/"status.json",{}) for t in tasks}
    signature=digest({name:{k:value.get(k) for k in ("status","completed_at","time")} for name,value in states.items()})
    old=read_json(out/"reports/numerical_summary.json",{})
    if old.get("signature")==signature:return old
    cubes={variant:np.full((3,5,4,32),np.nan) for variant in ("mmas","as")}
    counts=Counter();checks=[]
    for task in tasks:
        state=states[task["id"]];status=state.get("status","pending")
        counts[task["stage"]+":"+status]+=1
        if task["kind"]=="numeric_validation":
            checks.append({"id":task["id"],"status":status,"oracle":state.get("oracle_status"),
                "validation_status":state.get("validation_status")})
        if task["kind"]!="numeric_pair" or status!="completed":continue
        with np.load(out/"jobs"/task["id"]/"paired.npz",allow_pickle=False) as data:
            np.testing.assert_array_equal(data["numeric_modes"],NUMERIC_MODES)
            for j,i in enumerate(task["indices"]):cubes[task["variant"]][:,task["replicate"],:,i]=data["gap"][:,:,j]
    rows=[]
    for variant,cube in cubes.items():
        # 配对任务完成才进入矩阵，三种模式不存在选择性缺组。
        count=np.isfinite(cube[0,:,0,:]).sum(axis=0);used=count>0
        if not used.any():continue
        complete=bool(np.all(count==5))
        for mode_index,mode in enumerate(NUMERIC_MODES):
            for model_index in range(4):
                gap=cube[mode_index,:,model_index,:]
                delta=gap-cube[0,:,model_index,:]
                net=gap-cube[mode_index,:,0,:]
                per_instance=lambda x:np.nansum(x,axis=0)[used]/count[used]
                effect=per_instance(delta);lo=hi=None
                if complete:
                    rng=np.random.default_rng(2026091702)
                    means=effect[rng.integers(0,32,size=(10000,32))].mean(axis=1)
                    lo,hi=map(float,np.quantile(means,[.025,.975]))
                rows.append({"variant":variant,"mode":mode,"model":"baseline" if model_index==0 else str(81000+model_index),
                    "instances":int(used.sum()),"seeds_min":int(count[used].min()),"seeds_max":int(count[used].max()),
                    "complete_32x5":complete,"mean_reference_gap":float(per_instance(gap).mean()),
                    "gap_change_vs_legacy_pp":float(effect.mean()),"gap_minus_baseline_pp":float(per_instance(net).mean()),
                    "paired_change_ci95_low":lo,"paired_change_ci95_high":hi})
    report={"created_at":now(),"signature":signature,"counts":dict(counts),"checks":checks,"rows":rows,
        "scope":"开发集、固定冠军；完整 32×5 后提供实例级配对 bootstrap 点态区间，不作多重比较显著性声明"}
    folder=out/"reports";folder.mkdir(exist_ok=True)
    if rows:
        path=folder/"numeric_quality.csv";temporary=path.with_suffix(".tmp")
        with temporary.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temporary.replace(path)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
        for ax,variant in zip(axes,("mmas","as")):
            for mode in NUMERIC_MODES:
                selected=[r for r in rows if r["variant"]==variant and r["mode"]==mode]
                if selected:ax.plot([r["model"] for r in selected],[r["mean_reference_gap"] for r in selected],"o-",label=mode)
            ax.set_title(variant.upper()+" (development set)");ax.set_ylabel("Mean reference gap (%)");ax.legend()
        fig.savefig(folder/"numeric_quality.png",dpi=180);fig.savefig(folder/"numeric_quality.pdf");plt.close(fig)
    atomic_json(folder/"numerical_summary.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUT)
    a=p.parse_args();print(summarize(a.output)["counts"])
