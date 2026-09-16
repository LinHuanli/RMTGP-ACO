"""完整诊断产物的独立核验与描述性表图；不自动产生因果结论。"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from .common import atomic_json,file_hash,read_json
from .diagnostics import PH_FIELDS,TR_FIELDS,validate_sample
from .probes import tour_edges,tour_hash


def edge_fingerprint(tour):
    """独立 CPU 重算 CUDA 双校验码，并用 SHA256 规范边表检查别名。"""
    n=len(tour)-1; mask=(1<<64)-1
    def mix(v):
        v=(int(v)+0x9E3779B97F4A7C15)&mask
        v=((v^(v>>30))*0xBF58476D1CE4E5B9)&mask
        v=((v^(v>>27))*0x94D049BB133111EB)&mask
        return v^(v>>31)
    a=b=0
    for u,v in zip(tour[:-1],tour[1:]):
        code=min(int(u),int(v))*n+max(int(u),int(v))
        a^=mix(code); b=(b+mix(code^0xD6E8FEB86659FD93))&mask
    return np.array([a,b],dtype=np.uint64)


def check_program_outputs(arrays,programs,flat_indices,batch_size,tolerance=5e-5):
    """在 CPU PyTorch 重算 GP 程序；这是数学复核，不声称跨设备 bitwise 相等。"""
    import torch
    maximum=0.
    for t,flat in enumerate(flat_indices):
        tr,ph=programs[int(flat)//batch_size]
        for program,data,names,k,valid in (
            (ph,arrays["ph"][t],PH_FIELDS,12,arrays["source_valid"][t]),
            (tr,arrays["tr"][t],TR_FIELDS,16,np.isfinite(arrays["tr"][t,...,19]))):
            values=data[valid]
            if program is None or program.is_exact_zero: raw=np.zeros_like(values[...,k])
            else:
                context={name:torch.from_numpy(values[...,i].copy()).double() for i,name in enumerate(names[:k])}
                raw=program.evaluate(context).numpy()
            error=float(np.max(abs(raw-values[...,k])))
            maximum=max(maximum,error)
            np.testing.assert_allclose(raw,values[...,k],atol=tolerance,rtol=tolerance)
    return maximum


def sample_rows(data,iteration,flat_indices,seen_hashes):
    validation=validate_sample(data); rows=[]
    for t,flat in enumerate(flat_indices):
        valid=data["source_valid"][t]; ph=data["ph"][t,valid]
        source_tours=data["source_tours"][t,valid]
        hashes=data["source_hash"][t,valid]
        source_sets=[]
        for tour,h in zip(source_tours,hashes):
            np.testing.assert_array_equal(edge_fingerprint(tour),h)
            key=tuple(map(int,h)); exact=tour_hash(tour)
            if key in seen_hashes and seen_hashes[key]!=exact: raise ValueError("来源校验码碰撞")
            seen_hashes[key]=exact; source_sets.append(tour_edges(tour))
        n=source_tours.shape[-1]-1
        post=data["post_tours"][t]; pre=data["pre_tours"][t]
        ib=tour_edges(post[int(np.argmin(data["colony_lengths"][t]))]); gb=tour_edges(data["best_tours"][t])
        info=data["source_info"][t,valid]; weights=info[:,5]/info[:,5].sum(dtype=np.float64)
        entropy=[];maximum=[];fallback=[]
        for p,context in enumerate(data["context"][t]):
            prob=data["tr"][t,p,:,19]; prob=prob[np.isfinite(prob)]
            entropy.append(float(-np.sum(prob*np.log(np.maximum(prob,1e-30)))))
            maximum.append(float(prob.max())); fallback.append(int(context[6]))
        # 把多来源沉积聚合到规范无向边，再计算集中度。不能把来源槽当不同边。
        total={};zero={}
        for tour,deposits,budget in zip(source_tours,data["deposit"][t,valid],info[:,5]):
            for u,v,w in zip(tour[:-1],tour[1:],deposits):
                key=(min(int(u),int(v)),max(int(u),int(v)))
                total[key]=total.get(key,0.)+float(w)
                zero[key]=zero.get(key,0.)+float(budget)/n
        mass=np.asarray(list(total.values()));normalized=mass/mass.sum()
        rows.append({"flat_index":int(flat),"iteration":iteration,
            "ph_saturation_end":float((abs(ph[...,13])>=.99).mean()),
            "ph_raw_mean_end":float(ph[...,12].mean()),"ph_raw_std_end":float(ph[...,12].std()),
            "ph_factor_std_end":float(ph[...,14].std()),
            "source_unique_end":len(set(tuple(h) for h in hashes)),
            "source_age_weighted_end":float(weights@(iteration-info[:,3])),
            "source_ib_overlap_end":float(weights@np.array([len(s&ib)/n for s in source_sets])),
            "source_gb_overlap_end":float(weights@np.array([len(s&gb)/n for s in source_sets])),
            "post_unique_edge_sets_end":len(set(tour_hash(tour) for tour in post)),
            "pre_post_retention_end":float(np.mean([len(tour_edges(a)&tour_edges(b))/n for a,b in zip(pre,post)])),
            "deposit_undirected_cv_end":float(mass.std()/mass.mean()),
            "deposit_effective_edge_count_end":float(1/(normalized**2).sum()),
            "deposit_ad_same_sources_budget":sum(abs(v-zero[k]) for k,v in total.items())/mass.sum(),
            "tr_entropy_probe_end":float(np.mean(entropy)),"tr_maxp_probe_end":float(np.mean(maximum)),
            "tr_greedy_fallback_probe_end":float(np.mean(fallback)),
            "tau_floor_fraction_iteration":float(data["trace"][t,6]/data["candidate_arc_count"][t]),
            "tau_relative_l1_after_deposit":float(data["tau_relative_l1"][t,2]),
            "tau_above_nominal_fraction_after_deposit":float(data["tau_above_nominal_max"][t,2]/(n*(n-1)))})
    return rows,validation


def summarize(directory):
    directory=Path(directory);index=read_json(directory/"index.json")
    if not index.get("completed"): raise ValueError("诊断未完整提交；不能出完整机制报告")
    rows=[];windows=[];events=[];seen_hashes={};by_task={};checks=[]
    for name,record in sorted(index["files"].items()):
        if file_hash(directory/name)!=record["sha256"]: raise ValueError(f"文件损坏: {name}")
        meta=record["metadata"];kind=meta["kind"]
        if kind not in ("sample","iterations"): continue
        with np.load(directory/name,allow_pickle=False) as archive: data=dict(archive)
        if kind=="sample":
            values,check=sample_rows(data,meta["iteration"],meta["flat_indices"],seen_hashes)
            rows.extend(values);checks.append(check)
        else:
            for t,flat in enumerate(meta["flat_indices"]):
                by_task.setdefault(int(flat),[]).append((meta["start"],{k:v[t] for k,v in data.items()}))
    for flat,blocks in by_task.items():
        blocks.sort(key=lambda b:b[0]);data={k:np.concatenate([b[1][k] for b in blocks]) for k in blocks[0][1]}
        trace=data["trace"];h=len(trace);previous=None;run=0;switches=np.zeros(h,dtype=int);repetition=np.zeros(h,dtype=int)
        for i,(info,hashes) in enumerate(zip(data["source_info"],data["source_hash"])):
            ids=tuple(sorted(tuple(map(int,v)) for v in hashes[np.isfinite(info[:,0])]))
            same=ids==previous;run=run+1 if same else 1;switches[i]=int(previous is not None and not same)
            repetition[i]=run;previous=ids
        for lo in range(0,h,25):
            hi=min(lo+25,h);ls=data["ls"][lo:hi];moments=data["ph_moments"][lo:hi].sum(axis=(0,1))
            count=moments[0];rawmean=moments[2]/count
            windows.append({"flat_index":flat,"start":lo+1,"end":hi,
                "pre_mean":float(ls[...,0].mean()),"post_mean":float(ls[...,1].mean()),
                "ls_improved_fraction":float((ls[...,1]<ls[...,0]).mean()),
                "ph_saturation_window":float(moments[1]/count),"ph_raw_mean_window":float(rawmean),
                "ph_raw_std_window":float(np.sqrt(max(0,moments[3]/count-rawmean**2))),
                "source_switches":int(switches[lo:hi].sum()),"source_repeat_duration_end":int(repetition[hi-1]),
                "actual_uniform_count":int(data["counters"][lo:hi,3].sum()),
                "candidate_exhausted_count":int(data["counters"][lo:hi,0].sum()),
                "restart_count":int(trace[lo:hi,8].sum()),"would_restart_count":int(trace[lo:hi,9].sum()),
                "floor_clip_count":int(trace[lo:hi,6].sum()),"upper_clip_count":int(trace[lo:hi,7].sum())})
        for i in np.flatnonzero((trace[:,8]>0)|(trace[:,9]>0)):
            event={"flat_index":flat,"iteration":int(i+1),"executed":int(trace[i,8]),"would_trigger":int(trace[i,9]),
                "branch_factor":float(trace[i,10]),"stagnation":float(trace[i,15])}
            for distance in (25,100,500):
                event[f"quality_change_after_{distance}"]=float(data["anytime"][i+distance]-data["anytime"][i]) if i+distance<h else None
            events.append(event)
    target=directory/"analysis";target.mkdir(exist_ok=True)
    for name,values in (("sample_endpoints",rows),("windows",windows),("restart_events",events)):
        if values:
            with (target/f"{name}.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    atomic_json(target/"validation.json",{"status":"passed","samples":len(checks),
        "max_budget_relative_error":max(x["budget_max_relative_error"] for x in checks),
        "hashes_checked":len(seen_hashes),"scope":"采样原始记录重算；不是全部 terminal 数值 oracle 或机制因果验收"})
    plot(target,rows,windows)
    return target


def plot(target,rows,windows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    panels=((windows,"end","ph_saturation_window","PH saturation, window"),
            (rows,"iteration","source_age_weighted_end","Source age, endpoint"),
            (rows,"iteration","tr_entropy_probe_end","Actual TR entropy, probes"),
            (rows,"iteration","post_unique_edge_sets_end","Unique post-LS edge sets"),
            (rows,"iteration","tau_relative_l1_after_deposit","Tau relative L1, post-deposit"),
            (windows,"end","source_repeat_duration_end","Source repetition duration"))
    for ax,(values,x,y,title) in zip(axes.flat,panels):
        for flat in sorted(set(v["flat_index"] for v in values)):
            selected=sorted((v for v in values if v["flat_index"]==flat),key=lambda v:v[x])
            ax.plot([v[x] for v in selected],[v[y] for v in selected],alpha=.65,lw=.8)
        ax.set(title=title,xlabel="ACO iteration")
    fig.savefig(target/"mechanisms.pdf");fig.savefig(target/"mechanisms.png",dpi=160);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);a=p.parse_args()
    print(summarize(a.directory))
