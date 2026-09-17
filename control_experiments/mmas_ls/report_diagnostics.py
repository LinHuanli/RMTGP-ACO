"""历史八格诊断的向量化汇总。保留实例轴，不将边或轮次当成独立样本。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from .common import atomic_json, atomic_npz, digest, file_hash, read_json
from .report_inputs import diagnostic_coverage, frozen_factorial_tasks

WINDOW_FIELDS = (
    "pre_gap", "post_gap", "ls_gain_pp", "ls_retention", "ls_improved_fraction",
    "ph_saturation", "ph_raw_std", "ph_tanh_std", "source_age", "source_switch_fraction",
    "source_repeat_duration_end", "source_ib_fraction", "source_rb_fraction", "source_gb_fraction",
    "floor_fraction", "restart_count", "would_restart_count", "ls_moves_per_ant", "ls_checks_per_ant")
SAMPLE_FIELDS = ("tr_entropy_probe", "tr_max_probability_probe", "tr_greedy_fallback_probe",
                 "deposit_ad_same_source_budget", "deposit_cv", "tau_relative_l1_after_deposit",
                 "tau_above_nominal_fraction_after_deposit")


def iteration_metrics(data, reference, previous, repetition):
    """每个 100 轮提交块切为 25 轮窗口；来源重复计数跨提交块连续。"""
    ls = data["ls"].astype(np.float64)
    info = data["source_info"]
    if info.shape[2] != 1 or not np.isfinite(info).all():
        raise ValueError("本分析只接受 MMAS 全因子的单实际来源")
    hashes = data["source_hash"][:,:,0]
    t, w = hashes.shape[:2]
    switch = np.zeros((t,w))
    duration = np.zeros((t,w))
    for i in range(w):
        same = np.all(hashes[:,i] == previous, axis=-1) if previous is not None else np.zeros(t,dtype=bool)
        switch[:,i] = ~same if previous is not None else 0
        repetition = np.where(same, repetition+1, 1)
        duration[:,i] = repetition
        previous = hashes[:,i]
    trace = data["trace"]
    counts = data["ls_counts"]
    start = data["start"]
    windows = []
    for lo in range(0,w,25):
        hi = lo+25
        x = ls[:,lo:hi]
        moments = data["ph_moments"][:,lo:hi].sum(axis=(1,2),dtype=np.float64)
        count = moments[:,0]
        if np.any(count <= 0):
            raise ValueError("PH 窗口没有实际来源边")
        pre = 100*(x[...,0]/reference[:,None,None]-1)
        post = 100*(x[...,1]/reference[:,None,None]-1)
        kinds = info[:,lo:hi,0,0]
        # 各来源类型按真正被强化的来源统计；native_source_kind 不替代它。
        rows = [pre.mean((1,2)), post.mean((1,2)), (pre-post).mean((1,2)),
                (x[...,2]/500).mean((1,2)), (x[...,1]<x[...,0]).mean((1,2)),
                moments[:,1]/count, np.sqrt(np.maximum(0,moments[:,3]/count-(moments[:,2]/count)**2)),
                np.sqrt(np.maximum(0,moments[:,5]/count-(moments[:,4]/count)**2)),
                (np.arange(start+lo,start+hi)[None,:]-info[:,lo:hi,0,3]).mean(1),
                switch[:,lo:hi].mean(1), duration[:,hi-1],
                (kinds==0).mean(1), (kinds==1).mean(1), (kinds==2).mean(1),
                (trace[:,lo:hi,6]/(500*20)).mean(1), trace[:,lo:hi,8].sum(1),
                trace[:,lo:hi,9].sum(1), counts[:,lo:hi,:,0].mean((1,2)), counts[:,lo:hi,:,1].mean((1,2))]
        windows.append(np.stack(rows,axis=-1))
    return np.stack(windows,axis=1), previous, repetition


def sample_metrics(archive):
    """固定探针的真实选择概率与同来源、同预算的 PH 沉积差。"""
    tr = archive["tr"]
    context = archive["context"]
    valid = np.isfinite(tr[...,19])
    probs = np.where(valid, tr[...,19], 0).astype(np.float64)
    if np.any(probs<0) or np.any(context[...,0]<0):
        raise ValueError("无效 TR 概率或缺失探针")
    np.testing.assert_array_equal(valid.sum(-1), context[...,7])
    np.testing.assert_allclose(probs.sum(-1), 1, atol=2e-6, rtol=0)
    entropy = -np.sum(probs*np.log(np.maximum(probs,1e-30)),axis=-1).mean(-1)
    deposit = archive["deposit"][:,0].astype(np.float64)
    if archive["source_valid"].shape[1] != 1 or not archive["source_valid"].all():
        raise ValueError("全因子 PH 来源映射不一致")
    budget = archive["source_info"][:,0,5].astype(np.float64)
    if np.any(budget<=0) or not np.isfinite(deposit).all():
        raise ValueError("PH 预算非正或沉积无效")
    np.testing.assert_allclose(deposit.sum(-1), archive["trace"][:,4], rtol=1e-5, atol=1e-9)
    n = deposit.shape[-1]
    return np.stack([entropy,probs.max(-1).mean(-1),context[...,6].mean(-1),
                     abs(deposit-budget[:,None]/n).sum(-1)/budget,
                     deposit.std(-1)/deposit.mean(-1), archive["tau_relative_l1"][:,2],
                     archive["tau_above_nominal_max"][:,2]/(n*(n-1))],axis=-1)


def summarize_one(args):
    out, task, cache_dir = args
    out, cache_dir = Path(out), Path(cache_dir)
    directory = out/"jobs"/task["id"]/"diagnostics"
    index = read_json(directory/"index.json")
    flats = diagnostic_coverage(index)
    code_hash = digest({p.name:file_hash(p) for p in (Path(__file__),Path(__file__).with_name("report_inputs.py"))})
    key = digest({"index":file_hash(directory/"index.json"),"analysis":code_hash,
                  "raw":file_hash(directory.parent/"raw.npz")})
    output = cache_dir/(task["id"]+".npz")
    meta_path = output.with_suffix(".json")
    prior = read_json(meta_path,{})
    if prior.get("key")==key and output.exists() and file_hash(output)==prior.get("sha256"):
        return {"id":task["id"],"path":str(output),"cache_reused":True}
    with np.load(directory.parent/"raw.npz",allow_pickle=False) as raw:
        alias = raw["behavior_alias"]
        reference = raw["reference"]
        curves = raw["anytime"]
    b = len(reference)
    p = int(alias.max())+1
    if flats != set(range(p*b)):
        raise ValueError("诊断执行求解覆盖与行为别名不一致")
    windows = np.full((p*b,200,len(WINDOW_FIELDS)),np.nan)
    sample_iterations = np.array([1]+list(range(25,5001,25)))
    samples = np.full((p*b,len(sample_iterations),len(SAMPLE_FIELDS)),np.nan)
    events=[]
    for shard in index["shards"]:
        records = [(name,r["metadata"]) for name,r in index["files"].items() if r["metadata"].get("shard")==shard]
        previous=None; repetition=0
        for name,meta in sorted((r for r in records if r[1]["kind"]=="iterations"),key=lambda r:r[1]["start"]):
            selected = np.array(meta["flat_indices"])
            with np.load(directory/name,allow_pickle=False) as archive:
                data={k:archive[k] for k in ("ls","ls_counts","source_info","source_hash","trace","ph_moments")}
            data["start"]=meta["start"]
            values,previous,repetition=iteration_metrics(data,reference[selected%b],previous,repetition)
            first=(meta["start"]-1)//25
            windows[selected,first:first+values.shape[1]]=values
            for row,iteration in np.argwhere((data["trace"][...,8]>0)|(data["trace"][...,9]>0)):
                flat=int(selected[row]); absolute=meta["start"]+int(iteration)
                logical=np.flatnonzero(alias==flat//b)
                for model in logical:
                    event={"task":task["id"],"condition":task["condition"],"replicate":task["replicate"],
                           "model":int(model),"instance":task["indices"][flat%b],"iteration":absolute,
                           "executed":int(data["trace"][row,iteration,8]),"would_trigger":int(data["trace"][row,iteration,9])}
                    for offset in (25,100,500):
                        event[f"change_after_{offset}_pp"]=(float(100*(curves[model,flat%b,absolute-1+offset]-curves[model,flat%b,absolute-1])/reference[flat%b])
                                                               if absolute+offset<=5000 else None)
                    events.append(event)
        for name,meta in (r for r in records if r[1]["kind"]=="sample"):
            with np.load(directory/name,allow_pickle=False) as archive:
                values=sample_metrics(archive)
            samples[np.array(meta["flat_indices"]),int(np.searchsorted(sample_iterations,meta["iteration"]))]=values
    if not np.isfinite(windows).all() or not np.isfinite(samples).all():
        raise ValueError("机制汇总存在缺失或非有限值")
    atomic_npz(output,windows=windows.reshape(p,b,200,-1)[alias],
               samples=samples.reshape(p,b,201,-1)[alias],window_fields=np.array(WINDOW_FIELDS),
               sample_fields=np.array(SAMPLE_FIELDS),sample_iterations=sample_iterations)
    atomic_json(meta_path,{"key":key,"sha256":file_hash(output),"events":events,
                         "scope":"完整每轮日志的25轮窗口；所有预定TR探针和PH末点；描述性统计"})
    return {"id":task["id"],"path":str(output),"cache_reused":False}


def run(out,report_dir,workers=4):
    tasks=frozen_factorial_tasks(out,"diagnosis_dev")
    cache=Path(report_dir)/".cache"/"mechanisms"
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(summarize_one,(out,t,cache)):t["id"] for t in tasks}
        for future in as_completed(pending):
            results.append(future.result())
            print(f"mechanisms {len(results)}/{len(tasks)} {results[-1]['id']}",flush=True)
            atomic_json(Path(report_dir)/"mechanism_progress.json",{"done":len(results),"total":len(tasks),"status":"running"})
    atomic_json(Path(report_dir)/"mechanism_progress.json",{"done":len(results),"total":len(tasks),"status":"completed"})
    return results


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    run(args.output,args.report_dir,args.workers)
