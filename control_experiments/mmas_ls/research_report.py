"""一次性研究报告入口：核验、实例级配对统计、图表和中文报告。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import subprocess
import numpy as np

from .common import ROOT, atomic_json, atomic_npz, digest, file_hash, now, read_json, jsonable
from .report_inputs import checked_status, frozen_factorial_tasks, validate_manifest, validate_raw
from .statistics import CONDITIONS, classify, contrasts, simultaneous_interval, write_csv

MODES=("legacy","centered_fp32","centered_fp64")
VARIANTS=("as","mmas")
CURVE_ITERATIONS=np.array([1]+list(range(25,5001,25)))
DEFAULT_OUT=ROOT/"control_experiments/mmas_ls/artifacts/numerical-v1"
DEFAULT_REPORT=ROOT/"control_experiments/mmas_ls/reports/numerical-v1"


def point_interval(values):
    """沿用数值对照的点态 percentile 区间；不用于多重显著性筛选。"""
    values=np.asarray(values,dtype=np.float64)
    if values.shape[0]!=32 or not np.isfinite(values).all():
        raise ValueError("点态区间需要全部 32 个实例")
    rng=np.random.default_rng(2026091702)
    draws=rng.integers(0,32,size=(10000,32))
    means=values[draws].mean(axis=1)
    return np.quantile(means,[.025,.975],axis=0)


def paired_summary(gap,baseline):
    """gap、baseline 均为 [instance,aco_seed]，不得提前混合实例和种子。"""
    per_instance=(np.asarray(gap)-baseline).mean(-1)
    lo,hi=point_interval(per_instance)
    return {"mean_gap":float(np.mean(gap)),"delta_pp":float(per_instance.mean()),
            "ci_low_pp":float(lo),"ci_high_pp":float(hi),
            "wins":int((per_instance<-.01).sum()),"ties":int((abs(per_instance)<=.01).sum()),
            "losses":int((per_instance>.01).sum()),"worst10_delta_pp":float(np.sort(per_instance)[-4:].mean())}


def source_audit(out):
    """冻结源码、模型和 dev 输入独立校验；只读取本轮开发集。"""
    freeze=read_json(out/"protocol/queue_freeze.json")
    snapshot=Path(freeze["snapshot"])
    for name,expected in freeze["source"]["files"].items():
        if file_hash(snapshot/name)!=expected:
            raise ValueError(f"冻结源码损坏: {name}")
    for model in read_json(out/"manifests/checkpoints.json"):
        if file_hash(model["checkpoint"])!=model["file_hash"]:
            raise ValueError(f"冠军文件变化: {model['id']}")
    records=read_json(out/"manifests/diagnosis_dev.json")["records"]
    if len(records)!=32 or len({r["coordinate_hash"] for r in records})!=32:
        raise ValueError("开发集数量或唯一性错误")
    with np.load(out/"inputs/diagnosis_dev.npz",allow_pickle=False) as data:
        coords=data["coords"]
        ref=data["reference_tour"]
        points=coords[np.arange(32)[:,None],ref]
        lengths=np.linalg.norm(np.diff(points,axis=1),axis=-1).sum(-1)
        np.testing.assert_allclose(lengths,data["reference_length"],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(data["hashes"],[r["coordinate_hash"] for r in records])
    return freeze,records,coords


def load_quality(out):
    """验证所有正式质量结果及 N1 的三个子任务；不从目录扫描挑选结果。"""
    from rmtgp_aco.mechanisms import MechanismConfig,InstrumentationConfig
    out=Path(out)
    freeze,records,coords=source_audit(out)
    queue=[read_json(p) for p in sorted((out/"queue").glob("*.json"))]
    if Counter(q["task"]["kind"] for q in queue)!=Counter(historical_mechanism=40,numeric_pair=40,numeric_validation=18):
        raise ValueError("本报告要求本轮完整 98 个冻结任务")
    factorial=np.full((8,4,32,5),np.nan); f_auc=np.full_like(factorial,np.nan)
    f_curve=np.full((8,4,32,5,201),np.nan)
    numeric=np.full((2,3,4,32,5),np.nan); n_auc=np.full_like(numeric,np.nan)
    n_curve=np.full((2,3,4,32,5,201),np.nan)
    seen=set();times=[];inputs={};raw_rows=[];validation=[];oracle_rows=[]
    models=["baseline","81001","81002","81003"]

    def read_arm(task):
        path=out/"jobs"/task["id"]
        status=checked_status(path,("raw.npz","manifest.json","diagnostics/index.json"))
        meta=read_json(path/"manifest.json")
        validate_manifest(out,task,meta,freeze["source"]["source_hash"])
        if status["scientific_hash"]!=meta["scientific_hash"]:
            raise ValueError("完成状态与科学身份不一致")
        idx=read_json(path/"diagnostics/index.json")
        if idx["scientific_hash"]!=meta["scientific_hash"]:
            raise ValueError("诊断与质量结果身份不一致")
        inputs[task["id"]]=status["files"]
        with np.load(path/"raw.npz",allow_pickle=False) as archive:
            data={k:archive[k] for k in ("gap","length","reference","auc","anytime","tour","model_ids","modes","instance_hashes")}
        validate_raw(data,task,records)
        points=coords[np.asarray(task["indices"])[None,:,None],data["tour"]]
        exact=np.linalg.norm(np.diff(points,axis=2),axis=-1).sum(-1)
        np.testing.assert_allclose(data["length"],exact,atol=1e-10,rtol=0)
        # 恢复运行的最终 wall_seconds 只覆盖最后一次尝试；不作为完整计时。
        stages=meta.get("stage_timings",[])
        timed_flats=[i for stage in stages for i in stage["flat_indices"]]
        full=(bool(stages) and all(s["first_iteration"]==1 and s["last_iteration"]==5000 for s in stages)
              and len(timed_flats)==meta["executed_solves"] and len(set(timed_flats))==len(timed_flats))
        times.append({"task":task["id"],"cohort":"P1" if task["kind"]=="historical_mechanism" else "N1",
                      "variant":task["variant"],"condition":task["condition"],
                      "mode":task["mechanism"]["terminal_statistics"],"instances":len(task["indices"]),
                      "gpu_model":meta["backend_metrics"]["device_names"],"host":meta["host"],
                      "visible_gpu":meta["environment"]["cuda_visible_devices"],
                      "wall_seconds_last_attempt":meta["wall_seconds"],"full_horizon_timing":full,
                      "diagnostic_bytes":sum(r["compressed_bytes"] for r in idx["files"].values())})
        return data,meta

    valid_factorial={t["id"]:t for t in frozen_factorial_tasks(out,"diagnosis_dev")}
    for number,q in enumerate(queue,1):
        task=q["task"];path=out/"jobs"/task["id"]
        if q["source_hash"]!=freeze["source"]["source_hash"]:
            raise ValueError("队列源码身份不一致")
        if task["kind"]=="numeric_validation":
            status=read_json(path/"status.json")
            if status.get("status")!="completed" or status.get("validation_status")!="passed":
                raise ValueError("数值验收未通过")
            if status["specification"]["source"]["source_hash"]!=q["source_hash"]:
                raise ValueError("数值验收源码不一致")
            if task["numeric_mode"]!="legacy" and status["oracle_status"]!="passed":
                raise ValueError("稳定模式 oracle 未通过")
            item=status["rows"][0]
            validation.append({"task":task["id"],"gpu_model":task["required_gpu_model"],"variant":task["variant"],
                               "mode":task["numeric_mode"],"validation":status["validation_status"],
                               "oracle":status["oracle_status"],"instances":task["instances"],"iterations":task["steps"],
                               "off_seconds":item["wall_seconds"]["off"],"audit_seconds":item["wall_seconds"]["mechanism_v3"],
                               "audit_wall_ratio":item["wall_ratio"]})
            oracle=read_json(path/task["variant"]/"analysis/terminal_oracle.json")
            for name,values in oracle["fields"].items():
                oracle_rows.append({"task":task["id"],"field":name,**values})
            inputs[task["id"]]={"status.json":file_hash(path/"status.json"),
                               "terminal_oracle.json":file_hash(path/task["variant"]/"analysis/terminal_oracle.json")}
            continue
        if task["split"]!="diagnosis_dev" or not 0<=task["replicate"]<5:
            raise ValueError("报告不接受确认集或额外随机重复")
        if len(set(task["indices"]))!=len(task["indices"]) or any(not 0<=i<32 for i in task["indices"]):
            raise ValueError("重复或越界实例")
        rep=task["replicate"]
        if task["kind"]=="historical_mechanism":
            if valid_factorial.get(task["id"])!=task:
                raise ValueError("非法全因子任务")
            data,meta=read_arm(task)
            c=CONDITIONS.index(task["condition"])
            for j,i in enumerate(task["indices"]):
                key=("P1",c,i,rep)
                if key in seen: raise ValueError("重复全因子结果")
                seen.add(key)
                factorial[c,:,i,rep]=data["gap"][:,j];f_auc[c,:,i,rep]=data["auc"][:,j]
                f_curve[c,:,i,rep]=100*(data["anytime"][:,j,CURVE_ITERATIONS-1]/data["reference"][j]-1)
        else:
            checked_status(path,("paired.npz","comparison.json"))
            comparison=read_json(path/"comparison.json")
            if comparison["task"]!=task or set(comparison["children"])!=set(MODES):
                raise ValueError("数值对照任务不完整")
            v=VARIANTS.index(task["variant"]);arms=[];devices=[]
            with np.load(path/"paired.npz",allow_pickle=False) as pair:
                np.testing.assert_array_equal(pair["numeric_modes"],MODES)
                np.testing.assert_array_equal(pair["instance_hashes"],[records[i]["coordinate_hash"] for i in task["indices"]])
                for m,mode in enumerate(MODES):
                    child=jsonable({**task,"id":task["id"]+"--"+mode,"kind":"numeric_arm","condition":"C111",
                           "mechanism":asdict(MechanismConfig(terminal_statistics=mode)),
                           "instrumentation":asdict(InstrumentationConfig(profile="mechanism_v3",schema_version=3)),"modes":["full"]})
                    data,meta=read_arm(child);arms.append(data)
                    if comparison["children"][mode]["files"]!=inputs[child["id"]]:
                        raise ValueError("数值对照引用的子任务哈希不一致")
                    np.testing.assert_array_equal(pair["model_ids"],data["model_ids"])
                    np.testing.assert_array_equal(pair["gap"][m],data["gap"])
                    devices.append((meta["host"],meta["environment"]["cuda_visible_devices"]))
                    for j,i in enumerate(task["indices"]):
                        key=("N1",v,m,i,rep)
                        if key in seen: raise ValueError("重复数值对照结果")
                        seen.add(key)
                        numeric[v,m,:,i,rep]=data["gap"][:,j];n_auc[v,m,:,i,rep]=data["auc"][:,j]
                        n_curve[v,m,:,i,rep]=100*(data["anytime"][:,j,CURVE_ITERATIONS-1]/data["reference"][j]-1)
            if len(set(devices))!=1:
                raise ValueError("同一数值配对的最终执行设备不一致")
            for arm in arms[1:]:
                for name in ("tour","anytime","length","gap"):
                    np.testing.assert_array_equal(arm[name][0],arms[0][name][0])
            inputs[task["id"]]=read_json(path/"status.json")["files"]
        print(f"quality {number}/{len(queue)} {task['id']}",flush=True)
    if any(not np.isfinite(x).all() for x in (factorial,f_auc,f_curve,numeric,n_auc,n_curve)):
        raise ValueError("完整配对矩阵有缺失")
    for c,condition in enumerate(CONDITIONS):
        for m,model in enumerate(models):
            for i in range(32):
                for r in range(5):raw_rows.append({"cohort":"P1","variant":"mmas","condition":condition,"mode":"legacy",
                    "model":model,"instance_hash":records[i]["coordinate_hash"],"replicate":r,
                    "gap":factorial[c,m,i,r],"auc":f_auc[c,m,i,r]})
    for v,variant in enumerate(VARIANTS):
        for s,mode in enumerate(MODES):
            for m,model in enumerate(models):
                for i in range(32):
                    for r in range(5):raw_rows.append({"cohort":"N1","variant":variant,"condition":"C111","mode":mode,
                        "model":model,"instance_hash":records[i]["coordinate_hash"],"replicate":r,
                        "gap":numeric[v,s,m,i,r],"auc":n_auc[v,s,m,i,r]})
    deployment={}
    for model in read_json(out/"manifests/checkpoints.json"):
        decisions=[line.split(":",1)[1].strip() for line in model["expression"].splitlines() if line.startswith("final_deployed:")]
        if len(decisions)!=1 or decisions[0] not in ("True","False"):
            raise ValueError("冻结冠军缺失明确部署决策")
        deployment[model["id"]]=decisions[0]=="True"
    return {"factorial":factorial,"f_auc":f_auc,"f_curve":f_curve,"numeric":numeric,"n_auc":n_auc,"n_curve":n_curve}, {
        "inputs":inputs,"source":freeze["source"]["source_hash"],"frozen_commit":freeze["source"]["commit"],
        "times":times,"validation":validation,"oracle_rows":oracle_rows,"raw_rows":raw_rows,"deployment":deployment}


def quality_tables(arrays):
    gaps=arrays["factorial"];numeric=arrays["numeric"]
    delta=(gaps[:,1:]-gaps[:,:1]).mean(axis=(1,3)).T
    names,matrix=contrasts();mean,low,high,se,critical=simultaneous_interval(delta@matrix.T)
    effects=[{"contrast":name,"mean_pp":mean[i],"ci_low_pp":low[i],"ci_high_pp":high[i],
              "standard_error":se[i],"excludes_zero":bool(low[i]>0 or high[i]<0),
              "practical_class":classify(low[i],high[i])} for i,name in enumerate(names)]
    factorial=[];champions=[];main=[];numerical=[]
    for c,condition in enumerate(CONDITIONS):
        d=paired_summary(gaps[c,1:].mean(0),gaps[c,0])
        d.update(condition=condition,baseline_gap=float(gaps[c,0].mean()),
                 baseline_change_pp=float((gaps[c,0]-gaps[0,0]).mean()),
                 gp_change_pp=float((gaps[c,1:]-gaps[0,1:]).mean()),
                 baseline_auc=float(arrays["f_auc"][c,0].mean()),gp_auc=float(arrays["f_auc"][c,1:].mean()),
                 ci_low_pp=low[10+c],ci_high_pp=high[10+c])
        factorial.append(d)
    for p in range(3):
        m,l,h,_,_=simultaneous_interval((gaps[:,p+1]-gaps[:,0]).mean(-1).T@matrix.T)
        champions.extend({"champion":str(81001+p),"contrast":name,"mean_pp":m[i],"ci_low_pp":l[i],"ci_high_pp":h[i],
                          "family":"separate 18 contrasts for this champion"} for i,name in enumerate(names))
    for v,variant in enumerate(VARIANTS):
        for s,mode in enumerate(MODES):
            for m,label in enumerate(("baseline","81001","81002","81003","GP_mean")):
                g=numeric[v,s,m] if m<4 else numeric[v,s,1:].mean(0)
                ref=numeric[v,0,m] if m<4 else numeric[v,0,1:].mean(0)
                row={"variant":variant,"mode":mode,"model":label,**paired_summary(g,numeric[v,s,0]),
                     "mean_auc":float(arrays["n_auc"][v,s,m].mean() if m<4 else arrays["n_auc"][v,s,1:].mean()),
                     "interval_scope":"pointwise descriptive; not multiplicity adjusted"}
                main.append(row)
                effect=(g-ref).mean(-1);lo,hi=point_interval(effect)
                numerical.append({"variant":variant,"mode":mode,"model":label,"mean_gap":float(g.mean()),
                                  "change_vs_legacy_pp":float(effect.mean()),"ci_low_pp":float(lo),"ci_high_pp":float(hi)})
    return {"factorial":factorial,"contrasts":effects,"per_champion":champions,"main_results":main,
            "numerical_effects":numerical,"max_t_critical":critical}


def deployment_table(numeric,decisions):
    """依据冻结训练 gate 展示 legacy 部署，不在开发集上重新选择。"""
    rows=[]
    for v,variant in enumerate(VARIANTS):
        passed=[decisions[f"{variant}-{s}"] for s in (81001,81002,81003)]
        deployed=np.stack([numeric[v,0,p+1] if flag else numeric[v,0,0] for p,flag in enumerate(passed)])
        label="raw champions" if all(passed) else "baseline fallback" if not any(passed) else "mixed frozen gates"
        rows.append({"variant":variant,"mode":"legacy","deployment":label,"passed_champions":sum(passed),"gap":float(deployed.mean())})
    return rows


def timing_tables(times):
    groups=defaultdict(list)
    for row in times:
        groups[(row["cohort"],row["variant"],row["mode"],row["gpu_model"],row["instances"])].append(row)
    result=[]
    for key,rows in sorted(groups.items()):
        complete=[r["wall_seconds_last_attempt"] for r in rows if r["full_horizon_timing"]]
        result.append(dict(zip(("cohort","variant","mode","gpu_model","instances"),key),tasks=len(rows),
                           complete_timed_tasks=len(complete),resumed_or_incomplete_timing=len(rows)-len(complete),
                           median_seconds=float(np.median(complete)) if complete else None,
                           min_seconds=float(min(complete)) if complete else None,
                           max_seconds=float(max(complete)) if complete else None))
    return result


def aggregate_mechanisms(out, report_dir):
    from .report_diagnostics import WINDOW_FIELDS,SAMPLE_FIELDS
    windows=np.full((8,4,32,5,200,len(WINDOW_FIELDS)),np.nan)
    samples=np.full((8,4,32,5,201,len(SAMPLE_FIELDS)),np.nan);events=[]
    for task in frozen_factorial_tasks(out,"diagnosis_dev"):
        path=report_dir/".cache/mechanisms"/(task["id"]+".npz")
        meta=read_json(path.with_suffix(".json"))
        if file_hash(path)!=meta["sha256"]:raise ValueError("机制派生缓存损坏")
        c=CONDITIONS.index(task["condition"]);r=task["replicate"]
        with np.load(path,allow_pickle=False) as data:
            np.testing.assert_array_equal(data["window_fields"],WINDOW_FIELDS)
            for j,i in enumerate(task["indices"]):
                windows[c,:,i,r]=data["windows"][:,j];samples[c,:,i,r]=data["samples"][:,j]
        events.extend(meta["events"])
    if not np.isfinite(windows).all() or not np.isfinite(samples).all():
        raise ValueError("机制汇总覆盖不完整")
    rows=[];traces=[];event_summary=[]
    for c,condition in enumerate(CONDITIONS):
        for model,label in enumerate(("baseline","81001","81002","81003","GP_mean")):
            w=windows[c,model] if model<4 else windows[c,1:].mean(0)
            s=samples[c,model] if model<4 else samples[c,1:].mean(0)
            row={"condition":condition,"model":label}
            row.update({name:float(w[...,i].mean()) for i,name in enumerate(WINDOW_FIELDS)})
            row.update({name:float(s[...,i].mean()) for i,name in enumerate(SAMPLE_FIELDS)})
            # window restart_count 为窗口计数；整次求解的期望次数单独列出。
            row["restarts_per_solve"]=float(w[...,WINDOW_FIELDS.index("restart_count")].sum(-1).mean())
            rows.append(row)
            for k,iteration in enumerate(range(25,5001,25)):
                traces.append({"condition":condition,"model":label,"sampling":"window","iteration":iteration,
                               **{name:float(w[:,:,k,i].mean()) for i,name in enumerate(WINDOW_FIELDS)}})
            selected=[e for e in events if e["condition"]==condition and e["model"]==model] if model<4 else []
            if model<4:
                executed=[e for e in selected if e["executed"]]
                event_row={"condition":condition,"model":label,"events":len(executed),
                           "instances_with_events":len({e["instance"] for e in executed}),"solves_with_events":len({(e["instance"],e["replicate"]) for e in executed})}
                for offset in (25,100,500):
                    valid=[e for e in executed if e[f"change_after_{offset}_pp"] is not None]
                    # 每次求解内平均事件，再在实例内平均有事件的种子，最后平均实例。
                    by_solve=defaultdict(list)
                    for e in valid:by_solve[(e["instance"],e["replicate"])].append(e[f"change_after_{offset}_pp"])
                    by_instance=defaultdict(list)
                    for (i,r),v in by_solve.items():by_instance[i].append(np.mean(v))
                    event_row[f"valid_events_{offset}"]=len(valid)
                    event_row[f"valid_instances_{offset}"]=len(by_instance)
                    event_row[f"conditional_change_{offset}_pp"]=float(np.mean([np.mean(v) for v in by_instance.values()])) if by_instance else None
                event_summary.append(event_row)
    return {"windows":windows,"samples":samples},rows,traces,events,event_summary


def audit_validation_logs(out,report_dir,workers):
    """验收日志与正式质量日志分表核验；预热记录不进入科学样本数。"""
    from concurrent.futures import ProcessPoolExecutor
    from . import report_audit
    from .report_audit import audit_one
    code=digest({p.name:file_hash(p) for p in (Path(report_audit.__file__),Path(__file__).with_name("report_inputs.py"))})
    jobs=[]
    for path in sorted((out/"queue").glob("*.json")):
        task=read_json(path)["task"]
        if task["kind"]!="numeric_validation":continue
        jobs.append((task["id"],out/"jobs"/task["id"]/task["variant"],report_dir/".cache/audit",code))
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(audit_one,jobs))
    summary={"status":"passed","jobs":len(results),"files":sum(r["files"] for r in results),
             "bytes":sum(r["bytes"] for r in results),"results":results,
             "scope":"18 个 100 轮正式验收的诊断索引与引用文件；预热不计入质量结果"}
    atomic_json(report_dir/"validation_integrity.json",summary)
    return summary


def generate(out,report_dir,workers=4,phase="all"):
    out,report_dir=Path(out).resolve(),Path(report_dir).resolve()
    if report_dir.is_relative_to(out):raise ValueError("报告不得写入原始 cohort")
    report_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(report_dir/"report_manifest.json",{"status":"running","phase":phase,"started_at":now()})
    arrays,meta=load_quality(out)
    tables=quality_tables(arrays)
    tables["legacy_deployment"]=deployment_table(arrays["numeric"],meta["deployment"])
    tables["timing"]=timing_tables(meta["times"])
    for name in ("factorial","contrasts","per_champion","main_results","numerical_effects","legacy_deployment","timing"):
        write_csv(report_dir/(name+".csv"),tables[name])
    for name,key in (("quality_per_solve","raw_rows"),("timing_per_task","times"),("numeric_validation","validation"),("terminal_oracle","oracle_rows")):
        write_csv(report_dir/(name+".csv"),meta[key])
    atomic_json(report_dir/"statistics.json",{**tables,"instances":32,"aco_seeds":5,"fixed_champions":3,
                "epsilon_pp":.01,"bootstrap_replicates":30000,"bootstrap_seed":2026091701,
                "numeric_bootstrap_replicates":10000,"numeric_bootstrap_seed":2026091702,
                "scope":"开发集；固定冠军；全因子18项同时区间，数值对照与主表点态描述区间"})
    atomic_npz(report_dir/".cache/quality_arrays.npz",**arrays)
    atomic_json(report_dir/".cache/quality_metadata.json",meta)
    from .report_render import plot_quality, plot_mechanisms, render
    plot_quality(report_dir,arrays,tables)
    if phase=="quality":
        atomic_json(report_dir/"report_manifest.json",{"status":"quality_only","scope":"质量表图完成；不宣告完整诊断报告完成"})
        return report_dir
    from .report_audit import run as audit
    from .report_diagnostics import run as diagnostics
    audit_result=audit(out,report_dir,workers)
    validation_audit=audit_validation_logs(out,report_dir,workers)
    diagnostics(out,report_dir,workers)
    mechanism,rows,traces,events,event_summary=aggregate_mechanisms(out,report_dir)
    write_csv(report_dir/"mechanism_summary.csv",rows)
    write_csv(report_dir/"mechanism_windows.csv",traces)
    if events:write_csv(report_dir/"restart_events.csv",events)
    write_csv(report_dir/"restart_summary.csv",event_summary)
    plot_mechanisms(report_dir,mechanism)
    provenance={"status":"complete","generated_at":now(),"cohort":str(out),"source_hash":meta["source"],
                "frozen_source_commit":meta["frozen_commit"],"analysis_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                "analysis_files":{p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob("report*.py"))},
                "inputs":meta["inputs"],"tasks":98,"diagnostic_integrity":{k:v for k,v in audit_result.items() if k!="results"},
                "validation_integrity":{k:v for k,v in validation_audit.items() if k!="results"},
                "scope":"仅 numerical-v1 开发集；不自动放行任何后续科学门禁"}
    provenance["analysis_files"]["research_report.py"]=file_hash(__file__)
    provenance["analysis_files"]["statistics.py"]=file_hash(Path(__file__).with_name("statistics.py"))
    atomic_json(report_dir/"provenance.json",provenance)
    render(report_dir,tables,meta,rows,event_summary,provenance)
    atomic_json(report_dir/"report_manifest.json",{"status":"complete","files":{p.name:file_hash(p) for p in sorted(report_dir.iterdir())
                if p.is_file() and p.name not in ("report_manifest.json",) and not p.name.endswith((".log","_progress.json"))}})
    return report_dir


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--report-dir",type=Path,default=DEFAULT_REPORT)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--phase",choices=("quality","all"),default="all")
    args=parser.parse_args()
    try:
        print(generate(args.output,args.report_dir,args.workers,args.phase))
    except Exception as error:
        if not args.report_dir.resolve().is_relative_to(args.output.resolve()):
            atomic_json(args.report_dir/"report_manifest.json",{"status":"failed","error":repr(error),"failed_at":now()})
        raise
