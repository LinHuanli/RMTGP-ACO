"""数值稳定性独立队列；不覆盖旧 cohort，不依据质量挑选精度。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import numpy as np
from .common import ROOT,OUT,atomic_json,atomic_npz,read_json,file_hash,now,evaluation_seed
from .campaign import ALLOWED_GPU_MODELS,snapshot
from .evaluate import factorial_tasks,run_task
from rmtgp_aco.mechanisms import MechanismConfig,InstrumentationConfig

NUMERIC_MODES=("legacy","centered_fp32","centered_fp64")
MODEL_LABELS=dict(zip(ALLOWED_GPU_MODELS,("a5000","ada4000","a4000")))
DEFAULT_OUT=ROOT/"control_experiments/mmas_ls/artifacts/numerical-v1"


def validation_ids(model,variant,modes=NUMERIC_MODES):
    return [f"numeric-check-{MODEL_LABELS[model]}-{variant}-{mode}" for mode in modes]


def ready(task,out,gpu_model=None):
    """每种卡独立验收；A4000 忙不能阻止已验收 A5000 的开发集实验。"""
    out=Path(out)
    if read_json(out/"protocol/numerical.json",{}).get("authorized_scope")!="development_numeric_and_historical":return False
    historical=task["kind"]=="historical_mechanism"
    if historical and read_json(out/"gates/historical_development.json",{}).get("status")!="approved":return False
    modes=("legacy",) if historical else NUMERIC_MODES
    candidates=(gpu_model,) if gpu_model else ALLOWED_GPU_MODELS
    for model in candidates:
        if model not in MODEL_LABELS:continue
        ok=True
        for name in validation_ids(model,task["variant"],modes):
            status=read_json(out/"jobs"/name/"status.json",{})
            queue=read_json(out/"queue"/(name+".json"),{})
            if (status.get("status")!="completed" or status.get("validation_status")!="passed"
                or status.get("specification",{}).get("source",{}).get("source_hash")!=queue.get("source_hash")):
                ok=False;break
        if ok:return True
    return False


def prepare(out=DEFAULT_OUT,smoke=False):
    out=Path(out).resolve()
    if (out/"queue").exists():raise ValueError("数值队列已冻结；修改源代码后必须新建 cohort")
    for relative in ("inputs/diagnosis_dev.npz","manifests/diagnosis_dev.json","manifests/checkpoints.json",
                     "validation/data_provenance.json"):
        src=OUT/relative;dst=out/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    old=ROOT/"control_experiments/mmas_ls/artifacts/v3-r2"
    historical_snapshot=read_json(old/"protocol/queue_freeze.json")["snapshot"]
    atomic_json(out/"protocol/numerical.json",{
        "authorized_scope":"development_numeric_and_historical","created_at":now(),
        "modes":NUMERIC_MODES,"instances":32,"aco_seeds":5,"ants":32,"iterations":5000,
        "historical_snapshot":historical_snapshot,"confirmation_allowed":False,"retraining_allowed":False,
        "selection_rule":"先通过全部 terminal oracle、轨迹/分块/预算验收；再比较预热计时；不按质量选择数值模式",
        "interpretation":"同一已训练冠军的数值环境干预；不能解释为稳定环境重新训练的收益",
        "recording":"全部组使用 schema3 完整记录；旧文件和旧 gate 保留",
        "disk_guard":"启动任务前至少保留 150 GiB；不足时等待，不删数据、不降低采样"})
    dest,source=snapshot(out);tasks=[]
    # A5000 验收先行，其余型号一旦空闲即可补齐，互不形成全局等待。
    for model in (ALLOWED_GPU_MODELS[:1] if smoke else ALLOWED_GPU_MODELS):
        for variant in ("mmas","as"):
            for mode in NUMERIC_MODES:
                tasks.append({"id":validation_ids(model,variant,(mode,))[0],"kind":"numeric_validation",
                    "stage":"N0","split":"diagnosis_dev","variant":variant,"numeric_mode":mode,
                    "instances":2 if smoke else 32,"steps":25 if smoke else 100,"required_gpu_model":model,
                    "historical_snapshot":historical_snapshot})
    for rep in range(0 if smoke else 5):
        for start in range(0,32,8):
            for variant in ("mmas","as"):
                order=list(NUMERIC_MODES)
                np.random.default_rng(evaluation_seed("numeric-order",rep*100+start)).shuffle(order)
                tasks.append({"id":f"numeric-pair-{variant}-s{rep}-b{start:02d}","kind":"numeric_pair",
                    "stage":"N1","split":"diagnosis_dev","indices":list(range(start,start+8)),
                    "replicate":rep,"variant":variant,"numeric_order":order,"iterations":5000})
    # 仅研究原实现行为的开发集，不把有数值缺陷的 terminal 认证为理想数学输入。
    for task in ([] if smoke else factorial_tasks()):
        tasks.append({**task,"kind":"historical_mechanism","instrumentation":asdict(
            InstrumentationConfig(profile="mechanism_v3",schema_version=3)),
            "interpretation":"原实现及固定冠军的开发集行为归因；不得直接泛化到数值修正版或学习方法"})
    for order,task in enumerate(tasks):
        atomic_json(out/"queue"/(task["id"]+".json"),{"task":task,"order":order,
            "snapshot":str(dest),"source_hash":source["source_hash"]})
    atomic_json(out/"protocol/queue_freeze.json",{"created_at":now(),"source":source,"snapshot":str(dest),
        "task_count":len(tasks),"numeric_paired_logical_solves":0 if smoke else 32*5*2*4*3,
        "historical_factorial_logical_solves":0 if smoke else 32*5*8*4,"confirmation_frozen":False,
        "smoke_only":smoke})
    return out


def paired_run(task,out):
    """三种模式同一 worker 顺序执行；模型和实例并行，保留完整逐步记录。"""
    out=Path(out);target=out/"jobs"/task["id"];results={};times={};children={}
    inst=asdict(InstrumentationConfig(profile="mechanism_v3",schema_version=3))
    for mode in task["numeric_order"]:
        child={**task,"id":task["id"]+"--"+mode,"kind":"numeric_arm","condition":"C111",
               "mechanism":asdict(MechanismConfig(terminal_statistics=mode)),"instrumentation":inst,
               "modes":["full"]}
        status=run_task(child,out)
        with np.load(out/"jobs"/child["id"]/"raw.npz",allow_pickle=False) as data:
            results[mode]={name:data[name].copy() for name in ("gap","length","tour","anytime","instance_hashes","model_ids")}
        times[mode]=status["wall_seconds"];children[mode]={"id":child["id"],"files":status["files"]}
    rows=[];legacy=results["legacy"]
    for mode in NUMERIC_MODES[1:]:
        actual=results[mode]
        np.testing.assert_array_equal(actual["instance_hashes"],legacy["instance_hashes"])
        np.testing.assert_array_equal(actual["model_ids"],legacy["model_ids"])
        np.testing.assert_array_equal(actual["tour"][0],legacy["tour"][0])
        np.testing.assert_array_equal(actual["anytime"][0],legacy["anytime"][0])
        for k,model in enumerate(actual["model_ids"]):
            rows.append({"mode":mode,"model":str(model),"mean_gap":float(actual["gap"][k].mean()),
                "legacy_mean_gap":float(legacy["gap"][k].mean()),
                "paired_gap_change_pp":float((actual["gap"][k]-legacy["gap"][k]).mean()),
                "relative_to_baseline_pp":float((actual["gap"][k]-actual["gap"][0]).mean()),
                "tour_order_changed_fraction":float(np.any(actual["tour"][k]!=legacy["tour"][k],axis=-1).mean())})
    atomic_npz(target/"paired.npz",gap=np.stack([results[m]["gap"] for m in NUMERIC_MODES]),
        instance_hashes=legacy["instance_hashes"],model_ids=legacy["model_ids"],numeric_modes=np.asarray(NUMERIC_MODES))
    atomic_json(target/"comparison.json",{"task":task,"rows":rows,"wall_seconds":times,"children":children,
        "timing_scope":"完整记录和首次运行成本；不作为无审计部署计时或数值模式选择依据",
        "interpretation":"负的 paired_gap_change_pp 表示稳定输入下质量改善；不是重新训练后的收益"})
    atomic_json(target/"status.json",{"status":"completed","completed_at":now(),"rows":rows,
        "files":{name:file_hash(target/name) for name in ("paired.npz","comparison.json")}})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=DEFAULT_OUT)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args();print(prepare(a.output,a.smoke))
