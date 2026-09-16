"""独立 v3 cohort；旧 v2 数据与队列保持原样，正式任务仍受验收门禁控制。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
from .common import OUT,ROOT,atomic_json,file_hash,now,read_json
from .campaign import ALLOWED_GPU_MODELS,snapshot
from .evaluate import factorial_tasks
from rmtgp_aco.mechanisms import InstrumentationConfig,factorial_conditions

V3=ROOT/"control_experiments/mmas_ls/artifacts/v3-r2"


def prepare(out=V3):
    out=Path(out)
    if (out/"queue").exists() and any((out/"queue").iterdir()):
        raise ValueError("已有 v3 队列不能覆盖；修改后请创建新的验收 cohort")
    for folder in ("inputs","manifests"):
        for src in (OUT/folder).glob("*"):
            if not src.is_file(): continue
            dest=out/folder/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and file_hash(dest)!=file_hash(src):raise ValueError("冻结数据不一致")
            if not dest.exists():shutil.copy2(src,dest)
    provenance=read_json(OUT/"validation/data_provenance.json")
    if provenance.get("status")!="passed":raise ValueError("数据验收尚未通过")
    atomic_json(out/"validation/data_provenance.json",provenance)
    atomic_json(out/"protocol/diagnostics.json",{
        "schema":3,"old_cohort":str(OUT),"old_scope":"P0 复现及工程验收，不是完整机制记录",
        "new_scope":"完整诊断重新运行，不回填旧任务的不可观测状态", "created_at":now(),
        "formal_gate":"诊断完整验收与 P0 科学审阅均通过后才放行 P1"})
    atomic_json(out/"validation/diagnostics.json",{"status":"pending","complete_acceptance":False,
        "required":["三种允许型号的 TSP500 配对验收","32 实例、重排及分块等价",
                    "所有 terminal 的独立输入重算","故障恢复及受控退化分支",
                    "正式 batch 开销与磁盘预算","重启快照及同状态分叉验收"],
        "rule":"逐项附证据；不能因 pilot 进程结束自动通过"})
    destination,source=snapshot(out)
    tasks=[]
    for model,label in zip(ALLOWED_GPU_MODELS,("a5000","rtx4000ada","a4000")):
        tasks.append({"id":f"diagnostic-pilot-{label}","kind":"diagnostic_pilot","stage":"D0",
            "split":"diagnosis_dev","instances":2,"steps":100,"required_gpu_model":model})
    for label,instances,steps in (("batch32",32,100),("checkpoint500",2,500)):
        tasks.append({"id":f"diagnostic-{label}-a5000","kind":"diagnostic_pilot","stage":"D0",
            "split":"diagnosis_dev","instances":instances,"steps":steps,"required_gpu_model":ALLOWED_GPU_MODELS[0]})
    inst=asdict(InstrumentationConfig(profile="mechanism_v3",schema_version=3))
    for task in factorial_tasks():
        tasks.append({**task,"instrumentation":inst})
    # 与正式 dev 使用相同实例和 seed；重型子集不是额外独立统计样本。
    heavy={**inst,"level":"heavy","snapshot_iterations":[1,100,250,500,1000,2500]}
    for rep in range(3):
        for condition,mechanism in factorial_conditions().items():
            tasks.append({"id":f"heavy-{condition}-s{rep}","stage":"P3","kind":"mechanism_heavy",
                "split":"diagnosis_dev","indices":list(range(8)),"replicate":rep,"condition":condition,
                "variant":"mmas","mechanism":asdict(mechanism),"iterations":5000,"modes":["full"],
                "instrumentation":heavy})
    for order,task in enumerate(tasks):
        atomic_json(out/"queue"/(task["id"]+".json"),{"task":task,"order":order,
            "snapshot":str(destination),"source_hash":source["source_hash"]})
    atomic_json(out/"protocol/queue_freeze.json",{"source":source,"snapshot":str(destination),
        "task_count":len(tasks),"heavy_logical_solves":8*3*8*4,"created_at":now(),
        "stages":["D0","P1","P3-heavy"],"confirmation_frozen":False})
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=V3);a=p.parse_args();print(prepare(a.output))
