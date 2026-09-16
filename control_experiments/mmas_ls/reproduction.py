"""保留历史实例 ID 和原 seeds，复核旧结果；不参与新确认性统计。"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,atomic_json,atomic_npz,digest,file_hash,models,read_json
from .prepare import record_instance


def prepare_reproduction(out=OUT):
    from rmtgp_aco.data import build_line_offsets,load_indexed_instances
    out=Path(out)
    for distribution in ("uniform","cluster","gaussian"):
        historical=ROOT/f"runs/tsp500-2opt-ls-v2/final-test/shards/tsp500_{distribution}-mmas-seed-00.npz"
        with np.load(historical,allow_pickle=False) as saved:
            ids=saved["instance_ids"].tolist(); coords=saved["coordinate_hashes"].tolist()
        instances=[]; records=[]
        for identifier,coord in zip(ids,coords):
            source,row=identifier.rsplit(":",1);path=Path(source)
            if not path.exists(): path=ROOT/"Datasets/TSP"/source.split("Datasets/TSP/",1)[1]
            obj=load_indexed_instances(path,build_line_offsets(path),[int(row)-1])[0]
            if obj.coordinate_hash!=coord: raise ValueError("原历史数据已变化")
            record=record_instance(obj,path,int(row),file_hash(path)); record["id"]=identifier
            instances.append(obj); records.append(record)
        split=f"reproduction_{distribution}"
        payload={"split":split,"distribution":distribution,"records":records,"role":"historical_only"}
        payload["manifest_hash"]=digest(payload)
        atomic_json(out/f"manifests/{split}.json",payload)
        atomic_npz(out/f"inputs/{split}.npz",coords=np.stack([i.coords for i in instances]),
            reference_tour=np.stack([i.reference_tour for i in instances]),
            reference_length=np.array([i.reference_length for i in instances]),ids=np.array(ids),hashes=np.array(coords))


def tasks():
    from rmtgp_aco.mechanisms import MechanismConfig
    result=[]
    for distribution in ("uniform","cluster","gaussian"):
        for variant in ("as","mmas"):
            expected=[m["structural_hash"] for m in models(variant)]
            for rep in range(3):
                original=ROOT/f"runs/tsp500-2opt-ls-v2/final-test/shards/tsp500_{distribution}-{variant}-seed-{rep:02d}.npz"
                with np.load(original,allow_pickle=False) as saved:
                    if list(saved["candidate_hashes"])!=expected: raise ValueError("历史冠军哈希与当前冻结冠军不同")
                    seed=int(saved["seed"])
                result.append({"id":f"reproduction-{distribution}-{variant}-s{rep:02d}","stage":"P0",
                    "split":f"reproduction_{distribution}","variant":variant,"indices":list(range(32)),
                    "replicate":rep,"seed":seed,"condition":"C111","mechanism":asdict(MechanismConfig()),
                    "iterations":5000,"modes":["full"],"instrumentation":"light", "historical_file":str(original),
                    "historical_sha256":file_hash(original)})
    return result


def compare(out=OUT):
    """硬件改变允许路径分叉；保存逐实例偏差，不擅自设质量验收阈值。"""
    rows=[];details={}
    for task in tasks():
        folder=Path(out)/"jobs"/task["id"]
        if read_json(folder/"status.json",{}).get("status")!="completed":
            return {"status":"pending","waiting_for":task["id"]}
        with np.load(task["historical_file"],allow_pickle=False) as old,np.load(folder/"raw.npz",allow_pickle=False) as new:
            before=np.vstack([old["baseline_2opt_final_gap"],old["candidate_final_gap"]])
            diff=new["gap"]-before
            details[task["id"]]=diff
            rows.append({"task":task["id"],"variant":task["variant"],"distribution":task["split"],
                "old_baseline_gap":float(before[0].mean()),"new_baseline_gap":float(new["gap"][0].mean()),
                "old_gp_net_pp":float((before[1:]-before[:1]).mean()),
                "new_gp_net_pp":float((new["gap"][1:]-new["gap"][:1]).mean()),
                "per_instance_max_absolute_shift_pp":float(np.max(np.abs(diff))),
                "mean_shift_pp":diff.mean(axis=1).tolist()})
    report={"status":"review_required","rows":rows,"old_tours_available":False,
        "limitation":"旧文件未保存 tours，不能声称逐路径复现；跨硬件质量差须人工评估。原部署 gate 与固定 raw 冠军分开。"}
    atomic_json(Path(out)/"validation/reproduction.json",report)
    atomic_npz(Path(out)/"validation/reproduction_differences.npz",**details)
    return report
