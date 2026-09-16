"""核对历史使用记录，冻结独立实例和原始冠军。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .common import ROOT,HERE,OUT,SEEDS,atomic_json,atomic_npz,digest,file_hash,models,now,read_json,source_manifest


def history_usage():
    """保守纳入历史所有 JSON 的坐标哈希和 instance ID；不依据结果好坏筛选。"""
    hashes=set(); ids=set(); files=[]; errors=[]; coverage=[]; npz_files=[]
    def visit(value,key=""):
        if isinstance(value,dict):
            for k,v in value.items(): visit(v,k)
        elif isinstance(value,list):
            for v in value: visit(v,key)
        elif isinstance(value,str):
            if re.fullmatch(r"[0-9a-f]{64}",value): hashes.add(value)
            if ".txt:" in value and "Datasets/TSP/" in value:
                ids.add(value.split("Datasets/TSP/",1)[1])
    for path in sorted((ROOT/"runs").rglob("*.json")):
        if "snapshots" in path.parts: continue
        try:
            payload=read_json(path);visit(payload); files.append(str(path.relative_to(ROOT)))
            config=payload.get("configuration",{}) if isinstance(payload,dict) else {}
            if path.name=="manifest.json" and "gp" in config:
                scales=config.get("train_scales",[])
                schedule=path.parent/"schedule.json"
                item={"run":str(path.parent.relative_to(ROOT)),"train_scales":scales,"schedule_present":schedule.exists()}
                if 500 in scales:
                    if not schedule.exists():errors.append({"path":str(path),"error":"TSP500 training 缺失逐实例 schedule"})
                    else:
                        records=read_json(schedule).get("records",[])
                        relevant=[r for r in records if r.get("scale")==500]
                        item["tsp500_records"]=len(relevant)
                        if not relevant or any(not r.get("coordinate_hashes") for r in relevant):
                            errors.append({"path":str(schedule),"error":"TSP500 schedule 缺失坐标哈希"})
                coverage.append(item)
        except (ValueError,OSError) as error: errors.append({"path":str(path),"error":str(error)})
    for path in sorted((ROOT/"runs").rglob("*.npz")):
        if "snapshots" in path.parts:continue
        try:
            with np.load(path,allow_pickle=False) as arrays:
                keys=[k for k in arrays.files if "hash" in k or k=="instance_ids"]
                for key in keys:
                    try:visit(arrays[key].tolist())
                    except ValueError:
                        errors.append({"path":str(path),"error":f"历史 {key} 需要 pickle，未认证"})
            npz_files.append(str(path.relative_to(ROOT)))
        except (ValueError,OSError) as error:errors.append({"path":str(path),"error":str(error)})
    # 已有全部 validation/test 不作为新确认集，即使部分逐行结果已迁走。
    return hashes,ids,{"json_files":files,"npz_files":npz_files,"training_coverage":coverage,
        "errors":errors,"hashes":len(hashes),"instance_ids":len(ids),
        "excluded_entire_splits":["validation","test"],"policy":"历史使用记录缺失的来源池不认证为未用",
        "scope":"仓库 runs 下全部现存 JSON/NPZ 与有 manifest 的训练；不声明覆盖仓库外或被删除的记录"}


def record_instance(instance,source,line,source_hash):
    ref_hash=hashlib.sha256(np.asarray(instance.reference_tour,dtype="<i8").tobytes()).hexdigest()
    return {"id":"sha256:"+instance.coordinate_hash,"coordinate_hash":instance.coordinate_hash,
            "source":str(source),"line":int(line),"source_sha256":source_hash,
            "reference_hash":ref_hash,"reference_length":float(instance.reference_length),
            "reference_status":"feasible_reference_continuous_fp64","distance":"continuous_euclidean"}


def prepare_uniform(out):
    from rmtgp_aco.data import build_line_offsets,load_indexed_instances
    hashes,ids,audit=history_usage()
    atomic_json(out/"protocol/history_usage.json",audit)
    if audit["errors"]: raise RuntimeError("历史 JSON 有无法读取项，先修复历史清单再冻结旧数据")
    files=sorted((ROOT/"Datasets/TSP/train_dataset/tsp/tsp_500").glob("tsp500_uniform_16k_*.txt"))
    rng=np.random.default_rng(20260917)
    chosen=[]; instances=[]
    for path in files:
        offsets=build_line_offsets(path); file_digest=file_hash(path)
        for index in rng.permutation(len(offsets)):
            relative=str(path.relative_to(ROOT/"Datasets/TSP"))+f":{int(index)+1}"
            if relative in ids: continue
            instance=load_indexed_instances(path,offsets,[int(index)])[0]
            if instance.coordinate_hash in hashes: continue
            hashes.add(instance.coordinate_hash)
            chosen.append(record_instance(instance,path,int(index)+1,file_digest)); instances.append(instance)
            if len(chosen)==160: break
        if len(chosen)==160: break
    if len(chosen)<160: raise RuntimeError("可核实未用 Uniform 不足；需运行 generate-reference 补充")
    for name,start,stop in (("diagnosis_dev",0,32),("confirm_uniform",32,160)):
        payload={"split":name,"distribution":"uniform","records":chosen[start:stop],
            "root_seed":20260917,"history_audit_hash":digest(audit),"selection":"未用 training pool 留出；禁止未来再用于训练"}
        payload["manifest_hash"]=digest(payload)
        atomic_json(out/f"manifests/{name}.json",payload)
        subset=instances[start:stop]
        atomic_npz(out/f"inputs/{name}.npz",coords=np.stack([x.coords for x in subset]),
            reference_tour=np.stack([x.reference_tour for x in subset]),
            reference_length=np.asarray([x.reference_length for x in subset]),
            ids=np.asarray([r["id"] for r in chosen[start:stop]]),
            hashes=np.asarray([r["coordinate_hash"] for r in chosen[start:stop]]))


def prepare(out=OUT):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    existing=out/"manifests/checkpoints.json"
    frozen=[]
    for variant in ("as","mmas"):
        frozen.extend([{k:v for k,v in m.items() if k!="program"} for m in models(variant)])
    if existing.exists() and read_json(existing)!=frozen:
        raise RuntimeError("原 checkpoint 已变化；禁止覆盖冻结模型清单")
    atomic_json(existing,frozen)
    if not (out/"manifests/confirm_uniform.json").exists(): prepare_uniform(out)
    atomic_json(out/"protocol/source_at_prepare.json",source_manifest())
    atomic_json(out/"preparation.json",{"prepared_at":now(),"uniform_ready":True,
        "ood_ready":all((out/f"inputs/confirm_{d}.npz").exists() for d in ("cluster","gaussian"))})


def audit_existing(out=OUT):
    """扩充扫描后重新检验冻结集合；不能静默替换污染实例。"""
    hashes,ids,audit=history_usage();overlap=[]
    for split in ("diagnosis_dev","confirm_uniform"):
        for row in read_json(Path(out)/f"manifests/{split}.json")["records"]:
            identifier=row["source"].split("Datasets/TSP/",1)[1]+f":{row['line']}"
            if row["coordinate_hash"] in hashes or identifier in ids:overlap.append(row)
    audit["overlap"]=overlap;audit["status"]="passed" if not audit["errors"] and not overlap else "failed"
    atomic_json(Path(out)/"validation/data_provenance.json",audit)
    if audit["status"]!="passed":raise RuntimeError("数据使用审计失败，见 validation/data_provenance.json")
    return audit


def batch(split, indices, out=OUT, legacy=False, candidate_size=20):
    from rmtgp_aco.data import TSPInstance,make_problem_batch
    data=np.load(Path(out)/f"inputs/{split}.npz",allow_pickle=False)
    rows=[]
    for i in indices:
        rows.append(TSPInstance(str(data["ids"][i]),data["coords"][i],data["reference_tour"][i],
                               float(data["reference_length"][i]),str(data["hashes"][i])))
    return make_problem_batch(rows,candidate_size=candidate_size)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args(); prepare(args.output)
