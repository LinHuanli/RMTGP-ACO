"""在原实现保存的同一状态上重算 PH；不把不同轨迹的差异当作直接效应。"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from .common import atomic_json,read_json,experiment,now
from .diagnostics import PH_FIELDS,validate_completed
from .terminal_oracle import ph_reference
from .evaluate import program_entries


def analyze(directory,variant,instances):
    from rmtgp_aco.aco_cuda import _active_and_representative_programs
    directory=Path(directory);index=validate_completed(directory)
    specification=read_json(directory/"specification.json")
    mechanism=specification.get("task",{}).get("mechanism",{})
    aco,_=experiment(variant);entries=program_entries(variant);programs=[e["program"] for e in entries]
    _,_,_,_,representatives,_=_active_and_representative_programs(programs,aco)
    executed=[entries[i] for i in representatives];geometries={};rows=[]
    for name,record in index["files"].items():
        meta=record["metadata"]
        if meta["kind"]!="inputs":continue
        geo={}
        if meta.get("geometry_file"):
            with np.load(directory/meta["geometry_file"]) as data:geo.update(dict(data))
        with np.load(directory/name) as data:geo.update(dict(data))
        geometries[meta["shard"]]=geo
    for name,record in index["files"].items():
        meta=record["metadata"]
        if meta["kind"]!="sample":continue
        with np.load(directory/name) as archive:data=dict(archive)
        for t,flat in enumerate(meta["flat_indices"]):
            entry=executed[int(flat)//instances];program=entry["program"][1]
            valid=data["source_valid"][t];actual=data["ph"][t,valid]
            reference=ph_reference(data,geometries[meta["shard"]],t,meta["iteration"],aco,mechanism)
            def evaluate(values):
                if program is None or program.is_exact_zero:return np.zeros(values.shape[:-1])
                context={key:torch.from_numpy(values[...,k].copy()).double() for k,key in enumerate(PH_FIELDS[:12])}
                return program.evaluate(context).numpy()
            original=evaluate(actual);corrected=evaluate(reference)
            budgets=data["source_info"][t,valid,5].astype(float);n=reference.shape[1]
            tours=data["source_tours"][t,valid].astype(np.int64)
            code=np.minimum(tours[:,:-1],tours[:,1:])*n+np.maximum(tours[:,:-1],tours[:,1:])
            def distribution(raw):
                weights=1+aco.gamma_pheromone*np.tanh(raw)
                deposits=weights/weights.sum(axis=-1,keepdims=True)*budgets[:,None]
                return np.bincount(code.ravel(),weights=deposits.ravel(),minlength=n*n)
            old_deposit=distribution(original);new_deposit=distribution(corrected)
            recorded=np.bincount(code.ravel(),weights=data["deposit"][t,valid].ravel(),minlength=n*n)
            rows.append({"iteration":meta["iteration"],"flat_index":int(flat),"model":entry["id"],
                "edge_tau_mean_gpu":float(actual[...,1].mean()),"edge_tau_mean_reference":float(reference[...,1].mean()),
                "protected_raw_change_mean_abs":float(np.abs(original-corrected).mean()),
                "ph_saturation_old":float((abs(np.tanh(original))>=.99).mean()),
                "ph_saturation_corrected":float((abs(np.tanh(corrected))>=.99).mean()),
                "deposit_AD_same_state_cpu_fp64":float(abs(new_deposit-old_deposit).sum()/budgets.sum()),
                "old_cpu_vs_recorded_deposit_AD":float(abs(old_deposit-recorded).sum()/budgets.sum())})
    target=directory/"analysis";target.mkdir(exist_ok=True)
    with (target/"numeric_same_state_ph.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={"created_at":now(),"variant":variant,"rows":rows,
        "scope":"相同来源路径、信息素、colony 和预算上的 CPU FP64 数学重算；不是稳定 CUDA 的逐位输出，不是最终质量或重训收益",
        "intervention":"同时更正全部 PH terminal 的数学值；GP 表达式和总预算固定"}
    atomic_json(target/"numeric_same_state_ph.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);p.add_argument("--variant",choices=("mmas","as"),required=True)
    p.add_argument("--instances",type=int,required=True);a=p.parse_args()
    print(analyze(a.directory,a.variant,a.instances)["scope"])
