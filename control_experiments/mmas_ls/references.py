"""生成独立 OOD 实例，并用独立 LKH 求可行 reference；不宣称连续距离最优。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np
from .common import OUT,atomic_json,atomic_npz,digest,file_hash,now,read_json,validate_tours


def generate(distribution,index):
    rng=np.random.default_rng(np.random.SeedSequence([20260917,110 if distribution=="cluster" else 111,index]))
    if distribution=="cluster":
        centers=rng.uniform(.1,.9,(5,2)); labels=rng.integers(5,size=500)
        coordinates=centers[labels]+rng.normal(0,.05,(500,2))
        # 拒绝越界点，不裁剪到边界，避免人为重合。
        bad=np.any((coordinates<0)|(coordinates>1),axis=1)
        while bad.any():
            coordinates[bad]=centers[labels[bad]]+rng.normal(0,.05,(int(bad.sum()),2))
            bad=np.any((coordinates<0)|(coordinates>1),axis=1)
    else:
        coordinates=rng.normal(.5,.2,(500,2));bad=np.any((coordinates<0)|(coordinates>1),axis=1)
        while bad.any():
            coordinates[bad]=rng.normal(.5,.2,(int(bad.sum()),2));bad=np.any((coordinates<0)|(coordinates>1),axis=1)
    return coordinates


def solve_reference(distribution,index,solver,out):
    from rmtgp_aco.data import coordinate_hash
    folder=Path(out)/"references"/distribution/f"{index:03d}";folder.mkdir(parents=True,exist_ok=True)
    coords=generate(distribution,index);coord_hash=coordinate_hash(coords)
    frozen={"distribution":distribution,"index":index,"coordinate_hash":coord_hash,"generator_sha256":file_hash(Path(__file__)),
        "solver_sha256":file_hash(solver),"solver":"LKH-3.0.13","runs":10,"max_trials":10000,"distance_scale":1000000,
        "lkh_seed":20260917+index,"distance_definition":"EUC_2D at scale 1e6; final reference rescore continuous FP64"}
    old=read_json(folder/"status.json",{})
    if old.get("status")=="completed":
        if old["specification"]!=frozen:raise RuntimeError("独立 reference 规格已变化，不能覆盖")
        return
    problem=folder/"problem.tsp";tour=folder/"reference.tour";parameters=folder/"run.par"
    header=["NAME : independent-reference","TYPE : TSP","DIMENSION : 500","EDGE_WEIGHT_TYPE : EUC_2D","NODE_COORD_SECTION"]
    # 以下为生成的数据文件，不是算法源码。保留小数坐标，边长才按 EUC_2D 取整。
    problem.write_text("\n".join(header+[f"{i+1} {x*1e6:.10f} {y*1e6:.10f}" for i,(x,y) in enumerate(coords)]+["EOF",""]))
    parameters.write_text(f"PROBLEM_FILE = {problem}\nOUTPUT_TOUR_FILE = {tour}\nRUNS = 10\nMAX_TRIALS = 10000\nSEED = {20260917+index}\nTRACE_LEVEL = 1\n")
    atomic_json(folder/"status.json",{"status":"running","specification":frozen,"time":now()})
    started=time.perf_counter()
    with (folder/"solver.log").open("w") as stream:
        subprocess.run([str(solver),str(parameters)],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=21600)
    tokens=tour.read_text().split("TOUR_SECTION",1)[1].split();order=[]
    for token in tokens:
        if token in ("-1","EOF"):break
        order.append(int(token)-1)
    reference=np.r_[order,order[0]].astype(np.int64);validate_tours(reference,500)
    length=float(np.linalg.norm(coords[reference[1:]]-coords[reference[:-1]],axis=1).sum(dtype=np.float64))
    atomic_npz(folder/"raw.npz",coords=coords,reference_tour=reference,reference_length=np.asarray(length))
    atomic_json(folder/"status.json",{"status":"completed","specification":frozen,"seconds":time.perf_counter()-started,
        "reference_length":length,"reference_status":"independent_feasible_not_proven_optimal","time":now(),"raw_sha256":file_hash(folder/"raw.npz")})


def run(solver,workers=4,out=OUT):
    out=Path(out);solver=Path(solver).resolve()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(solve_reference,d,i,solver,out):(d,i) for d in ("cluster","gaussian") for i in range(128)}
        for future in as_completed(pending):future.result();print("reference complete",*pending[future],flush=True)
    # 只在该分布的全部 reference 完成后发布 immutable 输入，不发布半成品确认集。
    seen=set()
    for split in ("diagnosis_dev","confirm_uniform"):
        seen.update(r["coordinate_hash"] for r in read_json(out/f"manifests/{split}.json")["records"])
    for distribution in ("cluster","gaussian"):
        records=[];coordinates=[];tours=[];lengths=[]
        for i in range(128):
            folder=out/"references"/distribution/f"{i:03d}";status=read_json(folder/"status.json")
            coord=status["specification"]["coordinate_hash"]
            if coord in seen:raise ValueError("新实例重复")
            seen.add(coord)
            with np.load(folder/"raw.npz",allow_pickle=False) as data:
                coordinates.append(data["coords"]);tours.append(data["reference_tour"]);lengths.append(float(data["reference_length"]))
                records.append({"id":"sha256:"+coord,"coordinate_hash":coord,"reference_length":lengths[-1],
                    "reference_hash":hashlib.sha256(np.asarray(tours[-1],dtype="<i8").tobytes()).hexdigest(),
                    "reference_status":"independent_LKH_feasible_not_proven_optimal","source":str(folder),
                    "reference_provenance":status["specification"]})
        split="confirm_"+distribution
        payload={"split":split,"distribution":distribution,"records":records,"root_seed":20260917,
            "generator":"5 centers U(.1,.9), weights 1/5, sigma=.05, reject outside [0,1]^2" if distribution=="cluster" else "N(.5,.2^2 I), reject outside [0,1]^2",
            "note":"新定义的明确 OOD 分布；不声称与旧文件未知生成器完全同分布"}
        payload["manifest_hash"]=digest(payload)
        atomic_npz(out/f"inputs/{split}.npz",coords=np.stack(coordinates),reference_tour=np.stack(tours),
            reference_length=np.asarray(lengths),ids=np.asarray([r["id"] for r in records]),hashes=np.asarray([r["coordinate_hash"] for r in records]))
        atomic_json(out/f"manifests/{split}.json",payload)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUT);p.add_argument("--workers",type=int,default=4)
    p.add_argument("--solver",type=Path,default=OUT/"vendor/LKH-3.0.13/LKH")
    a=p.parse_args();run(a.solver,a.workers,a.output)
