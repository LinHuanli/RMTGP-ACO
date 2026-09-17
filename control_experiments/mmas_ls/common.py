"""实验的路径、原子写入、哈希与冻结模型读取。"""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("RMTGP_CONTROL_ROOT",str(CODE_ROOT)))
HERE = Path(__file__).resolve().parent
OUT = ROOT / "control_experiments/mmas_ls/artifacts/v2"
FORMAL = ROOT / "runs/tsp500-2opt-ls-v2/formal"
SEEDS = (81001,81002,81003)
BASE_COMMIT = "2b847698225e1d0fff36a25c1015a080350a7f00"


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [jsonable(v) for v in value]
    if isinstance(value,Path): return str(value)
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value


def atomic_json(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(jsonable(payload),ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def read_json(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if default is not None: return default
        raise


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def source_manifest():
    files=sorted(list((CODE_ROOT/"src/rmtgp_aco").glob("*.py"))+list((CODE_ROOT/"src/rmtgp_aco/cuda").glob("*.cu"))
                 +list(HERE.glob("*.py"))+list(HERE.glob("*.yaml"))
                 +list((HERE/"tests").glob("*.py"))
                 +([HERE/"MECHANISM_EXPLANATION.md"] if (HERE/"MECHANISM_EXPLANATION.md").exists() else []))
    hashes={str(p.relative_to(CODE_ROOT)):file_hash(p) for p in files}
    return {"source_hash":digest(hashes),"files":hashes,
            "commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True)}


def evaluation_seed(split, replicate):
    # 不包含 variant、条件、模型、GPU、batch 和执行顺序。
    return int.from_bytes(hashlib.sha256(f"mmas-ls-v1:20260917:{split}:{replicate}".encode()).digest()[:8],"little") % (2**63-1)


def models(variant):
    from rmtgp_aco.evaluation import load_champion,compile_champion
    result=[]
    for seed in SEEDS:
        run=FORMAL/"train"/variant/f"seed-{seed}"
        manifest=read_json(run/"manifest.json")
        if manifest["status"]!="completed": raise RuntimeError(f"模型未完成: {run}")
        path=run/"selected_candidate.pkl"
        champion=load_champion(path)
        result.append({"id":f"{variant}-{seed}","seed":seed,"checkpoint":str(path),
            "file_hash":file_hash(path),"structural_hash":champion.structural_hash,
            "training_git":manifest.get("source_code",manifest.get("git")),"program":compile_champion(champion),
            "expression":(run/"selected_candidate_expression.txt").read_text()})
    return result


def experiment(variant, iterations=5000, precision="fp32_fast"):
    from rmtgp_aco.spec import load_run_spec
    from rmtgp_aco.config import CudaPrecision
    spec=load_run_spec(FORMAL/"configs"/f"tsp500-2opt-ls-v2-{variant}-seed-81001.yaml")
    aco=replace(spec.experiment.aco,iterations=iterations)
    runtime=replace(spec.experiment.runtime,gpu_devices=(0,),cuda_tuning_manifest=None,
                    cuda_precision=CudaPrecision(precision),gpu_task_chunk_size=0)
    return aco,runtime


def atomic_npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary,**arrays)
    temporary.replace(path)


def validate_tours(tours,n):
    values=np.asarray(tours).reshape(-1,n+1)
    if not np.all(values[:,0]==values[:,-1]) or not np.all(np.sort(values[:,:-1],axis=1)==np.arange(n)):
        raise ValueError("返回的 tour 不是合法闭合 Hamiltonian cycle")


def environment():
    import importlib.metadata
    import platform
    import sys
    payload={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
             "packages":{p:importlib.metadata.version(p) for p in ("torch","numpy","numba","deap","cupy-cuda13x")}}
    try:
        payload["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version","--format=csv,noheader"],text=True)
    except (OSError,subprocess.CalledProcessError):payload["gpu"]=None
    payload["cuda_visible_devices"]=os.environ.get("CUDA_VISIBLE_DEVICES")
    return payload
