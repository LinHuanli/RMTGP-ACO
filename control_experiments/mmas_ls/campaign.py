"""不可变源码、共享文件锁、每 GPU 单 worker 的可恢复后台队列。"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from .common import ROOT,CODE_ROOT,OUT,atomic_json,digest,file_hash,now,read_json,source_manifest

ALLOWED_GPU_MODELS=("NVIDIA RTX A5000","NVIDIA RTX 4000 Ada Generation","NVIDIA RTX A4000")


@contextmanager
def lock(path,blocking=False):
    # 当前共享盘 flock 仅在单机生效；mkdir 是跨主机原子操作。
    # 崩溃遗留锁不自动按时间抢占，须核实远程进程和子任务都已退出后恢复。
    path=Path(str(path)+".d");path.parent.mkdir(parents=True,exist_ok=True)
    while True:
        try:path.mkdir();break
        except FileExistsError:
            if not blocking:yield False;return
            time.sleep(1)
    try:
        atomic_json(path/"owner.json",{"host":socket.gethostname(),"pid":os.getpid(),"time":now()})
        yield True
    finally:
        (path/"owner.json").unlink(missing_ok=True);path.rmdir()


def snapshot(out=OUT):
    """只复制源码；数据与模型仍由冻结文件哈希校验。"""
    manifest=source_manifest(); destination=Path(out)/"snapshots"/manifest["source_hash"]
    if not (destination/"snapshot.json").exists():
        for relative in manifest["files"]:
            src=CODE_ROOT/relative; dst=destination/relative
            dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        for relative in ("control_experiments/__init__.py",):
            dst=destination/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CODE_ROOT/relative,dst)
        atomic_json(destination/"snapshot.json",manifest)
    for relative,expected in manifest["files"].items():
        if file_hash(destination/relative)!=expected: raise RuntimeError("冻结源码被修改")
    return destination,manifest


def freeze(stages,out=OUT):
    from .evaluate import factorial_tasks
    from .reproduction import tasks as reproduction_tasks
    destination,manifest=snapshot(out)
    tasks=[]
    if "P0" in stages:
        from dataclasses import asdict
        from rmtgp_aco.mechanisms import factorial_conditions
        for rep in range(3):
            for condition in ("C111","C011","C101","C110"):
                for backend in ("fp32_fast","fp32"):
                    tasks.append({"id":f"precision-{backend}-{condition}-s{rep}","kind":"precision","backend":backend,
                        "stage":"P0","split":"diagnosis_dev","indices":list(range(8)),"replicate":rep,
                        "condition":condition,"mechanism":asdict(factorial_conditions()[condition])})
        tasks.extend(reproduction_tasks())
    if "P1" in stages: tasks.extend(factorial_tasks())
    if "P2" in stages: tasks.extend(factorial_tasks("confirm_uniform"))
    if "P3" in stages or "P4" in stages:
        from .extensions import extension_tasks
        tasks.extend(t for t in extension_tasks(out) if t["stage"] in stages)
    for order,task in enumerate(tasks):
        payload={"task":task,"snapshot":str(destination),"source_hash":manifest["source_hash"],"order":order}
        path=Path(out)/"queue"/(task["id"]+".json")
        if path.exists() and read_json(path)!=payload: raise ValueError("已有队列科学配置不可覆盖；请新建 cohort")
        atomic_json(path,payload)
    atomic_json(Path(out)/"protocol/queue_freeze.json",{"stages":stages,"task_count":len(tasks),
        "source":manifest,"snapshot":str(destination),"created_at":now()})
    print(f"frozen {len(tasks)} tasks; {destination}",flush=True)


def parse_devices(output):
    """只解析用户允许的完整型号，不把 A40/PRO 4000 等型号误配进去。"""
    result=[]
    for line in output.splitlines():
        fields=line.split()
        if len(fields)>3 and fields[0]=="IDLE" and any(line.strip().endswith(m) for m in ALLOWED_GPU_MODELS):
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*",fields[1]) and fields[2].isdigit():
                result.append((fields[1],int(fields[2])))
    return list(dict.fromkeys(result))


def devices():
    """gpu-free 只报告瞬时空闲；启动前必须在远端再核验。"""
    output=subprocess.check_output(["/home/linbocheng/bin/gpu-free"],text=True,timeout=90)
    result=parse_devices(output)
    return result,output


def gpu_info(device):
    args=["nvidia-smi",f"--id={device}","--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version","--format=csv,noheader,nounits"]
    fields=[x.strip() for x in subprocess.check_output(args,text=True,timeout=15).strip().split(",")]
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True,timeout=15)
    pids=[int(row.split(",")[1]) for row in processes.splitlines() if row.split(",")[0].strip()==fields[0]]
    return {"uuid":fields[0],"name":fields[1],"memory_mib":int(fields[2]),"utilization":int(fields[3]),"driver":fields[4],"pids":pids}


def eligible(task,out):
    stage=task["stage"]
    if read_json(Path(out)/"validation/native.json",{}).get("status")!="passed": return False
    if read_json(Path(out)/"validation/data_provenance.json",{}).get("status")!="passed": return False
    if stage!="P0" and read_json(Path(out)/"gates/P0.json",{}).get("status")!="approved": return False
    if stage in ("P2","P3","P4") and read_json(Path(out)/"gates/P1.json",{}).get("status")!="approved": return False
    if stage in ("P2","P3","P4") and not (Path(out)/"protocol/extensions_frozen.json").exists(): return False
    if not (Path(out)/f"inputs/{task['split']}.npz").exists(): return False
    if task.get("replay_from") and read_json(Path(out)/"jobs"/task["replay_from"]/"status.json",{}).get("status")!="completed": return False
    return True


def worker(device,out=OUT):
    """遇到外部占用让出 GPU；只终止自己启动的子进程，不操作其他用户。"""
    out=Path(out); initial=gpu_info(device)
    if initial["name"] not in ALLOWED_GPU_MODELS or initial["pids"] or initial["memory_mib"]>1024 or initial["utilization"]>5:
        raise RuntimeError(f"设备不是允许型号的空闲卡: {initial}")
    label=f"{socket.gethostname()}-{initial['uuid']}"
    with lock(out/"locks"/(label+".lock")) as acquired:
        if not acquired: raise RuntimeError("本实验已占用此 GPU")
        atomic_json(out/"workers"/(label+".json"),{"pid":os.getpid(),"device":initial,"status":"started","time":now()})
        while True:
            if (out/"STOP").exists(): return
            work=False
            for path in sorted((out/"queue").glob("*.json"),key=lambda p:read_json(p)["order"]):
                payload=read_json(path); task=payload["task"]; folder=out/"jobs"/task["id"]
                status=read_json(folder/"status.json",{})
                if status.get("status")=="completed" or not eligible(task,out): continue
                with lock(out/"locks/tasks"/(task["id"]+".lock")) as claimed:
                    if not claimed: continue
                    if read_json(folder/"status.json",{}).get("status")=="completed": continue
                    attempt=read_json(folder/"attempts.json",{"count":0})
                    if attempt["count"]>=3: continue
                    info=gpu_info(device)
                    if info["uuid"]!=initial["uuid"] or info["pids"]: return
                    attempt={"count":attempt["count"]+1,"host":socket.gethostname(),"gpu":info,"started_at":now()}
                    atomic_json(folder/"attempts.json",attempt)
                    atomic_json(folder/"status.json",{"status":"running","started_at":now(),"host":socket.gethostname(),"device":initial["uuid"]})
                    atomic_json(out/"workers"/(label+".json"),{"pid":os.getpid(),"status":"running","task":task["id"],"time":now()})
                    for relative,expected in read_json(Path(payload["snapshot"])/"snapshot.json")["files"].items():
                        if file_hash(Path(payload["snapshot"])/relative)!=expected: raise RuntimeError("快照文件变化")
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=initial["uuid"],RMTGP_CONTROL_ROOT=str(ROOT),
                        PYTHONPATH=str(Path(payload["snapshot"])/"src"),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMBA_NUM_THREADS="16")
                    cmd=[str(ROOT/".venv/bin/python"),"-m","control_experiments.mmas_ls.campaign","execute","--task",str(path),"--output",str(out)]
                    with (folder/f"attempt-{attempt['count']}.log").open("w") as log:
                        child=subprocess.Popen(cmd,cwd=payload["snapshot"],env=env,stdout=log,stderr=subprocess.STDOUT)
                        print(f"start {task['id']} pid={child.pid}",flush=True)
                        conflict=False
                        while child.poll() is None:
                            time.sleep(10)
                            try:
                                active=gpu_info(device)
                                unavailable=any(pid!=child.pid for pid in active["pids"])
                            except (OSError,ValueError,subprocess.SubprocessError):
                                # 监控失效时保守退出，不能留下无人监管的 GPU 子进程。
                                unavailable=True
                            if unavailable or (out/"STOP").exists():
                                conflict=True;child.terminate()
                                try:child.wait(timeout=20)
                                except subprocess.TimeoutExpired:child.kill();child.wait()
                        if conflict:
                            atomic_json(folder/"status.json",{"status":"resource_paused","time":now()})
                            # 外部资源变化不消耗数值失败重试次数。
                            attempt["count"]-=1;atomic_json(folder/"attempts.json",attempt)
                            return
                        if child.returncode:
                            atomic_json(folder/"status.json",{"status":"failed","returncode":child.returncode,"time":now()})
                    work=True
                    atomic_json(out/"workers"/(label+".json"),{"pid":os.getpid(),"status":"idle","last_task":task["id"],"time":now()})
            if not work:
                atomic_json(out/"workers"/(label+".json"),{"pid":os.getpid(),"status":"finished_available_queue","time":now()})
                return


def launch_device(host,gpu,out=OUT):
    """远程 nohup worker，返回启动 PID；真正领取仍受跨主机目录锁保护。"""
    out=Path(out).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*",host) or gpu<0:raise ValueError("无效设备地址")
    log=out/"logs"/f"{host}-gpu{gpu}-{time.time_ns()}.log";log.parent.mkdir(parents=True,exist_ok=True)
    command=f"cd {shlex.quote(str(ROOT))} && (nohup env PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {shlex.quote(str(ROOT/'.venv/bin/python'))} -m control_experiments.mmas_ls.campaign worker --device {gpu} --output {shlex.quote(str(out))} > {shlex.quote(str(log))} 2>&1 < /dev/null & echo $!)"
    result=subprocess.check_output(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",host,command],text=True,timeout=30)
    pid=int(result.strip().splitlines()[-1])
    record={"host":host,"gpu":gpu,"pid":pid,"log":str(log),"time":now()}
    print(f"launched {host} GPU{gpu} pid={pid}: {log}",flush=True)
    return record


def launch(out=OUT):
    available,listing=devices(); out=Path(out)
    atomic_json(out/"protocol/gpu_discovery.json",{"time":now(),"output":listing,"selected":available})
    for host,gpu in available:
        launch_device(host,gpu,out)


def status(out=OUT):
    counts={};seconds=[]
    for path in (Path(out)/"queue").glob("*.json"):
        task=read_json(path)["task"];s=read_json(Path(out)/"jobs"/task["id"]/"status.json",{})
        name=s.get("status","pending" if eligible(task,out) else "gated")
        counts[task["stage"]+":"+name]=counts.get(task["stage"]+":"+name,0)+1
        if s.get("wall_seconds"):seconds.append(s["wall_seconds"])
    return {"counts":counts,"completed_search_seconds":sum(seconds),"workers":[read_json(p) for p in (Path(out)/"workers").glob("*.json")]}


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=("freeze","launch","worker","execute","status"))
    p.add_argument("--output",type=Path,default=OUT);p.add_argument("--stages",nargs="+",default=["P0","P1"])
    p.add_argument("--device",type=int);p.add_argument("--task",type=Path)
    a=p.parse_args()
    if a.action=="freeze":freeze(a.stages,a.output)
    elif a.action=="launch":launch(a.output)
    elif a.action=="worker":worker(a.device,a.output)
    elif a.action=="status":print(status(a.output))
    else:
        from .evaluate import run_task
        try:
            task=read_json(a.task)["task"]
            if task.get("kind")=="precision":
                from .precision import run
                run(task["backend"],task["condition"],task["replicate"],a.output)
                result=read_json(a.output/"precision"/f"{task['backend']}-{task['condition']}-s{task['replicate']}"/"status.json")
                atomic_json(a.output/"jobs"/task["id"]/"status.json",result)
            else:print(run_task(task,a.output),flush=True)
        except BaseException:
            traceback.print_exc();raise


if __name__=="__main__":main()
