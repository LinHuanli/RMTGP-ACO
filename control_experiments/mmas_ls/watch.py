"""持续发现空闲 GPU 并补充 worker；不修改科学队列或阶段验收。"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from .common import ROOT,OUT,atomic_json,now,read_json
from .campaign import ALLOWED_GPU_MODELS,devices,eligible,launch_device,lock,affinity_path


def runnable_tasks(out,gpu_model=None):
    """排除已完成、正在领取、达到失败上限和未过验收的任务。"""
    tasks=[];out=Path(out)
    for path in (out/"queue").glob("*.json"):
        task=read_json(path)["task"];folder=out/"jobs"/task["id"]
        if gpu_model is not None and task.get("required_gpu_model",gpu_model)!=gpu_model:continue
        affinity=affinity_path(task,out)
        if gpu_model is not None and affinity and read_json(affinity,{}).get("gpu_model",gpu_model)!=gpu_model:continue
        if read_json(folder/"status.json",{}).get("status")=="completed":continue
        if (out/"locks/tasks"/(task["id"]+".lock.d")).exists():continue
        if read_json(folder/"attempts.json",{"count":0})["count"]>=3:continue
        allowed=(eligible(task,out,gpu_model) if task.get("kind") in
                 ("numeric_validation","numeric_pair","historical_mechanism") else eligible(task,out))
        if allowed:tasks.append(task["id"])
    return tasks


def scan(out,last_launch,cooldown=300):
    out=Path(out);available,listing=devices()
    atomic_json(out/"monitor/discovery.json",{"time":now(),"output":listing,"idle_allowed":available})
    count=len(runnable_tasks(out));launched=[];errors=[];remaining=count
    for host,gpu in available:
        if remaining==0:break
        # 单型号验收不能被另一型号 worker 反复领取。
        model=next((m for line in listing.splitlines() for m in ALLOWED_GPU_MODELS
                    if len(line.split())>2 and line.split()[0]=="IDLE" and line.split()[1:3]==[host,str(gpu)]
                    and line.strip().endswith(m)),None)
        if model is not None and not runnable_tasks(out,model): continue
        key=f"{host}:{gpu}";stamp=time.time()
        if stamp-last_launch.get(key,0)<cooldown:continue
        # 即使 SSH 返回失败也冷却，避免环境故障时每分钟重复启动。
        last_launch[key]=stamp
        try:
            launched.append(launch_device(host,gpu,out));remaining-=1
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            errors.append({"host":host,"gpu":gpu,"error":repr(error)})
    return {"time":now(),"idle_allowed":available,"runnable_unclaimed_tasks":count,"launched":launched,"errors":errors}


def run(out=OUT,interval=60,cooldown=300,once=False):
    out=Path(out).resolve()
    if interval<10 or cooldown<interval:raise ValueError("扫描间隔至少10秒；启动冷却不少于扫描间隔")
    with lock(out/"locks/resource-monitor.lock") as acquired:
        if not acquired:
            print("资源监控已在运行；不重复启动。",flush=True);return
        stopped=False
        def request_stop(signum,frame):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
        state_path=out/"monitor/status.json"
        base={"pid":os.getpid(),"host":socket.gethostname(),"interval_seconds":interval,
            "launch_cooldown_seconds":cooldown,"allowed_models":ALLOWED_GPU_MODELS,"started_at":now()}
        last_launch=read_json(out/"monitor/last_launch.json",{})
        atomic_json(state_path,{**base,"status":"starting"})
        try:
            while not stopped and not (out/"WATCH_STOP").exists() and not (out/"STOP").exists():
                started=time.monotonic()
                try:
                    cycle=scan(out,last_launch,cooldown)
                    atomic_json(out/"monitor/last_launch.json",last_launch)
                    payload={**base,"status":"watching","last_scan":cycle}
                except Exception as error:
                    payload={**base,"status":"retrying_discovery","error":repr(error),"time":now()}
                atomic_json(state_path,payload)
                with (out/"monitor/events.jsonl").open("a") as stream:
                    import json
                    stream.write(json.dumps(payload,ensure_ascii=False)+"\n")
                print(payload,flush=True)
                if once:break
                # 后台进程按秒响应停止；不会阻塞交互助手。
                while not stopped and time.monotonic()-started<interval:
                    if (out/"WATCH_STOP").exists() or (out/"STOP").exists():break
                    time.sleep(1)
        finally:atomic_json(state_path,{**base,"status":"stopped","ended_at":now()})


def start(out=OUT,interval=60,cooldown=300):
    out=Path(out).resolve()
    if (out/"WATCH_STOP").exists() or (out/"STOP").exists():raise RuntimeError("存在停止标记；不会自动解除")
    if (out/"locks/resource-monitor.lock.d").exists():
        print("监控锁已存在。请查 monitor/status.json；不自动抢占遗留锁。",flush=True);return
    path=out/"monitor/nohup.log";path.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    with path.open("a") as stream:
        child=subprocess.Popen(["nohup",str(ROOT/".venv/bin/python"),"-m","control_experiments.mmas_ls.watch","run",
            "--output",str(out),"--interval",str(interval),"--cooldown",str(cooldown)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(f"资源监控已启动 pid={child.pid}; log={path}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("start","run","once","status"))
    parser.add_argument("--output",type=Path,default=OUT);parser.add_argument("--interval",type=int,default=60)
    parser.add_argument("--cooldown",type=int,default=300);args=parser.parse_args()
    if args.action=="start":start(args.output,args.interval,args.cooldown)
    elif args.action=="status":print(read_json(args.output/"monitor/status.json",{}))
    else:run(args.output,args.interval,args.cooldown,args.action=="once")
