"""用户批准后解除指定未完成配对任务的型号绑定，保留中断证据并整组重跑。"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from pathlib import Path
import time
from .common import read_json,atomic_json,digest,now,file_hash
from .campaign import lock,affinity_path


def relax_pairs(out,task_ids):
    out=Path(out).resolve()
    if (out/"locks/resource-monitor.lock.d").exists():
        raise RuntimeError("先正常停止资源扫描器，再修改调度策略；不停止其他用户任务")
    if len(set(task_ids))!=len(task_ids):raise ValueError("重复任务")
    policy_path=out/"protocol/resource_policy.json"
    policy=read_json(policy_path,{"schema_version":1,"unrestricted_tasks":{}})
    prepared=[]
    with ExitStack() as stack:
        for task_id in sorted(task_ids):
            if Path(task_id).name!=task_id:raise ValueError("无效任务 id")
            queue=out/"queue"/(task_id+".json");payload=read_json(queue);task=payload["task"]
            if task.get("kind")!="numeric_pair":raise ValueError("仅允许迁移完整数值配对任务")
            if task_id in policy["unrestricted_tasks"]:raise ValueError("已有迁移记录，不重复移动产物")
            acquired=stack.enter_context(lock(out/"locks/tasks"/(task_id+".lock")))
            if not acquired:raise RuntimeError(f"任务仍被 worker 领取: {task_id}")
            parent=out/"jobs"/task_id;status=read_json(parent/"status.json",{})
            if status.get("status") not in (None,"resource_paused","failed"):
                raise ValueError(f"不移动运行中或已完成任务: {task_id}")
            previous_affinity=affinity_path(task,out)
            folders=[parent]+[out/"jobs"/(task_id+"--"+mode) for mode in task["numeric_order"]]
            prepared.append((task_id,task,queue,folders,status,
                read_json(previous_affinity,{}) if previous_affinity else {}))
        for task_id,task,queue,folders,status,affinity in prepared:
            archive=out/"resource_pauses"/"model-migration"/task_id/str(time.time_ns())
            archive.mkdir(parents=True)
            record={"task":task_id,"task_sha256":digest(task),"queue_file_sha256":file_hash(queue),
                "approved_at":now(),"authorization":"用户明确要求任意两个空闲 GPU，不再锁定型号",
                "old_status":status,"old_affinity":affinity,"archive":str(archive),
                "restart_policy":"三种模式整组从头运行；同一任务内不拼接旧设备日志；冻结任务和源码不变",
                "moved":[]}
            atomic_json(archive/"migration.json",record)
            for folder in folders:
                if folder.exists():
                    folder.rename(archive/folder.name);record["moved"].append(folder.name)
                    atomic_json(archive/"migration.json",record)
            policy["unrestricted_tasks"][task_id]=record
        policy["updated_at"]=now();atomic_json(policy_path,policy)
    return policy


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tasks",nargs="+",required=True);a=p.parse_args()
    print(relax_pairs(a.output,a.tasks))
