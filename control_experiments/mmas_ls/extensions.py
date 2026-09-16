"""P3/P4 的预注册工单。正式确认前冻结，不能根据 P2 结果选择条件。"""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
from rmtgp_aco.mechanisms import MechanismConfig,factorial_conditions
from .common import OUT,atomic_json,digest,read_json,now
from .evaluate import factorial_tasks


def extension_specification():
    native=MechanismConfig(); result=[]
    for policy in ("restart_best_only","global_best_only","native_slots_use_restart_best","native_slots_use_global_best"):
        result.append((policy,replace(native,source_policy=policy)))
    for probability in (0.,.25,.5,.75,1.):
        result.append((f"history-p{probability:g}",replace(native,source_policy="history_probability",p_history=probability)))
    for k in (1,4,8,32):result.append((f"top-{k}",replace(native,source_policy="top_k",source_count=k)))
    for scale in (.25,4.):
        for field in ("floor_scale","initial_tau_scale","restart_tau_scale"):
            result.append((f"{field}-{scale:g}",replace(native,**{field:scale})))
    result.extend((("new-hard-upper",replace(native,hard_upper_clip=True)),
                   ("headroom-zero",replace(native,tau_headroom_override=0.))))
    for tau in (False,True):
        for memory in (False,True):
            result.append((f"replay-Y{int(tau)}{int(memory)}",replace(native,restart_policy="replay",reset_pheromone=tau,reset_epoch_memory=memory)))
    return result


def extension_tasks(out=OUT):
    tasks=[]
    # P3 开发性机制扩展：不依据结果删减来源、剂量或重启分解。
    for rep in range(5):
        parent=f"diagnosis_dev-mmas-C111-s{rep:02d}-b000"
        for name,mechanism in extension_specification():
            task={"id":f"P3-dev-{name}-s{rep:02d}","stage":"P3","split":"diagnosis_dev",
                "indices":list(range(32)),"replicate":rep,"condition":name,"variant":"mmas",
                "mechanism":asdict(mechanism),"iterations":5000,"modes":["full"],"instrumentation":"light"}
            if mechanism.restart_policy=="replay" or mechanism.source_policy.startswith("native_slots"):
                task["replay_from"]=parent
            tasks.append(task)
        # 全部 8 个背景下拆分两树，不只选择净收益最大的背景。
        for condition,mechanism in factorial_conditions().items():
            tasks.append({"id":f"P3-trees-{condition}-s{rep:02d}","stage":"P3","split":"diagnosis_dev",
                "indices":list(range(32)),"replicate":rep,"condition":condition,"variant":"mmas",
                "mechanism":asdict(mechanism),"iterations":5000,"modes":["full","tr_only","ph_only"],"instrumentation":"light"})
    for split in ("confirm_cluster","confirm_gaussian"):tasks.extend(factorial_tasks(split,out))
    for rep in range(10):
        for start in range(0,128,32):
            for name,policy in (("AS-native","native_schedule"),("AS-IB","iteration_best"),("AS-history","global_calendar")):
                tasks.append({"id":f"P4-{name}-s{rep:02d}-b{start:03d}","stage":"P4","split":"confirm_uniform",
                    "indices":list(range(start,start+32)),"replicate":rep,"condition":name,"variant":"as",
                    "mechanism":asdict(MechanismConfig(source_policy=policy)),"iterations":5000,
                    "modes":["full"],"instrumentation":"light"})
            # 独立短预算和时间 terminal 控制分开；不把 5000 轮前缀冒充短预算。
            for horizon in (100,200,500,1000):
                for label,normalizer,clip in (("native",None,False),("fixed5000",5000,False),("train500clip",500,True)):
                    tasks.append({"id":f"P4-H{horizon}-{label}-s{rep:02d}-b{start:03d}","stage":"P4","split":"confirm_uniform",
                        "indices":list(range(start,start+32)),"replicate":rep,"condition":f"H{horizon}-{label}","variant":"mmas",
                        "mechanism":asdict(MechanismConfig(terminal_normalization_horizon=normalizer,terminal_clip=clip)),
                        "iterations":horizon,"modes":["full"],"instrumentation":"light"})
    return tasks


def freeze_extensions(out=OUT):
    """应在读取任何 P2 结果前调用；现阶段先固定工单，不自动解除 P0/P1 门禁。"""
    out=Path(out)
    if any((out/"jobs"/t["id"]/"raw.npz").exists() for t in factorial_tasks("confirm_uniform",out)):
        raise RuntimeError("P2 已有结果，不能将新扩展伪称为事前冻结")
    specification={"tasks":extension_tasks(out),"heavy_instances":list(range(8)),"heavy_seeds":[0,1,2],
        "snapshot_iterations":[1,100,250,500,1000,2500],"fork_iterations":[1,25,100,500],
        "same_state_start_owners":["baseline","each_champion"],
        "scope":"P3 在开发集探索；独立 P3 确认必须在看 P2 前另行锁定，不能用开发显著性作正式证据"}
    path=out/"protocol/extensions_frozen.json"; frozen={"hash":digest(specification),"specification":specification,"time":now()}
    if path.exists():
        if read_json(path)["hash"]!=frozen["hash"]:raise ValueError("扩展规格已经冻结")
        return
    atomic_json(path,frozen)
