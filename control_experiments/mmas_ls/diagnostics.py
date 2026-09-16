"""版本化机制诊断：GPU 采样、异步传输、原子分块提交和完整性校验。

未使用 terminal 也计算影子观测，但 required_mask 单独保留。影子量不能
解释为程序实际使用的输入。所有统计保留逻辑求解轴，不混合冠军和实例。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import threading
import time
import numpy as np

from .common import atomic_json, atomic_npz, digest, file_hash, now, read_json

TR_FIELDS = ("RTau", "REta", "BaseConf", "DistRank", "Entropy", "ConstructProg", "ACOProg", "Stagnation",
             "Tau", "Distance", "MeanTau", "MeanDistance", "Size", "FeasibleCount", "MutualRank", "TurnCos",
             "protected_raw", "base_score", "actual_score", "actual_probability", "rng_stream3", "candidate_position")
PH_FIELDS = ("EdgeEta", "EdgeTau", "NNRank", "ColonyFreq", "SourceQuality", "ACOProg", "Stagnation",
             "LSGain", "Origin", "TauHeadroom", "PreFreq", "PostFreq", "protected_raw", "tanh_raw",
             "residual_factor", "pre_normalization_deposit", "source_normalization_scale", "base_deposit")
SOURCE_FIELDS = ("kind", "ant_index", "length", "metadata_generation_iteration", "ls_gain", "actual_budget")
CONTEXT_FIELDS = ("iteration", "ant", "step", "current", "previous", "chosen", "candidate_exhausted",
                  "feasible_count", "greedy_branch", "actual_uniform_branch")
COUNTER_FIELDS = ("candidate_exhausted_greedy", "base_degenerate", "residual_degenerate", "actual_uniform")
QUANTILES = (0., .01, .05, .25, .5, .75, .95, .99, 1.)
PERMANENT_ITERATIONS = (1, 100, 250, 500, 1000, 2500)


def schema():
    """字段字典是产物的一部分；shape 中 T 为实际执行 task，S=32，N=城市数。"""
    fields = {
        "trace": ("[T,W,26]", "float32", "每轮", "legacy TRACE_FIELDS，长度为 GPU FP32"),
        "source_hash": ("[T,W,S,2]", "uint64", "每轮实际来源", "规范无向边双校验码；非密码学身份"),
        "source_info": ("[T,W,S,6]", "float32", "每轮实际来源", SOURCE_FIELDS),
        "counters": ("[T,W,4]", "uint32", "每轮全部构造决策", COUNTER_FIELDS),
        "ls": ("[T,W,32,3]", "float32", "每轮每只蚂蚁", ("pre_length", "post_length", "retained_edge_count")),
        "ls_counts": ("[T,W,32,4]", "uint64", "每轮每只蚂蚁", ("accepted_moves", "evaluated_moves", "improved_by_LS_tolerance", "passes")),
        "ph_moments": ("[T,W,S,6]", "float64", "每轮实际来源的全部边", ("count", "saturated_count", "raw_sum", "raw_square_sum", "tanh_sum", "tanh_square_sum")),
        "restart_before": ("[T,W,6]", "float64", "每轮沉积后、重启判断前", ("RB_length", "RB_found_iteration", "epoch_iteration", "RB_ls_gain", "stagnation", "GB_found_iteration")),
        "restart_after": ("[T,W,6]", "float64", "每轮重启判断后", ("RB_length", "RB_found_iteration", "epoch_iteration", "RB_ls_gain", "stagnation", "GB_found_iteration")),
        "diagnostics": ("[T,W,8]", "uint64", "每轮累计计数", "旧计数；uniform 字段不可解释为均匀抽样次数"),
        "ph": ("[T,S,N,18]", "float32", "t=1 及每 25 轮的全部来源边", PH_FIELDS),
        "deposit": ("[T,S,N]", "float32", "采样轮实际来源边", "最终单源单向边量；对称写入两个方向；无效槽用 NaN"),
        "source_tours": ("[T,S,N+1]", "uint16", "采样轮实际来源", "有效来源数见 trace.source_count"),
        "source_origin": ("[T,S,N]", "int8", "采样轮实际来源", "来源生成时的 Origin，不是当前轮重新计算"),
        "source_gain": ("[T,S,N]", "float32", "采样轮实际来源", "来源生成时的逐边或逐路径 LSGain"),
        "tr": ("[T,16,N,22]", "float32", "固定 4 蚂蚁 × 4 步；仅当前可行候选有效", TR_FIELDS),
        "context": ("[T,16,10]", "int32", "同 TR", CONTEXT_FIELDS),
        "visited": ("[T,16,ceil(N/64)]", "uint64", "同 TR", "构造选择前的 bitset；RNG 使用实例 ID 键"),
        "tau_quantiles": ("[T,4,9]", "float32", "窗口末点的全部非对角有向边", "更新前/蒸发及floor后/沉积后/重启及可选上界后"),
        "tau_moments": ("[T,4,4]", "float64", "同 tau_quantiles", ("mean", "std", "l1", "l2")),
        "tau_above_nominal_max": ("[T,4]", "int64", "同 tau_quantiles", "名义 tau_max 不是 native LS 的硬上界"),
        "pre_tours": ("[T,32,N+1]", "uint16", "heavy 采样轮", "2-opt 前路径"),
        "post_tours": ("[T,32,N+1]", "uint16", "heavy 采样轮", "2-opt 后路径；边集相同不等于证明 basin 相同"),
        "required_masks": ("[T,2]", "uint64", "每块常量", "TR/PH 实际需求；影子计算与实际使用分开"),
    }
    return {"version": 3, "profile": "mechanism_v3", "fields": {
        k:dict(shape=v[0], dtype=v[1], population=v[2], definition=v[3]) for k,v in fields.items()},
        "quantiles": QUANTILES, "missing": "浮点 NaN / 整数 -1 / 显式 source_count；不能填零冒充观测",
        "saturation": "abs(tanh(protected_raw)) >= 0.99",
        "alias": "PH ColonyFreq 和 PostFreq 为同一后 LS 边频率定义",
        "normalization": "每源归一化后再做干预总预算校正；预算按无向 tour edge 计一次",
        "window": "W<=100；25 轮窗口均值/计数从逐轮数据计算；末点分布不能称作窗口分布",
        "types": {"TR":"TrField", "PH":"PhField", "definition":"DEAP 类型标记；分别对应候选域和来源边域，非数值 dtype"},
        "snapshot_phases": ["post_ls", "iteration_end"]}


def array_digest(arrays):
    h=hashlib.sha256()
    for k,v in sorted(arrays.items()):
        v=np.ascontiguousarray(v)
        h.update(k.encode()); h.update(v.dtype.str.encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
    return h.hexdigest()


class Journal:
    """先落数据再提交索引。崩溃留下的未索引文件不会被当成完成结果。"""
    def __init__(self, directory, specification, min_free_bytes=2*1024**3):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.key=digest(specification); self.min_free_bytes=min_free_bytes
        self.path=self.directory/"index.json"
        self.index=read_json(self.path,{"version":3,"scientific_hash":self.key,"files":{},"completed":False})
        if self.index["scientific_hash"]!=self.key:
            raise ValueError("诊断科学配置变化，必须使用新目录")
        self.verify()
        atomic_json(self.directory/"specification.json",specification)
        atomic_json(self.directory/"schema.json",schema())

    def verify(self):
        for name,record in self.index["files"].items():
            if file_hash(self.directory/name)!=record["sha256"]:
                raise ValueError(f"诊断文件损坏: {name}")

    def write(self,name,arrays,metadata):
        if metadata.get("kind")=="sample":
            try: validate_sample(arrays)
            except Exception as error:
                # 验收失败也保留原始证据，但绝不写入成功提交索引。
                atomic_npz(self.directory/"quarantine"/name,**arrays)
                atomic_json(self.directory/"quarantine"/(name+".json"),{"error":repr(error),"metadata":metadata})
                raise
        size=sum(v.nbytes for v in arrays.values())
        if shutil.disk_usage(self.directory).free < self.min_free_bytes+size:
            raise OSError("空间不足：暂停诊断，不降低采样频率或删减字段")
        content=array_digest(arrays)
        previous=self.index["files"].get(name)
        if previous:
            if previous["content_hash"]!=content or previous["metadata"]!=metadata:
                raise ValueError(f"恢复重放与已提交块不一致: {name}")
            return
        began=time.perf_counter()
        atomic_npz(self.directory/name,**arrays)
        self.index["files"][name]={"sha256":file_hash(self.directory/name),"content_hash":content,
            "arrays":{k:{"shape":list(v.shape),"dtype":v.dtype.str} for k,v in arrays.items()},
            "uncompressed_bytes":size,"compressed_bytes":(self.directory/name).stat().st_size,
            "write_seconds":time.perf_counter()-began,"metadata":metadata}
        atomic_json(self.path,self.index)
        if metadata.get("kind")=="recovery":
            # 仅清理本 journal 已校验的旧恢复副本；永久机制快照不删除。
            recovery=sorted((v["metadata"]["iteration"],k) for k,v in self.index["files"].items()
                if v["metadata"].get("kind")=="recovery" and v["metadata"]["shard"]==metadata["shard"])
            expired=[k for _,k in recovery[:-2]]
            if expired:
                for key in expired: del self.index["files"][key]
                atomic_json(self.path,self.index)
                for key in expired: (self.directory/key).unlink()

    def complete(self, expected_shards, horizon):
        recorded={v["metadata"]["shard"] for v in self.index["files"].values() if v["metadata"].get("kind")=="iterations"}
        if recorded!=set(expected_shards):raise ValueError("提交包含未预期的执行分块")
        for shard in expected_shards:
            ranges=sorted((v["metadata"]["start"],v["metadata"]["end"])
                for v in self.index["files"].values() if v["metadata"].get("kind")=="iterations"
                and v["metadata"]["shard"]==shard)
            next_iteration=1
            for lo,hi in ranges:
                if lo!=next_iteration or hi<lo: raise ValueError("逐轮记录重复或缺失")
                next_iteration=hi+1
            if next_iteration!=horizon+1: raise ValueError("诊断尚未覆盖完整 horizon")
            observed=sorted(v["metadata"]["iteration"] for v in self.index["files"].values()
                if v["metadata"].get("kind")=="sample" and v["metadata"]["shard"]==shard)
            if observed!=[1]+list(range(25,horizon+1,25)): raise ValueError("末点采样缺失或重复")
        self.verify(); self.index.update(completed=True,completed_at=now(),shards=expected_shards,horizon=horizon)
        atomic_json(self.path,self.index)

    def latest(self, selected):
        """恢复只接受完全相同的执行分块。任务自动改变分块会明确失败。"""
        shard=digest(np.asarray(selected).tolist())[:16]
        choices=[(v["metadata"]["iteration"],name,v) for name,v in self.index["files"].items()
                 if v["metadata"].get("kind")=="recovery" and v["metadata"]["shard"]==shard]
        if not choices: return None
        iteration,name,record=max(choices)
        with np.load(self.directory/name,allow_pickle=False) as data:
            return {"iteration":iteration,"phase":"iteration_end","flat_indices":np.asarray(selected),
                    "arrays":{k:data[k].copy() for k in data.files}}


def validate_completed(directory):
    directory=Path(directory);index=read_json(directory/"index.json")
    if index.get("version")!=3 or not index.get("completed"):
        raise ValueError("缺少完整 schema 3 诊断")
    for name,record in index["files"].items():
        if file_hash(directory/name)!=record["sha256"]: raise ValueError(f"诊断文件损坏: {name}")
    return index


class DiagnosticRecorder:
    """CUDA 流上复制到 pinned host；压缩、哈希、写盘在单独线程执行。

    最多两个待写块，反压仅发生于提交边界。异常通过 futures 返回主线程。
    不从 observer 内改变任何决策工作区，不生成随机数。
    """
    def __init__(self,directory,specification,instrumentation):
        self.journal=Journal(directory,specification)
        self.instrumentation=instrumentation
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="mechanism-writer")
        self.pending=[]; self.shards=[]; self.last_commit={}; self.timings=[]
        self.restart_candidates={}
        self.transfer_seconds=0.

    def _submit(self,name,arrays,metadata,restart_only=False):
        import cupy as cp
        # 限制 pinned RAM。异常不被后台线程吞掉。
        while len(self.pending)>=2: self.pending.pop(0).result()
        device=cp.cuda.runtime.getDevice(); host={}; keep=[]
        transfer_start=cp.cuda.Event();transfer_start.record()
        for key,value in arrays.items():
            value=cp.ascontiguousarray(value)
            memory=cp.cuda.alloc_pinned_memory(value.nbytes)
            target=np.frombuffer(memory,dtype=value.dtype,count=value.size).reshape(value.shape)
            value.get(out=target,blocking=False)
            host[key]=target; keep.extend((memory,value))
        ready=cp.cuda.Event(); ready.record()
        def write():
            with cp.cuda.Device(device): ready.synchronize()
            self.transfer_seconds+=cp.cuda.get_elapsed_time(transfer_start,ready)/1000.
            if restart_only and not np.any(host["restart_event_trace"][:,8]>0):return len(keep)
            self.journal.write(name,host,metadata)
            # keep 的闭包保留源数组与 pinned allocation 直至传输和写盘完成。
            return len(keep)
        self.pending.append(self.pool.submit(write))

    def __call__(self,phase,iteration,state):
        import cupy as cp
        shard=digest(state["flat_indices"].tolist())[:16]
        if phase=="initialised":
            if shard in self.shards: raise ValueError("重复执行相同 task 分块")
            selected=set(map(int,state["flat_indices"]))
            for record in self.journal.index["files"].values():
                meta=record["metadata"]
                if meta.get("kind")=="inputs" and meta["shard"]!=shard and selected.intersection(meta["flat_indices"]):
                    raise ValueError("恢复不得改变 task 分块；不能把已有前缀混入另一种执行分块")
            self.shards.append(shard); self.last_commit[shard]=0
            count=state["count"];ants=state["tour_workspace"].shape[1]
            # 纳入恢复快照；由求解器通用恢复循环还原。
            state["audit_ls_ring"]=cp.full((count,100,ants,3),cp.nan,dtype=cp.float32)
            state["audit_diagnostics_ring"]=cp.zeros((count,100,8),dtype=cp.uint64)
            state["audit_restart_after_ring"]=cp.zeros((count,100,6),dtype=cp.float64)
            geometry=state["audit_geometry"]
            self._submit(f"{shard}/inputs.npz",{**geometry,"task_instance":state["task_instance"],
                "task_program":state["task_program"]},
                {"kind":"inputs","shard":shard,"flat_indices":state["flat_indices"].tolist(),
                 "hardware":state["audit_hardware"]})
            return
        if phase=="resumed":
            self.last_commit[shard]=iteration
            return
        if phase=="post_ls":
            slot=(iteration-1)%100
            state["audit_ls_ring"][:,slot,:,0]=state["length_before_workspace"]
            state["audit_ls_ring"][:,slot,:,1]=state["length_workspace"]
            state["audit_ls_ring"][:,slot,:,2]=(state["origin_workspace"]>0).sum(axis=-1)
        heavy=self.instrumentation.level=="heavy"
        # 原生重启检查是每 100 轮。固定规则下预留更新前状态，写线程只提交真实重启。
        # 保存阶段准确标作 post_ls，而不冒充 kernel 内的 pre_restart 阶段。
        if heavy and phase=="post_ls" and iteration%100==0:
            self.restart_candidates[shard]={k:v.copy() for k,v in state.items() if isinstance(v,cp.ndarray)}
        if heavy and iteration in PERMANENT_ITERATIONS and phase in ("post_ls","iteration_end"):
            self._snapshot(state,shard,iteration,phase,"permanent")
        if phase=="iteration_end":
            state["audit_diagnostics_ring"][:,(iteration-1)%100]=state["diagnostics"]
            state["audit_restart_after_ring"][:,(iteration-1)%100]=cp.stack(tuple(state[k] for k in
                ("restart_best_lengths","restart_found_best","restart_iteration","restart_best_ls_gain",
                 "stagnation","best_iterations")),axis=-1)
            if shard in self.restart_candidates:
                before=self.restart_candidates.pop(shard)
                event=state["mechanism_trace"][:,iteration-1].copy()
                for when,values in (("post_ls",before),("iteration_end",{k:v for k,v in state.items() if isinstance(v,cp.ndarray)})):
                    self._submit(f"{shard}/restart-{iteration:05d}-{when}.npz",
                        {**values,"restart_event_trace":event},{"kind":"restart_snapshot","shard":shard,
                            "iteration":iteration,"phase":when,"flat_indices":state["flat_indices"].tolist()},restart_only=True)
            if iteration==1 or iteration%25==0: self._sample(state,shard,iteration,heavy)
            if iteration%100==0: self._commit(state,shard,iteration)
            if iteration%500==0: self._snapshot(state,shard,iteration,phase,"recovery")
        elif phase=="chunk_end":
            if self.last_commit[shard]<iteration: self._commit(state,shard,iteration)

    def _snapshot(self,state,shard,iteration,phase,kind):
        import cupy as cp
        arrays={k:v for k,v in state.items() if isinstance(v,cp.ndarray)}
        self._submit(f"{shard}/{kind}-{iteration:05d}-{phase}.npz",arrays,
            {"kind":kind,"shard":shard,"iteration":iteration,"phase":phase,
             "flat_indices":state["flat_indices"].tolist()})

    def _commit(self,state,shard,iteration):
        lo=self.last_commit[shard]+1; width=iteration-lo+1
        if width<1 or width>100 or (lo-1)//100!=(iteration-1)//100:
            raise ValueError("提交区间不在同一环形块内")
        slot=(lo-1)%100; section=slice(slot,slot+width)
        capacity=state["audit_source_capacity"]
        arrays={"source_info":state["audit_source_info"][:,section,:capacity],
                "source_hash":state["audit_source_hash"][:,section,:capacity],
                "counters":state["audit_counters"][:,section],"ls":state["audit_ls_ring"][:,section],
                "diagnostics":state["audit_diagnostics_ring"][:,section],
                "trace":state["mechanism_trace"][:,lo-1:iteration],
                "anytime":state["anytime"][:,lo-1:iteration],
                "ph_moments":state["audit_ph_moments"][:,section,:capacity],
                "ls_counts":state["audit_ls_counts"][:,section],
                "restart_before":state["audit_restart_state"][:,section],
                "restart_after":state["audit_restart_after_ring"][:,section]}
        self._submit(f"{shard}/iterations-{lo:05d}-{iteration:05d}.npz",arrays,
            {"kind":"iterations","shard":shard,"start":lo,"end":iteration,
             "flat_indices":state["flat_indices"].tolist()})
        self.last_commit[shard]=iteration

    def _sample(self,state,shard,iteration,heavy):
        import cupy as cp
        n=state["pheromone_workspace"].shape[-1]; count=state["count"]
        tau=cp.concatenate((state["audit_tau"],state["pheromone_workspace"][:,None]),axis=1)
        mask=~cp.eye(n,dtype=cp.bool_); values=tau[:,:,mask]
        values64=values.astype(cp.float64)
        source_count=state["mechanism_trace"][:,iteration-1,3].astype(cp.int32)
        # 仅去除配置上永远不可能使用的 padding 槽；没有降低有效来源或采样密度。
        capacity=state["audit_source_capacity"]
        valid=cp.arange(capacity)[None,:]<source_count[:,None]
        deposit=cp.where(valid[:,:,None],state["deposit_workspace"][:,:capacity],cp.nan)
        source_tours=state["audit_sources"][:,:capacity].astype(cp.int32)
        edge_tau=state["audit_tau"][cp.arange(count)[:,None,None],0,
                                    source_tours[:,:,:-1],source_tours[:,:,1:]]
        arrays={"ph":state["audit_ph"][:,:capacity],"tr":state["audit_tr"],"context":state["audit_context"],
            "visited":state["audit_visited"],"deposit":deposit,"source_tours":state["audit_sources"][:,:capacity],
            "source_origin":state["audit_source_origin"][:,:capacity],"source_gain":state["audit_source_gain"][:,:capacity],
            "source_count":source_count,"source_valid":valid,
            "source_info":state["audit_source_info"][:,(iteration-1)%100,:capacity],
            "source_hash":state["audit_source_hash"][:,(iteration-1)%100,:capacity],
            "tau_quantiles":cp.quantile(values,cp.asarray(QUANTILES),axis=-1).transpose(1,2,0).astype(cp.float32),
            "tau_moments":cp.stack((values64.mean(-1),values64.std(-1),abs(values64).sum(-1),
                                    cp.sqrt((values64**2).sum(-1))),axis=-1),
            "tau_above_nominal_max":(values>state["task_tau_max"][:,None,None]).sum(-1),
            "tau_relative_l1":abs(values64-values64[:,:1]).sum(-1)/cp.maximum(abs(values64[:,:1]).sum(-1),1e-30),
            "tau_relative_l2":cp.sqrt(((values64-values64[:,:1])**2).sum(-1))/cp.maximum(cp.sqrt((values64[:,:1]**2).sum(-1)),1e-30),
            "required_masks":cp.stack((state["tr_masks"][state["task_program"]],state["ph_masks"][state["task_program"]]),-1),
            "program_active":cp.stack((state["tr_is_active"][state["task_program"]],state["ph_is_active"][state["task_program"]]),-1),
            "trace":state["mechanism_trace"][:,iteration-1],
            "candidate_arc_count":cp.full(count,n*state["candidate_size"],dtype=cp.int32),
            "source_edge_tau_before":edge_tau,
            "colony_lengths":state["length_workspace"],
            "pre_tours":state["pre_tour_workspace"],"post_tours":state["tour_workspace"],
            "best_tours":state["best_tours"]}
        # 来源边元数据和 LS 结构为诊断必要字段；light 也保存采样时全部路径。
        if heavy: arrays["tau_matrices"]=tau
        self._submit(f"{shard}/sample-{iteration:05d}.npz",arrays,
            {"kind":"sample","shard":shard,"iteration":iteration,"heavy":heavy,
             "flat_indices":state["flat_indices"].tolist(),"seed":state["seed"]})

    def finish(self,horizon):
        for future in self.pending: future.result()
        self.pending.clear(); self.pool.shutdown(wait=True)
        self.journal.index["pipeline_timing"]={"d2h_and_contiguous_copy_stream_seconds":self.transfer_seconds,
            "background_npz_write_seconds":sum(r["write_seconds"] for r in self.journal.index["files"].values()),
            "scope":"传输、GPU 和后台写盘可能重叠，不能直接相加为总耗时"}
        self.journal.complete(self.shards,horizon)
        return self.journal.index

    def close(self):
        # 异常退出也等待已发出的传输，避免挂起的 CUDA allocation 被提前释放。
        try:
            for future in self.pending: future.result()
        finally: self.pool.shutdown(wait=True)


def validate_sample(arrays, budget_tolerance=1e-5):
    """独立主机重算；不把 NaN 当零，也不将假定的均匀分支冒充真实概率。"""
    ph=arrays["ph"]; valid=arrays["source_valid"]
    if not np.isfinite(ph[valid]).all(): raise ValueError("有效 PH 观测非有限")
    # CUDA --use_fast_math 的 tanhf 与 NumPy libm 不逐位一致。
    # 此阈值仅用于超越函数复核；预算守恒仍使用严格 1e-5 相对误差。
    np.testing.assert_allclose(np.tanh(ph[valid][...,12]),ph[valid][...,13],atol=2e-5,rtol=2e-5)
    # 程序外的 shadow-budget 校正会再次缩放，单独验证源内比例与总预算。
    before=ph[...,15]*ph[...,16]
    after=arrays["deposit"]
    for t in range(len(valid)):
        for s in np.flatnonzero(valid[t]):
            scale=np.sum(after[t,s],dtype=np.float64)/np.sum(before[t,s],dtype=np.float64)
            np.testing.assert_allclose(after[t,s],before[t,s]*scale,rtol=budget_tolerance,atol=1e-9)
    budget=np.nansum(after,axis=(1,2),dtype=np.float64)
    np.testing.assert_allclose(budget,arrays["trace"][:,4],rtol=budget_tolerance,atol=1e-9)
    for t in range(len(valid)):
        for p,row in enumerate(arrays["context"][t]):
            if row[0]<0: raise ValueError("固定 TR 探针缺失")
            candidate=np.isfinite(arrays["tr"][t,p,:,19])
            probs=arrays["tr"][t,p,candidate,19]
            if len(probs)!=row[7]: raise ValueError("候选集合不完整")
            np.testing.assert_allclose(probs.sum(dtype=np.float64),1.,atol=2e-6)
            if row[8] and (np.count_nonzero(probs)!=1 or arrays["tr"][t,p,row[5],19]!=1):
                raise ValueError("贪心概率没有对应实际选择")
            if row[9]: np.testing.assert_allclose(probs,1./row[7],rtol=2e-6)
            if not row[8] and not row[9]:
                scores=arrays["tr"][t,p,candidate,18]
                np.testing.assert_allclose(probs,scores/scores.sum(dtype=np.float64),rtol=3e-6,atol=1e-7)
    return {"valid_sources":int(valid.sum()),"probes":int(len(valid)*16),"budget_max_relative_error":float(np.max(abs(budget/arrays["trace"][:,4]-1)))}
