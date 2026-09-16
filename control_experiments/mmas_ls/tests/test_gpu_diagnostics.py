"""显式空闲 GPU 上执行，不在 CPU 测试中自动占用设备。"""
from dataclasses import replace
import numpy as np
import pytest
import torch
from control_experiments.mmas_ls.tests.test_gpu_controls import example,settings
from rmtgp_aco.aco_cuda import solve_population_cuda_anytime,cuda_available
from rmtgp_aco.mechanisms import SolverControl,InstrumentationConfig,MechanismConfig
from control_experiments.mmas_ls.diagnostics import DiagnosticRecorder,validate_sample

pytestmark=pytest.mark.skipif(not cuda_available(),reason="需要 CUDA")


@pytest.mark.parametrize("variant",["as","mmas"])
def test_full_audit_preserves_solver(tmp_path,variant):
    b=example();c,r=settings(variant,30)
    original=solve_population_cuda_anytime(b,c,[(None,None)],seed=43,runtime=r)
    inst=InstrumentationConfig(profile="mechanism_v3",schema_version=3)
    writer=DiagnosticRecorder(tmp_path,{"seed":43},inst)
    try:
        control=SolverControl(instrumentation=inst,observer=writer,collected=[])
        result=solve_population_cuda_anytime(b,c,[(None,None)],seed=43,runtime=r,control=control)
        writer.finish(30)
    finally: writer.close()
    assert torch.equal(original.best_tour,result.best_tour)
    assert torch.equal(original.anytime_best,result.anytime_best)
    for name,row in writer.journal.index["files"].items():
        if row["metadata"]["kind"]=="sample":
            with np.load(tmp_path/name) as data: validate_sample(dict(data))


def test_real_gp_audit_and_both_phase_resume(tmp_path):
    import cupy as cp
    from control_experiments.mmas_ls.evaluate import program_entries
    b=example();c,r=settings(iterations=30)
    entries=program_entries("mmas")
    programs=[e["program"] for e in entries]
    original=solve_population_cuda_anytime(b,c,programs,seed=43,runtime=r)
    saved={}
    def observe(phase,iteration,state):
        if iteration==25 and phase in ("post_ls","iteration_end"):
            saved.setdefault(phase,{})[tuple(state["flat_indices"]) ]={"phase":phase,"iteration":iteration,"flat_indices":state["flat_indices"].copy(),
                "arrays":{k:cp.asnumpy(v) for k,v in state.items() if isinstance(v,cp.ndarray)}}
    inst=InstrumentationConfig(profile="mechanism_v3",schema_version=3)
    result=solve_population_cuda_anytime(b,c,programs,seed=43,runtime=r,
        control=SolverControl(instrumentation=inst,observer=observe))
    assert torch.equal(original.best_tour,result.best_tour)
    assert torch.equal(original.anytime_best,result.anytime_best)
    for snapshots in saved.values():
        resumed=solve_population_cuda_anytime(b,c,programs,seed=43,runtime=r,
            control=SolverControl(instrumentation=inst,resume=lambda selected:snapshots[tuple(selected)]))
        assert torch.equal(result.best_tour,resumed.best_tour)
        assert torch.equal(result.anytime_best,resumed.anytime_best)


def test_disk_checkpoint_resume_no_missing_or_duplicate_blocks(tmp_path):
    b=example();c,r=settings(iterations=530)
    inst=InstrumentationConfig(profile="mechanism_v3",schema_version=3)
    original=solve_population_cuda_anytime(b,c,[(None,None)],seed=81,runtime=r)
    writer=DiagnosticRecorder(tmp_path,{"seed":81},inst)
    try:
        solve_population_cuda_anytime(b,c,[(None,None)],seed=81,runtime=r,
            control=SolverControl(instrumentation=inst,observer=writer,stop_iteration=510))
    finally: writer.close()
    # 已提交的部分末块不能与恢复后的整块并存。真实异常只会在提交边界留下完整块；
    # 此测试模拟 510 轮硬中断，移走显式 stop_iteration 才会写出的尾块。
    from control_experiments.mmas_ls.common import atomic_json
    index=writer.journal.index
    for name,row in list(index["files"].items()):
        if row["metadata"].get("kind")=="iterations" and row["metadata"]["end"]==510:
            (tmp_path/name).rename(tmp_path/("interrupted-"+name.replace("/","-")))
            del index["files"][name]
    atomic_json(tmp_path/"index.json",index)
    resumed_writer=DiagnosticRecorder(tmp_path,{"seed":81},inst)
    try:
        resumed=solve_population_cuda_anytime(b,c,[(None,None)],seed=81,runtime=r,
            control=SolverControl(instrumentation=inst,observer=resumed_writer,resume=resumed_writer.journal.latest))
        resumed_writer.finish(530)
    finally: resumed_writer.close()
    assert torch.equal(original.best_tour,resumed.best_tour)
    assert torch.equal(original.anytime_best,resumed.anytime_best)


def test_forced_uniform_restart_floor_and_source_metadata(tmp_path):
    import cupy as cp
    b=example();c,r=settings(iterations=30)
    inst=InstrumentationConfig(profile="mechanism_v3",schema_version=3)
    replay=np.zeros((2,30),dtype=np.int8);replay[:,24]=1
    saved={}
    def observe(phase,iteration,state):
        if phase=="initialised":state["pheromone_workspace"].fill(0)
        if phase=="iteration_end" and iteration in (1,25,30):
            saved[iteration]={k:cp.asnumpy(state[k]) for k in ("audit_counters","audit_source_info",
                "audit_sources","audit_source_origin","global_best_origin","best_tours","mechanism_trace")}
    mechanism=MechanismConfig(restart_policy="replay",source_policy="global_best_only",floor_scale=10.)
    solve_population_cuda_anytime(b,c,[(None,None)],seed=83,runtime=r,
        control=SolverControl(mechanism,inst,replay_restarts=replay,observer=observe))
    assert (saved[1]["audit_counters"][:,0,3]>0).all()
    assert (saved[25]["mechanism_trace"][:,24,8]==1).all()
    assert saved[30]["mechanism_trace"][...,6].sum()>0
    np.testing.assert_array_equal(saved[25]["audit_sources"][:,0],saved[25]["best_tours"])
    np.testing.assert_array_equal(saved[25]["audit_source_origin"][:,0],saved[25]["global_best_origin"])
    assert (saved[30]["audit_source_info"][:,29,0,3]<30).all()


def test_heavy_snapshot_v3_fork_entry(tmp_path,monkeypatch):
    from control_experiments.mmas_ls import diagnostic_forks as forks
    from control_experiments.mmas_ls.common import atomic_npz,source_manifest,read_json
    from dataclasses import asdict
    b=example();c,r=settings(iterations=600)
    job=tmp_path/"jobs"/"owner"
    specification={"source":source_manifest()["source_hash"],"seed":43,"models":[],
        "task":{"indices":[0,1],"variant":"mmas","split":"diagnosis_dev","mechanism":asdict(MechanismConfig())}}
    inst=InstrumentationConfig("heavy",profile="mechanism_v3",schema_version=3)
    writer=DiagnosticRecorder(job/"diagnostics",specification,inst)
    try:
        solve_population_cuda_anytime(b,c,[(None,None)],seed=43,runtime=r,
            control=SolverControl(instrumentation=inst,observer=writer,stop_iteration=1))
        writer.finish(1)
    finally:writer.close()
    atomic_npz(job/"raw.npz",behavior_alias=np.array([0]))
    name=next(k for k,v in writer.journal.index["files"].items()
        if v["metadata"]["kind"]=="permanent" and v["metadata"]["phase"]=="post_ls")
    monkeypatch.setattr(forks,"batch",lambda *a,**k:b)
    monkeypatch.setattr(forks,"experiment",lambda *a,**k:(c,r))
    target=forks.fork(job,name,0,81002)
    assert read_json(target/"manifest.json")["original_horizon"]==600
    transfer=read_json(target/"ph_transfer.json")
    assert len(transfer["rows"])==2
    assert all(0<=row["cpu_fp64_full_gp_tv"]<=1 for row in transfer["rows"])
