"""仅在显式指定的空闲 GPU 上运行；覆盖原生等价与实际控制效果。"""
from dataclasses import replace
import numpy as np
import pytest
import torch
from rmtgp_aco.aco_cuda import solve_population_cuda_anytime, cuda_available
from rmtgp_aco.config import ACOConfig, RuntimeConfig, ExecutionBackend, GPUMode
from rmtgp_aco.data import TSPInstance, coordinate_hash, make_problem_batch
from rmtgp_aco.mechanisms import MechanismConfig, InstrumentationConfig, SolverControl, TRACE_FIELDS, factorial_conditions

pytestmark = pytest.mark.skipif(not cuda_available(), reason="需要显式空闲 CUDA 设备")


def example(n=20):
    instances = []
    for seed in (37,89):
        coords = np.random.default_rng(seed).random((n,2))
        tour = np.r_[np.arange(n),0]
        length = np.linalg.norm(coords[tour[1:]]-coords[tour[:-1]],axis=1).sum()
        instances.append(TSPInstance(f"smoke-{seed}",coords,tour,length,coordinate_hash(coords)))
    return make_problem_batch(instances,candidate_size=min(10,n-1))


def settings(variant="mmas", iterations=280):
    config = replace(ACOConfig.acotsp_local_search_default(variant,local_search="two_opt",iterations=iterations),
                     ants=32,candidate_size=10,local_search_candidate_size=10)
    runtime = RuntimeConfig(aco_backend=ExecutionBackend.CUDA_TILED_V2,gpu_mode=GPUMode.SINGLE,gpu_devices=(0,),
                            cuda_generated_gp=True,cuda_candidate_lanes=8,gpu_memory_fraction=0.7)
    return config,runtime


@pytest.mark.parametrize("variant",["as","mmas"])
def test_native_instrumentation_and_zero_tree(variant):
    batch=example(); config,runtime=settings(variant,60)
    original=solve_population_cuda_anytime(batch,config,[(None,None)],seed=43,runtime=runtime)
    for level in ("off","light","heavy"):
        control=SolverControl(instrumentation=InstrumentationConfig(level),collected=[])
        result=solve_population_cuda_anytime(batch,config,[(None,None)],seed=43,runtime=runtime,control=control)
        assert torch.equal(original.best_tour,result.best_tour)
        assert torch.equal(original.anytime_best,result.anytime_best)
        if level!="off":
            trace=control.collected[0]["trace"]
            assert np.max(trace[...,TRACE_FIELDS.index("budget_error")]) < 1e-5
            assert np.all(trace[...,7]==0)


def test_all_switches_and_source_budgets():
    batch=example(); config,runtime=settings()
    for name,mechanism in factorial_conditions().items():
        control=SolverControl(mechanism=mechanism,collected=[])
        result=solve_population_cuda_anytime(batch,config,[(None,None)],seed=31,runtime=runtime,control=control)
        trace=control.collected[0]["trace"]
        assert np.max(trace[...,20])<1e-5
        assert torch.isfinite(result.best_length).all()
        if name[1]=="0": assert np.all(trace[...,8]==0)
        if name[2]=="0": assert np.all(trace[...,6]==0)
        if name[3]=="0": assert np.all(trace[...,0]==0)
    for variant in ("as","mmas"):
        config,runtime=settings(variant,35)
        for policy in ("top_k","global_best_only","restart_best_only","history_probability","global_calendar"):
            mechanism=MechanismConfig(source_policy=policy,source_count=4 if policy=="top_k" else 1)
            control=SolverControl(mechanism=mechanism,collected=[])
            result=solve_population_cuda_anytime(batch,config,[(None,None)],seed=31,runtime=runtime,control=control)
            assert np.max(control.collected[0]["trace"][...,20])<1e-5


def test_snapshot_resume():
    import cupy as cp
    batch=example(); config,runtime=settings(iterations=40)
    saved={}
    def observe(phase,iteration,state):
        if iteration==20 and phase in ("post_ls","iteration_end"):
            saved[phase]={"iteration":iteration,"phase":phase,"flat_indices":state["flat_indices"].copy(),
                          "arrays":{k:cp.asnumpy(v) for k,v in state.items() if isinstance(v,cp.ndarray)}}
    result=solve_population_cuda_anytime(batch,config,[(None,None)],seed=43,runtime=runtime,
        control=SolverControl(instrumentation=InstrumentationConfig("heavy"),observer=observe))
    for snapshot in saved.values():
        resumed=solve_population_cuda_anytime(batch,config,[(None,None)],seed=43,runtime=runtime,
            control=SolverControl(instrumentation=InstrumentationConfig("heavy"),resume=snapshot))
        assert torch.equal(result.best_tour,resumed.best_tour)
        assert torch.equal(result.anytime_best,resumed.anytime_best)
