"""FP64 原求解器的默认等价与干预预算。"""
from dataclasses import replace
import numpy as np
import torch
from rmtgp_aco.aco_numba import solve_numba
from rmtgp_aco.config import ACOConfig
from rmtgp_aco.data import TSPInstance, coordinate_hash, make_problem_batch
from rmtgp_aco.mechanisms import SolverControl, factorial_conditions


def test_fp64_native_and_factorial():
    coords=np.random.default_rng(31).random((12,2))
    tour=np.r_[np.arange(12),0]
    batch=make_problem_batch([TSPInstance("cpu-oracle",coords,tour,10.,coordinate_hash(coords))],candidate_size=6)
    config=replace(ACOConfig.acotsp_local_search_default("mmas",local_search="two_opt",iterations=35),
                   ants=32,candidate_size=6,local_search_candidate_size=6)
    native=solve_numba(batch,config,seed=7)
    for name,mechanism in factorial_conditions().items():
        control=SolverControl(mechanism=mechanism,collected=[])
        result=solve_numba(batch,config,seed=7,control=control)
        if name=="C111":
            assert torch.equal(result.best_tour,native.best_tour)
            assert torch.equal(result.anytime_best,native.anytime_best)
        trace=control.collected[0]["trace"]
        assert np.max(trace[...,20])<1e-10
        if name[1]=="0": assert np.all(trace[...,8]==0)
        if name[2]=="0": assert np.all(trace[...,6]==0)
        if name[3]=="0": assert np.all(trace[...,0]==0)
