"""独立数值对照的 GPU 验收，不把 legacy 的已知错误标为数学正确。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .common import atomic_json,atomic_npz,read_json,experiment,now
from .diagnostic_validation import pilot
from .terminal_oracle import inspect,normalized
from .prepare import batch
from .evaluate import program_entries


def operator_probe(mode,directory):
    """直接调用生产 CUDA 的稳定统计 helper，覆盖常量、近常量和大动态范围。"""
    import cupy as cp
    from rmtgp_aco.aco_cuda import _CUDA_SOURCE
    from rmtgp_aco.mechanisms import MechanismConfig
    source=MechanismConfig(terminal_statistics=mode).cuda_prefix(False)+_CUDA_SOURCE.read_text()+r'''
extern "C" __global__ void numeric_probe(const float* input,float* output,int n,int logarithmic) {
    const float* row=input+blockIdx.x*n;
    CenteredTerminal s{};
    for(int j=0;j<n;++j) s.mean+=logarithmic?terminal_log_ratio(row[j],row[0],1e-12f):static_cast<TerminalReal>(row[j])-row[0];
    s.mean/=static_cast<TerminalReal>(n);
    for(int j=0;j<n;++j) {
        const TerminalReal x=logarithmic?terminal_log_ratio(row[j],row[0],1e-12f):static_cast<TerminalReal>(row[j])-row[0];
        const TerminalReal d=x-s.mean;s.deviation+=d*d;
    }
    s.deviation=terminal_sqrt(s.deviation/static_cast<TerminalReal>(n));
    for(int j=0;j<n;++j) {
        const TerminalReal x=logarithmic?terminal_log_ratio(row[j],row[0],1e-12f):static_cast<TerminalReal>(row[j])-row[0];
        output[blockIdx.x*n+j]=terminal_normalized(x,s);
    }
}
'''
    kernel=cp.RawModule(code=source,options=("--std=c++17","--use_fast_math"),name_expressions=("numeric_probe",)).get_function("numeric_probe")
    rows=[];arrays={}
    for n in (1,2,20,500):
        x=np.stack([np.full(n,.281851381,dtype=np.float32),
                    np.linspace(.281851381,.281851381+1e-6,n,dtype=np.float32),
                    np.geomspace(1e-12,1e12,n,dtype=np.float32)])
        for logarithmic in (0,1):
            actual=cp.empty_like(cp.asarray(x));kernel((3,),(1,),(cp.asarray(x),actual,np.int32(n),np.int32(logarithmic)))
            actual=cp.asnumpy(actual)
            expected=normalized(np.log(x.astype(float)) if logarithmic else x.astype(float))
            error=float(np.max(abs(actual-expected)))
            passed=bool(np.allclose(actual,expected,atol=2e-5,rtol=2e-5))
            rows.append({"n":n,"logarithmic":bool(logarithmic),"max_abs_error":error,"passed":passed})
            arrays[f"input_{n}_{logarithmic}"]=x;arrays[f"actual_{n}_{logarithmic}"]=actual
            arrays[f"reference_{n}_{logarithmic}"]=expected
    atomic_npz(Path(directory)/"operator_probe.npz",**arrays)
    atomic_json(Path(directory)/"operator_probe.json",rows)
    if not all(r["passed"] for r in rows):raise ArithmeticError("稳定统计算子没有通过 FP64 oracle")
    return rows


def validate(task,out):
    from rmtgp_aco import aco_cuda as cuda
    from rmtgp_aco.mechanisms import SolverControl,InstrumentationConfig
    out=Path(out);target=out/"jobs"/task["id"];mode=task["numeric_mode"];variant=task["variant"]
    target.mkdir(parents=True,exist_ok=True)
    if mode!="legacy":operator_probe(mode,target)
    report=pilot(target,task["instances"],task["steps"],(variant,),True,mode,out)
    oracle=inspect(target/variant)
    if mode!="legacy" and oracle["status"]!="passed":
        raise ArithmeticError(f"{mode} 的 terminal 输入重算失败；不能放行质量实验")
    problem=batch("diagnosis_dev",range(task["instances"]),out)
    aco,runtime=experiment(variant)
    programs=[e["program"] for e in program_entries(variant)]
    with np.load(target/f"{variant}-off.npz") as data:
        expected_tour=data["tour"].copy();expected_curve=data["anytime"].copy()
    if mode=="legacy":
        # 同一 Python 包装和同一设备，只替换为原冻结 CUDA 文件；不修改快照。
        old=Path(task["historical_snapshot"])/"src/rmtgp_aco/cuda"
        names=("_CUDA_SOURCE","_CUDA_V2_SOURCE","_CUDA_LS_SOURCE")
        saved={name:getattr(cuda,name) for name in names}
        try:
            for name in names:setattr(cuda,name,old/saved[name].name)
            reference=cuda.solve_population_cuda_anytime(problem,aco,programs,seed=57231,runtime=runtime,
                control=SolverControl(instrumentation=InstrumentationConfig("off"),stop_iteration=task["steps"]))
        finally:
            for name,value in saved.items():setattr(cuda,name,value)
        np.testing.assert_array_equal(expected_tour,reference.best_tour.numpy())
        np.testing.assert_array_equal(expected_curve,reference.anytime_best.numpy()[...,:task["steps"]])
        equivalence="historical_frozen_cuda_exact"
    else:
        reference=cuda.solve_population_cuda_anytime(problem,aco,[(None,None)],seed=57231,runtime=runtime,
            control=SolverControl(instrumentation=InstrumentationConfig("off"),stop_iteration=task["steps"]))
        np.testing.assert_array_equal(expected_tour[0],reference.best_tour.numpy()[0])
        np.testing.assert_array_equal(expected_curve[0],reference.anytime_best.numpy()[0,:,:task["steps"]])
        equivalence="baseline_unchanged_by_numeric_mode"
    report.update(status="completed",validation_status="passed",numeric_mode=mode,
                  oracle_status=oracle["status"],equivalence=equivalence,completed_at=now(),
                  scope="数值对照验收；legacy 已知偏差保留为被研究对象，不认证数学正确，不放行确认集")
    atomic_json(target/"status.json",report)
    return report
