"""报告读取、配对单位、诊断覆盖与输出的一致性测试；不需要 CUDA。"""
from dataclasses import asdict
from pathlib import Path
import copy
import numpy as np
import pytest

from control_experiments.mmas_ls.common import atomic_json, atomic_npz, digest, evaluation_seed, file_hash
from control_experiments.mmas_ls.report_inputs import checked_status, diagnostic_coverage, frozen_factorial_tasks, validate_manifest
from control_experiments.mmas_ls.report_diagnostics import iteration_metrics, sample_metrics, WINDOW_FIELDS
from control_experiments.mmas_ls.research_report import paired_summary, timing_tables, quality_tables, deployment_table
from control_experiments.mmas_ls.report_render import markdown_table
from control_experiments.mmas_ls.statistics import load_factorial, CONDITIONS
from rmtgp_aco.mechanisms import factorial_conditions, InstrumentationConfig


@pytest.fixture
def frozen(tmp_path):
    """两个实例、两个随机重复的完整八格，沿用正式 5000 轮数据结构。"""
    records=[{"coordinate_hash":f"instance-{i}","reference_length":10.} for i in range(2)]
    atomic_json(tmp_path/"manifests/diagnosis_dev.json",{"records":records,"manifest_hash":"manifest"})
    atomic_npz(tmp_path/"inputs/diagnosis_dev.npz",coords=np.zeros((2,500,2)))
    models=[{"id":f"mmas-{s}","file_hash":f"file-{s}","structural_hash":f"tree-{s}","seed":s} for s in (81001,81002,81003)]
    atomic_json(tmp_path/"manifests/checkpoints.json",models)
    entries=[{"id":"baseline","mode":"baseline","seed":None,"hash":"baseline"}]+[
        {"id":m["id"],"mode":"full","seed":m["seed"],"hash":m["structural_hash"],"file_hash":m["file_hash"]} for m in models]
    for c,mechanism in factorial_conditions().items():
        for r in range(2):
            task={"id":f"{c}-{r}","kind":"historical_mechanism","stage":"P1","split":"diagnosis_dev",
                  "variant":"mmas","condition":c,"replicate":r,"indices":[0,1],"iterations":5000,
                  "mechanism":asdict(mechanism),"modes":["full"],
                  "instrumentation":asdict(InstrumentationConfig(profile="mechanism_v3",schema_version=3))}
            atomic_json(tmp_path/"queue"/(task["id"]+".json"),{"task":task,"source_hash":"source"})
            scientific={"task":task,"seed":evaluation_seed("diagnosis_dev",r),"source":"source","manifest":"manifest",
                        "models":entries,"protocol":"protocol","trace_schema":[],
                        "input_sha256":file_hash(tmp_path/"inputs/diagnosis_dev.npz")}
            aco={"variant":"mmas","ants":32,"iterations":5000,"candidate_size":20,"local_search":"two_opt",
                 "local_search_candidate_size":20,"local_search_profile":"acotsp","local_search_dlb":True,
                 "rho":.2,"alpha":1.,"beta":2.,"gamma_transition":1/3,"gamma_pheromone":1/3,
                 "transition_integration":"residual","pheromone_integration":"budget_residual"}
            path=tmp_path/"jobs"/task["id"]
            atomic_json(path/"manifest.json",{**scientific,"scientific_hash":digest(scientific),"aco":aco,"runtime":{"cuda_precision":"fp32_fast"}})
            tour=np.broadcast_to(np.r_[np.arange(500),0],(4,2,501)).copy()
            atomic_npz(path/"raw.npz",gap=np.zeros((4,2)),length=np.full((4,2),10.),reference=np.full(2,10.),
                       auc=np.zeros((4,2)),anytime=np.full((4,2,5000),10.),tour=tour,
                       model_ids=np.array([m["id"] for m in entries]),modes=np.array(["baseline","full","full","full"]),
                       instance_hashes=np.array([r["coordinate_hash"] for r in records]))
            atomic_json(path/"status.json",{"status":"completed","files":{name:file_hash(path/name) for name in ("raw.npz","manifest.json")}})
    return tmp_path


def test_frozen_schema3_loader_and_missing(frozen):
    gaps,auc=load_factorial(frozen,"diagnosis_dev",2,2)
    assert gaps.shape==(8,4,2,2)
    assert not gaps.any() and not auc.any()
    # 缺失任务不能从可用子样本推断完整结果。
    (frozen/"queue/C111-0.json").unlink()
    with pytest.raises(ValueError,match="缺失结果"):
        load_factorial(frozen,"diagnosis_dev",2,2)


def test_duplicate_pair_rejected(frozen):
    import json
    path=frozen/"queue/C111-0.json"
    data=json.loads(path.read_text());data["task"]["indices"]=[0,0]
    atomic_json(path,data)
    with pytest.raises((ValueError,AssertionError)):
        load_factorial(frozen,"diagnosis_dev",2,2)


def test_condition_and_corruption_rejected(frozen):
    import json
    path=frozen/"queue/C111-0.json";data=json.loads(path.read_text())
    data["task"]["mechanism"]["floor_scale"]=0.
    atomic_json(path,data)
    with pytest.raises(ValueError,match="机制不一致"):
        frozen_factorial_tasks(frozen,"diagnosis_dev")
    raw=frozen/"jobs/C110-0/raw.npz"
    raw.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="损坏"):
        checked_status(raw.parent,("raw.npz",))


@pytest.mark.parametrize("mutation",["seed","model","source","ants"])
def test_manifest_rejects_scientific_changes(frozen,mutation):
    import json
    task=json.loads((frozen/"queue/C111-0.json").read_text())["task"]
    meta=json.loads((frozen/"jobs/C111-0/manifest.json").read_text())
    if mutation=="seed":meta["seed"]+=1
    elif mutation=="model":meta["models"][1]["file_hash"]="wrong"
    elif mutation=="source":meta["source"]="wrong"
    else:meta["aco"]["ants"]=64
    with pytest.raises(ValueError):validate_manifest(frozen,task,meta,"source")


def index_fixture():
    files={}
    for lo in (1,101):
        files[f"iter-{lo}"]={"metadata":{"kind":"iterations","shard":"a","start":lo,"end":lo+99,"flat_indices":[0,1]}}
    for i in [1]+list(range(25,201,25)):
        files[f"sample-{i}"]={"metadata":{"kind":"sample","shard":"a","iteration":i,"flat_indices":[0,1]}}
    return {"version":3,"completed":True,"horizon":200,"shards":["a"],"files":files}


def test_diagnostic_coverage_missing_and_overlap():
    index=index_fixture()
    assert diagnostic_coverage(index)=={0,1}
    damaged=copy.deepcopy(index);damaged["files"]["iter-101"]["metadata"]["start"]=100
    with pytest.raises(ValueError):diagnostic_coverage(damaged)
    del index["files"]["sample-25"]
    with pytest.raises(ValueError):diagnostic_coverage(index)


def test_pairing_unit_and_tail():
    baseline=np.zeros((32,5));gap=np.repeat(np.linspace(-.1,.1,32)[:,None],5,axis=1)
    row=paired_summary(gap,baseline)
    assert abs(row["delta_pp"])<1e-15
    assert row["wins"]+row["ties"]+row["losses"]==32
    assert row["worst10_delta_pp"]==pytest.approx(gap[-4:].mean())
    # 保持实例均值不变的种子重排不影响 bootstrap 区间。
    shuffled=gap+np.array([-.2,-.1,0,.1,.2])[None,:]
    other=paired_summary(shuffled,baseline)
    assert other["ci_low_pp"]==pytest.approx(row["ci_low_pp"])
    assert other["ci_high_pp"]==pytest.approx(row["ci_high_pp"])


def test_cross_block_repetition_and_ls_denominator():
    ls=np.zeros((1,25,32,3));ls[...,0]=12;ls[...,1]=10;ls[...,2]=250
    info=np.zeros((1,25,1,6));info[...,3]=np.arange(1,26)[None,:,None]
    moments=np.zeros((1,25,1,6));moments[...,0]=500
    data={"ls":ls,"source_info":info,"source_hash":np.ones((1,25,1,2),dtype=np.uint64),
          "ph_moments":moments,"trace":np.zeros((1,25,26)),"ls_counts":np.zeros((1,25,32,4)),"start":1}
    first,previous,repeat=iteration_metrics(data,np.array([10.]),None,0)
    assert first[0,0,WINDOW_FIELDS.index("ls_gain_pp")]==pytest.approx(20.)
    assert first[0,0,WINDOW_FIELDS.index("ls_retention")]==.5
    data["start"]=26
    second,_,_=iteration_metrics(data,np.array([10.]),previous,repeat)
    assert second[0,0,WINDOW_FIELDS.index("source_repeat_duration_end")]==50
    assert second[0,0,WINDOW_FIELDS.index("source_switch_fraction")]==0


def test_same_source_deposit_and_probe_missing():
    tr=np.full((1,16,5,22),np.nan);tr[...,19]=.2
    context=np.zeros((1,16,10),dtype=int);context[...,7]=5
    info=np.zeros((1,1,6));info[...,5]=2
    trace=np.zeros((1,26));trace[:,4]=2
    data={"tr":tr,"context":context,"deposit":np.full((1,1,5),.4),"source_valid":np.ones((1,1),dtype=bool),
          "source_info":info,"trace":trace,"tau_relative_l1":np.zeros((1,4)),"tau_above_nominal_max":np.zeros((1,4))}
    result=sample_metrics(data)
    assert result[0,0]==pytest.approx(np.log(5))
    assert result[0,3]==pytest.approx(0.)
    context[...,0]=-1
    with pytest.raises(ValueError):sample_metrics(data)


def test_incomplete_timing_excluded_and_markdown_units():
    base={"cohort":"N1","variant":"as","mode":"legacy","gpu_model":"A5000","instances":8}
    rows=timing_tables([{**base,"full_horizon_timing":True,"wall_seconds_last_attempt":100},
                        {**base,"full_horizon_timing":False,"wall_seconds_last_attempt":1}])
    assert rows[0]["median_seconds"]==100
    assert rows[0]["resumed_or_incomplete_timing"]==1
    text=markdown_table([{"effect":"J_RF|H=1","x":None}],[("effect","效应"),("x","pp")])
    assert "J_RF\\|H=1" in text and "—" in text


def test_all_tables_share_paired_aggregation_and_family():
    # baseline 在三个冠军平均中不能被重复加权；每实例的种子先求均值。
    base=np.linspace(1,2,32)[None,:,None]+np.zeros((8,32,5))
    gap=np.stack((base,base+.1,base+.2,base+.3),axis=1)
    numeric=np.zeros((2,3,4,32,5))
    numeric[:]=gap[0]
    tables=quality_tables({"factorial":gap,"f_auc":gap,"numeric":numeric,"n_auc":numeric})
    assert len(tables["contrasts"])==18
    assert len(tables["per_champion"])==54
    assert all(row["delta_pp"]==pytest.approx(.2) for row in tables["factorial"])
    for row in tables["main_results"]:
        assert row["wins"]+row["ties"]+row["losses"]==32
        if row["model"]=="GP_mean":assert row["delta_pp"]==pytest.approx(.2)
    assert all(row["change_vs_legacy_pp"]==0 for row in tables["numerical_effects"])


def test_deployment_uses_frozen_gate_not_development_quality():
    numeric=np.zeros((2,3,4,32,5));numeric[:,:,1:]=3.
    flags={f"{v}-{s}":s==81002 for v in ("as","mmas") for s in (81001,81002,81003)}
    rows=deployment_table(numeric,flags)
    assert all(r["passed_champions"]==1 and r["gap"]==1 for r in rows)
