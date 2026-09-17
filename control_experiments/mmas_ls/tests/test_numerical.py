"""数值实验必须显式分版，验收按显卡型号隔离。"""
import pytest
from rmtgp_aco.mechanisms import MechanismConfig
from control_experiments.mmas_ls.common import atomic_json
from control_experiments.mmas_ls.numerical_campaign import ready,validation_ids,NUMERIC_MODES
from control_experiments.mmas_ls.campaign import ALLOWED_GPU_MODELS


def test_numeric_configuration_is_explicit():
    assert MechanismConfig().terminal_statistics=="legacy"
    configs=[MechanismConfig(terminal_statistics=m) for m in NUMERIC_MODES]
    assert len({c.digest for c in configs})==3
    assert len({c.cuda_prefix(False) for c in configs})==3
    with pytest.raises(ValueError):MechanismConfig(terminal_statistics="automatic_best_gap")


def test_numeric_gate_is_model_specific_and_source_checked(tmp_path):
    task={"kind":"numeric_pair","variant":"mmas"}
    atomic_json(tmp_path/"protocol/numerical.json",{"authorized_scope":"development_numeric_and_historical"})
    assert not ready(task,tmp_path)
    for name in validation_ids(ALLOWED_GPU_MODELS[0],"mmas"):
        atomic_json(tmp_path/"queue"/(name+".json"),{"source_hash":"frozen"})
        atomic_json(tmp_path/"jobs"/name/"status.json",{"status":"completed","validation_status":"passed",
            "specification":{"source":{"source_hash":"frozen"}}})
    assert ready(task,tmp_path,ALLOWED_GPU_MODELS[0])
    assert ready(task,tmp_path)
    assert not ready(task,tmp_path,ALLOWED_GPU_MODELS[1])
    name=validation_ids(ALLOWED_GPU_MODELS[0],"mmas")[1]
    atomic_json(tmp_path/"queue"/(name+".json"),{"source_hash":"changed"})
    assert not ready(task,tmp_path)


def test_historical_gate_does_not_assert_math_correctness(tmp_path):
    task={"kind":"historical_mechanism","variant":"mmas"}
    atomic_json(tmp_path/"protocol/numerical.json",{"authorized_scope":"development_numeric_and_historical"})
    name=validation_ids(ALLOWED_GPU_MODELS[0],"mmas",("legacy",))[0]
    atomic_json(tmp_path/"queue"/(name+".json"),{"source_hash":"frozen"})
    atomic_json(tmp_path/"jobs"/name/"status.json",{"status":"completed","validation_status":"passed",
        "oracle_status":"mismatch","specification":{"source":{"source_hash":"frozen"}}})
    assert not ready(task,tmp_path)
    atomic_json(tmp_path/"gates/historical_development.json",{"status":"approved"})
    assert ready(task,tmp_path)
    assert not ready({**task,"kind":"numeric_pair"},tmp_path)


def test_paused_validation_preserves_artifacts_and_logs(tmp_path):
    from control_experiments.mmas_ls.campaign import archive_paused_validation
    from control_experiments.mmas_ls.common import read_json
    task={"id":"check","kind":"numeric_validation"};folder=tmp_path/"jobs/check"
    atomic_json(folder/"status.json",{"status":"resource_paused"})
    atomic_json(folder/"partial.json",{"saved":True})
    (folder/"attempt-1.log").write_text("interrupt evidence")
    target=archive_paused_validation(task,tmp_path)
    assert read_json(target/"partial.json")["saved"]
    assert (folder/"attempt-1.log").read_text()=="interrupt evidence"
    assert not (folder/"partial.json").exists()
    assert archive_paused_validation(task,tmp_path) is None
    atomic_json(folder/"status.json",{"status":"completed"})
    atomic_json(folder/"partial.json",{"saved":True})
    assert archive_paused_validation(task,tmp_path) is None


def test_resource_migration_preserves_evidence_and_frozen_queue(tmp_path):
    from control_experiments.mmas_ls.resource_migration import relax_pairs
    from control_experiments.mmas_ls.campaign import affinity_path
    from control_experiments.mmas_ls.common import read_json,file_hash
    task={"id":"pair","kind":"numeric_pair","split":"dev","replicate":4,"indices":[8,9],
          "numeric_order":list(NUMERIC_MODES)}
    queue=tmp_path/"queue/pair.json";atomic_json(queue,{"task":task,"source_hash":"frozen"})
    original_hash=file_hash(queue);affinity=affinity_path(task,tmp_path)
    atomic_json(affinity,{"gpu_model":"NVIDIA RTX A4000"})
    atomic_json(tmp_path/"jobs/pair/status.json",{"status":"resource_paused"})
    atomic_json(tmp_path/"jobs/pair--legacy/evidence.json",{"keep":True})
    policy=relax_pairs(tmp_path,["pair"]);record=policy["unrestricted_tasks"]["pair"]
    from pathlib import Path
    assert read_json(Path(record["archive"])/"pair--legacy/evidence.json")["keep"]
    assert file_hash(queue)==original_hash and affinity.exists()
    assert not (tmp_path/"jobs/pair").exists() and affinity_path(task,tmp_path) is None
    with pytest.raises(ValueError):affinity_path({**task,"replicate":5},tmp_path)
    assert affinity_path({**task,"id":"another"},tmp_path) is not None
    with pytest.raises(ValueError):relax_pairs(tmp_path,["pair"])
