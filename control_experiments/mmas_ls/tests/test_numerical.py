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
