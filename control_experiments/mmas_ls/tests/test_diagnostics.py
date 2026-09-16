"""诊断产物事务、版本边界与严格缺失检查，不依赖 GPU。"""
import numpy as np
import pytest
from control_experiments.mmas_ls.diagnostics import Journal, array_digest, schema
from rmtgp_aco.mechanisms import InstrumentationConfig


def test_profile_is_explicit():
    assert not InstrumentationConfig("heavy").detailed
    with pytest.raises(ValueError): InstrumentationConfig(profile="mechanism_v3")
    assert InstrumentationConfig(profile="mechanism_v3",schema_version=3).detailed
    with pytest.raises(ValueError): InstrumentationConfig(profile="mechanism_v3",schema_version=3,aggregate_every=50)


def test_commit_and_replay(tmp_path):
    j=Journal(tmp_path,{"seed":1},min_free_bytes=0)
    values={"x":np.array([1,np.nan],dtype=np.float32)}
    meta={"kind":"iterations","shard":"a","start":1,"end":2}
    j.write("x.npz",values,meta); j.write("x.npz",values,meta)
    assert len(j.index["files"])==1
    with pytest.raises(ValueError): j.write("x.npz",{"x":np.zeros(2)},meta)
    with pytest.raises(ValueError): Journal(tmp_path,{"seed":2})
    with pytest.raises(ValueError): j.complete(["a"],2)  # 缺少固定采样
    (tmp_path/"x.npz").write_bytes(b"damaged")
    with pytest.raises(ValueError): j.verify()


def test_schema_distinguishes_missing_and_sampling():
    value=schema()
    assert value["version"]==3
    assert "每轮" in value["fields"]["source_hash"]["population"]
    assert "末点" in value["fields"]["tau_quantiles"]["population"]
    assert array_digest({"x":np.ones(1,dtype=np.float32)})!=array_digest({"x":np.ones(1,dtype=np.float64)})


def test_json_profile_and_fingerprint():
    import json
    from dataclasses import asdict
    from control_experiments.mmas_ls.diagnostic_analysis import edge_fingerprint
    config=InstrumentationConfig(profile="mechanism_v3",schema_version=3)
    assert InstrumentationConfig(**json.loads(json.dumps(asdict(config))))==config
    np.testing.assert_array_equal(edge_fingerprint([0,1,2,3,0]),edge_fingerprint([2,1,0,3,2]))
    assert not np.array_equal(edge_fingerprint([0,1,2,3,0]),edge_fingerprint([0,2,1,3,0]))
