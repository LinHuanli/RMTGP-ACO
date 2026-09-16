"""缓存、精度、来源和任务清单的 CPU 侧检查。"""
import numpy as np
import pytest
from control_experiments.mmas_ls.common import atomic_json,read_json,digest,evaluation_seed
from control_experiments.mmas_ls.campaign import lock
from control_experiments.mmas_ls.probes import tour_hash,tour_edges
from control_experiments.mmas_ls.references import generate
from control_experiments.mmas_ls.evaluate import factorial_tasks
from control_experiments.mmas_ls.extensions import extension_specification


def test_atomic_lock_excludes_other_claim_and_releases(tmp_path):
    path=tmp_path/"lock"
    with lock(path) as a:
        assert a
        with lock(path) as b:assert not b
    with lock(path) as c:assert c


def test_hashes_and_seeds():
    assert digest({"b":2,"a":1})==digest({"a":1,"b":2})
    assert evaluation_seed("confirm_uniform",0)!=evaluation_seed("confirm_uniform",1)
    assert tour_hash([0,1,2,3,0])==tour_hash([2,1,0,3,2])
    assert tour_hash([0,1,2,3,0])!=tour_hash([0,2,1,3,0])


def test_task_pairing_and_sizes():
    tasks=factorial_tasks("confirm_uniform")
    assert len(tasks)==320
    assert len(set(t["id"] for t in tasks))==320
    assert sum(len(t["indices"])*4 for t in tasks)==40960
    assert len(factorial_tasks("diagnosis_dev"))==40
    extension_specification()  # 所有干预参数必须通过严格配置验证。


@pytest.mark.parametrize("distribution",["cluster","gaussian"])
def test_independent_generator(distribution):
    a=generate(distribution,0); b=generate(distribution,1)
    assert a.shape==(500,2) and np.all((a>0)&(a<1))
    assert np.array_equal(a,generate(distribution,0))
    assert not np.array_equal(a,b)
    assert len(np.unique(a,axis=0))==500
