"""统计符号、零方差与相关多重比较的模拟验收。"""
import numpy as np
from control_experiments.mmas_ls.statistics import contrasts,simultaneous_interval,classify


def test_contrasts_and_interactions():
    names,m=contrasts()
    assert m.shape==(18,8)
    assert np.all(m[:10].sum(axis=1)==0)
    assert np.array_equal(m[10:],np.eye(8))
    delta=np.array([0,3,2,5,1,4,3,6])
    assert np.array_equal((m@delta)[:3],[1,2,3])
    assert np.allclose((m@delta)[6:10],0)


def test_zero_variance_and_reproducibility():
    x=np.zeros((32,18)); x[:,0]=.125
    a=simultaneous_interval(x,1000); b=simultaneous_interval(x,1000)
    assert np.array_equal(a[0],a[1]); assert np.array_equal(a[1],a[2])
    assert np.array_equal(a[0],b[0]); assert classify(-.008,.009)=="等效区间内"


def test_simulated_family_coverage():
    # 100 个预定模拟数据集；容差为 Monte Carlo 诊断而非精确覆盖率证明。
    rng=np.random.default_rng(81982); covered=0
    _,m=contrasts()
    for _ in range(100):
        x=rng.normal(size=(128,8))+rng.normal(size=(128,1))
        _,low,high,_,_=simultaneous_interval(x@m.T,1000,seed=712)
        covered+=int(np.all(low<=0) and np.all(high>=0))
    assert 85<=covered<=100
