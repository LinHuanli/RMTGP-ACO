"""机制日程与配置的 CPU 单元测试。"""
from dataclasses import replace
import pytest
from rmtgp_aco.mechanisms import MechanismConfig, factorial_conditions, native_schedule


def test_factorial_is_complete_and_independent():
    cells = factorial_conditions()
    assert len(cells) == 8
    assert cells["C111"] == MechanismConfig()
    for name, c in cells.items():
        assert (c.restart_policy == "native") == (name[1] == "1")
        assert c.floor_scale == int(name[2])
        assert (c.source_policy == "native_schedule") == (name[3] == "1")


@pytest.mark.parametrize("age,period", [(24,25),(25,5),(74,5),(75,3),(124,3),(125,2),(249,2),(250,1)])
def test_schedule_boundaries(age, period):
    assert native_schedule(age+2, 1, age+2)[0] == period


def test_global_best_boundary():
    assert native_schedule(500, 1, 450) == (1,1)
    assert native_schedule(500, 1, 449) == (1,2)


def test_invalid_controls_rejected_and_hashes_separate():
    with pytest.raises(ValueError):
        MechanismConfig(floor_scale=-1)
    with pytest.raises(ValueError):
        MechanismConfig(source_count=4)
    assert len({c.digest for c in factorial_conditions().values()}) == 8
    assert replace(MechanismConfig(), floor_scale=0).cuda_prefix(True) != MechanismConfig().cuda_prefix(True)
