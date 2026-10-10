"""局部搜索清单、资格门禁、调度比例与配对证据的非GPU检查。"""

import sys
from collections import Counter

import pytest
import yaml

from gpaco.config import ExecutionPlan, SearchConfig
from gpaco.data import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from local_search_pool import choose_task, mark_dependencies, tasks_for
from run_local_search_worker import identical_pair


def config():
    return yaml.safe_load((ROOT / "configs/workloads/local_search_queue.yaml").read_text())


def test_registered_ls_matrix_and_no_test_access():
    cfg = config()
    tasks = tasks_for(cfg)
    assert Counter(t["kind"] for t in tasks) == {
        "tuning": 12,
        "smoke": 12,
        "training": 36,
        "profile": 12,
        "performance": 90,
    }
    assert len({t["id"] for t in tasks}) == 162
    assert cfg["training"]["generations"] == 50
    assert cfg["training"]["search"]["iterations"] == 500
    assert cfg["training"]["population"] == 100
    assert not cfg["standard_test_opened"]
    assert all(t["seed"] in (1001, 1002, 1003) for t in tasks if t["kind"] == "training")
    assert {t["variant"] for t in tasks} == {"as", "acs", "mmas"}


def test_tuning_and_smoke_are_required_and_failure_not_retried():
    tasks = tasks_for(config())
    assert {t["kind"] for t in mark_dependencies(tasks)} == {"tuning"}
    tuning = tasks[0]
    tuning["status"] = "completed"
    ready = mark_dependencies(tasks)
    assert any(t["kind"] == "smoke" for t in ready)
    assert not any(t["kind"] == "training" for t in ready)
    smoke = next(t for t in tasks if t["kind"] == "smoke" and tuning["id"] in t["dependencies"])
    smoke["status"] = "excluded_contended"
    mark_dependencies(tasks)
    assert all(
        t["status"] == "blocked_dependency" for t in tasks if smoke["id"] in t["dependencies"]
    )


def test_dispatch_two_training_one_measurement_and_borrow_slots():
    cfg = config()
    ready = [dict(id="a", kind="training", n=500), dict(id="b", kind="performance", n=100)]
    assert [choose_task(ready, p, cfg)["kind"] for p in range(3)] == [
        "training",
        "training",
        "performance",
    ]
    assert choose_task(ready[:1], 2, cfg)["kind"] == "training"


def test_full_budget_pair_rejects_changed_output():
    a = dict(
        status="completed",
        program_hashes=["x"],
        input_manifest_sha256="input",
        seed=2,
        search={"local_search": "two_opt"},
        requested_tasks=3200,
        lengths_array_sha256="same",
        tours_array_sha256="same",
    )
    assert identical_pair([a, dict(a)])
    with pytest.raises(ValueError, match="改变了固定语义"):
        identical_pair([a, {**a, "tours_array_sha256": "different"}])
    assert not identical_pair([a, {"status": "infeasible"}])


def test_invalid_ls_configuration_rejected():
    with pytest.raises(ValueError):
        SearchConfig(local_search="three_opt", candidate_size=5, ls_candidate_size=20)
    with pytest.raises(ValueError):
        ExecutionPlan(ls_executor="best_improvement")
    assert SearchConfig(candidate_size=3).local_search == "none"
