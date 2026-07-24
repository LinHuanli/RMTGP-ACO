"""YAML 规范、manifest 与无放回采样测试。"""

from __future__ import annotations

from rmtgp_aco.cli import _apply_runtime_overrides
from rmtgp_aco.config import ExecutionBackend, TransitionIntegration
from rmtgp_aco.manifest import (
    build_manifest,
    verify_manifest,
)
from rmtgp_aco.sampling import (
    ScaleStratifiedSampler,
    iter_problem_batches,
    pools_from_paths,
)
from rmtgp_aco.spec import load_run_spec

SQUARE = "0 0 1 0 1 1 0 1 output 1 2 3 4 1"


def test_repository_protocol_configs_resolve() -> None:
    for name in ("as", "acs", "mmas"):
        spec = load_run_spec(f"configs/{name}_protocol_a.yaml")
        assert set(spec.data.training_paths()) == {50, 100}
        assert len(spec.data.training_paths()[50]) == 10
        assert spec.data.test["tsplib_le500"].max_scale == 500
        assert spec.experiment.runtime.aco_backend is ExecutionBackend.NUMBA_BATCH
        assert spec.experiment.runtime.processes == 1
        assert spec.experiment.runtime.cpu_threads == 16
        assert spec.data.train_instances_per_scale == 16
        assert spec.data.baseline_policy == "require"

    for scale in (50, 100):
        spec = load_run_spec(f"configs/acs_protocol_a_tsp{scale}_only.yaml")
        assert set(spec.data.training_paths()) == {scale}
        assert spec.experiment.train_scales == (scale,)
        assert spec.data.train_instances_per_scale == 32


def test_cli_legacy_profile_keeps_common_budget() -> None:
    from argparse import Namespace

    spec = load_run_spec("configs/as_protocol_a.yaml")
    converted = _apply_runtime_overrides(
        spec,
        Namespace(
            root_seed=999,
            processes=2,
            method_profile="legacy",
        ),
    )
    assert converted.experiment.root_seed == 999
    assert converted.experiment.runtime.processes == 2
    assert converted.experiment.gp.transition_profile == "legacy"
    assert not converted.experiment.gp.train_pheromone
    assert (
        converted.experiment.aco.transition_integration
        is TransitionIntegration.REPLACEMENT
    )


def test_quick_manifest_inventory_and_verification(tmp_path) -> None:
    root = tmp_path / "TSP"
    train = root / "train_dataset" / "tsp" / "tsp_50"
    validation = root / "val_dataset" / "tsp"
    test = root / "test_dataset" / "tsp"
    train.mkdir(parents=True)
    validation.mkdir(parents=True)
    test.mkdir(parents=True)
    (train / "tsp50_uniform_128k_1.txt").write_text(SQUARE + "\n", encoding="utf-8")
    (validation / "tsp50_uniform_val.txt").write_text(SQUARE + "\n", encoding="utf-8")
    (test / "tsp50_concorde_5.688.txt").write_text(SQUARE + "\n", encoding="utf-8")
    (test / "tsp100_concorde_7.756 copy.txt").write_text(
        SQUARE + "\n",
        encoding="utf-8",
    )
    manifest = build_manifest(root, full_hashes=False)
    assert len(manifest.files) == 3
    assert all("copy" not in record.path for record in manifest.files)
    assert verify_manifest(
        manifest,
        root=root,
        verify_hashes=False,
    ) == []


def test_scale_sampler_uses_without_replacement_stream(tmp_path) -> None:
    path = tmp_path / "scale4.txt"
    path.write_text("\n".join([SQUARE] * 4) + "\n", encoding="utf-8")
    pools = pools_from_paths({4: [path]})
    sampler = ScaleStratifiedSampler(
        pools,
        root_seed=3,
        candidate_size=2,
        instances_per_scale=1,
    )
    identifiers = [
        sampler.cases_for_generation(generation)[0].batch.instance_ids[0]
        for generation in range(1, 5)
    ]
    assert len(set(identifiers)) == 4

    batches = list(
        iter_problem_batches(
            [path],
            batch_size=2,
            candidate_size=2,
            max_instances=3,
        )
    )
    assert sum(batch.batch_size for batch in batches) == 3


def test_scale_sampler_state_roundtrip(tmp_path) -> None:
    path = tmp_path / "scale4.txt"
    path.write_text("\n".join([SQUARE] * 4) + "\n", encoding="utf-8")
    pools = pools_from_paths({4: [path]})
    first = ScaleStratifiedSampler(
        pools,
        root_seed=17,
        candidate_size=2,
        instances_per_scale=1,
    )
    first.cases_for_generation(1)
    state = first.state_dict()
    expected = first.cases_for_generation(2)[0]

    restored = ScaleStratifiedSampler(
        pools,
        root_seed=17,
        candidate_size=2,
        instances_per_scale=1,
    )
    restored.load_state_dict(state)
    actual = restored.cases_for_generation(2)[0]
    assert actual.batch.instance_ids == expected.batch.instance_ids
    assert actual.seed == expected.seed
