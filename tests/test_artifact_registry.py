"""证据等级及存储边界不能依赖随手起的目录名称。"""

import pytest

from gpaco import artifact_registry as registry
from gpaco.data import ROOT


def test_registry_ids_and_evidence_tiers_are_explicit():
    entries = registry.entries()
    assert len({r["id"] for r in entries}) == len(entries)
    assert not any(r["tier"] == "formal" for r in entries)
    assert registry.identity("E01-p01-cpu-baselines")["evidence_tier"] == "pilot"
    assert registry.identity("E01-p01-work-diagnostics")["evidence_tier"] == "diagnostic"


def test_output_rejects_ad_hoc_formal_and_smoke_destinations():
    for path in (
        "artifacts/my-new-results",
        "artifacts/runs/formal/E01/p01/cpu-1",
        "artifacts/runs/smoke/E00/p01/cpu-1",
        "artifacts/runs/pilot/E01/p01/cpu-baselines/../../wrong",
    ):
        with pytest.raises(ValueError, match="输出必须"):
            registry.require_output(ROOT / path, "E01-p01-cpu-baselines")
    path = ROOT / "artifacts/runs/pilot/E01/p01/cpu-baselines/tsp100-as-g001-a01"
    assert registry.require_output(path, "E01-p01-cpu-baselines") == path


def test_resolution_never_silently_chooses_duplicate_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    row = {"id": "test", "path": "target", "legacy_path": "legacy"}
    monkeypatch.setattr(registry, "entry", lambda _: row)
    (tmp_path / "legacy").mkdir()
    assert registry.resolve("test") == tmp_path / "legacy"
    (tmp_path / "target").mkdir()
    with pytest.raises(ValueError, match="两份"):
        registry.resolve("test")


def test_explicit_compatibility_alias_is_not_duplicate_data(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    monkeypatch.setattr(registry, "entry", lambda _: {"path": "target", "legacy_path": "legacy"})
    (tmp_path / "target").mkdir()
    (tmp_path / "legacy").symlink_to("target", target_is_directory=True)
    assert registry.resolve("example") == tmp_path / "target"


def test_moved_inputs_remain_content_verified():
    from gpaco.benchmark_inputs import BenchmarkInputs

    path = registry.resolve("shared-frozen-population-p01") / "tsp100/as"
    if not path.exists():
        pytest.skip("输入包尚未生成")
    programs, problem = BenchmarkInputs(path).load(1, 0)
    assert len(programs) == 100 and problem.size == 32
    assert not problem.coords.flags.writeable


def test_existing_reports_keep_evidence_tiers_separate():
    import json

    for tier, experiment in [("pilot", "E01"), ("diagnostic", "E01"), ("historical", "E02")]:
        directory = ROOT / f"docs/results/{tier}/{experiment}/p01"
        manifest = directory / "provenance.json"
        if not manifest.exists():
            pytest.skip("尚未生成分类报告")
        record = json.loads(manifest.read_text())
        assert record["evidence_tier"] == tier
        assert record["formal_result"] is False
        assert (directory / "README.md").exists()
        assert (directory / "tables").is_dir()
        assert not list(directory.glob("*.npz"))
        assert not list(directory.glob("*.png"))
