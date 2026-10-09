"""报告回归检查：独立ACO随机block不得在类别轴上相互覆盖。"""

import sys

from gpaco.data import ROOT, write_json

sys.path.insert(0, str(ROOT / "scripts"))
import report_baseline_diagnostics as report


def test_diagnostic_blocks_are_separate_and_incomplete_attempt_is_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "resolve", lambda _: tmp_path)
    monkeypatch.setattr(report, "file_hash", lambda _: "fixture-sha")
    monkeypatch.setattr(report, "export", lambda *args: None)
    fields = (
        "candidate_list_probe_positions",
        "fallback_rank_visited_checks",
        "fallback_rank_distance_pairs",
        "candidate_lane0_cycles_sum",
        "stats_lane0_cycles_sum",
        "feature_gp_score_lane0_cycles_sum",
        "selection_sync_acs_lane0_cycles_sum",
    )
    for block in range(3):
        attempt = tmp_path / f"block-{block}"
        write_json(attempt / "job.json", {})
        if block < 2:
            write_json(attempt / "COMPLETE.json", {"clean": True})
        write_json(
            attempt / "measurement/record.json",
            {
                "status": "completed",
                "n": 100,
                "variant": "as",
                "generation": 1,
                "block": block,
                "counts": {"effective_transitions": 10, **dict.fromkeys(fields, 10)},
                "result_sha256": "fixture-sha",
                "device_search_s": 1.0,
                "state_replay": {"snapshots": 1, "fallback_snapshots": 0},
            },
        )
        write_json(
            attempt / "measurement/instrumentation_pair.json",
            {
                "bitwise_equal_tours": True,
                "bitwise_equal_lengths": True,
                "plain": {},
                "instrumented": {},
            },
        )

    def check_figure(fig, *args):
        axis = fig.axes[1]
        assert len(axis.patches) == 8
        centers = {round(p.get_x() + p.get_width() / 2, 6) for p in axis.patches}
        assert centers == {0.0, 1.0}
        assert [t.get_text() for t in axis.get_xticklabels()] == [
            "100/as/g1/b0",
            "100/as/g1/b1",
        ]
        report.plt.close(fig)

    monkeypatch.setattr(report, "save", check_figure)
    assert len(report.deep_diagnostics(tmp_path)) == 2
