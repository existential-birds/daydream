"""Supplementary contracts for bounded-patch metadata and snapshot evidence."""

from pathlib import Path

from daydream.deep.routing_record import read_routing_record, write_routing_record


def test_snapshot_evidence_replaces_stale_metrics_and_preserves_phase_slices(tmp_path: Path) -> None:
    write_routing_record(tmp_path, {
        "risk": {"signals": {"diff_bytes": 99}, "stale_metric": 1},
        "diff_population": {"full_patch": {"bytes": 99, "stale_metric": 1}, "stale_population": {}},
        "wonder": {"outcome": "run", "effort": "high"},
        "arbiter": {"sharded": True},
    })
    risk = {"signals": {"diff_bytes": 31}, "floors": []}
    population = {"diff_key": "captured-key", "full_patch": {"bytes": 31}}

    write_routing_record(tmp_path, {"risk": risk, "diff_population": population})
    write_routing_record(tmp_path, {"wonder": {"reason": "resumed phase"}})

    record = read_routing_record(tmp_path)
    assert record["risk"] == risk
    assert record["diff_population"] == population
    assert record["wonder"] == {"outcome": "run", "effort": "high", "reason": "resumed phase"}
    assert record["arbiter"] == {"sharded": True}
