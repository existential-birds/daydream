"""Supplementary contracts for bounded-patch metadata and snapshot evidence."""

from pathlib import Path

import pytest

from daydream.deep.diff import bound_deep_diff
from daydream.deep.routing_record import read_routing_record, write_routing_record


@pytest.mark.parametrize("patch, expected_bytes, expected_blocks", [
    pytest.param("", 0, 0, id="empty"),
    pytest.param("diff --git a/x b/x\n+++ b/x\n+é\n", 31, 1, id="utf8"),
    pytest.param("diff --git a/x b/x\n+++ b/x\n+diff --git é\n", 42, 1, id="header-text-in-content"),
])
def test_fitting_patch_reports_complete_retention(
    patch: str, expected_bytes: int, expected_blocks: int,
) -> None:
    bounded, info = bound_deep_diff(patch)

    assert bounded == patch
    assert info.truncated is False
    assert info.original_bytes == expected_bytes
    assert info.retained_bytes == expected_bytes
    assert info.total_blocks == info.retained_blocks == expected_blocks
    assert info.marker is None
    assert info.dropped_paths == info.oversize_paths == []


def test_oversized_first_block_counts_only_retained_patch_bytes() -> None:
    first = "diff --git a/x b/x\n+++ b/x\n+é\n"
    later = "diff --git a/y b/y\n+++ b/y\n+z\n"

    bounded, info = bound_deep_diff(first + later, budget=30)

    assert info.truncated is True
    assert info.original_bytes == 61
    assert info.retained_bytes == 31
    assert (info.total_blocks, info.retained_blocks) == (2, 1)
    assert info.oversize_paths == ["x"]
    assert info.dropped_paths == ["y"]
    assert info.marker is not None
    assert bounded == info.marker + first
    assert len(bounded.encode("utf-8")) == 31 + len(info.marker.encode("utf-8"))


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
