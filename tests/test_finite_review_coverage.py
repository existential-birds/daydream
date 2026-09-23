"""Host-supplied source needs a completed verdict before it credits coverage."""

import json
from pathlib import Path

import pytest

from daydream.deep.artifacts import per_stack_failures_path, per_stack_records_path
from daydream.deep.coverage import _receipt_covered_files
from daydream.deep.review_steps import _reconcile_stack_verdicts
from daydream.review_budget import review_warnings


def test_source_packet_credits_only_own_completed_files(tmp_path: Path) -> None:
    per_stack_records_path(tmp_path, "python").write_text(json.dumps({
        "issues": [],
        "verdicts": [
            {"path": "api.py", "verdict": "clean"},
            {"path": "pending.py", "verdict": "not_reviewed"},
        ],
    }))
    receipts = {
        "python": {"source_packet_files": ["api.py", "pending.py"]},
        "other": {"source_packet_files": ["unread.py"]},
    }
    covered, counts = _receipt_covered_files(
        ["api.py", "pending.py", "unread.py"], receipts, tmp_path,
    )
    assert covered == {"api.py"}
    assert counts["source_packet_reviewed"] == 1


def test_unavailable_evidence_remains_a_reported_warning(tmp_path: Path) -> None:
    per_stack_failures_path(tmp_path).write_text(json.dumps({
        "python": "evidence incomplete: client.py unavailable",
    }))
    assert review_warnings(tmp_path) == (
        "python: evidence incomplete: client.py unavailable",
    )


def test_packet_reconciliation_requires_own_receipt_and_completed_verdict(
    tmp_path: Path,
) -> None:
    deep_dir = tmp_path / "deep"
    deep_dir.mkdir()
    (deep_dir / "coverage-receipts.json").write_text(json.dumps({
        "python": {"source_packet_files": ["api.py", "pending.py", "lib/other.py"]},
        "other": {"source_packet_files": ["other.py"]},
    }))
    paths = ["api.py", "pending.py", "other.py", "unread.py"]
    verdicts = _reconcile_stack_verdicts(
        tmp_path, None, "python", assigned_files=paths,
        declared_verdicts=[
            {"path": path, "verdict": "not_reviewed" if path == "pending.py" else "clean"}
            for path in paths
        ], parsed_records=[],
    )
    assert [entry["verdict"] for entry in verdicts] == [
        "clean", "not_reviewed", "not_reviewed", "not_reviewed",
    ]


def test_invalid_packet_receipt_does_not_discard_finding(tmp_path: Path) -> None:
    deep_dir = tmp_path / "deep"
    deep_dir.mkdir()
    (deep_dir / "coverage-receipts.json").write_text("invalid json")
    verdicts = _reconcile_stack_verdicts(
        tmp_path, None, "python", assigned_files=["api.py"],
        declared_verdicts=[], parsed_records=[{"file": "api.py"}],
    )
    assert verdicts[0]["verdict"] == "has_findings"


@pytest.mark.parametrize("packet_files", [[], ["api.py"]])
def test_finite_incomplete_file_with_proven_finding_remains_uncovered(
    tmp_path: Path, packet_files: list[str],
) -> None:
    deep_dir = tmp_path / "deep"
    deep_dir.mkdir()
    receipts = {"python": {"source_packet_files": packet_files}}
    (deep_dir / "coverage-receipts.json").write_text(json.dumps(receipts))
    verdicts = _reconcile_stack_verdicts(
        tmp_path, None, "python", assigned_files=["api.py"],
        declared_verdicts=[{"path": "api.py", "verdict": "not_reviewed"}],
        parsed_records=[{"file": "api.py"}],
    )
    assert verdicts[0]["verdict"] == "not_reviewed"
    assert verdicts[0]["n_findings"] == 1
    per_stack_records_path(deep_dir, "python").write_text(json.dumps({
        "issues": [{"file": "api.py"}], "verdicts": verdicts,
    }))
    covered, _ = _receipt_covered_files(["api.py"], receipts, deep_dir)
    assert covered == set()
