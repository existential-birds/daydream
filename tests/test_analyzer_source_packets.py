"""Evaluation recognizes completed host evidence without fabricating tool reads."""

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.eval.analyzer import analyze_coverage


def _artifacts(tmp_path: Path, receipts: Any, records: Any) -> Path:
    deep = tmp_path / "deep"
    deep.mkdir()
    (tmp_path / "diff.patch").write_text("diff --git a/api.py b/api.py\n")
    (deep / "coverage-receipts.json").write_text(json.dumps(receipts))
    (deep / "stack-python-records.json").write_text(json.dumps(records))
    return tmp_path


def test_completed_packet_counts_coverage_without_tool_reads(tmp_path: Path) -> None:
    directory = _artifacts(tmp_path, {
        "python": {"assigned_files": ["api.py"], "source_packet_files": ["api.py"]},
    }, {"issues": [], "verdicts": [{"path": "api.py", "verdict": "clean"}]})

    coverage = analyze_coverage({"main": None, "forked": []}, directory)

    assert coverage["coverage_ratio"] == 1.0
    assert coverage["files_reviewed"] == 1
    assert coverage["source_packet_reviewed"] == 1
    assert coverage["files_read_by_reviewers"] == 0
    assert coverage["artifact_reads_rejected"] == 0
    assert coverage["uncovered_files"] == []


def test_sharded_stack_packet_counts_coverage(tmp_path: Path) -> None:
    directory = _artifacts(tmp_path, {
        "python#1": {"assigned_files": ["api.py"], "source_packet_files": ["api.py"]},
    }, {"issues": [], "verdicts": [{"path": "api.py", "verdict": "clean"}]})
    (directory / "deep/stack-python-records.json").rename(directory / "deep/stack-python#1-records.json")
    assert analyze_coverage({"main": None, "forked": []}, directory)["coverage_ratio"] == 1.0


@pytest.mark.parametrize(("receipts", "records"), [
    ({}, {"verdicts": [{"path": "api.py", "verdict": "clean"}]}),
    ({"other": {"assigned_files": ["api.py"], "source_packet_files": ["api.py"]}},
     {"verdicts": [{"path": "api.py", "verdict": "clean"}]}),
    ({"python": {"assigned_files": [], "source_packet_files": ["api.py"]}},
     {"verdicts": [{"path": "api.py", "verdict": "clean"}]}),
    ({"python": {"assigned_files": ["api.py"], "source_packet_files": []}},
     {"verdicts": [{"path": "api.py", "verdict": "clean"}]}),
    ({"python": {"assigned_files": ["api.py"], "source_packet_files": ["api.py"]}},
     {"issues": [{"file": "api.py"}], "verdicts": []}),
    ({"python": {"assigned_files": ["api.py"], "source_packet_files": ["api.py"]}},
     {"issues": [{"file": "api.py"}],
      "verdicts": [{"path": "api.py", "verdict": "not_reviewed"}]}),
])
def test_packet_requires_same_stack_assignment_and_final_verdict(
    tmp_path: Path, receipts: Any, records: Any,
) -> None:
    directory = _artifacts(tmp_path, receipts, records)

    coverage = analyze_coverage({"main": None, "forked": []}, directory)

    assert coverage["coverage_ratio"] == 0.0
    assert coverage["files_reviewed"] == 0
    assert coverage["source_packet_reviewed"] == 0
    assert coverage["uncovered_files"] == ["api.py"]
