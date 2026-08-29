"""Stage-0 gate refusal (M4): a Stage-3 run may not start without gate evidence.

Every path here is fail-closed: a missing, unreadable, or failed gate report
refuses the run and names the reason. There is no default-to-allowed branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream_review_v1.gate_refusal import Stage0GateRefused, require_stage0_gate


def test_start_refused_without_gate_evidence(tmp_path: Path) -> None:
    with pytest.raises(Stage0GateRefused, match="gate"):
        require_stage0_gate(gate_report_path=tmp_path / "missing.json")  # M4: refuse when evidence missing


def test_start_refused_on_unparseable_gate_report(tmp_path: Path) -> None:
    """A corrupt gate report raises, never defaults to allowed."""
    (tmp_path / "gate.json").write_text("{not json")
    with pytest.raises(Stage0GateRefused, match="gate"):
        require_stage0_gate(gate_report_path=tmp_path / "gate.json")


def test_start_refused_on_failed_gate(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text('{"passed": false, "separation": 0.01}')
    with pytest.raises(Stage0GateRefused, match="failed"):
        require_stage0_gate(gate_report_path=tmp_path / "gate.json")


def test_start_allowed_on_passed_gate(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text('{"passed": true, "evidence_digest": "abc"}')
    report = require_stage0_gate(gate_report_path=tmp_path / "gate.json")  # no raise
    assert report["evidence_digest"] == "abc"


def test_coordinator_gate_report_is_consumed_unmodified(tmp_path: Path) -> None:
    """Stage-boundary contract audit: the gate-report.json the coordinator's
    Stage-0 stage writes must satisfy require_stage0_gate verbatim — no
    reshaping at the handoff, no nested wrapper that reads as passed=None."""
    from daydream.training.coordinator import PipelineConfig, run_pipeline

    fixture = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "training" / "records-50" / "records.jsonl"
    manifest = run_pipeline(
        PipelineConfig(corpus=fixture, out_dir=tmp_path, stages=("stage0",)), dry_run=True
    )
    report_path = tmp_path / "stage0" / "gate-report.json"
    assert manifest["stages"]["stage0"]["gate"]["evidence_digest"] == (
        __import__("json").loads(report_path.read_text())["evidence_digest"]
    )
    report = require_stage0_gate(gate_report_path=report_path)  # no raise
    assert report["passed"] is True
