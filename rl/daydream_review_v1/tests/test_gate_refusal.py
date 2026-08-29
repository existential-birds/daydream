"""Stage-0 gate refusal (M4): a Stage-3 run may not start without gate evidence.

Every path here is fail-closed: a missing, unreadable, or failed gate report
refuses the run and names the reason. There is no default-to-allowed branch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from daydream_review_v1.gate_refusal import (
    Stage0GateRefused,
    require_outcome_model_bound,
    require_stage0_gate,
)

MODEL_STATE = {
    "weights": {"bug": 1.0, "race": 0.5, "regression": 0.75},
    "bias": -0.25,
    "split_digest": "split-digest",
    "label_ratio_reported": 0.5,
    "train_rows": 10,
    "held_out_rows": 4,
    "held_out_accuracy": 0.75,
    "model_fingerprint": "abc12345",
}


def _evidence_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _bound_gate_report() -> dict[str, object]:
    evidence = {
        "split_digest": MODEL_STATE["split_digest"],
        "model_fingerprint": MODEL_STATE["model_fingerprint"],
        "thresholds": {"min_separation": 0.1, "min_calibration": 0.5},
        "held_out_rows": MODEL_STATE["held_out_rows"],
        "separation": 0.2,
        "calibration": 0.75,
        "accepted_ratio": 0.5,
    }
    return {
        "passed": True,
        "separation": evidence["separation"],
        "calibration": evidence["calibration"],
        "accepted_ratio": evidence["accepted_ratio"],
        "evidence_digest": _evidence_digest(evidence),
        "thresholds": evidence["thresholds"],
        "held_out_rows": evidence["held_out_rows"],
    }


def _write_checkpoint(tmp_path: Path, *, split_digest: str = "split-digest") -> Path:
    p = tmp_path / "outcome-model.json"
    p.write_text(json.dumps({**MODEL_STATE, "split_digest": split_digest}), encoding="utf-8")
    return p


def test_start_refused_without_gate_evidence(tmp_path: Path) -> None:
    gate_report_path = tmp_path / "missing.json"
    with pytest.raises(Stage0GateRefused, match="missing") as exc_info:
        require_stage0_gate(gate_report_path=gate_report_path)  # M4: refuse when evidence missing
    assert str(gate_report_path) in str(exc_info.value)  # contract: the message names the reason and the path


def test_start_refused_on_unparseable_gate_report(tmp_path: Path) -> None:
    """A corrupt gate report raises, never defaults to allowed."""
    gate_report_path = tmp_path / "gate.json"
    gate_report_path.write_text("{not json")
    with pytest.raises(Stage0GateRefused, match="unreadable") as exc_info:
        require_stage0_gate(gate_report_path=gate_report_path)
    assert str(gate_report_path) in str(exc_info.value)  # contract: the message names the reason and the path


def test_start_refused_on_failed_gate(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text('{"passed": false, "separation": 0.01}')
    with pytest.raises(Stage0GateRefused, match="failed"):
        require_stage0_gate(gate_report_path=tmp_path / "gate.json")


def test_start_allowed_on_passed_gate(tmp_path: Path) -> None:
    (tmp_path / "gate.json").write_text('{"passed": true, "evidence_digest": "abc"}')
    report = require_stage0_gate(gate_report_path=tmp_path / "gate.json")  # no raise
    assert report["evidence_digest"] == "abc"


def test_checkpoint_bound_to_passed_report(tmp_path: Path) -> None:
    """The fixture pair (report + matching checkpoint) passes the binding."""
    report = _bound_gate_report()
    require_outcome_model_bound(report, _write_checkpoint(tmp_path))  # no raise


def test_checkpoint_refused_when_missing(tmp_path: Path) -> None:
    """A configured but absent checkpoint refuses: nothing to bind the digest to."""
    with pytest.raises(Stage0GateRefused, match="missing") as exc_info:
        require_outcome_model_bound(_bound_gate_report(), tmp_path / "no-model.json")
    assert str(tmp_path / "no-model.json") in str(exc_info.value)


def test_checkpoint_refused_on_unparseable_state(tmp_path: Path) -> None:
    """A corrupt checkpoint refuses, never defaults to intrinsic-only."""
    p = tmp_path / "outcome-model.json"
    p.write_text("{not json")
    with pytest.raises(Stage0GateRefused, match="unreadable"):
        require_outcome_model_bound(_bound_gate_report(), p)


def test_checkpoint_refused_on_digest_mismatch(tmp_path: Path) -> None:
    """M4 binding: a passed report plus an unrelated checkpoint is a refusal."""
    p = _write_checkpoint(tmp_path, split_digest="some-other-split")
    with pytest.raises(Stage0GateRefused, match="does not bind") as exc_info:
        require_outcome_model_bound(_bound_gate_report(), p)
    assert str(p) in str(exc_info.value)


def test_checkpoint_refused_on_report_without_measurements(tmp_path: Path) -> None:
    """A hand-rolled report lacking the recomputable fields cannot bind."""
    report = {"passed": True, "evidence_digest": "fixture-digest"}
    with pytest.raises(Stage0GateRefused, match="lacks the recomputable evidence"):
        require_outcome_model_bound(report, _write_checkpoint(tmp_path))


def test_coordinator_gate_report_is_consumed_unmodified(tmp_path: Path) -> None:
    """Stage-boundary contract audit: the gate-report.json the coordinator's
    Stage-0 stage writes must satisfy require_stage0_gate verbatim — no
    reshaping at the handoff, no nested wrapper that reads as passed=None."""
    from daydream.training.coordinator import PipelineConfig, run_pipeline

    fixture = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "training" / "records-50" / "records.jsonl"
    run_pipeline(
        PipelineConfig(corpus=fixture, out_dir=tmp_path, stages=("stage0",)), dry_run=True
    )
    report_path = tmp_path / "stage0" / "gate-report.json"
    # The behavioral contract: the on-disk gate-report.json (manifest "gate" and the
    # file both derive from the same report.to_dict(), so a manifest-vs-file digest
    # comparison is tautological) must satisfy the Stage-3 boundary consumer verbatim.
    report = require_stage0_gate(gate_report_path=report_path)  # no raise
    assert report["passed"] is True
