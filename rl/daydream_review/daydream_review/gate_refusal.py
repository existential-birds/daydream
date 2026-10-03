"""Stage-3 taskset loading requires a passed Stage-0 held-out gate report. Re-read it unconditionally
before scheduling: missing, unreadable, or failed reports raise Stage0GateRefused. No flag or
default-allow path may admit an unvalidated reward model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.training.gate import _evidence_digest as _evidence_digest
from daydream.training.reward_model import OutcomeModel


class Stage0GateRefused(ValueError):
    """Raised when a Stage-3 run is refused because the Stage-0 gate has not passed."""


def require_stage0_gate(gate_report_path: Path) -> dict[str, Any]:
    """Read a GateReport.to_dict payload and return it for provenance stamping. Raise Stage0GateRefused
    for missing, unreadable, unparseable, or non-passed reports; diagnostics name the path and
    reason.
    """
    if not gate_report_path.is_file():
        raise Stage0GateRefused(
            f"Stage-0 gate report missing at {gate_report_path}: the offline gate has not run, "
            "so no rollout may be scheduled. Run the Stage-0 gate first and point "
            "--taskset.gate-report-path at its report."
        )
    try:
        report = json.loads(gate_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0GateRefused(
            f"Stage-0 gate report at {gate_report_path} is unreadable: {exc}. "
            "A corrupt gate report is a refusal, never an implicit pass."
        ) from exc
    if not isinstance(report, dict) or report.get("passed") is not True:
        passed = report.get("passed") if isinstance(report, dict) else None
        raise Stage0GateRefused(
            f"Stage-0 gate failed: report at {gate_report_path} records passed={passed!r}. "
            "The learned reward model did not clear the offline gate, so Stage-3 training is refused."
        )
    return report


def require_outcome_model_bound(gate_report: dict[str, Any], outcome_model_path: Path) -> OutcomeModel:
    """Capture the OutcomeModel checkpoint bound to the passed gate. Recompute
    evidence_digest from checkpoint split_digest/model_fingerprint and report thresholds,
    held_out_rows, separation, calibration, and accepted_ratio.

    Raise Stage0GateRefused for a missing/unparseable checkpoint, absent report measurements, or a
    mismatched digest; a passed report cannot authorize an unrelated model.
    """
    if not outcome_model_path.is_file():
        raise Stage0GateRefused(
            f"Stage-0 outcome model missing at {outcome_model_path}: a checkpoint is configured "
            "but absent, so the report's evidence_digest cannot be bound to any model. "
            "A rollout set may not score against a model the gate never evaluated."
        )
    try:
        state = json.loads(outcome_model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0GateRefused(
            f"Stage-0 outcome model at {outcome_model_path} is unreadable: {exc}. "
            "A corrupt checkpoint is a refusal, never an implicit pass."
        ) from exc
    if not isinstance(state, dict) or not state.get("split_digest"):
        raise Stage0GateRefused(
            f"Stage-0 outcome model at {outcome_model_path} carries no split_digest; "
            "the checkpoint cannot be bound to the gate report's evidence."
        )
    fields = ("thresholds", "held_out_rows", "separation", "calibration", "accepted_ratio")
    missing = [name for name in fields if name not in gate_report]
    if missing:
        raise Stage0GateRefused(
            f"Stage-0 gate report lacks the recomputable evidence {'/'.join(missing)}; "
            "the checkpoint cannot be bound to it. A hand-rolled report is a refusal, "
            "never an implicit pass."
        )
    recomputed = _evidence_digest(
        {
            "split_digest": state["split_digest"],
            "model_fingerprint": state.get("model_fingerprint", ""),
            "thresholds": gate_report["thresholds"],
            "held_out_rows": gate_report["held_out_rows"],
            "separation": gate_report["separation"],
            "calibration": gate_report["calibration"],
            "accepted_ratio": gate_report["accepted_ratio"],
        }
    )
    if recomputed != gate_report.get("evidence_digest"):
        raise Stage0GateRefused(
            f"Stage-0 outcome model at {outcome_model_path} does not bind to the gate report: "
            f"evidence_digest recomputes to {recomputed}, not the report's "
            f"{gate_report.get('evidence_digest')!r}. The checkpoint is not the model the "
            "gate evaluated, so Stage-3 training is refused."
        )
    try:
        return OutcomeModel(**state)
    except (TypeError, ValueError) as exc:
        raise Stage0GateRefused(
            f"Stage-0 outcome model at {outcome_model_path} has invalid checkpoint state: {exc}. "
            "A malformed checkpoint cannot schedule scored rollouts."
        ) from exc
