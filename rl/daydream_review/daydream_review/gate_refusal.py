"""Stage-0 gate refusal (M4): a Stage-3 rollout set may not start without gate evidence.

The offline Stage-0 gate (``daydream/training/gate.py``) validates the learned
outcome model on a frozen held-out split and writes a ``GateReport``. Before
any rollout can be scheduled, the taskset load path re-reads that report here,
unconditionally, and refuses closed on any doubt: a missing file, an
unreadable payload, or a report that did not pass all stop the run with a
:class:`Stage0GateRefused`. There is no flag, no bypass, and no
default-to-allowed branch — an unvalidated reward model never trains a policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class Stage0GateRefused(ValueError):
    """Raised when a Stage-3 run is refused because the Stage-0 gate has not passed."""


def require_stage0_gate(gate_report_path: Path) -> dict[str, Any]:
    """Read and validate the Stage-0 gate report at *gate_report_path*.

    Args:
        gate_report_path: Path to a ``GateReport.to_dict()`` payload written by
            the Stage-0 offline gate.

    Returns:
        The parsed gate report (so the caller can stamp provenance, e.g. the
        ``evidence_digest``, into run artifacts).

    Raises:
        Stage0GateRefused: When the file is missing, unreadable or unparseable,
            or when the report does not record ``passed: true``. The message
            names the reason and the report path.
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


def _evidence_digest(payload: dict[str, Any]) -> str:
    """SHA-256 over the sorted evidence payload (mirror of ``gate.py``)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_outcome_model_bound(gate_report: dict[str, Any], outcome_model_path: Path) -> None:
    """Bind the Stage-0 outcome model checkpoint to the passed gate report (M4).

    The gate's ``evidence_digest`` covers exactly ``{split_digest,
    model_fingerprint, thresholds, held_out_rows, separation, calibration,
    accepted_ratio}`` (``daydream/training/gate.py``): the first two come from
    the checkpoint state, the rest are the report's clear-text measurements, so
    the digest is recomputable at load and must match. A passed report plus an
    unrelated checkpoint must not cross the Stage-3 boundary — only the exact
    model the gate evaluated may schedule rollouts.

    Args:
        gate_report: The passed report returned by :func:`require_stage0_gate`.
        outcome_model_path: Path to the Stage-0 outcome model checkpoint
            (``OutcomeModel.state_dict()`` payload).

    Raises:
        Stage0GateRefused: When the checkpoint is missing or unparseable, when
            the report lacks the recomputable measurements, or when the
            recomputed digest does not equal the report's ``evidence_digest``.
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
