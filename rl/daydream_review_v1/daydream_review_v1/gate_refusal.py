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
