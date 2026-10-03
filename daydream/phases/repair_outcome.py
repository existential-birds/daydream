"""The bounded vocabulary of a single test-repair turn, and how a host abort maps onto it.

A repair turn has exactly five possible fates, and the host — not the model —
decides which one happened. This module is pure vocabulary plus classification:
no I/O, no state, and no new public stop reason (every interruption converges on
an existing :class:`~daydream.review_result.ReasonCode`).
"""

from __future__ import annotations

from enum import StrEnum

from daydream.review_result import ReasonCode, reason_for_budget


class RepairOutcome(StrEnum):
    """What the host observed a repair turn to be.

    ``CANDIDATE_COMPLETE`` and ``SCOPE_BLOCKED`` are reachable only from the
    explicit scope-request and host-validation paths, never from
    :func:`classify_repair_outcome` — an interrupted turn never claims either.
    """

    CANDIDATE_COMPLETE = "candidate_complete"
    DIAGNOSIS_UNRESOLVED = "diagnosis_unresolved"
    BUDGET_INTERRUPTED = "budget_interrupted"
    EXECUTION_ERROR = "execution_error"
    SCOPE_BLOCKED = "scope_blocked"


# The stop vocabulary `reason_for_budget` owns. A reason already outside that
# table is a non-budget host stop, never a guess about which budget ran out.
_BUDGET_REASON_CODES = frozenset({
    ReasonCode.MODEL_BUDGET_EXHAUSTION,
    ReasonCode.HOST_WALL_BUDGET_EXHAUSTION,
    ReasonCode.HOST_TOOL_BUDGET_EXHAUSTION,
    ReasonCode.HOST_PIPELINE_BUDGET_EXHAUSTION,
})


def repair_reason_code(reason: str | None) -> ReasonCode | None:
    """Map a repair turn's host abort reason onto the existing public reason vocabulary.

    The budget table is delegated, never re-implemented: an unrecognised reason
    is reported as the generic backend failure rather than a budget the host
    cannot prove. ``None`` — a turn that ended on its own — stays ``None``.
    """
    if reason is None:
        return None
    try:
        return reason_for_budget(reason)
    except KeyError:
        pass
    try:
        return ReasonCode(reason)
    except ValueError:
        return ReasonCode.BACKEND_FAILURE


def classify_repair_outcome(abort_reason: str | None, output: str) -> RepairOutcome:
    """Classify a turn the host aborted or let run to its own end.

    A host abort always wins over whatever the turn said: prose claiming a fix
    inside an interrupted turn is partial diagnosis, not completion. Absent an
    abort, a blank turn is an unresolved diagnosis — the host cannot read success
    out of silence.
    """
    if abort_reason is not None:
        code = repair_reason_code(abort_reason)
        if code in _BUDGET_REASON_CODES:
            return RepairOutcome.BUDGET_INTERRUPTED
        return RepairOutcome.EXECUTION_ERROR
    if not output.strip():
        # Silence is not a success claim: the turn is recorded and the job re-runs.
        return RepairOutcome.DIAGNOSIS_UNRESOLVED
    return RepairOutcome.DIAGNOSIS_UNRESOLVED
