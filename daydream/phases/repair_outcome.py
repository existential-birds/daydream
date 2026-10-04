"""The bounded vocabulary of a single test-repair turn, and how a host abort maps onto it.

A repair turn has exactly five possible fates, and the host — not the model —
decides which one happened. This module is pure vocabulary plus classification:
no I/O, no state, and no new public stop reason (every interruption converges on
an existing :class:`~daydream.review_result.ReasonCode`).

It also owns the one parser a repair turn's *scope request* enters through. A
turn may ask to widen the authorization policy when the failure it worked from
points at a path the run never authorized. That text is untrusted, so
:func:`repair_scope_request` canonicalizes every requested value through
``repository_paths`` — which rejects absolute paths, traversal, and symlink
crossings with a non-reflective error — and reports the evidence the turn
supplied alongside the path. Nothing here decides whether the request is
granted: that is the coordinator's call, against the footprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from daydream.redaction import redact_text
from daydream.repository_paths import InvalidRepositoryFilePath, canonicalize_repository_file_path
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


#: A recorded evidence excerpt is a name of nothing: it is bounded and redacted
#: exactly like the checkpoint's, because it is model prose either way.
SCOPE_REQUEST_EVIDENCE_MAX_CHARS = 500


@dataclass(frozen=True)
class ScopeRequest:
    """One repair turn's request to widen the authorization policy.

    ``requested`` is already canonical, so no later stage has to re-validate
    model text. ``evidence`` links each requested path to the failure the turn
    worked from; a path with none is still requested, and the coordinator —
    not this parser — decides whether an unattested request widens anything.
    """

    requested: tuple[str, ...]
    evidence: Mapping[str, str]
    evidence_source: str


def repair_scope_request(repo: Path, payload: object) -> ScopeRequest | None:
    """Parse a repair turn's scope request, or ``None`` when it asked for nothing.

    Fail closed in both directions. A payload that is not an object, or that
    carries no ``paths`` list, is a turn that requested nothing. A payload whose
    ``paths`` is present but not a list of strings, or that names a path the
    repository cannot confine, raises
    :class:`~daydream.repository_paths.InvalidRepositoryFilePath` rather than
    quietly dropping the entry: a malformed request must never become a
    narrower-than-asked authorization, and it must never be silently erased
    either. Redacted per-path evidence is kept under the canonical path so the
    audit event can name why a widening was approved.
    """
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise InvalidRepositoryFilePath("invalid scope request payload")
    raw_paths = payload.get("paths")
    if raw_paths is None:
        return None
    if not isinstance(raw_paths, list):
        raise InvalidRepositoryFilePath("invalid scope request paths")
    raw_evidence = payload.get("evidence")
    if raw_evidence is not None and not isinstance(raw_evidence, Mapping):
        raise InvalidRepositoryFilePath("invalid scope request evidence")
    requested: list[str] = []
    evidence: dict[str, str] = {}
    for value in raw_paths:
        normalized = canonicalize_repository_file_path(repo, value)
        if normalized in requested:
            continue
        requested.append(normalized)
        note = raw_evidence.get(normalized) if raw_evidence is not None else None
        if isinstance(note, str) and note.strip():
            evidence[normalized] = redact_text(note.strip())[-SCOPE_REQUEST_EVIDENCE_MAX_CHARS:]
    # Sorted, not input order: two turns naming the same paths in a different
    # order are the same request, and the record must read identically.
    requested.sort()
    source = payload.get("evidence_source")
    return ScopeRequest(
        requested=tuple(requested),
        evidence=evidence,
        evidence_source=source.strip() if isinstance(source, str) and source.strip() else "",
    )


__all__ = [
    "SCOPE_REQUEST_EVIDENCE_MAX_CHARS",
    "RepairOutcome",
    "ScopeRequest",
    "classify_repair_outcome",
    "repair_reason_code",
    "repair_scope_request",
]
