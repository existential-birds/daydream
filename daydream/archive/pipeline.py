"""Per-phase pipeline state derivation for archived runs.

Derives per-phase terminal states (merge/fix/test) and the ``pipeline_status``
aggregate from existing deep artifacts + phase events — no new runtime
instrumentation. Separates pipeline outcome from archive finalization so a run
that merged-failed and never tested is never archived as unqualified success.

Artifact reads are best-effort: absent/empty/malformed artifacts mean "we
cannot know the phase ran", reported honestly as ``absent``/neutral — these
functions never raise on bad input.

Exports:
    derive_phase_states: Per-phase terminal states from artifacts + events.
    derive_pipeline_status: Aggregate pipeline outcome from archive state +
        per-phase states.
"""

from pathlib import Path
from typing import Any

from daydream.archive import _read_json_artifact
from daydream.trajectory import DaydreamPhase

# Phase status values shared by phase_states entries and pipeline_status.
_SUCCEEDED = "succeeded"
_FAILED = "failed"
_PARTIAL = "partial"
_ABSENT = "absent"
_UNKNOWN = "unknown"


def _deep_dir(target_dir: Path) -> Path:
    return target_dir / ".daydream" / "deep"


def _merge_state(target_dir: Path) -> dict[str, Any]:
    """Derive the merge phase terminal state from deep artifacts.

    Discriminator (issue #762 spike): ``per-stack-failures.json`` is written
    with ``{"__merge__": ...}`` ONLY on the merge-failure consolidation path,
    while ``merged-items.json`` is written on BOTH the failure path and the
    success paths — so the merge key wins over merged-items presence.
    """
    failures = _read_json_artifact(_deep_dir(target_dir) / "per-stack-failures.json", dict)
    if failures is not None and "__merge__" in failures:
        return {"ran": True, "status": _FAILED}
    if (_deep_dir(target_dir) / "merged-items.json").is_file():
        return {"ran": True, "status": _SUCCEEDED}
    return {"ran": False, "status": _ABSENT}


def _fix_state(target_dir: Path, phase_events: list[Any]) -> dict[str, Any]:
    """Derive the fix terminal state from ``fix-failures.json`` + phase events.

    A present non-empty ``fix-failures.json`` means fix groups were
    dropped/reverted (partial); otherwise a ``phase_start`` for
    ``DaydreamPhase.FIX`` marks the fix phase as having run successfully.
    """
    fix_failures = _read_json_artifact(_deep_dir(target_dir) / "fix-failures.json", dict)
    if fix_failures:
        return {"ran": True, "status": _PARTIAL}
    for ev in phase_events or []:
        phase = getattr(ev, "phase", None)
        if phase is DaydreamPhase.FIX and getattr(ev, "event", None) == "phase_start":
            return {"ran": True, "status": _SUCCEEDED}
    return {"ran": False, "status": _ABSENT}


def _test_state(target_dir: Path) -> dict[str, Any]:
    """Derive the test terminal state from ``test-verdict.json``.

    The verdict is written for BOTH outcomes before the failure early-return, so
    presence ⇔ the test phase ran; ``passed: false`` ⇔ failed.
    """
    verdict = _read_json_artifact(_deep_dir(target_dir) / "test-verdict.json", dict)
    if verdict is None:
        return {"ran": False, "status": _ABSENT}
    if verdict.get("passed") is False:
        return {"ran": True, "status": _FAILED}
    return {"ran": True, "status": _SUCCEEDED}


def _absent() -> dict[str, Any]:
    """Return the neutral per-phase state for a phase this run did not execute."""
    return {"ran": False, "status": _ABSENT}


def derive_phase_states(
    target_dir: Path,
    *,
    phase_events: list[Any],
    runs_merge: bool = True,
    runs_fix: bool = True,
    runs_test: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return per-phase terminal states for ``merge``, ``fix``, and ``test``.

    Each entry is ``{"ran": bool, "status": str}`` with status one of
    ``succeeded``/``failed``/``partial``/``absent``/``unknown``. Pure
    derivation over on-disk deep artifacts + recorder phase events; never
    raises on absent/malformed artifacts (they read as ``absent``).

    The deep artifacts (``per-stack-failures.json`` / ``merged-items.json`` /
    ``test-verdict.json`` / ``fix-failures.json``) are session-agnostic -- they
    live in ``target_dir/.daydream/deep`` with no run-bound ``session_id``. A
    non-deep flow run against a previously deep-reviewed repo would otherwise
    inherit a PRIOR run's artifacts as its own pipeline state. ``runs_merge`` /
    ``runs_fix`` / ``runs_test`` gate each phase read to only the phases the
    current flow actually executes; a phase the flow never runs reads
    ``absent`` (neutral) regardless of what stale artifacts sit on disk.
    """
    return {
        "merge": _merge_state(target_dir) if runs_merge else _absent(),
        "fix": _fix_state(target_dir, phase_events) if runs_fix else _absent(),
        "test": _test_state(target_dir) if runs_test else _absent(),
    }


def _phase(phase_states: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a phase's state dict, defaulting to ``{}`` when absent/malformed.

    Unifies the two key styles used across ``derive_pipeline_status`` (reading
    ``"status"`` vs ``"ran"``) into one spelling, so an absent phase entry
    reads as an empty dict: ``.get("status")`` -> ``None`` and ``.get("ran")``
    -> ``None`` both degrade to the all-absent case instead of raising.
    """
    return phase_states.get(name) or {}


def derive_pipeline_status(
    archive_status: str,
    fix_failures: dict[str, Any] | None,
    phase_states: dict[str, Any],
    *,
    runs_fix: bool = False,
    runs_test: bool = False,
) -> str:
    """Aggregate pipeline outcome from the archive state + per-phase states.

    Precedence:
    1. ``cancelled`` when the archive is ``partial`` with no fix failures
       (``write_partial`` signal flush — the run stopped early, nothing failed).
    2. ``failed`` when merge or test reports failed.
    3. ``partial`` when ``fix_failures`` are present.
    4. ``partial`` when a phase the flow runs never ran (run stopped early).
    5. ``succeeded`` when every phase is succeeded or absent AND at least one
       phase actually ran (a flow that runs neither fix nor test surfaces no
       derivable phase signal, so an early-aborted/failed run must not be
       archived as unqualified success).
    6. else ``unknown``.
    """
    if archive_status == "partial" and not fix_failures:
        return "cancelled"
    merge_status = _phase(phase_states, "merge").get("status")
    test_status = _phase(phase_states, "test").get("status")
    if merge_status == _FAILED or test_status == _FAILED:
        return _FAILED
    if fix_failures:
        return _PARTIAL
    if (runs_test and _phase(phase_states, "test").get("ran") is False) or (
        runs_fix and _phase(phase_states, "fix").get("ran") is False
    ):
        return _PARTIAL
    for name in ("merge", "fix", "test"):
        status = _phase(phase_states, name).get("status")
        if status is not None and status not in (_SUCCEEDED, _ABSENT):
            return _UNKNOWN
    # Only claim success when at least one derivable phase actually ran. A flow
    # that runs neither fix nor test (TTT review, improve-only) surfaces no
    # phase evidence, so every phase reading ``absent`` means we cannot tell a
    # clean profile from an early aborted/failed run — report ``unknown``.
    if any(
        _phase(phase_states, name).get("status") == _SUCCEEDED
        for name in ("merge", "fix", "test")
    ):
        return _SUCCEEDED
    return _UNKNOWN
