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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from daydream.archive import _read_json_artifact
from daydream.timeutil import parse_iso_timestamp
from daydream.trajectory import (
    DaydreamPhase,
    LifecycleReasonCode,
    LifecycleStatus,
)

# Phase status values shared by phase_states entries and pipeline_status.
_SUCCEEDED = "succeeded"
_FAILED = "failed"
_PARTIAL = "partial"
_ABSENT = "absent"
_UNKNOWN = "unknown"


def _deep_dir(target_dir: Path) -> Path:
    return target_dir / ".daydream" / "deep"


def _event_value(event: Any, key: str) -> Any:
    """Read one raw or typed phase-event field without normalizing it away."""
    if isinstance(event, Mapping):
        return event.get(key)
    return getattr(event, key, None)


def _phase_value(value: Any) -> str | None:
    if isinstance(value, DaydreamPhase):
        return value.value
    return value if isinstance(value, str) else None


def _enum_value(value: Any) -> str | None:
    if isinstance(value, (LifecycleStatus, LifecycleReasonCode)):
        enum_value = value.value
        return enum_value if isinstance(enum_value, str) else None
    return value if isinstance(value, str) else None


def _event_rows(phase_events: Any) -> tuple[Sequence[Any], bool]:
    """Return raw rows plus whether the containing value itself is malformed."""
    if phase_events is None:
        return (), False
    if isinstance(phase_events, Sequence) and not isinstance(phase_events, (str, bytes, bytearray)):
        return phase_events, False
    return (), True


def _unknown() -> dict[str, Any]:
    return {"ran": True, "status": _UNKNOWN}


def _legacy_merge_state(target_dir: Path) -> dict[str, Any]:
    """Read the strict pre-identity artifact fallback."""
    deep = _deep_dir(target_dir)
    failures_path = deep / "per-stack-failures.json"
    if failures_path.is_file():
        failures = _read_json_artifact(failures_path, dict)
        if failures is None:
            return _unknown()
        if "__merge__" in failures:
            merge_failure = failures["__merge__"]
            if not isinstance(merge_failure, dict):
                return _unknown()
            message = merge_failure.get("message")
            if not isinstance(message, str) or not message.strip():
                return _unknown()
            return {"ran": True, "status": _FAILED}

    items_path = deep / "merged-items.json"
    if items_path.is_file():
        items = _read_json_artifact(items_path, dict)
        if items is None or not isinstance(items.get("items"), list):
            return _unknown()
        return {"ran": True, "status": _SUCCEEDED}
    return {"ran": False, "status": _ABSENT}


def _merge_state(
    target_dir: Path,
    phase_events: Any,
    *,
    session_id: str | None,
) -> dict[str, Any]:
    """Derive merge truth from a valid current-session pair or strict legacy artifacts."""
    rows, malformed_container = _event_rows(phase_events)
    if session_id is None:
        if malformed_container:
            return _unknown()
        if any(
            _event_value(event, "session_id") is not None
            or _event_value(event, "scope_id") is not None
            for event in rows
        ):
            return _unknown()
        return _legacy_merge_state(target_dir)

    if not session_id:
        return _unknown()
    if malformed_container:
        return _unknown()

    current: list[Any] = []
    malformed_current = False
    for event in rows:
        if _phase_value(_event_value(event, "phase")) != DaydreamPhase.MERGE.value:
            continue
        event_session = _event_value(event, "session_id")
        if isinstance(event_session, str) and event_session and event_session != session_id:
            continue
        if event_session != session_id:
            malformed_current = True
            continue
        current.append(event)

    if malformed_current:
        return _unknown()
    if not current:
        return {"ran": False, "status": _ABSENT}

    scopes: dict[str, dict[str, list[Any]]] = {}
    for event in current:
        scope_id = _event_value(event, "scope_id")
        kind = _event_value(event, "event")
        timestamp = _event_value(event, "timestamp")
        if (
            not isinstance(scope_id, str)
            or not scope_id
            or not isinstance(kind, str)
            or kind not in {"phase_start", "phase_end"}
        ):
            return _unknown()
        if not isinstance(timestamp, str):
            return _unknown()
        try:
            parse_iso_timestamp(timestamp)
        except ValueError:
            return _unknown()
        bucket = scopes.setdefault(scope_id, {"phase_start": [], "phase_end": []})
        bucket[kind].append(event)

    if len(scopes) != 1:
        return _unknown()
    pair = next(iter(scopes.values()))
    if len(pair["phase_start"]) != 1 or len(pair["phase_end"]) != 1:
        return _unknown()
    start = pair["phase_start"][0]
    end = pair["phase_end"][0]
    try:
        end_before_start = parse_iso_timestamp(
            _event_value(end, "timestamp")
        ) < parse_iso_timestamp(_event_value(start, "timestamp"))
    except (TypeError, ValueError):
        return _unknown()
    if end_before_start:
        return _unknown()
    if _event_value(start, "status") is not None:
        return _unknown()
    reason = _event_value(end, "reason_code")
    if reason is not None and _enum_value(reason) not in {item.value for item in LifecycleReasonCode}:
        return _unknown()
    status = _enum_value(_event_value(end, "status"))
    if status == LifecycleStatus.SUCCEEDED.value:
        return {"ran": True, "status": _SUCCEEDED}
    if status == LifecycleStatus.PARTIAL.value:
        return {"ran": True, "status": _PARTIAL}
    if status == LifecycleStatus.SKIPPED.value:
        return {"ran": False, "status": _ABSENT}
    if status in {
        LifecycleStatus.FAILED.value,
        LifecycleStatus.CANCELLED.value,
        LifecycleStatus.TIMED_OUT.value,
    }:
        return {"ran": True, "status": _FAILED}
    return _unknown()


def _matching_stabilization_failure(
    target_dir: Path, session_id: str | None
) -> bool:
    """Return whether a well-formed terminal failure belongs to this run."""
    failure = _read_json_artifact(
        _deep_dir(target_dir) / "stabilization-failed.json", dict
    )
    return bool(
        failure is not None
        and session_id is not None
        and failure.get("session_id") == session_id
        and isinstance(failure.get("reason"), str)
        and failure["reason"].strip()
    )


def _fix_state(
    target_dir: Path,
    phase_events: Any,
    *,
    stabilization_failed: bool,
) -> dict[str, Any]:
    """Derive fix state from stabilization/fix failures and phase events.

    A current-session stabilization failure wins because the retained fix was
    deliberately rejected.  Otherwise non-empty group failures are partial and
    a FIX phase start is successful.
    """
    if stabilization_failed:
        return {"ran": True, "status": _FAILED}
    fix_failures = _read_json_artifact(_deep_dir(target_dir) / "fix-failures.json", dict)
    if fix_failures:
        return {"ran": True, "status": _PARTIAL}
    rows, _malformed = _event_rows(phase_events)
    for ev in rows:
        phase = _phase_value(_event_value(ev, "phase"))
        if phase == DaydreamPhase.FIX.value and _event_value(ev, "event") == "phase_start":
            return {"ran": True, "status": _SUCCEEDED}
    return {"ran": False, "status": _ABSENT}


def _test_state(
    target_dir: Path,
    session_id: str | None,
    *,
    stabilization_failed: bool,
) -> dict[str, Any]:
    """Derive test state from final stabilization and ``test-verdict.json``.

    The pre-finalization verdict is not sufficient when a matching terminal
    stabilization artifact rejects that evidence.
    """
    if stabilization_failed:
        return {"ran": True, "status": _FAILED}
    verdict = _read_json_artifact(_deep_dir(target_dir) / "test-verdict.json", dict)
    if verdict is None or session_id is None or verdict.get("session_id") != session_id:
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
    phase_events: Any,
    runs_merge: bool = True,
    runs_fix: bool = True,
    runs_test: bool = True,
    session_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return per-phase terminal states for ``merge``, ``fix``, and ``test``.

    Each entry is ``{"ran": bool, "status": str}`` with status one of
    ``succeeded``/``failed``/``partial``/``absent``/``unknown``. Pure
    derivation over on-disk deep artifacts + frozen phase events; absent or
    malformed evidence degrades to ``absent`` or ``unknown`` rather than
    raising.

    Most deep artifacts are repository-local rather than run-qualified.
    ``test-verdict.json`` and ``stabilization-failed.json`` therefore carry a
    ``session_id`` and are accepted only when they match this archive session;
    stale or unbound evidence reads as absent. Merge instead uses one valid
    current-session event pair, retaining its artifact heuristic only for an
    explicitly unbound legacy caller. ``runs_merge`` / ``runs_fix`` /
    ``runs_test`` gate reads to phases the current flow executes, so a skipped
    phase remains neutral regardless of artifacts left by prior runs.
    """
    stabilization_failed = _matching_stabilization_failure(target_dir, session_id)
    return {
        "merge": (
            _merge_state(target_dir, phase_events, session_id=session_id)
            if runs_merge
            else _absent()
        ),
        "fix": (
            _fix_state(
                target_dir,
                phase_events,
                stabilization_failed=stabilization_failed,
            )
            if runs_fix
            else _absent()
        ),
        "test": (
            _test_state(
                target_dir,
                session_id,
                stabilization_failed=stabilization_failed,
            )
            if runs_test
            else _absent()
        ),
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
    runs_merge: bool = False,
    runs_fix: bool = False,
    runs_test: bool = False,
) -> str:
    """Aggregate pipeline outcome from the archive state + per-phase states.

    Precedence:
    1. ``cancelled`` when the archive is ``partial`` with no fix failures
       (``write_partial`` signal flush — the run stopped early, nothing failed).
    2. ``failed`` when merge, fix stabilization, or test reports failed.
    3. ``partial`` when ``fix_failures`` are present.
    4. ``partial`` for an explicit partial phase, or when a phase the flow runs
       never ran (run stopped early).
    5. ``succeeded`` when every phase is succeeded or absent AND at least one
       phase actually ran (a flow that runs no merge, fix, or test surfaces no
       derivable phase signal, so an early-aborted/failed run must not be
       archived as unqualified success).
    6. else ``unknown``.
    """
    if archive_status == "partial" and not fix_failures:
        return "cancelled"
    merge_status = _phase(phase_states, "merge").get("status")
    test_status = _phase(phase_states, "test").get("status")
    fix_status = _phase(phase_states, "fix").get("status")
    if merge_status == _FAILED or fix_status == _FAILED or test_status == _FAILED:
        return _FAILED
    if fix_failures:
        return _PARTIAL
    if any(
        _phase(phase_states, name).get("status") == _PARTIAL
        for name in ("merge", "fix", "test")
    ):
        return _PARTIAL
    if (runs_merge and _phase(phase_states, "merge").get("ran") is False) or (
        runs_test and _phase(phase_states, "test").get("ran") is False
    ) or (
        runs_fix and _phase(phase_states, "fix").get("ran") is False
    ):
        return _PARTIAL
    for name in ("merge", "fix", "test"):
        status = _phase(phase_states, name).get("status")
        if status is not None and status not in (_SUCCEEDED, _ABSENT):
            return _UNKNOWN
    # Only claim success when at least one derivable phase actually ran. A flow
    # that runs none of merge/fix/test (TTT review, improve-only) surfaces no
    # phase evidence, so every phase reading ``absent`` means we cannot tell a
    # clean profile from an early aborted/failed run — report ``unknown``.
    if any(
        _phase(phase_states, name).get("status") == _SUCCEEDED
        for name in ("merge", "fix", "test")
    ):
        return _SUCCEEDED
    return _UNKNOWN
