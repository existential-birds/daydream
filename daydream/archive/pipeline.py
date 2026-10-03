"""Derive phase and pipeline outcomes from frozen events and existing artifacts.

Archive finalization is independent of workflow success. Stale unbound
artifacts count as absent; malformed current evidence degrades to partial.
Bad or incomplete evidence never turns a phase green.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from daydream.archive import _read_json_artifact
from daydream.remote_ci.artifacts import read_terminal_remote_ci_verdict
from daydream.remote_ci.evidence import (
    _normalize_repository as _normalize_remote_repository,
    _require_sha as _require_remote_sha,
    _required_text,
)
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

_REMOTE_INCOMPLETE = frozenset(
    {"pending", "missing", "timed_out", "unavailable", "superseded", "cancelled"}
)


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


def _merge_state(
    phase_events: Any,
    *,
    session_id: str | None,
) -> dict[str, Any]:
    """Derive merge truth solely from a valid current-session lifecycle pair."""
    rows, malformed_container = _event_rows(phase_events)
    if not session_id or malformed_container:
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
    """Derive fix state from stabilization/fix failures and phase events."""
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
    """Derive test state from final stabilization and ``test-verdict.json``."""
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


def _phase_started(phase_events: Any, phase: DaydreamPhase) -> bool:
    """Return whether the current recorder observed this phase start."""
    rows, _malformed = _event_rows(phase_events)
    return any(
        _phase_value(_event_value(event, "phase")) == phase.value
        and _event_value(event, "event") == "phase_start"
        for event in rows
    )


def _bounded_text(value: object) -> str | None:
    try:
        return _required_text(value, "artifact text")
    except ValueError:
        return None


def _repository(value: object) -> str | None:
    try:
        normalized = _normalize_remote_repository(value, "repository")
    except ValueError:
        return None
    return normalized if value == normalized else None


def _configured_repository(value: object) -> str | None:
    """Validate and case-fold one operator-supplied repository slug."""
    try:
        return _normalize_remote_repository(value, "configured repository")
    except ValueError:
        return None


def _sha(value: object) -> str | None:
    try:
        return _require_remote_sha(value, "commit SHA")
    except ValueError:
        return None


def _push_payload(
    target_dir: Path, session_id: str | None
) -> tuple[dict[str, Any] | None, bool]:
    path = _deep_dir(target_dir) / "push-verdict.json"
    if _bounded_text(session_id) is None:
        return None, False
    payload = _read_json_artifact(path, dict)
    if payload is None:
        return None, path.is_file()
    if payload.get("session_id") != session_id:
        return None, False
    required_text = ("remote", "branch", "started_at", "updated_at")
    repository = payload.get("pushed_repository")
    valid_repository = repository is None or _repository(repository) is not None
    valid = (
        type(payload.get("schema_version")) is int
        and payload.get("schema_version") == 1
        and isinstance(payload.get("status"), str)
        and payload.get("status") in {"succeeded", "failed"}
        and all(_bounded_text(payload.get(key)) is not None for key in required_text)
        and _sha(payload.get("pushed_sha")) is not None
        and valid_repository
        and (
            payload.get("status") != "failed"
            or _bounded_text(payload.get("diagnostic")) is not None
        )
    )
    return (payload if valid else None), True


def _push_state(
    target_dir: Path,
    phase_events: list[Any],
    session_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return current-session push state and its validated receipt."""
    payload, current_or_malformed = _push_payload(target_dir, session_id)
    started = _phase_started(phase_events, DaydreamPhase.PUSH)
    if payload is None:
        if started or current_or_malformed:
            return {"ran": True, "status": _PARTIAL}, None
        return _absent(), None
    if payload["status"] == "failed":
        return {"ran": True, "status": _FAILED}, None
    return {"ran": True, "status": _SUCCEEDED}, payload


def _remote_details(payload: dict[str, Any]) -> dict[str, Any]:
    observations = payload.get("advisory_observations")
    failures: list[str] = []
    if isinstance(observations, list):
        for item in observations:
            if not isinstance(item, dict) or item.get("state") != "fail":
                continue
            context = _bounded_text(item.get("context"))
            if context is not None:
                failures.append(context)
    return {"advisory_failures": failures}


def _remote_ci_state(
    target_dir: Path,
    phase_events: list[Any],
    session_id: str | None,
    push: dict[str, Any] | None,
    *,
    pr_repo: str | None,
    pr_number: int | None,
) -> dict[str, Any]:
    """Derive remote state only from a matching current push and CI artifact."""
    started = _phase_started(phase_events, DaydreamPhase.REMOTE_CI)
    if push is None:
        return {"ran": True, "status": _PARTIAL} if started else _absent()

    path = _deep_dir(target_dir) / "remote-ci-verdict.json"
    payload = _read_json_artifact(path, dict)
    if payload is None or payload.get("session_id") != session_id:
        return {"ran": True, "status": _PARTIAL}
    status = payload.get("status")
    if not isinstance(status, str):
        return {"ran": True, "status": _PARTIAL}
    if status in _REMOTE_INCOMPLETE:
        return {
            "ran": True,
            "status": _PARTIAL,
            "details": _remote_details(payload),
        }
    try:
        verdict = read_terminal_remote_ci_verdict(payload, target_dir=target_dir)
    except (ValueError, TypeError, KeyError, AttributeError):
        return {"ran": True, "status": _PARTIAL}
    target = verdict.target
    if (
        target is None or target.head_repository != push.get("pushed_repository")
        or target.head_ref != push["branch"] or target.remote != push["remote"]
        or target.pushed_sha != push["pushed_sha"]
        or (pr_repo is not None and _configured_repository(pr_repo) != target.base_repository)
        or (pr_number is not None and pr_number != target.pr_number)
    ):
        return {"ran": True, "status": _PARTIAL}
    return {
        "ran": True,
        "status": _FAILED if status == "failed" else _SUCCEEDED,
        "details": _remote_details(payload),
    }


def derive_phase_states(
    target_dir: Path,
    *,
    phase_events: Any,
    runs_merge: bool = True,
    runs_fix: bool = True,
    runs_test: bool = True,
    runs_push: bool = False,
    runs_remote_ci: bool = False,
    session_id: str | None = None,
    pr_repo: str | None = None,
    pr_number: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Return terminal states for merge, fix, test, push, and remote CI."""
    stabilization_failed = _matching_stabilization_failure(target_dir, session_id)
    push_state, push = (
        _push_state(target_dir, phase_events, session_id)
        if runs_push
        else (_absent(), None)
    )
    return {
        "merge": (
            _merge_state(phase_events, session_id=session_id)
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
        "push": push_state,
        "remote_ci": (
            _remote_ci_state(
                target_dir,
                phase_events,
                session_id,
                push,
                pr_repo=pr_repo,
                pr_number=pr_number,
            )
            if runs_remote_ci
            else _absent()
        ),
    }


def _phase(phase_states: dict[str, Any], name: str) -> dict[str, Any]:
    """Read a phase state, treating missing or malformed entries as empty."""
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
    """Aggregate pipeline outcome from the archive state + per-phase states."""
    if archive_status == "partial" and not fix_failures:
        return "cancelled"
    phase_names = ("merge", "fix", "test", "push", "remote_ci")
    if any(_phase(phase_states, name).get("status") == _FAILED for name in phase_names):
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
    if any(
        _phase(phase_states, name).get("ran") is True
        and _phase(phase_states, name).get("status") == _PARTIAL
        for name in ("push", "remote_ci")
    ):
        return _PARTIAL
    for name in phase_names:
        status = _phase(phase_states, name).get("status")
        if status is not None and status not in (_SUCCEEDED, _ABSENT):
            return _UNKNOWN
    # Only claim success when at least one derivable phase actually ran. A flow
    # with no derivable phase evidence cannot distinguish a clean profile from
    # an early aborted/failed run, so all-absent remains ``unknown``.
    if any(
        _phase(phase_states, name).get("status") == _SUCCEEDED
        for name in phase_names
    ):
        return _SUCCEEDED
    return _UNKNOWN
