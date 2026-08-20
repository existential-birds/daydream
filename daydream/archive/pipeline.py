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

import json
from pathlib import Path
from typing import Any

from daydream.trajectory import DaydreamPhase

# Phase status values shared by phase_states entries and pipeline_status.
_SUCCEEDED = "succeeded"
_FAILED = "failed"
_PARTIAL = "partial"
_ABSENT = "absent"
_UNKNOWN = "unknown"


def _read_json_artifact(path: Path, expected_type: type) -> Any | None:
    """Read a JSON artifact, returning ``None`` when absent, empty, or malformed.

    Mirrors ``daydream/archive/__init__.py:_read_json_artifact``: any read
    failure degrades to ``None`` ("we cannot know"), never a raise.
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(data, expected_type) or not data:
        return None
    return data


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


def _fix_state(target_dir: Path, phase_events: list) -> dict[str, Any]:
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


def derive_phase_states(target_dir: Path, *, phase_events: list) -> dict[str, dict]:
    """Return per-phase terminal states for ``merge``, ``fix``, and ``test``.

    Each entry is ``{"ran": bool, "status": str}`` with status one of
    ``succeeded``/``failed``/``partial``/``absent``/``unknown``. Pure
    derivation over on-disk deep artifacts + recorder phase events; never
    raises on absent/malformed artifacts (they read as ``absent``).
    """
    return {
        "merge": _merge_state(target_dir),
        "fix": _fix_state(target_dir, phase_events),
        "test": _test_state(target_dir),
    }


def derive_pipeline_status(
    archive_status: str,
    fix_failures: dict | None,
    phase_states: dict,
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
    5. ``succeeded`` when every phase is succeeded or absent.
    6. else ``unknown``.
    """
    if archive_status == "partial" and not fix_failures:
        return "cancelled"
    merge_status = (phase_states.get("merge") or {}).get("status")
    test_status = (phase_states.get("test") or {}).get("status")
    if merge_status == _FAILED or test_status == _FAILED:
        return _FAILED
    if fix_failures:
        return _PARTIAL
    if (runs_test and (phase_states.get("test") or {}).get("ran") is False) or (
        runs_fix and (phase_states.get("fix") or {}).get("ran") is False
    ):
        return _PARTIAL
    for name in ("merge", "fix", "test"):
        status = (phase_states.get(name) or {}).get("status")
        if status is not None and status not in (_SUCCEEDED, _ABSENT):
            return _UNKNOWN
    return _SUCCEEDED
