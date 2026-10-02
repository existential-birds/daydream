"""Serialize bounded CI verdicts and operator handoffs without raw REST data."""

from __future__ import annotations

import platform
from dataclasses import asdict
from pathlib import Path

from daydream.json_utils import atomic_write_json
from daydream.remote_ci.evidence import (
    DEFAULT_LIMITS,
    PRCIBinding,
    RemoteCILimits,
    RemoteCITarget,
    RemoteCIVerdict,
    _is_finite_number,
    _is_positive_int,
    _required_text,
)


def _target_payload(target: RemoteCITarget | None) -> dict[str, object] | None:
    if target is None:
        return None
    payload = asdict(target)
    del payload["target_dir"]
    return payload


def _binding_payload(binding: PRCIBinding | None) -> dict[str, object] | None:
    return None if binding is None else asdict(binding)


def remote_ci_verdict_payload(
    verdict: RemoteCIVerdict,
    *,
    session_id: str,
    poll_count: int,
    started_at: str,
    updated_at: str,
    discovery_deadline: float,
    completion_deadline: float,
    limits: RemoteCILimits = DEFAULT_LIMITS,
) -> dict[str, object]:
    """Return the deterministic v1 artifact without raw GitHub data."""
    _required_text(session_id, "session id")
    _required_text(started_at, "start timestamp")
    _required_text(updated_at, "update timestamp")
    unobserved = (
        verdict.status in {"pending", "unavailable"}
        and verdict.binding is None
        and verdict.policy is None
        and verdict.active_workflow_count is None
        and verdict.evidence_sha is None
        and not verdict.required_observations
        and not verdict.advisory_observations
        and verdict.stable_polls == 0
        and isinstance(poll_count, int)
        and not isinstance(poll_count, bool)
        and poll_count == 0
    )
    if not unobserved and not _is_positive_int(poll_count):
        raise ValueError("poll count must be positive, or zero before CI observation")
    for value in (discovery_deadline, completion_deadline):
        if (
            not _is_finite_number(value) or value < 0
        ):
            raise ValueError("deadlines must be nonnegative numbers")
    return {
        "schema_version": 1,
        "session_id": session_id,
        "status": verdict.status,
        "reason": verdict.reason,
        "target": _target_payload(verdict.target),
        "binding": _binding_payload(verdict.binding),
        "head_sha": None if verdict.binding is None else verdict.binding.head_sha,
        "merge_sha": None if verdict.binding is None else verdict.binding.merge_sha,
        "evidence_sha": verdict.evidence_sha,
        "polling": {
            "poll_count": poll_count,
            "stable_polls": verdict.stable_polls,
            "started_at": started_at,
            "updated_at": updated_at,
            "elapsed_seconds": verdict.elapsed_seconds,
            "discovery_deadline": discovery_deadline,
            "completion_deadline": completion_deadline,
            "discovery_seconds": limits.discovery_seconds,
            "completion_seconds": limits.completion_seconds,
            "required_stable_polls": limits.stable_polls,
        },
        "policy": (
            None
            if verdict.policy is None
            else {
                "strict": verdict.policy.strict,
                "contexts": [
                    asdict(item)
                    for item in verdict.policy.contexts
                ],
            }
        ),
        "active_workflow_count": verdict.active_workflow_count,
        "required_observations": [
            asdict(item) for item in verdict.required_observations
        ],
        "advisory_observations": [
            asdict(item) for item in verdict.advisory_observations
        ],
        "failing_contexts": list(verdict.failing_contexts),
        "pending_contexts": list(verdict.pending_contexts),
        "missing_contexts": list(verdict.missing_contexts),
        "urls": list(verdict.urls),
        "diagnostic": verdict.diagnostic,
        "limitations": list(verdict.limitations),
        "archive_state": verdict.archive_state,
    }


def write_remote_ci_verdict(
    path: Path,
    verdict: RemoteCIVerdict,
    *,
    session_id: str,
    poll_count: int,
    started_at: str,
    updated_at: str,
    discovery_deadline: float,
    completion_deadline: float,
    limits: RemoteCILimits = DEFAULT_LIMITS,
) -> None:
    """Atomically replace the session- and SHA-bound remote CI verdict."""
    atomic_write_json(
        path,
        remote_ci_verdict_payload(
            verdict,
            session_id=session_id,
            poll_count=poll_count,
            started_at=started_at,
            updated_at=updated_at,
            discovery_deadline=discovery_deadline,
            completion_deadline=completion_deadline,
            limits=limits,
        ),
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )


def remote_ci_handoff_payload(
    verdict: RemoteCIVerdict, *, session_id: str
) -> dict[str, object]:
    """Build the bounded operator handoff for a non-success outcome."""
    _required_text(session_id, "session id")
    if verdict.status in {"passed", "no_ci"}:
        raise ValueError("successful remote CI does not have a handoff")
    return {
        "schema_version": 1,
        "session_id": session_id,
        "status": verdict.status,
        "outcome": "failed" if verdict.status == "failed" else "incomplete",
        "summary": verdict.reason,
        "target": _target_payload(verdict.target),
        "binding": _binding_payload(verdict.binding),
        "evidence_sha": verdict.evidence_sha,
        "failing_contexts": list(verdict.failing_contexts),
        "missing_contexts": list(verdict.missing_contexts),
        "pending_contexts": list(verdict.pending_contexts),
        "urls": list(verdict.urls),
        "next_action": (
            "Inspect CI; make a new local-verified ordinary push; rerun Daydream."
        ),
    }


def write_remote_ci_handoff(
    path: Path, verdict: RemoteCIVerdict, *, session_id: str
) -> None:
    """Atomically write one failed/incomplete remote-CI handoff."""
    atomic_write_json(
        path,
        remote_ci_handoff_payload(verdict, session_id=session_id),
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )


def local_host_facts() -> dict[str, str]:
    """Report only the native host identity; never infer coverage elsewhere."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
