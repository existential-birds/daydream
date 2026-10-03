"""Serialize bounded CI verdicts and operator handoffs without raw REST data."""

from __future__ import annotations

import platform
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, cast

from daydream.json_utils import atomic_write_json
from daydream.remote_ci.evaluation import (
    _fixed_identity_matches,
    observed_ci_outcome,
    partition_required_observations,
)
from daydream.remote_ci.evidence import (
    DEFAULT_LIMITS,
    CIObservation,
    PRCIBinding,
    RemoteCILimits,
    RemoteCITarget,
    RemoteCIVerdict,
    RequiredContext,
    RequiredPolicy,
    _is_finite_number,
    _is_positive_int,
    _mapping,
    _require_sha,
    _required_text,
    _sequence,
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


def read_terminal_remote_ci_verdict(
    payload: dict[str, Any], *, target_dir: Path
) -> RemoteCIVerdict:
    """Decode a v1 receipt and prove its terminal outcome from its bounded evidence.

    Receipt identity is already canonical: normalization must never repair tampered
    persisted identities. Extra fields remain forward-compatible, but partition,
    identity, polling limits and the producer's outcome policy must all agree.
    """
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported CI receipt schema")
    status = payload.get("status")
    if status not in {"passed", "no_ci", "failed"}:
        raise ValueError("CI receipt is not terminal")
    reason = _required_text(payload.get("reason"), "reason")
    target_row = _mapping(payload.get("target"), "target")
    binding_row = _mapping(payload.get("binding"), "binding")
    target = RemoteCITarget(target_dir=target_dir.absolute(), **cast(dict[str, Any], {
        item.name: target_row[item.name]
        for item in fields(RemoteCITarget) if item.name != "target_dir"
    }))
    binding = PRCIBinding(**cast(dict[str, Any], {
        item.name: binding_row.get(item.name) for item in fields(PRCIBinding)
    }))
    if any(target_row.get(key) != value for key, value in (_target_payload(target) or {}).items()):
        raise ValueError("CI target identity is not canonical")
    if any(binding_row.get(key) != value for key, value in asdict(binding).items()):
        raise ValueError("CI binding identity is not canonical")
    evidence_sha = _require_sha(payload.get("evidence_sha"), "evidence SHA")
    if (
        binding.state != "open" or not _fixed_identity_matches(target, binding)
        or target.pushed_sha != binding.head_sha
        or payload.get("head_sha") != binding.head_sha
        or payload.get("merge_sha") != binding.merge_sha
        or evidence_sha not in {target.pushed_sha, binding.merge_sha}
    ):
        raise ValueError("CI receipt identities disagree")
    polling = _mapping(payload.get("polling"), "polling")
    poll_count = polling.get("poll_count")
    stable_polls = polling.get("stable_polls")
    if not _is_positive_int(poll_count) or not _is_positive_int(stable_polls) or stable_polls > poll_count:
        raise ValueError("invalid CI polling counts")
    for key in ("started_at", "updated_at"):
        _required_text(polling.get(key), key)
    for key in ("elapsed_seconds", "discovery_deadline", "completion_deadline"):
        value = polling.get(key)
        if not _is_finite_number(value) or cast(float, value) < 0:
            raise ValueError(f"invalid CI {key}")
    limits = RemoteCILimits(
        discovery_seconds=cast(float, polling.get("discovery_seconds")),
        completion_seconds=cast(float, polling.get("completion_seconds")),
        stable_polls=cast(int, polling.get("required_stable_polls")),
    )
    policy_row = _mapping(payload.get("policy"), "policy")
    contexts = tuple(
        RequiredContext(cast(str, _mapping(item, "context").get("context")),
                        cast(int | None, _mapping(item, "context").get("app_id")))
        for item in _sequence(policy_row.get("contexts"), "contexts")
    )
    policy = RequiredPolicy(contexts, cast(bool, policy_row.get("strict")))
    if len(set(contexts)) != len(contexts):
        raise ValueError("duplicate CI required context")

    def observations(key: str) -> tuple[CIObservation, ...]:
        return tuple(CIObservation(**cast(dict[str, Any], {
            field.name: _mapping(item, "observation").get(field.name)
            for field in fields(CIObservation)
        })) for item in _sequence(payload.get(key), key))

    required = observations("required_observations")
    advisory = observations("advisory_observations")
    all_observations = (*required, *advisory)
    producer_keys = {
        (item.source, item.context.casefold() if item.source == "status" else item.context, item.app_id)
        for item in all_observations
    }
    if len(producer_keys) != len(all_observations):
        raise ValueError("duplicate CI producer")
    evidence = partition_required_observations(contexts, all_observations)

    def texts(key: str) -> tuple[str, ...]:
        return tuple(_required_text(item, key) for item in _sequence(payload.get(key), key))

    failing, pending, missing = (texts(key) for key in ("failing_contexts", "pending_contexts", "missing_contexts"))
    if (
        set(evidence.required) != set(required) or evidence.advisory != advisory
        or (evidence.failing, evidence.pending, evidence.missing) != (failing, pending, missing)
    ):
        raise ValueError("CI evidence does not match required policy")
    active_count = payload.get("active_workflow_count")
    if not isinstance(active_count, int) or isinstance(active_count, bool) or active_count < 0:
        raise ValueError("invalid CI active workflow count")
    diagnostic = payload.get("diagnostic")
    if diagnostic is not None:
        diagnostic = _required_text(diagnostic, "diagnostic")
    urls, limitations = texts("urls"), texts("limitations")
    elapsed = cast(float, polling["elapsed_seconds"])
    expected, _ = observed_ci_outcome(
        policy, evidence, active_workflow_count=active_count, elapsed=elapsed,
        stable_polls=stable_polls, limits=limits,
    )
    if status != expected or (status == "no_ci" and (urls or diagnostic is not None)):
        raise ValueError("CI terminal outcome is not supported by its evidence")
    return RemoteCIVerdict(
        status=expected, reason=reason, target=target, binding=binding, policy=policy,
        active_workflow_count=active_count, evidence_sha=evidence_sha,
        required_observations=required, advisory_observations=advisory,
        failing_contexts=failing, pending_contexts=pending, missing_contexts=missing,
        urls=urls, diagnostic=diagnostic, stable_polls=stable_polls,
        elapsed_seconds=elapsed, limitations=limitations,
    )


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
