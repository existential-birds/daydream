"""Derive phase and pipeline outcomes from frozen events and existing artifacts.

Archive finalization is independent of workflow success. Stale unbound
artifacts count as absent; malformed current evidence degrades to partial.
Bad or incomplete evidence never turns a phase green.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from daydream.remote_ci import (
    CIObservation,
    RemoteCILimits,
    RequiredContext,
    RequiredPolicy,
)
from daydream.remote_ci.evaluation import partition_required_observations
from daydream.remote_ci.evidence import (
    _is_finite_number,
    _is_positive_int,
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

if TYPE_CHECKING:
    from daydream.run_snapshot import ArchiveRunSnapshot

# Phase status values shared by phase_states entries and pipeline_status.
_SUCCEEDED = "succeeded"
_FAILED = "failed"
_PARTIAL = "partial"
_ABSENT = "absent"
_UNKNOWN = "unknown"

_REMOTE_INCOMPLETE = frozenset(
    {"pending", "missing", "timed_out", "unavailable", "superseded", "cancelled"}
)


def _read_json_artifact(path: Path, expected_type: type) -> Any | None:
    """Read a JSON artifact from *path*, returning ``None`` when absent, empty, or malformed."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, expected_type) or not data:
        return None
    return data


def _read_fix_failures(target_dir: Path) -> dict[str, str] | None:
    """Read the deep fix phase's ``{file_group: reason}`` map via `_read_json_artifact`.

    No recorded failures leaves run status unchanged.
    """
    # Keep deep imports lazy for non-deep runs.
    from daydream.deep.artifacts import DeepArtifact

    data = _read_json_artifact(DeepArtifact.FIX_FAILURES.at(target_dir / ".daydream" / "deep"), dict)
    if data is None:
        return None
    return {str(k): str(v) for k, v in data.items()}


def _read_fix_leftover_untracked(target_dir: Path) -> list[str] | None:
    """Read paths left untracked by failed fix passes via `_read_json_artifact`."""
    from daydream.deep.artifacts import DeepArtifact

    data = _read_json_artifact(DeepArtifact.FIX_LEFTOVER_UNTRACKED.at(target_dir / ".daydream" / "deep"), list)
    if data is None:
        return None
    return [str(p) for p in data]


def _read_session_bound_json_artifact(
    target_dir: Path, session_id: str | None, resolver: Callable[[Path], Path]
) -> dict[str, Any] | None:
    """Read a deep sidecar only when its ``session_id`` matches this run.

    Return ``None`` for absent, empty, malformed, unbound, or stale artifacts,
    or when this run has no session ID. Prior runs' sidecars cannot be attributed
    to the current run. ``resolver`` receives ``<target_dir>/.daydream/deep``.
    """
    if session_id is None:
        return None
    data: dict[str, Any] | None = _read_json_artifact(
        resolver(target_dir / ".daydream" / "deep"), dict
    )
    if data is None or data.get("session_id") != session_id:
        return None
    return data





def _manifest_state(
    *,
    target_dir: Path,
    run: ArchiveRunSnapshot,
    frozen_extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the status, fix, and pipeline manifest fields for one run tree.

    Derivation is gated to the phases this registered flow can execute, and
    every sidecar read is session-bound, so a non-deep or interrupted run never
    adopts prior state. The ``derive_*`` helpers never raise on absent or
    malformed artifacts, so this can never abort an archive.
    """
    from daydream.deep.artifacts import DeepArtifact
    from daydream.retry_policy import derive_retry_summary

    recorder_provenance = run.recorder_provenance
    phases = run.identity.phases
    session_id = recorder_provenance.session_id
    status = run.trajectories.status
    runs_merge = phases.merge
    runs_fix = phases.fix
    runs_test = phases.test
    runs_push = phases.push
    runs_remote_ci = phases.remote_ci
    # A deep fix run that hit per-group failures left partial/reverted edits in
    # the tree; the run is NOT "complete".
    fix_failures = _read_fix_failures(target_dir) if runs_fix else None
    # The recommended-capture sidecar's ``capture_point`` identifies which tree
    # produced ``recommended.patch``.
    recommended = (
        _read_session_bound_json_artifact(target_dir, session_id, DeepArtifact.RECOMMENDED_CAPTURE.at)
        if runs_fix
        else None
    )
    if fix_failures or frozen_extra.get("partial") is True:
        status = "partial"
    phase_states = derive_phase_states(
        target_dir,
        phase_events=frozen_extra.get("phase_events"),
        runs_merge=runs_merge,
        runs_fix=runs_fix,
        runs_test=runs_test,
        runs_push=runs_push,
        runs_remote_ci=runs_remote_ci,
        session_id=session_id,
        pr_repo=recorder_provenance.pr_repo,
        pr_number=recorder_provenance.pr_number,
    )
    return {
        "status": status,
        "fix_failures": fix_failures,
        "fix_leftover_untracked": _read_fix_leftover_untracked(target_dir) if runs_fix else None,
        # The quality-gate sidecar's ``{enabled, session_id, rounds}`` payload holds
        # per-file erosion and verbosity deltas.
        "fix_quality_gate": (
            _read_session_bound_json_artifact(target_dir, session_id, DeepArtifact.FIX_QUALITY_GATE.at)
            if runs_fix
            else None
        ),
        "recommended_capture": (recommended or {}).get("capture_point"),
        "phase_states": phase_states,
        "retry_summary": derive_retry_summary(frozen_extra.get("phase_events")),
        "pipeline_status": derive_pipeline_status(
            status,
            fix_failures,
            phase_states,
            runs_merge=runs_merge,
            runs_fix=runs_fix,
            runs_test=runs_test,
        ),
    }


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


def _identity_mapping(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _nonnegative_number(value: object) -> TypeGuard[int | float]:
    return _is_finite_number(value) and isinstance(value, (int, float)) and value >= 0


def _string_list(value: object) -> bool:
    return isinstance(value, list) and all(_bounded_text(item) is not None for item in value)


def _remote_evidence_consistent(payload: dict[str, Any]) -> bool:
    """Decode normalized evidence and recheck the producer's policy partition."""
    policy = payload["policy"]
    contexts = tuple(RequiredContext(item["context"], item.get("app_id")) for item in policy["contexts"])
    RequiredPolicy(contexts, policy["strict"])
    if len(set(contexts)) != len(contexts):
        return False

    def observations(key: str) -> tuple[CIObservation, ...]:
        rows = payload[key]
        if not isinstance(rows, list):
            raise ValueError("CI observations must be an array")
        return tuple(
            CIObservation(
                source=item["source"], context=item["context"], app_id=item.get("app_id"),
                state=item["state"], raw_state=item["raw_state"], url=item.get("url"),
                diagnostic=item.get("diagnostic"),
            )
            for item in rows
        )

    required = observations("required_observations")
    advisory = observations("advisory_observations")
    all_observations = (*required, *advisory)
    producer_keys = {
        (item.source, item.context.casefold() if item.source == "status" else item.context, item.app_id)
        for item in all_observations
    }
    if len(producer_keys) != len(all_observations):
        return False
    evidence = partition_required_observations(contexts, all_observations)
    return (
        set(evidence.required) == set(required)
        and evidence.advisory == advisory
        and list(evidence.failing) == payload["failing_contexts"]
        and list(evidence.pending) == payload["pending_contexts"]
        and list(evidence.missing) == payload["missing_contexts"]
    )


def _terminal_remote_shape(payload: dict[str, Any], status: str) -> bool:
    """Validate normalized verdict facts without trusting ``archive_state``."""
    polling = _identity_mapping(payload.get("polling"))
    policy = _identity_mapping(payload.get("policy"))
    if polling is None or policy is None:
        return False
    poll_count = polling.get("poll_count")
    stable_polls = polling.get("stable_polls")
    required_stable_polls = polling.get("required_stable_polls")
    discovery_seconds = polling.get("discovery_seconds")
    completion_seconds = polling.get("completion_seconds")
    active_count = payload.get("active_workflow_count")
    contexts = policy.get("contexts")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or _bounded_text(payload.get("reason")) is None
        or not _is_positive_int(poll_count)
        or not _is_positive_int(stable_polls)
        or stable_polls > poll_count
        or not _is_positive_int(required_stable_polls)
        or _bounded_text(polling.get("started_at")) is None
        or _bounded_text(polling.get("updated_at")) is None
        or not _nonnegative_number(polling.get("elapsed_seconds"))
        or not _nonnegative_number(polling.get("discovery_deadline"))
        or not _nonnegative_number(polling.get("completion_deadline"))
        or not isinstance(policy.get("strict"), bool)
        or not isinstance(contexts, list)
        or not isinstance(active_count, int)
        or isinstance(active_count, bool)
        or active_count < 0
        or not _string_list(payload.get("failing_contexts"))
        or not _string_list(payload.get("pending_contexts"))
        or not _string_list(payload.get("missing_contexts"))
        or not _string_list(payload.get("urls"))
        or not _string_list(payload.get("limitations"))
        or (
            payload.get("diagnostic") is not None
            and _bounded_text(payload.get("diagnostic")) is None
        )
    ):
        return False
    try:
        RemoteCILimits(
            discovery_seconds=cast(float, discovery_seconds), completion_seconds=cast(float, completion_seconds),
            stable_polls=required_stable_polls,
        )
        if not _remote_evidence_consistent(payload):
            return False
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
    if status in {"passed", "no_ci"} and stable_polls < required_stable_polls:
        return False
    if status == "no_ci":
        return (
            contexts == []
            and polling["elapsed_seconds"] >= discovery_seconds
            and active_count == 0
            and payload["required_observations"] == []
            and payload["advisory_observations"] == []
            and payload["failing_contexts"] == []
            and payload["pending_contexts"] == []
            and payload["missing_contexts"] == []
            and payload["urls"] == []
            and payload["diagnostic"] is None
        )
    required = payload["required_observations"]
    advisory = payload["advisory_observations"]
    if status == "passed":
        return (
            payload["failing_contexts"] == []
            and payload["pending_contexts"] == []
            and payload["missing_contexts"] == []
            and all(item["state"] == "pass" for item in required)
            and (
                bool(contexts)
                or (bool(advisory) and all(item["state"] != "pending" for item in advisory))
            )
        )
    return bool(payload["failing_contexts"]) and any(
        item["state"] == "fail" for item in required
    )


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


def _remote_identity_matches(
    payload: dict[str, Any],
    push: dict[str, Any],
    *,
    pr_repo: str | None,
    pr_number: int | None,
) -> bool:
    target = _identity_mapping(payload.get("target"))
    binding = _identity_mapping(payload.get("binding"))
    if target is None or binding is None:
        return False

    pushed_repository = _repository(push.get("pushed_repository"))
    if pushed_repository is None:
        return False
    for identity in (target, binding):
        if (
            not _is_positive_int(identity.get("pr_number"))
            or any(_repository(identity.get(key)) is None for key in ("base_repository", "head_repository"))
            or any(_bounded_text(identity.get(key)) is None for key in ("pr_url", "base_ref", "head_ref"))
        ):
            return False
    if any(_sha(value) is None for value in (
        target.get("pushed_sha"), binding.get("head_sha"), payload.get("head_sha"), payload.get("evidence_sha"),
    )):
        return False
    merge_sha = binding.get("merge_sha")
    payload_merge = payload.get("merge_sha")
    if any(value is not None and _sha(value) is None for value in (merge_sha, payload_merge)):
        return False
    identities_match = (
        binding.get("state") == "open"
        and target["base_repository"] == binding["base_repository"]
        and target["head_repository"] == binding["head_repository"] == pushed_repository
        and target["base_ref"] == binding["base_ref"]
        and target["head_ref"] == binding["head_ref"] == push["branch"]
        and target["pr_number"] == binding["pr_number"]
        and target["pr_url"] == binding["pr_url"]
        and target.get("remote") == push["remote"]
        and target["pushed_sha"] == binding["head_sha"] == payload["head_sha"] == push["pushed_sha"]
        and payload_merge == merge_sha
        and payload["evidence_sha"] in {target["pushed_sha"], merge_sha}
    )
    if not identities_match:
        return False
    if pr_repo is not None and _configured_repository(pr_repo) != target["base_repository"]:
        return False
    return pr_number is None or pr_number == target["pr_number"]


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
    if (
        status not in {"passed", "no_ci", "failed"}
        or not _terminal_remote_shape(payload, status)
        or not _remote_identity_matches(
            payload, push, pr_repo=pr_repo, pr_number=pr_number
        )
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
    """Return terminal states for merge, fix, test, push, and remote CI.

    Each entry is ``{"ran": bool, "status": str}`` with status one of
    ``succeeded``/``failed``/``partial``/``absent``/``unknown``. Pure
    derivation over on-disk deep artifacts + frozen phase events; absent or
    malformed evidence degrades to ``absent`` or ``unknown`` rather than
    raising.

    Most deep artifacts are repository-local rather than run-qualified.
    ``test-verdict.json`` and ``stabilization-failed.json`` therefore carry a
    ``session_id`` and are accepted only when they match this archive session;
    stale or unbound evidence reads as absent. The two remote artifacts carry
    the same session plus exact pushed/PR identities. All five ``runs_*`` flags
    gate reads to phases the current flow executes, so a skipped phase remains
    neutral regardless of artifacts left by prior runs.
    Merge requires one valid current-session event pair; absent run identity
    cannot inherit a result from on-disk artifacts.
    """
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
    """Aggregate pipeline outcome from the archive state + per-phase states.

    Precedence:
    1. ``cancelled`` when the archive is ``partial`` with no fix failures
       (``write_partial`` signal flush — the run stopped early, nothing failed).
    2. ``failed`` when any local, push, or remote phase reports failed.
    3. ``partial`` when ``fix_failures`` are present.
    4. ``partial`` when a required local phase never ran or an attempted push/
       remote phase is incomplete.
    5. ``succeeded`` when every phase is succeeded or absent AND at least one
       phase actually ran (a flow that runs none of these phases surfaces no
       derivable phase signal, so an early-aborted/failed run must not be
       archived as unqualified success).
    6. else ``unknown``.
    """
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
