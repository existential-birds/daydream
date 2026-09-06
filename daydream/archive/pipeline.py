"""Per-phase pipeline state derivation for archived runs.

Derives terminal states (merge/fix/test/push/remote CI) and ``pipeline_status``
aggregate from existing deep artifacts + phase events — no new runtime
instrumentation. Separates pipeline outcome from archive finalization so a run
that merged-failed and never tested is never archived as unqualified success.

Artifact reads are best-effort: stale unbound artifacts are absent, while
malformed evidence for an observed/current phase is partial. These functions
never raise on bad input and never turn incomplete evidence green.

Exports:
    derive_phase_states: Per-phase terminal states from artifacts + events.
    derive_pipeline_status: Aggregate pipeline outcome from archive state +
        per-phase states.
"""

import re
from math import isfinite
from pathlib import Path
from typing import Any, TypeGuard

from daydream.archive import _read_json_artifact
from daydream.remote_ci import (
    CIObservation,
    RequiredContext,
    required_context_label,
    required_context_matches,
)
from daydream.trajectory import DaydreamPhase

# Phase status values shared by phase_states entries and pipeline_status.
_SUCCEEDED = "succeeded"
_FAILED = "failed"
_PARTIAL = "partial"
_ABSENT = "absent"
_UNKNOWN = "unknown"

_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_RE = re.compile(
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z"
)
_REMOTE_INCOMPLETE = frozenset(
    {"pending", "missing", "timed_out", "unavailable", "superseded", "cancelled"}
)


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
    phase_events: list[Any],
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
    for ev in phase_events or []:
        phase = getattr(ev, "phase", None)
        if phase is DaydreamPhase.FIX and getattr(ev, "event", None) == "phase_start":
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


def _phase_started(phase_events: list[Any], phase: DaydreamPhase) -> bool:
    """Return whether the current recorder observed this phase start."""
    return any(
        getattr(event, "phase", None) is phase
        and getattr(event, "event", None) == "phase_start"
        for event in phase_events or []
    )


def _bounded_text(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2_000
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        return None
    return value


def _repository(value: object) -> str | None:
    text = _bounded_text(value)
    if text is None or _REPOSITORY_RE.fullmatch(text) is None:
        return None
    if any(piece in {".", ".."} for piece in text.split("/")):
        return None
    normalized = text.lower()
    return normalized if text == normalized else None


def _sha(value: object) -> str | None:
    if isinstance(value, str) and _SHA_RE.fullmatch(value) is not None:
        return value
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
        payload.get("schema_version") == 1
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
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    )


def _string_list(value: object) -> bool:
    return isinstance(value, list) and all(_bounded_text(item) is not None for item in value)


def _observation_list(value: object) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        source = item.get("source")
        app_id = item.get("app_id")
        if (
            not isinstance(source, str)
            or source not in {"check_run", "status"}
            or _bounded_text(item.get("context")) is None
            or not isinstance(item.get("state"), str)
            or item.get("state") not in {"pass", "pending", "fail"}
            or _bounded_text(item.get("raw_state")) is None
            or (item.get("url") is not None and _bounded_text(item.get("url")) is None)
            or (
                item.get("diagnostic") is not None
                and _bounded_text(item.get("diagnostic")) is None
            )
        ):
            return False
        if source == "check_run":
            if not isinstance(app_id, int) or isinstance(app_id, bool) or app_id <= 0:
                return False
        elif app_id is not None:
            return False
    return True


def _remote_evidence_consistent(payload: dict[str, Any]) -> bool:
    """Recheck the serialized policy partition with the producer's match rules."""
    contexts = tuple(
        RequiredContext(item["context"], item.get("app_id"))
        for item in payload["policy"]["contexts"]
    )
    if len(set(contexts)) != len(contexts):
        return False

    def observations(key: str) -> tuple[CIObservation, ...]:
        return tuple(
            CIObservation(
                source=item["source"],
                context=item["context"],
                app_id=item.get("app_id"),
                state=item["state"],
                raw_state=item["raw_state"],
                url=item.get("url"),
                diagnostic=item.get("diagnostic"),
            )
            for item in payload[key]
        )

    required = observations("required_observations")
    advisory = observations("advisory_observations")
    all_observations = (*required, *advisory)
    producer_keys = {
        (
            item.source,
            item.context.casefold() if item.source == "status" else item.context,
            item.app_id,
        )
        for item in all_observations
    }
    if len(producer_keys) != len(all_observations):
        return False
    if any(
        not any(required_context_matches(context, item) for context in contexts)
        for item in required
    ) or any(
        any(required_context_matches(context, item) for context in contexts)
        for item in advisory
    ):
        return False

    failing: list[str] = []
    pending: list[str] = []
    missing: list[str] = []
    for context in contexts:
        matches = [item for item in required if required_context_matches(context, item)]
        label = required_context_label(context)
        if not matches:
            missing.append(label)
        elif any(item.state == "fail" for item in matches):
            failing.append(label)
        elif any(item.state == "pending" for item in matches):
            pending.append(label)
    return bool(
        payload["failing_contexts"] == failing
        and payload["pending_contexts"] == pending
        and payload["missing_contexts"] == missing
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
        payload.get("schema_version") != 1
        or _bounded_text(payload.get("reason")) is None
        or not isinstance(poll_count, int)
        or isinstance(poll_count, bool)
        or poll_count <= 0
        or not isinstance(stable_polls, int)
        or isinstance(stable_polls, bool)
        or stable_polls <= 0
        or stable_polls > poll_count
        or not isinstance(required_stable_polls, int)
        or isinstance(required_stable_polls, bool)
        or required_stable_polls <= 0
        or _bounded_text(polling.get("started_at")) is None
        or _bounded_text(polling.get("updated_at")) is None
        or not _nonnegative_number(polling.get("elapsed_seconds"))
        or not _nonnegative_number(polling.get("discovery_deadline"))
        or not _nonnegative_number(polling.get("completion_deadline"))
        or not _nonnegative_number(discovery_seconds)
        or discovery_seconds <= 0
        or not _nonnegative_number(completion_seconds)
        or completion_seconds < discovery_seconds
        or not isinstance(policy.get("strict"), bool)
        or not isinstance(contexts, list)
        or not isinstance(active_count, int)
        or isinstance(active_count, bool)
        or active_count < 0
        or not _observation_list(payload.get("required_observations"))
        or not _observation_list(payload.get("advisory_observations"))
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
    for item in contexts:
        if not isinstance(item, dict) or _bounded_text(item.get("context")) is None:
            return False
        app_id = item.get("app_id")
        if app_id is not None and (
            not isinstance(app_id, int) or isinstance(app_id, bool) or app_id <= 0
        ):
            return False
    try:
        if not _remote_evidence_consistent(payload):
            return False
    except ValueError:
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
    target_base = _repository(target.get("base_repository"))
    target_head = _repository(target.get("head_repository"))
    binding_base = _repository(binding.get("base_repository"))
    binding_head = _repository(binding.get("head_repository"))
    target_number = target.get("pr_number")
    binding_number = binding.get("pr_number")
    target_url = _bounded_text(target.get("pr_url"))
    binding_url = _bounded_text(binding.get("pr_url"))
    target_base_ref = _bounded_text(target.get("base_ref"))
    binding_base_ref = _bounded_text(binding.get("base_ref"))
    target_head_ref = _bounded_text(target.get("head_ref"))
    binding_head_ref = _bounded_text(binding.get("head_ref"))
    target_sha = _sha(target.get("pushed_sha"))
    binding_head_sha = _sha(binding.get("head_sha"))
    binding_merge = binding.get("merge_sha")
    merge_sha = None if binding_merge is None else _sha(binding_merge)
    payload_head = _sha(payload.get("head_sha"))
    payload_merge = payload.get("merge_sha")
    normalized_payload_merge = None if payload_merge is None else _sha(payload_merge)
    evidence_sha = _sha(payload.get("evidence_sha"))

    if (
        pushed_repository is None
        or target_base is None
        or target_head is None
        or binding_base is None
        or binding_head is None
        or not isinstance(target_number, int)
        or isinstance(target_number, bool)
        or target_number <= 0
        or not isinstance(binding_number, int)
        or isinstance(binding_number, bool)
        or binding_number <= 0
        or target_url is None
        or binding_url is None
        or target_base_ref is None
        or binding_base_ref is None
        or target_head_ref is None
        or binding_head_ref is None
        or target_sha is None
        or binding_head_sha is None
        or (binding_merge is not None and merge_sha is None)
        or payload_head is None
        or (payload_merge is not None and normalized_payload_merge is None)
        or evidence_sha is None
    ):
        return False

    valid_evidence = {target_sha}
    if merge_sha is not None:
        valid_evidence.add(merge_sha)
    identities_match = (
        binding.get("state") == "open"
        and target_base == binding_base
        and target_head == binding_head == pushed_repository
        and target_base_ref == binding_base_ref
        and target_head_ref == binding_head_ref == push["branch"]
        and target_number == binding_number
        and target_url == binding_url
        and target.get("remote") == push["remote"]
        and target_sha == binding_head_sha == payload_head == push["pushed_sha"]
        and normalized_payload_merge == merge_sha
        and evidence_sha in valid_evidence
    )
    if not identities_match:
        return False
    if pr_repo is not None and _repository(pr_repo) != target_base:
        return False
    return pr_number is None or pr_number == target_number


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
    phase_events: list[Any],
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
    derivation over on-disk deep artifacts + recorder phase events; never
    raises on absent/malformed artifacts. Unbound stale artifacts are absent;
    malformed evidence for an observed/current phase is partial.

    Most deep artifacts are repository-local rather than run-qualified.
    ``test-verdict.json`` and ``stabilization-failed.json`` therefore carry a
    ``session_id`` and are accepted only when they match this archive session;
    stale or unbound evidence reads as absent. The two remote artifacts carry
    the same session plus exact pushed/PR identities. All five ``runs_*`` flags
    gate reads to phases the current flow executes, so a skipped phase remains
    neutral regardless of artifacts left by prior runs.
    """
    stabilization_failed = _matching_stabilization_failure(target_dir, session_id)
    push_state, push = (
        _push_state(target_dir, phase_events, session_id)
        if runs_push
        else (_absent(), None)
    )
    return {
        "merge": _merge_state(target_dir) if runs_merge else _absent(),
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
    runs_push: bool = False,
    runs_remote_ci: bool = False,
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
    if (runs_test and _phase(phase_states, "test").get("ran") is False) or (
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
