"""Strict, session-bindable evaluation of GitHub's remote CI evidence.

This module deliberately separates untrusted REST parsing from orchestration.
Only normalized observations cross the boundary; raw GitHub response objects
are never written to Daydream artifacts.
"""

from __future__ import annotations

import platform
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import anyio

from daydream.git_ops import (
    DeadlineExpired,
    GitError,
    GitHubPageLimits,
    GitHubRequestBudget,
    gh_actions_workflows,
    gh_active_branch_rules,
    gh_classic_required_checks,
    gh_commit_check_runs,
    gh_commit_statuses,
    gh_pr_ci_snapshot,
)
from daydream.json_utils import atomic_write_json
from daydream.trajectory import redact_structured_text

RemoteCIStatus = Literal[
    "pending",
    "passed",
    "no_ci",
    "failed",
    "missing",
    "timed_out",
    "unavailable",
    "superseded",
    "cancelled",
]
ObservationState = Literal["pass", "pending", "fail"]

_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_SLUG_PART_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")
_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "key",
        "password",
        "secret",
        "sig",
        "signature",
        "token",
    }
)
_CHECK_PASS = frozenset({"success", "neutral", "skipped"})
_CHECK_FAIL = frozenset(
    {"failure", "cancelled", "timed_out", "action_required", "stale", "startup_failure"}
)
_CHECK_PENDING = frozenset({"queued", "in_progress", "requested", "waiting", "pending"})
_URL_CHARS = 2_000
_WORKFLOW_STATES = frozenset(
    {"active", "deleted", "disabled_fork", "disabled_inactivity", "disabled_manually"}
)
_LIMITATION = (
    "GitHub-reported checks and statuses only; no operating-system or coverage inference."
)


class RemoteCIIdentityMismatch(ValueError):
    """The live REST pull request no longer names the fixed push target."""


@dataclass(frozen=True)
class RemoteCILimits:
    """Finite polling and diagnostic bounds for remote CI verification."""

    poll_seconds: float = 10.0
    discovery_seconds: float = 120.0
    completion_seconds: float = 1800.0
    request_seconds: float = 30.0
    stable_polls: int = 2
    per_page: int = 100
    max_pages: int = 10
    diagnostic_chars: int = 2_000

    def __post_init__(self) -> None:
        finite_positive = (
            self.poll_seconds,
            self.discovery_seconds,
            self.completion_seconds,
            self.request_seconds,
        )
        if any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not isfinite(item)
            or item <= 0
            for item in finite_positive
        ):
            raise ValueError("remote CI time limits must be positive")
        if self.completion_seconds < self.discovery_seconds:
            raise ValueError("completion deadline must not precede discovery deadline")
        for item in (self.stable_polls, self.per_page, self.max_pages, self.diagnostic_chars):
            if not _is_positive_int(item):
                raise ValueError("remote CI count limits must be positive integers")


@dataclass(frozen=True)
class RequiredContext:
    context: str
    app_id: int | None

    def __post_init__(self) -> None:
        _required_text(self.context, "required context")
        if self.app_id is not None and not _is_positive_int(self.app_id):
            raise ValueError("required context app id must be a positive integer or null")


@dataclass(frozen=True)
class RequiredPolicy:
    contexts: tuple[RequiredContext, ...]
    strict: bool

    def __post_init__(self) -> None:
        if not isinstance(self.contexts, tuple) or not all(
            isinstance(item, RequiredContext) for item in self.contexts
        ):
            raise ValueError("required policy contexts must be normalized")
        if not isinstance(self.strict, bool):
            raise ValueError("required policy strict must be boolean")


@dataclass(frozen=True)
class RemoteCITarget:
    target_dir: Path
    base_repository: str
    base_ref: str
    head_repository: str
    head_ref: str
    pr_number: int
    pr_url: str
    remote: str
    pushed_sha: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_dir, Path) or not self.target_dir.is_absolute():
            raise ValueError("remote CI target directory must be an absolute Path")
        object.__setattr__(
            self, "base_repository", _normalize_repository(self.base_repository, "base repository")
        )
        object.__setattr__(
            self, "head_repository", _normalize_repository(self.head_repository, "head repository")
        )
        _required_text(self.base_ref, "base ref")
        _required_text(self.head_ref, "head ref")
        if not _is_positive_int(self.pr_number):
            raise ValueError("PR number must be a positive integer")
        object.__setattr__(self, "pr_url", _safe_url(self.pr_url, _URL_CHARS, "PR URL"))
        _required_text(self.remote, "remote")
        _require_sha(self.pushed_sha, "pushed SHA")


@dataclass(frozen=True)
class PRCIBinding:
    """Latest REST identity for the fixed pull request and pushed head."""

    pr_number: int
    pr_url: str
    base_repository: str
    base_ref: str
    head_repository: str
    head_ref: str
    head_sha: str
    merge_sha: str | None
    state: Literal["open", "closed"]

    def __post_init__(self) -> None:
        if not _is_positive_int(self.pr_number):
            raise ValueError("PR binding number must be a positive integer")
        object.__setattr__(self, "pr_url", _safe_url(self.pr_url, _URL_CHARS, "PR URL"))
        object.__setattr__(
            self, "base_repository", _normalize_repository(self.base_repository, "base repository")
        )
        object.__setattr__(
            self, "head_repository", _normalize_repository(self.head_repository, "head repository")
        )
        _required_text(self.base_ref, "base ref")
        _required_text(self.head_ref, "head ref")
        _require_sha(self.head_sha, "head SHA")
        if self.merge_sha is not None:
            _require_sha(self.merge_sha, "merge SHA")
        if self.state not in {"open", "closed"}:
            raise ValueError("PR state must be open or closed")


@dataclass(frozen=True)
class CIObservation:
    source: Literal["check_run", "status"]
    context: str
    app_id: int | None
    state: ObservationState
    raw_state: str
    url: str | None
    diagnostic: str | None

    def __post_init__(self) -> None:
        if self.source not in {"check_run", "status"}:
            raise ValueError("unknown CI observation source")
        _required_text(self.context, "observation context")
        if self.source == "check_run":
            if not _is_positive_int(self.app_id):
                raise ValueError("check-run observation requires a positive app id")
        elif self.app_id is not None:
            raise ValueError("legacy status observation cannot have an app id")
        if self.state not in {"pass", "pending", "fail"}:
            raise ValueError("unknown CI observation state")
        _required_text(self.raw_state, "raw observation state")


@dataclass(frozen=True)
class RemoteCISnapshot:
    """One complete, trusted GitHub observation poll."""

    target: RemoteCITarget
    binding: PRCIBinding
    policy: RequiredPolicy
    active_workflows: tuple[dict[str, str | int], ...]
    head_observations: tuple[CIObservation, ...]
    merge_observations: tuple[CIObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target, RemoteCITarget):
            raise ValueError("snapshot target must be normalized")
        if not isinstance(self.binding, PRCIBinding):
            raise ValueError("snapshot binding must be normalized")
        if not isinstance(self.policy, RequiredPolicy):
            raise ValueError("snapshot policy must be normalized")
        if not all(isinstance(item, CIObservation) for item in self.head_observations):
            raise ValueError("head observations must be normalized")
        if not all(isinstance(item, CIObservation) for item in self.merge_observations):
            raise ValueError("merge observations must be normalized")


@dataclass(frozen=True)
class RemoteCIVerdict:
    """Normalized state-machine result safe for console and persistence."""

    status: RemoteCIStatus
    reason: str
    target: RemoteCITarget | None
    binding: PRCIBinding | None
    policy: RequiredPolicy | None
    active_workflow_count: int | None
    evidence_sha: str | None
    required_observations: tuple[CIObservation, ...]
    advisory_observations: tuple[CIObservation, ...]
    failing_contexts: tuple[str, ...]
    pending_contexts: tuple[str, ...]
    missing_contexts: tuple[str, ...]
    urls: tuple[str, ...]
    diagnostic: str | None
    stable_polls: int
    elapsed_seconds: float
    limitations: tuple[str, ...] = (_LIMITATION,)

    def __post_init__(self) -> None:
        if self.target is None and self.status != "unavailable":
            raise ValueError("only a pre-target unavailable verdict may omit target identity")

    @property
    def archive_state(self) -> Literal["succeeded", "failed", "partial"]:
        if self.status in {"passed", "no_ci"}:
            return "succeeded"
        if self.status == "failed":
            return "failed"
        return "partial"


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


DEFAULT_LIMITS = RemoteCILimits()


def _required_text(value: object, name: str, *, limit: int = 2_000) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"{name} must be a nonempty bounded string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _require_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase full commit SHA")
    return value


def _normalize_repository(value: object, name: str) -> str:
    text = _required_text(value, name, limit=202)
    pieces = text.split("/")
    if (
        len(pieces) != 2
        or any(part in {".", ".."} for part in pieces)
        or any(_SLUG_PART_RE.fullmatch(part) is None for part in pieces)
    ):
        raise ValueError(f"{name} must be an owner/repository GitHub slug")
    return text.lower()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _field(row: Mapping[str, object], name: str, owner: str) -> object:
    if name not in row:
        raise ValueError(f"{owner} is missing {name}")
    return row[name]


def _safe_url(value: object, limit: int, name: str) -> str:
    text = _required_text(value, name, limit=limit)
    if any(char.isspace() for char in text):
        raise ValueError(f"{name} contains whitespace")
    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise ValueError(f"{name} is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"{name} must be an HTTP(S) URL without userinfo")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    query = urlencode(
        [
            (key, "[REDACTED_CREDENTIAL]" if _sensitive_query_key(key) else item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ],
        doseq=True,
    )
    sanitized = urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    if len(sanitized) > limit:
        raise ValueError(f"{name} exceeds its length bound")
    redacted = redact_structured_text(sanitized)
    if redacted == "[REDACTION_FAILED]" or len(redacted) > limit:
        raise ValueError(f"{name} could not be safely redacted")
    return redacted


def _sensitive_query_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_QUERY_KEYS or normalized.endswith(
        ("_token", "_secret", "_password", "_signature", "_credential", "_api_key")
    )


def _optional_url(value: object, limit: int, name: str) -> str | None:
    if value is None:
        return None
    return _safe_url(value, limit, name)


def _safe_diagnostic(parts: Sequence[object], limit: int) -> str | None:
    if not _is_positive_int(limit):
        raise ValueError("diagnostic limit must be a positive integer")
    texts: list[str] = []
    for value in parts:
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError("diagnostic fields must be strings or null")
        if value:
            texts.append(value)
    if not texts:
        return None
    return redact_structured_text(" — ".join(texts))[:limit]


def parse_pr_ci_binding(raw: object, target: RemoteCITarget) -> PRCIBinding:
    """Validate the latest REST PR row against the one fixed target."""
    row = _mapping(raw, "pull request")
    number = _field(row, "number", "pull request")
    if not _is_positive_int(number):
        raise ValueError("pull request number is invalid")
    if number != target.pr_number:
        raise RemoteCIIdentityMismatch("pull request number changed")
    pr_url = _safe_url(_field(row, "html_url", "pull request"), _URL_CHARS, "PR URL")
    if pr_url != target.pr_url:
        raise RemoteCIIdentityMismatch("pull request URL changed")
    state = _field(row, "state", "pull request")
    if state not in {"open", "closed"}:
        raise ValueError("pull request state is invalid")

    base = _mapping(_field(row, "base", "pull request"), "pull request base")
    head = _mapping(_field(row, "head", "pull request"), "pull request head")
    base_repo = _mapping(_field(base, "repo", "pull request base"), "base repository")
    head_repo = _mapping(_field(head, "repo", "pull request head"), "head repository")
    base_repository = _normalize_repository(
        _field(base_repo, "full_name", "base repository"), "base repository"
    )
    head_repository = _normalize_repository(
        _field(head_repo, "full_name", "head repository"), "head repository"
    )
    base_ref = _required_text(_field(base, "ref", "pull request base"), "base ref")
    head_ref = _required_text(_field(head, "ref", "pull request head"), "head ref")
    if (
        base_repository != target.base_repository
        or base_ref != target.base_ref
        or head_repository != target.head_repository
        or head_ref != target.head_ref
    ):
        raise RemoteCIIdentityMismatch("pull request base/head identity changed")

    head_sha = _require_sha(_field(head, "sha", "pull request head"), "head SHA")
    raw_merge = _field(row, "merge_commit_sha", "pull request")
    merge_sha = None if raw_merge is None else _require_sha(raw_merge, "merge SHA")
    return PRCIBinding(
        pr_number=number,
        pr_url=pr_url,
        base_repository=base_repository,
        base_ref=base_ref,
        head_repository=head_repository,
        head_ref=head_ref,
        head_sha=head_sha,
        merge_sha=merge_sha,
        state=state,
    )


def _parse_required_entry(raw: object, *, app_key: str) -> RequiredContext:
    row = _mapping(raw, "required status check")
    context = _required_text(_field(row, "context", "required status check"), "required context")
    app_id = _field(row, app_key, "required status check")
    if app_id is not None and not _is_positive_int(app_id):
        raise ValueError("required status-check app id must be positive or null")
    return RequiredContext(context=context, app_id=cast(int | None, app_id))


def _normalized_policy(contexts: Sequence[RequiredContext], strict: bool) -> RequiredPolicy:
    by_name: dict[str, set[int | None]] = {}
    for item in contexts:
        by_name.setdefault(item.context, set()).add(item.app_id)
    normalized: list[RequiredContext] = []
    for context, app_ids in by_name.items():
        pinned = sorted(item for item in app_ids if item is not None)
        if pinned:
            normalized.extend(RequiredContext(context, item) for item in pinned)
        else:
            normalized.append(RequiredContext(context, None))
    normalized.sort(key=lambda item: (item.context, -1 if item.app_id is None else item.app_id))
    return RequiredPolicy(tuple(normalized), strict)


def parse_required_policy(active_rules: object, classic: object | None) -> RequiredPolicy:
    """Union active ruleset and classic branch-protection requirements."""
    contexts: list[RequiredContext] = []
    strict = False
    for raw_rule in _sequence(active_rules, "active branch rules"):
        rule = _mapping(raw_rule, "active branch rule")
        rule_type = _required_text(_field(rule, "type", "active branch rule"), "rule type")
        if rule_type != "required_status_checks":
            continue
        parameters = _mapping(
            _field(rule, "parameters", "required status-check rule"),
            "required status-check parameters",
        )
        raw_strict = _field(
            parameters,
            "strict_required_status_checks_policy",
            "required status-check parameters",
        )
        if not isinstance(raw_strict, bool):
            raise ValueError("ruleset strict policy must be boolean")
        strict = strict or raw_strict
        for item in _sequence(
            _field(parameters, "required_status_checks", "required status-check parameters"),
            "required status checks",
        ):
            contexts.append(_parse_required_entry(item, app_key="integration_id"))

    if classic is not None:
        classic_row = _mapping(classic, "classic required checks")
        raw_strict = _field(classic_row, "strict", "classic required checks")
        if not isinstance(raw_strict, bool):
            raise ValueError("classic strict policy must be boolean")
        strict = strict or raw_strict
        for raw_context in _sequence(
            _field(classic_row, "contexts", "classic required checks"), "classic contexts"
        ):
            contexts.append(RequiredContext(_required_text(raw_context, "classic context"), None))
        for item in _sequence(
            _field(classic_row, "checks", "classic required checks"), "classic checks"
        ):
            contexts.append(_parse_required_entry(item, app_key="app_id"))
    return _normalized_policy(contexts, strict)


def parse_active_workflows(rows: object) -> tuple[dict[str, str | int], ...]:
    """Validate the Actions inventory and retain active workflow identity only."""
    active: list[dict[str, str | int]] = []
    seen_ids: set[int] = set()
    for raw in _sequence(rows, "workflow inventory"):
        row = _mapping(raw, "workflow")
        row_id = _field(row, "id", "workflow")
        if not _is_positive_int(row_id) or row_id in seen_ids:
            raise ValueError("workflow id must be unique and positive")
        seen_ids.add(cast(int, row_id))
        name = _required_text(_field(row, "name", "workflow"), "workflow name")
        path = _required_text(_field(row, "path", "workflow"), "workflow path")
        state = _field(row, "state", "workflow")
        if not isinstance(state, str) or state not in _WORKFLOW_STATES:
            raise ValueError("workflow state is unsupported")
        if state == "active":
            active.append({"id": cast(int, row_id), "name": name, "path": path, "state": state})
    active.sort(key=lambda row: cast(int, row["id"]))
    return tuple(active)


def _parse_check_state(status: object, conclusion: object) -> tuple[ObservationState, str]:
    if not isinstance(status, str):
        raise ValueError("check-run status must be a string")
    if status == "completed":
        if not isinstance(conclusion, str):
            raise ValueError("completed check run requires a conclusion")
        if conclusion in _CHECK_PASS:
            return "pass", conclusion
        if conclusion in _CHECK_FAIL:
            return "fail", conclusion
        raise ValueError("check-run conclusion is unsupported")
    if status in _CHECK_PENDING:
        if conclusion is not None:
            raise ValueError("incomplete check run cannot have a conclusion")
        return "pending", status
    raise ValueError("check-run status is unsupported")


def _parse_status_state(state: object) -> ObservationState:
    if state == "success":
        return "pass"
    if state == "pending":
        return "pending"
    if state in {"failure", "error"}:
        return "fail"
    raise ValueError("legacy status state is unsupported")


def parse_observations(
    check_runs: object,
    statuses: object,
    *,
    expected_sha: str,
    limit: int,
) -> tuple[CIObservation, ...]:
    """Strictly parse and reduce observations for exactly one commit SHA."""
    _require_sha(expected_sha, "expected SHA")
    if not _is_positive_int(limit):
        raise ValueError("diagnostic limit must be positive")
    checks_by_producer: dict[tuple[str, int], tuple[int, CIObservation]] = {}
    statuses_by_context: dict[str, CIObservation] = {}
    seen_check_ids: set[int] = set()
    seen_status_ids: set[int] = set()

    for raw in _sequence(check_runs, "check runs"):
        row = _mapping(raw, "check run")
        row_id = _field(row, "id", "check run")
        if not _is_positive_int(row_id) or row_id in seen_check_ids:
            raise ValueError("check-run id must be unique and positive")
        seen_check_ids.add(cast(int, row_id))
        name = _required_text(_field(row, "name", "check run"), "check-run name")
        if _require_sha(_field(row, "head_sha", "check run"), "check-run SHA") != expected_sha:
            raise ValueError("check-run evidence belongs to a different SHA")
        app = _mapping(_field(row, "app", "check run"), "check-run app")
        app_id = _field(app, "id", "check-run app")
        if not _is_positive_int(app_id):
            raise ValueError("check-run app id must be positive")
        state, raw_state = _parse_check_state(
            _field(row, "status", "check run"), _field(row, "conclusion", "check run")
        )
        output = _mapping(_field(row, "output", "check run"), "check-run output")
        diagnostic = _safe_diagnostic(
            (
                _field(output, "title", "check-run output"),
                _field(output, "summary", "check-run output"),
            ),
            limit,
        )
        observation = CIObservation(
            source="check_run",
            context=name,
            app_id=cast(int, app_id),
            state=state,
            raw_state=raw_state,
            url=_optional_url(
                _field(row, "details_url", "check run"), _URL_CHARS, "check-run URL"
            ),
            diagnostic=diagnostic,
        )
        key = (name, cast(int, app_id))
        prior = checks_by_producer.get(key)
        if prior is None or cast(int, row_id) > prior[0]:
            checks_by_producer[key] = (cast(int, row_id), observation)

    for raw in _sequence(statuses, "legacy statuses"):
        row = _mapping(raw, "legacy status")
        row_id = _field(row, "id", "legacy status")
        if not _is_positive_int(row_id) or row_id in seen_status_ids:
            raise ValueError("legacy status id must be unique and positive")
        seen_status_ids.add(cast(int, row_id))
        context = _required_text(_field(row, "context", "legacy status"), "status context")
        if _require_sha(_field(row, "sha", "legacy status"), "status SHA") != expected_sha:
            raise ValueError("legacy status evidence belongs to a different SHA")
        status_key = context.casefold()
        if status_key in statuses_by_context:
            continue
        legacy_raw_state = _field(row, "state", "legacy status")
        state = _parse_status_state(legacy_raw_state)
        statuses_by_context[status_key] = CIObservation(
            source="status",
            context=context,
            app_id=None,
            state=state,
            raw_state=cast(str, legacy_raw_state),
            url=_optional_url(
                _field(row, "target_url", "legacy status"), _URL_CHARS, "status URL"
            ),
            diagnostic=_safe_diagnostic(
                (_field(row, "description", "legacy status"),), limit
            ),
        )

    observations = [item[1] for item in checks_by_producer.values()]
    observations.extend(statuses_by_context.values())
    observations.sort(
        key=lambda item: (
            0 if item.source == "check_run" else 1,
            item.context if item.source == "check_run" else item.context.casefold(),
            -1 if item.app_id is None else item.app_id,
        )
    )
    return tuple(observations)


def _fixed_identity_matches(target: RemoteCITarget, binding: PRCIBinding) -> bool:
    return (
        binding.pr_number == target.pr_number
        and binding.pr_url == target.pr_url
        and binding.base_repository == target.base_repository
        and binding.base_ref == target.base_ref
        and binding.head_repository == target.head_repository
        and binding.head_ref == target.head_ref
    )


def _required_label(item: RequiredContext) -> str:
    if item.app_id is None:
        return item.context
    return f"{item.context} (app {item.app_id})"


def _matches(item: RequiredContext, observation: CIObservation) -> bool:
    if item.app_id is not None:
        return (
            observation.source == "check_run"
            and observation.context == item.context
            and observation.app_id == item.app_id
        )
    if observation.source == "check_run":
        return observation.context == item.context
    return observation.context.casefold() == item.context.casefold()


def _verdict(
    snapshot: RemoteCISnapshot,
    *,
    status: RemoteCIStatus,
    reason: str,
    evidence_sha: str | None,
    required_observations: Sequence[CIObservation] = (),
    advisory_observations: Sequence[CIObservation] = (),
    failing: Sequence[str] = (),
    pending: Sequence[str] = (),
    missing: Sequence[str] = (),
    stable_polls: int,
    elapsed: float,
) -> RemoteCIVerdict:
    all_observations = (*required_observations, *advisory_observations)
    urls = tuple(sorted({item.url for item in all_observations if item.url is not None}))
    return RemoteCIVerdict(
        status=status,
        reason=reason,
        target=snapshot.target,
        binding=snapshot.binding,
        policy=snapshot.policy,
        active_workflow_count=len(snapshot.active_workflows),
        evidence_sha=evidence_sha,
        required_observations=tuple(required_observations),
        advisory_observations=tuple(advisory_observations),
        failing_contexts=tuple(failing),
        pending_contexts=tuple(pending),
        missing_contexts=tuple(missing),
        urls=urls,
        diagnostic=None,
        stable_polls=stable_polls,
        elapsed_seconds=elapsed,
    )


def evaluate_remote_ci(
    snapshot: RemoteCISnapshot,
    *,
    elapsed: float,
    stable_polls: int,
    limits: RemoteCILimits,
) -> RemoteCIVerdict:
    """Evaluate one complete snapshot under the resolved remote-CI policy."""
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("elapsed time must be nonnegative")
    if not isinstance(stable_polls, int) or isinstance(stable_polls, bool) or stable_polls < 1:
        raise ValueError("stable poll count must be positive")
    binding = snapshot.binding
    if binding.state != "open" or not _fixed_identity_matches(snapshot.target, binding):
        return _verdict(
            snapshot,
            status="superseded",
            reason="the fixed pull request identity changed or closed",
            evidence_sha=None,
            stable_polls=stable_polls,
            elapsed=elapsed,
        )
    if binding.head_sha != snapshot.target.pushed_sha:
        status: RemoteCIStatus = "missing" if elapsed >= limits.discovery_seconds else "pending"
        return _verdict(
            snapshot,
            status=status,
            reason="the pushed commit is not yet the pull request head",
            evidence_sha=None,
            stable_polls=stable_polls,
            elapsed=elapsed,
        )

    if binding.merge_sha is not None and snapshot.merge_observations:
        evidence_sha = binding.merge_sha
        observations = snapshot.merge_observations
    else:
        evidence_sha = snapshot.target.pushed_sha
        observations = snapshot.head_observations

    required_observations: list[CIObservation] = []
    matched_ids: set[int] = set()
    failing: list[str] = []
    pending: list[str] = []
    missing: list[str] = []
    for required in snapshot.policy.contexts:
        matches = [item for item in observations if _matches(required, item)]
        if not matches:
            missing.append(_required_label(required))
            continue
        required_observations.extend(matches)
        matched_ids.update(id(item) for item in matches)
        label = _required_label(required)
        if any(item.state == "fail" for item in matches):
            failing.append(label)
        elif any(item.state == "pending" for item in matches):
            pending.append(label)
    required_observations = list(dict.fromkeys(required_observations))
    advisory = [item for item in observations if id(item) not in matched_ids]

    def make(status: RemoteCIStatus, reason: str) -> RemoteCIVerdict:
        return _verdict(
            snapshot,
            status=status,
            reason=reason,
            evidence_sha=evidence_sha,
            required_observations=required_observations,
            advisory_observations=advisory,
            failing=failing,
            pending=pending,
            missing=missing,
            stable_polls=stable_polls,
            elapsed=elapsed,
        )

    if failing:
        return make("failed", "a required CI producer failed")
    if pending:
        if elapsed >= limits.completion_seconds:
            return make("timed_out", "required CI remained pending")
        return make("pending", "required CI is pending")
    if missing:
        if elapsed >= limits.discovery_seconds:
            return make("missing", "required CI was not reported")
        return make("pending", "waiting for required CI")

    if snapshot.policy.contexts:
        if stable_polls < limits.stable_polls:
            return make("pending", "required CI identity is stabilizing")
        return make("passed", "all required CI passed")

    if observations:
        if any(item.state == "pending" for item in observations):
            if elapsed >= limits.completion_seconds:
                return make("timed_out", "reported CI remained pending")
            return make("pending", "reported CI is pending")
        if stable_polls < limits.stable_polls:
            return make("pending", "reported CI identity is stabilizing")
        return make("passed", "all reported CI reached a terminal state")

    if snapshot.active_workflows:
        if elapsed >= limits.discovery_seconds:
            return make("missing", "active workflows reported no CI for the pushed commit")
        return make("pending", "waiting for active workflows")
    if elapsed < limits.discovery_seconds:
        return make("pending", "discovering remote CI")
    if stable_polls < limits.stable_polls:
        return make("pending", "empty CI identity is stabilizing")
    return make("no_ci", "no remote CI is configured")


class RemoteCIFetcher(Protocol):
    """One complete remote-CI poll under a shared absolute request budget."""

    async def fetch(
        self, target: RemoteCITarget, *, budget: GitHubRequestBudget
    ) -> RemoteCISnapshot: ...


@dataclass(frozen=True)
class GitHubRemoteCIFetcher:
    """Fetch normalized CI evidence through the sole async GitHub boundary."""

    limits: RemoteCILimits = field(default_factory=RemoteCILimits)

    async def fetch(
        self, target: RemoteCITarget, *, budget: GitHubRequestBudget
    ) -> RemoteCISnapshot:
        owner, name = target.base_repository.split("/", 1)
        pages = GitHubPageLimits(
            per_page=self.limits.per_page, max_pages=self.limits.max_pages
        )
        try:
            binding = parse_pr_ci_binding(
                await gh_pr_ci_snapshot(
                    target.target_dir,
                    owner,
                    name,
                    target.pr_number,
                    budget=budget,
                ),
                target,
            )
            active_rules = await gh_active_branch_rules(
                target.target_dir,
                owner,
                name,
                target.base_ref,
                limits=pages,
                budget=budget,
            )
            classic = await gh_classic_required_checks(
                target.target_dir,
                owner,
                name,
                target.base_ref,
                budget=budget,
            )
            policy = parse_required_policy(active_rules, classic)
            workflows = parse_active_workflows(
                await gh_actions_workflows(
                    target.target_dir,
                    owner,
                    name,
                    limits=pages,
                    budget=budget,
                )
            )
            head = await self._observations(
                target, owner, name, target.pushed_sha, pages=pages, budget=budget
            )
            merge: tuple[CIObservation, ...] = ()
            if binding.merge_sha is not None and binding.merge_sha != target.pushed_sha:
                merge = await self._observations(
                    target,
                    owner,
                    name,
                    binding.merge_sha,
                    pages=pages,
                    budget=budget,
                )
        except RemoteCIIdentityMismatch:
            raise
        except ValueError as exc:
            raise GitError("GitHub remote CI response failed strict validation") from exc
        return RemoteCISnapshot(
            target=target,
            binding=binding,
            policy=policy,
            active_workflows=workflows,
            head_observations=head,
            merge_observations=merge,
        )

    async def _observations(
        self,
        target: RemoteCITarget,
        owner: str,
        name: str,
        sha: str,
        *,
        pages: GitHubPageLimits,
        budget: GitHubRequestBudget,
    ) -> tuple[CIObservation, ...]:
        checks = await gh_commit_check_runs(
            target.target_dir,
            owner,
            name,
            sha,
            limits=pages,
            budget=budget,
        )
        statuses = await gh_commit_statuses(
            target.target_dir,
            owner,
            name,
            sha,
            limits=pages,
            budget=budget,
        )
        return parse_observations(
            checks, statuses, expected_sha=sha, limit=self.limits.diagnostic_chars
        )


def _effective_observations(snapshot: RemoteCISnapshot) -> tuple[CIObservation, ...]:
    if snapshot.binding.merge_sha is not None and snapshot.merge_observations:
        return snapshot.merge_observations
    return snapshot.head_observations


def _snapshot_stability_key(snapshot: RemoteCISnapshot) -> tuple[object, ...]:
    workflows = tuple(
        (row["id"], row["name"], row["path"], row["state"])
        for row in snapshot.active_workflows
    )
    observations = tuple(
        (item.source, item.context, item.app_id, item.state, item.raw_state)
        for item in _effective_observations(snapshot)
    )
    return (snapshot.binding, snapshot.policy, workflows, observations)


def _external_verdict(
    target: RemoteCITarget,
    *,
    status: Literal["unavailable", "superseded", "cancelled"],
    reason: str,
    detail: str | None,
    elapsed: float,
    stable_polls: int,
    prior: RemoteCIVerdict | None,
    limit: int,
) -> RemoteCIVerdict:
    diagnostic = None
    if detail:
        diagnostic = redact_structured_text(detail)[:limit]
    if prior is not None:
        return replace(
            prior,
            status=status,
            reason=reason,
            diagnostic=diagnostic,
            elapsed_seconds=elapsed,
            stable_polls=max(stable_polls, 0),
        )
    return RemoteCIVerdict(
        status=status,
        reason=reason,
        target=target,
        binding=None,
        policy=None,
        active_workflow_count=None,
        evidence_sha=None,
        required_observations=(),
        advisory_observations=(),
        failing_contexts=(),
        pending_contexts=(),
        missing_contexts=(),
        urls=(),
        diagnostic=diagnostic,
        stable_polls=max(stable_polls, 0),
        elapsed_seconds=elapsed,
    )


def unavailable_remote_ci_verdict(
    *,
    reason: str,
    diagnostic: str | None = None,
    target: RemoteCITarget | None = None,
    elapsed_seconds: float = 0.0,
    limit: int = 2_000,
) -> RemoteCIVerdict:
    """Create an honest fail-closed verdict before a full PR target exists."""
    if not _is_positive_int(limit):
        raise ValueError("diagnostic limit must be positive")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("elapsed time must be a finite nonnegative number")
    safe_reason = redact_structured_text(_required_text(reason, "unavailable reason"))[:limit]
    safe_diagnostic = None
    if diagnostic:
        safe_diagnostic = redact_structured_text(diagnostic)[:limit]
    return RemoteCIVerdict(
        status="unavailable",
        reason=safe_reason,
        target=target,
        binding=None,
        policy=None,
        active_workflow_count=None,
        evidence_sha=None,
        required_observations=(),
        advisory_observations=(),
        failing_contexts=(),
        pending_contexts=(),
        missing_contexts=(),
        urls=(),
        diagnostic=safe_diagnostic,
        stable_polls=0,
        elapsed_seconds=elapsed_seconds,
    )


def pending_remote_ci_verdict(
    target: RemoteCITarget,
    reason: str = "discovering remote CI",
) -> RemoteCIVerdict:
    """Create the current-target marker written before the first CI request."""
    if not isinstance(target, RemoteCITarget):
        raise ValueError("pending remote CI requires a normalized target")
    safe_reason = redact_structured_text(_required_text(reason, "pending reason"))[
        :2_000
    ]
    return RemoteCIVerdict(
        status="pending",
        reason=safe_reason,
        target=target,
        binding=None,
        policy=None,
        active_workflow_count=None,
        evidence_sha=None,
        required_observations=(),
        advisory_observations=(),
        failing_contexts=(),
        pending_contexts=(),
        missing_contexts=(),
        urls=(),
        diagnostic=None,
        stable_polls=0,
        elapsed_seconds=0.0,
    )


def _deadline_verdict(
    snapshot: RemoteCISnapshot,
    *,
    elapsed: float,
    stable_polls: int,
    registered: bool,
    limits: RemoteCILimits,
) -> RemoteCIVerdict:
    verdict = evaluate_remote_ci(
        snapshot, elapsed=elapsed, stable_polls=stable_polls, limits=limits
    )
    if verdict.status != "pending":
        return verdict
    if registered:
        return replace(
            verdict,
            status="timed_out",
            reason="remote CI did not stabilize before the completion deadline",
        )
    return replace(
        verdict,
        status="missing",
        reason="remote CI could not be established before the discovery deadline",
    )


async def wait_for_remote_ci(
    target: RemoteCITarget,
    *,
    fetcher: RemoteCIFetcher,
    limits: RemoteCILimits = DEFAULT_LIMITS,
    monotonic: Callable[[], float] = anyio.current_time,
    monotonic_started_at: float | None = None,
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep,
    on_snapshot: Callable[[RemoteCIVerdict], None],
) -> RemoteCIVerdict:
    """Poll under immutable deadlines and emit only complete trusted states."""
    clock_now = monotonic()
    if not isfinite(clock_now):
        raise ValueError("remote CI monotonic clock must be finite")
    if monotonic_started_at is None:
        started = clock_now
    else:
        if (
            isinstance(monotonic_started_at, bool)
            or not isinstance(monotonic_started_at, (int, float))
            or not isfinite(monotonic_started_at)
            or monotonic_started_at > clock_now
        ):
            raise ValueError("remote CI monotonic start must be finite and not in the future")
        started = float(monotonic_started_at)
    discovery_deadline = started + limits.discovery_seconds
    completion_deadline = started + limits.completion_seconds
    if not isfinite(discovery_deadline) or not isfinite(completion_deadline):
        raise ValueError("remote CI absolute deadlines must be finite")
    last_snapshot: RemoteCISnapshot | None = None
    last_verdict: RemoteCIVerdict | None = None
    last_key: tuple[object, ...] | None = None
    stable_polls = 0
    registered = False
    bound_head = False
    cancelled_error = anyio.get_cancelled_exc_class()

    try:
        now = started
        while now < completion_deadline:
            now = monotonic()
            if not isfinite(now):
                raise ValueError("remote CI monotonic clock must remain finite")
            active_deadline = completion_deadline if registered else discovery_deadline
            if now >= active_deadline:
                if last_snapshot is None:
                    verdict = _external_verdict(
                        target,
                        status="unavailable",
                        reason="remote CI deadline expired before one complete poll",
                        detail=None,
                        elapsed=max(0.0, now - started),
                        stable_polls=stable_polls,
                        prior=last_verdict,
                        limit=limits.diagnostic_chars,
                    )
                else:
                    verdict = _deadline_verdict(
                        last_snapshot,
                        elapsed=max(0.0, now - started),
                        stable_polls=stable_polls,
                        registered=registered,
                        limits=limits,
                    )
                on_snapshot(verdict)
                return verdict

            budget = GitHubRequestBudget(
                deadline=active_deadline,
                per_request_seconds=limits.request_seconds,
                monotonic=monotonic,
            )
            try:
                snapshot = await fetcher.fetch(target, budget=budget)
            except DeadlineExpired:
                now = monotonic()
                if last_snapshot is None:
                    verdict = _external_verdict(
                        target,
                        status="unavailable",
                        reason="remote CI deadline expired during the first poll",
                        detail=None,
                        elapsed=max(0.0, now - started),
                        stable_polls=stable_polls,
                        prior=None,
                        limit=limits.diagnostic_chars,
                    )
                else:
                    verdict = _deadline_verdict(
                        last_snapshot,
                        elapsed=max(0.0, now - started),
                        stable_polls=stable_polls,
                        registered=registered,
                        limits=limits,
                    )
                on_snapshot(verdict)
                return verdict
            except RemoteCIIdentityMismatch as exc:
                verdict = _external_verdict(
                    target,
                    status="superseded",
                    reason="the fixed pull request identity changed",
                    detail=str(exc),
                    elapsed=max(0.0, monotonic() - started),
                    stable_polls=stable_polls,
                    prior=last_verdict,
                    limit=limits.diagnostic_chars,
                )
                on_snapshot(verdict)
                return verdict
            except GitError as exc:
                now = monotonic()
                if last_snapshot is not None and now >= active_deadline:
                    verdict = _deadline_verdict(
                        last_snapshot,
                        elapsed=max(0.0, now - started),
                        stable_polls=stable_polls,
                        registered=registered,
                        limits=limits,
                    )
                else:
                    verdict = _external_verdict(
                        target,
                        status="unavailable",
                        reason="GitHub remote CI evidence is unavailable",
                        detail=str(exc),
                        elapsed=max(0.0, now - started),
                        stable_polls=stable_polls,
                        prior=last_verdict,
                        limit=limits.diagnostic_chars,
                    )
                on_snapshot(verdict)
                return verdict

            now = monotonic()
            if not isfinite(now):
                raise ValueError("remote CI monotonic clock must remain finite")
            if bound_head and snapshot.binding.head_sha != target.pushed_sha:
                current = evaluate_remote_ci(
                    snapshot,
                    elapsed=max(0.0, now - started),
                    stable_polls=max(stable_polls, 1),
                    limits=limits,
                )
                verdict = replace(
                    current,
                    status="superseded",
                    reason="the pull request head changed after binding to the pushed commit",
                )
                on_snapshot(verdict)
                return verdict
            if snapshot.binding.head_sha == target.pushed_sha:
                bound_head = True

            key = _snapshot_stability_key(snapshot)
            stable_polls = stable_polls + 1 if key == last_key else 1
            last_key = key
            last_snapshot = snapshot
            relevant_observations = _effective_observations(snapshot)
            verdict = evaluate_remote_ci(
                snapshot,
                elapsed=max(0.0, now - started),
                stable_polls=stable_polls,
                limits=limits,
            )
            last_verdict = verdict
            if bound_head and snapshot.policy.contexts:
                registered = bool(verdict.required_observations) and not bool(
                    verdict.missing_contexts
                )
            else:
                registered = bound_head and bool(relevant_observations)
            if verdict.status != "pending":
                on_snapshot(verdict)
                return verdict

            active_deadline = completion_deadline if registered else discovery_deadline
            if now >= active_deadline:
                verdict = _deadline_verdict(
                    snapshot,
                    elapsed=max(0.0, now - started),
                    stable_polls=stable_polls,
                    registered=registered,
                    limits=limits,
                )
                on_snapshot(verdict)
                return verdict
            on_snapshot(verdict)
            delay = min(limits.poll_seconds, active_deadline - now)
            if delay <= 0 or not isfinite(delay):
                continue
            await sleep(delay)
        verdict = _external_verdict(
            target,
            status="unavailable",
            reason="remote CI completion deadline elapsed without a trusted terminal state",
            detail=None,
            elapsed=max(0.0, now - started),
            stable_polls=stable_polls,
            prior=last_verdict,
            limit=limits.diagnostic_chars,
        )
        on_snapshot(verdict)
        return verdict
    except cancelled_error:
        now = monotonic()
        verdict = _external_verdict(
            target,
            status="cancelled",
            reason="remote CI verification was cancelled",
            detail=None,
            elapsed=max(0.0, now - started),
            stable_polls=stable_polls,
            prior=last_verdict,
            limit=limits.diagnostic_chars,
        )
        with anyio.CancelScope(shield=True):
            on_snapshot(verdict)
        raise
    except KeyboardInterrupt:
        now = monotonic()
        verdict = _external_verdict(
            target,
            status="cancelled",
            reason="remote CI verification was interrupted",
            detail=None,
            elapsed=max(0.0, now - started),
            stable_polls=stable_polls,
            prior=last_verdict,
            limit=limits.diagnostic_chars,
        )
        with anyio.CancelScope(shield=True):
            on_snapshot(verdict)
        raise


def _target_payload(target: RemoteCITarget | None) -> dict[str, object] | None:
    if target is None:
        return None
    return {
        "base_repository": target.base_repository,
        "base_ref": target.base_ref,
        "head_repository": target.head_repository,
        "head_ref": target.head_ref,
        "pr_number": target.pr_number,
        "pr_url": target.pr_url,
        "remote": target.remote,
        "pushed_sha": target.pushed_sha,
    }


def _binding_payload(binding: PRCIBinding | None) -> dict[str, object] | None:
    if binding is None:
        return None
    return {
        "pr_number": binding.pr_number,
        "pr_url": binding.pr_url,
        "base_repository": binding.base_repository,
        "base_ref": binding.base_ref,
        "head_repository": binding.head_repository,
        "head_ref": binding.head_ref,
        "head_sha": binding.head_sha,
        "merge_sha": binding.merge_sha,
        "state": binding.state,
    }


def _observation_payload(item: CIObservation) -> dict[str, object]:
    return {
        "source": item.source,
        "context": item.context,
        "app_id": item.app_id,
        "state": item.state,
        "raw_state": item.raw_state,
        "url": item.url,
        "diagnostic": item.diagnostic,
    }


def remote_ci_verdict_payload(
    verdict: RemoteCIVerdict,
    *,
    session_id: str,
    poll_count: int,
    started_at: str,
    updated_at: str,
    discovery_deadline: float,
    completion_deadline: float,
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
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
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
        },
        "policy": (
            None
            if verdict.policy is None
            else {
                "strict": verdict.policy.strict,
                "contexts": [
                    {"context": item.context, "app_id": item.app_id}
                    for item in verdict.policy.contexts
                ],
            }
        ),
        "active_workflow_count": verdict.active_workflow_count,
        "required_observations": [
            _observation_payload(item) for item in verdict.required_observations
        ],
        "advisory_observations": [
            _observation_payload(item) for item in verdict.advisory_observations
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
