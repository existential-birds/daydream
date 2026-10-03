"""Normalized CI identities, limits, observations, and bounded value validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, Mapping, Sequence, TypeGuard, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from daydream.redaction import redact_structured_text

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
            not _is_finite_number(item) or item <= 0
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


def _normalize_remote_identity(target: RemoteCITarget | PRCIBinding) -> None:
    """Normalize the repository/ref/url identity shared by remote-CI targets."""
    object.__setattr__(
        target,
        "base_repository",
        _normalize_repository(target.base_repository, "base repository"),
    )
    object.__setattr__(
        target,
        "head_repository",
        _normalize_repository(target.head_repository, "head repository"),
    )
    _required_text(target.base_ref, "base ref")
    _required_text(target.head_ref, "head ref")
    object.__setattr__(target, "pr_url", _safe_url(target.pr_url, _URL_CHARS, "PR URL"))


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
        _normalize_remote_identity(self)
        if not _is_positive_int(self.pr_number):
            raise ValueError("PR number must be a positive integer")
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
        _normalize_remote_identity(self)
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
        for name in ("url", "diagnostic"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, f"observation {name}")


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

    @property
    def evidence(self) -> tuple[str, tuple[CIObservation, ...]]:
        """Prefer populated merge-commit evidence; otherwise use the pushed head."""
        if self.binding.merge_sha is not None and self.merge_observations:
            return self.binding.merge_sha, self.merge_observations
        return self.target.pushed_sha, self.head_observations


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


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _is_positive_int(value: object) -> TypeGuard[int]:
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
