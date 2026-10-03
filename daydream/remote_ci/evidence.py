"""Normalized CI identities, limits, observations, and bounded value validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Annotated, Literal, Mapping, Sequence, TypeGuard, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BeforeValidator, ConfigDict, Field, StrictFloat, StrictInt
from pydantic.dataclasses import dataclass as validated_dataclass

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


# Native values and receipt reconstruction use the same strict scalar declarations.
# Live REST normalization and canonical receipt identity comparison remain distinct.
_CI_VALUE_CONFIG = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)
_CIText = Annotated[str, Field(strict=True, min_length=1, max_length=2_000, pattern=r"^[^\x00-\x1f\x7f]+\z")]
_PositiveInt = Annotated[StrictInt, Field(gt=0)]


def _positive_number(value: object) -> int | float:
    if not _is_finite_number(value) or value <= 0:
        raise ValueError("remote CI time limits must be positive")
    return value


_PositiveNumber = Annotated[StrictInt | StrictFloat, BeforeValidator(_positive_number)]
_CommitSHA = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{40}\z")]


class RemoteCIIdentityMismatch(ValueError):
    """The live REST pull request no longer names the fixed push target."""


@validated_dataclass(frozen=True, config=_CI_VALUE_CONFIG)
class RemoteCILimits:
    """Finite polling and diagnostic bounds for remote CI verification."""

    poll_seconds: _PositiveNumber = 10.0
    discovery_seconds: _PositiveNumber = 120.0
    completion_seconds: _PositiveNumber = 1800.0
    request_seconds: _PositiveNumber = 30.0
    stable_polls: _PositiveInt = 2
    per_page: _PositiveInt = 100
    max_pages: _PositiveInt = 10
    diagnostic_chars: _PositiveInt = 2_000

    def __post_init__(self) -> None:
        if self.completion_seconds < self.discovery_seconds:
            raise ValueError("completion deadline must not precede discovery deadline")


@validated_dataclass(frozen=True, config=_CI_VALUE_CONFIG)
class RequiredContext:
    context: _CIText
    app_id: _PositiveInt | None


@validated_dataclass(frozen=True, config=_CI_VALUE_CONFIG)
class RequiredPolicy:
    contexts: tuple[RequiredContext, ...]
    strict: bool


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
    object.__setattr__(target, "pr_url", _safe_url(target.pr_url, _URL_CHARS, "PR URL"))


@validated_dataclass(frozen=True, config=_CI_VALUE_CONFIG)
class RemoteCITarget:
    target_dir: Path
    base_repository: str
    base_ref: _CIText
    head_repository: str
    head_ref: _CIText
    pr_number: _PositiveInt
    pr_url: str
    remote: _CIText
    pushed_sha: _CommitSHA

    def __post_init__(self) -> None:
        if not self.target_dir.is_absolute():
            raise ValueError("remote CI target directory must be an absolute Path")
        _normalize_remote_identity(self)


@validated_dataclass(frozen=True, config=_CI_VALUE_CONFIG)
class PRCIBinding:
    """Latest REST identity for the fixed pull request and pushed head."""

    pr_number: _PositiveInt
    pr_url: str
    base_repository: str
    base_ref: _CIText
    head_repository: str
    head_ref: _CIText
    head_sha: _CommitSHA
    merge_sha: _CommitSHA | None
    state: Literal["open", "closed"]

    def __post_init__(self) -> None:
        _normalize_remote_identity(self)


@validated_dataclass(frozen=True, config=_CI_VALUE_CONFIG)
class CIObservation:
    source: Literal["check_run", "status"]
    context: _CIText
    app_id: _PositiveInt | None
    state: ObservationState
    raw_state: _CIText
    url: _CIText | None
    diagnostic: _CIText | None

    def __post_init__(self) -> None:
        if self.source == "check_run":
            if self.app_id is None:
                raise ValueError("check-run observation requires a positive app id")
        elif self.app_id is not None:
            raise ValueError("legacy status observation cannot have an app id")


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
