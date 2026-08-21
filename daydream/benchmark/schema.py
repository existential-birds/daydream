"""Strict Pydantic schemas for the private benchmark workspace.

This module owns the workspace's ``benchmark.yaml`` manifest shape and its
invariants: host normalization, the ``source``/``privacy`` blocks, the PR
ledger (``pull_requests``), and the case index (``cases``). Every model uses
``extra="forbid"`` so an unknown field is a schema violation, never silently
ignored. Later tasks add the snapshot union, case/gold/provenance/exclusion
models, derived state, transitions, and the ``0/2/1`` validation classifier.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "Source",
    "Privacy",
    "PullRequestEntry",
    "CaseIndexEntry",
    "BenchmarkManifest",
    "normalize_hostname",
]

_REPOSITORY_SHAPE = re.compile(r"^[^/]+/[^/]+$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def normalize_hostname(raw: str) -> str:
    """Normalize a DNS hostname, stripping scheme/credentials/port/path.

    Lowers the host, drops ``<scheme>://``, ``user:pass@``, ``:port`` and a
    trailing ``/path``. Rejects empty strings, wildcards, embedded whitespace,
    and a result with no dot-bearing host segment (so a bare single label like
    ``localhost`` is rejected, but ``api.anthropic.com`` is kept).
    """
    if not isinstance(raw, str):
        raise ValueError(f"hostname must be a string, got {raw!r}")
    host = raw
    if "://" in host:
        host = host.split("://", 1)[1]
    if "@" in host:
        host = host.split("@", 1)[1]
    if ":" in host:
        host = host.split(":", 1)[0]
    host = host.split("/", 1)[0]
    host = host.strip().lower()
    if not host:
        raise ValueError(f"invalid hostname {raw!r}")
    if "*" in host:
        raise ValueError(f"wildcard hostname not allowed: {raw!r}")
    if any(ch.isspace() for ch in host):
        raise ValueError(f"hostname must not contain whitespace: {raw!r}")
    if "." not in host:
        raise ValueError(f"hostname has no dot-bearing host segment: {raw!r}")
    return host


def _normalize_host_list(values: list[str], what: str) -> list[str]:
    if not values:
        raise ValueError(f"{what} must not be empty")
    return [normalize_hostname(str(h)) for h in values]


class Source(BaseModel):
    """Immutable repository identity for the workspace's forge (github.com)."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["github"]
    hostname: str
    repository: str
    repository_id: int | None = None
    visibility: Literal["unresolved", "public", "private"] = "unresolved"

    @field_validator("hostname")
    @classmethod
    def _github_only(cls, v: str) -> str:
        if v != "github.com":
            raise ValueError(f"v1 supports only the github.com forge, got {v!r}")
        return v

    @field_validator("repository")
    @classmethod
    def _repo_shape(cls, v: str) -> str:
        if not v or not _REPOSITORY_SHAPE.match(v):
            raise ValueError(f"repository must be OWNER/REPO, got {v!r}")
        return v


class Privacy(BaseModel):
    """Privacy/egress configuration: classification, reviewer/judge data + hosts."""

    model_config = ConfigDict(extra="forbid")

    classification: str
    reviewer_data: str
    reviewer_allowed_hosts: list[str]
    judge_data: str
    judge_allowed_hosts: list[str]
    archive: str
    uploads: str

    @field_validator("reviewer_allowed_hosts")
    @classmethod
    def _reviewer_hosts(cls, v: list[str]) -> list[str]:
        return _normalize_host_list(v, "reviewer_allowed_hosts")

    @field_validator("judge_allowed_hosts")
    @classmethod
    def _judge_hosts(cls, v: list[str]) -> list[str]:
        return _normalize_host_list(v, "judge_allowed_hosts")


class PullRequestEntry(BaseModel):
    """One entry in the ``pull_requests[]`` ledger."""

    model_config = ConfigDict(extra="forbid")

    number: int
    import_state: Literal["pending", "fetched", "fetch_failed"]
    import_file: str | None = None
    import_sha256: str | None = None
    error: dict[str, str] | None = None
    requested_heads: list[str] = []
    case_ids: list[str] = []

    @model_validator(mode="after")
    def _conditional(self) -> "PullRequestEntry":
        if self.import_state == "fetched":
            if self.import_file is None or self.import_sha256 is None:
                raise ValueError("fetched import requires import_file and import_sha256")
            if not _HEX64.fullmatch(self.import_sha256):
                raise ValueError(f"import_sha256 must be 64-hex, got {self.import_sha256!r}")
            if self.error is not None:
                raise ValueError("fetched import must not carry an error")
        elif self.import_state == "fetch_failed":
            if self.error is None:
                raise ValueError("fetch_failed import requires an error")
            if self.import_file is not None or self.import_sha256 is not None:
                raise ValueError("fetch_failed import must not set import_file/import_sha256")
        else:  # pending
            if self.import_file is not None or self.import_sha256 is not None or self.error is not None:
                raise ValueError("pending import must not set import_file/import_sha256/error")
        if self.import_file is not None and not self.import_file:
            raise ValueError("import_file must not be blank")
        return self


class CaseIndexEntry(BaseModel):
    """One entry of the ``cases[]`` index."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    pr_number: int
    case_file: str


class BenchmarkManifest(BaseModel):
    """The ``benchmark.yaml`` workspace manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    benchmark_id: UUID
    created_at: datetime
    source: Source
    privacy: Privacy
    pull_requests: list[PullRequestEntry] = []
    cases: list[CaseIndexEntry] = []

    @field_validator("created_at")
    @classmethod
    def _created_at_utc(cls, v: str | datetime) -> datetime:
        value = v if isinstance(v, datetime) else datetime.fromisoformat(v)
        if value.tzinfo is None:
            raise ValueError(f"created_at must carry a UTC offset, got {v!r}")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _cases_ordered(self) -> "BenchmarkManifest":
        ordered = sorted(self.cases, key=lambda c: (c.pr_number, c.case_id))
        if [c.case_id for c in ordered] != [c.case_id for c in self.cases]:
            raise ValueError("cases[] index must be sorted by (pr_number, case_id)")
        return self
