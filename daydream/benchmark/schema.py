"""Strict workspace, case, import, snapshot, and evidence schemas. Unknown fields are
forbidden. Identity derivation, hostname normalization, provenance, exclusion, and
self-marker checks belong here.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Literal, get_args
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from daydream.reviews.identity import FINDING_MARKER_RE
from daydream.severity import SeverityLevel


class _StrictModel(BaseModel):
    """Shared schema base: every benchmark model rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "Source",
    "Privacy",
    "PullRequestEntry",
    "CaseIndexEntry",
    "BenchmarkManifest",
    "normalize_hostname",
    "case_id_for",
    "CaseDocument",
    "Snapshot",
    "SnapshotReady",
    "SnapshotUnreplayable",
    "SnapshotImported",
    "AuthoringAnchor",
    "EvidenceRecord",
    "Candidate",
    "PrioritizationFacts",
    "PrioritizationCandidate",
    "ImportDocument",
    "PullRequestMeta",
    "CaseSource",
    "derive_finding_id",
    "derive_gold_status",
    "derive_gold_mode",
    "EVIDENCE_REASONS",
    "CASE_EXCLUSION_REASONS",
]

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _rfc3339(value: str | datetime) -> datetime:
    """Parse an RFC3339 timestamp (UTC) into a timezone-aware datetime."""
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must carry a UTC offset, got {value!r}")
    return parsed.astimezone(timezone.utc)


def rfc3339_now() -> str:
    """Current UTC time as an RFC3339 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_ts(v: str | datetime | None) -> datetime | None:
    if v is None:
        return None
    return _rfc3339(v)


_SOURCE_ID_RE = re.compile(r"\Agithub:(review|inline_comment|thread_comment|issue_comment):\d+\Z")


Sha40 = Annotated[str, Field(pattern=r"\A[0-9a-f]{40}\z")]
Sha64 = Annotated[str, Field(pattern=r"\A[0-9a-f]{64}\z")]
Sha64OrEmpty = Annotated[str, Field(pattern=r"\A(?:[0-9a-f]{64})?\z")]
NullableSha40 = Sha40 | None
PositiveLine = Annotated[int, Field(ge=1)] | None
Timestamp = Annotated[datetime, BeforeValidator(_validate_ts)]
OptionalTimestamp = Annotated[datetime | None, BeforeValidator(_validate_ts)]
SourceId = Annotated[str, Field(pattern=_SOURCE_ID_RE)]


def normalize_hostname(raw: str) -> str:
    """Normalize a DNS hostname, stripping scheme/credentials/port/query path."""
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


# manifest blocks


class Source(_StrictModel):
    """Immutable repository identity for the workspace's forge (github.com)."""

    provider: Literal["github"]
    hostname: Annotated[str, Field(pattern=r"\Agithub\.com\z")]
    repository: Annotated[str, Field(pattern=r"^[^/]+/[^/]+$")]
    repository_id: str | None = None
    visibility: Literal["unresolved", "public", "private"] = "unresolved"

    @field_validator("repository_id")
    @classmethod
    def _repository_id_nonblank_opaque(cls, v: str | None) -> str | None:
        # None is the unresolved sentinel (unchanged); blank and numeric-only
        # strings are never valid GitHub node ids (e.g. R_kgD...).
        if v is None:
            return v
        stripped = v.strip()
        if not stripped or stripped.isdigit():
            raise ValueError(f"repository_id must be a nonblank opaque string, got {v!r}")
        return stripped


class Privacy(_StrictModel):
    """Privacy / egress configuration for a private benchmark."""

    classification: Literal["confidential"]
    reviewer_data: Literal["source_snapshot"]
    reviewer_allowed_hosts: list[str]
    judge_data: Literal["finding_text_and_location_only"]
    judge_allowed_hosts: list[str]
    archive: Literal["disabled"]
    uploads: Literal["disabled"]

    @field_validator("reviewer_allowed_hosts", "judge_allowed_hosts")
    @classmethod
    def _allowed_hosts(cls, value: list[str], info: ValidationInfo) -> list[str]:
        return _normalize_host_list(value, str(info.field_name))


class PullRequestEntry(_StrictModel):
    """One entry in the ``pull_requests[]`` ledger."""

    number: int
    import_state: Literal["pending", "fetched", "fetch_failed"]
    import_file: Annotated[str, Field(min_length=1)] | None = None
    import_sha256: Sha64 | None = None
    error: dict[str, str] | None = None
    latest_error: dict[str, str] | None = None
    requested_heads: list[str] = []
    case_ids: list[str] = []

    @model_validator(mode="after")
    def _conditional(self) -> "PullRequestEntry":
        if self.import_state == "fetched":
            if self.import_file is None or self.import_sha256 is None:
                raise ValueError("fetched import requires import_file and import_sha256")
            if self.error is not None:
                raise ValueError("fetched import must not carry an error")
        elif self.import_state == "fetch_failed":
            if self.error is None:
                raise ValueError("fetch_failed import requires an error")
            if self.import_file is not None or self.import_sha256 is not None:
                raise ValueError("fetch_failed import must not set import_file/import_sha256")
            if self.latest_error is not None:
                raise ValueError("fetch_failed import must not carry a latest_error")
        else:  # pending
            if (
                self.import_file is not None
                or self.import_sha256 is not None
                or self.error is not None
                or self.latest_error is not None
            ):
                raise ValueError(
                    "pending import must not set import_file/import_sha256/error/latest_error"
                )
        return self


class CaseIndexEntry(_StrictModel):
    """One entry of the ``cases[]`` index."""

    case_id: str
    pr_number: int
    case_file: str


class BenchmarkManifest(_StrictModel):
    """The ``benchmark.yaml`` workspace manifest."""

    schema_version: Literal[1] = 1
    benchmark_id: UUID
    created_at: Timestamp
    source: Source
    privacy: Privacy
    pull_requests: list[PullRequestEntry] = []
    cases: list[CaseIndexEntry] = []

    @model_validator(mode="after")
    def _cases_ordered(self) -> "BenchmarkManifest":
        def _cases_key(c: CaseIndexEntry) -> tuple[int, str, str]:
            return (c.pr_number, head_sha_from_case_id(c.case_id), c.case_id)

        ordered = sorted(self.cases, key=_cases_key)
        if [c.case_id for c in ordered] != [c.case_id for c in self.cases]:
            raise ValueError("cases[] index must be sorted by (pr_number, head-sha, case_id)")
        ids = [c.case_id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("cases[] index must not contain duplicate case_id rows")
        return self


# ID derivation


def case_id_for(pr_number: int, head_sha: str) -> str:
    """Derive the canonical ``case_id`` ``pr-<6-digit>-<first-12-hex>``."""
    if not _HEX40.fullmatch(head_sha):
        raise ValueError(f"head SHA must be lowercase 40-hex, got {head_sha!r}")
    return f"pr-{pr_number:06d}-{head_sha[:12]}"


def head_sha_from_case_id(case_id: str) -> str:
    """Extract the 12-hex head-sha prefix from a canonical ``case_id``."""
    return case_id.rsplit("-", 1)[-1]


def _loc_parts(loc: "Location | dict[str, Any] | None") -> tuple[str, str, str]:
    if loc is None:
        return ("", "", "")
    if isinstance(loc, dict):
        return (
            str(loc.get("path") or ""),
            str(loc.get("start_line") or ""),
            str(loc.get("end_line") or ""),
        )
    return (str(loc.path), str(loc.start_line), str(loc.end_line))


def _field_of(value: "Finding | dict[str, Any]", name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def derive_finding_id(finding: "Finding | dict[str, Any]", *, case_id: str) -> str:
    """Hash case-scoped content/location, normalizing nulls to empty strings for stable
    finding identity.
    """
    payload = "\x1f".join(
        [
            str(case_id or ""),
            str(_field_of(finding, "title") or ""),
            str(_field_of(finding, "body") or ""),
            str(_field_of(finding, "severity") or ""),
            *_loc_parts(_field_of(finding, "location")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# snapshot union


class _SnapshotBase(_StrictModel):

    status: str
    policy: Literal["final_pr_head", "explicit_head"]
    requested_head: str


class SnapshotReady(_SnapshotBase):
    status: Literal["ready"]
    base_resolution: Literal["merge_base_v1"]
    # original_base_sha is the true merge base of base-tip and head (the
    # bundle's synthetic base commit); requested_base_sha is the selected
    # base-branch tip the merge base was resolved against.
    original_base_sha: Sha40
    requested_base_sha: Sha40
    original_head_sha: Sha40
    base_tree_sha: Sha40
    head_tree_sha: Sha40
    diff_sha256: Sha64
    bundle_file: str
    bundle_sha256: Sha64
    error: None = None


_SNAPSHOT_ERROR_REASON = Literal[
    "head_unreachable",
    "head_not_on_pr",
    "base_unreachable",
    "missing_object",
    "equal_trees",
    "empty_diff",
    "bundle_failure",
    "base_drift",
]


class _SnapshotError(_StrictModel):
    reason: _SNAPSHOT_ERROR_REASON
    detail: str


class SnapshotUnreplayable(_SnapshotBase):
    status: Literal["unreplayable"]
    original_base_sha: NullableSha40 = None
    requested_base_sha: NullableSha40 = None
    original_head_sha: NullableSha40 = None
    base_tree_sha: None = None
    head_tree_sha: None = None
    diff_sha256: None = None
    bundle_file: None = None
    bundle_sha256: None = None
    error: _SnapshotError


class SnapshotImported(_SnapshotBase):
    """PR-known pins before snapshot freezing. Both base fields hold the PR tip; the
    merge-base and bundle are determined only when freezing transitions this record to
    ready or unreplayable.
    """

    status: Literal["imported"]
    original_base_sha: Sha40
    requested_base_sha: Sha40
    original_head_sha: Sha40
    error: None = None


Snapshot = Annotated[
    SnapshotReady | SnapshotUnreplayable | SnapshotImported,
    Field(discriminator="status"),
]

# location / finding / provenance / exclusions


def exact_git_tree_path(value: Any) -> str:
    """Validate a Git-tree path without normalizing legal filename bytes."""
    if not isinstance(value, str):
        raise ValueError("Git tree path must be a string")
    if not value:
        raise ValueError("Git tree path must not be blank")
    if value.startswith("/"):
        raise ValueError(f"Git tree path must be relative, got {value!r}")
    if "\x00" in value:
        raise ValueError("Git tree path must not contain NUL")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Git tree path contains an invalid component: {value!r}")
    return value


def _relative_path(v: str, *, what: str = "location") -> str:
    """A POSIX-relative source path (shared by Location and AuthoringAnchor)."""
    if not v:
        raise ValueError(f"{what} path must not be blank")
    if v.startswith("/") or (":" in v and not v.startswith("http")):
        raise ValueError(f"{what} path must be relative, got {v!r}")
    if v == ".." or v.startswith("../") or "/../" in v or v.endswith("/.."):
        raise ValueError(f"{what} path must not contain '..' segments: {v!r}")
    if "\x00" in v:
        raise ValueError(f"{what} path must not contain NUL")
    return v


class Location(_StrictModel):
    """A POSIX-relative source location with a positive ordered line span."""

    path: str
    start_line: Annotated[int, Field(ge=1)]
    end_line: Annotated[int, Field(ge=1)]

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        return _relative_path(v)

    @model_validator(mode="after")
    def _ordered(self) -> "Location":
        if self.start_line > self.end_line:
            raise ValueError("start_line must be <= end_line")
        return self


class _EvidenceAuthor(_StrictModel):
    """The author of an evidence record (login + GitHub user/bot type)."""

    login: str
    type: str


class AuthoringAnchor(_StrictModel):
    """Versioned original comment location derived from the authenticated mirror. Derived
    anchors carry commit/path/range; other statuses require all data fields unset.
    GitHub re-anchored fields are never substituted.
    """

    version: Literal[1]
    status: Literal["derived", "history-unavailable", "path-unavailable", "range-unavailable"]
    commit_id: NullableSha40
    path: str | None
    start_line: PositiveLine
    end_line: PositiveLine

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _relative_path(v, what="anchor")

    @model_validator(mode="after")
    def _derived_iff_populated(self) -> "AuthoringAnchor":
        populated = (
            self.commit_id is not None
            and self.path is not None
            and self.start_line is not None
            and self.end_line is not None
        )
        if self.status == "derived":
            if not populated:
                raise ValueError("derived anchor requires commit_id, path, start_line, and end_line")
            if self.start_line > self.end_line:  # type: ignore[operator]
                raise ValueError("start_line must be <= end_line")
        elif populated:
            raise ValueError(f"anchor status {self.status!r} must leave commit/path/lines None")
        return self


class EvidenceRecord(_StrictModel):
    """One normalized GitHub PR evidence record."""

    source_id: SourceId
    kind: Literal["review", "inline_comment", "thread_comment", "issue_comment"]
    database_id: int
    node_id: str
    author: _EvidenceAuthor
    body: str
    body_sha256: Sha64
    created_at: Timestamp
    updated_at: Timestamp
    submitted_at: OptionalTimestamp = None
    commit_id: NullableSha40 = None
    original_commit_id: NullableSha40 = None
    path: str | None = None
    original_path: str | None = None
    line: PositiveLine = None
    start_line: PositiveLine = None
    original_line: PositiveLine = None
    original_start_line: PositiveLine = None
    authoring_anchor: AuthoringAnchor | None = None
    review_id: str | None = None
    thread_id: str | None = None
    reply_to_id: str | None = None
    subject_type: Literal["line", "file"] | None = None
    side: Literal["LEFT", "RIGHT"] | None = None
    start_side: Literal["LEFT", "RIGHT"] | None = None
    resolved: bool = False
    outdated: bool = False
    dismissed: bool = False
    state: str | None = None
    is_bot: bool
    url: str

    @model_validator(mode="after")
    def _body_hash(self) -> "EvidenceRecord":
        if self.body and self.body_sha256 != hashlib.sha256(self.body.encode("utf-8")).hexdigest():
            raise ValueError("body_sha256 must equal sha256(body)")
        return self


class Candidate(_StrictModel):
    """An import-time deterministic candidate projected from one evidence record."""

    source_id: SourceId
    title: str
    body: str
    severity: None = None
    location: Location | None = None
    exact_acceptable: bool
    not_exact_reason: str | None = None

    @model_validator(mode="after")
    def _reason(self) -> "Candidate":
        if self.exact_acceptable is False and self.not_exact_reason is None:
            raise ValueError("not_exact_reason is required when exact_acceptable is False")
        if self.exact_acceptable and self.not_exact_reason is not None:
            raise ValueError("not_exact_reason must be blank when exact_acceptable is True")
        return self


class _ImportRepository(_StrictModel):
    """The resolved repository identity captured at import time."""

    id: str
    name_with_owner: str
    visibility: Literal["public", "private"]

    @field_validator("id")
    @classmethod
    def _id_is_opaque_string(cls, value: str) -> str:
        if value.strip().isdigit():
            raise ValueError("repository id must be an opaque string, not numeric")
        return value


class _FetchInfo(_StrictModel):
    """The fetch bookkeeping of one import document."""

    fetched_at: str
    etag: str | None = None
    payload_sha256: Sha64


class _PrRef(_StrictModel):
    """A base/head ref-sha pair of a pull request."""

    sha: str
    ref: str | None = None


class PullRequestMeta(_StrictModel):
    """Shared PR identity with strict required fields and backward-compatible additive
    metadata. Legacy imports default optional body/hash/url/merge/close fields to empty
    or None; unknown fields remain forbidden.
    """

    number: int
    url: str
    title: str
    state: str
    base: _PrRef
    head: _PrRef
    created_at: Timestamp
    updated_at: Timestamp
    author: _EvidenceAuthor
    html_url: str = ""
    body: str = ""
    title_sha256: Sha64OrEmpty = ""
    body_sha256: Sha64OrEmpty = ""
    merged_at: OptionalTimestamp = None
    closed_at: OptionalTimestamp = None
    changed_files: list[str] | None = None

    @field_validator("changed_files", mode="before")
    @classmethod
    def _canonical_changed_files(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("changed_files must be a list or null")
        paths = [exact_git_tree_path(path) for path in value]
        if paths != sorted(set(paths)):
            raise ValueError("changed_files must be sorted and duplicate-free")
        return paths

    @model_validator(mode="after")
    def _body_hash_consistency(self) -> "PullRequestMeta":
        if self.body_sha256 != "" and self.body_sha256 != hashlib.sha256(
            self.body.encode("utf-8")
        ).hexdigest():
            raise ValueError("body_sha256 must equal sha256(body)")
        if self.title_sha256 != "" and self.title_sha256 != hashlib.sha256(
            self.title.encode("utf-8")
        ).hexdigest():
            raise ValueError("title_sha256 must equal sha256(title)")
        return self


class CaseSource(_StrictModel):
    """The typed source block of a case document (import provenance)."""

    import_file: str
    import_sha256: Sha64


class ImportDocument(_StrictModel):
    """A normalized, verifiable import of one PR's full evidence set."""

    schema_version: Literal[1] = 1
    repository: _ImportRepository
    pull_request: PullRequestMeta
    evidence: list[EvidenceRecord] = []
    fetch: _FetchInfo


class Provenance(_StrictModel):
    """Where a finding came from (historical review output, edited, or authored)."""

    kind: Literal["historical", "edited", "authored"]
    source_ids: list[str] = []

    @model_validator(mode="after")
    def _cardinality(self) -> "Provenance":
        if self.kind == "historical" and len(self.source_ids) != 1:
            raise ValueError("historical provenance requires exactly one source ID")
        if self.kind == "edited" and len(self.source_ids) < 1:
            raise ValueError("edited provenance requires at least one source ID")
        return self


class Finding(_StrictModel):
    """A single gold finding (or an authored/edited candidate)."""

    finding_id: Sha64
    title: str
    body: str
    severity: SeverityLevel | None = None
    location: Location | None = None
    provenance: Provenance

    @field_validator("title", "body")
    @classmethod
    def _text_limits(cls, value: str, info: ValidationInfo) -> str:
        field = info.field_name
        if not value.strip():
            raise ValueError(f"{field} must not be blank")
        if "\x00" in value:
            raise ValueError(f"{field} must not contain NUL")
        if field == "title" and len(value) > 500:
            raise ValueError("title exceeds 500 characters")
        if field == "body" and len(value.encode("utf-8")) > 8 * 1024:
            raise ValueError("body exceeds 8 KiB")
        return value

    @model_validator(mode="after")
    def _historical_marker(self) -> "Finding":
        if self.provenance.kind == "historical":
            text = f"{self.title}\n{self.body}"
            if FINDING_MARKER_RE.search(text):
                raise ValueError("historical findings must not carry the Daydream self-marker")
        return self


_EVIDENCE_REASON = Literal[
    "fixed_before_snapshot",
    "not_actionable",
    "incorrect",
    "duplicate",
    "style_only",
    "out_of_scope",
    "other",
]


class _NoteForOther(_StrictModel):
    """Require a note on an exclusion model when ``reason == "other"``."""

    _exclusion_noun: ClassVar[str] = "exclusion"

    reason: str
    note: str | None = None

    @model_validator(mode="after")
    def _note_for_other(self) -> "_NoteForOther":
        if self.reason == "other" and not self.note:
            raise ValueError(f"{self._exclusion_noun} with reason 'other' requires a note")
        return self


class EvidenceExclusion(_NoteForOther):
    """A reason an individual finding/evidence item was excluded from gold."""

    _exclusion_noun: ClassVar[str] = "evidence exclusion"

    source_id: str
    reason: _EVIDENCE_REASON


_CASE_EXCLUSION_REASON = Literal["unreplayable", "not_suitable", "duplicate_case", "other"]

# Runtime spelling of the exclusion vocabularies above; the Literals remain the
# single owner so a reason cannot be declared in one place and rejected in the other.
EVIDENCE_REASONS = frozenset(get_args(_EVIDENCE_REASON))
CASE_EXCLUSION_REASONS = frozenset(get_args(_CASE_EXCLUSION_REASON))


class CaseExclusion(_NoteForOther):
    """Why an entire case was excluded from the dataset."""

    _exclusion_noun: ClassVar[str] = "case exclusion"

    reason: _CASE_EXCLUSION_REASON


# Single source of truth for the snapshot-comparison fact extraction version.
# Defined here (next to :class:`PrioritizationFacts`) so both github_import (the
# writer) and curation (the reader) can import it without a circular import.
EXTRACTION_VERSION = 1


class PrioritizationCandidate(_StrictModel):
    """Per-evidence comparison facts against the pinned snapshot head."""

    commit_relation: Literal["at_head", "ancestor", "non_ancestor", "unavailable"]
    anchor_delta: Literal[
        "unchanged", "changed", "renamed", "deleted", "binary", "locationless", "unavailable"
    ]


class PrioritizationFacts(_StrictModel):
    """Additive per-case prioritization facts (schema_version stays 2)."""

    extraction_version: int
    head_sha: Sha40
    candidates: dict[str, PrioritizationCandidate] = {}
    non_candidates: dict[str, PrioritizationCandidate] = {}


class Curation(_StrictModel):
    """Curated gold state for one case."""

    state: Literal["draft", "ready", "stale", "excluded", "unreplayable"]
    snapshot_attested: bool = False
    clean_attested: bool = False
    gold_status: Literal["findings", "clean"] | None = None
    findings: list[Finding] = []
    exclusions: list[EvidenceExclusion] = []
    case_exclusion: CaseExclusion | None = None
    task_spec_sha256: Sha64 | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Curation":
        if self.case_exclusion is not None and self.state != "excluded":
            raise ValueError("case_exclusion is only valid when state == 'excluded'")
        if self.state == "ready" and self.task_spec_sha256 is None:
            raise ValueError("ready curation requires task_spec_sha256 (approved spec digest)")
        if self.state == "ready" and self.snapshot_attested is not True:
            raise ValueError("ready curation requires snapshot_attested=True")
        if self.state == "stale" and self.snapshot_attested is not False:
            raise ValueError("stale curation requires snapshot_attested=False")
        if self.gold_status == "findings":
            if not self.findings or self.clean_attested:
                raise ValueError("gold_status 'findings' requires >=1 finding and clean_attested=False")
        elif self.gold_status == "clean":
            if self.findings or not self.clean_attested:
                raise ValueError("gold_status 'clean' requires zero findings and clean_attested=True")
        elif self.state == "draft":
            if self.gold_status is not None or self.clean_attested:
                raise ValueError("draft curation must have gold_status None and clean_attested=False")
        return self


def _schema_ready(raw: dict[str, Any]) -> dict[str, Any]:
    """Strip audit fields and backfill missing legacy-ready task-spec hashes. Render from
    raw case content using the same deterministic contract as compilation, so strict
    readers can load legacy cases without rewriting them.
    """
    doc = dict(raw)
    curation = dict(raw.get("curation") or {})
    curation.pop("gold_mode", None)
    curation.pop("task_spec_approved_at", None)
    if curation.get("state") == "ready" and "task_spec_sha256" not in curation:
        from daydream.benchmark.harbor.build import (
            task_spec_digest,
        )

        curation["task_spec_sha256"] = task_spec_digest(doc)
    doc["curation"] = curation
    return doc


class CaseDocument(_StrictModel):
    """One ``cases/<case-id>.yaml`` document."""

    schema_version: Literal[1, 2] = 2
    case_id: str
    pull_request: PullRequestMeta
    snapshot: Snapshot
    source: CaseSource
    curation: Curation
    candidates: list[Candidate] = []
    prioritization: PrioritizationFacts | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "CaseDocument":
        """Validate cross-document invariants in stable failure order."""
        ids = [c.source_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("case contains duplicate candidate source_ids")
        pr_number = self.pull_request.number
        head = self.snapshot.original_head_sha
        if head is None:
            raise ValueError("snapshot carries no head SHA to derive case_id")
        expected = case_id_for(pr_number, head)
        if self.case_id != expected:
            raise ValueError(f"case_id {self.case_id!r} mismatches {expected!r}")
        if self.schema_version == 2:
            for i, f in enumerate(self.curation.findings):
                if f.finding_id != derive_finding_id(f, case_id=self.case_id):
                    raise ValueError(
                        f"finding[{i}] finding_id is not the canonical sha256 for case {self.case_id}"
                    )
        ids = [f.finding_id for f in self.curation.findings]
        if len(set(ids)) != len(ids):
            raise ValueError("case contains duplicate canonical findings")
        explicitly_excluded = (
            self.curation.state == "excluded" and self.curation.case_exclusion is not None
        )
        if not explicitly_excluded and (
            (self.curation.state == "unreplayable")
            != (self.snapshot.status == "unreplayable")
        ):
            raise ValueError(
                "unreplayable snapshot and curation states must match unless explicitly excluded"
            )
        requested = self.snapshot.requested_base_sha
        if requested is not None and requested != self.pull_request.base.sha:
            raise ValueError("snapshot requested_base_sha must match pull_request.base.sha")
        return self


def derive_gold_status(curation: Curation) -> str | None:
    """Yes: findings (>=1 finding), clean (0 findings + attested), else draft none."""
    if curation.findings:
        return "findings"
    if curation.clean_attested:
        return "clean"
    return None


def derive_gold_mode(curation: Curation) -> str:
    """Derive the gold provenance mode of a curation's findings."""
    kinds = {f.provenance.kind for f in curation.findings}
    if not kinds:
        return "clean"
    if kinds == {"authored"}:
        return "authored"
    if "authored" in kinds:
        return "mixed"
    return "historical"


# state transitions, derived workspace state, 0/2/1 classifier


class TransitionError(Exception):
    """An invalid PR-import or case-curation state transition."""

    def __init__(self, frm: str, to: str):
        super().__init__(f"invalid transition {frm!r} -> {to!r}")
        self.frm = frm
        self.to = to


class PreflightLedger(_StrictModel):
    """Private repository verification ledger written only after exact identity and read
    access succeed.
    """

    schema_version: Literal[1] = 1
    last_verified_at: str
    repository: str
    repository_id: str | None
    visibility: Literal["public", "private"]
    matched: bool


_PR_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"fetched", "fetch_failed"},
    "fetch_failed": {"fetched", "fetch_failed"},
    # Failed refresh retains fetched linkage and writes latest_error. Only first-import
    # failures with no linkage transition to fetch_failed.
    "fetched": {"fetched"},
}

_CASE_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"ready", "excluded", "unreplayable"},
    "ready": {"stale", "draft"},
    "stale": {"ready", "excluded"},
    "unreplayable": {"excluded"},
    "excluded": {"draft", "unreplayable"},
}


def _validate_transition(table: dict[str, set[str]], frm: str, to: str) -> None:
    if to not in table.get(frm, set()):
        raise TransitionError(frm, to)


def validate_pr_transition(frm: str, to: str) -> None:
    """Raise :class:`TransitionError` unless ``frm -> to`` is a valid PR ledger move."""
    _validate_transition(_PR_TRANSITIONS, frm, to)


def validate_case_transition(frm: str, to: str) -> None:
    """Raise :class:`TransitionError` unless ``frm -> to`` is a valid curation move."""
    _validate_transition(_CASE_TRANSITIONS, frm, to)


def derive_workspace_state(
    *,
    pull_requests: list[dict[str, Any]] | None = None,
    cases: list[dict[str, Any]] | None = None,
) -> str:
    """Prioritize collecting, curating, stale, ready, then empty from ledger and cases.
    Callers classify corruption separately through classify_validation.
    """
    pr_states = [pr.get("import_state") for pr in pull_requests or []]
    case_states = [case.get("curation_state") for case in cases or []]
    for present, result in (
        (any(state in ("pending", "fetch_failed") for state in pr_states), "collecting"),
        (any(state in ("draft", "unreplayable") for state in case_states), "curating"),
        ("stale" in case_states, "stale"),
        ("ready" in case_states, "ready"),
    ):
        if present:
            return result
    return "empty"


def classify_validation(*, ready: bool, corrupt: bool) -> int:
    """Map readiness to a ``0``/``2``/``1`` validation exit code."""
    if corrupt:
        return 1
    if ready:
        return 0
    return 2
