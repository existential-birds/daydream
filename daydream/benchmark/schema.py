"""Strict Pydantic schemas for the private benchmark workspace.

This module owns the ``benchmark.yaml`` manifest and ``cases/*.yaml`` case
schemas plus their invariants:

* ``normalize_hostname`` and the manifest ``source``/``privacy`` blocks.
* the ``pull_requests[]`` ledger and ``cases[]`` index.
* the snapshot ``ready | unreplayable`` union, the case/gold/provenance/
  exclusion models, ``case_id`` / finding-id derivation, and the Daydream
  self-marker rule.

Every model uses ``extra="forbid"`` so an unknown field is a schema violation.
Later tasks add derived workspace state, transitions, and the ``0/2/1``
classifier.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from daydream.pr_review import FINDING_MARKER_RE

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
    "derive_finding_id",
    "derive_gold_status",
    "derive_gold_mode",
]

_REPOSITORY_SHAPE = re.compile(r"^[^/]+/[^/]+$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_EMPTY_OR_NUL = re.compile(r"[\x00]")


def normalize_hostname(raw: str) -> str:
    """Normalize a DNS hostname, stripping scheme/credentials/port/query path.

    Lowers the host, drops ``<scheme>://``, ``user:pass@``, ``:port`` and a
    trailing ``/path``. Rejects empty strings, wildcards, embedded whitespace,
    and a result with no dot-bearing host segment.
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


# ---------------------------------------------------------------------------
# manifest blocks
# ---------------------------------------------------------------------------


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
    """Privacy / egress configuration for a private benchmark."""

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
        def _cases_key(c: CaseIndexEntry) -> tuple[int, str, str]:
            return (c.pr_number, head_sha_from_case_id(c.case_id), c.case_id)

        ordered = sorted(self.cases, key=_cases_key)
        if [c.case_id for c in ordered] != [c.case_id for c in self.cases]:
            raise ValueError("cases[] index must be sorted by (pr_number, head-sha, case_id)")
        return self


# ---------------------------------------------------------------------------
# ID derivation
# ---------------------------------------------------------------------------


def case_id_for(pr_number: int, head_sha: str) -> str:
    """Derive the canonical ``case_id`` ``pr-<6-digit>-<first-12-hex>``."""
    if not _HEX40.fullmatch(head_sha):
        raise ValueError(f"head SHA must be lowercase 40-hex, got {head_sha!r}")
    return f"pr-{pr_number:06d}-{head_sha[:12]}"


def head_sha_from_case_id(case_id: str) -> str:
    """Extract the 12-hex head-sha prefix from a canonical ``case_id``."""
    return case_id.rsplit("-", 1)[-1]


def _loc_parts(loc: "Location | dict | None") -> tuple[str, str, str]:
    if loc is None:
        return ("", "", "")
    if isinstance(loc, dict):
        return (
            str(loc.get("path") or ""),
            str(loc.get("start_line") or ""),
            str(loc.get("end_line") or ""),
        )
    return (str(loc.path), str(loc.start_line), str(loc.end_line))


def _field_of(value: "Finding | dict", name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def derive_finding_id(finding: "Finding | dict") -> str:
    """sha256 over the canonical (title, body, severity, path, start/end) tuple.

    Nulls are normalized to the empty string; ``finding_id`` must equal this
    digest so a case's findings are content-addressable and dedupe-friendly.
    """
    payload = "\x1f".join(
        [
            str(_field_of(finding, "title") or ""),
            str(_field_of(finding, "body") or ""),
            str(_field_of(finding, "severity") or ""),
            *_loc_parts(_field_of(finding, "location")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# snapshot union
# ---------------------------------------------------------------------------


class _SnapshotBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    policy: str
    requested_head: str

    @field_validator("policy")
    @classmethod
    def _policy_nonblank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("policy must not be blank")
        return v


class SnapshotReady(_SnapshotBase):
    status: Literal["ready"]
    original_base_sha: str
    original_head_sha: str
    base_tree_sha: str
    head_tree_sha: str
    diff_sha256: str
    bundle_file: str
    bundle_sha256: str
    error: None = None

    @field_validator(
        "original_base_sha", "original_head_sha", "base_tree_sha", "head_tree_sha"
    )
    @classmethod
    def _sha40(cls, v: str) -> str:
        if not _HEX40.fullmatch(v):
            raise ValueError(f"SHA must be lowercase 40-hex, got {v!r}")
        return v

    @field_validator("diff_sha256", "bundle_sha256")
    @classmethod
    def _sha64(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(f"digest must be lowercase 64-hex, got {v!r}")
        return v


_SNAPSHOT_ERROR_REASON = Literal[
    "head_unreachable",
    "head_not_on_pr",
    "base_unreachable",
    "missing_object",
    "equal_trees",
    "empty_diff",
    "bundle_failure",
]


class _SnapshotError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: _SNAPSHOT_ERROR_REASON
    detail: str


class SnapshotUnreplayable(_SnapshotBase):
    status: Literal["unreplayable"]
    original_base_sha: str | None = None
    original_head_sha: str | None = None
    base_tree_sha: None = None
    head_tree_sha: None = None
    diff_sha256: None = None
    bundle_file: None = None
    bundle_sha256: None = None
    error: _SnapshotError

    @field_validator("original_base_sha", "original_head_sha")
    @classmethod
    def _sha40_nullable(cls, v: str | None) -> str | None:
        if v is not None and not _HEX40.fullmatch(v):
            raise ValueError(f"SHA must be lowercase 40-hex, got {v!r}")
        return v


Snapshot = Annotated[SnapshotReady | SnapshotUnreplayable, Field(discriminator="status")]

# ---------------------------------------------------------------------------
# location / finding / provenance / exclusions
# ---------------------------------------------------------------------------


class Location(BaseModel):
    """A POSIX-relative source location with a positive ordered line span."""

    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int
    end_line: int

    @field_validator("path")
    @classmethod
    def _relative_path(cls, v: str) -> str:
        if not v:
            raise ValueError("location path must not be blank")
        if v.startswith("/") or (":" in v and not v.startswith("http")):
            raise ValueError(f"location path must be relative, got {v!r}")
        if v == ".." or v.startswith("../") or "/../" in v or v.endswith("/.."):
            raise ValueError(f"location path must not contain '..' segments: {v!r}")
        if "\x00" in v:
            raise ValueError("location path must not contain NUL")
        return v

    @model_validator(mode="after")
    def _ordered(self) -> "Location":
        if self.start_line < 1 or self.end_line < 1:
            raise ValueError("start_line/end_line must be positive")
        if self.start_line > self.end_line:
            raise ValueError("start_line must be <= end_line")
        return self


_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_./:-]+$")


class Provenance(BaseModel):
    """Where a finding came from (historical review output, edited, or authored)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["historical", "edited", "authored"]
    source_ids: list[str] = []

    @model_validator(mode="after")
    def _cardinality(self) -> "Provenance":
        if self.kind == "historical" and len(self.source_ids) != 1:
            raise ValueError("historical provenance requires exactly one source ID")
        if self.kind == "edited" and len(self.source_ids) < 1:
            raise ValueError("edited provenance requires at least one source ID")
        return self


class Finding(BaseModel):
    """A single gold finding (or an authored/edited candidate)."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    title: str
    body: str
    severity: Literal["high", "medium", "low"] | None = None
    location: Location | None = None
    provenance: Provenance

    @field_validator("finding_id")
    @classmethod
    def _id_hex(cls, v: str) -> str:
        if not _HEX64.fullmatch(v):
            raise ValueError(f"finding_id must be 64-hex, got {v!r}")
        return v

    @field_validator("title")
    @classmethod
    def _title_limit(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title must not be blank")
        if "\x00" in v:
            raise ValueError("title must not contain NUL")
        if len(v.encode("utf-8")) > 500:
            raise ValueError("title exceeds 500 UTF-8 bytes")
        return v

    @field_validator("body")
    @classmethod
    def _body_limit(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("body must not be blank")
        if "\x00" in v:
            raise ValueError("body must not contain NUL")
        if len(v.encode("utf-8")) > 8 * 1024:
            raise ValueError("body exceeds 8 KiB")
        return v

    @model_validator(mode="after")
    def _canonical_id(self) -> "Finding":
        if self.finding_id != derive_finding_id(self):
            raise ValueError("finding_id is not the canonical sha256")
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


class _NoteForOther(BaseModel):
    """Require a note on an exclusion model when ``reason == "other"``.

    ``reason`` / ``note`` are declared here so the shared validator typechecks;
    concrete subclasses narrow ``reason`` to their own Literal.
    """

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

    model_config = ConfigDict(extra="forbid")
    _exclusion_noun: ClassVar[str] = "evidence exclusion"

    source_id: str
    reason: _EVIDENCE_REASON
    note: str | None = None


_CASE_EXCLUSION_REASON = Literal["unreplayable", "not_suitable", "duplicate_case", "other"]


class CaseExclusion(_NoteForOther):
    """Why an entire case was excluded from the dataset."""

    model_config = ConfigDict(extra="forbid")
    _exclusion_noun: ClassVar[str] = "case exclusion"

    reason: _CASE_EXCLUSION_REASON
    note: str | None = None


class Curation(BaseModel):
    """Curated gold state for one case."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["draft", "ready", "stale", "excluded", "unreplayable"]
    snapshot_attested: bool = False
    clean_attested: bool = False
    gold_status: Literal["findings", "clean"] | None = None
    findings: list[Finding] = []
    exclusions: list[EvidenceExclusion] = []
    case_exclusion: CaseExclusion | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Curation":
        if self.case_exclusion is not None and self.state != "excluded":
            raise ValueError("case_exclusion is only valid when state == 'excluded'")
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


class CaseDocument(BaseModel):
    """One ``cases/<case-id>.yaml`` document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str
    pull_request: dict
    snapshot: Snapshot
    source: dict
    curation: Curation

    @model_validator(mode="after")
    def _case_id_matches(self) -> "CaseDocument":
        pr_number = int(self.pull_request["number"])
        head = snapshot_head_sha(self.snapshot)
        if head is None:
            raise ValueError("snapshot carries no head SHA to derive case_id")
        expected = case_id_for(pr_number, head)
        if self.case_id != expected:
            raise ValueError(f"case_id {self.case_id!r} mismatches {expected!r}")
        return self

    @model_validator(mode="after")
    def _unique_findings(self) -> "CaseDocument":
        ids = [f.finding_id for f in self.curation.findings]
        if len(set(ids)) != len(ids):
            raise ValueError("case contains duplicate canonical findings")
        return self


def snapshot_head_sha(snapshot: "SnapshotReady | SnapshotUnreplayable") -> str | None:
    """The 40-hex head SHA of a snapshot, or None when unknown (unreplayable)."""
    return snapshot.original_head_sha


def derive_gold_status(curation: Curation) -> str | None:
    """Yes: findings (>=1 finding), clean (0 findings + attested), else draft none."""
    if curation.findings:
        return "findings"
    if not curation.findings and curation.clean_attested:
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


# ---------------------------------------------------------------------------
# state transitions, derived workspace state, 0/2/1 classifier
# ---------------------------------------------------------------------------


class TransitionError(Exception):
    """An invalid PR-import or case-curation state transition."""

    def __init__(self, frm: str, to: str):
        super().__init__(f"invalid transition {frm!r} -> {to!r}")
        self.frm = frm
        self.to = to


_PR_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"fetched", "fetch_failed"},
    "fetch_failed": {"fetched"},
    "fetched": {"fetched"},
}

_CASE_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"ready", "excluded", "unreplayable"},
    "ready": {"stale", "draft"},
    "stale": {"ready", "excluded"},
    "unreplayable": {"excluded"},
    "excluded": {"draft", "unreplayable"},
}


def validate_pr_transition(frm: str, to: str) -> None:
    """Raise :class:`TransitionError` unless ``frm -> to`` is a valid PR ledger move."""
    allowed = _PR_TRANSITIONS.get(frm, set())
    if to not in allowed:
        raise TransitionError(frm, to)


def validate_case_transition(frm: str, to: str) -> None:
    """Raise :class:`TransitionError` unless ``frm -> to`` is a valid curation move."""
    allowed = _CASE_TRANSITIONS.get(frm, set())
    if to not in allowed:
        raise TransitionError(frm, to)


def derive_workspace_state(
    *,
    pull_requests: list[dict] | None = None,
    cases: list[dict] | None = None,
) -> str:
    """Derive the workspace state from ledger + case index.

    Priority per §5: ``collecting`` > ``stale`` > ``ready`` > ``empty``. A
    ``corrupt`` flag (schema/checksum/path/bundle) is surfaced by the caller
    via ``classify_validation``; here we only reason over the ledger/index.
    """
    pull_requests = pull_requests or []
    cases = cases or []
    any_collecting = False
    any_stale = False
    any_ready = False
    any_draft_or_unreplayable = False

    for pr in pull_requests:
        state = pr.get("import_state")
        if state in ("pending", "fetch_failed"):
            any_collecting = True

    for c in cases:
        cs = c.get("curation_state")
        if cs in ("draft", "unreplayable"):
            any_draft_or_unreplayable = True
        elif cs == "stale":
            any_stale = True
        elif cs == "ready":
            any_ready = True

    if any_collecting:
        return "collecting"
    if any_draft_or_unreplayable:
        return "curating"
    if any_stale:
        return "stale"
    if any_ready:
        return "ready"
    return "empty"


def classify_validation(*, ready: bool, incomplete: bool, corrupt: bool) -> int:
    """Map readiness to a ``0``/``2``/``1`` validation exit code.

    ``0`` ready; ``2`` structurally valid but incomplete; ``1`` corrupt.
    ``corrupt`` always takes precedence over ``incomplete``.
    """
    if corrupt:
        return 1
    if ready:
        return 0
    return 2
