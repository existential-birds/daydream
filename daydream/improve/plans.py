"""Persistent plan-directory state for the improve advisor flow."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.improve.prioritize import member_alias, plan_priority
from daydream.improve.render import (
    _redact_model_value,
    markdown_cell,
    plan_slug,
    render_plan,
)
from daydream.trajectory import redact_text

REJECTIONS_SCHEMA_VERSION = 1
PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION = 1
PLAN_INDEX_SCHEMA_VERSION = 1
PLAN_INDEX_FILENAME = ".index.json"
_FINGERPRINT_MARKER = re.compile(
    r"<!--\s*fingerprint:([^\s>]+)\s*-->"
)
_NUMBERED_PLAN = re.compile(r"^(\d{3})-[a-z0-9-]+\.md$")
_SAFE_ERROR_DETAIL = re.compile(r"^[A-Za-z0-9_.;=-]{1,80}$")
_HOST_BLOCKED_STATUS = re.compile(
    r"^BLOCKED \(PLAN_(?:WRITER|VALIDATION)_FAILED: [^()\r\n]+\)$"
)
# Plan | Title | Priority | Effort | Status
_INDEX_COLUMNS = 5
_INDEX_ROW_NUMBER = re.compile(r"\b(\d{3})\b")
# The slug class admits no separator or dot, so a recovered link can never name
# anything but a sibling plan file.
_INDEX_ROW_LINK = re.compile(r"\[\d{3}\]\(\d{3}-([a-z0-9-]+)\.md\)")
# Re-anchor worktree directory names are built from the run session id, so only
# a filesystem-safe run id may reach the path; anything else falls back to an
# anchor-derived name.
_SAFE_DIRNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
# Re-anchor worktree dirnames end in this suffix; the prune pass and the
# re-anchor write share it so a rename can never silently stop pruning.
_REANCHOR_DIR_SUFFIX = "-reanchor"
REANCHORED_STATUS_PREFIX = "REANCHORED"
_REANCHORED_LANDED = re.compile(rf"^{re.escape(REANCHORED_STATUS_PREFIX)} \(landed at (.+)\)$")


def load_rejections(plans_dir: Path) -> dict[str, dict[str, Any]]:
    """Load durable rejections keyed by fingerprint.

    An absent, unreadable, malformed, or structurally invalid file is treated
    as empty so stale user-authored state cannot prevent a fresh audit.
    """
    path = plans_dir / "rejected.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REJECTIONS_SCHEMA_VERSION
        or not isinstance(payload.get("rejected"), list)
    ):
        return {}

    rejections: dict[str, dict[str, Any]] = {}
    for entry in payload["rejected"]:
        if not isinstance(entry, dict):
            continue
        fingerprint = entry.get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            rejections[fingerprint] = entry
    return rejections


def record_rejections(
    plans_dir: Path, entries: Sequence[dict[str, Any]]
) -> None:
    """Append rejection entries to the versioned durable envelope."""
    if not entries:
        return
    rejected = [
        _redact_model_value(entry)
        for entry in load_rejections(plans_dir).values()
    ]
    rejected.extend(
        _redact_model_value(dict(entry))
        for entry in entries
    )
    plans_dir.mkdir(parents=True, exist_ok=True)
    (plans_dir / "rejected.json").write_text(
        json.dumps(
            {
                "schema_version": REJECTIONS_SCHEMA_VERSION,
                "rejected": rejected,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


_SAFE_METADATA_LABEL = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")


def _safe_metadata_label(value: Any, *, fallback: str) -> str:
    text = redact_text(str(value or "").strip())
    if not _SAFE_METADATA_LABEL.fullmatch(text):
        return fallback
    return text


def _received_metadata(value: Any) -> dict[str, Any]:
    received_type = (
        "null"
        if value is None
        else "object"
        if isinstance(value, dict)
        else "array"
        if isinstance(value, list)
        else type(value).__name__
    )
    metadata: dict[str, Any] = {
        "type": received_type,
        "object_count": 0,
        "array_count": 0,
        "string_count": 0,
        "string_length": 0,
        "top_level_count": (
            len(value) if isinstance(value, (dict, list)) else None
        ),
    }

    def count_shape(item: Any) -> None:
        if isinstance(item, dict):
            metadata["object_count"] += 1
            for child in item.values():
                count_shape(child)
        elif isinstance(item, list):
            metadata["array_count"] += 1
            for child in item:
                count_shape(child)
        elif isinstance(item, str):
            metadata["string_count"] += 1
            metadata["string_length"] += len(item)

    count_shape(value)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        metadata["sha256"] = None
        metadata["serialized_length"] = None
    else:
        metadata["sha256"] = hashlib.sha256(serialized).hexdigest()
        metadata["serialized_length"] = len(serialized)
    return metadata


def _validation_error(code_with_pointer: str) -> dict[str, str]:
    code, separator, remainder = code_with_pointer.partition("@")
    embedded_pointer, _, detail = remainder.partition("#")
    # Assembly issues always carry their own pointer; the host codes raised
    # around them are plan-wide.
    pointer = (
        embedded_pointer
        if separator and embedded_pointer.startswith("/")
        else "/"
    )
    if detail and _SAFE_ERROR_DETAIL.fullmatch(detail):
        return {"code": code, "pointer": pointer, "detail": detail}
    return {"code": code, "pointer": pointer}


# Codes emitted by assemble._collect_issues (seam 2, model authoring defects).
_AUTHORING_CODES = frozenset(
    {
        "AUTHOR_SCHEMA_INVALID",
        "MALFORMED_APPENDED_ARGS",
        "MALFORMED_PATH",
        "PATH_OUTSIDE_REPOSITORY",
        "EMPTY_SCOPE",
        "EXISTING_PATH_MISSING",
        "EXISTING_PATH_NOT_QUOTED",
        "NEW_PATH_ALREADY_EXISTS",
        "EXCERPT_ANCHOR_INVALID",
        "EXCERPT_PATH_MISSING",
        "RECON_COMMAND_UNKNOWN",
        "CREATE_PATH_NOT_NEW",
        "CHANGE_PATH_NOT_EXISTING",
        "TEST_EXEMPLAR_INVALID",
        "STOP_PATH_UNKNOWN",
    }
)


def _validation_stage(errors: Sequence[str]) -> str:
    codes = [error.partition("@")[0] for error in errors]
    if any(code == "RENDER_FAILED" for code in codes):
        return "render"
    if any(code in _AUTHORING_CODES for code in codes):
        return "authoring"
    return "semantic"


def _attempt_diagnostic(
    *,
    finding: dict[str, Any],
    attempt: dict[str, Any] | None,
    received: Any,
    disposition: str,
    stage: str,
    errors: Sequence[str] = (),
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempt = attempt or {}
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "finding": {
            "fingerprint": str(finding.get("fingerprint") or ""),
            "title": redact_text(
                str(finding.get("title") or "Selected finding")
            ),
        },
        "planner": {
            "descriptor": _safe_metadata_label(
                attempt.get("descriptor"),
                fallback="plan-writer",
            ),
            "backend": _safe_metadata_label(
                attempt.get("backend"),
                fallback="unknown-backend",
            ),
            "model": _safe_metadata_label(
                attempt.get("model"),
                fallback="unknown-model",
            ),
        },
        "disposition": disposition,
        "stage": stage,
        "errors": [_validation_error(error) for error in errors],
        "validation_errors": [_validation_error(error) for error in errors],
        "received": _received_metadata(received),
        "artifact": artifact,
    }


def record_plan_write_diagnostics(
    path: Path,
    attempts: Sequence[dict[str, Any]],
    *,
    artifact_provenance: dict[str, str] | None = None,
) -> None:
    """Append sanitized plan-attempt metadata without retaining model content."""
    existing_attempts: list[dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("schema_version")
            == PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION
            and isinstance(existing.get("attempts"), list)
            and (
                artifact_provenance is None
                or existing.get("artifact_provenance")
                == artifact_provenance
            )
        ):
            existing_attempts = [
                _redact_model_value(item)
                for item in existing["attempts"]
                if isinstance(item, dict)
            ]
    payload = {
        "schema_version": PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION,
        "artifact_type": "daydream.plan-write-diagnostics",
        **(
            {"artifact_provenance": dict(artifact_provenance)}
            if artifact_provenance is not None
            else {}
        ),
        "attempts": [
            _redact_model_value(item)
            for item in [*existing_attempts, *attempts]
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


@dataclass(frozen=True)
class PlanIndexEntry:
    """One plan's durable record in ``daydream_plans/.index.json``."""

    number: int
    slug: str
    title: str
    fingerprint: str
    package_fingerprint: str
    member_fingerprints: tuple[str, ...]
    member_aliases: tuple[str, ...]
    priority: str
    effort: str
    risk: str
    category: str
    planned_at: str
    status: str
    host_blocked: bool
    change_shape: str
    maintenance_signals: tuple[str, ...]
    reuse_target: str

    @property
    def path(self) -> str | None:
        """The plan file this entry names, or ``None`` when none was written."""
        return f"{self.number:03d}-{self.slug}.md" if self.slug else None

    @property
    def landing_path(self) -> str | None:
        """The repo-relative landing path of a re-anchored plan, or ``None``.

        ``None`` when the status lacks the ``(landed at ...)`` suffix — the
        path is parsed tolerantly and never synthesized.
        """
        match = _REANCHORED_LANDED.match(self.status)
        return match.group(1) if match else None


def _index_field(value: Any) -> str:
    """Normalize a model- or operator-supplied index field for durable storage."""
    return redact_text(str(value or "").strip())


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(item.strip() for item in value if isinstance(item, str) and item.strip()))


def _string_sequence(value: Any) -> tuple[str, ...]:
    """Normalize a string sequence while preserving meaningful multiplicity."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _finding_package_fingerprint(finding: dict[str, Any]) -> str:
    package = finding.get("package_fingerprint")
    if isinstance(package, str) and package:
        return package
    fingerprint = finding.get("fingerprint")
    return fingerprint if isinstance(fingerprint, str) else ""


def _entry_fingerprints(entry: PlanIndexEntry) -> frozenset[str]:
    return frozenset(
        value
        for value in (
            entry.package_fingerprint,
            entry.fingerprint,
            *entry.member_fingerprints,
            *entry.member_aliases,
        )
        if value
    )


def _finding_member_fingerprints(finding: dict[str, Any], *, fallback: str) -> tuple[str, ...]:
    members = _string_tuple(finding.get("member_fingerprints"))
    return members or ((fallback,) if fallback else ())


def _finding_member_aliases(finding: dict[str, Any]) -> tuple[str, ...]:
    return _string_sequence(finding.get("member_aliases"))


def _finding_member_identities(
    finding: dict[str, Any],
) -> tuple[frozenset[str], ...]:
    """Return the alternative durable IDs for each current package member."""
    nested = finding.get("members")
    if isinstance(nested, list):
        valid_members = [item for item in nested if isinstance(item, dict)]
        semantic_aliases = [member_alias(item) for item in valid_members]
        alias_counts = Counter(semantic_aliases)
        groups: list[frozenset[str]] = []
        for item, semantic_alias in zip(valid_members, semantic_aliases, strict=True):
            identities = {
                value
                for value in (
                    item.get("fingerprint"),
                    (semantic_alias if alias_counts[semantic_alias] == 1 else None),
                )
                if isinstance(value, str) and value
            }
            if identities:
                groups.append(frozenset(identities))
        if groups:
            return tuple(groups)

    fingerprints = _finding_member_fingerprints(
        finding,
        fallback=_finding_package_fingerprint(finding),
    )
    aliases = _finding_member_aliases(finding)
    if aliases and len(aliases) == len(fingerprints):
        alias_counts = Counter(aliases)
        groups = [
            frozenset(
                value
                for value in (
                    fingerprint,
                    alias if alias_counts[alias] == 1 else None,
                )
                if value
            )
            for fingerprint, alias in zip(fingerprints, aliases, strict=True)
        ]
    else:
        groups = [frozenset((fingerprint,)) for fingerprint in fingerprints]
    # A singleton's semantic package ID is itself a valid cross-run alias.
    # For multi-member packages it must not stand in for every member: doing so
    # would silently suppress newly added work after a membership change.
    if len(groups) == 1:
        package = _finding_package_fingerprint(finding)
        if package:
            groups[0] = groups[0] | {package}
        if aliases and len(aliases) != len(fingerprints):
            groups[0] = groups[0] | set(aliases)
    return tuple(groups)


def _entry_member_coverage(entry: PlanIndexEntry) -> frozenset[str]:
    values = {*entry.member_fingerprints, *entry.member_aliases}
    if len(entry.member_fingerprints) <= 1:
        values.update((entry.package_fingerprint, entry.fingerprint))
    return frozenset(value for value in values if value)


def _fully_covered(identities: tuple[frozenset[str], ...], coverage: set[str] | frozenset[str]) -> bool:
    return bool(identities) and all(group & coverage for group in identities)


def _entry_payload(entry: PlanIndexEntry) -> dict[str, Any]:
    return {
        "number": entry.number,
        "slug": entry.slug,
        "title": entry.title,
        "fingerprint": entry.fingerprint,
        "package_fingerprint": entry.package_fingerprint,
        "member_fingerprints": list(entry.member_fingerprints),
        "member_aliases": list(entry.member_aliases),
        "priority": entry.priority,
        "effort": entry.effort,
        "risk": entry.risk,
        "category": entry.category,
        "planned_at": entry.planned_at,
        "status": entry.status,
        "host_blocked": entry.host_blocked,
        "change_shape": entry.change_shape,
        "maintenance_signals": list(entry.maintenance_signals),
        "reuse_target": entry.reuse_target,
    }


def _entry_from_payload(payload: Any) -> PlanIndexEntry | None:
    if not isinstance(payload, dict):
        return None
    number = payload.get("number")
    fingerprint = payload.get("fingerprint")
    package_fingerprint = payload.get("package_fingerprint")
    slug = _index_field(payload.get("slug"))
    status = _index_field(payload.get("status"))
    if (
        not isinstance(number, int)
        or isinstance(number, bool)
        or not 0 < number < 1000
        or not isinstance(fingerprint, str)
        or not fingerprint
        or not status
        or (slug and _NUMBERED_PLAN.fullmatch(f"{number:03d}-{slug}.md") is None)
    ):
        return None
    if not isinstance(package_fingerprint, str) or not package_fingerprint:
        package_fingerprint = fingerprint
    member_fingerprints = _string_tuple(payload.get("member_fingerprints"))
    if not member_fingerprints:
        member_fingerprints = (fingerprint,)
    member_aliases = _string_sequence(payload.get("member_aliases"))
    return PlanIndexEntry(
        number=number,
        slug=slug,
        title=_index_field(payload.get("title")),
        fingerprint=fingerprint,
        package_fingerprint=package_fingerprint,
        member_fingerprints=member_fingerprints,
        member_aliases=member_aliases,
        priority=_index_field(payload.get("priority")),
        effort=_index_field(payload.get("effort")),
        risk=_index_field(payload.get("risk")),
        category=_index_field(payload.get("category")),
        planned_at=_index_field(payload.get("planned_at")),
        status=status,
        host_blocked=bool(payload.get("host_blocked")),
        change_shape=_index_field(payload.get("change_shape") or "unknown"),
        maintenance_signals=_string_tuple(payload.get("maintenance_signals")),
        reuse_target=_index_field(payload.get("reuse_target")),
    )


def load_plan_index(plans_dir: Path) -> list[PlanIndexEntry]:
    """Load the durable plan index.

    An absent, unreadable, malformed, or structurally invalid sidecar yields no
    entries; the run then recovers what it can from the rendered index and from
    the plan files on disk rather than failing.
    """
    try:
        payload = json.loads(
            (plans_dir / PLAN_INDEX_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PLAN_INDEX_SCHEMA_VERSION
        or not isinstance(payload.get("plans"), list)
    ):
        return []
    return [
        entry
        for item in payload["plans"]
        if (entry := _entry_from_payload(item)) is not None
    ]


def reanchored_plan_rows(plans_dir: Path) -> list[PlanIndexEntry]:
    """Return every re-anchored plan recorded in the durable index.

    Reuses :func:`load_plan_index`, which treats an absent/malformed sidecar
    as empty, so this helper never raises on a missing or broken index.
    """
    return [
        entry
        for entry in load_plan_index(plans_dir)
        if entry.status.startswith(REANCHORED_STATUS_PREFIX)
    ]


def _index_text(plans_dir: Path) -> str:
    try:
        return (plans_dir / "README.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _rendered_index_entries(plans_dir: Path) -> dict[str, PlanIndexEntry]:
    """Recover index rows from the rendered README, keyed by fingerprint.

    ``README.md`` is render-only output with one standing exception: its Status
    cell is what ``render_plan``'s Finishing section tells an executor to edit,
    so a hand-edited status outranks the sidecar. Whole rows are recovered too,
    which is how a run survives a deleted sidecar or an index written before the
    sidecar existed.
    """
    entries: dict[str, PlanIndexEntry] = {}
    for line in _index_text(plans_dir).splitlines():
        if not line.startswith("|"):
            continue
        cells = [
            cell.strip().replace("\\|", "|")
            for cell in re.split(r"(?<!\\)\|", line.strip("|"))
        ]
        if len(cells) != _INDEX_COLUMNS:
            continue
        marker = _FINGERPRINT_MARKER.search(cells[0])
        number = _INDEX_ROW_NUMBER.search(cells[0])
        status = _index_field(cells[-1])
        if marker is None or number is None or not status:
            continue
        link = _INDEX_ROW_LINK.search(cells[0])
        entries[marker.group(1)] = PlanIndexEntry(
            number=int(number.group(1)),
            slug=link.group(1) if link is not None else "",
            title=_index_field(cells[1]),
            fingerprint=marker.group(1),
            package_fingerprint=marker.group(1),
            member_fingerprints=(marker.group(1),),
            member_aliases=(),
            priority=_index_field(cells[2]),
            effort=_index_field(cells[3]),
            risk="",
            category="",
            planned_at="",
            status=status,
            host_blocked=_HOST_BLOCKED_STATUS.fullmatch(status) is not None,
            change_shape="unknown",
            maintenance_signals=(),
            reuse_target="",
        )
    return entries


def _merged_index(plans_dir: Path) -> dict[int, PlanIndexEntry]:
    """Durable entries keyed by plan number, with README statuses applied."""
    rendered = _rendered_index_entries(plans_dir)
    merged: dict[int, PlanIndexEntry] = {}
    for entry in load_plan_index(plans_dir):
        override = rendered.pop(entry.fingerprint, None)
        if override is not None and override.status != entry.status:
            entry = replace(
                entry,
                status=override.status,
                host_blocked=override.host_blocked,
            )
        merged.setdefault(entry.number, entry)
    for entry in rendered.values():
        merged.setdefault(entry.number, entry)
    return merged


def _has_plan_file(plans_dir: Path, entry: PlanIndexEntry) -> bool:
    filename = entry.path
    if filename is not None and (plans_dir / filename).is_file():
        return True
    return any(plans_dir.glob(f"{entry.number:03d}-*.md"))


def _is_retryable(plans_dir: Path, entry: PlanIndexEntry) -> bool:
    """A host-blocked attempt whose number never produced a plan file."""
    return entry.host_blocked and not _has_plan_file(plans_dir, entry)


def planned_fingerprints(plans_dir: Path) -> set[str]:
    """Return package identities and aliases with durable plan status."""
    return {
        fingerprint
        for entry in _merged_index(plans_dir).values()
        if not _is_retryable(plans_dir, entry)
        for fingerprint in _entry_fingerprints(entry)
    }


def _worktrees_dir(repo: Path) -> Path:
    """Return the ``.daydream/worktrees`` base directory for a repo.

    Shared by every re-anchor worktree consumer so the base path cannot be
    broadened in one place while others drift.
    """
    return repo / ".daydream" / "worktrees"


def _iter_reanchor_worktrees(repo: Path) -> Iterable[Path]:
    """Yield existing ``*-reanchor`` worktree directories under ``.daydream/worktrees``.

    Single source of truth for which worktrees the automatic prune removes and
    the manual list reports, so the discovery contract cannot drift. Existing
    directories only; non-directory entries are skipped.
    """
    return (
        path
        for path in _worktrees_dir(repo).glob(f"*{_REANCHOR_DIR_SUFFIX}")
        if path.is_dir()
    )


def prune_stale_reanchor_worktrees(repo: Path) -> int:
    """Remove leftover ``*-reanchor`` worktrees from prior plan runs.

    Repeated head-drift runs accumulate detached worktrees under
    ``.daydream/worktrees/<run_id>-reanchor``; each new plan run prunes them so
    the directory does not grow unboundedly. Individual failures are tolerated
    so one stale worktree never blocks a plan run.
    """
    removed = 0
    for path in _iter_reanchor_worktrees(repo):
        try:
            git_ops.worktree_remove(repo, path, force=True)
        except git_ops.GitError:
            continue
        removed += 1
    return removed


# Verdicts for a named prune of a single re-anchor worktree. Distinct outcomes
# let the caller report precisely what happened instead of a bare success/fail.
PRUNE_REMOVED = "removed"
PRUNE_NOT_FOUND = "not-found"
PRUNE_NOT_REANCHOR = "not-reanchor"
PRUNE_UNSAFE_NAME = "unsafe-name"
PRUNE_GIT_FAILURE = "git-failure"


@dataclass(frozen=True)
class NamedPruneOutcome:
    """Outcome of a prune of one named re-anchor worktree.

    ``verdict`` is one of :data:`PRUNE_REMOVED`, :data:`PRUNE_NOT_FOUND`,
    :data:`PRUNE_NOT_REANCHOR`, :data:`PRUNE_UNSAFE_NAME`, or
    :data:`PRUNE_GIT_FAILURE`. ``plan_count``
    is best-effort metadata for the removal notice only, never a gate.
    """

    verdict: str
    plan_count: int = 0


def prune_named_reanchor_worktree(repo: Path, name: str) -> NamedPruneOutcome:
    """Remove the single ``-reanchor`` worktree named *name*, returning a verdict.

    The worktree lives at ``.daydream/worktrees/<name>``. The name is
    validated against the shared filesystem-safe dirname rules before any
    filesystem or git access. A ``daydream_plans`` directory's ``*.md`` count
    is captured (*before* removal) purely as blast-radius metadata for the
    removal notice; it never affects removal. Removal failures are reported as
    :data:`PRUNE_GIT_FAILURE`, never coerced to success.

    Returns :data:`PRUNE_UNSAFE_NAME` for a name that fails the
    filesystem-safe dirname rules, :data:`PRUNE_NOT_REANCHOR` for a safe name
    lacking the ``-reanchor`` suffix, :data:`PRUNE_NOT_FOUND` when the named
    directory is absent, :data:`PRUNE_REMOVED` on a successful removal, and
    :data:`PRUNE_GIT_FAILURE` when removal itself fails.
    """
    if _SAFE_DIRNAME.fullmatch(name) is None:
        return NamedPruneOutcome(PRUNE_UNSAFE_NAME)
    if not name.endswith(_REANCHOR_DIR_SUFFIX):
        return NamedPruneOutcome(PRUNE_NOT_REANCHOR)
    path = _worktrees_dir(repo) / name
    if not path.is_dir():
        return NamedPruneOutcome(PRUNE_NOT_FOUND)
    plans = path / "daydream_plans"
    if plans.is_dir():
        plan_count = sum(1 for _ in plans.glob("*.md"))
    else:
        plan_count = 0
    try:
        git_ops.worktree_remove(repo, path, force=True)
    except git_ops.GitError:
        return NamedPruneOutcome(PRUNE_GIT_FAILURE, plan_count)
    return NamedPruneOutcome(PRUNE_REMOVED, plan_count)


def list_reanchor_worktrees(repo: Path) -> list[Path]:
    """List the ``*.{_REANCHOR_DIR_SUFFIX}`` directories under ``.daydream/worktrees``.

    Returns the exact names the automatic prune would remove, so an operator
    can discover them before pruning. Existing directories only; non-directory
    entries are skipped, matching ``prune_stale_reanchor_worktrees``.
    """
    return list(_iter_reanchor_worktrees(repo))


def _highest_plan_number(
    plans_dir: Path, entries: Iterable[PlanIndexEntry]
) -> int:
    """Highest number claimed by the index or already taken on disk.

    The filesystem is consulted unconditionally: a deleted, truncated, or stale
    sidecar must never hand back a number that would overwrite a plan file.
    """
    numbers = [
        int(match.group(1))
        for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
        if (match := _NUMBERED_PLAN.match(path.name)) is not None
    ]
    numbers.extend(entry.number for entry in entries)
    return max(numbers, default=0)


def _render_index(
    rows: Sequence[str],
    *,
    plans_dir: Path,
    planned_on: date,
    non_interactive_default: bool,
    run_session_id: str | None,
) -> str:
    rejections = load_rejections(plans_dir)
    default_note = (
        "\nThe non-interactive default selected the top-N vetted defect "
        "findings by leverage.\n"
        if non_interactive_default
        else ""
    )
    rejected_lines = [
        f"- {markdown_cell(entry.get('title'))}: "
        f"{markdown_cell(entry.get('reason') or 'rejected during vetting')} "
        f"<!-- fingerprint:{fingerprint} -->"
        for fingerprint, entry in rejections.items()
    ]
    return (
        "# Implementation Plans\n\n"
        f"Generated by daydream improve on {planned_on.isoformat()}. Execute "
        "in the order below. Read each plan fully, honor its STOP conditions, "
        "and update its row when done.\n"
        + (
            f"\nDaydream run: `{run_session_id}`\n"
            if run_session_id is not None
            else ""
        )
        +
        f"{default_note}\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Status |\n"
        "|------|-------|----------|--------|--------|\n"
        + ("\n".join(rows) if rows else "| — | No plans written. | — | — | — |")
        + "\n\nStatus values: TODO | IN PROGRESS | DONE | BLOCKED "
        "(with one-line reason) | REJECTED (with one-line rationale) | "
        "REANCHORED (with one-line landing path)\n\n"
        "## Findings considered and rejected\n\n"
        + ("\n".join(rejected_lines) if rejected_lines else "- None.")
        + "\n"
    )


def _index_row(
    entry: PlanIndexEntry, *, plans_dir: Path | None = None
) -> str:
    """Render one durable entry as an execution-order row.

    When *plans_dir* is given, an entry whose plan file is not present there
    renders as the bare number: the file lives elsewhere (e.g. a re-anchored
    plan's sibling inside the worktree index), so a link would dangle.
    """
    filename = entry.path
    if (
        plans_dir is not None
        and filename is not None
        and not _has_plan_file(plans_dir, entry)
    ):
        plan_cell = f"{entry.number:03d}"
    else:
        plan_cell = (
            f"[{entry.number:03d}]({filename})"
            if filename is not None
            else f"{entry.number:03d}"
        )
    return (
        f"| {plan_cell} <!-- fingerprint:{entry.fingerprint} --> | "
        f"{markdown_cell(entry.title)} | {markdown_cell(entry.priority)} | "
        f"{markdown_cell(entry.effort)} | {markdown_cell(entry.status)} |"
    )


def _blocked_entry(
    *,
    number: int,
    fingerprint: str,
    finding: dict[str, Any],
    status: str,
    planned_at: str,
) -> PlanIndexEntry:
    """Record a blocked attempt without consulting rejected planner metadata."""
    return PlanIndexEntry(
        number=number,
        slug="",
        title=_index_field(finding.get("title") or "Selected finding"),
        fingerprint=fingerprint,
        package_fingerprint=_finding_package_fingerprint(finding) or fingerprint,
        member_fingerprints=_finding_member_fingerprints(finding, fallback=fingerprint),
        member_aliases=_finding_member_aliases(finding),
        priority=plan_priority(finding),
        effort=_index_field(finding.get("effort")),
        risk=_index_field(finding.get("risk")),
        category=_index_field(finding.get("category")),
        planned_at=planned_at,
        status=status,
        host_blocked=_HOST_BLOCKED_STATUS.fullmatch(status) is not None,
        change_shape=_index_field(finding.get("change_shape") or "unknown"),
        maintenance_signals=_string_tuple(finding.get("maintenance_signals")),
        reuse_target=_index_field(finding.get("reuse_target")),
    )


def _index_entry(
    *,
    number: int,
    slug: str,
    title: str,
    fingerprint: str,
    finding: dict[str, Any],
    planned_at: str,
    status: str,
    host_blocked: bool = False,
) -> PlanIndexEntry:
    """Build one durable index entry from a landed plan's finding fields."""
    return PlanIndexEntry(
        number=number,
        slug=slug,
        title=_index_field(title),
        fingerprint=fingerprint,
        package_fingerprint=_finding_package_fingerprint(finding) or fingerprint,
        member_fingerprints=_finding_member_fingerprints(finding, fallback=fingerprint),
        member_aliases=_finding_member_aliases(finding),
        priority=plan_priority(finding),
        effort=_index_field(finding.get("effort")),
        risk=_index_field(finding.get("risk")),
        category=_index_field(finding.get("category")),
        planned_at=planned_at,
        status=status,
        host_blocked=host_blocked,
        change_shape=_index_field(finding.get("change_shape") or "unknown"),
        maintenance_signals=_string_tuple(finding.get("maintenance_signals")),
        reuse_target=_index_field(finding.get("reuse_target")),
    )


@dataclass(frozen=True)
class PlanReservation:
    """A plan number claimed before any plan writer has produced output.

    Numbers are handed out in the order the caller reserves them, so the
    filename a finding gets never depends on which writer finishes first.
    ``number`` is ``None`` when the finding is already planned or rejected and
    therefore consumes no number.
    """

    index: int
    fingerprint: str
    number: int | None
    existing_number: int | None = None
    existing_path: str | None = None
    existing_package_fingerprint: str | None = None
    existing_member_fingerprints: tuple[str, ...] = ()
    existing_member_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanOutcome:
    """What a single :meth:`PlanWriteSession.commit` did on disk."""

    status: str
    number: int | None
    path: str | None
    title: str


class PlanWriteSession:
    """Reconcile plan-writer results into files and the durable index.

    The session owns every piece of plan-directory state: number reservation
    (including reuse of a host-blocked attempt's number), validation,
    rendering, blocked-attempt rows, and index reconciliation. Callers reserve
    numbers once in a deterministic order, then commit each result as its
    writer completes, so a finished plan is on disk while slower writers are
    still running.

    Durable state lives in ``daydream_plans/.index.json``; ``README.md`` is
    rendered from it and is never parsed back except for an operator's Status
    edit (see :func:`_rendered_index_entries`).

    ``commit`` is synchronous on purpose: called from concurrent async tasks it
    runs to completion without an await point, so the shared entry/number state
    needs no lock.
    """

    def __init__(
        self,
        plans_dir: Path,
        *,
        planned_at: str,
        non_interactive_default: bool = False,
        run_session_id: str | None = None,
    ) -> None:
        self._plans_dir = plans_dir
        self._repo = plans_dir.parent
        self._planned_at = planned_at
        self._planned_on = date.today()
        self._run_session_id = run_session_id
        plans_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = plans_dir / "README.md"
        self._sidecar_path = plans_dir / PLAN_INDEX_FILENAME
        self._non_interactive_default = (
            non_interactive_default or "non-interactive default" in _index_text(plans_dir).lower()
        )
        self._entries = _merged_index(plans_dir)
        self._rejected = load_rejections(plans_dir)
        self._next_number = _highest_plan_number(plans_dir, self._entries.values()) + 1
        self._reserved_count = 0
        self._written: list[tuple[int, dict[str, Any]]] = []
        self._skipped: list[tuple[int, dict[str, Any]]] = []
        self._failed: list[tuple[int, dict[str, Any]]] = []
        self._diagnostics: list[tuple[int, dict[str, Any]]] = []
        self._reanchored: dict[int, PlanIndexEntry] = {}
        self._reanchor_worktree: Path | None = None
        self._planned_at_errors: tuple[str, ...] = ()
        commit = _git(self._repo, "cat-file", "-e", f"{planned_at}^{{commit}}")
        if commit.returncode != 0:
            self._planned_at_errors = ("PLANNED_AT_INVALID",)
        else:
            ancestor = _git(
                self._repo, "merge-base", "--is-ancestor", planned_at, "HEAD"
            )
            if ancestor.returncode != 0:
                self._planned_at_errors = ("PLANNED_AT_NOT_ANCESTOR",)

    def reserve(
        self, findings: Sequence[dict[str, Any] | None]
    ) -> list[PlanReservation]:
        """Claim one plan number per finding, in the order given."""
        reservations: list[PlanReservation] = []
        for finding in findings:
            index = self._reserved_count
            self._reserved_count += 1
            if not isinstance(finding, dict):
                reservations.append(PlanReservation(index=index, fingerprint="", number=None))
                continue
            fingerprint = _finding_package_fingerprint(finding)
            identities = _finding_member_identities(finding)
            existing = self._existing_plan(identities)
            if existing is not None:
                reservations.append(
                    PlanReservation(
                        index=index,
                        fingerprint=fingerprint,
                        number=None,
                        existing_number=existing.number,
                        existing_path=(
                            existing.path
                            if existing.path is not None and _has_plan_file(self._plans_dir, existing)
                            else None
                        ),
                        existing_package_fingerprint=existing.package_fingerprint,
                        existing_member_fingerprints=existing.member_fingerprints,
                        existing_member_aliases=existing.member_aliases,
                    )
                )
                continue
            durable_coverage = set(self._rejected)
            for entry in self._entries.values():
                if not _is_retryable(self._plans_dir, entry):
                    durable_coverage.update(_entry_member_coverage(entry))
            if _fully_covered(identities, durable_coverage):
                reservations.append(PlanReservation(index=index, fingerprint=fingerprint, number=None))
                continue
            reserved_numbers = [
                entry.number
                for entry in self._entries.values()
                if _is_retryable(self._plans_dir, entry) and _fully_covered(identities, _entry_member_coverage(entry))
            ]
            for reserved in reserved_numbers:
                del self._entries[reserved]
            if reserved_numbers:
                number = min(reserved_numbers)
            else:
                number = self._next_number
                self._next_number += 1
            reservations.append(PlanReservation(index=index, fingerprint=fingerprint, number=number))
        return reservations

    def _existing_plan(self, identities: tuple[frozenset[str], ...]) -> PlanIndexEntry | None:
        """Resolve one durable plan only when it covers every current member."""
        candidates = [
            entry
            for entry in self._entries.values()
            if not _is_retryable(self._plans_dir, entry) and _fully_covered(identities, _entry_member_coverage(entry))
        ]
        return candidates[0] if len(candidates) == 1 else None

    def commit(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
    ) -> PlanOutcome:
        """Land one plan-writer result, writing its file when it is complete."""
        safe = _redact_model_value(selection)
        if not isinstance(safe, dict):
            return PlanOutcome("ignored", None, None, "")
        finding = safe.get("finding")
        if not isinstance(finding, dict):
            return PlanOutcome("ignored", None, None, "")
        title = str(finding.get("title") or "Selected finding")
        if reservation.number is None:
            skipped: dict[str, Any] = {"finding": finding}
            if reservation.existing_number is not None:
                skipped["number"] = reservation.existing_number
                if reservation.existing_path is not None:
                    skipped["path"] = reservation.existing_path
                if reservation.existing_package_fingerprint is not None:
                    skipped["package_fingerprint"] = reservation.existing_package_fingerprint
                    skipped["member_fingerprints"] = list(reservation.existing_member_fingerprints)
                    skipped["member_aliases"] = list(reservation.existing_member_aliases)
            self._skipped.append((reservation.index, skipped))
            attempt = self._attempt_of(safe)
            if attempt is not None:
                self._diagnostics.append(
                    (
                        reservation.index,
                        _attempt_diagnostic(
                            finding=finding,
                            attempt=attempt,
                            received=_plan_payload(safe),
                            disposition="skipped",
                            stage="reconciliation",
                            errors=("ALREADY_PLANNED_OR_REJECTED",),
                        ),
                    )
                )
            return PlanOutcome(
                "skipped",
                reservation.existing_number,
                reservation.existing_path,
                title,
            )
        return self._land(reservation, safe)

    def finish(self) -> dict[str, list[dict[str, Any]]]:
        """Reconcile the index and return what this session landed."""
        self._write_index()
        self._release_reanchor_worktree()
        return {
            "written": _by_reservation(self._written),
            "skipped": _by_reservation(self._skipped),
            "failed": _by_reservation(self._failed),
            "diagnostics": _by_reservation(self._diagnostics),
        }

    def _release_reanchor_worktree(self) -> None:
        """Best-effort release of the re-anchor worktree's git lock.

        A failed unlock must never surface or fail the plan run, so the unlock
        is swallowed via :class:`~root.git_ops.GitError`. Clear the reference
        regardless so a later release is a no-op.
        """
        if self._reanchor_worktree is None:
            return
        try:
            git_ops.worktree_unlock(self._repo, self._reanchor_worktree)
        except git_ops.GitError:
            pass
        self._reanchor_worktree = None

    @staticmethod
    def _attempt_of(selection: dict[str, Any]) -> dict[str, Any] | None:
        attempt = selection.get("_attempt")
        return attempt if isinstance(attempt, dict) else None

    def _block(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
        *,
        number: int,
        finding: dict[str, Any],
        status: str,
        stage: str,
        errors: Sequence[str],
        received: Any,
    ) -> PlanOutcome:
        self._entries[number] = _blocked_entry(
            number=number,
            fingerprint=reservation.fingerprint,
            finding=finding,
            status=status,
            planned_at=self._planned_at,
        )
        self._failed.append((reservation.index, finding))
        self._diagnostics.append(
            (
                reservation.index,
                _attempt_diagnostic(
                    finding=finding,
                    attempt=self._attempt_of(selection),
                    received=received,
                    disposition="blocked",
                    stage=stage,
                    errors=errors,
                ),
            )
        )
        self._write_index()
        return PlanOutcome(
            "blocked",
            number,
            None,
            str(finding.get("title") or "Selected finding"),
        )

    def _record_written(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
        *,
        number: int,
        title: str,
        finding: dict[str, Any],
        attempt: dict[str, Any] | None,
        plan_result: dict[str, Any],
        path: str,
        artifact: dict[str, Any],
    ) -> PlanOutcome:
        """Record a successfully landed plan and return its outcome."""
        self._written.append(
            (
                reservation.index,
                {**selection, "number": number, "path": path},
            )
        )
        self._diagnostics.append(
            (
                reservation.index,
                _attempt_diagnostic(
                    finding=finding,
                    attempt=attempt,
                    received=plan_result,
                    disposition="success",
                    stage="success",
                    artifact=artifact,
                ),
            )
        )
        return PlanOutcome("written", number, path, title)

    def _land(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
    ) -> PlanOutcome:
        finding = selection["finding"]
        assert reservation.number is not None  # commit() gates on the number
        number = reservation.number
        title = str(finding.get("title") or "Selected finding")
        attempt = self._attempt_of(selection)
        slug = plan_slug(selection.get("title"))
        if selection.get("error"):
            raw_errors = attempt.get("errors") if attempt is not None else None
            if not isinstance(raw_errors, (list, tuple)) and attempt is not None:
                legacy_code = attempt.get("transport_error_code")
                raw_errors = (legacy_code,) if isinstance(legacy_code, str) else ()
            error_entries = tuple(
                entry
                for entry in (
                    raw_errors if isinstance(raw_errors, (list, tuple)) else ()
                )
                if isinstance(entry, str)
                and re.fullmatch(
                    r"[A-Z][A-Z0-9_]{1,63}", entry.partition("@")[0]
                )
            )
            if not error_entries:
                error_entries = ("UNKNOWN",)
            error_codes = tuple(
                entry.partition("@")[0] for entry in error_entries
            )
            if attempt is not None and attempt.get("validation"):
                status = (
                    "BLOCKED (PLAN_VALIDATION_FAILED: "
                    f"{','.join(error_codes)})"
                )
                stage = _validation_stage(error_entries)
            else:
                status = f"BLOCKED (PLAN_WRITER_FAILED: {error_codes[0]})"
                stage = "transport"
            return self._block(
                reservation,
                selection,
                number=number,
                finding=finding,
                status=status,
                stage=stage,
                errors=error_entries,
                received=(
                    attempt.get("received_result")
                    if attempt is not None
                    else None
                ),
            )

        plan_result = _plan_payload(selection)
        if self._planned_at_errors:
            return self._block(
                reservation,
                selection,
                number=number,
                finding=finding,
                status=(
                    "BLOCKED (PLAN_VALIDATION_FAILED: "
                    f"{','.join(self._planned_at_errors)})"
                ),
                stage=_validation_stage(self._planned_at_errors),
                errors=self._planned_at_errors,
                received=plan_result,
            )

        try:
            current_head = git_ops.head_sha(self._repo)
        except git_ops.GitError:
            current_head = None
        if current_head != self._planned_at:
            return self._reanchor_and_write(
                reservation,
                selection,
                plan_result,
                number=number,
                title=title,
                slug=slug,
            )

        filename = f"{number:03d}-{slug}.md"
        try:
            text = render_plan(
                finding,
                plan=plan_result,
                planned_at=self._planned_at,
                number=number,
                planned_on=self._planned_on,
                run_session_id=self._run_session_id,
            )
        except Exception:  # noqa: BLE001 - persist a safe render disposition
            return self._block(
                reservation,
                selection,
                number=number,
                finding=finding,
                status="BLOCKED (PLAN_VALIDATION_FAILED: RENDER_FAILED)",
                stage="render",
                errors=("RENDER_FAILED",),
                received=plan_result,
            )
        (self._plans_dir / filename).write_text(text, encoding="utf-8")
        self._entries[number] = _index_entry(
            number=number,
            slug=slug,
            title=selection.get("title") or title,
            fingerprint=reservation.fingerprint,
            finding=finding,
            planned_at=self._planned_at,
            status="TODO",
        )
        outcome = self._record_written(
            reservation,
            selection,
            number=number,
            title=title,
            finding=finding,
            attempt=attempt,
            plan_result=plan_result,
            path=filename,
            artifact={"path": filename, "status": "TODO"},
        )
        self._write_index()
        return outcome

    def _reanchor_and_write(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
        plan_result: dict[str, Any],
        *,
        number: int,
        title: str,
        slug: str,
    ) -> PlanOutcome:
        """Write a finished plan into a fresh detached worktree at current HEAD.

        The plan's content is complete; only the ``planned_at`` anchor is stale
        because HEAD advanced past the commit captured at session start. The
        plan lands in a fresh detached worktree at the current HEAD, re-anchored
        to it, plus a durable copy in the main index's ``daydream_plans/`` so it
        survives the next run's worktree pruning, and returns ``written`` with
        the full worktree path it landed at — a stale anchor alone must not
        discard a valid plan. Any sub-failure falls back to :meth:`_block` with
        ``PLAN_REANCHOR_FAILED``; nothing is silently dropped.
        """
        finding = selection["finding"]
        attempt = self._attempt_of(selection)
        filename = f"{number:03d}-{slug}.md"

        def _reanchor_failed() -> PlanOutcome:
            return self._block(
                reservation,
                selection,
                number=number,
                finding=finding,
                status="BLOCKED (PLAN_WRITER_FAILED: PLAN_REANCHOR_FAILED)",
                stage="transport",
                errors=("PLAN_REANCHOR_FAILED",),
                received=plan_result,
            )

        try:
            new_head = git_ops.head_sha(self._repo)
        except git_ops.GitError:
            return _reanchor_failed()
        try:
            worktree = self._reanchor_worktree
            if worktree is None:
                run_id = self._run_session_id or f"run-{self._planned_at[:12]}"
                if _SAFE_DIRNAME.fullmatch(run_id) is None:
                    run_id = f"run-{self._planned_at[:12]}"
                worktree = _worktrees_dir(self._repo) / f"{run_id}{_REANCHOR_DIR_SUFFIX}"
                git_ops.worktree_add(
                    self._repo, worktree, new_head, detach=True
                )
                # Arm the git worktree lock exactly once, at creation, so a
                # concurrent run's start-of-run prune cannot destroy the live
                # worktree while the plan is mid-write. Released best-effort on
                # finish()/failure; a lock failure blocks the plan (git refuses
                # to remove a locked worktree with a single --force).
                git_ops.worktree_lock(self._repo, worktree, reason=run_id)
                self._reanchor_worktree = worktree
            text = render_plan(
                finding,
                plan=plan_result,
                planned_at=new_head,
                number=number,
                planned_on=self._planned_on,
                run_session_id=self._run_session_id,
            )
            plans_dir = worktree / "daydream_plans"
            plans_dir.mkdir(parents=True, exist_ok=True)
            (plans_dir / filename).write_text(text, encoding="utf-8")
            # The plan text is worktree-independent: land the durable copy in the
            # main index too, so it survives the next run's worktree pruning.
            (self._plans_dir / filename).write_text(text, encoding="utf-8")
            self._reanchored[number] = _index_entry(
                number=number,
                slug=slug,
                title=selection.get("title") or title,
                fingerprint=reservation.fingerprint,
                finding=finding,
                planned_at=new_head,
                status="TODO",
            )
            entries = dict(self._entries)
            entries.update(self._reanchored)
            self._write_index_files(
                plans_dir,
                [entries[index] for index in sorted(entries)],
                check_links=True,
            )
            # The re-anchor worktree is pruned at the start of the next plan run,
            # so the durable status must point at the surviving copy in the main
            # index rather than a path that will no longer exist.
            landed_rel = (
                (self._plans_dir / filename).relative_to(self._repo).as_posix()
            )
            self._entries[number] = _index_entry(
                number=number,
                slug=slug,
                title=selection.get("title") or title,
                fingerprint=reservation.fingerprint,
                finding=finding,
                planned_at=new_head,
                status=f"{REANCHORED_STATUS_PREFIX} (landed at {landed_rel})",
            )
            # Index the durable main copy immediately so an interrupted run can
            # never leave a plan file without a main-index entry (which the next
            # run would silently re-plan and orphan). Mirror _land/finish().
            self._write_index()
        except Exception:  # noqa: BLE001 - persist a safe re-anchor disposition
            return _reanchor_failed()

        landed_path = (worktree / "daydream_plans" / filename).as_posix()
        return self._record_written(
            reservation,
            selection,
            number=number,
            title=title,
            finding=finding,
            attempt=attempt,
            plan_result=plan_result,
            path=landed_path,
            artifact={
                "path": landed_path,
                "status": "TODO",
                "reanchored": True,
                "planned_at": new_head,
            },
        )

    def _write_index(self) -> None:
        """Rewrite the sidecar and its rendered index from the entries so far.

        The sidecar lands first: it is the durable record, and rewriting both on
        every landing leaves an interrupted run with state that matches the plan
        files already on disk.
        """
        entries = [self._entries[number] for number in sorted(self._entries)]
        self._write_index_files(self._plans_dir, entries)

    def _write_index_files(
        self,
        plans_dir: Path,
        entries: Sequence[PlanIndexEntry],
        *,
        check_links: bool = False,
    ) -> None:
        """Write the sidecar and its rendered index for *entries* into *plans_dir*."""
        (plans_dir / PLAN_INDEX_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": PLAN_INDEX_SCHEMA_VERSION,
                    "artifact_type": "daydream.plan-index",
                    "plans": [_entry_payload(entry) for entry in entries],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (plans_dir / "README.md").write_text(
            _render_index(
                [
                    _index_row(
                        entry,
                        plans_dir=plans_dir if check_links else None,
                    )
                    for entry in entries
                ],
                plans_dir=plans_dir,
                planned_on=self._planned_on,
                non_interactive_default=self._non_interactive_default,
                run_session_id=self._run_session_id,
            ),
            encoding="utf-8",
        )


def _plan_payload(selection: dict[str, Any]) -> dict[str, Any]:
    """Return the authored plan fields, without host bookkeeping keys."""
    return {
        key: value
        for key, value in selection.items()
        if key not in {"finding", "error"} and not key.startswith("_")
    }


def _by_reservation(
    entries: Sequence[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [entry for _, entry in sorted(entries, key=lambda item: item[0])]
