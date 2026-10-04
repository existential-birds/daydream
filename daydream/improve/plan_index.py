"""Durable plan identities, rejection history, and operator-edited index recovery."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from daydream.improve.prioritize import member_alias, plan_priority
from daydream.improve.render import (
    markdown_cell,
)
from daydream.redaction import redact_value
from daydream.trajectory import redact_text

REJECTIONS_SCHEMA_VERSION = 1
PLAN_INDEX_SCHEMA_VERSION = 1
PLAN_INDEX_FILENAME = ".index.json"
_FINGERPRINT_MARKER = re.compile(
    r"<!--\s*fingerprint:([^\s>]+)\s*-->"
)
_NUMBERED_PLAN = re.compile(r"^(\d{3})-[a-z0-9-]+\.md$")
_HOST_BLOCKED_STATUS = re.compile(
    r"^BLOCKED \(PLAN_(?:WRITER|VALIDATION)_FAILED: [^()\r\n]+\)$"
)


# Plan | Title | Priority | Effort | Status
_INDEX_COLUMNS = 5
_INDEX_ROW_NUMBER = re.compile(r"\b(\d{3})\b")


# The slug class admits no separator or dot, so a recovered link can never name
# anything but a sibling plan file.
_INDEX_ROW_LINK = re.compile(r"\[\d{3}\]\(\d{3}-([a-z0-9-]+)\.md\)")
REANCHORED_STATUS_PREFIX = "REANCHORED"
_REANCHORED_LANDED = re.compile(rf"^{re.escape(REANCHORED_STATUS_PREFIX)} \(landed at (.+)\)$")


def load_rejections(plans_dir: Path) -> dict[str, dict[str, Any]]:
    """Load durable rejections; absent, unreadable, or invalid envelopes count as empty."""
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
        redact_value(entry)
        for entry in load_rejections(plans_dir).values()
    ]
    rejected.extend(
        redact_value(dict(entry))
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
    planned_at: str
    status: str
    host_blocked: bool

    @property
    def path(self) -> str | None:
        """The plan file this entry names, or ``None`` when none was written."""
        return f"{self.number:03d}-{self.slug}.md" if self.slug else None

    @property
    def landing_path(self) -> str | None:
        """Parse the recorded landing path; never synthesize one from a status without it."""
        match = _REANCHORED_LANDED.match(self.status)
        return match.group(1) if match else None


def _index_field(value: Any) -> str:
    """Normalize a model- or operator-supplied index field for durable storage."""
    return redact_text(str(value or "").strip())


def _string_sequence(value: Any) -> tuple[str, ...]:
    """Normalize a string sequence while preserving meaningful multiplicity."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _string_tuple(value: Any) -> tuple[str, ...]:
    """Normalize string values and retain only their first occurrence."""
    return tuple(dict.fromkeys(_string_sequence(value)))


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
    aliases = _string_sequence(finding.get("member_aliases"))
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
        planned_at=_index_field(payload.get("planned_at")),
        status=status,
        host_blocked=bool(payload.get("host_blocked")),
    )


def load_plan_index(plans_dir: Path) -> list[PlanIndexEntry]:
    """Load valid sidecar entries; unreadable or invalid envelopes leave recovery to README and plan files."""
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
    """Read re-anchored entries, treating a missing or malformed sidecar as empty."""
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
    """Recover README rows by fingerprint, including operator-edited statuses.

    Status edits override the sidecar because executors are instructed to update
    that cell. Whole rows also recover a deleted or pre-sidecar index.
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
            planned_at="",
            status=status,
            host_blocked=_HOST_BLOCKED_STATUS.fullmatch(status) is not None,
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


def _highest_plan_number(
    plans_dir: Path, entries: Iterable[PlanIndexEntry]
) -> int:
    """Include disk filenames even with a missing or stale index, preventing overwrites."""
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
    """Render an index row, suppressing dangling links when plans_dir is supplied."""
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
    return _index_entry(
        number=number,
        slug="",
        title=finding.get("title") or "Selected finding",
        fingerprint=fingerprint,
        finding=finding,
        planned_at=planned_at,
        status=status,
        host_blocked=_HOST_BLOCKED_STATUS.fullmatch(status) is not None,
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
        member_aliases=_string_sequence(finding.get("member_aliases")),
        priority=plan_priority(finding),
        effort=_index_field(finding.get("effort")),
        planned_at=planned_at,
        status=status,
        host_blocked=host_blocked,
    )
