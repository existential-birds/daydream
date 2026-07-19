"""Persistent plan-directory state for the improve advisor flow."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

REJECTIONS_SCHEMA_VERSION = 1
_FINGERPRINT_MARKER = re.compile(
    r"<!--\s*fingerprint:([^\s>]+)\s*-->"
)
_NUMBERED_PLAN = re.compile(r"^(\d{3})-[a-z0-9-]+\.md$")
_SECTION = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
_BODY_SECTIONS = (
    "Why this matters",
    "Current state",
    "Scope",
    "Steps",
    "Test plan",
    "Done criteria",
    "STOP conditions",
)
_REQUIRED_PLAN_SECTIONS = (
    "Status",
    "Why this matters",
    "Current state",
    "Commands you will need",
    "Scope",
    "Steps",
    "Test plan",
    "Done criteria",
    "STOP conditions",
)


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
    rejected = list(load_rejections(plans_dir).values())
    rejected.extend(dict(entry) for entry in entries)
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


def _markdown_cell(value: Any) -> str:
    """Render a value safely inside a Markdown table cell."""
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


def _section_content(markdown: str) -> dict[str, str]:
    matches = list(_SECTION.finditer(markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        sections[match.group(1)] = markdown[match.end() : end].strip()
    return sections


def missing_required_sections(markdown: str) -> tuple[str, ...]:
    """Return required plan sections that are absent or empty."""
    sections = _section_content(markdown)
    return tuple(
        section
        for section in _REQUIRED_PLAN_SECTIONS
        if not sections.get(section, "").strip()
    )


def split_plan_at_status(markdown: str) -> tuple[str, str]:
    """Split a plan into ``(head, body)`` at the ``## Status`` heading.

    ``head`` is everything before the ``## Status`` line: the title and the
    host-stamped drift-check / Executor-instructions blockquote that
    :func:`render_plan` emits. ``body`` is the ``## Status`` line through end
    of document. ``_review_plan`` re-stamps ``head`` from the original plan on
    a round-trip (the blockquote is not a ``##`` section, so
    :func:`missing_required_sections` cannot detect its loss) and takes
    ``body`` from the tightened output. Returns ``("", markdown)`` when no
    ``## Status`` heading is present.
    """
    match = re.search(r"^## Status\b", markdown, re.MULTILINE)
    if match is None:
        return "", markdown
    return markdown[: match.start()], markdown[match.start():]


def resolve_review_plan_path(repo: Path, requested: str) -> Path:
    """Resolve a review target confined to ``repo/daydream_plans``."""
    plans_dir = (repo / "daydream_plans").resolve()
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(plans_dir):
        raise ValueError(
            "review-plan only accepts files under "
            f"{plans_dir}; received {requested!r}"
        )
    if not candidate.is_file():
        raise ValueError(f"review-plan file does not exist: {candidate}")
    return candidate


def _scope_paths(finding: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    path = finding.get("path")
    if isinstance(path, str) and path:
        paths.append(path)
    evidence = finding.get("evidence")
    if isinstance(evidence, list):
        for entry in evidence:
            if not isinstance(entry, str):
                continue
            evidence_path = entry.strip("`").split(":", 1)[0]
            if evidence_path and evidence_path not in paths:
                paths.append(evidence_path)
    return paths


def _default_section(
    section: str,
    finding: dict[str, Any],
    paths: list[str],
) -> str:
    if section == "Why this matters":
        return str(finding.get("body") or "Address the selected vetted finding.")
    if section == "Current state":
        evidence = finding.get("evidence")
        entries = evidence if isinstance(evidence, list) else []
        return "\n".join(f"- `{entry}`" for entry in entries) or "- See the cited finding."
    if section == "Scope":
        rendered = "\n".join(f"- `{path}`" for path in paths)
        return (
            f"**In scope**\n\n"
            f"{rendered or '- The cited implementation surface.'}\n\n"
            "**Out of scope**\n\n"
            "- All unrelated files and behavior."
        )
    if section == "Steps":
        return (
            "### Step 1: Implement the selected finding\n\n"
            "Follow the vetted fix sketch and preserve repository conventions."
        )
    if section == "Test plan":
        return "- Add regression coverage for the selected finding.\n- Run the verified repository commands."
    if section == "Done criteria":
        return (
            "- [ ] The selected finding is resolved.\n"
            "- [ ] Verified repository commands pass.\n"
            "- [ ] No out-of-scope files are modified."
        )
    return (
        "- The live code does not match the Current state section.\n"
        "- A verification command fails twice after a reasonable fix attempt.\n"
        "- The change requires an out-of-scope file.\n"
        "- A load-bearing assumption in this plan is false."
    )


def _commands_table(commands: dict[str, str]) -> str:
    lines = [
        "| Purpose | Command | Expected on success |",
        "|---------|---------|---------------------|",
    ]
    if commands:
        lines.extend(
            f"| {_markdown_cell(purpose)} | `{command}` | exit 0 |"
            for purpose, command in commands.items()
        )
    else:
        lines.append("| Verification | No command established during recon | — |")
    return "\n".join(lines)


def render_plan(
    finding: dict[str, Any],
    *,
    body_markdown: str,
    planned_at: str,
    number: int,
    slug: str,
    commands: dict[str, str],
    title: str | None = None,
    priority: str | None = None,
    depends_on: Sequence[str] = (),
) -> str:
    """Render one self-contained, host-stamped implementation plan."""
    sections = _section_content(body_markdown)
    paths = _scope_paths(finding)
    diff_paths = " ".join(shlex.quote(path) for path in paths) or "."
    plan_title = title or str(finding.get("title") or slug)
    plan_priority = priority or {
        "HIGH": "P1",
        "MED": "P2",
        "LOW": "P3",
    }.get(str(finding.get("impact")), "P2")
    dependencies = ", ".join(depends_on) or "none"
    direction_note = (
        "\n\n> This direction finding is framed as a design/spike plan; "
        "do not expand it into a build-everything implementation."
        if finding.get("category") == "direction"
        else ""
    )

    body: list[str] = []
    for section in ("Why this matters", "Current state"):
        content = sections.get(section) or _default_section(section, finding, paths)
        if section == "Why this matters":
            content += direction_note
        body.append(f"## {section}\n\n{content}")
    body.append("## Commands you will need\n\n" + _commands_table(commands))
    if optional_content := sections.get("Suggested executor toolkit"):
        body.append(f"## Suggested executor toolkit\n\n{optional_content}")
    for section in ("Scope", "Git workflow", "Steps", "Test plan", "Done criteria", "STOP conditions"):
        section_content = sections.get(section)
        if section_content is None and section in _BODY_SECTIONS:
            section_content = _default_section(section, finding, paths)
        if section_content:
            body.append(f"## {section}\n\n{section_content}")
    if optional_content := sections.get("Maintenance notes"):
        body.append(f"## Maintenance notes\n\n{optional_content}")

    return (
        f"# Plan {number:03d}: {plan_title}\n\n"
        "> **Executor instructions**: Follow this plan step by step. Run every\n"
        "> verification command and confirm the expected result before moving to the\n"
        "> next step. If anything in the \"STOP conditions\" section occurs, stop and\n"
        "> report — do not improvise. When done, update the status row for this plan\n"
        "> in `daydream_plans/README.md` unless a reviewer maintains the index.\n"
        ">\n"
        f"> **Drift check (run first)**: `git diff --stat {planned_at}..HEAD -- {diff_paths}`\n"
        "> If any in-scope file changed since this plan was written, compare the\n"
        "> Current state excerpts against live code. A mismatch is a STOP condition.\n\n"
        "## Status\n\n"
        f"- **Priority**: {plan_priority}\n"
        f"- **Effort**: {finding.get('effort', '—')}\n"
        f"- **Risk**: {finding.get('risk', '—')}\n"
        f"- **Depends on**: {dependencies}\n"
        f"- **Category**: {finding.get('category', '—')}\n"
        f"- **Planned at**: commit `{planned_at[:7]}`, {date.today().isoformat()}\n\n"
        + "\n\n".join(body)
        + "\n"
    )


def _existing_index_rows(index_text: str) -> list[str]:
    rows: list[str] = []
    for line in index_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6 or cells[0] in {"Plan", "------"}:
            continue
        if set(cells[0]) == {"-"}:
            continue
        rows.append(line)
    return rows


def planned_fingerprints(plans_dir: Path) -> set[str]:
    """Return fingerprints already represented by a plan index row."""
    index_path = plans_dir / "README.md"
    if not index_path.is_file():
        return set()
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return set()
    rows = _existing_index_rows(index_text)
    return {
        match.group(1)
        for row in rows
        if (match := _FINGERPRINT_MARKER.search(row)) is not None
    }


def _highest_plan_number(plans_dir: Path, rows: Sequence[str]) -> int:
    numbers = [
        int(match.group(1))
        for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
        if (match := _NUMBERED_PLAN.match(path.name)) is not None
    ]
    for row in rows:
        match = re.search(r"\b(\d{3})\b", row)
        if match is not None:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0)


def _render_index(
    rows: Sequence[str],
    *,
    plans_dir: Path,
    non_interactive_default: bool,
    dependency_notes: Sequence[str],
) -> str:
    rejections = load_rejections(plans_dir)
    default_note = (
        "\nThe non-interactive default selected the top-N vetted defect "
        "findings by leverage.\n"
        if non_interactive_default
        else ""
    )
    rejected_lines = [
        f"- {_markdown_cell(entry.get('title'))}: "
        f"{_markdown_cell(entry.get('reason') or 'rejected during vetting')} "
        f"<!-- fingerprint:{fingerprint} -->"
        for fingerprint, entry in rejections.items()
    ]
    return (
        "# Implementation Plans\n\n"
        f"Generated by daydream improve on {date.today().isoformat()}. Execute "
        "in the order below unless dependencies say otherwise. Read each plan "
        "fully, honor its STOP conditions, and update its row when done.\n"
        f"{default_note}\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Depends on | Status |\n"
        "|------|-------|----------|--------|------------|--------|\n"
        + ("\n".join(rows) if rows else "| — | No plans written. | — | — | — | — |")
        + "\n\nStatus values: TODO | IN PROGRESS | DONE | BLOCKED "
        "(with one-line reason) | REJECTED (with one-line rationale)\n\n"
        "## Dependency notes\n\n"
        + (
            "\n".join(f"- {note}" for note in dependency_notes)
            if dependency_notes
            else "- None recorded."
        )
        + "\n\n## Findings considered and rejected\n\n"
        + ("\n".join(rejected_lines) if rejected_lines else "- None.")
        + "\n"
    )


def write_plans(
    plans_dir: Path,
    selections: Sequence[dict[str, Any]],
    *,
    planned_at: str,
    commands: dict[str, str] | None = None,
    non_interactive_default: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Reconcile selected plan-writer results into files and the durable index."""
    plans_dir.mkdir(parents=True, exist_ok=True)
    index_path = plans_dir / "README.md"
    index_text = (
        index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
    )
    rows = _existing_index_rows(index_text)
    fingerprints = set(_FINGERPRINT_MARKER.findall(index_text))
    rejected = load_rejections(plans_dir)
    next_number = _highest_plan_number(plans_dir, rows) + 1
    result: dict[str, list[dict[str, Any]]] = {
        "written": [],
        "skipped": [],
        "failed": [],
    }
    dependency_notes: list[str] = []

    for selection in selections:
        finding = selection.get("finding")
        if not isinstance(finding, dict):
            continue
        fingerprint = str(finding.get("fingerprint") or "")
        if fingerprint in fingerprints or fingerprint in rejected:
            result["skipped"].append(finding)
            continue

        number = next_number
        next_number += 1
        slug = str(selection.get("slug") or "plan")
        title = str(selection.get("title") or finding.get("title") or slug)
        priority = str(selection.get("priority") or "P2")
        depends_on_raw = selection.get("depends_on")
        depends_on = (
            [str(item) for item in depends_on_raw]
            if isinstance(depends_on_raw, list)
            else []
        )
        marker = f"<!-- fingerprint:{fingerprint} -->"
        if error := selection.get("error"):
            rows.append(
                f"| {number:03d} {marker} | {_markdown_cell(title)} | "
                f"{_markdown_cell(priority)} | "
                f"{_markdown_cell(finding.get('effort'))} | "
                f"{_markdown_cell(', '.join(depends_on))} | "
                f"BLOCKED (plan-writing failed: {_markdown_cell(error)}) |"
            )
            result["failed"].append(finding)
            fingerprints.add(fingerprint)
            continue

        filename = f"{number:03d}-{slug}.md"
        text = render_plan(
            finding,
            body_markdown=str(selection.get("markdown") or ""),
            planned_at=planned_at,
            number=number,
            slug=slug,
            commands=commands or {},
            title=title,
            priority=priority,
            depends_on=depends_on,
        )
        (plans_dir / filename).write_text(text, encoding="utf-8")
        rows.append(
            f"| [{number:03d}]({filename}) {marker} | "
            f"{_markdown_cell(title)} | {_markdown_cell(priority)} | "
            f"{_markdown_cell(finding.get('effort'))} | "
            f"{_markdown_cell(', '.join(depends_on))} | TODO |"
        )
        if depends_on:
            dependency_notes.append(
                f"{number:03d} depends on {', '.join(depends_on)}."
            )
        result["written"].append({**selection, "number": number, "path": filename})
        fingerprints.add(fingerprint)

    index_path.write_text(
        _render_index(
            rows,
            plans_dir=plans_dir,
            non_interactive_default=(
                non_interactive_default
                or "non-interactive default" in index_text.lower()
            ),
            dependency_notes=dependency_notes,
        ),
        encoding="utf-8",
    )
    return result
