"""Render canonical merged finding items into the human-readable markdown report.

`render_report` is a pure function: it takes the canonical item list (the single
source of truth produced by the cross-stack merge) and produces the same
``review-output.md`` layout the merge agent used to emit as prose. Mirrors the
mandatory report format defined in ``daydream/deep/prompts.py`` (the
``## Structural Review`` / ``## Issues`` / ``## Cross-Stack Issues`` sections,
the ``[cross-stack]`` title prefix, and the unbolded ``N. [FILE:LINE] DESC``
head-line rule). No LLM, no I/O.

Exports:
    render_report: list[dict] -> str
    render_held_section: list[dict] -> str
    insert_diagrams_section: (report text, diagram blocks) -> str
"""

from __future__ import annotations

from typing import Any

# Heading of the grounded-diagram section (issue #1113). The section is written
# by ``render_report`` on the merge write and re-applied textually by the
# diagram step, which runs after two other writers have already produced
# ``review-output.md``; ``insert_diagrams_section`` is what makes that second
# application idempotent.
_DIAGRAMS_HEADING = "## Diagrams"
_REVIEW_HEADING = "# Review"


def _finding_line(item: dict[str, Any], *, prefix: str = "") -> str:
    """Format one finding as the unbolded ``N. [prefix][FILE:LINE] DESCRIPTION`` line.

    The numbered head line is plain text — never wrapped in bold markers — per
    the report-format rules in ``deep/prompts.py``.
    """
    return f"{item['id']}. {prefix}[{item['file']}:{item['line']}] {item['description']}"


def _is_heading(line: str) -> bool:
    """True for a top-level (``# ``) or section-level (``## ``) markdown heading.

    The rendered diagram blocks can never produce one: their headings are HTML
    (``<h3>``), their table rows start with ``|``, and every mermaid line is
    indented or is a bare keyword -- the label sanitizer drops ``#`` outright.
    So this is a safe section terminator to scan for.
    """
    return line.startswith("# ") or line.startswith("## ")


def _remove_diagrams_section(lines: list[str]) -> list[str]:
    """Drop an existing ``## Diagrams`` section, leaving one blank separator.

    Consumes the heading and everything up to the next markdown heading (or end
    of file), then normalizes the blank lines around the hole so the result is
    byte-identical to a report that never had the section. That exact-inverse
    property is what makes ``insert_diagrams_section`` idempotent.
    """
    out: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        if lines[index].strip() != _DIAGRAMS_HEADING:
            out.append(lines[index])
            index += 1
            continue
        index += 1
        while index < total and not _is_heading(lines[index]):
            index += 1
        while out and out[-1] == "":
            out.pop()
        # One blank separator, but only between two surviving neighbours -- a
        # leading or trailing blank line would make the removal a non-inverse
        # of the insertion and break idempotence.
        if out and index < total:
            out.append("")
    return out


def insert_diagrams_section(report_text: str, blocks: str) -> str:
    """Insert (or replace) the ``## Diagrams`` section in a rendered report.

    Pure and idempotent: applying it twice with the same ``blocks`` yields the
    same bytes, because an existing section is removed before the new one is
    inserted. The section goes directly after the ``# Review`` heading, so it
    reads above the findings; every other section, including the ``## Coverage``
    block that ``_append_coverage_section`` appends later, is left untouched.

    Args:
        report_text: The rendered report. Its trailing-newline convention is
            preserved.
        blocks: The rendered diagram blocks. Empty or whitespace-only removes
            the section instead of writing an empty one.

    Returns:
        The report text with the section inserted, replaced, or removed.
    """
    trailing_newline = report_text.endswith("\n")
    lines = _remove_diagrams_section(report_text.split("\n"))
    body = blocks.strip("\n")
    if body.strip():
        section = [_DIAGRAMS_HEADING, *body.split("\n")]
        anchor = next((i for i, line in enumerate(lines) if line.strip() == _REVIEW_HEADING), None)
        if anchor is None:
            lines = [*section, "", *lines]
        else:
            lines = [*lines[: anchor + 1], "", *section, *lines[anchor + 1 :]]
    text = "\n".join(lines).rstrip("\n")
    return f"{text}\n" if trailing_newline else text


def render_report(items: list[dict[str, Any]], *, diagram_blocks: str | None = None) -> str:
    """Render canonical items into the deep-review markdown report.

    Groups items by ``lens`` and emits, in order: ``## Structural Review``
    (only when structural items exist), ``## Issues`` (per-stack lens),
    ``## Cross-Stack Issues`` (cross-stack lens, each title prefixed with the
    literal ``[cross-stack]``), and ``## Wonder Findings`` (wonder lens). A
    section is omitted entirely when it has no items. Each finding line is
    ``N. [FILE:LINE] DESCRIPTION``, unbolded, where ``N`` is the item's
    canonical ``id``.

    When ``diagram_blocks`` is given, a ``## Diagrams`` section carrying the
    rendered grounded-diagram blocks is inserted directly after ``# Review``,
    via the same :func:`insert_diagrams_section` the diagram step re-applies to
    the on-disk report -- one code path, so the two can never drift.

    Args:
        items: Canonical merged finding items, each carrying ``id``, ``lens``,
            ``file``, ``line``, and ``description``.
        diagram_blocks: Rendered diagram blocks, or ``None``/``""`` for no
            diagram section. Keyword-only with a default so no existing call
            site changes.

    Returns:
        The rendered markdown report as a string.
    """
    sections: list[str] = ["# Review"]

    # lens -> (section title, line prefix). One arm per lens; a section is
    # emitted only when it has items.
    lens_sections = [
        ("structural", "## Structural Review", ""),
        ("per-stack", "## Issues", ""),
        ("cross-stack", "## Cross-Stack Issues", "[cross-stack] "),
        ("wonder", "## Wonder Findings", ""),
    ]

    for lens, title, prefix in lens_sections:
        rows = [i for i in items if i.get("lens") == lens]
        if rows:
            body = "\n".join(_finding_line(i, prefix=prefix) for i in rows)
            sections.append(f"{title}\n{body}")

    text = "\n\n".join(sections) + "\n"
    if diagram_blocks is not None:
        text = insert_diagrams_section(text, diagram_blocks)
    return text


def render_held_section(held: list[dict[str, Any]]) -> str:
    """Render findings withheld from the actionable report."""
    if not held:
        return ""
    body = "\n".join(_finding_line(item) for item in held)
    return f"## Held Findings\n{body}"
