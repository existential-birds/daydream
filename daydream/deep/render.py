"""Pure Markdown rendering of canonical merged findings and grounded diagram sections."""

from __future__ import annotations

from typing import Any

# User-visible pipeline stages (exploration is a pre-stage banner, not counted).
_PIPELINE_STAGE_NAMES: list[str] = [
    "TTT intent",
    "TTT alternative-review",
    "per-stack reviews",
    "cross-stack merge",
    "optional fix gate",
]

# Merge and diagram steps share this heading for idempotent replacement.
_DIAGRAMS_HEADING = "## Diagrams"
_REVIEW_HEADING = "# Review"


def _finding_line(item: dict[str, Any], *, prefix: str = "") -> str:
    """Render the canonical id and location as an unbolded numbered finding line."""
    return f"{item['id']}. {prefix}[{item['file']}:{item['line']}] {item['description']}"


def _is_heading(line: str) -> bool:
    """Recognize section terminators: rendered diagram blocks cannot emit top-level #/## headings."""
    return line.startswith("# ") or line.startswith("## ")


def _remove_diagrams_section(lines: list[str]) -> list[str]:
    """Remove diagram sections through the next heading and restore one separator between neighbors.

    This exactly reverses insertion, preserving idempotence.
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
        # Only surviving neighbors need a separator; edge blanks would break the inverse.
        if out and index < total:
            out.append("")
    return out


def insert_diagrams_section(report_text: str, blocks: str) -> str:
    """Replace diagrams after # Review, or prepend when that heading is absent.

    Pure and byte-idempotent. Empty/whitespace blocks remove the section; retain
    other sections and the report's trailing-newline convention.
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


def render_report(items: list[dict[str, Any]]) -> str:
    """Render nonempty lens groups in structural, per-stack, cross-stack, then wonder order.

    Use canonical ids, plain numbered location lines, and [cross-stack] title prefixes.
    """
    sections: list[str] = ["# Review"]

    # Omit empty lenses; each entry specifies its heading and finding prefix.
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
    return text


def render_held_section(held: list[dict[str, Any]]) -> str:
    """Render findings withheld from the actionable report."""
    if not held:
        return ""
    body = "\n".join(_finding_line(item) for item in held)
    return f"## Held Findings\n{body}"
