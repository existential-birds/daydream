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
"""

from __future__ import annotations

from typing import Any


def _finding_line(item: dict[str, Any], *, prefix: str = "") -> str:
    """Format one finding as the unbolded ``N. [prefix][FILE:LINE] DESCRIPTION`` line.

    The numbered head line is plain text — never wrapped in bold markers — per
    the report-format rules in ``deep/prompts.py``.
    """
    return f"{item['id']}. {prefix}[{item['file']}:{item['line']}] {item['description']}"


def render_report(items: list[dict[str, Any]]) -> str:
    """Render canonical items into the deep-review markdown report.

    Groups items by ``lens`` and emits, in order: ``## Structural Review``
    (only when structural items exist), ``## Issues`` (per-stack lens), and
    ``## Cross-Stack Issues`` (cross-stack lens, each title prefixed with the
    literal ``[cross-stack]``). A section is omitted entirely when it has no
    items. Each finding line is ``N. [FILE:LINE] DESCRIPTION``, unbolded, where
    ``N`` is the item's canonical ``id``.

    Args:
        items: Canonical merged finding items, each carrying ``id``, ``lens``,
            ``file``, ``line``, and ``description``.

    Returns:
        The rendered markdown report as a string.
    """
    structural = [i for i in items if i.get("lens") == "structural"]
    per_stack = [i for i in items if i.get("lens") == "per-stack"]
    cross_stack = [i for i in items if i.get("lens") == "cross-stack"]

    sections: list[str] = ["# Review"]

    if structural:
        body = "\n".join(_finding_line(i) for i in structural)
        sections.append(f"## Structural Review\n{body}")

    if per_stack:
        body = "\n".join(_finding_line(i) for i in per_stack)
        sections.append(f"## Issues\n{body}")

    if cross_stack:
        body = "\n".join(_finding_line(i, prefix="[cross-stack] ") for i in cross_stack)
        sections.append(f"## Cross-Stack Issues\n{body}")

    return "\n\n".join(sections) + "\n"
