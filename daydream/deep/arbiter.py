"""Arbiter selection logic for deep-mode per-stack reviews (issue #168).

Sonnet runs the N per-stack reviews; a single Opus *arbiter* re-reviews only the
findings that warrant a heavyweight second opinion. This module holds the pure,
side-effect-free predicate that decides which parsed per-stack records reach the
arbiter, so it can be unit-tested against adversarial shapes (mixed-severity,
multi-stack, same-``file:line`` collisions) independent of any agent call.

A record is selected when EITHER:
  - it is ``severity == "high"`` (heavy findings always get the Opus second look), OR
  - it is *contested*: the same ``(file, line)`` location is surfaced by two or
    more distinct stacks that disagree on severity. Divergent severity at one
    location is exactly the case a cheaper model is most likely to mis-rank.

Low/medium uncontested findings never reach the arbiter — that is the whole
point of the cost split.

Residual risk: a genuinely-high issue that a cheaper per-stack model under-ranked
as an isolated, uncontested medium/low at a unique location is also never
arbitrated — an accepted cost trade-off of the high-OR-contested selection scope.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _severity(record: dict[str, Any]) -> str:
    """Normalize a record's severity to a lowercase string ("" when absent)."""
    value = record.get("severity")
    return value.lower() if isinstance(value, str) else ""


def select_arbiter_targets(
    records: list[dict[str, Any]],
    sources: list[str],
) -> list[int]:
    """Return the indices of records that need arbiter re-review.

    Args:
        records: Parsed per-stack records (each ideally carrying ``severity``,
            ``file``, ``line``). Records missing ``severity`` are treated as
            non-high and only become selectable through the contested path.
        sources: Per-record originating stack name, positionally aligned with
            ``records`` (``len(sources) == len(records)``).

    Returns:
        Sorted, de-duplicated list of indices into ``records`` selected for the
        arbiter: every high-severity record, plus every record at a
        ``(file, line)`` location contested across >=2 stacks with divergent
        severity.

    Raises:
        ValueError: If ``records`` and ``sources`` differ in length.
    """
    if len(records) != len(sources):
        raise ValueError(
            f"records/sources length mismatch: {len(records)} != {len(sources)}"
        )

    selected: set[int] = set()

    # High severity: always arbitrated.
    for i, record in enumerate(records):
        if _severity(record) == "high":
            selected.add(i)

    # Contested: same (file, line) reported by >=2 distinct stacks that disagree
    # on severity. Group by location, then test cross-stack severity divergence.
    by_location: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_location[(record.get("file"), record.get("line"))].append(i)

    for indices in by_location.values():
        stacks = {sources[i] for i in indices}
        severities = {_severity(records[i]) for i in indices if _severity(records[i])}
        if len(stacks) >= 2 and len(severities) >= 2:
            selected.update(indices)

    return sorted(selected)
