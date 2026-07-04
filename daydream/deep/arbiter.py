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
from collections.abc import Iterable
from typing import Any


def _severity(record: dict[str, Any]) -> str:
    """Normalize a record's severity to a lowercase string ("" when absent)."""
    value = record.get("severity")
    return value.lower() if isinstance(value, str) else ""


def _confidence(record: dict[str, Any]) -> str:
    """Normalize a record's confidence to an uppercase string ("" when absent)."""
    value = record.get("confidence")
    return value.upper() if isinstance(value, str) else ""


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


def select_suppression_targets(
    records: list[dict[str, Any]],
    sources: list[str],
    exclude: Iterable[int] = (),
) -> list[int]:
    """Return indices of borderline, uncontested records for the suppression pass (#232).

    The precision-mode suppression pass gives a skeptical LLM second opinion to
    *evidenced-but-minor* findings the arbiter never scrutinizes: records that are
    ``confidence == "LOW"`` and/or ``severity == "low"`` and are neither
    high-severity nor contested. It mirrors :func:`select_arbiter_targets` as a
    pure, side-effect-free predicate so it can be unit-tested against adversarial
    shapes independent of any agent call.

    High-severity and contested records reach the *arbiter* (fail-open); this pass
    must never touch them, so callers pass the arbiter's target indices as
    ``exclude``. Because that set already covers every high-severity and contested
    record, excluding it leaves only low/medium uncontested records -- of which the
    LOW-confidence / low-severity ones are the borderline findings selected here.

    Args:
        records: Parsed per-stack records (each ideally carrying ``severity`` and
            ``confidence``).
        sources: Per-record originating stack name, positionally aligned with
            ``records`` (``len(sources) == len(records)``). Accepted for signature
            symmetry with :func:`select_arbiter_targets`; contestedness is handled
            entirely through ``exclude``.
        exclude: Indices to skip (the arbiter target set). A record already routed
            to the arbiter is never a suppression target.

    Returns:
        Sorted, de-duplicated list of indices into ``records`` selected for the
        suppression pass: every ``exclude``-free record that is LOW-confidence
        or low-severity.

    Raises:
        ValueError: If ``records`` and ``sources`` differ in length.
    """
    if len(records) != len(sources):
        raise ValueError(
            f"records/sources length mismatch: {len(records)} != {len(sources)}"
        )

    excluded = set(exclude)
    selected: list[int] = []
    for i, record in enumerate(records):
        if i in excluded:
            continue
        if _confidence(record) == "LOW" or _severity(record) == "low":
            selected.append(i)
    return selected
