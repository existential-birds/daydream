"""Arbiter selection logic for deep-mode per-stack reviews (issue #168).

Sonnet runs the N per-stack reviews; a single Opus *arbiter* re-reviews only the
findings that warrant a heavyweight second opinion. This module holds the pure,
side-effect-free predicate that decides which parsed per-stack records reach the
arbiter, so it can be unit-tested against adversarial shapes (mixed-severity,
multi-stack, same-``file:line`` collisions) independent of any agent call.

A record is selected when EITHER:
  - its severity is at or above the ``min_severity`` knob (default ``"high"``;
    heavy findings always get the Opus second look), OR
  - it is *contested*: the same ``(file, line)`` location is surfaced by two or
    more distinct stacks that disagree on severity. Divergent severity at one
    location is exactly the case a cheaper model is most likely to mis-rank.

With the default knob, low/medium uncontested findings never reach the arbiter —
that is the whole point of the cost split. Lowering ``min_severity`` (the
profile's ``Arbitration.min_severity``) widens the severity branch; the contested
branch is unaffected.

Residual risk: a genuinely-high issue that a cheaper per-stack model under-ranked
as an isolated, uncontested medium/low at a unique location is also never
arbitrated — an accepted cost trade-off of the high-OR-contested selection scope.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from daydream.severity import CANONICAL_LEVELS, SEVERITY_RANK

# Canonical confidence vocabulary (uppercase HIGH|MEDIUM|LOW schema enum,
# mirroring ``review_profile._CONFIDENCE_LEVELS``).
_CONFIDENCE_LEVELS: frozenset[str] = frozenset(("HIGH", "MEDIUM", "LOW"))


def _severity(record: dict[str, Any]) -> str:
    """Normalize a record's severity to a lowercase string ("" when absent).

    Unified fallback policy for the arbiter's own view (issue #972 R3.1): an
    empty string means "absent severity ⇒ not selectable by severity"; the
    contested path still applies to such records. No severity value is
    fabricated here.
    """
    value = record.get("severity")
    return value.lower() if isinstance(value, str) else ""


def _confidence(record: dict[str, Any]) -> str:
    """Normalize a record's confidence to an uppercase string ("" when absent)."""
    value = record.get("confidence")
    return value.upper() if isinstance(value, str) else ""


def _at_or_above(min_severity: str) -> frozenset[str]:
    """Return every canonical level ranked at or above ``min_severity``.

    Raises:
        ValueError: If ``min_severity`` is not a canonical severity level
            (fail loud — an unknown knob value must never silently widen or
            narrow selection).
    """
    if min_severity not in SEVERITY_RANK:
        raise ValueError(
            f"min_severity must be one of {', '.join(CANONICAL_LEVELS)}; got {min_severity!r}"
        )
    rank = SEVERITY_RANK[min_severity]
    return frozenset(level for level, level_rank in SEVERITY_RANK.items() if level_rank <= rank)


def select_arbiter_targets(
    records: list[dict[str, Any]],
    sources: list[str],
    min_severity: str = "high",
    contested_location: bool = True,
    contested_only: Iterable[int] = (),
) -> list[int]:
    """Return the indices of records that need arbiter re-review.

    Args:
        records: Parsed per-stack records (each ideally carrying ``severity``,
            ``file``, ``line``). Records missing ``severity`` are treated as
            below the severity knob and only become selectable through the
            contested path.
        sources: Per-record originating stack name, positionally aligned with
            ``records`` (``len(sources) == len(records)``).
        min_severity: Lowest canonical severity level the severity branch
            selects (``"high"`` by default; ``"medium"`` also selects medium
            records, ``"low"`` selects everything). The contested branch is
            independent of this knob.
        contested_location: Whether the contested branch (same location
            reported by >=2 distinct stacks with divergent severity) selects
            records (default ``True``; the profile's
            ``Arbitration.contested_location`` governs this in the production
            path).
        contested_only: Indices the severity branch must skip. They stay fully
            eligible through the contested branch. The deep pipeline passes the
            structural meta-stack's records here (issue #1103) so a structural
            finding is adjudicated against the language-stack finding that
            restates it, without widening severity-based arbitration to a lens
            that is high-conviction by construction.

    Returns:
        Sorted, de-duplicated list of indices into ``records`` selected for the
        arbiter: every record at or above ``min_severity`` (excluding
        ``contested_only``), plus every record at a location contested across
        >=2 stacks with divergent severity (when ``contested_location`` is
        enabled).

    Contested locations are ``(file, line)`` pairs, with one widening: a record
    anchored at ``line: 0`` is a whole-file finding (the reserved whole-file
    anchor), so it co-locates with *every* line reported in that file rather
    than only with other ``line: 0`` records. Without that, a whole-file
    structural finding and the line-anchored language finding describing the
    same defect land in different groups and can never contest each other
    (issue #1103, case B).

    Raises:
        ValueError: If ``records`` and ``sources`` differ in length, or if
            ``min_severity`` is not a canonical severity level.
    """
    if len(records) != len(sources):
        raise ValueError(
            f"records/sources length mismatch: {len(records)} != {len(sources)}"
        )

    selected: set[int] = set()
    severity_exempt = set(contested_only)

    # Severity branch: everything at or above the ``min_severity`` knob
    # (default "high" — the profile's ``Arbitration.min_severity`` governs this
    # in the production path). ``contested_only`` records opt out of this
    # branch and reach the arbiter only by being contested.
    eligible = _at_or_above(min_severity)
    for i, record in enumerate(records):
        if i not in severity_exempt and _severity(record) in eligible:
            selected.add(i)

    # Contested: same location reported by >=2 distinct stacks that disagree
    # on severity. Group by location, then test cross-stack severity divergence.
    if contested_location:
        by_location: dict[tuple[Any, Any], list[int]] = defaultdict(list)
        # ``line: 0`` is the reserved whole-file anchor, not a line number, so a
        # record carrying it is about the whole file and belongs to every group
        # in that file (issue #1103). Collect them separately, then fold each
        # file's whole-file records into that file's line groups.
        whole_file: dict[Any, list[int]] = defaultdict(list)
        for i, record in enumerate(records):
            if record.get("line") == 0:
                whole_file[record.get("file")].append(i)
            else:
                by_location[(record.get("file"), record.get("line"))].append(i)
        for (file, _line), indices in by_location.items():
            indices.extend(whole_file.get(file, ()))
        # A file whose only records are whole-file ones still forms one group:
        # two whole-file findings from different stacks can contest each other.
        for file, indices in whole_file.items():
            by_location.setdefault((file, 0), list(indices))

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
    severity_classes: tuple[str, ...] = ("low",),
    confidence_classes: tuple[str, ...] = ("LOW",),
) -> list[int]:
    """Return indices of borderline, uncontested records for the suppression pass (#232).

    The precision-mode suppression pass gives a skeptical LLM second opinion to
    *evidenced-but-minor* findings the arbiter never scrutinizes: records that are
    in ``confidence_classes`` and/or low-severity (per ``severity_classes``) and are
    neither high-severity nor contested. It mirrors :func:`select_arbiter_targets` as a
    pure, side-effect-free predicate so it can be unit-tested against adversarial
    shapes independent of any agent call.

    High-severity and contested records reach the *arbiter* (fail-open); this pass
    must never touch them, so callers pass the arbiter's target indices as
    ``exclude``. Because that set already covers every high-severity and contested
    record, excluding it leaves only higher-severity uncontested records -- of
    which the ones in ``confidence_classes`` and those in ``severity_classes`` are the
    borderline findings selected here.

    Args:
        records: Parsed per-stack records (each ideally carrying ``severity`` and
            ``confidence``).
        sources: Per-record originating stack name, positionally aligned with
            ``records`` (``len(sources) == len(records)``). Accepted for signature
            symmetry with :func:`select_arbiter_targets`; contestedness is handled
            entirely through ``exclude``.
        exclude: Indices to skip (the arbiter target set). A record already routed
            to the arbiter is never a suppression target.

        severity_classes: Canonical severity levels the severity branch selects
            (default ``("low",)``; the profile's ``Suppression.severity_classes``
            governs this in the production path).
        confidence_classes: Canonical uppercase confidence levels the confidence
            branch selects (default ``("LOW",)``; the profile's
            ``Suppression.confidence_classes`` governs this in the production
            path).

    Returns:
        Sorted, de-duplicated list of indices into ``records`` selected for the
        suppression pass: every ``exclude``-free record that is in
        ``confidence_classes`` or low-severity (per ``severity_classes``).

    Raises:
        ValueError: If ``records`` and ``sources`` differ in length, if
            ``severity_classes`` contains a non-canonical severity value, or if
            ``confidence_classes`` contains a non-canonical confidence value.
    """
    classes = frozenset(
        cls.lower() for cls in severity_classes
    )
    unknown = classes - frozenset(CANONICAL_LEVELS)
    if unknown:
        raise ValueError(
            f"severity_classes must be a subset of {', '.join(sorted(CANONICAL_LEVELS))}; "
            f"got unknown value(s): {', '.join(sorted(unknown))}"
        )
    confidence = frozenset(cnf.upper() for cnf in confidence_classes)
    unknown_conf = confidence - _CONFIDENCE_LEVELS
    if unknown_conf:
        raise ValueError(
            f"confidence_classes must be a subset of {', '.join(sorted(_CONFIDENCE_LEVELS))}; "
            f"got unknown value(s): {', '.join(sorted(unknown_conf))}"
        )
    if len(records) != len(sources):
        raise ValueError(
            f"records/sources length mismatch: {len(records)} != {len(sources)}"
        )

    excluded = set(exclude)
    selected: list[int] = []
    for i, record in enumerate(records):
        if i in excluded:
            continue
        if _confidence(record) in confidence or _severity(record) in classes:
            selected.append(i)
    return selected
