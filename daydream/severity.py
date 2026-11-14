"""Canonical severity vocabulary and explicit boundary normalization.

This leaf module must not import other daydream modules. CANONICAL_LEVELS owns
the levels; normalize_severity returns None for unknown or absent values.
Each boundary decides how to handle None: report/approval paths preserve it
without blocking approval; only the structural lens may default to high.
"""

CANONICAL_LEVELS: tuple[str, ...] = ("low", "medium", "high")
"""The canonical severity vocabulary: the only declaration of the levels."""


def model_facing_levels() -> tuple[str, ...]:
    """Return the canonical levels in descending, model-facing order."""
    return tuple(reversed(CANONICAL_LEVELS))


SEVERITY_RANK: dict[str, int] = {"high": 0, "medium": 1, "low": 2}
"""Sort rank for the canonical levels (high < medium < low).

The single source of truth consumed by both the deep arbiter (target
selection) and the deep/shallow fix-loop sort site (``phases.severity_sorted``),
so a reorder anywhere can never silently diverge. Unknown or absent values
rank as medium (``1``) at the sort site via ``SEVERITY_RANK.get(value, 1)``.
This is a sort rank, not a severity assignment: it does not constitute a
fallback severity policy.
"""


"""Host-owned severity rubric appended to every severity-assigning prompt.

Defines all three canonical levels in observable, checkable terms. It is
appended AFTER all profile strategy text (P-RUBRIC) and is never routed
through profile-owned strategy content (R1.4): builders import this constant
from this module, so no builder can inline a divergent copy or expose the
rubric to profile override.
"""


def normalize_severity(value: object) -> str | None:
    """Return a canonical lowercase level, tolerating case and whitespace.

    Unknown, non-string, or absent values return None for explicit caller handling.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in CANONICAL_LEVELS else None


