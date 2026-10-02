"""Canonical severity vocabulary and explicit boundary normalization.

This leaf module must not import other daydream modules. CANONICAL_LEVELS owns
the levels; normalize_severity returns None for unknown or absent values.
Each boundary decides how to handle None: report/approval paths preserve it
without blocking approval; only the structural lens may default to high.
"""

from typing import Literal, TypeAlias

CANONICAL_LEVELS: tuple[str, ...] = ("low", "medium", "high")
"""The canonical severity vocabulary: the only declaration of the levels."""

SeverityLevel: TypeAlias = Literal["high", "medium", "low"]
"""Static mirror of the model-facing levels, bound to the runtime declaration by tests."""


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


SEVERITY_RUBRIC = (
    "## Severity Rubric\n"
    "\n"
    "Assign exactly one level per finding:\n"
    "\n"
    "- high: the defect breaks a primary user journey, causes data loss or "
    "corruption, introduces a security vulnerability, or leaves the changed "
    "code incorrect as merged. A finding is high only when it meets one of "
    "these conditions.\n"
    "- medium: a real defect with a workaround or a limited blast radius -- "
    "wrong in a secondary path, an edge case, or recoverable at runtime.\n"
    "- low: a style, clarity, naming, or minor robustness issue with no "
    "behavioral break. Requests for work outside this diff are not findings; "
    "when surfaced at all, they are low.\n"
    "\n"
    "Maintainability, readability, and structural-erosion findings are never "
    "high: they do not break a primary user journey, lose or corrupt data, open a "
    "security vulnerability, or make the merged code incorrect."
)
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


def is_high_severity(value: object) -> bool:
    """Whether *value* normalizes to the ``high`` level (unknown/absent is never high)."""
    return normalize_severity(value) == "high"


def stronger_severity(a: object, b: object) -> str | None:
    """Return the stronger canonical input when folding duplicate findings.

    Normalize both inputs; unknown/absent values cannot win or fabricate a level.
    Return None only when neither input is canonical.
    """
    left = normalize_severity(a)
    right = normalize_severity(b)
    if left is None:
        return right
    if right is None:
        return left
    return left if SEVERITY_RANK[left] <= SEVERITY_RANK[right] else right
