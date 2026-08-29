"""Single source of truth for finding severity vocabulary and policy.

This module is the one place to look for the canonical severity levels, the
sort-rank mapping, and the missing-severity fallback policy. It is a leaf
module: it must not import from any other daydream module, so it stays
importable everywhere in the pipeline.

P6 rule: off-vocabulary severity values are never silently passed through.
A boundary site that encounters an unknown or absent value must map it
explicitly (via :func:`normalize_severity`, which returns ``None`` for
unknown/absent) and decide what that means in its own context.

Default severity policy (R3.1, documented once):

- A missing severity stays ``None`` on report and approval paths; ``None``
  deliberately does not block approval.
- The only ``"high"`` default permitted anywhere is the structural-lens
  ``setdefault("severity", "high")`` (structural high-conviction invariant).
- No other fallback severity value is permitted.
"""

CANONICAL_LEVELS: tuple[str, ...] = ("low", "medium", "high")
"""The canonical severity vocabulary: the only declaration of the levels."""

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
    """Normalize a severity value to the canonical vocabulary.

    Returns the canonical lowercase value for known levels (case- and
    whitespace-tolerant), and ``None`` for unknown, non-string, or absent
    values. Callers must handle ``None`` explicitly — an unknown value is
    never mapped implicitly.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in CANONICAL_LEVELS else None
