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
"""Sort rank mirroring ``phases._SEVERITY_RANK`` exactly.

Unknown or absent values rank as medium (``1``) at the sort site via
``SEVERITY_RANK.get(value, 1)``. This is a sort rank, not a severity
assignment: it does not constitute a fallback severity policy.
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
