"""Single source of truth for adjudication disposition classification (issue #1078).

Shared by the adjudication queue builder, the materialization step, and the
corpus v2 tier classifier so the decisive/non-decisive split is defined once.
"""

from typing import Final

__all__ = ["DECISIVE_DISPOSITIONS", "NON_DECISIVE_DISPOSITIONS", "is_decisive"]

NON_DECISIVE_DISPOSITIONS: Final[frozenset[str]] = frozenset({"ambiguous", "unanswered", "missing"})
DECISIVE_DISPOSITIONS: Final[frozenset[str]] = frozenset({"accepted", "rejected"})


def is_decisive(disposition: str) -> bool:
    """Return True when ``disposition`` is a decisive outcome (accepted/rejected)."""
    return disposition in DECISIVE_DISPOSITIONS
