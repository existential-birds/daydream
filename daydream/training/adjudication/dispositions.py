"""Single source of truth for adjudication disposition classification (issue #1078).

Re-export shim: the source of truth now lives at ``daydream.training.dispositions``
so the corpus v2 tier classifier can consume it without triggering this package's
eager import chain. Kept for backward-compatible imports.
"""

from daydream.training.dispositions import (
    DECISIVE_DISPOSITIONS,
    NON_DECISIVE_DISPOSITIONS,
    is_decisive,
)

__all__ = ["DECISIVE_DISPOSITIONS", "NON_DECISIVE_DISPOSITIONS", "is_decisive"]
