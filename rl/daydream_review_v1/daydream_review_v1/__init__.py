"""verifiers v1 environment for the daydream deep review-fix-test loop.

The loader resolves a third-party taskset by importing the top-level module whose
name equals the taskset id with hyphens turned into underscores, then picking the
single ``Taskset`` subclass and the single ``Harness`` subclass out of ``__all__``
(verifiers v0.2.1 ``verifiers/v1/loaders.py:33-72``). Exporting the config classes
alongside them is fine — they subclass neither base.
"""

from daydream_review_v1.harness import DaydreamReviewHarness, DaydreamReviewHarnessConfig
from daydream_review_v1.taskset import DaydreamReviewConfig, DaydreamReviewTask, DaydreamReviewTaskset

__all__ = [
    "DaydreamReviewTaskset",
    "DaydreamReviewTask",
    "DaydreamReviewConfig",
    "DaydreamReviewHarness",
    "DaydreamReviewHarnessConfig",
]
