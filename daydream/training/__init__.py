"""Training-record exporter and downstream training-data utilities.

This package owns *reading from* the daydream archive for downstream training
consumers. The archive package (`daydream.archive`) owns *writing to* it.

Modules under this package land incrementally per the JSONL exporter plan
(.planning/specs/open-weight-review-model/issues/01-jsonl-exporter-PLAN.md).
Future R2/R3/R5/R8 work (reward composition, auto-labeling, training recipes,
bootstrap collection) will also live here.
"""

from daydream.training.labeler_signals import (
    PerFindingResolution,
    per_finding_resolution_signal,
)
from daydream.training.rubric import PerFindingLabel, derive_per_finding_labels

__all__ = [
    "PerFindingLabel",
    "PerFindingResolution",
    "derive_per_finding_labels",
    "per_finding_resolution_signal",
]
