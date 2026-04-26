"""ATIF v1.6 trajectory models and validator (vendored from Harbor v0.5.0).

Re-exports the vendored Harbor surface so callers import a single namespace::

    from daydream.atif import Trajectory, Step, validate

See daydream/atif/NOTICE for provenance and Apache-2.0 attribution.
"""

from pathlib import Path
from typing import Any

from daydream.atif.models import (
    Agent,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from daydream.atif.validator import TrajectoryValidator


def validate(trajectory: dict[str, Any] | str | Path, *, validate_images: bool = True) -> bool:
    """Validate an ATIF trajectory (dict, JSON string, or path).

    Pure passthrough to a freshly-constructed TrajectoryValidator (CONTEXT.md D-08).
    Returns True iff the trajectory matches the ATIF v1.0–v1.6 schema accepted by
    the vendored validator. Returns False on any validation failure.

    For detailed error messages, instantiate TrajectoryValidator() directly and
    inspect `.errors` after calling `.validate()`.

    Args:
        trajectory: A trajectory dict, JSON string, or filesystem path.
        validate_images: Whether to verify ImageSource.path entries exist on
            disk (default True; pass False when validating in-memory dicts
            that lack a filesystem anchor).

    Returns:
        True if the trajectory passes ATIF schema + cross-reference checks;
        False otherwise.

    """
    return TrajectoryValidator().validate(trajectory, validate_images=validate_images)


__all__ = [
    "Agent",
    "ContentPart",
    "FinalMetrics",
    "ImageSource",
    "Metrics",
    "Observation",
    "ObservationResult",
    "Step",
    "SubagentTrajectoryRef",
    "ToolCall",
    "Trajectory",
    "TrajectoryValidator",
    "validate",
]
