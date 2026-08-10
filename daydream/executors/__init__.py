"""``daydream.executors``: the DAYDREAM_SERVICE_V1 executor seam.

Public surface for registered compute/workspace adapters and their conformance
tests. Importable by both daydream core and external adapter packages:

- :mod:`daydream.executors.contract` — neutral models, capabilities, errors,
  capability admission (no vendor fields in common schema).
- :mod:`daydream.executors.protocol` — the ``ReviewExecutor`` port.
- :mod:`daydream.executors.local` — hermetic filesystem/time-based reference
  adapter (development/test infrastructure).
- :mod:`daydream.executors.scripted` — a structurally different in-memory,
  step-based conformance adapter.
- :mod:`daydream.executors.sprites` — optional hosted Sprites adapter (kept
  adapter-scoped; live execution is separately credentialed).
"""

from daydream.executors.contract import (
    DAYDREAM_SERVICE_V1,
    MIN_SUPPORTED_DAYDREAM_SERVICE_V1,
    REQUIRED_CAPABILITIES,
    ArtifactEnvelope,
    CancelError,
    ExecutionOutcome,
    ExecutionRef,
    ExecutionSnapshot,
    ExecutionStatus,
    ExecutorCapability,
    ExecutorError,
    ExecutorInfrastructureError,
    ExecutorJob,
    UnknownExecutionError,
    is_terminal,
    map_vendor_error,
    require_capabilities,
)
from daydream.executors.local import LocalExecutor
from daydream.executors.protocol import ReviewExecutor, is_review_executor
from daydream.executors.scripted import ScriptedExecutor
from daydream.executors.sprites import SpritesExecutor, sprite_staging_enabled

__all__ = [
    "ArtifactEnvelope",
    "CancelError",
    "DAYDREAM_SERVICE_V1",
    "ExecutionOutcome",
    "ExecutionRef",
    "ExecutionSnapshot",
    "ExecutionStatus",
    "ExecutorCapability",
    "ExecutorError",
    "ExecutorInfrastructureError",
    "ExecutorJob",
    "LocalExecutor",
    "MIN_SUPPORTED_DAYDREAM_SERVICE_V1",
    "REQUIRED_CAPABILITIES",
    "ReviewExecutor",
    "ScriptedExecutor",
    "SpritesExecutor",
    "UnknownExecutionError",
    "is_review_executor",
    "is_terminal",
    "map_vendor_error",
    "require_capabilities",
    "sprite_staging_enabled",
]
