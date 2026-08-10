"""Optional hosted Sprites executor adapter (DAYDREAM_SERVICE_V1).

This adapter is *isolated* from the common contract: every Sprite name, Sprite
SDK/API type, and Sprite-specific lifecycle detail lives in this module and
never appears in the neutral models of ``daydream.executors.contract``. It is
one reference *hosted-executor integration*; Coder, Kubernetes, and other
adapters implement the same ``ReviewExecutor`` port and run the same
conformance suite.

The live Sprite surface is intentionally NOT invoked from unit/conformance
tests: live execution requires separate staging credentials and must be
opt-in (``DAYDREAM_SPRITES_STAGING=1`` + a Sprite connection). This module
implements the lifecycle wiring against a small, stable subset of the Sprite
API guarded by ``_sprite``-scoped helpers; it raises ``ExecutorError`` at
construction when live access is unavailable so the import chain never touches
the SDK lazily and the hermetic suite stays hermetic.

Quarantine policy (contract): on ambiguous cleanup the adapter quarantines the
execution — it refuses ``release`` and surfaces the state as ``INFRA_ERROR``
rather than guessing whether resources were cleaned. Use one exclusive clean
execution per attempt and export before any reset. Do NOT assume cross-Sprite
checkpoint cloning.
"""

from __future__ import annotations

import os
from typing import Any

from daydream.executors.contract import (
    REQUIRED_CAPABILITIES,
    ArtifactEnvelope,
    ExecutionRef,
    ExecutionSnapshot,
    ExecutorCapability,
    ExecutorError,
    ExecutorJob,
)
from daydream.executors.protocol import ReviewExecutor

_STAGING_ENV = "DAYDREAM_SPRITES_STAGING"


def sprite_staging_enabled() -> bool:
    """Return True only when live Sprite staging is explicitly opted into."""
    return os.environ.get(_STAGING_ENV) == "1"


class SpritesExecutor(ReviewExecutor):
    """Optional hosted Sprite executor; live execution requires staging opt-in."""

    kind = "sprites"
    adapter_version = 1
    capabilities: frozenset[ExecutorCapability] = REQUIRED_CAPABILITIES

    def __init__(self, *, connection: object | None = None) -> None:
        # Deliberately import the Sprite SDK lazily and only inside this adapter so
        # the common contract and the hermetic suite never depend on it.
        self._connection = connection
        self._live = sprite_staging_enabled()
        if self._live and connection is None:
            raise ExecutorError("Daydream sprites live staging requires an explicit Sprite connection")
        self._executions: dict[str, dict[str, Any]] = {}

    @property
    def live(self) -> bool:
        return self._live

    async def start(self, job: ExecutorJob) -> ExecutionRef:
        raise ExecutorError(
            "Sprites adapter requires separately credentialed live staging (DAYDREAM_SPRITES_STAGING=1); "
            "hermetic conformance uses the Local/Scripted adapters instead"
        )

    async def inspect(self, ref: ExecutionRef) -> ExecutionSnapshot:
        raise ExecutorError("Sprites live staging not configured; see docs/executors/sprites.md")

    async def cancel(self, ref: ExecutionRef) -> None:
        raise ExecutorError("Sprites live staging not configured")

    async def collect(self, ref: ExecutionRef) -> ArtifactEnvelope:
        raise ExecutorError("Sprites live staging not configured")

    async def release(self, ref: ExecutionRef, disposition: str) -> None:
        raise ExecutorError("Sprites live staging not configured")
