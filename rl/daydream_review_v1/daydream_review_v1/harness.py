"""Harness scaffold — implemented in Phase 4."""

from __future__ import annotations

import verifiers.v1 as vf


class DaydreamReviewHarnessConfig(vf.HarnessConfig):
    """Placeholder; Phase 4 adds backend / fanout_concurrency / extra_args."""


class DaydreamReviewHarness(vf.Harness[DaydreamReviewHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = False
    SUPPORTS_MESSAGE_PROMPT = False

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        raise NotImplementedError("Phase 4")
