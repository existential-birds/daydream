"""Native tool-free review calls preserve Pi reasoning and large prompt transport."""

from pathlib import Path

import pytest

from daydream.agent import run_agent
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend


async def test_tools_disabled_request_fails_before_dispatch_on_unsupported_backend(tmp_path: Path) -> None:
    backend = ScriptedBackend()
    with pytest.raises(NotImplementedError, match="tools_disabled"):
        await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.DEEP, tools_disabled=True)
    assert backend.calls == []

async def test_review_system_instructions_require_native_tool_free_support(tmp_path: Path) -> None:
    backend = ScriptedBackend(supports_tools_disabled=True)
    with pytest.raises(ValueError, match="tools_disabled"):
        await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.DEEP,
                        review_system_instructions="Stop after the evidence batch.")
    with pytest.raises(NotImplementedError, match="review_instructions"):
        await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.DEEP,
                        tools_disabled=True, review_system_instructions="Stop after the evidence batch.")
    assert backend.calls == []
