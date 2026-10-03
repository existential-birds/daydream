"""Shared assertions for the external protocol-CLI backend contract tests.

Pi and Osprey drive the same subprocess transport boundary, so the common
observation/terminal-envelope invariants and the cancel lifecycle probe live
here once instead of being copied into each backend's test module.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

from daydream.backends import ResultEvent, TextEvent
from daydream.backends._transport import CliTransport
from daydream.backends.osprey import OspreyBackend, OspreyConfig
from daydream.backends.pi import PiBackend


def assert_protocol_cli_invariants(fixture: Any, target: Path, prompt: str, events: list[Any], backend: Any
) -> dict[str, Any]:
    """Assert the transport invariants every protocol-CLI backend must hold.

    Returns the recorded observation so the caller can add its backend-specific
    argv assertions.
    """
    observation: dict[str, Any] = fixture.read_observations()[0]
    assert observation["effective_cwd"] == str(target)
    assert observation["inherited_cwd"] == str(target)
    assert observation["stdin_bytes"] == 0
    assert observation["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert observation["cwd_canaries"]["SOURCE_CANARY"] is True
    assert observation["walk_truncated"] is False
    assert any(isinstance(event, TextEvent) and event.text == "CURRENT_REASONING_CANARY" for event in events)
    assert len([event for event in events if isinstance(event, ResultEvent)]) == 1
    assert backend._transports == []
    return observation


def make_cancel_probe(kind: Literal["pi", "osprey"]) -> tuple[Any, MagicMock]:
    """A backend with one tracked transport whose process times out then reaps."""
    backend: Any
    if kind == "pi":
        backend = PiBackend(model="glm-5.2")
        transport = CliTransport("pi", ["pi", "--mode", "json"], limit=1024)
    else:
        backend = OspreyBackend(OspreyConfig(osprey_binary="fake"))
        transport = CliTransport("osprey", ["osprey", "agent"], limit=1024)
    proc = MagicMock()
    proc.returncode = None
    proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), 0])
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    transport._proc = proc
    backend._transports = [transport]
    return backend, proc
