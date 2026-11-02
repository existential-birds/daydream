"""Tests for the shared Claude-SDK mock/patch harness.

The harness stands in for the SDK types ``ClaudeBackend`` dispatches on, so its
own contract — the eight patched names, message replay, options capture, and the
``assistant_message`` / ``result_message`` override seam the metrics tests rely
on — is pinned here rather than inferred from its consumers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.backends.claude import ClaudeBackend
from tests.harness.claude_sdk import (
    patch_claude_sdk,
    scripted_client,
)


async def _drive(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> list[Any]:
    patch_claude_sdk(monkeypatch, scripted_client(messages))
    backend = ClaudeBackend(model="opus")
    return [event async for event in backend.execute(Path("/tmp"), "go")]

