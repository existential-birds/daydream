"""Tests for the shared Codex mock-process builder."""

import pytest

from tests.harness.codex_replay import make_mock_process


@pytest.mark.asyncio
async def test_make_mock_process_replays_lines():
    proc = make_mock_process(['{"type": "thread.started"}'])
    assert (await proc.stdout.readline()).strip() == b'{"type": "thread.started"}'
    assert await proc.stdout.readline() == b""
