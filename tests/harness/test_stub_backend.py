"""Tests for the shared deep-pipeline ``StubBackend`` harness."""

from __future__ import annotations

import pytest

from tests.harness.stub_backend import StubBackend


@pytest.mark.asyncio
async def test_execute_records_max_turns(tmp_path) -> None:
    backend = StubBackend(tmp_path)

    _ = [event async for event in backend.execute(tmp_path, "go", max_turns=7)]

    assert backend.calls[0]["max_turns"] == 7
