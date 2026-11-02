"""Shared fan-out admission across the CLI backends."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from daydream.backends import (
    effective_fanout_concurrency,
    resolve_fanout_concurrency,
)


@pytest.mark.parametrize("raw,expected", [("6", 6), ("0", 10)])
def test_endpoint_fanout_admission_and_workflow_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str,
    expected: int,
) -> None:
    """The shared positive rule owns admission; actual backend journeys own defaults and forwarding."""
    monkeypatch.setenv("DAYDREAM_PI_FANOUT_CONCURRENCY", raw)
    hint = resolve_fanout_concurrency("DAYDREAM_PI_FANOUT_CONCURRENCY", 10)
    assert hint == expected
    assert effective_fanout_concurrency(2, SimpleNamespace(fanout_concurrency=hint)) == 2
    assert ("must be positive; using default 10" in caplog.text) is (raw == "0")
