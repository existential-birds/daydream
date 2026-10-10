"""Partial Codex usage must not invent complete per-turn metrics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from daydream.backends import (
    CostEvent,
    MetricsEvent,
)
from daydream.backends.codex import CodexBackend
from tests.harness.codex_replay import make_mock_process_from_fixture as _make_mock_process


async def _execute_events(model: str, fixture: str) -> list[object]:
    backend = CodexBackend(model=model)
    mock_proc = _make_mock_process(fixture)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        return [event async for event in backend.execute(Path("/tmp"), "test")]


async def test_partial_usage_skips_metrics_event() -> None:
    events = await _execute_events("fixture-model", "turn_completed_partial_usage.jsonl")
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 0
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert cost.input_tokens == 200
    assert cost.output_tokens is None
