"""Tests for Codex backend MetricsEvent emission (EVNT-07, D-16).

Verifies that the Codex backend emits a MetricsEvent at every
``turn.completed`` event with the documented parity gap:

- ``cost_usd`` is ALWAYS None (Codex CLI does not report USD cost; D-16
  decision: do NOT synthesize from a token-price table — Pitfall 6).
- ``cached_tokens`` is ALWAYS None (Codex ``turn.completed.usage`` has
  no cached-token field).
- ``message_id`` is the empty string "" (Codex has no per-message id
  surface; D-16).

The legacy CostEvent emission is preserved so FinalMetrics aggregation
still works for Codex runs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from daydream.backends import (
    CostEvent,
    MetricsEvent,
)
from daydream.backends.codex import CodexBackend

# Reuse the shared JSONL-stream mock helper from tests/harness — single
# source of truth for the mock-process shape (Pitfall: do NOT re-implement
# readline). Imported directly so the existing pattern is preserved.
from tests.harness.codex_replay import make_mock_process_from_fixture as _make_mock_process


@pytest.mark.asyncio
async def test_metrics_event_emitted_at_turn_completed():
    """turn.completed with full usage produces MetricsEvent with EVNT-02 field names (EVNT-07)."""
    backend = CodexBackend(model="gpt-5.3-codex")
    mock_proc = _make_mock_process("turn_completed_with_usage.jsonl")
    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "test"):
            events.append(event)
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 1
    m = metrics[0]
    assert m.message_id == ""              # D-16: Codex parity gap — no per-message id
    assert m.prompt_tokens == 200          # EVNT-02 verbatim (NOT input_tokens)
    assert m.completion_tokens == 100      # EVNT-02 verbatim (NOT output_tokens)
    assert m.cached_tokens is None         # D-16: Codex parity gap
    assert m.cost_usd is None              # D-16: Codex parity gap


@pytest.mark.asyncio
async def test_cost_event_still_emitted():
    """The legacy CostEvent emission is preserved (so FinalMetrics aggregation works for Codex too)."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = _make_mock_process("turn_completed_with_usage.jsonl")
    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "test"):
            events.append(event)
    cost = [e for e in events if isinstance(e, CostEvent)]
    assert len(cost) == 1
    assert cost[0].input_tokens == 200     # CostEvent keeps SDK boundary names
    assert cost[0].output_tokens == 100
    assert cost[0].cached_tokens is None
    assert cost[0].cost_usd is None


@pytest.mark.asyncio
async def test_partial_usage_skips_metrics_event():
    """usage missing output_tokens => no MetricsEvent emitted (EVNT-02 requires both as int)."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = _make_mock_process("turn_completed_partial_usage.jsonl")
    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "test"):
            events.append(event)
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 0
    # CostEvent still emitted with the partial signal.
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert cost.input_tokens == 200
    assert cost.output_tokens is None


@pytest.mark.asyncio
async def test_codex_parity_gap_d16():
    """D-16: cost_usd and cached_tokens are ALWAYS None for Codex; no token-price-table synthesis."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = _make_mock_process("turn_completed_with_usage.jsonl")
    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "test"):
            events.append(event)
    for e in events:
        if isinstance(e, (MetricsEvent, CostEvent)):
            assert e.cost_usd is None
            assert e.cached_tokens is None
