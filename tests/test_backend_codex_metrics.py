"""Tests for Codex backend MetricsEvent emission (EVNT-07).

Verifies that the Codex backend emits a MetricsEvent at every
``turn.completed`` event with the documented shape:

- ``cost_usd`` is synthesized from tokens via the #61 price table for known
  models (#194 reverses D-16 — Codex now matches Claude/Pi by populating
  cost at the backend layer). It stays None for models unknown to the table
  (#156 observable marker).
- ``cached_tokens`` mirrors ``cached_input_tokens`` from usage (None when the
  CLI omits the field).
- ``message_id`` is the empty string "" (Codex has no per-message id; D-04).

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
from daydream.pricing import compute_cost, load_user_prices, resolve_prices

# Reuse the shared JSONL-stream mock helper from tests/harness — single
# source of truth for the mock-process shape (Pitfall: do NOT re-implement
# readline). Imported directly so the existing pattern is preserved.
from tests.harness.codex_replay import make_mock_process_from_fixture as _make_mock_process


@pytest.mark.asyncio
async def test_metrics_event_emitted_at_turn_completed():
    """turn.completed with full usage produces MetricsEvent with EVNT-02 field names (EVNT-07).

    Model gpt-5.3-codex is in MODEL_PRICES, so #194 synthesizes cost at the
    backend layer; the value must match compute_cost for these tokens.
    """
    backend = CodexBackend(model="gpt-5.3-codex")
    mock_proc = _make_mock_process("turn_completed_with_usage.jsonl")
    with patch("daydream.backends.codex.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "test"):
            events.append(event)
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 1
    m = metrics[0]
    assert m.message_id == ""              # D-04: Codex has no per-message id
    assert m.prompt_tokens == 200          # EVNT-02 verbatim (NOT input_tokens)
    assert m.completion_tokens == 100      # EVNT-02 verbatim (NOT output_tokens)
    assert m.cached_tokens is None         # fixture carries no cached_input_tokens
    expected = compute_cost(
        model="gpt-5.3-codex",
        input_tokens=200,
        cached_input_tokens=0,
        output_tokens=100,
        prices=resolve_prices(load_user_prices()),
    )
    assert expected is not None
    assert m.cost_usd is not None          # #194: synthesized at the backend layer
    assert m.cost_usd == pytest.approx(expected)


@pytest.mark.asyncio
async def test_cost_event_still_emitted():
    """The legacy CostEvent emission is preserved (so FinalMetrics aggregation works for Codex too).

    Model ``fixture-model`` is unknown to the price table, so cost_usd stays
    None here (#156); a known model would synthesize (see test_metrics_event
    above and test_backend_codex.py::test_codex_synthesizes_cost_for_known_model).
    """
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
    assert cost[0].cost_usd is None        # fixture-model unknown → #156 marker


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
