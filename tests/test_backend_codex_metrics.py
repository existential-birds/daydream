"""Codex emits per-turn MetricsEvent plus terminal CostEvent. Known models use token pricing; unknown cost
stays None. Cache counts mirror usage, and absent message identity remains empty.
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
from tests.harness.codex_replay import make_mock_process_from_fixture as _make_mock_process


async def _execute_events(model: str, fixture: str) -> list[object]:
    backend = CodexBackend(model=model)
    mock_proc = _make_mock_process(fixture)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        return [event async for event in backend.execute(Path("/tmp"), "test")]

async def test_metrics_event_emitted_at_turn_completed() -> None:
    """Known-model turn metrics must match compute_cost for the emitted usage."""
    events = await _execute_events("gpt-5.3-codex", "turn_completed_with_usage.jsonl")
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 1
    m = metrics[0]
    assert m.message_id == ""              # D-04: Codex has no per-message id
    assert m.prompt_tokens == 200          # EVNT-02 verbatim (NOT input_tokens)
    assert m.completion_tokens == 100      # EVNT-02 verbatim (NOT output_tokens)
    assert m.cached_tokens is None         # fixture carries no cached_input_tokens
    expected = compute_cost(
        model="gpt-5.3-codex", input_tokens=200, cached_input_tokens=0, output_tokens=100,
        prices=resolve_prices(load_user_prices()),
    )
    assert expected is not None
    assert m.cost_usd is not None          # #194: synthesized at the backend layer
    assert m.cost_usd == pytest.approx(expected)

async def test_cost_event_still_emitted() -> None:
    """Terminal CostEvent remains available to FinalMetrics; fixture-model intentionally has no known
    price.
    """
    events = await _execute_events("fixture-model", "turn_completed_with_usage.jsonl")
    cost = [e for e in events if isinstance(e, CostEvent)]
    assert len(cost) == 1
    assert cost[0].input_tokens == 200     # CostEvent keeps SDK boundary names
    assert cost[0].output_tokens == 100
    assert cost[0].cached_tokens is None
    assert cost[0].cost_usd is None        # fixture-model unknown → #156 marker

async def test_partial_usage_skips_metrics_event() -> None:
    events = await _execute_events("fixture-model", "turn_completed_partial_usage.jsonl")
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    assert len(metrics) == 0
    cost = [e for e in events if isinstance(e, CostEvent)][0]
    assert cost.input_tokens == 200
    assert cost.output_tokens is None
