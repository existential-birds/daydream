"""Run Claude, Codex, and Pi replay drivers against shared event-vocabulary, tool-correlation, metrics,
read-only, and tool-permission contracts. KNOWN_DELTAS declares the permitted driver differences.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

import pytest

from daydream.backends import (
    AgentEvent,
    CostEvent,
    MetricsEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from tests.contract._loaders import claude_loader, codex_loader, pi_loader

# The canonical replay script provides shared step parity.
CANONICAL_SCRIPT: dict[str, Any] = json.loads(
    (Path(__file__).parent / "contract" / "fixtures" / "canonical_script.json").read_text()
)

# Permitted driver differences: Codex and Pi do not provide per-message identity. Codex
# fixture-model pricing is unknown, so cost stays None. Pi's explicitly reported zero cost takes
# precedence over computed pricing.
KNOWN_DELTAS: dict[str, dict[str, Any]] = {
    "codex": {"metrics_message_id": "", "cost_usd": None}, "pi": {"metrics_message_id": "", "cost_usd": 0.0},
    "claude": {},
}

Loader = Callable[..., AsyncIterator[AgentEvent]]

_DRIVER_OF: dict[str, str] = {"claude_loader": "claude", "codex_loader": "codex", "pi_loader": "pi"}

def _driver(loader: Loader) -> str:
    return _DRIVER_OF[loader.__name__]

def _vocabulary(events: list[AgentEvent]) -> set[str]:
    return {type(e).__name__ for e in events}

@pytest.mark.parametrize("loader", [claude_loader, codex_loader, pi_loader])
async def test_backend_conformance(loader: Loader) -> None:
    events = [e async for e in loader(CANONICAL_SCRIPT)]
    types = _vocabulary(events)
    assert {"TextEvent", "ToolStartEvent", "ToolResultEvent"} <= types
    starts = {e.id for e in events if isinstance(e, ToolStartEvent)}
    results = {e.id for e in events if isinstance(e, ToolResultEvent)}
    assert results <= starts
    assert any(isinstance(e, (MetricsEvent, CostEvent)) for e in events)
    read_only_events = [e async for e in loader(CANONICAL_SCRIPT, read_only=True)]
    assert _vocabulary(read_only_events) == types
    assert {"TextEvent", "ToolStartEvent", "ToolResultEvent"} <= _vocabulary(read_only_events)
    driver = _driver(loader)
    deltas = KNOWN_DELTAS[driver]
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    costs = [e for e in events if isinstance(e, CostEvent)]
    if driver == "claude":
        assert all(m.message_id != "" for m in metrics)
    else:
        # Compare float values by equality, not identity, and require both metric lists nonempty to
        # prevent a vacuous pass.
        assert metrics
        assert costs
        assert all(m.message_id == deltas["metrics_message_id"] for m in metrics)
        assert all(c.cost_usd == deltas["cost_usd"] for c in costs)
