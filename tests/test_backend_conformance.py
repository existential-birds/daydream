"""Parametrized backend conformance suite with a documented-delta allow-list.

Both real backend drivers (Claude SDK, Codex CLI) are exercised through their
canonical-script loaders and asserted against one behavior contract:

- the documented ``AgentEvent`` vocabulary is present (TextEvent,
  ToolStartEvent, ToolResultEvent);
- every ``ToolResultEvent`` pairs with a prior ``ToolStartEvent`` (by id);
- at least one metrics-bearing event (``MetricsEvent`` or ``CostEvent``) is
  emitted;
- ``read_only=True`` is accepted by ``execute`` and does not change the
  observable vocabulary;
- ``format_skill_invocation`` yields ``/{key}`` for Claude and ``${name}`` for
  Codex (namespace stripped) — asserted against the real backend instances.

Per-driver divergences are documented in ``KNOWN_DELTAS`` and the assertions
consult that allow-list instead of demanding strict cross-driver equivalence.
The tool-id attribute on ``ToolStartEvent``/``ToolResultEvent`` is ``.id``
(confirmed in ``daydream/backends/__init__.py``), not ``tool_id``.
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
from daydream.backends.claude import ClaudeBackend
from daydream.backends.codex import CodexBackend
from tests.contract._loaders import claude_loader, codex_loader

# The canonical agent script the contract suite drives both backends against
# (see tests/contract/test_backend_step_parity.py). One source of truth.
CANONICAL_SCRIPT: dict[str, Any] = json.loads(
    (Path(__file__).parent / "contract" / "fixtures" / "canonical_script.json").read_text()
)

# Per-driver divergences accepted by the conformance contract. Each entry is
# grounded in live code, cited inline. Assertions consult this allow-list
# rather than demanding strict cross-driver equivalence on these fields.
#
# - Codex MetricsEvent.message_id == "": Codex has no per-message id; the
#   backend emits MetricsEvent with message_id="" once per turn.completed
#   (daydream/backends/codex.py:333).
# - Codex CostEvent.cost_usd is None: Codex's turn.completed.usage carries no
#   cost; the backend always yields CostEvent(cost_usd=None)
#   (daydream/backends/codex.py:341).
KNOWN_DELTAS: dict[str, dict[str, Any]] = {
    "codex": {
        "metrics_message_id": "",
        "cost_usd": None,
    },
    "claude": {},
}

Loader = Callable[..., AsyncIterator[AgentEvent]]

# Map each loader to its driver key (for KNOWN_DELTAS lookups).
_DRIVER_OF: dict[str, str] = {
    "claude_loader": "claude",
    "codex_loader": "codex",
}


def _driver(loader: Loader) -> str:
    return _DRIVER_OF[loader.__name__]


def _vocabulary(events: list[AgentEvent]) -> set[str]:
    return {type(e).__name__ for e in events}


@pytest.mark.parametrize("loader", [claude_loader, codex_loader])
async def test_backend_conformance(loader: Loader) -> None:
    """Documented vocabulary present, tool results pair with starts, a metrics
    event is emitted, and per-driver metrics deltas honor the allow-list."""
    events = [e async for e in loader(CANONICAL_SCRIPT)]

    # Vocabulary: the documented AgentEvent shapes are all present.
    types = _vocabulary(events)
    assert {"TextEvent", "ToolStartEvent", "ToolResultEvent"} <= types

    # Pairing: every ToolResultEvent id pairs with a prior ToolStartEvent id.
    starts = {e.id for e in events if isinstance(e, ToolStartEvent)}
    results = {e.id for e in events if isinstance(e, ToolResultEvent)}
    assert results <= starts

    # Metrics: at least one metrics-bearing event.
    assert any(isinstance(e, (MetricsEvent, CostEvent)) for e in events)

    # Per-driver metrics deltas consult the allow-list rather than demanding
    # strict cross-driver equivalence.
    driver = _driver(loader)
    deltas = KNOWN_DELTAS[driver]
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    costs = [e for e in events if isinstance(e, CostEvent)]
    if driver == "codex":
        # Codex carries no per-message id and no cost.
        assert all(m.message_id == deltas["metrics_message_id"] for m in metrics)
        assert all(c.cost_usd is deltas["cost_usd"] for c in costs)
    else:
        # Claude metrics carry a real per-message id (the AssistantMessage id).
        assert all(m.message_id != "" for m in metrics)


@pytest.mark.parametrize("loader", [claude_loader, codex_loader])
async def test_read_only_preserves_vocabulary(loader: Loader) -> None:
    """read_only=True is accepted by execute() and does not change the
    observable AgentEvent vocabulary."""
    default_events = [e async for e in loader(CANONICAL_SCRIPT)]
    read_only_events = [e async for e in loader(CANONICAL_SCRIPT, read_only=True)]
    assert _vocabulary(read_only_events) == _vocabulary(default_events)
    assert {"TextEvent", "ToolStartEvent", "ToolResultEvent"} <= _vocabulary(read_only_events)


def test_format_skill_invocation_per_driver() -> None:
    """Claude yields /{key}; Codex yields ${name} with the namespace stripped.

    Asserted against the real backend instances, not mocks.
    """
    key = "beagle-python:review-python"
    claude = ClaudeBackend(model="claude-test-model")
    codex = CodexBackend(model="codex-test-model")
    assert claude.format_skill_invocation(key) == f"/{key}"
    assert codex.format_skill_invocation(key) == "$review-python"
