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
- no backend exposes a Daydream-owned skill-invocation surface.

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
    create_backend,
)
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
# - Codex CostEvent.cost_usd is None: the conformance loader drives a sentinel
#   model (``codex-test-model``) that is absent from the price table, so #194's
#   backend-layer synthesis yields None (#156). A production priced model
#   (e.g. gpt-5.5) would synthesize a non-None cost; that path is covered by
#   tests/test_backend_codex.py and tests/test_codex_real_cli_contract.py.
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

    types = _vocabulary(events)
    assert {"TextEvent", "ToolStartEvent", "ToolResultEvent"} <= types

    starts = {e.id for e in events if isinstance(e, ToolStartEvent)}
    results = {e.id for e in events if isinstance(e, ToolResultEvent)}
    assert results <= starts

    assert any(isinstance(e, (MetricsEvent, CostEvent)) for e in events)

    # Per-driver metrics deltas consult the allow-list, not strict cross-driver equivalence.
    driver = _driver(loader)
    deltas = KNOWN_DELTAS[driver]
    metrics = [e for e in events if isinstance(e, MetricsEvent)]
    costs = [e for e in events if isinstance(e, CostEvent)]
    if driver == "codex":
        # Codex carries no per-message id; cost is None only because the
        # conformance loader's sentinel model is unpriced (see KNOWN_DELTAS).
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


def test_backends_have_no_skill_method() -> None:
    """M13: no backend formats/resolves/registers/permits/invokes a skill."""
    from daydream.backends import Backend

    assert not hasattr(Backend, "format_skill_invocation")
    backends = (
        create_backend("claude", model="test-model"),
        create_backend("codex", model="test-model"),
        create_backend("pi", model="test-model"),
        create_backend("osprey", model="test-model", osprey_binary="fake-osprey"),
    )
    for backend in backends:
        assert not hasattr(backend, "format_skill_invocation")
