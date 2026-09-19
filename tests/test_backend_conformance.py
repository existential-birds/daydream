"""Parametrized backend conformance suite with a documented-delta allow-list.

Both real backend drivers (Claude SDK, Codex CLI, Pi CLI) are exercised through
their canonical-script loaders and asserted against one behavior contract:

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
from tests.contract._loaders import claude_loader, codex_loader, pi_loader

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
# - Pi MetricsEvent.message_id == "": Pi has no per-message id; the backend
#   emits MetricsEvent with message_id="" once per turn_end (daydream/backends/pi.py).
# - Pi CostEvent.cost_usd == 0.0: the canonical script declares
#   ``usage.cost.total == 0.0`` and the loader drives the sentinel model
#   ``pi-test-model``, so the backend carries the scripted zero through rather
#   than synthesizing a priced cost (measured against the loader's script).
KNOWN_DELTAS: dict[str, dict[str, Any]] = {
    "codex": {
        "metrics_message_id": "",
        "cost_usd": None,
    },
    "pi": {
        "metrics_message_id": "",
        "cost_usd": 0.0,
    },
    "claude": {},
}

Loader = Callable[..., AsyncIterator[AgentEvent]]

# Map each loader to its driver key (for KNOWN_DELTAS lookups).
_DRIVER_OF: dict[str, str] = {
    "claude_loader": "claude",
    "codex_loader": "codex",
    "pi_loader": "pi",
}


def _driver(loader: Loader) -> str:
    return _DRIVER_OF[loader.__name__]


def _vocabulary(events: list[AgentEvent]) -> set[str]:
    return {type(e).__name__ for e in events}


@pytest.mark.parametrize("loader", [claude_loader, codex_loader, pi_loader])
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
    if driver == "claude":
        # Claude metrics carry a real per-message id (the AssistantMessage id).
        assert all(m.message_id != "" for m in metrics)
    else:
        # Codex and Pi consult their declared deltas: equality (not identity)
        # so a computed 0.0 cost cannot pass or fail on float identity.  The
        # lists must be non-empty or ``all(...)`` above would hold vacuously
        # (the earlier ``any`` guard only requires one of the two event kinds).
        assert metrics
        assert costs
        assert all(m.message_id == deltas["metrics_message_id"] for m in metrics)
        assert all(c.cost_usd == deltas["cost_usd"] for c in costs)


@pytest.mark.parametrize("loader", [claude_loader, codex_loader, pi_loader])
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
