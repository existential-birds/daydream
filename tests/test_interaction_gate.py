"""Pure resolution truth table for the orthogonal interaction gate.

``resolve_gate`` collapses two orthogonal axes — *assume* (a forced
yes/no answer, e.g. ``--yes``) and *interactivity* (may we block on
stdin?) — into a single decision: ``True``/``False`` to use directly, or
``None`` to fall back to an interactive prompt. This is a fast pure unit
test that *supplements* the real-path fix-gate test in
``test_deep_orchestrator.py`` (it never replaces it).
"""

from __future__ import annotations

import pytest

from daydream.agent import resolve_gate


@pytest.mark.parametrize(
    "assume,interactive,expected",
    [
        (None, True, None),  # interactive, no assumption -> prompt
        (None, False, False),  # unattended, no assumption -> safe default (decline)
        ("yes", True, True),  # explicit yes wins even on a TTY
        ("yes", False, True),  # CI --yes -> unattended auto-apply
        ("no", False, False),  # explicit no
        ("no", True, False),  # explicit no wins even on a TTY
    ],
)
def test_resolve_gate(assume: str | None, interactive: bool, expected: bool | None) -> None:
    assert resolve_gate(assume=assume, interactive=interactive, safe_default=False) is expected


def test_resolve_gate_safe_default_true_when_unattended() -> None:
    # A gate whose unattended safe default is "yes" (e.g. auto-commit) returns
    # True when there's no assumption and we cannot prompt.
    assert resolve_gate(assume=None, interactive=False, safe_default=True) is True
