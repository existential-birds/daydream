"""Truth table for forced answers versus permission to prompt.

resolve_gate returns True/False directly or None for interactive fallback.
Real fix-gate behavior is covered in deep_orchestrator/test_fix_gate_cleanup_and_precision.py.
"""

from __future__ import annotations

import pytest

from daydream.agent import resolve_gate


@pytest.mark.parametrize("assume,interactive,expected",
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
