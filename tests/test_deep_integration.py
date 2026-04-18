"""Deep-mode backend parity integration tests (D-38..D-40).

Most tests are xfail(strict=True) until plan 05-10 wires the orchestrator
against both Claude- and Codex-shaped mock backends.
``test_existing_tests_still_collect`` is NOT xfail — it is a live
regression guard that enforces D-40: the 50+ existing tests must keep
importing cleanly while Phase 5 lands.
"""

from pathlib import Path

import pytest


@pytest.mark.xfail(reason="Wave 5 plan 05-10 not yet implemented", strict=True)
def test_claude_shape_backend(multi_stack_target: Path) -> None:
    """D-38: run_deep works with Claude-shaped MockBackend."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-10 not yet implemented", strict=True)
def test_codex_shape_backend(multi_stack_target: Path) -> None:
    """D-38: run_deep works with Codex-shaped MockBackend (no agents kwarg support)."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-10 not yet implemented", strict=True)
def test_phase_primitives_unmodified() -> None:
    """D-39: phase_understand_intent, phase_alternative_review, phase_parse_feedback invoked unchanged."""
    raise NotImplementedError


def test_existing_tests_still_collect() -> None:
    """D-40: smoke check that existing tests still import."""
    import tests.test_cli  # noqa: F401
    import tests.test_integration  # noqa: F401
    import tests.test_loop  # noqa: F401
    import tests.test_phases  # noqa: F401
