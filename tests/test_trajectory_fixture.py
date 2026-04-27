"""Cross-test bleed prevention for the recorder ContextVar (CORE-10 / D-17).

These two tests cooperate: ``test_a_leak_recorder_var`` deliberately sets
``_RECORDER_VAR`` and never unsets it; ``test_b_recorder_var_starts_clean``
then asserts the ContextVar reads as ``None`` at test entry. Without the
suite-wide autouse ``_reset_trajectory_recorder`` fixture in
``tests/conftest.py``, this asserts FAILS — proving the fixture is the
sole guard against cross-test bleed.

Naming: pytest collects in source order; ``a_`` runs before ``b_``.
"""

from __future__ import annotations

from typing import Any

# Import the private ContextVar through a leak helper so the test does
# not violate the public-only-import rule documented in the trajectory
# module (D-10) for production callers — a test deliberately inspecting
# the contextvar primitive is the documented exception.
from daydream.trajectory import _RECORDER_VAR, get_current_recorder


class _SentinelRecorder:
    """Stand-in for an active TrajectoryRecorder — never used as one.

    The test only needs ``_RECORDER_VAR.set(value)`` to be observable to
    the next test if the autouse reset fixture is missing.
    """


def test_a_leak_recorder_var() -> None:
    """Set the ContextVar without cleanup — exercises the leak path.

    If the autouse ``_reset_trajectory_recorder`` fixture is in place in
    ``tests/conftest.py``, the AFTER-yield reset call clears this leak
    before the next test starts. If the fixture is absent, the leak
    propagates and ``test_b_recorder_var_starts_clean`` fails.
    """
    sentinel: Any = _SentinelRecorder()
    _RECORDER_VAR.set(sentinel)
    assert get_current_recorder() is sentinel


def test_b_recorder_var_starts_clean() -> None:
    """Assert the ContextVar is None at the START of the test.

    The autouse fixture's BEFORE-yield reset gives every test a clean
    slate. Without the fixture, ``test_a_leak_recorder_var`` above
    leaves the sentinel installed and this assertion fails.
    """
    assert get_current_recorder() is None
