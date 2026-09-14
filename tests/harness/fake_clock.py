"""Fake monotonic clock test double for the ``daydream.clock`` seam.

:class:`FakeClock` replaces ``daydream.clock.monotonic`` with a value the test
controls via :meth:`advance`, so deadline/budget tests observe expiry
deterministically without sleeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from daydream import clock


@dataclass
class FakeClock:
    """A controllable monotonic clock; never sleeps."""

    monotonic_value: float = 0.0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeClock:
        """Patch ``daydream.clock.monotonic`` to this clock; return ``self``."""
        monkeypatch.setattr(clock, "monotonic", self.monotonic)
        return self

    def monotonic(self) -> float:
        """Current fake time in seconds."""
        return self.monotonic_value

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.monotonic_value += seconds
