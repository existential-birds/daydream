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


def patch_retry_sleep(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> list[float]:
    """Replace the retry backoff sleep with a recording, clock-advancing stub.

    Patches ``daydream.agent.anyio.sleep`` so a retry ladder consumes injected
    time instead of wall time: each call appends the delay to the returned list
    and advances *clock* by the same amount. The list is the assertion seam for
    deterministic backoff tests.
    """
    delays: list[float] = []

    async def _sleeper(delay: float) -> None:
        delays.append(delay)
        clock.advance(delay)

    monkeypatch.setattr("daydream.agent.anyio.sleep", _sleeper)
    return delays
