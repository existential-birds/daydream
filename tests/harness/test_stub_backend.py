"""Tests for the shared deep-pipeline ``StubBackend`` harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.fake_clock import FakeClock
from tests.harness.stub_backend import StubBackend


async def test_stub_advances_the_injected_clock_on_a_runaway_fix_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runaway burst charges the injected clock instead of sleeping.

    The stub itself enforces no deadline -- it only advances the shared clock,
    exactly like a real slow backend consumes real time; the cut belongs to
    ``run_agent``.
    """
    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    stub = StubBackend(tmp_path)
    stub.runaway_single_fix_file = "api.py"
    stub.clock_advance = fake.advance
    stub.clock_advance_per_event_s = 2.0

    events = [e async for e in stub.execute(tmp_path, "Fix this issue:\nFile: api.py\nLine: 1")]

    assert len(events) == 500 and fake.monotonic_value == 2_000.0  # 1000 + 500 * 2.0
