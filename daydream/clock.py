"""Shared process-local monotonic clock for deadlines and budgets.

Call clock.monotonic() through the module, never a from-import, so one test
monkeypatch advances every consumer without sleeping.
"""

from __future__ import annotations

import time


def monotonic() -> float:
    """Return the current monotonic time in seconds."""
    return time.monotonic()
