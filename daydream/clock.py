"""The single process-local monotonic clock seam.

Every deadline and budget read goes through :func:`monotonic` so a test can
install a fake clock and advance time instead of sleeping. Consumers must call
it as ``clock.monotonic()`` (a module attribute), never via a ``from`` import,
so one monkeypatch reaches every reader.
"""

from __future__ import annotations

import time


def monotonic() -> float:
    """Return the current monotonic time in seconds."""
    return time.monotonic()
