"""Shared process-group waits for subprocess-cancellation tests.

A group-signal kill reaps the direct child synchronously, but a grandchild
reparented to PID 1 lingers as a zombie until init reaps it -- and a zombie
still answers ``killpg(pgid, 0)``. Asserting ``ProcessLookupError`` in the same
event-loop tick as the kill therefore races the kernel's reap: on a loaded host
(CI runners, parallel suites) the window is wide enough to fail intermittently.
Polling until the group is gone makes the assertion deterministic -- the
observable outcome is "no process remains in the group", not "the group vanished
by the next instruction".
"""

from __future__ import annotations

import asyncio
import os


async def wait_for_process_group_gone(pgid: int, *, timeout_s: float = 30.0) -> None:
    """Await *pgid*'s disappearance (a readiness wait, not a fixed sleep).

    The loop exits the moment ``killpg`` raises ``ProcessLookupError`` or
    ``PermissionError`` (a recycled pgid); ``timeout_s`` is a failure bound, not
    a synchronization delay. Raises ``TimeoutError`` if the group outlives it.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            # EPERM means the pgid was recycled by a foreign-uid process,
            # i.e. our same-uid group exited.
            return
        if loop.time() > deadline:
            raise TimeoutError(f"process group {pgid} still alive after {timeout_s}s")
        await asyncio.sleep(0.01)


async def wait_for_process_group_exit(pgid: int) -> None:
    """Assertion-flavored :func:`wait_for_process_group_gone` for cancellation tests."""
    try:
        await wait_for_process_group_gone(pgid, timeout_s=2.0)
    except TimeoutError as exc:
        raise AssertionError(f"process group {pgid} survived cancellation") from exc
