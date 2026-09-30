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
import json
import os
from pathlib import Path

# A CLI that forks a `sleep` grandchild (which inherits the stdout pipe),
# reports readiness by printing "UP" after the fork, then hangs. A python CLI is
# used because bash defers SIGTERM while a child runs on macOS, which would
# stall every test on the TERMINATE_GRACE_S window. The print-after-fork makes
# these tests deterministic: the code under test is always exercised against a
# live process group, never racy against the CLI's fork.
GROUP_HOLDER_CLI = (
    "import subprocess, time; "
    "subprocess.Popen(['sleep', '30']); "
    "print('UP', flush=True); "
    "time.sleep(1000)"
)


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


def fd_count() -> int | None:
    """Return the open-descriptor count from ``/dev/fd``, or ``None`` if absent."""
    fd_dir = Path("/dev/fd")
    if not fd_dir.is_dir():
        return None
    return len(list(fd_dir.iterdir()))


async def wait_for_fd_baseline(baseline: int | None, *, timeout_s: float = 30.0) -> None:
    """Await the open-descriptor count's return to *baseline*.

    A readiness wait, not a fixed sleep: the loop exits the moment the count
    matches and ``timeout_s`` is a failure bound. Matches ``None`` when
    ``/dev/fd`` is unavailable, so a caller that cannot observe the count does
    not wait.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while fd_count() != baseline:
        if loop.time() > deadline:
            raise TimeoutError(
                f"fd count {fd_count()} did not return to baseline {baseline} after {timeout_s}s"
            )
        await asyncio.sleep(0.01)


async def wait_for_process_ids(path: Path, *, timeout_s: float = 5.0) -> dict[str, int]:
    """Await the ``direct``/``grandchild`` pid JSON a fake blocking CLI publishes.

    Times out with an :class:`AssertionError` naming *path*, so a failed wait is
    reported as a test failure rather than a bare timeout.
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded.get("direct"), int) and isinstance(loaded.get("grandchild"), int):
                return {"direct": loaded["direct"], "grandchild": loaded["grandchild"]}
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            pass
        await asyncio.sleep(0.01)
    raise AssertionError(f"blocking fake gh did not publish {path}")
