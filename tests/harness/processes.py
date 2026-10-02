"""Readiness waits for cancellation tests. Reparented grandchildren can remain zombies that answer
killpg(pgid, 0) after the direct child is reaped. Poll group disappearance to tolerate kernel
reaping delay under load.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

# Fork a sleep grandchild inheriting stdout, then print readiness and hang. Python avoids Bash
# deferring SIGTERM on macOS; readiness after the fork ensures tests exercise a live process group.
GROUP_HOLDER_CLI = ("import subprocess, time; "
    "subprocess.Popen(['sleep', '30']); "
    "print('UP', flush=True); "
    "time.sleep(1000)"
)


async def wait_for_process_group_gone(pgid: int, *, timeout_s: float = 30.0) -> None:
    """Await group disappearance or PermissionError from a recycled PGID. timeout_s is the failure
    bound, not a synchronization delay; exceeding it raises TimeoutError.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            # EPERM means a foreign-UID process recycled the PGID after our group exited.
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
    """Await the descriptor count returning to baseline; timeout_s bounds failure. If /dev/fd is
    unavailable, None matches immediately.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while fd_count() != baseline:
        if loop.time() > deadline:
            raise TimeoutError(f"fd count {fd_count()} did not return to baseline {baseline} after {timeout_s}s")
        await asyncio.sleep(0.01)


async def wait_for_process_ids(path: Path, *, timeout_s: float = 5.0) -> dict[str, int]:
    """Await the fake CLI's direct/grandchild PID JSON; timeout raises AssertionError naming the path.

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
