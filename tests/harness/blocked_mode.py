"""Drive a blocked-mode protocol fixture from the test body.

A ``response_mode="block"`` fixture publishes an atomic ``entered`` marker naming
its pid and then blocks reading the ``release`` FIFO, so the host can act while
the child is provably mid-turn. These helpers own that handshake once: poll the
marker, de-duplicate pids, optionally replace a destination before the first
release, and release each blocked invocation. Releaser-thread failures are
collected and re-raised in the test body rather than dying silently in a daemon
thread.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for the fixture type
    from tests.harness.protocol_cli import ProtocolCli


def _entered_pid(fixture: ProtocolCli) -> int | None:
    """Read the pid from the atomic marker, or None while it is not readable yet."""
    try:
        entered = json.loads(fixture.entered.read_text(encoding="utf-8"))
        pid = entered["pid"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
    return pid if isinstance(pid, int) else None


def release_blocked_invocations(fixture: ProtocolCli, expected_pids: int, stop: threading.Event,
    failures: list[BaseException], *, replacement: tuple[Path, bytes] | None = None, timeout_s: float = 15,
) -> None:
    """Release each blocked invocation through the FIFO; record any failure.

    ``replacement`` overwrites the destination exactly once, before the first
    release, so the blocked child still observes the swapped bytes. Intended to
    run as a thread target; use :func:`blocking_releaser` for the full
    start/stop/join lifecycle.
    """
    seen: set[int] = set()
    replaced = False
    deadline = time.monotonic() + timeout_s
    try:
        while len(seen) < expected_pids:
            if stop.is_set():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("blocked protocol invocation did not enter")
            pid = _entered_pid(fixture)
            if pid is None or pid in seen:
                stop.wait(0.01)
                continue
            seen.add(pid)
            if not replaced and replacement is not None:
                destination, payload = replacement
                destination.write_bytes(payload)
                replaced = True
            with fixture.release.open("wb", buffering=0) as fifo:
                fifo.write(b"release")
    except BaseException as exc:  # surfaced by the test body
        failures.append(exc)


@contextmanager
def blocking_releaser(fixture: ProtocolCli, expected_pids: int, *, replacement: tuple[Path, bytes] | None = None,
    timeout_s: float = 15, join_timeout_s: float = 15, name: str = "protocol-cli-releaser",
) -> Iterator[threading.Thread]:
    """Run :func:`release_blocked_invocations` for the duration of the body.

    On a clean exit the releaser is stopped, joined, and required to be dead
    with no recorded failures.
    """
    stop = threading.Event()
    failures: list[BaseException] = []
    thread = threading.Thread(
        target=release_blocked_invocations, args=(fixture, expected_pids, stop, failures),
        kwargs={"replacement": replacement, "timeout_s": timeout_s}, name=name, daemon=True,
    )
    thread.start()
    try:
        yield thread
    finally:
        stop.set()
        thread.join(timeout=join_timeout_s)
    assert not thread.is_alive(), f"releaser {name} did not stop"
    assert failures == [], f"releaser {name} failed: {failures[0]!r}"


async def wait_for_entered(fixture: ProtocolCli, timeout_s: float = 30.0) -> int:
    """Await the blocked fixture's atomic entered marker; return its pid."""
    deadline = time.monotonic() + timeout_s
    while True:
        pid = _entered_pid(fixture)
        if pid is not None:
            return pid
        if time.monotonic() >= deadline:
            raise AssertionError("blocked protocol invocation never published entered")
        await anyio.sleep(0.02)
