"""Shared subprocess lifecycle helpers for the CLI backends (codex, pi)."""

from __future__ import annotations

import asyncio
import os
import signal

import anyio

from daydream.backends import _parsed_nonnegative_float

# Reset idle detection on each complete line. Codex can remain silent for a
# whole generation/tool call, so use 2700s (2.7x its largest observed turn).
# Pi token updates permit a shorter response window below.
DEFAULT_STREAM_IDLE_TIMEOUT_S = 2700.0
DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S = 600.0
STREAM_IDLE_TIMEOUT_ENV = "DAYDREAM_STREAM_IDLE_TIMEOUT_S"

# Grace between SIGTERM and SIGKILL when reaping a subprocess. A cooperative CLI
# exits on SIGTERM long before this; the window only bounds how long we wait on a
# child that ignores it before forcing the kill.
TERMINATE_GRACE_S = 5.0


class StreamStalledError(Exception):
    """Raised when a backend CLI produces no stdout for the idle window.

    A stalled request may be a transient provider/network failure, so the
    normal bounded backend retry loop gets a chance to re-arm the subprocess.
    """

    retryable = True
    max_retries = 1

    def __init__(self, cli: str, timeout_s: float) -> None:
        super().__init__(
            f"{cli} CLI produced no output for {timeout_s:g}s; the stream is stalled. "
            f"The subprocess was terminated. Set {STREAM_IDLE_TIMEOUT_ENV} to widen the "
            "window (0 disables idle detection entirely)."
        )
        self.cli = cli
        self.timeout_s = timeout_s


def stream_idle_timeout_s(
    *, default: float = DEFAULT_STREAM_IDLE_TIMEOUT_S
) -> float | None:
    """Read DAYDREAM_STREAM_IDLE_TIMEOUT_S; zero disables detection.

    Malformed, nonfinite, or negative values warn and use the default. A blank
    value stays silent (absent), matching the rest of the env handling.
    """
    if not os.environ.get(STREAM_IDLE_TIMEOUT_ENV):
        return default
    value = _parsed_nonnegative_float(os.environ, STREAM_IDLE_TIMEOUT_ENV, default)
    return None if value == 0 else value


async def readline_with_idle_timeout(
    stdout: asyncio.StreamReader, *, cli: str, timeout_s: float | None
) -> bytes:
    """Read one line, restarting the idle window after each completed line.

    Raise StreamStalledError on timeout; the caller terminates its child in finally.
    """
    if timeout_s is None:
        return await stdout.readline()
    try:
        async with asyncio.timeout(timeout_s):
            return await stdout.readline()
    except TimeoutError as exc:
        raise StreamStalledError(cli, timeout_s) from exc


async def terminate_process(
    proc: asyncio.subprocess.Process, timeout: float | None = None
) -> None:
    """Shielded, idempotent process-group teardown: SIGTERM, grace, SIGKILL, reap, close.

    CLI children use start_new_session=True, so pgid == pid. Signal the group
    before the direct child dies: macOS may deny signaling orphaned grandchildren,
    which would retain pipe writers. Close the underlying transport explicitly
    because surviving writers can prevent StreamReader EOF.

    Shield the entire teardown from level-triggered AnyIO cancellation; otherwise
    a cancelled generator's first await can abort reaping or SIGKILL. Repeated
    calls skip terminate/wait after the child is reaped and safely close pipes.
    """
    grace = TERMINATE_GRACE_S if timeout is None else timeout
    with anyio.CancelScope(shield=True):
        _kill_process_group(proc, signal.SIGTERM)
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace)
            except asyncio.TimeoutError:
                # The CLI ignored SIGTERM. SIGKILL the group while it is still
                # alive so the grandchildren die with it — killing just the
                # direct child first would orphan them and a later group kill
                # would fail with EPERM.
                _kill_process_group(proc, signal.SIGKILL)
                proc.kill()
                await proc.wait()
            else:
                # The CLI exited on SIGTERM; reap any surviving grandchildren.
                _kill_process_group(proc, signal.SIGKILL)
        _close_process_io(proc)


def _kill_process_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Signal an integer process-group id; ignore vanished or macOS-orphaned groups."""
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _close_process_io(proc: asyncio.subprocess.Process) -> None:
    """Close the fd-owning transport and stdin, even without StreamReader EOF."""
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        transport.close()
    if proc.stdin is not None:
        proc.stdin.close()


async def cancel_processes(processes: list[asyncio.subprocess.Process]) -> None:
    """Shield and join teardown of a process snapshot, releasing all groups and pipes."""
    snapshot = list(processes)
    with anyio.CancelScope(shield=True):
        await asyncio.gather(*(terminate_process(process) for process in snapshot))
