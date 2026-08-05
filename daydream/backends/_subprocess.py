# daydream/backends/_subprocess.py
"""Shared subprocess lifecycle helpers for the CLI backends (codex, pi)."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal

import anyio

logger = logging.getLogger(__name__)

# Idle-stall detection for the CLI backends' stdout stream.
#
# Fires on the ABSENCE of stream activity, never on slow output: the window
# restarts on every line, so a CLI that trickles tokens is never interrupted.
#
# The default is calibrated from two measurements (July 2026):
#
# 1. 403 archived runs (~/.daydream/archive/runs). Grouping ATIF steps by
#    ``extra.subtrajectories`` gives the wall-clock span of a single backend
#    turn, which upper-bounds the silent window inside it. Largest healthy
#    within-invocation span: codex 989.4s (41 runs), pi 1925.0s (186 runs).
# 2. Live capture of both CLIs with per-line arrival timestamps. pi streams
#    token-level ``message_update`` deltas (16901 lines over a 259s reasoning
#    block; largest gap 11.7s) and ``tool_execution_update`` per output chunk,
#    so its only silent construct is an output-silent tool call. codex 0.144.6
#    ``--experimental-json`` emits NO ``item.updated`` at all: it was silent for
#    the full 151s of a 7038-token generation and for the full 120s of a chatty
#    ticking shell command. codex is therefore the binding constraint — its
#    stream is legitimately dead for the whole duration of any single tool call
#    or generation block.
#
# 2700s is 2.7x codex's largest observed turn span and 1.4x pi's, and it sits
# deliberately ABOVE ``config.DEFAULT_WALL_BUDGET_S`` (1800s) so it can never
# act as a second, shorter turn cap on phases that already carry a wall budget.
# It is a backstop for the turns nothing else bounds — the improve phases run
# with no wall budget at all — and for stalls that outlive the process.
DEFAULT_STREAM_IDLE_TIMEOUT_S = 2700.0
STREAM_IDLE_TIMEOUT_ENV = "DAYDREAM_STREAM_IDLE_TIMEOUT_S"

# Grace between SIGTERM and SIGKILL when reaping a subprocess. A cooperative CLI
# exits on SIGTERM long before this; the window only bounds how long we wait on a
# child that ignores it before forcing the kill.
TERMINATE_GRACE_S = 5.0


class StreamStalledError(Exception):
    """Raised when a backend CLI produces no stdout for the idle window.

    ``retryable`` is ``False``: the idle window is a terminal bound for a
    subprocess invocation, so a persistent stalled stream cannot consume the
    generic retry budget.
    """

    retryable = False

    def __init__(self, cli: str, timeout_s: float) -> None:
        super().__init__(
            f"{cli} CLI produced no output for {timeout_s:g}s; the stream is stalled. "
            f"The subprocess was terminated. Set {STREAM_IDLE_TIMEOUT_ENV} to widen the "
            "window (0 disables idle detection entirely)."
        )
        self.cli = cli
        self.timeout_s = timeout_s


def stream_idle_timeout_s() -> float | None:
    """Resolve the stdout idle timeout in seconds.

    Returns ``None`` when idle detection is disabled — an explicit ``0`` in
    ``$DAYDREAM_STREAM_IDLE_TIMEOUT_S``. A malformed, non-finite, or negative
    value logs a warning and falls back to the default (mirroring
    ``DAYDREAM_PI_RETRY_ATTEMPTS`` handling in :mod:`daydream.backends.pi`).
    """
    raw = os.environ.get(STREAM_IDLE_TIMEOUT_ENV)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "%s=%r is not a valid float; using default %g",
                STREAM_IDLE_TIMEOUT_ENV, raw, DEFAULT_STREAM_IDLE_TIMEOUT_S,
            )
        else:
            if not math.isfinite(value):
                logger.warning(
                    "%s=%r is not finite; using default %g",
                    STREAM_IDLE_TIMEOUT_ENV, raw, DEFAULT_STREAM_IDLE_TIMEOUT_S,
                )
            elif value < 0:
                logger.warning(
                    "%s=%r is negative; using default %g",
                    STREAM_IDLE_TIMEOUT_ENV, raw, DEFAULT_STREAM_IDLE_TIMEOUT_S,
                )
            elif value == 0:
                return None
            else:
                return value
    return DEFAULT_STREAM_IDLE_TIMEOUT_S


async def readline_with_idle_timeout(
    stdout: asyncio.StreamReader, *, cli: str, timeout_s: float | None
) -> bytes:
    """Read one line from *stdout*, bounded by the idle window.

    The window covers the wait for a single line, so it restarts on every line
    that arrives: only a stream that goes fully silent for *timeout_s* trips it.

    Raises:
        StreamStalledError: When no byte arrives within *timeout_s*. The caller
            terminates the subprocess in its ``finally`` block.
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
    """SIGTERM *proc*, wait up to *timeout* seconds, then SIGKILL if still running.

    The signals reach the whole process group, not just the direct child: the
    CLI is spawned with ``start_new_session=True``, making it the session leader
    (``pgid == pid``), so a group signal reaches shell tools the CLI spawned as
    grandchildren. The group is signalled *before* the direct child can die and
    orphan them — a ``killpg`` on a group whose members have been reparented to
    PID 1 fails with ``PermissionError`` on macOS, leaving the grandchildren to
    hold the pipe write ends open (the Errno 24 leak).

    The subprocess pipes are closed explicitly at the end: ``proc.stdout`` /
    ``proc.stderr`` are ``StreamReader`` with no public ``close()``, and the fd
    owner is ``proc._transport``, whose ``close()`` releases the pipe read ends
    regardless of EOF (a surviving grandchild keeps EOF from ever arriving).

    Shielded from cancellation. The backend reads its subprocess inside an async
    generator, so a wall-budget scope firing mid-read unwinds the ``CancelledError``
    straight into the generator's ``finally`` — here — with the parent scope still
    cancelled. anyio cancellation is level-triggered, so without the shield the
    first ``await`` below re-raises before the wait/SIGKILL/reap can run: the child
    is signalled but never reaped, and a SIGTERM-ignoring child is not even killed.
    The shield lets this teardown run to completion before the cancel resumes.

    Idempotent: on a second call the process is already reaped (``returncode`` is
    set), so the terminate/wait sequence is skipped and the group signal and pipe
    closes are no-ops.
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
    """Send *sig* to every process in *proc*'s process group (idempotent).

    ``ProcessLookupError`` (group already gone — double-call, or the direct
    child's death took it down) and ``PermissionError`` (the group holds
    members orphaned into another session after the CLI died; macOS refuses the
    signal) are both swallowed. ``proc.pid`` may be a non-int on test mocks,
    which are skipped.
    """
    pid = getattr(proc, "pid", None)
    if isinstance(pid, int):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _close_process_io(proc: asyncio.subprocess.Process) -> None:
    """Close the subprocess transport and stdin, releasing the pipe fds.

    ``proc.stdout``/``proc.stderr`` are ``asyncio.StreamReader`` with no public
    ``close()`` — the fd owner is ``proc._transport``, whose ``close()``
    releases the pipe read ends regardless of EOF. ``proc.stdin`` is an
    ``asyncio.StreamWriter`` whose ``close()`` is a no-op after EOF.
    """
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        transport.close()
    if proc.stdin is not None:
        proc.stdin.close()


async def cancel_processes(processes: list[asyncio.subprocess.Process]) -> None:
    """Cancel every tracked subprocess, reaping its group and pipes.

    Delegates to :func:`terminate_process` per process: SIGTERM, grace, SIGKILL,
    then group signal and pipe close. The observable contract is that every
    process group is gone and every pipe fd released, idempotent on double-call.

    Shielded for the same reason as :func:`terminate_process`: the reap must
    finish even when the caller is already being cancelled.
    """
    snapshot = list(processes)
    with anyio.CancelScope(shield=True):
        await asyncio.gather(*(terminate_process(process) for process in snapshot))
