# daydream/backends/_subprocess.py
"""Shared subprocess lifecycle helpers for the CLI backends (codex, pi)."""

from __future__ import annotations

import asyncio
import logging
import math
import os

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


class StreamStalledError(Exception):
    """Raised when a backend CLI produces no stdout for the idle window.

    ``retryable`` is ``True``: a stalled stream is the most common symptom of a
    flaky endpoint, and every provider (Anthropic included) drops connections
    often enough that treating a stall as terminal makes one blip kill a whole
    run. ``agent.run_agent``'s retry loop re-arms a fresh subprocess per attempt,
    which is exactly what recovers from a dead connection. Each attempt is still
    bounded by the idle window, so the worst case is ``attempts × window`` of
    dead air only when the endpoint stays dead for the entire retry budget.
    """

    retryable = True

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


async def terminate_process(proc: asyncio.subprocess.Process, timeout: float = 5.0) -> None:
    """SIGTERM *proc*, wait up to *timeout* seconds, then SIGKILL if still running."""
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def cancel_processes(processes: list[asyncio.subprocess.Process]) -> None:
    """Cancel every tracked subprocess: SIGTERM all first, then wait/SIGKILL each."""
    snapshot = list(processes)
    for process in snapshot:
        process.terminate()
    for process in snapshot:
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
