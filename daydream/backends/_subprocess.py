# daydream/backends/_subprocess.py
"""Shared subprocess lifecycle helpers for the CLI backends (codex, pi)."""

from __future__ import annotations

import asyncio


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
