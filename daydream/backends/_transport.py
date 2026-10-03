"""Own CLI spawning, stdin, idle reads, stderr drains, exit handling, and shielded teardown.
Adapters interpret protocols/errors; transport exposes decoded lines and exit codes without fallbacks.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anyio

from daydream.backends._subprocess import (
    cancel_processes,
    readline_with_idle_timeout,
    terminate_process,
)


class CliTransport:
    """Spawn/stream decoded stdout; propagate stalls, oversized lines, and spawn errors for adapter handling."""

    def __init__(
        self,
        cli: str,
        argv: list[str],
        *,
        stdin_data: bytes | None = None,
        stderr_sink: Callable[[str], None] | None = None,
        decode_errors: str = "strict",
        limit: int,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self._cli = cli
        self._argv = argv
        self._stdin_data = stdin_data
        self._stderr_sink = stderr_sink
        # Backend decode policy: codex/pi were strict pre-transport; osprey
        # decoded with errors="replace" and must keep doing so (see the
        # backend's ``decode_errors="replace"`` construction below).
        self._decode_errors = decode_errors
        self._drain_task: asyncio.Task[None] | None = None
        self.processes: list[asyncio.subprocess.Process] = []
        self._proc: asyncio.subprocess.Process | None = None
        self._spawn_kwargs: dict[str, object] = {
            "limit": limit,
            "env": env,
            "cwd": cwd,
        }
        self.stdin_closed = False

    async def start(self) -> None:
        """Spawn the child, write+close stdin when piped, start stderr drain."""
        stdin = (
            asyncio.subprocess.PIPE
            if self._stdin_data is not None
            else asyncio.subprocess.DEVNULL
        )
        stderr = (
            asyncio.subprocess.STDOUT
            if self._stderr_sink is None
            else asyncio.subprocess.PIPE
        )
        # Spawn OSError propagates: the caller maps it to its backend error type.
        proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            start_new_session=True,
            **self._spawn_kwargs,  # type: ignore[arg-type]
        )
        self._proc = proc
        self.processes.append(proc)

        if self._stdin_data is not None:
            stdin_writer = proc.stdin
            if stdin_writer is None:  # pragma: no cover - PIPE guarantees stdin
                raise OSError("child stdin is not writable despite piped input")
            stdin_writer.write(self._stdin_data or b"")
            stdin_writer.close()
            self.stdin_closed = True

        if proc.stderr is not None:
            self._drain_task = asyncio.create_task(self._drain_stderr(proc.stderr))

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        while True:
            try:
                raw = await stderr.readline()
            except ValueError:
                # ``StreamReader.readline`` clears an over-limit unterminated
                # line before raising. Stderr is diagnostic-only, so note the
                # discard and keep draining instead of failing an otherwise
                # valid JSONL session during teardown.
                if self._stderr_sink is not None:
                    self._stderr_sink("stderr diagnostic line exceeded stream limit and was discarded")
                continue
            if not raw:
                break
            line = raw.decode(errors="replace").strip()
            if line and self._stderr_sink is not None:
                self._stderr_sink(line)

    @property
    def returncode(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.returncode

    async def lines(
        self, timeout_for_line: Callable[[], float | None]
    ) -> AsyncIterator[str]:
        """Read stripped lines with a fresh response/tool-active idle window per line. Propagate stalls
        and oversized-line ValueError; decoding stays strict unless the adapter requests replacement.
        """
        if self._proc is None:
            raise RuntimeError("transport not started; call start() first")
        stdout = self._proc.stdout
        if stdout is None:  # pragma: no cover - stdout is always PIPE
            return
        while True:
            raw = await readline_with_idle_timeout(
                stdout, cli=self._cli, timeout_s=timeout_for_line()
            )
            if not raw:
                return
            yield raw.decode(errors=self._decode_errors).strip()

    async def wait(self) -> int:
        """Reap the child and return its actual status; adapters interpret failures."""
        if self._proc is None:
            raise RuntimeError("transport not started; call start() first")
        await self._proc.wait()
        assert self._proc.returncode is not None
        return self._proc.returncode

    async def drain_finished(self) -> None:
        """Shield and join the stderr task without masking the caller's primary error."""
        if self._drain_task is not None:
            task, self._drain_task = self._drain_task, None
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task, return_exceptions=True)

    async def terminate(self) -> None:
        """Group-signal, reap, and close pipes; shielded and idempotent."""
        if self._proc is not None:
            await terminate_process(self._proc)

    @classmethod
    async def cancel_all(cls, transports: list[CliTransport]) -> None:
        """Shield and join teardown and stderr drains for every tracked live process."""
        await cancel_processes([
            proc for t in transports for proc in t.processes if proc.returncode is None
        ])
        with anyio.CancelScope(shield=True):
            # Iterate a snapshot: a backend's teardown ``finally`` removes its
            # transport from this caller-owned list in place while we await, so
            # a live iteration can raise 'list changed size during iteration'.
            for t in list(transports):
                await t.drain_finished()


# Number of captured non-JSON lines printed in a PROCESS_EXIT message. The
# count reported in the message is the number of lines actually printed, not
# the size of the capture window.
PROCESS_EXIT_EXCERPT_MAX_LINES = 10


async def teardown(transport: CliTransport, transports: list[CliTransport]) -> None:
    """Shielded, idempotent terminate/drain/remove for a tracked transport.

    Osprey invokes this before its finally block as well as during cleanup.
    """
    await transport.terminate()
    await transport.drain_finished()
    if transport in transports:
        transports.remove(transport)


def process_exit_message(*, display: str, returncode: int, lines: list[str]) -> str:
    """Render Codex/Pi exit diagnostics with the last ten lines or a no-output fallback.

    The header counts printed lines; Osprey owns its different message shape.
    """
    if lines:
        shown = lines[-PROCESS_EXIT_EXCERPT_MAX_LINES:]
        detail = (
            f"\n{display} CLI output (last {len(shown)} non-JSON lines):\n"
            + "\n".join(shown)
        )
    else:
        detail = f"\n(no non-JSON output captured — {display.lower()} may have crashed before writing to stdout)"
    return f"{display} CLI exited with return code {returncode}.{detail}"


def write_temp_json_schema(schema: dict[str, Any], *, prefix: str) -> str:
    """Write a caller-owned temporary schema; unlink on serialization failure, otherwise after child exit."""
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", prefix=prefix, delete=False
    )
    completed = False
    try:
        json.dump(schema, handle, separators=(",", ":"))
        handle.write("\n")
        completed = True
        return handle.name
    finally:
        try:
            handle.close()
        finally:
            if not completed:
                Path(handle.name).unlink(missing_ok=True)
