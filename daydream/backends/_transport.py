"""Injectable subprocess/JSONL transport for the CLI backends (codex, pi, osprey).

One owner for spawn, stdin policy, idle-timeout line reads, stderr handling,
exit-code surfacing, and shielded teardown, built on the primitives in
:mod:`daydream.backends._subprocess`. Backends keep all protocol mapping: the
transport yields raw decoded lines and surfaces only the exit code, so backend
error messages stay byte-identical to what they were before the transport.
"""

from __future__ import annotations

import asyncio
import enum
from collections.abc import AsyncIterator, Callable

import anyio

from daydream.backends._subprocess import (
    cancel_processes,
    readline_with_idle_timeout,
    terminate_process,
)


class StdinMode(enum.Enum):
    """How the child's stdin is wired at spawn."""

    DEVNULL = enum.auto()
    PIPE = enum.auto()


class StderrPolicy(enum.Enum):
    """Where the child's stderr goes.

    ``MERGE_INTO_STDOUT`` folds stderr into the JSONL stream (codex, pi).
    ``DRAIN_TASK`` keeps stderr a separate pipe drained by a background task
    (osprey), whose lines are handed to ``stderr_sink``; the backend awaits
    :meth:`drain_finished` after ``wait()``/``terminate()`` so the drain task
    can never outlive the transport.
    """

    MERGE_INTO_STDOUT = enum.auto()
    DRAIN_TASK = enum.auto()


class TransportExitError(Exception):
    """The child exited non-zero.

    The transport only surfaces the code and the caller-noted diagnostics —
    the backend formats its own user-visible message from these.
    """

    def __init__(self, cli: str, returncode: int, diagnostics: list[str]) -> None:
        self.cli = cli
        self.returncode = returncode
        self.diagnostics = diagnostics


class CliTransport:
    """Spawn a CLI subprocess and stream its stdout as decoded JSONL lines.

    Transport-internal errors propagate as raised: :class:`StreamStalledError`
    on stream silence, :class:`ValueError` on oversized lines, ``OSError`` from
    spawn — the caller maps each to its own backend error type. No fallbacks.
    """

    def __init__(
        self,
        cli: str,
        argv: list[str],
        *,
        stdin_mode: StdinMode = StdinMode.DEVNULL,
        stdin_data: bytes | None = None,
        stderr_policy: StderrPolicy = StderrPolicy.MERGE_INTO_STDOUT,
        stderr_sink: Callable[[str], None] | None = None,
        diagnostics_sink: Callable[[str], None] | None = None,
        limit: int,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        if stdin_mode is StdinMode.PIPE and stdin_data is None:
            raise ValueError("stdin_mode=PIPE requires stdin_data")
        self._cli = cli
        self._argv = argv
        self._stdin_mode = stdin_mode
        self._stdin_data = stdin_data
        self._stderr_policy = stderr_policy
        self._stderr_sink = stderr_sink
        self._diagnostics: list[str] = []
        self._diagnostics_sink = diagnostics_sink
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
            if self._stdin_mode is StdinMode.PIPE
            else asyncio.subprocess.DEVNULL
        )
        stderr = (
            asyncio.subprocess.STDOUT
            if self._stderr_policy is StderrPolicy.MERGE_INTO_STDOUT
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

        if self._stdin_mode is StdinMode.PIPE:
            stdin_writer = proc.stdin
            if stdin_writer is None:  # pragma: no cover - PIPE guarantees stdin
                raise OSError("child stdin is not writable despite StdinMode.PIPE")
            stdin_writer.write(self._stdin_data or b"")
            stdin_writer.close()
            self.stdin_closed = True

        if self._stderr_policy is StderrPolicy.DRAIN_TASK and proc.stderr is not None:
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
        """Yield decoded, stripped stdout lines under per-line idle windows.

        The callable is invoked per line, so a dual-window policy (response vs
        tool-active) can switch mid-stream. Silence within the window raises
        :class:`StreamStalledError` via the shared primitive; an oversized line
        raises ``ValueError`` unchanged.
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
            yield raw.decode().strip()

    def note_diagnostic(self, line: str) -> None:
        """Record *line* as a diagnostic via the caller's sink."""
        self._diagnostics.append(line)
        if self._diagnostics_sink is not None:
            self._diagnostics_sink(line)

    async def wait(self) -> int:
        """Await the child and return its exit code.

        Raises:
            TransportExitError: On a non-zero exit; ``.returncode`` and
                ``.diagnostics`` carry the code and caller-noted lines.
        """
        if self._proc is None:
            raise RuntimeError("transport not started; call start() first")
        returncode = await self._proc.wait()
        if returncode != 0:
            raise TransportExitError(self._cli, returncode, self._diagnostics)
        return returncode

    async def drain_finished(self) -> None:
        """Await the stderr drain task (a no-op under MERGE_INTO_STDOUT).

        Shielded so a drain awaited in a backend's teardown ``finally`` still
        completes when the caller's scope is already cancelled — the same
        shield the cancel-sweep relies on.
        """
        if self._drain_task is not None:
            task, self._drain_task = self._drain_task, None
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task)

    async def terminate(self) -> None:
        """Group-signal, reap, and close pipes; shielded and idempotent."""
        if self._proc is not None:
            await terminate_process(self._proc)

    @classmethod
    async def cancel_all(cls, transports: list[CliTransport]) -> None:
        """Cancel every tracked transport, mirroring
        :func:`daydream.backends._subprocess.cancel_processes`.

        Delegates the per-transport work to :meth:`terminate_process` via
        :func:`cancel_processes` over the tracked live processes, so the reap is
        shielded from the caller's cancellation and idempotent on double-call —
        the same contract the backends' ``cancel()`` delegation relies on.
        """
        await cancel_processes([
            proc for t in transports for proc in t.processes if proc.returncode is None
        ])
        with anyio.CancelScope(shield=True):
            # Iterate a snapshot: a backend's teardown ``finally`` removes its
            # transport from this caller-owned list in place while we await, so
            # a live iteration can raise 'list changed size during iteration'.
            for t in list(transports):
                await t.drain_finished()
