"""Injectable subprocess/JSONL transport for the CLI backends (codex, pi, osprey).

One owner for spawn, stdin policy, idle-timeout line reads, stderr handling,
exit-code surfacing, and shielded teardown, built on the primitives in
:mod:`daydream.backends._subprocess`. The module also owns the shared
reap / exit-check / teardown sequence and the exit-diagnostic message builder
(:func:`reap`, :func:`raise_for_exit`, :func:`teardown`,
:func:`process_exit_message`), so each CLI adapter contributes only its own
wording and parameters. Backends keep all protocol mapping: the transport
yields raw decoded lines and surfaces only the exit code, so each backend
owns its error wording. The shared PROCESS_EXIT builder reports the number
of lines actually printed.
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

    The backend formats its own user-visible message from the exit code.
    """

    def __init__(self, cli: str, returncode: int) -> None:
        self.cli = cli
        self.returncode = returncode


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
        decode_errors: str = "strict",
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
        raises ``ValueError`` unchanged. Lines decode with the configured
        ``decode_errors`` (strict by default), keeping each backend's
        historical decode contract.
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
        """Await the child and return its exit code.

        Raises:
            TransportExitError: On a non-zero exit; ``.returncode`` carries
                the code.
        """
        if self._proc is None:
            raise RuntimeError("transport not started; call start() first")
        returncode = await self._proc.wait()
        if returncode != 0:
            raise TransportExitError(self._cli, returncode)
        return returncode

    async def drain_finished(self) -> None:
        """Await the stderr drain task (a no-op under MERGE_INTO_STDOUT).

        Shielded so a drain awaited in a backend's teardown ``finally`` still
        completes when the caller's scope is already cancelled — the same
        shield the cancel-sweep relies on. Drain-task exceptions are folded
        into the gathered result so teardown cannot mask the caller's original
        backend error.
        """
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


# Number of captured non-JSON lines printed in a PROCESS_EXIT message. The
# count reported in the message is the number of lines actually printed, not
# the size of the capture window.
PROCESS_EXIT_EXCERPT_MAX_LINES = 10


async def reap(transport: CliTransport) -> int | None:
    """Await *transport*'s child and return its exit code, however it exited.

    :meth:`CliTransport.wait` raises :class:`TransportExitError` on a non-zero
    exit; this suppresses that signal and returns the code so the caller can
    run its own exit check after any between-the-two steps (codex yields its
    final diagnostics, pi its terminal events). No fallback value is
    substituted: the returned code is exactly ``transport.returncode``.
    """
    try:
        await transport.wait()
    except TransportExitError:
        pass
    return transport.returncode


def raise_for_exit(
    returncode: int | None,
    *,
    error_type: Callable[..., Exception],
    category: str,
    build_message: Callable[[int], str],
    retryable: bool | None = None,
) -> None:
    """Raise ``error_type`` for a non-zero *returncode*, else return.

    The adapter owns the error class and the wording; this owns the shared
    guard. ``retryable`` is forwarded only when the adapter passes it (pi does;
    codex/osprey construct their error with exactly today's kwargs).
    """
    if returncode is None or returncode == 0:
        return
    kwargs: dict[str, object] = {"category": category}
    if retryable is not None:
        kwargs["retryable"] = retryable
    raise error_type(build_message(returncode), **kwargs)


async def teardown(transport: CliTransport, transports: list[CliTransport]) -> None:
    """Signal, drain and drop *transport* from the caller-owned *transports*.

    Idempotent: the reap (:meth:`CliTransport.terminate`) and the stderr drain
    (:meth:`CliTransport.drain_finished`) are each shielded and safe to repeat.
    A second call re-drains nothing; :meth:`terminate_process` still issues its
    group SIGTERM before checking ``returncode``, but the process is already
    reaped, so the extra signal is harmless and the wait is a no-op. Osprey
    calls this early as well as in its ``finally``; the ``finally`` call is a
    no-op.
    """
    await transport.terminate()
    await transport.drain_finished()
    if transport in transports:
        transports.remove(transport)


def process_exit_message(*, display: str, returncode: int, lines: list[str]) -> str:
    """Build the codex/pi PROCESS_EXIT message, parameterized by *display*.

    Always leads with ``<display> CLI exited with return code <n>.`` and then
    either the last :data:`PROCESS_EXIT_EXCERPT_MAX_LINES` captured lines (the
    header reports the number of lines actually printed) or the display's
    no-output fallback sentence. Osprey builds its structurally different
    message itself.
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
