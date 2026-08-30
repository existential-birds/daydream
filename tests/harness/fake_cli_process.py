"""Stall-capable in-process stand-in for a backend CLI subprocess.

:func:`tests.harness.pi_replay.make_mock_process` replays a finite happy-path
stream. This harness models the *pathological* process shapes the idle-stall
and teardown code paths exist for — a stream that goes permanently silent, a
child that ignores SIGTERM — without spawning any OS process and without
racing scripted sleeps against a timeout window. Silence is modeled as a
``readline()`` that never resolves, so the only timer in a test is the one
under test and the outcome cannot depend on host load.

OS semantics modeled by :class:`FakeCliProcess`:

- ``terminate()`` exits the child (``-SIGTERM``) unless ``ignore_sigterm``;
- ``kill()`` always exits it (``-SIGKILL``);
- exiting closes stdout (EOF), like a real pipe when the writer dies;
- ``wait()`` resolves only once the child has exited, and marks it
  ``reaped`` — the fake equivalent of "gone from the process table".

:func:`install_fake_cli_process` patches ``asyncio.create_subprocess_exec`` at
the transport seam (``daydream.backends._transport``), where the codex/pi
backends now spawn via :class:`~daydream.backends._transport.CliTransport`,
as seen by the backend module, so everything of daydream's runs for real —
argv construction, the readline loop, the idle window, the shielded
SIGTERM→SIGKILL teardown — only the OS fork is replaced (the same seam
treatment as ``tests/harness/fake_gh.py``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

SIGTERM_RC = -15
SIGKILL_RC = -9


class _FakeStdin:
    """The write/close surface CodexBackend drives to deliver the prompt."""

    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    def close(self) -> None:
        self.closed = True


class _FakePipeTransport:
    """The fd owner ``terminate_process`` closes to release the pipe read ends.

    Closing releases the read ends regardless of EOF — the mechanism that lets
    a stderr drain finish even when a surviving descendant holds the fd open.
    """

    def __init__(self, proc: FakeCliProcess) -> None:
        self._proc = proc

    def close(self) -> None:
        self._proc._release_pipes()


class FakeCliProcess:
    """An ``asyncio.subprocess.Process`` stand-in with modeled exit semantics."""

    def __init__(
        self,
        lines: list[str],
        *,
        hang: bool = False,
        exit_code: int = 0,
        ignore_sigterm: bool = False,
        stderr_lines: list[str] | None = None,
        stderr_held_open: bool = False,
        stdout_reader: asyncio.StreamReader | None = None,
        stderr_reader: asyncio.StreamReader | None = None,
    ) -> None:
        if stdout_reader is not None:
            self.stdout: asyncio.StreamReader = stdout_reader
        else:
            self.stdout = asyncio.StreamReader()
            for line in lines:
                self.stdout.feed_data((line + "\n").encode())
        self.stderr: asyncio.StreamReader | None = stderr_reader
        if self.stderr is None and (stderr_lines is not None or stderr_held_open):
            self.stderr = asyncio.StreamReader()
            for line in stderr_lines or []:
                self.stderr.feed_data((line + "\n").encode())
        self._stderr_held_open = stderr_held_open
        self.stdin = _FakeStdin()
        self.returncode: int | None = None
        self.reaped = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self._ignore_sigterm = ignore_sigterm
        self._exited = asyncio.Event()
        self._transport = _FakePipeTransport(self)
        if not hang:
            self._exit(exit_code)

    def _exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
            self.stdout.feed_eof()
            # A descendant holding the inherited stderr fd keeps its EOF from
            # arriving; the child's own death alone does not end the drain.
            if not self._stderr_held_open:
                self._release_pipes()
            self._exited.set()

    def _release_pipes(self) -> None:
        """Model the teardown's fd release (real: ``proc._transport.close()``)."""
        if self.stderr is not None:
            self.stderr.feed_eof()

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self._ignore_sigterm:
            self._exit(SIGTERM_RC)

    def kill(self) -> None:
        self.kill_calls += 1
        self._exit(SIGKILL_RC)

    async def wait(self) -> int:
        await self._exited.wait()
        self.reaped = True
        assert self.returncode is not None
        return self.returncode


@dataclass
class FakeCliSpawner:
    """Records every launch the backend attempted (the spawn counter)."""

    procs: list[FakeCliProcess] = field(default_factory=list)
    argvs: list[tuple[str, ...]] = field(default_factory=list)
    kwargs: list[dict[str, Any]] = field(default_factory=list)


def install_fake_cli_process(
    monkeypatch: pytest.MonkeyPatch,
    cli: str,
    *,
    lines: list[str],
    hang: bool = False,
    exit_code: int = 0,
    ignore_sigterm: bool = False,
    stderr_lines: list[str] | None = None,
    stderr_held_open: bool = False,
    stdout_reader: asyncio.StreamReader | None = None,
    stderr_reader: asyncio.StreamReader | None = None,
) -> FakeCliSpawner:
    """Patch ``create_subprocess_exec`` as seen by the *cli* backend module.

    Every launch gets a fresh :class:`FakeCliProcess` with the given shape and
    is recorded on the returned spawner, so tests can assert how many
    subprocesses were actually started (e.g. "a stall must not relaunch").
    The ``stderr_*`` options model osprey's separate stderr pipe: lines fed to
    the drain, optionally held open past the child's exit (a surviving
    descendant inheriting the fd) until the teardown releases the fds.
    """
    spawner = FakeCliSpawner()

    async def fake_exec(*args: Any, **kwargs: Any) -> FakeCliProcess:
        proc = FakeCliProcess(
            lines,
            hang=hang,
            exit_code=exit_code,
            ignore_sigterm=ignore_sigterm,
            stderr_lines=stderr_lines,
            stderr_held_open=stderr_held_open,
            stdout_reader=stdout_reader,
            stderr_reader=stderr_reader,
        )
        spawner.procs.append(proc)
        spawner.argvs.append(tuple(str(a) for a in args))
        spawner.kwargs.append(kwargs)
        return proc

    monkeypatch.setattr(
        "daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec
    )
    return spawner
