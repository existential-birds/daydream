"""Real-path tests for subprocess termination: process-group reaping and fd release."""

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from daydream import runner
from daydream.backends import AUDIT_ROOT_ISOLATION
from daydream.backends._subprocess import terminate_process
from daydream.backends._transport import CliTransport, teardown
from daydream.backends.codex import CodexBackend
from tests.harness.processes import (
    GROUP_HOLDER_CLI,
    fd_count,
    wait_for_fd_baseline,
    wait_for_process_group_gone,
)

if TYPE_CHECKING:
    from daydream.run_config import RunConfig

MakeConfig = Callable[..., "RunConfig"]


async def _spawn_holder() -> asyncio.subprocess.Process:
    """Spawn a long-running CLI that keeps a `sleep` child holding the pipe."""
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", GROUP_HOLDER_CLI, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, start_new_session=True,
    )
    stdout = proc.stdout
    assert stdout is not None
    assert await stdout.readline() == b"UP\n"
    return proc

async def test_terminate_process_kills_whole_group() -> None:
    """Grandchildren cannot outlive the CLI: the whole group dies on terminate."""
    proc = await _spawn_holder()
    pgid = os.getpgid(proc.pid)
    assert pgid == proc.pid  # session leader => pid is the group id

    await terminate_process(proc)
    await wait_for_process_group_gone(pgid)

async def test_terminate_process_releases_fds() -> None:
    """No fd growth after an aborted run, even with a grandchild holding the pipe."""
    base = fd_count()
    proc = await _spawn_holder()
    await terminate_process(proc)
    await wait_for_fd_baseline(base)

async def test_terminate_process_is_idempotent() -> None:
    """Calling terminate twice (cancel + finally both fire) is a no-op."""
    proc = await _spawn_holder()
    await terminate_process(proc)
    await terminate_process(proc)  # must not raise

async def test_cancel_transports_kills_groups_and_releases_fds() -> None:
    """Native transport cancellation reaps every group, not just direct children."""
    base = fd_count()
    transports: list[CliTransport] = []
    pgids: list[int] = []
    try:
        for _ in range(2):
            transport = CliTransport("python3", ["python3", "-c", GROUP_HOLDER_CLI], limit=1024)
            transports.append(transport)
            await transport.start()
            lines = transport.lines(lambda: None)
            assert await anext(lines) == "UP"
            await lines.aclose()
            assert transport._proc is not None
            pgids.append(os.getpgid(transport._proc.pid))
        await CliTransport.cancel_all(transports)
        assert transports == []
        for pgid in pgids:
            await wait_for_process_group_gone(pgid)
        await wait_for_fd_baseline(base)
    finally:
        await CliTransport.cancel_all(transports)


async def test_cancel_transports_preserves_pending_spawn_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    native_spawn = asyncio.create_subprocess_exec
    captured: list[asyncio.subprocess.Process] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_capture(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        child = await native_spawn(*args, **kwargs)
        captured.append(child)
        entered.set()
        await release.wait()
        return child

    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", delayed_capture)
    transport = CliTransport("fixture", [sys.executable, "-c", "import time; time.sleep(30)"], limit=1024)
    transports = [transport]
    startup = asyncio.create_task(transport.start())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert captured[0].returncode is None
        await CliTransport.cancel_all(transports)
        assert transports == [transport]
        assert captured[0].returncode is None
        release.set()
        await startup
        await CliTransport.cancel_all(transports)
        assert transports == []
        assert await asyncio.wait_for(captured[0].wait(), timeout=5) < 0
    finally:
        release.set()
        await startup
        await teardown(transport, transports)


async def test_cancel_transports_closes_exited_root_inherited_pipes(tmp_path: Path) -> None:
    ready = tmp_path / "descendant-ready"
    release = tmp_path / "descendant-release"
    finished = tmp_path / "descendant-finished"
    descendant = (
        f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); p=pathlib.Path({str(release)!r});\n"
        "while not p.exists(): time.sleep(0.01)\n"
        f"pathlib.Path({str(finished)!r}).touch()"
    )
    # An independent pipe writer keeps this proof independent of platform rules
    # for signaling a group whose leader already exited; group reaping is tested above.
    root = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {descendant!r}], start_new_session=True)"
    transport = CliTransport("fixture", [sys.executable, "-c", root], stderr_sink=lambda _: None, limit=1024)
    transports = [transport]
    try:
        await transport.start()
        await _wait_for_file(ready, timeout_s=5)
        async with asyncio.timeout(5):
            while transport.returncode is None:
                await asyncio.sleep(0.01)
        assert transport.returncode == 0
        assert transport._drain_task is not None and not transport._drain_task.done()
        await asyncio.wait_for(CliTransport.cancel_all(transports), timeout=5)
        assert transports == []
        assert transport._drain_task is None
        assert not release.exists()
    finally:
        release.touch()
        await teardown(transport, transports)
        await _wait_for_file(finished, timeout_s=5)


async def _wait_for_file(path: Path, *, timeout_s: float = 60.0) -> None:
    """Await *path*'s creation (a readiness wait, not a fixed sleep).

    The fake backend CLI writes *path* only AFTER forking its grandchild, the
    same guarantee the ``UP`` readiness line gives the direct-helper tests
    above. The production backends consume the CLI's stdout, so a file marker
    is the real-path stand-in for that line. Polling with a sub-interval yield
    is the event-loop-safe way to wait on a filesystem condition; the loop
    exits the moment the file appears, so the wait is bounded only by the
    timeout (a failure bound, not a synchronization delay).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not path.exists():
        if loop.time() > deadline:
            raise TimeoutError(f"timed out waiting for the backend CLI marker at {path}")
        await asyncio.sleep(0.01)

async def test_runner_run_aborted_improve_reaps_group_and_releases_fds(
    improve_monorepo_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    silence_console: Callable[..., None],
) -> None:
    """Real-path: an aborted ``--improve`` run through ``runner.run`` reaps the
    backend CLI's process group and returns the process to the fd baseline.

    Enters the production entrypoint (``runner.run``) over a real temp git repo
    with a real event loop; only the network/API side is mocked. ``create_backend``
    is pinned to a REAL :class:`CodexBackend` whose ``codex`` CLI is a fake on
    ``$PATH`` — it forks a grandchild that inherits the piped stdout (the Errno
    24 shape) and then blocks forever, so the run sits mid-execute at the abort
    point. The run task is cancelled once the CLI reports ready (marker file
    written after the grandchild's fork — see ``_wait_for_file``), driving
    ``run_agent``'s shutdown path (``except BaseException`` -> ``backend.cancel()``
    -> native transport teardown -> group signal + transport close).

    Assertions are the issue #303 contract: the CLI's process group no longer
    exists (``os.killpg(pgid, 0)`` raises ``ProcessLookupError`` — no orphaned
    grandchildren) and the fd count returns to the pre-run baseline.
    """
    silence_console("daydream.runner")
    for module in ("recon", "audit", "planning", "issue_publication", "reporting"):
        silence_console(f"daydream.improve.{module}")
    silence_console("daydream.agent")

    # A fake `codex` CLI that is a genuine subprocess: it forks a `sleep`
    # grandchild (which holds the piped stdout open after the CLI dies), writes
    # its own pid to the readiness marker, then blocks forever. A python CLI is
    # used because bash defers SIGTERM while a child runs on macOS, which would
    # stall on the TERMINATE_GRACE_S window (same rationale as GROUP_HOLDER_CLI).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "codex-ready"
    cli = bin_dir / "codex"
    # The marker is written to a temp path and renamed into place so it never
    # exists empty: `open(marker, 'w')` creates the file before the pid write
    # lands, and `_wait_for_file` polls bare existence, so a non-atomic write
    # races the reader into `int('')` on a loaded host.
    cli.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, time\n"
        "subprocess.Popen(['sleep', '300'])\n"
        f"tmp = {str(marker)!r} + '.tmp'\n"
        "with open(tmp, 'w') as f:\n"
        "    f.write(str(os.getpid()))\n"
        f"os.replace(tmp, {str(marker)!r})\n"
        "time.sleep(1000)\n"
    )
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    # The fake CLI is intentionally silent; the only exit from its readline is
    # the test's cancellation, so disable the idle-stall window entirely.
    monkeypatch.setenv("DAYDREAM_STREAM_IDLE_TIMEOUT_S", "0")

    # The backend seam (the single mock seam the testing standard permits): the
    # network/API is mocked by the fake CLI on $PATH, but the subprocess spawn
    # stays real so the OS process group actually exists for the assertion.
    backend = CodexBackend(model="test-model")

    def _factory(*_args: object, **kwargs: object) -> CodexBackend:
        audit_root = kwargs.get("audit_root")
        setattr(backend, "audit_root", audit_root)
        setattr(backend, "audit_root_isolation", AUDIT_ROOT_ISOLATION if isinstance(audit_root, Path) else None,)
        return backend

    monkeypatch.setattr("daydream.runner.create_backend", _factory)

    base_fds = fd_count()
    run_task = asyncio.create_task(runner.run(make_config(improve_monorepo_target, flow_name="improve")))
    pgid: int | None = None
    try:
        await _wait_for_file(marker, timeout_s=60)
        pid = int(marker.read_text())
        pgid = os.getpgid(pid)
        assert pgid == pid  # start_new_session => pid is the group id
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
    finally:
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except BaseException:
                pass

    assert pgid is not None
    await wait_for_process_group_gone(pgid)
    await wait_for_fd_baseline(base_fds)
