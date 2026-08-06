"""Real-path tests for subprocess termination: process-group reaping and fd release."""

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from daydream.backends._subprocess import terminate_process

if TYPE_CHECKING:
    from daydream.runner import RunConfig

MakeConfig = Callable[..., "RunConfig"]

_HOLDER_SCRIPT = (
    # The `sleep 30` grandchild inherits the stdout pipe write end, so an
    # aborted run keeps the pipe open even after the CLI dies (the Errno 24
    # leak). The `print("UP", flush=True)` runs only after the grandchild is
    # forked, so awaiting the first stdout line makes these tests deterministic
    # — the terminating code is always exercised against a live process group,
    # never racy against the CLI's fork. A python CLI is used because bash
    # defers SIGTERM while a child runs on macOS, which would stall every test
    # on the TERMINATE_GRACE_S window.
    "import subprocess, time; "
    "subprocess.Popen(['sleep', '30']); "
    "print('UP', flush=True); "
    "time.sleep(1000)"
)


def _fd_count() -> int:
    return len(os.listdir("/dev/fd"))


async def _spawn_holder() -> asyncio.subprocess.Process:
    """Spawn a long-running CLI that keeps a `sleep` child holding the pipe."""
    proc = await asyncio.create_subprocess_exec(
        "python3",
        "-c",
        _HOLDER_SCRIPT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
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

    await _wait_for_group_gone(pgid)


async def test_terminate_process_releases_fds() -> None:
    """No fd growth after an aborted run, even with a grandchild holding the pipe."""
    base = _fd_count()
    proc = await _spawn_holder()
    await terminate_process(proc)
    await _wait_for_fd_count(base)


async def test_terminate_process_is_idempotent() -> None:
    """Calling terminate twice (cancel + finally both fire) is a no-op."""
    proc = await _spawn_holder()
    await terminate_process(proc)
    await terminate_process(proc)  # must not raise


async def test_cancel_processes_kills_groups_and_releases_fds() -> None:
    """cancel_processes reaps every tracked process group, not just direct children."""
    from daydream.backends._subprocess import cancel_processes

    base = _fd_count()
    procs = [await _spawn_holder() for _ in range(2)]
    pgids = [os.getpgid(p.pid) for p in procs]

    await cancel_processes(procs)

    for pgid in pgids:
        await _wait_for_group_gone(pgid)
    await _wait_for_fd_count(base)


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


async def _wait_for_group_gone(pgid: int, *, timeout_s: float = 10.0) -> None:
    """Await *pgid*'s disappearance (a readiness wait, not a fixed sleep).

    A group-signal kill reaps the direct child synchronously, but a grandchild
    reparented to PID 1 lingers as a zombie until init reaps it — and a zombie
    still answers ``killpg(pgid, 0)``. Asserting ``ProcessLookupError`` in the
    same event-loop tick as the kill therefore races the kernel's reap: on a
    loaded host (CI runners, parallel suites) the window is wide enough to fail
    intermittently. Polling until the group is gone makes the assertion
    deterministic — the observable outcome is "no process remains in the group",
    not "the group vanished by the next instruction". The loop exits the moment
    ``killpg`` raises; the timeout is a failure bound, not a synchronization
    delay (same contract as ``_wait_for_file``).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        if loop.time() > deadline:
            raise TimeoutError(f"process group {pgid} still alive after {timeout_s}s")
        await asyncio.sleep(0.01)


async def _wait_for_fd_count(base: int, *, timeout_s: float = 10.0) -> None:
    """Await the fd count's return to *base* (a readiness wait, not a fixed sleep).

    The transport close releases the pipe fds, but the release lands in the
    event loop's connection_lost processing — asserting equality in the same
    tick as teardown races the loop on a loaded host (CI runners, parallel
    suites), the same failure mode ``_wait_for_group_gone`` documents. Polling
    until the count returns makes the assertion deterministic; the loop exits
    the moment the baseline is reached and the timeout is a failure bound, not
    a synchronization delay.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while _fd_count() != base:
        if loop.time() > deadline:
            raise TimeoutError(
                f"fd count {_fd_count()} did not return to baseline {base} after {timeout_s}s"
            )
        await asyncio.sleep(0.01)


async def test_runner_run_aborted_improve_reaps_group_and_releases_fds(
    improve_monorepo_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
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
    -> ``cancel_processes`` -> group signal + transport close).

    Assertions are the issue #303 contract: the CLI's process group no longer
    exists (``os.killpg(pgid, 0)`` raises ``ProcessLookupError`` — no orphaned
    grandchildren) and the fd count returns to the pre-run baseline.
    """
    from daydream import runner
    from daydream.backends.codex import CodexBackend

    silence_console("daydream.runner")
    silence_console("daydream.improve.orchestrator")
    silence_console("daydream.agent")

    # A fake `codex` CLI that is a genuine subprocess: it forks a `sleep`
    # grandchild (which holds the piped stdout open after the CLI dies), writes
    # its own pid to the readiness marker, then blocks forever. A python CLI is
    # used because bash defers SIGTERM while a child runs on macOS, which would
    # stall on the TERMINATE_GRACE_S window (same rationale as _HOLDER_SCRIPT).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "codex-ready"
    cli = bin_dir / "codex"
    cli.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, time\n"
        "subprocess.Popen(['sleep', '300'])\n"
        f"open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
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
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_a, **_k: backend)

    base_fds = _fd_count()
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
    await _wait_for_group_gone(pgid)
    await _wait_for_fd_count(base_fds)
