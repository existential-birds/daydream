"""Real-path tests for subprocess termination: process-group reaping and fd release."""

import asyncio
import os

import pytest

from daydream.backends._subprocess import terminate_process

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

    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


async def test_terminate_process_releases_fds() -> None:
    """No fd growth after an aborted run, even with a grandchild holding the pipe."""
    base = _fd_count()
    proc = await _spawn_holder()
    await terminate_process(proc)
    assert _fd_count() == base


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
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
    assert _fd_count() == base
