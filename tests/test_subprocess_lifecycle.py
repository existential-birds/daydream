"""Real-path tests for subprocess termination: process-group reaping and fd release."""

import asyncio
import os
import signal
import sys

from daydream.backends._subprocess import cancel_processes, terminate_process
from tests.harness.processes import (
    fd_count,
    wait_for_fd_baseline,
    wait_for_process_group_gone,
)

# A group leader whose same-group ``sleep`` child also holds the pipe.
_GROUP_HOLDER = "import subprocess, time; subprocess.Popen(['sleep', '30']); print('UP', flush=True); time.sleep(1000)"


async def test_cancel_processes_kills_groups_and_releases_fds() -> None:
    """Cancellation reaps every tracked process group, not just direct children; teardown may repeat."""
    base = fd_count()
    procs = []
    for _ in range(2):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _GROUP_HOLDER, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        )
        assert proc.stdout is not None and await proc.stdout.readline() == b"UP\n"
        procs.append(proc)
    pgids = [os.getpgid(p.pid) for p in procs]
    assert pgids == [p.pid for p in procs]
    await cancel_processes(procs)
    for pgid in pgids:
        await wait_for_process_group_gone(pgid)
    await terminate_process(procs[0])  # cancel + finally may both close the same process
    await wait_for_fd_baseline(base)


async def test_terminate_process_does_not_wait_for_escaped_pipe_writer() -> None:
    """Actual direct-child exit bounds cleanup even when an escaped writer retains stdout."""
    child = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "writer = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "start_new_session=True)\n"
        "print(writer.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    base = fd_count()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", child, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, start_new_session=True,
    )
    assert proc.stdout is not None
    async with asyncio.timeout(5):
        writer_pid = int(await proc.stdout.readline())
    # Short TERM grace forces escalation; give the real OS child watcher a
    # separate cleanup allowance under parallel coverage instrumentation. The
    # escaped writer's 60s lifetime still exceeds both bounded waits below.
    cleanup = asyncio.create_task(terminate_process(proc, timeout=0.05, cleanup_timeout_s=1))
    try:
        async with asyncio.timeout(3):
            await asyncio.shield(cleanup)
        assert proc.returncode == -signal.SIGKILL
        await wait_for_fd_baseline(base, timeout_s=5)
    finally:
        # The escaped session intentionally cannot be cleaned by killpg(proc.pid).
        # Release it even when the baseline remains blocked in proc.wait().
        try:
            os.killpg(writer_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        async with asyncio.timeout(5):
            await cleanup
