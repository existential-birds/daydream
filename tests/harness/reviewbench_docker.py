"""Own one offline seccomp probe container and its attached Docker client."""
from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from pathlib import Path


async def _capture(stream: asyncio.StreamReader, retained: bytearray) -> None:
    while chunk := await stream.read(8192):
        retained.extend(chunk[:max(0, 65536 - len(retained))])


async def _remove_container(name: str) -> None:
    child = None
    try:
        async with asyncio.timeout(10):
            child = await asyncio.create_subprocess_exec('docker', 'rm', '--force', name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await child.communicate()
        if child.returncode != 0 and b'No such container' not in stderr:
            raise AssertionError(f'owned seccomp container cleanup failed: {stdout[:1024]!r} {stderr[:1024]!r}')
    finally:
        if child is not None and child.returncode is None:
            with suppress(ProcessLookupError):
                child.kill()
            await asyncio.wait_for(child.wait(), timeout=2)


async def run_seccomp_probe(
    image: str, profile: Path, script: str, *, startup_timeout: float = 180, execution_timeout: float = 45,
) -> str:
    """Bound startup independently; authorize exactly 45s of filter execution.

    The Python probe must emit readiness after bootstrap and wait for GO before
    exercising the filter. Container ownership uses a fresh UUID name; all paths
    remove only that container and kill/reap the attached CLI independently.
    """
    name = 'daydream-seccomp-' + uuid.uuid4().hex
    child = None
    readers: list[asyncio.Task[None]] = []
    stdout, stderr = bytearray(), bytearray()
    phase = 'startup'
    failure: BaseException | None = None
    try:
        async with asyncio.timeout(startup_timeout):
            child = await asyncio.create_subprocess_exec('docker', 'run', '--name', name, '-i',
                '--network', 'none', '--cap-drop', 'ALL', '--security-opt', f'seccomp={profile}',
                '--entrypoint', '/opt/daydream/bin/python', image, '-I', '-c', script,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            assert child.stdout is not None and child.stderr is not None and child.stdin is not None
            readers.append(asyncio.create_task(_capture(child.stderr, stderr)))
            ready = await child.stdout.readline()
            if ready != b'DAYDREAM_SECCOMP_READY\n':
                stdout.extend(ready[:65536])
                raise AssertionError('Python did not report readiness')
        phase = 'filter execution'
        async with asyncio.timeout(execution_timeout):
            readers.append(asyncio.create_task(_capture(child.stdout, stdout)))
            child.stdin.write(b'GO\n')
            await child.stdin.drain()
            child.stdin.close()
            await child.wait()
            await asyncio.gather(*readers)
            assert child.returncode == 0, f'probe exit {child.returncode}'
    except BaseException as exc:
        failure = exc
    finally:
        cleanup_error: BaseException | None = None
        if child is not None:
            try:
                if child.returncode is None:
                    with suppress(ProcessLookupError):
                        child.kill()
                await asyncio.wait_for(child.wait(), timeout=2)
            except (OSError, TimeoutError) as exc:
                cleanup_error = exc
            try:
                await _remove_container(name)
            except (OSError, TimeoutError, AssertionError) as exc:
                cleanup_error = exc
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        if cleanup_error is not None:
            message = f'owned seccomp cleanup unavailable: {type(cleanup_error).__name__}: {str(cleanup_error)[:2048]}'
            if failure is None:
                raise AssertionError(message) from cleanup_error
            failure.add_note(message)
    if isinstance(failure, asyncio.CancelledError):
        raise failure
    if failure is not None:
        raise AssertionError(f'seccomp {phase} failed ({type(failure).__name__}); '
            f'stdout={bytes(stdout[:2048])!r} stderr={bytes(stderr[:2048])!r}') from failure
    return stdout.decode()
