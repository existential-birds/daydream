"""Supervise one reviewer and its descendants across Harbor cancellation; retain bounded redacted output."""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import select
import signal
import stat
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import psutil

from daydream.redaction import redact_structured_text

SHUTDOWN_GRACE_SECONDS = 20.0
FORCE_REAP_SECONDS = 2.0
OUTPUT_LIMIT_BYTES = 65536
STOP_REASONS = ('host_task_cancelled', 'host_execution_failed')


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_diagnostic(directory: int, name: str, flags: int) -> int:
    """Preserve symlink rejection for descriptor-relative diagnostic opens."""
    try:
        return os.open(name, flags, dir_fd=directory)
    except OSError as exc:
        try:
            symbolic = stat.S_ISLNK(os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode)
        except OSError:
            symbolic = False
        if symbolic:
            raise ValueError('diagnostic symlink rejected') from exc
        raise


@contextmanager
def _absolute_directory(path: Path, *, create: bool = False) -> Iterator[int]:
    """Reject symlinked ancestry, including operator-supplied readout parents."""
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            if part in {'', '.', '..'} or '/' in part or '\\' in part:
                raise ValueError('unsafe diagnostic directory component')
            if create:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
            child = _open_diagnostic(descriptor, part, _DIRECTORY_FLAGS)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def publish_diagnostic_bytes(path: Path, body: bytes) -> None:
    """Publish one host diagnostic without following ancestry or replacing bytes."""
    with _absolute_directory(path.parent, create=True) as directory:
        temporary = '.diagnostic-' + uuid.uuid4().hex + '.tmp'
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            os.fsync(directory)
        finally:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)


def read_diagnostic_bytes(path: Path, limit: int = 65536) -> bytes:
    """Read bounded regular diagnostic bytes through no-follow anchored parents."""
    with _absolute_directory(path.parent) as directory:
        descriptor = _open_diagnostic(directory, path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('diagnostic file must be regular')
            if info.st_size > limit:
                raise ValueError('diagnostic file exceeds read limit')
            body = stream.read(limit + 1)
            if len(body) > limit:
                raise ValueError('diagnostic file exceeds read limit')
            return body


def _enable_subreaper() -> bool:
    """Keep orphaned invocation descendants owned by this dedicated supervisor."""
    if sys.platform != 'linux':
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'reviewer subreaper containment unavailable')
    return True


def cancellation_marker(logs: Path, invocation: str) -> Path:
    """Exact host/container shared control path; never a native exchange path."""
    if str(uuid.UUID(invocation)) != invocation:
        raise ValueError('invalid reviewer invocation identity')
    return logs / ('reviewer-cancel-' + invocation + '.json')


def _requested_interruption(environment: dict[str, str], invocation: str) -> str | None:
    evidence = Path(environment.get('DAYDREAM_RB_EVIDENCE', '/logs/agent/daydream'))
    with suppress(OSError, ValueError):
        marker = json.loads(read_diagnostic_bytes(cancellation_marker(evidence.parent, invocation), limit=1024))
        for reason in STOP_REASONS:
            if marker == {'invocation_id': invocation, 'interruption': reason}:
                return reason
    return None


def _descendants(parent: psutil.Process, reviewer_pid: int) -> dict[int, psutil.Process]:
    """Admit current descendants independently of the process enumeration snapshot."""
    owned = {}
    for process in parent.children(recursive=True):
        if process.pid == reviewer_pid:
            continue
        with suppress(psutil.NoSuchProcess):
            if parent in process.parents():
                owned[process.pid] = process
    return owned


def _signal_owned(owned: dict[int, psutil.Process], signum: int) -> None:
    for process in owned.values():
        with suppress(psutil.NoSuchProcess):
            process.send_signal(signum)


async def _join_owned_processes(
    child: asyncio.subprocess.Process, owned: dict[int, psutil.Process], parent: psutil.Process, *, subreaper: bool,
) -> None:
    """Discover pipe-holding descendants before joining the reviewer's exit."""
    while True:
        owned.update(_descendants(parent, child.pid))
        _signal_owned(owned, signal.SIGKILL)
        if subreaper:
            for process in owned.values():
                if process.is_running():
                    with suppress(ChildProcessError):
                        os.waitpid(process.pid, os.WNOHANG)
        owned.update(_descendants(parent, child.pid))
        if child.returncode is not None and not any(process.is_running() for process in owned.values()):
            # A wait registered before exit can depend on inherited pipe EOF.
            # Observe exit and clean late adoptions before registering that wait.
            await child.wait()
            return
        await asyncio.sleep(0.02)


async def _capture(stream: asyncio.StreamReader, retained: bytearray, flags: dict[str, bool]) -> None:
    while chunk := await stream.read(4096):
        if flags.get('discarded'):
            continue
        if len(retained) + len(chunk) > OUTPUT_LIMIT_BYTES:
            # Never cut through a credential prefix and then redact its suffix.
            # Oversize diagnostics are unavailable rather than partially exposed.
            retained.clear()
            flags['discarded'] = True
        else:
            retained.extend(chunk)


def _safe_output(retained: bytearray, flags: dict[str, bool]) -> str:
    if flags.get('discarded'):
        return '[reviewer output unavailable: stream exceeded diagnostic limit]\n'
    text = retained.decode(errors='replace')
    if text and not text.endswith('\n'):
        text = text[:text.rfind('\n') + 1]
        flags['incomplete_line_discarded'] = True
    return redact_structured_text(text)


def _publish_process_evidence(environment: dict[str, str], output: bytes, receipt: dict[str, Any]) -> None:
    evidence = Path(environment.get('DAYDREAM_RB_EVIDENCE', '/logs/agent/daydream'))
    logs = evidence.parent
    publish_diagnostic_bytes(logs / 'reviewer-entrypoint.log',
        redact_structured_text(output.decode(errors='replace')).encode())
    publish_diagnostic_bytes(logs / 'reviewer-process.json',
        (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode())


def warn_diagnostics_unavailable(surface: str, error: Exception) -> None:
    """Emit bounded availability facts without exception text or filesystem paths."""
    error_number = error.errno if isinstance(error, OSError) else None
    code = str(error_number) if type(error_number) is int and 0 <= error_number <= 2_147_483_647 else 'unavailable'
    body = f'ReviewBench {surface} diagnostics unavailable ({type(error).__name__[:64]}; errno={code}).\n'.encode()
    try:
        descriptor = sys.stderr.fileno()
        if select.select([], [descriptor], [], 0)[1]:
            os.write(descriptor, body)
    except Exception:
        pass  # Even the warning channel is optional; never buffer for exit-time reflush.


def _try_publish_process_evidence(environment: dict[str, str], output: bytes, receipt: dict[str, Any]) -> None:
    try:
        _publish_process_evidence(environment, output, receipt)
    except Exception as exc:
        warn_diagnostics_unavailable('process', exc)


async def supervise(invocation: str, environment: dict[str, str], outer_timeout: float | None = None) -> int:
    """Own the exact child, stop inference on cancellation, bound export and reap."""
    cancellation_marker(Path("/"), invocation)  # Validate before starting a reviewer.
    subreaper = _enable_subreaper()
    parent = psutil.Process()
    parent.children(recursive=True)  # Refuse startup if descendant ownership is unavailable.
    started = time.monotonic()
    output, stderr = bytearray(), bytearray()
    output_flags: dict[str, bool] = {}
    stderr_flags: dict[str, bool] = {}
    receipt: dict[str, Any] = {'schema': 1, 'invocation_id': invocation,
        'interruption': None, 'cleanup_outcome': 'not_requested',
        'cleanup_allowance_seconds': SHUTDOWN_GRACE_SECONDS,
        'force_reap_allowance_seconds': FORCE_REAP_SECONDS,
        'outer_deadline_seconds': outer_timeout,
        'outer_deadline_availability': 'declared trial config' if outer_timeout is not None else 'unavailable',
        'output_limit_bytes_per_stream': OUTPUT_LIMIT_BYTES, 'child_returncode': None,
        'descendant_containment': 'linux_subreaper' if subreaper else 'observed_parentage_only'}
    if (reason := _requested_interruption(environment, invocation)) is not None:
        receipt.update(interruption='host_cancellation' if reason == 'host_task_cancelled' else reason,
                       cleanup_outcome='cancelled_before_start',
                       elapsed_seconds=time.monotonic() - started)
        _try_publish_process_evidence(environment, b'reviewer cancelled before startup\n', receipt)
        return 130
    worker_environment = {**environment, 'DAYDREAM_RB_INVOCATION': invocation}
    child = await asyncio.create_subprocess_exec(sys.executable, '-I', '-m',
        'daydream.benchmark.reviewbench.harbor_entrypoint', env=worker_environment,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    assert child.stdout is not None and child.stderr is not None
    readers = [asyncio.create_task(_capture(child.stdout, output, output_flags)),
               asyncio.create_task(_capture(child.stderr, stderr, stderr_flags))]
    owned: dict[int, psutil.Process] = {}
    cancellation_at: float | None = None
    force_at: float | None = None
    try:
        while child.returncode is None:
            owned.update(_descendants(parent, child.pid))
            reason = _requested_interruption(environment, invocation)
            if reason is not None and cancellation_at is None:
                cancellation_at = time.monotonic()
                receipt.update(interruption='host_cancellation' if reason == 'host_task_cancelled' else reason,
                    interruption_elapsed_seconds=cancellation_at - started, cleanup_outcome='graceful_pending')
                # Stop native inference immediately, then let the reviewer join
                # its writers and export. Children may have independent sessions.
                child.send_signal(signal.SIGTERM)
                _signal_owned(owned, signal.SIGTERM)
                # Later descendants can be Git reads needed to finalize evidence.
                # Keep discovering them for bounded reaping after exit or grace expiry.
            if cancellation_at is not None and time.monotonic() - cancellation_at >= SHUTDOWN_GRACE_SECONDS:
                force_at = time.monotonic()
                receipt['cleanup_outcome'] = 'forced_termination'
                _signal_owned(owned, signal.SIGKILL)
                child.kill()
                break
            await asyncio.sleep(0.05)
        async with asyncio.timeout(FORCE_REAP_SECONDS):
            await _join_owned_processes(child, owned, parent, subreaper=subreaper)
            receipt['owned_descendants_cleanup'] = 'joined' if subreaper else 'observed_exits'
            await asyncio.gather(*readers)
        if cancellation_at is not None and force_at is None:
            receipt['cleanup_outcome'] = 'graceful_exit'
    except Exception as exc:
        receipt['cleanup_outcome'] = 'reap_or_output_expired' if isinstance(exc, TimeoutError) else 'supervisor_failed'
        receipt['cleanup_error'] = type(exc).__name__
        if child.returncode is None:
            child.kill()
        try:
            async with asyncio.timeout(FORCE_REAP_SECONDS):
                try:
                    await _join_owned_processes(child, owned, parent, subreaper=subreaper)
                except Exception:
                    receipt['owned_descendants_cleanup'] = 'unavailable_process_table'
                    await child.wait()
                else:
                    receipt['owned_descendants_cleanup'] = 'joined' if subreaper else 'observed_exits'
        except TimeoutError:
            receipt['child_reap'] = 'unavailable_after_force_bound'
    finally:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        receipt.update(child_returncode=child.returncode, elapsed_seconds=time.monotonic() - started,
            output_truncated=bool(output_flags.get('discarded') or stderr_flags.get('discarded')),
            terminal_evidence_available=Path(environment.get('DAYDREAM_RB_EVIDENCE',
                '/logs/agent/daydream')).joinpath('execution.json').is_file())
        safe_output = _safe_output(output, output_flags) + _safe_output(stderr, stderr_flags)
        receipt['stdout_capture'] = output_flags
        receipt['stderr_capture'] = stderr_flags
        _try_publish_process_evidence(environment, safe_output.encode(), receipt)
    return child.returncode if child.returncode is not None and child.returncode >= 0 else 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--outer-timeout', type=float)
    arguments = parser.parse_args()
    environment = dict(os.environ)
    try:
        return asyncio.run(supervise(arguments.invocation, environment, arguments.outer_timeout))
    except Exception as exc:
        _try_publish_process_evidence(environment, b'reviewer supervisor failed before startup\n', {
            'schema': 1, 'invocation_id': arguments.invocation, 'cleanup_outcome': 'startup_failed',
            'startup_error': type(exc).__name__[:80],
            'errno': exc.errno if isinstance(exc, OSError) and type(exc.errno) is int else None,
            'child_returncode': None,
        })
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
