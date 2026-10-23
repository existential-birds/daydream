"""Invocation-owned process supervision across Harbor's detached Docker await.

Only cancellation tombstones live in private temporary storage. Native exchange
storage stays private to the runner. The supervisor publishes bounded, redacted
process output and a factual cleanup receipt independently of reviewer finalizers.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import signal
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from daydream.benchmark.reviewbench.diagnostics import publish_diagnostic_bytes
from daydream.redaction import redact_structured_text

SHUTDOWN_GRACE_SECONDS = 20.0
FORCE_REAP_SECONDS = 2.0
OUTPUT_LIMIT_BYTES = 65536


def invocation_directory(invocation: str) -> Path:
    """Resolve one validated invocation capability, refusing redirected storage."""
    if str(uuid.UUID(invocation)) != invocation:
        raise ValueError('invalid reviewer invocation identity')
    root = Path(tempfile.gettempdir()).resolve() / 'daydream-reviewbench-invocations'
    root.mkdir(mode=0o700, exist_ok=True)
    _require_private_directory(root)
    directory = root / invocation
    directory.mkdir(mode=0o700, exist_ok=True)
    _require_private_directory(directory)
    return directory


def _require_private_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('reviewer invocation storage is not private')


def request_cancellation(invocation: str) -> None:
    """A startup tombstone prevents a late reviewer from starting model work."""
    directory = invocation_directory(invocation)
    marker = directory / 'cancel'
    try:
        publish_diagnostic_bytes(marker, b'cancelled\n')
    except FileExistsError:
        if marker.is_symlink() or marker.read_bytes() != b'cancelled\n':
            raise ValueError('reviewer cancellation marker conflicts')



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


def _cancellation_requested(directory: Path, environment: dict[str, str], invocation: str) -> bool:
    if (directory / 'cancel').exists():
        return True
    evidence = Path(environment.get('DAYDREAM_RB_EVIDENCE', '/logs/agent/daydream'))
    path = cancellation_marker(evidence.parent, invocation)
    try:
        if path.is_symlink() or path.stat().st_size > 1024:
            return False
        marker = json.loads(path.read_bytes())
        return isinstance(marker, dict) and marker == {
            'invocation_id': invocation, 'interruption': 'host_task_cancelled'}
    except (OSError, ValueError):
        return False


def _reap_adopted(owned: dict[int, str], reviewer_pid: int) -> None:
    for pid in owned:
        if pid == reviewer_pid:
            continue  # asyncio owns the direct reviewer's exit/reap.
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass

def _process_table() -> dict[int, tuple[int, str]]:
    """Read parentage and start identity; no executable-name matching or killing."""
    if Path('/proc/self/stat').is_file():
        result: dict[int, tuple[int, str]] = {}
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal():
                continue
            try:
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                result[int(path.name)] = int(fields[1]), fields[19]
            except (OSError, ValueError, IndexError):
                continue
        return result
    if sys.platform == 'darwin':
        return _darwin_process_table()
    raise OSError('reviewer process ownership unavailable on this platform')


class _DarwinProcessInfo(ctypes.Structure):
    # macOS SDK sys/proc_info.h proc_bsdinfo (PROC_PIDTBSDINFO). Keep the
    # declared layout so start identities retain microsecond precision.
    _fields_ = [
        ('flags', ctypes.c_uint32), ('status', ctypes.c_uint32), ('exit_status', ctypes.c_uint32),
        ('pid', ctypes.c_uint32), ('ppid', ctypes.c_uint32), ('uid', ctypes.c_uint32),
        ('gid', ctypes.c_uint32), ('real_uid', ctypes.c_uint32), ('real_gid', ctypes.c_uint32),
        ('saved_uid', ctypes.c_uint32), ('saved_gid', ctypes.c_uint32), ('reserved', ctypes.c_uint32),
        ('command', ctypes.c_char * 16), ('name', ctypes.c_char * 32),
        ('file_count', ctypes.c_uint32), ('process_group', ctypes.c_uint32),
        ('job_control', ctypes.c_uint32), ('terminal_device', ctypes.c_uint32),
        ('terminal_group', ctypes.c_uint32), ('nice', ctypes.c_int32),
        ('started_seconds', ctypes.c_uint64), ('started_microseconds', ctypes.c_uint64),
    ]


def _darwin_process_table() -> dict[int, tuple[int, str]]:
    """Read kernel ownership directly, without a subprocess deadline under load."""
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    library.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.proc_listallpids.restype = ctypes.c_int
    library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    library.proc_pidinfo.restype = ctypes.c_int
    count = library.proc_listallpids(None, 0)
    if count <= 0:
        raise OSError(ctypes.get_errno(), 'reviewer process enumeration unavailable')
    # Kernel process counts can grow between sizing and reading. A bounded
    # retry avoids silently accepting a truncated ownership table.
    for _ in range(2):
        capacity = count + 128
        processes = (ctypes.c_int * capacity)()
        copied = library.proc_listallpids(processes, ctypes.sizeof(processes))
        if 0 < copied < capacity:
            break
        count = capacity * 2
    else:
        raise OSError('reviewer process enumeration did not settle')
    result: dict[int, tuple[int, str]] = {}
    for pid in processes[:copied]:
        if pid <= 0:
            continue
        information = _DarwinProcessInfo()
        size = ctypes.sizeof(information)
        if library.proc_pidinfo(pid, 3, 0, ctypes.byref(information), size) != size:
            continue  # A process can exit between enumeration and its query.
        if int(information.pid) != pid:
            raise OSError('reviewer process identity differs')
        result[pid] = (int(information.ppid),
            f'{int(information.started_seconds)}:{int(information.started_microseconds)}')
    if os.getpid() not in result:
        raise OSError('reviewer process identity unavailable')
    return result


def _descendants(pid: int, table: dict[int, tuple[int, str]]) -> dict[int, str]:
    owned: dict[int, str] = {}
    parents = {pid}
    while True:
        found = {child for child, (parent, _) in table.items() if parent in parents and child not in owned}
        if not found:
            return owned
        owned.update((child, table[child][1]) for child in found)
        parents = found


def _signal_owned(owned: dict[int, str], signum: int) -> None:
    table = _process_table()
    for pid, identity in owned.items():
        if table.get(pid, (None, None))[1] != identity:
            continue
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


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


async def supervise(invocation: str, environment: dict[str, str], outer_timeout: float | None = None) -> int:
    """Own the exact child, stop inference on cancellation, bound export and reap."""
    directory = invocation_directory(invocation)
    subreaper = _enable_subreaper()
    _process_table()  # Refuse startup if exact descendant ownership is unavailable.
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
    if _cancellation_requested(directory, environment, invocation):
        receipt.update(interruption='host_cancellation', cleanup_outcome='cancelled_before_start',
                       elapsed_seconds=time.monotonic() - started)
        _publish_process_evidence(environment, b'reviewer cancelled before startup\n', receipt)
        return 130
    worker_environment = {**environment, 'DAYDREAM_RB_INVOCATION': invocation}
    if outer_timeout is not None:
        worker_environment['DAYDREAM_RB_DECLARED_OUTER_TIMEOUT'] = str(outer_timeout)
    child = await asyncio.create_subprocess_exec(sys.executable, '-I', '-m',
        'daydream.benchmark.reviewbench.harbor_entrypoint', env=worker_environment,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    assert child.stdout is not None and child.stderr is not None
    readers = [asyncio.create_task(_capture(child.stdout, output, output_flags)),
               asyncio.create_task(_capture(child.stderr, stderr, stderr_flags))]
    owned: dict[int, str] = {}
    cancellation_at: float | None = None
    force_at: float | None = None
    signalled: dict[int, str] = {}
    try:
        while child.returncode is None:
            table = _process_table()
            owned.update(_descendants(os.getpid() if subreaper else child.pid, table))
            owned.pop(child.pid, None)
            if _cancellation_requested(directory, environment, invocation) and cancellation_at is None:
                cancellation_at = time.monotonic()
                receipt.update(interruption='host_cancellation',
                    interruption_elapsed_seconds=cancellation_at - started, cleanup_outcome='graceful_pending')
                # Stop native inference immediately, then let the reviewer join
                # its writers and export. Children may have independent sessions.
                child.send_signal(signal.SIGTERM)
                _signal_owned(owned, signal.SIGTERM)
                signalled.update(owned)
            if cancellation_at is not None:
                newly_owned = {pid: identity for pid, identity in owned.items() if signalled.get(pid) != identity}
                _signal_owned(newly_owned, signal.SIGTERM)
                signalled.update(newly_owned)
            if cancellation_at is not None and time.monotonic() - cancellation_at >= SHUTDOWN_GRACE_SECONDS:
                force_at = time.monotonic()
                receipt['cleanup_outcome'] = 'forced_termination'
                _signal_owned(owned, signal.SIGKILL)
                child.kill()
                break
            await asyncio.sleep(0.05)
        async with asyncio.timeout(FORCE_REAP_SECONDS):
            await child.wait()
            # A native TERM handler can fork then exit between table reads.
            # Linux adoption keeps that late child attached to this invocation
            # even after the reviewer and its original native parent exit.
            while True:
                table = _process_table()
                owned.update(_descendants(os.getpid() if subreaper else child.pid, table))
                owned.pop(child.pid, None)
                _signal_owned(owned, signal.SIGKILL)
                if subreaper:
                    _reap_adopted(owned, child.pid)
                after_reap = _process_table()
                remaining = {pid: identity for pid, identity in owned.items()
                             if after_reap.get(pid, (None, None))[1] == identity}
                if not remaining:
                    break
                await asyncio.sleep(0.02)
            receipt['owned_descendants_cleanup'] = 'joined' if subreaper else 'observed_exits'
            await asyncio.gather(*readers)
        if cancellation_at is not None and force_at is None:
            receipt['cleanup_outcome'] = 'graceful_exit'
    except Exception as exc:
        receipt['cleanup_outcome'] = 'reap_or_output_expired' if isinstance(exc, TimeoutError) else 'supervisor_failed'
        receipt['cleanup_error'] = type(exc).__name__
        try:
            _signal_owned(owned, signal.SIGKILL)
        except Exception:
            receipt['owned_descendants_cleanup'] = 'unavailable_process_table'
        if child.returncode is None:
            child.kill()
        try:
            async with asyncio.timeout(FORCE_REAP_SECONDS):
                await child.wait()
                if subreaper:
                    _reap_adopted(owned, child.pid)
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
        _publish_process_evidence(environment, safe_output.encode(), receipt)
        print(safe_output, end='')
    return child.returncode if child.returncode is not None and child.returncode >= 0 else 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--cancel', action='store_true')
    parser.add_argument('--outer-timeout', type=float)
    arguments = parser.parse_args()
    if arguments.cancel:
        request_cancellation(arguments.invocation)
        return 0
    return asyncio.run(supervise(arguments.invocation, dict(os.environ), arguments.outer_timeout))


if __name__ == '__main__':
    raise SystemExit(main())
