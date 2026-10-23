"""Atomic checkpoints from exact recorder bytes; native exchange directories stay private."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import Any

from daydream.redaction import redact_structured_text
from daydream.trajectory import PARTIAL_SUFFIX, RUN_DOCUMENT_NAME, RunWriteSnapshot, run_directory


def _json_bytes(value: Any) -> bytes:
    return (redact_structured_text(json.dumps(value, indent=2, allow_nan=False)) + '\n').encode()


def _relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {'', '.', '..'} for part in value.split('/')):
        raise ValueError('unsafe checkpoint document path')
    return Path(*path.parts)


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


@contextmanager
def _directory(parent: int, parts: Sequence[str], *, create: bool = False) -> Iterator[int]:
    """Walk each checkpoint directory component relative to an anchored descriptor."""
    descriptor = os.dup(parent)
    try:
        for part in parts:
            if part in {'', '.', '..'} or '/' in part or '\\' in part:
                raise ValueError('unsafe checkpoint directory component')
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                try:
                    symbolic = stat.S_ISLNK(os.stat(part, dir_fd=descriptor, follow_symlinks=False).st_mode)
                except OSError:
                    symbolic = False
                if symbolic:
                    raise ValueError('checkpoint symlink rejected') from exc
                raise
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _absolute_directory(path: Path, *, create: bool = False) -> Iterator[int]:
    """Reject symlinked ancestry, including operator-supplied readout parents."""
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        with _directory(descriptor, absolute.parts[1:], create=create) as opened:
            yield opened
    finally:
        os.close(descriptor)


def _identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    return metadata.st_dev, metadata.st_ino


def _create_bytes(directory: int, relative: Path, body: bytes) -> None:
    """Flush and exclusively link one flat file below an anchored directory."""
    temporary = '.checkpoint-' + uuid.uuid4().hex + '.tmp'
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, relative.name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
    finally:
        with suppress(OSError):
            os.unlink(temporary, dir_fd=directory)


def publish_diagnostic_bytes(path: Path, body: bytes) -> None:
    """Publish one host diagnostic without following ancestry or replacing bytes."""
    with _absolute_directory(path.parent, create=True) as directory:
        _create_bytes(directory, Path(path.name), body)


def _read_bytes(directory: int, relative: Path, limit: int | None = None) -> bytes:
    """Read only a regular file through no-follow descriptor-relative parents."""
    try:
        descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
    except OSError as exc:
        try:
            symbolic = stat.S_ISLNK(os.stat(relative.name, dir_fd=directory, follow_symlinks=False).st_mode)
        except OSError:
            symbolic = False
        if symbolic:
            raise ValueError('checkpoint symlink rejected') from exc
        raise
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('checkpoint file must be regular')
        if limit is not None and info.st_size > limit:
            raise ValueError('diagnostic file exceeds read limit')
        body = stream.read() if limit is None else stream.read(limit + 1)
        if limit is not None and len(body) > limit:
            raise ValueError('diagnostic file exceeds read limit')
        return body


def read_diagnostic_bytes(path: Path, limit: int = 65536) -> bytes:
    """Read bounded regular diagnostic bytes through no-follow anchored parents."""
    with _absolute_directory(path.parent) as directory:
        return _read_bytes(directory, Path(path.name), limit)


def _optional_json(path: Path, availability: dict[str, str]) -> Any:
    try:
        with _absolute_directory(path.parent) as directory:
            try:
                value = json.loads(_read_bytes(directory, Path(path.name)))
                if not isinstance(value, dict):
                    raise ValueError('diagnostic receipt must be an object')
            except (OSError, ValueError) as exc:
                availability[path.name] = 'unavailable:' + type(exc).__name__
                return None
    except OSError as exc:
        availability[path.name] = 'unavailable:' + type(exc).__name__
        return None
    availability[path.name] = 'available'
    return value


class InterruptedReviewDiagnostics:
    """Retain two immutable checkpoints; optional failures never change review authority."""

    def __init__(self, evidence: Path, metadata: dict[str, Any]) -> None:
        self.checkpoint_directory = evidence.with_name(evidence.name + '-checkpoints')
        with _absolute_directory(self.checkpoint_directory.parent, create=True) as parent:
            os.mkdir(self.checkpoint_directory.name, mode=0o700, dir_fd=parent)
            with _directory(parent, [self.checkpoint_directory.name]) as directory:
                self._directory_identity = _identity(directory)
                _create_bytes(directory, Path('metadata.json'), _json_bytes(metadata))
        self.last_error: str | None = None
        self._sequence = 0
        self._retained: list[tuple[Path, bytes]] = []

    @contextmanager
    def _root(self) -> Iterator[int]:
        with _absolute_directory(self.checkpoint_directory) as directory:
            if _identity(directory) != self._directory_identity:
                raise ValueError('checkpoint directory identity changed')
            yield directory

    def retain(self, snapshot: RunWriteSnapshot) -> None:
        """Commit exact immutable root and sibling bytes at one recorder cutoff."""
        try:
            self._retain(snapshot)
            self.last_error = None
        except Exception as exc:
            self.last_error = 'checkpoint publication failed: ' + type(exc).__name__
            # Separate immutable receipts avoid overwriting competing bytes and
            # allow the reader to distinguish a stale checkpoint from silence.
            try:
                with self._root() as directory:
                    _create_bytes(directory, Path(f'failure-{self._sequence:08d}.json'),
                                  _json_bytes({'reason': self.last_error, 'cutoff_at': snapshot.cutoff_at}))
            except (OSError, ValueError):
                pass

    def _retain(self, snapshot: RunWriteSnapshot) -> None:
        session = snapshot.root_trajectory_id
        snapshot.validate(session)
        root = next(document for document in snapshot.documents if document.trajectory_id == session)
        documents = {}
        for document in snapshot.documents:
            relative = document.path.relative_to(root.path.parent)
            relative = relative.with_name(relative.name.removesuffix(PARTIAL_SUFFIX))
            name = (run_directory(Path(), session) / _relative(relative.as_posix())).as_posix()
            if name in documents:
                raise ValueError('duplicate checkpoint path')
            documents[name] = {'sha256': hashlib.sha256(document.json_bytes).hexdigest(),
                               'json': document.json_bytes.decode('utf-8')}
        bundle = {
            'schema': 2, 'kind': 'checkpoint', 'recorder_status': snapshot.status,
            'cutoff_at': snapshot.cutoff_at, 'session_id': session,
            'root': (run_directory(Path(), session) / RUN_DOCUMENT_NAME).as_posix(),
            'terminal_evidence': 'unavailable until joined final export', 'documents': documents,
        }
        body = (json.dumps(bundle, allow_nan=False) + '\n').encode()
        _decode_checkpoint(body)
        self._sequence += 1
        generation = f'{self._sequence:08d}'
        with self._root() as directory:
            checkpoint = Path(f'checkpoint-{generation}.json')
            _create_bytes(directory, checkpoint, body)
            self._retained.append((checkpoint, body))
            while len(self._retained) > 2:
                checkpoint, body = self._retained.pop(0)
                try:
                    if _read_bytes(directory, checkpoint) == body:
                        os.unlink(checkpoint.name, dir_fd=directory)
                except (OSError, ValueError):
                    pass  # Preserve competing bytes and symlinks.


def read_checkpoint(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a committed generation, rejecting symlinks, corruption and bad refs."""
    with _absolute_directory(path.parent) as descriptor:
        return _decode_checkpoint(_read_bytes(descriptor, Path(path.name)))


def _decode_checkpoint(body: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(body)
    if (not isinstance(manifest, dict) or type(manifest.get('schema')) is not int or manifest['schema'] != 2
            or manifest.get('kind') != 'checkpoint' or manifest.get('recorder_status') not in ('partial', 'complete')
            or any(not isinstance(manifest.get(key), str) or not manifest[key]
                   for key in ('cutoff_at', 'session_id', 'root'))):
        raise ValueError('unsupported or malformed checkpoint')
    inventory = manifest.get('documents')
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('checkpoint document inventory is malformed')
    session = manifest['session_id']
    expected_root = (run_directory(Path(), session) / RUN_DOCUMENT_NAME).as_posix()
    if manifest['root'] != expected_root or expected_root not in inventory:
        raise ValueError('checkpoint root unavailable')
    by_path: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    for relative, entry in inventory.items():
        _relative(relative)
        if not isinstance(entry, dict) or not isinstance(entry.get('json'), str):
            raise ValueError('checkpoint document is malformed')
        if hashlib.sha256(entry['json'].encode()).hexdigest() != entry.get('sha256'):
            raise ValueError('checkpoint bytes differ')
        document = json.loads(entry['json'])
        if (not isinstance(document, dict) or document.get('session_id') != session
                or not isinstance(document.get('trajectory_id'), str) or not document['trajectory_id']
                or document['trajectory_id'] in identities):
            raise ValueError('checkpoint identity differs')
        extra = document.get('extra', {})
        if not isinstance(extra, dict) or any(
            not isinstance(extra.get(key, []), list) or any(not isinstance(value, dict) for value in extra[key])
            for key in ('subtrajectories', 'phase_events', 'active_dispatches') if key in extra
        ):
            raise ValueError('checkpoint activity is malformed')
        identities.add(document['trajectory_id'])
        by_path[relative] = document
    for document in by_path.values():
        for entry in document.get('extra', {}).get('subtrajectories', []):
            if 'sibling_trajectory_ref' in entry:
                reference = entry['sibling_trajectory_ref']
                if (not isinstance(reference, str) or reference not in by_path
                        or entry.get('trajectory_id') != by_path[reference]['trajectory_id']):
                    raise ValueError('checkpoint reference unavailable or mismatched')
    if by_path[expected_root]['trajectory_id'] != session:
        raise ValueError('checkpoint root unavailable')
    manifest['documents'] = {path: entry['sha256'] for path, entry in inventory.items()}
    return manifest, list(by_path.values())


def interrupted_readout(evidence: Path) -> dict[str, Any]:
    """Summarize retained observations without inference or scoring interrupted work."""
    checkpoints = evidence.with_name(evidence.name + '-checkpoints')
    availability: dict[str, str] = {}
    result: dict[str, Any] = {'terminal_evidence': 'unavailable', 'checkpoint': 'unavailable',
                              'input_availability': availability}
    for label, path in {'identity_and_policy': checkpoints / 'metadata.json',
                        'execution': evidence / 'execution.json',
                        'process': evidence.parent / 'reviewer-process.json',
                        'host': evidence.parent / 'reviewer-host.json'}.items():
        if (value := _optional_json(path, availability)) is not None:
            result[label] = value
    if 'execution' in result:
        result['terminal_evidence'] = 'joined export available; inspect execution status'
    # Harbor alone labels an outer timeout. SIGTERM/task cancellation in the
    # reviewer cannot establish whether its caller timed out or was cancelled.
    trial = evidence.parent.parent / 'result.json'
    if (trial_result := _optional_json(trial, availability)) is not None:
        exception = trial_result.get('exception_info')
        if exception is not None and not isinstance(exception, dict):
            availability[trial.name] = 'unavailable:ValueError'
        else:
            exception = exception or {}
            result['harbor_outcome'] = {
                'exception_type': exception.get('exception_type'),
                'occurred_at': exception.get('occurred_at'),
                'agent_execution': trial_result.get('agent_execution'),
            }
    failures = []
    try:
        with _absolute_directory(checkpoints) as directory:
            names = os.listdir(directory)
    except FileNotFoundError:
        names = []
    for name in sorted((name for name in names if re.fullmatch(r'checkpoint-[0-9]{8}\.json', name)), reverse=True):
        generation = checkpoints / name
        try:
            manifest, documents = read_checkpoint(generation)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({'generation': generation.stem.removeprefix('checkpoint-'), 'reason': type(exc).__name__})
            continue
        result['checkpoint'] = {'generation': generation.stem.removeprefix('checkpoint-'), **manifest}
        result['activity'] = [{
            'trajectory_id': document['trajectory_id'],
            'phase_events': document.get('extra', {}).get('phase_events', []),
            'invocations': document.get('extra', {}).get('subtrajectories', []),
            'active_dispatches': document.get('extra', {}).get('active_dispatches', []),
        } for document in documents]
        break
    result['unavailable_checkpoints'] = failures
    result['publication_failures'] = [_optional_json(checkpoints / name, availability) for name in sorted(names)
                                      if re.fullmatch(r'failure-[0-9]{8}\.json', name)]
    result['interpretation'] = (
        'Host public observations only. Silence is not reasoning time. Missing usage and terminal evidence '
        'remain unavailable. Checkpoints do not establish completed analysis, findings, scores, '
        'who cancelled a run, or a provider root cause.'
    )
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m daydream.benchmark.reviewbench.diagnostics PATH_TO_DAYDREAM_EVIDENCE')
    print(redact_structured_text(json.dumps(interrupted_readout(Path(sys.argv[1])), indent=2)))
