"""Atomic checkpoints from exact recorder bytes; native exchange directories stay private."""
from __future__ import annotations

import hashlib
import json
import os
import stat
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
        identities = {}
        references: list[dict[str, Any]] = []
        for document in snapshot.documents:
            relative = document.path.relative_to(root.path.parent)
            relative = relative.with_name(relative.name.removesuffix(PARTIAL_SUFFIX))
            name = (run_directory(Path(), session) / _relative(relative.as_posix())).as_posix()
            if name in documents:
                raise ValueError('duplicate checkpoint path')
            payload = document.validated_payload(session)
            extra = payload.get('extra', {})
            if not isinstance(extra, dict) or any(
                not isinstance(extra.get(key, []), list) or any(not isinstance(value, dict) for value in extra[key])
                for key in ('subtrajectories', 'phase_events', 'active_dispatches') if key in extra
            ):
                raise ValueError('checkpoint activity is malformed')
            references.extend(extra.get('subtrajectories', []))
            identities[name] = document.trajectory_id
            documents[name] = {'sha256': hashlib.sha256(document.json_bytes).hexdigest(),
                               'json': document.json_bytes.decode('utf-8')}
        root_name = (run_directory(Path(), session) / RUN_DOCUMENT_NAME).as_posix()
        if identities.get(root_name) != session:
            raise ValueError('checkpoint root unavailable')
        for reference in references:
            if 'sibling_trajectory_ref' in reference:
                target = reference['sibling_trajectory_ref']
                if (not isinstance(target, str) or target not in identities
                        or identities[target] != reference.get('trajectory_id')):
                    raise ValueError('checkpoint sibling reference unavailable or mismatched')
        bundle = {
            'schema': 2, 'kind': 'checkpoint', 'recorder_status': snapshot.status,
            'cutoff_at': snapshot.cutoff_at, 'session_id': session,
            'root': root_name,
            'terminal_evidence': 'unavailable until joined final export', 'documents': documents,
        }
        body = (json.dumps(bundle, allow_nan=False) + '\n').encode()
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
