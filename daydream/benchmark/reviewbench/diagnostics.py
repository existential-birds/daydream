"""Immutable ReviewBench checkpoints and an offline public-evidence readout.

Only recorder-supplied bytes cross this boundary. Native exchange directories
are never scanned or copied. A checkpoint manifest commits its entire document
set; an interrupted install cannot supersede the previous valid checkpoint.
"""
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
    """Install flushed bytes without following any parent or replacing a leaf."""
    with _directory(directory, relative.parts[:-1], create=True) as parent:
        temporary = '.checkpoint-' + uuid.uuid4().hex + '.tmp'
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, relative.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            os.fsync(parent)
        finally:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=parent)


def publish_diagnostic_bytes(path: Path, body: bytes) -> None:
    """Publish one host diagnostic without following ancestry or replacing bytes."""
    with _absolute_directory(path.parent, create=True) as directory:
        _create_bytes(directory, Path(path.name), body)


def _read_bytes(directory: int, relative: Path) -> bytes:
    """Read only a regular file through no-follow descriptor-relative parents."""
    with _directory(directory, relative.parts[:-1]) as parent:
        try:
            descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=parent)
        except OSError as exc:
            try:
                symbolic = stat.S_ISLNK(os.stat(relative.name, dir_fd=parent, follow_symlinks=False).st_mode)
            except OSError:
                symbolic = False
            if symbolic:
                raise ValueError('checkpoint symlink rejected') from exc
            raise
        with os.fdopen(descriptor, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('checkpoint file must be regular')
            return stream.read()


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
    """Publish at most two committed generations with no-clobber owned paths.

    The diagnostic sink is optional: failure records an explicit availability
    reason and never changes review authority. Every generation is a new tree;
    there is no mutable latest pointer and no copied live artifact tree.
    """

    def __init__(self, evidence: Path, metadata: dict[str, Any]) -> None:
        self.checkpoint_directory = evidence.with_name(evidence.name + '-checkpoints')
        with _absolute_directory(self.checkpoint_directory.parent, create=True) as parent:
            os.mkdir(self.checkpoint_directory.name, mode=0o700, dir_fd=parent)
            with _directory(parent, [self.checkpoint_directory.name]) as directory:
                self._directory_identity = _identity(directory)
                _create_bytes(directory, Path('metadata.json'), _json_bytes(metadata))
        self.last_error: str | None = None
        self._sequence = 0
        self._retained: list[tuple[str, tuple[int, int], dict[Path, bytes]]] = []

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
        files: dict[Path, bytes] = {}
        for document in snapshot.documents:
            # Paths are used only for logical layout, never read from disk.
            relative = document.path.relative_to(root.path.parent)
            name = relative.name.removesuffix(PARTIAL_SUFFIX)
            relative = _relative(relative.with_name(name).as_posix())
            destination = run_directory(Path(), session) / relative
            if destination in files:
                raise ValueError('duplicate checkpoint path')
            files[destination] = document.json_bytes
        for document in snapshot.documents:
            payload = document.validated_payload(session)
            for entry in payload.get('extra', {}).get('subtrajectories', []):
                if 'sibling_trajectory_ref' in entry and _relative(entry['sibling_trajectory_ref']) not in files:
                    raise ValueError('checkpoint sibling reference unavailable')
        self._sequence += 1
        generation = f'{self._sequence:08d}'
        manifest = {
            'schema': 1, 'kind': 'checkpoint', 'recorder_status': snapshot.status,
            'cutoff_at': snapshot.cutoff_at, 'session_id': session,
            'root': (run_directory(Path(), session) / RUN_DOCUMENT_NAME).as_posix(),
            'terminal_evidence': 'unavailable until joined final export',
            'documents': {path.as_posix(): hashlib.sha256(body).hexdigest() for path, body in files.items()},
        }
        # checkpoint.json is the commit record. Readers ignore directories
        # without it, including a SIGKILL halfway through a document install.
        with self._root() as directory:
            os.mkdir(generation, mode=0o700, dir_fd=directory)
            with _directory(directory, [generation]) as opened:
                identity = _identity(opened)
                for path, body in files.items():
                    _create_bytes(opened, path, body)
                # Refuse an ancestor substituted during document installation.
                # No path-based traversal may publish or prune through its target.
                for path, body in files.items():
                    if _read_bytes(opened, path) != body:
                        raise ValueError('checkpoint document identity changed')
                observed = os.stat(generation, dir_fd=directory, follow_symlinks=False)
                if (observed.st_dev, observed.st_ino) != identity or not stat.S_ISDIR(observed.st_mode):
                    raise ValueError('checkpoint generation identity changed')
                marker = Path('checkpoint.json')
                files[marker] = _json_bytes(manifest)
                _create_bytes(opened, marker, files[marker])
            self._retained.append((generation, identity, files))
            while len(self._retained) > 2:
                name, identity, owned = self._retained.pop(0)
                self._discard(directory, name, identity, owned)

    @staticmethod
    def _discard(root: int, name: str, identity: tuple[int, int], owned: dict[Path, bytes]) -> None:
        """Prune only matching files and directories below anchored no-follow parents."""
        try:
            with _directory(root, [name]) as directory:
                if _identity(directory) != identity:
                    return
                if any(_read_bytes(directory, path) != body for path, body in owned.items()):
                    return
                for path, body in owned.items():
                    with _directory(directory, path.parts[:-1]) as parent:
                        if _read_bytes(parent, Path(path.name)) != body:
                            return
                        os.unlink(path.name, dir_fd=parent)
                parents = {parent for path in owned for parent in path.parents if parent.parts}
                for path in sorted(parents, key=lambda path: len(path.parts), reverse=True):
                    with _directory(directory, path.parts[:-1]) as parent:
                        os.rmdir(path.name, dir_fd=parent)
            observed = os.stat(name, dir_fd=root, follow_symlinks=False)
            if (observed.st_dev, observed.st_ino) == identity and stat.S_ISDIR(observed.st_mode):
                os.rmdir(name, dir_fd=root)
        except (OSError, ValueError):
            return


def read_checkpoint(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a committed generation, rejecting symlinks, corruption and bad refs."""
    with _absolute_directory(directory) as descriptor:
        return _read_checkpoint(descriptor)


def _read_checkpoint(descriptor: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(_read_bytes(descriptor, Path('checkpoint.json')))
    if manifest.get('schema') != 1 or manifest.get('kind') != 'checkpoint':
        raise ValueError('unsupported checkpoint')
    documents = []
    identities = set()
    for relative, digest in manifest['documents'].items():
        body = _read_bytes(descriptor, _relative(relative))
        if hashlib.sha256(body).hexdigest() != digest:
            raise ValueError('checkpoint bytes differ')
        document = json.loads(body)
        if document['session_id'] != manifest['session_id'] or document['trajectory_id'] in identities:
            raise ValueError('checkpoint identity differs')
        identities.add(document['trajectory_id'])
        documents.append(document)
        for entry in document.get('extra', {}).get('subtrajectories', []):
            if 'sibling_trajectory_ref' in entry and entry['sibling_trajectory_ref'] not in manifest['documents']:
                raise ValueError('checkpoint reference unavailable')
    if manifest['session_id'] not in identities or manifest['root'] not in manifest['documents']:
        raise ValueError('checkpoint root unavailable')
    return manifest, documents


def interrupted_readout(evidence: Path) -> dict[str, Any]:
    """Summarize retained observations without inference or scoring interrupted work."""
    checkpoints = evidence.with_name(evidence.name + '-checkpoints')
    availability: dict[str, str] = {}
    result: dict[str, Any] = {'terminal_evidence': 'unavailable', 'checkpoint': 'unavailable',
                              'input_availability': availability}
    metadata = checkpoints / 'metadata.json'
    if (value := _optional_json(metadata, availability)) is not None:
        result['identity_and_policy'] = value
    execution = evidence / 'execution.json'
    if (value := _optional_json(execution, availability)) is not None:
        result['execution'] = value
        result['terminal_evidence'] = 'joined export available; inspect execution status'
    process = evidence.parent / 'reviewer-process.json'
    if (value := _optional_json(process, availability)) is not None:
        result['process'] = value
    host = evidence.parent / 'reviewer-host.json'
    if (value := _optional_json(host, availability)) is not None:
        result['host'] = value
    # Harbor alone labels an outer timeout. SIGTERM/task cancellation in the
    # reviewer cannot establish whether its caller timed out or was cancelled.
    trial = evidence.parent.parent / 'result.json'
    if (trial_result := _optional_json(trial, availability)) is not None:
        exception = trial_result.get('exception_info') or {}
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
    for name in sorted((name for name in names if re.fullmatch(r'[0-9]{8}', name)), reverse=True):
        generation = checkpoints / name
        try:
            manifest, documents = read_checkpoint(generation)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({'generation': generation.name, 'reason': type(exc).__name__})
            continue
        result['checkpoint'] = {'generation': generation.name, **manifest}
        result['activity'] = [{
            'trajectory_id': document['trajectory_id'],
            'phase_events': document.get('extra', {}).get('phase_events', []),
            'invocations': document.get('extra', {}).get('subtrajectories', []),
            'active_dispatches': document.get('extra', {}).get('active_dispatches', []),
        } for document in documents]
        result['last_observed'] = []
        for document in documents:
            extra = document.get('extra', {})
            invocations = extra.get('subtrajectories', [])
            active = [invocation for invocation in invocations
                      if invocation.get('invocation_id') and invocation.get('ended_at') is None]
            phases = extra.get('phase_events', [])
            result['last_observed'].append({
                'trajectory_id': document['trajectory_id'],
                'last_phase_event': phases[-1] if phases else None,
                'active_invocations': active,
                'active_dispatches': extra.get('active_dispatches', []),
                'completed_phase_timings': [event for event in phases if event.get('event') == 'phase_end'],
            })
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
