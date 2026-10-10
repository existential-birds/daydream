"""Immutable invocation-local source authority, separate from supporting artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from daydream.git_ops.models import GitError
from daydream.git_ops.source import frozen_source


def source_line_bytes(body: bytes) -> list[bytes]:
    """Source offsets follow Git/Pi LF lines, retaining exact original bytes."""
    parts = body.split(b'\n')
    return [part + b'\n' for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


@dataclass(frozen=True)
class SourceWindow:
    """Real source bytes bound to an original location and covered assignment units."""

    target_ids: tuple[str, ...]
    file: str
    source_path: str
    side: Literal['before', 'after']
    revision: str
    content_sha256: str
    blob_oid: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    body: str
    projection: Path | None = None
    read_required: bool = True

    def metadata(self) -> dict[str, Any]:
        """Public source identity; host projection locations are intentionally separate."""
        return {name: getattr(self, name) for name in (
            'file', 'source_path', 'side', 'revision', 'content_sha256', 'blob_oid',
            'start_line', 'end_line', 'start_byte', 'end_byte',
        )} | {'target_ids': list(self.target_ids)}

    def covers(self, other: SourceWindow) -> bool:
        """Coverage requires the same frozen side and matching enclosed source bytes."""
        identity = ('file', 'source_path', 'side', 'revision', 'content_sha256', 'blob_oid')
        if (any(getattr(self, name) != getattr(other, name) for name in identity)
                or not self.start_byte <= other.start_byte <= other.end_byte <= self.end_byte
                or not self.start_line <= other.start_line <= other.end_line <= self.end_line):
            return False
        start, end = other.start_byte - self.start_byte, other.end_byte - self.start_byte
        return self.body.encode()[start:end] == other.body.encode()


@dataclass(frozen=True)
class SourceRecipe:
    """An explicit packet grant contains only this invocation's frozen windows."""

    windows: tuple[SourceWindow, ...]
    repo: Path
    allowed_files: tuple[str, ...] = ()
    head_revision: str | None = None
    repository_files: tuple[str, ...] = ()
    # Host-only authority: strict complete enumeration of this exact HEAD.
    # Legacy/default empty inventories cannot establish absence.
    repository_inventory_revision: str | None = None

    def unavailable_lookup(self, name: str, data: dict[str, Any], cwd: Path,
                           supporting_paths: set[Path]) -> bool:
        """Prove source-free lookup intent from frozen authority and no-follow absence.

        Native completion/provenance is checked by the receipt owner. Before-side
        logical paths do not imply that a normal current-side operand exists.
        """
        if name == 'read_source':
            target, side = data.get('target_id'), data.get('side')
            return (set(data) == {'target_id', 'side'} and isinstance(target, str)
                    and bool(target) and len(target.encode()) <= 2048 and isinstance(side, str)
                    and side in {'before', 'after'}
                    and not any(target in window.target_ids and side == window.side for window in self.windows)
                    and not (side == 'after' and self.permits_repository_read(target)))
        if name != 'read' or set(data) - {'path', 'offset', 'limit'}:
            return False
        path = data.get('path')
        offset, limit = data.get('offset', 1), data.get('limit')
        if (not isinstance(path, str) or not path or '\x00' in path or len(path.encode()) > 2048
                or type(offset) is not int or offset < 1
                or limit is not None and (type(limit) is not int or limit < 1)
                or '..' in Path(path).parts or not path.isascii() or "'" in path
                or path.startswith(('~', '@', '//')) or any(ord(char) < 32 for char in path)
                or path.casefold().startswith('file:')
                or any(marker in path.lower() for marker in (' am.', ' pm.'))):
            return False
        try:
            lexical = Path(os.path.abspath(cwd / path))
            if any(parent.is_symlink() for parent in (lexical, *lexical.parents)):
                return False
            # lstat distinguishes absent operands from broken symlinks and I/O errors.
            try:
                lexical.lstat()
            except FileNotFoundError:
                pass
            else:
                return False
            def normalize(value: str) -> str:
                return unicodedata.normalize('NFD', value).casefold()

            if any(normalize(str(lexical)) == normalize(str(path)) for path in supporting_paths) or any(
                window.projection is not None and normalize(str(lexical)) == normalize(str(window.projection.resolve()))
                for window in self.windows
            ):
                return False
            if not lexical.is_relative_to(cwd):
                return (Path(path).is_absolute()
                        and not normalize(str(lexical)).startswith(normalize(str(cwd)) + '/'))
            relative = lexical.relative_to(cwd).as_posix()
            if (self.head_revision is None or not re.fullmatch(r'[0-9a-f]{40}', self.head_revision)
                    or self.repository_inventory_revision != self.head_revision):
                return False
            # Case/Unicode aliases of a frozen identity never establish absence.
            return not any(normalize(relative) == normalize(known) for known in self.repository_files) and not any(
                window.side == 'after' and normalize(relative) == normalize(window.source_path)
                for window in self.windows
            )
        except (OSError, ValueError, RuntimeError):
            return False

    def permits_repository_read(self, path: str) -> bool:
        """Normal repository reads may investigate frozen tracked dependencies.

        Tracked paths also select frozen current-side dependencies, never artifacts.
        Every admitted range still needs independent regular-blob verification.
        """
        return path in self.allowed_files or path in self.repository_files

    def _repository_window(self, path: str) -> SourceWindow | None:
        if self.head_revision is None or not self.permits_repository_read(path):
            return None
        try:
            oid, raw = frozen_source(self.repo, self.head_revision, path)
            body = raw.decode('utf-8')
        except (GitError, UnicodeError):
            return None
        return SourceWindow((path,), path, path, 'after', self.head_revision,
                            hashlib.sha256(raw).hexdigest(), oid, 1,
                            max(1, len(source_line_bytes(raw))), 0, len(raw), body,
                            read_required=False)

    def selector(self, target_id: str, side: str) -> SourceWindow | None:
        if (not isinstance(target_id, str) or not target_id or len(target_id.encode()) > 2048
                or not isinstance(side, str) or side not in {'before', 'after'}):
            return None
        matches = [w for w in self.windows if target_id in w.target_ids and w.side == side]
        return matches[0] if len(matches) == 1 else None

    def read_source(self, target_id: str, side: str) -> str:
        """Serve one independently verified frozen window, without executing agent shell."""
        window = self.selector(target_id, side)
        if (window is None and side == 'after' and isinstance(target_id, str)
                and not any(target_id in candidate.target_ids and side == candidate.side
                            for candidate in self.windows)):
            window = self._repository_window(target_id)
        if window is None or not self.verify_window(window):
            raise ValueError('Frozen source selector unavailable')
        text = json.dumps({'source': window.metadata(), 'body': window.body}, ensure_ascii=False)
        if len(text.encode()) > 2 * 1024 * 1024:
            raise ValueError('Frozen source window exceeds bound')
        return text

    def verify_window(self, window: SourceWindow) -> bool:
        """Revalidate identities and bytes independently of tool-returned provenance."""
        identity = ('file', 'source_path', 'side', 'revision', 'content_sha256', 'blob_oid')
        allowed = any(
            all(getattr(allowed, name) == getattr(window, name) for name in identity)
            and set(window.target_ids) <= set(allowed.target_ids) for allowed in self.windows
        ) or (window.side == 'after' and window.file == window.source_path
              and self.permits_repository_read(window.file) and window.revision == self.head_revision
              and set(window.target_ids) <= {window.file})
        if len(window.body.encode()) > 2 * 1024 * 1024 or not allowed:
            return False
        try:
            oid, raw = frozen_source(self.repo, window.revision, window.source_path)
            lines = source_line_bytes(raw)
            start = sum(len(line) for line in lines[:window.start_line - 1])
            end = sum(len(line) for line in lines[:window.end_line])
            # Continuations may begin/end within a long UTF-8 line.
            return (oid == window.blob_oid and hashlib.sha256(raw).hexdigest() == window.content_sha256
                    and 1 <= window.start_line <= window.end_line <= max(1, len(lines))
                    and start <= window.start_byte <= window.end_byte <= end
                    and raw[window.start_byte:window.end_byte] == window.body.encode())
        except (GitError, OSError, UnicodeError):
            return False

    def revalidate(self) -> None:
        from daydream.prompt_budget import SanctionedInputTransport, SourceAccessUnavailable, _capture_input

        for window in self.windows:
            if not self.verify_window(window):
                raise SourceAccessUnavailable('frozen source identity or bytes changed')
            if window.projection is not None:
                lexical = Path(os.path.abspath(window.projection))
                if any(parent.is_symlink() for parent in (lexical, *lexical.parents)):
                    raise SourceAccessUnavailable('frozen source projection is not confined')
                captured = _capture_input('source', window.projection, SanctionedInputTransport.EXACT_PATHS, 0,
                                          text_budget=2 * 1024 * 1024)
                if captured.text != window.body or captured.sha256 != hashlib.sha256(window.body.encode()).hexdigest():
                    raise SourceAccessUnavailable('frozen source projection changed')

    def native_result(self, call_input: dict[str, Any], output: str) -> SourceWindow | None:
        if set(call_input) != {'target_id', 'side'}:
            return None
        try:
            value = json.loads(output)
        except (ValueError, TypeError):
            return None
        if not isinstance(value, dict) or set(value) != {'source', 'body'}:
            return None
        target, side = call_input.get('target_id'), call_input.get('side')
        window = self.selector(target, side) if isinstance(target, str) and isinstance(side, str) else None
        if (window is None and isinstance(target, str) and side == 'after'
                and not any(target in candidate.target_ids and side == candidate.side for candidate in self.windows)):
            window = self._repository_window(target)
        if (window is None
                or json.dumps(value['source'], sort_keys=True) != json.dumps(window.metadata(), sort_keys=True)
                or value['body'] != window.body):
            return None
        return window if self.verify_window(window) else None

    def match_read(self, call_input: dict[str, Any], output: str, cwd: Path) -> SourceWindow | None:
        """Accept only known native read ranges and the exact frozen body codec."""
        command = call_input.get('command', call_input.get('cmd'))
        if isinstance(command, str):
            try:
                words = shlex.split(command)
            except ValueError:
                return None
            if len(words) == 3 and words[:2] == ['git', 'show'] and ':' in words[2]:
                revision, source_path = words[2].split(':', 1)
                for window in self.windows:
                    if window.revision == revision and window.source_path == source_path:
                        full = replace(window, body=output, start_byte=0, end_byte=len(output.encode()),
                                       start_line=1, end_line=max(1, len(source_line_bytes(output.encode()))))
                        if self.verify_window(full):
                            return full
                if revision == self.head_revision:
                    dependency = self._repository_window(source_path)
                    if dependency is not None and dependency.body == output and self.verify_window(dependency):
                        return dependency
            if len(words) == 2 and words[0] == 'cat':
                return self.match_read({'file_path': words[1]}, output, cwd)
            return None
        path = call_input.get('file_path', call_input.get('path', call_input.get('file')))
        if not isinstance(path, str) or '..' in Path(path).parts:
            return None
        lexical = Path(path) if Path(path).is_absolute() else cwd / path
        # Ignore platform root aliases such as macOS /tmp by canonicalizing cwd
        # and the declared projection root during host creation, never a tool path.
        lexical = Path(os.path.abspath(lexical))
        if lexical.is_symlink():
            return None
        resolved = lexical.resolve()
        windows = list(self.windows)
        if (resolved.is_relative_to(cwd) and self.permits_repository_read(resolved.relative_to(cwd).as_posix())
                and self.head_revision is not None):
            path = resolved.relative_to(cwd).as_posix()
            if not any(window.side == 'after' and window.source_path == path for window in windows):
                dependency = self._repository_window(path)
                if dependency is not None:
                    windows.append(dependency)
        for window in windows:
            projection = window.projection is not None and resolved == window.projection.resolve()
            repository = (window.side == 'after' and resolved == (cwd / window.source_path).resolve())
            if not projection and not repository:
                continue
            trusted = window.projection.parent if projection and window.projection is not None else cwd
            trusted = trusted.resolve()
            if not lexical.is_relative_to(trusted) or any(
                parent.is_symlink() for parent in lexical.parents if parent.is_relative_to(trusted)
            ):
                continue
            # Pi read and Claude Read share one-based offset/limit line semantics.
            offset = call_input.get('offset', 1)
            limit = call_input.get('limit')
            if type(offset) is not int or offset < 1 or limit is not None and (type(limit) is not int or limit < 1):
                continue
            if set(call_input) - {'path', 'file_path', 'file', 'offset', 'limit'}:
                continue
            basis = window
            if repository:
                try:
                    _, raw = frozen_source(self.repo, window.revision, window.source_path)
                    body = raw.decode('utf-8')
                except (GitError, UnicodeError):
                    continue
                basis = replace(window, start_line=1, end_line=max(1, len(source_line_bytes(raw))),
                                start_byte=0, end_byte=len(raw), body=body)
            lines = [line.decode('utf-8') for line in source_line_bytes(basis.body.encode())]
            selected = lines[offset - 1:None if limit is None else offset - 1 + limit]
            body = ''.join(selected)
            decoded = output == body
            if not decoded and 'path' in call_input:
                # Pi's bounded Read codec joins LF lines and appends an advisory
                # footer when the explicit limit leaves more file lines. This
                # is complete delivery of the requested range, not truncation.
                # Derive the entire representation independently; never strip
                # an arbitrary notice or trust its claimed counts/positions.
                native_lines = basis.body.split('\n')
                end_line = len(native_lines) if limit is None else min(offset - 1 + limit, len(native_lines))
                native_body = '\n'.join(native_lines[offset - 1:end_line])
                if limit is not None and end_line < len(native_lines):
                    native_body += (f'\n\n[{len(native_lines) - end_line} more lines in file. '
                                    f'Use offset={end_line + 1} to continue.]')
                decoded = output == native_body
            if not decoded and 'file_path' in call_input:
                # Claude's Read codec numbers consecutive source lines. Verify
                # every number and payload against independently retrieved bytes;
                # arbitrary wrappers, skipped lines and suffixes remain opaque.
                numbered = [re.fullmatch(r'[ \t]*(\d+)→(.*)', line) for line in output.splitlines()]
                decoded = (len(numbered) == len(selected) and all(
                    match is not None and int(match[1]) == offset + index
                    and match[2] == expected.rstrip('\n')
                    for index, (match, expected) in enumerate(zip(numbered, selected, strict=True))))
            if not decoded or not body and basis.body:
                continue
            start = basis.start_byte + len(''.join(lines[:offset - 1]).encode())
            candidate = replace(basis, body=body, start_byte=start, end_byte=start + len(body.encode()),
                                start_line=basis.start_line + offset - 1,
                                end_line=max(basis.start_line + offset - 1,
                                             basis.start_line + offset + len(selected) - 2))
            if self.verify_window(candidate):
                return candidate
        return None
