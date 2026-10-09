"""Frozen canonical changes projected into bounded, complete stage assignments."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from daydream.artifact_visibility import ArtifactSession, artifact_dir_for
from daydream.backends import Backend
from daydream.deep.detection import StackAssignment
from daydream.deep.diff import _DIFF_MINUS_HEADER, _DIFF_PLUS_HEADER, iter_diff_blocks
from daydream.git_ops.models import GitError
from daydream.git_ops.queries import ls_tree_files
from daydream.git_ops.source import frozen_source
from daydream.hunk_index import _HUNK_HEADER, _unquote_git_path
from daydream.json_utils import atomic_write_bytes
from daydream.prompt_budget import (
    SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES,
    AdvisoryCandidate,
    PreparedSanctionedInput,
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    SanctionedInputUnavailable,
    SourceAccessUnavailable,
    _capture_input,
    inline_section_emitted_bytes,
    prepare_sanctioned_inputs,
    sanctioned_transport_for,
    select_advisory_inputs,
)
from daydream.review_source import SourceRecipe, SourceWindow, source_line_bytes
from daydream.workspace import WorkContext


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class _Part:
    assignment: dict[str, Any]
    diff: str


class StageInputFactory:
    """Required work takes priority over advisory context on either transport.

    The logical batches fit the actual inline allowance, including wrappers and
    binding metadata. Exact-path backends receive those same assignments, never
    the canonical whole-change artifacts as their first-pass inputs.
    """

    def __init__(
        self, backend: Backend, work: WorkContext, stack: StackAssignment, *,
        diff_path: Path, hunk_index_path: Path, shared_paths: dict[str, Path | None],
        revision: dict[str, Any], artifact_session: ArtifactSession | None,
        allow_standalone: bool, read_only: bool,
        bundle_capable: bool = False,
    ) -> None:
        self.backend, self.work, self.stack = backend, work, stack
        self.session, self.allow_standalone, self.read_only = artifact_session, allow_standalone, read_only
        self.shared_paths = shared_paths
        self.bundle_capable = bundle_capable
        self.transport = sanctioned_transport_for(backend, work.repo, read_only=read_only)
        # Capture hashes through the existing no-follow, bounded streaming
        # capture. These identities stay host-only, regardless of transport.
        self.canonical = tuple(self._canonical(label, path) for label, path in (
            ("canonical-diff", diff_path), ("canonical-index", hunk_index_path),
        ))
        self.full_diff = diff_path.read_text(encoding="utf-8")
        index_text = hunk_index_path.read_text(encoding="utf-8")
        self.index = json.loads(index_text)
        self._revalidate_canonical()
        if hashlib.sha256(self.full_diff.encode()).hexdigest() != self.canonical[0].sha256:
            raise SanctionedInputUnavailable("canonical change bytes changed during preparation")
        if hashlib.sha256(index_text.encode()).hexdigest() != self.canonical[1].sha256:
            raise SanctionedInputUnavailable("canonical index bytes changed during preparation")
        if not isinstance(self.index, dict):
            raise SanctionedInputUnavailable("canonical hunk index is malformed")
        self.binding = {"analyzed_revision": revision,
                        "canonical_inputs": {item.label: item.sha256 for item in self.canonical}}
        self.scope_name = hashlib.sha256(stack.stack_name.encode()).hexdigest()[:20]
        self.parts = self._parts()
        self.by_id = {part.assignment["target_id"]: part for part in self.parts}
        self.assignment_batches = self._batches()
        self.current_paths: dict[str, Path] = {}
        self.blocks = dict(iter_diff_blocks(self.full_diff))
        # Normal source tools already permit repository dependency inspection.
        # Bind those receipts to this captured tree without expanding the Pi
        # packet or publishing the repository inventory in every prompt.
        self.repository_files = tuple(ls_tree_files(work.repo, revision['head_sha'], strict=True))

    @staticmethod
    def _canonical(label: str, path: Path) -> PreparedSanctionedInput:
        return _capture_input(label, path, SanctionedInputTransport.EXACT_PATHS, 0, pointer_only=True)

    def _revalidate_canonical(self) -> None:
        for expected in self.canonical:
            if self._canonical(expected.label, expected.path) != expected:
                raise SanctionedInputUnavailable("canonical review inputs changed")

    def _render(self, parts: list[_Part]) -> dict[str, str]:
        files = {part.assignment["file"] for part in parts}
        # A part's ranges are authoritative for that assignment; do not expose
        # the rest of a large file's index as required first-pass work.
        index: dict[str, Any] = {}
        for path in sorted(files):
            assignments = [part.assignment for part in parts if part.assignment["file"] == path]
            canonical = self.index.get(path, {})
            hunks = canonical.get('hunks', [])
            if any(part['kind'] == 'file' for part in assignments):
                selected = hunks
            else:
                selected = []
                seen_ranges: set[tuple[int, int, int, int]] = set()
                for part in assignments:
                    bounds = (part['old_start'], part['old_start'] + part['old_count'] - 1,
                              part['new_start'], part['new_start'] + part['new_count'] - 1)
                    if bounds in seen_ranges:
                        continue
                    seen_ranges.add(bounds)
                    keys = ('old_start', 'old_end', 'new_start', 'new_end')
                    matching = next((hunk for hunk in hunks
                                     if tuple(hunk[key] for key in keys) == bounds), None)
                    if matching is not None:
                        selected.append(matching)
                    elif part['new_count'] == 0:
                        # The canonical posting index intentionally omits empty
                        # new-side ranges. Review assignments still retain the
                        # exact frozen diff's old-side deletion authority.
                        selected.append({**dict(zip(keys, bounds, strict=True)),
                                         'added': 0, 'removed': part['old_count'], 'old_only': True})
                    else:
                        raise SanctionedInputUnavailable("canonical hunk ranges do not match assigned work")
            index[path] = {"hunks": selected, "assignments": assignments,
                           "status": "partial" if any(part['part_count'] > 1 for part in assignments) else "complete"}
        return {"diff": "".join(part.diff for part in parts), "hunk-index": _json(index),
                "input-binding": _json(self.binding)}

    def _bundled(self, contents: dict[str, str]) -> dict[str, str]:
        if not self.bundle_capable or not contents:
            return contents
        return {'review-assignment': '\n\n'.join(f'### {label} (supporting)\n{text}'
                                                 for label, text in contents.items())}

    def _fits(self, contents: dict[str, str]) -> bool:
        contents = self._bundled(contents)
        return inline_section_emitted_bytes([(label, len(text.encode())) for label, text in contents.items()]) <= (
            SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES
        )

    def _parts(self) -> list[_Part]:
        result: list[_Part] = []
        reserved_targets = set(self.stack.files)
        next_part = 0
        blocks: dict[str, str] = {}
        for path, block in iter_diff_blocks(self.full_diff):
            blocks[path] = blocks.get(path, "") + block
        for path in sorted(self.stack.files):
            whole = _Part({"target_id": path, "file": path, "part_index": 1, "part_count": 1,
                           "kind": "file", "hunk_index": None}, blocks.get(path, ""))
            if self._fits(self._render([whole])):
                result.append(whole)
                continue
            pieces = self._split_file(path, whole.diff)
            for index, part in enumerate(pieces, 1):
                while True:
                    next_part += 1
                    target_id = f"part:{next_part:06d}"
                    if target_id not in reserved_targets:
                        break
                reserved_targets.add(target_id)
                part.assignment.update(target_id=target_id,
                                       part_index=index, part_count=len(pieces))
            result.extend(pieces)
        return result

    def _split_file(self, path: str, block: str) -> list[_Part]:
        lines = block.splitlines(keepends=True)
        starts = [index for index, line in enumerate(lines) if _HUNK_HEADER.match(line)]
        # Binary/rename metadata is also required work, split explicitly if huge.
        if not starts:
            return self._segments(path, "", block, 0, (0, 0, 0, 0))
        prefix = "".join(lines[:starts[0]])
        parts: list[_Part] = []
        for ordinal, start in enumerate(starts):
            end = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(lines)
            match = _HUNK_HEADER.match(lines[start])
            assert match is not None
            bounds = (int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1))
            header = prefix + lines[start]
            body = "".join(lines[start + 1:end])
            parts.extend(self._segments(path, header, body, ordinal, bounds))
        return parts

    def _segments(
        self, path: str, header: str, body: str, hunk_index: int, bounds: tuple[int, int, int, int],
    ) -> list[_Part]:
        old_start, old_count, new_start, new_count = bounds
        base = {"target_id": "part:000000", "file": path, "part_index": 999999, "part_count": 999999,
                "kind": "hunk", "hunk_index": hunk_index,
                "old_start": old_start, "old_count": old_count, "new_start": new_start, "new_count": new_count}
        if self._fits(self._render([_Part(base, header + body)])):
            return [_Part(base, header + body)]
        # Prefer complete diff lines. Very long single lines use ordered UTF-8
        # fragments with exact byte offsets; none is called a complete hunk.
        chunks: list[tuple[str, int, int, int, int, int]] = []
        rest = body
        byte_offset = 0
        old_line, new_line = old_start, new_start
        line_offset = 0
        line_kind = body[:1]
        while rest:
            low, high = 0, len(rest)
            meta = {**base, "kind": "continuation", "segment_index": 999999, "segment_count": 999999,
                    "fragment_offset": byte_offset, "old_line": old_line, "new_line": new_line,
                    "fragment_line_offset": line_offset, "fragment_bytes": 999999999}
            notice = "[Ordered continuation: this is a partial hunk; see assignment mapping.]\n"
            while low < high:
                mid = (low + high + 1) // 2
                if self._fits(self._render([_Part(meta, notice + header + rest[:mid])])):
                    low = mid
                else:
                    high = mid - 1
            if low == 0:
                raise SanctionedInputUnavailable("required change header exceeds the inline allowance")
            boundary = rest.rfind("\n", 0, low)
            take = boundary + 1 if boundary >= 0 else low
            chunk = rest[:take]
            chunks.append((chunk, byte_offset, old_line, new_line, len(chunk.encode()), line_offset))
            # Count only completed lines. Fragment offsets retain exact mapping
            # for lines split inside a segment; all ranges remain the full hunk.
            for line in chunk.splitlines(keepends=True):
                if line_offset == 0:
                    line_kind = line[:1]
                if line.endswith("\n"):
                    old_line += int(line_kind not in {"+", "\\"})
                    new_line += int(line_kind not in {"-", "\\"})
                    line_offset = 0
                else:
                    line_offset += len(line.encode())
            byte_offset += len(chunk.encode())
            rest = rest[take:]
        return [_Part({**base, "kind": "continuation", "segment_index": index, "segment_count": len(chunks),
                       "fragment_offset": offset, "fragment_bytes": size, "old_line": old, "new_line": new,
                       "fragment_line_offset": line_offset},
                      "[Ordered continuation: this is a partial hunk; see assignment mapping.]\n" + header + chunk)
                for index, (chunk, offset, old, new, size, line_offset) in enumerate(chunks, 1)]

    def _batches(self) -> list[list[dict[str, Any]]]:
        batches: list[list[_Part]] = []
        current: list[_Part] = []
        for part in self.parts:
            if current and (len({p.assignment['file'] for p in [*current, part]}) > 4
                            or not self._fits(self._render([*current, part]))):
                batches.append(current)
                current = []
            if not self._fits(self._render([part])):
                raise SanctionedInputUnavailable("required assignment exceeds the inline allowance")
            current.append(part)
        if current:
            batches.append(current)
        return [[part.assignment for part in batch] for batch in batches]

    def _write(self, relative: Path, text: str) -> Path:
        if self.session is not None:
            return self.session.write_review_input(self.work.repo, relative, text)
        root = artifact_dir_for(self.work.repo, allow_standalone=self.allow_standalone) / "deep" / "stage-inputs"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, text.encode(), fsync=True, dir_fsync=True, mode=0o600)
        return path

    def _supporting_catalog(
        self, directory: Path, entries: list[dict[str, Any]], *, namespace: str = 'supporting',
        entry_key: str = 'parts', count_key: str = 'part_count',
    ) -> tuple[Path, dict[str, Path]]:
        """Publish a bounded exact-pointer catalog without exposing a directory grant."""
        captured: dict[str, Path] = {}
        ordinal = 0

        def write_groups(key: str, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
            nonlocal ordinal
            groups: list[list[dict[str, Any]]] = []
            current: list[dict[str, Any]] = []
            for value in values:
                if len(_json({key: [value]}).encode()) > 12_000:
                    raise SanctionedInputUnavailable('supporting catalog entry exceeds its bounded allowance')
                if current and len(_json({key: [*current, value]}).encode()) > 12_000:
                    groups.append(current)
                    current = []
                current.append(value)
            if current or not groups:
                groups.append(current)
            result: list[dict[str, Any]] = []
            for group in groups:
                label = f'{namespace}-catalog-{ordinal:06d}'
                ordinal += 1
                path = self._write(directory / f'{label}.json', _json({key: group}))
                captured[label] = path
                result.append({'path': str(path), count_key: (
                    len(group) if key == entry_key else sum(child[count_key] for child in group))})
            return result

        level = write_groups(entry_key, entries)
        while len(level) > 1:
            following = write_groups('catalogs', level)
            if len(following) >= len(level):
                raise SanctionedInputUnavailable('supporting catalog cannot fit its bounded index')
            level = following
        return Path(level[0]['path']), captured

    def _access_guide(self, state: dict[str, Any], prepared: PreparedSanctionedInputs) -> str:
        """Persist only captured exact authority, without clipping required access."""
        guide: dict[str, Any] = {'stage': state['stage'], 'contexts': []}
        for key in ('source_catalog', 'supporting_catalog', 'supporting_bundle'):
            if key in state:
                guide[key] = state[key]
        guide['required_source'] = [
            {key: entry[key] for key in ('file', 'side', 'start_line', 'end_line', 'start_byte', 'end_byte', 'access')}
            for entry in state['source_access'] if entry['read_required'] and state['stage'] != 'integration'
        ]
        if state['stage'] == 'integration' and 'source_catalog' in guide:
            guide['required_source'] = 'Use catalog windows marked read_required; relevant source still needs receipts.'
        if len(_json(guide).encode()) > 8192:
            raise SanctionedInputUnavailable('required persistent source access exceeds its bounded allowance')
        omitted: list[str] = []
        for item in prepared.inputs:
            if item.source is not None or not item.prompt_visible:
                continue
            context: dict[str, Any] = {'label': item.label, 'status': 'complete'}
            if prepared.transport is SanctionedInputTransport.EXACT_PATHS:
                context['path'] = str(item.path)
                if item.inline_advisory:
                    context['initial_context'] = 'complete inline'
            else:
                context['transport'] = 'complete inline'
            candidate = {**guide, 'contexts': [*guide['contexts'], context]}
            # Reserve room to report optional omissions explicitly.
            if len(_json(candidate).encode()) <= 7168:
                guide = candidate
            else:
                omitted.append(item.label)
        if omitted:
            guide['unrepresented_optional_context'] = omitted
        if state['stage'] == 'integration' and 'source_catalog' not in guide:
            # Custom builders keep their full initial inventory and transport.
            # Do not turn an additive guide into a new whole-change input gate,
            # or invent a catalog grant on an unsupported/inline transport.
            guide['source_inventory'] = {'transport': 'initial supplied source_access',
                                          'windows': len(state['source_access']),
                                          'read_required_windows': sum(entry['read_required']
                                                                       for entry in state['source_access']),
                                          'persistent_arguments': 'unrepresented; use supplied source_access'}
        if len(_json(guide).encode()) > 8192:
            raise SanctionedInputUnavailable('persistent access guide exceeds its bounded allowance')
        return _json(guide)

    def _source_windows(self, parts: list[_Part], directory: Path,
                        statuses: list[dict[str, Any]], *, required_windows: bool | None = None) -> SourceRecipe:
        """Freeze side-specific source, with bounded enclosing context for large parts."""
        windows: list[SourceWindow] = []
        frozen: dict[tuple[str, str], tuple[str, bytes]] = {}
        for part in parts:
            assignment = part.assignment
            path = assignment['file']
            block = self.blocks.get(path, '')
            side_paths: dict[str, str | None] = {}
            for side, pattern, prefix in (('before', _DIFF_MINUS_HEADER, 'a/'),
                                          ('after', _DIFF_PLUS_HEADER, 'b/')):
                match = pattern.search(block)
                selected = _unquote_git_path(match[1].rstrip('\t')) if match else path
                side_paths[side] = None if selected == '/dev/null' else selected.removeprefix(prefix)
            # Rename-only blocks have no ---/+++ headers.
            for side, marker in (('before', 'rename from '), ('after', 'rename to ')):
                rename = next((line[len(marker):] for line in block.splitlines() if line.startswith(marker)), None)
                if rename is not None:
                    side_paths[side] = _unquote_git_path(rename)
            old_only = assignment.get('new_count') == 0 and assignment.get('old_count', 0) > 0
            if assignment['kind'] == 'continuation':
                # _HUNK_HEADER is line-oriented; split to retain byte exact origins.
                chunks: list[str] = []
                for line in block.splitlines(keepends=True):
                    if _HUNK_HEADER.match(line):
                        chunks.append('')
                    elif chunks:
                        chunks[-1] += line
                ordinal = assignment.get('hunk_index', 0)
                if ordinal < len(chunks):
                    hunk_body = chunks[ordinal].encode()
                    first = assignment.get('fragment_offset', 0)
                    selected = hunk_body[first:first + assignment.get('fragment_bytes', 0)]
                    selected_lines = selected.splitlines()
                    kinds = {line[:1] for line in selected_lines[
                        1 if assignment.get('fragment_line_offset', 0) else 0:]}
                    if assignment.get('fragment_line_offset', 0):
                        origin = hunk_body.rfind(b'\n', 0, first) + 1
                        kinds.add(hunk_body[origin:origin + 1])
                    old_only = b'-' in kinds and b'+' not in kinds
            required_side = 'before' if side_paths['after'] is None or old_only else 'after'
            for side in ('before', 'after'):
                source_path = side_paths[side]
                if source_path is None:
                    continue
                revision = self.binding['analyzed_revision']['merge_base_sha' if side == 'before' else 'head_sha']
                try:
                    key = (revision, source_path)
                    if key not in frozen:
                        frozen[key] = frozen_source(self.work.repo, revision, source_path)
                    oid, raw = frozen[key]
                except GitError:
                    if side == required_side:
                        raise SourceAccessUnavailable('required frozen source is unavailable') from None
                    statuses.append({'label': f'source-{side}', 'file': path, 'side': side,
                                     'status': 'unavailable', 'read_required': False})
                    continue
                try:
                    raw.decode('utf-8')
                except UnicodeError:
                    if side == required_side:
                        raise SourceAccessUnavailable('required frozen source is not UTF-8') from None
                    continue
                start, end = self._window_bounds(raw, part, side)
                body = raw[start:end].decode('utf-8')
                start_line = raw[:start].count(b'\n') + 1
                end_line = max(start_line, raw[:end].count(b'\n') + int(end > 0 and raw[end - 1:end] != b'\n'))
                window = SourceWindow(
                    target_ids=(assignment['target_id'],), file=path, source_path=source_path,
                    side='before' if side == 'before' else 'after', revision=revision,
                    content_sha256=hashlib.sha256(raw).hexdigest(), blob_oid=oid,
                    start_line=start_line, end_line=end_line, start_byte=start, end_byte=end, body=body,
                    read_required=side == required_side if required_windows is None else required_windows,
                )
                # Reuse one full source projection across multiple units in this invocation.
                same = next((i for i, prior in enumerate(windows)
                             if replace(prior, target_ids=window.target_ids, projection=None) == window), None)
                if same is not None:
                    windows[same] = replace(windows[same], target_ids=tuple(dict.fromkeys(
                        (*windows[same].target_ids, *window.target_ids))))
                    continue
                if self.transport is SanctionedInputTransport.EXACT_PATHS:
                    projection = self._write(directory / f'source-{len(windows):06d}.txt', body)
                    window = replace(window, projection=projection)
                windows.append(window)
        if sum(len(window.body.encode()) for window in windows) > 8 * 1024 * 1024:
            raise SanctionedInputUnavailable('invocation source recipe exceeds retained source bound')
        recipe = SourceRecipe(tuple(windows), self.work.repo,
                              tuple(sorted(set(self.stack.files + self.stack.frontier_files))),
                              self.binding['analyzed_revision']['head_sha'], self.repository_files,
                              repository_inventory_revision=self.binding['analyzed_revision']['head_sha'])
        recipe.revalidate()
        return recipe

    def _source_access(self, window: SourceWindow) -> dict[str, Any]:
        access: dict[str, Any] = {}
        if window.projection is not None:
            access['path'] = str(window.projection)
        if getattr(self.backend, 'supports_source_recipe', False) is True:
            access.update(tool='read_source', arguments={'target_id': window.target_ids[0], 'side': window.side})
        elif not access and getattr(self.backend, 'read_only_disposable_clone', False) is True:
            access.update(tool='exec_command', arguments={'cmd': 'git show ' + shlex.quote(
                f'{window.revision}:{window.source_path}')})
        elif not access and window.side == 'after':
            access.update(tool='Read', arguments={'file_path': window.source_path, 'offset': window.start_line,
                                                  'limit': window.end_line - window.start_line + 1})
        elif not access:
            access['status'] = 'unavailable'
        return window.metadata() | {'read_required': window.read_required, 'access': access}

    @staticmethod
    def _window_bounds(raw: bytes, part: _Part, side: str) -> tuple[int, int]:
        if len(raw) <= 128 * 1024:
            return 0, len(raw)
        lines = source_line_bytes(raw)
        key = 'old' if side == 'before' else 'new'
        assignment = part.assignment
        first = max(1, assignment.get(f'{key}_line', assignment.get(f'{key}_start', 1)))
        count = assignment.get(f'{key}_count', len(lines))
        # Ordered continuation payloads keep their own line/fragment origin.
        if assignment['kind'] == 'continuation':
            fragment = part.diff.split('\n')
            header = next((index for index, line in enumerate(fragment) if _HUNK_HEADER.match(line)), -1)
            count = max(1, len(fragment[header + 1:]))
        begin_line, end_line = max(1, first - 32), min(len(lines), first + max(1, count) + 32)
        start = sum(len(line) for line in lines[:begin_line - 1])
        end = sum(len(line) for line in lines[:end_line])
        if end - start > 128 * 1024:
            origin = sum(len(line) for line in lines[:first - 1])
            offset = max(0, assignment.get('fragment_line_offset', 0) - 1)
            start = max(start, origin + offset - 4096)
            end = min(end, origin + offset + max(assignment.get('fragment_bytes', 0), 48 * 1024) + 4096)
            # Move inward only to valid UTF-8 boundaries.
            while start < end and raw[start] & 0xC0 == 0x80:
                start += 1
            while end < len(raw) and end > start and raw[end] & 0xC0 == 0x80:
                end -= 1
        return start, end

    def prepare(self, state: dict[str, Any]) -> PreparedSanctionedInputs | None:
        self._revalidate_canonical()
        shared_paths = {} if state['stage'] == 'triage' else self.shared_paths
        parts = [self.by_id[target] for target in state['assigned_target_ids'] if target in self.by_id]
        if state['stage'] in {'triage', 'integration'}:
            # The existing file selectors are the source inventory for these duties.
            parts = [_Part({'target_id': path, 'file': path, 'kind': 'file'}, self.blocks.get(path, ''))
                     for path in state['assigned_files']]
        statuses: list[dict[str, str]] = []
        deferred_contents: dict[str, str] = {}
        if state['stage'] == 'triage':
            contents = {}
        elif state['stage'] == 'integration':
            # Structure remains one interaction assignment. Inventory is honest
            # about all changed files; bounded diff parts are supporting context.
            inventory = {path: {"hunks": len(self.index.get(path, {}).get('hunks', [])),
                                "parts": sum(p.assignment['file'] == path for p in self.parts)}
                         for path in self.stack.files}
            contents = {"diff": "Whole-change interaction assignment. Use the inventory and targeted source reads.\n",
                        "hunk-index": _json(inventory), "input-binding": _json(self.binding)}
            if not self._fits(contents):
                contents['hunk-index'] = _json({"status": "partial", "file_count": len(inventory),
                                               "canonical_sha256": self.canonical[1].sha256})
                statuses.append({"label": "hunk-index", "status": "partial"})
            # Expose bounded complete supporting parts under exact confinement;
            # INLINE selects whole parts fitting the same aggregate allowance.
            for index, part in enumerate(self.parts):
                label = f"diff-part-{index:06d}"
                if self.bundle_capable and self.transport is SanctionedInputTransport.EXACT_PATHS:
                    deferred_contents[label] = part.diff
                    continue
                candidate = {**contents, label: part.diff}
                if (self.transport is SanctionedInputTransport.EXACT_PATHS and len(contents) < 128
                        and sum(len(text.encode()) for text in candidate.values()) <= 1024 * 1024
                        or self._fits(candidate)):
                    contents[label] = part.diff
                else:
                    statuses.append({"label": label, "status": "unavailable"})
        else:
            contents = self._render(parts)
        identity = hashlib.sha256(_json([state['stage'], state['assigned_target_ids'],
                                        state.get('attempt', 1)]).encode()).hexdigest()[:20]
        directory = Path(self.scope_name) / identity
        paths = {label: self._write(directory / ("diff.patch" if label == "diff" else
                                                "hunk-index.json" if label == "hunk-index" else f"{label}.txt"), text)
                 for label, text in contents.items()}
        legacy_paths = dict(paths)
        if self.bundle_capable and contents:
            bundled = self._bundled(contents)
            paths = {label: self._write(directory / f'{label}.txt', text) for label, text in bundled.items()}
            contents = bundled
            state['supporting_bundle'] = (
                {'path': str(paths['review-assignment'])} if self.transport is SanctionedInputTransport.EXACT_PATHS
                else {'label': 'review-assignment', 'transport': 'inline'})
        deferred_paths = {label: self._write(directory / f'{label}.patch', text)
                          for label, text in deferred_contents.items()}
        catalog_paths: dict[str, Path] = {}
        if deferred_paths:
            catalog_entries = [{'file': part.assignment['file'], 'target_id': part.assignment['target_id'],
                                'path': str(deferred_paths[f'diff-part-{index:06d}'])}
                               for index, part in enumerate(self.parts)]
            catalog, catalog_paths = self._supporting_catalog(directory, catalog_entries)
            state['supporting_catalog'] = {'path': str(catalog), 'part_count': len(catalog_entries),
                                           'status': 'complete', 'read_required': False}
        assigned_source_files = {part.assignment['file'] for part in parts}
        compact_structure = state['stage'] == 'integration' and self.bundle_capable
        source_parts = [*parts, *self.parts] if compact_structure else parts
        recipe = self._source_windows(source_parts, directory, statuses,
                                      required_windows=False if compact_structure else None)
        source_catalog_paths: dict[str, Path] = {}
        if (state['stage'] == 'integration' and getattr(self.backend, 'supports_source_recipe', False) is True
                and self.transport is SanctionedInputTransport.EXACT_PATHS):
            source_entries = [self._source_access(window) for window in recipe.windows]
            for entry in source_entries:
                entry['access'].pop('path', None)
            source_catalog, source_catalog_paths = self._supporting_catalog(
                directory, source_entries, namespace='source', entry_key='windows', count_key='window_count')
            state['source_catalog'] = {'path': str(source_catalog), 'window_count': len(source_entries),
                                       'status': 'complete', 'read_required': False}
        # The shared selector preserves required priority and uses the actual
        # transport allowance. Reuse existing identity-bound artifacts instead
        # of copying advisory bytes or following an unchecked symlink.
        selection = select_advisory_inputs(
            self.backend, self.work.repo,
            [AdvisoryCandidate(label, path) for label, path in paths.items()]
            + [AdvisoryCandidate(label, path) for label, path in shared_paths.items() if path is not None],
            read_only=self.read_only,
        )
        admitted = selection.selected_paths()
        if not paths.keys() <= admitted.keys():
            raise SanctionedInputUnavailable("required stage inputs exceed the transport allowance")
        aggregate = sum(len(text.encode()) for text in contents.values())
        for label, path in shared_paths.items():
            if path is None:
                continue
            if label not in admitted:
                statuses.append({"label": label, "status": "unavailable"})
                continue
            try:
                captured = _capture_input(label, path, self.transport, aggregate)
            except SanctionedInputUnavailable:
                statuses.append({"label": label, "status": "unavailable"})
                continue
            aggregate += captured.size
            paths[label] = path
        paths.update(deferred_paths)
        paths.update(catalog_paths)
        paths.update(source_catalog_paths)
        paths.update({f'source-{index:06d}': window.projection for index, window in enumerate(recipe.windows)
                      if window.projection is not None})
        prepared = prepare_sanctioned_inputs(self.backend, self.work.repo, paths, read_only=self.read_only,
                                             source_recipe=recipe)
        prepared = replace(prepared, inputs=tuple(replace(item, prompt_visible=False)
                                                  if (item.label in deferred_paths or item.label in catalog_paths
                                                      or item.label in source_catalog_paths)
                                                  else item
                                                  for item in prepared.inputs))
        if self.bundle_capable and self.transport is SanctionedInputTransport.EXACT_PATHS:
            # Whole small shared artifacts travel in the prompt within the same
            # logical supporting allowance. They need no repeated native read.
            entries = [(label, len(text.encode())) for label, text in contents.items()]
            inline_shared: dict[str, str] = {}
            for item in prepared.inputs:
                if item.label not in shared_paths:
                    continue
                inline_entries = [*entries, (item.label, item.size)]
                if inline_section_emitted_bytes(inline_entries) > SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES:
                    continue
                captured = _capture_input(item.label, item.path, self.transport, 0, text_budget=item.size)
                if captured.sha256 != item.sha256:
                    raise SanctionedInputUnavailable('shared supporting input changed during preparation')
                inline_shared[item.label] = captured.text or ''
                entries = inline_entries
            prepared = replace(prepared, inputs=tuple(
                replace(item, text=inline_shared[item.label], inline_advisory=True)
                if item.label in inline_shared else item for item in prepared.inputs))
            state['context_inline_labels'] = list(inline_shared)
        self.current_paths = {**legacy_paths, **paths}
        declared_status = {item['label'] for item in statuses}
        statuses.extend({"label": label, "status": "complete"} for label in paths if label not in declared_status)
        if compact_structure:
            state['context_availability'] = {
                'source_windows': len(recipe.windows), 'supporting_parts': len(deferred_paths),
                'supporting_catalogs': len(catalog_paths),
                'source_catalogs': len(source_catalog_paths),
            }
            statuses = [status for status in statuses if status['status'] != 'complete' or
                        not (status['label'].startswith('source-') or status['label'] in deferred_paths
                             or status['label'] in catalog_paths)]
        state.update(context_inputs=list(paths), context_transport=prepared.transport.value, context_statuses=statuses,
                     canonical_input_identities=self.binding['canonical_inputs'])
        if deferred_paths:
            state['context_inputs'] = [label for label in paths if label not in deferred_paths
                                      and label not in catalog_paths and label not in source_catalog_paths
                                      and not label.startswith('source-')]
        state['source_access'] = [self._source_access(window) for window in recipe.windows
                                  if window.file in assigned_source_files]
        state['available_source_files'] = {
            'files': sorted(set(recipe.allowed_files) - assigned_source_files),
            'side': 'after', 'revision': recipe.head_revision,
            'access': 'normal repository reads; frozen before windows use source_access only',
        }
        if state['stage'] == 'integration' and self.bundle_capable:
            state['source_access'] = [{key: value for key, value in entry.items()
                                       if key in {'target_ids', 'file', 'source_path', 'side', 'start_line', 'end_line',
                                                  'start_byte', 'end_byte', 'access'}} | {'read_required': False}
                                      for entry in state['source_access']]
            if getattr(self.backend, 'supports_source_recipe', False) is True:
                for entry in state['source_access']:
                    entry['access'].pop('path', None)
        units = sum(len(batch) for batch in self.assignment_batches)
        decided = set(state.get('completed_target_ids', []))
        remaining = [part for part in self.parts if part.assignment['target_id'] not in decided]
        state['remaining_work'] = {
            'assignment_units': 1 if state['stage'] == 'integration' else units - len(decided),
            'files': len({part.assignment['file'] for part in remaining}),
            'stages': 1 if state['stage'] == 'integration' else sum(
                any(part['target_id'] not in decided for part in batch) for batch in self.assignment_batches),
            'current_stage_source_windows': sum(window.read_required for window in recipe.windows),
            'current_stage_mandatory_read_estimate': (0 if self.transport is SanctionedInputTransport.INLINE else
                                         1 if self.bundle_capable else 3) + sum(
                                             window.read_required for window in recipe.windows),
            'transport_floor_only': True,
        }
        state['access_guide'] = self._access_guide(state, prepared)
        return prepared
