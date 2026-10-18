"""Bounded correspondence references to frozen review inputs, never read receipts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from daydream.git_ops.models import GitError
from daydream.git_ops.source import frozen_source
from daydream.hunk_index import _HUNK_HEADER
from daydream.json_utils import validates_schema
from daydream.output_schema import strict_object
from daydream.repository_paths import git_observed_path_is_confined

SOURCE_REFS_MAX_BYTES = 8192
_SUPPORTING_SOURCE_MAX_BYTES = 64 * 1024
_DIGEST = {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
_RANGE = strict_object({key: {'type': 'integer', 'minimum': 1} for key in ('start', 'end')})
_FRAGMENT = strict_object({
    'offset': {'type': 'integer', 'minimum': 0}, 'bytes': {'type': 'integer', 'minimum': 0},
    'sha256': _DIGEST,
})
SOURCE_REF_SCHEMA = strict_object({
    'artifact_sha256': _DIGEST,
    'part_sha256': {'anyOf': [_DIGEST, {'type': 'null'}]},
    'revision': {'type': 'string', 'minLength': 1, 'maxLength': 128},
    # Assigned selectors originate in Git, whose filenames include characters
    # excluded from model-authored finding paths. Semantic validation below
    # rejects unsafe components; exact host-selector matching owns admission.
    'path': {'type': 'string', 'minLength': 1, 'maxLength': 4096},
    'side': {'enum': ['old', 'new', 'diff']},
    'source_range': {'anyOf': [_RANGE, {'type': 'null'}]},
    'fragment': {'anyOf': [_FRAGMENT, {'type': 'null'}]},
    'basis': {'enum': ['supplied_diff', 'source_inspection', 'executed_check']},
    'role': {'enum': ['assigned', 'supporting']},
})
SOURCE_REFS_SCHEMA = {'type': 'array', 'maxItems': 4, 'items': SOURCE_REF_SCHEMA}


def _sha(body: str | bytes) -> str:
    return hashlib.sha256(body.encode('utf-8') if isinstance(body, str) else body).hexdigest()


def lf_lines(text: str) -> list[str]:
    """Preserve exact Git lines: only LF terminates a diff/source line."""
    pieces = text.split('\n')
    return [piece + '\n' for piece in pieces[:-1]] + ([pieces[-1]] if pieces[-1] else [])


def _side_summary(body: str, start: int, end: int, old: int, new: int, *,
                  first_line_offset: int = 0) -> dict[str, Any]:
    """Map a byte interval by walking original lines, including split UTF-8 lines."""
    selected: dict[str, dict[str, Any] | None] = {'old': None, 'new': None}
    offset = 0
    for line in lf_lines(body):
        encoded = line.encode('utf-8')
        marker = line[:1]
        source_offset = first_line_offset if offset == 0 else 0
        left, right = max(start, offset + 1), min(end, offset + len(encoded))
        if marker in {' ', '-', '+'} and left < right:
            for side, line_number in (('old', old), ('new', new)):
                if marker == ('+' if side == 'old' else '-'):
                    continue
                prior = selected[side]
                selected[side] = {
                    'start': prior['start'] if prior else line_number, 'end': line_number,
                    'start_byte': prior['start_byte'] if prior else left - offset - 1 + source_offset,
                    'end_byte': right - offset - 1 + source_offset,
                    'partial_start': prior['partial_start'] if prior else bool(source_offset) or left > offset + 1,
                    'partial_end': right < offset + len(encoded),
                }
        if marker in {' ', '-'}:
            old += 1
        if marker in {' ', '+'}:
            new += 1
        offset += len(encoded)
    return selected


def part_source_mapping(diff: str, assignment: dict[str, Any], *, artifact_sha256: str,
                        revision: dict[str, Any], old_path: str, new_path: str,
                        body: str | bytes | None = None) -> dict[str, Any]:
    """Expose exact selectable references and side summaries without source contents."""
    identity = {'artifact_sha256': artifact_sha256, 'part_sha256': _sha(diff),
                'basis': 'supplied_diff', 'role': 'assigned'}
    paths = {'old': old_path, 'new': new_path}
    revisions = {'old': revision['merge_base_sha'], 'new': revision['head_sha']}
    refs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    if assignment['kind'] == 'continuation' and body is not None:
        source = body.encode('utf-8') if isinstance(body, str) else body
        offset, size = assignment['fragment_offset'], assignment['fragment_bytes']
        fragment = source[offset:offset + size]
        line_offset = assignment['fragment_line_offset']
        # Existing continuation cursors locate its first source line. Restore
        # only a split line's marker, never rescan the preceding/full hunk.
        selected = (source[offset - line_offset:offset - line_offset + 1] if line_offset else b'') + fragment
        end = len(selected)
        if offset + size < len(source) and not fragment.endswith(b'\n'):
            selected += b'x'  # The original line continues beyond this exact fragment.
        summary = _side_summary(selected.decode('utf-8'), 0, end,
            assignment['old_line'], assignment['new_line'], first_line_offset=max(0, line_offset - 1))
        summaries.append(summary)
        for side in ('old', 'new'):
            span = summary[side]
            if span is not None and paths[side] != '/dev/null':
                refs.append({**identity, 'revision': revisions[side], 'path': paths[side], 'side': side,
                             'source_range': None,
                             'fragment': {'offset': offset, 'bytes': size, 'sha256': _sha(fragment)}})
    else:
        lines = lf_lines(diff)
        starts = [index for index, line in enumerate(lines) if _HUNK_HEADER.match(line)]
        for ordinal, start in enumerate(starts):
            match = _HUNK_HEADER.match(lines[start])
            assert match is not None
            hunk_body = ''.join(lines[start + 1:starts[ordinal + 1] if ordinal + 1 < len(starts) else len(lines)])
            summary = _side_summary(hunk_body, 0, len(hunk_body.encode()), int(match[1]), int(match[3]))
            summaries.append(summary)
            for side in ('old', 'new'):
                span = summary[side]
                if span is not None and paths[side] != '/dev/null':
                    refs.append({**identity, 'revision': revisions[side], 'path': paths[side], 'side': side,
                                 'source_range': {'start': span['start'], 'end': span['end']}, 'fragment': None})
    if not refs:
        # Binary, pure rename and header-only assignments have genuine diff metadata.
        metadata = diff.encode('utf-8')
        offset = 0
        if assignment['kind'] == 'continuation' and body is not None:
            offset = assignment['fragment_offset']
            source = body.encode('utf-8') if isinstance(body, str) else body
            metadata = source[offset:offset + assignment['fragment_bytes']]
        refs.append({**identity, 'revision': revision['head_sha'], 'path': assignment['file'], 'side': 'diff',
                     'source_range': None, 'fragment': {'offset': offset,
                     'bytes': len(metadata), 'sha256': _sha(metadata)}})
    return {'source_refs': refs, 'source_summary': summaries,
            'fragment_offset_basis': 'UTF-8 bytes in the original diff hunk body; never a source line number'}


def validate_source_refs(refs: Any, expected_refs: Iterable[dict[str, Any]], *, revision: dict[str, Any],
                         source_root: Path | None = None, require_assigned: bool = True) -> bool:
    """Validate all references before admission; labels assert basis, not execution."""
    if not validates_schema(refs, SOURCE_REFS_SCHEMA):
        return False
    try:
        if len(json.dumps(refs, ensure_ascii=False).encode('utf-8')) > SOURCE_REFS_MAX_BYTES:
            return False
    except UnicodeEncodeError:
        return False
    assigned = False
    selectors = [{key: value for key, value in ref.items() if key not in {'basis', 'role'}} for ref in refs]
    matching = [False] * len(refs)
    # Whole-change assignments may have many parts. Stream their existing host
    # identities while retaining only the at-most-four submitted selectors.
    for expected in expected_refs:
        for index, selector in enumerate(selectors):
            bounds = selector['source_range']
            same_identity = all(value == expected[key] for key, value in selector.items() if key != 'source_range')
            matching[index] |= same_identity and (
                isinstance(bounds, dict) and isinstance(expected['source_range'], dict)
                and expected['source_range']['start'] <= bounds['start'] <= bounds['end']
                <= expected['source_range']['end']
                or bounds is None and expected['source_range'] is None)
        if all(matching):
            break
    for index, ref in enumerate(refs):
        path = ref['path']
        if (path.startswith('/') or '\0' in path or len(path.encode('utf-8')) > 4096
                or any(part in {'', '.', '..'} for part in path.split('/'))):
            return False
        if matching[index]:
            assigned |= ref['role'] == 'assigned'
            continue
        if (ref['role'] != 'supporting' or ref['basis'] == 'supplied_diff' or source_root is None
                or ref['part_sha256'] is not None or ref['fragment'] is not None
                or ref['side'] not in {'old', 'new'} or not isinstance(ref['source_range'], dict)
                or not git_observed_path_is_confined(source_root, ref['path'])):
            return False
        expected_revision = revision['merge_base_sha' if ref['side'] == 'old' else 'head_sha']
        if ref['revision'] != expected_revision:
            return False
        try:
            _, contents = frozen_source(source_root, expected_revision, ref['path'])
        except (GitError, OSError, UnicodeError, ValueError):
            return False
        bounds = ref['source_range']
        line_count = contents.count(b'\n') + int(bool(contents) and not contents.endswith(b'\n'))
        if (len(contents) > _SUPPORTING_SOURCE_MAX_BYTES or ref['artifact_sha256'] != _sha(contents)
                or not 1 <= bounds['start'] <= bounds['end'] <= line_count):
            return False
    return assigned or not require_assigned
