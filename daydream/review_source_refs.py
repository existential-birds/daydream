"""Bounded correspondence references to frozen review inputs, never read receipts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.git_ops.models import GitError, PathAbsentError
from daydream.git_ops.source import FrozenSourceRejected, frozen_source
from daydream.host_tools import HostToolFatal, HostToolRejected
from daydream.hunk_index import _HUNK_HEADER
from daydream.json_utils import validates_schema
from daydream.output_schema import strict_object
from daydream.redaction import redact_text
from daydream.repository_paths import git_observed_path_is_confined

if TYPE_CHECKING:
    from daydream.git_ops.captured_source import CapturedSource

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
_BASIS = {'enum': ['supplied_diff', 'source_inspection', 'executed_check']}
SOURCE_SELECTORS_SCHEMA = {'type': 'array', 'maxItems': 4, 'items': {'anyOf': [
    strict_object({
        'source_id': {'type': 'string', 'pattern': '^source:(?:[1-9][0-9]*|[0-9a-f]{64})$', 'maxLength': 71},
        'basis': _BASIS, 'source_range': {'anyOf': [_RANGE, {'type': 'null'}]},
    }),
    strict_object({
        'path': {'type': 'string', 'minLength': 1, 'maxLength': 4096},
        'side': {'enum': ['old', 'new']}, 'source_range': _RANGE,
        'basis': {'enum': ['source_inspection', 'executed_check']},
    }),
]}}


def _sha(body: str | bytes) -> str:
    return hashlib.sha256(body.encode('utf-8') if isinstance(body, str) else body).hexdigest()


def source_revision(revision: Mapping[str, Any], side: str) -> str:
    """Keep actual HEAD provenance separate from captured working-content identity."""
    identity: str = revision['merge_base_sha'] if side == 'old' else revision.get('source_id', revision['head_sha'])
    return identity


def _source_contents(root: Path, revision: Mapping[str, Any], side: str, path: str,
                     captured_source: CapturedSource | None) -> tuple[str, bytes]:
    if captured_source is not None:
        if captured_source.revision_for(side) != source_revision(revision, side):
            raise GitError('captured source differs from analyzed revision')
        return captured_source.read(side, path)
    if 'source_id' in revision:
        raise GitError('captured working source is unavailable; restart review')
    return frozen_source(root, source_revision(revision, side), path)


@dataclass(frozen=True)
class _SupportingFacts:
    digest: str
    size: int
    lines: int
    utf8: bool


def _supporting_facts(root: Path, revision: Mapping[str, Any], side: str, path: str,
                      captured: CapturedSource | None) -> _SupportingFacts:
    _, contents = _source_contents(root, revision, side, path, captured)
    try:
        contents.decode('utf-8')
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    return _SupportingFacts(_sha(contents), len(contents),
        contents.count(b'\n') + int(bool(contents) and not contents.endswith(b'\n')), utf8)


class _SubmissionSources:
    """Callback-local host facts, revoked before the factory validates and installs."""

    def __init__(self, revision: dict[str, Any], root: Path, aliases: Mapping[str, str],
                 captured: CapturedSource | None) -> None:
        self.revision, self.root, self.aliases, self.captured = revision, root, aliases, captured
        self._facts: dict[tuple[str, str], _SupportingFacts] = {}
        self._active = True

    def _supporting(self, side: str, path: str) -> _SupportingFacts:
        if not self._active:
            raise RuntimeError('submission source resolver has closed')
        key = side, path
        if key not in self._facts:
            self._facts[key] = _supporting_facts(self.root, self.revision, side, path, self.captured)
        return self._facts[key]

    def canonicalize(self, selectors: Any, expected: Iterable[dict[str, Any]], *,
                     require_assigned: bool = True) -> list[dict[str, Any]]:
        if not self._active:
            raise RuntimeError('submission source resolver has closed')
        return canonicalize_source_selectors(selectors, expected, revision=self.revision,
            source_root=self.root, require_assigned=require_assigned, source_aliases=self.aliases,
            captured_source=self.captured, _submission=self)

    def close(self) -> None:
        self._active = False
        self._facts.clear()


def source_id(reference: dict[str, Any]) -> str:
    """Name one exact host-assigned identity independently of the model's basis claim."""
    identity = {key: value for key, value in reference.items() if key not in {'basis', 'role'}}
    return 'source:' + _sha(json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(',', ':')))


def source_options(references: Iterable[dict[str, Any]], *,
                   source_aliases: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Describe every supplied source without asking a model to reproduce its mechanics."""
    aliases = source_aliases or {}
    options = []
    for reference in references:
        identifier = source_id(reference)
        options.append({'source_id': aliases.get(identifier, identifier),
                        **{key: deepcopy(reference[key]) for key in ('path', 'side', 'source_range')},
                        'basis': 'supplied_diff'})
    return options


def canonicalize_source_selectors(
    selectors: Any, expected_refs: Iterable[dict[str, Any]], *, revision: dict[str, Any],
    source_root: Path, require_assigned: bool = True,
    source_aliases: Mapping[str, str] | None = None,
    captured_source: CapturedSource | None = None,
    _submission: _SubmissionSources | None = None,
) -> list[dict[str, Any]]:
    """Resolve a complete selection atomically against frozen assignment/Git identities.

    A null assigned range retains the original range, including exact continuation
    fragments. Supporting selections cannot provide hashes or satisfy assigned work.
    No selected source is evidence that a model read or understood its contents.
    """
    if not validates_schema(selectors, SOURCE_SELECTORS_SCHEMA):
        raise HostToolRejected(
            'evidence_refs: supply at most four selectors with either source_id, basis, source_range '
            'or path, side, source_range, basis; remove hashes and other fields and use the supplied schema.')
    field = 'evidence_refs'
    try:
        if len(json.dumps(selectors, ensure_ascii=False).encode('utf-8')) > SOURCE_REFS_MAX_BYTES:
            raise HostToolRejected(
                f'evidence_refs: the whole selector array exceeds {SOURCE_REFS_MAX_BYTES} UTF-8 bytes; '
                'remove optional supporting selectors or choose shorter assigned source_id selectors.')
        requested = {selector['source_id'] for selector in selectors if 'source_id' in selector}
        assigned: dict[str, dict[str, Any]] = {}
        for expected in expected_refs:
            identifier = source_id(expected)
            alias = source_aliases.get(identifier) if source_aliases is not None else None
            for offered in (identifier, alias):
                if offered is not None and offered in requested:
                    assigned[offered] = expected
            if requested <= assigned.keys():
                break
        if requested != assigned.keys():
            unknown = next(selector['source_id'] for selector in selectors
                           if 'source_id' in selector and selector['source_id'] not in assigned)
            raise HostToolRejected(
                f'evidence_refs source_id {unknown}: outside this target\'s assigned inventory; '
                'choose a source_id from this target\'s source_options, including its assigned deferred parts.')
        resolved: list[dict[str, Any]] = []
        for index, selector in enumerate(selectors):
            field = f'evidence_refs[{index}]'
            if 'source_id' in selector:
                reference = deepcopy(assigned[selector['source_id']])
                bounds = selector['source_range']
                original = reference['source_range']
                if bounds is not None:
                    if (not isinstance(original, dict)
                            or not original['start'] <= bounds['start'] <= bounds['end'] <= original['end']):
                        allowed = (f'{original["start"]}..{original["end"]}' if isinstance(original, dict)
                                   else 'no line range (this is an exact fragment)')
                        raise HostToolRejected(
                            f'{field}.source_range for {selector["source_id"]}: '
                            f'allowed {allowed}; use null to retain the exact assigned identity '
                            'or choose a contained range with start <= end.')
                    reference['source_range'] = deepcopy(bounds)
                reference.update(basis=selector['basis'], role='assigned')
            else:
                path, side, bounds = selector['path'], selector['side'], selector['source_range']
                field += f' supporting path {redact_text(path)[:160]!r} ({side})'
                if not git_observed_path_is_confined(source_root, path):
                    raise HostToolRejected(
                        f'{field}: path must remain inside the repository; choose a confined repository path '
                        'or remove this optional supporting selector.')
                pinned_revision = source_revision(revision, side)
                try:
                    facts = (_submission._supporting(side, path) if _submission is not None else
                             _supporting_facts(source_root, revision, side, path, captured_source))
                except (FrozenSourceRejected, PathAbsentError) as exc:
                    raise HostToolRejected(
                        f'{field}: no available regular blob at the pinned revision; choose a regular file '
                        f'on this side within {_SUPPORTING_SOURCE_MAX_BYTES} whole-blob bytes, '
                        'use an assigned source_id, or remove this optional supporting selector.') from exc
                if facts.size > _SUPPORTING_SOURCE_MAX_BYTES:
                    raise HostToolRejected(
                        f'{field}: the whole pinned blob is {facts.size} bytes, above '
                        f'{_SUPPORTING_SOURCE_MAX_BYTES}; narrowing source_range cannot reduce this whole-blob '
                        'limit. Use an assigned source_id or remove this optional supporting selector.')
                if not facts.utf8:
                    raise HostToolRejected(
                        f'{field}: the whole pinned blob is not UTF-8 text; choose a UTF-8 regular file '
                        'or remove this optional supporting selector.')
                line_count = facts.lines
                if not 1 <= bounds['start'] <= bounds['end'] <= line_count:
                    raise HostToolRejected(
                        f'{field}.source_range: allowed 1..{line_count} with start <= end; '
                        'choose a contained range or remove this optional supporting selector.')
                reference = {
                    'artifact_sha256': facts.digest, 'part_sha256': None, 'revision': pinned_revision,
                    'path': path, 'side': side, 'source_range': deepcopy(bounds), 'fragment': None,
                    'basis': selector['basis'], 'role': 'supporting',
                }
            resolved.append(reference)
        if require_assigned and not any(reference['role'] == 'assigned' for reference in resolved):
            raise HostToolRejected(
                'evidence_refs: a reviewed target requires at least one assigned source_id from its '
                'source_options; supply that selector. Supporting paths alone cannot identify assigned work.')
        # Canonical evidence has its own retained bounds and contract, independent
        # of the smaller model-facing selector schema.
        if not validate_source_refs(resolved, assigned.values(), revision=revision, source_root=source_root,
                                    require_assigned=require_assigned, raise_unavailable=True,
                                    captured_source=captured_source, _submission=_submission):
            raise HostToolRejected(
                f'evidence_refs: canonical references must fit four items and {SOURCE_REFS_MAX_BYTES} UTF-8 '
                'bytes in total; remove optional supporting selectors or use fewer assigned selectors.')
    except (GitError, OSError) as exc:
        raise HostToolFatal(
            f'{field}: pinned supporting source storage is unavailable during source validation. '
            'The host cannot resolve its captured revision; this review invocation must stop.',
            code='input_unavailable') from exc
    except UnicodeError as exc:
        raise HostToolRejected(
            'evidence_refs: the selected supporting source cannot be resolved; choose an assigned source_id '
            'or remove the optional supporting selector.') from exc
    return resolved


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
    revisions = {side: source_revision(revision, side) for side in ('old', 'new')}
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
        refs.append({**identity, 'revision': source_revision(revision, 'diff'),
                     'path': assignment['file'], 'side': 'diff',
                     'source_range': None, 'fragment': {'offset': offset,
                     'bytes': len(metadata), 'sha256': _sha(metadata)}})
    return {'source_refs': refs, 'source_options': source_options(refs), 'source_summary': summaries,
            'fragment_offset_basis': 'UTF-8 bytes in the original diff hunk body; never a source line number'}


def validate_source_refs(refs: Any, expected_refs: Iterable[dict[str, Any]], *, revision: dict[str, Any],
                         source_root: Path | None = None, require_assigned: bool = True,
                         raise_unavailable: bool = False, captured_source: CapturedSource | None = None,
                         _submission: _SubmissionSources | None = None) -> bool:
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
    spans: list[tuple[int, int] | None] = [None] * len(refs)
    # Whole-change assignments may have many parts. Stream their existing host
    # identities while retaining only the at-most-four submitted selectors.
    for expected in expected_refs:
        for index, selector in enumerate(selectors):
            if any(value != expected[key] for key, value in selector.items() if key != 'source_range'):
                continue
            bounds, allowed = selector['source_range'], expected['source_range']
            if isinstance(bounds, dict) and isinstance(allowed, dict):
                # Hunks of one part share its identity, so a range may join them.
                start, end = spans[index] or (allowed['start'], allowed['end'])
                start, end = spans[index] = min(start, allowed['start']), max(end, allowed['end'])
                matching[index] |= start <= bounds['start'] <= bounds['end'] <= end
            else:
                matching[index] |= bounds is None and allowed is None
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
        expected_revision = source_revision(revision, ref['side'])
        if ref['revision'] != expected_revision:
            return False
        try:
            facts = (_submission._supporting(ref['side'], ref['path']) if _submission is not None else
                     _supporting_facts(source_root, revision, ref['side'], ref['path'], captured_source))
        except (GitError, OSError, UnicodeError, ValueError) as exc:
            if raise_unavailable:
                raise HostToolFatal(
                    f'evidence_refs[{index}] supporting path {redact_text(ref["path"])[:160]!r} '
                    f'({ref["side"]}): pinned supporting source storage became unavailable during '
                    'source validation; this review invocation must stop.', code='input_unavailable') from exc
            return False
        bounds = ref['source_range']
        line_count = facts.lines
        if (facts.size > _SUPPORTING_SOURCE_MAX_BYTES or ref['artifact_sha256'] != facts.digest
                or not 1 <= bounds['start'] <= bounds['end'] <= line_count):
            return False
    return assigned or not require_assigned
