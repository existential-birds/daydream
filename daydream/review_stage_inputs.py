"""Frozen canonical changes projected into bounded, complete stage assignments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from daydream import prompt_budget as inputs
from daydream.backends import Backend
from daydream.deep.detection import StackAssignment
from daydream.deep.diff import _DIFF_MINUS_HEADER, _DIFF_PLUS_HEADER, iter_diff_blocks
from daydream.git_ops.models import GitError
from daydream.hunk_index import _HUNK_HEADER, _unquote_git_path
from daydream.json_utils import canonical_json as _json
from daydream.review_source_refs import (
    _SubmissionSources,
    lf_lines,
    part_source_mapping,
    source_id,
    source_options,
)
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.git_ops.captured_source import CapturedSource

_EXACT_PATH_ASSIGNMENT_MAX_BYTES = 24 * 1024


@dataclass
class _Part:
    assignment: dict[str, Any]
    diff: str


def _bounded_groups[T](values: list[T], fits: Callable[[list[T]], bool], error: str) -> list[list[T]]:
    """Pack whole values in order, rejecting any value that cannot fit alone."""
    groups: list[list[T]] = []
    current: list[T] = []
    for value in values:
        if not fits([value]):
            raise inputs.SanctionedInputUnavailable(error)
        if current and not fits([*current, value]):
            groups.append(current)
            current = []
        current.append(value)
    if current:
        groups.append(current)
    return groups


def _capture_canonical(label: str, path: Path) -> inputs.PreparedSanctionedInput:
    return inputs._capture_input(label, path, inputs.SanctionedInputTransport.EXACT_PATHS, 0, pointer_only=True)


def _revalidate(canonical: tuple[inputs.PreparedSanctionedInput, ...], captured_source: CapturedSource | None) -> None:
    if captured_source is not None:
        try:
            captured_source.validate()
        except (GitError, OSError) as exc:
            raise inputs.SanctionedInputUnavailable(f'captured source changed or is unavailable: {exc}') from exc
    for expected in canonical:
        if _capture_canonical(expected.label, expected.path) != expected:
            raise inputs.SanctionedInputUnavailable(f"sanctioned input {expected.label!r} changed after capture")


def _block_paths(file: str, first: str, last: str) -> tuple[str, str]:
    """Resolve the old side from a path's first block and the new side from its last block."""
    paths = []
    for pattern, prefix, block in ((_DIFF_MINUS_HEADER, "a/", first), (_DIFF_PLUS_HEADER, "b/", last)):
        header = pattern.search(block)
        paths.append(_unquote_git_path(header[1].rstrip("\t")).removeprefix(prefix) if header else file)
    rename_from = next((line.removeprefix("rename from ") for line in first.split('\n')
                        if line.startswith("rename from ")), None)
    return (_unquote_git_path(rename_from) if rename_from is not None else paths[0]), paths[1]


@dataclass(frozen=True)
class CanonicalChange:
    """One validated canonical patch and hunk index, shared read-only by every scope of a review phase.

    Fields are immutable; index entries stay canonical JSON text so each scope
    parses private copies of only the files it is assigned.
    """

    canonical: tuple[inputs.PreparedSanctionedInput, ...]
    text: str
    diff_paths: Mapping[str, tuple[str, str]]
    index_entries: Mapping[str, str]

    @classmethod
    def capture(cls, diff_path: Path, hunk_index_path: Path,
                captured_source: CapturedSource | None = None) -> CanonicalChange:
        """Capture identities with bounded no-follow streaming, then bind the bytes read to them once."""
        canonical = tuple(_capture_canonical(label, path) for label, path in (
            ("canonical-diff", diff_path), ("canonical-index", hunk_index_path)))
        text = captured_source.diff if captured_source is not None else None
        if text is None or hashlib.sha256(text.encode()).hexdigest() != canonical[0].sha256:
            text = diff_path.read_text(encoding="utf-8")
        index_text = hunk_index_path.read_text(encoding="utf-8")
        index = json.loads(index_text)
        # Each factory also validates the captured source before using this change.
        _revalidate(canonical, None)
        if hashlib.sha256(text.encode()).hexdigest() != canonical[0].sha256:
            raise inputs.SanctionedInputUnavailable("canonical change bytes changed during preparation")
        if hashlib.sha256(index_text.encode()).hexdigest() != canonical[1].sha256:
            raise inputs.SanctionedInputUnavailable("canonical index bytes changed during preparation")
        if not isinstance(index, dict) or not all(isinstance(entry, dict) for entry in index.values()):
            raise inputs.SanctionedInputUnavailable("canonical hunk index is malformed")
        firsts: dict[str, str] = {}
        lasts: dict[str, str] = {}
        for path, block in iter_diff_blocks(text):
            firsts.setdefault(path, block)
            lasts[path] = block
        diff_paths = {path: _block_paths(path, firsts[path], lasts[path]) for path in firsts}
        return cls(
            canonical=canonical, text=text, diff_paths=MappingProxyType(diff_paths),
            index_entries=MappingProxyType({path: _json(entry) for path, entry in index.items()}))

    def paths(self, file: str) -> tuple[str, str]:
        return self.diff_paths.get(file, (file, file))

    def index_slice(self, files: Iterable[str]) -> dict[str, Any]:
        """Parse private copies of the requested index entries."""
        return {path: json.loads(self.index_entries[path]) for path in files if path in self.index_entries}


class StageInputFactory:
    """Project scoped first-pass assignments with required-work priority; see the assignment limits in this module."""

    def __init__(self, backend: Backend, work: WorkContext, stack: StackAssignment, *,
        change: CanonicalChange, revision: dict[str, Any], read_only: bool) -> None:
        self.backend, self.work, self.stack = backend, work, stack
        self.captured_source = work.captured_source
        self.revision = revision
        self.transport = inputs.sanctioned_transport_for(backend, work.repo, read_only=read_only)
        self.change = change
        # Identities stay host-only on both transports.
        self.canonical = change.canonical
        self._revalidate_canonical()
        self.index = change.index_slice(stack.files)
        self.binding = {"analyzed_revision": revision,
                        "canonical_inputs": {item.label: item.sha256 for item in self.canonical}}
        self.parts = self._parts()
        # Scope-local handles stay fixed when later projections select only some parts.
        self._source_aliases: dict[str, str] = {}
        for part in self.parts:
            for reference in part.assignment['source_refs']:
                identifier = source_id(reference)
                self._source_aliases.setdefault(identifier, f'source:{len(self._source_aliases) + 1}')
        for part in self.parts:
            part.assignment['source_options'] = source_options(
                part.assignment['source_refs'], source_aliases=self._source_aliases)
        self.by_id = {part.assignment["target_id"]: part for part in self.parts}
        self.assignment_batches = self._batches()


    def _mapped(self, part: _Part, *, body: str | bytes | None = None) -> _Part:
        old_path, new_path = self.change.paths(part.assignment['file'])
        part.assignment.update(part_source_mapping(part.diff, part.assignment,
            artifact_sha256=self.canonical[0].sha256, revision=self.revision,
            old_path=old_path, new_path=new_path, body=body))
        return part

    def _revalidate_canonical(self) -> None:
        _revalidate(self.canonical, self.captured_source)


    def admit_submission[T](self, prepare: Callable[[_SubmissionSources], Callable[[], T]], *,
                            validated: bool = False) -> T:
        """Resolve a prospective payload, then synchronously install it after one complete input check.

        The resolver retains bounded host facts, never bodies or success across
        callbacks. The Review owner's closure rechecks current authority after
        all I/O and owns the only domain state transition. The check runs here,
        after resolving, unless ``validated`` says the caller's endpoint ran it
        for this submission just before this call.
        """
        sources = _SubmissionSources(self.revision, self.work.repo, self._source_aliases, self.captured_source)
        try:
            install = prepare(sources)
        finally:
            sources.close()
        if not validated:
            self._revalidate_canonical()
        return install()

    def _render(self, parts: list[_Part]) -> dict[str, str]:
        files = {part.assignment["file"] for part in parts}
        # Part ranges own the assignment; the rest of a large file's index is not required first-pass work.
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
                    matching = next((hunk for hunk in hunks if tuple(hunk[key] for key in keys) == bounds), None)
                    if matching is not None:
                        selected.append(matching)
                    elif part['new_count'] == 0:
                        # Canonical posting omits empty new-side ranges; assignments retain the frozen diff's
                        # exact old-side deletion authority.
                        selected.append({**dict(zip(keys, bounds, strict=True)),
                                         'added': 0, 'removed': part['old_count'], 'old_only': True})
                    else:
                        raise inputs.SanctionedInputUnavailable("canonical hunk ranges do not match assigned work")
            index[path] = {"hunks": selected, "assignments": assignments,
                           "status": "partial" if any(part['part_count'] > 1 for part in assignments) else "complete"}
        return {"diff": "".join(part.diff for part in parts), "hunk-index": _json(index),
                "input-binding": _json(self.binding)}

    def _remaining_bytes(self, contents: dict[str, str], *, dispatch: bool = False) -> int:
        cap = (_EXACT_PATH_ASSIGNMENT_MAX_BYTES if self.transport is inputs.SanctionedInputTransport.EXACT_PATHS
               else inputs.STAGED_REVIEW_INLINE_MAX_BYTES if dispatch
               else inputs.SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES)
        return cap - inputs.inline_section_emitted_bytes(
            [(label, len(text.encode())) for label, text in contents.items()])

    def _parts(self) -> list[_Part]:
        result: list[_Part] = []
        reserved_targets = set(self.stack.files)
        next_part = 0
        # Keep only this scope's blocks; a path may have several (a type change is a deletion then an addition).
        blocks: dict[str, list[str]] = {path: [] for path in self.stack.files}
        for path, block in iter_diff_blocks(self.change.text):
            if path in blocks:
                blocks[path].append(block)
        for path in sorted(self.stack.files):
            whole = self._mapped(_Part({"target_id": path, "file": path, "part_index": 1, "part_count": 1,
                           "kind": "file", "hunk_index": None}, "".join(blocks[path])))
            if self._remaining_bytes(self._render([whole])) >= 0:
                result.append(whole)
                continue
            pieces = self._split_file(path, blocks[path])
            for index, part in enumerate(pieces, 1):
                while True:
                    next_part += 1
                    target_id = f"part:{next_part:06d}"
                    if target_id not in reserved_targets:
                        break
                reserved_targets.add(target_id)
                part.assignment.update(target_id=target_id, part_index=index, part_count=len(pieces))
            result.extend(pieces)
        return result

    def _split_file(self, path: str, blocks: list[str]) -> list[_Part]:
        """Split each block under its own header; hunk ordinals continue across a path's blocks."""
        parts: list[_Part] = []
        ordinal = 0
        for block in blocks:
            lines = lf_lines(block)
            starts = [index for index, line in enumerate(lines) if _HUNK_HEADER.match(line)]
            # Binary/rename metadata is also required work, split explicitly if huge.
            if not starts:
                parts.extend(self._segments(path, "", block, ordinal, (0, 0, 0, 0)))
                continue
            prefix = "".join(lines[:starts[0]])
            for index, start in enumerate(starts):
                end = starts[index + 1] if index + 1 < len(starts) else len(lines)
                match = _HUNK_HEADER.match(lines[start])
                assert match is not None
                bounds = (int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1))
                body = "".join(lines[start + 1:end])
                parts.extend(self._segments(path, prefix + lines[start], body, ordinal, bounds))
                ordinal += 1
        return parts

    def _segments(self, path: str, header: str, body: str, hunk_index: int, bounds: tuple[int, int, int, int],
    ) -> list[_Part]:
        old_start, old_count, new_start, new_count = bounds
        base = {"target_id": "part:000000", "file": path, "part_index": 999999, "part_count": 999999,
                "kind": "hunk", "hunk_index": hunk_index,
                "old_start": old_start, "old_count": old_count, "new_start": new_start, "new_count": new_count}
        whole = self._mapped(_Part(base, header + body))
        if self._remaining_bytes(self._render([whole])) >= 0:
            return [whole]
        # Prefer whole diff lines; oversized lines use ordered UTF-8 fragments at exact byte offsets, never whole hunks.
        chunks: list[_Part] = []
        source_body = body.encode('utf-8')
        rest = body
        byte_offset = 0
        old_line, new_line = old_start, new_start
        line_offset = 0
        line_kind = body[:1]
        notice = "[Ordered continuation: this is a partial hunk; see assignment mapping.]\n"
        while rest:
            meta = {**base, "kind": "continuation", "segment_index": 999999, "segment_count": 999999,
                    "fragment_offset": byte_offset, "fragment_bytes": 999999999,
                    "old_line": old_line, "new_line": new_line, "fragment_line_offset": line_offset}
            # Reserve the descriptor/summary wrappers before choosing bytes. This
            # avoids adding unbudgeted attribution after splitting a full input.
            probe_bytes = len(inputs.truncate_utf8_to_budget(rest[:256], 256).encode('utf-8'))
            probe = self._mapped(_Part({**meta, 'fragment_bytes': probe_bytes}, notice + header), body=source_body)
            allowance = self._remaining_bytes(self._render([probe])) - 1536
            # A budget of N bytes never needs more than N characters: bound the
            # encoded prefix so multi-MB hunks are not re-encoded per fragment.
            low = len(inputs.truncate_utf8_to_budget(rest[:max(0, allowance)], max(0, allowance)))
            if low == 0:
                raise inputs.SanctionedInputUnavailable("required change header exceeds the assignment allowance")
            boundary = rest.rfind("\n", 0, low)
            take = boundary + 1 if boundary >= 0 else low
            chunk = rest[:take]
            part = self._mapped(_Part({**meta, "segment_index": len(chunks) + 1,
                                      "fragment_bytes": len(chunk.encode())}, notice + header + chunk),
                                body=source_body)
            if self._remaining_bytes(self._render([part])) < 0:
                raise inputs.SanctionedInputUnavailable('required source attribution exceeds the assignment allowance')
            chunks.append(part)
            # Count completed lines only; fragment offsets map split lines exactly, while ranges retain the full hunk.
            for line in lf_lines(chunk):
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
        for part in chunks:
            part.assignment['segment_count'] = len(chunks)
        return chunks

    def _batches(self) -> list[list[dict[str, Any]]]:
        batches = _bounded_groups(self.parts,
            lambda parts: len({p.assignment['file'] for p in parts}) <= 4
            and self._remaining_bytes(self._render(parts), dispatch=True) >= 0,
            "required assignment exceeds the assignment allowance")
        return [[part.assignment for part in batch] for batch in batches]
