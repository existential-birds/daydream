"""Frozen canonical changes projected into bounded, complete stage assignments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream import prompt_budget as inputs
from daydream.artifact_visibility import ArtifactSession, artifact_dir_for
from daydream.backends import Backend
from daydream.backends.osprey import OspreyBackend
from daydream.backends.pi import PiBackend
from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import GENERIC_STACK, StackAssignment, base_stack_name
from daydream.deep.diff import _DIFF_MINUS_HEADER, _DIFF_PLUS_HEADER, iter_diff_blocks
from daydream.git_ops.models import GitError
from daydream.git_ops.source import frozen_source
from daydream.hunk_index import _HUNK_HEADER, _unquote_git_path
from daydream.json_utils import atomic_write_bytes, canonical_json as _json
from daydream.review_source_refs import (
    _SubmissionSources,
    canonicalize_source_selectors,
    lf_lines,
    part_source_mapping,
    source_id,
    source_options,
    validate_source_refs,
)
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.extensions.registry import Registry

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


class StageInputFactory:
    """Project scoped first-pass assignments with required-work priority; see the assignment limits in this module."""

    def __init__(self, backend: Backend, work: WorkContext, stack: StackAssignment, *,
        diff_path: Path, hunk_index_path: Path, shared_paths: dict[str, Path | None],
        revision: dict[str, Any], artifact_session: ArtifactSession | None, allow_standalone: bool, read_only: bool,
        bundle_capable: bool = False) -> None:
        self.backend, self.work, self.stack = backend, work, stack
        self.captured_source = work.captured_source
        self.revision = revision
        self.session, self.allow_standalone, self.read_only = artifact_session, allow_standalone, read_only
        self.shared_paths = shared_paths
        self.bundle_capable = bundle_capable
        self.transport = inputs.sanctioned_transport_for(backend, work.repo, read_only=read_only)
        # Capture identities with bounded no-follow streaming; keep them host-only on both transports.
        self.canonical = tuple(self._canonical(label, path) for label, path in (
            ("canonical-diff", diff_path), ("canonical-index", hunk_index_path)))
        self.full_diff = diff_path.read_text(encoding="utf-8")
        index_text = hunk_index_path.read_text(encoding="utf-8")
        self.index = json.loads(index_text)
        self._revalidate_canonical()
        if hashlib.sha256(self.full_diff.encode()).hexdigest() != self.canonical[0].sha256:
            raise inputs.SanctionedInputUnavailable("canonical change bytes changed during preparation")
        if hashlib.sha256(index_text.encode()).hexdigest() != self.canonical[1].sha256:
            raise inputs.SanctionedInputUnavailable("canonical index bytes changed during preparation")
        if not isinstance(self.index, dict):
            raise inputs.SanctionedInputUnavailable("canonical hunk index is malformed")
        self.binding = {"analyzed_revision": revision,
                        "canonical_inputs": {item.label: item.sha256 for item in self.canonical}}
        self.scope_name = hashlib.sha256(stack.stack_name.encode()).hexdigest()[:20]
        self.blocks = dict(iter_diff_blocks(self.full_diff))
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
        self.current_paths: dict[str, Path] = {}

    @classmethod
    def for_review(cls, backend: Backend, work: WorkContext, stack: StackAssignment, *,
        diff_path: Path, shared_paths: dict[str, Path | None], revision: dict[str, Any],
        registry: Registry, artifact_session: ArtifactSession | None, allow_standalone: bool,
    ) -> StageInputFactory:
        """Use identical transport and prompt capabilities for live work and restored assertions."""
        from daydream.deep.prompts import (
            build_generic_fallback_prompt,
            build_per_stack_prompt,
            build_structural_prompt,
        )

        prompt_name = {STRUCTURE_STACK_NAME: 'structural', GENERIC_STACK: 'generic-fallback'}.get(
            base_stack_name(stack.stack_name), 'per-stack')
        builder = registry.prompt(prompt_name)
        bundle_capable = (any(builder is builtin for builtin in (
            build_generic_fallback_prompt, build_per_stack_prompt, build_structural_prompt))
            or getattr(builder, 'review_input_bundle', False) is True)
        read_only = (getattr(backend, 'read_only_disposable_clone', False) is True
                     or inputs.uses_diff_reference(backend, work.repo, read_only=True))
        return cls(backend, work, stack, diff_path=diff_path, hunk_index_path=diff_path.parent / 'hunk-index.json',
            shared_paths=shared_paths, revision=revision, artifact_session=artifact_session,
            allow_standalone=allow_standalone, read_only=read_only, bundle_capable=bundle_capable)

    def validate_retained_decisions(self, decisions: Any, *, complete: bool) -> None:
        """Rebind restored assertions to every original target and its frozen source selectors."""
        from daydream.review_result import validate_target_decisions

        validate_target_decisions(decisions, analyzed_revision=self.revision)
        structural = self.stack.stack_name == STRUCTURE_STACK_NAME
        expected = {f'integration:{self.stack.stack_name}'} if structural else set(self.by_id)
        actual = {decision['target_id'] for decision in decisions}
        if not actual <= expected or (complete and (actual != expected or any(
                decision['status'] != 'reviewed' for decision in decisions))):
            raise ValueError('retained target decisions differ from frozen assignment inventory')
        for decision in decisions:
            refs = (ref for part in self.parts for ref in part.assignment['source_refs']) if structural else (
                self.by_id[decision['target_id']].assignment['source_refs'])
            if not validate_source_refs(decision['evidence_refs'], refs, revision=self.revision,
                    source_root=self.work.repo, require_assigned=decision['status'] == 'reviewed',
                    captured_source=self.captured_source):
                raise ValueError('retained source attribution differs from frozen assignment')

    def _mapped(self, part: _Part, *, body: str | bytes | None = None) -> _Part:
        old_path, new_path = self._diff_paths(part.assignment['file'])
        part.assignment.update(part_source_mapping(part.diff, part.assignment,
            artifact_sha256=self.canonical[0].sha256, revision=self.revision,
            old_path=old_path, new_path=new_path, body=body))
        return part

    @staticmethod
    def _canonical(label: str, path: Path) -> inputs.PreparedSanctionedInput:
        return inputs._capture_input(label, path, inputs.SanctionedInputTransport.EXACT_PATHS, 0, pointer_only=True)

    def _revalidate_canonical(self) -> None:
        if self.captured_source is not None:
            try:
                self.captured_source.validate()
            except (GitError, OSError) as exc:
                raise inputs.SanctionedInputUnavailable(f'captured source changed or is unavailable: {exc}') from exc
        for expected in self.canonical:
            if self._canonical(expected.label, expected.path) != expected:
                raise inputs.SanctionedInputUnavailable(f"sanctioned input {expected.label!r} changed after capture")

    def canonicalize_source_selectors(self, selectors: Any, expected_refs: Iterable[dict[str, Any]], *,
                                      require_assigned: bool = True) -> list[dict[str, Any]]:
        """Recheck the original canonical inputs immediately before resolving selectors."""
        self._revalidate_canonical()
        return canonicalize_source_selectors(selectors, expected_refs, revision=self.revision,
            source_root=self.work.repo, require_assigned=require_assigned, source_aliases=self._source_aliases,
            captured_source=self.captured_source)

    def admit_submission[T](self, prepare: Callable[[_SubmissionSources], Callable[[], T]]) -> T:
        """Resolve a prospective payload, validate once, then synchronously install it.

        The resolver retains bounded host facts, never bodies or success across
        callbacks. The Review owner's closure rechecks current authority after
        all I/O and owns the only domain state transition.
        """
        sources = _SubmissionSources(self.revision, self.work.repo, self._source_aliases, self.captured_source)
        try:
            install = prepare(sources)
        finally:
            sources.close()
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

    def _bundled(self, contents: dict[str, str]) -> dict[str, str]:
        if not self.bundle_capable or not contents:
            return contents
        return {'review-assignment': '\n\n'.join(f'### {label} (supporting)\n{text}'
                                                 for label, text in contents.items())}

    def _remaining_bytes(self, contents: dict[str, str], *, dispatch: bool = False) -> int:
        contents = self._bundled(contents)
        cap = (_EXACT_PATH_ASSIGNMENT_MAX_BYTES if self.transport is inputs.SanctionedInputTransport.EXACT_PATHS
               else inputs.STAGED_REVIEW_INLINE_MAX_BYTES if dispatch
               else inputs.SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES)
        return cap - inputs.inline_section_emitted_bytes(
            [(label, len(text.encode())) for label, text in contents.items()])

    def _parts(self) -> list[_Part]:
        result: list[_Part] = []
        reserved_targets = set(self.stack.files)
        next_part = 0
        blocks: dict[str, str] = {}
        for path, block in iter_diff_blocks(self.full_diff):
            blocks[path] = blocks.get(path, "") + block
        for path in sorted(self.stack.files):
            whole = self._mapped(_Part({"target_id": path, "file": path, "part_index": 1, "part_count": 1,
                           "kind": "file", "hunk_index": None}, blocks.get(path, "")))
            if self._remaining_bytes(self._render([whole])) >= 0:
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
                part.assignment.update(target_id=target_id, part_index=index, part_count=len(pieces))
            result.extend(pieces)
        return result

    def _split_file(self, path: str, block: str) -> list[_Part]:
        lines = lf_lines(block)
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

    def _write(self, relative: Path, text: str) -> Path:
        if self.session is not None:
            return self.session.write_review_input(self.work.repo, relative, text)
        root = artifact_dir_for(self.work.repo, allow_standalone=self.allow_standalone) / "deep" / "stage-inputs"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, text.encode(), fsync=True, dir_fsync=True, mode=0o600)
        return path

    def _diff_paths(self, file: str) -> tuple[str, str]:
        block = self.blocks.get(file, "")
        paths = []
        for pattern, prefix in ((_DIFF_MINUS_HEADER, "a/"), (_DIFF_PLUS_HEADER, "b/")):
            header = pattern.search(block)
            paths.append(_unquote_git_path(header[1].rstrip("\t")).removeprefix(prefix) if header else file)
        rename_from = next((line.removeprefix("rename from ") for line in block.split('\n')
                            if line.startswith("rename from ")), None)
        return (_unquote_git_path(rename_from) if rename_from is not None else paths[0]), paths[1]

    def _before_context(self, files: set[str], directory: Path, statuses: list[dict[str, str]]) -> dict[str, Path]:
        """Offer bounded ordinary file context for old paths file-only tools cannot reach."""
        if not isinstance(self.backend, (PiBackend, OspreyBackend)):
            return {}
        candidates: dict[str, Path] = {}
        for index, file in enumerate(sorted(files)):
            old_path, new_path = self._diff_paths(file)
            if old_path == "/dev/null" or not (new_path == "/dev/null" or old_path != new_path):
                continue
            label = f"before-context-{index:04d}"
            try:
                _, raw = (self.captured_source.read('old', old_path) if self.captured_source is not None else
                          frozen_source(self.work.repo, self.revision["merge_base_sha"], old_path))
                text = raw.decode("utf-8")
                if len(raw) > 64 * 1024:
                    raise UnicodeError("optional context exceeds its bound")
            except (GitError, UnicodeError):
                statuses.append({"label": label, "status": "omitted", "file": file})
                continue
            captured_context = (
                f"# Captured ordinary context\n# original_path: {old_path}\n# side: before\n"
                f"# revision: {self.revision['merge_base_sha']}\n\n{text}"
            )
            candidates[label] = self._write(directory / f"{label}.txt", captured_context)
        return candidates

    def _supporting_catalog(self, directory: Path, entries: list[dict[str, Any]]) -> tuple[Path, dict[str, Path]]:
        """Publish a bounded exact-pointer catalog without exposing a directory grant."""
        captured: dict[str, Path] = {}
        ordinal = 0

        def write_groups(key: str, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
            nonlocal ordinal
            groups = _bounded_groups(values, lambda group: len(_json({key: group}).encode()) <= 12_000,
                                     'supporting catalog entry exceeds its bounded allowance') or [[]]
            result: list[dict[str, Any]] = []
            for group in groups:
                label = f'supporting-catalog-{ordinal:06d}'
                ordinal += 1
                path = self._write(directory / f'{label}.json', _json({key: group}))
                captured[label] = path
                result.append({'path': str(path), 'part_count': (
                    len(group) if key == 'parts' else sum(child['part_count'] for child in group))})
            return result

        level = write_groups('parts', entries)
        while len(level) > 1:
            following = write_groups('catalogs', level)
            if len(following) >= len(level):
                raise inputs.SanctionedInputUnavailable('supporting catalog cannot fit its bounded index')
            level = following
        return Path(level[0]['path']), captured

    def prepare(self, state: dict[str, Any]) -> inputs.PreparedSanctionedInputs:
        self._revalidate_canonical()
        shared_paths = {} if state['stage'] == 'triage' else self.shared_paths
        statuses: list[dict[str, str]] = []
        deferred_contents: dict[str, str] = {}
        if state['stage'] == 'triage':
            contents = {}
        elif state['stage'] == 'integration':
            # Structure is one interaction assignment: inventory names all changed files; bounded diff parts support it.
            inventory = {path: {"hunks": len(self.index.get(path, {}).get('hunks', [])),
                                "parts": sum(p.assignment['file'] == path for p in self.parts)}
                         for path in self.stack.files}
            contents = {"diff": "Whole-change interaction assignment. Use the inventory and targeted source reads.\n",
                        "hunk-index": _json(inventory), "input-binding": _json(self.binding)}
            if self._remaining_bytes(contents, dispatch=True) < 0:
                contents['hunk-index'] = _json({"status": "partial", "file_count": len(inventory),
                                               "canonical_sha256": self.canonical[1].sha256})
                statuses.append({"label": "hunk-index", "status": "partial"})
            # Expose bounded whole parts under exact confinement; INLINE selects those within its aggregate allowance.
            for index, part in enumerate(self.parts):
                label = f"diff-part-{index:06d}"
                if self.bundle_capable and self.transport is inputs.SanctionedInputTransport.EXACT_PATHS:
                    deferred_contents[label] = part.diff
                    continue
                descriptor = _json({'target_id': part.assignment['target_id'],
                                    'source_refs': part.assignment['source_refs'],
                                    'source_options': part.assignment['source_options']})
                candidate = {**contents, label: part.diff, f'{label}-sources': descriptor}
                if (self.transport is inputs.SanctionedInputTransport.EXACT_PATHS and len(candidate) <= 128
                        and sum(len(text.encode()) for text in candidate.values()) <= 1024 * 1024
                        or self._remaining_bytes(candidate, dispatch=True) >= 0):
                    contents = candidate
                else:
                    statuses.append({"label": label, "status": "unavailable"})
        else:
            contents = self._render([self.by_id[target] for target in state['assigned_target_ids']
                                     if target in self.by_id])
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
                {'path': str(paths['review-assignment'])}
                if self.transport is inputs.SanctionedInputTransport.EXACT_PATHS
                else {'label': 'review-assignment', 'transport': 'inline'})
        deferred_paths = {label: self._write(directory / f'{label}.patch', text)
                          for label, text in deferred_contents.items()}
        catalog_paths: dict[str, Path] = {}
        compact_structure = state['stage'] == 'integration' and self.bundle_capable
        # Use the transport allowance with required inputs first. Reuse identity-bound artifacts to avoid
        # copying advisory bytes or following unchecked symlinks.
        before_files = set(state["assigned_files"])
        for file in self.blocks:
            old_path, new_path = self._diff_paths(file)
            if new_path == "/dev/null" and old_path != "/dev/null":
                before_files.add(file)
        before_context = self._before_context(before_files, directory, statuses)
        selection = inputs.select_advisory_inputs(self.backend, self.work.repo,
            [inputs.AdvisoryCandidate(label, path) for label, path in paths.items()]
            + [inputs.AdvisoryCandidate(label, path) for label, path in shared_paths.items() if path is not None]
            + [inputs.AdvisoryCandidate(label, path) for label, path in before_context.items()],
            read_only=self.read_only, staged_review=True)
        admitted = selection.selected_paths()
        if not paths.keys() <= admitted.keys():
            raise inputs.SanctionedInputUnavailable("required stage inputs exceed the transport allowance")
        for label, before_path in before_context.items():
            if label in admitted:
                paths[label] = before_path
            else:
                statuses.append({"label": label, "status": "omitted", "reason": "exceeds-transport-allowance"})
        aggregate = sum(len(text.encode()) for text in contents.values())
        for label, path in shared_paths.items():
            if path is None:
                continue
            if label not in admitted:
                statuses.append({"label": label, "status": "unavailable"})
                continue
            try:
                captured = inputs._capture_input(label, path, self.transport, aggregate, staged_review=True)
            except inputs.SanctionedInputUnavailable:
                statuses.append({"label": label, "status": "unavailable"})
                continue
            aggregate += captured.size
            paths[label] = path
        supporting_count = len(deferred_paths)
        if deferred_paths:
            supporting = inputs.select_advisory_inputs(self.backend, self.work.repo,
                [inputs.AdvisoryCandidate(label, path) for label, path in {**paths, **deferred_paths}.items()],
                read_only=self.read_only, staged_review=True).selected_paths()
            deferred_paths = {label: path for label, path in deferred_paths.items() if label in supporting}
            while deferred_paths:
                catalog_entries = []
                for index, part in enumerate(self.parts):
                    label = f'diff-part-{index:06d}'
                    if label not in deferred_paths:
                        continue
                    entry: dict[str, Any] = {'file': part.assignment['file'], 'target_id': part.assignment['target_id'],
                             'path': str(deferred_paths[label]), 'source_refs': part.assignment['source_refs'],
                             'source_options': part.assignment['source_options']}
                    if len(_json({'parts': [entry]}).encode()) <= 12_000:
                        catalog_entries.append(entry)
                    else:
                        deferred_paths.pop(label)
                if not deferred_paths:
                    break
                catalog, catalog_paths = self._supporting_catalog(
                    directory / f'catalog-{len(deferred_paths)}', catalog_entries)
                all_paths = {**paths, **deferred_paths, **catalog_paths}
                count = len(all_paths)
                size = sum(path.stat().st_size for path in all_paths.values())
                if (count <= inputs.SANCTIONED_EXACT_INPUT_MAX_FILES
                        and size <= inputs.SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES):
                    break
                while deferred_paths and (count > inputs.SANCTIONED_EXACT_INPUT_MAX_FILES
                        or size > inputs.SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES):
                    _, omitted_path = deferred_paths.popitem()
                    count -= 1
                    size -= omitted_path.stat().st_size
                catalog_paths = {}
            omitted_parts = supporting_count - len(deferred_paths)
            if deferred_paths:
                state['supporting_catalog'] = {
                    'path': str(catalog), 'part_count': len(deferred_paths),
                    'guidance': (
                        "The displayed integration diff is an inventory. For before/after patch context, read this "
                        "catalog and its listed exact paths, including bounded child catalogs, even without a Git "
                        "execution tool. These frozen patch parts do not guarantee complete old source files. "
                        "Use source-first investigation when useful; read only parts relevant to concrete concerns."),
                    'status': 'partial' if omitted_parts else 'complete', 'omitted_part_count': omitted_parts}
            if omitted_parts:
                statuses.append({'label': 'supporting-parts', 'status': 'partial' if deferred_paths else 'unavailable',
                                 'reason': 'exceeds-transport-allowance'})
        paths.update(deferred_paths)
        paths.update(catalog_paths)
        hidden_labels = deferred_paths.keys() | catalog_paths.keys()
        prepared = inputs.prepare_sanctioned_inputs(self.backend, self.work.repo, paths, read_only=self.read_only,
            staged_review=True)
        prepared = replace(prepared, inputs=tuple(
            replace(item, prompt_visible=False) if item.label in hidden_labels else item for item in prepared.inputs))
        if self.bundle_capable and self.transport is inputs.SanctionedInputTransport.EXACT_PATHS:
            # Required assignments travel as exact pointers. Only shared bytes
            # actually inlined consume the inline allowance, including wrappers.
            entries: list[tuple[str, int]] = []
            inline_shared: dict[str, str] = {}
            for item in (item for item in prepared.inputs if item.label in shared_paths):
                inline_entries = [*entries, (item.label, item.size)]
                if (inputs.inline_section_emitted_bytes(inline_entries)
                        > inputs.SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES):
                    continue
                captured = inputs._capture_input(item.label, item.path, self.transport, 0, text_budget=item.size)
                if captured.sha256 != item.sha256:
                    raise inputs.SanctionedInputUnavailable('shared supporting input changed during preparation')
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
                'supporting_parts': len(deferred_paths), 'supporting_catalogs': len(catalog_paths),
                'omitted_supporting_parts': supporting_count - len(deferred_paths)}
            statuses = [status for status in statuses if status['status'] != 'complete' or
                        status['label'] not in hidden_labels]
        state.update(context_inputs=[label for label in paths if not deferred_paths or label not in hidden_labels],
                     context_transport=prepared.transport.value, context_statuses=statuses,
                     canonical_input_identities=self.binding['canonical_inputs'])
        decided = set(state.get('completed_target_ids', []))
        remaining = [part for part in self.parts if part.assignment['target_id'] not in decided]
        state['remaining_work'] = {
            'assignment_units': 1 if state['stage'] == 'integration' else len(self.parts) - len(decided),
            'files': len({part.assignment['file'] for part in remaining}),
            'stages': 1 if state['stage'] == 'integration' else sum(
                any(part['target_id'] not in decided for part in batch) for batch in self.assignment_batches)}
        exact_paths = prepared.transport is inputs.SanctionedInputTransport.EXACT_PATHS
        contexts = [{"label": item.label, **({"path": str(item.path)} if exact_paths else {"transport": "inline"})}
                    for item in prepared.inputs if item.prompt_visible]
        access_guide: dict[str, Any] = {"stage": state["stage"], "contexts": contexts}
        if 'supporting_catalog' in state:
            access_guide['supporting_catalog'] = state['supporting_catalog']
        state['access_guide'] = _json(access_guide)
        return prepared
