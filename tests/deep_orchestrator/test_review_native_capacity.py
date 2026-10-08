"""Production-sized assignment transport with real reads and a late dependency judgment."""
from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ToolResultEvent, ToolStartEvent
from tests.deep_orchestrator.test_review_capture_and_retry import read_source, source_review
from tests.deep_orchestrator.test_review_completion import scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, stage_ends
from tests.deep_orchestrator.test_review_native_sources import NativeSourceBackend
from tests.harness.git_helpers import seed_feature_branch
from tests.harness.stub_backend import completed_stage_reads
from tests.test_deep_orchestrator import _sanctioned_inputs


def _workload(wired: bool) -> tuple[dict[str, str], dict[str, str]]:
    before, after = {}, {}
    for index in range(41):
        path = f'package_{index // 5}/module_{index:02}.py'
        old, new = '', ''
        sections = 8 if index == 13 else 4 if index < 13 else 0
        for section in range(sections):
            stable = ''.join(f'# Unchanged enclosing context {section}:{line}\n' for line in range(18))
            old += stable + f'WINDOW_{section} = 0\n' + ''.join(
                f'# Old contract {section}:{line:02} ' + 'o' * 72 + '\n' for line in range(18))
            new += stable + f'WINDOW_{section} = 1\n' + ''.join(
                f'# New contract {section}:{line:02} ' + 'n' * 72 + '\n' for line in range(18))
        if not sections:
            old, new = 'VALUE = 0\n', 'VALUE = 1\n'
        if index == 0:
            old += 'def parse_flags(args):\n    return {}\n'
            new += 'def parse_flags(args):\n    return {"dry_run": "--dry-run" in args}\n'
        if index == 40:
            old += 'def build_request(options):\n    return {}\n'
            new += ('from package_0.module_00 import parse_flags\n'
                    'def build_request(options):\n'
                    + ('    return {"dry_run": options["dry_run"]}\n' if wired else '    return {}\n'))
        before[path], after[path] = old, new
    before['00_change-guide.md'] = '# Request contract\nDry-run must propagate to the request.\n'
    after['00_change-guide.md'] = before['00_change-guide.md'] + ''.join(
        f'Change context {index:04}: ' + 'c' * 100 + '\n' for index in range(800))
    return before, after


@pytest.mark.parametrize('wired', [False, True], ids=['late-cross-file-defect', 'correctly-wired-clean'])
async def test_all_41_files_and_late_contract_are_reviewed_within_192_native_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wired: bool,
) -> None:
    repo = tmp_path / 'production_sized_work'
    before, after = _workload(wired)
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    supporting_reads: list[str] = []
    source_reads: list[str] = []
    reviewed_units: list[str] = []
    judged_late: list[bool] = []
    late = 'package_8/module_40.py'

    def read(path: Path, call: str) -> Iterable[AgentEvent]:
        body = path.read_text()
        yield ToolStartEvent(id=call, name='Read', input={'file_path': str(path)})
        yield ToolResultEvent(id=call, output=body, is_error=False)

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            yield from completed_stage_reads(repo, stage)
            return
        assert stage['remaining_tool_calls'] <= 192
        bodies = {block['files'][0]: block['excerpt'] for block in stage['evidence']
                  if len(block['files']) == 1 and not block['partial']}
        pointers = _sanctioned_inputs(review.backend.calls[-1]['prompt'])
        for label, path in pointers.items():
            if label.startswith('source-'):
                continue
            supporting_reads.append(label)
            yield from read(path, f'supporting-{len(supporting_reads)}')
            if label == 'review-assignment':
                bundle = path.read_text()
                assert '### diff (supporting)' in bundle and '### input-binding (supporting)' in bundle
        for window in stage.get('source_access', []):
            if window['side'] != 'after' or not window['read_required']:
                continue
            source_reads.append(window['file'])
            path = Path(window['access']['path'])
            bodies[window['file']] = path.read_text()
            yield from read(path, f'window-{len(source_reads)}')
        if not stage.get('source_access'):
            for path in stage['assigned_files']:
                source_reads.append(path)
                bodies[path] = (repo / path).read_text()
                yield from read(repo / path, f'legacy-source-{len(source_reads)}')
        for part in stage['assignment_parts']:
            path = part['file']
            if path not in bodies:
                # A complete admitted receipt can have a clipped prompt view.
                # Fresh source is needed when the actual code is unavailable here.
                window = next(window for window in stage['source_access']
                              if window['file'] == path and window['side'] == 'after')
                projection = Path(window['access']['path'])
                bodies[path] = projection.read_text()
                source_reads.append(path)
                yield from read(projection, f'visible-code-{len(source_reads)}')
            # Interpret actual changed declarations, rather than trusting targets.
            tree = ast.parse(bodies[path])
            values = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Constant)]
            assert values and all(value == 1 for value in values)
            reviewed_units.append(part['target_id'])
        output['notes'] = 'Changed numeric window declarations preserve the value-one contract.'
        if late in stage['assigned_files']:
            producer = repo / 'package_0/module_00.py'
            yield from read(producer, 'late-import-producer')
            source_reads.append('package_0/module_00.py')
            assert '"dry_run": "--dry-run" in args' in producer.read_text()
            tree = ast.parse(bodies[late])
            request = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name == 'build_request')
            returned = request.body[0]
            assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Dict)
            carries_flag = any(isinstance(key, ast.Constant) and key.value == 'dry_run'
                               for key in returned.value.keys)
            judged_late.append(carries_flag)
            if not carries_flag:
                finding = {'id': 1, 'file': late, 'line': 4, 'severity': 'medium', 'confidence': 'MEDIUM',
                           'description': 'Late request builder drops the parsed dry-run flag',
                           'rationale': 'parse_flags creates dry_run but build_request does not send it.',
                           'evidence': f'{late}:4 returns an empty request; package_0/module_00.py defines dry_run.'}
                output['candidates'] = [{'candidate_id': '', 'file': late, 'line': 4,
                                         'trigger': 'Parse --dry-run and build the request',
                                         'consequence': 'The request omits dry_run', 'grounds': finding['evidence'],
                                         'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = response
    await review.finish('python', findings=() if wired else ('Late request builder drops the parsed dry-run flag',))
    data = review.load()
    assert len(scopes(data)['python']['files']) == 41 and late in scopes(data)['python']['files']
    assert len(reviewed_units) == len(set(reviewed_units)) == 87
    assert judged_late == [wired]
    metadata = [phase['metadata'] for phase in stage_ends(review, 'python')]
    assert metadata[0]['hard_tool_call_allowance'] == 192
    assert metadata[-1]['observed_tool_starts'] <= 192
    assert source_reads.count(late) == 1 and len(set(source_reads)) == 41
    assert 42 <= len(source_reads) <= 46
    assert supporting_reads.count('review-assignment') == len(metadata)
    assert len(supporting_reads) <= 2 * len(metadata)
    assert all(phase['admitted'] for phase in metadata)
    assert len((repo / '.daydream/diff.patch').read_bytes()) > 256 * 1024


async def test_small_complete_shared_context_is_inline_and_needs_one_supporting_pointer_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = source_review(tmp_path, monkeypatch, source_bytes=400, files=2)
    reads: list[str] = []
    prompts: list[str] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        prompt = review.backend.calls[-1]['prompt']
        prompts.append(prompt)
        for label, path in _sanctioned_inputs(prompt).items():
            if label.startswith('source-'):
                continue
            reads.append(label)
            body = path.read_text()
            yield ToolStartEvent(id=f'supporting-{label}', name='Read', input={'file_path': str(path)})
            yield ToolResultEvent(id=f'supporting-{label}', output=body, is_error=False)
        for path in stage['assigned_files']:
            yield from read_source(review, path, f'judged-{path}')
        assert "return 'universe'" in (review.repo / 'api.py').read_text()
        assert 'VALUE = 1' in (review.repo / 'module_00.py').read_text()
        output['notes'] = 'Current greeting and module value agree with the declared change intent.'

    review.backend.stage_response = response
    await review.finish('python')
    assert reads == ['review-assignment']
    assert '<sanctioned-input label="intent">' in prompts[0]
    assert 'The PR updates greetings across stacks.' in prompts[0]
