"""Production-sized assignment transport with a late cross-file judgment."""
from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, PiRequestConfig, RequestEvent, ToolResultEvent, ToolStartEvent
from tests.deep_orchestrator.test_review_capture_and_retry import part_identity
from tests.deep_orchestrator.test_review_completion import scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, candidate, read_events, stage_ends
from tests.deep_orchestrator.test_review_stage_reservations import NativeStageBackend
from tests.harness.git_helpers import seed_feature_branch, tracked_source_state


def _workload(wired: bool, *, guide_lines: int = 800) -> tuple[dict[str, str], dict[str, str]]:
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
            new += ("from package_0.module_00 import parse_flags\ndef build_request(options):\n"
                    + ('    return {"dry_run": options["dry_run"]}\n' if wired else '    return {}\n'))
        before[path], after[path] = old, new
    before['00_change-guide.md'] = '# Request contract\nDry-run must propagate to the request.\n'
    after['00_change-guide.md'] = before['00_change-guide.md'] + ''.join(
        f'Change context {index:04}: ' + 'c' * 100 + '\n' for index in range(guide_lines))
    return before, after


@pytest.mark.parametrize('wired', [False, True], ids=['late-cross-file-defect', 'correctly-wired-clean'])
@pytest.mark.parametrize('inline', [False, True], ids=['exact-paths', 'inline'])
async def test_late_cross_file_judgment_fits_cumulative_native_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wired: bool, inline: bool) -> None:
    repo = tmp_path / 'production_sized_work'
    before, after = _workload(wired)
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    monkeypatch.setattr(review.backend, 'read_only_disposable_clone', inline, raising=False)
    late = 'package_8/module_40.py'
    finding = {'id': 1, 'file': late, 'line': 4, 'severity': 'medium', 'confidence': 'MEDIUM',
               'description': 'Late request builder drops the parsed dry-run flag',
               'rationale': 'The changed request builder omits the dry-run value.',
               'evidence': f'{late}:4 omits dry_run required by the changed request contract.'}
    assigned_files: set[str] = set()
    assignment_units = 0
    original_parts: list[dict[str, Any]] = []
    inline_sizes: list[int] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal assignment_units
        yield RequestEvent(prompt=review.backend.calls[-1]['prompt'],
                           output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        assert stage['remaining_tool_calls'] <= 192
        assert stage['context_transport'] == ('inline' if inline else 'exact_paths')
        assigned_files.update(stage['assigned_files'])
        assignment_units += len(stage['assignment_parts'])
        original_parts.extend(stage['assignment_parts'])
        assert len(stage['assigned_files']) <= 4
        if inline:
            prompt = review.backend.calls[-1]['prompt']
            inline_sizes.append(len(prompt[prompt.index('Sanctioned phase inputs (captured verbatim):'):].encode()))
            assert inline_sizes[-1] <= 64 * 1024
        output['notes'] = 'The late request builder is assessed against the changed dry-run contract.'
        if not wired and late in stage['assigned_files']:
            output['candidates'] = [candidate(disposition='confirmed', finding=finding, file=late, line=4,
                                             trigger='Parse --dry-run and build the request',
                                             consequence='The request omits dry_run', grounds=finding['evidence'])]
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    data = await review.finish('python', deep_shard_enabled=False,
                               findings=() if wired else ('Late request builder drops the parsed dry-run flag',))
    records = json.loads((repo / '.daydream/deep/stack-python-records.json').read_text())
    assert records['issues'] == ([] if wired else [dict(finding, uid='python:1')])
    assert records['analyzed_revision'] == data['terminal_result']['analyzed_revision']
    assert records['originating_run_id'] == data['terminal_result']['run_id']
    assert len(scopes(data)['python']['files']) == 41 and late in scopes(data)['python']['files']
    assert assigned_files == set(after for after in before if after.endswith('.py'))
    assert assignment_units == (87 if inline else 48)
    if not wired:
        assert part_identity(original_parts) == (
            '1bce80a2a82762289d9663dd38cbb730b13218f4058ce01b65fdbb91f524d02b' if inline else
            '9dab8a79d66313a9b9d8f4dafdbd68bf392ff9915109a582ffeccb2f54812c5d')
    if inline:
        assert max(inline_sizes) > 12 * 1024
        assert len(stage_ends(review, 'python')) < 37
    metadata = [phase['metadata'] for phase in stage_ends(review, 'python')]
    assert metadata[0]['hard_tool_call_allowance'] == 192
    assert metadata[-1]['observed_tool_starts'] <= 192
    assert sum(phase['submission_starts'] for phase in metadata) == len(metadata)
    assert all(phase['admitted'] for phase in metadata)
    assert len((repo / '.daydream/diff.patch').read_bytes()) > 256 * 1024


@pytest.mark.parametrize('wired', [False, True], ids=['dropped-flag', 'wired-flag'])
async def test_structure_traces_early_flag_to_late_request_under_original_scope_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wired: bool) -> None:
    repo = tmp_path / 'structure_flag_flow'
    before, after = _workload(wired, guide_lines=20)
    seed_feature_branch(repo, base=before, feature=after)
    source_before = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    backend = NativeStageBackend(repo)
    review.backend = backend
    early, late = 'package_0/module_00.py', 'package_8/module_40.py'
    evidence_requests: list[tuple[str, str]] = []

    @backend.script('structure')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        yield RequestEvent(prompt=backend.calls[-1]['prompt'], output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        assert stage['analyzed_revision']['head_sha'] == source_before['head']
        if stage['stage'] == 'integration':
            assert stage['assigned_target_ids'] == ['integration:structure']
            assert set(stage['assigned_files']) == set(after)
            assert stage['observed_tool_starts'] == 0
            assert stage['remaining_tool_calls'] == stage['hard_tool_call_allowance'] == 96
        else:
            assert stage['stage'] == 'triage'
            assert stage['completed_target_ids'] == ['integration:structure']
            assert stage['remaining_tool_calls'] == stage['hard_tool_call_allowance'] == 93
            assert stage['observed_tool_starts'] == 3
            assert stage['assigned_candidate_ids'] == ['structure:candidate:1']
            assert stage['closed_candidate_ids'] == []
        namespace: dict[str, Any] = {}
        for path in (early, late):
            evidence_requests.append((stage['stage'], path))
            for event in read_events(repo, path, event_id=f"{stage['stage']}-{path}"):
                if isinstance(event, ToolResultEvent):
                    functions: list[ast.stmt] = [node for node in ast.parse(event.output).body
                                                if isinstance(node, ast.FunctionDef)]
                    exec(compile(ast.Module(body=functions, type_ignores=[]), path, 'exec'), namespace)
                yield event
        options = namespace['parse_flags'](['--dry-run'])
        assert options == {'dry_run': True}
        propagated = namespace['build_request'](options).get('dry_run') is True
        output['notes'] = 'Traced the parsed dry-run option through request construction.'
        if not propagated:
            finding = {'id': 1, 'file': late, 'line': 4, 'severity': 'medium', 'confidence': 'MEDIUM',
                       'description': 'Late request builder drops the parsed dry-run flag',
                       'rationale': 'Parsing --dry-run sets the option but request construction omits it.',
                       'evidence': f'{early} parses dry_run=True; {late}:4 returns a request without dry_run.'}
            if stage['stage'] == 'integration':
                output['candidates'] = [candidate(file=late, line=4, trigger='Parse --dry-run and build the request',
                    consequence='The request omits dry_run', grounds=finding['evidence'])]
            else:
                pending, = stage['candidates']
                assert pending['file'] == late and pending['grounds'] == finding['evidence']
                output['candidates'] = [dict(pending, disposition='confirmed', finding=finding)]
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    data = await review.finish('structure', deep_shard_enabled=False,
                               findings=() if wired else ('Late request builder drops the parsed dry-run flag',))
    assert set(scopes(data)) == {'python', 'generic', 'structure'}
    assert set(scopes(data)['structure']['files']) == set(after)
    assert data['terminal_result']['analysis_state'] == 'complete'
    stages = [stage for stage in backend.stages if stage['scope_id'] == 'structure']
    assert [stage['stage'] for stage in stages] == (['integration'] if wired else ['integration', 'triage'])
    assert evidence_requests == [(stage, path) for stage in ('integration',) + (() if wired else ('triage',))
                                 for path in (early, late)]
    metadata = [event['metadata'] for event in stage_ends(review, 'structure')]
    assert [item['observed_tool_starts'] for item in metadata] == ([3] if wired else [3, 6])
    assert [item['remaining_tool_calls'] for item in metadata] == ([93] if wired else [93, 90])
    assert all(item['admitted'] and item['submission_starts'] == item['successful_submissions'] == 1
               for item in metadata)
    records = json.loads((repo / '.daydream/deep/stack-structure-records.json').read_text())
    assert records['analyzed_revision'] == data['terminal_result']['analyzed_revision']
    assert [item['uid'] for item in records['issues']] == ([] if wired else ['structure:1'])
    assert 64 * 1024 < len((repo / '.daydream/diff.patch').read_bytes()) <= 256 * 1024
    assert tracked_source_state(repo) == source_before
