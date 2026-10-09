"""Snapshot-bound terminal contracts through the real runner and Git."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from daydream import git_ops, json_utils, runner
from daydream.backends import AgentEvent, MaxTurnsError, ResultEvent, TextEvent, ToolStartEvent
from daydream.dataset import LocalRecordStore
from daydream.findings import FindingsValidationError, load_findings_artifact
from daydream.phases import findings
from daydream.phases.review import ReviewOutputError
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.console import collapse_panel_text
from tests.harness.dataset import read_records
from tests.harness.fake_clock import FakeClock
from tests.harness.git_helpers import git
from tests.harness.stub_backend import completed_stage_reads, review_stage_result, review_stage_state
from tests.test_deep_orchestrator import _pin_findings_pr, _profile_with_pipeline, _record


class ReviewRun:
    def __init__(self, repo: Path, tmp: Path, patch: pytest.MonkeyPatch) -> None:
        self.repo, self.tmp, self.patch = repo, tmp, patch
        self.pr = _pin_findings_pr(patch, repo)
        self.output = tmp / 'findings.json'
        self.backend = EmptyReviewBackend(repo)
        self.config = empty_review_config(repo, tmp / 'trajectory.json', findings_out=str(self.output), pr_number=7)
        patch.setattr('daydream.runner.create_backend', lambda *_a, **_k: self.backend)

    async def run(self, **overrides: Any) -> int:
        return await runner.run(replace(self.config, **overrides))

    def load(self) -> dict[str, Any]:
        artifact = load_findings_artifact(self.output, expected_repo='o/r', expected_pr_number=7,
                                         expected_head_sha=self.pr.head_sha)
        assert artifact.schema_version == 2 and artifact.head_sha == self.pr.head_sha
        data: dict[str, Any] = json.loads(self.output.read_text())
        revision = data['terminal_result']['analyzed_revision']
        assert (data['head_sha'], revision['head_sha'], revision['merge_base_sha']) == (
            self.pr.head_sha, self.pr.head_sha, self.pr.base_sha)
        return data


@pytest.fixture
def review(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReviewRun:
    return ReviewRun(multi_stack_target, tmp_path, monkeypatch)


def scopes(data: dict[str, Any]) -> dict[str, Any]:
    return {s['scope_id']: s for s in data['terminal_result']['stack_outcomes']}


def record(line: int = 2) -> dict[str, Any]:
    return _record(description='Grounded defect', file='api.py', line=line, severity='medium', confidence='MEDIUM',
                   rationale='Boundary evidence', evidence=f'api.py:{line}')


def reviewer(prompt: str) -> bool:
    return 'you are reviewing the ' in prompt.lower() or 'you are the structural reviewer' in prompt.lower()


class AuthError(RuntimeError):
    category = 'AUTH_CONFIG'


@pytest.mark.parametrize(('case', 'state', 'reason'), [
    ('empty', 'complete', None), ('backend', 'incomplete', 'backend_failure'),
    ('auth', 'incomplete', 'authentication_failure'), ('model', 'incomplete', 'model_budget_exhaustion'),
    ('structural', 'complete', None), ('mixed', 'incomplete', 'backend_failure'),
    ('total', 'failed', 'backend_failure'), ('demoted', 'complete', None),
    ('language', 'complete', None), ('archive', 'incomplete', 'backend_failure'),
])
async def test_outcomes(review: ReviewRun, archive_dir: Path, case: str, state: str, reason: str | None) -> None:
    nonempty = case in {'structural', 'mixed', 'demoted', 'language'}
    errors = {'auth': AuthError('rejected'), 'model': MaxTurnsError('spent turns')}
    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
        review_by_stack={'python' if case == 'language' else 'structure': [record(500 if case == 'demoted' else 2)]}
        if nonempty else {}, fail_stack='python' if reason and case != 'total' else None,
        stack_error=errors.get(case), responder=lambda p: [RuntimeError('unavailable')]
        if case == 'total' and reviewer(p) else None)
    review.backend.merge_echo_records = True
    if case == 'language':
        review.backend.merge_items = None
    assert await review.run(archive=case == 'archive') == 0
    data = review.load()
    result, inventory = data['terminal_result'], scopes(data)
    assert (result['analysis_state'], result['pipeline_state']) == (state, 'completed')
    assert set(inventory) == {'python', 'react', 'generic', 'structure'}
    assert bool(data['findings']) is nonempty
    assert {name: s['status'] for name, s in inventory.items()} == {
        name: 'incomplete' if name == 'python' and case == 'model' else
        'failed' if case == 'total' or (name == 'python' and reason) else 'complete' for name in inventory}
    if reason:
        assert inventory['python']['reason_codes'] == [reason]
    else:
        assert result['reason_codes'] == []
    if nonempty:
        assert len(data['findings']) == 1 and data['findings'][0]['title'] == 'Grounded defect'
    if case == 'demoted':
        assert (data['findings'][0]['confidence'], data['findings'][0]['location_distrust']) == ('LOW', True)
    if case == 'language':
        canonical = json.loads((review.repo / '.daydream/deep/merged-items.json').read_text())
        assert canonical['items'][0]['source_uids'] == ['python:1']
    if case == 'empty':
        stages = [stage_state for c in review.backend.calls
                  if (stage_state := review_stage_state(c['prompt'])) is not None]
        assert {stage['scope_id'] for stage in stages} == set(inventory)
        assert [stage['stage'] for stage in stages if stage['scope_id'] == 'structure'] == ['integration']
        assert not any('cross-stack merge agent' in c['prompt'].lower() or
                       'supervisor adjudication' in c['prompt'].lower() for c in review.backend.calls)
    if case == 'archive':
        archived = archive_dir / 'runs' / result['run_id']
        assert json.loads((archived / 'manifest.json').read_text())['archive_status'] == 'complete'
        assert json.loads((archived / 'findings.json').read_text()) == data
        paths = list(archived.rglob('review-coverage.json'))
        assert len(paths) == 1
        saved = json.loads(paths[0].read_text())
        assert (saved['stack_outcomes'], saved['analyzed_revision']) == (result['stack_outcomes'],
                                                                      result['analyzed_revision'])
        assert json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text()) == saved


@pytest.mark.parametrize(('mode', 'reason', 'payload'), [
    ('missing', 'missing_output', None), ('malformed', 'malformed_output', {'unexpected': []}),
    ('invalid', 'malformed_output', {'issues': [{'id': 1}, 'invalid']}),
    ('rejected-native', 'malformed_output', {'unexpected': []}), ('text-only', None, None),
    *[('invalid-envelope', 'malformed_output', payload) for payload in
      ({}, {'issues': 'bad'}, {'issues': [{'description': 'unfinished'}]})],
])
async def test_schema_output(review: ReviewRun, mode: str, reason: str | None, payload: Any) -> None:

    def response(prompt: str) -> list[AgentEvent] | None:
        state = review_stage_state(prompt)
        if state is None or state['scope_id'] != 'python':
            return None
        return [*completed_stage_reads(review.repo, state),
                TextEvent(text=json.dumps(review_stage_result(prompt, []))
                          if mode in {'text-only', 'rejected-native'} else ''),
                ResultEvent(structured_output=payload, continuation=None)]

    review.backend.responder = response
    assert await review.run() == 0
    data = review.load()
    assert data['terminal_result']['analysis_state'] == ('complete' if reason is None else 'incomplete')
    assert scopes(data)['python']['reason_codes'] == ([] if reason is None else [reason])
    assert data['findings'] == []
    saved = json.loads((review.repo / '.daydream/deep/stack-python-records.json').read_text())
    assert saved['issues'] == [] and bool(saved.get('incomplete')) is (reason is not None)


@pytest.mark.parametrize('case', ['live', 'dirty', 'no-diff', 'dirty-no-diff', 'interactive', 'mismatch',
                                  'base-tip', 'shards', 'pipeline-budget'])
async def test_snapshot_boundaries(review: ReviewRun, request: pytest.FixtureRequest, case: str) -> None:
    overrides: dict[str, Any] = {}
    store = LocalRecordStore(review.tmp / 'records')
    if case in {'mismatch', 'no-diff'}:
        overrides.update(dataset_capture=True, dataset_store_path=store.root)
    prior = {'.review-output.md': 'prior completed review\n', '.daydream/deep/history.json': '{"prior": true}\n',
             '.daydream/deep/merged-items.json': '{"items": [{"item_uid": "item:old"}]}\n',
             '.daydream/recommended.patch': 'prior recommended patch\n'}
    if case == 'mismatch':
        for name, text in prior.items():
            path = review.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    live, calls = review.pr, []
    if case in {'no-diff', 'dirty-no-diff', 'base-tip'}:
        git(review.repo, 'checkout', 'main')
        if case == 'base-tip':
            (review.repo / 'base-only.txt').write_text('advanced\n')
            git(review.repo, 'add', 'base-only.txt')
            git(review.repo, 'commit', '-m', 'advance base')
            tip = git_ops.head_sha(review.repo)
            git(review.repo, 'checkout', 'feature')
            live = replace(review.pr, pr_base_sha=tip)
        else:
            review.pr = _pin_findings_pr(review.patch, review.repo)
            live = review.pr
    if case in {'dirty', 'dirty-no-diff', 'interactive'}:
        (review.repo / 'api.py').write_text('DIRTY = True\n')
    if case == 'interactive':
        overrides['findings_out'] = None
    if case == 'shards':
        shard_target = cast(Path, request.getfixturevalue('shard_many_python_target'))
        review = ReviewRun(shard_target, review.tmp, review.patch)
        overrides.update(deep_shard_enabled=True, deep_shard_max_files=1)
    if case == 'pipeline-budget':
        overrides['review_profile'] = _profile_with_pipeline(review_wall_budget_s=0)
    if case in {'live', 'mismatch', 'base-tip'}:
        def lookup(*_a: Any, **_k: Any) -> Any:
            calls.append(live)
            return replace(live, head_sha=live.base_sha) if case == 'mismatch' else live
        review.patch.setattr('daydream.pr_review.find_pr_by_number', lookup)
        def advance(_p: str) -> None:
            nonlocal live
            live = replace(review.pr, head_sha=review.pr.base_sha)
        if case == 'live':
            review.backend = EmptyReviewBackend(review.repo, forbid_supervise=False,
                review_by_stack={'structure': [record()]}, responder=advance)
    if case == 'no-diff':
        assert await review.run(dataset_store_path=store.root) == 0 and not store.root.exists()
    assert await review.run(**overrides) == (1 if case in {'dirty', 'dirty-no-diff', 'mismatch'} else 0)
    if case in {'mismatch', 'no-diff'}:
        raw = read_records(store).runs[0]
        assert raw['outcome'] == ('failed' if case == 'mismatch' else 'success')
        if case == 'mismatch':
            assert raw['original_task']['status'] == 'unavailable'
            assert all(raw[name]['status'] == 'unproduced'
                       for name in ('trajectories', 'findings', 'scoring', 'recommended_patch'))
            assert all((review.repo / name).read_text() == text for name, text in prior.items())
        else:
            assert raw['original_task']['value']['diff'] == '' and raw['findings']['value']['items'] == []
            assert raw['scoring']['status'] == 'unproduced'
    if case in {'interactive', 'mismatch'}:
        assert not review.output.exists()
        assert bool(review.backend.calls) is (case == 'interactive')
        if case == 'interactive':
            assert 'DIRTY = True' in (review.repo / '.daydream/diff.patch').read_text()
        return
    data = review.load()
    result = data['terminal_result']
    failed = case in {'dirty', 'dirty-no-diff', 'pipeline-budget'}
    assert result['analysis_state'] == ('failed' if failed else 'complete')
    if case in {'dirty', 'dirty-no-diff', 'no-diff', 'pipeline-budget'}:
        assert review.backend.calls == []
    if case.startswith('dirty'):
        assert 'dirty_snapshot' in result['reason_codes']
    if case == 'pipeline-budget':
        assert 'host_pipeline_budget_exhaustion' in result['reason_codes']
        assert all(s['status'] == 'uncovered' for s in scopes(data).values())
    if case == 'no-diff':
        assert result['stack_outcomes'] == [] and result['phase_outcomes'][0]['noop'] is True and result['run_id']
    if case == 'live':
        assert live.head_sha == review.pr.base_sha and git_ops.head_sha(review.repo) == review.pr.head_sha
        assert git_ops.show(review.repo, live.head_sha, 'api.py') and len(calls) == 1
        assert (data['findings'][0]['placement'], data['findings'][0]['line']) == ('inline', 2)
        assert result['analyzed_revision']['pr_base_sha'] == review.pr.base_sha
    if case == 'base-tip':
        assert result['analyzed_revision']['pr_base_sha'] == tip != review.pr.base_sha
    if case == 'shards':
        assert set(scopes(data)) == {'python#0', 'python#1', 'python#2', 'generic', 'structure'}
        assert all(s['status'] == 'complete' for s in scopes(data).values())
        assert {f for n, s in scopes(data).items() if n != 'structure' for f in s['files']} == {
            'mod0.py', 'mod1.py', 'mod2.py', 'README.md'}


@pytest.mark.parametrize('budget', ['tool', 'wall', 'model'])
@pytest.mark.parametrize('nonempty', [False, True])
async def test_unsuccessful_stage_discards_checkpoints(
    review: ReviewRun, budget: str, nonempty: bool,
) -> None:
    fake = FakeClock().install(review.patch)
    def response(prompt: str) -> Any:
        if reviewer(prompt):
            yield ResultEvent(structured_output=review_stage_result(prompt, [record()] if nonempty else []),
                              continuation=None)
            if budget == 'model':
                yield MaxTurnsError('spent turns')
            else:
                state = review_stage_state(prompt)
                assert state is not None
                for index in range(state['remaining_tool_calls'] + 1):
                    if budget == 'wall':
                        fake.advance(601)
                    yield ToolStartEvent(id=f'budget-{index}', name='Read', input={'file_path': 'api.py'})
    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
                                       responder=lambda p: response(p) if reviewer(p) else None)
    review.backend.merge_echo_records = True
    assert await review.run(review_profile=_profile_with_pipeline(review_wall_budget_s=99999)) == 0
    data = review.load()
    result = data['terminal_result']
    reason = 'model_budget_exhaustion' if budget == 'model' else f'host_{budget}_budget_exhaustion'
    assert result['analysis_state'] == 'failed' and result['completed_stacks'] == []
    assert reason in result['reason_codes'] and data['findings'] == []
    saved = json.loads((review.repo / '.daydream/deep/stack-python-records.json').read_text())
    assert saved['issues'] == [] and saved['incomplete'] is True
    for scope in scopes(data).values():
        assert (scope['status'], scope['partial_evidence']) == ('incomplete', False)
        assert scope['reason_codes'] == [reason]


@pytest.mark.parametrize('fault', ['missing', 'corrupt', 'shape', 'revision', 'scope', 'origin'])
async def test_loaded_artifact_faults(review: ReviewRun, fault: str) -> None:
    write, read = Path.write_text, Path.read_text
    malformed = '{invalid' if fault == 'corrupt' else '{"issues": "invalid"}'
    def faulty_write(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        if path.name == 'stack-python-records.json' and fault in {'corrupt', 'shape'}:
            text = malformed
        count = write(path, text, *args, **kwargs)
        if path.name == 'stack-python-records.json' and fault == 'missing':
            path.unlink()
        return count
    def faulty_read(path: Path, *args: Any, **kwargs: Any) -> str:
        text = read(path, *args, **kwargs)
        if path.name == 'stack-python-records.json' and fault in {'revision', 'scope', 'origin'}:
            data = json.loads(text)
            if fault == 'revision':
                data['analyzed_revision']['head_sha'] = 'foreign-head'
            elif fault == 'scope':
                data['scope_id'] = 'foreign-scope'
            else:
                data.pop('originating_run_id')
            return json.dumps(data)
        return text
    review.patch.setattr(Path, 'write_text', faulty_write)
    review.patch.setattr(Path, 'read_text', faulty_read)
    if fault in {'missing', 'corrupt', 'shape'}:
        review.backend.review_by_stack = {'react': [_record(
            description='Surviving sibling defect', file='App.tsx', line=1)]}
    assert await review.run(dataset_capture=True, dataset_store_path=review.tmp / 'records') == 1
    if fault in {'corrupt', 'shape'}:
        raw = read_records(LocalRecordStore(review.tmp / 'records')).runs[0]
        assert raw['outcome'] == 'failed' and raw['completeness']['artifact_acquisition'] == 'failed'
        diagnostic = raw['provenance']['collection_diagnostics']['stack-python-records.json']
        assert diagnostic['reason'] == ('malformed_json' if fault == 'corrupt' else 'malformed_shape')
        assert diagnostic.get('raw_text' if fault == 'corrupt' else 'raw_json') == (
            malformed if fault == 'corrupt' else {'issues': 'invalid'})
        scoring = raw['scoring']['value']
        assert scoring['format_valid'] is (fault == 'shape')
        assert scoring['persisted_breakdown']['correctness_per_finding'] is None
        assert scoring['persisted_breakdown']['composite'] == (0.0 if fault == 'corrupt' else None)
    data = review.load()
    result, inventory = data['terminal_result'], scopes(data)
    reason = 'missing_artifact' if fault == 'missing' else 'malformed_artifact'
    assert (result['pipeline_state'], result['analysis_state'], result['projection_valid']) == (
        'failed', 'incomplete', True)
    assert set(inventory) == {'python', 'react', 'generic', 'structure'} and reason in result['reason_codes']
    assert {n: s['status'] for n, s in inventory.items()} == dict.fromkeys(inventory, 'complete') | {'python': 'failed'}
    assert inventory['python']['reason_codes'] == [reason]
    if fault in {'missing', 'corrupt', 'shape'}:
        assert [(f['title'], f['placement']) for f in data['findings']] == [('Surviving sibling defect', 'inline')]
    else:
        assert data['findings'] == []
    assert not any('cross-stack merge agent' in c['prompt'].lower() for c in review.backend.calls)


@pytest.mark.parametrize(('phase', 'reason'), [('intent', 'backend_failure'), ('alternatives', 'backend_failure'),
    ('merge', 'synthesis_failure'), ('merge-low', 'synthesis_failure'),
    ('missing', 'missing_output'), ('malformed', 'malformed_output'),
    ('omitted', 'evidence_incomplete')])
async def test_required_phase_faults(review: ReviewRun, phase: str, reason: str) -> None:
    from tests.harness.review_profile import independent_alternatives_profile
    from tests.harness.review_result import merge_result
    def response(prompt: str) -> Any:
        lower = prompt.lower()
        if (phase == 'intent' and 'present your understanding concisely' in lower) or (
                phase == 'alternatives' and 'evaluate the implementation' in lower):
            return [RuntimeError(f'{phase} unavailable')]
        if phase in {'missing', 'malformed', 'omitted'} and 'supervisor adjudication' in lower:
            payload = None if phase == 'missing' else {'verdicts': [None] if phase == 'malformed' else []}
            return [ResultEvent(structured_output=payload, continuation=None)]
        if phase == 'merge-low' and 'cross-stack merge agent' in lower:
            return [ResultEvent(structured_output=merge_result([
                dict(record(), confidence='LOW', lens='per-stack'),
            ]), continuation=None)]
    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
        review_by_stack={'python' if phase in {'merge', 'merge-low'} else 'structure': [record()]}, responder=response)
    review.backend.merge_echo_records = True
    review.backend.merge_emit_str = 'invalid merge output' if phase == 'merge' else None
    if phase in {'intent', 'alternatives', 'merge', 'merge-low'}:
        review.config = replace(review.config, review_profile=independent_alternatives_profile())
    if phase in {'merge', 'merge-low', 'omitted'}:
        assert await review.run() == (1 if phase in {'merge', 'merge-low'} else 0)
    else:
        error_type = ReviewOutputError if phase in {'missing', 'malformed'} else RuntimeError
        with pytest.raises(error_type, match=None if error_type is ReviewOutputError else f'{phase} unavailable'):
            await review.run()
    data = review.load()
    result = data['terminal_result']
    assert (result['pipeline_state'], result['analysis_state']) == (
        'completed' if phase == 'omitted' else 'failed',
        'failed' if phase in {'intent', 'alternatives'} else 'incomplete')
    assert reason in result['reason_codes']
    if phase in {'merge', 'merge-low'}:
        outcome = next(p for p in result['phase_outcomes'] if p['phase'] == 'merge')
        assert outcome['status'] == 'failed'
        assert 'synthesis_failure' in outcome['reason_codes']
    expected_titles = [] if phase in {'intent', 'alternatives'} else ['Grounded defect']
    assert [f['title'] for f in data['findings']] == expected_titles
    if phase in {'missing', 'malformed', 'omitted'}:
        outcome = next(p for p in result['phase_outcomes'] if p['phase'] == 'supervision')
        expected = 'incomplete' if phase == 'omitted' else 'failed'
        assert (outcome['status'], outcome['reason_codes']) == (expected, [reason])
        assert any('supervisor adjudication' in c['prompt'].lower() for c in review.backend.calls)


@pytest.mark.parametrize('late_event', ['legacy-result', 'native-start'])
@pytest.mark.parametrize('offset', [0, 1], ids=['at-deadline', 'after-deadline'])
async def test_late_supervision_events_cannot_supply_reserved_finalization_or_replace_prior_findings(
    review: ReviewRun, ext_dir: ExtDir, late_event: str, offset: int,
) -> None:
    import re

    from daydream.backends import PiRequestConfig, RequestEvent
    from daydream.phases.schemas import SUPERVISE_SCHEMA

    fake = FakeClock().install(review.patch)
    finalized: list[str] = []
    marker = review.tmp / 'supervised-native-start.json'
    ext_dir.write_module(
        'from pathlib import Path\n'
        'from daydream.extensions import ToolDecision\n'
        'def supervise(name, tool_input, *, phase):\n'
        '    if name == "structured_output":\n'
        f'        Path({str(marker)!r}).write_text(name)\n'
        '    return ToolDecision(veto=False)\n'
        'def register(registry): registry.register_tool_supervisor(supervise)\n'
    )
    late_title = 'LATE_RESULT_MUST_NOT_REPLACE_PRIOR_FINDING'

    def response(prompt: str) -> Any:
        if prompt.startswith('INVESTIGATION HAS ENDED.'):
            finalized.append(prompt)
            assert late_title not in prompt
            yield ResultEvent(structured_output={'verdicts': []}, continuation=None)
        elif 'supervisor adjudication' in prompt.lower():
            if late_event == 'native-start':
                yield RequestEvent(prompt=prompt, output_schema=SUPERVISE_SCHEMA,
                                   config=PiRequestConfig(schema_emulated=False, no_tools=False))
            allowance = re.search(r'Investigation allowance: at most ([\d.e+-]+) seconds', prompt)
            assert allowance is not None
            fake.advance(float(allowance[1]) + offset)
            proposal = {'verdicts': [{'id': 1, 'action': 'edit', 'reason': 'Late proposal',
                                     'severity': None, 'confidence': None, 'description': late_title,
                                     'rationale': None, 'evidence': None}]}
            yield (ToolStartEvent(id='expired-native-submit', name='structured_output', input=proposal)
                   if late_event == 'native-start' else ResultEvent(structured_output=proposal, continuation=None))

    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
                                        review_by_stack={'structure': [record()]},
                                        responder=lambda prompt: response(prompt) if (
                                            prompt.startswith('INVESTIGATION HAS ENDED.')
                                            or 'supervisor adjudication' in prompt.lower()) else None)
    review.backend.merge_echo_records = True
    assert await review.run() == 0
    data = review.load()
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    assert len(finalized) == 1, 'A result arriving after expiry must not become the legacy fallback checkpoint'
    outcome = next(phase for phase in data['terminal_result']['phase_outcomes'] if phase['phase'] == 'supervision')
    assert outcome['status'] == 'incomplete' and outcome['reason_codes'] == ['host_wall_budget_exhaustion']
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    if late_event == 'native-start':
        assert marker.read_text() == 'structured_output'
        trajectory = json.loads((review.tmp / 'trajectory.json').read_text())
        starts = [call['tool_call_id'] for trace in [trajectory, *trajectory['extra']['subtrajectories']]
                  for step in trace.get('steps', []) for call in step.get('tool_calls') or []]
        assert starts.count('expired-native-submit') == 1


@pytest.mark.parametrize('fault', ['absent-install', 'prior-install', 'staging'])
async def test_atomic_public_failure(review: ReviewRun, fault: str) -> None:
    if fault == 'prior-install':
        assert await review.run() == 0
    prior = review.output.read_bytes() if review.output.exists() else None
    link, stage = os.link, json_utils._stage_bytes
    injected: list[bool] = []
    def fail_link(source: Any, destination: Any, **kwargs: Any) -> None:
        if Path(destination) == review.output and not injected:
            injected.append(True)
            raise OSError('public install failed')
        link(source, destination, **kwargs)
    def fail_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        if b'"terminal_result"' in content:
            injected.append(True)
            raise OSError('findings staging failed')
        return stage(path, content, **kwargs)
    review.patch.setattr(os, 'link', fail_link)
    if fault == 'staging':
        review.patch.setattr(json_utils, '_stage_bytes', fail_stage)
    assert await review.run() == 1 and injected
    assert review.output.read_bytes() == prior if prior else not review.output.exists()


@pytest.mark.parametrize('fault', ['coverage', 'fix', 'salvage'])
async def test_finalization_boundary(review: ReviewRun, capsys: pytest.CaptureFixture[str], fault: str) -> None:
    error = RuntimeError('original provider error')
    stage, write, salvage = json_utils._stage_bytes, Path.write_text, findings._write_single_stack_merged_items
    state: dict[str, Any] = {'failed': False, 'merged': None, 'rejected': False, 'writes': []}
    def observe_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        if path.name == 'review-coverage.json':
            state['writes'].append((state['failed'], content))
            if fault == 'coverage':
                raise OSError('coverage staging failed')
        return stage(path, content, **kwargs)
    def observe_write(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        if path.name == 'merged-items.json':
            state['merged'] = path
        return write(path, text, *args, **kwargs)
    def reject_salvage(*args: Any, **kwargs: Any) -> None:
        if state['failed'] and fault == 'salvage':
            state['rejected'] = True
            raise ValueError('rejected projection')
        salvage(*args, **kwargs)
    def response(prompt: str) -> Any:
        trigger = {'coverage': 'present your understanding concisely', 'fix': 'fix this issue',
                   'salvage': 'supervisor adjudication'}[fault]
        if trigger in prompt.lower() or (fault == 'fix' and prompt.lower().startswith('fix these')):
            state['failed'] = True
            if fault == 'salvage':
                assert json.loads(state['merged'].read_text())['items']
                state['merged'].unlink()
            return [error]
    review.patch.setattr(json_utils, '_stage_bytes', observe_stage)
    review.patch.setattr(Path, 'write_text', observe_write)
    review.patch.setattr(findings, '_write_single_stack_merged_items', reject_salvage)
    review.backend = EmptyReviewBackend(review.repo, forbid_supervise=False,
                                       review_by_stack={'structure': [record()]}, responder=response)
    if fault == 'fix':
        assert await review.run(output_mode='loop', findings_out=None) == 1
        saved_path = review.repo / '.daydream/deep/review-coverage.json'
        assert state['failed'] and state['writes'] and all(not after for after, _ in state['writes'])
        assert saved_path.read_bytes() == state['writes'][-1][1]
        saved = json.loads(saved_path.read_text())
        assert all(s['status'] == 'complete' for s in saved['stack_outcomes'] + saved['phase_outcomes'])
        assert 'pipeline' not in saved['required_phases']
        return
    with pytest.raises(RuntimeError) as raised:
        await review.run()
    assert raised.value is error
    if fault == 'coverage':
        assert state['writes'] and not review.output.exists()
        assert 'Terminal review finalization failed: OSError' in collapse_panel_text(capsys)
        with pytest.raises(FindingsValidationError):
            review.load()
    else:
        data = review.load()
        result = data['terminal_result']
        assert state['rejected'] and data['findings'] == []
        assert (result['pipeline_state'], result['analysis_state'], result['projection_valid']) == (
            'failed', 'failed', False)
        assert all(s['status'] == 'complete' for s in scopes(data).values())
        phases = {p['phase']: p for p in result['phase_outcomes']}
        assert phases['supervision']['status'] == phases['findings']['status'] == 'failed'
        assert (phases['supervision']['reason_codes'], phases['findings']['reason_codes']) == (
            ['backend_failure'], ['malformed_artifact'])
