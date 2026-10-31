"""Snapshot-bound terminal contracts through the real runner and Git."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from rich.console import Console

from daydream import git_ops, json_utils, runner
from daydream.backends import MaxTurnsError, ResultEvent, TextEvent, ToolStartEvent
from daydream.dataset import LocalRecordStore
from daydream.deep import review_terminal
from daydream.findings import FindingsValidationError, load_findings_artifact
from daydream.phases.review import ReviewOutputError
from daydream.review_result import ReviewCoverage
from daydream.ui.messages import print_warning
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.console import collapse_panel_text
from tests.harness.dataset import read_records
from tests.harness.fake_clock import FakeClock
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git, init_repo
from tests.harness.scripts import cli_main
from tests.harness.stub_backend import review_stage_result, review_stage_state
from tests.test_deep_orchestrator import _pin_findings_pr, _profile_with_pipeline, _record
from tests.test_pr_review import _recording_fake_submit


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


@pytest.mark.parametrize('corrupt_capture', [False, True], ids=['intact', 'late-source-corruption'])
async def test_comment_pins_pr_head_independently_and_excludes_local_edits(
    review: ReviewRun, capsys: pytest.CaptureFixture[str],
    corrupt_capture: bool,
) -> None:
    """Pin PR bytes independently; late capture corruption cannot authorize clean approval.

    Capture-owner tests miss settlement after the final model response and posting.
    """
    from collections.abc import AsyncIterator

    from daydream.backends import AgentEvent
    from daydream.pr_review import ClassifiedReviewPlan, ReviewEvent

    committed = git_ops.show(review.repo, review.pr.head_sha, 'api.py').decode()
    (review.repo / 'api.py').write_text('LOCAL_COMMIT = True\n')
    git(review.repo, 'add', 'api.py')
    git(review.repo, 'commit', '-m', 'local change beyond PR')
    local_head = git_ops.head_sha(review.repo)
    (review.repo / 'api.py').write_text('LOCAL_INDEX = True\n')
    git(review.repo, 'add', 'api.py')
    (review.repo / 'api.py').write_text('LOCAL_WORKING = True\n')
    (review.repo / 'untracked.py').write_text('LOCAL_UNTRACKED = True\n')
    before = (review.repo / '.git/index').read_bytes()
    plans: dict[str, ClassifiedReviewPlan] = {}
    captured_roots: list[Path] = []

    class CommittedReader(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            assert cwd != review.repo and (cwd / 'api.py').read_text() == committed
            assert not (cwd / 'untracked.py').exists()
            captured_roots.append(cwd)
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = CommittedReader(review.repo)
    if corrupt_capture:
        finalize = review_terminal.finalize_review

        def corrupt_before_finalization(ctx: Any, *args: Any, **kwargs: Any) -> int:
            (ctx.work.repo / 'api.py').unlink()
            return finalize(ctx, *args, **kwargs)

        review.patch.setattr(review_terminal, 'finalize_review', corrupt_before_finalization)
    review.patch.setattr('daydream.pr_review.post_classified_review', _recording_fake_submit(plans))
    assert await review.run(output_mode='comment', findings_out=None, approve_on_clean=True) == 0
    plan = plans['plan']
    assert plan.pr.head_sha == review.pr.head_sha != local_head
    assert plan.pr.base_sha == review.pr.base_sha and plan.pr.pr_base_sha == review.pr.base_sha
    assert plan.event is (ReviewEvent.COMMENT if corrupt_capture else ReviewEvent.APPROVE)
    coverage = ReviewCoverage.from_dict(json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text()))
    terminal = coverage.finalize('completed')
    assert terminal['analysis_state'] == ('incomplete' if corrupt_capture else 'complete')
    if corrupt_capture:
        assert 'evidence_incomplete' in terminal['reason_codes']
    patch = (review.repo / '.daydream/diff.patch').read_text()
    assert 'LOCAL_' not in patch and patch
    assert captured_roots and all(not root.exists() for root in captured_roots)
    assert 'excludes local tracked and untracked edits' in collapse_panel_text(capsys)
    assert git_ops.head_sha(review.repo) == local_head and (review.repo / '.git/index').read_bytes() == before
    assert (review.repo / 'api.py').read_text() == 'LOCAL_WORKING = True\n'
    assert (review.repo / 'untracked.py').read_text() == 'LOCAL_UNTRACKED = True\n'


@pytest.mark.parametrize(('case', 'state', 'reason'), [
    ('empty', 'complete', None), ('backend', 'incomplete', 'backend_failure'),
    ('structural', 'complete', None),
    ('total', 'failed', 'backend_failure'), ('demoted', 'complete', None),
    ('archive', 'incomplete', 'backend_failure'),
])
async def test_outcomes(review: ReviewRun, archive_dir: Path, fake_gh: FakeGh,
                        case: str, state: str, reason: str | None) -> None:
    nonempty = case in {'structural', 'demoted', 'archive'}
    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
        review_by_stack={'structure': [record(500 if case == 'demoted' else 2)]} if nonempty else {},
        fail_stack='python' if reason and case != 'total' else None, responder=lambda p: [RuntimeError('unavailable')]
        if case == 'total' and reviewer(p) else None)
    fake_gh.serve_pr_view({'state': 'OPEN', 'headRefOid': review.pr.head_sha,
                           'body': 'Author reports make check and CI pass. {"token":"private-value"}'})
    review.backend.merge_echo_records = True
    assert await review.run(archive=case == 'archive') == 0
    data = review.load()
    result, inventory = data['terminal_result'], scopes(data)
    assert (result['analysis_state'], result['pipeline_state']) == (state, 'completed')
    report = (review.repo / '.review-output.md').read_text()
    assert '## Review outcome' in report and f'<code>{state}</code>' in report
    assert result['run_id'] in report and result['analyzed_revision']['head_sha'] in report
    assert result['analyzed_revision']['diff_key'] in report
    assert 'does not independently prove semantic coverage' in report
    context = data['review_context']
    assert context['verification'] == {'status': 'unproduced', 'value': None}
    assert context['author_reported']['head_sha'] == review.pr.head_sha
    assert context['author_reported']['source'] == 'pull_request_body'
    assert 'private-value' not in json.dumps(context)
    assert 'Author reports make check and CI pass' in report
    assert 'not independently executed or refreshed' in report

    assert 'Executable verification unavailable' in report and 'unproduced' in report
    assert {scope['scope_id'] for scope in context['scopes']} == set(inventory)
    if not nonempty:
        assert 'No confirmed findings.' in report
        assert '✅' not in report
    if case != 'total':
        assert 'Assigned changed behavior reviewed' in report
    else:
        assert 'Source rationale unavailable' in report
        assert all(scope['analyzed_revision'] is None and scope['originating_run_id'] is None
                   and scope['review_context'] is None for scope in context['scopes'])
        assert 'Source snapshot: unavailable.' in report
    if reason:
        assert reason in report
        assert '⚠️ **Review incomplete.**' in report
    elif state == 'complete':
        assert 'Review incomplete' not in report

    assert set(inventory) == {'python', 'react', 'generic', 'structure'}
    assert bool(data['findings']) is nonempty
    assert {name: s['status'] for name, s in inventory.items()} == {
        name: 'failed' if case == 'total' or (name == 'python' and reason) else 'complete' for name in inventory}
    if reason:
        assert inventory['python']['reason_codes'] == [reason]
    else:
        assert result['reason_codes'] == []
    if nonempty:
        assert len(data['findings']) == 1 and data['findings'][0]['title'] == 'Grounded defect'
    if case == 'demoted':
        assert (data['findings'][0]['confidence'], data['findings'][0]['location_distrust']) == ('LOW', True)
        canonical = json.loads((review.repo / '.daydream/deep/merged-items.json').read_text())['items'][0]
        assert (canonical['line'], canonical['location_cited_line'], canonical['severity'],
                canonical['severity_before_demotion'], canonical['location_distrust']) == (
                    500, 500, 'low', 'medium', True)
        assert 'cited line 500 in api.py' in canonical['location_note']
        assert 'demoted to low' in canonical['location_note']
    if case in {'empty', 'backend', 'total'}:
        fake_gh.serve_prior_threads(fingerprints=[], thread_ids=[], viewer_did_author=True)
        assert cli_main(['post-findings', str(review.output), '--pr', '7', '--head-sha', review.pr.head_sha,
                         '--repo', 'o/r', '--target', str(review.repo)]) == 0
        posts = fake_gh.calls('POST', '/repos/o/r/pulls/7/reviews')
        if case == 'empty':
            assert posts == [], 'report context must not create publishing authority'
        else:
            assert len(posts) == 1 and posts[0].payload['event'] == 'COMMENT'
            body = posts[0].payload['body']
            assert body.startswith('Actionable comments posted: 0\n\n')
            assert f'⚠️ **Review {state}.**' in body and 'python:' in body
            assert f"Run:\n```text\n{result['run_id']}\n```" in body
            assert all(detail not in body for detail in (
                'Review outcome', 'No confirmed findings.', 'Executable verification unavailable',
                result['analyzed_revision']['diff_key'],
            ))
    deep = review.repo / '.daydream/deep'
    merge_calls = sum('cross-stack merge agent' in c['prompt'].lower() for c in review.backend.calls)
    assert merge_calls == (1 if case == 'total' else 0)
    if case in {'empty', 'structural'}:
        assert (deep / 'review-output.md').read_bytes() == (review.repo / '.review-output.md').read_bytes()
        assert not (deep / 'merge-failed.txt').exists()
        assert json.loads((deep / 'review-coverage.json').read_text())['stack_outcomes'] == result['stack_outcomes']
    if case == 'structural':
        items = json.loads((deep / 'merged-items.json').read_text())['items']
        assert [(i['description'], i['lens'], i['file'], i['line'], i['source_uids']) for i in items] == [
            ('Grounded defect', 'structural', 'api.py', 2, ['structure:1'])]
        records = json.loads((deep / 'stack-structure-records.json').read_text())['issues']
        assert records[0]['uid'] == 'structure:1' and 'Grounded defect' in report
        assert sum('supervisor adjudication' in c['prompt'].lower() for c in review.backend.calls) == 1
    if case == 'empty':
        assert json.loads((deep / 'merged-items.json').read_text()) == {'items': [], 'held': []}
        events = json.loads((review.tmp / 'trajectory.json').read_text())['extra']['phase_events']
        for phase, stage in (('merge', 'cross-stack-agent'), ('deep', 'supervise')):
            lifecycle = [event for event in events if event['phase'] == phase
                         and event.get('metadata', {}).get('stage') == stage]
            assert [event['event'] for event in lifecycle] == ['phase_start', 'phase_end']
            assert lifecycle[-1]['status'] == 'succeeded'
        stages = [stage_state for c in review.backend.calls
                  if (stage_state := review_stage_state(c['prompt'])) is not None]
        assert {stage['scope_id'] for stage in stages} == set(inventory)
        assert [stage['stage'] for stage in stages if stage['scope_id'] == 'structure'] == ['integration']
        assert not any('supervisor adjudication' in c['prompt'].lower() for c in review.backend.calls)
    if case == 'archive':
        archived = archive_dir / 'runs' / result['run_id']
        assert json.loads((archived / 'manifest.json').read_text())['archive_status'] == 'complete'
        assert json.loads((archived / 'findings.json').read_text()) == data
        paths = list(archived.rglob('review-coverage.json'))
        assert len(paths) == 1
        saved = json.loads(paths[0].read_text())
        assert ReviewCoverage.from_dict(saved).finalize('completed') == result
        assert json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text()) == saved


async def test_automatic_shard_failure_preserves_successful_sibling_findings_and_revision(
    sibling_frontier_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review = ReviewRun(sibling_frontier_target, tmp_path, monkeypatch)
    finding = dict(record(1), file='core.py', evidence='core.py:1 defines core_helper')
    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
        review_by_stack={'python#0': [finding]}, fail_stack='python#1',
        responder=lambda p: (
            [AuthError('rejected')] if (stage := review_stage_state(p)) and stage['scope_id'] == 'python#2'
            else [MaxTurnsError('spent turns')] if stage and stage['scope_id'] == 'structure' else None))
    review.backend.merge_echo_records = True
    assert await review.run(review_cache_enabled=True) == 0
    data = review.load()
    result, inventory = data['terminal_result'], scopes(data)
    assert list(inventory) == ['python#0', 'python#1', 'python#2', 'structure']
    assert (result['analysis_state'], result['pipeline_state']) == ('incomplete', 'completed')
    assert {name: (scope['status'], scope['reason_codes']) for name, scope in inventory.items()} == {
        'python#0': ('complete', []), 'python#1': ('failed', ['backend_failure']),
        'python#2': ('failed', ['authentication_failure']), 'structure': ('incomplete', ['model_budget_exhaustion'])}
    assert not any(inventory[name]['partial_evidence'] for name in ('python#1', 'python#2', 'structure'))
    assert [item['title'] for item in data['findings']] == ['Grounded defect']
    deep = review.repo / '.daydream/deep'
    persisted = json.loads((deep / 'review-coverage.json').read_text())
    assert persisted['stack_outcomes'] == result['stack_outcomes']
    assert persisted['phase_outcomes'] == result['phase_outcomes']
    assert persisted['analyzed_revision'] == result['analyzed_revision']
    assert [scope['scope_id'] for scope in persisted['planned_scopes']] == list(inventory)
    folded = next(p for p in result['phase_outcomes'] if p['phase'] == 'alternatives')
    assert (folded['status'], folded['reason_codes'], folded['noop'], folded['usable_evidence']) == (
        'incomplete', ['model_budget_exhaustion'], False, False)
    records = json.loads((deep / 'stack-python#0-records.json').read_text())
    assert records['scope_id'] == 'python#0' and records['analyzed_revision'] == result['analyzed_revision']
    assert records['issues'][0]['uid'] == 'python#0:1'
    structural = json.loads((deep / 'stack-structure-records.json').read_text())
    assert structural['issues'] == [] and structural['incomplete'] is True
    assert structural['scope_id'] == 'structure' and structural['analyzed_revision'] == result['analyzed_revision']
    assert {path.name for path in deep.glob('stack-*-records.json')} == {
        'stack-python#0-records.json', 'stack-structure-records.json'}
    units = {json.loads(path.read_text())['unit']
             for path in (deep.parent / 'review-cache/entries').glob('*/manifest.json')}
    assert units == set(), 'Admitted sibling evidence cannot become reusable review authority.'


@pytest.mark.parametrize('case', ['live', 'dirty', 'no-diff', 'dirty-no-diff', 'interactive',
                                  'base-tip', 'shards', 'pipeline-budget'])
async def test_snapshot_boundaries(review: ReviewRun, request: pytest.FixtureRequest, case: str) -> None:
    overrides: dict[str, Any] = {}
    store = LocalRecordStore(review.tmp / 'records')
    if case == 'no-diff':
        overrides.update(dataset_capture=True, dataset_store_path=store.root)
    prior = {'.review-output.md': 'prior completed review\n', '.daydream/deep/history.json': '{"prior": true}\n',
             '.daydream/deep/merged-items.json': '{"items": [{"item_uid": "item:old"}]}\n',
             '.daydream/recommended.patch': 'prior recommended patch\n'}
    if case in {'no-diff', 'dirty-no-diff'}:
        for name, text in prior.items():
            path = review.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (review.repo / '.review-output.md').write_text(
            '# Review\n\n## Issues\n1. [obsolete.py:9] PRIOR_FINDING\n\n'
            '## Held Findings\n2. [obsolete.py:10] PRIOR_HELD\n\n'
            '## Diagrams\nPRIOR_DIAGRAM\n')
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
    if case in {'live', 'base-tip'}:
        def lookup(*_a: Any, **_k: Any) -> Any:
            calls.append(live)
            return live
        review.patch.setattr('daydream.pr_review.find_pr_by_number', lookup)
        def advance(_p: str) -> None:
            nonlocal live
            live = replace(review.pr, head_sha=review.pr.base_sha)
        if case == 'live':
            review.backend = EmptyReviewBackend(review.repo, forbid_supervise=False,
                review_by_stack={'structure': [record()]}, responder=advance)
    if case == 'no-diff':
        assert await review.run(dataset_store_path=store.root) == 0 and not store.root.exists()
    assert await review.run(**overrides) == 0
    if case == 'no-diff':
        raw = read_records(store).runs[0]
        assert raw['outcome'] == 'success'
        assert raw['original_task']['value']['diff'] == '' and raw['findings']['value']['items'] == []
        assert raw['scoring']['status'] == 'available'
        assert 'No confirmed findings.' in raw['scoring']['value']['review_text']
        assert raw['verification']['status'] == 'unproduced'
    if case == 'interactive':
        assert not review.output.exists()
        assert review.backend.calls
        assert 'DIRTY = True' in (review.repo / '.daydream/diff.patch').read_text()
        return
    data = review.load()
    result = data['terminal_result']
    failed = case == 'pipeline-budget'
    assert result['analysis_state'] == ('failed' if failed else 'complete')
    if case in {'dirty-no-diff', 'no-diff', 'pipeline-budget'}:
        assert review.backend.calls == []
    if case.startswith('dirty'):
        assert (review.repo / 'api.py').read_text() == 'DIRTY = True\n'
        assert 'DIRTY = True' not in (review.repo / '.daydream/diff.patch').read_text()
    if case == 'pipeline-budget':
        assert 'host_pipeline_budget_exhaustion' in result['reason_codes']
        assert all(s['status'] == 'uncovered' for s in scopes(data).values())
        assert data['findings'] == [] and data['review_warnings']
        assert 'Review incomplete' in (review.repo / '.review-output.md').read_text()
    if case == 'no-diff':
        assert result['stack_outcomes'] == [] and result['phase_outcomes'][0]['noop'] is True and result['run_id']
    if case in {'no-diff', 'dirty-no-diff'}:
        report = (review.repo / '.review-output.md').read_text()
        assert all(marker not in report for marker in ('PRIOR_FINDING', 'PRIOR_HELD', 'PRIOR_DIAGRAM'))
        assert 'No confirmed findings.' in report and result['run_id'] in report
        assert (review.repo / '.daydream/deep/review-output.md').read_text() == report
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


@pytest.mark.parametrize(
    ("budget", "nonempty", "structure_only"),
    [
        *[(budget, nonempty, False) for budget in ("tool", "wall", "model") for nonempty in (True, None)],
        *[(budget, True, True) for budget in ("tool", "backend")],
    ],
)
async def test_unsuccessful_stage_discards_checkpoints(
    review: ReviewRun,
    budget: str,
    nonempty: bool | None,
    structure_only: bool,
) -> None:
    from tests.deep_orchestrator.test_review_capture_and_retry import supporting_contents

    source = git(review.repo, 'diff', 'HEAD')
    fake = FakeClock().install(review.patch)
    intent = 'PRESERVE_AUTHOR_INTENT'
    review.patch.setattr(git_ops, 'gh_pr_view', lambda *_a, **_k: {
        'body': intent, 'state': 'OPEN', 'headRefOid': review.pr.head_sha})

    def response(prompt: str) -> Any:
        if 'understand the intent of these changes' in prompt.lower():
            yield TextEvent(text=intent)
            yield ResultEvent(structured_output=None, continuation=None)
            return
        if reviewer(prompt):
            state = review_stage_state(prompt)
            assert state is not None
            if state['scope_id'] == 'structure':
                supporting = supporting_contents(prompt)
                assert state['stage'] == 'integration' and 'review-assignment' in supporting
                assert supporting['intent'] == intent and intent in prompt and 'AUTHORITATIVE' in prompt
                assert 'api.py' in prompt and 'Reviewer dispatch-time snapshot:' in prompt
                assert str(review.repo / '.daydream/diff.patch') not in prompt
            if nonempty is not None:
                yield ResultEvent(structured_output=review_stage_result(prompt, [record()] if nonempty else []),
                                  continuation=None)
            if budget == 'backend':
                yield RuntimeError('structural provider unavailable')
            elif budget == 'model':
                yield MaxTurnsError('spent turns')
            else:
                for index in range(state['remaining_tool_calls'] + 1):
                    if budget == 'wall':
                        fake.advance(601)
                    yield ToolStartEvent(id=f'budget-{index}', name='Read', input={'file_path': 'api.py'})

    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
        review_by_stack={'python': [record()]} if structure_only else {},
        responder=lambda p: response(p) if (reviewer(p) and (not structure_only or
            'you are the structural reviewer' in p.lower())) or 'understand the intent' in p.lower() else None)
    review.backend.merge_echo_records = True
    if budget == 'wall':
        # One fake clock is shared by every scope, so siblings must not overlap and charge each other's budgets.
        review.backend.fanout_concurrency = 1
    assert await review.run(review_profile=_profile_with_pipeline(review_wall_budget_s=99999)) == 0
    data = review.load()
    result = data['terminal_result']
    reason = ('backend_failure' if budget == 'backend' else 'model_budget_exhaustion'
              if budget == 'model' else f'host_{budget}_budget_exhaustion')
    assert result['analysis_state'] == ('incomplete' if structure_only else 'failed')
    assert result['completed_stacks'] == (['generic', 'python', 'react'] if structure_only else [])
    assert reason in result['reason_codes']
    assert [f['title'] for f in data['findings']] == (['Grounded defect'] if structure_only else [])
    stages = [state for call in review.backend.calls if (state := review_stage_state(call['prompt'])) is not None]
    assert [stage['stage'] for stage in stages if stage['scope_id'] == 'structure'] == ['integration']
    assert not any(call['prompt'].startswith('INVESTIGATION HAS ENDED.') for call in review.backend.calls)
    assert {stage['scope_id'] for stage in stages} == {'python', 'react', 'generic', 'structure'}
    assert not any('would you have done this differently' in call['prompt'].lower()
                   or 'evaluate the implementation' in call['prompt'].lower() for call in review.backend.calls)
    assert git(review.repo, 'diff', 'HEAD') == source
    folded = next(p for p in result['phase_outcomes'] if p['phase'] == 'alternatives')
    assert (folded['status'], folded['reason_codes'], folded['noop'], folded['usable_evidence']) == (
        'failed' if budget == 'backend' else 'incomplete', [reason], False, False)
    persisted = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    assert persisted['phase_outcomes'] == result['phase_outcomes']
    assert persisted['stack_outcomes'] == result['stack_outcomes']
    assert 'folded into Structure' in persisted['diagnostics']['phases']['alternatives']
    for name, scope in scopes(data).items():
        if structure_only and name != 'structure':
            assert scope['status'] == 'complete'
            continue
        records_path = review.repo / f'.daydream/deep/stack-{name}-records.json'
        if budget == 'backend':
            assert not records_path.exists()
        else:
            saved = json.loads(records_path.read_text())
            assert saved['issues'] == [] and saved['incomplete'] is True
        assert (scope['status'], scope['partial_evidence']) == (
            'failed' if budget == 'backend' else 'incomplete', False)
        assert scope['reason_codes'] == [reason]


@pytest.mark.parametrize(('fault', 'owner'), [
    *[(fault, 'python') for fault in ('corrupt', 'shape', 'revision', 'scope', 'origin')],
    *[(fault, 'structure') for fault in ('missing', 'shape', 'duplicate')],
])
async def test_loaded_artifact_faults(review: ReviewRun, fault: str, owner: str) -> None:
    source = git(review.repo, 'diff', 'HEAD')
    artifact_name = f'stack-{owner}-records.json'
    write, read = Path.write_text, Path.read_text
    malformed = '{invalid' if fault == 'corrupt' else '{"issues": "invalid"}'
    def faulty_write(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        if path.name == artifact_name and fault in {'corrupt', 'shape'}:
            text = malformed
        if path.name == artifact_name and fault == 'duplicate':
            payload = json.loads(text)
            payload['issues'].append(dict(payload['issues'][0]))
            text = json.dumps(payload)
        count = write(path, text, *args, **kwargs)
        if path.name == artifact_name and fault == 'missing':
            path.unlink()
        return count
    def faulty_read(path: Path, *args: Any, **kwargs: Any) -> str:
        text = read(path, *args, **kwargs)
        if path.name == artifact_name and fault in {'revision', 'scope', 'origin'}:
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
    if fault in {'missing', 'corrupt', 'shape', 'duplicate'}:
        review.backend.review_by_stack = {'react': [_record(
            description='Surviving sibling defect', file='App.tsx', line=1)]}
    if fault == 'duplicate':
        review.backend.review_by_stack['structure'] = [record()]
    assert await review.run(dataset_capture=True, dataset_store_path=review.tmp / 'records') == 1
    if fault in {'corrupt', 'shape'}:
        raw = read_records(LocalRecordStore(review.tmp / 'records')).runs[0]
        assert raw['outcome'] == 'failed' and raw['completeness']['artifact_acquisition'] == 'failed'
        diagnostic = raw['provenance']['collection_diagnostics'][artifact_name]
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
        'failed', 'incomplete' if fault != 'duplicate' else 'failed', fault != 'duplicate')
    assert set(inventory) == {'python', 'react', 'generic', 'structure'} and reason in result['reason_codes']
    assert {n: s['status'] for n, s in inventory.items()} == dict.fromkeys(inventory, 'complete') | {owner: 'failed'}
    assert inventory[owner]['reason_codes'] == [reason]
    if fault in {'missing', 'corrupt', 'shape'}:
        assert [(f['title'], f['placement']) for f in data['findings']] == [('Surviving sibling defect', 'inline')]
    else:
        assert data['findings'] == []
    assert not any('cross-stack merge agent' in c['prompt'].lower() for c in review.backend.calls)
    assert git(review.repo, 'diff', 'HEAD') == source
    if owner == 'structure':
        folded = next(p for p in result['phase_outcomes'] if p['phase'] == 'alternatives')
        assert (folded['status'], folded['reason_codes'], folded['noop'], folded['usable_evidence']) == (
            'failed', [reason], False, False)
        persisted = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
        assert persisted['phase_outcomes'] == result['phase_outcomes']
        assert persisted['stack_outcomes'] == result['stack_outcomes']
        assert 'folded into Structure' in persisted['diagnostics']['phases']['alternatives']
        assert not any('would you have done this differently' in c['prompt'].lower()
                       or 'evaluate the implementation' in c['prompt'].lower() for c in review.backend.calls)


@pytest.mark.parametrize(('phase', 'reason'), [('intent', 'backend_failure'), ('intent-empty', 'missing_output'),
    ('alternatives', 'backend_failure'), ('merge', 'synthesis_failure'), ('merge-low', 'synthesis_failure'),
    ('missing', 'missing_output'), ('malformed', 'malformed_output'),
    ('omitted', 'evidence_incomplete')])
async def test_required_phase_faults(review: ReviewRun, phase: str, reason: str) -> None:
    from tests.harness.review_profile import independent_alternatives_profile
    from tests.harness.review_result import merge_result
    intent_fault = phase.startswith('intent')
    def response(prompt: str) -> Any:
        lower = prompt.lower()
        if phase == 'intent-empty' and 'present your understanding concisely' in lower:
            return [ResultEvent(structured_output=None, continuation=None)]
        if (phase == 'intent' and 'present your understanding concisely' in lower) or (
                phase == 'alternatives' and 'evaluate the implementation' in lower):
            return [RuntimeError(f'{phase} unavailable')]
        if phase in {'missing', 'malformed', 'omitted'} and 'supervisor adjudication' in lower:
            payload = None if phase == 'missing' else {'verdicts': [None] if phase == 'malformed' else []}
            return [ResultEvent(structured_output=payload, continuation=None)]
        if phase == 'merge-low' and 'cross-stack merge agent' in lower:
            return [ResultEvent(structured_output=merge_result([dict(record(), confidence='LOW', lens='per-stack'),
            ]), continuation=None)]
    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
        review_by_stack={'python' if phase in {'merge', 'merge-low'} else 'structure': [record()]}, responder=response)
    review.backend.merge_echo_records = True
    review.backend.merge_emit_str = 'invalid merge output' if phase == 'merge' else None
    if intent_fault or phase in {'alternatives', 'merge', 'merge-low'}:
        review.config = replace(review.config, review_profile=independent_alternatives_profile())
    if phase in {'merge', 'merge-low', 'omitted'}:
        assert await review.run() == (1 if phase in {'merge', 'merge-low'} else 0)
    else:
        error_type = ReviewOutputError if phase in {'intent-empty', 'missing', 'malformed'} else RuntimeError
        with pytest.raises(error_type, match=None if error_type is ReviewOutputError else
                           f'{phase} unavailable') as raised:
            await review.run()
        if phase == 'intent-empty':
            assert cast(ReviewOutputError, raised.value).reason_code == 'missing_output'
    data = review.load()
    result = data['terminal_result']
    assert (result['pipeline_state'], result['analysis_state']) == (
        'completed' if phase == 'omitted' else 'failed',
        'failed' if intent_fault or phase == 'alternatives' else 'incomplete')
    assert reason in result['reason_codes']
    if phase in {'merge', 'merge-low'}:
        outcome = next(p for p in result['phase_outcomes'] if p['phase'] == 'merge')
        assert outcome['status'] == 'failed'
        assert 'synthesis_failure' in outcome['reason_codes']
    expected_titles = [] if intent_fault or phase == 'alternatives' else ['Grounded defect']
    assert [f['title'] for f in data['findings']] == expected_titles
    if phase in {'missing', 'malformed', 'omitted'}:
        outcome = next(p for p in result['phase_outcomes'] if p['phase'] == 'supervision')
        expected = 'incomplete' if phase == 'omitted' else 'failed'
        assert (outcome['status'], outcome['reason_codes']) == (expected, [reason])
        assert any('supervisor adjudication' in c['prompt'].lower() for c in review.backend.calls)


@pytest.mark.parametrize(('late_event', 'offset'), [
    ('legacy-result', 0), ('native-start', 1)], ids=['at-deadline-legacy-result', 'after-deadline-native-start'])
async def test_late_supervision_events_cannot_supply_reserved_finalization_or_replace_prior_findings(
    review: ReviewRun, ext_dir: ExtDir, late_event: str, offset: int) -> None:
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
            assert 'hidden evaluation' not in prompt
            assert 'planted' not in prompt
            assert 'an empty findings result can be successful when substantiated' in prompt
            assert 'Do not call tools or obtain new evidence' in prompt
            yield ResultEvent(structured_output={'verdicts': []}, continuation=None)
        elif 'supervisor adjudication' in prompt.lower():
            assert 'hidden evaluation' not in prompt
            assert 'planted' not in prompt
            assert 'Reopen a rejected candidate only when new evidence' in prompt
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
    output_parent = review.output.parent.stat()
    link, stage, allocate = os.link, json_utils._stage_bytes, tempfile.mkstemp
    injected: list[bool] = []
    staging_findings = False
    def fail_link(source: Any, destination: Any, **kwargs: Any) -> None:
        parent_fd = kwargs.get('dst_dir_fd')
        anchored_output = False
        if parent_fd is not None and Path(destination) == Path(review.output.name):
            parent = os.fstat(parent_fd)
            anchored_output = (parent.st_dev, parent.st_ino) == (output_parent.st_dev, output_parent.st_ino)
        if (Path(destination) == review.output or anchored_output) and not injected:
            injected.append(True)
            raise OSError('public install failed')
        link(source, destination, **kwargs)
    def observe_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        nonlocal staging_findings
        staging_findings = b'"terminal_result"' in content
        try:
            return stage(path, content, **kwargs)
        finally:
            staging_findings = False
    def fail_allocation(*args: Any, **kwargs: Any) -> tuple[int, str]:
        if staging_findings:
            injected.append(True)
            raise OSError('findings staging failed')
        return allocate(*args, **kwargs)
    review.patch.setattr(os, 'link', fail_link)
    if fault == 'staging':
        review.patch.setattr(json_utils, '_stage_bytes', observe_stage)
        review.patch.setattr(tempfile, 'mkstemp', fail_allocation)
    assert await review.run() == 1 and injected
    assert review.output.read_bytes() == prior if prior else not review.output.exists()


@pytest.mark.parametrize(('export', 'fault'), [
    (True, 'markdown-read'), (True, 'context-write'), (True, 'report-write'), (False, 'report-write')],
    ids=['export-markdown-read', 'export-context-write', 'export-report-write', 'no-export-report-write'])
async def test_optional_presentation_failure_preserves_canonical_export(
    review: ReviewRun, fault: str, export: bool,
) -> None:
    warning_console = Console(width=120)
    review.patch.setattr(review_terminal, 'console', warning_console)
    review.patch.setattr(review_terminal, 'print_warning', print_warning)
    stage, read = json_utils._stage_bytes, Path.read_text
    refreshing = False
    injected: list[bool] = []
    def faulty_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        nonlocal refreshing
        if path.name == 'review-report-context.json':
            refreshing = True
            if fault == 'context-write':
                injected.append(True)
                raise OSError('optional context staging failed')
        if refreshing and fault == 'report-write' and path.name in {'review-output.md', '.review-output.md'}:
            injected.append(True)
            raise OSError('optional report staging failed')
        return stage(path, content, **kwargs)
    def faulty_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if refreshing and fault == 'markdown-read' and path.name == 'review-output.md':
            injected.append(True)
            raise PermissionError('optional Markdown unavailable')
        return read(path, *args, **kwargs)
    review.patch.setattr(json_utils, '_stage_bytes', faulty_stage)
    review.patch.setattr(Path, 'read_text', faulty_read)
    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
                                       review_by_stack={'structure': [record()]})
    review.backend.merge_echo_records = True
    with warning_console.capture() as warning_output:
        assert await review.run(findings_out=str(review.output) if export else None) == 0 and injected
    error = 'PermissionError' if fault == 'markdown-read' else 'OSError'
    assert f'Review report unavailable: {error}' in warning_output.get()
    if not export:
        saved = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
        assert saved['run_id'] and saved['analyzed_revision']['head_sha'] == review.pr.head_sha
        assert all(scope['status'] == 'complete' for scope in saved['stack_outcomes'])
        canonical = json.loads((review.repo / '.daydream/deep/merged-items.json').read_text())
        assert len(canonical['items']) == 1 and canonical['items'][0]['description'] == 'Grounded defect'
        assert not review.output.exists()
        return
    data = review.load()
    assert len(data['findings']) == 1 and data['findings'][0]['title'] == 'Grounded defect'
    assert data['terminal_result']['analysis_state'] == 'complete'
    assert data['terminal_result']['projection_valid']
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    assert 'review_context' not in data


@pytest.mark.parametrize('fault', ['coverage', 'fix', 'salvage', 'presentation', 'source-loss', 'source-loss-salvage',
                                  'source-loss-worktree', 'source-loss-placement-worktree',
                                  'placement-git-error-worktree', 'placement-replacement-worktree'])
async def test_finalization_boundary(review: ReviewRun, capsys: pytest.CaptureFixture[str], fault: str) -> None:
    force_worktree = fault.endswith('-worktree')
    if force_worktree:
        git(review.repo, 'remote', 'add', 'origin', str(bare_remote(review.tmp / 'origin.git')))
        git(review.repo, 'push', 'origin', 'main')
    error = RuntimeError('original provider error')
    stage, write, read, allocate = json_utils._stage_bytes, Path.write_text, Path.read_text, tempfile.mkstemp
    show = git_ops.show
    state: dict[str, Any] = {'failed': False, 'merged': None, 'rejected': False, 'writes': [], 'export': None}
    staging_coverage = False
    real_run = subprocess.run
    placement_fault = 'placement' in fault
    def fault_at_placement(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if (placement_fault and state['failed']
                and args == ['git', 'diff', '--name-only', '-z', review.pr.base_sha, review.pr.head_sha]):
            assert not state.get('dispatch_injected')
            state['dispatch_injected'] = True
            if fault == 'placement-git-error-worktree':
                oid = review.pr.base_sha
                captured_root = Path(kwargs['cwd'])
                assert captured_root != review.repo
                (captured_root / '.git' / 'objects' / oid[:2] / oid[2:]).rename(review.tmp / 'removed-commit-object')
            else:
                review.repo.rename(review.tmp / 'removed-source')
                if fault == 'placement-replacement-worktree':
                    init_repo(review.repo)
                    (review.repo / 'replacement.bin').write_bytes(b'replacement\x00\xff')
                    from tests.test_artifact_visibility import _walk_entry
                    state['replacement_git'] = []
                    _walk_entry(review.repo, review.repo / '.git', state['replacement_git'])
        return real_run(args, **kwargs)
    def observe_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        nonlocal staging_coverage
        if fault == 'presentation' and state['failed'] and path.name == 'review-report-context.json':
            state['rejected'] = True
            raise OSError('optional context staging failed after provider error')
        staging_coverage = path.name == 'review-coverage.json'
        if path.name == 'review-coverage.json':
            state['writes'].append((state['failed'], content))
        if b'"terminal_result"' in content:
            state['export'] = (path, content)
        try:
            return stage(path, content, **kwargs)
        finally:
            staging_coverage = False
    def fail_allocation(*args: Any, **kwargs: Any) -> tuple[int, str]:
        if fault == 'coverage' and staging_coverage:
            raise OSError('coverage staging failed')
        return allocate(*args, **kwargs)
    def observe_write(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        if path.name == 'merged-items.json':
            state['merged'] = path
        return write(path, text, *args, **kwargs)
    def faulty_read(path: Path, *args: Any, **kwargs: Any) -> str:
        text = read(path, *args, **kwargs)
        if state['failed'] and fault == 'salvage' and path.name == 'stack-structure-records.json':
            state['rejected'] = True
            return '{"issues": "invalid"}'
        return text
    def retained_source_only(target: Path, *args: Any, **kwargs: Any) -> bytes:
        if target == review.repo and fault.startswith('source-loss') and state['failed']:
            pytest.fail('terminal export attempted to read live Git after source loss')
        return show(target, *args, **kwargs)
    def response(prompt: str) -> Any:
        trigger = {'coverage': 'present your understanding concisely', 'fix': 'fix this issue'}.get(
            fault, 'supervisor adjudication')
        if trigger in prompt.lower() or (fault == 'fix' and prompt.lower().startswith('fix these')):
            state['failed'] = True
            if fault == 'salvage':
                assert json.loads(state['merged'].read_text())['items']
                state['merged'].unlink()
            if fault.startswith('source-loss'):
                state['admitted'] = json.loads(state['merged'].read_text())['items'][0]
                if fault == 'source-loss-salvage':
                    state['merged'].unlink()
                if fault != 'source-loss-placement-worktree':
                    review.repo.rename(review.tmp / 'removed-source')
            return [error]
    review.patch.setattr(json_utils, '_stage_bytes', observe_stage)
    review.patch.setattr(tempfile, 'mkstemp', fail_allocation)
    review.patch.setattr(Path, 'write_text', observe_write)
    review.patch.setattr(Path, 'read_text', faulty_read)
    review.patch.setattr(git_ops, 'show', retained_source_only)
    review.patch.setattr('daydream.git_ops.process.subprocess.run', fault_at_placement)
    stack = 'python' if force_worktree or fault == 'source-loss' else 'structure'
    review.backend = EmptyReviewBackend(review.repo, forbid_supervise=False, forbid_merge=stack != 'python',
                                       review_by_stack={stack: [record()]}, responder=response)
    review.backend.merge_echo_records = True
    if fault == 'fix':
        assert await review.run(output_mode='loop', findings_out=None, test_command=f'{sys.executable} -c pass') == 1
        saved_path = review.repo / '.daydream/deep/review-coverage.json'
        assert state['failed'] and state['writes'] and all(not after for after, _ in state['writes'])
        assert saved_path.read_bytes() == state['writes'][-1][1]
        saved = json.loads(saved_path.read_text())
        assert all(s['status'] == 'complete' for s in saved['stack_outcomes'] + saved['phase_outcomes'])
        assert 'pipeline' not in saved['required_phases']
        return
    with pytest.raises(RuntimeError) as raised:
        await review.run(dataset_capture=fault.startswith('source-loss'),
                         dataset_store_path=review.tmp / 'records', force_worktree=force_worktree)
    assert raised.value is error
    if placement_fault:
        assert state.get('dispatch_injected')
    if fault.startswith('placement-'):
        assert not review.output.exists()
        if fault == 'placement-git-error-worktree':
            assert state['export'] is None
        else:
            assert state['export'] is not None
            private_path, private_bytes = state['export']
            assert private_path.read_bytes() == private_bytes
            retained = json.loads(private_bytes)
            assert retained['terminal_result']['analysis_state'] == 'incomplete'
            assert retained['findings'][0]['title'] == 'Grounded defect'
        warning = collapse_panel_text(capsys)
        if fault == 'placement-git-error-worktree':
            assert 'Terminal review finalization failed: GitError' in warning
        else:
            assert 'Terminal review finalization failed' not in warning
            assert any('ArtifactVisibilityError' in note for note in error.__notes__)
        if fault == 'placement-replacement-worktree':
            from tests.test_artifact_visibility import _Entry, _walk_entry
            after: list[_Entry] = []
            _walk_entry(review.repo, review.repo / '.git', after)
            assert after == state['replacement_git']
            assert (review.repo / 'replacement.bin').read_bytes() == b'replacement\x00\xff'
        return
    if fault.startswith('source-loss'):
        assert not review.repo.exists()
        if force_worktree:
            operational = {call['cwd'] for call in review.backend.calls if 'cwd' in call}
            assert operational and all(path != review.repo and not path.exists() for path in operational)
        assert not review.output.exists()
        raw = read_records(LocalRecordStore(review.tmp / 'records')).runs
        assert len(raw) == 1 and raw[0]['outcome'] == 'failed'
        findings = raw[0]['findings']['value']
        assert len(findings['items']) == 1
        assert findings['items'][0]['description'] == 'Grounded defect'
        assert findings['items'][0]['item_uid'] == state['admitted']['item_uid']
        assert findings['items'][0]['source_uids'] == state['admitted']['source_uids']
        coverage = findings['terminal_coverage']
        assert coverage['run_id'] == raw[0]['run_id']
        assert coverage['analyzed_revision']['head_sha'] == review.pr.head_sha
        assert coverage['analyzed_revision']['merge_base_sha'] == review.pr.base_sha
        assert coverage['analyzed_revision']['diff_key'] == raw[0]['original_task']['value']['diff_sha256']
        assert all(scope['status'] == 'complete' for scope in coverage['stack_outcomes'])
        assert state['export'] is not None, 'validated terminal findings were not installed privately'
        export_path, exported_bytes = state['export']
        assert export_path.read_bytes() == exported_bytes
        exported = json.loads(exported_bytes)
        assert len(exported['findings']) == 1 and exported['findings'][0]['title'] == 'Grounded defect'
        assert (exported['findings'][0]['placement'], exported['findings'][0]['line']) == ('inline', 2)
        result = exported['terminal_result']
        assert result['stack_outcomes'] == coverage['stack_outcomes']
        assert result['run_id'] == coverage['run_id']
        assert result['analyzed_revision'] == coverage['analyzed_revision']
        assert (result['pipeline_state'], result['analysis_state'], result['projection_valid']) == (
            'failed', 'incomplete', True)
        assert any('secondary base failure' in note for note in error.__notes__)
        assert 'Terminal review finalization failed' not in collapse_panel_text(capsys)
        return
    if fault == 'coverage':
        assert state['writes'] and not review.output.exists()
        assert 'Terminal review finalization failed: OSError' in collapse_panel_text(capsys)
        with pytest.raises(FindingsValidationError):
            review.load()
    elif fault == 'presentation':
        data = review.load()
        assert state['rejected'] and len(data['findings']) == 1 and 'review_context' not in data
        result = data['terminal_result']
        assert (result['pipeline_state'], result['analysis_state'], result['projection_valid']) == (
            'failed', 'incomplete', True)
        assert all(s['status'] == 'complete' for s in scopes(data).values())
        assert 'Review report unavailable:' in collapse_panel_text(capsys)
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
