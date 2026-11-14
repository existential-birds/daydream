"""Snapshot-bound terminal contracts through the real runner and Git."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream import git_ops, json_utils, runner
from daydream.backends import ResultEvent, ToolStartEvent
from daydream.dataset import LocalRecordStore
from daydream.deep import review_terminal
from daydream.findings import FindingsValidationError, load_findings_artifact
from daydream.phases.review import ReviewOutputError
from daydream.pr_review import post_findings_from_artifact
from daydream.review_result import ReviewCoverage
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.console import collapse_panel_text
from tests.harness.dataset import read_records
from tests.harness.fake_clock import FakeClock
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git, init_repo
from tests.test_artifact_visibility import _Entry, _walk_entry
from tests.test_deep_orchestrator import _pin_findings_pr, _record
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


async def test_comment_pins_pr_head_independently_and_excludes_local_edits(
    review: ReviewRun, capsys: pytest.CaptureFixture[str],
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
    assert plan.event is ReviewEvent.COMMENT
    coverage = ReviewCoverage.from_dict(json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text()))
    terminal = coverage.finalize('completed')
    assert terminal['analysis_state'] == 'incomplete'
    assert 'evidence_incomplete' in terminal['reason_codes']
    patch = (review.repo / '.daydream/diff.patch').read_text()
    assert 'LOCAL_' not in patch and patch
    assert captured_roots and all(not root.exists() for root in captured_roots)
    assert 'excludes local tracked and untracked edits' in collapse_panel_text(capsys)
    assert git_ops.head_sha(review.repo) == local_head and (review.repo / '.git/index').read_bytes() == before
    assert (review.repo / 'api.py').read_text() == 'LOCAL_WORKING = True\n'
    assert (review.repo / 'untracked.py').read_text() == 'LOCAL_UNTRACKED = True\n'


@pytest.mark.parametrize('case', ['dirty-no-diff', 'live'])
async def test_snapshot_boundaries(review: ReviewRun, fake_gh: FakeGh, case: str) -> None:
    """Analysis stays bound to the captured snapshot: dirty edits and a mid-run PR head advance never rebind it.

    'live' is a runner row: only a real review whose PR head moves after capture proves the pinned
    head is still analyzed and placed, with one lookup. Its produced artifact then meets the poster.
    """
    if case == 'live':
        live, calls = review.pr, []

        def lookup(*_a: Any, **_k: Any) -> Any:
            calls.append(live)
            return live

        def advance(_prompt: str) -> None:
            nonlocal live
            live = replace(review.pr, head_sha=review.pr.base_sha)

        review.patch.setattr('daydream.pr_review.find_pr_by_number', lookup)
        review.backend = EmptyReviewBackend(review.repo, responder=advance)
        review.backend.review_claims = [{'title': 'Grounded defect', 'file': 'api.py', 'line': 2}]
        assert await review.run() == 0
        data = review.load()
        result = data['terminal_result']
        assert result['analysis_state'] == 'complete'
        assert live.head_sha == review.pr.base_sha and git_ops.head_sha(review.repo) == review.pr.head_sha
        assert git_ops.show(review.repo, live.head_sha, 'api.py') and len(calls) == 1
        assert (data['findings'][0]['placement'], data['findings'][0]['line']) == ('inline', 2)
        assert result['analyzed_revision']['pr_base_sha'] == review.pr.base_sha
        # The poster takes that produced artifact only for the event it was produced for: a moved head, another
        # repository or another pull request is refused before any GitHub call.
        printed = StringIO()

        def post(artifact: Path = review.output, **facts: Any) -> int:
            return post_findings_from_artifact(review.repo, artifact, console=Console(file=printed, width=200), **{
                'pr_number': 7, 'head_sha': review.pr.head_sha, 'repo': 'o/r', 'approve_on_clean': True, **facts})

        produced = len(fake_gh.process_calls())
        for changed in ({'head_sha': live.head_sha}, {'repo': 'o/other'}, {'pr_number': 8}):
            assert post(**changed) == 1, changed
        assert len(fake_gh.process_calls()) == produced
        assert post() == 0
        (posted,) = fake_gh.calls('POST', 'repos/o/r/pulls/7/reviews')
        # A finding without severity withholds the opted-in approval.
        assert (posted.payload['event'], posted.payload['commit_id']) == ('COMMENT', review.pr.head_sha)
        assert [(comment['path'], comment['line']) for comment in posted.payload['comments']] == [('api.py', 2)]
        # An incomplete review still exports its admitted finding with truthful coverage, and never approves.
        review.backend.review_undecided = 1
        live, partial = review.pr, review.tmp / 'incomplete.json'
        assert await review.run(findings_out=str(partial)) == 0
        exported = json.loads(partial.read_text())
        assert exported['terminal_result']['analysis_state'] == 'incomplete'
        assert [finding['title'] for finding in exported['findings']] == ['Grounded defect']
        assert post(partial) == 0
        notice = fake_gh.calls('POST', 'repos/o/r/pulls/7/reviews')[-1].payload
        assert notice['event'] == 'COMMENT' and len(notice['comments']) == 1
        assert '**Review incomplete.**\n\n- committed:00000: evidence_incomplete' in notice['body']
        # A delivery GitHub does not confirm is reported as unconfirmed, never as posted.
        fake_gh.set_response('POST', 'repos/o/r/pulls/7/reviews', {'__error__': 'HTTP 422: unprocessable'})
        assert post() == 1 and 'Could not confirm the PR review was posted' in ' '.join(printed.getvalue().split())
        return
    prior = {'.daydream/deep/history.json': '{"prior": true}\n',
             '.daydream/deep/merged-items.json': '{"items": [{"item_uid": "item:old"}]}\n',
             '.daydream/recommended.patch': 'prior recommended patch\n',
             '.review-output.md': '# Review\n\n## Issues\n1. [obsolete.py:9] PRIOR_FINDING\n\n'
                                  '## Held Findings\n2. [obsolete.py:10] PRIOR_HELD\n\n## Diagrams\nPRIOR_DIAGRAM\n'}
    for name, text in prior.items():
        path = review.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(review.repo, 'checkout', 'main')
    review.pr = _pin_findings_pr(review.patch, review.repo)
    (review.repo / 'api.py').write_text('DIRTY = True\n')
    assert await review.run() == 0
    result = review.load()['terminal_result']
    assert result['analysis_state'] == 'complete'
    assert review.backend.calls == []
    assert (review.repo / 'api.py').read_text() == 'DIRTY = True\n'
    assert 'DIRTY = True' not in (review.repo / '.daydream/diff.patch').read_text()
    report = (review.repo / '.review-output.md').read_text()
    assert all(marker not in report for marker in ('PRIOR_FINDING', 'PRIOR_HELD', 'PRIOR_DIAGRAM'))
    assert 'No confirmed findings.' in report and result['run_id'] in report
    assert (review.repo / '.daydream/deep/review-output.md').read_text() == report


@pytest.mark.parametrize(('fault', 'owner'), [
    ('corrupt', 'python'), *[(fault, 'structure') for fault in ('missing', 'shape', 'duplicate')],
])
async def test_loaded_artifact_faults(review: ReviewRun, fault: str, owner: str) -> None:
    source = git(review.repo, 'diff', 'HEAD')
    artifact_name = f'stack-{owner}-records.json'
    write = Path.write_text
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
    review.patch.setattr(Path, 'write_text', faulty_write)
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


@pytest.mark.parametrize(('phase', 'reason'), [('intent-empty', 'missing_output'), ('malformed', 'malformed_output')])
async def test_required_phase_faults(review: ReviewRun, phase: str, reason: str) -> None:
    """Required-phase output faults are typed; only post-admission faults keep admitted findings.

    The supervisor row is a runner row: only real admission followed by a failed required
    supervision phase proves that admitted findings survive with an incomplete analysis.
    """
    from tests.harness.review_profile import independent_alternatives_profile

    def response(prompt: str) -> Any:
        lower = prompt.lower()
        if phase == 'intent-empty' and 'present your understanding concisely' in lower:
            return [ResultEvent(structured_output=None, continuation=None)]
        if phase == 'malformed' and 'supervisor adjudication' in lower:
            return [ResultEvent(structured_output={'verdicts': [None]}, continuation=None)]
    review.backend = EmptyReviewBackend(review.repo, forbid_merge=False, forbid_supervise=False,
                                        review_by_stack={'structure': [record()]}, responder=response)
    review.backend.merge_echo_records = True
    if phase == 'intent-empty':
        review.config = replace(review.config, review_profile=independent_alternatives_profile())
    with pytest.raises(ReviewOutputError) as raised:
        await review.run()
    assert raised.value.reason_code == reason
    data = review.load()
    result = data['terminal_result']
    admitted = phase == 'malformed'
    assert (result['pipeline_state'], result['analysis_state']) == ('failed', 'incomplete' if admitted else 'failed')
    assert reason in result['reason_codes']
    assert [f['title'] for f in data['findings']] == (['Grounded defect'] if admitted else [])
    if admitted:
        outcome = next(p for p in result['phase_outcomes'] if p['phase'] == 'supervision')
        assert (outcome['status'], outcome['reason_codes']) == ('failed', [reason])
        assert any('supervisor adjudication' in c['prompt'].lower() for c in review.backend.calls)


# Runner rows: only the real supervision investigation deadline decides whether a late result is admitted.
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
    """A failed public install or staging leaves the destination absent or byte-identical to the prior export."""
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


# Source-loss and replacement-checkout rows are runner rows: only the real terminal export after
# admission can lose its checkout, read retained bytes, and install privately.
@pytest.mark.parametrize('fault', ['coverage', 'fix', 'salvage', 'source-loss-salvage',
                                   'placement-replacement-worktree'])
async def test_finalization_boundary(review: ReviewRun, capsys: pytest.CaptureFixture[str], fault: str) -> None:
    """Provider failure after admission publishes typed coverage atomically, never live or replaced Git."""
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
            review.repo.rename(review.tmp / 'removed-source')
            init_repo(review.repo)
            (review.repo / 'replacement.bin').write_bytes(b'replacement\x00\xff')
            state['replacement_git'] = []
            _walk_entry(review.repo, review.repo / '.git', state['replacement_git'])
        return real_run(args, **kwargs)
    def observe_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        nonlocal staging_coverage
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
        if target == review.repo and fault == 'source-loss-salvage' and state['failed']:
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
            if fault == 'source-loss-salvage':
                state['admitted'] = json.loads(state['merged'].read_text())['items'][0]
                state['merged'].unlink()
                review.repo.rename(review.tmp / 'removed-source')
            return [error]
    review.patch.setattr(json_utils, '_stage_bytes', observe_stage)
    review.patch.setattr(tempfile, 'mkstemp', fail_allocation)
    review.patch.setattr(Path, 'write_text', observe_write)
    review.patch.setattr(Path, 'read_text', faulty_read)
    review.patch.setattr(git_ops, 'show', retained_source_only)
    review.patch.setattr('daydream.git_ops.process.subprocess.run', fault_at_placement)
    stack = 'python' if force_worktree else 'structure'
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
        await review.run(dataset_capture=fault == 'source-loss-salvage',
                         dataset_store_path=review.tmp / 'records', force_worktree=force_worktree)
    assert raised.value is error
    if placement_fault:
        assert state.get('dispatch_injected')
    if placement_fault:
        assert not review.output.exists()
        assert state['export'] is not None
        private_path, private_bytes = state['export']
        assert private_path.read_bytes() == private_bytes
        retained = json.loads(private_bytes)
        assert retained['terminal_result']['analysis_state'] == 'incomplete'
        assert retained['findings'][0]['title'] == 'Grounded defect'
        assert 'Terminal review finalization failed' not in collapse_panel_text(capsys)
        assert any('ArtifactVisibilityError' in note for note in error.__notes__)
        after: list[_Entry] = []
        _walk_entry(review.repo, review.repo / '.git', after)
        assert after == state['replacement_git']
        assert (review.repo / 'replacement.bin').read_bytes() == b'replacement\x00\xff'
        return
    if fault == 'source-loss-salvage':
        assert not review.repo.exists()
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
