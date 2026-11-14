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
from daydream.dataset import LocalRecordStore
from daydream.deep import review_terminal
from daydream.findings import FindingsValidationError, load_findings_artifact
from daydream.pr_review import post_findings_from_artifact
from daydream.review_result import ReviewCoverage
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.console import collapse_panel_text
from tests.harness.dataset import read_records
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
@pytest.mark.parametrize('fault', ['coverage', 'fix', 'projection', 'source-loss', 'placement-replacement-worktree'])
async def test_finalization_boundary(review: ReviewRun, capsys: pytest.CaptureFixture[str], fault: str) -> None:
    """A failure after admission publishes typed coverage atomically, never live or replaced Git."""
    force_worktree = fault.endswith('-worktree')
    if force_worktree:
        git(review.repo, 'remote', 'add', 'origin', str(bare_remote(review.tmp / 'origin.git')))
        git(review.repo, 'push', 'origin', 'main')
    error = RuntimeError('original provider error')
    stage, read, allocate, show = json_utils._stage_bytes, Path.read_text, tempfile.mkstemp, git_ops.show
    state: dict[str, Any] = {'failed': False, 'writes': [], 'export': None}
    staging_coverage = False
    real_run = subprocess.run
    def fault_at_placement(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if ('placement' in fault and state['failed'] and 'replacement_git' not in state
                and args == ['git', 'diff', '--name-only', '-z', review.pr.base_sha, review.pr.head_sha]):
            review.repo.rename(review.tmp / 'removed-source')
            init_repo(review.repo)
            (review.repo / 'replacement.bin').write_bytes(b'replacement\x00\xff')
            state['replacement_git'] = []
            _walk_entry(review.repo, review.repo / '.git', state['replacement_git'])
        return real_run(args, **kwargs)
    def observe_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        nonlocal staging_coverage
        staging_coverage = path.name == 'review-coverage.json'
        if staging_coverage:
            state['writes'].append((state['failed'], content))
        if b'"terminal_result"' in content and path.name != 'committed-review.json':
            state['export'] = (path, content)
        try:
            return stage(path, content, **kwargs)
        finally:
            staging_coverage = False
    def fail_allocation(*args: Any, **kwargs: Any) -> tuple[int, str]:
        if fault == 'coverage' and staging_coverage:
            raise OSError('coverage staging failed')
        return allocate(*args, **kwargs)
    def faulty_read(path: Path, *args: Any, **kwargs: Any) -> str:
        text = read(path, *args, **kwargs)
        return '{"items": "invalid"}' if fault == 'projection' and path.name == 'merged-items.json' else text
    def retained_source_only(target: Path, *args: Any, **kwargs: Any) -> bytes:
        if target == review.repo and fault == 'source-loss' and state['failed']:
            pytest.fail('terminal export attempted to read live Git after source loss')
        return show(target, *args, **kwargs)
    def fail_fix(prompt: str) -> Any:
        if fault == 'fix' and prompt.lower().startswith(('fix this issue', 'fix these')):
            state['failed'] = True
            return [error]
        return None
    def after_acceptance(kind: str) -> None:
        # The challenge's verdict is admitted; the native conversation then fails and the checkout is lost.
        if kind == 'challenge' and fault in ('source-loss', 'placement-replacement-worktree'):
            state['failed'] = True
            if fault == 'source-loss':
                review.repo.rename(review.tmp / 'removed-source')
            raise error
    review.patch.setattr(json_utils, '_stage_bytes', observe_stage)
    review.patch.setattr(tempfile, 'mkstemp', fail_allocation)
    review.patch.setattr(Path, 'read_text', faulty_read)
    review.patch.setattr(git_ops, 'show', retained_source_only)
    review.patch.setattr('daydream.git_ops.process.subprocess.run', fault_at_placement)
    review.backend = EmptyReviewBackend(review.repo, responder=fail_fix)
    review.backend.review_claims = [{'title': 'Grounded defect', 'file': 'api.py', 'line': 2}]
    review.backend.review_after_acceptance = after_acceptance
    if fault == 'fix':
        assert await review.run(output_mode='loop', findings_out=None, test_command=f'{sys.executable} -c pass') == 1
        saved_path = review.repo / '.daydream/deep/review-coverage.json'
        assert state['failed'] and state['writes'] and all(not after for after, _ in state['writes'])
        assert saved_path.read_bytes() == state['writes'][-1][1]
        saved = json.loads(saved_path.read_text())
        assert all(s['status'] == 'complete' for s in saved['stack_outcomes'] + saved['phase_outcomes'])
        return
    # The reviewer retains its own failures, so every row ends as an exit status, never as an exception.
    assert await review.run(dataset_capture=fault == 'source-loss', dataset_store_path=review.tmp / 'records',
                            force_worktree=force_worktree) == 1
    panel = collapse_panel_text(capsys)
    if fault == 'coverage':
        assert state['writes'] and state['export'] is None and not review.output.exists()
        assert 'Terminal review finalization failed: coverage staging failed' in panel
        with pytest.raises(FindingsValidationError):
            review.load()
        return
    if fault == 'projection':
        data = review.load()
        result = data['terminal_result']
        assert data['findings'] == [] and result['reason_codes'] == ['malformed_artifact']
        assert (result['pipeline_state'], result['analysis_state'], result['projection_valid']) == (
            'completed', 'failed', False)
        assert all(s['status'] == 'complete' for s in scopes(data).values())
        return
    assert not review.output.exists() and state['export'] is not None
    private_path, private_bytes = state['export']
    assert private_path.read_bytes() == private_bytes
    retained = json.loads(private_bytes)
    result = retained['terminal_result']
    assert (result['analysis_state'], result['projection_valid']) == ('incomplete', True)
    assert [(finding['title'], finding['placement'], finding['line']) for finding in retained['findings']] == [
        ('Grounded defect', 'inline', 2)]
    assert 'Terminal review finalization failed' not in panel
    if fault == 'source-loss':
        assert not review.repo.exists() and 'artifact source is unavailable' in panel
        raw = read_records(LocalRecordStore(review.tmp / 'records')).runs
        assert len(raw) == 1 and raw[0]['outcome'] == 'success'
        assert raw[0]['completeness']['terminal_review'] == 'available'
        findings = raw[0]['findings']['value']
        assert [item['description'] for item in findings['items']] == ['Grounded defect']
        coverage = findings['terminal_coverage']
        assert coverage['run_id'] == raw[0]['run_id'] == result['run_id']
        assert [(p['phase'], p['status'], p['reason_codes']) for p in coverage['phase_outcomes']] == [
            ('challenge', 'incomplete', ['backend_failure']), ('cross_assignment', 'complete', [])]
        assert coverage['analyzed_revision'] == result['analyzed_revision']
        assert coverage['analyzed_revision']['head_sha'] == review.pr.head_sha
        assert coverage['stack_outcomes'] == result['stack_outcomes']
        return
    assert 'artifact source directory identity changed' in panel
    after: list[_Entry] = []
    _walk_entry(review.repo, review.repo / '.git', after)
    assert after == state['replacement_git']
    assert (review.repo / 'replacement.bin').read_bytes() == b'replacement\x00\xff'
