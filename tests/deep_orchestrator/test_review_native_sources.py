"""Ordinary file inputs keep useful before-side context for file-only backends."""
from __future__ import annotations

import copy
import hashlib
import json
import tracemalloc
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent, ToolResultEvent, ToolStartEvent
from daydream.git_ops import queries
from daydream.git_ops.models import GitError
from daydream.git_ops.source import frozen_source
from daydream.host_tools import HostToolFatal, HostToolRejected
from daydream.review_source_refs import canonicalize_source_selectors, source_id
from tests.deep_orchestrator.test_review_completion import ReviewRun, record, scopes
from tests.deep_orchestrator.test_review_investigation import (
    InvestigationRun,
    candidate,
    many_file_review,
    read_events,
    stage_ends,
)
from tests.deep_orchestrator.test_review_stage_reservations import NativeStageBackend
from tests.harness.git_helpers import git, seed_feature_branch, tracked_source_state


async def test_local_capture_instability_refuses_dispatch_without_changing_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Capture retries alone miss a runner dispatching models after preparation fails."""
    repo = tmp_path / 'unstable_local'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'}, feature={'api.py': 'VALUE = 1\n'})
    review = ReviewRun(repo, tmp_path, monkeypatch)
    before = tracked_source_state(repo)
    index = (repo / '.git/index').read_bytes()
    enumerate_untracked = queries._list_untracked_z
    observed = []

    def concurrent_write(source: Path) -> list[str]:
        paths = enumerate_untracked(source)
        if source == repo:
            observed.append(source)
            path = source / 'api.py'
            # Real writes with unchanged bytes still invalidate observations
            # accepted before this enumeration/materialization boundary.
            path.write_bytes(path.read_bytes())
        return paths

    monkeypatch.setattr(queries, '_list_untracked_z', concurrent_write)
    assert await review.run(findings_out=None) == 1
    from tests.harness.console import collapse_panel_text

    assert 'stop concurrent writers and rerun the review' in collapse_panel_text(capsys)
    assert len(observed) > 1 and review.backend.calls == []
    assert (repo / '.git/index').read_bytes() == index
    assert tracked_source_state(repo) == before
    assert not review.output.exists()


@pytest.mark.parametrize('cancelled', [False, True], ids=['final-working', 'net-cancellation'])
async def test_local_runner_uses_one_capture_for_diff_targets_and_supporting_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool,
) -> None:
    """Source-owner tests miss runner cwd routing and the no-model net-empty outcome."""
    from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend

    repo = tmp_path / 'local_subject'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n', 'support.py': 'HELPER = 0\n'},
                        feature={'api.py': 'VALUE = 1\n', 'support.py': 'HELPER = 0\n'})
    (repo / 'api.py').write_text('VALUE = 2\n')
    git(repo, 'add', 'api.py')
    accepted = 'VALUE = 0\n' if cancelled else 'VALUE = 3\n'
    (repo / 'api.py').write_text(accepted)
    (repo / 'support.py').write_text('HELPER = 2\n')
    if not cancelled:
        (repo / 'new.py').write_text('NEW = 3\n')
    index = (repo / '.git/index').read_bytes()
    original_head = git(repo, 'rev-parse', 'HEAD')
    seen: list[Path] = []
    review = ReviewRun(repo, tmp_path, monkeypatch)

    class CapturedReader(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            assert cwd != repo
            assert (cwd / 'api.py').read_text() == accepted
            assert (cwd / 'support.py').read_text() == 'HELPER = 2\n'
            assert (cwd / 'new.py').read_text() == 'NEW = 3\n'
            if not seen:
                (repo / 'api.py').write_text('VALUE = 9\n')
                (repo / 'support.py').write_text('HELPER = 9\n')
                (repo / 'new.py').write_text('NEW = 9\n')
            seen.append(cwd)
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = CapturedReader(repo)
    assert await review.run(findings_out=None, ignore_paths=['support.py']) == 0
    diff = (repo / '.daydream/diff.patch').read_text()
    result = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    assert result['analyzed_revision']['head_sha'] == original_head
    assert len(result['analyzed_revision']['source_id']) == 64
    if cancelled:
        assert diff == '' and not seen and result['planned_scopes'] == []
    else:
        assert seen and all(not root.exists() for root in seen)
        assert 'VALUE = 3' in diff and 'NEW = 3' in diff and 'VALUE = 2' not in diff
        assert 'support.py' not in diff
        assert {path for scope in result['planned_scopes'] for path in scope['files']} == {'api.py', 'new.py'}
    assert (repo / '.git/index').read_bytes() == index
    assert git(repo, 'rev-parse', 'HEAD') == original_head


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-paths', 'inline'])
@pytest.mark.parametrize('legacy', [False, True], ids=['compact', 'full-hash'])
async def test_compact_assigned_selector_completes_on_the_last_native_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool, legacy: bool,
) -> None:
    """Long-hash fixtures miss a provider selecting a compact handle with no correction call left."""
    review = many_file_review(tmp_path, monkeypatch, count=1, docs_count=0)
    review.backend = NativeStageBackend(review.repo)
    review.backend.sandbox = sandbox
    diff = git_ops.diff(review.repo, 'main')
    digest = hashlib.sha256(diff.encode()).hexdigest()

    @review.backend.script('python')
    def respond(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        binding = review.backend.calls[-1]['host_tools']
        for index in range(stage['hard_tool_call_allowance'] - 1):
            yield from read_events(review.repo, 'api.py', event_id=f'read-{index}')
        identifier = source_id(stage['source_refs']['api.py'][0]) if legacy else 'source:1'
        output['targets'][0]['evidence_refs'] = [
            {'source_id': identifier, 'basis': 'supplied_diff', 'source_range': None}]
        yield ToolStartEvent(id='compact-submit', name='structured_output', input=output)
        yield ToolResultEvent(id='compact-submit', output='Submitted.', is_error=False)
        assert binding.accepted and binding.fatal_error is None
        assert [option['source_id'] for option in stage['source_options']['api.py']] == ['source:1', 'source:2']

    data = await review.finish('python')
    decision, = scopes(data)['python']['target_decisions']
    assert decision['evidence_refs'] == [{
        'artifact_sha256': digest, 'part_sha256': digest, 'revision': review.pr.base_sha,
        'path': 'api.py', 'side': 'old', 'source_range': {'start': 1, 'end': 2}, 'fragment': None,
        'basis': 'supplied_diff', 'role': 'assigned',
    }]
    event, = stage_ends(review, 'python')
    assert event['metadata']['admitted'] is True
    assert event['metadata']['remaining_tool_calls'] == 0
    assert event['metadata']['successful_submissions'] == 1


async def test_oversized_renamed_source_is_omitted_without_buffering_the_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'oversized_renamed_source'
    oversized = '#' + 'x' * (9 * 1024 * 1024)
    seed_feature_branch(repo, base={'retired.py': oversized, 'consumer.py': 'VALUE = 0\n'},
                        feature={'modern.py': oversized, 'consumer.py': 'VALUE = 1\n'})
    git(repo, 'rm', 'retired.py')
    git(repo, 'commit', '-m', 'rename oversized source')
    base = git(repo, 'rev-parse', 'main')
    tracemalloc.start()
    try:
        with pytest.raises(GitError, match='bounded retrieval limit'):
            frozen_source(repo, base, 'retired.py')
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeStageBackend(repo)
    omitted = False

    @review.backend.script('python')
    def respond(stage: dict[str, Any], output: dict[str, Any]) -> None:
        nonlocal omitted
        contexts = json.loads(stage['access_guide'])['contexts']
        assert not any(item['label'].startswith('before-context-') for item in contexts)
        assert any(status.get('file') == 'modern.py' and status['status'] == 'omitted'
                   for status in stage['context_statuses'])
        omitted = True

    assert await review.run() == 0
    assert scopes(review.load())['python']['status'] == 'complete'
    assert omitted
    assert not (repo / 'retired.py').exists()
    assert (repo / 'modern.py').exists()


@pytest.mark.parametrize('change', ['deleted', 'renamed', 'pure-rename'])
@pytest.mark.parametrize('broken', [False, True], ids=['compatible', 'consumer-defect'])
async def test_before_source_is_furnished_as_a_private_ordinary_file_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, broken: bool,
) -> None:
    repo = tmp_path / 'before_source'
    renamed = change != 'deleted'
    old_path = 'legacy.py' if renamed else 'retired.py'
    old = (''.join(f'# Stable API contract context {index:02}\n' for index in range(30)) if renamed else
           '# Public greeting used by retained consumers.\n') + 'def hello():\n    return "world"\n'
    base_consumer = 'EXPECTED_GREETING = "world"\n' if renamed else (
        'from retired import hello\nprint(hello())\n' if broken else 'print("world")\n')
    consumer = 'EXPECTED_GREETING = "universe"\n' if renamed and broken else base_consumer
    after = {'consumer.py': consumer + '# Existing public contract retained.\n'}
    if renamed:
        after['modern.py'] = old + ('' if change == 'pure-rename' else
                                    '# Relocated implementation retains the greeting.\n')
    seed_feature_branch(repo, base={old_path: old, 'consumer.py': base_consumer}, feature=after)
    git(repo, 'rm', old_path)
    git(repo, 'commit', '-m', 'remove old greeting path')
    before = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeStageBackend(repo)
    title = ('Renamed greeting no longer satisfies the consumer contract' if renamed else
             'Retained consumer imports a deleted greeting module')
    submitted: dict[str, list[dict[str, Any]]] = {}

    @review.backend.script('python')
    def respond(stage: dict[str, Any], output: dict[str, Any]) -> None:
        contexts = json.loads(stage['access_guide'])['contexts']
        context, = [item for item in contexts if item['label'].startswith('before-context-')]
        path = Path(context['path'])
        captured = path.read_text()
        assert context['label'] in stage['context_inputs']
        assert f'# original_path: {old_path}' in captured and '# side: before' in captured
        assert captured.endswith(old) and not path.is_relative_to(repo)
        for target in output['targets']:
            offered = stage['source_refs'][target['target_id']]
            if any(ref['path'] in {old_path, 'modern.py'} for ref in offered):
                # A supported Pi subclass needs the same frozen before context.
                # Submit only IDs; production callback admission must retain the
                # rename, absent-side and metadata identities asserted below.
                part = next(part for part in stage['assignment_parts'] if part['target_id'] == target['target_id'])
                options = part['source_options']
                assert len(options) == len(offered)
                assert all(option['basis'] == 'supplied_diff' for option in options)
                target['evidence_refs'] = [
                    {'source_id': option['source_id'], 'basis': 'supplied_diff', 'source_range': None}
                    for option in options]
                submitted[target['target_id']] = copy.deepcopy(offered)
        if broken:
            finding = dict(record(1), file='consumer.py', description=title,
                           rationale='The retained consumer disagrees with the removed source contract.',
                           evidence=('consumer.py:1 expects universe; legacy.py:32 returned world.' if renamed else
                                     'consumer.py:1 imports retired.hello; the supplied before-context defined it.'))
            output['candidates'] = [candidate(disposition='confirmed', finding=finding, file='consumer.py', line=1,
                                             trigger='Run the retained consumer against the greeting contract',
                                             consequence=('The greeting contract disagrees' if renamed else
                                                          'ModuleNotFoundError'),
                                             grounds=finding['evidence'])]

    data = await review.finish('python', findings=(title,) if broken else ())
    assert submitted, 'the removed/renamed source must produce an assigned source decision'
    retained = {decision['target_id']: decision['evidence_refs']
                for decision in scopes(data)['python']['target_decisions']}
    for target_id, refs in submitted.items():
        assert retained[target_id] == refs
        if change == 'pure-rename':
            # A metadata-only assignment has no source lines to cite. Bind the
            # reference to independently known Git bytes, rather than echoing
            # the offered selector as proof that generation was correct.
            metadata = (f'diff --git a/{old_path} b/modern.py\nsimilarity index 100%\n'
                        f'rename from {old_path}\nrename to modern.py\n').encode()
            ref, = refs
            assert (ref['path'], ref['side'], ref['revision']) == ('modern.py', 'diff', review.pr.head_sha)
            assert ref['part_sha256'] == hashlib.sha256(metadata).hexdigest()
            assert ref['fragment'] == {'offset': 0, 'bytes': len(metadata),
                                       'sha256': hashlib.sha256(metadata).hexdigest()}
            assert ref['source_range'] is None and ref['role'] == 'assigned'
            continue
        assert {(ref['path'], ref['side'], ref['revision']) for ref in refs} == (
            {(old_path, 'old', review.pr.base_sha), ('modern.py', 'new', review.pr.head_sha)} if renamed else
            {(old_path, 'old', review.pr.base_sha)})
        assert [ref['source_range'] for ref in refs] == (
            [{'start': 30, 'end': 32}, {'start': 30, 'end': 33}] if renamed else [{'start': 1, 'end': 3}])
        assert all(ref['role'] == 'assigned' and ref['fragment'] is None for ref in refs)
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    persisted = next(scope for scope in coverage['stack_outcomes'] if scope['scope_id'] == 'python')
    assert persisted['target_decisions'] == scopes(data)['python']['target_decisions']
    assert not (repo / old_path).exists()
    assert (repo / 'modern.py').is_file() is renamed
    assert tracked_source_state(repo) == before


@pytest.mark.parametrize('fault', [None, 'contained-range', 'foreign-id', 'assigned-range', 'supporting-range',
                                  'supporting-only', 'forged-hash', 'missing', 'traversal', 'binary', 'oversized',
                                  'symlink', 'storage-reread'])
def test_source_selectors_construct_supporting_evidence_from_the_pinned_git_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None,
) -> None:
    """Canonical references miss selector feedback; vague errors cannot guide a valid bounded correction."""
    repo = tmp_path / 'pinned_selectors'
    supported = 'FIRST = "界"\u2028STILL_FIRST = True\nSECOND = True\n'
    before = 'UNCHANGED = True\nVALUE = 0\nTRAILING = True\n'
    seed_feature_branch(repo, base={'api.py': before, 'support.py': supported},
                        feature={'api.py': before.replace('VALUE = 0', 'VALUE = 1'), 'support.py': supported})
    # Frozen regular-blob validation must own binary and large supporting inputs;
    # the model can choose a path/range, never supply replacement metadata.
    # NUL makes Git retain a binary diff header while the invalid UTF-8 byte
    # still reaches the supporting-source decoder at selector resolution.
    (repo / 'binary.dat').write_bytes(b'\xff\0\n')
    (repo / 'oversized.py').write_text('#' * (64 * 1024 + 1))
    (repo / 'symlink.py').symlink_to('support.py')
    git(repo, 'add', 'binary.dat', 'oversized.py', 'symlink.py')
    git(repo, 'commit', '-m', 'add bounded source boundary cases')
    (repo / 'symlink.py').unlink()
    revision = {'head_sha': git(repo, 'rev-parse', 'HEAD'), 'merge_base_sha': git(repo, 'rev-parse', 'main')}
    part = git(repo, 'diff', 'main', '--', 'api.py') + '\n'
    assigned = {'artifact_sha256': hashlib.sha256(git_ops.diff(repo, 'main').encode()).hexdigest(),
                'part_sha256': hashlib.sha256(part.encode()).hexdigest(), 'revision': revision['head_sha'],
                'path': 'api.py', 'side': 'new', 'source_range': {'start': 1, 'end': 3}, 'fragment': None,
                'basis': 'supplied_diff', 'role': 'assigned'}
    choices: list[dict[str, Any]] = [
        {'source_id': source_id(assigned), 'basis': 'supplied_diff', 'source_range': None},
        {'path': 'support.py', 'side': 'new', 'source_range': {'start': 1, 'end': 2}, 'basis': 'source_inspection'},
    ]
    # The worktree disagrees after capture. Its bytes must never become the
    # canonical supporting digest or LF range authority.
    (repo / 'support.py').write_text('MUTABLE_WORKTREE_CONTENT\n')
    if fault == 'contained-range':
        choices[0]['source_range'] = {'start': 2, 'end': 2}
    elif fault == 'foreign-id':
        choices[0]['source_id'] = 'source:' + '0' * 64
    elif fault == 'assigned-range':
        choices[0]['source_range'] = {'start': 1, 'end': 4}
    elif fault == 'supporting-range':
        choices[1]['source_range'] = {'start': 2, 'end': 1}
    elif fault == 'supporting-only':
        choices.pop(0)
    elif fault == 'forged-hash':
        choices[1]['artifact_sha256'] = '0' * 64
    elif fault in {'missing', 'traversal', 'binary', 'oversized', 'symlink'}:
        choices[1]['path'] = {'missing': 'missing.py', 'traversal': '../support.py',
                              'binary': 'binary.dat', 'oversized': 'oversized.py', 'symlink': 'symlink.py'}[fault]
        # Keep the range valid for each blob so a missing size/UTF-8/regular-file
        # guard cannot pass this regression at an unrelated range check.
        choices[1]['source_range'] = {'start': 1, 'end': 1}
    if fault == 'storage-reread':
        # A later assigned selector makes the last processed field differ from
        # the supporting reference whose real Git reread loses storage. Other
        # selector cases cover admission, but miss this diagnostic attribution.
        choices.append({'source_id': source_id(assigned), 'basis': 'supplied_diff', 'source_range': None})
        real_show = queries.show
        unavailable = repo / '.git-unavailable'

        def lose_storage(source_repo: Path, ref: str, path: str) -> bytes:
            contents = real_show(source_repo, ref, path)
            if source_repo == repo and path == 'support.py':
                (repo / '.git').rename(unavailable)
            return contents

        monkeypatch.setattr(queries, 'show', lose_storage)
        try:
            with pytest.raises(HostToolFatal) as fatal:
                canonicalize_source_selectors(choices, [assigned], revision=revision, source_root=repo)
        finally:
            if unavailable.exists():
                unavailable.rename(repo / '.git')
        assert fatal.value.code == 'input_unavailable' and fatal.value.preacceptance
        assert isinstance(fatal.value.__cause__, GitError)
        diagnostic = str(fatal.value)
        assert 'evidence_refs[1]' in diagnostic and 'support.py' in diagnostic and '(new)' in diagnostic
        assert 'evidence_refs[2]' not in diagnostic
        assert 'Changing the submission cannot resolve' in diagnostic and str(repo) not in diagnostic
        return
    if fault not in {None, 'contained-range'}:
        with pytest.raises(HostToolRejected) as rejected:
            canonicalize_source_selectors(choices, [assigned], revision=revision, source_root=repo)
        message = str(rejected.value)
        assert 'evidence_refs' in message
        assert any(word in message.lower() for word in ('choose', 'use', 'remove', 'supply'))
        if fault == 'foreign-id':
            assert choices[0]['source_id'] in message and 'assigned' in message
        elif fault == 'assigned-range':
            assert '1..3' in message and 'null' in message
        elif fault == 'supporting-range':
            assert '1..2' in message and 'start <= end' in message
        elif fault == 'oversized':
            assert 'whole pinned blob' in message and '65536' in message
            assert 'narrowing source_range cannot' in message
        elif fault == 'supporting-only':
            assert 'reviewed' in message and 'source_id' in message
        return
    canonical = canonicalize_source_selectors(choices, [assigned], revision=revision, source_root=repo)
    expected_assigned = {**assigned, 'source_range': {'start': 2, 'end': 2}} if fault == 'contained-range' else assigned
    assert canonical == [expected_assigned, {
        'artifact_sha256': hashlib.sha256(supported.encode()).hexdigest(), 'part_sha256': None,
        'revision': revision['head_sha'], 'path': 'support.py', 'side': 'new',
        'source_range': {'start': 1, 'end': 2}, 'fragment': None,
        'basis': 'source_inspection', 'role': 'supporting',
    }]
    assert (repo / 'support.py').read_text() == 'MUTABLE_WORKTREE_CONTENT\n'
