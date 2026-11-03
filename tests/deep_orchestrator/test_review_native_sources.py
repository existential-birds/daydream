"""Ordinary file inputs keep useful before-side context for file-only backends."""
from __future__ import annotations

import copy
import hashlib
import json
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.deep import orchestrator
from daydream.git_ops import queries
from daydream.git_ops.models import GitError
from daydream.git_ops.source import frozen_source
from daydream.host_tools import HostToolFatal, HostToolRejected
from daydream.review_source_refs import (
    canonicalize_source_selectors,
    source_id,
    validate_source_refs,
)
from tests.deep_orchestrator.test_review_completion import ReviewRun, record, scopes
from tests.deep_orchestrator.test_review_investigation import (
    InvestigationRun,
    candidate,
)
from tests.deep_orchestrator.test_review_stage_reservations import NativeStageBackend
from tests.harness.git_helpers import git, seed_feature_branch, tracked_source_state


@pytest.mark.usefixtures('fake_gh')
async def test_local_capture_failures_refuse_dispatch_without_changing_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A captured source tampered before setup refuses model dispatch with an actionable cause."""
    from tests.harness.console import collapse_panel_text

    repo = tmp_path / 'unstable_local'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'}, feature={'api.py': 'VALUE = 1\n'})
    review = ReviewRun(repo, tmp_path, monkeypatch)
    before = tracked_source_state(repo)
    index = (repo / '.git/index').read_bytes()
    observed: list[Path] = []
    setup = orchestrator.run_deep

    async def corrupt_before_setup(config: Any, work: Any, *args: Any, **kwargs: Any) -> int:
        assert work.captured_source is not None and work.repo != repo
        observed.append(work.repo)
        (work.repo / 'api.py').write_bytes(b'TAMPER = True\n')
        return await setup(config, work, *args, **kwargs)

    monkeypatch.setattr(orchestrator, 'run_deep', corrupt_before_setup)
    assert await review.run(findings_out=None) == 1
    assert 'captured native source integrity failed' in collapse_panel_text(capsys)
    assert review.backend.calls == []
    assert len(observed) == 1 and not observed[0].exists()
    assert (repo / '.git/index').read_bytes() == index
    assert tracked_source_state(repo) == before
    assert not review.output.exists()
    assert not (repo / '.review-output.md').exists()
    assert not (repo / '.daydream/deep/review-coverage.json').exists()


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


async def test_before_source_is_furnished_as_a_private_ordinary_file_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure rename furnishes the removed side privately and binds its metadata-only decision to Git bytes."""
    repo = tmp_path / 'before_source'
    old_path = 'legacy.py'
    old = ''.join(f'# Stable API contract context {index:02}\n' for index in range(30)) + (
        'def hello():\n    return "world"\n')
    base_consumer = 'EXPECTED_GREETING = "world"\n'
    after = {'consumer.py': 'EXPECTED_GREETING = "universe"\n# Existing public contract retained.\n',
             'modern.py': old}
    seed_feature_branch(repo, base={old_path: old, 'consumer.py': base_consumer}, feature=after)
    git(repo, 'rm', old_path)
    git(repo, 'commit', '-m', 'remove old greeting path')
    before = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeStageBackend(repo)
    title = 'Renamed greeting no longer satisfies the consumer contract'
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
            offered = stage['source_options'][target['target_id']]
            if any(option['path'] in {old_path, 'modern.py'} for option in offered):
                # A supported Pi subclass needs the same frozen before context.
                # Submit only IDs; production callback admission must retain the
                # rename, absent-side and metadata identities asserted below.
                part = next(part for part in stage['assignment_parts'] if part['target_id'] == target['target_id'])
                options = part['source_options']
                assert options == offered
                assert all(option['basis'] == 'supplied_diff' for option in options)
                target['evidence_refs'] = [
                    {'source_id': option['source_id'], 'basis': 'supplied_diff', 'source_range': None}
                    for option in options]
                submitted[target['target_id']] = copy.deepcopy(offered)
        finding = dict(record(1), file='consumer.py', description=title,
                       rationale='The retained consumer disagrees with the removed source contract.',
                       evidence='consumer.py:1 expects universe; legacy.py:32 returned world.')
        output['candidates'] = [candidate(disposition='confirmed', finding=finding, file='consumer.py', line=1,
                                         trigger='Run the retained consumer against the greeting contract',
                                         consequence='The greeting contract disagrees', grounds=finding['evidence'])]

    data = await review.finish('python', findings=(title,))
    assert submitted, 'the renamed source must produce an assigned source decision'
    retained = {decision['target_id']: decision['evidence_refs']
                for decision in scopes(data)['python']['target_decisions']}
    # A metadata-only assignment has no source lines to cite. Bind the reference
    # to independently known Git bytes, rather than echoing the offered selector.
    metadata = (f'diff --git a/{old_path} b/modern.py\nsimilarity index 100%\n'
                f'rename from {old_path}\nrename to modern.py\n').encode()
    for target_id, options in submitted.items():
        ref, = retained[target_id]
        assert len(options) == 1
        assert (ref['path'], ref['side'], ref['revision']) == ('modern.py', 'diff', review.pr.head_sha)
        assert ref['part_sha256'] == hashlib.sha256(metadata).hexdigest()
        assert ref['fragment'] == {'offset': 0, 'bytes': len(metadata),
                                   'sha256': hashlib.sha256(metadata).hexdigest()}
        assert ref['source_range'] is None and ref['role'] == 'assigned'
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    persisted = next(scope for scope in coverage['stack_outcomes'] if scope['scope_id'] == 'python')
    assert persisted['target_decisions'] == scopes(data)['python']['target_decisions']
    assert not (repo / old_path).exists() and (repo / 'modern.py').is_file()
    assert tracked_source_state(repo) == before


def test_source_selectors_construct_supporting_evidence_from_the_pinned_git_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    # The worktree disagrees after capture. Its bytes must never become the
    # canonical supporting digest or LF range authority.
    (repo / 'support.py').write_text('MUTABLE_WORKTREE_CONTENT\n')

    def selectors() -> list[dict[str, Any]]:
        return [{'source_id': source_id(assigned), 'basis': 'supplied_diff', 'source_range': None},
                {'path': 'support.py', 'side': 'new', 'source_range': {'start': 1, 'end': 2},
                 'basis': 'source_inspection'}]

    contained = selectors()
    contained[0]['source_range'] = {'start': 2, 'end': 2}
    canonical = canonicalize_source_selectors(contained, [assigned], revision=revision, source_root=repo)
    assert canonical == [{**assigned, 'source_range': {'start': 2, 'end': 2}}, {
        'artifact_sha256': hashlib.sha256(supported.encode()).hexdigest(), 'part_sha256': None,
        'revision': revision['head_sha'], 'path': 'support.py', 'side': 'new',
        'source_range': {'start': 1, 'end': 2}, 'fragment': None,
        'basis': 'source_inspection', 'role': 'supporting',
    }]
    assert (repo / 'support.py').read_text() == 'MUTABLE_WORKTREE_CONTENT\n'
    # A submission-scoped resolver reuses one pinned read and is revoked before installation.
    from daydream.review_source_refs import _SubmissionSources
    sources = _SubmissionSources(revision, repo, {}, None)
    twice = [*contained, {**contained[1], 'source_range': {'start': 2, 'end': 2}}]
    assert sources.canonicalize(twice, [assigned])[:2] == canonical
    sources.close()
    for revoked in (lambda: sources.canonicalize(contained, [assigned]), lambda: sources._supporting('new', 'x')):
        with pytest.raises(RuntimeError, match='resolver has closed'):
            revoked()
    # Non-native admission revalidates the same pinned blob and rejects unavailable or out-of-range support.
    supporting = canonical[1]
    for reference, admitted in ((supporting, True), ({**supporting, 'source_range': {'start': 1, 'end': 99}}, False),
                                ({**supporting, 'path': 'missing.py'}, False),
                                ({**canonical[0], 'path': 'foreign.py'}, False)):
        assert validate_source_refs([canonical[0], reference], [assigned], revision=revision,
                                    source_root=repo) is admitted, reference
    for fault, expected in (('assigned-range', ('1..3', 'null')), ('supporting-range', ('1..2', 'start <= end')),
                            ('supporting-only', ('reviewed', 'source_id')), ('forged-hash', ()),
                            ('traversal', ()), ('binary', ()), ('symlink', ()),
                            ('oversized', ('whole pinned blob', '65536', 'narrowing source_range cannot'))):
        choices = selectors()
        if fault == 'assigned-range':
            choices[0]['source_range'] = {'start': 1, 'end': 4}
        elif fault == 'supporting-range':
            choices[1]['source_range'] = {'start': 2, 'end': 1}
        elif fault == 'supporting-only':
            choices.pop(0)
        elif fault == 'forged-hash':
            choices[1]['artifact_sha256'] = '0' * 64
        else:
            choices[1]['path'] = {'traversal': '../support.py', 'binary': 'binary.dat',
                                  'oversized': 'oversized.py', 'symlink': 'symlink.py'}[fault]
            # Keep the range valid for each blob so a missing size/UTF-8/regular-file
            # guard cannot pass this regression at an unrelated range check.
            choices[1]['source_range'] = {'start': 1, 'end': 1}
        with pytest.raises(HostToolRejected) as rejected:
            canonicalize_source_selectors(choices, [assigned], revision=revision, source_root=repo)
        message = str(rejected.value)
        assert 'evidence_refs' in message, fault
        assert any(word in message.lower() for word in ('choose', 'use', 'remove', 'supply')), fault
        assert all(part in message for part in expected), (fault, message)
    # A later assigned selector makes the last processed field differ from
    # the supporting reference whose real Git reread loses storage. Other
    # selector cases cover admission, but miss this diagnostic attribution.
    choices = [*selectors(), {'source_id': source_id(assigned), 'basis': 'supplied_diff', 'source_range': None}]
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


def test_stage_inputs_rebind_retained_decisions_and_refuse_changed_canonical_bytes(tmp_path: Path) -> None:
    """Restored judgments must cite this frozen assignment; changed canonical bytes stop preparation.

    Runner resume journeys restore only matching attribution, so the mismatch and late-change
    refusals are proved directly at the stage-input owner.
    """
    from daydream.backends.pi import PiBackend
    from daydream.deep.detection import StackAssignment
    from daydream.hunk_index import write_hunk_index
    from daydream.prompt_budget import SanctionedInputUnavailable
    from daydream.review_stage_inputs import StageInputFactory
    from tests.harness.git_helpers import work_context

    repo = tmp_path / 'repo'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'}, feature={'api.py': 'VALUE = 1\n'})
    root = tmp_path / 'inputs'
    root.mkdir()
    base, head = git(repo, 'rev-parse', 'main'), git(repo, 'rev-parse', 'HEAD')
    patch = git_ops.diff(repo, base, head)
    diff_path = root / 'diff.patch'
    diff_path.write_text(patch)
    revision = {'head_sha': head, 'merge_base_sha': base, 'diff_key': hashlib.sha256(patch.encode()).hexdigest()}
    factory = StageInputFactory(PiBackend(model='unexecuted'), work_context(repo, run_id='inputs'),
        StackAssignment('python', ['api.py']), diff_path=diff_path, hunk_index_path=write_hunk_index(root, patch),
        shared_paths={}, revision=revision, artifact_session=None, allow_standalone=True, read_only=True)
    part = factory.assignment_batches[0][0]
    reference = next(ref for ref in part['source_refs'] if ref['side'] == 'new')
    decision = {'target_id': part['target_id'], 'status': 'reviewed', 'reason': '', 'evidence_refs': [reference]}
    factory.validate_retained_decisions([decision], complete=True)
    with pytest.raises(ValueError, match='differ from frozen assignment inventory'):
        factory.validate_retained_decisions([], complete=True)
    forged = {**decision, 'evidence_refs': [{**reference, 'part_sha256': '0' * 64}]}
    with pytest.raises(ValueError, match='source attribution differs'):
        factory.validate_retained_decisions([forged], complete=True)
    diff_path.write_text(patch + '# changed after capture\n')
    with pytest.raises(SanctionedInputUnavailable, match='changed after capture'):
        factory.prepare({'stage': 'triage', 'assigned_target_ids': []})
