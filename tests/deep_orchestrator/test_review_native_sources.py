"""Ordinary file inputs keep useful before-side context for file-only backends."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.deep import orchestrator
from daydream.git_ops import queries
from daydream.git_ops.models import GitError
from daydream.host_tools import HostToolFatal, HostToolRejected
from daydream.review_source_refs import (
    canonicalize_source_selectors,
    source_id,
    validate_source_refs,
)
from tests.deep_orchestrator.test_review_completion import ReviewRun
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


def test_stage_inputs_isolate_scopes_and_refuse_changed_canonical_bytes(tmp_path: Path) -> None:
    """Scopes share one immutable change without sharing mutable inputs; changed canonical bytes stop dispatch.

    The reviewer revalidates the canonical change before every dispatch, so the late-change refusal is
    proved directly at the stage-input owner.
    """
    from daydream.backends.pi import PiBackend
    from daydream.deep.detection import StackAssignment
    from daydream.hunk_index import write_hunk_index
    from daydream.prompt_budget import SanctionedInputUnavailable
    from daydream.review_stage_inputs import CanonicalChange, StageInputFactory
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
    change = CanonicalChange.capture(diff_path, write_hunk_index(root, patch))

    def scope_inputs() -> StageInputFactory:
        return StageInputFactory(PiBackend(model='unexecuted'), work_context(repo, run_id='inputs'),
            StackAssignment('python', ['api.py']), change=change, shared_paths={},
            revision=revision, artifact_session=None, allow_standalone=True, read_only=True)

    factory, sibling = scope_inputs(), scope_inputs()
    pristine = copy.deepcopy((factory.parts[0].assignment, factory.index, factory._source_aliases))
    sibling.parts[0].assignment['source_refs'].clear()
    sibling.index['api.py']['hunks'].clear()
    sibling._source_aliases.clear()
    assert (factory.parts[0].assignment, factory.index, factory._source_aliases) == pristine
    with pytest.raises(TypeError):
        change.index_entries['api.py'] = '{}'  # type: ignore[index]
    diff_path.write_text(patch + '# changed after capture\n')
    with pytest.raises(SanctionedInputUnavailable, match='changed after capture'):
        factory._revalidate_canonical()


@pytest.mark.parametrize(('lines', 'to_symlink'), [(3, True), (3000, True), (3000, False)],
                         ids=['whole-file', 'split', 'split-to-regular'])
def test_type_changed_path_keeps_old_side_refs_under_each_block_header(
    tmp_path: Path, lines: int, to_symlink: bool,
) -> None:
    """A type change is a deletion block then an addition block for one path.

    Removed lines stay citable on the old side, and a split part carries only its own block's header,
    whether the oversized block comes first (regular file to symlink) or last (symlink to regular file).
    """
    from daydream.backends.pi import PiBackend
    from daydream.deep.detection import StackAssignment
    from daydream.hunk_index import write_hunk_index
    from daydream.review_stage_inputs import CanonicalChange, StageInputFactory
    from tests.harness.git_helpers import commit, init_repo, work_context

    repo = tmp_path / 'repo'
    init_repo(repo)
    (repo / 'target').write_text('t\n')
    regular = ''.join(f'line {index} ' + 'y' * 40 + '\n' for index in range(lines))
    for side, symlink in (('base', not to_symlink), ('type change', to_symlink)):
        (repo / 'x').unlink(missing_ok=True)
        if symlink:
            (repo / 'x').symlink_to('target')
        else:
            (repo / 'x').write_text(regular)
        git(repo, 'add', '.')
        head = commit(repo, side)
    base = git(repo, 'rev-parse', 'HEAD~1')
    patch = git_ops.diff(repo, base, head)
    root = tmp_path / 'inputs'
    root.mkdir()
    diff_path = root / 'diff.patch'
    diff_path.write_text(patch)
    revision = {'head_sha': head, 'merge_base_sha': base, 'diff_key': hashlib.sha256(patch.encode()).hexdigest()}
    factory = StageInputFactory(PiBackend(model='unexecuted'), work_context(repo, run_id='type-change'),
        StackAssignment('python', ['x']), change=CanonicalChange.capture(diff_path, write_hunk_index(root, patch)),
        shared_paths={}, revision=revision, artifact_session=None, allow_standalone=True, read_only=True)
    refs = [{(ref['side'], ref['path']) for ref in part.assignment['source_refs']} for part in factory.parts]
    if lines == 3:
        assert refs == [{('old', 'x'), ('new', 'x')}]
        return
    headers = [[line for line in part.diff.splitlines() if line.startswith(('diff --git', 'new file', 'deleted file'))]
               for part in factory.parts]
    deleted = [['diff --git a/x b/x', 'deleted file mode ' + ('100644' if to_symlink else '120000')]]
    added = [['diff --git a/x b/x', 'new file mode ' + ('120000' if to_symlink else '100644')]]
    split = len(factory.parts) - 1
    assert split > 1
    if to_symlink:
        assert (headers, refs) == (deleted * split + added, [{('old', 'x')}] * split + [{('new', 'x')}])
    else:
        assert (headers, refs) == (deleted + added * split, [{('old', 'x')}] + [{('new', 'x')}] * split)
