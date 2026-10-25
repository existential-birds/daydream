"""Ordinary file inputs keep useful before-side context for file-only backends."""
from __future__ import annotations

import copy
import hashlib
import json
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from daydream.git_ops.models import GitError
from daydream.git_ops.source import frozen_source
from tests.deep_orchestrator.test_review_completion import record, scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, StagedBackend, candidate
from tests.harness.git_helpers import git, seed_feature_branch, tracked_source_state
from tests.harness.table_cases import isolated_case


class PiBackend(StagedBackend):
    """File-only transport name activates the runner's ordinary before-side input path."""


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
    review.backend = PiBackend(repo)
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


async def _case_before_source_is_furnished_as_a_private_ordinary_file_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, broken: bool
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
    review.backend = PiBackend(repo)
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
                target['evidence_refs'] = copy.deepcopy(offered)
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


async def test_before_source_is_furnished_as_a_private_ordinary_file_input(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[str] = ["deleted", "renamed", "pure-rename"]
    cases_1: Sequence[bool] = [False, True]
    case_index = 0
    for change in cases_0:
        for broken in cases_1:
            with subtests.test(change=change, broken=broken), isolated_case(tmp_path, str(case_index)) as scenario:
                case_index += 1
                await _case_before_source_is_furnished_as_a_private_ordinary_file_input(
                    tmp_path=scenario.tmp_path, monkeypatch=scenario.monkeypatch, change=change, broken=broken
                )
