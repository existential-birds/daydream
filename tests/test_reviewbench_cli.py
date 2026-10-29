"""Pinned ReviewBench source and supplied task identities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from daydream.benchmark.reviewbench import contract
from daydream.benchmark.reviewbench.contract import Case, load_cases, verify_source
from tests.harness.git_helpers import commit, git, init_repo, write_and_stage


@pytest.fixture
def supplied_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / 'supplied'
    init_repo(source)
    entries = [asdict(Case('https://github.com/example/project', number, 'a' * 40, 'b' * 40))
               for number in (1, 2)]
    write_and_stage(source, 'corpus/manifest.json', json.dumps(entries))
    write_and_stage(source, 'corpus/test/test.json', json.dumps(entries[1:]))
    write_and_stage(source, 'scripts/eval/scorer.ts', 'synthetic input\n')
    monkeypatch.setattr(contract, 'REVIEWBENCH_REVISION', commit(source, 'synthetic upstream contract'))
    return source


@pytest.mark.parametrize('fault', [
    'revision', 'modified', 'manifest', 'staged', 'deleted',
    'untracked', 'shadow-module', 'ignored', 'hidden-untracked', 'hidden-whitespace',
])
def test_source_pin_refuses_changed_inputs(supplied_source: Path, fault: str) -> None:
    assert verify_source(supplied_source)['revision'] == contract.REVIEWBENCH_REVISION
    if fault == 'revision':
        write_and_stage(supplied_source, 'new', 'other revision')
        commit(supplied_source, 'different revision')
    elif fault in {'modified', 'manifest'}:
        filename = 'scripts/eval/scorer.ts' if fault == 'modified' else 'corpus/manifest.json'
        (supplied_source / filename).write_text('changed input')
    elif fault == 'staged':
        write_and_stage(supplied_source, 'scripts/eval/added.ts', 'new evaluator input')
    elif fault == 'deleted':
        (supplied_source / 'scripts/eval/scorer.ts').unlink()
    else:
        filename = {
            'untracked': 'scripts/extra.ts',
            'shadow-module': 'scripts/eval/input-validation.js',
            'ignored': 'scripts/eval/package.json',
            'hidden-untracked': 'scripts/nested/extra.ts',
            'hidden-whitespace': 'scripts/   ',
        }[fault]
        path = supplied_source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        if fault == 'ignored':
            (supplied_source / '.git/info/exclude').write_text(filename + '\n')
        if fault.startswith('hidden'):
            git(supplied_source, 'config', 'status.showUntrackedFiles', 'no')
    with pytest.raises(ValueError, match='pinned|modified|untracked|scripts'):
        verify_source(supplied_source)


def test_source_pin_allows_unrelated_untracked_files(supplied_source: Path) -> None:
    for filename in ('notes.txt', 'scratch/notes.txt', '   ', 'résumé.md'):
        path = supplied_source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('local notes')
    git(supplied_source, 'config', 'status.showUntrackedFiles', 'no')
    assert verify_source(supplied_source) == {
        'repository': contract.REVIEWBENCH_REPOSITORY, 'revision': contract.REVIEWBENCH_REVISION,
    }


def test_supplied_task_keys_are_unique(supplied_source: Path) -> None:
    assert [case.pr_number for case in load_cases(supplied_source, 'full')] == [1, 2]
    assert [case.pr_number for case in load_cases(supplied_source, 'test')] == [2]
    path = supplied_source / 'corpus/test/test.json'
    values = json.loads(path.read_text())
    values.append(values[0])
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='duplicated'):
        load_cases(supplied_source, 'test')
