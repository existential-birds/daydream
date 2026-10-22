"""Frozen supplied source and exact full-identity membership contracts."""

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
               for number in range(1, 220)]
    write_and_stage(source, 'corpus/manifest.json', json.dumps(entries))
    write_and_stage(source, 'corpus/test/test.json', json.dumps(entries[:25]))
    for filename in ('package-lock.json', 'scripts/eval/scorer.ts', 'scripts/eval/print-summary.ts',
                     'scripts/eval/eval.ts', 'scripts/eval/matcher.ts', 'scripts/classifier/prompts.ts'):
        write_and_stage(source, filename, 'synthetic input\n')
    monkeypatch.setattr(contract, 'REVIEWBENCH_REVISION', commit(source, 'synthetic upstream contract'))
    return source


@pytest.mark.parametrize('fault', ['revision', 'modified', 'untracked', 'hidden-untracked', 'hidden-whitespace'])
def test_source_pin_refuses_changed_inputs(supplied_source: Path, fault: str) -> None:
    """Pin evaluator bytes even when Git configuration hides untracked inputs."""
    assert verify_source(supplied_source)['hashes']['scripts/eval/scorer.ts']
    if fault == 'revision':
        write_and_stage(supplied_source, 'new', 'other revision')
        commit(supplied_source, 'different revision')
    elif fault == 'modified':
        (supplied_source / 'scripts/eval/scorer.ts').write_text('changed metrics')
    else:
        path = supplied_source / 'extra-reference.json'
        if fault.startswith('hidden'):
            git(supplied_source, 'config', 'status.showUntrackedFiles', 'no')
            path = (supplied_source / '   ' if fault == 'hidden-whitespace'
                    else supplied_source / 'nested' / 'extra-reference.json')
            path.parent.mkdir(exist_ok=True)
        path.write_text('{}')
    with pytest.raises(ValueError, match='pinned|modified|untracked'):
        verify_source(supplied_source)


@pytest.mark.parametrize('fault', ['duplicate', 'subset', 'identity'])
def test_supplied_membership_is_exact(supplied_source: Path, fault: str) -> None:
    """Select exact supplied membership; evaluator tests start with selected cases."""
    assert len(load_cases(supplied_source, 'full')) == 219
    starter = load_cases(supplied_source, 'test')
    assert len(starter) == 25 and starter[-1].pr_number == 25
    path = supplied_source / 'corpus/test/test.json'
    values = json.loads(path.read_text())
    if fault == 'duplicate':
        values[-1] = values[0]
    elif fault == 'subset':
        values.pop()
    else:
        values[-1]['base'] = 'c' * 40
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='membership|identity'):
        load_cases(supplied_source, 'test')
