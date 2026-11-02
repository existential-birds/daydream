"""Pinned ReviewBench source and supplied task identities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from daydream.benchmark.reviewbench import contract
from daydream.benchmark.reviewbench.contract import Case, load_cases, verify_source
from tests.harness.git_helpers import commit, init_repo, write_and_stage


def _supplied(source: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    init_repo(source)
    entries = [asdict(Case('https://github.com/example/project', number, 'a' * 40, 'b' * 40))
               for number in (1, 2)]
    write_and_stage(source, 'corpus/manifest.json', json.dumps(entries))
    write_and_stage(source, 'corpus/test/test.json', json.dumps(entries[1:]))
    write_and_stage(source, 'scripts/eval/scorer.ts', 'synthetic input\n')
    monkeypatch.setattr(contract, 'REVIEWBENCH_REVISION', commit(source, 'synthetic upstream contract'))
    return source


def test_source_pin_refuses_changed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for fault in ('revision', 'modified', 'shadow-module'):
        supplied_source = _supplied(tmp_path / fault, monkeypatch)
        assert verify_source(supplied_source)['revision'] == contract.REVIEWBENCH_REVISION
        if fault == 'revision':
            write_and_stage(supplied_source, 'new', 'other revision')
            commit(supplied_source, 'different revision')
        elif fault == 'modified':
            (supplied_source / 'scripts/eval/scorer.ts').write_text('changed input')
        else:
            (supplied_source / 'scripts/eval/input-validation.js').write_text('{}')
        with pytest.raises(ValueError, match='pinned|modified|untracked|scripts'):
            verify_source(supplied_source)


def test_supplied_task_keys_are_unique(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    supplied_source = _supplied(tmp_path / 'supplied', monkeypatch)
    assert [case.pr_number for case in load_cases(supplied_source, 'full')] == [1, 2]
    assert [case.pr_number for case in load_cases(supplied_source, 'test')] == [2]
    path = supplied_source / 'corpus/test/test.json'
    values = json.loads(path.read_text())
    values.append(values[0])
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='duplicated'):
        load_cases(supplied_source, 'test')
