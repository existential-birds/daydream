"""Frozen source, supplied case membership and independent execution accounting."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.git_ops.process import _run_git

REVIEWBENCH_REVISION = '4f04392d6bf4941ab6f8a3ea0c15b449f4458864'
REVIEWBENCH_REPOSITORY = 'https://github.com/review-bench/ReviewBench.git'


@dataclass(frozen=True)
class Case:
    """Full supplied PR identity; key spelling is ReviewBench's prKey contract."""

    repo: str
    pr_number: int
    base: str
    head: str

    def __post_init__(self) -> None:
        if (not isinstance(self.repo, str)
                or not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', self.repo)
                or type(self.pr_number) is not int or self.pr_number < 1
                or not all(isinstance(sha, str) and re.fullmatch('[a-f0-9]{40}', sha)
                           for sha in (self.base, self.head))):
            raise ValueError('invalid full ReviewBench PR identity')

    @property
    def key(self) -> str:
        return f'{self.repo.removeprefix("https://github.com/").replace("/", "_")}_{self.pr_number}-{self.head[:8]}'

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Case:
        try:
            return cls(**{key: value[key] for key in ('repo', 'pr_number', 'base', 'head')})
        except (KeyError, TypeError) as exc:
            raise ValueError('missing or malformed ReviewBench PR identity') from exc


def file_digest(path: Path) -> str:
    """Hash actual immutable input bytes, not paths or labels."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source: Path) -> dict[str, Any]:
    """Refuse another revision or modified executable/reference source."""
    if git_ops.head_sha(source) != REVIEWBENCH_REVISION:
        raise ValueError(f'ReviewBench source must be pinned at {REVIEWBENCH_REVISION}')
    status = _run_git(source, ['status', '--porcelain', '--untracked-files=all'],
                      error_context='ReviewBench status failed')
    if status.stdout:
        raise ValueError('ReviewBench source is modified or contains untracked inputs')
    paths = ('corpus/manifest.json', 'corpus/test/test.json', 'package-lock.json',
             'scripts/eval/scorer.ts', 'scripts/eval/print-summary.ts', 'scripts/eval/eval.ts',
             'scripts/eval/matcher.ts', 'scripts/classifier/prompts.ts')
    return {'repository': REVIEWBENCH_REPOSITORY, 'revision': REVIEWBENCH_REVISION,
            'hashes': {path: file_digest(source / path) for path in paths}}


def load_cases(source: Path, selection: str) -> list[Case]:
    """Use actual supplied starter/full membership, never judge --limit."""
    if selection not in {'test', 'full'}:
        raise ValueError('ReviewBench set must be test or full')
    filename = 'corpus/test/test.json' if selection == 'test' else 'corpus/manifest.json'
    raw = json.loads((source / filename).read_text())
    if not isinstance(raw, list):
        raise ValueError('ReviewBench set must be an array')
    cases = [Case.from_dict(value) for value in raw]
    if len(cases) != (25 if selection == 'test' else 219) or len({case.key for case in cases}) != len(cases):
        raise ValueError('ReviewBench supplied set membership is incomplete or duplicated')
    if selection == 'test':
        full = {case.key: case for case in load_cases(source, 'full')}
        if any(full.get(case.key) != case for case in cases):
            raise ValueError('starter identity differs from full manifest')
    return cases


def execution_accounting(expected: list[str], outcomes: Mapping[str, Mapping[str, Any]],
                         produced: list[str], evaluated: list[str] | None = None) -> dict[str, Any]:
    """Keep omitted work visible independently of upstream overlap-only scoring."""
    if len(expected) != len(set(expected)) or set(outcomes) - set(expected) or set(produced) - set(expected):
        raise ValueError('unexpected or duplicate case membership')
    completed = [key for key in expected if outcomes.get(key, {}).get('status') in {'complete', 'completed_empty'}]
    failed = [key for key in expected if key in outcomes and key not in completed]
    missing = [key for key in expected if key not in outcomes]
    missing_artifacts = [key for key in completed if key not in produced]
    conversion_limited = [key for key in expected if outcomes.get(key, {}).get('status') == 'conversion_limited']
    omitted_findings = sum(len(outcome.get('conversion_omissions', [])) for outcome in outcomes.values())
    scored = evaluated if evaluated is not None else []
    if len(scored) != len(set(scored)) or set(scored) - set(produced):
        raise ValueError('unexpected evaluated case membership')
    accounting = {'expected': expected, 'produced': produced, 'completed': completed,
                  'completed_empty': [key for key in completed if outcomes[key]['status'] == 'completed_empty'],
                  'failed': failed, 'missing': missing, 'missing_artifacts': missing_artifacts, 'evaluated': scored,
                  'conversion_limited': conversion_limited}
    return {**accounting, 'counts': {**{key: len(values) for key, values in accounting.items()},
                                    'conversion_omitted_findings': omitted_findings},
            'complete_generation': not failed and not missing and set(produced) == set(expected),
            'complete_evaluation': not failed and not missing and set(scored) == set(expected)}
