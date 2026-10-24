"""Frozen source and supplied case membership."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream import git_ops

REVIEWBENCH_REVISION = '4f04392d6bf4941ab6f8a3ea0c15b449f4458864'
REVIEWBENCH_REPOSITORY = 'https://github.com/review-bench/ReviewBench.git'
PUBLIC_PROFILE_SCHEMA = 4


def canonical_digest(value: Any) -> str:
    """Digest JSON policy and identities, never credential contents."""
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Case:
    """Supplied PR identity with ReviewBench's prKey spelling."""

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


def verify_source(source: Path) -> dict[str, str]:
    """Require pinned source and unchanged benchmark inputs."""
    if git_ops.head_sha(source) != REVIEWBENCH_REVISION:
        raise ValueError(f'ReviewBench source must be pinned at {REVIEWBENCH_REVISION}')
    if git_ops.changed_paths_z(source, 'HEAD', include_untracked=False):
        raise ValueError('ReviewBench source is modified')
    tracked = set(git_ops.ls_files(source, strict=True))
    if set(git_ops.search_files(source, ['scripts'], include_ignored=True)) - tracked:
        raise ValueError('ReviewBench scripts contain untracked inputs')
    return {'repository': REVIEWBENCH_REPOSITORY, 'revision': REVIEWBENCH_REVISION}


def load_cases(source: Path, selection: str) -> list[Case]:
    """Load supplied membership from a verified source checkout."""
    if selection not in {'test', 'full'}:
        raise ValueError('ReviewBench set must be test or full')
    filename = 'corpus/test/test.json' if selection == 'test' else 'corpus/manifest.json'
    raw = json.loads((source / filename).read_text())
    if not isinstance(raw, list):
        raise ValueError('ReviewBench set must be an array')
    cases = [Case.from_dict(value) for value in raw]
    if len({case.key for case in cases}) != len(cases):
        raise ValueError('ReviewBench supplied task keys are duplicated')
    return cases
