"""The deterministic throwaway repository the tests and the fixture image share.

Its history is byte-for-byte reproducible — fixed author and committer identity,
fixed timestamps, fixed tree contents, no signing — so its commit SHAs are
constants. ``tests/fixtures/corpus-mini/`` pins exactly those SHAs, and so does
``images/manifest.toml``: one repository, one set of SHAs, everywhere.

Lives in the package rather than in ``conftest.py`` because three callers need
it: the test suite, ``images/build_images.py`` (the ``fixture://`` clone-URL
sentinel), and anyone staging the local smoke rollout by hand.

    python -m daydream_review_v1.fixture /tmp/daydream-rl-smoke/repo
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FIXTURE_SLUG = "existential-birds/daydream-rl-fixture"

FIXTURE_TEST_COMMAND = "python -m unittest discover -q"
"""Stdlib only, so the image needs no dependency install and no network."""

# Commit SHAs of the history build_fixture_repo() produces. Asserted by
# tests/test_taskset.py::test_fixture_repo_is_deterministic.
FIXTURE_BASE_SHA = "a225d61f1ada3bd03f06cdf8a3f3f2d00870f6c5"
FIXTURE_PR1_HEAD_SHA = "1ba756e6743d833dc361177c2fb4946e76015985"
FIXTURE_PR2_HEAD_SHA = "9b92381663058612621b186545f91bfb3a54079c"

_IDENTITY = {
    "GIT_AUTHOR_NAME": "Daydream Fixture",
    "GIT_AUTHOR_EMAIL": "fixture@daydream.invalid",
    "GIT_COMMITTER_NAME": "Daydream Fixture",
    "GIT_COMMITTER_EMAIL": "fixture@daydream.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}

# Every real repository ignores its build artifacts, and so must this one:
# `capture_recommended_patch` appends a creation hunk for each untracked,
# NON-ignored file (daydream/git_ops.py:842 -> list_untracked, which passes
# --exclude-standard). Without this file, stray .pyc bytecode written during the
# review would land in recommended.patch and make an empty fix look like a real
# one — the exact signal fix_tests_pass gates on.
_GITIGNORE = """__pycache__/
*.py[cod]
.daydream/
"""

_README = """# daydream-rl-fixture

A deterministic throwaway repository used by the `daydream-review-v1` verifiers
environment's tests and by its fixture container image. Never published.
"""

_CALC_V1 = '''"""Tiny arithmetic helpers used by the daydream RL fixture repo."""


def add(a: int, b: int) -> int:
    """Return the sum of *a* and *b*."""
    return a + b
'''

_CALC_V2 = _CALC_V1 + '''

def divide(a: int, b: int) -> float:
    """Return *a* divided by *b*."""
    return a / b
'''

_CALC_V3 = _CALC_V2 + '''

def mean(values: list[int]) -> float:
    """Return the arithmetic mean of *values*."""
    return sum(values) / len(values)
'''

_TESTS_V1 = '''import unittest

from calc import add


class TestCalc(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(2, 2), 4)


if __name__ == "__main__":
    unittest.main()
'''

_TESTS_V2 = '''import unittest

from calc import add, divide


class TestCalc(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(2, 2), 4)

    def test_divide(self) -> None:
        self.assertEqual(divide(6, 3), 2.0)


if __name__ == "__main__":
    unittest.main()
'''

_TESTS_V3 = '''import unittest

from calc import add, divide, mean


class TestCalc(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(2, 2), 4)

    def test_divide(self) -> None:
        self.assertEqual(divide(6, 3), 2.0)

    def test_mean(self) -> None:
        self.assertEqual(mean([1, 2, 3]), 2.0)


if __name__ == "__main__":
    unittest.main()
'''

_TESTS_RED = _TESTS_V3.replace("self.assertEqual(add(2, 2), 4)", "self.assertEqual(add(2, 2), 5)")


@dataclass(frozen=True)
class FixtureRepo:
    """A built fixture repository and the SHAs of its three commits."""

    path: Path
    base_sha: str
    pr1_head_sha: str
    pr2_head_sha: str


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **_IDENTITY},
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for rel, content in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--no-gpg-sign", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def build_fixture_repo(dest: Path, *, red: bool = False) -> FixtureRepo:
    """Build the deterministic three-commit fixture repository at *dest*.

    Args:
        dest: New or empty directory to create the repository in; created if
            absent. An occupied destination (an existing file or a non-empty
            directory) is rejected before any mutation.
        red: Plant a failing assertion in the final commit's test file, to prove
            the image build's green-baseline gate actually fails.

    Returns:
        The built repository and its commit SHAs.
    """
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ValueError(f"fixture destination must be a new or empty directory: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, "init", "--quiet", "--initial-branch", "main")
    _git(dest, "config", "commit.gpgsign", "false")

    base = _commit(
        dest,
        {
            ".gitignore": _GITIGNORE,
            "README.md": _README,
            "calc.py": _CALC_V1,
            "tests/__init__.py": "",
            "tests/test_calc.py": _TESTS_V1,
        },
        "Initial commit: calc.add",
    )
    pr1 = _commit(dest, {"calc.py": _CALC_V2, "tests/test_calc.py": _TESTS_V2}, "Add calc.divide")
    pr2 = _commit(
        dest,
        {"calc.py": _CALC_V3, "tests/test_calc.py": _TESTS_RED if red else _TESTS_V3},
        "Add calc.mean",
    )
    return FixtureRepo(path=dest, base_sha=base, pr1_head_sha=pr1, pr2_head_sha=pr2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path, help="directory to build the repository in")
    parser.add_argument("--red", action="store_true", help="plant a failing test in the head commit")
    args = parser.parse_args(argv)
    repo = build_fixture_repo(args.dest, red=args.red)
    print(f"repo       {repo.path}")
    print(f"base_sha   {repo.base_sha}")
    print(f"pr1_head   {repo.pr1_head_sha}")
    print(f"pr2_head   {repo.pr2_head_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
