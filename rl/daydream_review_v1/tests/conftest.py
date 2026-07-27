"""Shared fixtures: a deterministic fixture git repo and a stub upstream server.

The fixture repo's history is byte-for-byte reproducible (fixed author/committer
identity, fixed timestamps, fixed tree contents, no gpg signing), so its commit
SHAs are constants. ``tests/fixtures/corpus-mini/`` pins exactly those SHAs, and
``images/manifest.toml`` pins them too — one repo, one set of SHAs, everywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

FIXTURE_SLUG = "existential-birds/daydream-rl-fixture"

# Commit SHAs of the deterministic history built by build_fixture_repo().
# Asserted in test_fixture_repo_is_deterministic; if git's object format ever
# changes these fail loudly rather than drifting silently.
FIXTURE_BASE_SHA = "f1f3ba6e1ca9af4ee1dd29939bf65885a98305c0"
FIXTURE_PR1_HEAD_SHA = "24399731a2119b7d5c02a7c55cbd4c644717174a"
FIXTURE_PR2_HEAD_SHA = "0cfd97ece2d3bb19e4fa805f3faaae68ce70a9b4"

FIXTURE_TEST_COMMAND = "python -m unittest discover -q"

_IDENTITY = {
    "GIT_AUTHOR_NAME": "Daydream Fixture",
    "GIT_AUTHOR_EMAIL": "fixture@daydream.invalid",
    "GIT_COMMITTER_NAME": "Daydream Fixture",
    "GIT_COMMITTER_EMAIL": "fixture@daydream.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}

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

_README = """# daydream-rl-fixture

A deterministic throwaway repository used by the `daydream-review-v1` verifiers
environment's tests and by its fixture container image. Never published.
"""


@dataclass(frozen=True)
class FixtureRepo:
    """A built fixture repository and the SHAs of its three commits."""

    path: Path
    base_sha: str
    pr1_head_sha: str
    pr2_head_sha: str


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, **_IDENTITY}
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
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
        dest: Directory to create the repository in. Created if absent.
        red: Plant a failing assertion in the final commit's test file. Used to
            prove the image build's green-baseline gate actually fails.

    Returns:
        The built repository and its commit SHAs.
    """
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, "init", "--quiet", "--initial-branch", "main")
    _git(dest, "config", "commit.gpgsign", "false")

    base = _commit(
        dest,
        {"README.md": _README, "calc.py": _CALC_V1, "tests/__init__.py": "", "tests/test_calc.py": _TESTS_V1},
        "Initial commit: calc.add",
    )
    pr1 = _commit(dest, {"calc.py": _CALC_V2, "tests/test_calc.py": _TESTS_V2}, "Add calc.divide")
    pr2 = _commit(
        dest,
        {"calc.py": _CALC_V3, "tests/test_calc.py": _TESTS_RED if red else _TESTS_V3},
        "Add calc.mean",
    )
    return FixtureRepo(path=dest, base_sha=base, pr1_head_sha=pr1, pr2_head_sha=pr2)


@pytest.fixture
def fixture_repo(tmp_path: Path) -> FixtureRepo:
    """A freshly built deterministic fixture repository."""
    return build_fixture_repo(tmp_path / "daydream-rl-fixture")


@pytest.fixture(scope="session")
def corpus_mini_dir() -> Path:
    """The committed 2-PR harvested-format corpus over the fixture repo."""
    return Path(__file__).parent / "fixtures" / "corpus-mini"


@pytest.fixture(scope="session")
def fixture_manifest_path() -> Path:
    """The committed images manifest, which carries the fixture repo entry."""
    return Path(__file__).parent.parent / "images" / "manifest.toml"


class _StubUpstreamHandler(BaseHTTPRequestHandler):
    """Answer any Chat Completions / Responses / Messages POST with canned text."""

    reply: str = "I have no findings."

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = json.dumps(
            {
                "id": "stub-1",
                "object": "chat.completion",
                "created": 0,
                "model": "stub/canned",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_upstream() -> Iterator[str]:
    """Run a canned OpenAI-compatible upstream; yield its ``/v1`` base URL."""
    server = HTTPServer(("127.0.0.1", 0), _StubUpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
