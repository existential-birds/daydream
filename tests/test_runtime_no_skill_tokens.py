"""M24: scan active runtime/help text for skill-invocation tokens."""

import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


HELP_TOKENS = re.compile(
    r"\b(beagle|SKILL\.md|--skill|--no-skills|/skill:|override_skill|"
    r"format_skill_invocation|pr-feedback|fetch-pr-feedback)\b",
    re.IGNORECASE,
)


def test_runtime_help_has_no_skill_tokens(repo_root: Path) -> None:
    """M24: active runtime/help text carries no Beagle installation or skill-invocation guidance."""
    r = subprocess.run(
        [sys.executable, "-m", "daydream", "--help-all"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=60,
    )
    combined = r.stdout + r.stderr
    hits = HELP_TOKENS.findall(combined)
    assert not hits, f"runtime help contains skill tokens: {sorted(set(hits))}"
