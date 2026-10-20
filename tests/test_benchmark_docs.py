"""Keep private benchmark instructions free of credentials and private PR identities."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "benchmark.md"


def test_privacy_placeholders_only() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "OWNER/REPO" in text
    assert re.search(r"sk-[A-Za-z0-9]", text) is None
    assert re.search(r"ghp_[A-Za-z0-9]", text) is None
    assert re.search(r"github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/pull/\d+", text) is None

def test_harbor_env_policy_docs_point_at_the_declaration() -> None:
    """S2: the docs name the single source without restating the sets."""
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "daydream/benchmark/harbor/env_policy.py" in runbook
    guidance = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "daydream/benchmark/harbor/env_policy.py" in guidance

