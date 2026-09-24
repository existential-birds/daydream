"""Documentation-contract tests for the private PR Harbor benchmark runbook (issue #784).

Pins Daydream-owned docs/CLI/schema/privacy contracts only. No Harbor runtime,
Docker, or paid hosted-model command is ever executed here; paid commands
(calibrate-judge, run --oracle) are only asserted to be visibly marked as paid
gates and are never run in CI. Mirrors the existing doc-contract patterns in
tests/test_docs_contract.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "benchmark.md"


def _json_blocks(text: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"```json\n(.*?)```", text, re.S)
    return [json.loads(b) for b in blocks if b.strip()]


def _objective_example() -> dict[str, Any]:
    for block in _json_blocks(RUNBOOK.read_text(encoding="utf-8")):
        if "objective" in block and "identity" in block:
            return block
    pytest.fail("runbook has no ```json objective example with an identity block")


def test_privacy_placeholders_only() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "OWNER/REPO" in text
    assert re.search(r"sk-[A-Za-z0-9]", text) is None
    assert re.search(r"ghp_[A-Za-z0-9]", text) is None
    assert re.search(r"github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/pull/\d+", text) is None


def test_harbour_env_policy_docs_point_at_the_declaration() -> None:
    """S2: the docs name the single source without restating the sets."""
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "daydream/benchmark/harbor/env_policy.py" in runbook
    guidance = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "daydream/benchmark/harbor/env_policy.py" in guidance


# --- Objectives & suites (MH-9, MH-10) ---

def test_objective_example_is_privacy_safe() -> None:
    blob = json.dumps(_objective_example())
    assert "OWNER" not in blob and "github.com" not in blob
    assert not re.search(r"pull/\d+|pr-\d", blob)


