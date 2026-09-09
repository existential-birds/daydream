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

from daydream.benchmark.cli import _build_benchmark_parser

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "benchmark.md"
README = ROOT / "README.md"
CLAUDE = ROOT / "CLAUDE.md"

# The 12 shipped subcommands (MH-3), matching _build_benchmark_parser()
# in daydream/benchmark/cli.py:469-613.
EXPECTED_SUBCOMMANDS = {
    "init", "status", "validate", "build-harbor", "upgrade", "import-prs",
    "curate", "calibrate-judge", "run", "clean", "objective", "aggregate",
}

# Legacy runtime tokens MH-2 forbids as active instructions in the runbook.
FORBIDDEN_LEGACY_TOKENS = ("martian", "MARTIAN", "CodeRabbit", "anthropic-direct")


def _parser_choices() -> set[str]:
    subparsers = _build_benchmark_parser()._subparsers
    if subparsers is None:
        raise AssertionError("benchmark parser has no subparsers")
    choices = subparsers._group_actions[0].choices
    if choices is None:
        raise AssertionError("benchmark parser subcommands are empty")
    return set(choices)


def _code_lines(text: str) -> list[str]:
    """Nonblank, non-comment lines inside fenced ```bash/```sh blocks."""
    out: list[str] = []
    in_block = False
    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            line = raw.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def _json_blocks(text: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"```json\n(.*?)```", text, re.S)
    return [json.loads(b) for b in blocks if b.strip()]


def _objective_example() -> dict[str, Any]:
    for block in _json_blocks(RUNBOOK.read_text(encoding="utf-8")):
        if "objective" in block and "identity" in block:
            return block
    pytest.fail("runbook has no ```json objective example with an identity block")


def _suite_manifest_example() -> dict[str, Any]:
    for block in _json_blocks(RUNBOOK.read_text(encoding="utf-8")):
        if "entries" in block:
            return block
    pytest.fail("runbook has no ```json suite manifest example with entries")


# --- CLI / runbook command set (MH-3, MH-15) ---

def test_privacy_placeholders_only() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "OWNER/REPO" in text
    assert re.search(r"sk-[A-Za-z0-9]", text) is None
    assert re.search(r"ghp_[A-Za-z0-9]", text) is None
    assert re.search(r"github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/pull/\d+", text) is None


# --- Objectives & suites (MH-9, MH-10) ---

def test_objective_example_is_privacy_safe() -> None:
    blob = json.dumps(_objective_example())
    assert "OWNER" not in blob and "github.com" not in blob
    assert not re.search(r"pull/\d+|pr-\d", blob)


