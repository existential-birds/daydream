"""Invariant tests for .github/dependabot.yml (issue #936).

The config's real acceptance happens GitHub-side (Dependency graph registration,
first grouped PRs). These tests pin the structural invariants a careless edit
could silently break: exactly four ecosystems at the exact directories, weekly
Monday schedule on one timezone, PR limit 5, grouping on the three
multi-dependency blocks, and the reviewer-facing docs that make the
config's volume bounds discoverable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEPENDABOT_PATH = _REPO_ROOT / ".github" / "dependabot.yml"
WORKFLOWS_README = _REPO_ROOT / ".github" / "workflows" / "README.md"


def load_dependabot() -> dict[str, Any]:
    loaded = yaml.safe_load(DEPENDABOT_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), "dependabot.yml did not parse to a mapping"
    return loaded


def updates(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return doc["updates"]


def test_four_ecosystems_at_exact_directories() -> None:
    blocks = {(u["package-ecosystem"], u["directory"]) for u in updates(load_dependabot())}
    assert blocks == {
        ("uv", "/"),
        ("uv", "/rl/daydream_review_v1"),
        ("github-actions", "/"),
        ("npm", "/.github/workflows"),  # the npm manifest trick — must NOT be "/"
    }


def test_weekly_monday_one_timezone_limit_five_everywhere() -> None:
    for u in updates(load_dependabot()):
        sched = u["schedule"]
        assert sched["interval"] == "weekly"
        assert sched["day"] == "monday"
        assert sched["timezone"] == "Australia/Brisbane"
        assert u["open-pull-requests-limit"] == 5


def test_grouping_on_multi_dependency_blocks_only() -> None:
    by_eco = {u["package-ecosystem"]: u for u in updates(load_dependabot())}
    for eco in ("uv", "github-actions"):
        group = by_eco[eco]["groups"]
        assert list(group) == ["dependencies"], f"{eco} needs a single group"
        pattern = group["dependencies"]["patterns"]
        assert "*" in pattern, f"{eco} group must cover all deps (minor+patch together)"
    assert "groups" not in by_eco["npm"], "npm tracks a single dep — no group"


def test_commit_message_prefix_everywhere() -> None:
    for u in updates(load_dependabot()):
        assert u["commit-message"]["prefix"] == "chore(deps)"


def test_config_documents_ownership_and_gates() -> None:
    header = DEPENDABOT_PATH.read_text(encoding="utf-8").split("updates:", 1)[0]
    assert "CODEOWNERS" in header, "config comment must point at reviewer ownership"
    assert "make check" in header, "config comment must name the validation gate"
