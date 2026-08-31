"""Invariant tests for .github/dependabot.yml (issue #936).

The config's real acceptance happens GitHub-side (Dependency graph registration,
first grouped PRs). These tests pin the structural invariants a careless edit
could silently break: exactly four ecosystems at the exact directories, weekly
Monday schedule on one timezone, PR limit 5, grouping on the three
multi-dependency blocks, and the reviewer-facing docs that make the
config's volume bounds discoverable.
"""

from __future__ import annotations

import re
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
    found = doc["updates"]
    assert isinstance(found, list), "updates must be a list"
    return found


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
    # Key on (ecosystem, directory): the two uv blocks must each be asserted,
    # not collapsed by ecosystem alone (the root uv block's grouping would
    # otherwise never be inspected).
    by_block = {(u["package-ecosystem"], u["directory"]): u for u in updates(load_dependabot())}
    for key in (("uv", "/"), ("uv", "/rl/daydream_review_v1"), ("github-actions", "/")):
        group = by_block[key]["groups"]
        assert list(group) == ["dependencies"], f"{key} needs a single group"
        pattern = group["dependencies"]["patterns"]
        assert "*" in pattern, f"{key} group must cover all deps (minor+patch together)"
    assert "groups" not in by_block[("npm", "/.github/workflows")], "npm tracks a single dep — no group"


def test_codex_pin_matches_tracked_package_json() -> None:
    """CI installs codex from a hardcoded string in daydream-review.yml, not
    from the tracked package.json served to the npm Dependabot block. Merging a
    Dependabot bump PR changes nothing in CI unless the workflow string moves
    too — assert the two stay in lockstep so silent divergence fails here."""
    package_json = _REPO_ROOT / ".github" / "workflows" / "package.json"
    manifest = yaml.safe_load(package_json.read_text(encoding="utf-8"))
    manifest_version = manifest["dependencies"]["@openai/codex"]

    workflow = (_REPO_ROOT / ".github" / "workflows" / "daydream-review.yml").read_text(
        encoding="utf-8"
    )
    match = re.search(r"npm install -g @openai/codex@([^\s]+)", workflow)
    assert match is not None, (
        "daydream-review.yml must install codex via `npm install -g "
        "@openai/codex@<version>` so the pin stays reconcilable"
    )
    assert match.group(1) == manifest_version, (
        f"codex pin {match.group(1)!r} in daydream-review.yml diverged from "
        f"package.json version {manifest_version!r} — bump them in lockstep"
    )


def test_commit_message_prefix_everywhere() -> None:
    for u in updates(load_dependabot()):
        assert u["commit-message"]["prefix"] == "chore(deps)"


def test_readme_documents_dependabot_volume_and_gates() -> None:
    text = WORKFLOWS_README.read_text(encoding="utf-8")
    for needle in (
        "## Dependabot dependency updates",
        ".github/dependabot.yml",
        "Australia/Brisbane",
        "open-pull-requests-limit",
        "`make check`",
        "CODEOWNERS",
    ):
        assert needle in text, f"README missing: {needle}"


def test_config_documents_ownership_and_gates() -> None:
    header = DEPENDABOT_PATH.read_text(encoding="utf-8").split("updates:", 1)[0]
    assert "CODEOWNERS" in header, "config comment must point at reviewer ownership"
    assert "make check" in header, "config comment must name the validation gate"
