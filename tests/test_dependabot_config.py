"""Invariant tests for .github/dependabot.yml (issue #936).

The config's real acceptance happens GitHub-side (Dependency graph registration,
first grouped PRs). These tests pin the structural invariants a careless edit
could silently break: exactly three ecosystems at the exact directories, weekly
Monday schedule on one timezone, PR limit 5, grouping on the two
multi-dependency uv blocks, and the reviewer-facing docs that make the
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
        ("npm", "/.github/workflows"),  # the npm manifest trick — must NOT be "/"
    }
    # The github-actions ecosystem was deliberately removed (PR 1065 follow-up):
    # the repo pins every third-party action to a SHA registered in
    # test_workflow_templates.py::_PINNED_ACTION_VERSIONS, which Dependabot
    # cannot update, so its bump PRs were structurally unmergeable.


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
    for key in (("uv", "/"), ("uv", "/rl/daydream_review_v1")):
        group = by_block[key]["groups"]
        assert list(group) == ["dependencies"], f"{key} needs a single group"
        pattern = group["dependencies"]["patterns"]
        assert "*" in pattern, f"{key} group must cover all deps (minor+patch together)"
        update_types = group["dependencies"]["update-types"]
        assert set(update_types) == {"minor", "patch"}, (
            f"{key} group must scope update-types to minor+patch (found {update_types})"
        )
    assert "groups" not in by_block[("npm", "/.github/workflows")], "npm tracks a single dep — no group"


def test_codex_install_tracks_package_json() -> None:
    """CI installs codex at the version pinned in the tracked workflows
    package.json — the file the npm Dependabot block updates. A hardcoded
    version string in daydream-review.yml would silently leave CI on the
    stale pin after a bump PR; assert the install reads the manifest so a
    Dependabot bump reaches CI end-to-end."""
    package_json = _REPO_ROOT / ".github" / "workflows" / "package.json"
    manifest = yaml.safe_load(package_json.read_text(encoding="utf-8"))
    assert "@openai/codex" in manifest["dependencies"], (
        "package.json must track @openai/codex so Dependabot can bump it"
    )

    workflow = (_REPO_ROOT / ".github" / "workflows" / "daydream-review.yml").read_text(
        encoding="utf-8"
    )
    # No literal pin may exist anywhere in the workflow — the version must
    # come from the tracked manifest, so Dependabot bumps flow through to CI.
    assert not re.search(r"@openai/codex@\d+\.\d+\.\d+", workflow), (
        "daydream-review.yml hardcodes a codex version — CI would stay on the "
        "stale pin after a Dependabot bump of package.json; install from the "
        "tracked manifest instead"
    )
    install = re.search(
        r"npm install -g .*@openai/codex@\$\(.*package\.json.*\)", workflow
    )
    assert install is not None, (
        "daydream-review.yml must install codex at the version read from the "
        "tracked .github/workflows/package.json (the npm Dependabot block's "
        "target) so bump PRs reach CI end-to-end"
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
        "end-to-end",
    ):
        assert needle in text, f"README missing: {needle}"


def test_config_documents_ownership_and_gates() -> None:
    header = DEPENDABOT_PATH.read_text(encoding="utf-8").split("updates:", 1)[0]
    assert "CODEOWNERS" in header, "config comment must point at reviewer ownership"
    assert "make check" in header, "config comment must name the validation gate"
