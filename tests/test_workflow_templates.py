"""Invariant tests for the shipped GitHub Actions workflow templates.

A workflow's behavior can only be validated by running it in CI, so these tests
deliberately do NOT restate the YAML (trigger lists, exact ``if:`` strings,
permission dicts). They guard only the handful of properties a single file-read
cannot verify and that a careless edit could silently break:

- No untrusted event data is interpolated into a ``run:`` body (injection).
- The daydream install stays pinned to the current release tag (cross-file
  drift against ``pyproject.toml``).
- The privilege split holds: the job that checks out untrusted PR code never
  holds the App key, and the privileged jobs never check out PR code.
- The repo's own Codex dogfood workflow persists ``codex login`` before the
  review runs (``codex exec`` does not read ``OPENAI_API_KEY`` for auth).

PyYAML parses the bare ``on:`` key as boolean ``True``; ``wf_on()`` normalizes it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "daydream" / "templates" / "workflows"
REPO_WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"

_SECRET_REF_RE = re.compile(r"secrets\.([A-Za-z0-9_]+)")


def load_workflow(path: Path) -> dict[str, Any]:
    """Parse a workflow template into its YAML tree."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path.name} did not parse to a mapping"
    return loaded


def job_steps(wf: dict[str, Any], job: str) -> list[dict[str, Any]]:
    """Return the steps list for ``job``."""
    steps = wf["jobs"][job]["steps"]
    assert isinstance(steps, list) and steps
    return steps


def has_checkout(job: dict[str, Any]) -> bool:
    return any("actions/checkout" in s.get("uses", "") for s in job["steps"])


# Injection guard (all templates): untrusted event data must reach run: via env:,
# never ${{ }} interpolation, which would splice attacker-controlled text into the
# shell.

_EVENT_INTERP = re.compile(
    r"\$\{\{[^}]*github\.event\.(comment|issue|pull_request|workflow_run|review)[^}]*\}\}"
)


@pytest.mark.parametrize("wf_path", sorted(TEMPLATES_DIR.rglob("*.yml")), ids=lambda p: p.name)
def test_no_event_data_interpolated_into_run_steps(wf_path: Path) -> None:
    wf = load_workflow(wf_path)
    for job_name, job in wf["jobs"].items():
        for step in job["steps"]:
            if "run" in step:
                assert not _EVENT_INTERP.search(step["run"]), (
                    f"{wf_path.name}:{job_name}: event data must reach run: via env:, "
                    f"never ${{{{ }}}} interpolation"
                )


# Install-pin drift guard: the bot must install a pinned daydream release, never
# the moving `main` tip. Fails on release until the template pin is bumped in
# lockstep with the package version.

_INSTALL_RE = re.compile(
    r"uv tool install\s+git\+https://github\.com/existential-birds/daydream(?P<ref>@\S+)?"
)


def _package_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["version"]


@pytest.mark.parametrize("name", ["daydream-review.yml", "daydream-post.yml", "single/daydream.yml"])
def test_daydream_install_is_pinned_to_current_release_tag(name: str) -> None:
    text = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    refs = [m.group("ref") for m in _INSTALL_RE.finditer(text)]
    assert refs, f"{name} must install daydream via `uv tool install git+…`"
    expected = f"@v{_package_version()}"
    for ref in refs:
        assert ref == expected, (
            f"{name} pins the daydream install to {ref or '(unpinned main)'}, but must pin to "
            f"{expected}. Bump the template pin in lockstep with the package version on release."
        )


# Privilege split — the security invariant the whole design exists to enforce:
# no job ever holds both untrusted PR code and the App key.


def test_split_setup_preserves_privilege_split() -> None:
    review = load_workflow(TEMPLATES_DIR / "daydream-review.yml")
    review_text = (TEMPLATES_DIR / "daydream-review.yml").read_text(encoding="utf-8")
    command_text = (TEMPLATES_DIR / "daydream-command.yml").read_text(encoding="utf-8")
    post = load_workflow(TEMPLATES_DIR / "daydream-post.yml")
    post_text = (TEMPLATES_DIR / "daydream-post.yml").read_text(encoding="utf-8")

    # Phase A runs untrusted PR code: read-only, and its only secret is the API key.
    assert review["permissions"] == {"contents": "read"}
    assert set(_SECRET_REF_RE.findall(review_text)) == {"ANTHROPIC_API_KEY"}

    # The two App-key holders never check out untrusted code: the command job
    # never checks out at all; the post job may only take the trusted default
    # branch (no PR / workflow_run ref override).
    assert "actions/checkout" not in command_text
    for job in post["jobs"].values():
        for step in job["steps"]:
            if "actions/checkout" in step.get("uses", ""):
                assert "workflow_run" not in str(step.get("with", {}).get("ref", ""))
    assert set(_SECRET_REF_RE.findall(post_text)) == {"DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"}


def test_single_setup_preserves_privilege_split() -> None:
    wf = load_workflow(TEMPLATES_DIR / "single" / "daydream.yml")
    text = (TEMPLATES_DIR / "single" / "daydream.yml").read_text(encoding="utf-8")

    # analyze is the only job that touches untrusted PR code: read-only, no App
    # key, and it must not leave the GITHUB_TOKEN persisted in .git/config.
    analyze = wf["jobs"]["analyze"]
    assert analyze["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "DAYDREAM_APP" not in yaml.safe_dump(analyze)
    checkout = next(s for s in analyze["steps"] if "actions/checkout" in s.get("uses", ""))
    assert checkout["with"]["persist-credentials"] is False

    # The App-key holders never check out code, and nothing dispatches, so no job
    # needs actions: write.
    for job_name in ("gate", "post", "surface-failure"):
        assert not has_checkout(wf["jobs"][job_name])
    assert "permission-actions" not in text

    # surface-failure fires on non-success analyze results, so its if: must carry
    # a status function; without always() GitHub injects an implicit success() and
    # skips the job on the very failures it exists to surface.
    assert "always()" in wf["jobs"]["surface-failure"]["if"]

    # Same three secrets as the split setup, nothing more.
    assert set(_SECRET_REF_RE.findall(text)) == {
        "ANTHROPIC_API_KEY",
        "DAYDREAM_APP_ID",
        "DAYDREAM_APP_PRIVATE_KEY",
    }


# Repo dogfood workflow (Codex): the one non-obvious behavioral constraint worth
# guarding — `codex exec` does not read OPENAI_API_KEY for model-API auth, so
# `codex login --with-api-key` must persist auth.json BEFORE the review runs.


def test_repo_review_authenticates_codex_before_running() -> None:
    wf = load_workflow(REPO_WORKFLOWS_DIR / "daydream-review.yml")
    text = (REPO_WORKFLOWS_DIR / "daydream-review.yml").read_text(encoding="utf-8")
    assert set(_SECRET_REF_RE.findall(text)) == {"OPENAI_API_KEY"}

    steps = job_steps(wf, "analyze")
    login = next(s for s in steps if "codex login --with-api-key" in s.get("run", ""))
    assert "OPENAI_API_KEY" in login["env"]
    review_idx = next(i for i, s in enumerate(steps) if "daydream --review" in s.get("run", ""))
    assert steps.index(login) < review_idx, "Codex auth must be persisted before the review runs"
    # The review step authenticates via auth.json, not a redundant env secret.
    assert "OPENAI_API_KEY" not in steps[review_idx].get("env", {})
