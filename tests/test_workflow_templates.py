"""Structure tests for the shipped GitHub Actions workflow templates.

Parses each template under ``daydream/templates/workflows/`` with ``yaml.safe_load`` and
asserts on the parsed tree; raw-text assertions ("this string never appears")
read the file directly. PyYAML parses the bare ``on:`` key as boolean ``True``;
``wf_on()`` normalizes it back.

Security contracts under test (roadmap §"Sub-project #2 security design",
revised by Task 0 spike findings):

- Review workflow (Phase A) is unprivileged: ``contents: read`` only, no App
  secrets anywhere, ``ANTHROPIC_API_KEY`` is the only ``secrets.*`` reference.
- Command workflow never checks out code; the App token it mints (spike Step 1:
  a ``GITHUB_TOKEN`` dispatch never fires downstream ``workflow_run``) carries
  ``permission-actions: write`` (dispatch) + ``permission-pull-requests: write``
  (reaction) and reaches ``gh`` via ``env:`` only — never a ``run:`` body.
- The 👀 reaction is signed by the App token (not ``GITHUB_TOKEN``) so it is
  attributed to the bot identity, not ``github-actions[bot]``; the token is
  minted before the reaction step. ``pull-requests: write`` is the right scope
  (spike Step 4: ``issues: write`` is neither sufficient nor necessary for PR
  comments). The job itself declares ``permissions: {}`` — the default token is
  unused.
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


def wf_on(wf: dict[Any, Any]) -> dict[str, Any]:
    """Return the trigger mapping, normalizing PyYAML's ``on:`` -> ``True`` key."""
    triggers = wf[True] if True in wf else wf["on"]
    assert isinstance(triggers, dict)
    return triggers


def job_steps(wf: dict[str, Any], job: str) -> list[dict[str, Any]]:
    """Return the steps list for ``job``."""
    steps = wf["jobs"][job]["steps"]
    assert isinstance(steps, list) and steps
    return steps


@pytest.fixture(scope="module")
def review_wf() -> dict[str, Any]:
    return load_workflow(TEMPLATES_DIR / "daydream-review.yml")


@pytest.fixture(scope="module")
def review_text() -> str:
    return (TEMPLATES_DIR / "daydream-review.yml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def command_wf() -> dict[str, Any]:
    return load_workflow(TEMPLATES_DIR / "daydream-command.yml")


@pytest.fixture(scope="module")
def command_text() -> str:
    return (TEMPLATES_DIR / "daydream-command.yml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def post_wf() -> dict[str, Any]:
    return load_workflow(TEMPLATES_DIR / "daydream-post.yml")


@pytest.fixture(scope="module")
def post_text() -> str:
    return (TEMPLATES_DIR / "daydream-post.yml").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# daydream-review.yml (Phase A — unprivileged analyze)
# ---------------------------------------------------------------------------


def test_review_workflow_is_unprivileged(review_wf: dict[str, Any], review_text: str) -> None:
    assert review_wf["permissions"] == {"contents": "read"}
    assert "DAYDREAM_APP" not in review_text  # no App secrets, ever (footgun 5)
    triggers = wf_on(review_wf)
    assert triggers["pull_request"]["types"] == ["opened", "ready_for_review"]
    assert "pr_number" in triggers["workflow_dispatch"]["inputs"]
    assert triggers["workflow_dispatch"]["inputs"]["pr_number"]["required"] is True


def test_review_auto_job_gates_forks_and_toggle(review_wf: dict[str, Any]) -> None:
    cond = review_wf["jobs"]["analyze"]["if"]
    assert "head.repo.full_name == github.repository" in cond  # footgun 4 (fork gate)
    assert "DAYDREAM_AUTO_REVIEW" in cond  # operator toggle
    assert "workflow_dispatch" in cond  # dispatch path bypasses the auto toggle


def test_review_only_secret_is_anthropic_api_key(review_text: str) -> None:
    assert set(_SECRET_REF_RE.findall(review_text)) == {"ANTHROPIC_API_KEY"}


def test_review_checkout_pins_pr_head_per_event(review_wf: dict[str, Any]) -> None:
    steps = job_steps(review_wf, "analyze")
    # pull_request shape: checkout pins the PR head SHA explicitly.
    pr_checkouts = [
        s
        for s in steps
        if "actions/checkout" in s.get("uses", "")
        and "github.event.pull_request.head.sha" in str(s.get("with", {}).get("ref", ""))
    ]
    assert len(pr_checkouts) == 1
    assert pr_checkouts[0]["if"] == "github.event_name == 'pull_request'"
    # workflow_dispatch shape: the run sits on the default branch (spike Shape 3),
    # so the PR head ref is fetched explicitly, PR number via env, never inline.
    dispatch_fetches = [
        s for s in steps if "run" in s and "refs/pull/${PR_NUMBER}/head" in s["run"]
    ]
    assert len(dispatch_fetches) == 1
    fetch = dispatch_fetches[0]
    assert fetch["if"] == "github.event_name == 'workflow_dispatch'"
    assert fetch["env"]["PR_NUMBER"] == "${{ inputs.pr_number }}"
    assert "inputs.pr_number" not in fetch["run"]  # env-only (footgun 2)


def test_review_uploads_findings_artifact(review_wf: dict[str, Any]) -> None:
    uploads = [
        s for s in job_steps(review_wf, "analyze") if "actions/upload-artifact" in s.get("uses", "")
    ]
    assert len(uploads) == 1
    assert uploads[0]["with"]["name"] == "daydream-findings"


def test_review_run_step_takes_event_data_via_env_only(review_wf: dict[str, Any]) -> None:
    daydream_runs = [
        s for s in job_steps(review_wf, "analyze") if "daydream --review" in s.get("run", "")
    ]
    assert len(daydream_runs) == 1
    step = daydream_runs[0]
    assert "ANTHROPIC_API_KEY" in step["env"]
    run = step["run"]
    assert '--non-interactive' in run
    assert '--pr-number "$PR_NUMBER"' in run
    assert "--findings-out findings/findings.json" in run
    assert '--base "origin/$BASE_REF"' in run
    assert "${{" not in run  # event data reaches run: via env:, never interpolation


# ---------------------------------------------------------------------------
# daydream-command.yml (gatekeeper — privileged dispatch, never touches code)
# ---------------------------------------------------------------------------


def test_command_workflow_gates_and_never_touches_code(
    command_wf: dict[str, Any], command_text: str
) -> None:
    cond = command_wf["jobs"]["dispatch"]["if"].replace('"', "'")
    for assoc in ("OWNER", "MEMBER", "COLLABORATOR"):
        assert assoc in cond  # footgun 4 (author gate)
    assert "comment.user.type != 'Bot'" in cond  # no self-loops
    assert "issue.pull_request" in cond  # PR comments only
    assert "actions/checkout" not in command_text  # privileged job: no code


def test_command_workflow_default_token_is_unprivileged(command_wf: dict[str, Any]) -> None:
    # Both writes (reaction + dispatch) flow through the App token, so the
    # default GITHUB_TOKEN needs no permissions at all.
    assert command_wf["permissions"] == {}


def test_command_app_token_mints_actions_and_pull_requests_write(command_wf: dict[str, Any]) -> None:
    mints = [
        s
        for s in job_steps(command_wf, "dispatch")
        if "actions/create-github-app-token" in s.get("uses", "")
    ]
    assert len(mints) == 1
    with_ = mints[0]["with"]
    grants = {k: v for k, v in with_.items() if k.startswith("permission-")}
    # actions: write dispatches the review (spike Step 1); pull-requests: write
    # posts the 👀 reaction (spike Step 4). Least privilege for this job's two
    # operations — exactly these, nothing more.
    assert grants == {"permission-actions": "write", "permission-pull-requests": "write"}
    assert with_["app-id"] == "${{ secrets.DAYDREAM_APP_ID }}"
    assert with_["private-key"] == "${{ secrets.DAYDREAM_APP_PRIVATE_KEY }}"


def test_command_reaction_is_attributed_to_bot_identity(command_wf: dict[str, Any]) -> None:
    # The bug this guards: a reaction signed by ${{ github.token }} posts as
    # github-actions[bot], not the operator's bot. It must use the minted App
    # token, and the mint must precede the reaction so the token exists.
    steps = job_steps(command_wf, "dispatch")
    reactions = [s for s in steps if "content=eyes" in s.get("run", "")]
    assert len(reactions) == 1
    reaction = reactions[0]
    assert reaction["env"]["GH_TOKEN"] == "${{ steps.token.outputs.token }}"
    assert "github.token" not in str(reaction["env"])
    mint_idx = next(
        i for i, s in enumerate(steps) if "actions/create-github-app-token" in s.get("uses", "")
    )
    assert mint_idx < steps.index(reaction)  # token exists before the reaction fires


def test_command_minted_token_reaches_gh_via_env_only(command_wf: dict[str, Any]) -> None:
    steps = job_steps(command_wf, "dispatch")
    for step in steps:  # never logged: the token never appears in a run: body
        assert "steps.token.outputs.token" not in step.get("run", "")
    dispatches = [s for s in steps if "gh workflow run daydream-review.yml" in s.get("run", "")]
    assert len(dispatches) == 1
    assert dispatches[0]["env"]["GH_TOKEN"] == "${{ steps.token.outputs.token }}"


def test_command_match_is_exact_and_body_env_only(command_wf: dict[str, Any]) -> None:
    matches = [s for s in job_steps(command_wf, "dispatch") if "grep -Eq" in s.get("run", "")]
    assert len(matches) == 1
    step = matches[0]
    assert step["env"]["BODY"] == "${{ github.event.comment.body }}"  # footgun 2
    assert step["env"]["BOT_HANDLE"] == "${{ vars.DAYDREAM_BOT_HANDLE }}"
    assert '(^|[[:space:]])@${ESCAPED_HANDLE}[[:space:]]+review([[:space:]]|$)' in step["run"]
    assert "github.event" not in step["run"]


# ---------------------------------------------------------------------------
# daydream-post.yml (Phase B — privileged post, never touches PR code)
# ---------------------------------------------------------------------------


def test_post_workflow_token_is_least_privilege(post_wf: dict[str, Any], post_text: str) -> None:
    step = next(
        s for s in post_wf["jobs"]["post"]["steps"] if "create-github-app-token" in s.get("uses", "")
    )
    grants = {k: v for k, v in step["with"].items() if k.startswith("permission-")}
    assert grants == {
        "permission-pull-requests": "write",
        "permission-contents": "read",
        "permission-metadata": "read",
    }  # footgun 3, exactly
    for job in post_wf["jobs"].values():  # never logged: token reaches
        for s in job["steps"]:  # gh via env:, not echo/run
            assert "steps.token.outputs.token" not in s.get("run", "")
    # Phase B holds App material only — the analyze key never appears here.
    assert set(_SECRET_REF_RE.findall(post_text)) == {"DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"}


def test_post_workflow_never_checks_out_pr_code(post_wf: dict[str, Any]) -> None:
    checkouts = [
        s
        for job in post_wf["jobs"].values()
        for s in job["steps"]
        if "actions/checkout" in s.get("uses", "")
    ]
    for step in checkouts:  # core invariant
        assert "workflow_run" not in str(step.get("with", {}).get("ref", ""))
    assert wf_on(post_wf)["workflow_run"]["workflows"] == ["Daydream Review"]
    assert wf_on(post_wf)["workflow_run"]["types"] == ["completed"]


def test_post_workflow_derives_target_from_event_only(
    post_wf: dict[str, Any], post_text: str
) -> None:
    assert "github.event.workflow_run.head_sha" in post_text  # footgun 1: event-derived
    assert "post-findings" in post_text and "--head-sha" in post_text
    # Spike Step 2 revision: on the workflow_dispatch shape the event's
    # head_sha is the default-branch tip, so the derive step must fetch the
    # LIVE PR from the API — the GitHub API is the trust anchor, never the
    # artifact alone.
    derive = next(s for s in job_steps(post_wf, "post") if s.get("id") == "target")
    assert "repos/" in derive["run"] and "/pulls/" in derive["run"]


def test_post_workflow_gate_and_permissions(post_wf: dict[str, Any]) -> None:
    # GITHUB_TOKEN only downloads the artifact; the App token carries writes.
    assert post_wf["permissions"] == {"actions": "read"}
    assert post_wf["jobs"]["post"]["if"] == "github.event.workflow_run.conclusion == 'success'"


def test_post_workflow_downloads_artifact_from_triggering_run(post_wf: dict[str, Any]) -> None:
    downloads = [
        s for s in job_steps(post_wf, "post") if "actions/download-artifact" in s.get("uses", "")
    ]
    assert len(downloads) == 1
    assert downloads[0]["with"]["name"] == "daydream-findings"
    assert downloads[0]["with"]["run-id"] == "${{ github.event.workflow_run.id }}"


def test_post_workflow_surfaces_failures(post_wf: dict[str, Any]) -> None:
    # Post-job failure: a final if: failure() step comments via the minted
    # token, values via env only. The condition may include additional guards
    # (e.g. checking that the download succeeded) beyond the bare failure().
    failure_steps = [s for s in job_steps(post_wf, "post") if "failure()" in str(s.get("if", ""))]
    assert len(failure_steps) == 1
    step = failure_steps[0]
    assert step["env"]["GH_TOKEN"] == "${{ steps.token.outputs.token }}"
    assert "daydream review failed" in step["run"]
    assert "${{" not in step["run"]  # values via env, never interpolation
    # Analyze failure: routes to the surface job instead of a silent skip.
    surface = post_wf["jobs"]["surface-analyze-failure"]
    assert surface["if"] == "github.event.workflow_run.conclusion == 'failure'"
    comment_steps = [s for s in surface["steps"] if "daydream review failed" in s.get("run", "")]
    assert len(comment_steps) == 1
    assert comment_steps[0]["env"]["GH_TOKEN"] == "${{ steps.token.outputs.token }}"


# ---------------------------------------------------------------------------
# Injection scan — footgun 2, all templates (env:-only event data in run:)
# ---------------------------------------------------------------------------


_EVENT_INTERP = re.compile(
    r"\$\{\{[^}]*github\.event\.(comment|issue|pull_request|workflow_run|review)[^}]*\}\}"
)


@pytest.mark.parametrize("wf_path", sorted(TEMPLATES_DIR.glob("*.yml")), ids=lambda p: p.name)
def test_no_event_data_interpolated_into_run_steps(wf_path) -> None:
    wf = load_workflow(wf_path)
    for job_name, job in wf["jobs"].items():
        for step in job["steps"]:
            if "run" in step:
                assert not _EVENT_INTERP.search(step["run"]), (
                    f"{wf_path.name}:{job_name}: event data must reach run: via env:, "
                    f"never ${{{{ }}}} interpolation"
                )


# ---------------------------------------------------------------------------
# Install-pin drift guard — the bot must install a pinned daydream release,
# never the moving `main` tip (a stale uv cache or main/template drift would
# otherwise feed operators a daydream whose CLI no longer matches the workflow).
# This test fails on release until the template pin is bumped in lockstep with
# the package version — closing the gap that broke a live install.
# ---------------------------------------------------------------------------


_INSTALL_RE = re.compile(
    r"uv tool install\s+git\+https://github\.com/existential-birds/daydream(?P<ref>@\S+)?"
)


def _package_version() -> str:
    """Read the declared package version from pyproject.toml (the single source)."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["version"]


@pytest.mark.parametrize("name", ["daydream-review.yml", "daydream-post.yml"])
def test_daydream_install_is_pinned_to_current_release_tag(name: str) -> None:
    text = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    refs = [m.group("ref") for m in _INSTALL_RE.finditer(text)]
    assert refs, f"{name} must install daydream via `uv tool install git+…`"
    expected = f"@v{_package_version()}"
    for ref in refs:
        assert ref == expected, (
            f"{name} pins the daydream install to {ref or '(unpinned main)'}, "
            f"but must pin to {expected}. An unpinned/stale install lets the bot run a "
            f"daydream whose CLI has drifted from this workflow. Bump the template pin in "
            f"lockstep with the package version on every release."
        )


# ---------------------------------------------------------------------------
# .github/workflows/daydream-review.yml (live repo dogfood workflow — Codex)
#
# The repo's own review CI has intentionally diverged from the shipped template:
# operators get the Anthropic-backed template, but daydream dogfoods itself on the
# Codex backend (Anthropic disallows subscription auth for automations). The
# template tests above never load this file, so its Codex contract is asserted
# here directly.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def repo_review_wf() -> dict[str, Any]:
    return load_workflow(REPO_WORKFLOWS_DIR / "daydream-review.yml")


@pytest.fixture(scope="module")
def repo_review_text() -> str:
    return (REPO_WORKFLOWS_DIR / "daydream-review.yml").read_text(encoding="utf-8")


def test_repo_review_only_secret_is_openai_api_key(repo_review_text: str) -> None:
    assert set(_SECRET_REF_RE.findall(repo_review_text)) == {"OPENAI_API_KEY"}


def test_repo_review_installs_codex_cli(repo_review_wf: dict[str, Any]) -> None:
    codex_installs = [
        s
        for s in job_steps(repo_review_wf, "analyze")
        if "Codex" in s.get("name", "") and "npm install -g @openai/codex" in s.get("run", "")
    ]
    assert len(codex_installs) == 1
    # The backend parses `codex exec --experimental-json` output, whose event
    # shape can drift between CLI releases, so the install must be version-pinned.
    assert "npm install -g @openai/codex@" in codex_installs[0]["run"]


def test_repo_review_runs_codex_backend_non_interactive(repo_review_wf: dict[str, Any]) -> None:
    daydream_runs = [
        s for s in job_steps(repo_review_wf, "analyze") if "daydream --review" in s.get("run", "")
    ]
    assert len(daydream_runs) == 1
    run = daydream_runs[0]["run"]
    assert "--backend codex" in run
    assert "--non-interactive" in run


def test_repo_review_authenticates_codex_before_run(repo_review_wf: dict[str, Any]) -> None:
    # `codex exec` does NOT read OPENAI_API_KEY from the environment for its own
    # model-API auth (CLI 0.139.0): without persisted auth state it sends no
    # bearer and every call 401s. Auth must be persisted via an explicit
    # `codex login --with-api-key` step before the review runs, and the review
    # step must NOT re-declare the secret (auth flows through $CODEX_HOME/auth.json).
    steps = job_steps(repo_review_wf, "analyze")
    login_steps = [s for s in steps if "codex login --with-api-key" in s.get("run", "")]
    assert len(login_steps) == 1, "expected exactly one `codex login --with-api-key` step"
    login = login_steps[0]
    assert "OPENAI_API_KEY" in login["env"]

    # The login step must come before the daydream review step so auth.json
    # exists when `codex exec` runs.
    login_idx = steps.index(login)
    review_idx = next(i for i, s in enumerate(steps) if "daydream --review" in s.get("run", ""))
    assert login_idx < review_idx, "Codex auth must be persisted before the review step runs"

    # The review step authenticates via auth.json, not a redundant env secret.
    assert "OPENAI_API_KEY" not in steps[review_idx].get("env", {})
