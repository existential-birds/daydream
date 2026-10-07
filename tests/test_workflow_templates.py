"""Workflow identity, permissions, injection, and failure-reporting contracts.

PyYAML parses bare ``on:`` as True; _wf_triggers normalizes that key."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = _REPO_ROOT / "daydream" / "templates" / "workflows"
REPO_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"

# Each packaged workflow is checked against the live repo copy, because the
# packaged template must not drift from what the bot actually runs.
_TEMPLATE_AND_LIVE_IDS = ["template", "live"]
_COMMAND_WORKFLOW_PATHS = [TEMPLATES_DIR / "daydream-command.yml", REPO_WORKFLOWS_DIR / "daydream-command.yml"]
_REVIEW_WORKFLOW_PATHS = [TEMPLATES_DIR / "daydream-review.yml", REPO_WORKFLOWS_DIR / "daydream-review.yml"]
_POST_WORKFLOW_PATHS = [TEMPLATES_DIR / "daydream-post.yml", REPO_WORKFLOWS_DIR / "daydream-post.yml"]

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


def _wf_triggers(wf: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` trigger map, normalizing PyYAML's boolean key."""
    on: Any = wf.get("on")
    if on is None:
        on = cast(dict[Any, Any], wf).get(True)
    return on if isinstance(on, dict) else {}


def has_checkout(job: dict[str, Any]) -> bool:
    return any("actions/checkout" in s.get("uses", "") for s in job["steps"])


def _dispatch_step(wf: dict[str, Any]) -> dict[str, Any]:
    """Return the `dispatch` job step that triggers the review workflow."""
    return next(step
        for step in job_steps(wf, "dispatch")
        if "gh workflow run daydream-review.yml" in step.get("run", "")
    )


# Non-local actions require full commit SHAs and release-version comments; local actions are exempt.

_BOT_WORKFLOW_PATHS = sorted([*REPO_WORKFLOWS_DIR.glob("daydream-*.yml"), *TEMPLATES_DIR.rglob("*.yml")],
    key=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)

_PINNED_ACTION_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")


def _action_references(wf: dict[str, Any]) -> list[str]:
    """Return job-level and step-level ``uses:`` executable references, in document order."""
    refs: list[str] = []
    for job in wf["jobs"].values():
        job_uses = job.get("uses")
        if isinstance(job_uses, str):
            refs.append(job_uses)
        steps = job.get("steps")
        if isinstance(steps, list):
            for step in steps:
                step_uses = step.get("uses")
                if isinstance(step_uses, str):
                    refs.append(step_uses)
    return refs


# Pass untrusted event data through env; expression interpolation would splice it into shell code.

_EVENT_INTERP = re.compile(r"\$\{\{[^}]*github\.event\.(comment|issue|pull_request|workflow_run|review)[^}]*\}\}")

@pytest.mark.parametrize("wf_path", _COMMAND_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_command_workflow_dispatches_approved_head(wf_path: Path) -> None:
    wf = load_workflow(wf_path)
    dispatch = _dispatch_step(wf)

    assert "gh api" in dispatch["run"] and ".head.sha" in dispatch["run"]
    assert '-f approved_head_sha="$HEAD_SHA"' in dispatch["run"]
    assert '-f approved_at="$COMMENT_CREATED_AT"' in dispatch["run"]
    assert "PR_NUMBER" in dispatch["env"]
    assert "COMMENT_CREATED_AT" in dispatch["env"]
    assert not any(_EVENT_INTERP.search(step.get("run", "")) for step in job_steps(wf, "dispatch"))
    assert "actions/checkout" not in wf_path.read_text(encoding="utf-8")

@pytest.mark.parametrize("wf_path", _COMMAND_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_command_workflow_acknowledges_only_after_successful_dispatch(wf_path: Path) -> None:
    """A failed dispatch creates no review run or failure report. Acknowledge only
    after success so the reaction cannot falsely signal a running review."""
    wf = load_workflow(wf_path)
    steps = job_steps(wf, "dispatch")
    dispatch = _dispatch_step(wf)
    ack = next(step for step in steps if step.get("name") == "Acknowledge with eyes reaction")

    assert steps.index(dispatch) < steps.index(ack), (
        f"{wf_path.name}: the 👀 acknowledgement must come after the dispatch step "
        "so a dispatch failure never mis-signals success"
    )

    # Preserve implicit success() and reject continue-on-error so failed dispatch cannot acknowledge.
    ack_if = ack.get("if", "")
    assert not any(fn in ack_if for fn in ("always()", "failure()", "cancelled()")), (
        f"{wf_path.name}: the acknowledgement must stay gated on dispatch success "
        "(its if: must keep GitHub's implicit success())"
    )
    assert "continue-on-error" not in dispatch, (
        f"{wf_path.name}: the dispatch step must not continue-on-error, or a failed "
        "dispatch would still post the 👀 reaction"
    )

@pytest.mark.parametrize("wf_path", _REVIEW_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_review_workflow_head_bound_gate(wf_path: Path) -> None:
    """Bind checkout and review to the approved head; live Codex and packaged
    Anthropic workflows retain their distinct credential lifecycles."""
    wf = load_workflow(wf_path)
    text = wf_path.read_text(encoding="utf-8")

    assert "pull_request" not in _wf_triggers(wf)
    inputs = _wf_triggers(wf)["workflow_dispatch"]["inputs"]
    assert "approved_head_sha" in inputs
    assert "approved_at" in inputs

    steps = job_steps(wf, "analyze")
    verify = next(step
        for step in steps
        if "approved_head_sha" in step.get("run", "") and "exit 1" in step.get("run", "")
    )
    # Bind push time to the PR head ref. Equal-second timestamps cannot prove approval preceded
    # the push, so require strict-before rather than using repository-wide pushed_at.
    assert ".head.ref" in verify["run"]
    assert "activity" in verify["run"]
    assert r'\< "$APPROVED_AT"' in verify["run"]
    assert "APPROVED_AT" in verify["env"]
    checkout_idx = next(i for i, step in enumerate(steps) if "actions/checkout" in step.get("uses", ""))
    assert steps.index(verify) < checkout_idx

    review = next(step for step in steps if "daydream --review" in step.get("run", ""))
    assert "--approved-head-sha" in review["run"]
    assert "APPROVED_HEAD_SHA" in review["env"]

    if wf_path == REPO_WORKFLOWS_DIR / "daydream-review.yml":
        cleanup = next(step for step in steps if "auth.json" in step.get("run", ""))
        assert steps.index(review) < steps.index(cleanup)
        assert cleanup.get("if", "") == "always()"
        assert set(_SECRET_REF_RE.findall(text)) == {"OPENAI_API_KEY"}
    else:
        # The packaged template never persists auth and uses only its model credential.
        assert not any("auth.json" in step.get("run", "") for step in steps)
        assert set(_SECRET_REF_RE.findall(text)) == {"ANTHROPIC_API_KEY"}


def _assert_single_reject_funnel(run: str, *, label: str) -> None:
    """Every drift rejection must persist PR identity and its diagnostic before
    exiting; otherwise the post workflow loses the instructive failure message."""
    lines = run.splitlines()
    write_idx = [i for i, ln in enumerate(lines) if "> findings/failure.json" in ln]
    exit_idx = [i for i, ln in enumerate(lines) if ln.strip() == "exit 1"]
    reject_calls = [i for i, ln in enumerate(lines) if re.match(r"reject \"", ln.strip())]
    assert len(write_idx) == 1 and len(exit_idx) == 1, (
        f"{label}: every drift rejection must funnel through the single "
        "failure-context helper (one write, one exit) (issue #336)"
    )
    assert write_idx[0] < exit_idx[0], (f"{label}: the helper must record the failure context before exiting")
    assert '"message"' in lines[write_idx[0]], (
        f"{label}: the recorded failure context must carry the instructive message (issue #336)"
    )
    assert len(reject_calls) >= 3, (f"{label}: the drift gate must reject head changes, unresolvable "
        "push times, and pushes at/after the approving comment"
    )

@pytest.mark.parametrize("wf_path", _REVIEW_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_review_workflow_persists_failure_context(wf_path: Path) -> None:
    """workflow_run cannot identify a dispatch run's PR. The failure artifact
    must supply its number and diagnostic to the post workflow."""
    wf = load_workflow(wf_path)
    steps = job_steps(wf, "analyze")

    verify = next(step
        for step in steps
        if "approved_head_sha" in step.get("run", "") and "exit 1" in step.get("run", "")
    )
    _assert_single_reject_funnel(verify["run"], label=wf_path.name)

    # Other failures use the job-level context handler.
    record = next(step for step in steps if step.get("name") == "Record failure context")
    assert record.get("if", "") == "failure()"
    assert "failure.json" in record["run"]

    upload = next(step for step in steps if step.get("name") == "Upload failure context")
    assert upload.get("if", "") == "failure()"
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "daydream-findings-failure"
    assert upload["with"]["path"] == "findings/failure.json"

@pytest.mark.parametrize("wf_path",
    [TEMPLATES_DIR / "daydream-review.yml", REPO_WORKFLOWS_DIR / "daydream-review.yml",
        TEMPLATES_DIR / "single" / "daydream.yml",
    ], ids=["template", "live", "single"],
)
def test_findings_upload_survives_late_stage_failure(wf_path: Path) -> None:
    """Egress or archive failure can follow a successful findings write. Upload
    findings unconditionally; the separate failure-context upload uses failure()."""
    wf = load_workflow(wf_path)
    steps = job_steps(wf, "analyze")

    upload = next(step for step in steps if step.get("name") == "Upload findings artifact")
    assert upload.get("if", "") == "always()", (
        f"{wf_path.name}: the findings upload must be if: always() so a late-stage "
        "failure does not discard an already-written findings.json"
    )
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "daydream-findings"
    assert upload["with"]["path"] == "findings/"

    # Upload after the review writes findings; unconditional earlier upload would always be empty.
    review = next(step for step in steps if "--findings-out" in step.get("run", ""))
    assert steps.index(review) < steps.index(upload)

@pytest.mark.parametrize("wf_path", _POST_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_surface_analyze_failure_resolves_dispatch_run_pr(wf_path: Path) -> None:
    """Resolve dispatch-run PR identity from failure context; warn and continue
    when no trustworthy target is available."""
    wf = load_workflow(wf_path)
    job = wf["jobs"]["surface-analyze-failure"]
    steps = job["steps"]

    # Cross-run artifact download uses GITHUB_TOKEN actions: read; the App token writes comments.
    effective_perms = job.get("permissions", wf.get("permissions", {}))
    assert effective_perms == {"actions": "read"}

    download = next(step for step in steps if step.get("name") == "Download failure context")
    assert download["uses"].startswith("actions/download-artifact@")
    assert download["with"]["name"] == "daydream-findings-failure"
    assert "run-id" in download["with"]
    assert "github-token" in download["with"]
    assert download.get("continue-on-error") is True

    comment = next(step for step in steps if step.get("name") == "Comment on the PR")
    run = comment["run"]
    assert "findings/failure.json" in run
    assert ".pr_number // empty" in run
    assert ".message // empty" in run
    guard = 'echo "no PR resolvable for the failed analyze run; skipping comment" >&2'
    assert guard in run
    assert "exit 0" in run
    assert run.index(guard) < run.index("exit 0")

def test_single_workflow_head_bound_gate() -> None:
    path = TEMPLATES_DIR / "single" / "daydream.yml"
    wf = load_workflow(path)

    assert "pull_request" not in _wf_triggers(wf)

    gate = wf["jobs"]["gate"]
    assert "approved_head_sha" in gate["outputs"]
    assert "approved_at" in gate["outputs"]
    decide = next(step for step in gate["steps"] if step.get("name") == "Decide and resolve PR")
    assert "approved_head_sha" in decide.get("outputs", {}) or "head.sha" in decide.get("run", "")
    assert "approved_at" in decide.get("outputs", {}) or "approved_at=" in decide.get("run", "")

    # Head resolution must succeed before acknowledgment; otherwise a failed lookup would signal a review.
    ack = next(step for step in gate["steps"] if step.get("name") == "Acknowledge with eyes reaction")
    assert gate["steps"].index(ack) > gate["steps"].index(decide)

    steps = wf["jobs"]["analyze"]["steps"]
    verify = next(step
        for step in steps
        if "approved_head_sha" in step.get("run", "") and "exit 1" in step.get("run", "")
    )
    assert ".head.ref" in verify["run"]
    assert "activity" in verify["run"]
    assert r'\< "$APPROVED_AT"' in verify["run"]
    assert "APPROVED_AT" in wf["jobs"]["analyze"]["env"]

    _assert_single_reject_funnel(verify["run"], label="single/daydream.yml")

    checkout_idx = next(i for i, step in enumerate(steps) if "actions/checkout" in step.get("uses", ""))
    assert steps.index(verify) < checkout_idx
    review = next(step
        for step in steps
        if "daydream" in step.get("run", "") and "--approved-head-sha" in step.get("run", "")
    )
    assert "APPROVED_HEAD_SHA" in review["env"]


# Execute the workflow match step under GitHub's actual shell, testing outputs rather than copied
# regexes. Locate it by stable id, not display name. Live, packaged and single-file copies must agree.

_MATCH_STEP_SOURCES = [pytest.param(REPO_WORKFLOWS_DIR / "daydream-command.yml", "dispatch", id="live-command"),
    pytest.param(TEMPLATES_DIR / "daydream-command.yml", "dispatch", id="template-command"),
    pytest.param(TEMPLATES_DIR / "single" / "daydream.yml", "gate", id="single"),
]

# Empty expected command means no run. Plurals, spacing, handle boundaries and casing matter.
_MATCH_CASES = [pytest.param("@bot review", "bot", "review", id="review"),
    pytest.param("@bot add sequence diagram", "bot", "sequence", id="sequence-full"),
    pytest.param("@bot add sequence", "bot", "sequence", id="sequence-alias"),
    pytest.param("@bot add flowchart", "bot", "flowchart", id="flowchart"),
    pytest.param("@bot add  flowchart", "bot", "flowchart", id="flowchart-extra-space"),
    pytest.param("@bot add sequence\tdiagram", "bot", "sequence", id="sequence-tab"),
    pytest.param("@bot add sequence diagram please", "bot", "sequence", id="sequence-trailing-text"),
    pytest.param("@bot review please", "bot", "review", id="review-trailing-text"),
    pytest.param("please @bot review", "bot", "review", id="review-leading-text"),
    pytest.param("some context\n@bot add flowchart\nthanks", "bot", "flowchart", id="flowchart-mid-body"),
    pytest.param("some context\n@bot add flowchart\n" + "🌐" * 30_000, "bot", "flowchart",
                 id="flowchart-large-unicode-body"),
    pytest.param("@bot add flowchart\r\n", "bot", "flowchart", id="flowchart-crlf"),
    # grep is line-oriented, so a mention split across lines falls back to the
    # `add sequence` alias rather than matching nothing.
    pytest.param("@bot add sequence\ndiagram", "bot", "sequence", id="sequence-alias-across-newline"),
    # Branch precedence is review > sequence > flowchart.
    pytest.param("@bot review\n@bot add flowchart", "bot", "review", id="review-wins-over-flowchart"),
    # Regex metacharacters in handles must match literally.
    pytest.param("@my.bot+1 review", "my.bot+1", "review", id="metachar-handle-review"),
    pytest.param("@my.bot+1 add sequence diagram", "my.bot+1", "sequence", id="metachar-handle-sequence"),
    pytest.param("@my.bot+1 add flowchart", "my.bot+1", "flowchart", id="metachar-handle-flowchart"),
    pytest.param("@zzz add flowchart", ".*", "", id="metachar-handle-not-a-wildcard"),
    pytest.param("@bot reviews", "bot", "", id="no-reviews-plural"),
    pytest.param("@bot add flowcharts", "bot", "", id="no-flowcharts-plural"),
    pytest.param("@bot add sequence diagrams", "bot", "", id="no-sequence-diagrams-plural"),
    pytest.param("@bot add sequence diagrams please", "bot", "", id="no-sequence-diagrams-plural-trailing"),
    pytest.param("@bot addflowchart", "bot", "", id="no-missing-space"),
    pytest.param("x@bot add flowchart", "bot", "", id="no-handle-mid-word"),
    pytest.param("@bot ADD FLOWCHART", "bot", "", id="no-uppercase"),
    # The sequence veto covers the whole body: a malformed mention vetoes a valid one too.
    pytest.param("@bot add sequence\n@bot add sequence diagrams", "bot", "", id="sequence-veto-fails-closed"),
]


def _run_match_step(path: Path, job: str, body: str, handle: str, out: Path) -> dict[str, str]:
    """Execute *path*'s own `id: match` step and return the outputs it wrote."""
    wf = load_workflow(path)
    step = next(s for s in job_steps(wf, job) if s.get("id") == "match")
    workflow_shell = wf.get("defaults", {}).get("run", {}).get("shell")
    job_shell = wf["jobs"][job].get("defaults", {}).get("run", {}).get("shell", workflow_shell)
    assert step.get("shell", job_shell) is None, "match-step harness requires the workflow's unspecified shell"
    # GitHub's unspecified shell is `bash -e {0}`. Explicit `shell: bash` adds
    # pipefail, which can turn grep's successful early exit into a false match.
    script = out.with_suffix(".sh")
    script.write_text(step["run"], encoding="utf-8")
    out.write_text("", encoding="utf-8")
    subprocess.run(["bash", "-e", str(script)],
        env={"PATH": os.environ["PATH"], "BODY": body, "BOT_HANDLE": handle, "GITHUB_OUTPUT": str(out)}, check=True,
        capture_output=True, text=True,
    )
    parsed: dict[str, str] = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        key, _, value = line.partition("=")
        parsed[key] = value
    return parsed

@pytest.mark.parametrize(("wf_path", "job"), _MATCH_STEP_SOURCES)
@pytest.mark.parametrize(("body", "handle", "expected"), _MATCH_CASES)
def test_match_step_recognizes_exactly_the_three_bot_commands(
    wf_path: Path, job: str, body: str, handle: str, expected: str, tmp_path: Path
) -> None:
    outputs = _run_match_step(wf_path, job, body, handle, tmp_path / "gh-output")

    # Command is always written for the single-file job output; matched remains the downstream gate.
    assert outputs["command"] == expected
    assert outputs["matched"] == ("true" if expected else "false")

@pytest.mark.parametrize("wf_path", _COMMAND_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_command_workflow_dispatches_the_matched_command(wf_path: Path) -> None:
    """Pass the matched command through env, preserving approval binding and
    keeping event interpolation out of shell code."""
    wf = load_workflow(wf_path)
    dispatch = _dispatch_step(wf)

    assert dispatch["env"]["COMMAND"] == "${{ steps.match.outputs.command }}"
    assert '-f command="$COMMAND"' in dispatch["run"]
    assert not _EVENT_INTERP.search(dispatch["run"])

@pytest.mark.parametrize("wf_path", _REVIEW_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_review_workflow_command_input_is_a_bounded_choice(wf_path: Path) -> None:
    """Only approved commands may select a credential-bearing run. Installed
    workflow validation enforces the same three-option bound."""
    inputs = _wf_triggers(load_workflow(wf_path))["workflow_dispatch"]["inputs"]
    command = inputs["command"]

    assert command["type"] == "choice"
    assert command["options"] == ["review", "sequence", "flowchart"]
    assert command["default"] == "review"
    assert command["required"] is False

@pytest.mark.parametrize("wf_path", _REVIEW_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_review_workflow_branches_on_command_and_fails_closed(wf_path: Path) -> None:
    """Diagram and review runs share head-binding and artifact contracts; reject
    unknown commands rather than dispatching a different run."""
    steps = job_steps(load_workflow(wf_path), "analyze")
    run_step = next(step for step in steps if "daydream --review" in step.get("run", ""))
    run = run_step["run"]

    assert run_step["env"]["COMMAND"] == "${{ inputs.command }}"
    assert 'case "$COMMAND" in' in run
    assert 'daydream --diagram-only "$COMMAND"' in run
    for flag in ("--non-interactive", '--pr-number "$PR_NUMBER"', '--approved-head-sha "$APPROVED_HEAD_SHA"',
        "--findings-out findings/findings.json", '--base "origin/$BASE_REF"',
    ):
        assert flag in run
    assert "exit 1" in run
    assert "esac" in run

def test_single_workflow_exposes_the_matched_command_to_analyze() -> None:
    """The match step always writes command, even for non-matching comments.
    The review branch keeps the default deep pipeline without --review."""
    wf = load_workflow(TEMPLATES_DIR / "single" / "daydream.yml")

    assert wf["jobs"]["gate"]["outputs"]["command"] == "${{ steps.match.outputs.command }}"
    analyze = wf["jobs"]["analyze"]
    assert analyze["env"]["COMMAND"] == "${{ needs.gate.outputs.command }}"

    run = next(step["run"] for step in analyze["steps"] if "daydream --non-interactive" in step.get("run", ""))
    assert 'case "$COMMAND" in' in run
    assert 'daydream --diagram-only "$COMMAND"' in run
    assert "daydream --review" not in run
    assert "exit 1" in run and "esac" in run

@pytest.mark.parametrize("wf_path", sorted(TEMPLATES_DIR.rglob("*.yml")), ids=lambda p: p.name)
def test_no_event_data_interpolated_into_run_steps(wf_path: Path) -> None:
    wf = load_workflow(wf_path)
    for job_name, job in wf["jobs"].items():
        for step in job["steps"]:
            if "run" in step:
                assert not _EVENT_INTERP.search(step["run"]), (
                    f"{wf_path.name}:{job_name}: event data must reach run: via env:, never ${{{{ }}}} interpolation"
                )

@pytest.mark.parametrize("wf_path", _BOT_WORKFLOW_PATHS, ids=lambda p: p.relative_to(_REPO_ROOT).as_posix(),)
def test_bot_workflow_action_references_are_pinned_to_commit_shas(wf_path: Path) -> None:
    wf = load_workflow(wf_path)
    rel = wf_path.relative_to(_REPO_ROOT).as_posix()
    for ref in _action_references(wf):
        if ref.startswith("./"):
            continue
        assert _PINNED_ACTION_RE.fullmatch(ref), (f"{rel}: non-local action reference {ref!r} is not a full commit SHA "
            f"(expected owner/repo@<40 hex chars>)"
        )


# All workflows install pinned revisions. Released surfaces follow the package release; diagram
# surfaces may need a reviewed newer commit.

@pytest.mark.parametrize("post_path", _POST_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_split_setup_preserves_privilege_split(post_path: Path) -> None:
    review = load_workflow(TEMPLATES_DIR / "daydream-review.yml")
    review_text = (TEMPLATES_DIR / "daydream-review.yml").read_text(encoding="utf-8")
    command_text = (TEMPLATES_DIR / "daydream-command.yml").read_text(encoding="utf-8")
    post = load_workflow(post_path)
    post_text = post_path.read_text(encoding="utf-8")

    # Phase A runs untrusted PR code: read-only, and its only secret is the API key.
    assert review["permissions"] == {"contents": "read"}
    assert set(_SECRET_REF_RE.findall(review_text)) == {"ANTHROPIC_API_KEY"}

    # No privileged command or post job may check out code.
    assert "actions/checkout" not in command_text
    for job in post["jobs"].values():
        assert not has_checkout(job)
    assert set(_SECRET_REF_RE.findall(post_text)) == {"DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"}

@pytest.mark.parametrize("wf_path",
    [TEMPLATES_DIR / "daydream-post.yml", REPO_WORKFLOWS_DIR / "daydream-post.yml",
        TEMPLATES_DIR / "single" / "daydream.yml",
    ], ids=["template", "live", "single"],
)
def test_post_job_token_can_read_head_evidence(wf_path: Path) -> None:
    """The App-key job avoids PR checkout. Diagram citations therefore require
    contents: read for the contents API at the approved head SHA."""
    steps = [step
        for step in load_workflow(wf_path)["jobs"]["post"]["steps"]
        if str(step.get("uses", "")).startswith("actions/create-github-app-token@")
    ]
    assert steps
    assert all(step["with"]["permission-contents"] == "read" for step in steps)

@pytest.mark.parametrize("wf_path", _POST_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_post_findings_step_exports_bot_login(wf_path: Path) -> None:
    """Both post workflows must pass the deposited bot login. Without it, REST
    author dedup cannot run and GraphQL falls back to viewerDidAuthor alone."""
    text = wf_path.read_text(encoding="utf-8")
    assert "BOT_LOGIN: ${{ vars.DAYDREAM_BOT_HANDLE }}" in text, (
        f"{wf_path.name}: Post findings step must export BOT_LOGIN from vars.DAYDREAM_BOT_HANDLE (issue #254)"
    )
    assert '--bot-login "$BOT_LOGIN"' in text, f"{wf_path.name}: Post findings step must pass --bot-login explicitly"

@pytest.mark.parametrize("wf_path", _POST_WORKFLOW_PATHS, ids=_TEMPLATE_AND_LIVE_IDS)
def test_failure_comment_target_never_uses_findings_artifact(wf_path: Path) -> None:
    """Failure handlers trust only derived or event PR identity: findings.json
    is unvalidated here. Missing both identities must exit without posting."""
    wf = load_workflow(wf_path)
    steps = job_steps(wf, "post")
    handler = next(s for s in steps if s.get("name") == "Surface failure on the PR")
    run = handler["run"]

    assert run.count("PR_NUMBER=") == 1, (f"{wf_path.name}: failure handler must assign PR_NUMBER exactly once")
    assert 'PR_NUMBER="${DERIVED_PR_NUMBER:-$EVENT_PR_NUMBER}"' in run, (
        f"{wf_path.name}: failure target must come only from DERIVED_PR_NUMBER or EVENT_PR_NUMBER (issue #384)"
    )

    assert "findings/findings.json" not in run, (
        f"{wf_path.name}: failure handler must not read findings/findings.json (issue #384)"
    )
    assert "jq" not in run, (f"{wf_path.name}: failure handler must not run jq over an artifact (issue #384)")

    guard = 'echo "no PR resolvable; cannot surface the failure" >&2'
    exit_guard = "exit 0"
    assert guard in run, f"{wf_path.name}: empty-result diagnostic must be present"
    assert exit_guard in run, f"{wf_path.name}: exit 0 must be present in the empty-result guard"
    assert run.index(guard) < run.index(exit_guard), (f"{wf_path.name}: empty-result diagnostic must precede exit 0")
    assert run.index(exit_guard) < run.index('gh api "repos/${REPO}/issues/${PR_NUMBER}/comments"'), (
        f"{wf_path.name}: exit 0 must precede the comment write (issue #384)"
    )

def test_single_setup_preserves_privilege_split() -> None:
    wf = load_workflow(TEMPLATES_DIR / "single" / "daydream.yml")
    text = (TEMPLATES_DIR / "single" / "daydream.yml").read_text(encoding="utf-8")

    # Only analyze touches untrusted PR code: read-only, no App key, and no persisted GitHub token.
    analyze = wf["jobs"]["analyze"]
    assert analyze["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "DAYDREAM_APP" not in yaml.safe_dump(analyze)
    checkout = next(s for s in analyze["steps"] if "actions/checkout" in s.get("uses", ""))
    assert checkout["with"]["persist-credentials"] is False

    # App-key jobs never check out code; no dispatch means actions: write is unnecessary.
    for job_name in ("gate", "post", "surface-failure"):
        assert not has_checkout(wf["jobs"][job_name])
    assert "permission-actions" not in text

    # always() overrides GitHub's implicit success() so analyze failure still reaches the failure job.
    assert "always()" in wf["jobs"]["surface-failure"]["if"]

    assert set(_SECRET_REF_RE.findall(text)) == {"ANTHROPIC_API_KEY", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"}
