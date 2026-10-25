"""Approval, privilege and artifact contracts for complete shipped workflow bundles.

The split installer lands command, review and post together. Bind their inputs
and artifacts in each actual bundle; exercise the shared Bash matcher separately.
"""

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
_SECRET_REF_RE = re.compile(r"secrets\.([A-Za-z0-9_]+)")
_PINNED_ACTION_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")
_EVENT_INTERP = re.compile(r"\$\{\{[^}]*github\.event\.(comment|issue|pull_request|workflow_run|review)[^}]*\}\}")


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
    return next(
        step for step in job_steps(wf, "dispatch") if "gh workflow run daydream-review.yml" in step.get("run", "")
    )


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
    assert write_idx[0] < exit_idx[0], f"{label}: the helper must record the failure context before exiting"
    assert '"message"' in lines[write_idx[0]], (
        f"{label}: the recorded failure context must carry the instructive message (issue #336)"
    )
    assert len(reject_calls) >= 3, (
        f"{label}: the drift gate must reject head changes, unresolvable "
        "push times, and pushes at/after the approving comment"
    )


def _assert_execution_policy(*workflows: dict[str, Any]) -> None:
    """Each job in the bundle keeps hosted runners, pinned actions and env-only event input."""
    for workflow in workflows:
        for job in workflow["jobs"].values():
            assert job["runs-on"] == "ubuntu-24.04"
            assert not any(_EVENT_INTERP.search(step.get("run", "")) for step in job["steps"])
        for reference in _action_references(workflow):
            assert reference.startswith("./") or _PINNED_ACTION_RE.fullmatch(reference), reference


@pytest.mark.parametrize(
    ("directory", "credential"),
    [(TEMPLATES_DIR, "ANTHROPIC_API_KEY"), (REPO_WORKFLOWS_DIR, "OPENAI_API_KEY")],
    ids=["packaged", "live"],
)
def test_split_bundle_binds_approval_review_artifacts_and_posting(directory: Path, credential: str) -> None:
    """Each installed three-file bundle joins immutable approval to unprivileged review and App-only posting."""
    texts = {role: (directory / f"daydream-{role}.yml").read_text() for role in ("command", "review", "post")}
    command, review, post = (
        load_workflow(directory / f"daydream-{role}.yml") for role in ("command", "review", "post")
    )
    _assert_execution_policy(command, review, post)
    assert {role: set(_SECRET_REF_RE.findall(text)) for role, text in texts.items()} == {
        "command": {"DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"},
        "review": {credential},
        "post": {"DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"},
    }
    assert review["permissions"] == {"contents": "read"}
    assert "actions/checkout" not in texts["command"]
    assert not any(has_checkout(job) for job in post["jobs"].values())
    assert "pull_request" not in _wf_triggers(review)
    assert _wf_triggers(post)["workflow_run"]["workflows"] == [review["name"]] == ["Daydream Review"]

    # The dispatched values must arrive at the same bundle's bounded review inputs.
    dispatch_steps = job_steps(command, "dispatch")
    dispatch = _dispatch_step(command)
    assert "gh workflow run daydream-review.yml" in dispatch["run"]
    assert "gh api" in dispatch["run"] and ".head.sha" in dispatch["run"]
    assert '-f approved_head_sha="$HEAD_SHA"' in dispatch["run"]
    assert '-f approved_at="$COMMENT_CREATED_AT"' in dispatch["run"]
    assert '-f command="$COMMAND"' in dispatch["run"]
    assert {"PR_NUMBER", "COMMENT_CREATED_AT"} <= dispatch["env"].keys()
    assert dispatch["env"]["COMMAND"] == "${{ steps.match.outputs.command }}"
    ack = next(step for step in dispatch_steps if step.get("name") == "Acknowledge with eyes reaction")
    assert dispatch_steps.index(dispatch) < dispatch_steps.index(ack)
    assert not any(fn in ack.get("if", "") for fn in ("always()", "failure()", "cancelled()"))
    assert "continue-on-error" not in dispatch
    inputs = _wf_triggers(review)["workflow_dispatch"]["inputs"]
    assert {"approved_head_sha", "approved_at"} <= inputs.keys()
    assert {key: inputs["command"][key] for key in ("type", "options", "default", "required")} == {
        "type": "choice",
        "options": ["review", "sequence", "flowchart"],
        "default": "review",
        "required": False,
    }

    steps = job_steps(review, "analyze")
    by_name = {step.get("name"): step for step in steps}
    verify = next(step for step in steps if "approved_head_sha" in step.get("run", "") and "exit 1" in step["run"])
    assert ".head.ref" in verify["run"] and "activity" in verify["run"]
    assert r'\< "$APPROVED_AT"' in verify["run"] and "APPROVED_AT" in verify["env"]
    assert verify["env"]["APPROVED_HEAD_SHA"] == "${{ inputs.approved_head_sha }}"
    checkout = next(step for step in steps if "actions/checkout" in step.get("uses", ""))
    assert steps.index(verify) < steps.index(checkout)
    assert checkout["with"]["ref"] == "${{ inputs.approved_head_sha }}"
    _assert_single_reject_funnel(verify["run"], label=directory.as_posix())
    run_step = next(step for step in steps if "daydream --review" in step.get("run", ""))
    run = run_step["run"]
    assert run_step["env"]["APPROVED_HEAD_SHA"] == "${{ inputs.approved_head_sha }}"
    assert run_step["env"]["COMMAND"] == "${{ inputs.command }}"
    assert 'case "$COMMAND" in' in run and 'daydream --diagram-only "$COMMAND"' in run
    assert all(
        flag in run
        for flag in (
            "--non-interactive",
            '--pr-number "$PR_NUMBER"',
            '--approved-head-sha "$APPROVED_HEAD_SHA"',
            "--findings-out findings/findings.json",
            '--base "origin/$BASE_REF"',
        )
    )
    assert "exit 1" in run and "esac" in run
    if credential == "OPENAI_API_KEY":
        cleanup = next(step for step in steps if "auth.json" in step.get("run", ""))
        assert steps.index(run_step) < steps.index(cleanup) and cleanup.get("if") == "always()"
    else:
        assert not any("auth.json" in step.get("run", "") for step in steps)

    # Success and failure artifacts join their actual downstream consumers, even after a late failure.
    upload = by_name["Upload findings artifact"]
    failure_upload = by_name["Upload failure context"]
    record = by_name["Record failure context"]
    assert record.get("if") == "failure()" and "failure.json" in record["run"]
    assert upload.get("if") == "always()" and steps.index(run_step) < steps.index(upload)
    assert failure_upload.get("if") == "failure()"
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert failure_upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["path"] == "findings/"
    assert failure_upload["with"]["path"] == "findings/failure.json"
    post_steps = job_steps(post, "post")
    download = next(step for step in post_steps if step.get("name") == "Download findings artifact")
    assert upload["with"]["name"] == download["with"]["name"] == "daydream-findings"
    assert download["with"]["run-id"] == "${{ github.event.workflow_run.id }}"
    failure_job = post["jobs"]["surface-analyze-failure"]
    assert failure_job.get("permissions", post.get("permissions", {})) == {"actions": "read"}
    failure_download = next(step for step in failure_job["steps"] if step.get("name") == "Download failure context")
    assert failure_download["uses"].startswith("actions/download-artifact@")
    assert failure_upload["with"]["name"] == failure_download["with"]["name"] == "daydream-findings-failure"
    assert failure_download["with"]["run-id"] == "${{ github.event.workflow_run.id }}"
    assert "github-token" in failure_download["with"] and failure_download.get("continue-on-error") is True
    comment = next(step for step in failure_job["steps"] if step.get("name") == "Comment on the PR")["run"]
    assert all(
        fragment in comment for fragment in ("findings/failure.json", ".pr_number // empty", ".message // empty")
    )
    guard = 'echo "no PR resolvable for the failed analyze run; skipping comment" >&2'
    assert guard in comment and "exit 0" in comment and comment.index(guard) < comment.index("exit 0")
    posting_tokens = [
        step for step in post_steps if step.get("uses", "").startswith("actions/create-github-app-token@")
    ]
    assert posting_tokens and all(step["with"]["permission-contents"] == "read" for step in posting_tokens)
    assert "BOT_LOGIN: ${{ vars.DAYDREAM_BOT_HANDLE }}" in texts["post"]
    assert '--bot-login "$BOT_LOGIN"' in texts["post"]
    failure = next(step["run"] for step in post_steps if step.get("name") == "Surface failure on the PR")
    assert failure.count("PR_NUMBER=") == 1 and 'PR_NUMBER="${DERIVED_PR_NUMBER:-$EVENT_PR_NUMBER}"' in failure
    assert "findings/findings.json" not in failure and "jq" not in failure
    guard = 'echo "no PR resolvable; cannot surface the failure" >&2'
    assert guard in failure and "exit 0" in failure
    assert (
        failure.index(guard)
        < failure.index("exit 0")
        < failure.index('gh api "repos/${REPO}/issues/${PR_NUMBER}/comments"')
    )


def test_single_bundle_joins_gate_analyze_and_privileged_publication() -> None:
    path = TEMPLATES_DIR / "single" / "daydream.yml"
    text = path.read_text()
    wf = load_workflow(path)
    _assert_execution_policy(wf)
    assert "pull_request" not in _wf_triggers(wf)
    assert set(_SECRET_REF_RE.findall(text)) == {"ANTHROPIC_API_KEY", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"}
    gate, analyze = wf["jobs"]["gate"], wf["jobs"]["analyze"]
    assert {"approved_head_sha", "approved_at"} <= gate["outputs"].keys()
    assert gate["outputs"]["command"] == "${{ steps.match.outputs.command }}"
    decide = next(step for step in gate["steps"] if step.get("name") == "Decide and resolve PR")
    assert "approved_head_sha" in decide.get("outputs", {}) or "head.sha" in decide.get("run", "")
    assert "approved_at" in decide.get("outputs", {}) or "approved_at=" in decide.get("run", "")
    ack = next(step for step in gate["steps"] if step.get("name") == "Acknowledge with eyes reaction")
    assert gate["steps"].index(decide) < gate["steps"].index(ack)
    assert analyze["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "DAYDREAM_APP" not in yaml.safe_dump(analyze)
    assert analyze["env"]["COMMAND"] == "${{ needs.gate.outputs.command }}"
    assert analyze["env"]["APPROVED_AT"] == "${{ needs.gate.outputs.approved_at }}"
    steps = job_steps(wf, "analyze")
    verify = next(step for step in steps if "approved_head_sha" in step.get("run", "") and "exit 1" in step["run"])
    assert ".head.ref" in verify["run"] and "activity" in verify["run"] and r'\< "$APPROVED_AT"' in verify["run"]
    _assert_single_reject_funnel(verify["run"], label=path.as_posix())
    checkout = next(step for step in steps if "actions/checkout" in step.get("uses", ""))
    assert steps.index(verify) < steps.index(checkout) and checkout["with"]["persist-credentials"] is False
    review = next(step for step in steps if "daydream --non-interactive" in step.get("run", ""))
    assert "APPROVED_HEAD_SHA" in review["env"]
    assert 'case "$COMMAND" in' in review["run"] and 'daydream --diagram-only "$COMMAND"' in review["run"]
    assert "daydream --review" not in review["run"] and "exit 1" in review["run"] and "esac" in review["run"]
    upload = next(step for step in steps if step.get("name") == "Upload findings artifact")
    assert steps.index(review) < steps.index(upload) and upload.get("if") == "always()"
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "daydream-findings" and upload["with"]["path"] == "findings/"
    for name in ("gate", "post", "surface-failure"):
        assert not has_checkout(wf["jobs"][name])
    assert "permission-actions" not in text and "always()" in wf["jobs"]["surface-failure"]["if"]
    tokens = [
        step for step in job_steps(wf, "post") if step.get("uses", "").startswith("actions/create-github-app-token@")
    ]
    assert tokens and all(step["with"]["permission-contents"] == "read" for step in tokens)


_MATCH_STEP_SOURCES = [
    (REPO_WORKFLOWS_DIR / "daydream-command.yml", "dispatch"),
    (TEMPLATES_DIR / "daydream-command.yml", "dispatch"),
    (TEMPLATES_DIR / "single" / "daydream.yml", "gate"),
]

# Empty expected command means no run. Plurals, spacing, handle boundaries and casing matter.
_MATCH_CASES = [
    pytest.param("@bot review", "bot", "review", id="review"),
    pytest.param("@bot add sequence diagram", "bot", "sequence", id="sequence-full"),
    pytest.param("@bot add sequence", "bot", "sequence", id="sequence-alias"),
    pytest.param("@bot add flowchart", "bot", "flowchart", id="flowchart"),
    pytest.param("@bot add  flowchart", "bot", "flowchart", id="flowchart-extra-space"),
    pytest.param("@bot add sequence\tdiagram", "bot", "sequence", id="sequence-tab"),
    pytest.param("@bot add sequence diagram please", "bot", "sequence", id="sequence-trailing-text"),
    pytest.param("@bot review please", "bot", "review", id="review-trailing-text"),
    pytest.param("please @bot review", "bot", "review", id="review-leading-text"),
    pytest.param(
        "some context\n@bot add flowchart\n" + "🌐" * 30_000, "bot", "flowchart", id="flowchart-large-unicode-body"
    ),
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


@pytest.fixture(scope="module")
def command_match_script() -> str:
    """Byte and shell parity make every behavioral assertion cover all shipped copies."""
    scripts = []
    for path, job in _MATCH_STEP_SOURCES:
        wf = load_workflow(path)
        step = next(s for s in job_steps(wf, job) if s.get("id") == "match")
        workflow_shell = wf.get("defaults", {}).get("run", {}).get("shell")
        job_shell = wf["jobs"][job].get("defaults", {}).get("run", {}).get("shell", workflow_shell)
        assert step.get("shell", job_shell) is None, path
        scripts.append(step["run"])
    assert scripts[0] == scripts[1] == scripts[2]
    return str(scripts[0])


@pytest.mark.parametrize(("body", "handle", "expected"), _MATCH_CASES)
def test_match_step_recognizes_exactly_the_three_bot_commands(
    command_match_script: str,
    body: str,
    handle: str,
    expected: str,
    tmp_path: Path,
) -> None:
    # GitHub's unspecified shell is `bash -e {0}`. Explicit `shell: bash` adds
    # pipefail, which can turn grep's successful early exit into a false match.
    out = tmp_path / "gh-output"
    script = out.with_suffix(".sh")
    script.write_text(command_match_script, encoding="utf-8")
    out.write_text("", encoding="utf-8")
    subprocess.run(
        ["bash", "-e", str(script)],
        env={"PATH": os.environ["PATH"], "BODY": body, "BOT_HANDLE": handle, "GITHUB_OUTPUT": str(out)},
        check=True,
        capture_output=True,
        text=True,
    )
    parsed: dict[str, str] = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        key, _, value = line.partition("=")
        parsed[key] = value
    # Command is always written for the single-file job output; matched remains the downstream gate.
    assert parsed["command"] == expected
    assert parsed["matched"] == ("true" if expected else "false")
