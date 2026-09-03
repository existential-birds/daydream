"""Invariant tests for the shipped GitHub Actions workflow templates.

A workflow's behavior can only be validated by running it in CI, so these tests
deliberately do NOT restate the YAML (trigger lists, exact ``if:`` strings,
permission dicts). They guard only the handful of properties a single file-read
cannot verify and that a careless edit could silently break:

- No untrusted event data is interpolated into a ``run:`` body (injection).
- Every non-local action ``uses:`` in the live and shipped bot workflows
  resolves to a full commit SHA (never a mutable tag/branch/expression).
- The daydream install stays pinned to an immutable release commit (cross-file
  drift against ``pyproject.toml``), in every live and shipped workflow.
- Every App-token action in the live and packaged posting workflows stays pinned to the approved v3.2.0 commit.
- The privilege split holds: the job that checks out untrusted PR code never
  holds the App key, and the privileged jobs never check out PR code.
- The repo's own Codex dogfood workflow persists ``codex login`` before the
  review runs (``codex exec`` does not read ``OPENAI_API_KEY`` for auth), and
  the repository workflow README names ``OPENAI_API_KEY`` as the credential the
  live Codex workflow consumes.
- The repository workflow README declares these files as repository-only Codex
  dogfood configuration and points to the packaged install guide (never copies
  ``ANTHROPIC_API_KEY``).
- The CI actionlint step and the Makefile actionlint target both reference the
  actionlint image by immutable OCI digest (``rhysd/actionlint:1.7.7@sha256:…``),
  so a revert to a mutable tag, or drift between the two, fails the suite.
- The CI actionlint step covers every workflow the project ships — the repo's
  own top-level workflows plus all recursively discovered template workflows
  (the nested ``single/daydream.yml`` included) — and each selector still has
  to match at least one real workflow file (no stale selectors).

PyYAML parses the bare ``on:`` key as boolean ``True``; ``wf_on()`` normalizes it.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = _REPO_ROOT / "daydream" / "templates" / "workflows"
REPO_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
# The digest below matches the immutable rhysd/actionlint:1.7.7 manifest digest
# verified against the live OCI registry (`docker buildx imagetools inspect
# rhysd/actionlint:1.7.7` / `docker manifest inspect …@sha256:887a…147e9`;
# both the tagged ref and the raw sha256 resolve to it). The registry is the only
# authoritative source: this constant is a pin, not a verification of itself.
# Any future edit to the digest MUST be re-verified against that registry before
# landing, or CI `docker pull` will fail at run time.
_ACTIONLINT_IMAGE = "rhysd/actionlint:1.7.7@sha256:887a259a5a534f3c4f36cb02dca341673c6089431057242cdc931e9f133147e9"

_SECRET_REF_RE = re.compile(r"secrets\.([A-Za-z0-9_]+)")
_ACTIONLINT_REF_RE = re.compile(r"rhysd/actionlint:[^\s`]+")


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


# Action-ref policy (all bot workflows, live + shipped): every non-local
# `uses:` must resolve to a full commit SHA, never a mutable tag, branch,
# expression, Docker reference, short hash, or non-hex revision, and must carry
# the human-readable `# vX.Y.Z` inline comment naming the release that SHA pins.
# Repo-local `./…` actions are exempt. Rides the root pytest suite in ci.yml.

_BOT_WORKFLOW_PATHS = sorted(
    [*REPO_WORKFLOWS_DIR.glob("daydream-*.yml"), *TEMPLATES_DIR.rglob("*.yml")],
    key=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)

_PINNED_ACTION_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")

# Approved SHA → release mapping every pinned action ref must match, so the
# `# vX.Y.Z` inline comment is verifiable rather than decorative (yaml.safe_load
# strips it, so the comment can only be checked against the raw text). Concrete
# pairs, in the style of _APP_TOKEN_ACTION below: a refloated pin or a mistyped
# comment fails loudly instead of being silently absorbed by a wildcard.
_PINNED_ACTION_VERSIONS = {
    "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5": "v4.3.1",
    "astral-sh/setup-uv@38f3f104447c67c051c4a08e39b64a148898af3a": "v4.2.0",
    "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02": "v4.6.2",
    "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093": "v4.3.0",
    "actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1": "v3.2.0",
}

_USES_LINE_RE = re.compile(r"^\s*uses:\s*(?P<ref>\S+)(?:\s*#\s*(?P<comment>\S+))?$")

# Release tag → PEELED commit SHA for every daydream release the workflow pins
# may reference. Values are the peeled (`^{}` target) commits of annotated tags,
# NOT tag-object SHAs — `git ls-remote origin 'refs/tags/vX.Y.Z'` on an
# annotated tag reports the tag object (e.g. `9abbaeb3…` for v0.28.0), which is
# the classic trap; use `git ls-remote origin 'refs/tags/vX.Y.Z^{}'` instead.
# History is retained (never prune old entries) for provenance. A cross-check
# enforces both sides: every entry is either pinned by a workflow install ref
# or a strictly older release retained for provenance. Values are also
# verified offline against this repo's own release-tag refs (peeled targets),
# the only way to tell the annotated-tag OBJECT sha from the peeled commit; a
# checkout that carries no tags skips that check, so the refs are trusted,
# never fetched or verified against GitHub (intentionally offline).
_DAYDREAM_RELEASE_COMMITS: dict[str, str] = {
    "v0.27.0": "805fd0f105fe803a90a6a8b2c2d9646a4041eccc",
    "v0.28.0": "e7741f17fc998a675ed2fe3f364d2e646cde5518",
}

_RELEASE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_DAYDREAM_INSTALL_WORKFLOW_PATHS = [
    REPO_WORKFLOWS_DIR / "daydream-review.yml",
    REPO_WORKFLOWS_DIR / "daydream-post.yml",
    TEMPLATES_DIR / "daydream-review.yml",
    TEMPLATES_DIR / "daydream-post.yml",
    TEMPLATES_DIR / "single" / "daydream.yml",
]


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


# Injection guard (all templates): untrusted event data must reach run: via env:,
# never ${{ }} interpolation, which would splice attacker-controlled text into the
# shell.

_EVENT_INTERP = re.compile(r"\$\{\{[^}]*github\.event\.(comment|issue|pull_request|workflow_run|review)[^}]*\}\}")


def test_command_workflows_dispatch_approved_head() -> None:
    """The live trusted command workflow binds the PR head at approval time."""
    path = REPO_WORKFLOWS_DIR / "daydream-command.yml"
    wf = load_workflow(path)
    dispatch = next(
        step
        for step in job_steps(wf, "dispatch")
        if "gh workflow run daydream-review.yml" in step.get("run", "")
    )

    assert "gh api" in dispatch["run"] and ".head.sha" in dispatch["run"]
    assert '-f approved_head_sha="$HEAD_SHA"' in dispatch["run"]
    assert '-f approved_at="$COMMENT_CREATED_AT"' in dispatch["run"]
    assert "PR_NUMBER" in dispatch["env"]
    assert "COMMENT_CREATED_AT" in dispatch["env"]
    assert not any(
        _EVENT_INTERP.search(step.get("run", ""))
        for step in job_steps(wf, "dispatch")
    )


def test_template_command_workflow_dispatches_approved_head() -> None:
    """The packaged command template binds the PR head at approval time."""
    path = TEMPLATES_DIR / "daydream-command.yml"
    wf = load_workflow(path)
    dispatch = next(
        step
        for step in job_steps(wf, "dispatch")
        if "gh workflow run daydream-review.yml" in step.get("run", "")
    )
    assert "approved_head_sha" in dispatch["run"]
    assert "approved_at" in dispatch["run"]
    assert "gh api" in dispatch["run"] and ".head.sha" in dispatch["run"]
    assert "PR_NUMBER" in dispatch["env"]
    assert "COMMENT_CREATED_AT" in dispatch["env"]
    assert "actions/checkout" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "wf_path",
    [TEMPLATES_DIR / "daydream-command.yml", REPO_WORKFLOWS_DIR / "daydream-command.yml"],
    ids=["template", "live"],
)
def test_command_workflow_acknowledges_only_after_successful_dispatch(wf_path: Path) -> None:
    """The 👀 acknowledgement must come after the dispatch step and stay gated
    on its success. A failed dispatch (unresolvable head SHA or a `gh workflow
    run` error) creates no Daydream Review run, so surface-analyze-failure
    never fires and nothing surfaces on the PR — a reaction posted before or
    regardless of the dispatch outcome would mis-signal success. The step
    ordering IS the mechanism, so a reordering that moves the ack above the
    dispatch (or drops its success gate) must fail here.
    """
    wf = load_workflow(wf_path)
    steps = job_steps(wf, "dispatch")
    dispatch = next(
        step
        for step in steps
        if "gh workflow run daydream-review.yml" in step.get("run", "")
    )
    ack = next(step for step in steps if step.get("name") == "Acknowledge with eyes reaction")

    # The ack must follow the dispatch in step order.
    assert steps.index(dispatch) < steps.index(ack), (
        f"{wf_path.name}: the 👀 acknowledgement must come after the dispatch step "
        "so a dispatch failure never mis-signals success"
    )

    # The ack's if: must keep GitHub's implicit success() gate — no status
    # function (always()/failure()/cancelled()) may let it run after a failed
    # dispatch, and the dispatch step may not continue-on-error (which would
    # keep the job alive past its failure and still post the reaction).
    ack_if = ack.get("if", "")
    assert not any(fn in ack_if for fn in ("always()", "failure()", "cancelled()")), (
        f"{wf_path.name}: the acknowledgement must stay gated on dispatch success "
        "(its if: must keep GitHub's implicit success())"
    )
    assert "continue-on-error" not in dispatch, (
        f"{wf_path.name}: the dispatch step must not continue-on-error, or a failed "
        "dispatch would still post the 👀 reaction"
    )


@pytest.mark.parametrize(
    "wf_path",
    [TEMPLATES_DIR / "daydream-review.yml", REPO_WORKFLOWS_DIR / "daydream-review.yml"],
    ids=["template", "live"],
)
def test_review_workflow_head_bound_gate(wf_path: Path) -> None:
    """The review workflow is comment-only and enforces the approved head;
    backend-specific auth handling differs between the packaged Anthropic
    template and the repo's live Codex workflow.
    """
    wf = load_workflow(wf_path)
    text = wf_path.read_text(encoding="utf-8")

    assert "pull_request" not in _wf_triggers(wf)
    inputs = _wf_triggers(wf)["workflow_dispatch"]["inputs"]
    assert "approved_head_sha" in inputs
    assert "approved_at" in inputs

    steps = job_steps(wf, "analyze")
    verify = next(
        step
        for step in steps
        if "approved_head_sha" in step.get("run", "") and "exit 1" in step.get("run", "")
    )
    # The drift gate anchors on the approved commit's push time scoped to the
    # PR's head ref (the head repository's activity feed), not the repo-wide
    # head.repo.pushed_at, and rejects a push at-or-after the comment: the
    # second-granularity timestamps cannot distinguish a same-second push, so
    # equality must fail closed (\< strict-before guard) rather than pass.
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
        # Live Codex workflow: persisted codex auth is cleaned up after the review.
        cleanup = next(step for step in steps if "auth.json" in step.get("run", ""))
        assert steps.index(review) < steps.index(cleanup)
        assert cleanup.get("if", "") == "always()"
        assert set(_SECRET_REF_RE.findall(text)) == {"OPENAI_API_KEY"}
    else:
        # Packaged template: nothing persists auth, and the model credential is
        # the only secret.
        assert not any("auth.json" in step.get("run", "") for step in steps)
        assert set(_SECRET_REF_RE.findall(text)) == {"ANTHROPIC_API_KEY"}


@pytest.mark.parametrize(
    "wf_path",
    [TEMPLATES_DIR / "daydream-review.yml", REPO_WORKFLOWS_DIR / "daydream-review.yml"],
    ids=["template", "live"],
)
def test_review_workflow_persists_failure_context(wf_path: Path) -> None:
    """A failed analyze run must leave the PR number where the post workflow can
    resolve it: on any failure the analyze job records ``failure.json`` and
    uploads it as ``daydream-findings-failure``, so surface-analyze-failure can
    comment on the PR even though the workflow_run event cannot identify a
    workflow_dispatch run's PR (issue #336).
    """
    wf = load_workflow(wf_path)
    steps = job_steps(wf, "analyze")

    # The drift gate funnels every rejection through one helper that records
    # the failure context (pr_number + the instructive message) before exiting
    # — head changed, unresolvable push time, push at-or-after the approving
    # comment — so surface-analyze-failure comments the instructive message on
    # the PR (issue #336). A rejection that exits without the write would
    # degrade the comment to the generic fallback body. The single write/exit
    # point lives in the helper, so a new rejection branch only has to call it
    # (rather than re-triplicating the mkdir/printf/exit skeleton).
    verify = next(
        step
        for step in steps
        if "approved_head_sha" in step.get("run", "") and "exit 1" in step.get("run", "")
    )
    lines = verify["run"].splitlines()
    write_idx = [i for i, ln in enumerate(lines) if "> findings/failure.json" in ln]
    exit_idx = [i for i, ln in enumerate(lines) if ln.strip() == "exit 1"]
    reject_calls = [i for i, ln in enumerate(lines) if re.match(r"reject \"", ln.strip())]
    assert len(write_idx) == 1 and len(exit_idx) == 1, (
        f"{wf_path.name}: every drift rejection must funnel through the single "
        "failure-context helper (one write, one exit) (issue #336)"
    )
    assert write_idx[0] < exit_idx[0], (
        f"{wf_path.name}: the helper must record the failure context before exiting"
    )
    assert '"message"' in lines[write_idx[0]], (
        f"{wf_path.name}: the recorded failure context must carry the "
        "instructive message (issue #336)"
    )
    assert len(reject_calls) >= 3, (
        f"{wf_path.name}: the drift gate must reject head changes, unresolvable "
        "push times, and pushes at/after the approving comment"
    )

    # Any other failure still records the PR number via the job-level handler.
    record = next(step for step in steps if step.get("name") == "Record failure context")
    assert record.get("if", "") == "failure()"
    assert "failure.json" in record["run"]

    upload = next(step for step in steps if step.get("name") == "Upload failure context")
    assert upload.get("if", "") == "failure()"
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "daydream-findings-failure"
    assert upload["with"]["path"] == "findings/failure.json"


@pytest.mark.parametrize(
    "wf_path",
    [TEMPLATES_DIR / "daydream-post.yml", REPO_WORKFLOWS_DIR / "daydream-post.yml"],
    ids=["template", "live"],
)
def test_surface_analyze_failure_resolves_dispatch_run_pr(wf_path: Path) -> None:
    """surface-analyze-failure must resolve the PR of a failed workflow_dispatch
    run from the failure-context artifact (the event cannot identify it) and
    post the recorded message on the PR; an unresolvable run keeps the
    warn-and-continue guard (issue #336).
    """
    wf = load_workflow(wf_path)
    job = wf["jobs"]["surface-analyze-failure"]
    steps = job["steps"]

    # GITHUB_TOKEN needs actions: read for the cross-run artifact download
    # (the App token carries the comment write).
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
    """The single-file setup is comment-only and enforces the approved head."""
    path = TEMPLATES_DIR / "single" / "daydream.yml"
    wf = load_workflow(path)

    assert "pull_request" not in _wf_triggers(wf)

    gate = wf["jobs"]["gate"]
    assert "approved_head_sha" in gate["outputs"]
    assert "approved_at" in gate["outputs"]
    decide = next(step for step in gate["steps"] if step.get("name") == "Decide and resolve PR")
    assert "approved_head_sha" in decide.get("outputs", {}) or "head.sha" in decide.get("run", "")
    assert "approved_at" in decide.get("outputs", {}) or "approved_at=" in decide.get("run", "")

    # Acknowledge only AFTER head resolution succeeds (mirroring the split
    # workflow's ack-after-dispatch ordering): a fallible gh api exit in the
    # decide step must abort the job before the ack, so a transient
    # head-resolution failure surfaces as a failed run rather than a 👀 with no
    # review or comment (issue #336).
    ack = next(step for step in gate["steps"] if step.get("name") == "Acknowledge with eyes reaction")
    assert gate["steps"].index(ack) > gate["steps"].index(decide)

    steps = wf["jobs"]["analyze"]["steps"]
    verify = next(
        step
        for step in steps
        if "approved_head_sha" in step.get("run", "") and "exit 1" in step.get("run", "")
    )
    assert ".head.ref" in verify["run"]
    assert "activity" in verify["run"]
    assert r'\< "$APPROVED_AT"' in verify["run"]
    assert "APPROVED_AT" in wf["jobs"]["analyze"]["env"]

    # The drift gate funnels every rejection through one helper that records
    # the failure context (pr_number + the instructive message) before exiting
    # — head changed, unresolvable push time, push at-or-after the approving
    # comment — so surface-failure comments the instructive message instead of
    # the generic fallback body (issue #336). The single write/exit point lives
    # in the helper, so a new rejection branch only has to call it.
    lines = verify["run"].splitlines()
    write_idx = [i for i, ln in enumerate(lines) if "> findings/failure.json" in ln]
    exit_idx = [i for i, ln in enumerate(lines) if ln.strip() == "exit 1"]
    reject_calls = [i for i, ln in enumerate(lines) if re.match(r"reject \"", ln.strip())]
    assert len(write_idx) == 1 and len(exit_idx) == 1, (
        "single/daydream.yml: every drift rejection must funnel through the "
        "single failure-context helper (one write, one exit) (issue #336)"
    )
    assert write_idx[0] < exit_idx[0]
    assert '"message"' in lines[write_idx[0]]
    assert len(reject_calls) >= 3

    checkout_idx = next(i for i, step in enumerate(steps) if "actions/checkout" in step.get("uses", ""))
    assert steps.index(verify) < checkout_idx
    review = next(
        step
        for step in steps
        if "daydream" in step.get("run", "") and "--approved-head-sha" in step.get("run", "")
    )
    assert "APPROVED_HEAD_SHA" in review["env"]


@pytest.mark.parametrize("wf_path", sorted(TEMPLATES_DIR.rglob("*.yml")), ids=lambda p: p.name)
def test_no_event_data_interpolated_into_run_steps(wf_path: Path) -> None:
    wf = load_workflow(wf_path)
    for job_name, job in wf["jobs"].items():
        for step in job["steps"]:
            if "run" in step:
                assert not _EVENT_INTERP.search(step["run"]), (
                    f"{wf_path.name}:{job_name}: event data must reach run: via env:, never ${{{{ }}}} interpolation"
                )


@pytest.mark.parametrize(
    "wf_path",
    _BOT_WORKFLOW_PATHS,
    ids=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)
def test_bot_workflow_action_references_are_pinned_to_commit_shas(wf_path: Path) -> None:
    wf = load_workflow(wf_path)
    rel = wf_path.relative_to(_REPO_ROOT).as_posix()
    for ref in _action_references(wf):
        if ref.startswith("./"):
            continue
        assert _PINNED_ACTION_RE.fullmatch(ref), (
            f"{rel}: non-local action reference {ref!r} is not a full commit SHA "
            f"(expected owner/repo@<40 hex chars>)"
        )
    # yaml.safe_load strips inline comments, so the declared `# vX.Y.Z` version
    # comment is only visible in the raw text. Every non-local uses: line must
    # carry the approved release comment for its pinned SHA.
    for line in wf_path.read_text(encoding="utf-8").splitlines():
        m = _USES_LINE_RE.match(line)
        if m is None:
            continue
        ref, comment = m.group("ref"), m.group("comment")
        if ref.startswith("./"):
            continue
        expected = _PINNED_ACTION_VERSIONS.get(ref)
        assert expected is not None, (
            f"{rel}: non-local action reference {ref!r} is not in the approved "
            f"pinned-action map; add it (with its release version) or its version "
            f"comment cannot be verified"
        )
        assert comment == expected, (
            f"{rel}: action reference {ref!r} carries version comment {comment!r}, "
            f"but must carry the approved {expected!r} inline comment"
        )


# Install-pin drift guard: the bot must install a pinned daydream release, never
# the moving `main` tip, across every live and shipped workflow. Fails on
# release until the pin is bumped in lockstep with the package version.

_INSTALL_RE = re.compile(r"uv tool install\s+git\+https://github\.com/existential-birds/daydream(?P<ref>@\S+)?")


def _package_version() -> str:
    pyproject = _REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return cast(str, data["project"]["version"])


@pytest.mark.parametrize(
    "wf_path",
    _DAYDREAM_INSTALL_WORKFLOW_PATHS,
    ids=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)
def test_daydream_install_is_pinned_to_release_commit(wf_path: Path) -> None:
    text = wf_path.read_text(encoding="utf-8")
    refs = [m.group("ref") for m in _INSTALL_RE.finditer(text)]
    rel = wf_path.relative_to(_REPO_ROOT).as_posix()
    assert refs, f"{rel} must install daydream via `uv tool install git+…`"
    version = _package_version()
    key = f"v{version}"
    commit = _DAYDREAM_RELEASE_COMMITS.get(key)
    assert commit is not None, (
        f"No release→commit map entry for {key}. The release process is manual: "
        f"(1) bump project.version in pyproject.toml, (2) tag the release, "
        f"(3) get the PEELED commit: git ls-remote origin 'refs/tags/{key}^{{}}' "
        f"(annotated tags report a tag-object SHA on the bare ref — use the ^{{}} target), "
        f"(4) add the entry to _DAYDREAM_RELEASE_COMMITS in tests/test_workflow_templates.py, "
        f"(5) update all six workflow install refs in lockstep "
        f"(.github/workflows/daydream-review.yml, .github/workflows/daydream-post.yml, "
        f"daydream/templates/workflows/daydream-review.yml, "
        f"daydream/templates/workflows/daydream-post.yml, "
        f"daydream/templates/workflows/single/daydream.yml ×2)."
    )
    expected = f"@{commit}"
    for ref in refs:
        assert ref == expected, (
            f"{rel} pins the daydream install to {ref or '(unpinned main)'}, but must pin to "
            f"the immutable release commit {expected} for {key}. Update all six install refs "
            f"in lockstep with the _DAYDREAM_RELEASE_COMMITS entry."
        )


def test_release_commit_map_values_are_immutable_full_shas() -> None:
    for tag, commit in _DAYDREAM_RELEASE_COMMITS.items():
        assert _RELEASE_COMMIT_RE.fullmatch(commit), (
            f"_DAYDREAM_RELEASE_COMMITS[{tag!r}] = {commit!r} is not a full lowercase "
            f"40-char hex commit SHA. Mutable refs (tags, branches), short hashes, and "
            f"uppercase hex are rejected; the form gate can't tell an annotated-tag "
            f"OBJECT sha from a peeled commit, so record the peeled commit: "
            f"git ls-remote origin 'refs/tags/{tag}^{{}}'."
        )


def _repo_release_tags() -> list[str]:
    """Release tag names present in this checkout's local refs. A shallow,
    tagless checkout (e.g. CI's plain ``actions/checkout``) yields an empty
    list, which the peel cross-check treats as ``cannot verify offline``,
    never as ``no releases exist``.
    """
    result = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return result.stdout.splitlines()


def _peeled_commit_for_tag(tag: str) -> str | None:
    """Resolve ``refs/tags/<tag>`` to its peeled (``^{}``) commit via the repo's
    own local refs, or None when the tag does not exist in this checkout. On an
    annotated tag the bare ref resolves to the tag OBJECT; the ``^{}`` target
    is the commit the release actually points at.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{}}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def test_release_commit_map_values_are_peeled_release_commits() -> None:
    """Every map value must be the PEELED commit of its release tag, verified
    offline against this repo's own refs: the only check that can tell an
    annotated-tag OBJECT sha (what the bare ref reports) from its peeled
    commit, and the only check that distinguishes a real released version from
    a phantom key masquerading as retained history. A checkout that carries no
    release tags at all (CI's shallow clone) skips: there is no local data to
    verify against, and the check is intentionally offline (never GitHub).
    """
    tags = _repo_release_tags()
    if not tags:
        pytest.skip(
            "this checkout carries no release tags (e.g. a shallow CI clone), "
            "so the offline peel cross-check cannot run"
        )
    tag_set = set(tags)
    for tag, commit in _DAYDREAM_RELEASE_COMMITS.items():
        assert tag in tag_set, (
            f"_DAYDREAM_RELEASE_COMMITS key {tag!r} is not a real release tag "
            f"in this repo (no refs/tags/{tag}), so it cannot be retained "
            f"provenance: remove the entry or name a version that was actually released."
        )
        peeled = _peeled_commit_for_tag(tag)
        assert peeled is not None  # tag_set membership already proved the ref exists
        assert peeled == commit, (
            f"_DAYDREAM_RELEASE_COMMITS[{tag!r}] = {commit!r} is not the peeled "
            f"commit of refs/tags/{tag} (got {peeled!r}): on an annotated tag the "
            f"bare ref reports the tag OBJECT sha, not the commit — record the "
            f"peeled commit: git ls-remote origin 'refs/tags/{tag}^{{}}'."
        )


def _release_version(tag: str) -> tuple[int, int, int] | None:
    """Parse a vX.Y.Z release tag into a sortable version tuple."""
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def test_release_commit_map_entries_are_pinned_or_retained_provenance() -> None:
    """Cross-check _DAYDREAM_RELEASE_COMMITS against the workflow install refs
    so an entry cannot ship as dead data: every entry's commit must be referenced
    by at least one install pin, or the entry must be a strictly older release
    deliberately retained for provenance (the map never prunes history). A
    malformed tag key, a pin that lost its map entry, or a stale commit that outlived
    its installs all fail here — the provenance entries are accounted for, not
    unnoticed.
    """
    current_version = _release_version(f"v{_package_version()}")
    assert current_version is not None
    pinned_commits = {
        m.group("ref")[1:]
        for wf_path in _DAYDREAM_INSTALL_WORKFLOW_PATHS
        for m in _INSTALL_RE.finditer(wf_path.read_text(encoding="utf-8"))
        if m.group("ref") is not None
    }
    assert pinned_commits, "no daydream install pins to cross-check the map against"
    for tag, commit in _DAYDREAM_RELEASE_COMMITS.items():
        if commit in pinned_commits:
            continue
        version = _release_version(tag)
        assert version is not None and version < current_version, (
            f"_DAYDREAM_RELEASE_COMMITS[{tag!r}] = {commit!r} is neither pinned by "
            f"any workflow install nor an older release retained for provenance, so "
            f"it ships as dead data: reference it with an install pin or remove it."
        )


# Privilege split — the security invariant the whole design exists to enforce:
# no job ever holds both untrusted PR code and the App key.


@pytest.mark.parametrize(
    "post_path",
    [TEMPLATES_DIR / "daydream-post.yml", REPO_WORKFLOWS_DIR / "daydream-post.yml"],
    ids=["template", "live"],
)
def test_split_setup_preserves_privilege_split(post_path: Path) -> None:
    review = load_workflow(TEMPLATES_DIR / "daydream-review.yml")
    review_text = (TEMPLATES_DIR / "daydream-review.yml").read_text(encoding="utf-8")
    command_text = (TEMPLATES_DIR / "daydream-command.yml").read_text(encoding="utf-8")
    post = load_workflow(post_path)
    post_text = post_path.read_text(encoding="utf-8")

    # Phase A runs untrusted PR code: read-only, and its only secret is the API key.
    assert review["permissions"] == {"contents": "read"}
    assert set(_SECRET_REF_RE.findall(review_text)) == {"ANTHROPIC_API_KEY"}

    # The App-key holders never check out code: the command workflow never checks
    # out at all, and every job in each privileged post workflow performs no
    # checkout.
    assert "actions/checkout" not in command_text
    for job in post["jobs"].values():
        assert not has_checkout(job)
    assert set(_SECRET_REF_RE.findall(post_text)) == {"DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"}


_APP_TOKEN_ACTION = "actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1"

# Every token-minting workflow, shipped or live, pins every App-token action to
# the approved v3.2.0 commit. Lists the concrete (job, action) pairs so a renamed
# job, a refloated pin, or a newly added unpinned token action fails loudly rather
# than being silently absorbed by a wildcard.
_APP_TOKEN_PIN_CASES = [
    (
        TEMPLATES_DIR / "daydream-post.yml",
        "template-post",
        [("post", _APP_TOKEN_ACTION), ("surface-analyze-failure", _APP_TOKEN_ACTION)],
    ),
    (
        REPO_WORKFLOWS_DIR / "daydream-post.yml",
        "live-post",
        [("post", _APP_TOKEN_ACTION), ("surface-analyze-failure", _APP_TOKEN_ACTION)],
    ),
    (TEMPLATES_DIR / "daydream-command.yml", "template-command", [("dispatch", _APP_TOKEN_ACTION)]),
    (REPO_WORKFLOWS_DIR / "daydream-command.yml", "live-command", [("dispatch", _APP_TOKEN_ACTION)]),
    (
        TEMPLATES_DIR / "single" / "daydream.yml",
        "single",
        [("gate", _APP_TOKEN_ACTION), ("post", _APP_TOKEN_ACTION), ("surface-failure", _APP_TOKEN_ACTION)],
    ),
]


@pytest.mark.parametrize(
    "wf_path,expected", [(p, e) for p, _id, e in _APP_TOKEN_PIN_CASES], ids=[_id for _, _id, _ in _APP_TOKEN_PIN_CASES]
)
def test_workflows_pin_create_github_app_token(wf_path: Path, expected: list[Any]) -> None:
    wf = load_workflow(wf_path)
    token_action_uses = [
        (job_name, str(step.get("uses", "")))
        for job_name, job in wf["jobs"].items()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/create-github-app-token@")
    ]
    assert sorted(token_action_uses) == sorted(expected)


@pytest.mark.parametrize(
    "wf_path",
    [TEMPLATES_DIR / "daydream-post.yml", REPO_WORKFLOWS_DIR / "daydream-post.yml"],
    ids=["template", "live"],
)
def test_post_findings_step_exports_bot_login(wf_path: Path) -> None:
    """The Post findings step must export BOT_LOGIN from the deposited
    ``DAYDREAM_BOT_HANDLE`` variable and pass it to ``daydream post-findings``
    via ``--bot-login`` so the author filter has a bot login to match against
    (issue #254). Without it, the post job degrades to viewerDidAuthor-only
    GraphQL dedup and suppresses nothing on the REST side.

    Pinned on BOTH the shipped template and the repo's own live workflow —
    the two files retain distinct failure-surfacing conditions, but this
    bot-login invariant must hold for both.
    """
    text = wf_path.read_text(encoding="utf-8")
    assert "BOT_LOGIN: ${{ vars.DAYDREAM_BOT_HANDLE }}" in text, (
        f"{wf_path.name}: Post findings step must export BOT_LOGIN from vars.DAYDREAM_BOT_HANDLE (issue #254)"
    )
    assert '--bot-login "$BOT_LOGIN"' in text, f"{wf_path.name}: Post findings step must pass --bot-login explicitly"


@pytest.mark.parametrize(
    "wf_path",
    [TEMPLATES_DIR / "daydream-post.yml", REPO_WORKFLOWS_DIR / "daydream-post.yml"],
    ids=["template", "live"],
)
def test_failure_comment_target_never_uses_findings_artifact(wf_path: Path) -> None:
    """The 'Surface failure on the PR' handler must derive its comment target
    only from DERIVED_PR_NUMBER or EVENT_PR_NUMBER — never from the
    findings/findings.json artifact, which is unvalidated in this failure
    path (issue #384). When both sources are empty it logs the diagnostic
    and exits 0 without writing a comment.

    Pinned on BOTH the shipped template and the repo's own live workflow —
    the two files retain distinct failure-surfacing conditions, but this
    target-source invariant must hold for both.
    """
    wf = load_workflow(wf_path)
    steps = job_steps(wf, "post")
    handler = next(s for s in steps if s.get("name") == "Surface failure on the PR")
    run = handler["run"]

    # Exactly one PR_NUMBER assignment, sourced only from the two allowed env vars.
    assert run.count("PR_NUMBER=") == 1, (
        f"{wf_path.name}: failure handler must assign PR_NUMBER exactly once"
    )
    assert 'PR_NUMBER="${DERIVED_PR_NUMBER:-$EVENT_PR_NUMBER}"' in run, (
        f"{wf_path.name}: failure target must come only from DERIVED_PR_NUMBER or EVENT_PR_NUMBER (issue #384)"
    )

    # No unvalidated artifact read / jq extraction in the failure handler.
    assert "findings/findings.json" not in run, (
        f"{wf_path.name}: failure handler must not read findings/findings.json (issue #384)"
    )
    assert "jq" not in run, (
        f"{wf_path.name}: failure handler must not run jq over an artifact (issue #384)"
    )

    # The empty-result guard exits 0 before the comment write, so no write
    # happens when neither source yields a number.
    guard = 'echo "no PR resolvable; cannot surface the failure" >&2'
    exit_guard = "exit 0"
    assert guard in run, f"{wf_path.name}: empty-result diagnostic must be present"
    assert exit_guard in run, f"{wf_path.name}: exit 0 must be present in the empty-result guard"
    assert run.index(guard) < run.index(exit_guard), (
        f"{wf_path.name}: empty-result diagnostic must precede exit 0"
    )
    assert run.index(exit_guard) < run.index('gh api "repos/${REPO}/issues/${PR_NUMBER}/comments"'), (
        f"{wf_path.name}: exit 0 must precede the comment write (issue #384)"
    )


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


# Repo dogfood workflow (Codex). The full rationale — the `codex exec` auth
# gap, the `codex login --with-api-key` persistence, and the README's naming of
# the live credential — lives in the module docstring and is exercised by
# test_repo_workflow_readme_documents_codex_credential; this test asserts the
# login-persistence ordering that rationale requires.


def _assert_repo_workflow_uses_openai_credential() -> None:
    """Assert the repo's live review workflow uses OPENAI_API_KEY as its only secret."""
    text = (REPO_WORKFLOWS_DIR / "daydream-review.yml").read_text(encoding="utf-8")
    assert set(_SECRET_REF_RE.findall(text)) == {"OPENAI_API_KEY"}


def test_repo_review_authenticates_codex_before_running() -> None:
    wf = load_workflow(REPO_WORKFLOWS_DIR / "daydream-review.yml")
    _assert_repo_workflow_uses_openai_credential()

    steps = job_steps(wf, "analyze")
    login = next(s for s in steps if "codex login --with-api-key" in s.get("run", ""))
    assert "OPENAI_API_KEY" in login["env"]
    review_idx = next(i for i, s in enumerate(steps) if "daydream --review" in s.get("run", ""))
    assert steps.index(login) < review_idx, "Codex auth must be persisted before the review runs"
    # The review step authenticates via auth.json, not a redundant env secret.
    assert "OPENAI_API_KEY" not in steps[review_idx].get("env", {})


def test_repo_workflow_readme_declares_codex_and_points_to_canonical_install() -> None:
    text = (REPO_WORKFLOWS_DIR / "README.md").read_text(encoding="utf-8")

    # Presence of the corrected contract. Prose checks are kept to the substance
    # (rather than verbatim phrasing) so an innocuous reword does not break the
    # test: heading mentions dogfood workflows, and the repo-only Codex dogfood
    # stance is declared.
    first_heading = next((ln for ln in text.splitlines() if ln.startswith("#")), "")
    assert "dogfood" in first_heading.lower() and "workflows" in first_heading.lower()
    assert "codex dogfood" in text.lower() and "repository-only" in text.lower()

    # The stable technical contract (not prose, so safe to pin verbatim).
    assert "daydream --review --backend codex" in text
    assert "OPENAI_API_KEY" in text

    # The canonical (packaged) install guide is linked to rather than duplicated:
    # resolve the relative link against THIS README's own directory (so the target
    # path is derived from the link itself, not reconstructed from the repo root,
    # which would let a moved README's stale link pass) and assert the target
    # exists with an ## Install anchor so the marketed link cannot rot.
    install_link = "../../daydream/templates/workflows/README.md#install"
    canonical = (REPO_WORKFLOWS_DIR / install_link.split("#")[0]).resolve()
    assert install_link in text
    assert canonical.exists()
    assert re.search(r"^## ?Install\b", canonical.read_text(encoding="utf-8"), re.M)

    # Absence of the stale strings.
    for stale in ("Copy the three workflow files", "Install step 1", "ANTHROPIC_API_KEY"):
        assert stale not in text


def test_makefile_and_ci_pin_actionlint_image_by_digest() -> None:
    # Caveat (matches _PINNED_ACTION_VERSIONS, which also cannot verify a
    # SHA-->release mapping against its upstream): these assertions only prove the
    # three copies agree with one another. They cannot, and are not intended to,
    # re-derive the manifest digest from the registry. _ACTIONLINT_IMAGE is a
    # golden pin whose correctness was verified against the live OCI registry at
    # write time (see its definition comment); if it is ever changed to a
    # different 64-hex value without such a re-verification, this suite stays
    # green and the failure surfaces only at CI runtime on `docker pull`.
    wf = load_workflow(REPO_WORKFLOWS_DIR / "ci.yml")
    steps = job_steps(wf, "check")
    actionlint = next(s for s in steps if s.get("name") == "Lint workflows with actionlint")

    ci_refs = _ACTIONLINT_REF_RE.findall(actionlint["run"])
    makefile_text = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    make_refs = _ACTIONLINT_REF_RE.findall(makefile_text)

    assert ci_refs == [_ACTIONLINT_IMAGE], (
        "CI actionlint step must reference the digest-pinned image exactly once: "
        f"found {ci_refs}"
    )
    assert make_refs == [_ACTIONLINT_IMAGE], (
        "Makefile actionlint target must carry the digest-pinned image exactly "
        f"once: found {make_refs}"
    )


# CI coverage guard: the actionlint step in .github/workflows/ci.yml must
# receive EVERY workflow the project ships — the repo's own top-level workflows
# plus all recursively discovered template workflows (the nested
# single/daydream.yml included). Reads the live selectors out of the ci.yml
# actionlint step and expands them, so a selector that stops covering a shipped
# file (or a newly nested template) fails this test rather than silently
# shipping un-linted workflows.


def test_makefile_actionlint_selectors_match_ci() -> None:
    wf = load_workflow(REPO_WORKFLOWS_DIR / "ci.yml")
    steps = job_steps(wf, "check")
    actionlint = next(s for s in steps if s.get("name") == "Lint workflows with actionlint")
    ci_selectors = [
        tok for tok in actionlint["run"].split()
        if tok.endswith(".yml") and not tok.startswith("-")
    ]
    mk = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "rhysd/actionlint" in mk
    # Every CI selector appears verbatim in the Makefile's actionlint recipe,
    # and vice versa: adding a directory on either side strands the other gate.
    for sel in ci_selectors:
        assert sel in mk, f"selector {sel!r} missing from Makefile actionlint target"
    # Parse ONLY the actionlint target's recipe (between the target line and the
    # next top-level target), not comments, which may mention ci.yml in prose.
    mk_lines = mk.splitlines()
    start = next(i for i, line in enumerate(mk_lines) if line.strip() == "actionlint:")
    recipe: list[str] = []
    for line in mk_lines[start + 1:]:
        if line and not line.startswith(("\t", " ")):
            break
        recipe.append(line)
    mk_toks = [t.strip("\\\t ") for t in " ".join(recipe).split()]
    mk_selectors = {t for t in mk_toks if t.endswith(".yml") and t != "\\"}
    for sel in mk_selectors:
        assert sel in ci_selectors, f"Makefile-only selector {sel!r} escapes CI"


def test_ci_actionlint_covers_all_workflow_sources() -> None:
    wf = load_workflow(REPO_WORKFLOWS_DIR / "ci.yml")
    steps = job_steps(wf, "check")
    actionlint = next(s for s in steps if s.get("name") == "Lint workflows with actionlint")
    selectors = [
        tok for tok in actionlint["run"].split() if tok.endswith(".yml") and not tok.startswith("-")
    ]

    actual: set[Path] = set()
    for selector in selectors:
        matches = set(_REPO_ROOT.glob(selector))
        assert matches, (
            f"actionlint selector {selector!r} in .github/workflows/ci.yml matches "
            "no workflow files; drop the stale selector or fix its glob"
        )
        actual |= matches
    expected = {*REPO_WORKFLOWS_DIR.glob("*.yml"), *TEMPLATES_DIR.rglob("*.yml")}
    actual_rel = sorted(p.relative_to(_REPO_ROOT).as_posix() for p in actual)
    expected_rel = sorted(p.relative_to(_REPO_ROOT).as_posix() for p in expected)
    assert actual == expected, (
        "actionlint selectors in .github/workflows/ci.yml cover a different set "
        "of workflows than the project ships. "
        f"actual={actual_rel} expected={expected_rel}. "
        "Extend the actionlint run's selectors so the glob-expanded set equals "
        "the repo workflows plus all shipped template workflows (nested included)."
    )
