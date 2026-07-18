"""Real-path tests that the posterior-capture loop closes.

A finding is only labelable if it reaches ``/pulls/{n}/comments`` as a
top-level, repliable thread carrying :data:`DAYDREAM_FOOTER` and
``finding_marker(fingerprint)``. The review *body* carries those markers too,
but that surface is invisible to :func:`index_pr_review_comments` — so a
finding folded into the body is permanently unlabelable.

These tests cover the two halves of that loop:

* placement — a finding with no diff-line home whose file IS in the PR diff
  must be classified ``"file"``, not ``"body"`` (driven through
  ``runner.run`` with a real git worktree);
* capture — driving ``daydream post-findings`` from ``cli.main`` with a real
  ``gh`` subprocess seam must produce a comment on ``/pulls/{n}/comments``
  that the labeler's own signals read back, count, and resolve on reply.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from daydream import cli, git_ops
from daydream.backends import ResultEvent, TextEvent
from daydream.findings import FINDINGS_SCHEMA_VERSION, write_findings_artifact
from daydream.pr_review import parse_finding_markers
from daydream.runner import RunConfig, run
from daydream.training.labeler_signals import (
    comment_resolution_signal,
    index_pr_review_comments,
    per_finding_resolution_signal,
)
from tests.harness.phase_backend import PhaseDispatchBackend

FILE_FINGERPRINT = "f" * 64


def _never_fetch(*_args, **_kwargs):
    """A gh_api that must never be called: `threads=` makes the fetch unnecessary."""
    raise AssertionError("gh_api must not be called when threads= is supplied")


def cli_main(argv: list[str]) -> int:
    """Drive ``cli.main`` with ``argv`` and return its exit code."""
    import sys

    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    raise AssertionError("cli.main() must exit via sys.exit")


# --- Placement: an unplaceable finding on a changed file goes file-level -----


@contextmanager
def _review_run_env(repo: Path, monkeypatch, out: Path, backend, fake_gh):
    """`--review --findings-out` real-path setup, mocking ONLY the backend seam.

    PR discovery runs for real through ``git_ops``' ``gh`` subprocess seam
    against the fake ``gh`` binary, so a regression in PR lookup surfaces here
    instead of being patched out.
    """
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    fake_gh.serve_pr_view(
        {
            "number": 7,
            "state": "OPEN",
            "headRefName": "feature",
            "baseRefName": "main",
            "headRefOid": git_ops.head_sha(repo),
            "baseRefOid": git_ops.merge_base(repo, "main"),
            "url": "https://github.com/acme/widgets/pull/7",
            "body": "",
        }
    )
    config = RunConfig(
        target=str(repo),
        output_mode="review",
        pr_number=7,
        findings_out=str(out),
        non_interactive=True,
    )
    with patch("daydream.runner.create_backend", return_value=backend):
        yield config


async def test_unanchorable_finding_on_changed_file_is_placed_file_level(
    feature_branch_repo, monkeypatch, tmp_path, fake_gh
):
    """A finding whose anchors match nothing still gets a trackable placement.

    Enters from ``runner.run`` against a real git worktree. The scripted issue
    targets ``main.py`` — a file genuinely in the PR diff — but its text
    contains no token present in that file, so anchor resolution yields no
    line. Before the fix this fell through to ``placement="body"``, which no
    posterior signal can ever read back. It must now be ``"file"``.
    """
    out = tmp_path / "findings.json"
    issue = {
        "id": 1,
        "title": "Module lacks a rollback barrier",
        "description": "No `quiescent_rollback_barrier` guards this module",
        "recommendation": "Introduce a `quiescent_rollback_barrier`",
        "severity": "medium",
        "confidence": "HIGH",
        "files": ["main.py"],
        "rationale": "",
    }
    backend = PhaseDispatchBackend(
        events=[
            TextEvent(text="Review complete."),
            ResultEvent(structured_output={"issues": [issue]}, continuation=None),
        ]
    )
    with _review_run_env(feature_branch_repo, monkeypatch, out, backend, fake_gh) as config:
        assert await run(config) == 0

    findings = json.loads(out.read_text())["findings"]
    assert findings, "scripted issue must survive to the artifact"
    main_py = [f for f in findings if f["path"] == "main.py"]
    assert main_py, "the finding must target the changed file"
    assert [f["placement"] for f in main_py] == ["file"], (
        "a finding on a file in the PR diff with no resolvable line must be "
        f"placed file-level, not folded into the invisible review body: {main_py}"
    )


# --- Capture: the posted comment is read back by the labeler's own signals ---


def _artifact(path: Path, findings: list[dict]) -> Path:
    write_findings_artifact(
        path,
        {
            "schema_version": FINDINGS_SCHEMA_VERSION,
            "repo": "o/r",
            "pr_number": 7,
            "head_sha": "h" * 40,
            "run_info": "test run info",
            "findings": findings,
        },
    )
    return path


@pytest.fixture
def file_level_artifact(tmp_path: Path) -> Path:
    """One finding with ``placement="file"`` on a path inside the PR diff."""
    return _artifact(
        tmp_path / "findings.json",
        [
            {
                "fingerprint": FILE_FINGERPRINT,
                "path": "b.py",
                "line": None,
                "placement": "file",
                "title": "Cross-cutting concern in b.py",
                "body": "Body text",
                "severity": "high",
                "confidence": "HIGH",
                "is_cross_stack": True,
            }
        ],
    )


def _posted_comments(fake_gh) -> list[dict]:
    """Rebuild the ``/comments`` GET payload from what actually crossed the gh boundary."""
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/comments")
    return [
        {
            "id": 9000 + i,
            "in_reply_to_id": None,
            "user": {"login": "daydream-review"},
            "body": call.payload["body"],
        }
        for i, call in enumerate(posts, start=1)
    ]


def test_file_level_finding_is_captured_and_resolvable(fake_gh, file_level_artifact) -> None:
    """The full loop: post → read back → count → resolve on reply.

    Drives ``daydream post-findings`` from ``cli.main`` with the real
    ``git_ops`` subprocess seam (only the ``gh`` binary is faked), then feeds
    the comments that actually crossed that boundary into the labeler's own
    signals. Asserts the rubric-visible outcome, not that anything was called.
    """
    fake_gh.set_response("diff-paths", value=["b.py"])
    assert cli_main(
        ["post-findings", str(file_level_artifact), "--pr", "7", "--head-sha", "h" * 40, "--repo", "o/r"]
    ) == 0

    # 1. The finding reached /pulls/{n}/comments carrying footer + marker.
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/comments")
    assert len(posts) == 1, "the file-level finding must be posted as its own comment"
    body = posts[0].payload["body"]
    assert posts[0].payload["subject_type"] == "file"
    assert "<sub>🧙 Posted by [daydream v" in body, "footer identifies daydream authorship"
    assert parse_finding_markers(body) == [FILE_FINGERPRINT]

    # 2. The labeler reads it back off that endpoint.
    comments = _posted_comments(fake_gh)
    row = {"pr_repo": "o/r", "pr_number": 7}
    threads = index_pr_review_comments(row, gh_api=lambda *a, **k: comments)
    assert threads is not None
    unreplied = comment_resolution_signal(row, gh_api=_never_fetch, threads=threads)
    assert unreplied.total == 1, "the finding must be a trackable top-level comment"
    assert unreplied.unresolved == 1

    # 3. A maintainer reply moves it to resolved — per-finding, by fingerprint.
    reply = {
        "id": 9999,
        "in_reply_to_id": comments[0]["id"],
        "user": {"login": "kevin"},
        "body": "fixed",
    }
    replied = [*comments, reply]
    threads = index_pr_review_comments(row, gh_api=lambda *a, **k: replied)
    assert threads is not None
    assert comment_resolution_signal(row, gh_api=_never_fetch, threads=threads).unresolved == 0
    per_finding = per_finding_resolution_signal(
        row, recorded_fingerprints=[FILE_FINGERPRINT], gh_api=_never_fetch, threads=threads
    )
    assert [(r.fingerprint, r.resolved) for r in per_finding] == [(FILE_FINGERPRINT, True)]


def test_file_level_post_rejected_falls_back_to_review_body(fake_gh, file_level_artifact) -> None:
    """A finding GitHub refuses a file-level comment for is never silently dropped.

    ``diff-paths`` excludes ``b.py``, so the shim 422s the file-level POST the
    way real GitHub does for a path outside the PR diff. The finding must then
    appear in the review body — worse for capture, but delivered.
    """
    fake_gh.set_response("diff-paths", value=["other.py"])
    assert cli_main(
        ["post-findings", str(file_level_artifact), "--pr", "7", "--head-sha", "h" * 40, "--repo", "o/r"]
    ) == 0

    reviews = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(reviews) == 1
    assert parse_finding_markers(reviews[0].payload["body"]) == [FILE_FINGERPRINT], (
        "a rejected file-level finding must fall back into the review body"
    )


def test_review_failure_still_reports_live_file_level_comments(
    fake_gh, file_level_artifact, capsys
) -> None:
    """A failed review POST must not claim nothing was posted.

    File-level comments post *before* the consolidated review, so when the
    review POST fails they are already live on the PR. Reporting "no comments
    were posted" would send the operator looking for a clean slate that does
    not exist.
    """
    fake_gh.set_response("diff-paths", value=["b.py"])
    fake_gh.set_response("POST", "/repos/o/r/pulls/7/reviews", value=None)

    rc = cli_main(
        ["post-findings", str(file_level_artifact), "--pr", "7", "--head-sha", "h" * 40, "--repo", "o/r"]
    )

    assert rc == 1, "a failed review POST is still a failure"
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/comments")) == 1
    out = capsys.readouterr().out
    assert "1 file-level comment(s) were already posted" in out, out
    assert "No comments were posted" not in out, out
