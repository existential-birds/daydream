"""Place findings in repliable threads and capture their posterior labels.

Only top-level /pulls/{n}/comments carrying the Daydream footer and finding
marker are visible to the labeler; review-body findings are not. Real runner
placement and post-findings CLI tests exercise posting, readback, counting,
and reply-based resolution through the fake gh process.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream import git_ops
from daydream.findings import write_findings_artifact
from daydream.reviews.identity import parse_finding_markers
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.training.labeler_signals import (
    comment_resolution_signal,
    index_pr_review_comments,
    per_finding_resolution_signal,
)
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.review_result import findings_artifact
from tests.harness.scripts import cli_main

FILE_FINGERPRINT = "f" * 64


def _never_fetch(*_args: Any, **_kwargs: Any) -> None:
    """A gh_api that must never be called: `threads=` makes the fetch unnecessary."""
    raise AssertionError("gh_api must not be called when threads= is supplied")


# --- Placement: an unplaceable finding on a changed file goes file-level -----


@contextmanager
def _review_run_env(repo: Path, monkeypatch: pytest.MonkeyPatch, out: Path, backend: Any, fake_gh: FakeGh,
) -> Iterator[Any]:
    """Set up review/findings output with a fake backend and real Git/gh discovery."""
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    fake_gh.serve_pr_view({"number": 7, "state": "OPEN", "headRefName": "feature", "baseRefName": "main",
            "headRefOid": git_ops.head_sha(repo),
            "headRepository": {"name": "widgets", "nameWithOwner": "acme/widgets"},
            "headRepositoryOwner": {"login": "acme"}, "url": "https://github.com/acme/widgets/pull/7", "body": "",
        }
    )
    config = RunConfig(target=str(repo), output_mode="review", pr_number=7, findings_out=str(out), non_interactive=True,
    )
    with patch("daydream.runner.create_backend", return_value=backend):
        yield config


def _scripted_review_backend(repo: Path, issue: dict[str, Any]) -> EmptyReviewBackend:
    """Report one schema-valid per-stack issue through the real merge path."""
    record = {key: issue[key] for key in (
        "id", "description", "file", "line", "severity", "confidence", "rationale", "evidence")}
    backend = EmptyReviewBackend(repo, forbid_merge=False, forbid_supervise=False,
                                 review_by_stack={"python": [record]})
    backend.merge_echo_records = True
    backend.merge_items = None
    return backend


def _issue(*, line: int) -> dict[str, Any]:
    """A scripted review issue citing ``main.py`` at ``line``."""
    return {"id": 1, "title": "Module lacks a rollback barrier",
        "description": "No `quiescent_rollback_barrier` guards this module",
        "recommendation": "Introduce a `quiescent_rollback_barrier`", "severity": "medium", "confidence": "HIGH",
        "files": ["main.py"], "file": "main.py", "line": line, "rationale": "", "evidence": f"main.py:{line}",
    }

async def test_unanchorable_finding_on_changed_file_is_placed_file_level(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gh: FakeGh,
) -> None:
    """A changed-file finding with no usable anchor remains a trackable file comment.

    Cite beyond EOF and every hunk: an in-hunk citation would be authoritative
    and exercise inline placement instead of this fallback.
    """
    out = tmp_path / "findings.json"
    issue = _issue(line=9)
    backend = _scripted_review_backend(feature_branch_repo, issue)
    with _review_run_env(feature_branch_repo, monkeypatch, out, backend, fake_gh) as config:
        assert await run(config) == 0

    findings = json.loads(out.read_text())["findings"]
    assert findings, "scripted issue must survive to the artifact"
    main_py = [f for f in findings if f["path"] == "main.py"]
    assert main_py, "the finding must target the changed file"
    assert all(f["placement"] == "file" for f in main_py), (
        "a finding on a file in the PR diff with no resolvable line must be "
        f"placed file-level, not folded into the invisible review body: {main_py}"
    )

async def test_in_hunk_citation_is_placed_inline_without_an_anchor_match(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gh: FakeGh,
) -> None:
    """A valid in-hunk citation stays inline despite having no matching text anchor."""
    out = tmp_path / "findings.json"
    issue = _issue(line=1)
    backend = _scripted_review_backend(feature_branch_repo, issue)
    with _review_run_env(feature_branch_repo, monkeypatch, out, backend, fake_gh) as config:
        assert await run(config) == 0

    findings = json.loads(out.read_text())["findings"]
    main_py = [f for f in findings if f["path"] == "main.py"]
    assert main_py, "the finding must target the changed file"
    assert all(f["placement"] == "inline" and f["line"] == 1 for f in main_py), (
        "an in-hunk citation must be posted inline on the line the reviewer "
        f"cited, not relocated or demoted: {main_py}"
    )
    assert all("**Placement:**" not in f["body"] for f in main_py), (
        "nothing moved, so no relocation note belongs on the finding"
    )


# --- Capture: the posted comment is read back by the labeler's own signals ---


def _artifact(path: Path, findings: list[dict[str, Any]]) -> Path:
    write_findings_artifact(path, findings_artifact(findings, run_info="test run info"))
    return path


@pytest.fixture
def file_level_artifact(tmp_path: Path) -> Path:
    """One finding with ``placement="file"`` on a path inside the PR diff."""
    return _artifact(tmp_path / "findings.json",
        [{"fingerprint": FILE_FINGERPRINT, "path": "b.py", "line": None, "placement": "file",
                "title": "Cross-cutting concern in b.py", "body": "Body text", "severity": "high", "confidence": "HIGH",
                "is_cross_stack": True,
            }
        ],
    )


def _posted_comments(fake_gh: FakeGh) -> list[dict[str, Any]]:
    """Rebuild the ``/comments`` GET payload from what actually crossed the gh boundary."""
    posts = fake_gh.calls("POST", "/repos/o/r/pulls/7/comments")
    return [{"id": 9000 + i, "in_reply_to_id": None, "user": {"login": "daydream-review"}, "body": call.payload["body"]}
        for i, call in enumerate(posts, start=1)
    ]

def test_file_level_finding_is_captured_and_resolvable(fake_gh: FakeGh, file_level_artifact: Path) -> None:
    """Post through the CLI/gh seam, then read, count, and resolve the actual comment."""
    fake_gh.set_response("diff-paths", value=["b.py"])
    assert cli_main(["post-findings", str(file_level_artifact), "--pr", "7", "--head-sha", "h" * 40, "--repo", "o/r"]
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
    reply = {"id": 9999, "in_reply_to_id": comments[0]["id"], "user": {"login": "kevin", "type": "User"},
        "author_association": "MEMBER", "body": "Fixed in abc123",
    }
    replied = [*comments, reply]
    threads = index_pr_review_comments(row, gh_api=lambda *a, **k: replied)
    assert threads is not None
    assert comment_resolution_signal(row, gh_api=_never_fetch, threads=threads).unresolved == 0
    per_finding = per_finding_resolution_signal(
        row, recorded_fingerprints=[FILE_FINGERPRINT], gh_api=_never_fetch, threads=threads
    )
    assert [(r.fingerprint, r.disposition) for r in per_finding] == [(FILE_FINGERPRINT, "accepted")]

def test_file_level_post_rejected_falls_back_to_review_body(fake_gh: FakeGh, file_level_artifact: Path) -> None:
    """A 422 for a path outside the PR diff retains the finding in the review body."""
    fake_gh.set_response("diff-paths", value=["other.py"])
    assert cli_main(["post-findings", str(file_level_artifact), "--pr", "7", "--head-sha", "h" * 40, "--repo", "o/r"]
    ) == 0

    reviews = fake_gh.calls("POST", "/repos/o/r/pulls/7/reviews")
    assert len(reviews) == 1
    assert parse_finding_markers(reviews[0].payload["body"]) == [FILE_FINGERPRINT], (
        "a rejected file-level finding must fall back into the review body"
    )

def test_review_failure_still_reports_live_file_level_comments(
    fake_gh: FakeGh, file_level_artifact: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed review POST must acknowledge file comments already posted before it."""
    fake_gh.set_response("diff-paths", value=["b.py"])
    fake_gh.set_response("POST", "/repos/o/r/pulls/7/reviews", value=None)

    rc = cli_main(["post-findings", str(file_level_artifact), "--pr", "7", "--head-sha", "h" * 40, "--repo", "o/r"])

    assert rc == 1, "a failed review POST is still a failure"
    assert len(fake_gh.calls("POST", "/repos/o/r/pulls/7/comments")) == 1
    out = capsys.readouterr().out
    assert "1 file-level comment(s) were already posted" in out, out
    assert "No comments were posted" not in out, out
