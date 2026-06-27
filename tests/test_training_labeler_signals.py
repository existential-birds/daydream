"""Tests for posterior-signal extractors.

Each signal is a pure function over ``(manifest_row, fetcher)`` — no LLM,
no I/O beyond fetchers. Each signal has a positive/negative test pair
driven by ``_fake_gh_responder`` and a ``_simple_diff_adding`` helper.
"""

from __future__ import annotations

from pathlib import Path

from daydream.pr_review import DAYDREAM_FOOTER
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PRMergeSignal,
    comment_resolution_signal,
    fix_applied_signal,
    local_commit_applied_signal,
    pr_link_signal,
    pr_merge_signal,
    reviewer_logins_signal,
)


def _simple_diff_adding(line: str) -> str:
    """Return a one-hunk unified diff that adds ``line`` to ``app.py``."""
    return (
        "diff --git a/app.py b/app.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        f"+{line}\n"
    )


def _fake_gh_responder(responses):
    def responder(repo, endpoint, **kwargs):
        return responses[(repo, endpoint)]

    return responder


def test_pr_merge_signal_positive() -> None:
    row = {"pr_repo": "org/repo", "pr_number": 42}
    gh = _fake_gh_responder(
        {
            ("org/repo", "repos/org/repo/pulls/42"): {
                "merged": True,
                "merged_at": "2026-01-01T00:00:00Z",
            },
        }
    )
    assert pr_merge_signal(row, gh_api=gh) == PRMergeSignal(merged=True, merged_at="2026-01-01T00:00:00Z")


def test_pr_merge_signal_no_pr() -> None:
    row = {"pr_repo": None, "pr_number": None}
    assert pr_merge_signal(row, gh_api=_fake_gh_responder({})) == PRMergeSignal(merged=False, merged_at=None)


def test_fix_applied_signal_layered_cascade_returns_applied(tmp_path: Path) -> None:
    """Hunk content from diff.patch appears verbatim in a post-head commit
    on the default branch."""
    (tmp_path / "diff.patch").write_text(_simple_diff_adding("foo = 1"))
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "base_branch": "main",
        "archive_path": str(tmp_path),
    }
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: ["app.py"],
        commits_in_window_fetcher=lambda repo, base, head: ["commit1"],
        file_at_fetcher=lambda repo, path, sha: "foo = 1\n",
    )
    assert isinstance(sig, FixAppliedSignal)
    assert sig.verdict == "applied"
    assert sig.hunks_applied == 1
    assert sig.hunks_total == 1


def test_fix_applied_signal_empty_window_returns_unknown(tmp_path: Path) -> None:
    (tmp_path / "diff.patch").write_text(_simple_diff_adding("foo = 1"))
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "base_branch": "main",
        "archive_path": str(tmp_path),
    }
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: [],
        commits_in_window_fetcher=lambda repo, base, head: [],  # empty window
        file_at_fetcher=lambda repo, path, sha: "",
    )
    assert sig.verdict == "unknown"


def test_fix_applied_signal_no_file_overlap_returns_not_applied(tmp_path: Path) -> None:
    (tmp_path / "diff.patch").write_text(_simple_diff_adding("foo = 1"))
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "base_branch": "main",
        "archive_path": str(tmp_path),
    }
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: ["unrelated.py"],
        commits_in_window_fetcher=lambda repo, base, head: ["c1"],
        file_at_fetcher=lambda repo, path, sha: "",
    )
    assert sig.verdict == "not_applied"


def test_fix_applied_signal_50pct_hunk_threshold(tmp_path: Path) -> None:
    """≥50% hunks applied → applied; below → not_applied."""
    (tmp_path / "diff.patch").write_text(
        _simple_diff_adding("foo = 1") + _simple_diff_adding("bar = 2") + _simple_diff_adding("baz = 3")
    )
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "base_branch": "main",
        "archive_path": str(tmp_path),
    }
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: ["app.py"],
        commits_in_window_fetcher=lambda repo, base, head: ["c1"],
        file_at_fetcher=lambda repo, path, sha: "foo = 1\nbar = 2\n",
    )
    # 2 of 3 hunks applied → applied
    assert sig.verdict == "applied"
    assert sig.hunks_applied == 2
    assert sig.hunks_total == 3


def test_comment_resolution_signal_all_resolved() -> None:
    row = {"pr_repo": "org/repo", "pr_number": 42}
    gh = _fake_gh_responder(
        {
            ("org/repo", "repos/org/repo/pulls/42/comments"): [
                {
                    "id": 1,
                    "in_reply_to_id": None,
                    "user": {"login": "daydream-runner"},
                    "body": f"finding\n\n{DAYDREAM_FOOTER}",
                },
                {"id": 2, "in_reply_to_id": 1, "user": {"login": "human"}, "body": "ack"},
            ],
        }
    )
    assert comment_resolution_signal(row, gh_api=gh) == CommentResolutionSignal(total=1, replied=1, unresolved=0)


def test_comment_resolution_counts_footer_comment_from_human_author() -> None:
    row = {"pr_repo": "org/repo", "pr_number": 42}
    gh = _fake_gh_responder(
        {
            ("org/repo", "repos/org/repo/pulls/42/comments"): [
                {
                    "id": 1,
                    "in_reply_to_id": None,
                    "user": {"login": "kevin"},
                    "body": f"finding\n\n{DAYDREAM_FOOTER}",
                },
            ],
        }
    )
    assert comment_resolution_signal(row, gh_api=gh) == CommentResolutionSignal(total=1, replied=0, unresolved=1)


def test_comment_resolution_ignores_non_daydream_bot_comment() -> None:
    row = {"pr_repo": "org/repo", "pr_number": 42}
    gh = _fake_gh_responder(
        {
            ("org/repo", "repos/org/repo/pulls/42/comments"): [
                {"id": 1, "in_reply_to_id": None, "user": {"login": "coderabbitai[bot]"}, "body": "nit"},
            ],
        }
    )
    assert comment_resolution_signal(row, gh_api=gh) == CommentResolutionSignal(total=0, replied=0, unresolved=0)


def test_local_commit_applied_signal_positive(tmp_path: Path) -> None:
    """When the diff.patch content appears in a local commit on the branch ≥ head_sha."""
    (tmp_path / "diff.patch").write_text(_simple_diff_adding("foo = 1"))
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "branch": "feat/x",
        "archive_path": str(tmp_path),
    }
    sig = local_commit_applied_signal(
        row,
        repo_clone=tmp_path,
        commits_since_fetcher=lambda repo, branch, since_sha: ["c1"],
        file_at_fetcher=lambda repo, path, sha: "foo = 1\n",
    )
    assert sig == LocalCommitAppliedSignal(verdict="applied")


def _fake_gh_reviews():
    """gh_api stub mirroring the reviews + comments endpoints.

    /reviews → alice (human, approved) + octobot[bot] (commented).
    /comments → a daydream-runner top-level comment whose body carries
    DAYDREAM_FOOTER, with bob replying to it (so bob is a reviewer).
    """
    responses = {
        ("o/r", "repos/o/r/pulls/7/reviews"): [
            {"user": {"login": "alice"}, "state": "APPROVED"},
            {"user": {"login": "octobot[bot]"}, "state": "COMMENTED"},
        ],
        ("o/r", "repos/o/r/pulls/7/comments"): [
            {
                "id": 100,
                "in_reply_to_id": None,
                "user": {"login": "daydream-runner"},
                "body": f"Some review finding.\n\n{DAYDREAM_FOOTER}",
            },
            {
                "id": 101,
                "in_reply_to_id": 100,
                "user": {"login": "bob"},
                "body": "Good catch, fixed.",
            },
        ],
    }

    def responder(repo, endpoint, **kwargs):
        return responses[(repo, endpoint)]

    return responder


def test_reviewer_logins_signal_collects_humans_excludes_bots_and_daydream() -> None:
    logins = reviewer_logins_signal({"pr_repo": "o/r", "pr_number": 7}, gh_api=_fake_gh_reviews())
    assert logins == ["alice", "bob"]  # sorted, deduped, humans only
    assert "octobot[bot]" not in logins  # [bot] excluded
    assert "daydream-runner" not in logins  # author of the footer comment excluded


def test_local_commit_applied_signal_no_local_commits_returns_rejected(tmp_path: Path) -> None:
    (tmp_path / "diff.patch").write_text(_simple_diff_adding("foo = 1"))
    row = {
        "repo_slug": "org/repo",
        "head_sha": "abc",
        "branch": "feat/x",
        "archive_path": str(tmp_path),
    }
    sig = local_commit_applied_signal(
        row,
        repo_clone=tmp_path,
        commits_since_fetcher=lambda repo, branch, since_sha: [],
        file_at_fetcher=lambda repo, path, sha: "",
    )
    assert sig == LocalCommitAppliedSignal(verdict="rejected")


def _fake_commits_pulls(pulls):
    # pulls: list returned by repos/{slug}/commits/{sha}/pulls
    def responder(repo, endpoint, **kwargs):
        assert endpoint == "repos/org/repo/commits/abc123/pulls"
        return pulls

    return responder


def test_pr_link_signal_matches_pr_by_head_sha() -> None:
    row = {"repo_slug": "org/repo", "branch": "feat/x", "head_sha": "abc123", "pr_number": None}
    gh = _fake_commits_pulls([{"number": 7, "head": {"sha": "abc123"}}])
    assert pr_link_signal(row, gh_api=gh) == (7, "org/repo")


def test_pr_link_signal_disambiguates_multiple_pulls_by_head_sha() -> None:
    row = {"repo_slug": "org/repo", "branch": "feat/x", "head_sha": "abc123", "pr_number": None}
    gh = _fake_commits_pulls(
        [{"number": 5, "head": {"sha": "other"}}, {"number": 7, "head": {"sha": "abc123"}}]
    )
    assert pr_link_signal(row, gh_api=gh) == (7, "org/repo")


def test_pr_link_signal_returns_none_when_no_head_sha_match() -> None:
    row = {"repo_slug": "org/repo", "branch": "feat/x", "head_sha": "abc123", "pr_number": None}
    gh = _fake_commits_pulls([{"number": 5, "head": {"sha": "forcepushed"}}])
    assert pr_link_signal(row, gh_api=gh) is None


def test_pr_link_signal_returns_none_without_required_fields() -> None:
    gh = _fake_commits_pulls([])  # must NOT be called (missing head_sha short-circuits)
    assert pr_link_signal({"repo_slug": "org/repo"}, gh_api=gh) is None
