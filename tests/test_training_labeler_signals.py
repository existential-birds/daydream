"""Test pure posterior-signal extraction with all external reads supplied by fetchers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream import pr_review
from daydream.pr_review import DAYDREAM_FOOTER, PRInfo, finding_marker
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PerFindingResolution,
    PRCommentThreads,
    PRMergeSignal,
    comment_resolution_signal,
    fix_applied_signal,
    index_pr_review_comments,
    local_commit_applied_signal,
    per_finding_resolution_signal,
    pr_link_signal,
    pr_merge_signal,
    reviewer_logins_signal,
)
from tests.harness.review_payload import payload_for
from tests.harness.trajectory import diff_adding


def _fake_gh_responder(responses: Any) -> Any:
    def responder(repo: Any, endpoint: Any, **kwargs: Any) -> Any:
        return responses[(repo, endpoint)]

    return responder


def _row(**overrides: Any) -> dict[str, Any]:
    """Captured labeler inputs; callers supply the recommendation and branches."""
    return {"repo_slug": "org/repo", "head_sha": "abc", "recommended_patch": "", **overrides}


def test_reviewed_commit_line_does_not_break_daydream_footer_detection() -> None:
    """Reviewed-commit metadata must preserve detection and exactly one stable footer."""
    pr = PRInfo(
        number=1,
        head_sha="f" * 40,
        base_sha="0" * 40,
        base_ref="main",
        head_ref="feature",
        owner="acme",
        repo="widgets",
        url="https://github.com/acme/widgets/pull/1",
    )
    payload = payload_for(
        pr,
        pr_review.ClassifiedIssues(),
        renderers=pr_review.ReviewRenderers(pr_review.default_render_finding, pr_review.default_render_summary),
        run_info="Fixture run info",
    )
    body = payload["body"]
    assert "- **Reviewed commit:**" in body  # precondition: new line present
    assert body.count(DAYDREAM_FOOTER) == 1
    gh = _fake_gh_responder({("acme/widgets", "repos/acme/widgets/pulls/1/comments"): [
        _comment(1, body), _comment(2, "Acknowledged", in_reply_to=1),
    ]})
    assert comment_resolution_signal({"pr_repo": "acme/widgets", "pr_number": 1}, gh_api=gh) == (
        CommentResolutionSignal(total=1, replied=1, unresolved=0)
    )


def test_pr_merge_signal_positive() -> None:
    row = {"pr_repo": "org/repo", "pr_number": 42}
    gh = _fake_gh_responder(
        {("org/repo", "repos/org/repo/pulls/42"): {"merged": True, "merged_at": "2026-01-01T00:00:00Z"}}
    )
    assert pr_merge_signal(row, gh_api=gh) == PRMergeSignal(
        merged=True, merged_at="2026-01-01T00:00:00Z", state="merged"
    )


def test_pr_merge_signal_no_pr() -> None:
    row = {"pr_repo": None, "pr_number": None}
    assert pr_merge_signal(row, gh_api=_fake_gh_responder({})) == PRMergeSignal(merged=False, merged_at=None)


def test_fix_applied_signal_captured_recommendation_returns_applied(tmp_path: Path) -> None:
    """Hunk content from the captured recommendation appears verbatim in a post-head commit
    on the default branch."""
    row = _row(recommended_patch=diff_adding("foo = 1"), base_branch="main")
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


@pytest.mark.parametrize(
    ("files", "commits", "verdict"),
    [([], [], "unknown"), (["unrelated.py"], ["c1"], "not_applied")],
    ids=["empty-window", "no-file-overlap"],
)
def test_fix_applied_signal_without_matching_changes(
    tmp_path: Path, files: list[str], commits: list[str], verdict: str
) -> None:
    row = _row(recommended_patch=diff_adding("foo = 1"), base_branch="main")
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: files,
        commits_in_window_fetcher=lambda repo, base, head: commits,
        file_at_fetcher=lambda repo, path, sha: "",
    )
    assert sig.verdict == verdict


def test_fix_applied_signal_50pct_hunk_threshold(tmp_path: Path) -> None:
    # Distinct new-side line numbers assign each added line to exactly one hunk.
    row = _row(recommended_patch=(
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        "+foo = 1\n"
        "@@ -10,1 +11,2 @@\n"
        " existing\n"
        "+bar = 2\n"
        "@@ -20,1 +21,2 @@\n"
        " existing\n"
        "+baz = 3\n"
    ), base_branch="main")
    sig = fix_applied_signal(
        row,
        changed_files=["app.py"],
        repo_clone=tmp_path,
        diff_fetcher=lambda repo, base, head: ["app.py"],
        commits_in_window_fetcher=lambda repo, base, head: ["c1"],
        file_at_fetcher=lambda repo, path, sha: "foo = 1\nbar = 2\n",
    )
    assert sig.verdict == "applied"
    assert sig.hunks_applied == 2
    assert sig.hunks_total == 3


FP = "a" * 64


def test_comment_resolution_signal_counts_top_level_threads() -> None:
    comments = [
        _comment(1, f"finding\n\n{DAYDREAM_FOOTER}"),
        _comment(2, "ack", in_reply_to=1),
        _comment(3, f"finding\n\n{DAYDREAM_FOOTER}"),
    ]
    gh = _fake_gh_responder({("org/repo", "repos/org/repo/pulls/42/comments"): comments})
    assert comment_resolution_signal({"pr_repo": "org/repo", "pr_number": 42}, gh_api=gh) == (
        CommentResolutionSignal(total=2, replied=1, unresolved=1)
    )


def _scoped_threads(replies: list[tuple[str, dict[str, Any]]]) -> PRCommentThreads:
    comments = [_comment(10, _daydream_body(FP))] + [
        _comment(100 + i, body, in_reply_to=10, **over) for i, (body, over) in enumerate(replies)
    ]
    gh = _fake_gh_responder({("org/repo", "repos/org/repo/pulls/11/comments"): comments})
    threads = index_pr_review_comments({"pr_repo": "org/repo", "pr_number": 11}, gh_api=gh, session_fingerprints=[FP])
    assert threads is not None
    return threads


def _resolve(replies: list[tuple[str, dict[str, Any]]]) -> list[PerFindingResolution]:
    threads = _scoped_threads(replies)
    return per_finding_resolution_signal(
        {"pr_repo": "org/repo", "pr_number": 11},
        recorded_fingerprints=[FP],
        gh_api=None,
        threads=threads,
    )


def test_disposition_accepted_on_qualifying_accept() -> None:
    (res,) = _resolve([("Fixed in abc123", {"login": "maint", "assoc": "OWNER"})])
    assert res.disposition == "accepted"
    assert res.comment_id == 10
    ev = res.evidence[0]
    assert ev["reply_id"] == 100 and ev["author"] == "maint"
    assert ev["author_association"] == "OWNER" and ev["reason"] == "assoc:OWNER"


def test_disposition_rejected_on_false_positive() -> None:
    (res,) = _resolve([("False positive, path is unreachable", {"login": "dev", "assoc": "MEMBER"})])
    assert res.disposition == "rejected"


def test_disposition_ambiguous_on_question() -> None:
    (res,) = _resolve([("Is this still true on 3.12?", {"login": "dev", "assoc": "MEMBER"})])
    assert res.disposition == "ambiguous"


def test_disposition_non_qualifying_author_does_not_vote() -> None:
    """A non-qualifying author's decisive-looking reply stays excluded evidence: no vote, no decisive
    timestamp, and an unanswered disposition.
    """
    (res,) = _resolve([("Fixed in abc123", {"login": "dev", "assoc": "NONE"})])
    assert res.disposition == "unanswered"
    assert res.evidence[0]["reason"] == "excluded:non-qualifying"


def test_disposition_unanswered_on_bot_only_and_self_replies() -> None:
    (res,) = _resolve(
        [
            ("Fixed in abc", {"bot": "Bot", "login": "app[bot]", "assoc": "NONE"}),
            ("Fixed in abc", {"login": "daydream-agent", "assoc": "NONE", "_self": True}),
        ]
    )
    assert res.disposition == "unanswered"
    # Persist excluded evidence so the reason remains auditable.
    assert len(res.evidence) == 2
    assert all("excluded" in ev["reason"] for ev in res.evidence)


def test_local_commit_applied_signal_positive(tmp_path: Path) -> None:
    row = _row(recommended_patch=diff_adding("foo = 1"), branch="feat/x")
    sig = local_commit_applied_signal(
        row,
        repo_clone=tmp_path,
        commits_since_fetcher=lambda repo, branch, since_sha: ["c1"],  # noqa
        file_at_fetcher=lambda repo, path, sha: "foo = 1\n",
    )
    assert sig == LocalCommitAppliedSignal(verdict="applied")


def _fake_gh_reviews() -> Any:
    """Serve human/bot reviews and a footer-marked finding with a human reply, exercising reviewer
    extraction from both endpoints.
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
            {"id": 101, "in_reply_to_id": 100, "user": {"login": "bob"}, "body": "Good catch, fixed."},
        ],
    }

    return _fake_gh_responder(responses)


def test_reviewer_logins_signal_collects_humans_excludes_bots_and_daydream() -> None:
    logins = reviewer_logins_signal({"pr_repo": "o/r", "pr_number": 7}, gh_api=_fake_gh_reviews())
    assert logins == ["alice", "bob"]  # sorted, deduped, humans only
    assert "octobot[bot]" not in logins  # [bot] excluded
    assert "daydream-runner" not in logins  # author of the footer comment excluded


def test_local_commit_applied_signal_no_local_commits_returns_rejected(tmp_path: Path) -> None:
    row = _row(recommended_patch=diff_adding("foo = 1"), branch="feat/x")
    sig = local_commit_applied_signal(
        row,
        repo_clone=tmp_path,
        commits_since_fetcher=lambda repo, branch, since_sha: [],  # noqa
        file_at_fetcher=lambda repo, path, sha: "",
    )
    assert sig == LocalCommitAppliedSignal(verdict="rejected")


def test_local_commit_applied_signal_unreadable_window_returns_unknown(tmp_path: Path) -> None:
    """An unreadable commit window is unknown, not rejection. The fixture also lacks the change on base, so
    fallback cannot upgrade it.
    """
    row = _row(recommended_patch=diff_adding("foo = 1"), branch="feat/x", base_branch="main")
    sig = local_commit_applied_signal(
        row,
        repo_clone=tmp_path,
        commits_since_fetcher=lambda repo, branch, since_sha: None,  # noqa
        file_at_fetcher=lambda repo, path, sha: "",
    )
    assert sig == LocalCommitAppliedSignal(verdict="unknown")


@pytest.mark.parametrize(
    ("available_ref", "expected_refs"),
    [
        pytest.param("origin/main", ["origin/main"], id="deleted-branch-remote-tip"),
        pytest.param("main", ["origin/main", "main"], id="remote-missing-local-fallback"),
    ],
)
def test_local_commit_applied_signal_base_branch_fallback(
    tmp_path: Path,
    available_ref: str,
    expected_refs: list[str],
) -> None:
    """A deleted branch uses the remote base before the possibly stale local tip."""
    row = _row(recommended_patch=diff_adding("foo = 1"), branch="feat/squashed-away", base_branch="main")
    seen_refs: list[str] = []

    def _file_at(repo: Path, path: str, ref: str) -> str:
        seen_refs.append(ref)
        return "existing\nfoo = 1\n" if ref == available_ref else ""

    sig = local_commit_applied_signal(
        row,
        repo_clone=tmp_path,
        commits_since_fetcher=lambda _repo, _branch, _since_sha: None,
        file_at_fetcher=_file_at,
    )
    assert sig == LocalCommitAppliedSignal(verdict="applied")
    assert seen_refs == expected_refs


def test_local_commit_applied_signal_unreadable_window_no_hunks_is_unknown(tmp_path: Path) -> None:
    """No recommended hunks means no applied change; an empty search must not vacuously prove success."""
    row = _row(recommended_patch="", branch="feat/squashed-away", base_branch="main")
    sig = local_commit_applied_signal(
        row,
        repo_clone=tmp_path,
        commits_since_fetcher=lambda repo, branch, since_sha: None,  # noqa
        file_at_fetcher=lambda repo, path, ref: "anything at all\n",
    )
    assert sig == LocalCommitAppliedSignal(verdict="unknown")


def _fake_commits_pulls(pulls: Any) -> Any:
    # pulls: list returned by repos/{slug}/commits/{sha}/pulls
    def responder(repo: Any, endpoint: str, **kwargs: Any) -> Any:
        assert endpoint == "repos/org/repo/commits/abc123/pulls"
        return pulls

    return responder


@pytest.mark.parametrize(
    "pulls",
    [
        pytest.param([{"number": 7, "head": {"sha": "abc123"}}], id="single-match"),
        pytest.param(
            [{"number": 5, "head": {"sha": "other"}}, {"number": 7, "head": {"sha": "abc123"}}],
            id="disambiguated-match",
        ),
    ],
)
def test_pr_link_signal_matches_pr_by_head_sha(pulls: list[dict[str, Any]]) -> None:
    row = {"repo_slug": "org/repo", "branch": "feat/x", "head_sha": "abc123", "pr_number": None}
    gh = _fake_commits_pulls(pulls)
    assert pr_link_signal(row, gh_api=gh) == (7, "org/repo")


def test_pr_link_signal_returns_none_when_no_head_sha_match() -> None:
    row = {"repo_slug": "org/repo", "branch": "feat/x", "head_sha": "abc123", "pr_number": None}
    gh = _fake_commits_pulls([{"number": 5, "head": {"sha": "forcepushed"}}])
    assert pr_link_signal(row, gh_api=gh) is None


def test_pr_link_signal_returns_none_without_required_fields() -> None:
    gh = _fake_commits_pulls([])  # must NOT be called (missing head_sha short-circuits)
    assert pr_link_signal({"repo_slug": "org/repo"}, gh_api=gh) is None


_FP_A = "a" * 64
_FP_B = "b" * 64
_FP_C = "c" * 64


def _daydream_finding_comment(comment_id: int, fingerprint: str) -> dict[str, Any]:
    """A top-level daydream review comment carrying the footer + finding marker."""
    return {
        "id": comment_id,
        "in_reply_to_id": None,
        "user": {"login": "daydream-runner"},
        "body": f"A finding.\n\n{finding_marker(fingerprint)}\n\n{DAYDREAM_FOOTER}",
    }


def test_per_finding_resolution_signal_mixed_outcomes() -> None:
    row = {"pr_repo": "org/repo", "pr_number": 42}
    gh = _fake_gh_responder(
        {
            ("org/repo", "repos/org/repo/pulls/42/comments"): [
                _daydream_finding_comment(1, _FP_A),
                _daydream_finding_comment(2, _FP_B),
                _comment(3, "Fixed in abc123", in_reply_to=1, login="human"),
            ],
        }
    )
    result = per_finding_resolution_signal(row, recorded_fingerprints=[_FP_A, _FP_B], gh_api=gh)
    assert [(r.fingerprint, r.comment_id, r.disposition) for r in result] == [
        (_FP_A, 1, "accepted"),
        (_FP_B, 2, "unanswered"),
    ]
    assert result[0].evidence and not result[1].evidence


def test_per_finding_resolution_signal_deleted_comment() -> None:
    row = {"pr_repo": "org/repo", "pr_number": 42}
    gh = _fake_gh_responder({("org/repo", "repos/org/repo/pulls/42/comments"): [_daydream_finding_comment(1, _FP_A)]})
    result = per_finding_resolution_signal(row, recorded_fingerprints=[_FP_A, _FP_C], gh_api=gh)
    assert [(r.fingerprint, r.comment_id, r.disposition) for r in result] == [
        (_FP_A, 1, "unanswered"),
        (_FP_C, None, "missing"),
    ]


def test_per_finding_resolution_signal_single_finding() -> None:
    """Avoid a pre-matching negation token in the reject reply: sentence-level "not" would negate "already
    handled" and make the result ambiguous.
    """
    row = {"pr_repo": "org/repo", "pr_number": 42}
    gh = _fake_gh_responder(
        {
            ("org/repo", "repos/org/repo/pulls/42/comments"): [
                _daydream_finding_comment(9, _FP_A),
                _comment(10, "This is already handled upstream", in_reply_to=9, login="human"),
            ],
        }
    )
    result = per_finding_resolution_signal(row, recorded_fingerprints=[_FP_A], gh_api=gh)
    assert len(result) == 1
    assert (result[0].comment_id, result[0].disposition) == (9, "rejected")


def test_per_finding_resolution_signal_no_pr() -> None:
    row = {"pr_repo": None, "pr_number": None}
    result = per_finding_resolution_signal(row, recorded_fingerprints=[_FP_A], gh_api=_fake_gh_responder({}))
    assert result == []


def _comment(
    cid: int,
    body: str,
    in_reply_to: int | None = None,
    login: str = "alice",
    assoc: str = "MEMBER",
    bot: str = "User",
    created: str = "2026-08-01T00:00:00Z",
    **extra: Any,
) -> dict[str, Any]:
    self_reply = extra.pop("_self", False)
    comment: dict[str, Any] = {
        "id": cid,
        "in_reply_to_id": in_reply_to,
        "user": {"login": login, "type": bot},
        "author_association": assoc,
        "body": body,
        "created_at": created,
    }
    if self_reply:
        comment["is_self_reply"] = True
    comment.update(extra)
    return comment


def _daydream_body(*fps: str) -> str:
    return "\n".join(finding_marker(fp) for fp in fps) + "\n" + DAYDREAM_FOOTER


def test_pr_merge_signal_preserves_state() -> None:
    gh = _fake_gh_responder(
        {
            ("org/repo", "repos/org/repo/pulls/1"): {
                "merged": False,
                "merged_at": None,
                "state": "open",
                "draft": False,
            },
            ("org/repo", "repos/org/repo/pulls/2"): {
                "merged": False,
                "merged_at": None,
                "state": "closed",
                "draft": False,
            },
        }
    )
    assert pr_merge_signal({"pr_repo": "org/repo", "pr_number": 1}, gh_api=gh).state == "open"
    sig2 = pr_merge_signal({"pr_repo": "org/repo", "pr_number": 2}, gh_api=gh)
    assert sig2.state == "closed" and sig2.merged is False


def test_pr_merge_signal_legacy_payload_defaults() -> None:
    """A payload without state/draft (cached fixtures) degrades to safe defaults, not 'closed'."""
    gh = _fake_gh_responder({("org/repo", "repos/org/repo/pulls/3"): {"merged": False, "merged_at": None}})
    sig = pr_merge_signal({"pr_repo": "org/repo", "pr_number": 3}, gh_api=gh)
    assert sig.state == "unknown" and sig.draft is False


def test_thread_index_keeps_full_reply_objects() -> None:
    fp_a, fp_b = "a" * 64, "b" * 64
    comments = [
        _comment(10, _daydream_body(fp_a)),
        _comment(11, _daydream_body(fp_b)),
        _comment(20, "Fixed in abc123", in_reply_to=10, login="maint", assoc="OWNER", created="2026-08-02T10:00:00Z"),
    ]
    gh = _fake_gh_responder({("org/repo", "repos/org/repo/pulls/5/comments"): comments})
    threads = index_pr_review_comments({"pr_repo": "org/repo", "pr_number": 5}, gh_api=gh)
    assert threads is not None
    replies = threads.replies_by_comment[10]
    assert len(replies) == 1
    r = replies[0]
    assert r["user"]["login"] == "maint" and r["author_association"] == "OWNER"
    assert r["body"] == "Fixed in abc123" and r["created_at"] == "2026-08-02T10:00:00Z"


def test_thread_index_scopes_to_fingerprints() -> None:
    fp_mine, fp_other = "c" * 64, "d" * 64
    comments = [
        _comment(30, _daydream_body(fp_mine)),
        _comment(31, _daydream_body(fp_other)),  # another daydream run's finding
        _comment(40, "already handled", in_reply_to=31),  # reply to OTHER run's thread
        _comment(41, "fixed in abc", in_reply_to=30),
    ]
    gh = _fake_gh_responder({("org/repo", "repos/org/repo/pulls/7/comments"): comments})
    threads = index_pr_review_comments(
        {"pr_repo": "org/repo", "pr_number": 7},
        gh_api=gh,
        session_fingerprints=[fp_mine],
    )
    assert threads is not None
    assert set(threads.comment_id_by_fingerprint) == {fp_mine}
    assert 31 not in threads.top_level_daydream_ids
    assert threads.replies_by_comment.get(31) is None  # other-run thread not in evidence


def test_comment_resolution_signal_becomes_fingerprint_scoped_aggregate() -> None:
    fp_mine, fp_other = "e" * 64, "f" * 64
    comments = [
        _comment(50, _daydream_body(fp_mine)),
        _comment(51, _daydream_body(fp_other)),
        _comment(60, "thanks", in_reply_to=51),
    ]
    gh = _fake_gh_responder({("org/repo", "repos/org/repo/pulls/9/comments"): comments})
    row = {"pr_repo": "org/repo", "pr_number": 9}
    threads = index_pr_review_comments(row, gh_api=gh, session_fingerprints=[fp_mine])
    sig = comment_resolution_signal(row, gh_api=gh, threads=threads)
    assert (sig.total, sig.replied, sig.unresolved) == (1, 0, 1)
