"""Tests for ``daydream benchmark import-prs``.

Task 0 spike: the load-bearing claim that the importer's ``gh``/``git``
calls, made through :mod:`daydream.git_ops` with ``cwd=<workspace root>``,
are intercepted by the in-process ``fake_gh`` router. Subsequent
collection/normalization/projection/orchestration tasks build on this seam.
"""

import json

import pytest

_PR_HEADER = {
    "number": 101,
    "url": "https://github.com/o/r/pull/101",
    "title": "Fix cache",
    "state": "open",
    "base": {"ref": "main", "sha": "b" * 40},
    "head": {"ref": "feature/cache", "sha": "a" * 40},
    "merged_at": None,
    "closed_at": None,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "user": {"login": "alice", "type": "User"},
}


def test_preflight_gh_and_ls_remote_route_through_fake(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi

    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    ws = tmp_path / "ws"
    ws.mkdir()
    status = gi._run_gh_preflight_status(ws)       # gh auth status --hostname github.com
    login = gi._run_gh_api_user(ws)                # gh api user
    cred = gi._gh_auth_git_credential(ws)          # gh auth git-credential
    refs = gi._git_ls_remote(ws, "https://github.com/o/r.git")  # git ls-remote
    assert status.returncode == 0
    assert login == {"login": "octocat", "type": "User"}
    assert "password=" in cred
    assert "refs/heads/head" in refs


def test_fetch_normalizes_all_rest_evidence(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response(
        "GET",
        "repos/o/r/pulls/101/reviews",
        [
            {"id": 1, "node_id": "PRR_1", "user": {"login": "alice", "type": "User"},
             "body": "approved", "state": "APPROVED", "commit_id": "a" * 40,
             "submitted_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#pullrequestreview-1"},
        ],
    )
    fake_gh.set_response(
        "GET",
        "repos/o/r/pulls/101/comments",
        [
            {"id": 7, "node_id": "DIFF_7", "user": {"login": "bot[bot]", "type": "Bot"},
             "body": "please fix", "commit_id": "a" * 40, "original_commit_id": "a" * 40,
             "path": "a.py", "original_position": 3, "line": 4, "original_line": 3,
             "subject_type": "line", "side": "RIGHT", "created_at": "2026-01-01T00:00:00Z",
             "updated_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#discussion_r7"},
        ],
    )
    fake_gh.set_response(
        "GET",
        "repos/o/r/issues/101/comments",
        [
            {"id": 9, "user": {"login": "carol", "type": "User"}, "body": "question",
             "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#issuecomment-9"},
        ],
    )
    doc = gi.fetch_and_normalize(ws, "o/r", 101, heads=["final"])
    kinds = {e.kind for e in doc.evidence}
    assert kinds == {"review", "inline_comment", "issue_comment"}
    assert doc.evidence[0].source_id == "github:review:1"
    assert doc.evidence[0].is_bot is False
    assert doc.evidence[1].is_bot is True      # bot classification retained, not dropped
    assert doc.evidence[1].subject_type == "line" and doc.evidence[1].side == "RIGHT"
    assert all("--paginate" in (c.argv or []) for c in fake_gh.calls("GET"))
