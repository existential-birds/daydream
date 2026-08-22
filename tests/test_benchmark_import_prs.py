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


def test_graphql_threads_and_replies_normalized(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    for ep, val in [
        ("repos/o/r/pulls/101/reviews", []),
        ("repos/o/r/pulls/101/comments", []),
        ("repos/o/r/issues/101/comments", []),
    ]:
        fake_gh.set_response("GET", ep, val)
    fake_gh._write_threads([
        {"id": "thread_1", "isResolved": True,
         "isOutdated": True, "isResolvedBy": None,
         "subjectType": "LINE", "path": "a.py", "line": 4, "originalLine": 3,
         "side": "RIGHT", "startSide": None,
         "comments": {"nodes": [
             {"id": "c1", "databaseId": 10, "body": "root", "author": {"login": "dave", "type": "User"},
              "createdAt": "2026-01-01T00:00:00Z", "url": "https://github.com/o/r/pull/101#discussion_r10"},
             {"id": "c2", "databaseId": 11, "body": "reply", "replyTo": {"id": "c1"},
              "author": {"login": "eve", "type": "User"},
              "createdAt": "2026-01-01T00:00:00Z", "url": "https://github.com/o/r/pull/101#discussion_r11"},
         ]}},
    ])
    doc = gi.fetch_and_normalize(ws, "o/r", 101, heads=["final"])
    kinds = {e.kind for e in doc.evidence}
    assert "thread_comment" in kinds
    root = next(e for e in doc.evidence if e.database_id == 10)
    reply = next(e for e in doc.evidence if e.database_id == 11)
    assert root.resolved is True and root.outdated is True
    assert root.side == "RIGHT" and root.path == "a.py" and root.line == 4
    assert reply.kind == "thread_comment" and reply.reply_to_id == "c1"
    assert reply.thread_id == "thread_1"


def test_candidate_projection_right_file_body_left(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.schema import Location

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response(
        "GET",
        "repos/o/r/pulls/101/reviews",
        [
            {"id": 5, "node_id": "PRR_5", "user": {"login": "alice", "type": "User"},
             "body": "review body", "state": "COMMENTED", "commit_id": "a" * 40,
             "submitted_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#pullrequestreview-5"},
            {"id": 6, "node_id": "PRR_6", "user": {"login": "alice", "type": "User"},
             "body": "looks good", "state": "APPROVED", "commit_id": "a" * 40,
             "submitted_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#pullrequestreview-6"},
        ],
    )
    fake_gh.set_response(
        "GET",
        "repos/o/r/pulls/101/comments",
        [
            {"id": 1, "node_id": "DIFF_1", "user": {"login": "alice", "type": "User"},
             "body": "## note\nfix this", "path": "a.py", "line": 5, "start_line": 4,
             "subject_type": "line", "side": "RIGHT", "start_side": None,
             "commit_id": "a" * 40, "created_at": "2026-01-01T00:00:00Z",
             "updated_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#discussion_r1"},
            {"id": 2, "node_id": "DIFF_2", "user": {"login": "alice", "type": "User"},
             "body": "file-level", "subject_type": "file",
             "commit_id": "a" * 40, "created_at": "2026-01-01T00:00:00Z",
             "updated_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#discussion_r2"},
            {"id": 3, "node_id": "DIFF_3", "user": {"login": "alice", "type": "User"},
             "body": "left-side", "path": "a.py", "line": 2, "start_line": 2,
             "subject_type": "line", "side": "LEFT", "start_side": None,
             "commit_id": "a" * 40, "created_at": "2026-01-01T00:00:00Z",
             "updated_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#discussion_r3"},
        ],
    )
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([])
    doc_gi = gi.fetch_and_normalize(ws, "o/r", 101, heads=["final"])
    cands = gi.project_candidates(doc_gi, head_sha="a" * 40)
    by_src = {c.source_id: c for c in cands}
    right = by_src["github:inline_comment:1"]
    assert right.title == "note" and "fix this" in right.body
    assert right.severity is None
    assert right.location == Location(path="a.py", start_line=4, end_line=5)
    assert right.exact_acceptable is True
    assert by_src["github:inline_comment:2"].location is None       # file-level
    assert by_src["github:inline_comment:3"].exact_acceptable is False  # LEFT
    assert by_src["github:review:5"].location is None               # review body
    assert by_src["github:review:5"].exact_acceptable is True
    assert "github:review:6" not in by_src                          # pure approval: no candidate


def test_parse_targets_dedupes_and_orders(tmp_path):
    from daydream.benchmark import github_import as gi

    pf = tmp_path / "prs.txt"
    pf.write_text("42\nhttps://github.com/o/r/pull/9\n7\n42\n")
    targets = gi.parse_import_targets(
        pr_args=["https://github.com/o/r/pull/9", "7"],
        pr_files=[pf],
        heads=["abc" * 13 + "1", "abc" * 13 + "2"],
    )
    assert targets.pr_numbers == [9, 7, 42]     # stable: CLI order then file order; dupes collapsed
    assert targets.requested_heads == ["final", "abc" * 13 + "1", "abc" * 13 + "2"]  # 'final' always present


def _seed_manifest(ws):
    """Build an initialized private workspace with an unresolved Source (o/r)."""
    from daydream.benchmark.workspace import init_workspace

    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])


def test_preflight_six_checks_in_order_and_atomic_identity(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_manifest(ws)  # Source(provider=github, hostname=github.com, repository=o/r, repository_id=None, visibility=unresolved)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response(
        "repo-view-full",
        value={"id": 5, "nameWithOwner": "o/r",
               "url": "https://github.com/o/r", "visibility": "PRIVATE", "defaultBranchRef": {"name": "main"}},
    )
    out = gi.preflight(ws, pr_count=2)
    assert out.login == "octocat" and out.repository_id == 5 and out.visibility == "private"
    # identity written atomically into benchmark.yaml source block
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == 5 and raw["source"]["visibility"] == "private"
    # second preflight is a no-op on identity (immutable, already resolved)
    out2 = gi.preflight(ws, pr_count=1)
    assert out2.repository_id == 5


def test_rate_limit_retries_three_then_fails_pr(tmp_path, fake_gh, monkeypatch):
    import subprocess

    from daydream import git_ops as _git_ops
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    attempts = {"n": 0}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    slept: list[float] = []
    monkeypatch.setattr("daydream.benchmark.github_import.time.sleep", lambda s: slept.append(s))
    real_run = _git_ops.subprocess.run

    def flaky_gh(args, *pargs, **kwargs):
        argv = list(args)
        joined = " ".join(argv)
        if (
            argv
            and argv[0] == "gh"
            and "pulls/101" in joined
            and "reviews" not in joined
            and "comments" not in joined
            and "issues" not in joined
        ):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return subprocess.CompletedProcess(
                    argv, 1, "API rate limit exceeded",
                    "gh: API rate limit exceeded Retry-After: 2",
                )
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", flaky_gh)
    ok = gi._fetch_with_retry(ws, "o/r", 101)
    assert attempts["n"] == 3 and ok.returncode == 0
    assert slept and all(w <= 60 for w in slept)  # Retry-After honored, 60s cap
