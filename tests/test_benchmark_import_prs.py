"""Tests for ``daydream benchmark import-prs``.

Covers the full import surface through the in-process ``fake_gh`` router:
target parsing, six-check preflight + immutable identity, REST + GraphQL
fetch/import, candidate projection, bounded rate-limit retry, atomic
ledger/case transactions, refresh/staleness (curation preservation), snapshot
freeze wiring (ready|unreplayable cases + bundle staging), and both e2e
acceptance paths including partial-failure persistence. All ``gh`` calls route
through the ``fake_gh`` router; freeze mirror fetches hit a real local bare
origin (no network).
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.harness import github_schema as gs
from tests.harness.fake_gh import FakeGh

# Deterministic seed identity so a local bare origin's commits are stable and
# reproducible (mirrors tests/test_benchmark_snapshot.py::_SEED_ENV).
_SEED_ENV = {
    "GIT_AUTHOR_NAME": "Tester",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Tester",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}

_PR_HEADER = {
    "number": 101,
    "url": "https://github.com/o/r/pull/101",
    "html_url": "https://github.com/o/r/pull/101",
    "title": "Fix cache",
    "body": "",
    "state": "open",
    "base": {"ref": "main", "sha": "b" * 40},
    "head": {"ref": "feature/cache", "sha": "a" * 40},
    "merged_at": None,
    "closed_at": None,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "user": {"login": "alice", "type": "User"},
}

_REPO_ID = "R_kgDOABC123"
_REPO_VIEW = {
    "id": _REPO_ID,
    "nameWithOwner": "o/r",
    "url": "https://github.com/o/r",
    "visibility": "PRIVATE",
    "defaultBranchRef": {"name": "main"},
}


def test_preflight_gh_and_ls_remote_wire_command_scoped_helper(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    ws.mkdir()
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    assert gi._run_gh_preflight_status(ws).returncode == 0
    assert gi._run_gh_api_user(ws) == {"login": "octocat", "type": "User"}
    refs = gi._git_ls_remote(ws, "https://github.com/o/r.git")
    assert "refs/heads/head" in refs
    ls = fake_gh.command_calls("git ls-remote")[-1]
    joined = " ".join(ls.argv)
    assert "-c" in ls.argv and any(a.startswith("credential.helper=") for a in ls.argv)
    assert "gh auth git-credential" in joined and "password=" not in joined
    assert ls.env is not None and ls.env.get("GIT_TERMINAL_PROMPT") == "0"


def test_fetch_persists_complete_pr_header(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    header = dict(_PR_HEADER)
    header["body"] = "fixes the cache\n\nand tests"
    header["html_url"] = "https://github.com/o/r/pull/101"
    header["merged_at"] = "2026-01-02T00:00:00Z"
    header["closed_at"] = "2026-01-02T00:00:00Z"
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    for ep in ("repos/o/r/pulls/101/reviews", "repos/o/r/pulls/101/comments",
               "repos/o/r/issues/101/comments"):
        fake_gh.set_response("GET", ep, [])
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    pr = doc.pull_request
    assert pr.body == "fixes the cache\n\nand tests"
    assert pr.html_url == "https://github.com/o/r/pull/101"
    assert pr.title_sha256 == hashlib.sha256(b"Fix cache").hexdigest()
    assert pr.body_sha256 == hashlib.sha256("fixes the cache\n\nand tests".encode()).hexdigest()
    assert pr.head.ref == "feature/cache"          # head.ref parity with base.ref
    assert pr.merged_at is not None and pr.closed_at is not None
    assert pr.number == 101 and pr.author.login == "alice"


def test_materialized_case_carries_full_pr_header(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)                  # REST + canned PR for pr 101
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    pr = case["pull_request"]
    assert pr["head"]["ref"] == "feature/cache"    # head.ref persisted in the case YAML
    assert pr["body"] == ""                         # _PR_HEADER has no body -> empty
    assert pr["title_sha256"] and pr["body_sha256"]
    assert "merged_at" in pr and "closed_at" in pr and "html_url" in pr


def test_import_only_snapshot_records_requested_base_sha(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Import-only (root=None, no freeze) writes SnapshotImported with both
    SHAs = the PR base tip (origin/head SHAs are the PR-known values; the merge
    base is not yet computed and diverges on imported -> ready).
    """
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)                 # REST + canned PR for pr 101
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    snapshot = case["snapshot"]
    assert snapshot["status"] == "imported"
    # both base SHAs carry the PR base tip at import
    assert snapshot["requested_base_sha"] == snapshot["original_base_sha"]
    assert snapshot["original_base_sha"] == "b" * 40
    assert snapshot["requested_base_sha"] == "b" * 40


def test_fetch_normalizes_null_body_to_empty_string(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    header = dict(_PR_HEADER)
    header["body"] = None                          # GitHub returns null for empty
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    for ep in ("repos/o/r/pulls/101/reviews", "repos/o/r/pulls/101/comments",
               "repos/o/r/issues/101/comments"):
        fake_gh.set_response("GET", ep, [])
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    assert doc.pull_request.body == ""             # null -> empty string, never "None"


@pytest.mark.parametrize("body_field,expected", [
    (None, ""),                      # null body -> empty
    ("", ""),                        # empty body
    ("héllo wörld \u00e9", "héllo wörld \u00e9"),          # Unicode preserved
    ("line1\nline2\nline3", "line1\nline2\nline3"),        # newlines preserved
    ("x" * 50000, "x" * 50000),      # over context-limit body (never bounded here; persisted whole)
])
def test_import_body_shape_preserved(tmp_path: Path, fake_gh: FakeGh, body_field: Any, expected: str) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    header = dict(_PR_HEADER)
    header["body"] = body_field
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    for ep in ("repos/o/r/pulls/101/reviews", "repos/o/r/pulls/101/comments",
               "repos/o/r/issues/101/comments"):
        fake_gh.set_response("GET", ep, [])
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    assert doc.pull_request.body == expected
    assert doc.pull_request.body_sha256 == hashlib.sha256(expected.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("state,merged_at,closed_at,expect_merged", [
    ("open", None, None, False),
    ("closed", None, "2026-01-02T00:00:00Z", False),     # closed-unmerged
    ("closed", "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z", True),  # merged
])
def test_import_merged_state_distinction(
    tmp_path: Path,
    fake_gh: FakeGh,
    state: Any,
    merged_at: Any,
    closed_at: Any,
    expect_merged: Any,
) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    header = dict(_PR_HEADER)
    header["state"] = state
    header["merged_at"] = merged_at
    header["closed_at"] = closed_at
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    for ep in ("repos/o/r/pulls/101/reviews", "repos/o/r/pulls/101/comments",
               "repos/o/r/issues/101/comments"):
        fake_gh.set_response("GET", ep, [])
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    pr = doc.pull_request
    assert pr.state == state
    assert (pr.merged_at is not None) == expect_merged
    assert (pr.closed_at is not None) == (closed_at is not None)


def test_import_no_comments_pr(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    header = dict(_PR_HEADER)
    header["body"] = "no comments here"
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    for ep in ("repos/o/r/pulls/101/reviews", "repos/o/r/pulls/101/comments",
               "repos/o/r/issues/101/comments"):
        fake_gh.set_response("GET", ep, [])
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    assert doc.evidence == [] and doc.pull_request.body == "no comments here"


def test_payload_digest_spans_header_and_evidence(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    def fetch_with(title: Any) -> Any:
        ws = tmp_path / "ws"
        (ws / "imports").mkdir(parents=True, exist_ok=True)
        header = dict(_PR_HEADER)
        header["title"] = title
        header["body"] = "b"
        fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
        for ep in ("repos/o/r/pulls/101/reviews", "repos/o/r/pulls/101/comments",
                   "repos/o/r/issues/101/comments"):
            fake_gh.set_response("GET", ep, [])
        return gi.fetch_and_normalize(ws, "o/r", 101)
    a = fetch_with("Fix cache")
    b = fetch_with("Fix cache EDITED")            # header-only change, same evidence
    assert a.fetch.payload_sha256 != b.fetch.payload_sha256
    # a header-only change must flip the digest even with identical evidence
    assert gi._evidence_signature_from_doc(a) == gi._evidence_signature_from_doc(b)


def test_fetch_normalizes_all_rest_evidence(tmp_path: Path, fake_gh: FakeGh) -> None:
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
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    kinds = {e.kind for e in doc.evidence}
    assert kinds == {"review", "inline_comment", "issue_comment"}
    assert doc.evidence[0].source_id == "github:review:1"
    assert doc.evidence[0].is_bot is False
    assert doc.evidence[1].is_bot is True      # bot classification retained, not dropped
    assert doc.evidence[1].subject_type == "line" and doc.evidence[1].side == "RIGHT"
    get_calls = fake_gh.calls("GET")
    header_args = [c.argv for c in get_calls if c.endpoint == "repos/o/r/pulls/101"]
    collection_args = [c.argv for c in get_calls if c.endpoint != "repos/o/r/pulls/101"]
    # list endpoints paginate; the singular header is fetched as one object, not array-flattened
    assert header_args and "@json" in (header_args[0] or []) and "--paginate" not in (header_args[0] or [])
    assert all("--paginate" in (a or []) for a in collection_args)


def test_review_thread_queries_request_only_schema_fields() -> None:
    """Both GraphQL queries may only request fields GitHub's schema defines.

    Aliases (`side: diffSide`, `type: __typename`) are allowed — the extractor
    validates the *real* field name — but a bare invented field must fail.
    """
    from daydream.benchmark import github_import as gi
    assert gs.unknown_query_fields(gi._REVIEW_THREADS_QUERY) == set()
    assert gs.unknown_query_fields(gi._THREAD_COMMENTS_QUERY) == set()


def test_fake_gh_rejects_invented_review_thread_fields(tmp_path: Path) -> None:
    """Reintroducing a field GitHub's schema does not define fails CI through the fake gh."""
    from tests.harness.fake_gh import _handle_api
    state = tmp_path / "state"
    state.mkdir()
    payload = state / "q.json"
    payload.write_text(json.dumps({
        "query": "query X($o:String!,$n:String!,$p:Int!){ repository(owner:$o,name:$n){"
                 " pullRequest(number:$p){ reviewThreads(first:50){ nodes{ id isBot } } } } }",
        "variables": {"o": "o", "n": "r", "p": 1},
    }))
    rc, _out, err = _handle_api(["api", "graphql", "--input", str(payload)], state)
    assert rc == 1
    assert "isBot" in err


def test_fake_gh_accepts_fixed_review_threads_query(tmp_path: Path) -> None:
    """The fixed production query routes through the fake without a schema rejection."""
    from daydream.benchmark import github_import as gi
    from tests.harness.fake_gh import _handle_api
    state = tmp_path / "state"
    state.mkdir()
    payload = state / "q.json"
    payload.write_text(json.dumps({
        "query": gi._REVIEW_THREADS_QUERY,
        "variables": {"o": "o", "n": "r", "number": 1},
    }))
    rc, _out, err = _handle_api(["api", "graphql", "--input", str(payload)], state)
    assert rc == 0 and "schema" not in err


def test_graphql_threads_and_replies_normalized(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    # REST comments for db 10 (root) and 11 (reply to 10)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [
        {"id": 10, "node_id": "DIFF_10", "user": {"login": "dave", "type": "User"},
         "body": "root", "commit_id": "a" * 40, "original_commit_id": "a" * 40,
         "path": "a.py", "original_path": "a.py", "line": 4, "original_line": 3,
         "subject_type": "line", "side": "RIGHT",
         "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#discussion_r10"},
        {"id": 11, "node_id": "DIFF_11", "user": {"login": "eve", "type": "User"},
         "body": "reply", "commit_id": "a" * 40, "path": "a.py", "line": 5,
         "subject_type": "line", "side": "RIGHT", "in_reply_to_id": 10,
         "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#discussion_r11"},
    ])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([
        {"id": "thread_1", "isResolved": True,
         "isOutdated": True,
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
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    by_db = {e.database_id: e for e in doc.evidence}
    root, reply = by_db[10], by_db[11]
    assert root.kind == "inline_comment" and root.source_id == "github:inline_comment:10"
    assert root.resolved is True and root.outdated is True
    assert root.thread_id == "thread_1" and root.side == "RIGHT" and root.path == "a.py"
    assert reply.kind == "inline_comment" and reply.thread_id == "thread_1"
    assert reply.reply_to_id == "10"          # REST in_reply_to_id (parent db id)
    assert not any(e.kind == "thread_comment" for e in doc.evidence)


def test_candidate_projection_right_file_body_left(tmp_path: Path, fake_gh: FakeGh) -> None:
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
    doc_gi = gi.fetch_and_normalize(ws, "o/r", 101)
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


def test_parse_targets_dedupes_and_orders(tmp_path: Path) -> None:
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


def test_parse_head_pr_sha_grammar_and_binding(tmp_path: Path) -> None:
    """``--head PR=<40-hex>`` binds the explicit head to that PR only.

    A bare 40-hex stays a back-compat superset; an unparseable RHS raises
    :class:`ImportTargetError`; a bound PR that is not imported is rejected so
    the binding can never be silently dropped.
    """
    import pytest

    from daydream.benchmark import github_import as gi

    sha = "a" * 40
    targets = gi.parse_import_targets(["101"], [], [f"101={sha}"])
    assert targets.requested_heads == ["final", sha]
    assert targets.pr_heads == {101: ["final", sha]}
    targets2 = gi.parse_import_targets(["101"], [], [sha])
    assert targets2.requested_heads == ["final", sha]
    with pytest.raises(gi.ImportTargetError):
        gi.parse_import_targets(["101"], [], ["101=nothex"])
    # a bound PR that is never requested cannot be honored, so it is rejected
    with pytest.raises(gi.ImportTargetError):
        gi.parse_import_targets(["100"], [], [f"101={sha}"])


def test_parse_heads_bound_per_pr_in_multi_import(tmp_path: Path) -> None:
    """A ``PR=<sha>`` head is honored for that PR only, never spread to others.

    Regression guard for the bug where ``--pr 100 --pr 101 --head 101=<sha>``
    misapplied ``<sha>`` to PR 100 too (the binding was parsed then dropped).
    """
    from daydream.benchmark import github_import as gi

    sha = "a" * 40
    targets = gi.parse_import_targets(["100", "101"], [], [f"101={sha}"])
    assert targets.pr_numbers == [100, 101]
    assert targets.pr_heads == {100: ["final"], 101: ["final", sha]}
    assert targets.requested_heads == ["final", sha]


def _seed_manifest(ws: Path) -> None:
    """Build an initialized private workspace with an unresolved Source (o/r).

    Idempotent: a caller may explicitly :func:`init_workspace` first (e.g. the
    snapshot-freeze test pins reviewer/judge hosts), in which case the manifest
    already exists and the scaffold is left untouched.
    """
    from daydream.benchmark.workspace import init_workspace

    if (ws / "benchmark.yaml").exists():
        return
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])


def test_preflight_six_checks_in_order_and_atomic_identity(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_manifest(ws)  # unresolved Source (repository=o/r)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    out = gi.preflight(ws, pr_count=2)
    assert out.login == "octocat" and out.repository_id == _REPO_ID and out.visibility == "private"
    # identity written atomically into benchmark.yaml source block
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == _REPO_ID and raw["source"]["visibility"] == "private"
    # a second preflight re-runs repo view and still matches (identity is
    # immutable but re-verified on every import/refresh)
    out2 = gi.preflight(ws, pr_count=1)
    assert out2.repository_id == _REPO_ID


def test_preflight_reverifies_identity_on_every_run_and_fails_closed(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_manifest(ws)                                  # unresolved Source (repository=repo)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    out = gi.preflight(ws, pr_count=2)
    assert out.login == "octocat" and out.repository_id == _REPO_ID and out.visibility == "private"
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == _REPO_ID and raw["source"]["visibility"] == "private"

    # a later run sees a renamed/moved repository (node id changed) -> must fail closed
    fake_gh.set_response("repo-view-full", value={**_REPO_VIEW, "id": "R_kgDDIFFERENT"})
    with pytest.raises(gi.PreflightError) as ei:
        gi.preflight(ws, pr_count=1)
    assert ei.value.code == "repo_mismatch"
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == _REPO_ID    # unchanged: no mutation staged


def test_preflight_rejects_numeric_node_id(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    _seed_manifest(ws)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value={**_REPO_VIEW, "id": 5})
    with pytest.raises(gi.PreflightError) as ex:
        gi.preflight(ws, pr_count=1)
    assert ex.value.code == "repo_unresolved"


def test_preflight_rejects_numeric_string_node_id(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    _seed_manifest(ws)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    # The legacy stale-int representation can arrive as a numeric-only *string*;
    # _verify_repo_view must fail closed on it too (isdigit() reject branch).
    fake_gh.set_response("repo-view-full", value={**_REPO_VIEW, "id": "12345"})
    with pytest.raises(gi.PreflightError) as ex:
        gi.preflight(ws, pr_count=1)
    assert ex.value.code == "repo_unresolved"


def test_status_reports_last_preflight_verification(
    tmp_path: Path,
    fake_gh: FakeGh,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.cli import _handle_benchmark_status
    from daydream.benchmark.storage import load_json_strict
    from daydream.benchmark.workspace import workspace_status

    ws = tmp_path / "ws"
    _seed_manifest(ws)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    gi.preflight(ws, pr_count=1)

    st = workspace_status(ws)
    assert st.last_preflight_verified_at is not None
    ledger = load_json_strict(ws / "runtime" / "preflight.json")
    assert ledger["repository_id"] == _REPO_ID and ledger["matched"] is True

    # a pre-fix / never-run workspace reports "not yet run"
    ws2 = tmp_path / "ws2"
    _seed_manifest(ws2)
    assert workspace_status(ws2).last_preflight_verified_at is None

    # the CLI status line surfaces verification ran / not yet run
    _handle_benchmark_status(ws)
    assert "repository identity/access verification: ran" in capsys.readouterr().out
    _handle_benchmark_status(ws2)
    assert "repository identity/access verification: not yet run" in capsys.readouterr().out




def test_rate_limit_retries_three_then_fails_pr(
    tmp_path: Path,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

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
    real_run = subprocess.run

    def flaky_gh(args: Any, *pargs: Any, **kwargs: Any) -> Any:
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
    assert attempts["n"] == 3 and ok["number"] == 101
    assert slept and all(w <= 60 for w in slept)  # Retry-After honored, 60s cap


def _seed_preflight(ws: Any, fake_gh: FakeGh, *, pull_header: Any=_PR_HEADER) -> None:
    """Seed an unresolved workspace + canned preflight/REST data for pr 101."""
    _seed_manifest(ws)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    fake_gh.set_response("GET", "repos/o/r/pulls/101", pull_header)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])


# ---------------------------------------------------------------------------
# real-git local-origin seed for snapshot-freeze wiring (no network)
# ---------------------------------------------------------------------------


def _seed_git(repo: Path, *args: str, check: bool = True, env: dict[str, str] | None = None) -> str:
    """Run git in *repo*, returning stripped stdout."""
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        env={**os.environ, **env} if env else os.environ.copy(), check=check,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _seed_write(repo: Path, name: str, content: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _seed_git(repo, "add", name)


def _seed_commit(repo: Path, message: str) -> str:
    _seed_git(repo, "commit", "-m", message, env=_SEED_ENV)
    return _seed_git(repo, "rev-parse", "HEAD")


def _seed_local_origin(tmp_path: Path, fake_gh: FakeGh) -> tuple[str, str, str]:
    """Build a real local bare origin whose base/head are the PR's SHAs.

    Uses the deterministic seed identity (same content/env as the snapshot
    module's ``_seed_origin``), producing real base2/head commits. The feature
    head is pushed as ``refs/pull/101/head`` and the canned PR 101 header is
    re-seeded so its ``base.sha``/``head.sha`` match the origin — the
    import-time freeze fetches real git objects (no network).

    Returns ``(origin_url, base_sha, head_sha)``.
    """
    import shutil as _sh

    repo = tmp_path / "local_wt"
    if repo.exists():
        _sh.rmtree(repo)
    repo.mkdir()
    _seed_git(repo, "init", "-b", "main")
    _seed_write(repo, "readme.txt", "base1\n")
    _seed_commit(repo, "base1")
    _seed_write(repo, "base.py", "BASE = 2\n")
    base_sha = _seed_commit(repo, "base2")
    _seed_write(repo, "beyond.py", "BEYOND = 3\n")
    _seed_commit(repo, "base3")
    _seed_git(repo, "checkout", "--detach", base_sha)
    (repo / "base.py").write_text("BASE = 20\n")
    _seed_git(repo, "add", "base.py")
    _seed_write(repo, "feature.py", "FEATURE = 1\n")
    head_sha = _seed_commit(repo, "feature")
    bare = tmp_path / "origin_local.git"
    if bare.exists():
        _sh.rmtree(bare)
    bare.mkdir()
    _seed_git(bare, "init", "--bare")
    _seed_git(repo, "remote", "add", "origin", str(bare))
    _seed_git(repo, "push", "origin", "main:main")
    _seed_git(repo, "push", "origin", f"{head_sha}:refs/pull/101/head", check=False)
    # Re-seed the canned PR header so base.sha/head.sha are the real origin SHAs.
    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_sha}
    header["head"] = {"ref": "feature/cache", "sha": head_sha}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    return str(bare), base_sha, head_sha


def test_import_freezes_cases_ready_with_bundle(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict, sha256_file
    from daydream.benchmark.workspace import init_workspace

    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["api.anthropic.com"], ["api.anthropic.com"])
    _seed_preflight(ws, fake_gh)                 # identity + canned PR
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh)
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url)
    assert rc == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    pr = raw["pull_requests"][0]
    assert pr["import_state"] == "fetched"
    case_id = pr["case_ids"][0]
    case = load_yaml_strict(ws / f"cases/{case_id}.yaml")
    assert case["snapshot"]["status"] == "ready"
    assert case["snapshot"]["original_base_sha"] == base_sha
    assert case["snapshot"]["requested_base_sha"] == base_sha
    assert case["snapshot"]["original_head_sha"] == head_sha
    bundle = ws / case["snapshot"]["bundle_file"]
    assert bundle.exists()
    assert sha256_file(bundle) == case["snapshot"]["bundle_sha256"]


def test_e2e_import_distinct_idempotent_explicit_head_and_shared_mirror(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Same PR via import is idempotent; a distinct explicit head is a new case.

    Also proves one shared ``cache/repository.git`` serves both without ref
    collision.
    """
    from daydream.benchmark import github_import as gi
    from daydream.benchmark import snapshot as sn
    from daydream.benchmark.storage import load_yaml_strict
    from daydream.benchmark.workspace import init_workspace

    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["api.anthropic.com"], ["api.anthropic.com"])
    _seed_preflight(ws, fake_gh)
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh)
    # default head only first
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    ids1 = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]["case_ids"]
    # same PR + same head again -> same idempotent case
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    ids2 = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]["case_ids"]
    assert ids1 == ids2
    # a distinct head (unreachable in this origin) -> a distinct case id
    alt_head = "cdef" * 10
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[alt_head], origin_url=origin_url) == 0
    ids3 = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]["case_ids"]
    assert len(ids3) == len(ids1) + 1 and ids3[-1].endswith(alt_head[:12])
    # one shared mirror, no ref collision: the PR-head ref still resolves to head
    assert (ws / "cache" / "repository.git").exists()
    assert sn.rev_parse(ws / "cache/repository.git", "refs/pull/101/head") == head_sha


def test_import_writes_atomic_unit_and_no_file_on_failure(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict, sha256_file

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)  # preflight + rest/graphql canned data for pr 101 (one head)
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None)
    assert rc == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    pr = raw["pull_requests"][0]
    assert pr["import_state"] == "fetched"
    assert pr["import_file"] == "imports/pr-000101.json"
    assert pr["import_sha256"] == sha256_file(ws / pr["import_file"])
    assert pr["requested_heads"] == ["final"]
    assert pr["case_ids"] == ["pr-000101-" + "a" * 12]   # head from _PR_HEADER
    assert (ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml").exists()


def test_failed_fetch_leaves_no_import_file_and_ledger_error(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh, pull_header=None)  # 404 -> fetch fails
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None)
    assert rc != 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    pr = raw["pull_requests"][0]
    assert pr["import_state"] == "fetch_failed"
    assert pr["error"]["code"] and pr["error"]["message"]
    assert pr["import_file"] is None and pr["import_sha256"] is None
    assert not (ws / "imports" / "pr-000101.json").exists()


def test_status_reflects_fetched_import_and_resolved_identity(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.workspace import workspace_status

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None)
    assert rc == 0
    st = workspace_status(ws)
    assert st.workspace_state != "empty"
    assert st.repository_identity_resolved is True


def test_cli_import_prs_drives_command(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]) -> None:
    from daydream.benchmark.cli import _handle_benchmark_command
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    rc = _handle_benchmark_command(["import-prs", str(ws), "--pr", "101", "--head", "a" * 40])
    assert rc == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["pull_requests"][0]["import_state"] == "fetched"
    out = capsys.readouterr().out
    assert "octocat" in out and "private" in out        # preflight print: identity + visibility
    assert "1" in out                                        # requested PR count
    assert str(ws / "imports") in out                      # local destination


def _curate_case(ws: Path, case_file: Any) -> None:
    """Mark a materialized case ready + attested with one historical finding."""
    import yaml

    from daydream.benchmark.schema import derive_finding_id
    from daydream.benchmark.storage import load_yaml_strict

    path = ws / "cases" / case_file
    raw = load_yaml_strict(path)
    finding = {
        "title": "bot asks to fix the cache",
        "body": "please fix",
        "severity": "low",
        "location": {"path": "a.py", "start_line": 4, "end_line": 4},
        "provenance": {"kind": "historical", "source_ids": ["github:inline_comment:1"]},
    }
    finding["finding_id"] = derive_finding_id(finding, case_id=raw["case_id"])
    raw["curation"] = {
        "state": "ready",
        "snapshot_attested": True,
        "clean_attested": False,
        "gold_status": "findings",
        "findings": [finding],
        "exclusions": [],
        "case_exclusion": None,
        "task_spec_sha256": "d" * 64,
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def test_refresh_body_only_change_stales_gold(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # state=ready, attested
    # body-only change: same evidence, edited PR body (feeds compiled context)
    hdr = dict(_PR_HEADER)
    hdr["body"] = "EDITED body that changes compiled context"
    _seed_preflight(ws, fake_gh, pull_header=hdr)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale"        # task-input contract changed -> stale


def test_refresh_metadata_only_change_updates_checksums_without_staling(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    before = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    before_import_sha = before["source"]["import_sha256"]
    # metadata-only change: updated_at + html_url, same title/body/base/head (no evidence change)
    hdr = dict(_PR_HEADER)
    hdr["updated_at"] = "2026-01-02T00:00:00Z"
    hdr["html_url"] = "https://github.com/o/r/pull/101"
    _seed_preflight(ws, fake_gh, pull_header=hdr)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"          # NOT staled
    assert case["source"]["import_sha256"] != before_import_sha   # import checksum updated
    assert case["curation"]["findings"]                  # curated gold preserved
    # The refreshed header metadata must actually propagate into the case-level
    # pull_request block; import_sha256 alone cannot prove it, since the digest
    # re-serializes fetch.fetched_at and flips on any refresh.
    assert case["pull_request"]["updated_at"] == "2026-01-02T00:00:00Z"
    assert case["pull_request"]["html_url"] == "https://github.com/o/r/pull/101"


def test_refresh_predate_import_metadata_change_does_not_stale(tmp_path: Path, fake_gh: FakeGh) -> None:
    """A predate import file (no body, no head.ref) must not stale gold on the
    first post-upgrade refresh: its task-input contract cannot be reconstructed,
    so only an evidence change can stale it until it is re-persisted."""
    import json

    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_json_strict, load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    # Rewrite the persisted import in the predate shape: head.ref dropped and the
    # additive body/digest/html_url/merged/closed fields absent.
    import_path = ws / "imports" / "pr-000101.json"
    raw = load_json_strict(import_path)
    pr = raw["pull_request"]
    pr.pop("body", None)
    pr.pop("html_url", None)
    pr.pop("title_sha256", None)
    pr.pop("body_sha256", None)
    pr.pop("merged_at", None)
    pr.pop("closed_at", None)
    pr["head"].pop("ref", None)
    import_path.write_text(json.dumps(raw, indent=2))
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    before = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    before_import_sha = before["source"]["import_sha256"]
    # metadata-only change: same title/body/base/head and evidence as the predate file
    hdr = dict(_PR_HEADER)
    hdr["updated_at"] = "2026-01-02T00:00:00Z"
    _seed_preflight(ws, fake_gh, pull_header=hdr)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"          # NOT staled by the metadata-only refresh
    assert case["source"]["import_sha256"] != before_import_sha   # import checksum updated
    assert case["curation"]["findings"]                  # curated gold preserved


def test_refresh_predate_canonical_format_drift_does_not_stale(tmp_path: Path, fake_gh: FakeGh) -> None:
    """A first ``--refresh`` after the canonical-record format change must NOT
    flip prior curated cases stale on pure format drift. Pre-canonicalization
    files persisted a comment that existed in both feeds twice (REST
    ``inline_comment`` + GraphQL ``thread_comment``) and thread-only comments
    under the ``thread_comment`` kind; the canonical format emits exactly one
    ``inline_comment`` per database id. With byte-identical GitHub content the
    only difference is the persisted shape, so the (database_id-keyed) evidence
    signature must compare equal and keep the curated case ready."""
    import json

    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_json_strict, load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    fake_gh.set_response(
        "GET",
        "repos/o/r/pulls/101/comments",
        [
            {"id": 1, "node_id": "DIFF_1", "user": {"login": "alice", "type": "User"},
             "body": "please fix", "commit_id": "a" * 40, "original_commit_id": "a" * 40,
             "path": "a.py", "line": 4, "subject_type": "line", "side": "RIGHT",
             "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#discussion_r1"},
        ],
    )
    fake_gh._write_threads(
        [
            {"id": "thread_1", "isResolved": False, "isOutdated": False,
             "subjectType": "LINE", "path": "a.py", "line": 4, "side": "RIGHT",
             "comments": {"nodes": [
                 {"id": "c1", "databaseId": 1, "body": "please fix",
                  "author": {"login": "alice", "type": "User"},
                  "createdAt": "2026-01-01T00:00:00Z",
                  "url": "https://github.com/o/r/pull/101#discussion_r1"},
                 {"id": "c2", "databaseId": 2, "body": "thread-only",
                  "author": {"login": "alice", "type": "User"},
                  "createdAt": "2026-01-01T00:00:00Z",
                  "url": "https://github.com/o/r/pull/101#discussion_r2"},
             ]}},
        ],
        number=101,
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")       # state=ready, attested
    # Rewrite the persisted import in the pre-canonicalization shape: the comment
    # that existed in both feeds (db 1) is stored twice (inline + thread_comment)
    # and the thread-only comment (db 2) under the thread_comment kind.
    import_path = ws / "imports" / "pr-000101.json"
    raw = load_json_strict(import_path)
    old_evidence: list[dict[str, Any]] = []
    for e in raw["evidence"]:
        if e.get("thread_id"):
            if e.get("commit_id"):
                old_evidence.append({**e,
                                     "source_id": f"github:inline_comment:{e['database_id']}",
                                     "kind": "inline_comment"})
            # The thread feed never carried commit anchors, so the pre-canonical
            # thread_comment copy has no commit_id/original_commit_id: the two
            # copies of db 1 must project DIFFERENTLY.  Spreading e verbatim
            # would make them hash identically and collapse onto one element,
            # under-modeling the drift (and hiding the spurious stale it used to
            # trigger on the first post-format refresh).
            old_evidence.append({
                **{k: v for k, v in e.items() if k not in ("commit_id", "original_commit_id")},
                "source_id": f"github:thread_comment:{e['database_id']}",
                "kind": "thread_comment"})
        else:
            old_evidence.append(e)
    assert len(old_evidence) == 3      # db 1 twice, db 2 once as thread_comment
    import_path.write_text(json.dumps({**raw, "evidence": old_evidence}, indent=2))
    # refresh with IDENTICAL GitHub content: only the persisted format changed
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"           # format drift must NOT stale gold
    assert case["curation"]["findings"]                   # curated gold preserved
    # the migration path rewrites the file to the canonical shape: exactly one
    # inline_comment per database id
    refreshed = load_json_strict(import_path)
    assert len(refreshed["evidence"]) == 2
    assert {e["kind"] for e in refreshed["evidence"]} == {"inline_comment"}


def test_refresh_marks_stale_and_never_overwrites_curation(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)   # seed REST with one evidence record via the comment fixture below
    fake_gh.set_response(
        "GET",
        "repos/o/r/pulls/101/comments",
        [
            {"id": 1, "node_id": "DIFF_1", "user": {"login": "bot[bot]", "type": "Bot"},
             "body": "please fix", "commit_id": "a" * 40, "original_commit_id": "a" * 40,
             "path": "a.py", "line": 4, "subject_type": "line", "side": "RIGHT",
             "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#discussion_r1"},
        ],
    )
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None)
    assert rc == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")   # curation.state=ready, snapshot_attested=True
    # refresh re-fetches; the referenced evidence now disappears
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None)
    assert rc == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale" and case["curation"]["snapshot_attested"] is False
    assert case["curation"]["findings"]              # prior curated findings preserved


def _pr_header(number: int) -> dict[str, Any]:
    header = dict(_PR_HEADER)
    header["number"] = number
    header["url"] = f"https://github.com/o/r/pull/{number}"
    return header


def _seed_identity(fake_gh: FakeGh) -> None:
    """Seed the preflight identity + repo-access responses (no PR data)."""
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))


def _seed_rest(gh: Any, number: int, *, reviews: Any, comments: Any, issue_comments: Any) -> None:
    """Seed one PR's REST evidence into the fake router."""
    gh.set_response("GET", f"repos/o/r/pulls/{number}", _pr_header(number))
    gh.set_response("GET", f"repos/o/r/pulls/{number}/reviews", reviews)
    gh.set_response("GET", f"repos/o/r/pulls/{number}/comments", comments)
    gh.set_response("GET", f"repos/o/r/issues/{number}/comments", issue_comments)


def test_e2e_paginated_human_bot_evidence_and_no_comment_pr(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark.cli import _handle_benchmark_command
    from daydream.benchmark.storage import load_json_strict, load_yaml_strict

    ws = tmp_path / "ws"
    _seed_manifest(ws)
    _seed_identity(fake_gh)
    _seed_rest(
        fake_gh, 101,
        reviews=[
            {"id": 1, "node_id": "PRR_1", "user": {"login": "cr[bot]", "type": "Bot"},
             "body": "Found a bug.", "state": "COMMENTED", "commit_id": "a" * 40,
             "submitted_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#pullrequestreview-1"},
            {"id": 2, "node_id": "PRR_2", "user": {"login": "carol", "type": "User"},
             "body": "Nice work.", "state": "COMMENTED", "commit_id": "a" * 40,
             "submitted_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#pullrequestreview-2"},
        ],
        comments=[
            {"id": 7, "node_id": "DIFF_7", "user": {"login": "bot[bot]", "type": "Bot"},
             "body": "please fix the cache", "path": "a.py", "line": 4,
             "subject_type": "line", "side": "RIGHT", "commit_id": "a" * 40,
             "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#discussion_r7"},
            {"id": 8, "node_id": "DIFF_8", "user": {"login": "dave", "type": "User"},
             "body": "Order matters here.", "path": "b.py", "line": 2,
             "subject_type": "line", "side": "RIGHT", "commit_id": "a" * 40,
             "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#discussion_r8"},
        ],
        issue_comments=[
            {"id": 9, "node_id": "IC_9", "user": {"login": "carol", "type": "User"},
             "body": "question", "created_at": "2026-01-01T00:00:00Z",
             "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#issuecomment-9"},
        ],
    )
    fake_gh._write_threads(
        [
            {"id": "thread_1", "isResolved": False, "isOutdated": False,
             "subjectType": "LINE", "path": "a.py", "line": 4, "side": "RIGHT",
             "comments": {"nodes": [
                 {"id": "c1", "databaseId": 10, "body": "root",
                  "author": {"login": "dave", "type": "User"},
                  "createdAt": "2026-01-01T00:00:00Z",
                  "url": "https://github.com/o/r/pull/101#discussion_r10"},
             ]}},
        ],
        number=101,
    )
    _seed_rest(fake_gh, 102, reviews=[], comments=[], issue_comments=[])
    rc = _handle_benchmark_command(
        ["import-prs", str(ws), "--pr", "101", "--pr", "102", "--pr", "https://github.com/o/r/pull/102"]
    )
    assert rc == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert [p["number"] for p in raw["pull_requests"]] == [101, 102]
    imp = load_json_strict(ws / "imports/pr-000101.json")
    kinds = {e["kind"] for e in imp["evidence"]}
    assert kinds == {"review", "inline_comment", "issue_comment"}
    # overlapping root (db 10) appears exactly once as a canonical inline_comment
    assert len([e for e in imp["evidence"] if e["database_id"] == 10]) == 1
    db10 = next(e for e in imp["evidence"] if e["database_id"] == 10)
    assert db10["kind"] == "inline_comment" and db10["source_id"] == "github:inline_comment:10"
    assert db10["thread_id"] == "thread_1"
    assert any(e["is_bot"] for e in imp["evidence"])      # bot author retained
    assert any(not e["is_bot"] for e in imp["evidence"])  # human author retained
    assert load_json_strict(ws / "imports/pr-000102.json")["evidence"] == []  # no-comment PR retained


def test_e2e_partial_failure_persists_ledger_and_exits_nonzero(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark.cli import _handle_benchmark_command
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_manifest(ws)
    _seed_identity(fake_gh)
    _seed_rest(fake_gh, 101, reviews=[], comments=[], issue_comments=[])
    fake_gh.set_response(
        "GET", "repos/o/r/pulls/102", {"__error__": "API rate limit exceeded Retry-After: 1"}
    )
    rc = _handle_benchmark_command(["import-prs", str(ws), "--pr", "101", "--pr", "102"])
    assert rc != 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    by_n = {p["number"]: p for p in raw["pull_requests"]}
    assert by_n[101]["import_state"] == "fetched"
    assert by_n[102]["import_state"] == "fetch_failed"
    assert by_n[102]["error"]["code"] == "rate_limit"
    assert (ws / "imports/pr-000101.json").exists()
    assert not (ws / "imports/pr-000102.json").exists()   # failed fetch: no import file


def test_benchmark_help_lists_import_prs() -> None:
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "daydream", "benchmark", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "import-prs" in r.stdout
def test_reimport_does_not_duplicate_cases_rows(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Re-importing the same PR (unchanged evidence) must not duplicate cases[] rows."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    raw1 = load_yaml_strict(ws / "benchmark.yaml")
    assert len(raw1["cases"]) == 1
    # re-import same PR, unchanged evidence (responses not changed) — fetched->fetched allowed
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    raw2 = load_yaml_strict(ws / "benchmark.yaml")
    ids = [c["case_id"] for c in raw2["cases"]]
    assert len(raw2["cases"]) == 1, f"cases[] grew to {len(raw2['cases'])}: {ids}"
    assert ids[0] == "pr-000101-aaaaaaaaaaaa"


def test_reimport_unchanged_evidence_preserves_curation(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Re-import with unchanged evidence must not wipe curated findings/attestation."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case_file = "pr-000101-aaaaaaaaaaaa.yaml"
    _curate_case(ws, case_file)  # state=ready, snapshot_attested=True, findings non-empty
    before = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert before["state"] == "ready" and before["snapshot_attested"] is True and before["findings"]
    # Re-import same PR WITHOUT refresh, unchanged evidence.
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    after = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert after["state"] in ("ready", "stale"), "curation must not reset to draft"
    assert after["findings"], "curated findings must not be wiped"
    assert after["snapshot_attested"] is True, "unchanged re-import must keep attestation"


def test_refresh_unchanged_signature_preserves_curation(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Refresh with an UNCHANGED evidence signature must keep curated findings."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case_file = "pr-000101-aaaaaaaaaaaa.yaml"
    _curate_case(ws, case_file)
    # refresh with IDENTICAL evidence responses (signature unchanged) -> changed=False
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    after = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert after["findings"], "curated findings must not be wiped on unchanged refresh"
    assert after["state"] != "draft", "curation must not reset to draft on unchanged refresh"


def test_graphql_review_threads_retries_rate_limit_then_fails(
    tmp_path: Path,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphQL reviewThreads honors the rate-limit retry policy (3x)."""
    import daydream.benchmark.github_import as gi_mod
    from daydream.git_ops import RateLimitError

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)

    calls = {"n": 0}

    def flaky_gh_api(*a: Any, **kw: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("graphql rate limited", retry_after=0.0)
        ok = {"repository": {"pullRequest": {"reviewThreads": {"nodes": [], "pageInfo": {"hasNextPage": False}}}}}
        return {"data": ok}

    monkeypatch.setattr("daydream.git_ops.gh_api", flaky_gh_api)
    monkeypatch.setattr(gi_mod, "time", type("_T", (), {"sleep": staticmethod(lambda _s: None)})())
    threads = gi_mod._graphql_review_threads(ws, "o/r", 101)
    assert threads == []
    assert calls["n"] == 3, "rate-limit retry should make 3 attempts"


def test_spike_nested_comments_paginate_past_100(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    for ep, val in [("repos/o/r/pulls/101/reviews", list[Any]()),
                    ("repos/o/r/pulls/101/comments", list[Any]()),
                    ("repos/o/r/issues/101/comments", list[Any]())]:
        fake_gh.set_response("GET", ep, val)
    # thread "thread_1" carries 150 replies split across nested pages of 100+50
    comments = [{"id": f"c{i}", "databaseId": 1000 + i, "body": f"r{i}",
                 "author": {"login": "dave", "type": "User"},
                 "createdAt": f"2026-01-01T00:00:{i % 60:02d}Z",
                 "url": f"https://github.com/o/r/pull/101#discussion_r{1000+i}"}
                for i in range(1, 151)]
    fake_gh._serve_thread_comments("thread_1", comments, page_size=100)
    fake_gh._write_threads([{"id": "thread_1", "isResolved": False,
        "isOutdated": False, "subjectType": "LINE", "path": "a.py", "line": 4,
        "side": "RIGHT", "comments": {"nodes": comments[:100]}}], number=101)
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    ids = {e.database_id for e in doc.evidence}
    assert {1000 + i for i in range(1, 151)} <= ids     # all 150 collected
    assert len([e for e in doc.evidence if 1000 <= e.database_id <= 1150]) == 150


def test_graphql_threads_replies_collect_past_100(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    for ep in ("repos/o/r/pulls/101/reviews", "repos/o/r/pulls/101/comments",
               "repos/o/r/issues/101/comments"):
        fake_gh.set_response("GET", ep, [])
    comments = [{"id": f"c{i}", "databaseId": 2000 + i, "body": f"reply {i}",
                 "author": {"login": "eve", "type": "User"},
                 "createdAt": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
                 "replyTo": {"id": "c1"},
                 "url": f"https://github.com/o/r/pull/101#discussion_r{2000+i}"}
                for i in range(1, 251)]     # 250 replies -> 3 nested pages
    fake_gh._serve_thread_comments("thread_9", comments, page_size=100)
    fake_gh._write_threads([{"id": "thread_9", "isResolved": False,
        "isOutdated": False, "subjectType": "LINE", "path": "a.py", "line": 4,
        "side": "RIGHT", "comments": {"nodes": comments[:100]}}], number=101)
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    replies = [e for e in doc.evidence if 2000 <= e.database_id <= 2250]
    assert len(replies) == 250
    assert len({e.database_id for e in replies}) == 250        # no dup
    # deterministic creation order preserved end to end
    assert [e.database_id for e in sorted(replies, key=lambda r: r.database_id)] \
           == sorted(range(2001, 2251))


def test_reconcile_inline_and_thread_into_one_record(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    # review id 5 with state DISMISSED (dismissal source for comment 10)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [
        {"id": 5, "node_id": "PRR_5", "user": {"login": "alice", "type": "User"},
         "body": "", "state": "DISMISSED", "commit_id": "a" * 40,
         "submitted_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#pullrequestreview-5"}])
    # REST inline comment 10 is the root of thread_1, belongs to review 5
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [
        {"id": 10, "node_id": "DIFF_10", "user": {"login": "dave", "type": "User"},
         "body": "root", "commit_id": "a" * 40, "original_commit_id": "a" * 40,
         "path": "a.py", "original_path": "a.py", "line": 4, "original_line": 3,
         "subject_type": "line", "side": "RIGHT",
         "pull_request_review_id": 5,
         "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#discussion_r10"}])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([{"id": "thread_1", "isResolved": True,
        "isOutdated": True, "subjectType": "LINE",
        "path": "a.py", "line": 4, "originalLine": 3, "side": "RIGHT",
        "startSide": None,
        "comments": {"nodes": [
            {"id": "c1", "databaseId": 10, "body": "root",
             "author": {"login": "dave", "type": "User"},
             "createdAt": "2026-01-01T00:00:00Z",
             "url": "https://github.com/o/r/pull/101#discussion_r10"}]}}],
        number=101)
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    by_db = {e.database_id: e for e in doc.evidence}
    rec = by_db[10]
    assert rec.kind == "inline_comment"
    assert rec.source_id == "github:inline_comment:10"
    assert rec.thread_id == "thread_1" and rec.resolved is True
    assert rec.outdated is True and rec.dismissed is True      # via review 5 DISMISSED
    assert rec.review_id == "5"
    assert rec.commit_id == "a" * 40 and rec.path == "a.py"     # REST anchors kept
    # exactly one record with database_id 10, no thread_comment kind anywhere
    assert len([e for e in doc.evidence if e.database_id == 10]) == 1
    assert not any(e.kind == "thread_comment" for e in doc.evidence)


def test_evidence_order_deterministic_across_page_sizes(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    # REST-only comments with distinct database ids + timestamps
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [
        {"id": 1, "node_id": "PRR_1", "user": {"login": "alice", "type": "User"},
         "body": "approved", "state": "APPROVED", "commit_id": "a" * 40,
         "submitted_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#pullrequestreview-1"}])
    comments = [
        {"id": 30, "node_id": "DIFF_30", "user": {"login": "dave", "type": "User"},
         "body": "first", "commit_id": "a" * 40, "path": "a.py", "line": 1,
         "subject_type": "line", "side": "RIGHT",
         "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#discussion_r30"},
        {"id": 7, "node_id": "DIFF_7", "user": {"login": "carol", "type": "User"},
         "body": "second", "commit_id": "a" * 40, "path": "b.py", "line": 2,
         "subject_type": "line", "side": "RIGHT",
         "created_at": "2026-01-02T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#discussion_r7"},
    ]
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", comments)
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([], number=101)
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    # deterministic order: sorted by (database_id, created_at)
    assert [e.database_id for e in doc.evidence] == [1, 7, 30]
    payload = doc.fetch.payload_sha256
    doc2 = gi.fetch_and_normalize(ws, "o/r", 101)     # refetch: identical digest
    assert doc2.fetch.payload_sha256 == payload


def test_outdated_root_not_exact_acceptable_via_joined_record(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    # REST copy of comment 40 is OUTDATED via the joined GraphQL thread state
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [
        {"id": 40, "node_id": "DIFF_40", "user": {"login": "dave", "type": "User"},
         "body": "outdated root", "commit_id": "a" * 40, "path": "a.py", "line": 5,
         "subject_type": "line", "side": "RIGHT",
         "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#discussion_r40"}])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([{"id": "thread_2", "isResolved": True,
        "isOutdated": True, "subjectType": "LINE",
        "path": "a.py", "line": 5, "originalLine": 4, "side": "RIGHT",
        "startSide": None,
        "comments": {"nodes": [
            {"id": "c40", "databaseId": 40, "body": "outdated root",
             "author": {"login": "dave", "type": "User"},
             "createdAt": "2026-01-01T00:00:00Z",
             "url": "https://github.com/o/r/pull/101#discussion_r40"}]}}],
        number=101)
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    cands = {c.source_id: c for c in gi.project_candidates(doc, head_sha="a" * 40)}
    cand = cands["github:inline_comment:40"]
    assert cand.exact_acceptable is False
    assert cand.not_exact_reason == "outdated"


def test_fixture_matrix_evidence_preserved_and_historical(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi

    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [
        {"id": 1, "node_id": "PRR_1", "user": {"login": "cr[bot]", "type": "Bot"},
         "body": "Found a bug.", "state": "COMMENTED", "commit_id": "a" * 40,
         "submitted_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#pullrequestreview-1"},   # non-pure review body
        {"id": 2, "node_id": "PRR_2", "user": {"login": "carol", "type": "User"},
         "body": "Nice work.", "state": "APPROVED", "commit_id": "a" * 40,
         "submitted_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#pullrequestreview-2"},   # pure approval
    ])
    # edited comment: updated_at != created_at
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [
        {"id": 7, "node_id": "DIFF_7", "user": {"login": "bot[bot]", "type": "Bot"},
         "body": "please fix", "commit_id": "a" * 40, "path": "a.py", "line": 4,
         "subject_type": "line", "side": "RIGHT",
         "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-03T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#discussion_r7"}])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [
        {"id": 9, "node_id": "IC_9", "user": {"login": "carol", "type": "User"},
         "body": "question", "created_at": "2026-01-01T00:00:00Z",
         "updated_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#issuecomment-9"}])
    fake_gh._write_threads([], number=101)
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    kinds = {e.kind for e in doc.evidence}
    assert kinds == {"review", "inline_comment", "issue_comment"}   # nothing dropped
    assert any(e.is_bot for e in doc.evidence)                        # bot actor retained
    assert any(not e.is_bot for e in doc.evidence)                    # human actor retained
    edited = next(e for e in doc.evidence if e.database_id == 7)
    assert edited.updated_at > edited.created_at                      # edit metadata preserved
    by_src = {e.source_id: e for e in doc.evidence}
    assert by_src["github:review:1"].state == "COMMENTED"             # non-pure review body retained
    assert by_src["github:review:2"].state == "APPROVED"              # pure approval retained as evidence
    cands = {c.source_id for c in gi.project_candidates(doc, head_sha="a" * 40)}
    assert "github:inline_comment:7" in cands                          # root comment is a candidate
    assert "github:review:2" not in cands                              # pure approval: evidence only


def test_graphql_review_threads_records_rate_limit_after_retries(
    tmp_path: Path,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausting GraphQL rate-limit retries surfaces _ImportRateLimitError (ledger rate_limit)."""
    import pytest

    import daydream.benchmark.github_import as gi_mod
    from daydream.git_ops import RateLimitError

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)

    def always_limited(*a: Any, **kw: Any) -> None:
        raise RateLimitError("graphql rate limited", retry_after=0.0)

    monkeypatch.setattr("daydream.git_ops.gh_api", always_limited)
    monkeypatch.setattr(gi_mod, "time", type("_T", (), {"sleep": staticmethod(lambda _s: None)})())
    with pytest.raises(gi_mod._ImportRateLimitError):
        gi_mod._graphql_review_threads(ws, "o/r", 101)


def test_corrupt_prior_import_fails_before_network(tmp_path: Path, fake_gh: FakeGh) -> None:
    # Seed a fetched ledger entry + a corrupt prior import, then refresh.
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, head = _seed_ready_case(tmp_path, fake_gh)     # valid prior state
    imp = next((ws / "imports").glob("*.json"))
    imp.write_bytes(b"{ corrupt json !!")                       # corrupt the persisted import
    with pytest.raises(WorkspaceCorrupt):                       # must fail, not heal to None
        gi._prior_import_state(ws, load_yaml_strict(ws / "benchmark.yaml"), 101)


def test_corrupt_prior_curation_fails_not_healed(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, head = _seed_ready_case(tmp_path, fake_gh)
    case = ws / "cases" / f"{case_id}.yaml"
    case.write_bytes(b"{ corrupt yaml !!")   # corruption the strict loader rejects
    with pytest.raises(WorkspaceCorrupt):
        gi._prior_import_state(ws, load_yaml_strict(ws / "benchmark.yaml"), 101)


def test_missing_prior_import_is_nonfatal_first_run(tmp_path: Path) -> None:
    # A never-imported PR (no ledger fetch) must not fail on prior-state discovery.
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict
    from daydream.benchmark.workspace import init_workspace

    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    raw = load_yaml_strict(ws / "benchmark.yaml")
    (
        prior_sig,
        prior_task_sig,
        curations,
        _,
        prior_pinned,
        prior_policy,
        prior_heads,
    ) = gi._prior_import_state(ws, raw, 202)
    assert prior_sig is None and prior_task_sig is None and curations == {}
    assert prior_pinned == {} and prior_policy == {} and prior_heads == []


def test_refresh_stale_clears_task_spec_approval(tmp_path: Path, fake_gh: FakeGh) -> None:
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict
    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)   # seed REST with one evidence comment
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [{"id": 1, "node_id": "DIFF_1", "user": {"login": "bot[bot]", "type": "Bot"},
          "body": "please fix", "commit_id": "a" * 40, "original_commit_id": "a" * 40,
          "path": "a.py", "line": 4, "subject_type": "line", "side": "RIGHT",
          "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
          "html_url": "https://github.com/o/r/pull/101#discussion_r1"}])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")   # ready + attested (with digest, per Task 4)
    # refresh: the referenced evidence disappears -> case flips stale
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale" and case["curation"]["snapshot_attested"] is False
    assert "task_spec_sha256" not in case["curation"] and "task_spec_approved_at" not in case["curation"]


# ---------------------------------------------------------------------------
# widened evidence signature (issue #813): projection-relevant provenance keyed
# per physical database_id; kind/source_id/url/timestamps excluded
# ---------------------------------------------------------------------------


def _one_evidence() -> dict[str, Any]:
    """One canonical inline-comment evidence record dict (projection fields)."""
    return {"database_id": 1, "body_sha256": "a" * 64, "body": "please fix",
            "path": "a.py", "line": 4, "commit_id": "a" * 40, "outdated": False,
            "resolved": False, "dismissed": False, "state": "COMMENTED",
            "subject_type": "line", "side": "RIGHT", "start_side": None,
            "original_path": "a.py", "original_line": 4, "original_commit_id": "a" * 40,
            "author": {"login": "alice", "type": "User"}}


def _sig(ev: dict[str, Any]) -> Any:
    """Signature over one raw evidence record: ``{"evidence": [ev]}`` wrapper."""
    from daydream.benchmark import github_import as gi

    return gi._evidence_signature_from_raw({"evidence": [ev]})


def test_signature_changes_on_anchor_move() -> None:
    base = _one_evidence()
    moved = {**base, "line": 7}                     # same body, moved anchor
    assert _sig(base) != _sig(moved)


def test_signature_changes_on_resolution_state() -> None:
    base = _one_evidence()
    assert _sig(base) != _sig({**base, "resolved": True})
    assert _sig(base) != _sig({**base, "outdated": True})
    assert _sig(base) != _sig({**base, "dismissed": True})
    assert _sig(base) != _sig({**base, "commit_id": "b" * 40})
    assert _sig(base) != _sig({**base, "author": {"login": "bob", "type": "User"}})


def test_signature_ignores_metadata_only_change() -> None:
    base = _one_evidence()
    meta = {**base, "updated_at": "2026-01-02T00:00:00Z", "url": "https://e.example/2"}
    assert _sig(base) == _sig(meta)


def test_signature_ignores_format_drift_duplicate_and_kind() -> None:
    from daydream.benchmark import github_import as gi

    base = _one_evidence()
    dup = [{**base, "kind": "inline_comment"},      # same database_id stored twice
           {**base, "kind": "thread_comment"}]
    canon = [base]
    assert gi._evidence_signature_from_raw({"evidence": dup}) \
        == gi._evidence_signature_from_raw({"evidence": canon})


# ---------------------------------------------------------------------------
# per-case staleness via reference intersection (issue #813): a case stales
# only when its own referenced evidence changed (or the PR-wide task input)
# ---------------------------------------------------------------------------


def _seed_discussion(db_id: int, body: str = "please fix", line: int = 4) -> dict[str, Any]:
    """One canonical REST inline-comment dict for the fake router."""
    return {"id": db_id, "node_id": f"DIFF_{db_id}", "user": {"login": "bot[bot]", "type": "Bot"},
            "body": body, "commit_id": "a" * 40, "original_commit_id": "a" * 40,
            "path": "a.py", "line": line, "subject_type": "line", "side": "RIGHT",
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
            "html_url": f"https://github.com/o/r/pull/101#discussion_r{db_id}"}


def test_refresh_unrelated_new_comment_does_not_stale(tmp_path: Path, fake_gh: FakeGh) -> None:
    # PR 101 imported with one referenced comment (db 1) and curated ready; a NEW
    # unrelated comment (db 99) must not stale the referenced case on refresh.
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_seed_discussion(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1
    fake_gh.set_response(
        "GET", "repos/o/r/pulls/101/comments",
        [_seed_discussion(1), {**_seed_discussion(99), "path": "b.py", "body": "unrelated nit"}],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"       # NOT staled by an unrelated new comment
    assert case["curation"]["findings"]


def test_refresh_changed_anchor_on_referenced_evidence_stales(tmp_path: Path, fake_gh: FakeGh) -> None:
    # Same body, moved anchor on the REFERENCED comment (db 1) -> the case stales
    # while its curated findings stay preserved.
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_seed_discussion(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_seed_discussion(1, line=7)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale"
    assert case["curation"]["findings"]               # curated findings preserved
    assert case["curation"]["snapshot_attested"] is False


# ---------------------------------------------------------------------------
# head-immutability (issue #813): an existing case resolves to its pinned head,
# so a live head advance reproduces the same case_id with no orphan
# ---------------------------------------------------------------------------


def test_refresh_after_head_advance_keeps_case_id(tmp_path: Path, fake_gh: FakeGh) -> None:
    # Import + curate PR 101 at head a*40, then change the live head to b*40
    # (the branch advanced) and refresh: the SAME case_id is reproduced, no
    # new case, no orphan, and the untouched pinned case stays ready.
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    hdr = dict(_PR_HEADER)
    hdr["head"] = {"ref": "feature/cache", "sha": "b" * 40}   # live head now advanced (valid 40-hex)
    _seed_preflight(ws, fake_gh, pull_header=hdr)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    ids = [c["case_id"] for c in raw["cases"]]
    assert ids == ["pr-000101-aaaaaaaaaaaa"]          # pinned, not advanced to b*40
    assert (ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml").exists()
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"       # unchanged evidence -> stays ready


# ---------------------------------------------------------------------------
# non-destructive failed refresh (issue #813): a failed refresh on an already-
# fetched PR preserves last-good linkage and records the attempt in latest_error
# ---------------------------------------------------------------------------


def test_refresh_failure_preserves_linkage_and_records_attempt(tmp_path: Path, fake_gh: FakeGh) -> None:
    # Import + curate PR 101 successfully, then make the refresh fetch fail: the
    # last-good import_file/import_sha256/case_ids are preserved and the attempt
    # is recorded separately in latest_error (NOT reset to fetch_failed).
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    before = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]
    fake_gh.set_response(
        "GET", "repos/o/r/pulls/101", {"__error__": "API rate limit exceeded Retry-After: 1"}
    )
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None)
    assert rc != 0
    after = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]
    assert after["import_state"] == "fetched"            # NOT reset to fetch_failed
    assert after["import_file"] == before["import_file"]  # last-good linkage preserved
    assert after["import_sha256"] == before["import_sha256"]
    assert after["case_ids"] == before["case_ids"]
    assert after["latest_error"]["code"] == "rate_limit"  # attempt recorded separately
    assert (ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml").exists()  # case still indexed


def test_refresh_unreachable_pinned_head_freezes_fails_without_clobber(tmp_path: Path, fake_gh: FakeGh) -> None:
    """A refresh whose re-freeze of a curated pinned head goes unreplayable
    (force-push/rebased branch made the head unreachable) must fail the refresh
    (rc != 0) and keep the curated ready case + its bundle intact and indexed —
    never write the unreplayable snapshot over the curated case (issue #813)."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict
    from daydream.benchmark.workspace import init_workspace, validate_workspace

    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["api.anthropic.com"], ["api.anthropic.com"])
    _seed_preflight(ws, fake_gh)
    origin_url, _base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    case_id = f"pr-000101-{head_sha[:12]}"
    _curate_case(ws, f"{case_id}.yaml")
    case_path = ws / "cases" / f"{case_id}.yaml"
    before = load_yaml_strict(case_path)
    assert before["snapshot"]["status"] == "ready"
    bundle_rel = before["snapshot"]["bundle_file"]

    # force-push: repoint refs/pull/101/head to a commit that does NOT contain
    # the pinned head (a rebased branch), so the pinned head is unreachable.
    repo = tmp_path / "local_wt"
    _seed_git(repo, "checkout", "main")
    _seed_write(repo, "rebased.py", "REBASED = 1\n")
    _seed_git(repo, "add", "rebased.py")
    new_head = _seed_commit(repo, "force-pushed rebased head")
    _seed_git(repo, "push", "-f", "origin", f"{new_head}:refs/pull/101/head", check=False)
    hdr = dict(_PR_HEADER)
    hdr["head"] = {"ref": "feature/cache", "sha": new_head}
    _seed_preflight(ws, fake_gh, pull_header=hdr)

    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=[], refresh=True, origin_url=origin_url)
    assert rc != 0
    after = load_yaml_strict(case_path)
    assert after["snapshot"]["status"] == "ready"   # NOT replaced with unreplayable
    assert after["curation"]["state"] == "ready"     # curated gold preserved
    assert (ws / bundle_rel).exists()                  # bundle still on disk and referenced
    raw = load_yaml_strict(ws / "benchmark.yaml")
    pr = raw["pull_requests"][0]
    assert pr["import_state"] == "fetched"            # last-good linkage preserved
    assert pr["latest_error"] is not None              # attempt recorded, not silent
    code, _label = validate_workspace(ws)
    assert code == 0                                    # no orphan bundle corruption


def test_refresh_noncanonical_referenced_source_id_fails_closed(tmp_path: Path, fake_gh: FakeGh) -> None:
    """A hand-edited/externally-mutated curation whose referenced source_id is
    non-canonical must fail the re-import (rc != 0) instead of being silently
    dropped from the per-case stale gate — never fail open (issue #813)."""
    import yaml

    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_seed_discussion(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case_path = ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml"
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1

    # hand-edit: a non-canonical referenced source_id (malformed reference).
    raw = load_yaml_strict(case_path)
    raw["curation"]["findings"][0]["provenance"]["source_ids"] = [
        "https://github.com/o/r/pull/101#discussion_r1"
    ]
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None)
    assert rc != 0                                      # fail closed, not silently skipped
    after = load_yaml_strict(ws / "benchmark.yaml")
    pr = after["pull_requests"][0]
    assert pr["latest_error"] is not None
    on_disk = load_yaml_strict(case_path)
    assert on_disk["curation"]["findings"][0]["provenance"]["source_ids"] == [
        "https://github.com/o/r/pull/101#discussion_r1"
    ]   # curation was NOT rewritten by the failed refresh


def test_refresh_gained_reply_status_flips_signature_and_stales(tmp_path: Path, fake_gh: FakeGh) -> None:
    """reply_to_id gates candidacy (replies are evidence, never candidates), so it
    must sit in the projection hash: a comment gaining reply status shifts the
    candidate set and must flip the signature, staling a referencing case."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_seed_discussion(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1

    # the same comment now carries a reply parent (gains reply status): its
    # projection hash must flip even though body/anchor fields are unchanged.
    reply = dict(_seed_discussion(1))
    reply["in_reply_to_id"] = 10
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [reply])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale"       # referenced id's projection changed
    assert case["curation"]["findings"]               # curated gold preserved


def test_reimport_changed_referenced_evidence_stales(tmp_path: Path, fake_gh: FakeGh) -> None:
    """A plain re-import (refresh=False) with changed referenced evidence must not
    silently keep the curated case ready — it routes through the same per-case
    stale decision as refresh (issue #813)."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    # Seed the referenced comment (db 1) at line 4 for the first import.
    fake_gh.set_response(
        "GET",
        "repos/o/r/pulls/101/comments",
        [
            {"id": 1, "node_id": "DIFF_1", "user": {"login": "bot[bot]", "type": "Bot"},
             "body": "please fix", "commit_id": "a" * 40, "original_commit_id": "a" * 40,
             "path": "a.py", "line": 4, "subject_type": "line", "side": "RIGHT",
             "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#discussion_r1"},
        ],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1
    # Re-seed the REFERENCED comment (db 1) with a moved anchor, same body.
    fake_gh.set_response(
        "GET",
        "repos/o/r/pulls/101/comments",
        [
            {"id": 1, "node_id": "DIFF_1", "user": {"login": "bot[bot]", "type": "Bot"},
             "body": "please fix", "commit_id": "a" * 40, "original_commit_id": "a" * 40,
             "path": "a.py", "line": 7, "subject_type": "line", "side": "RIGHT",
             "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#discussion_r1"},
        ],
    )
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=False, origin_url=None)
    assert rc == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale"        # cannot bypass refresh semantics
    assert case["curation"]["findings"]                # curated findings preserved


def test_refresh_precanon_duplicate_db_id_verdict_is_deterministic(tmp_path: Path, fake_gh: FakeGh) -> None:
    """A legacy raw import file storing the same database_id twice - a REST
    inline copy WITH ``commit_id`` and a GraphQL thread copy WITHOUT it (the
    pre-canonicalization duplicate) - must yield ONE deterministic changed
    verdict per id. The two copies' projection hashes differ (``commit_id`` is
    projected), and the thread copy's missing commit anchors are a pure format
    artifact: the fresh REST-derived canonical projection is still covered by a
    prior projection, so the first post-format refresh must NOT stale the
    curated case. The old ``dict(prior_sig)`` collapse left which tuple survives
    to frozenset iteration order (nondeterministic across processes under hash
    randomization), making the spurious stale a coin toss.
    """
    import json

    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_json_strict, load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_seed_discussion(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1

    # Rewrite the prior import file to the REAL historical shape: the same
    # database_id persisted twice — the REST inline copy (commit_id present) and
    # a GraphQL thread copy (commit_id absent) — whose projection hashes differ.
    import_path = ws / "imports/pr-000101.json"
    prior = load_json_strict(import_path)
    rest = next(e for e in prior["evidence"] if e["database_id"] == 1)
    thread = {k: v for k, v in rest.items() if k not in ("commit_id", "original_commit_id")}
    thread.update(kind="thread_comment", source_id="github:thread_comment:1")
    prior["evidence"] = [thread, rest]
    import_path.write_text(json.dumps(prior, indent=2))

    # Premise check: the duplicate's two tuples are genuinely distinct (the
    # thread copy lacks the projected commit_id), so a dict() collapse would
    # hand the survivor to frozenset iteration order instead of comparing sets.
    sig = gi._evidence_signature_from_raw(prior)
    assert len(sig) == 2 and len({h for _, h in sig}) == 2

    # Refresh against the canonical feed (one record per id, unchanged body):
    # the fresh REST-derived projection matches a prior projection, so the
    # per-id comparison (order-independent) leaves the curated case ready, and
    # the refreshed file carries exactly one record per id.
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_seed_discussion(1)])
    assert gi.run_import_prs(
        ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None
    ) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"      # format drift must NOT stale gold
    assert case["curation"]["findings"]              # curated findings preserved
    refreshed = load_json_strict(import_path)
    assert len([e for e in refreshed["evidence"] if e["database_id"] == 1]) == 1
