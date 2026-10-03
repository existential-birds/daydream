"""Import-PR integration coverage through the hermetic ``fake_gh`` router.

Snapshot freezes use a real local bare origin; no network is required.
"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from daydream import cli as top_cli, git_ops
from daydream.benchmark import (
    curation as cu,
    github_import as gi,
    github_transport as transport,
    schema,
    snapshot as sn,
    storage,
)
from daydream.benchmark.cli import _handle_benchmark_command, _handle_benchmark_status
from daydream.benchmark.harbor import build
from daydream.benchmark.harbor.build import task_spec_digest
from daydream.benchmark.schema import EXTRACTION_VERSION, Location, case_id_for, derive_finding_id
from daydream.benchmark.storage import WorkspaceCorrupt, load_json_strict, load_yaml_strict, sha256_file
from daydream.benchmark.workspace import init_workspace, validate_workspace, workspace_status
from daydream.git_ops import RateLimitError
from daydream.reviews.identity import FINDING_MARKER_RE, finding_marker
from tests.harness import github_schema as gs
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import (
    git as _seed_git,
    seed_pr_origin,
    seeded_commit as _seed_commit,
    write_and_stage as _seed_write,
)

_PR_HEADER = {"number": 101, "url": "https://github.com/o/r/pull/101", "html_url": "https://github.com/o/r/pull/101",
    "title": "Fix cache", "body": "", "state": "open", "base": {"ref": "main", "sha": "b" * 40},
    "head": {"ref": "feature/cache", "sha": "a" * 40}, "merged_at": None, "closed_at": None,
    "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
    "user": {"login": "alice", "type": "User"}, "changed_files": 0,
}

_FETCH_IDENTITY = schema.PreflightLedger(
    last_verified_at="2026-01-01T00:00:00Z", repository="o/r",
    repository_id=None, visibility="private", matched=True,
)

_REPO_ID = "R_kgDOABC123"
_REPO_VIEW = {"id": _REPO_ID, "nameWithOwner": "o/r", "url": "https://github.com/o/r", "visibility": "PRIVATE",
    "defaultBranchRef": {"name": "main"},
}


def _seed_empty_rest(fake_gh: FakeGh) -> None:
    """Serve a PR with no reviews or comments from every REST evidence endpoint."""
    for endpoint in ("repos/o/r/pulls/101/reviews", "repos/o/r/pulls/101/comments", "repos/o/r/issues/101/comments"):
        fake_gh.set_response("GET", endpoint, [])


def _fetch_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "imports").mkdir(parents=True)
    return ws


def test_preflight_gh_and_ls_remote_wire_command_scoped_helper(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    assert gi._run_gh_api_user(ws) == {"login": "octocat", "type": "User"}
    refs = git_ops.git_ls_remote(ws, "https://github.com/o/r.git")
    assert "refs/heads/head" in refs
    ls = fake_gh.command_calls("git ls-remote")[-1]
    joined = " ".join(ls.argv)
    assert "-c" in ls.argv and any(a.startswith("credential.helper=") for a in ls.argv)
    assert "gh auth git-credential" in joined and "password=" not in joined
    assert ls.env is not None and ls.env.get("GIT_TERMINAL_PROMPT") == "0"

def test_fetch_persists_complete_pr_header(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    header = dict(_PR_HEADER)
    header["body"] = "fixes the cache\n\nand tests"
    header["html_url"] = "https://github.com/o/r/pull/101"
    header["merged_at"] = "2026-01-02T00:00:00Z"
    header["closed_at"] = "2026-01-02T00:00:00Z"
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    _seed_empty_rest(fake_gh)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    pr = doc.pull_request
    assert pr.body == "fixes the cache\n\nand tests"
    assert pr.html_url == "https://github.com/o/r/pull/101"
    assert pr.title_sha256 == hashlib.sha256(b"Fix cache").hexdigest()
    assert pr.body_sha256 == hashlib.sha256("fixes the cache\n\nand tests".encode()).hexdigest()
    assert pr.head.ref == "feature/cache"          # head.ref parity with base.ref
    assert pr.merged_at is not None and pr.closed_at is not None
    assert pr.number == 101 and pr.author.login == "alice"

def test_fetch_changed_files_persists_complete_rename_union(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    header = {**_PR_HEADER, "changed_files": 2}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/files",
        [{"status": "modified", "filename": "src/a:b.py"},
            {"status": "renamed", "filename": "src/new name.py", "previous_filename": r"src\old.py"},
        ],
    )
    _seed_empty_rest(fake_gh)

    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101, include_changed_files=True)

    assert doc.pull_request.changed_files == ["src/a:b.py", "src/new name.py", r"src\old.py"]
    calls = fake_gh.calls("GET", "repos/o/r/pulls/101/files")
    assert len(calls) == 1 and "--paginate" in (calls[0].argv or [])

@pytest.mark.parametrize(("count", "rows"),
    [(2, [{"status": "modified", "filename": "a.py"}]), (3001, []),
        (1, [{"status": "renamed", "filename": "new.py"}]),
        (1, [{"status": "modified", "filename": "new.py", "previous_filename": "old.py"}]), (1, [{"filename": "a.py"}]),
        (1, [{"status": "invented", "filename": "a.py"}]), (1, [{"status": 17, "filename": "a.py"}]),
        (2, [{"status": "modified", "filename": "a.py"}, {"status": "modified", "filename": "a.py"}]),
        (1, [{"status": "modified", "filename": "../escape.py"}]),
    ],
)
def test_fetch_changed_files_fails_closed_on_incomplete_or_malformed_inventory(
    tmp_path: Path, fake_gh: FakeGh, count: int, rows: list[Any]
) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", {**_PR_HEADER, "changed_files": count})
    fake_gh.set_response("GET", "repos/o/r/pulls/101/files", rows)
    _seed_empty_rest(fake_gh)
    with pytest.raises(git_ops.GitError, match="changed.files|inventory|3000"):
        gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101, include_changed_files=True)

def test_final_only_fetch_does_not_request_changed_files(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    _seed_empty_rest(fake_gh)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    assert doc.pull_request.changed_files is None
    assert fake_gh.calls("GET", "repos/o/r/pulls/101/files") == []

def test_materialized_case_carries_full_pr_header(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)                  # REST + canned PR for pr 101
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    pr = case["pull_request"]
    assert pr["head"]["ref"] == "feature/cache"    # head.ref persisted in the case YAML
    assert pr["body"] == ""                         # _PR_HEADER has no body -> empty
    assert pr["title_sha256"] and pr["body_sha256"]
    assert "merged_at" in pr and "closed_at" in pr and "html_url" in pr

def test_import_only_snapshot_records_requested_base_sha(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Both base SHAs retain the PR base tip until freezing computes the merge base."""

    ws = _preflight_workspace(tmp_path, fake_gh)                 # REST + canned PR for pr 101
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    snapshot = case["snapshot"]
    assert snapshot["status"] == "imported"
    assert snapshot["requested_base_sha"] == snapshot["original_base_sha"]
    assert snapshot["original_base_sha"] == "b" * 40
    assert snapshot["requested_base_sha"] == "b" * 40

@pytest.mark.parametrize("body_field,expected", [
    (None, ""),                      # null body -> empty
    ("", ""),                        # empty body
    ("héllo wörld \u00e9", "héllo wörld \u00e9"),          # Unicode preserved
    ("line1\nline2\nline3", "line1\nline2\nline3"),        # newlines preserved
    ("x" * 50000, "x" * 50000),      # over context-limit body (never bounded here; persisted whole)
])
def test_import_body_shape_preserved(tmp_path: Path, fake_gh: FakeGh, body_field: Any, expected: str) -> None:
    ws = _fetch_workspace(tmp_path)
    header = dict(_PR_HEADER)
    header["body"] = body_field
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    _seed_empty_rest(fake_gh)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    assert doc.pull_request.body == expected
    assert doc.pull_request.body_sha256 == hashlib.sha256(expected.encode("utf-8")).hexdigest()

@pytest.mark.parametrize("state,merged_at,closed_at,expect_merged", [("open", None, None, False),
    ("closed", None, "2026-01-02T00:00:00Z", False),     # closed-unmerged
    ("closed", "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z", True),  # merged
])
def test_import_merged_state_distinction(
    tmp_path: Path, fake_gh: FakeGh, state: Any, merged_at: Any, closed_at: Any, expect_merged: Any,
) -> None:
    ws = _fetch_workspace(tmp_path)
    header = dict(_PR_HEADER)
    header["state"] = state
    header["merged_at"] = merged_at
    header["closed_at"] = closed_at
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    _seed_empty_rest(fake_gh)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    pr = doc.pull_request
    assert pr.state == state
    assert (pr.merged_at is not None) == expect_merged
    assert (pr.closed_at is not None) == (closed_at is not None)

def test_import_no_comments_pr(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    header = dict(_PR_HEADER)
    header["body"] = "no comments here"
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    _seed_empty_rest(fake_gh)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    assert doc.evidence == [] and doc.pull_request.body == "no comments here"

def test_payload_digest_spans_header_and_evidence(tmp_path: Path, fake_gh: FakeGh) -> None:
    def fetch_with(title: Any) -> Any:
        ws = tmp_path / "ws"
        (ws / "imports").mkdir(parents=True, exist_ok=True)
        header = dict(_PR_HEADER)
        header["title"] = title
        header["body"] = "b"
        fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
        _seed_empty_rest(fake_gh)
        return gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    a = fetch_with("Fix cache")
    b = fetch_with("Fix cache EDITED")            # header-only change, same evidence
    assert a.fetch.payload_sha256 != b.fetch.payload_sha256
    assert gi._evidence_signature_from_doc(a) == gi._evidence_signature_from_doc(b)

def test_fetch_normalizes_all_rest_evidence(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [_review(1, 'approved')])
    fake_gh.set_response(
        "GET", "repos/o/r/pulls/101/comments", [_rest_comment(7, original_line=3, original_position=3)],
    )
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments",
        [{"id": 9, "user": {"login": "carol", "type": "User"}, "body": "question",
             "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#issuecomment-9"},
        ],
    )
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    kinds = {e.kind for e in doc.evidence}
    assert kinds == {"review", "inline_comment", "issue_comment"}
    assert doc.evidence[0].source_id == "github:review:1"
    assert doc.evidence[0].is_bot is False
    assert doc.evidence[1].is_bot is True      # bot classification retained, not dropped
    assert doc.evidence[1].subject_type == "line" and doc.evidence[1].side == "RIGHT"
    get_calls = fake_gh.calls("GET")
    header_args = [c.argv for c in get_calls if c.endpoint == "repos/o/r/pulls/101"]
    collection_args = [c.argv for c in get_calls if c.endpoint != "repos/o/r/pulls/101"]
    assert header_args and "--paginate" not in (header_args[0] or [])
    assert all("--paginate" in (a or []) for a in collection_args)

def test_review_thread_queries_request_only_schema_fields() -> None:
    """Validate GraphQL's actual field names, accepting aliases such as ``side: diffSide``."""
    assert gs.unknown_query_fields(gi._REVIEW_THREADS_QUERY) == set()
    assert gs.unknown_query_fields(gi._THREAD_COMMENTS_QUERY) == set()

def test_graphql_threads_and_replies_normalized(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [
        _rest_comment(10, 'root', login='dave', user_type='User', original_line=3, original_path='a.py'),
        _rest_comment(11, 'reply', login='eve', user_type='User', original_commit_id=None, line=5, in_reply_to_id=10),
    ])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([{"id": "thread_1", "isResolved": True,
         "isOutdated": True, "subjectType": "LINE", "path": "a.py", "line": 4, "originalLine": 3,
         "side": "RIGHT", "startSide": None, "comments": {"nodes": [
             {"id": "c1", "databaseId": 10, "body": "root", "author": {"login": "dave", "type": "User"},
              "createdAt": "2026-01-01T00:00:00Z", "url": "https://github.com/o/r/pull/101#discussion_r10"},
             {"id": "c2", "databaseId": 11, "body": "reply", "replyTo": {"id": "c1"},
              "author": {"login": "eve", "type": "User"},
              "createdAt": "2026-01-01T00:00:00Z", "url": "https://github.com/o/r/pull/101#discussion_r11"},
         ]}},
    ])
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    by_db = {e.database_id: e for e in doc.evidence}
    root, reply = by_db[10], by_db[11]
    assert root.kind == "inline_comment" and root.source_id == "github:inline_comment:10"
    assert root.resolved is True and root.outdated is True
    assert root.thread_id == "thread_1" and root.side == "RIGHT" and root.path == "a.py"
    assert reply.kind == "inline_comment" and reply.thread_id == "thread_1"
    assert reply.reply_to_id == "10"          # REST in_reply_to_id (parent db id)
    assert not any(e.kind == "thread_comment" for e in doc.evidence)

def test_rest_inline_normalization_retains_original_range(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    # Original range belongs to the authoring commit; observed fields describe the re-anchor.
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(1, 'fix this', login='alice', user_type='User', commit_id='b' * 40, line=5, original_line=5,
                original_start_line=4, start_line=4
            ),
        ],
    )
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([])
    doc_gi = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    rec = {e.database_id: e for e in doc_gi.evidence}[1]
    assert rec.original_start_line == 4
    assert rec.original_commit_id == "a" * 40
    assert rec.original_line == 5

def test_graphql_thread_maps_original_start_line(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    # thread-only comment (no REST counterpart) canonicalized from thread fields
    fake_gh._write_threads([{"id": "thread_1", "isResolved": False, "isOutdated": False,
         "subjectType": "LINE", "path": "a.py", "line": 5, "originalLine": 5, "originalStartLine": 4,
         "side": "RIGHT", "startSide": None, "comments": {"nodes": [{"id": "c1", "databaseId": 1, "body": "fix this",
              "author": {"login": "alice", "type": "User"}, "createdAt": "2026-01-01T00:00:00Z",
              "url": "https://github.com/o/r/pull/101#discussion_r1"},
         ]}},
    ])
    doc_gi = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    rec = {e.database_id: e for e in doc_gi.evidence}[1]
    assert rec.kind == "inline_comment"
    assert rec.original_start_line == 4

head_sha = "a" * 40  # matches _PR_HEADER's head sha; the projection head in these tests

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _set_anchor(
    rec: Any, *, status: Literal["derived", "history-unavailable", "path-unavailable", "range-unavailable"] = "derived",
    commit_id: str = "a" * 40, path: str = "a.py", start_line: int = 4, end_line: int = 5,
) -> Any:
    """Attach a strict derived or closed anchor directly for projection tests."""

    if status == "derived":
        rec.authoring_anchor = schema.AuthoringAnchor(
            version=1, status="derived", commit_id=commit_id, path=path, start_line=start_line, end_line=end_line,
        )
    else:
        rec.authoring_anchor = schema.AuthoringAnchor(
            version=1, status=status, commit_id=None, path=None, start_line=None, end_line=None,
        )
    return rec


def _evidence_record(**over: Any) -> schema.EvidenceRecord:
    body = over.pop("body", "fix this")
    fields: dict[str, Any] = {
        "source_id": "github:inline_comment:1", "kind": "inline_comment", "database_id": 1, "node_id": "DIFF_1",
        "author": schema._EvidenceAuthor(login="alice", type="User"), "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "created_at": _TS, "updated_at": _TS,
        "commit_id": head_sha, "original_commit_id": head_sha, "is_bot": False,
        "url": "https://github.com/o/r/pull/101#discussion_r1",
    }
    fields.update(over)
    return schema.EvidenceRecord(**fields)


def _rec_dict(**over: Any) -> dict[str, Any]:
    rec = _evidence_record(
        path="a.py", original_path="a.py", line=4, start_line=4, original_line=4, original_start_line=4,
        subject_type="line", side="RIGHT",
    )
    d = rec.model_dump(mode="json")
    d.update(over)
    return d


def _project_from_anchor(*, anchor_commit: str | None = None,
    status: Literal["derived", "history-unavailable", "path-unavailable", "range-unavailable"] | None = None,
    rest_commit_id: str | None = None, original_commit_id: str | None = None, anchor: Any = None,
) -> Any:
    """Use observed new.py:9 and authoring old.py:4–5 to reveal which location wins."""

    if anchor is None:
        if status is not None:
            anchor = schema.AuthoringAnchor(
                version=1, status=status, commit_id=None, path=None, start_line=None, end_line=None,
            )
        elif anchor_commit is not None:
            anchor = schema.AuthoringAnchor(
                version=1, status="derived", commit_id=anchor_commit, path="old.py", start_line=4, end_line=5,
            )
    rec = _evidence_record(commit_id=rest_commit_id if rest_commit_id is not None else head_sha,
        original_commit_id=original_commit_id if original_commit_id is not None else head_sha, path="new.py",
        original_path="new.py", line=9, start_line=9, original_line=8, original_start_line=8, subject_type="line",
        side="RIGHT", authoring_anchor=anchor,
    )
    return gi._project_one(rec, head_sha=head_sha)


@pytest.mark.parametrize("outdated", [False, True])
@pytest.mark.parametrize(("raw_template", "expected_title", "expected_body"),
    [
        ("\n{marker}\r\n## Cache race\r\nDetails.\r\n", "Cache race", "\n\n## Cache race\nDetails."),
        ("Before {marker} after\n{marker}\n", "Before after", "Before  after"),
        ("{marker}", "", ""),
        ("<!-- daydream-finding: ab -->", "<!-- daydream-finding: ab -->", "<!-- daydream-finding: ab -->"),
    ],
)
def test_finding_marker_projection_preserves_raw_evidence_and_eligibility(
    raw_template: str, expected_title: str, expected_body: str, outdated: bool,
) -> None:
    raw_body = raw_template.format(marker=finding_marker("f" * 64))
    rec = _evidence_record(
        body=raw_body, commit_id="a" * 40, original_commit_id="a" * 40, path="feature.py", line=2, original_line=2,
        subject_type="line", side="RIGHT", authoring_anchor=schema.AuthoringAnchor(
            version=1, status="derived", commit_id="a" * 40, path="feature.py", start_line=2, end_line=2,
        ), outdated=outdated,
    )
    before = rec.model_dump()

    candidate = gi._project_one(rec, head_sha="a" * 40)

    assert candidate.title == expected_title
    assert candidate.body == expected_body
    assert not FINDING_MARKER_RE.search(candidate.title + "\n" + candidate.body)
    assert candidate.source_id == rec.source_id
    assert candidate.location == schema.Location(path="feature.py", start_line=2, end_line=2)
    assert candidate.exact_acceptable is (bool(expected_title) and not outdated)
    assert candidate.not_exact_reason == ("outdated" if outdated else "title" if not expected_title else None)
    assert rec.model_dump() == before

def test_exact_acceptance_from_authoring_anchor_matches_head() -> None:
    """GitHub's re-anchored commit matches head, but the authoring commit must gate acceptance."""
    cand = _project_from_anchor(anchor_commit="b" * 40, rest_commit_id=head_sha, original_commit_id="b" * 40)
    assert cand.exact_acceptable is False
    assert cand.not_exact_reason == "re-anchored"

def test_exact_acceptance_under_explicit_historical_snapshot() -> None:
    """An anchor matching the historical head remains exact after GitHub re-anchors the comment."""

    cand = _project_from_anchor(anchor_commit=head_sha, rest_commit_id="c" * 40, original_commit_id=head_sha)
    assert cand.exact_acceptable is True
    assert cand.not_exact_reason is None
    assert cand.location == Location(path="old.py", start_line=4, end_line=5)

def test_range_and_missing_anchor_fail_closed() -> None:
    """Preserve closed-anchor reasons; missing anchors mean history-unavailable."""
    assert _project_from_anchor(status="range-unavailable").not_exact_reason == "range-unavailable"
    assert _project_from_anchor(anchor=None).not_exact_reason == "history-unavailable"

def test_anchor_fields_flip_projection_signature() -> None:
    h1 = gi._evidence_projection_hash(_rec_dict())
    d = _rec_dict()
    d["authoring_anchor"] = {"version": 1, "status": "path-unavailable",
        "commit_id": None, "path": None, "start_line": None, "end_line": None}
    assert gi._evidence_projection_hash(d) != h1

def test_file_level_comment_exactness_gated_by_anchor() -> None:
    """Locationless comments require editing even with a derived anchor."""

    def project(anchor: schema.AuthoringAnchor | None) -> schema.Candidate:
        rec = _evidence_record(
            commit_id="a" * 40, original_commit_id="a" * 40, subject_type="file", authoring_anchor=anchor,
        )
        return gi._project_one(rec, head_sha="a" * 40)

    no_anchor = project(None)
    assert no_anchor.location is None
    assert no_anchor.exact_acceptable is False
    assert no_anchor.not_exact_reason == "history-unavailable"

    closed = project(schema.AuthoringAnchor(
        version=1, status="path-unavailable", commit_id=None, path=None, start_line=None, end_line=None,
    ))
    assert closed.location is None
    assert closed.exact_acceptable is False
    assert closed.not_exact_reason == "path-unavailable"

    derived_off_head = project(schema.AuthoringAnchor(
        version=1, status="derived", commit_id="b" * 40, path="a.py", start_line=4, end_line=5,
    ))
    assert derived_off_head.location is None
    assert derived_off_head.exact_acceptable is False
    assert derived_off_head.not_exact_reason == "range-unavailable"

    derived_at_head = project(schema.AuthoringAnchor(
        version=1, status="derived", commit_id="a" * 40, path="a.py", start_line=4, end_line=5,
    ))
    assert derived_at_head.location is None
    assert derived_at_head.exact_acceptable is False
    assert derived_at_head.not_exact_reason == "range-unavailable"

def test_derive_one_anchor_inverted_range_and_bad_path_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rec(*, original_start_line: int, original_line: int) -> schema.EvidenceRecord:
        return _evidence_record(commit_id="a" * 40, original_commit_id="a" * 40, path="a.py", original_path="a.py",
            original_line=original_line, original_start_line=original_start_line, subject_type="line", side="RIGHT",
        )

    mirror = tmp_path / "mirror.git"
    # inverted range: the guard fires before the mirror is consulted
    inverted = gi._derive_one_anchor(rec(original_start_line=8, original_line=4), mirror, "a" * 40)
    assert inverted.status == "range-unavailable"
    assert inverted.commit_id is None and inverted.path is None

    monkeypatch.setattr("daydream.benchmark.snapshot.derive_authoring_path", lambda *a, **k: "weird:file.py")
    bad_path = gi._derive_one_anchor(rec(original_start_line=4, original_line=5), mirror, "a" * 40)
    assert bad_path.status == "path-unavailable"
    assert bad_path.commit_id is None and bad_path.path is None

def test_candidate_projection_right_file_body_left(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response(
        "GET", "repos/o/r/pulls/101/reviews", [_review(5, 'review body', state='COMMENTED'), _review(6, 'looks good')],
    )
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(
                1, '## note\nfix this', login='alice', user_type='User', line=5, original_line=5,
                original_start_line=4, start_line=4
            ), {"id": 2, "node_id": "DIFF_2", "user": {"login": "alice", "type": "User"},
             "body": "file-level", "subject_type": "file", "commit_id": "a" * 40, "created_at": "2026-01-01T00:00:00Z",
             "updated_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#discussion_r2"},
            _rest_comment(3, 'left-side', login='alice', user_type='User', original_commit_id=None, line=2, side='LEFT',
                start_line=2
            ),
        ],
    )
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([])
    doc_gi = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    # Direct fetch/project bypasses materialization, so supply its normally derived anchor.
    _set_anchor(
        {e.database_id: e for e in doc_gi.evidence}[1], commit_id="a" * 40, path="a.py", start_line=4, end_line=5,
    )
    cands = gi.project_candidates(doc_gi, head_sha="a" * 40)
    by_src = {c.source_id: c for c in cands}
    right = by_src["github:inline_comment:1"]
    assert right.title == "note" and "fix this" in right.body
    assert right.severity is None
    assert right.location == Location(path="a.py", start_line=4, end_line=5)
    assert right.exact_acceptable is True
    assert by_src["github:inline_comment:2"].location is None       # file-level
    # This file-level record has no derived anchor.
    assert by_src["github:inline_comment:2"].exact_acceptable is False
    assert by_src["github:inline_comment:2"].not_exact_reason == "history-unavailable"
    assert by_src["github:inline_comment:3"].exact_acceptable is False  # LEFT
    assert by_src["github:review:5"].location is None               # review body
    assert by_src["github:review:5"].exact_acceptable is True
    assert "github:review:6" not in by_src                          # pure approval: no candidate

def test_parse_targets_dedupes_and_orders(tmp_path: Path) -> None:
    pf = tmp_path / "prs.txt"
    pf.write_text("42\nhttps://github.com/o/r/pull/9\n7\n42\n")
    targets = gi.parse_import_targets(
        pr_args=["https://github.com/o/r/pull/9", "7"], pr_files=[pf], heads=["abc" * 13 + "1", "abc" * 13 + "2"],
    )
    assert targets.pr_numbers == [9, 7, 42]     # stable: CLI order then file order; dupes collapsed
    assert targets.requested_heads == ["final", "abc" * 13 + "1", "abc" * 13 + "2"]  # 'final' always present

def test_parse_head_pr_sha_grammar_and_binding() -> None:
    sha = "a" * 40
    targets = gi.parse_import_targets(["101"], [], [f"101={sha}"])
    assert targets.requested_heads == ["final", sha]
    assert targets.pr_heads == {101: ["final", sha]}
    targets2 = gi.parse_import_targets(["101"], [], [sha])
    assert targets2.requested_heads == ["final", sha]
    with pytest.raises(gi.ImportTargetError):
        gi.parse_import_targets(["101"], [], ["101=nothex"])
    with pytest.raises(gi.ImportTargetError):
        gi.parse_import_targets(["100"], [], [f"101={sha}"])

def test_parse_heads_bound_per_pr_in_multi_import() -> None:
    sha = "a" * 40
    targets = gi.parse_import_targets(["100", "101"], [], [f"101={sha}"])
    assert targets.pr_numbers == [100, 101]
    assert targets.pr_heads == {100: ["final"], 101: ["final", sha]}
    assert targets.requested_heads == ["final", sha]


def _seed_manifest(ws: Path) -> None:
    """Initialize unresolved source o/r, preserving any caller-supplied workspace and host pins."""

    if (ws / "benchmark.yaml").exists():
        return
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])


def test_preflight_six_checks_in_order_and_atomic_identity(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)  # unresolved Source (repository=o/r)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    gi.preflight(ws, pr_count=2)
    assert "authenticated identity: octocat" in capsys.readouterr().out
    ledger = load_json_strict(ws / "runtime" / "preflight.json")
    assert ledger["repository_id"] == _REPO_ID and ledger["visibility"] == "private"
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == _REPO_ID and raw["source"]["visibility"] == "private"
    # Immutable repository identity is re-verified on every import.
    gi.preflight(ws, pr_count=1)
    assert load_json_strict(ws / "runtime" / "preflight.json")["repository_id"] == _REPO_ID

@pytest.mark.parametrize("repository_id", [_REPO_ID, "12345"])
def test_import_cli_preflight_reports_only_persisted_identity(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str], repository_id: str,
) -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)
    before = (ws / "benchmark.yaml").read_bytes()
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value={**_REPO_VIEW, "id": repository_id})
    fake_gh.set_response("GET", "repos/o/r/pulls/101", {"__error__": "Not Found (HTTP 404)"})
    with pytest.raises(SystemExit) as exit_info:
        top_cli.main(["benchmark", "import-prs", str(ws), "--pr", "101"])
    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    ledger_path = ws / "runtime" / "preflight.json"
    if repository_id == _REPO_ID:
        ledger = load_json_strict(ledger_path)
        assert ledger["repository_id"] == _REPO_ID
        assert ledger["visibility"] == "private" and ledger["matched"] is True
        assert captured.out.splitlines() == [
            "authenticated identity: octocat",
            "repository visibility: private",
            "requested PR count: 1",
            f"local destination: {ws / 'imports'}",
        ]
        assert "fetch_failed" in json.dumps(load_yaml_strict(ws / "benchmark.yaml"))
    else:
        assert not ledger_path.exists()
        assert (ws / "benchmark.yaml").read_bytes() == before
        assert captured.out == ""
        assert "repo_unresolved" in captured.err


def test_preflight_reverifies_identity_on_every_run_and_fails_closed(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)                                  # unresolved Source (repository=repo)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    gi.preflight(ws, pr_count=2)
    assert "authenticated identity: octocat" in capsys.readouterr().out
    ledger = load_json_strict(ws / "runtime" / "preflight.json")
    assert ledger["repository_id"] == _REPO_ID and ledger["visibility"] == "private"
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == _REPO_ID and raw["source"]["visibility"] == "private"

    fake_gh.set_response("repo-view-full", value={**_REPO_VIEW, "id": "R_kgDDIFFERENT"})
    with pytest.raises(gi.PreflightError) as ei:
        gi.preflight(ws, pr_count=1)
    assert ei.value.code == "repo_mismatch"
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == _REPO_ID    # unchanged: no mutation staged

@pytest.mark.parametrize("node_id", [5, "12345"], ids=["integer", "numeric-string"])
def test_preflight_rejects_numeric_node_id(tmp_path: Path, fake_gh: FakeGh, node_id: Any) -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value={**_REPO_VIEW, "id": node_id})
    with pytest.raises(gi.PreflightError) as ex:
        gi.preflight(ws, pr_count=1)
    assert ex.value.code == "repo_unresolved"

def test_status_reports_last_preflight_verification(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    gi.preflight(ws, pr_count=1)

    st = workspace_status(ws)
    assert st.last_preflight_verified_at is not None
    ledger = load_json_strict(ws / "runtime" / "preflight.json")
    assert ledger["repository_id"] == _REPO_ID and ledger["matched"] is True

    ws2 = tmp_path / "ws2"
    _seed_manifest(ws2)
    assert workspace_status(ws2).last_preflight_verified_at is None

    _handle_benchmark_status(ws)
    assert "repository identity/access verification: ran" in capsys.readouterr().out
    _handle_benchmark_status(ws2)
    assert "repository identity/access verification: not yet run" in capsys.readouterr().out

def test_rate_limit_retries_three_then_fails_pr(tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _fetch_workspace(tmp_path)
    attempts = {"n": 0}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    slept: list[float] = []
    monkeypatch.setattr("daydream.benchmark.github_transport.time.sleep", lambda s: slept.append(s))
    real_run = subprocess.run

    def flaky_gh(args: Any, *pargs: Any, **kwargs: Any) -> Any:
        argv = list(args)
        joined = " ".join(argv)
        if (argv
            and argv[0] == "gh"
            and "pulls/101" in joined
            and "reviews" not in joined
            and "comments" not in joined
            and "issues" not in joined
        ):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return subprocess.CompletedProcess(
                    argv, 1, "API rate limit exceeded", "gh: API rate limit exceeded Retry-After: 2",
                )
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr("daydream.git_ops.process.subprocess.run", flaky_gh)
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


def _preflight_workspace(
    tmp_path: Path, fake_gh: FakeGh, *, hosts: tuple[str, str] | None = None, pull_header: Any = _PR_HEADER,
) -> Path:
    """Seed PR 101 and its workspace, optionally overriding reviewer/judge hosts."""
    ws = tmp_path / "ws"
    if hosts is not None:
        init_workspace(ws, "o/r", [hosts[0]], [hosts[1]])
    _seed_preflight(ws, fake_gh, pull_header=pull_header)
    return ws



def _seed_local_origin(tmp_path: Path, fake_gh: FakeGh) -> tuple[str, str, str]:
    """Seed real PR refs through ``seed_pr_origin`` and synchronize the fake header SHAs.

    Return ``(origin_url, base_sha, head_sha)`` for network-free import freezes.
    """
    origin_url, base_sha, head_sha = seed_pr_origin(tmp_path)
    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_sha}
    header["head"] = {"ref": "feature/cache", "sha": head_sha}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    return origin_url, base_sha, head_sha


def _seed_stacked_origin(tmp_path: Path, fake_gh: FakeGh) -> tuple[str, str, str, str]:
    """Build an advanced-base PR whose historical head adds legacy.py and feature.py.

    The final head reverts legacy.py, distinguishing historical and final path inventories.
    """
    repo = tmp_path / "stacked_wt"
    repo.mkdir()
    _seed_git(repo, "init", "-b", "main")
    _seed_write(repo, "readme.txt", "base\n")
    base_sha = _seed_commit(repo, "base")
    _seed_write(repo, "upstream.py", "UPSTREAM = 1\n")
    base_tip = _seed_commit(repo, "advanced base")
    _seed_git(repo, "checkout", "--detach", base_sha)
    _seed_write(repo, "legacy.py", "LEGACY = 1\n")
    _seed_write(repo, "feature.py", "FEATURE = 1\n")
    explicit_sha = _seed_commit(repo, "historical feature")
    _seed_git(repo, "rm", "legacy.py")
    final_sha = _seed_commit(repo, "revert historical path")

    bare = tmp_path / "stacked_origin.git"
    bare.mkdir()
    _seed_git(bare, "init", "--bare")
    _seed_git(repo, "remote", "add", "origin", str(bare))
    _seed_git(repo, "push", "origin", "main:main")
    _seed_git(repo, "push", "origin", f"{final_sha}:refs/pull/101/head", check=False)
    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_tip}
    header["head"] = {"ref": "feature", "sha": final_sha}
    header["changed_files"] = 2
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/files",
        [{"status": "added", "filename": "feature.py"}, {"status": "added", "filename": "legacy.py"}],
    )
    return str(bare), base_tip, explicit_sha, final_sha


def _seed_anchor_origin(tmp_path: Path, fake_gh: FakeGh) -> tuple[str, str, str, str]:
    """Seed a.py edits followed by old.py -> new.py rename, and synchronize the PR header.

    Return ``(origin_url, base_sha, authoring_sha, head_sha)`` for mirror-based anchor tracing.
    """

    repo = tmp_path / "anchor_wt"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    _seed_git(repo, "init", "-b", "main")
    _seed_write(repo, "readme.txt", "README\n")
    _seed_write(repo, "a.py", "A1 = 1\nA2 = 1\nA3 = 1\nA4 = 1\nA5 = 1\n")
    _seed_write(repo, "old.py", "O1 = 1\nO2 = 1\n")
    _seed_commit(repo, "base")
    base_sha = _seed_git(repo, "rev-parse", "HEAD")
    _seed_git(repo, "checkout", "-b", "feature")
    _seed_write(repo, "a.py", "A1 = 1\nA1b = 1\nA2 = 1\nA3 = 1\nA4 = 1\nA5 = 1\n")
    authoring_sha = _seed_commit(repo, "edit a.py on feature")
    _seed_git(repo, "mv", "old.py", "new.py")
    head_sha = _seed_commit(repo, "rename old.py to new.py")
    bare = tmp_path / "anchor_origin.git"
    if bare.exists():
        shutil.rmtree(bare)
    bare.mkdir()
    _seed_git(bare, "init", "--bare")
    _seed_git(repo, "remote", "add", "origin", str(bare))
    _seed_git(repo, "push", "origin", "main:main")
    _seed_git(repo, "push", "origin", f"{head_sha}:refs/pull/101/head", check=False)
    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_sha}
    header["head"] = {"ref": "feature", "sha": head_sha}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    return str(bare), base_sha, authoring_sha, head_sha


def test_materialization_derives_anchors_per_comment(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Persist per-comment anchors from real Git history, including an earlier renamed path."""

    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, _base_sha, authoring_sha, head_sha = _seed_anchor_origin(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(
                1, 'fix at head', login='alice', user_type='User', original_commit_id=head_sha, commit_id=head_sha,
                line=5, original_line=5, original_start_line=4
            ), _rest_comment(2, 'fix old file', login='carol', user_type='User', original_commit_id=authoring_sha,
                commit_id=head_sha, path='new.py', line=2, original_line=2, original_start_line=1
            ),
        ],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    imp = load_json_strict(ws / "imports" / "pr-000101.json")
    anchors = {e["database_id"]: e["authoring_anchor"] for e in imp["evidence"]}
    at_head = anchors[1]
    assert at_head is not None and at_head["status"] == "derived"
    assert at_head["commit_id"] == head_sha and at_head["path"] == "a.py"
    assert at_head["start_line"] == 4 and at_head["end_line"] == 5
    # Mirror rename tracing recovers old.py from GitHub's observed new.py.
    earlier = anchors[2]
    assert earlier is not None and earlier["status"] == "derived"
    assert earlier["commit_id"] == authoring_sha and earlier["path"] == "old.py"
    assert earlier["start_line"] == 1 and earlier["end_line"] == 2

def test_materialization_fails_closed_without_mirror(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Without a freeze mirror, keep anchors absent instead of trusting GitHub's re-anchored data."""

    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(1, 'fix this', login='alice', user_type='User', original_line=4, original_start_line=3)],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    imp = load_json_strict(ws / "imports" / "pr-000101.json")
    rec = next(e for e in imp["evidence"] if e["database_id"] == 1)
    assert rec["kind"] == "inline_comment"
    assert rec["authoring_anchor"] is None

def test_materialization_inverted_authoring_range_fails_closed(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, _base_sha, authoring_sha, head_sha = _seed_anchor_origin(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(1, 'inverted range', login='alice', user_type='User', original_commit_id=authoring_sha,
                commit_id=head_sha, path='new.py', original_line=4, original_start_line=8
            ),
        ],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=origin_url) == 0
    imp = load_json_strict(ws / "imports" / "pr-000101.json")
    rec = next(e for e in imp["evidence"] if e["database_id"] == 1)
    assert rec["authoring_anchor"] == {"version": 1, "status": "range-unavailable",
        "commit_id": None, "path": None, "start_line": None, "end_line": None,
    }

def test_import_freezes_cases_ready_with_bundle(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
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
    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    ids1 = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]["case_ids"]
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    ids2 = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]["case_ids"]
    assert ids1 == ids2
    # a distinct head (unreachable in this origin) -> a distinct case id
    alt_head = "cdef" * 10
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[alt_head], origin_url=origin_url) == 0
    ids3 = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]["case_ids"]
    assert len(ids3) == len(ids1) + 1 and ids3[-1].endswith(alt_head[:12])
    assert (ws / "cache" / "repository.git").exists()
    assert sn.rev_parse(ws / "cache/repository.git", "refs/pull/101/head") == head_sha

def test_refresh_demotes_clean_draft_when_historical_head_leaves_pr_scope(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real import boundary applies final-inventory scope to retained heads."""

    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, _base_tip, explicit_sha, final_sha = _seed_stacked_origin(tmp_path, fake_gh)

    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[explicit_sha], origin_url=origin_url) == 0
    explicit_path = ws / f"cases/pr-000101-{explicit_sha[:12]}.yaml"
    explicit = load_yaml_strict(explicit_path)
    assert explicit["snapshot"]["status"] == "ready"
    prior_bundle = ws / explicit["snapshot"]["bundle_file"]
    assert prior_bundle.exists()
    explicit["curation"].update(
        state="draft", snapshot_attested=True, clean_attested=True, gold_status="clean", task_spec_sha256="d" * 64,
    )
    explicit_path.write_text(yaml.safe_dump(explicit, sort_keys=False))

    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": _base_tip}
    header["head"] = {"ref": "feature", "sha": final_sha}
    header["changed_files"] = 1
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/files", [{"status": "added", "filename": "feature.py"}])

    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 0
    explicit = load_yaml_strict(explicit_path)
    assert explicit["snapshot"]["status"] == "unreplayable"
    assert explicit["snapshot"]["error"]["reason"] == "base_drift"
    assert explicit["curation"]["state"] == "unreplayable"
    assert explicit["curation"]["snapshot_attested"] is False
    assert explicit["curation"]["clean_attested"] is False
    assert explicit["curation"]["gold_status"] is None
    assert "task_spec_sha256" not in explicit["curation"]

    final = load_yaml_strict(ws / f"cases/pr-000101-{final_sha[:12]}.yaml")
    assert final["snapshot"]["status"] == "ready"
    assert not prior_bundle.exists()

    assert validate_workspace(ws) == (
        2, "incomplete: workspace state curating; unreplayable snapshot reasons: base_drift",
    )
    capsys.readouterr()
    with pytest.raises(SystemExit) as status_exit:
        top_cli.main(["benchmark", "status", str(ws)])
    assert status_exit.value.code == 0
    assert "snapshot unreplayable (base_drift)" in capsys.readouterr().out
    with pytest.raises(SystemExit) as validate_exit:
        top_cli.main(["benchmark", "validate", str(ws)])
    assert validate_exit.value.code == 2
    assert "unreplayable snapshot reasons: base_drift" in capsys.readouterr().out

def test_explicit_head_path_probe_git_failure_isolated_to_that_case(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, _base_tip, explicit_sha, final_sha = _seed_stacked_origin(tmp_path, fake_gh)

    real_git = shutil.which("git")
    assert real_git is not None
    shim_dir = tmp_path / "git shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = diff ] && [ \"$2\" = --name-status ]; then\n"
        "  echo 'injected path inventory failure' >&2\n"
        "  exit 88\n"
        "fi\n"
        "exec \"$DAYDREAM_TEST_REAL_GIT\" \"$@\"\n"
    )
    shim.chmod(0o755)
    monkeypatch.setenv("DAYDREAM_TEST_REAL_GIT", real_git)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")

    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[explicit_sha], origin_url=origin_url) == 0
    manifest = load_yaml_strict(ws / "benchmark.yaml")
    assert manifest["pull_requests"][0]["import_state"] == "fetched"
    explicit = load_yaml_strict(ws / f"cases/pr-000101-{explicit_sha[:12]}.yaml")
    final = load_yaml_strict(ws / f"cases/pr-000101-{final_sha[:12]}.yaml")
    assert explicit["snapshot"]["status"] == "unreplayable"
    assert explicit["snapshot"]["error"]["reason"] == "bundle_failure"
    assert "injected path inventory failure" in explicit["snapshot"]["error"]["detail"]
    assert final["snapshot"]["status"] == "ready"

def test_refresh_legacy_ready_snapshot_requires_upgrade_before_retirement(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = tmp_path / "workspace with spaces"
    init_workspace(ws, "o/r", ["api.anthropic.com"], ["api.anthropic.com"])
    _seed_preflight(ws, fake_gh)
    origin_url, base_tip, explicit_sha, final_sha = _seed_stacked_origin(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[explicit_sha], origin_url=origin_url) == 0

    explicit_path = ws / f"cases/pr-000101-{explicit_sha[:12]}.yaml"
    prior = load_yaml_strict(explicit_path)
    prior["snapshot"].pop("base_resolution")
    explicit_path.write_text(yaml.safe_dump(prior, sort_keys=False))
    prior_bytes = explicit_path.read_bytes()
    prior_bundle = ws / prior["snapshot"]["bundle_file"]

    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_tip}
    header["head"] = {"ref": "feature", "sha": final_sha}
    header["changed_files"] = 1
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/files", [{"status": "added", "filename": "feature.py"}])

    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 1
    entry = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]
    assert entry["import_state"] == "fetched"
    message = entry["latest_error"]["message"]
    assert "daydream benchmark upgrade <workspace>" in message
    assert str(ws) in message
    assert "base_resolution" in message
    assert explicit_path.read_bytes() == prior_bytes
    assert prior_bundle.exists()

def test_bundle_retirement_preserves_a_ready_shared_reference(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, _base_tip, explicit_sha, _final_sha = _seed_stacked_origin(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[explicit_sha], origin_url=origin_url) == 0
    manifest = load_yaml_strict(ws / "benchmark.yaml")
    explicit_id = case_id_for(101, explicit_sha)
    explicit_path = ws / f"cases/{explicit_id}.yaml"
    explicit = load_yaml_strict(explicit_path)

    alias_head = "e" * 40
    alias_id = case_id_for(101, alias_head)
    alias = copy.deepcopy(explicit)
    alias["case_id"] = alias_id
    alias["snapshot"]["original_head_sha"] = alias_head
    alias["snapshot"]["requested_head"] = alias_head
    alias_path = ws / f"cases/{alias_id}.yaml"
    alias_path.write_text(yaml.safe_dump(alias, sort_keys=False))
    manifest["cases"].append({"case_id": alias_id, "pr_number": 101, "case_file": f"cases/{alias_id}.yaml"})

    transitioned = copy.deepcopy(explicit)
    transitioned["snapshot"]["status"] = "unreplayable"
    assert gi._retired_snapshot_bundles(ws, manifest, 101, [(explicit_id, f"cases/{explicit_id}.yaml", transitioned)],
    ) == []

def test_inventory_only_refresh_preserves_gold_when_snapshot_remains_in_scope(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Changed-file scope evidence is persisted but is not reviewer task input."""

    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, base_tip, explicit_sha, final_sha = _seed_stacked_origin(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[explicit_sha], origin_url=origin_url) == 0
    explicit_path = ws / f"cases/pr-000101-{explicit_sha[:12]}.yaml"
    explicit = load_yaml_strict(explicit_path)
    explicit["curation"].update(state="draft", snapshot_attested=False, clean_attested=True, gold_status="clean")
    explicit_path.write_text(yaml.safe_dump(explicit, sort_keys=False))
    before_curation = load_yaml_strict(explicit_path)["curation"]

    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_tip}
    header["head"] = {"ref": "feature", "sha": final_sha}
    header["changed_files"] = 3
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/files",
        [{"status": "added", "filename": "feature.py"}, {"status": "added", "filename": "legacy.py"},
            {"status": "modified", "filename": "unrelated.py"},
        ],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 0

    refreshed = load_yaml_strict(explicit_path)
    assert refreshed["snapshot"]["status"] == "ready"
    assert refreshed["pull_request"]["changed_files"] == ["feature.py", "legacy.py", "unrelated.py"]
    assert refreshed["curation"] == before_curation

def test_in_scope_explicit_and_final_heads_validate_and_compile(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, _base_tip, explicit_sha, _final_sha = _seed_stacked_origin(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[explicit_sha], origin_url=origin_url) == 0

    manifest = load_yaml_strict(ws / "benchmark.yaml")
    for row in manifest["cases"]:
        case_id = row["case_id"]
        case = load_yaml_strict(ws / row["case_file"])
        cu.CaseEditor(ws, case_id).attest_clean()
        cu.CaseEditor(ws, case_id).mark_ready(head_sha=case["snapshot"]["original_head_sha"])

    assert validate_workspace(ws) == (0, "ready")
    lock = build.compile_workspace(ws)
    assert len(lock["cases"]) == 2

def test_import_writes_atomic_unit_and_no_file_on_failure(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)  # preflight + rest/graphql canned data for pr 101 (one head)
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
    ws = _preflight_workspace(tmp_path, fake_gh, pull_header=None)  # 404 -> fetch fails
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None)
    assert rc != 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    pr = raw["pull_requests"][0]
    assert pr["import_state"] == "fetch_failed"
    assert pr["error"]["code"] and pr["error"]["message"]
    assert pr["import_file"] is None and pr["import_sha256"] is None
    assert not (ws / "imports" / "pr-000101.json").exists()

def test_status_reflects_fetched_import_and_resolved_identity(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None)
    assert rc == 0
    st = workspace_status(ws)
    assert st.workspace_state != "empty"
    assert st.repository_identity_resolved is True

def test_cli_import_prs_drives_command(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
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

    path = ws / "cases" / case_file
    raw = load_yaml_strict(path)
    finding = {"title": "bot asks to fix the cache", "body": "please fix", "severity": "low",
        "location": {"path": "a.py", "start_line": 4, "end_line": 4},
        "provenance": {"kind": "historical", "source_ids": ["github:inline_comment:1"]},
    }
    finding["finding_id"] = derive_finding_id(finding, case_id=raw["case_id"])
    raw["curation"] = {"state": "ready", "snapshot_attested": True, "clean_attested": False, "gold_status": "findings",
        "findings": [finding], "exclusions": [], "case_exclusion": None,
    }
    raw["curation"]["task_spec_sha256"] = task_spec_digest(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def test_refresh_body_only_change_stales_gold(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # state=ready, attested
    # PR body feeds compiled task context.
    hdr = dict(_PR_HEADER)
    hdr["body"] = "EDITED body that changes compiled context"
    _seed_preflight(ws, fake_gh, pull_header=hdr)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale"        # task-input contract changed -> stale

def test_refresh_metadata_only_change_updates_checksums_without_staling(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    before = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    before_import_sha = before["source"]["import_sha256"]
    hdr = dict(_PR_HEADER)
    hdr["updated_at"] = "2026-01-02T00:00:00Z"
    hdr["html_url"] = "https://github.com/o/r/pull/101"
    _seed_preflight(ws, fake_gh, pull_header=hdr)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"          # NOT staled
    assert case["source"]["import_sha256"] != before_import_sha   # import checksum updated
    assert case["curation"]["findings"]                  # curated gold preserved
    # import_sha256 changes with fetched_at; only these assertions prove metadata propagation.
    assert case["pull_request"]["updated_at"] == "2026-01-02T00:00:00Z"
    assert case["pull_request"]["html_url"] == "https://github.com/o/r/pull/101"

def test_refresh_predate_import_metadata_change_does_not_stale(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Legacy imports lack reconstructable task inputs; only evidence changes can stale them."""

    ws = _preflight_workspace(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    # Reconstruct the legacy header without additive metadata or head.ref.
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
    hdr = dict(_PR_HEADER)
    hdr["updated_at"] = "2026-01-02T00:00:00Z"
    _seed_preflight(ws, fake_gh, pull_header=hdr)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"          # NOT staled by the metadata-only refresh
    assert case["source"]["import_sha256"] != before_import_sha   # import checksum updated
    assert case["curation"]["findings"]                  # curated gold preserved

def test_refresh_predate_canonical_format_drift_does_not_stale(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Canonicalizing legacy duplicate IDs preserves curation when GitHub content is unchanged."""

    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1, login='alice', user_type='User')])
    fake_gh._write_threads([{"id": "thread_1", "isResolved": False, "isOutdated": False,
             "subjectType": "LINE", "path": "a.py", "line": 4, "side": "RIGHT",
             "comments": {"nodes": [{"id": "c1", "databaseId": 1, "body": "please fix",
                  "author": {"login": "alice", "type": "User"}, "createdAt": "2026-01-01T00:00:00Z",
                  "url": "https://github.com/o/r/pull/101#discussion_r1"},
                 {"id": "c2", "databaseId": 2, "body": "thread-only",
                  "author": {"login": "alice", "type": "User"}, "createdAt": "2026-01-01T00:00:00Z",
                  "url": "https://github.com/o/r/pull/101#discussion_r2"},
             ]}},
        ], number=101,
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")       # state=ready, attested
    # Legacy storage duplicated db 1 across feeds and classified db 2 as thread-only.
    import_path = ws / "imports" / "pr-000101.json"
    raw = load_json_strict(import_path)
    old_evidence: list[dict[str, Any]] = []
    for e in raw["evidence"]:
        if e.get("thread_id"):
            if e.get("commit_id"):
                old_evidence.append({**e,
                                     "source_id": f"github:inline_comment:{e['database_id']}",
                                     "kind": "inline_comment"})
            # Historical GraphQL copies lacked commit anchors. Preserve that difference
            # so db 1 produces two distinct hashes and actually exercises format drift.
            old_evidence.append({**{k: v for k, v in e.items() if k not in ("commit_id", "original_commit_id")},
                "source_id": f"github:thread_comment:{e['database_id']}", "kind": "thread_comment"})
        else:
            old_evidence.append(e)
    assert len(old_evidence) == 3      # db 1 twice, db 2 once as thread_comment
    import_path.write_text(json.dumps({**raw, "evidence": old_evidence}, indent=2))
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"           # format drift must NOT stale gold
    assert case["curation"]["findings"]                   # curated gold preserved
    refreshed = load_json_strict(import_path)
    assert len(refreshed["evidence"]) == 2
    assert {e["kind"] for e in refreshed["evidence"]} == {"inline_comment"}

def test_refresh_legacy_without_original_start_line_preserves_curation(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(1, login='alice', user_type='User', line=5, original_line=5, original_start_line=4, start_line=4
            ),
        ],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    # Legacy signatures encode the absent original_start_line as None.
    import_path = ws / "imports" / "pr-000101.json"
    raw = load_json_strict(import_path)
    raw["evidence"] = [{k: v for k, v in e.items() if k != "original_start_line"} for e in raw["evidence"]]
    assert all("original_start_line" not in e for e in raw["evidence"])
    import_path.write_text(json.dumps(raw, indent=2))
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"           # NOT staled by the field addition
    assert case["curation"]["findings"]                   # curated gold preserved

def test_refresh_marks_stale_and_never_overwrites_curation(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)   # seed REST with one evidence record via the comment fixture below
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None)
    assert rc == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")   # curation.state=ready, snapshot_attested=True
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
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))


def _seed_rest(gh: Any, number: int, *, reviews: Any, comments: Any, issue_comments: Any) -> None:
    gh.set_response("GET", f"repos/o/r/pulls/{number}", _pr_header(number))
    gh.set_response("GET", f"repos/o/r/pulls/{number}/reviews", reviews)
    gh.set_response("GET", f"repos/o/r/pulls/{number}/comments", comments)
    gh.set_response("GET", f"repos/o/r/issues/{number}/comments", issue_comments)


def test_e2e_paginated_human_bot_evidence_and_no_comment_pr(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)
    _seed_identity(fake_gh)
    _seed_rest(fake_gh, 101,
        reviews=[_review(1, 'Found a bug.', login='cr[bot]', user_type='Bot', state='COMMENTED'),
            _review(2, 'Nice work.', login='carol', state='COMMENTED'),
        ], comments=[_rest_comment(7, 'please fix the cache', original_commit_id=None),
            _rest_comment(
                8, 'Order matters here.', login='dave', user_type='User', original_commit_id=None, path='b.py', line=2
            ),
        ], issue_comments=[{"id": 9, "node_id": "IC_9", "user": {"login": "carol", "type": "User"},
             "body": "question", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
             "html_url": "https://github.com/o/r/pull/101#issuecomment-9"},
        ],
    )
    fake_gh._write_threads([{"id": "thread_1", "isResolved": False, "isOutdated": False,
             "subjectType": "LINE", "path": "a.py", "line": 4, "side": "RIGHT",
             "comments": {"nodes": [{"id": "c1", "databaseId": 10, "body": "root",
                  "author": {"login": "dave", "type": "User"}, "createdAt": "2026-01-01T00:00:00Z",
                  "url": "https://github.com/o/r/pull/101#discussion_r10"},
             ]}},
        ], number=101,
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
    assert len([e for e in imp["evidence"] if e["database_id"] == 10]) == 1
    db10 = next(e for e in imp["evidence"] if e["database_id"] == 10)
    assert db10["kind"] == "inline_comment" and db10["source_id"] == "github:inline_comment:10"
    assert db10["thread_id"] == "thread_1"
    assert any(e["is_bot"] for e in imp["evidence"])      # bot author retained
    assert any(not e["is_bot"] for e in imp["evidence"])  # human author retained
    assert load_json_strict(ws / "imports/pr-000102.json")["evidence"] == []  # no-comment PR retained

def test_e2e_partial_failure_persists_ledger_and_exits_nonzero(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)
    _seed_identity(fake_gh)
    _seed_rest(fake_gh, 101, reviews=[], comments=[], issue_comments=[])
    fake_gh.set_response("GET", "repos/o/r/pulls/102", {"__error__": "API rate limit exceeded Retry-After: 1"})
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
    r = subprocess.run([sys.executable, "-m", "daydream", "benchmark", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "import-prs" in r.stdout
def test_reimport_does_not_duplicate_cases_rows(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    raw1 = load_yaml_strict(ws / "benchmark.yaml")
    assert len(raw1["cases"]) == 1
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    raw2 = load_yaml_strict(ws / "benchmark.yaml")
    ids = [c["case_id"] for c in raw2["cases"]]
    assert len(raw2["cases"]) == 1, f"cases[] grew to {len(raw2['cases'])}: {ids}"
    assert ids[0] == "pr-000101-aaaaaaaaaaaa"

def test_reimport_unchanged_evidence_preserves_curation(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case_file = "pr-000101-aaaaaaaaaaaa.yaml"
    _curate_case(ws, case_file)  # state=ready, snapshot_attested=True, findings non-empty
    before = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert before["state"] == "ready" and before["snapshot_attested"] is True and before["findings"]
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    after = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert after["state"] in ("ready", "stale"), "curation must not reset to draft"
    assert after["findings"], "curated findings must not be wiped"
    assert after["snapshot_attested"] is True, "unchanged re-import must keep attestation"

def test_refresh_unchanged_signature_preserves_curation(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case_file = "pr-000101-aaaaaaaaaaaa.yaml"
    _curate_case(ws, case_file)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    after = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert after["findings"], "curated findings must not be wiped on unchanged refresh"
    assert after["state"] != "draft", "curation must not reset to draft on unchanged refresh"

def test_refresh_derived_anchor_projection_flip_stales_curated_case(tmp_path: Path, fake_gh: FakeGh,) -> None:
    """A new derived location stales curation without raw-evidence changes; further refresh is stable."""

    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, _base_sha, authoring_sha, head_sha = _seed_anchor_origin(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(1, 'fix at head', login='alice', user_type='User', original_commit_id=authoring_sha,
                commit_id=head_sha, line=5, original_line=5, original_start_line=4
            ),
        ],
    )
    # No origin means no mirror: reproduce an import without any authoring anchors.
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    imp = load_json_strict(ws / "imports" / "pr-000101.json")
    assert all(e.get("authoring_anchor") is None for e in imp["evidence"])
    raw = load_yaml_strict(ws / "benchmark.yaml")
    case_id = raw["cases"][0]["case_id"]
    prior_case = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert prior_case["candidates"][0]["location"] is None   # pre-anchor basis
    _curate_case(ws, f"{case_id}.yaml")
    prior_findings = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["findings"]
    prior_exclusions = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["exclusions"]
    # The mirror supplies a new location basis; preserved gold must require re-approval.
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 0
    imp = load_json_strict(ws / "imports" / "pr-000101.json")
    rec = next(e for e in imp["evidence"] if e["database_id"] == 1)
    assert rec["authoring_anchor"] == {
        "version": 1, "status": "derived", "commit_id": authoring_sha, "path": "a.py", "start_line": 4, "end_line": 5,
    }
    case = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert case["curation"]["state"] == "stale"
    assert case["curation"]["snapshot_attested"] is False
    assert "task_spec_sha256" not in case["curation"]  # approval invalidated
    assert case["curation"]["findings"] == prior_findings   # curation never overwritten
    assert case["curation"]["exclusions"] == prior_exclusions
    assert case["candidates"][0]["location"] == {"path": "a.py", "start_line": 4, "end_line": 5}
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 0
    case = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert case["curation"]["state"] == "stale"
    assert case["curation"]["findings"] == prior_findings
    assert case["curation"]["exclusions"] == prior_exclusions

def test_refresh_pre_anchor_projected_location_flip_stales_without_mirror(tmp_path: Path, fake_gh: FakeGh,) -> None:
    """Dropping a legacy observed location invalidates curation while preserving its gold findings."""

    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(1, 'fix this', login='alice', user_type='User', line=5, original_line=5, original_start_line=4,
                start_line=5
            ),
        ],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    imp = load_json_strict(ws / "imports" / "pr-000101.json")
    assert all(e.get("authoring_anchor") is None for e in imp["evidence"])
    raw = load_yaml_strict(ws / "benchmark.yaml")
    case_id = raw["cases"][0]["case_id"]
    case_path = ws / "cases" / f"{case_id}.yaml"
    # Reconstruct the observed-field location persisted before authoring anchors existed.
    prior_case = load_yaml_strict(case_path)
    assert prior_case["candidates"][0]["location"] is None      # current code, anchor-less
    prior_case["candidates"][0]["location"] = {"path": "a.py", "start_line": 5, "end_line": 5}
    case_path.write_text(yaml.safe_dump(prior_case, sort_keys=False))
    _curate_case(ws, f"{case_id}.yaml")
    prior_findings = load_yaml_strict(case_path)["curation"]["findings"]
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(case_path)
    assert case["curation"]["state"] == "stale"
    assert case["curation"]["snapshot_attested"] is False
    assert case["curation"]["findings"] == prior_findings   # curation never overwritten
    assert case["candidates"][0]["location"] is None        # fresh anchor-less basis

def test_graphql_review_threads_retries_rate_limit_then_fails(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)

    calls = {"n": 0}

    def flaky_gh_api(*a: Any, **kw: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("graphql rate limited", retry_after=0.0)
        ok = {"repository": {"pullRequest": {"reviewThreads": {"nodes": [], "pageInfo": {"hasNextPage": False}}}}}
        return {"data": ok}

    monkeypatch.setattr("daydream.git_ops.gh_api", flaky_gh_api)
    monkeypatch.setattr(transport, "time", type("_T", (), {"sleep": staticmethod(lambda _s: None)})())
    threads = gi._graphql_review_threads(ws, "o/r", 101)
    assert threads == []
    assert calls["n"] == 3, "rate-limit retry should make 3 attempts"

def test_graphql_threads_replies_collect_past_100(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    _seed_empty_rest(fake_gh)
    comments = [{"id": f"c{i}", "databaseId": 2000 + i, "body": f"reply {i}",
                 "author": {"login": "eve", "type": "User"}, "createdAt": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
                 "replyTo": {"id": "c1"}, "url": f"https://github.com/o/r/pull/101#discussion_r{2000+i}"}
                for i in range(1, 251)]     # 250 replies -> 3 nested pages
    fake_gh._serve_thread_comments("thread_9", comments, page_size=100)
    fake_gh._write_threads([{"id": "thread_9", "isResolved": False,
        "isOutdated": False, "subjectType": "LINE", "path": "a.py", "line": 4,
        "side": "RIGHT", "comments": {"nodes": comments[:100]}}], number=101)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    replies = [e for e in doc.evidence if 2000 <= e.database_id <= 2250]
    assert len(replies) == 250
    assert len({e.database_id for e in replies}) == 250        # no dup
    assert [e.database_id for e in sorted(replies, key=lambda r: r.database_id)] \
           == sorted(range(2001, 2251))

def test_reconcile_inline_and_thread_into_one_record(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [_review(5, state='DISMISSED')])
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(
            10, 'root', login='dave', user_type='User', original_line=3, original_path='a.py', pull_request_review_id=5
        )])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([{"id": "thread_1", "isResolved": True,
        "isOutdated": True, "subjectType": "LINE", "path": "a.py", "line": 4, "originalLine": 3, "side": "RIGHT",
        "startSide": None, "comments": {"nodes": [{"id": "c1", "databaseId": 10, "body": "root",
             "author": {"login": "dave", "type": "User"}, "createdAt": "2026-01-01T00:00:00Z",
             "url": "https://github.com/o/r/pull/101#discussion_r10"}]}}],
        number=101)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    by_db = {e.database_id: e for e in doc.evidence}
    rec = by_db[10]
    assert rec.kind == "inline_comment"
    assert rec.source_id == "github:inline_comment:10"
    assert rec.thread_id == "thread_1" and rec.resolved is True
    assert rec.outdated is True and rec.dismissed is True      # via review 5 DISMISSED
    assert rec.review_id == "5"
    assert rec.commit_id == "a" * 40 and rec.path == "a.py"     # REST anchors kept
    assert len([e for e in doc.evidence if e.database_id == 10]) == 1
    assert not any(e.kind == "thread_comment" for e in doc.evidence)

def test_evidence_order_deterministic_across_page_sizes(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [_review(1, 'approved')])
    comments = [_rest_comment(30, 'first', login='dave', user_type='User', original_commit_id=None, line=1),
        _rest_comment(7, 'second', login='carol', user_type='User', original_commit_id=None, path='b.py', line=2,
            created_at='2026-01-02T00:00:00Z', updated_at='2026-01-02T00:00:00Z'
        ),
    ]
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", comments)
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([], number=101)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    # deterministic order: sorted by (database_id, created_at)
    assert [e.database_id for e in doc.evidence] == [1, 7, 30]
    payload = doc.fetch.payload_sha256
    doc2 = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)     # refetch: identical digest
    assert doc2.fetch.payload_sha256 == payload

def test_outdated_root_not_exact_acceptable_via_joined_record(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    # REST copy of comment 40 is OUTDATED via the joined GraphQL thread state
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(
            40, 'outdated root', login='dave', user_type='User', line=5, original_line=5, original_start_line=4
        )])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([{"id": "thread_2", "isResolved": True,
        "isOutdated": True, "subjectType": "LINE", "path": "a.py", "line": 5, "originalLine": 4, "side": "RIGHT",
        "startSide": None, "comments": {"nodes": [{"id": "c40", "databaseId": 40, "body": "outdated root",
             "author": {"login": "dave", "type": "User"}, "createdAt": "2026-01-01T00:00:00Z",
             "url": "https://github.com/o/r/pull/101#discussion_r40"}]}}],
        number=101)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
    # Supply a valid anchor so only the joined outdated flag denies exact acceptance.
    _set_anchor({e.database_id: e for e in doc.evidence}[40], commit_id="a" * 40, path="a.py", start_line=4, end_line=5)
    cands = {c.source_id: c for c in gi.project_candidates(doc, head_sha="a" * 40)}
    cand = cands["github:inline_comment:40"]
    assert cand.exact_acceptable is False
    assert cand.not_exact_reason == "outdated"

def test_fixture_matrix_evidence_preserved_and_historical(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [
        _review(1, 'Found a bug.', login='cr[bot]', user_type='Bot', state='COMMENTED'),   # non-pure review body
        _review(2, 'Nice work.', login='carol'),   # pure approval
    ])
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [
        _rest_comment(7, original_commit_id=None, updated_at='2026-01-03T00:00:00Z')])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [
        {"id": 9, "node_id": "IC_9", "user": {"login": "carol", "type": "User"},
         "body": "question", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
         "html_url": "https://github.com/o/r/pull/101#issuecomment-9"}])
    fake_gh._write_threads([], number=101)
    doc = gi.fetch_and_normalize(ws, _FETCH_IDENTITY, 101)
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
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)

    def always_limited(*a: Any, **kw: Any) -> None:
        raise RateLimitError("graphql rate limited", retry_after=0.0)

    monkeypatch.setattr("daydream.git_ops.gh_api", always_limited)
    monkeypatch.setattr(transport, "time", type("_T", (), {"sleep": staticmethod(lambda _s: None)})())
    with pytest.raises(gi._ImportRateLimitError):
        gi._graphql_review_threads(ws, "o/r", 101)

def test_corrupt_prior_import_fails_before_network(tmp_path: Path, fake_gh: FakeGh) -> None:
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, head = _seed_ready_case(tmp_path, fake_gh)     # valid prior state
    imp = next((ws / "imports").glob("*.json"))
    imp.write_bytes(b"{ corrupt json !!")                       # corrupt the persisted import
    with pytest.raises(WorkspaceCorrupt):                       # must fail, not heal to None
        gi._prior_import_state(ws, load_yaml_strict(ws / "benchmark.yaml"), 101)

def test_corrupt_prior_curation_fails_not_healed(tmp_path: Path, fake_gh: FakeGh) -> None:
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, head = _seed_ready_case(tmp_path, fake_gh)
    case = ws / "cases" / f"{case_id}.yaml"
    case.write_bytes(b"{ corrupt yaml !!")   # corruption the strict loader rejects
    with pytest.raises(WorkspaceCorrupt):
        gi._prior_import_state(ws, load_yaml_strict(ws / "benchmark.yaml"), 101)

def test_missing_prior_import_is_nonfatal_first_run(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    raw = load_yaml_strict(ws / "benchmark.yaml")
    prior = gi._prior_import_state(ws, raw, 202)
    assert prior.evidence_signature is None and prior.task_signature is None
    assert prior.document is None and prior.cases == {}
    assert prior.pinned_head is None and prior.requested_heads == []


def test_refresh_stale_clears_task_spec_approval(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)   # seed REST with one evidence comment
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")   # ready + attested (with digest, per Task 4)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale" and case["curation"]["snapshot_attested"] is False
    assert "task_spec_sha256" not in case["curation"] and "task_spec_approved_at" not in case["curation"]

# Projection signatures key by physical database_id and exclude transport metadata.


def _one_evidence() -> dict[str, Any]:
    return {"database_id": 1, "body_sha256": "a" * 64, "body": "please fix",
            "path": "a.py", "line": 4, "commit_id": "a" * 40, "outdated": False,
            "resolved": False, "dismissed": False, "state": "COMMENTED",
            "subject_type": "line", "side": "RIGHT", "start_side": None,
            "original_path": "a.py", "original_line": 4, "original_commit_id": "a" * 40,
            "author": {"login": "alice", "type": "User"}}


def _sig(ev: dict[str, Any]) -> Any:
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
    base = _one_evidence()
    dup = [{**base, "kind": "inline_comment"},      # same database_id stored twice
           {**base, "kind": "thread_comment"}]
    canon = [base]
    assert gi._evidence_signature_from_raw({"evidence": dup}) \
        == gi._evidence_signature_from_raw({"evidence": canon})

# Stale decisions combine referenced-evidence changes with PR-wide task-input changes.


def _review(db_id: int, body: str = "", *, login: str = "alice", user_type: str = "User", state: str = "APPROVED",
    commit_id: str = "a" * 40, submitted_at: str = "2026-01-01T00:00:00Z",
) -> dict[str, Any]:
    return {"id": db_id, "node_id": f"PRR_{db_id}", "user": {"login": login, "type": user_type},
            "body": body, "state": state, "commit_id": commit_id, "submitted_at": submitted_at,
            "html_url": f"https://github.com/o/r/pull/101#pullrequestreview-{db_id}"}


def _rest_comment(
    db_id: int, body: str = "please fix", *, login: str = "bot[bot]", user_type: str = "Bot", commit_id: str = "a" * 40,
    original_commit_id: str | None = "a" * 40, path: str = "a.py", line: int = 4, subject_type: str = "line",
    side: str = "RIGHT", created_at: str = "2026-01-01T00:00:00Z", updated_at: str = "2026-01-01T00:00:00Z",
    original_line: int | None = None, original_start_line: int | None = None, start_line: int | None = None,
    start_side: str | None = None, original_path: str | None = None, original_position: int | None = None,
    in_reply_to_id: int | None = None, pull_request_review_id: int | None = None,
) -> dict[str, Any]:
    """Match REST payload omission: optional keys, including original_commit_id, appear only when set."""
    comment: dict[str, Any] = {"id": db_id, "node_id": f"DIFF_{db_id}", "user": {"login": login, "type": user_type},
        "body": body, "commit_id": commit_id, "path": path, "line": line,
    }
    if original_commit_id is not None:
        comment["original_commit_id"] = original_commit_id
    for name, value in (
        ("original_line", original_line), ("original_start_line", original_start_line), ("start_line", start_line),
        ("start_side", start_side), ("original_path", original_path), ("original_position", original_position),
        ("in_reply_to_id", in_reply_to_id), ("pull_request_review_id", pull_request_review_id),
    ):
        if value is not None:
            comment[name] = value
    comment.update({"subject_type": subject_type, "side": side,
                    "created_at": created_at, "updated_at": updated_at,
                    "html_url": f"https://github.com/o/r/pull/101#discussion_r{db_id}"})
    return comment


def test_refresh_unrelated_new_comment_does_not_stale(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(1), {**_rest_comment(99), "path": "b.py", "body": "unrelated nit"}],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"       # NOT staled by an unrelated new comment
    assert case["curation"]["findings"]

def test_refresh_changed_anchor_on_referenced_evidence_stales(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1, line=7)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale"
    assert case["curation"]["findings"]               # curated findings preserved
    assert case["curation"]["snapshot_attested"] is False



def test_refresh_after_head_advance_keeps_case_id(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
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



def test_refresh_failure_preserves_linkage_and_records_attempt(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    before = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]
    fake_gh.set_response("GET", "repos/o/r/pulls/101", {"__error__": "API rate limit exceeded Retry-After: 1"})
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None)
    assert rc != 0
    after = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]
    assert after["import_state"] == "fetched"            # NOT reset to fetch_failed
    assert after["import_file"] == before["import_file"]  # last-good linkage preserved
    assert after["import_sha256"] == before["import_sha256"]
    assert after["case_ids"] == before["case_ids"]
    assert after["latest_error"]["code"] == "rate_limit"  # attempt recorded separately
    assert (ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml").exists()  # case still indexed

def test_refresh_corrupt_prior_anchor_stages_ledger_failure(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    before = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]
    import_path = ws / "imports" / "pr-000101.json"
    imp = load_json_strict(import_path)
    rec = next(e for e in imp["evidence"] if e["database_id"] == 1)
    rec["authoring_anchor"] = {"version": 1, "status": "bogus"}
    import_path.write_text(json.dumps(imp, indent=2))
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None)
    assert rc != 0
    after = load_yaml_strict(ws / "benchmark.yaml")["pull_requests"][0]
    assert after["import_state"] == "fetched"            # NOT reset to fetch_failed
    assert after["import_file"] == before["import_file"]  # last-good linkage preserved
    assert after["latest_error"]["code"] == "fetch"
    assert "authoring_anchor" in after["latest_error"]["message"]
    assert (ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml").exists()  # case still indexed

def test_refresh_unreachable_pinned_head_freezes_fails_without_clobber(tmp_path: Path, fake_gh: FakeGh) -> None:
    """An unreachable pinned head fails refresh, preserving the indexed ready case and its bundle."""

    ws = _preflight_workspace(tmp_path, fake_gh, hosts=("api.anthropic.com", "api.anthropic.com"))
    origin_url, _base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    case_id = f"pr-000101-{head_sha[:12]}"
    _curate_case(ws, f"{case_id}.yaml")
    case_path = ws / "cases" / f"{case_id}.yaml"
    before = load_yaml_strict(case_path)
    assert before["snapshot"]["status"] == "ready"
    bundle_rel = before["snapshot"]["bundle_file"]

    # Rebase the PR ref onto a branch that cannot reach the pinned head.
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
    """Invalid curation references fail refresh before they can evade the per-case stale gate."""

    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case_path = ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml"
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1

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
    """Replies are evidence-only, so gaining reply status changes the projection hash and curation basis."""

    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1

    reply = dict(_rest_comment(1))
    reply["in_reply_to_id"] = 10
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [reply])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale"       # referenced id's projection changed
    assert case["curation"]["findings"]               # curated gold preserved

def test_reimport_changed_referenced_evidence_stales(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Plain re-import applies the same per-case stale gate as refresh."""

    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1, line=7)])
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=False, origin_url=None)
    assert rc == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale"        # cannot bypass refresh semantics
    assert case["curation"]["findings"]                # curated findings preserved

def test_refresh_precanon_duplicate_db_id_verdict_is_deterministic(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Matching any legacy projection preserves curation, independent of set order."""

    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")    # references github:inline_comment:1

    # Legacy REST and GraphQL copies shared an ID but differed in commit anchors.
    import_path = ws / "imports/pr-000101.json"
    prior = load_json_strict(import_path)
    rest = next(e for e in prior["evidence"] if e["database_id"] == 1)
    thread = {k: v for k, v in rest.items() if k not in ("commit_id", "original_commit_id")}
    thread.update(kind="thread_comment", source_id="github:thread_comment:1")
    prior["evidence"] = [thread, rest]
    import_path.write_text(json.dumps(prior, indent=2))

    # Distinct hashes matter: collapsing them into a dict would make the survivor
    # depend on frozenset iteration order rather than comparing all prior projections.
    sig = gi._evidence_signature_from_raw(prior)
    assert len(sig) == 2 and len({h for _, h in sig}) == 2

    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(1)])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "ready"      # format drift must NOT stale gold
    assert case["curation"]["findings"]              # curated findings preserved
    refreshed = load_json_strict(import_path)
    assert len([e for e in refreshed["evidence"] if e["database_id"] == 1]) == 1

def test_ready_import_persists_facts_per_candidate(tmp_path: Path, fake_gh: FakeGh) -> None:
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, candidate=True)
    case = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    facts = case["prioritization"]
    assert facts["extraction_version"] == 1
    assert facts["head_sha"] == case["snapshot"]["original_head_sha"]
    sid = case["candidates"][0]["source_id"]
    (rel, delta) = (facts["candidates"][sid]["commit_relation"], facts["candidates"][sid]["anchor_delta"])
    assert rel == "at_head" and delta == "unchanged"
    assert set(facts["candidates"]) | set(facts["non_candidates"]) == {"github:inline_comment:1"}

def test_imported_status_case_has_no_facts(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments",
        [_rest_comment(1, 'fix this', login='alice', user_type='User', original_line=4, original_start_line=3)],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert "prioritization" not in case or case["prioritization"] is None

def test_fact_extraction_failure_records_unavailable_and_import_still_succeeds(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old extraction version forces a failing probe without invalidating import or curation."""

    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, candidate=True)
    import_path = ws / load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["source"]["import_file"]
    before = load_json_strict(import_path)
    raw_case = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    raw_case["prioritization"]["extraction_version"] = 0
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw_case)

    # break the mirror; the refresh freeze re-populates it from the local bare origin
    shutil.rmtree(ws / "cache" / "repository.git")

    def boom(*a: Any, **kw: Any) -> str:
        raise git_ops.GitError("injected anchor_delta failure")

    monkeypatch.setattr("daydream.benchmark.snapshot.anchor_delta", boom)
    origin_url = str(tmp_path / "origin_local.git")
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 0
    case = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    sid = case["candidates"][0]["source_id"]
    assert case["prioritization"]["candidates"][sid]["anchor_delta"] == "unavailable"
    assert case["curation"]["state"] == "draft"
    # Ignore refresh transport metadata; extraction must preserve the import payload.
    after = load_json_strict(import_path)
    for d in (before, after):
        d["fetch"] = {k: v for k, v in d["fetch"].items() if k not in ("fetched_at", "etag")}
    assert gi._payload_sha256(after) == gi._payload_sha256(before)
    assert after["evidence"] == before["evidence"]

def test_facts_absent_from_every_hash_surface(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Advisory case facts cannot alter import digests or workspace validity."""
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, candidate=True)
    case_path = ws / "cases" / f"{case_id}.yaml"
    case = load_yaml_strict(case_path)
    import_path = ws / case["source"]["import_file"]
    digest_before = gi._payload_sha256(load_json_strict(import_path))
    code_before, _ = validate_workspace(ws)

    sid = case["candidates"][0]["source_id"]
    case["prioritization"]["candidates"][sid] = {"commit_relation": "non_ancestor", "anchor_delta": "deleted"}
    storage.atomic_write_yaml(case_path, case)

    assert gi._payload_sha256(load_json_strict(import_path)) == digest_before
    code_after, _ = validate_workspace(ws)
    assert code_after == code_before

def test_refresh_reuses_persisted_facts_and_preserves_curation(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-op refresh preserves facts, curation, and snapshot without any mirror probes."""
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, candidate=True)
    case_path = ws / "cases" / f"{case_id}.yaml"
    sid = load_yaml_strict(case_path)["candidates"][0]["source_id"]
    cu.CaseEditor(ws, case_id).accept_candidate(sid)     # curator action
    before_case = load_yaml_strict(case_path)
    origin_url = str(tmp_path / "origin_local.git")

    def boom(*a: Any, **kw: Any) -> str:
        raise AssertionError("mirror probe re-ran on a no-op refresh")

    monkeypatch.setattr(sn, "commit_relation", boom)
    monkeypatch.setattr(sn, "anchor_delta", boom)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 0
    after_case = load_yaml_strict(case_path)
    assert after_case["curation"] == before_case["curation"]  # carried forward unchanged
    assert after_case["snapshot"] == before_case["snapshot"]  # pinned head/bundle intact
    assert after_case["prioritization"] == before_case["prioritization"]
    assert after_case["prioritization"]["head_sha"] == after_case["snapshot"]["original_head_sha"]

def test_reuse_gate_verifies_candidate_split(tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recompute facts when projection changes the candidate split, even with identical raw evidence."""
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, candidate=True)
    case_path = ws / "cases" / f"{case_id}.yaml"
    before = load_yaml_strict(case_path)
    sid = before["candidates"][0]["source_id"]
    assert set(before["prioritization"]["candidates"]) == {sid}

    monkeypatch.setattr(gi, "project_candidates", lambda doc, head: [])

    calls: list[int] = []
    real = gi._extract_prioritization_facts

    def recorded(*a: Any, **kw: Any) -> Any:
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(gi, "_extract_prioritization_facts", recorded)
    origin_url = str(tmp_path / "origin_local.git")
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 0
    assert calls  # the gate rejected the stale split and recomputed
    after = load_yaml_strict(case_path)
    assert after["prioritization"]["candidates"] == {}
    assert set(after["prioritization"]["non_candidates"]) == {sid}

def test_facts_version_bump_alone_never_stales(tmp_path: Path, fake_gh: FakeGh) -> None:
    from tests.test_benchmark_curation import _seed_ready_case

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, candidate=True)
    case_path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(case_path)
    raw["prioritization"]["extraction_version"] = 0            # simulate an older facts version
    raw["curation"]["state"] = "ready"
    storage.atomic_write_yaml(case_path, raw)
    origin_url = str(tmp_path / "origin_local.git")
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=origin_url) == 0
    refreshed = load_yaml_strict(case_path)
    assert refreshed["curation"]["state"] == "ready"           # version bump alone does not stale

    assert refreshed["prioritization"]["extraction_version"] == EXTRACTION_VERSION

def test_equivalent_imports_produce_identical_facts_and_rank(tmp_path: Path, fake_gh: FakeGh) -> None:
    from tests.test_benchmark_curation import _seed_ready_case_mixed

    ws1, case_id1, _ = _seed_ready_case_mixed(tmp_path, fake_gh)
    ws2, case_id2, _ = _seed_ready_case_mixed(tmp_path, fake_gh)
    v1, v2 = cu.get_case(ws1, case_id1), cu.get_case(ws2, case_id2)
    assert v1["prioritized_evidence"] == v2["prioritized_evidence"]
    assert load_yaml_strict(ws1 / "cases" / f"{case_id1}.yaml")["prioritization"] == \
        load_yaml_strict(ws2 / "cases" / f"{case_id2}.yaml")["prioritization"]


@pytest.mark.parametrize("gate, code", [
    ("user", "auth_failed"), ("repository", "no_access"), ("git", "git_preflight_failed"),
])
@pytest.mark.parametrize("cli", [False, True])
@pytest.mark.parametrize("credential", [
    "api_key=private-preflight-canary", "Authorization: Bearer private-preflight-canary",
    "https://operator:private-preflight-canary@github.com/o/r",
])
def test_preflight_process_errors_are_private_before_direct_or_cli_reporting(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    gate: str, code: str, cli: bool, credential: str,
) -> None:
    import traceback

    ws = tmp_path / "ws"
    _seed_manifest(ws)
    before = (ws / "benchmark.yaml").read_bytes()
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    process_run = subprocess.run
    calls: list[list[str]] = []

    def process_error(args: Any, *pargs: Any, **kwargs: Any) -> Any:
        calls.append(list(args))
        failed = (
            (gate == "user" and args[:3] == ["gh", "api", "user"])
            or (gate == "repository" and args[:3] == ["gh", "repo", "view"])
            or (gate == "git" and args[0] == "git" and "ls-remote" in args)
        )
        if failed:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr=f"remote access denied: {credential}")
        return process_run(args, *pargs, **kwargs)

    monkeypatch.setattr("daydream.git_ops.process.subprocess.run", process_error)
    if cli:
        with pytest.raises(SystemExit) as exit_info:
            top_cli.main(["benchmark", "import-prs", str(ws), "--pr", "101"])
        assert exit_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        diagnostic = captured.err
        assert code in diagnostic
    else:
        with pytest.raises(gi.PreflightError) as failure:
            gi.preflight(ws, pr_count=1)
        assert failure.value.code == code
        diagnostic = "".join(traceback.format_exception(failure.value))
        assert "private-preflight-canary" not in failure.value.message
        captured = capsys.readouterr()
        assert captured.out == ""
    assert "remote access denied" in diagnostic
    assert "private-preflight-canary" not in diagnostic
    assert (ws / "benchmark.yaml").read_bytes() == before
    assert not (ws / "runtime/preflight.json").exists()
    assert not list((ws / "imports").iterdir())
    assert calls[0] == ["gh", "auth", "status", "--hostname", "github.com"]
    if gate == "git":
        argv = calls[-1]
        assert any(a.startswith("credential.helper=") for a in argv)
        assert "https://github.com/o/r.git" in argv and credential not in " ".join(argv)


def test_import_retains_preflight_repository_identity_through_acquisition(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)
    fetch = gi._fetch_with_retry

    def replace_manifest_after_fetch(root: Path, repo: str, number: int) -> dict[str, Any]:
        header = fetch(root, repo, number)
        manifest = storage.load_yaml_strict(root / "benchmark.yaml")
        manifest["source"].update(
            repository="other/repository", repository_id="R_changed", visibility="public",
        )
        (root / "benchmark.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
        return header

    monkeypatch.setattr(gi, "_fetch_with_retry", replace_manifest_after_fetch)
    assert gi.run_import_prs(ws, [101], origin_url=None) == 0
    imported = storage.load_json_strict(ws / "imports" / "pr-000101.json")
    assert imported["repository"] == {
        "id": _REPO_ID, "name_with_owner": "o/r", "visibility": "private",
    }
    assert storage.load_yaml_strict(ws / "benchmark.yaml")["source"]["repository"] == "o/r"
    ledger = storage.load_json_strict(ws / "runtime" / "preflight.json")
    assert ledger["repository_id"] == imported["repository"]["id"]
    assert (ws / "runtime" / "preflight.json").stat().st_mode & 0o777 == 0o600
