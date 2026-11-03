"""Import-PR integration coverage through the hermetic ``fake_gh`` router.

Snapshot freezes use a real local bare origin; no network is required.
"""
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from daydream import git_ops
from daydream.benchmark import github_import as gi, schema
from daydream.benchmark.cli import _handle_benchmark_status
from daydream.benchmark.harbor.build import task_spec_digest
from daydream.benchmark.schema import Location, case_id_for, derive_finding_id
from daydream.benchmark.storage import load_json_strict, load_yaml_strict, sha256_file
from daydream.benchmark.workspace import init_workspace, validate_workspace, workspace_status
from daydream.git_ops import RateLimitError
from daydream.pr_review import finding_marker
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


def _file_level_comment_exactness_gated_by_anchor() -> None:
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

def _derive_one_anchor_inverted_range_and_bad_path_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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

def _candidate_projection_right_file_body_left(tmp_path: Path, fake_gh: FakeGh) -> None:
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
    doc_gi = gi.fetch_and_normalize(ws, "o/r", 101)
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

def _parse_targets_dedupes_and_orders(tmp_path: Path) -> None:
    pf = tmp_path / "prs.txt"
    pf.write_text("42\nhttps://github.com/o/r/pull/9\n7\n42\n")
    targets = gi.parse_import_targets(
        pr_args=["https://github.com/o/r/pull/9", "7"], pr_files=[pf], heads=["abc" * 13 + "1", "abc" * 13 + "2"],
    )
    assert targets.pr_numbers == [9, 7, 42]     # stable: CLI order then file order; dupes collapsed
    assert targets.requested_heads == ["final", "abc" * 13 + "1", "abc" * 13 + "2"]  # 'final' always present

def _parse_head_pr_sha_grammar_and_binding() -> None:
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

def _seed_manifest(ws: Path) -> None:
    """Initialize unresolved source o/r, preserving any caller-supplied workspace and host pins."""

    if (ws / "benchmark.yaml").exists():
        return
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])


def _preflight_reverifies_identity_on_every_run_and_fails_closed(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)                                  # unresolved Source (repository=repo)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value=dict(_REPO_VIEW))
    out = gi.preflight(ws, pr_count=2)
    assert out.login == "octocat" and out.repository_id == _REPO_ID and out.visibility == "private"
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == _REPO_ID and raw["source"]["visibility"] == "private"

    assert gi.preflight(ws, pr_count=1).repository_id == _REPO_ID
    before = (ws / "benchmark.yaml").read_bytes()
    fake_gh.set_response("repo-view-full", value={**_REPO_VIEW, "id": "R_kgDDIFFERENT"})
    with pytest.raises(gi.PreflightError) as ei:
        gi.preflight(ws, pr_count=1)
    assert ei.value.code == "repo_mismatch"
    raw = load_yaml_strict(ws / "benchmark.yaml")
    assert raw["source"]["repository_id"] == _REPO_ID    # unchanged: no mutation staged
    assert (ws / "benchmark.yaml").read_bytes() == before

def _preflight_rejects_numeric_node_id(tmp_path: Path, fake_gh: FakeGh, node_id: Any = "12345") -> None:
    ws = tmp_path / "ws"
    _seed_manifest(ws)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value={**_REPO_VIEW, "id": node_id})
    with pytest.raises(gi.PreflightError) as ex:
        gi.preflight(ws, pr_count=1)
    assert ex.value.code == "repo_unresolved"

def _status_reports_last_preflight_verification(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
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
            ), _rest_comment(3, 'inverted range', login='alice', user_type='User', original_commit_id=authoring_sha,
                commit_id=head_sha, path='new.py', original_line=4, original_start_line=8
            ),
        ],
    )
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    imp = load_json_strict(ws / "imports" / "pr-000101.json")
    observed = {e["database_id"]: e for e in imp["evidence"]}
    assert observed[1]["original_commit_id"] == head_sha
    assert (observed[1]["original_start_line"], observed[1]["original_line"]) == (4, 5)
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
    assert anchors[3] == {"version": 1, "status": "range-unavailable",
        "commit_id": None, "path": None, "start_line": None, "end_line": None,
    }

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
    _bundle_retirement_preserves_a_ready_shared_reference(ws, explicit_sha)

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

def _bundle_retirement_preserves_a_ready_shared_reference(ws: Path, explicit_sha: str) -> None:
    """A transitioned case cannot retire a bundle another ready case still references."""
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
    manifest = copy.deepcopy(manifest)
    manifest["cases"].append({"case_id": alias_id, "pr_number": 101, "case_file": f"cases/{alias_id}.yaml"})

    transitioned = copy.deepcopy(explicit)
    transitioned["snapshot"]["status"] = "unreplayable"
    assert gi._retired_snapshot_bundles(ws, manifest, 101, [(explicit_id, f"cases/{explicit_id}.yaml", transitioned)],
    ) == []
    alias_path.unlink()  # unindexed case files are corruption for the following refresh


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
    assert [(e["database_id"], e["kind"], e["authoring_anchor"]) for e in imp["evidence"]] == [
        (1, "inline_comment", None),
    ]
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


def _graphql_threads_replies_collect_past_100(tmp_path: Path, fake_gh: FakeGh) -> None:
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
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    replies = [e for e in doc.evidence if 2000 <= e.database_id <= 2250]
    assert len(replies) == 250
    assert len({e.database_id for e in replies}) == 250        # no dup
    assert [e.database_id for e in sorted(replies, key=lambda r: r.database_id)] \
           == sorted(range(2001, 2251))

def _reconcile_inline_and_thread_into_one_record(tmp_path: Path, fake_gh: FakeGh) -> None:
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
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
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

def _outdated_root_not_exact_acceptable_via_joined_record(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws = _fetch_workspace(tmp_path)
    marker = finding_marker("f" * 64)
    fake_gh.set_response("GET", "repos/o/r/pulls/101", _PR_HEADER)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    # REST copy of comment 40 is OUTDATED via the joined GraphQL thread state
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [_rest_comment(
            40, marker, login='dave', user_type='User', line=5, original_line=5, original_start_line=4
        )])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])
    fake_gh._write_threads([{"id": "thread_2", "isResolved": True,
        "isOutdated": True, "subjectType": "LINE", "path": "a.py", "line": 5, "originalLine": 4, "side": "RIGHT",
        "startSide": None, "comments": {"nodes": [{"id": "c40", "databaseId": 40, "body": marker,
             "author": {"login": "dave", "type": "User"}, "createdAt": "2026-01-01T00:00:00Z",
             "url": "https://github.com/o/r/pull/101#discussion_r40"}]}}],
        number=101)
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
    # Supply a valid anchor so only the joined outdated flag denies exact acceptance.
    _set_anchor({e.database_id: e for e in doc.evidence}[40], commit_id="a" * 40, path="a.py", start_line=4, end_line=5)
    cands = {c.source_id: c for c in gi.project_candidates(doc, head_sha="a" * 40)}
    cand = cands["github:inline_comment:40"]
    assert cand.title == cand.body == "" and doc.evidence[0].body == marker
    assert cand.exact_acceptable is False
    assert cand.not_exact_reason == "outdated"

def _graphql_review_threads_records_rate_limit_after_retries(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _preflight_workspace(tmp_path, fake_gh)

    def always_limited(*a: Any, **kw: Any) -> None:
        raise RateLimitError("graphql rate limited", retry_after=0.0)

    monkeypatch.setattr("daydream.git_ops.gh_api", always_limited)
    monkeypatch.setattr(gi, "time", type("_T", (), {"sleep": staticmethod(lambda _s: None)})())
    with pytest.raises(gi._ImportRateLimitError):
        gi._graphql_review_threads(ws, "o/r", 101)

# Projection signatures key by physical database_id and exclude transport metadata.


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


def test_import_population_preserves_curated_refresh_and_acquisition_outcomes(
    tmp_path: Path, fake_gh: FakeGh,
) -> None:
    """One locked importer owns first acquisition, curated refresh and ordinary reimport."""
    ws = tmp_path / "ws"
    _seed_manifest(ws)
    _seed_identity(fake_gh)
    curated_numbers = list(range(101, 117))
    inventories = {
        201: (2, [{"status": "modified", "filename": "a.py"}]),
        202: (3001, []),
        203: (1, [{"status": "renamed", "filename": "new.py"}]),
        204: (1, [{"status": "modified", "filename": "new.py", "previous_filename": "old.py"}]),
        205: (1, [{"filename": "a.py"}]),
        206: (1, [{"status": "invented", "filename": "a.py"}]),
        207: (1, [{"status": 17, "filename": "a.py"}]),
        208: (2, [{"status": "modified", "filename": "a.py"}, {"status": "modified", "filename": "a.py"}]),
        209: (1, [{"status": "modified", "filename": "../escape.py"}]),
        210: (2, [{"status": "modified", "filename": "src/a:b.py"},
                  {"status": "renamed", "filename": "src/new name.py", "previous_filename": r"src\old.py"}]),
        211: (0, []),
    }
    inventory_errors = {
        201: "PR changed_files inventory count mismatch: header=2, rows=1",
        202: "PR changed_files count 3001 exceeds the 3000-file API inventory limit",
        203: "PR changed_files renamed row 0 is missing previous_filename",
        204: "PR changed_files row 0 has unexpected previous_filename",
        205: "PR changed_files row 0 has missing or unsupported status",
        206: "PR changed_files row 0 has missing or unsupported status",
        207: "PR changed_files row 0 has missing or unsupported status",
        208: "PR changed_files inventory repeats filename 'a.py'",
        209: "PR changed_files row 0 has invalid filename: Git tree path contains an invalid component: '../escape.py'",
    }
    numbers = [*curated_numbers, *inventories]
    headers = {n: copy.deepcopy(_pr_header(n)) for n in numbers}
    comments = {n: [_rest_comment(1)] for n in curated_numbers}
    for n in (101, 102, 104, 112, 113):
        comments[n] = []
    comments[105] = [_rest_comment(1, login="alice", user_type="User")]
    comments[106] = [_rest_comment(1, line=5, original_line=5, start_line=4, original_start_line=4)]
    comments[110] = [_rest_comment(1, line=5, original_line=5, start_line=5, original_start_line=4)]
    for n in numbers:
        headers[n]["html_url"] = f"https://github.com/o/r/pull/{n}"
        _seed_rest(fake_gh, n, reviews=[], comments=comments.get(n, []), issue_comments=[])
        if n in inventories:
            headers[n]["changed_files"] = inventories[n][0]
            fake_gh.set_response("GET", f"repos/o/r/pulls/{n}/files", inventories[n][1])
        fake_gh.set_response("GET", f"repos/o/r/pulls/{n}", headers[n])
    fake_gh._write_threads([{
        "id": "thread_105", "isResolved": False, "isOutdated": False,
        "subjectType": "LINE", "path": "a.py", "line": 4, "side": "RIGHT",
        "comments": {"nodes": [
            {"id": "c1", "databaseId": 1, "body": "please fix", "author": {"login": "alice", "type": "User"},
             "createdAt": "2026-01-01T00:00:00Z", "url": "https://github.com/o/r/pull/105#discussion_r1"},
            {"id": "c2", "databaseId": 2, "body": "thread-only", "author": {"login": "alice", "type": "User"},
             "createdAt": "2026-01-01T00:00:00Z", "url": "https://github.com/o/r/pull/105#discussion_r2"},
        ]},
    }], number=105)
    requested = {n: ["final", "a" * 40] for n in inventories}
    assert gi.run_import_prs(ws, pr_numbers=numbers, pr_heads=requested, origin_url=None) == 1
    first = load_yaml_strict(ws / "benchmark.yaml")
    first_rows = {row["number"]: row for row in first["pull_requests"]}
    first_good = [*curated_numbers, 210, 211]
    assert {n: row["import_state"] for n, row in first_rows.items()} == {
        **dict.fromkeys(first_good, "fetched"), **dict.fromkeys(inventory_errors, "fetch_failed")}
    assert {n: row["error"] for n, row in first_rows.items() if row["error"] is not None} == {
        n: {"code": "fetch", "message": message} for n, message in inventory_errors.items()}
    for n in inventory_errors:
        assert first_rows[n]["import_file"] is first_rows[n]["import_sha256"] is None
        assert first_rows[n]["case_ids"] == [] and first_rows[n]["latest_error"] is None
        assert not (ws / f"imports/pr-{n:06d}.json").exists()
        assert list((ws / "cases").glob(f"pr-{n:06d}-*.yaml")) == []
    initial_imports = {n: load_json_strict(ws / f"imports/pr-{n:06d}.json") for n in first_good}
    for n in first_good:
        case_id = f"pr-{n:06d}-aaaaaaaaaaaa"
        case = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
        assert first_rows[n]["case_ids"] == [case_id] and case["snapshot"]["status"] == "imported"
        assert case.get("prioritization") is None
        assert case["pull_request"] == initial_imports[n]["pull_request"]
        assert case["source"]["import_sha256"] == first_rows[n]["import_sha256"] == sha256_file(
            ws / f"imports/pr-{n:06d}.json")
    assert initial_imports[101]["pull_request"]["changed_files"] is None
    assert initial_imports[210]["pull_request"]["changed_files"] == ["src/a:b.py", "src/new name.py", r"src\old.py"]
    assert initial_imports[211]["pull_request"]["changed_files"] == []
    assert [row["case_id"] for row in first["cases"]] == [f"pr-{n:06d}-aaaaaaaaaaaa" for n in first_good]

    # Historical authoring inputs remain raw; the importer must decide each transition.
    for n in curated_numbers:
        _curate_case(ws, f"pr-{n:06d}-aaaaaaaaaaaa.yaml")
        path = ws / "cases" / f"pr-{n:06d}-aaaaaaaaaaaa.yaml"
        raw = load_yaml_strict(path)
        raw["curation"]["task_spec_approved_at"] = "2026-01-03T00:00:00Z"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
    legacy = copy.deepcopy(initial_imports[104])
    for key in ("body", "html_url", "title_sha256", "body_sha256", "merged_at", "closed_at"):
        legacy["pull_request"].pop(key, None)
    legacy["pull_request"]["head"].pop("ref")
    (ws / "imports/pr-000104.json").write_text(json.dumps(legacy, indent=2))
    old = copy.deepcopy(initial_imports[105])
    rest = next(e for e in old["evidence"] if e["database_id"] == 1)
    old["evidence"] = [rest, *[
        {**{k: v for k, v in e.items() if k not in ("commit_id", "original_commit_id")},
         "source_id": f"github:thread_comment:{e['database_id']}", "kind": "thread_comment"}
        for e in old["evidence"]]]
    assert len(old["evidence"]) == 3
    assert len({h for identity, h in gi._evidence_signature_from_raw(old) if identity == 1}) == 2
    (ws / "imports/pr-000105.json").write_text(json.dumps(old, indent=2))
    legacy_range = copy.deepcopy(initial_imports[106])
    assert legacy_range["evidence"][0]["original_start_line"] == 4
    assert legacy_range["evidence"][0]["original_line"] == legacy_range["evidence"][0]["line"] == 5
    for record in legacy_range["evidence"]:
        record.pop("original_start_line")
    assert all("original_start_line" not in record for record in legacy_range["evidence"])
    (ws / "imports/pr-000106.json").write_text(json.dumps(legacy_range, indent=2))
    path = ws / "cases/pr-000110-aaaaaaaaaaaa.yaml"
    old_case = load_yaml_strict(path)
    assert old_case["candidates"][0]["location"] is None
    assert all(e.get("authoring_anchor") is None for e in initial_imports[110]["evidence"])
    old_case["candidates"][0]["location"] = {"path": "a.py", "start_line": 5, "end_line": 5}
    path.write_text(yaml.safe_dump(old_case, sort_keys=False))
    broken = copy.deepcopy(initial_imports[114])
    broken["evidence"][0]["authoring_anchor"] = {"version": 1, "status": "bogus"}
    (ws / "imports/pr-000114.json").write_text(json.dumps(broken, indent=2))
    path = ws / "cases/pr-000115-aaaaaaaaaaaa.yaml"
    invalid_reference = load_yaml_strict(path)
    invalid_reference["curation"]["findings"][0]["provenance"]["source_ids"] = [
        "https://github.com/o/r/pull/115#discussion_r1"]
    path.write_text(yaml.safe_dump(invalid_reference, sort_keys=False))
    curated_before = {n: load_yaml_strict(ws / f"cases/pr-{n:06d}-aaaaaaaaaaaa.yaml")["curation"]
                      for n in curated_numbers}
    failed_bytes = {n: ((ws / f"imports/pr-{n:06d}.json").read_bytes(),
                        (ws / f"cases/pr-{n:06d}-aaaaaaaaaaaa.yaml").read_bytes()) for n in (113, 114, 115)}
    headers[102]["body"] = "EDITED body that changes compiled context"
    headers[103].update(updated_at="2026-01-02T00:00:00Z", html_url="https://github.com/o/r/pull/103#updated")
    headers[104]["updated_at"] = "2026-01-02T00:00:00Z"
    headers[112]["head"] = {"ref": "feature/cache", "sha": "b" * 40}
    headers[201]["changed_files"] = 1
    for n in (102, 103, 104, 112, 201):
        fake_gh.set_response("GET", f"repos/o/r/pulls/{n}", headers[n])
    comments[103][0].update(updated_at="2026-01-02T00:00:00Z",
                            html_url="https://github.com/o/r/pull/103#discussion_r1-updated")
    comments[107] = []
    comments[108].append(_rest_comment(99, "unrelated nit", path="b.py"))
    comments[109][0]["in_reply_to_id"] = 10
    comments[111][0]["line"] = 7
    for n in (103, 107, 108, 109, 111):
        fake_gh.set_response("GET", f"repos/o/r/pulls/{n}/comments", comments[n])
    fake_gh.set_response("GET", "repos/o/r/pulls/113", {"__error__": "API rate limit exceeded Retry-After: 0"})
    fake_gh.set_response("GET", "repos/o/r/pulls/201/files", [{"status": "modified", "filename": "a.py"}])
    assert gi.run_import_prs(ws, pr_numbers=numbers, pr_heads=requested, refresh=True, origin_url=None) == 1
    refreshed = load_yaml_strict(ws / "benchmark.yaml")
    rows = {row["number"]: row for row in refreshed["pull_requests"]}
    assert {n: row["import_state"] for n, row in rows.items()} == {
        **dict.fromkeys([*curated_numbers, 201, 210, 211], "fetched"),
        **dict.fromkeys(range(202, 210), "fetch_failed")}
    assert {n: row["error"] for n, row in rows.items() if row["error"] is not None} == {
        n: {"code": "fetch", "message": message} for n, message in inventory_errors.items() if n != 201}
    assert {n: row["latest_error"] for n, row in rows.items() if row["latest_error"] is not None} == {
        113: {"code": "rate_limit", "message":
              "gh api repos/o/r/pulls/113 rate limited: API rate limit exceeded Retry-After: 0"},
        114: {"code": "fetch", "message":
              "prior import evidence record has an invalid authoring_anchor or a missing database_id"},
        115: {"code": "fetch", "message":
              "curation references non-canonical source_id 'https://github.com/o/r/pull/115#discussion_r1'"},
    }
    for n in (113, 114, 115):
        assert {**rows[n], "latest_error": None} == first_rows[n]
        assert ((ws / f"imports/pr-{n:06d}.json").read_bytes(),
                (ws / f"cases/pr-{n:06d}-aaaaaaaaaaaa.yaml").read_bytes()) == failed_bytes[n]
    after_cases = {n: load_yaml_strict(ws / f"cases/pr-{n:06d}-aaaaaaaaaaaa.yaml") for n in curated_numbers}
    assert {n: (case["curation"]["state"], case["curation"]["snapshot_attested"],
                "task_spec_sha256" in case["curation"], "task_spec_approved_at" in case["curation"])
            for n, case in after_cases.items()} == {
        101: ("ready", True, True, True), 102: ("stale", False, False, False),
        103: ("ready", True, True, True), 104: ("ready", True, True, True),
        105: ("ready", True, True, True), 106: ("ready", True, True, True),
        107: ("stale", False, False, False), 108: ("ready", True, True, True),
        109: ("stale", False, False, False), 110: ("stale", False, False, False),
        111: ("stale", False, False, False), 112: ("ready", True, True, True),
        113: ("ready", True, True, True), 114: ("ready", True, True, True),
        115: ("ready", True, True, True), 116: ("ready", True, True, True),
    }
    transitions = {"state", "snapshot_attested", "task_spec_sha256", "task_spec_approved_at"}
    assert {n: {k: v for k, v in case["curation"].items() if k not in transitions}
            for n, case in after_cases.items()} == {
        n: {k: v for k, v in curated_before[n].items() if k not in transitions} for n in curated_numbers}
    ready = (101, 103, 104, 105, 106, 108, 112, 113, 114, 115, 116)
    assert {n: after_cases[n]["curation"] for n in ready} == {n: curated_before[n] for n in ready}
    good = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 116, 201, 210, 211]
    imports = {n: load_json_strict(ws / f"imports/pr-{n:06d}.json") for n in good}
    for n in good:
        case_id = f"pr-{n:06d}-aaaaaaaaaaaa"
        case = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
        assert rows[n]["import_file"] == f"imports/pr-{n:06d}.json"
        assert rows[n]["requested_heads"] == (["final", "a" * 40] if n in inventories else ["final"])
        assert rows[n]["case_ids"] == [case_id] and case["snapshot"]["status"] == "imported"
        assert case["source"]["import_sha256"] == rows[n]["import_sha256"] == sha256_file(ws / rows[n]["import_file"])
        assert case.get("prioritization") is None
    for n in (103, 104):
        assert after_cases[n]["source"]["import_sha256"] != first_rows[n]["import_sha256"]
        assert imports[n]["fetch"]["payload_sha256"] != initial_imports[n]["fetch"]["payload_sha256"]
        assert after_cases[n]["pull_request"]["updated_at"] == "2026-01-02T00:00:00Z"
    assert after_cases[103]["pull_request"]["html_url"] == "https://github.com/o/r/pull/103#updated"
    assert (imports[103]["evidence"][0]["updated_at"], imports[103]["evidence"][0]["url"]) == (
        "2026-01-02T00:00:00Z", "https://github.com/o/r/pull/103#discussion_r1-updated")
    assert gi._evidence_signature_from_raw(imports[103]) == gi._evidence_signature_from_raw(initial_imports[103])
    assert imports[102]["evidence"] == initial_imports[102]["evidence"]
    assert after_cases[102]["pull_request"]["body"] == "EDITED body that changes compiled context"
    assert [(e["database_id"], e["kind"]) for e in imports[105]["evidence"]] == [
        (1, "inline_comment"), (2, "inline_comment")]
    assert imports[106]["evidence"][0]["original_start_line"] == 4
    assert {e["database_id"] for e in imports[108]["evidence"]} == {1, 99}
    assert after_cases[109]["candidates"] == [] and imports[109]["evidence"][0]["reply_to_id"] == "10"
    assert imports[110]["evidence"] == initial_imports[110]["evidence"]
    assert after_cases[110]["candidates"][0]["location"] is None
    assert imports[111]["evidence"][0]["line"] == 7
    assert imports[112]["pull_request"]["head"]["sha"] == after_cases[112]["snapshot"]["original_head_sha"] == "a" * 40
    assert {n: imports[n]["pull_request"]["changed_files"] for n in (101, 201, 210, 211)} == {
        101: None, 201: ["a.py"], 210: ["src/a:b.py", "src/new name.py", r"src\old.py"], 211: []}
    assert rows[201]["error"] is rows[201]["latest_error"] is None
    for n in range(202, 210):
        assert not (ws / f"imports/pr-{n:06d}.json").exists()
        assert list((ws / "cases").glob(f"pr-{n:06d}-*.yaml")) == []
    index = [{"case_id": f"pr-{n:06d}-aaaaaaaaaaaa", "pr_number": n,
              "case_file": f"cases/pr-{n:06d}-aaaaaaaaaaaa.yaml"} for n in [*curated_numbers, 201, 210, 211]]
    assert refreshed["cases"] == index

    # Ordinary reimport must preserve an unchanged case and stale a fresh ready case.
    assert after_cases[116]["curation"]["state"] == "ready"
    comments[116][0]["line"] = 7
    fake_gh.set_response("GET", "repos/o/r/pulls/116/comments", comments[116])
    assert gi.run_import_prs(ws, pr_numbers=[101, 116], heads=["final"], origin_url=None) == 0
    last = load_yaml_strict(ws / "benchmark.yaml")
    assert last["cases"] == index
    assert load_yaml_strict(ws / "cases/pr-000101-aaaaaaaaaaaa.yaml")["curation"] == curated_before[101]
    final = load_yaml_strict(ws / "cases/pr-000116-aaaaaaaaaaaa.yaml")["curation"]
    assert final["state"] == "stale" and final["snapshot_attested"] is False
    assert final["findings"] == curated_before[116]["findings"]
    assert "task_spec_sha256" not in final and "task_spec_approved_at" not in final
    files_calls = [call for call in fake_gh.calls("GET") if call.endpoint.endswith("/files")]
    assert {call.endpoint: sum(c.endpoint == call.endpoint for c in files_calls) for call in files_calls} == {
        f"repos/o/r/pulls/{n}/files": 2 for n in inventories}
    assert all("--paginate" in (call.argv or []) for call in files_calls)


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


def test_target_parsing_and_anchor_projection_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _parse_targets_dedupes_and_orders(tmp_path)
    _parse_head_pr_sha_grammar_and_binding()
    _file_level_comment_exactness_gated_by_anchor()
    _derive_one_anchor_inverted_range_and_bad_path_fail_closed(tmp_path, monkeypatch)
    missing = gi._derive_one_anchor(_evidence_record(original_commit_id=None), tmp_path / "mirror.git", "a" * 40)
    assert missing.status == "history-unavailable"
    # Persisted facts are reused only for the same extraction version and candidate split.
    prior = {"extraction_version": schema.EXTRACTION_VERSION, "head_sha": "a" * 40, "candidates": {"x": {}},
             "non_candidates": {}}
    for stale, candidates in (({**prior, "extraction_version": "old"}, {"x"}), (prior, {"y"})):
        assert gi._reuse_prior_facts(stale, "a" * 40, set(), candidates, {"x", "y"}) is None


def test_preflight_reverifies_identity_and_status_reports_it(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("status", "reverify", "numeric"):
        (tmp_path / name).mkdir()
    _status_reports_last_preflight_verification(tmp_path / "status", fake_gh, capsys)
    _preflight_reverifies_identity_on_every_run_and_fails_closed(tmp_path / "reverify", fake_gh)
    _preflight_rejects_numeric_node_id(tmp_path / "numeric", fake_gh)


def test_fetch_normalizes_rest_and_graphql_evidence_into_projected_candidates(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every served GraphQL query is also validated by fake_gh against the checked-in schema subset."""
    for check in (_candidate_projection_right_file_body_left, _graphql_threads_replies_collect_past_100,
                  _reconcile_inline_and_thread_into_one_record, _outdated_root_not_exact_acceptable_via_joined_record):
        root = tmp_path / check.__name__
        root.mkdir()
        check(root, fake_gh)
    root = tmp_path / "rate-limit"
    root.mkdir()
    with pytest.raises(git_ops.GitError, match="pulls/999 failed"):   # an ordinary gh failure is not retried
        gi._fetch_with_retry(root, "o/r", 999)
    _graphql_review_threads_records_rate_limit_after_retries(root, fake_gh, monkeypatch)
