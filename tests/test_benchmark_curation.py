"""Tests for the UI-independent golden-review curation service.

Covers the real-path seed (a genuine frozen ``ready`` workspace built from a
real local bare origin), the head-tree line-count read source (a shared bare
mirror via ``git cat-file blob <head>:<path>``), and the full derivation /
rejection / transition surface of :mod:`daydream.benchmark.curation`.
"""

import os
import subprocess

import pytest
import yaml

from daydream import git_ops

# Deterministic seed identity so a local bare origin's commits are stable and
# reproducible (mirrors tests/test_benchmark_import_prs.py::_SEED_ENV).
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


def _seed_manifest(ws):
    """Build an initialized private workspace with an unresolved Source (o/r)."""
    from daydream.benchmark.workspace import init_workspace

    if (ws / "benchmark.yaml").exists():
        return
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])


def _seed_preflight(ws, fake_gh, *, pull_header=_PR_HEADER):
    """Seed an unresolved workspace + canned preflight/REST data for pr 101."""
    _seed_manifest(ws)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response(
        "repo-view-full",
        value={"id": 5, "nameWithOwner": "o/r",
               "url": "https://github.com/o/r", "visibility": "PRIVATE",
               "defaultBranchRef": {"name": "main"}},
    )
    fake_gh.set_response("GET", "repos/o/r/pulls/101", pull_header)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", [])
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", [])


def _seed_git(repo, *args: str, check: bool = True, env: dict[str, str] | None = None) -> str:
    """Run git in *repo*, returning stripped stdout."""
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        env={**os.environ, **env} if env else os.environ.copy(), check=check,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _seed_write(repo, name: str, content: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _seed_git(repo, "add", name)


def _seed_commit(repo, message: str) -> str:
    _seed_git(repo, "commit", "-m", message, env=_SEED_ENV)
    return _seed_git(repo, "rev-parse", "HEAD")


def _seed_local_origin(tmp_path, fake_gh, *, lines: int = 3) -> tuple[str, str, str]:
    """Build a real local bare origin whose base/head are the PR's SHAs.

    The feature head adds ``feature.py`` with exactly *lines* lines (line i is
    ``f"LINE {i}\\n"``), so the frozen head file's line count is deterministic
    for the location-vs-head assertions. Returns ``(origin_url, base_sha, head_sha)``.
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
    _seed_write(repo, "feature.py", "".join(f"LINE {i}\n" for i in range(1, lines + 1)))
    head_sha = _seed_commit(repo, "feature")
    bare = tmp_path / "origin_local.git"
    if bare.exists():
        _sh.rmtree(bare)
    bare.mkdir()
    _seed_git(bare, "init", "--bare")
    _seed_git(repo, "remote", "add", "origin", str(bare))
    _seed_git(repo, "push", "origin", "main:main")
    _seed_git(repo, "push", "origin", f"{head_sha}:refs/pull/101/head", check=False)
    # Seed the preflight identity + repo-access responses (idempotent — this
    # helper is used directly by the spike test, which does not call
    # ``_seed_preflight``), then re-seed the canned PR header so
    # base.sha/head.sha are the real origin SHAs.
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response(
        "repo-view-full",
        value={"id": 5, "nameWithOwner": "o/r",
               "url": "https://github.com/o/r", "visibility": "PRIVATE",
               "defaultBranchRef": {"name": "main"}},
    )
    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_sha}
    header["head"] = {"ref": "feature/cache", "sha": head_sha}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    return str(bare), base_sha, head_sha


def _seed_ready_case(tmp_path, fake_gh, *, lines: int = 3, candidate: bool = False):
    """Seed a genuine frozen ``ready`` workspace for one imported PR.

    Builds a real bare origin, runs the real import (which freezes a ready
    snapshot + bundle + shared ``cache/repository.git`` mirror), and returns
    ``(ws, case_id, head_sha)``. With *candidate* True, seeds one REST inline
    comment so the case has one exact-acceptable candidate.
    """
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict
    from daydream.benchmark.workspace import init_workspace

    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    _seed_preflight(ws, fake_gh)
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh, lines=lines)
    if candidate:
        comment = {
            "id": 1,
            "node_id": "DIFF_1",
            "user": {"login": "alice", "type": "User"},
            "body": "please fix",
            "commit_id": head_sha,
            "original_commit_id": head_sha,
            "path": "feature.py",
            "line": 2,
            "subject_type": "line",
            "side": "RIGHT",
            "in_reply_to_id": None,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "html_url": "https://github.com/o/r/pull/101#discussion_r1",
        }
        fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [comment])
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    case_id = raw["cases"][0]["case_id"]
    return ws, case_id, head_sha


def test_spike_head_file_line_count_from_mirror(tmp_path, fake_gh):
    """The frozen head tree is readable via ``git cat-file blob <head>:<path>``
    with cwd in the shared bare mirror — the location-vs-head read source."""
    from daydream.benchmark import github_import as gi, snapshot as sn
    from daydream.benchmark.workspace import init_workspace

    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh, lines=7)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    m = sn.mirror(ws)
    assert m.exists()
    proc = git_ops._run_git(m, ["cat-file", "blob", f"{head_sha}:feature.py"], retries=0)
    assert proc.returncode == 0
    assert len(proc.stdout.splitlines()) == 7
    assert base_sha != head_sha  # the seed produced a real base/head divergence


def test_list_cases_and_head_file_line_count(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=4)

    cases = cu.list_cases(ws)
    assert [c["case_id"] for c in cases] == [case_id]
    assert cases[0]["state"] == "draft" and cases[0]["gold_mode"] == "clean"

    view = cu.list_case(ws, case_id)
    assert view["snapshot"]["status"] == "ready"
    assert cu._head_file_line_count(ws, head_sha, "feature.py") == 4
    with pytest.raises(cu.CurationError):
        cu._head_file_line_count(ws, head_sha, "missing.py")


def test_validate_case_accepts_clean_and_rejects_duplicate_and_over_cap(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    from daydream.benchmark.schema import derive_finding_id
    from daydream.benchmark.storage import load_yaml_strict

    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    assert cu.validate_case(ws, case_id) is None

    # inject a duplicate canonical finding directly, then validate -> rejected
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    f1 = {"title": "dup", "body": "b", "severity": "low",
          "provenance": {"kind": "authored", "source_ids": []}}
    f1["finding_id"] = derive_finding_id(f1)
    raw["curation"]["findings"] = [f1, dict(f1)]   # same canonical -> duplicate
    raw["curation"]["state"] = "draft"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(cu.CurationError):
        cu.validate_case(ws, case_id)

    # >50 gold -> rejected
    raw["curation"]["findings"] = [
        {"title": f"f{i}", "body": "b", "severity": "low",
         "provenance": {"kind": "authored", "source_ids": []}} for i in range(51)]
    for f in raw["curation"]["findings"]:
        f["finding_id"] = derive_finding_id(f)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(cu.CurationError):
        cu.validate_case(ws, case_id)
