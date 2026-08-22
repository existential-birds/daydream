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

import os
import subprocess

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
    doc = gi.fetch_and_normalize(ws, "o/r", 101)
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


def test_parse_head_pr_sha_grammar_and_binding(tmp_path):
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


def test_parse_heads_bound_per_pr_in_multi_import(tmp_path):
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


def _seed_manifest(ws):
    """Build an initialized private workspace with an unresolved Source (o/r).

    Idempotent: a caller may explicitly :func:`init_workspace` first (e.g. the
    snapshot-freeze test pins reviewer/judge hosts), in which case the manifest
    already exists and the scaffold is left untouched.
    """
    from daydream.benchmark.workspace import init_workspace

    if (ws / "benchmark.yaml").exists():
        return
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])


def test_preflight_six_checks_in_order_and_atomic_identity(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_manifest(ws)  # unresolved Source (repository=o/r)
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
    assert attempts["n"] == 3 and ok["number"] == 101
    assert slept and all(w <= 60 for w in slept)  # Retry-After honored, 60s cap


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


# ---------------------------------------------------------------------------
# real-git local-origin seed for snapshot-freeze wiring (no network)
# ---------------------------------------------------------------------------


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


def _seed_local_origin(tmp_path, fake_gh) -> tuple[str, str, str]:
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


def test_import_freezes_cases_ready_with_bundle(tmp_path, fake_gh):
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
    assert case["snapshot"]["original_head_sha"] == head_sha
    bundle = ws / case["snapshot"]["bundle_file"]
    assert bundle.exists()
    assert sha256_file(bundle) == case["snapshot"]["bundle_sha256"]


def test_e2e_import_distinct_idempotent_explicit_head_and_shared_mirror(tmp_path, fake_gh):
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


def test_import_writes_atomic_unit_and_no_file_on_failure(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict, sha256_file

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)  # preflight + rest/graphql canned data for pr 101 (one head)
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"])
    assert rc == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    pr = raw["pull_requests"][0]
    assert pr["import_state"] == "fetched"
    assert pr["import_file"] == "imports/pr-000101.json"
    assert pr["import_sha256"] == sha256_file(ws / pr["import_file"])
    assert pr["requested_heads"] == ["final"]
    assert pr["case_ids"] == ["pr-000101-" + "a" * 12]   # head from _PR_HEADER
    assert (ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml").exists()


def test_failed_fetch_leaves_no_import_file_and_ledger_error(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh, pull_header=None)  # 404 -> fetch fails
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"])
    assert rc != 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    pr = raw["pull_requests"][0]
    assert pr["import_state"] == "fetch_failed"
    assert pr["error"]["code"] and pr["error"]["message"]
    assert pr["import_file"] is None and pr["import_sha256"] is None
    assert not (ws / "imports" / "pr-000101.json").exists()


def test_status_reflects_fetched_import_and_resolved_identity(tmp_path, fake_gh):
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.workspace import workspace_status

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"])
    assert rc == 0
    st = workspace_status(ws)
    assert st.workspace_state != "empty"
    assert st.repository_identity_resolved is True


def test_cli_import_prs_drives_command(tmp_path, fake_gh, capsys):
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


def _curate_case(ws, case_file):
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
    finding["finding_id"] = derive_finding_id(finding)
    raw["curation"] = {
        "state": "ready",
        "snapshot_attested": True,
        "clean_attested": False,
        "gold_status": "findings",
        "findings": [finding],
        "exclusions": [],
        "case_exclusion": None,
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def test_refresh_marks_stale_and_never_overwrites_curation(tmp_path, fake_gh):
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
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"])
    assert rc == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")   # curation.state=ready, snapshot_attested=True
    # refresh re-fetches; the referenced evidence now disappears
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [])
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True)
    assert rc == 0
    case = load_yaml_strict(ws / "cases" / "pr-000101-aaaaaaaaaaaa.yaml")
    assert case["curation"]["state"] == "stale" and case["curation"]["snapshot_attested"] is False
    assert case["curation"]["findings"]              # prior curated findings preserved


def _pr_header(number: int) -> dict:
    header = dict(_PR_HEADER)
    header["number"] = number
    header["url"] = f"https://github.com/o/r/pull/{number}"
    return header


def _seed_identity(fake_gh) -> None:
    """Seed the preflight identity + repo-access responses (no PR data)."""
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response(
        "repo-view-full",
        value={"id": 5, "nameWithOwner": "o/r",
               "url": "https://github.com/o/r", "visibility": "PRIVATE",
               "defaultBranchRef": {"name": "main"}},
    )


def _seed_rest(gh, number: int, *, reviews, comments, issue_comments) -> None:
    """Seed one PR's REST evidence into the fake router."""
    gh.set_response("GET", f"repos/o/r/pulls/{number}", _pr_header(number))
    gh.set_response("GET", f"repos/o/r/pulls/{number}/reviews", reviews)
    gh.set_response("GET", f"repos/o/r/pulls/{number}/comments", comments)
    gh.set_response("GET", f"repos/o/r/issues/{number}/comments", issue_comments)


def test_e2e_paginated_human_bot_evidence_and_no_comment_pr(tmp_path, fake_gh):
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
    assert kinds == {"review", "inline_comment", "issue_comment", "thread_comment"}
    assert any(e["is_bot"] for e in imp["evidence"])      # bot author retained
    assert any(not e["is_bot"] for e in imp["evidence"])  # human author retained
    assert load_json_strict(ws / "imports/pr-000102.json")["evidence"] == []  # no-comment PR retained


def test_e2e_partial_failure_persists_ledger_and_exits_nonzero(tmp_path, fake_gh):
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


def test_benchmark_help_lists_import_prs():
    import subprocess

    r = subprocess.run(["daydream", "benchmark", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "import-prs" in r.stdout
def test_reimport_does_not_duplicate_cases_rows(tmp_path, fake_gh):
    """Re-importing the same PR (unchanged evidence) must not duplicate cases[] rows."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"]) == 0
    raw1 = load_yaml_strict(ws / "benchmark.yaml")
    assert len(raw1["cases"]) == 1
    # re-import same PR, unchanged evidence (responses not changed) — fetched->fetched allowed
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"]) == 0
    raw2 = load_yaml_strict(ws / "benchmark.yaml")
    ids = [c["case_id"] for c in raw2["cases"]]
    assert len(raw2["cases"]) == 1, f"cases[] grew to {len(raw2['cases'])}: {ids}"
    assert ids[0] == "pr-000101-aaaaaaaaaaaa"


def test_reimport_unchanged_evidence_preserves_curation(tmp_path, fake_gh):
    """Re-import with unchanged evidence must not wipe curated findings/attestation."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"]) == 0
    case_file = "pr-000101-aaaaaaaaaaaa.yaml"
    _curate_case(ws, case_file)  # state=ready, snapshot_attested=True, findings non-empty
    before = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert before["state"] == "ready" and before["snapshot_attested"] is True and before["findings"]
    # Re-import same PR WITHOUT refresh, unchanged evidence.
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"]) == 0
    after = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert after["state"] in ("ready", "stale"), "curation must not reset to draft"
    assert after["findings"], "curated findings must not be wiped"
    assert after["snapshot_attested"] is True, "unchanged re-import must keep attestation"


def test_refresh_unchanged_signature_preserves_curation(tmp_path, fake_gh):
    """Refresh with an UNCHANGED evidence signature must keep curated findings."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"]) == 0
    case_file = "pr-000101-aaaaaaaaaaaa.yaml"
    _curate_case(ws, case_file)
    # refresh with IDENTICAL evidence responses (signature unchanged) -> changed=False
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True) == 0
    after = load_yaml_strict(ws / "cases" / case_file)["curation"]
    assert after["findings"], "curated findings must not be wiped on unchanged refresh"
    assert after["state"] != "draft", "curation must not reset to draft on unchanged refresh"


def test_graphql_review_threads_retries_rate_limit_then_fails(tmp_path, fake_gh, monkeypatch):
    """GraphQL reviewThreads honors the rate-limit retry policy (3x)."""
    import daydream.benchmark.github_import as gi_mod
    from daydream.git_ops import RateLimitError

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)

    calls = {"n": 0}

    def flaky_gh_api(*a, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("graphql rate limited", retry_after=0.0)
        ok = {"repository": {"pullRequest": {"reviewThreads": {"nodes": [], "pageInfo": {"hasNextPage": False}}}}}
        return {"data": ok}

    monkeypatch.setattr(gi_mod.git_ops, "gh_api", flaky_gh_api)
    monkeypatch.setattr(gi_mod, "time", type("_T", (), {"sleep": staticmethod(lambda _s: None)})())
    threads = gi_mod._graphql_review_threads(ws, "o/r", 101)
    assert threads == []
    assert calls["n"] == 3, "rate-limit retry should make 3 attempts"


def test_graphql_review_threads_records_rate_limit_after_retries(tmp_path, fake_gh, monkeypatch):
    """Exhausting GraphQL rate-limit retries surfaces _ImportRateLimitError (ledger rate_limit)."""
    import pytest

    import daydream.benchmark.github_import as gi_mod
    from daydream.git_ops import RateLimitError

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)

    def always_limited(*a, **kw):
        raise RateLimitError("graphql rate limited", retry_after=0.0)

    monkeypatch.setattr(gi_mod.git_ops, "gh_api", always_limited)
    monkeypatch.setattr(gi_mod, "time", type("_T", (), {"sleep": staticmethod(lambda _s: None)})())
    with pytest.raises(gi_mod._ImportRateLimitError):
        gi_mod._graphql_review_threads(ws, "o/r", 101)
