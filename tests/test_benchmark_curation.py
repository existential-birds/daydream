"""Tests for the UI-independent golden-review curation service.

Covers the real-path seed (a genuine frozen ``ready`` workspace built from a
real local bare origin), the head-tree line-count read source (a shared bare
mirror via ``git cat-file blob <head>:<path>``), and the full derivation /
rejection / transition surface of :mod:`daydream.benchmark.curation`.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from daydream import git_ops
from daydream.benchmark.schema import derive_finding_id
from daydream.benchmark.storage import atomic_write_yaml, load_yaml_strict

_REPO_ROOT = Path(__file__).resolve().parents[1]

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
        value={"id": "R_kgDOABC123", "nameWithOwner": "o/r",
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
        value={"id": "R_kgDOABC123", "nameWithOwner": "o/r",
               "url": "https://github.com/o/r", "visibility": "PRIVATE",
               "defaultBranchRef": {"name": "main"}},
    )
    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_sha}
    header["head"] = {"ref": "feature/cache", "sha": head_sha}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    return str(bare), base_sha, head_sha


_SEED_SEQ = {"n": 0}


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

    _SEED_SEQ["n"] += 1
    ws = tmp_path / f"ws-{_SEED_SEQ['n']}"
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
    from daydream.benchmark import github_import as gi
    from daydream.benchmark import snapshot as sn
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


def test_accept_candidate_produces_historical_derived_finding(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    view = cu.get_case(ws, case_id)
    cand = next(c for c in view["candidates"] if c["exact_acceptable"])

    cu.accept_candidate(ws, case_id, cand["source_id"])

    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f = raw["curation"]["findings"][0]
    assert len(raw["curation"]["findings"]) == 1
    assert f["provenance"]["kind"] == "historical"
    assert f["provenance"]["source_ids"] == [cand["source_id"]]
    assert f["title"] == cand["title"] and f["body"] == cand["body"]
    assert f["location"] == cand["location"]
    assert f["finding_id"] == derive_finding_id(f, case_id=case_id)      # derived, content-addressed
    assert raw["curation"]["gold_status"] == "findings"
    assert raw["curation"]["gold_mode"] == "historical"
    assert raw["curation"]["state"] == "draft"           # accept on draft stays draft

def test_add_finding_is_authored_and_replace_is_edited(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)

    cu.add_finding(ws, case_id, title="New concern", body="fresh wording",
                   severity="high", location={"path": "feature.py",
                                              "start_line": 1, "end_line": 1},
                   source_ids=[])
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f = raw["curation"]["findings"][0]
    assert f["provenance"]["kind"] == "authored" and f["provenance"]["source_ids"] == []
    assert raw["curation"]["gold_mode"] == "authored"

    # now replace it with a rewritten (edited) finding referencing one source
    cu.replace_findings(ws, case_id, f["finding_id"],
                        replacements=[{"title": "New concern (v2)", "body": "rewritten",
                                       "severity": "medium",
                                       "location": {"path": "feature.py",
                                                    "start_line": 2, "end_line": 2},
                                       "source_ids": ["github:inline_comment:1"]}])
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f2 = raw["curation"]["findings"][0]
    assert f2["provenance"]["kind"] == "edited" and f2["finding_id"] != f["finding_id"]
    assert f2["title"] == "New concern (v2)" and raw["curation"]["gold_mode"] == "historical"
    assert f2["finding_id"] == derive_finding_id(f2, case_id=case_id)

def test_exclude_evidence_reason_contract_and_other_requires_note(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    view = cu.get_case(ws, case_id)
    src = view["candidates"][0]["source_id"]

    cu.exclude_evidence(ws, case_id, src, reason="fixed_before_snapshot")
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    ex = raw["curation"]["exclusions"][0]
    assert ex == {"source_id": src, "reason": "fixed_before_snapshot", "note": None}

    with pytest.raises(cu.CurationError):
        cu.exclude_evidence(ws, case_id, src, reason="other")          # other needs a note
    with pytest.raises(cu.CurationError):
        cu.exclude_evidence(ws, case_id, src, reason="not_a_reason")   # literal contract
    # re-excluding the same source is last-wins: it replaces the existing row
    cu.exclude_evidence(ws, case_id, src, reason="incorrect")
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["exclusions"] == [{"source_id": src, "reason": "incorrect", "note": None}]


def test_reopen_for_mutation_transitions(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    cur.update({"state": "ready", "snapshot_attested": True})
    reopened = cu._reopen_for_mutation(cur)
    assert reopened["state"] == "draft" and reopened["snapshot_attested"] is False
    cur.update({"state": "stale", "snapshot_attested": True})
    reopened = cu._reopen_for_mutation(cur)
    assert reopened["state"] == "stale" and reopened["snapshot_attested"] is False


def test_mark_ready_requires_sha_and_attest_clean_never_ready(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    cu.accept_candidate(ws, case_id,
        next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])["source_id"])

    # wrong SHA is rejected; correct SHA attests
    with pytest.raises(cu.CurationError):
        cu.mark_ready(ws, case_id, head_sha="f" * 40)
    cu.mark_ready(ws, case_id, head_sha=head_sha)
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["state"] == "ready" and raw["curation"]["snapshot_attested"] is True

    # clean attest on an empty-gold case: sets clean_attested, never ready
    ws2, case_id2, head_sha2 = _seed_ready_case(tmp_path, fake_gh, lines=2)
    cu.attest_clean(ws2, case_id2)
    raw2 = load_yaml_strict(ws2 / "cases" / f"{case_id2}.yaml")
    assert raw2["curation"]["clean_attested"] is True
    assert raw2["curation"]["gold_status"] == "clean"
    assert raw2["curation"]["state"] == "draft" and raw2["curation"]["snapshot_attested"] is False


def test_mark_ready_clean_attested_empty_yields_ready(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    from daydream.benchmark.storage import load_yaml_strict
    from daydream.benchmark.workspace import validate_workspace
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=2)  # empty gold
    cu.attest_clean(ws, case_id)
    cu.mark_ready(ws, case_id, head_sha=head_sha)
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["state"] == "ready" and cur["snapshot_attested"] is True
    assert cur["clean_attested"] is True
    assert cur["gold_status"] == "clean" and cur["gold_mode"] == "clean"
    code, _label = validate_workspace(ws)     # ready-clean passes workspace validation
    assert code == 0


def test_mark_ready_empty_not_clean_attested_still_raises(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=2)  # empty gold, NOT attested
    with pytest.raises(cu.CurationError):
        cu.mark_ready(ws, case_id, head_sha=head_sha)
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["state"] == "draft" and cur["snapshot_attested"] is False


def test_mark_ready_clean_wrong_sha_is_non_mutating(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=2)
    cu.attest_clean(ws, case_id)
    with pytest.raises(cu.StaleStateError):
        cu.mark_ready(ws, case_id, head_sha="f" * 40)
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["state"] == "draft" and cur["snapshot_attested"] is False
    assert cur["clean_attested"] is True        # attestation preserved; only readiness failed


def test_ready_edit_reopens_draft_and_clears_attestation(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    # put the case in ready + attested with one historical finding
    cu.accept_candidate(ws, case_id,
        next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])["source_id"])
    cu.mark_ready(ws, case_id, head_sha=head_sha)
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["state"] == "ready" and raw["curation"]["snapshot_attested"] is True

    # an exclusion (gold/provenance/evidence mutation) on a ready case reopens draft
    src = next(c["source_id"] for c in raw["candidates"])
    cu.exclude_evidence(ws, case_id, src, reason="duplicate")
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["state"] == "draft"
    assert raw["curation"]["snapshot_attested"] is False


def test_exclude_and_reinclude_case_transitions(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    snap_status = cu.get_case(ws, case_id)["snapshot"]["status"]  # "ready"
    assert snap_status == "ready"

    cu.exclude_case(ws, case_id, reason="not_suitable")
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["state"] == "excluded"
    assert raw["curation"]["case_exclusion"] == {"reason": "not_suitable", "note": None}

    # a ready snapshot re-includes to draft (not back to ready)
    cu.reinclude_case(ws, case_id)
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["state"] == "draft"
    assert raw["curation"]["case_exclusion"] is None

    # other reason requires a note
    with pytest.raises(cu.CurationError):
        cu.exclude_case(ws, case_id, reason="other")
    # invalid literal rejected
    with pytest.raises(cu.CurationError):
        cu.exclude_case(ws, case_id, reason="nope")


def test_apply_gold_fragment_strips_forged_fields_and_never_ready(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _head = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    cand = next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    src = cand["source_id"]

    fragment = {
        "findings": [{
            # forged fields must be discarded and re-derived
            "finding_id": "f" * 64, "provenance": {"kind": "historical", "source_ids": [src]},
            "state": "ready", "gold_status": "findings", "gold_mode": "historical",
            "title": cand["title"], "body": cand["body"], "severity": None,
            "location": cand["location"], "source_ids": [src],
        }],
        "exclusions": [], "case_exclusion": None, "clean": False,
    }
    cu.apply_gold_fragment(ws, case_id, fragment)
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f = raw["curation"]["findings"][0]
    assert f["finding_id"] == derive_finding_id(f, case_id=case_id)          # forged id discarded
    assert f["provenance"]["kind"] == "historical"          # derived from candidate match
    assert raw["curation"]["state"] == "draft"              # never ready
    assert raw["curation"]["snapshot_attested"] is False
    assert raw["curation"]["gold_status"] == "findings"


def test_stable_curation_types_exported():
    import daydream.benchmark as bm
    assert callable(bm.apply_gold_fragment)
    assert callable(bm.accept_candidate)
    assert callable(bm.mark_ready)
    assert callable(bm.validate_case)
    assert issubclass(bm.CurationError, Exception)


def test_stale_case_edit_stays_stale_and_re_attests(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    src = next(c["source_id"] for c in cu.get_case(ws, case_id)["candidates"])
    # force the case into stale + attested (simulating a refresh that flipped ready->stale)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    raw["curation"].update({"state": "stale", "snapshot_attested": True})
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    cu.accept_candidate(ws, case_id, src)                        # stale edit stays stale
    cu.exclude_evidence(ws, case_id, src, reason="duplicate")   # stale edit stays stale
    raw = load_yaml_strict(path)
    assert raw["curation"]["state"] == "stale"
    assert raw["curation"]["snapshot_attested"] is False

    cu.mark_ready(ws, case_id, head_sha=head_sha)                # stale -> ready
    raw = load_yaml_strict(path)
    assert raw["curation"]["state"] == "ready" and raw["curation"]["snapshot_attested"] is True


def test_reject_before_persistence_leaves_file_unchanged(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()

    # invalid location path (not in head) on an authored finding -> rejected, unchanged
    with pytest.raises(cu.CurationError):
        cu.add_finding(ws, case_id, title="x", body="b", severity="low",
                       location={"path": "missing.py", "start_line": 1, "end_line": 1},
                       source_ids=[])
    assert path.read_bytes() == before

    # line beyond the head file's line count -> rejected, unchanged
    with pytest.raises(cu.CurationError):
        cu.add_finding(ws, case_id, title="x", body="b", severity="low",
                       location={"path": "feature.py", "start_line": 99, "end_line": 99},
                       source_ids=[])
    assert path.read_bytes() == before

    # forged provenance on the fragment is discarded (not rejected) but never persists state=ready
    frag = {"findings": [{"title": "x", "body": "b", "severity": "low",
                          "location": {"path": "feature.py", "start_line": 1, "end_line": 1},
                          "source_ids": [], "state": "ready"}],
            "exclusions": [], "case_exclusion": None, "clean": False}
    cu.apply_gold_fragment(ws, case_id, frag)
    raw = load_yaml_strict(path)
    assert raw["curation"]["state"] == "draft"


def test_list_cases_and_head_file_line_count(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=4)

    cases = cu.list_cases(ws)
    assert [c["case_id"] for c in cases] == [case_id]
    assert cases[0]["state"] == "draft" and cases[0]["gold_mode"] == "clean"

    view = cu.get_case(ws, case_id)
    assert view["snapshot"]["status"] == "ready"
    snapshot_doc = view["snapshot"]
    assert cu._head_file_line_count(ws, snapshot_doc, "feature.py") == 4
    with pytest.raises(cu.CurationError):
        cu._head_file_line_count(ws, snapshot_doc, "missing.py")


def test_list_cases_returns_evidence_count_and_changed_stats(tmp_path, fake_gh):
    # numstat + evidence-count claims confirmed by tests/test_spike_issue775_reads.py
    from daydream.benchmark import curation as cu
    ws, case_id, _head = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)

    cases = cu.list_cases(ws)
    assert cases[0]["evidence_count"] == 1          # one seeded candidate
    assert cases[0]["changed_files"] == 2           # base.py + feature.py
    assert cases[0]["changed_lines"] == 6           # 2 + 4 (base edit + feature add)
    # a non-ready snapshot degrades to zero stats without raising
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    raw["snapshot"] = {"status": "imported", "policy": "final_pr_head",
                       "requested_head": "final"}
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    c2 = cu.list_cases(ws)[0]
    assert c2["changed_files"] == 0 and c2["changed_lines"] == 0


def test_list_cases_ready_mirror_failure_returns_stats_from_bundle(tmp_path, fake_gh):
    """Deliberate behavior flip (issue #814): a ready case whose shared bare
    mirror is deleted still returns change stats — the reads come from a
    disposable clone of the frozen bundle, never the mirror."""
    from daydream.benchmark import curation as cu
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    import shutil
    shutil.rmtree(ws / "cache" / "repository.git")        # ready case, mirror gone
    cases = cu.list_cases(ws)
    assert cases[0]["changed_files"] == 2 and cases[0]["changed_lines"] == 6


def test_corrupt_bundle_path_fails_clean_with_curation_error(tmp_path, fake_gh):
    """A ready snapshot whose bundle_file is absolute / traversal must fail the
    read-only bundle-clone paths with the curated CurationError contract, never
    the storage WorkspaceCorrupt family — list_cases (and the TUI) and
    validate_case's location-vs-head read catch only CurationError."""
    from daydream.benchmark import curation as cu
    from daydream.benchmark.storage import load_yaml_strict

    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    # a located historical finding makes validate_case read the frozen head tree
    view = cu.get_case(ws, case_id)
    cand = next(c for c in view["candidates"] if c["exact_acceptable"])
    cu.accept_candidate(ws, case_id, cand["source_id"])

    path = ws / "cases" / f"{case_id}.yaml"
    good_bundle = load_yaml_strict(path)["snapshot"]["bundle_file"]
    for bad in (str(ws / "snapshots" / "bundle.bundle"), "../../escape.bundle"):
        raw = load_yaml_strict(path)
        raw["snapshot"]["bundle_file"] = bad
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        with pytest.raises(cu.CurationError):
            cu.list_cases(ws)
        with pytest.raises(cu.CurationError):
            cu.validate_case(ws, case_id)
    # restoring the original relative in-root bundle_file heals the read-only paths
    raw = load_yaml_strict(path)
    raw["snapshot"]["bundle_file"] = good_bundle
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert cu.list_cases(ws)[0]["changed_files"] == 2


def test_curate_and_validate_after_mirror_removal(tmp_path, fake_gh):
    """Acceptance (e): deleting the shared mirror never makes a case uncuratable.

    Curation location-vs-head reads and ``list_cases`` change stats must come
    from a disposable clone of the frozen bundle (origin/base vs origin/head),
    and ``validate_workspace`` fidelity is bundle-based too."""
    import shutil

    from daydream.benchmark import curation as cu
    from daydream.benchmark.workspace import validate_workspace

    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    shutil.rmtree(ws / "cache" / "repository.git")        # mirror gone
    # curation location-vs-head still works from the bundle clone
    cu.add_finding(ws, case_id, title="x", body="y", severity="high",
                   location={"path": "feature.py", "start_line": 1, "end_line": 1})
    # list_cases change stats still work (was CurationError before this issue)
    rows = cu.list_cases(ws)
    assert rows[0]["changed_files"] == 2 and rows[0]["changed_lines"] == 6
    # attest ready (re-exercises location-vs-head from the bundle), so the
    # workspace reaches the ``ready`` exit-0 state on validation
    cu.mark_ready(ws, case_id, head_sha=head_sha)
    # validate_workspace still passes (fidelity is bundle-based)
    code, label = validate_workspace(ws)
    assert code == 0 and label == "ready"


def test_bundle_clone_reused_across_findings_and_calls(tmp_path, fake_gh, monkeypatch):
    """Located-finding validation and list_cases share one bundle clone.

    Regression for the O(cases x findings) clone fan-out: every located finding
    used to open a fresh ``git clone --no-checkout`` of the frozen bundle, and
    every list_cases/validate_case call re-cloned. The reuse cache must serve
    all of them from a single clone per bundle file.
    """
    from daydream.benchmark import curation as cu

    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    real_run_git = git_ops._run_git
    clones = {"n": 0}

    def spy_run_git(repo, args, **kwargs):
        if args and args[0] == "clone":
            clones["n"] += 1
        return real_run_git(repo, args, **kwargs)

    monkeypatch.setattr(git_ops, "_run_git", spy_run_git)

    cu.add_finding(ws, case_id, title="a", body="b", severity="high",
                   location={"path": "feature.py", "start_line": 1, "end_line": 1})
    cu.add_finding(ws, case_id, title="c", body="d", severity="medium",
                   location={"path": "feature.py", "start_line": 2, "end_line": 2})
    assert clones["n"] == 1            # both located-finding mutations share one clone
    cu.validate_case(ws, case_id)      # two located findings, still one clone
    assert clones["n"] == 1
    cu.list_cases(ws)
    assert clones["n"] == 1


def test_get_case_attaches_evidence_projection(tmp_path, fake_gh):
    # evidence-join claim confirmed by tests/test_spike_issue775_reads.py
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)

    view = cu.get_case(ws, case_id)
    cand = next(c for c in view["candidates"] if c["exact_acceptable"])
    ev = cand["evidence"]
    assert ev["kind"] == "inline_comment"
    assert ev["author"] == {"login": "alice", "type": "User"}
    assert ev["commit_id"] == head_sha
    assert ev["resolved"] is False and ev["outdated"] is False
    # a candidate with no backing evidence record is tolerated (projection
    # absent) -- verified through get_case on a genuinely unmatched source_id,
    # not by hand-constructing a stripped view (which would pass even if the
    # projection logic silently attached an evidence dict unconditionally).
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    unmatched = {k: v for k, v in cand.items() if k != "evidence"}
    unmatched["source_id"] = "github:review:999"
    raw["candidates"].append(unmatched)
    atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)
    view2 = cu.get_case(ws, case_id)
    assert "evidence" not in view2["candidates"][-1]   # unmatched source, absent
    assert "evidence" in view2["candidates"][0]        # matched source, still joined


def test_validate_case_accepts_clean_and_rejects_duplicate_and_over_cap(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    assert cu.validate_case(ws, case_id) is None

    # a malformed curation block must not abort the whole read-only index
    path = ws / "cases" / f"{case_id}.yaml"
    pristine_yaml = path.read_bytes()
    raw = load_yaml_strict(path)
    raw["curation"] = dict(raw["curation"])
    raw["curation"]["state"] = "bogus"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cases = cu.list_cases(ws)
    assert cases[0]["state"] == "bogus" and cases[0]["gold_mode"] is None

    # a non-dict curation block must not abort the read-only index either
    path.write_text(yaml.safe_dump({**raw, "curation": "bogus"}, sort_keys=False))
    cases = cu.list_cases(ws)
    assert cases[0]["state"] is None and cases[0]["gold_mode"] is None

    # restore the pristine doc, then exercise a duplicate canonical finding, and
    # validate -> rejected
    path.write_bytes(pristine_yaml)
    raw = load_yaml_strict(path)
    f1 = {"title": "dup", "body": "b", "severity": "low",
          "provenance": {"kind": "authored", "source_ids": []}}
    f1["finding_id"] = derive_finding_id(f1, case_id=case_id)
    raw["curation"]["findings"] = [f1, dict(f1)]   # same canonical -> duplicates
    raw["curation"]["state"] = "draft"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(cu.CurationError):
        cu.validate_case(ws, case_id)

    # >50 gold -> rejected
    raw["curation"]["findings"] = [
        {"title": f"f{i}", "body": "b", "severity": "low",
         "provenance": {"kind": "authored", "source_ids": []}} for i in range(51)]
    for f in raw["curation"]["findings"]:
        f["finding_id"] = derive_finding_id(f, case_id=case_id)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(cu.CurationError):
        cu.validate_case(ws, case_id)


_WORKER = (
    "import sys\n"
    "from daydream.benchmark import curation as cu\n"
    "op, ws, cid = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "if op == 'add':\n"
    "    cu.add_finding(ws, cid, title=sys.argv[4], body='b', severity='low',\n"
    "                   location={'path': 'feature.py', 'start_line': 1, 'end_line': 1},\n"
    "                   source_ids=[])\n"
    "elif op == 'accept':\n"
    "    cu.accept_candidate(ws, cid, sys.argv[4])\n"
    "elif op == 'exclude':\n"
    "    cu.exclude_evidence(ws, cid, sys.argv[4], reason='duplicate')\n"
    "elif op == 'clean':\n"
    "    cu.attest_clean(ws, cid)\n"
    "else:\n"
    "    raise SystemExit('unknown op')\n"
)


def _spawn_worker(args: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _WORKER, *args],
        cwd=_REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_concurrent_accept_and_add_do_not_lose_updates(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    src = next(c["source_id"] for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    procs = [_spawn_worker(["accept", str(ws), case_id, src])]
    procs += [_spawn_worker(["add", str(ws), case_id, f"conc-{i}"]) for i in range(4)]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, (out, err)
    findings = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["findings"]
    assert len(findings) == 5                                   # all 5 concurrent updates landed
    assert {"conc-0", "conc-1", "conc-2", "conc-3"} <= {f["title"] for f in findings}
    hist = [f for f in findings if f["provenance"]["kind"] == "historical"]
    assert len(hist) == 1 and hist[0]["provenance"]["source_ids"] == [src]


def test_concurrent_excludes_serialize_to_single_row(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    src = next(c["source_id"] for c in cu.get_case(ws, case_id)["candidates"])
    # Mixed concurrent mutations on one case: 3 idempotent excludes of the same
    # source must serialize to a single row, AND 3 distinguishable adds must all
    # land — a lost update (if the workspace lock were removed) would drop one.
    procs = [_spawn_worker(["exclude", str(ws), case_id, src]) for _ in range(3)]
    procs += [_spawn_worker(["add", str(ws), case_id, f"mix-{i}"]) for i in range(3)]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, (out, err)
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["exclusions"] == [{"source_id": src, "reason": "duplicate", "note": None}]
    assert {"mix-0", "mix-1", "mix-2"} <= {f["title"] for f in raw["curation"]["findings"]}
    assert cu.validate_case(ws, case_id) is None                # case not corrupted by interleaving


def test_concurrent_clean_attestation_serializes(tmp_path, fake_gh):
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)   # empty gold
    procs = [_spawn_worker(["clean", str(ws), case_id]) for _ in range(3)]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, (out, err)
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["clean_attested"] is True and cur["gold_status"] == "clean"
    assert cur["state"] == "draft" and cur["snapshot_attested"] is False


def test_concurrent_adds_then_final_readiness_lands(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    procs = [_spawn_worker(["add", str(ws), case_id, f"r-{i}"]) for i in range(3)]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, (out, err)
    cu.mark_ready(ws, case_id, head_sha=head_sha)                 # final readiness on top of the concurrent adds
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    # All three concurrent adds land before readiness (order is lock-acquisition
    # order, so title order is nondeterministic — assert the set, not the order).
    assert sorted(f["title"] for f in raw["curation"]["findings"]) == ["r-0", "r-1", "r-2"]
    assert raw["curation"]["state"] == "ready" and raw["curation"]["snapshot_attested"] is True


def test_lock_file_and_error_text_contain_no_repo_evidence(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    src = next(c["source_id"] for c in cu.get_case(ws, case_id)["candidates"])
    cu.exclude_evidence(ws, case_id, src, reason="duplicate")   # acquires + releases the lock
    assert (ws / ".benchmark.lock").read_bytes() == b""         # lock file is empty: no repo evidence/credentials
    with pytest.raises(cu.CurationError) as ei:
        cu.exclude_evidence(ws, case_id, "not-a-source", reason="duplicate")
    msg = str(ei.value)
    assert "o/r" not in msg and "token" not in msg.lower() and "api_key" not in msg.lower()


def test_read_only_paths_run_concurrent_with_a_writer(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    lock_path = ws / ".benchmark.lock"
    # A writer holds the flock for a long window (30s) — far longer than any
    # plausible read-only path — so a differential probe can detect whether the
    # read-only paths block on the lock without a fragile wall-clock assertion.
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, time, sys\n"
         "fd = open(sys.argv[1], 'w')\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n"
         "print('held', flush=True)\n"
         "time.sleep(30)\n",
         str(lock_path)],
        stdout=subprocess.PIPE, text=True,
    )
    assert holder.stdout.readline().strip() == "held"            # another process now holds the flock
    try:
        cu.list_cases(ws)
        cu.get_case(ws, case_id)
        cu.validate_case(ws, case_id)
        # The writer must STILL be holding the flock after the reads returned:
        # had a read-only path blocked on the lock it could not finish until the
        # writer released (30s later).
        assert holder.poll() is None
    finally:
        holder.wait(timeout=30)


def test_locked_mutation_heals_interrupted_journal_before_new_write(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    from daydream.benchmark.storage import Transaction
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    mutated = dict(raw)
    mutated["curation"] = dict(raw["curation"])
    mutated["curation"]["state"] = "excluded"          # an interrupted mutation left in flight
    with Transaction(ws, op_id=f"curate-{case_id}", kind="curation:exclude-case") as tx:
        tx.stage(f"cases/{case_id}.yaml", yaml.safe_dump(mutated, sort_keys=False).encode("utf-8"))
        tx.inject_crash("target-1")                    # target applied under 'committing', then halt
    assert load_yaml_strict(path)["curation"]["state"] == "excluded"
    cu.add_finding(ws, case_id, title="recovered", body="b", severity="low",
                   location={"path": "feature.py", "start_line": 1, "end_line": 1}, source_ids=[])
    assert not list((ws / "transactions").iterdir())   # prior journal healed
    final = load_yaml_strict(path)
    assert final["curation"]["state"] == "draft"       # interrupted 'excluded' write was rolled back
    assert [f["title"] for f in final["curation"]["findings"]] == ["recovered"]


def test_stale_state_error_is_exported_curation_subtype():
    import daydream.benchmark as bm
    assert issubclass(bm.StaleStateError, bm.CurationError)


def test_stale_attestation_raises_stale_state_error_and_leaves_unchanged(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    src = next(c["source_id"] for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    cu.accept_candidate(ws, case_id, src)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    with pytest.raises(cu.StaleStateError):
        cu.mark_ready(ws, case_id, head_sha="f" * 40)   # stale attestation SHA
    assert path.read_bytes() == before                    # a rejected mutation writes nothing



def test_curation_ready_requires_task_spec_sha256():
    from daydream.benchmark.schema import Curation
    import pydantic
    base = dict(state="ready", snapshot_attested=True, gold_status="findings",
                clean_attested=False, exclusions=[], case_exclusion=None)
    f = {"finding_id": "f" * 64, "title": "t", "body": "b", "severity": "low",
         "location": None, "provenance": {"kind": "historical", "source_ids": ["s"]}}
    # ready with a digest validates; ready without a digest is rejected
    assert Curation(**base, findings=[f], task_spec_sha256="d" * 64).task_spec_sha256 == "d" * 64
    with pytest.raises(pydantic.ValidationError):
        Curation(**base, findings=[f], task_spec_sha256=None)
    # non-ready states (draft/stale) may be unset
    draft = Curation(state="draft", snapshot_attested=False, clean_attested=False,
                     gold_status=None, findings=[], exclusions=[], case_exclusion=None,
                     task_spec_sha256=None)
    assert draft.task_spec_sha256 is None


def test_task_spec_approved_at_is_stripped_before_validation():
    from daydream.benchmark import schema
    from daydream.benchmark.schema import Curation, _schema_ready
    from daydream.benchmark import curation as cu
    raw = {"curation": {"state": "draft", "snapshot_attested": False, "clean_attested": False,
                        "gold_status": None, "findings": [], "exclusions": [],
                        "case_exclusion": None, "gold_mode": "clean",
                        "task_spec_approved_at": "2026-08-23T00:00:00+00:00"}}
    ready = _schema_ready(raw)
    assert "task_spec_approved_at" not in ready["curation"]
    assert "gold_mode" not in ready["curation"]            # existing behaviour preserved
    # curation service path also drops it
    model = cu._curation_model(raw["curation"])
    assert isinstance(model, Curation)
    assert not hasattr(model, "task_spec_approved_at")
