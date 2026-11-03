"""Curation over real imported workspaces and immutable frozen bundles.

Exercise evidence derivation, transitions, and validation independently of
the shared mirror's continued availability.
"""

import fcntl
import hashlib
import os
import shutil
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import daydream.benchmark as bm
from daydream.benchmark import curation as cu, github_import as gi
from daydream.benchmark.curation import classify_evidence
from daydream.benchmark.harbor import build
from daydream.benchmark.schema import derive_finding_id
from daydream.benchmark.storage import (
    load_yaml_strict,
)
from daydream.benchmark.workspace import init_workspace
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import seed_pr_origin
from tests.harness.transaction_faults import TransactionFaultDriver
from tests.test_benchmark_import_prs import _PR_HEADER, _seed_preflight

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _seed_local_origin(tmp_path: Path, fake_gh: FakeGh, *, lines: int = 3) -> tuple[str, str, str]:
    """Seed real PR SHAs with exactly ``lines`` numbered lines in feature.py."""
    origin_url, base_sha, head_sha = seed_pr_origin(
        tmp_path, feature_body="".join(f"LINE {i}\n" for i in range(1, lines + 1))
    )
    # ``_seed_preflight`` already seeded the identity + repo-access responses
    # (called immediately before this helper); re-seed the canned PR header so
    # base.sha/head.sha are the real origin SHAs.
    header = dict(_PR_HEADER)
    header["base"] = {"ref": "main", "sha": base_sha}
    header["head"] = {"ref": "feature/cache", "sha": head_sha}
    fake_gh.set_response("GET", "repos/o/r/pulls/101", header)
    return origin_url, base_sha, head_sha


_SEED_SEQ = {"n": 0}


def _seed_ready_workspace(tmp_path: Path, fake_gh: FakeGh, *, lines: int = 3) -> tuple[Any, str, str, str]:
    """Initialize a workspace and seed a real local bare origin for PR 101.

    Returns ``(ws, origin_url, base_sha, head_sha)``; callers seed their
    evidence responses, then run :func:`_finish_import`.
    """
    _SEED_SEQ["n"] += 1
    ws = tmp_path / f"ws-{_SEED_SEQ['n']}"
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    _seed_preflight(ws, fake_gh)
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh, lines=lines)
    return ws, origin_url, base_sha, head_sha


def _finish_import(ws: Path, origin_url: str) -> str:
    """Run the real import and return the first frozen case id."""
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    return cast(str, load_yaml_strict(ws / "benchmark.yaml")["cases"][0]["case_id"])


def _import_ready_case(tmp_path: Path, fake_gh: FakeGh, *, lines: int = 3, candidate: bool = False) -> tuple[Any, ...]:
    """Import a real frozen workspace, optionally with one exact inline candidate.

    Return workspace, case ID, and head SHA.
    """
    ws, origin_url, _, head_sha = _seed_ready_workspace(tmp_path, fake_gh, lines=lines)
    if candidate:
        comment = {"id": 1, "node_id": "DIFF_1", "user": {"login": "alice", "type": "User"}, "body": "please fix",
            "commit_id": head_sha, "original_commit_id": head_sha, "path": "feature.py", "line": 2, "original_line": 2,
            "subject_type": "line", "side": "RIGHT", "in_reply_to_id": None, "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z", "html_url": "https://github.com/o/r/pull/101#discussion_r1",
        }
        fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", [comment])
    case_id = _finish_import(ws, origin_url)
    return ws, case_id, head_sha


def _import_ready_case_mixed(tmp_path: Path, fake_gh: FakeGh, *, lines: int = 3) -> tuple[Any, ...]:
    """Import five records: exact and editable candidates plus three evidence-only kinds."""
    ws, origin_url, _, head_sha = _seed_ready_workspace(tmp_path, fake_gh, lines=lines)
    reviews = [{"id": 100, "node_id": "PRR_100", "user": {"login": "carol", "type": "User"}, "body": "approval text",
            "state": "APPROVED", "commit_id": None, "submitted_at": "2026-01-01T00:01:00Z",
            "created_at": "2026-01-01T00:01:00Z", "updated_at": "2026-01-01T00:01:00Z",
            "html_url": "https://github.com/o/r/pull/101#pullrequestreview-100",
        }, {"id": 101, "user_id": "PRR_101", "user": {"login": "carol", "type": "User"}, "body": "please also fix this",
            "state": "COMMENTED", "commit_id": None, "submitted_at": "2026-01-01T00:02:00Z",
            "created_at": "2026-01-01T00:02:00Z", "updated_at": "2026-01-01T00:02:00Z",
            "html_url": "https://github.com/o/r/pull/101#pullrequestreview-101",
        },
    ]
    inline_comments = [{"id": 1, "node_id": "DIFF_1", "user": {"login": "alice", "type": "User"}, "body": "please fix",
            "commit_id": head_sha, "original_commit_id": head_sha, "path": "feature.py", "line": 2, "original_line": 2,
            "subject_type": "line", "side": "RIGHT", "in_reply_to_id": None, "created_at": "2026-01-01T00:03:00Z",
            "updated_at": "2026-01-01T00:03:00Z", "html_url": "https://github.com/o/r/pull/101#discussion_r1",
        }, {"id": 2, "node_id": "DIFF_2", "user": {"login": "bob", "type": "User"}, "body": "reply text",
            "commit_id": None, "path": None, "line": None, "subject_type": None, "side": None, "in_reply_to_id": 1,
            "created_at": "2026-01-01T00:04:00Z", "updated_at": "2026-01-01T00:04:00Z",
            "html_url": "https://github.com/o/r/pull/101#discussion_r2",
        },
    ]
    issue_comments = [{
            "id": 200, "node_id": "IC_200", "user": {"login": "dave", "type": "User"}, "body": "conversation text",
            "created_at": "2026-01-01T00:05:00Z", "updated_at": "2026-01-01T00:05:00Z",
            "html_url": "https://github.com/o/r/issues/101#issuecomment-200",
        }
    ]
    fake_gh.set_response("GET", "repos/o/r/pulls/101/reviews", reviews)
    fake_gh.set_response("GET", "repos/o/r/pulls/101/comments", inline_comments)
    fake_gh.set_response("GET", "repos/o/r/issues/101/comments", issue_comments)
    case_id = _finish_import(ws, origin_url)
    return ws, case_id, head_sha


# Curation never re-imports, so one real import per shape and worker is copied into each
# test's own workspace (only the mirror's FETCH_HEAD names the template origin path).
_IMPORTED: dict[tuple[str, int, bool], tuple[Path, str, str]] = {}


def _copy_imported(tmp_path: Path, key: tuple[str, int, bool], fake_gh: FakeGh) -> tuple[Any, ...]:
    if key not in _IMPORTED:
        template = tmp_path / ("seed-%s-%d-%d" % key)
        importer = _import_ready_case_mixed if key[0] == "mixed" else _import_ready_case
        options: dict[str, Any] = {"lines": key[1]} if key[0] == "mixed" else {"lines": key[1], "candidate": key[2]}
        _IMPORTED[key] = importer(template, fake_gh, **options)
    template_ws, case_id, head_sha = _IMPORTED[key]
    _SEED_SEQ["n"] += 1
    ws = tmp_path / f"ws-{_SEED_SEQ['n']}"
    shutil.copytree(template_ws, ws, symlinks=True)
    return ws, case_id, head_sha


def _seed_ready_case(tmp_path: Path, fake_gh: FakeGh, *, lines: int = 3, candidate: bool = False) -> tuple[Any, ...]:
    """Return a private copy of a real frozen workspace, optionally with one exact inline candidate.

    Return workspace, case ID, and head SHA.
    """
    return _copy_imported(tmp_path, ("single", lines, candidate), fake_gh)


def _seed_ready_case_mixed(tmp_path: Path, fake_gh: FakeGh, *, lines: int = 3) -> tuple[Any, ...]:
    """Return a private copy of five imported records: exact/editable candidates plus three evidence-only kinds."""
    return _copy_imported(tmp_path, ("mixed", lines, False), fake_gh)


def _non_candidate_evidence_is_citable_and_excludable(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _ = _seed_ready_case_mixed(tmp_path, fake_gh)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    with pytest.raises(cu.CurationError, match="no candidate"):
        cu.accept_candidate(ws, case_id, "github:inline_comment:2")
    assert path.read_bytes() == before
    src = "github:review:100"   # a pure approval, NOT a candidate
    cu.exclude_evidence(ws, case_id, src, reason="duplicate")
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["exclusions"][0]["source_id"] == src
    with pytest.raises(cu.CurationError):        # genuinely unknown still rejected
        cu.exclude_evidence(ws, case_id, "github:review:999", reason="duplicate")


def _mark_ready_empty_not_clean_attested_still_raises(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=2)  # empty gold, NOT attested
    with pytest.raises(cu.CurationError):
        cu.mark_ready(ws, case_id, task_spec_sha256="d" * 64, head_sha=head_sha)
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["state"] == "draft" and cur["snapshot_attested"] is False


def _exclude_and_reinclude_case_transitions(tmp_path: Path, fake_gh: FakeGh) -> None:
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

    with pytest.raises(cu.CurationError):
        cu.exclude_case(ws, case_id, reason="other")
    with pytest.raises(cu.CurationError):
        cu.exclude_case(ws, case_id, reason="nope")


def _reject_before_persistence_leaves_file_unchanged(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()

    # invalid location path (not in head) on an authored finding -> rejected, unchanged
    with pytest.raises(cu.CurationError):
        cu.add_finding(ws, case_id, title="x", body="b", severity="low",
                       location={"path": "missing.py", "start_line": 1, "end_line": 1}, source_ids=[])
    assert path.read_bytes() == before

    # line beyond the head file's line count -> rejected, unchanged
    with pytest.raises(cu.CurationError):
        cu.add_finding(ws, case_id, title="x", body="b", severity="low",
                       location={"path": "feature.py", "start_line": 5, "end_line": 5}, source_ids=[])
    assert path.read_bytes() == before

    # forged provenance on the fragment is discarded (not rejected) but never persists state=ready
    frag = {"findings": [{"title": "x", "body": "b", "severity": "low",
                          "location": {"path": "feature.py", "start_line": 4, "end_line": 4},
                          "source_ids": [], "state": "ready"}],
            "exclusions": [], "case_exclusion": None, "clean": False}
    cu.apply_gold_fragment(ws, case_id, frag)
    raw = load_yaml_strict(path)
    assert raw["curation"]["state"] == "draft"
    assert raw["curation"]["findings"][0]["location"] == {"path": "feature.py", "start_line": 4, "end_line": 4}


def _corrupt_bundle_path_fails_clean_with_curation_error(tmp_path: Path, fake_gh: FakeGh) -> None:
    """A ready snapshot whose bundle_file is absolute / traversal must fail the
    read-only bundle-clone paths with the curated CurationError contract, never
    the storage WorkspaceCorrupt family — list_cases (and the TUI) and
    validate_case's location-vs-head read catch only CurationError."""

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
    raw = load_yaml_strict(path)
    raw["snapshot"]["bundle_file"] = good_bundle
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert cu.list_cases(ws)[0]["changed_files"] == 2


def _validate_case_accepts_clean_and_rejects_duplicate_and_over_cap(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    cu.validate_case(ws, case_id)                     # raises if the case is invalid

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

    path.write_bytes(pristine_yaml)
    raw = load_yaml_strict(path)
    f1 = {"title": "dup", "body": "b", "severity": "low", "provenance": {"kind": "authored", "source_ids": []}}
    f1["finding_id"] = derive_finding_id(f1, case_id=case_id)
    raw["curation"]["findings"] = [f1, dict(f1)]   # same canonical -> duplicates
    raw["curation"]["state"] = "draft"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(cu.CurationError, match="duplicate"):
        cu.validate_case(ws, case_id)

    raw["curation"]["findings"] = [{"title": f"f{i}", "body": "b", "severity": "low",
         "provenance": {"kind": "authored", "source_ids": []}} for i in range(51)]
    for f in raw["curation"]["findings"]:
        f["finding_id"] = derive_finding_id(f, case_id=case_id)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(cu.CurationError):
        cu.validate_case(ws, case_id)


def _mutations_reopen_stale_and_derive_fragment_provenance(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _head = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    with pytest.raises(cu.StaleStateError, match="attestation SHA mismatch"):
        cu.mark_ready(ws, case_id, head_sha="0" * 40)
    with pytest.raises(cu.CurationError, match="atom 0 carries no source_ids"):
        cu.add_edited_findings(ws, case_id, atoms=[
            {"title": "t", "body": "b", "severity": "low", "location": None, "source_ids": []}])
    assert path.read_bytes() == before
    cand = next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    # A fragment matching its single source's candidate exactly is historical, whatever it claims.
    cu.apply_gold_fragment(ws, case_id, {"findings": [{
        "title": cand["title"], "body": cand["body"], "severity": cand["severity"], "location": cand["location"],
        "source_ids": [cand["source_id"]], "provenance": {"kind": "authored"}}]})
    assert load_yaml_strict(path)["curation"]["findings"][0]["provenance"]["kind"] == "historical"
    raw = load_yaml_strict(path)
    raw["curation"].update(state="stale", snapshot_attested=False, task_spec_sha256="d" * 64,
                           task_spec_approved_at="2026-01-01T00:00:00Z")
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cu.add_finding(ws, case_id, title="late", body="b", severity="low", location=None, source_ids=[])
    stale = load_yaml_strict(path)["curation"]
    assert stale["state"] == "stale" and stale["snapshot_attested"] is False
    assert "task_spec_sha256" not in stale and "task_spec_approved_at" not in stale


def _locked_mutation_waits_for_the_lock_and_heals_interrupted_journal(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    mutated = dict(raw)
    mutated["curation"] = dict(raw["curation"])
    mutated["curation"]["state"] = "excluded"          # an interrupted mutation left in flight
    faults = TransactionFaultDriver(ws, op_id=f"curate-{case_id}", kind="curation:exclude-case")
    with faults.transaction as tx:
        tx.stage(f"cases/{case_id}.yaml", yaml.safe_dump(mutated, sort_keys=False).encode("utf-8"))
        faults.halt_at("target-1")                     # target applied under 'committing', then halt
    assert load_yaml_strict(path)["curation"]["state"] == "excluded"
    # Another holder of the workspace lock (a separate open file description) blocks the mutation.
    holder = os.open(ws / ".benchmark.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    errors: list[BaseException] = []

    def add() -> None:
        try:
            cu.add_finding(ws, case_id, title="recovered", body="b", severity="low",
                           location={"path": "feature.py", "start_line": 1, "end_line": 1}, source_ids=[])
        except BaseException as exc:  # surfaced on the test thread below
            errors.append(exc)

    writer = threading.Thread(target=add, daemon=True)
    try:
        writer.start()
        writer.join(timeout=0.3)
        assert writer.is_alive(), "curation mutated without taking the workspace lock"
        assert load_yaml_strict(path)["curation"]["state"] == "excluded"
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)
    writer.join(timeout=10)
    assert not writer.is_alive() and not errors, errors
    assert not list((ws / "transactions").iterdir())   # prior journal healed before the new write
    final = load_yaml_strict(path)
    assert final["curation"]["state"] == "draft"       # interrupted 'excluded' write was rolled back
    assert [f["title"] for f in final["curation"]["findings"]] == ["recovered"]


def test_service_rejections_keep_case_bytes_and_curated_error_contract(tmp_path: Path, fake_gh: FakeGh) -> None:
    for check in (_non_candidate_evidence_is_citable_and_excludable, _mark_ready_empty_not_clean_attested_still_raises,
                  _exclude_and_reinclude_case_transitions, _reject_before_persistence_leaves_file_unchanged,
                  _corrupt_bundle_path_fails_clean_with_curation_error,
                  _validate_case_accepts_clean_and_rejects_duplicate_and_over_cap,
                  _mutations_reopen_stale_and_derive_fragment_provenance,
                  _locked_mutation_waits_for_the_lock_and_heals_interrupted_journal):
        root = tmp_path / check.__name__
        root.mkdir()
        check(root, fake_gh)


def test_curation_approval_survives_reopen_and_stale_reattestation(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    original = load_yaml_strict(path)
    import_path = ws / original["source"]["import_file"]
    import_bytes = import_path.read_bytes()
    candidate = next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    bm.accept_candidate(ws, case_id, candidate["source_id"])
    accepted = load_yaml_strict(path)["curation"]
    (finding,) = accepted["findings"]
    assert finding["provenance"] == {"kind": "historical", "source_ids": [candidate["source_id"]]}
    assert (finding["title"], finding["body"], finding["location"]) == (
        candidate["title"],
        candidate["body"],
        candidate["location"],
    )
    assert finding["finding_id"] == derive_finding_id(finding, case_id=case_id)
    assert (accepted["state"], accepted["gold_status"], accepted["gold_mode"]) == ("draft", "findings", "historical")
    bm.mark_ready(ws, case_id, head_sha=head_sha, task_spec_sha256="d" * 64)
    approved = load_yaml_strict(path)["curation"]
    assert approved["state"] == "ready" and approved["snapshot_attested"] is True
    assert approved["task_spec_sha256"] == "d" * 64 and approved["task_spec_approved_at"]

    cu.exclude_evidence(ws, case_id, candidate["source_id"], reason="duplicate")
    reopened = load_yaml_strict(path)["curation"]
    assert reopened["state"] == "draft" and reopened["snapshot_attested"] is False
    assert "task_spec_sha256" not in reopened and "task_spec_approved_at" not in reopened
    assert reopened["findings"] == accepted["findings"]
    cu.exclude_case(ws, case_id, reason="not_suitable")
    assert load_yaml_strict(path)["curation"]["state"] == "excluded"
    cu.reinclude_case(ws, case_id)
    reopened = load_yaml_strict(path)["curation"]
    assert reopened["state"] == "draft" and reopened["case_exclusion"] is None
    assert reopened["findings"] == accepted["findings"]

    raw = load_yaml_strict(path)
    raw["curation"].update(state="stale", snapshot_attested=False)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    bm.mark_ready(ws, case_id, head_sha=head_sha)  # derive the reviewed digest under the lock
    final = load_yaml_strict(path)
    expected = hashlib.sha256(build.render_task_spec(final, instruction=build.ASSIGNMENT_TEXT)).hexdigest()
    assert final["curation"]["state"] == "ready" and final["curation"]["snapshot_attested"] is True
    assert final["curation"]["task_spec_sha256"] == expected and final["curation"]["task_spec_approved_at"]
    bm.validate_case(ws, case_id)
    assert final["snapshot"] == original["snapshot"] and final["source"] == original["source"]
    assert import_path.read_bytes() == import_bytes


def test_classify_table_covers_precedence_and_dispositions() -> None:
    assert [band for band, _ in sorted(cu.BAND_RANK.items(), key=lambda kv: kv[1])] == [
        "review_first", "needs_judgment", "possibly_actioned", "likely_actioned", "withdrawn", "context", "decided"]
    assert list(cu.REASON_CODES) == ["resolved", "outdated", "anchor-delta-changed", "anchor-delta-deleted",
        "anchor-delta-renamed", "anchor-delta-binary", "pr-author-reply", "commit-non-ancestor", "commit-unavailable",
        "anchor-unavailable", "facts-missing", "dismissed", "decided-by-finding", "decided-by-exclusion",
        "decided-by-conflict", "non-candidate"]
    # (curation refs, is_candidate, dismissed, signals, facts_present, commit_relation,
    #  anchor_delta, expected_band, expected_disposition)
    cases = [({"finding", "exclusion"}, True, False, set(), True, "at_head", "unchanged", "decided", "conflict"),
        ({"finding"}, True, False, set(), True, "at_head", "unchanged", "decided", "finding"),
        (set(), False, False, set(), True, "at_head", "unchanged", "context", "n/a"),
        (set(), True, True, set(), True, "at_head", "unchanged", "withdrawn", "undecided"),
        (set(), True, False, {"resolved", "outdated"}, True, "at_head", "unchanged", "likely_actioned", "undecided"),
        (set(), True, False, {"resolved"}, True, "at_head", "unchanged", "possibly_actioned", "undecided"),
        (set(), True, False, set(), True, "non_ancestor", "unchanged", "needs_judgment", "undecided"),
        (set(), True, False, set(), True, "at_head", "unavailable", "needs_judgment", "undecided"),
        (set(), True, False, set(), False, "at_head", "unchanged", "needs_judgment", "undecided"),  # facts-missing
        (set(), True, False, set(), True, "at_head", "unchanged", "review_first", "undecided"),
        # a repeated identical signal still counts once:
        (set(), True, False, {"resolved"}, True, "at_head", "changed", "possibly_actioned", "undecided"),
    ]
    for refs, is_cand, dismissed, signals, has_facts, rel, delta, band, disp in cases:
        got_band, got_disp, _ = classify_evidence(refs=refs, is_candidate=is_cand, dismissed=dismissed, signals=signals,
            facts_present=has_facts, commit_relation=rel, anchor_delta=delta)
        assert got_band == band and got_disp == disp, (refs, signals, rel, delta, got_band)


def test_pr_author_reply_signal_respects_real_thread_identity() -> None:
    """REST replies belong to their chain root when GraphQL thread IDs are absent.

    Unrelated threadless records must never share a reply bucket.
    """

    def rec(source_id: str, kind: str, db_id: int, node_id: str, created: str,
            login: str, *, thread_id: str | None = None, reply_to_id: str | None = None) -> dict[str, object]:
        return {"source_id": source_id, "kind": kind, "database_id": db_id,
            "node_id": node_id, "created_at": created, "updated_at": created,
            "author": {"login": login, "type": "User"}, "thread_id": thread_id, "reply_to_id": reply_to_id,
        }

    # pr author login: alice
    records = [rec("github:review:100", "review", 100, "PRR_100", "2026-01-01T00:01:00Z", "carol"),
        rec("github:issue_comment:200", "issue_comment", 200, "IC_200", "2026-01-01T00:02:00Z", "dave"),
        # a thread-less, reply-less REST root in its own unrelated chain,
        # predating the later thread-less PR-author reply (the pre-fix
        # None-bucket scenario); a candidate, so a signal would render:
        rec("github:inline_comment:40", "inline_comment", 40, "DIFF_40", "2026-01-01T00:02:30Z", "bob"),
        # REST-only chain root + PR-author reply, both without thread ids:
        rec("github:inline_comment:10", "inline_comment", 10, "DIFF_10", "2026-01-01T00:03:00Z", "bob"),
        rec("github:inline_comment:11", "inline_comment", 11, "DIFF_11",
            "2026-01-01T00:04:00Z", "alice", reply_to_id="10"),
        # threaded chain (root + PR-author reply in one thread):
        rec("github:inline_comment:20", "inline_comment", 20, "DIFF_20", "2026-01-01T00:05:00Z", "bob", thread_id="T1"),
        rec("github:inline_comment:21", "inline_comment", 21, "DIFF_21",
            "2026-01-01T00:06:00Z", "alice", thread_id="T1", reply_to_id="DIFF_20"),
        # a different threaded thread with no PR-author reply:
        rec("github:inline_comment:30", "inline_comment", 30, "DIFF_30",
            "2026-01-01T00:07:00Z", "carol", thread_id="T2"),
    ]
    candidate_sids = [
        "github:inline_comment:40", "github:inline_comment:10", "github:inline_comment:20", "github:inline_comment:30",
    ]
    view = cu.prioritized_evidence({"evidence": records, "pull_request": {"author": {"login": "alice", "type": "User"}},
        "candidates": [{"source_id": sid} for sid in candidate_sids], "curation": {},
    })
    reasons = {sid: e["reasons"] for sid, e in view["by_source"].items()}
    assert "pr-author-reply" not in reasons["github:review:100"]
    assert "pr-author-reply" not in reasons["github:issue_comment:200"]
    assert "pr-author-reply" not in reasons["github:inline_comment:40"]
    assert "pr-author-reply" in reasons["github:inline_comment:10"]
    assert view["by_source"]["github:inline_comment:10"]["band"] == "possibly_actioned"
    assert "pr-author-reply" in reasons["github:inline_comment:20"]
    assert "pr-author-reply" not in reasons["github:inline_comment:30"]

    # the direct-caller fallback scan (no precomputed map) agrees:
    ctx = {"pr_author_login": "alice", "records": records}
    assert {"resolved", "outdated"} <= cu.collect_signals({**records[0], "resolved": True, "outdated": True}, ctx)
    assert "pr-author-reply" not in cu.collect_signals(records[0], ctx)
    assert "pr-author-reply" not in cu.collect_signals(records[1], ctx)
    assert "pr-author-reply" not in cu.collect_signals(records[2], ctx)
    assert "pr-author-reply" in cu.collect_signals(records[3], ctx)
    assert "pr-author-reply" in cu.collect_signals(records[5], ctx)
    assert "pr-author-reply" not in cu.collect_signals(records[7], ctx)
