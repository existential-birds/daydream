"""Preview writes a digest-pinned ledger; identical inputs yield byte-identical output (AC 7/8)."""
import json
import shutil
from pathlib import Path

import pytest

from daydream.archive.hydrate import HydrationError, MovingBranchError
from daydream.training.adjudication.harvest import AdjudicationDriftError, run_harvest
from daydream.training.adjudication.preview import run_preview


def _hydrated_index(tmp_path: Path, *, profiles: list[str] | None = None) -> Path:
    """Hydrated-index root in the sessions.jsonl shape the queue builder consumes
    (same fixture pattern as tests/test_cli_adjudicate.py: one ambiguous + one
    unanswered finding across two sessions)."""
    profiles = profiles or ["pr_review", "pr_review"]
    root = tmp_path / "index"
    root.mkdir()
    sessions = [
        {
            "session_id": "s1", "trajectory_id": "s1-traj", "segment_id": "s1-seg",
            "resolutions": [{
                "fingerprint": "fp-b", "disposition": "unanswered",
                "evidence": [{"reply_id": "r1", "body_sha256": "abc"}],
                "evidence_digest": "d2" * 32, "profile": profiles[0], "stack": "python",
            }],
        },
        {
            "session_id": "s2", "trajectory_id": "s2-traj", "segment_id": "s2-seg",
            "resolutions": [{
                "fingerprint": "fp-a", "disposition": "ambiguous",
                "evidence": [{"reply_id": "r2", "body_sha256": "abd"}],
                "evidence_digest": "d1" * 32, "profile": profiles[1], "stack": "python",
            }],
        },
    ]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    return root


def _mutate_one_digest(source: Path, target: Path) -> Path:
    """Copy the index and bump one finding's evidence_digest (digest drift)."""
    shutil.copytree(source, target)
    sessions_path = target / "sessions.jsonl"
    sessions = [json.loads(line) for line in sessions_path.read_text().splitlines() if line.strip()]
    for session in sessions:
        for resolution in session["resolutions"]:
            if resolution["fingerprint"] == "fp-a":
                resolution["evidence_digest"] = "ff" * 32
    sessions_path.write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    return target


def test_preview_ledger_is_deterministic_and_digest_pinned(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path)
    ledger_a = tmp_path / "ledger-a.json"
    ledger_b = tmp_path / "ledger-b.json"
    assert run_preview(root, ledger_a) == run_preview(root, ledger_b)
    a = json.loads(ledger_a.read_text())
    b = json.loads(ledger_b.read_text())
    assert a == b
    assert ledger_a.read_bytes() == ledger_b.read_bytes()
    item = a["items"][0]
    assert item["evidence_digest"] and item["record_id"]  # digest-pinned identity present
    assert item["evidence_digest"] in {"d1" * 32, "d2" * 32}
    assert len(a["items"]) == 2
    # sorted by record_id -> deterministic byte-identical output
    assert [i["record_id"] for i in a["items"]] == sorted(i["record_id"] for i in a["items"])
    assert a["ledger_digest"]


def test_preview_first_run_reports_no_drift(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path)
    result = run_preview(root, tmp_path / "ledger.json")
    assert result["drifted_record_ids"] == []


def test_preview_detects_evidence_drift(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path)
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    # Capture the pre-drift ledger BEFORE the re-preview overwrites it: the
    # pinned digests there are the ground truth drift detection must compare
    # against (re-reading the file after the second run_preview would return
    # the fresh post-drift ledger, proving nothing).
    prior_digests = {
        str(item["record_id"]): str(item["evidence_digest"])
        for item in json.loads(ledger.read_text(encoding="utf-8"))["items"]
    }
    drifted = _mutate_one_digest(root, tmp_path / "root2")  # copy + bump a digest
    result = run_preview(drifted, ledger)  # same ledger path: compares against prior preview
    assert result["drifted_record_ids"]  # drift surfaced, not merged silently
    # Drift names exactly the mutated finding (fp-a in session s2).
    from daydream.training.corpus_v2.identity import record_id as rid
    assert result["drifted_record_ids"] == [rid("s2", "s2-traj", "s2-seg", "fp-a")]
    fresh_digests = {
        str(item["record_id"]): str(item["evidence_digest"])
        for item in json.loads(ledger.read_text(encoding="utf-8"))["items"]
    }
    # The drifted record's pinned digest changed; the unchanged finding is not
    # flagged and its digest is pinned identically.
    for record in result["drifted_record_ids"]:
        assert prior_digests[record] != fresh_digests[record]
    assert len(result["drifted_record_ids"]) == 1
    unchanged = next(r for r in prior_digests if r not in result["drifted_record_ids"])
    assert prior_digests[unchanged] == fresh_digests[unchanged]


def test_preview_missing_sessions_file_raises_hydration_error(tmp_path: Path) -> None:
    with pytest.raises(HydrationError):
        run_preview(tmp_path, tmp_path / "ledger.json")


def test_preview_moving_branch_revision_is_rejected(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path)
    (root / "index-revision.txt").write_text("main\n", encoding="utf-8")
    with pytest.raises(MovingBranchError):
        run_preview(root, tmp_path / "ledger.json")


def test_preview_pinned_sha_revision_lands_in_ledger(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path)
    sha = "a" * 40
    (root / "index-revision.txt").write_text(sha + "\n", encoding="utf-8")
    result = run_preview(root, tmp_path / "ledger.json")
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert result["index_revision"] == ledger["index_revision"] == sha


def test_preview_malformed_evidence_raises_value_error_naming_source(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path)
    sessions_path = root / "sessions.jsonl"
    sessions = [json.loads(line) for line in sessions_path.read_text().splitlines() if line.strip()]
    del sessions[0]["resolutions"][0]["evidence_digest"]
    sessions_path.write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fp-b"):
        run_preview(root, tmp_path / "ledger.json")


def test_harvest_fails_closed_and_requeues_on_digest_drift(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path)
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    drifted = _mutate_one_digest(root, tmp_path / "root2")
    with pytest.raises(AdjudicationDriftError) as excinfo:
        run_harvest(drifted, ledger, tmp_path / "out")
    assert excinfo.value.requeued_record_ids  # affected findings requeued, nothing merged
    assert not (tmp_path / "out").exists()  # fail closed: export never written on drift


def test_harvest_identity_and_digests_stable_without_drift(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path)
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    summary_a = run_harvest(root, ledger, out_a)
    summary_b = run_harvest(root, ledger, out_b)
    assert summary_a["export_sha256"] == summary_b["export_sha256"]
    # Preview identities == harvest identities (AC 8 identity/digest stability).
    ledger_items = json.loads(ledger.read_text())["items"]
    exported = [
        json.loads(line) for line in (out_a / "adjudication.jsonl").read_text().splitlines()
    ]
    assert sorted(i["record_id"] for i in ledger_items) == sorted(e["record_id"] for e in exported)
    digests = {e["record_id"]: e["evidence_digest"] for e in exported}
    assert all(digests[i["record_id"]] == i["evidence_digest"] for i in ledger_items)


def test_preview_and_harvest_identity_digest_stability_gate(tmp_path: Path) -> None:
    """The parallel-implementation gate: preview and harvest must agree exactly."""
    root = _hydrated_index(tmp_path)
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    run_harvest(root, ledger, tmp_path / "out")
    exported = [
        json.loads(line) for line in (tmp_path / "out" / "adjudication.jsonl").read_text().splitlines()
    ]
    # Every queue item's identity AND digest are identical across preview ledger and harvest export.
    by_id = {e["record_id"]: e for e in exported}
    for item in json.loads(ledger.read_text())["items"]:
        assert item["record_id"] in by_id
        assert by_id[item["record_id"]]["evidence_digest"] == item["evidence_digest"]
    # record_id recomputation from the exported entries round-trips.
    from daydream.training.corpus_v2.identity import record_id as rid
    for e in exported:
        assert e["record_id"] == rid(e["session_id"], e["trajectory_id"], e["segment_id"], e["fingerprint"])


def test_preview_never_mutates_the_hydrated_index(tmp_path: Path) -> None:
    """Req 5: preview opens the SQLite index read-only, never appends
    label_observations, never writes resume-cache/complete markers."""
    import hashlib

    from tests.test_training_adjudication_materialize import _hydrated_sqlite_index

    root = _hydrated_sqlite_index(tmp_path)
    before = hashlib.sha256((root / "index.db").read_bytes()).hexdigest()
    before_tree = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    run_preview(root, tmp_path / "ledger.json")

    after = hashlib.sha256((root / "index.db").read_bytes()).hexdigest()
    after_tree = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert after == before  # index.db bytes untouched (no WAL checkpoint either)
    assert after_tree == before_tree  # no marker/cache files added anywhere


def test_posterior_feed_is_pr_review_only(tmp_path: Path) -> None:
    root = _hydrated_index(tmp_path, profiles=["pr_review", "task"])
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)

    # A decisive human verdict on every item makes both rows gold-eligible; the
    # pr_review-only posterior gate must then decide the posterior feed.
    from daydream.training.adjudication.observations import append_observation
    from daydream.training.adjudication.preview import _load_sessions
    from daydream.training.adjudication.queue import build_queue
    from daydream.training.labeler_versions import ADJUDICATION_LABELER_VERSION

    obs_path = tmp_path / "observations.jsonl"
    for item in build_queue(_load_sessions(root)[0]):
        append_observation(obs_path, {
            "record_id": str(item["record_id"]),
            "disposition": "accepted",
            "evidence_digest": str(item["evidence_digest"]),
            "evidence": item["evidence"],
            "labeler": "human-1",
            "role": "rater",
            "rationale": "confirmed by hand",
            "valid_at": "2026-08-30T10:00:00+00:00",
            "observed_at": "2026-08-30T10:00:01+00:00",
            "rubric_version": ADJUDICATION_LABELER_VERSION,
        })
    summary = run_harvest(root, ledger, tmp_path / "out", observations_path=obs_path)
    gold = [e for e in summary["exported"] if e["tier"] == "gold"]
    assert len(gold) == 2  # decisive human verdicts promote both rows to gold
    # The task-profile gold row must never enter the posterior feed...
    assert all(e["posterior_eligible"] is False for e in gold if e["profile"] != "pr_review")
    # ...while the pr_review gold row is the posterior feed.
    assert all(e["posterior_eligible"] for e in gold if e["profile"] == "pr_review")
