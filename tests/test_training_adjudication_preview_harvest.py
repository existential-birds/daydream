"""Preview writes a digest-pinned ledger; identical inputs yield byte-identical output (AC 7/8)."""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.hydrate import HydrationError, MovingBranchError
from daydream.training.adjudication.canonical import AnnotationDriftError
from daydream.training.adjudication.harvest import build_export_entries
from daydream.training.adjudication.observations import append_observation
from daydream.training.adjudication.preview import _load_sessions, run_preview
from daydream.training.adjudication.queue import build_queue
from daydream.training.corpus_projection.identity import record_id as rid
from daydream.training.labeler_versions import ADJUDICATION_LABELER_VERSION
from tests.harness.adjudication import write_sessions_index, write_sessions_jsonl
from tests.test_training_adjudication_materialize import _hydrated_sqlite_index


def _mutate_one_digest(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    sessions_path = target / "sessions.jsonl"
    sessions = [json.loads(line) for line in sessions_path.read_text().splitlines() if line.strip()]
    for session in sessions:
        for resolution in session["resolutions"]:
            if resolution["fingerprint"] == "fp-a":
                resolution["evidence_digest"] = "ff" * 32
    write_sessions_jsonl(target, sessions)
    return target


def test_preview_ledger_is_deterministic_and_digest_pinned(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
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
    assert [i["record_id"] for i in a["items"]] == sorted(i["record_id"] for i in a["items"])
    assert a["ledger_digest"]

def test_preview_first_run_reports_no_drift(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    result = run_preview(root, tmp_path / "ledger.json")
    assert result["drifted_record_ids"] == []

def test_preview_detects_evidence_drift(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    # Capture the old ledger before preview overwrites it; otherwise the comparison is vacuous.
    prior_digests = {str(item["record_id"]): str(item["evidence_digest"])
        for item in json.loads(ledger.read_text(encoding="utf-8"))["items"]
    }
    drifted = _mutate_one_digest(root, tmp_path / "root2")  # copy + bump a digest
    result = run_preview(drifted, ledger)  # same ledger path: compares against prior preview
    assert result["drifted_record_ids"]  # drift surfaced, not merged silently
    assert result["drifted_record_ids"] == [rid("s2", "s2-traj", "s2-seg", "fp-a")]
    fresh_digests = {str(item["record_id"]): str(item["evidence_digest"])
        for item in json.loads(ledger.read_text(encoding="utf-8"))["items"]
    }
    for record in result["drifted_record_ids"]:
        assert prior_digests[record] != fresh_digests[record]
    assert len(result["drifted_record_ids"]) == 1
    unchanged = next(r for r in prior_digests if r not in result["drifted_record_ids"])
    assert prior_digests[unchanged] == fresh_digests[unchanged]

def test_preview_missing_sessions_file_raises_hydration_error(tmp_path: Path) -> None:
    with pytest.raises(HydrationError):
        run_preview(tmp_path, tmp_path / "ledger.json")

def test_preview_moving_branch_revision_is_rejected(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    (root / "index-revision.txt").write_text("main\n", encoding="utf-8")
    with pytest.raises(MovingBranchError):
        run_preview(root, tmp_path / "ledger.json")

def test_preview_pinned_sha_revision_lands_in_ledger(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    sha = "a" * 40
    (root / "index-revision.txt").write_text(sha + "\n", encoding="utf-8")
    result = run_preview(root, tmp_path / "ledger.json")
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert result["index_revision"] == ledger["index_revision"] == sha

def test_preview_malformed_evidence_raises_value_error_naming_source(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    sessions_path = root / "sessions.jsonl"
    sessions = [json.loads(line) for line in sessions_path.read_text().splitlines() if line.strip()]
    del sessions[0]["resolutions"][0]["evidence_digest"]
    write_sessions_jsonl(root, sessions)
    with pytest.raises(ValueError, match="fp-b"):
        run_preview(root, tmp_path / "ledger.json")

def test_export_fails_closed_and_requeues_on_digest_drift(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    drifted = _mutate_one_digest(root, tmp_path / "root2")
    with pytest.raises(AnnotationDriftError) as excinfo:
        build_export_entries(drifted, ledger)
    assert excinfo.value.requeued_record_ids  # affected findings requeued, nothing merged

def test_export_identity_and_digests_stable_without_drift(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    rows_a = build_export_entries(root, ledger)
    rows_b = build_export_entries(root, ledger)
    assert rows_a == rows_b
    ledger_items = json.loads(ledger.read_text())["items"]
    assert sorted(i["record_id"] for i in ledger_items) == sorted(e["record_id"] for e in rows_a)
    digests = {e["record_id"]: e["evidence_digest"] for e in rows_a}
    assert all(digests[i["record_id"]] == i["evidence_digest"] for i in ledger_items)

def test_preview_and_export_identity_digest_stability_gate(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    rows = build_export_entries(root, ledger)
    by_id = {e["record_id"]: e for e in rows}
    for item in json.loads(ledger.read_text())["items"]:
        assert item["record_id"] in by_id
        assert by_id[item["record_id"]]["evidence_digest"] == item["evidence_digest"]
    for e in rows:
        assert e["record_id"] == rid(e["session_id"], e["trajectory_id"], e["segment_id"], e["fingerprint"])

def test_preview_never_mutates_the_hydrated_index(tmp_path: Path) -> None:
    """Preview must leave index rows, resume cache, and completion markers unchanged."""

    root = _hydrated_sqlite_index(tmp_path)
    before = hashlib.sha256((root / "index.db").read_bytes()).hexdigest()
    before_tree = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    run_preview(root, tmp_path / "ledger.json")

    after = hashlib.sha256((root / "index.db").read_bytes()).hexdigest()
    after_tree = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert after == before  # index.db bytes untouched (no WAL checkpoint either)
    assert after_tree == before_tree  # no marker/cache files added anywhere

def test_posterior_feed_is_pr_review_only(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index", profiles=["pr_review", "task"])
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)

    # Both rows are gold so posterior-profile eligibility alone decides the outcome.

    obs_path = tmp_path / "observations.jsonl"
    for item in build_queue(_load_sessions(root)[0]):
        append_observation(obs_path, {"record_id": str(item["record_id"]), "disposition": "accepted",
            "evidence_digest": str(item["evidence_digest"]), "evidence": item["evidence"], "labeler": "human-1",
            "role": "rater", "rationale": "confirmed by hand", "valid_at": "2026-08-30T10:00:00+00:00",
            "observed_at": "2026-08-30T10:00:01+00:00", "rubric_version": ADJUDICATION_LABELER_VERSION,
        })
    exported = build_export_entries(root, ledger, observations_path=obs_path)
    gold = [e for e in exported if e["tier"] == "gold"]
    assert len(gold) == 2  # decisive human verdicts promote both rows to gold
    assert all(e["posterior_eligible"] is False for e in gold if e["profile"] != "pr_review")
    assert all(e["posterior_eligible"] for e in gold if e["profile"] == "pr_review")


def test_export_requires_human_judgment_against_current_evidence(tmp_path: Path) -> None:
    root = write_sessions_index(tmp_path / "index")
    sessions = _load_sessions(root)[0]
    resolution = sessions[0]["resolutions"][0]
    resolution["disposition"] = "accepted"
    write_sessions_jsonl(root, sessions)
    ledger = tmp_path / "ledger.json"
    run_preview(root, ledger)
    item = next(i for i in build_queue(sessions, include_decisive=True) if i["disposition"] == "accepted")
    observations = tmp_path / "observations.jsonl"
    judgment = {
        "record_id": str(item["record_id"]), "disposition": "accepted", "evidence_digest": "stale",
        "evidence": item["evidence"], "labeler": "human-1", "role": "rater", "rationale": "checked",
        "valid_at": "2026-08-30T10:00:00+00:00", "observed_at": "2026-08-30T10:00:01+00:00",
        "rubric_version": ADJUDICATION_LABELER_VERSION,
    }
    append_observation(observations, judgment)
    stale = next(row for row in build_export_entries(root, ledger, observations_path=observations)
                 if row["record_id"] == item["record_id"])
    assert stale["disposition"] == "accepted"
    assert stale["tier"] == "task-only" and stale["posterior_eligible"] is False

    append_observation(observations, {
        **judgment, "evidence_digest": str(item["evidence_digest"]),
        "observed_at": "2026-08-30T10:00:02+00:00",
    })
    fresh = next(row for row in build_export_entries(root, ledger, observations_path=observations)
                 if row["record_id"] == item["record_id"])
    assert fresh["tier"] == "gold" and fresh["posterior_eligible"] is True


def test_session_revision_hash_binds_the_same_bytes_as_parsed_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = write_sessions_index(tmp_path / "index")
    revision_file = root / "index-revision.txt"
    revision_file.unlink(missing_ok=True)
    source = root / "sessions.jsonl"
    captured = source.read_bytes()
    replacement = captured.replace(b'"unanswered"', b'"missing"')
    read_text = Path.read_text
    read_bytes = Path.read_bytes
    replaced = False

    def replace_after_capture(path: Path) -> None:
        nonlocal replaced
        if path == source and not replaced:
            source.write_bytes(replacement)
            replaced = True

    def changing_text(path: Path, *args: Any, **kwargs: Any) -> str:
        value = read_text(path, *args, **kwargs)
        replace_after_capture(path)
        return value

    def changing_bytes(path: Path) -> bytes:
        value = read_bytes(path)
        replace_after_capture(path)
        return value

    monkeypatch.setattr(Path, "read_text", changing_text)
    monkeypatch.setattr(Path, "read_bytes", changing_bytes)
    sessions, revision = _load_sessions(root)
    assert sessions[0]["resolutions"][0]["disposition"] == "unanswered"
    assert revision == hashlib.sha256(captured).hexdigest()
    assert source.read_bytes() == replacement
