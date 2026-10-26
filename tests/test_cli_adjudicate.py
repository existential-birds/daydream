"""Real CLI annotation history and disposable queue behavior over records."""

import json
from pathlib import Path

import pytest

from daydream.dataset import LocalRecordStore
from daydream.training.record_evidence import sessions_from_snapshot
from tests.harness.adjudication import append_replies, judgment, record_store, reply_evidence, snapshot_id
from tests.harness.scripts import cli_main


def command(store: LocalRecordStore, verb: str, *args: str, pin: str | None = None) -> int:
    return cli_main(
        ["corpus", "adjudicate", verb, "--store", str(store.root), "--snapshot-id", pin or snapshot_id(store), *args]
    )


def test_cli_label_writes_durable_judgments_and_show_drains(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = record_store(tmp_path / "records")
    state = tmp_path / "state"
    assert command(store, "build", "--state-dir", str(state)) == 0
    queue = json.loads((state / "queue.json").read_text())
    assert len(queue) == 2 and all(item["item_uid"] for item in queue)
    assert (
        cli_main(
            [
                "corpus",
                "adjudicate",
                "label",
                "--state-dir",
                str(state),
                "--batch",
                "2",
                "--disposition",
                "accepted",
                "--rationale",
                "reproduced",
                "--labeler",
                "alice",
            ]
        )
        == 0
    )
    history = store.read_records()["observations"]
    assert len([o for o in history if o["source"] == "adjudication"]) == 2
    assert not (state / "observations.jsonl").exists()
    assert cli_main(["corpus", "adjudicate", "show", "--state-dir", str(state)]) == 0
    assert "unresolved: 0 / 2" in capsys.readouterr().out
    assert {
        r["disposition"]
        for s in sessions_from_snapshot(store.read_snapshot(snapshot_id(store)))
        for r in s["resolutions"]
    } == {"accepted"}


def test_cli_complete_population_report_export_materialize(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """One immutable population owns reporting, canonical identity, export and temporal pins.

    Reducing only prebuilt dictionaries misses a lost finding, model-only promotion, serializer
    provenance drift or mutation of the source store by one of the disposable CLI views.
    """
    from daydream.training import labeler_versions
    from daydream.training.record_identity import record_finding_id

    store = record_store(tmp_path / "records", ("unanswered", "ambiguous", "ambiguous", "unanswered", "missing"))
    assert command(store, "report") == 0
    initial_report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert initial_report["outcome_coverage"] == {"total": 0, "adjudicated": 0}
    assert initial_report["silver_task_only_count"] == 5
    assert initial_report["admission_gate"]["passes_80pct"] is False
    store.append_observation(judgment(store))
    store.append_observation(judgment(store, "rejected", item_index=1))
    store.append_observation(judgment(store, item_index=2, role="model-suggested", author="model"))
    state = tmp_path / "state"
    assert command(store, "build", "--state-dir", str(state)) == 0
    assert command(store, "report") == 0
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["outcome_coverage"] == {"total": 2, "adjudicated": 2}
    assert report["silver_task_only_count"] == 3
    assert report["class_balance"] == {"accepted": 1, "rejected": 1}
    assert report["unresolved"] == 0
    assert report["inter_rater"] == {"items": 0, "agreeing": 0}
    assert report["admission_gate"]["passes_80pct"] is True
    assert report["admission_gate"]["class_balance_ok"] is True
    assert report["strata"] == [{"stack": "python", "profile": "pr_review", "count": 5}]
    from daydream.training.adjudication.queue import build_queue

    sessions = sessions_from_snapshot(store.read_snapshot(snapshot_id(store)))
    open_queue = build_queue(sessions)
    complete_queue = build_queue(sessions, include_decisive=True)
    reversed_sessions = [{**session, "resolutions": list(reversed(session["resolutions"]))}
                         for session in reversed(sessions)]
    assert build_queue(reversed_sessions) == open_queue
    assert {item["item_uid"] for item in open_queue} == {"item:2", "item:3", "item:4"}
    assert {item["item_uid"] for item in complete_queue} == {"item:0", "item:1", "item:2", "item:3", "item:4"}
    assert [item["record_id"] for item in complete_queue] == sorted(str(item["record_id"]) for item in complete_queue)
    assert all(item["status"] == "open" for item in complete_queue)
    before = store.read_records()
    out = tmp_path / "materialized"
    assert command(store, "materialize", "--out-dir", str(out)) == 0
    records = [json.loads(line) for line in (out / "annotations.jsonl").read_text().splitlines()]
    assert len(records) == 5
    assert {row["item_uid"]: row["disposition"] for row in records} == {
        "item:0": "accepted",
        "item:1": "rejected",
        "item:2": "ambiguous",
        "item:3": "unanswered",
        "item:4": "missing",
    }
    assert [row["record_id"] for row in records] == sorted(
        record_finding_id("run-1", "run-1", "run-1", uid) for uid in ("item:0", "item:1", "item:2", "item:3", "item:4")
    )
    automatic = {row["item_uid"]: row for row in before["observations"] if row["role"] == "automatic"}
    for row in records:
        assert row["session_id"] == row["trajectory_id"] == row["segment_id"] == "run-1"
        assert row["profile"]["profile_name"] == "pr_review" and row["stack"] == "python"
        assert row["evidence_digest"] == automatic[row["item_uid"]]["evidence_digest"]
        assert row["classifier_version"] == labeler_versions.REPLY_CLASSIFIER_VERSION
        assert row["schema_version"] == f"annotation-snapshot/{labeler_versions.ANNOTATION_SNAPSHOT_SCHEMA_VERSION}"
        assert row["evidence_observed_at"] == "2100-01-01T00:00:00Z"
        assert "as_of" not in row
        assert row["reply_captures"][0]["text"] == "context"
        assert row["resolutions"][0]["reply_captures"] == row["reply_captures"]
    replay = tmp_path / "replay"
    assert command(store, "materialize", "--out-dir", str(replay)) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {p.name: p.read_bytes() for p in replay.iterdir()}
    cutoff = "2026-10-05T00:00:00Z"
    pinned = tmp_path / "pinned-valid-time"
    assert command(store, "materialize", "--out-dir", str(pinned), pin=snapshot_id(store, valid_before=cutoff)) == 0
    pinned_rows = [json.loads(line) for line in (pinned / "annotations.jsonl").read_text().splitlines()]
    assert pinned_rows == [{**row, "as_of": cutoff} for row in records]
    assert command(store, "harvest-snapshot", "--materialize-dir", str(out)) == 0
    assert command(store, "export", "--state-dir", str(state), "--out", str(tmp_path / "export.jsonl")) == 0
    exported = [json.loads(line) for line in (tmp_path / "export.jsonl").read_text().splitlines()]
    assert len(exported) == 5
    assert {row["record_id"] for row in exported} == {row["record_id"] for row in records}
    assert all(row["reply_captures"][0]["text"] == "context" for row in exported)
    assert store.read_records() == before
    assert not (store.root / "index.db").exists()
    materialized = {path.name: path.read_bytes() for path in out.iterdir()}
    append_replies(store)
    assert command(store, "harvest-snapshot", "--materialize-dir", str(out)) == 1
    assert "annotation evidence changed" in capsys.readouterr().out
    assert {path.name: path.read_bytes() for path in out.iterdir()} == materialized


def test_cli_retains_source_bound_correction_after_human_judgment(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records", ("unanswered",))
    automatic = store.read_records()["observations"][0]
    _semantic, capture = reply_evidence(automatic["item_uid"], "context")
    store.append_observation({**automatic, "observation_id": "captured", "correction": capture})
    store.append_observation(judgment(store))
    pin = snapshot_id(store)
    state, out = tmp_path / "state", tmp_path / "materialized"
    assert command(store, "build", "--state-dir", str(state), pin=pin) == 0
    assert json.loads((state / "queue.json").read_text())[0]["correction"] == capture
    assert command(store, "materialize", "--out-dir", str(out), pin=pin) == 0
    row = json.loads((out / "annotations.jsonl").read_text())
    assert row["disposition"] == "accepted" and row["correction"] == capture
    assert store.read_snapshot(pin).effective_judgment("run-1", automatic["item_uid"])["correction"] == capture


def test_cli_conflicting_raters_resolve_only_with_adjudicator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Persist precedence transitions and reload both current and frozen evidence generations.

    Rater chronology, conflict eligibility and adjudicator authority must survive the store/CLI
    join; direct reducer fixtures do not exercise that identity and digest binding.
    """
    from dataclasses import replace

    store = record_store(tmp_path / "records", ("unanswered",))
    automatic = store.read_snapshot(snapshot_id(store))
    assert automatic.effective_judgment("run-1", "item:0")["role"] == "automatic"
    store.append_observation(judgment(store, "accepted", author="alice", observed_at="2026-10-05T11:00:00Z"))
    first = store.read_snapshot(snapshot_id(store))
    assert first.effective_judgment("run-1", "item:0")["labeler"] == "alice"
    store.append_observation(judgment(store, "rejected", author="bob", observed_at="2026-10-05T11:00:00.500000+00:00"))
    conflicted = store.read_snapshot(snapshot_id(store))
    effective = conflicted.effective_judgment("run-1", "item:0")
    assert effective["labeler"] == "bob" and effective["disposition"] == "rejected"
    assert effective["conflict"] is True and effective["gold_eligible"] is False
    reversed_history = replace(conflicted, eligible_observations=tuple(reversed(conflicted.eligible_observations)))
    assert reversed_history.effective_judgment("run-1", "item:0") == effective
    assert sessions_from_snapshot(conflicted)[0]["resolutions"][0]["disposition"] == "ambiguous"
    assert command(store, "report", "--conflicts") == 0
    assert len(capsys.readouterr().out.strip()) == 64
    assert command(store, "report") == 0
    disputed_report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert disputed_report["inter_rater"] == {"items": 1, "agreeing": 0}
    assert disputed_report["outcome_coverage"] == {"adjudicated": 0, "total": 0}
    # Authority is independent of timestamp: an earlier explicit adjudicator wins over both
    # existing raters and a later rater appended after the resolution was persisted.
    store.append_observation(judgment(store, role="adjudicator", author="carol", observed_at="2026-10-05T10:30:00Z"))
    store.append_observation(judgment(store, "rejected", author="dave", observed_at="2026-10-05T12:00:00Z"))
    resolved = store.read_snapshot(snapshot_id(store))
    verdict = resolved.effective_judgment("run-1", "item:0")
    assert verdict["labeler"] == "carol" and verdict["disposition"] == "accepted"
    assert verdict["conflict"] is False and verdict["gold_eligible"] is True
    assert command(store, "report", "--conflicts") == 0
    assert capsys.readouterr().out == ""
    assert command(store, "report") == 0
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["inter_rater"] == {"items": 1, "agreeing": 1}
    assert report["outcome_coverage"] == {"adjudicated": 1, "total": 1}
    row = sessions_from_snapshot(resolved)[0]["resolutions"][0]
    assert row["disposition"] == "accepted" and row["gold_eligible"]
    assert store.read_snapshot(conflicted.snapshot["snapshot_id"]).effective_judgment("run-1", "item:0") == effective
    assert first.effective_judgment("run-1", "item:0")["disposition"] == "accepted"
    assert automatic.effective_judgment("run-1", "item:0")["disposition"] == "unanswered"


def test_cli_reopens_changed_evidence_without_reusing_human_judgment(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records", ("unanswered",), retain_replies=False)
    replies = (("9", "context\ncafé ☕\n"), ("10", "second perspective\n"))
    append_replies(store, replies, observed_at="2026-10-04T13:00:00Z", retain_replies=False)
    store.append_observation(judgment(store))
    first = snapshot_id(store)
    append_replies(store, replies, observation_id="captured", observed_at="2026-10-05T12:00:00Z")
    enriched = snapshot_id(store)
    enriched_resolution = sessions_from_snapshot(store.read_snapshot(enriched))[0]["resolutions"][0]
    assert enriched_resolution["disposition"] == "accepted"
    assert {capture["text"] for capture in enriched_resolution["reply_captures"]} == {text for _, text in replies}
    effective = store.read_snapshot(enriched).effective_judgment("run-1", "item:0")
    assert effective["reply_captures"] == enriched_resolution["reply_captures"]
    state = tmp_path / "state"
    assert command(store, "build", "--state-dir", str(state), pin=enriched) == 0
    assert json.loads((state / "queue.json").read_text())[0]["status"] == "open"
    # A later hash-only automatic observation must not discard acquired content.
    append_replies(store, replies, observation_id="hash-only-again", observed_at="2026-10-05T13:00:00Z",
                   retain_replies=False)
    still_captured = store.read_snapshot(snapshot_id(store)).effective_judgment("run-1", "item:0")
    assert still_captured["disposition"] == "accepted"
    assert still_captured["reply_captures"] == enriched_resolution["reply_captures"]
    append_replies(store, (("9", "new reply\n"), replies[1]), observation_id="edited-evidence")
    state = tmp_path / "state"
    assert command(store, "build", "--state-dir", str(state)) == 0
    row = json.loads((state / "queue.json").read_text())[0]
    assert row["status"] == "reopened" and row["prior_disposition"] == "accepted"
    assert sessions_from_snapshot(store.read_snapshot(first))[0]["resolutions"][0]["disposition"] == "accepted"
    assert sessions_from_snapshot(store.read_snapshot(first))[0]["resolutions"][0]["reply_captures"] == []
    assert sessions_from_snapshot(store.read_snapshot(enriched))[0]["resolutions"][0] == enriched_resolution
    latest = store.read_snapshot(snapshot_id(store))
    assert latest.effective_judgment("run-1", "item:0")["disposition"] == "unanswered"
    resolution = sessions_from_snapshot(latest)[0]["resolutions"][0]
    assert resolution["disposition"] == "unanswered"
    assert {capture["text"] for capture in row["reply_captures"]} == {"new reply\n", replies[1][1]}


@pytest.mark.parametrize("batch", ["0", "bad"])
def test_cli_invalid_batch_fails_before_state_reads(batch: str) -> None:
    assert (
        cli_main(
            [
                "corpus",
                "adjudicate",
                "label",
                "--state-dir",
                "/nonexistent",
                "--batch",
                batch,
                "--disposition",
                "accepted",
                "--rationale",
                "ok",
                "--labeler",
                "alice",
            ]
        )
        == 2
    )


def test_cli_label_preserves_canonical_evidence_digest_scheme(tmp_path: Path) -> None:
    from tests.harness.record_projection import seed_projection_store

    store = seed_projection_store(tmp_path, dispositions=("accepted",))
    state = tmp_path / "state"
    assert command(store, "build", "--state-dir", str(state)) == 0
    queued = json.loads((state / "queue.json").read_text())[0]
    assert queued["evidence"] and queued["evidence_digest_scheme"] == "canonical-json-v1"
    assert (
        cli_main(
            [
                "corpus",
                "adjudicate",
                "label",
                "--state-dir",
                str(state),
                "--record-id",
                queued["record_id"],
                "--disposition",
                "accepted",
                "--labeler",
                "chief",
                "--role",
                "adjudicator",
                "--rationale",
                "verified again",
            ]
        )
        == 0
    )
    observation = next(o for o in store.read_records()["observations"] if o["source"] == "adjudication")
    assert observation["evidence_digest"] == queued["evidence_digest"]
    assert observation["evidence_digest_scheme"] == "canonical-json-v1"


def test_cli_automatic_decisive_population_excluded_from_default_queue_preserved_in_materialization(
    tmp_path: Path,
) -> None:
    store = record_store(tmp_path / "records", ("accepted", "rejected", "unanswered"))
    state, out = tmp_path / "state", tmp_path / "out"
    assert command(store, "build", "--state-dir", str(state)) == 0
    queued = json.loads((state / "queue.json").read_text())
    assert len(queued) == 1 and queued[0]["disposition"] == "unanswered"
    assert command(store, "materialize", "--out-dir", str(out)) == 0
    assert len((out / "annotations.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("verb", ["build", "materialize", "export", "preview"])
def test_cli_derived_outputs_refuse_record_namespace_symlink_alias_without_mutation(tmp_path: Path, verb: str) -> None:
    store = record_store(tmp_path / "records")
    pin = snapshot_id(store)
    alias = tmp_path / "alias"
    alias.symlink_to(store.root / "runs", target_is_directory=True)
    before = {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    output = ["--out-dir", str(alias)] if verb == "materialize" else ["--state-dir", str(alias)]
    if verb == "export":
        output += ["--out", str(tmp_path / "export.jsonl")]
    assert command(store, verb, *output, pin=pin) == 1
    assert {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob("*") if p.is_file()} == before
