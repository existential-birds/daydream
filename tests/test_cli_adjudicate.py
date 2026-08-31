"""Real-path tests for the corpus adjudicate sub-verbs (style: test_cli_label.py)."""
import json
from pathlib import Path

import pytest

from daydream import cli


def _write_sessions(tmp_path: Path) -> Path:
    """One ambiguous + one unanswered finding across two sessions, in the
    hydrated-index shape the queue builder consumes (T1 session dicts)."""
    sessions = [
        {
            "session_id": "s1", "trajectory_id": "s1-traj", "segment_id": "s1-seg",
            "resolutions": [{
                "fingerprint": "fp-b", "disposition": "unanswered",
                "evidence": [{"reply_id": "r1", "body_sha256": "abc"}],
                "evidence_digest": "d2" * 32, "profile": "pr_review", "stack": "python",
            }],
        },
        {
            "session_id": "s2", "trajectory_id": "s2-traj", "segment_id": "s2-seg",
            "resolutions": [{
                "fingerprint": "fp-a", "disposition": "ambiguous",
                "evidence": [{"reply_id": "r2", "body_sha256": "abd"}],
                "evidence_digest": "d1" * 32, "profile": "pr_review", "stack": "python",
            }],
        },
    ]
    (tmp_path / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    return tmp_path


def test_adjudicate_label_records_human_observation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sessions(tmp_path)
    cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                "--state-dir", str(tmp_path / "adj")])
    queue = json.loads((tmp_path / "adj" / "queue.json").read_text())
    record_id = str(queue[0]["record_id"])
    rc = cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--record-id", record_id, "--disposition", "accepted",
         "--rationale", "reply confirms fix", "--labeler", "kevin"]
    )
    assert rc == 0
    lines = (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()
    obs = [json.loads(line) for line in lines]
    assert obs[-1]["disposition"] == "accepted" and obs[-1]["role"] == "rater"
    assert obs[-1]["record_id"] == record_id


def test_adjudicate_label_unknown_record_id_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sessions(tmp_path)
    cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                "--state-dir", str(tmp_path / "adj")])
    rc = cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--record-id", "a" * 64, "--disposition", "accepted",
         "--rationale", "reply confirms fix", "--labeler", "kevin"]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "a" * 64 in captured.out + captured.err


def test_adjudicate_label_batch_n_processes_unresolved_in_order(tmp_path: Path) -> None:
    _write_sessions(tmp_path)
    cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                "--state-dir", str(tmp_path / "adj")])
    rc = cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--batch", "1", "--disposition", "rejected", "--rationale", "stale finding",
         "--labeler", "kevin"]
    )
    assert rc == 0
    obs = [json.loads(line) for line in (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()]
    assert len(obs) == 1  # one observation per item; re-run advances, never duplicates
    rc2 = cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--batch", "1", "--disposition", "rejected", "--rationale", "stale finding",
         "--labeler", "kevin"]
    )
    assert rc2 == 0
    obs2 = [json.loads(line) for line in (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()]
    assert len(obs2) == 2 and obs2[0]["record_id"] != obs2[1]["record_id"]  # resume point advanced


def test_adjudicate_show_lists_queue_and_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sessions(tmp_path)
    cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                "--state-dir", str(tmp_path / "adj")])
    rc = cli._handle_corpus_command(["adjudicate", "show", "--state-dir", str(tmp_path / "adj")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ambiguous" in out and "unanswered" in out


def _seed_adjudicated(tmp_path: Path) -> tuple[Path, Path]:
    """Hydrated index + state dir with a built queue, one human decision, and
    a digest-pinned preview ledger (export harvest input)."""
    from daydream.training.adjudication.preview import run_preview

    root = tmp_path
    state = tmp_path / "adj"
    _write_sessions(root)
    assert cli._handle_corpus_command(
        ["adjudicate", "build", "--index-root", str(root), "--state-dir", str(state)]
    ) == 0
    queue = json.loads((state / "queue.json").read_text())
    record_id = str(queue[0]["record_id"])
    assert cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(state), "--record-id", record_id,
         "--disposition", "accepted", "--rationale", "reply confirms fix", "--labeler", "kevin"]
    ) == 0
    run_preview(root, state / "preview-ledger.json")
    return root, state


def _seed_with_conflict(tmp_path: Path) -> tuple[Path, Path]:
    """State dir where two records each have two disagreeing human raters;
    the first record's conflict is older than the second's."""
    from daydream.training.adjudication.observations import append_observation

    root = tmp_path
    state = tmp_path / "adj"
    _write_sessions(root)
    assert cli._handle_corpus_command(
        ["adjudicate", "build", "--index-root", str(root), "--state-dir", str(state)]
    ) == 0
    queue = json.loads((state / "queue.json").read_text())
    obs_path = state / "observations.jsonl"
    older_record, newer_record = queue[0], queue[1]
    common = {
        "rationale": "independent pass", "role": "rater",
        "valid_at": "2024-01-01T00:00:00+00:00", "rubric_version": "984-adjudicate-r1",
    }
    append_observation(obs_path, {
        "record_id": str(older_record["record_id"]), "disposition": "accepted",
        "evidence_digest": str(older_record["evidence_digest"]),
        "evidence": older_record["evidence"], "labeler": "rater-one",
        "observed_at": "2024-01-01T00:00:00+00:00", **common,
    })
    append_observation(obs_path, {
        "record_id": str(older_record["record_id"]), "disposition": "rejected",
        "evidence_digest": str(older_record["evidence_digest"]),
        "evidence": older_record["evidence"], "labeler": "old-conflict",
        "observed_at": "2024-01-02T00:00:00+00:00", **common,
    })
    append_observation(obs_path, {
        "record_id": str(newer_record["record_id"]), "disposition": "rejected",
        "evidence_digest": str(newer_record["evidence_digest"]),
        "evidence": newer_record["evidence"], "labeler": "rater-two",
        "observed_at": "2024-02-01T00:00:00+00:00", **common,
    })
    append_observation(obs_path, {
        "record_id": str(newer_record["record_id"]), "disposition": "accepted",
        "evidence_digest": str(newer_record["evidence_digest"]),
        "evidence": newer_record["evidence"], "labeler": "new-conflict",
        "observed_at": "2024-02-02T00:00:00+00:00", **common,
    })
    return root, state


def test_export_writes_projector_shape_and_dry_run_validates_only(tmp_path: Path) -> None:
    root, state = _seed_adjudicated(tmp_path)
    rc = cli._handle_corpus_command(
        ["adjudicate", "export", "--index-root", str(root), "--state-dir", str(state),
         "--out", str(tmp_path / "export.jsonl"), "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "export.jsonl").exists()  # dry-run validates without writing
    rc = cli._handle_corpus_command(
        ["adjudicate", "export", "--index-root", str(root), "--state-dir", str(state),
         "--out", str(tmp_path / "export.jsonl")])
    assert rc == 0
    rows = [json.loads(line) for line in (tmp_path / "export.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:  # loadable by the projector consumer without manual JSON editing
        assert set(row) >= {"record_id", "fingerprint", "disposition", "evidence",
                            "evidence_digest", "exclusion_reason"}


def test_export_requires_out_without_dry_run(tmp_path: Path) -> None:
    root, state = _seed_adjudicated(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        cli._handle_corpus_command(
            ["adjudicate", "export", "--index-root", str(root), "--state-dir", str(state)])
    assert excinfo.value.code == 2


def test_report_subverb_prints_coverage_and_strata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, state = _seed_adjudicated(tmp_path)
    rc = cli._handle_corpus_command(
        ["adjudicate", "report", "--index-root", str(root), "--state-dir", str(state)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "outcome-bearing" in out and "silver/task-only" in out
    assert "inter-rater" in out.lower()


def test_conflict_review_lists_disagreeing_raters_oldest_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root, state = _seed_with_conflict(tmp_path)
    rc = cli._handle_corpus_command(
        ["adjudicate", "report", "--index-root", str(root), "--state-dir", str(state),
         "--conflicts"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.index("old-conflict") < out.index("new-conflict")  # oldest-first ordering


def test_adjudicate_unknown_subverb_exits_2() -> None:
    with pytest.raises(SystemExit):
        cli._handle_corpus_command(["adjudicate", "bogus"])


def test_adjudicate_bare_invocation_exits_2() -> None:
    with pytest.raises(SystemExit):
        cli._handle_corpus_command(["adjudicate"])


def test_adjudicate_build_preserves_observations_and_is_idempotent(tmp_path: Path) -> None:
    _write_sessions(tmp_path)
    for _ in range(2):
        rc = cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                         "--state-dir", str(tmp_path / "adj")])
        assert rc == 0
    queue = json.loads((tmp_path / "adj" / "queue.json").read_text())
    assert len(queue) == 2


# ---- snapshot pipeline verbs (issue #1055, task 6) ----

from daydream.training.adjudication.cli import handle_adjudicate  # noqa: E402
from daydream.training.adjudication.materialize import run_materialize  # noqa: E402
from daydream.training.labeler_versions import (  # noqa: E402
    ADJUDICATION_LABELER_VERSION,
    REPLY_CLASSIFIER_VERSION,
    RUBRIC_SCHEMA_VERSION,
)

_PIN_ARGS = [
    "--curation-id", "cur-1",
    "--sanitized-hub-commit", "a" * 40,
    "--source-hub-commit", "b" * 40,
    "--evidence-observed-at", "2026-01-01T00:00:00+00:00",
    "--as-of", "2026-02-01T00:00:00+00:00",
]


def _cli_index(tmp_path: Path) -> Path:
    root = tmp_path / "index"
    root.mkdir(parents=True, exist_ok=True)
    sessions = [{
        "session_id": "s1", "trajectory_id": "t", "segment_id": "g",
        "resolutions": [{
            "fingerprint": "fp", "disposition": "unanswered",
            "evidence": [{"reply_id": 1, "body_sha256": "x"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
        }],
    }]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    return root


def test_cli_materialize_writes_sessions_and_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    code = handle_adjudicate([
        "materialize", "--index-root", str(_cli_index(tmp_path)),
        "--out-dir", str(out), "--archive-index-digest", "c" * 64, *_PIN_ARGS,
    ])
    assert code == 0
    assert (out / "sessions.jsonl").is_file()
    manifest = json.loads((out / "preview-manifest.json").read_text())
    assert manifest["curation_id"] == "cur-1"
    printed = capsys.readouterr().out
    assert "snapshot" in printed and "record" in printed  # S2 operator summary


def test_cli_materialize_matches_run_materialize_output(tmp_path: Path) -> None:
    out_cli = tmp_path / "out-cli"
    code = handle_adjudicate([
        "materialize", "--index-root", str(_cli_index(tmp_path)),
        "--out-dir", str(out_cli), "--archive-index-digest", "c" * 64, *_PIN_ARGS,
    ])
    assert code == 0
    out_direct = tmp_path / "out-direct"
    run_materialize(_cli_index(tmp_path / "direct"), out_direct, pin={
        "curation_id": "cur-1", "sanitized_hub_commit": "a" * 40,
        "source_hub_commit": "b" * 40, "archive_index_digest": "c" * 64,
        "evidence_observed_at": "2026-01-01T00:00:00+00:00",
        "as_of": "2026-02-01T00:00:00+00:00",
        "labeler_version": ADJUDICATION_LABELER_VERSION,
        "rubric_version": RUBRIC_SCHEMA_VERSION,
        "classifier_version": REPLY_CLASSIFIER_VERSION,
    })
    assert (out_cli / "sessions.jsonl").read_bytes() == (out_direct / "sessions.jsonl").read_bytes()
    assert (out_cli / "preview-manifest.json").read_bytes() == (out_direct / "preview-manifest.json").read_bytes()


def test_cli_materialize_missing_index_exits_1(tmp_path: Path) -> None:
    assert handle_adjudicate([
        "materialize", "--index-root", str(tmp_path / "nope"),
        "--out-dir", str(tmp_path / "out"), "--archive-index-digest", "c" * 64, *_PIN_ARGS,
    ]) == 1


def test_cli_materialize_missing_pin_component_exits_1(tmp_path: Path) -> None:
    assert handle_adjudicate([
        "materialize", "--index-root", str(_cli_index(tmp_path)),
        "--out-dir", str(tmp_path / "out"), "--archive-index-digest", "c" * 64,
        "--curation-id", "cur-1",  # missing the other pin flags
    ]) == 1


def test_cli_materialize_malformed_invocation_exits_2() -> None:
    with pytest.raises(SystemExit) as excinfo:
        handle_adjudicate(["materialize", "--index-root", "/tmp"])  # missing required pin flags
    assert excinfo.value.code == 2
