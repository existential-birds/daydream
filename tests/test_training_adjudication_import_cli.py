"""Drive adjudication CLI import against real SQLite archives written by production writers. Publication
tests replace only the external Hub client.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from daydream.archive.hydrate import PublicDestinationError
from daydream.archive.importer import REDACTED_PATH
from daydream.archive.index import _get_connection, append_label_observation, readonly_connection, upsert_run
from daydream.commands.corpus import _handle_corpus_command
from daydream.training.adjudication import cli as adjudication_cli
from daydream.training.adjudication.cli import handle_adjudicate
from daydream.training.adjudication.publish import publish_annotation_state, resume_annotation_state
from tests.fixtures.training.build_hub_snapshot import build_annotations_hub
from tests.harness.adjudication import write_sessions_jsonl
from tests.harness.trajectory import make_manifest

_OBSERVED = "2026-04-30T00:00:00+00:00"
_VALID_AT = "2026-04-29T00:00:00+00:00"


def _seed_session(root: Path, session_id: str, *, evidence_sha: str, labels: list[str], **observation_kwargs: Any,
) -> None:
    """Seed a run and automatic observation through real writers. Session-specific base/head SHAs prevent
    identity-fallback collisions; extra kwargs extend the observation.
    """
    head = hashlib.sha256(session_id.encode()).hexdigest()
    base = hashlib.sha256(("base-" + session_id).encode()).hexdigest()
    upsert_run(root, make_manifest(session_id=session_id, repo_slug="org/repo", head_sha=head, base_sha=base))
    append_label_observation(
        root, session_id, labels=labels, pr_state=None, labeler_version="980-rubric-r2", evidence_sha=evidence_sha,
        valid_at=_VALID_AT, reply_evidence_digest=None, reward_version=None, has_posterior=False, source="auto",
        observed_at=_OBSERVED, **observation_kwargs,
    )


def _source_row_count(roots: list[Path]) -> int:
    total = 0
    for root in roots:
        with closing(readonly_connection(root)) as conn:
            total += int(conn.execute("SELECT COUNT(*) FROM label_observations").fetchone()[0])
    return total


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_args(
    *roots: Path, state_dir: Path, extra: list[str], index_root: Path | None = None, archive_dir: Path | None = None,
) -> list[str]:
    argv: list[str] = ["import-local-observations"]
    for root in roots:
        argv += ["--archive-root", str(root)]
    if index_root is None:
        index_root = state_dir.parent / "idx"
        index_root.mkdir(exist_ok=True)
        # An independent pinned inventory is required: an empty index cannot
        # authorize backup rows by linking the backup to itself.
        sessions = []
        for source in roots:
            if not (source / "index.db").is_file():
                continue
            with closing(readonly_connection(source)) as conn:
                source_runs = conn.execute("SELECT * FROM runs").fetchall()
            for row in source_runs:
                upsert_run(index_root, make_manifest(session_id=row["session_id"], repo_slug=row["repo_slug"],
                    base_sha=row["base_sha"], head_sha=row["head_sha"],
                ))
                sessions.append({"session_id": row["session_id"], "trajectory_id": row["session_id"],
                    "segment_id": row["session_id"], "resolutions": [{
                        "fingerprint": "fp-" + row["session_id"], "disposition": "unanswered",
                        "evidence": [], "evidence_digest": "d" * 64,
                    }],
                })
        write_sessions_jsonl(index_root, sessions)
    if archive_dir is None:
        archive_dir = state_dir.parent / "archive"
    argv += ["--index-root", str(index_root), "--archive-dir", str(archive_dir), "--state-dir", str(state_dir), *extra]
    return argv


def _materialized_snapshot(root: Path, session: str, fingerprint: str) -> Path:
    """Materialize one session/finding in the hydrated-index shape used for import identity matching."""
    root.mkdir(parents=True, exist_ok=True)
    upsert_run(root, make_manifest(
        session_id=session, repo_slug="org/repo", head_sha=hashlib.sha256(session.encode()).hexdigest(),
        base_sha=hashlib.sha256(("base-" + session).encode()).hexdigest(),
    ))
    sessions = [{"session_id": session, "trajectory_id": session, "segment_id": session,
        "resolutions": [{"fingerprint": fingerprint, "disposition": "unanswered",
            "evidence": [{"reply_id": 1, "body_sha256": "abc", "created_at": "2026-01-01T00:00:00+00:00"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python", "comment_id": 7,
        }],
    }]
    write_sessions_jsonl(root, sessions)
    return root


def test_cli_import_writes_report_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root_a = tmp_path / "src-a"
    root_b = tmp_path / "src-b"
    _seed_session(root_a, "sess-a1", evidence_sha="e" * 64, labels=["accepted"])
    _seed_session(root_a, "sess-a2", evidence_sha="f" * 64, labels=["rejected"])
    _seed_session(root_b, "sess-b1", evidence_sha="1" * 64, labels=["accepted"])
    roots = [root_a, root_b]
    total = _source_row_count(roots)
    assert total == 3

    state = tmp_path / "state"
    rc = _handle_corpus_command(
        ["adjudicate", *_import_args(*roots, state_dir=state, extra=["--dry-run", "--json"])]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert sum(report["accounting"].values()) == total
    assert report["sources"] == [
        {"archive_root": str(root_a), "row_count": 2, "source_digest": _digest(root_a / "index.db")},
        {"archive_root": str(root_b), "row_count": 1, "source_digest": _digest(root_b / "index.db")},
    ]
    assert not state.exists()

def test_cli_real_path_real_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    _seed_session(src, "sess-2", evidence_sha="f" * 64, labels=["rejected"])
    before = (src / "index.db").read_bytes()

    state = tmp_path / "state"
    archive = tmp_path / "archive"
    rc = _handle_corpus_command(["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[])])
    assert rc == 0
    capsys.readouterr()  # drain the human-readable run before the --json re-run
    assert (src / "index.db").read_bytes() == before

    report = json.loads((state / "import-report.json").read_text(encoding="utf-8"))
    assert report["dry_run"] is False
    assert sum(report["accounting"].values()) == 2
    assert report["identity_summary"]["sess-1"]["matched_by"] == "repo_slug_sha"
    assert report["identity_summary"]["sess-2"]["matched_by"] == "repo_slug_sha"
    ledger = json.loads((state / "import-ledger.json").read_text(encoding="utf-8"))
    assert ledger["accounting"] == report["accounting"]
    assert {entry["session_id"] for entry in ledger["observations"]} == {"sess-1", "sess-2"}

    with closing(sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT session_id, evidence_sha FROM label_observations ORDER BY session_id").fetchall()
    assert rows == [("sess-1", "e" * 64), ("sess-2", "f" * 64)]
    assert not (state / "index.db").exists()

    rc = _handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=["--json"])]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["merge"]["appended"] == 0
    report_bytes = (state / "import-report.json").read_bytes()
    rc = _handle_corpus_command(["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[])])
    assert rc == 0
    assert (state / "import-report.json").read_bytes() == report_bytes
    assert (src / "index.db").read_bytes() == before

def test_cli_import_seeds_every_eligible_hydrated_run_for_checkpoint_resume(tmp_path: Path,) -> None:
    """Keep all eligible hydrated runs in the checkpoint, including sessions without surviving backup
    observations, so fresh-VM harvest can append without inventing history.
    """
    stage = tmp_path / "hydrated"
    for session_id in ("sess-a", "sess-b"):
        head = hashlib.sha256(session_id.encode()).hexdigest()
        base = hashlib.sha256(("base-" + session_id).encode()).hexdigest()
        upsert_run(stage, make_manifest(session_id=session_id, repo_slug="org/repo", head_sha=head, base_sha=base))
        rubric = {"per_finding_resolutions": [{
                    "fingerprint": f"fp-{session_id}", "comment_id": 7, "disposition": "accepted",
                    "evidence": [{"reply_id": 1, "body_sha256": "abc"}], "evidence_digest": "d" * 32,
                }
            ]
        }
        assert append_label_observation(
            stage, session_id, labels=["finding-accepted"], pr_state=None, labeler_version="980-rubric-r2",
            evidence_sha=head, rubric_json=json.dumps(rubric), source="auto", observed_at=_OBSERVED,
        )
    (stage / "downloads" / ("a" * 40)).mkdir(parents=True)

    backup = tmp_path / "backup"
    sess_a_head = hashlib.sha256("sess-a".encode()).hexdigest()
    _seed_session(backup, "sess-a", evidence_sha=sess_a_head, labels=["accepted"])
    target = tmp_path / "checkpoint-state"

    assert _handle_corpus_command(
        ["adjudicate", *_import_args(backup, state_dir=target, index_root=stage, archive_dir=target, extra=[])]
    ) == 0

    with closing(sqlite3.connect(f"file:{target / 'index.db'}?mode=ro", uri=True)) as conn:
        run_ids = {str(row[0]) for row in conn.execute("SELECT session_id FROM runs ORDER BY session_id")}
        observation_ids = {str(row[0])
            for row in conn.execute("SELECT DISTINCT session_id FROM label_observations ORDER BY session_id")
        }
    assert run_ids == {"sess-a", "sess-b"}
    assert observation_ids == {"sess-a"}

def test_cli_reimport_does_not_displace_newer_target_runs_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An older backup cannot displace newer target run fields or writer-owned cache mirrors; both runs and
    observations remain append-only.
    """
    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    state = tmp_path / "state"
    archive = tmp_path / "archive"
    assert _handle_corpus_command(["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[])]
    ) == 0

    head = hashlib.sha256("sess-1".encode()).hexdigest()
    base = hashlib.sha256(("base-" + "sess-1").encode()).hexdigest()
    upsert_run(archive,
        make_manifest(session_id="sess-1", repo_slug="org/repo", head_sha=head, base_sha=base,
            archived_at="2026-05-01T00:00:00+00:00", status="partial", profile_name="profile-v2", total_cost_usd=99.5,
        ),
    )
    def target_run() -> dict[str, Any]:
        with closing(readonly_connection(archive)) as conn:
            return dict(conn.execute(
                "SELECT archived_at, status, profile_name, total_cost_usd, "
                "outcome_labels, labeled_at, cost_per_finding_usd FROM runs WHERE session_id = 'sess-1'"
            ).fetchone())

    newer = target_run()
    args = _import_args(src, state_dir=state, archive_dir=archive, extra=["--json"])
    # Pinned inventory fills a NULL target value; populated target values win.
    conn = _get_connection(tmp_path / "idx")
    conn.execute("UPDATE runs SET cost_per_finding_usd = 2.5 WHERE session_id = 'sess-1'")
    conn.commit()
    conn.close()
    assert _handle_corpus_command(["adjudicate", *args]) == 0
    capsys.readouterr()
    after = target_run()
    populated = {key: value for key, value in newer.items() if value is not None}
    assert {key: after[key] for key in populated} == populated
    assert newer["cost_per_finding_usd"] is None
    assert after["cost_per_finding_usd"] == 2.5

def test_cli_overlapping_backups_dedupe_accounting(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root_a = tmp_path / "backup-a"
    root_b = tmp_path / "backup-b"
    _seed_session(root_a, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    _seed_session(root_a, "sess-2", evidence_sha="f" * 64, labels=["rejected"])
    for session_id, sha in (("sess-1", "e" * 64), ("sess-2", "f" * 64)):
        _seed_session(root_b, session_id, evidence_sha=sha, labels=["accepted" if sha[0] == "e" else "rejected"])
    state = tmp_path / "state"
    rc = _handle_corpus_command(
        ["adjudicate", *_import_args(root_a, root_b, state_dir=state, extra=["--dry-run", "--json"])]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["deduped_count"] == 2
    assert (sum(report["accounting"].values()) + report["deduped_count"]
        == _source_row_count([root_a, root_b])
    )


def _seed_publishable_state(tmp_path: Path) -> tuple[Path, Path]:
    """Build an empty queue through the real CLI and seed the required empty observation/preview payloads
    for checkpoint publication.
    """
    index_root = tmp_path / "index-root"
    index_root.mkdir()
    (index_root / "sessions.jsonl").write_text("", encoding="utf-8")
    state = tmp_path / "state"
    assert _handle_corpus_command(
        ["adjudicate", "build", "--index-root", str(index_root), "--state-dir", str(state)]
    ) == 0
    (state / "observations.jsonl").touch()
    (state / "preview-ledger.json").write_text("{}", encoding="utf-8")
    return index_root, state


def _write_manifest(tmp_path: Path) -> Path:
    p = tmp_path / "preview-manifest.json"
    p.write_text(json.dumps({"curation_id": "cur-import", "snapshot_id": "e" * 64}), encoding="utf-8")
    return p


def test_publish_then_resume_reproduces_queue_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake only the Hub client; publishing after merge/redaction and resuming on a fresh VM must reproduce
    the queue and report.
    """

    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    index_root, state = _seed_publishable_state(tmp_path)
    manifest = _write_manifest(tmp_path)

    hub = build_annotations_hub(curation_id="cur-import", snapshot_id="e" * 64)

    monkeypatch.setattr(adjudication_cli, "_make_client", lambda repo_id: hub)
    rc = _handle_corpus_command(["adjudicate",
            *_import_args(src, state_dir=state, index_root=index_root,
                archive_dir=tmp_path / "archive",  # distinct dir: publish stages the merged --archive-dir index
                extra=["--json", "--publish", "--manifest", str(manifest), "--hub-repo", "org/priv-ds"],
            ),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    revision = hub.repo_info("main").sha
    pointer_path = "annotations/cur-import/checkpoints/batch-latest.json"
    pointer = json.loads(hub.download_file(pointer_path, revision))
    batch_prefix = pointer["batch_prefix"]
    # Checkpoint the archive index as well as the queue so resume restores history.
    for name in ("queue.json", "observations.jsonl", "preview-ledger.json", "preview-manifest.json", "index.db"):
        assert batch_prefix + name in hub.files
    assert pointer_path in hub.files

    resumed_dir = tmp_path / "resumed"
    resumed = resume_annotation_state(
        hub, curation_id="cur-import", destination=resumed_dir, expected_snapshot_id="e" * 64,
    )
    assert set(resumed["restored"]) == {
        "queue.json", "observations.jsonl", "preview-ledger.json", "preview-manifest.json", "index.db",
    }
    for name in ("queue.json", "observations.jsonl", "preview-ledger.json", "index.db"):
        assert (resumed_dir / name).read_bytes() == (state / name).read_bytes()

    def _report(state_dir: Path) -> str:
        assert (_handle_corpus_command(
                ["adjudicate", "report", "--index-root", str(index_root), "--state-dir", str(state_dir)]
            )
            == 0
        )
        return capsys.readouterr().out

    original_report = _report(state)
    resumed_report = _report(resumed_dir)
    assert resumed_report == original_report

def test_publish_refuses_non_private_before_any_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:

    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    index_root, state = _seed_publishable_state(tmp_path)
    manifest = _write_manifest(tmp_path)

    hub = build_annotations_hub(curation_id="cur-import", snapshot_id="e" * 64, private=False)

    monkeypatch.setattr(adjudication_cli, "_make_client", lambda repo_id: hub)
    rc = _handle_corpus_command(["adjudicate",
            *_import_args(src, state_dir=state, index_root=index_root,
                archive_dir=tmp_path / "archive",  # distinct dir: publish stages the merged --archive-dir index
                extra=["--publish", "--manifest", str(manifest), "--hub-repo", "org/public-ds"],
            ),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "public Hub repository" in captured.out + captured.err
    prefix = f"annotations/cur-import/{'e' * 64}/"
    assert hub.uploaded_paths == []
    assert set(hub.files) == {prefix + "preview-manifest.json"}

    with pytest.raises(PublicDestinationError):
        publish_annotation_state(hub, state, manifest=manifest)

def test_publish_rejects_dry_run_and_missing_manifest() -> None:
    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(
            ["import-local-observations", "--archive-root", "/tmp", "--state-dir", "/tmp/s", "--publish", "--dry-run"]
        )
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(["import-local-observations", "--archive-root", "/tmp", "--state-dir", "/tmp/s", "--publish"])
    assert exc.value.code == 2

def test_cli_import_persists_redacted_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Persist prepared redacted metadata, never the credential-bearing original rubric."""

    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"],
        rubric_json=json.dumps({"workdir": "/Users/k/proj/build", "note": "ok"}),
    )

    state = tmp_path / "state"
    archive = tmp_path / "archive"
    rc = _handle_corpus_command(["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[])])
    assert rc == 0
    capsys.readouterr()
    with closing(sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT rubric_json FROM label_observations").fetchall()
    assert len(rows) == 1
    assert rows[0][0] is not None
    assert "/Users/k" not in rows[0][0]
    rubric = json.loads(rows[0][0])
    assert rubric["workdir"] == REDACTED_PATH
    assert rubric["note"] == "ok"


def test_local_import_retains_scanner_only_free_text_without_scan_scratch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from daydream.archive.index import label_observation_history

    src = tmp_path / "src"
    canary = "https://user:ghp_localimportcanary@github.com/o/r"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"], reviewer_logins=[canary])
    state = tmp_path / "state"
    archive = tmp_path / "archive"
    assert _handle_corpus_command([
        "adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[]),
    ]) == 0
    rows = label_observation_history(archive, "sess-1")
    assert json.loads(rows[0]["reviewer_logins"]) == [canary]
    assert not (state / "import-scan").exists()
    report = json.loads((state / "import-report.json").read_text())
    assert "redaction" not in report and "scan_summary" not in report
    assert canary not in "".join(capsys.readouterr())

def test_cli_import_reports_full_source_inventory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The success count includes deduped source rows as well as bucketed observations."""
    root_a = tmp_path / "backup-a"
    root_b = tmp_path / "backup-b"
    _seed_session(root_a, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    _seed_session(root_b, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    state = tmp_path / "state"
    rc = _handle_corpus_command(["adjudicate", *_import_args(root_a, root_b, state_dir=state, extra=[])])
    assert rc == 0
    captured = capsys.readouterr()
    report = json.loads((state / "import-report.json").read_text(encoding="utf-8"))
    total = sum(report["accounting"].values()) + report["deduped_count"]
    # Collapse Rich's word-wrapping so the message assertion is width-safe.
    assert f"{total} source row(s)" in " ".join(captured.out.split())

def test_cli_import_non_iso_stamp_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    # Corrupt SQL directly because the production writer always emits ISO timestamps.
    write = sqlite3.connect(src / "index.db")
    write.execute("UPDATE label_observations SET observed_at = ? WHERE session_id = 'sess-1'", ("2026-04-30 00:00:00",))
    write.commit()
    write.close()

    state = tmp_path / "state"
    rc = _handle_corpus_command(["adjudicate", *_import_args(src, state_dir=state, extra=[])])
    assert rc == 1
    captured = capsys.readouterr()
    assert "observed_at" in captured.out + captured.err
    assert (state / "index.db").exists() is False
    assert not (tmp_path / "archive" / "index.db").exists()

def test_cli_missing_archive_root_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(["import-local-observations", "--state-dir", "/tmp/x"])
    assert exc.value.code == 2

def test_cli_unknown_subverb_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(["import-local-observations-typo", "--archive-root", "/tmp"])
    assert exc.value.code == 2

def test_cli_inventory_failure_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    state = tmp_path / "state"
    rc = _handle_corpus_command(["adjudicate", *_import_args(broken, state_dir=state, extra=[])])
    assert rc == 1
    captured = capsys.readouterr()
    # Assert the token independently of Rich wrapping or path elision.
    assert "index.db" in captured.out + captured.err
    assert not state.exists()  # no placeholder success
    assert not (tmp_path / "archive" / "index.db").exists()  # no archive write either

def test_cli_import_links_against_hydrated_index_and_merges_into_archive_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Link against pinned hydrated and projected identities, then merge into archive-dir/index.db rather
    than state-dir.
    """

    src = tmp_path / "backup"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    index = tmp_path / "hydrated"
    conn = _get_connection(index)
    conn.execute("INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES ('sess-1', '2026-01-01T00:00:00+00:00', 'deep', 'archive/sess-1')"
    )
    conn.commit()
    conn.close()
    mat = _materialized_snapshot(tmp_path / "mat", session="sess-1", fingerprint="fp-1")

    state = tmp_path / "state"
    rc = _handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, index_root=mat, archive_dir=index, extra=[])]
    )
    assert rc == 0
    conn = _get_connection(index)
    rows = conn.execute("SELECT session_id, source FROM label_observations").fetchall()
    conn.close()
    assert [tuple(r) for r in rows] == [("sess-1", "auto")]
    assert not (state / "index.db").exists()

def test_cli_import_report_shows_mapping_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Report per-session identity validation. Use the harvester's head_sha evidence anchor so the per-
    finding identity match actually executes.
    """
    src = tmp_path / "backup"
    head = hashlib.sha256("sess-1".encode()).hexdigest()
    _seed_session(src, "sess-1", evidence_sha=head, labels=["accepted"])
    # identical derivative content on both sides -> links by session_id
    (src / "runs" / "sess-1").mkdir(parents=True)
    (src / "runs" / "sess-1" / "trajectory.json").write_text("{}", encoding="utf-8")
    mat = _materialized_snapshot(tmp_path / "mat", session="sess-1", fingerprint="fp-1")
    (mat / "runs" / "sess-1").mkdir(parents=True)
    (mat / "runs" / "sess-1" / "trajectory.json").write_text("{}", encoding="utf-8")

    state = tmp_path / "state"
    index = tmp_path / "hydrated"
    rc = _handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, index_root=mat, archive_dir=index, extra=["--json"])]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    summary = report["identity_summary"]["sess-1"]
    assert summary["matched_by"] == "session_id"
    # Match the row evidence head to the pinned run anchor exactly, routing the run-level row to its
    # per-finding bucket.
    assert summary["validation_outcome"] == "matched"

def test_cli_import_missing_identity_flags_exit_2() -> None:
    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(["import-local-observations", "--archive-root", "/tmp", "--state-dir", "/tmp/s"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(
            ["import-local-observations", "--archive-root", "/tmp", "--index-root", "/tmp/idx", "--state-dir", "/tmp/s"]
        )
    assert exc.value.code == 2

def test_cli_import_unreadable_index_root_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "backup"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    state = tmp_path / "state"
    missing = tmp_path / "no-such-index"
    rc = _handle_corpus_command(["adjudicate", *_import_args(src, state_dir=state, index_root=missing,
                                     archive_dir=tmp_path / "archive", extra=[])]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "no-such-index" in (captured.out + captured.err).replace("\n", " ") or \
        "index" in (captured.out + captured.err)
    assert not state.exists()
