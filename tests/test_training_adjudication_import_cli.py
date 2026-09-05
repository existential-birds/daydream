"""Real-path tests for ``corpus adjudicate import-local-observations`` (KD6).

Every test enters from the production CLI entrypoint
(``daydream.cli._handle_corpus_command`` / ``handle_adjudicate``) with real
temp archive roots (real SQLite ``index.db`` files written through the
production archive writer). No backend or Hub mocking — Task 10 owns the
publish wiring.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from daydream import cli
from daydream.archive.index import append_label_observation, upsert_run
from tests.harness.trajectory import make_manifest

_OBSERVED = "2026-04-30T00:00:00+00:00"
_VALID_AT = "2026-04-29T00:00:00+00:00"


def _seed_session(root: Path, session_id: str, *, evidence_sha: str, labels: list[str]) -> None:
    """One archived run + one auto label observation, via the real writer.

    The run gets per-session base/head SHAs so the identity fallback lookup
    (repo_slug, base_sha, head_sha) -> session_id never collides.
    """
    head = hashlib.sha256(session_id.encode()).hexdigest()
    base = hashlib.sha256(("base-" + session_id).encode()).hexdigest()
    upsert_run(
        root,
        make_manifest(
            session_id=session_id,
            repo_slug="org/repo",
            head_sha=head,
            base_sha=base,
        ),
    )
    append_label_observation(
        root,
        session_id,
        labels=labels,
        pr_state=None,
        labeler_version="980-rubric-r2",
        evidence_sha=evidence_sha,
        valid_at=_VALID_AT,
        reply_evidence_digest=None,
        reward_version=None,
        has_posterior=False,
        source="auto",
        observed_at=_OBSERVED,
    )


def _source_row_count(roots: list[Path]) -> int:
    total = 0
    for root in roots:
        conn = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
        try:
            total += int(conn.execute("SELECT COUNT(*) FROM label_observations").fetchone()[0])
        finally:
            conn.close()
    return total


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_args(
    *roots: Path,
    state_dir: Path,
    extra: list[str],
    index_root: Path | None = None,
    archive_dir: Path | None = None,
) -> list[str]:
    argv: list[str] = ["import-local-observations"]
    for root in roots:
        argv += ["--archive-root", str(root)]
    if index_root is None:
        index_root = state_dir.parent / "idx"
        index_root.mkdir(exist_ok=True)
        (index_root / "sessions.jsonl").write_text("", encoding="utf-8")
    if archive_dir is None:
        archive_dir = state_dir.parent / "archive"
    argv += [
        "--index-root", str(index_root),
        "--archive-dir", str(archive_dir),
        "--state-dir", str(state_dir),
        *extra,
    ]
    return argv


def _materialized_snapshot(root: Path, session: str, fingerprint: str) -> Path:
    """A materialized snapshot root (``sessions.jsonl`` in the hydrated-index
    session shape) with one projected finding for *session* — the projector
    shape the import links per-finding evidence against."""
    root.mkdir(parents=True, exist_ok=True)
    sessions = [{
        "session_id": session, "trajectory_id": session, "segment_id": session,
        "resolutions": [{
            "fingerprint": fingerprint, "disposition": "unanswered",
            "evidence": [{"reply_id": 1, "body_sha256": "abc",
                          "created_at": "2026-01-01T00:00:00+00:00"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
            "comment_id": 7,
        }],
    }]
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
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
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(*roots, state_dir=state, extra=["--dry-run", "--json"])]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    # Full reason-coded accounting: every source row lands in exactly one
    # import bucket (M7).
    assert sum(report["accounting"].values()) == total
    assert report["sources"] == [
        {"archive_root": str(root_a), "row_count": 2, "source_digest": _digest(root_a / "index.db")},
        {"archive_root": str(root_b), "row_count": 1, "source_digest": _digest(root_b / "index.db")},
    ]
    # Dry-run writes no state at all (S2).
    assert not state.exists()


def test_cli_real_path_real_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    _seed_session(src, "sess-2", evidence_sha="f" * 64, labels=["rejected"])
    before = (src / "index.db").read_bytes()

    state = tmp_path / "state"
    archive = tmp_path / "archive"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[])]
    )
    assert rc == 0
    capsys.readouterr()  # drain the human-readable run before the --json re-run
    # Read-only sources: byte-identical after a full (non-dry-run) import (M1).
    assert (src / "index.db").read_bytes() == before

    # Digest-stable report + hydrate-shaped ledger written into --state-dir.
    report = json.loads((state / "import-report.json").read_text(encoding="utf-8"))
    assert report["dry_run"] is False
    assert sum(report["accounting"].values()) == 2
    assert report["identity_summary"]["sess-1"]["matched_by"] == "repo_slug_sha"
    assert report["identity_summary"]["sess-2"]["matched_by"] == "repo_slug_sha"
    report = json.loads((state / "import-report.json").read_text(encoding="utf-8"))
    assert report["dry_run"] is False
    assert sum(report["accounting"].values()) == 2
    ledger = json.loads((state / "import-ledger.json").read_text(encoding="utf-8"))
    assert ledger["accounting"] == report["accounting"]
    assert {entry["session_id"] for entry in ledger["observations"]} == {"sess-1", "sess-2"}

    # The merge appended the imported observations into the hydrated
    # --archive-dir archive; the state-dir index.db is never written.
    conn = sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT session_id, evidence_sha FROM label_observations ORDER BY session_id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("sess-1", "e" * 64), ("sess-2", "f" * 64)]
    assert not (state / "index.db").exists()

    # Idempotent re-import (M4): identical sources, nothing new appended,
    # byte-identical report.
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=["--json"])]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["merge"]["appended"] == 0
    # Digest-stable (S1): once the state archive has absorbed the import, an
    # identical re-import produces a byte-identical report.
    report_bytes = (state / "import-report.json").read_bytes()
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[])]
    )
    assert rc == 0
    assert (state / "import-report.json").read_bytes() == report_bytes
    assert (src / "index.db").read_bytes() == before


def test_cli_reimport_does_not_displace_newer_target_runs_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An overlapping re-import of an older backup must not displace newer
    target runs state (status/archived_at/profile_*/cost metrics, plus the
    writer-owned cache mirrors): the run-row seeding is no-displacement like
    the observation merge, so append-only holds for both tables (M3)."""
    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    state = tmp_path / "state"
    archive = tmp_path / "archive"
    assert cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[])]
    ) == 0

    # Newer target state: a later archive refresh rewrites the same session
    # with a newer timestamp, evolved profile, and updated cost metrics.
    head = hashlib.sha256("sess-1".encode()).hexdigest()
    base = hashlib.sha256(("base-" + "sess-1").encode()).hexdigest()
    upsert_run(
        archive,
        make_manifest(
            session_id="sess-1",
            repo_slug="org/repo",
            head_sha=head,
            base_sha=base,
            archived_at="2026-05-01T00:00:00+00:00",
            status="partial",
            profile_name="profile-v2",
            total_cost_usd=99.5,
        ),
    )
    conn = sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        newer = dict(
            conn.execute(
                "SELECT archived_at, status, profile_name, total_cost_usd, "
                "outcome_labels, labeled_at FROM runs WHERE session_id = 'sess-1'"
            ).fetchone()
        )
    finally:
        conn.close()

    # Re-import the (older) source backup overlapping the same session: every
    # populated target column must survive untouched (only NULL columns may be
    # filled from the source snapshot).
    assert cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=["--json"])]
    ) == 0
    capsys.readouterr()
    conn = sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        after = dict(
            conn.execute(
                "SELECT archived_at, status, profile_name, total_cost_usd, "
                "outcome_labels, labeled_at FROM runs WHERE session_id = 'sess-1'"
            ).fetchone()
        )
    finally:
        conn.close()
    populated = {key: value for key, value in newer.items() if value is not None}
    assert {key: after[key] for key in populated} == populated


def test_cli_overlapping_backups_dedupe_accounting(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Two backups of the same archive: the shared rows dedupe, the accounting
    # still covers every source row across both roots (M4 + M7).
    root_a = tmp_path / "backup-a"
    root_b = tmp_path / "backup-b"
    _seed_session(root_a, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    _seed_session(root_a, "sess-2", evidence_sha="f" * 64, labels=["rejected"])
    for session_id, sha in (("sess-1", "e" * 64), ("sess-2", "f" * 64)):
        _seed_session(root_b, session_id, evidence_sha=sha, labels=["accepted" if sha[0] == "e" else "rejected"])
    state = tmp_path / "state"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(root_a, root_b, state_dir=state, extra=["--dry-run", "--json"])]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["deduped_count"] == 2
    # Every source row is either accounted in a bucket or dropped as a
    # byte-identical duplicate (M4 + M7).
    assert (
        sum(report["accounting"].values()) + report["deduped_count"]
        == _source_row_count([root_a, root_b])
    )


def _seed_publishable_state(tmp_path: Path) -> tuple[Path, Path]:
    """Produce the publishable adjudication state files via the real pipeline.

    ``queue.json`` comes from the real ``build`` verb over an empty hydrated
    index (empty ``sessions.jsonl`` -> empty queue). ``observations.jsonl``
    and ``preview-ledger.json`` are the empty-state defaults that publish's
    fixed payload set requires — their production-by-verb behavior is the
    publish/label/export verbs' own tested contract, not this test's subject.
    """
    index_root = tmp_path / "index-root"
    index_root.mkdir()
    (index_root / "sessions.jsonl").write_text("", encoding="utf-8")
    state = tmp_path / "state"
    assert cli._handle_corpus_command(
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
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--publish composes publish_annotation_state after the merge + redaction
    gate (M8), and a fresh-VM resume from the Hub checkpoint reproduces the
    identical queue + report (AC5). Only the Hub client is faked."""
    from daydream.training.adjudication.publish import resume_annotation_state
    from tests.fixtures.training.build_hub_snapshot import build_annotations_hub

    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    index_root, state = _seed_publishable_state(tmp_path)
    manifest = _write_manifest(tmp_path)

    hub = build_annotations_hub(curation_id="cur-import", snapshot_id="e" * 64)
    from daydream.training.adjudication import cli as adjudication_cli

    monkeypatch.setattr(adjudication_cli, "_make_client", lambda repo_id: hub)
    rc = cli._handle_corpus_command(
        [
            "adjudicate",
            *_import_args(
                src,
                state_dir=state,
                index_root=index_root,
                archive_dir=state,  # publish stages the merged index from --archive-dir
                extra=[
                    "--json",
                    "--publish",
                    "--manifest",
                    str(manifest),
                    "--hub-repo",
                    "org/priv-ds",
                ],
            ),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    prefix = f"annotations/cur-import/{'e' * 64}/"
    # The checkpoint is always written on --publish: it is the fresh-VM
    # resume anchor (AC5). The archive index (where the import's rows live)
    # is byte-published with the adjudication payload, so a fresh-VM resume
    # restores the import itself, not just the queue/report.
    for name in (
        "queue.json",
        "observations.jsonl",
        "preview-ledger.json",
        "preview-manifest.json",
        "index.db",
        "checkpoints/batch-latest.json",
    ):
        assert prefix + name in hub.files

    # Fresh-VM resume: empty stage dir, restore from the Hub checkpoint.
    resumed_dir = tmp_path / "resumed"
    resumed = resume_annotation_state(hub, manifest=manifest, stage_dir=resumed_dir)
    assert set(resumed["restored"]) == {
        "queue.json",
        "observations.jsonl",
        "preview-ledger.json",
        "preview-manifest.json",
        "index.db",
    }
    for name in ("queue.json", "observations.jsonl", "preview-ledger.json", "index.db"):
        assert (resumed_dir / name).read_bytes() == (state / name).read_bytes()

    # Identical queue + report (AC5): the report verb over the resumed state
    # reproduces the pre-publish report byte-for-byte.
    def _report(state_dir: Path) -> str:
        assert (
            cli._handle_corpus_command(
                ["adjudicate", "report", "--index-root", str(index_root), "--state-dir", str(state_dir)]
            )
            == 0
        )
        return capsys.readouterr().out

    original_report = _report(state)
    resumed_report = _report(resumed_dir)
    assert resumed_report == original_report


def test_publish_refuses_non_private_before_any_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-private destination is refused before any byte is written (M17):
    exit 1, the refusal named on stderr/stdout, and zero uploads (the hub
    keeps only its seeded manifest)."""
    from daydream.archive.hydrate import PublicDestinationError
    from daydream.training.adjudication.publish import publish_annotation_state
    from tests.fixtures.training.build_hub_snapshot import build_annotations_hub

    src = tmp_path / "src"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    index_root, state = _seed_publishable_state(tmp_path)
    manifest = _write_manifest(tmp_path)

    hub = build_annotations_hub(curation_id="cur-import", snapshot_id="e" * 64, private=False)
    from daydream.training.adjudication import cli as adjudication_cli

    monkeypatch.setattr(adjudication_cli, "_make_client", lambda repo_id: hub)
    rc = cli._handle_corpus_command(
        [
            "adjudicate",
            *_import_args(
                src,
                state_dir=state,
                index_root=index_root,
                archive_dir=state,  # publish stages the merged index from --archive-dir
                extra=["--publish", "--manifest", str(manifest), "--hub-repo", "org/public-ds"],
            ),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "not private" in captured.out + captured.err
    prefix = f"annotations/cur-import/{'e' * 64}/"
    assert hub.uploaded_paths == []
    assert set(hub.files) == {prefix + "preview-manifest.json"}

    # The composition seam itself hard-fails with the typed error, not a
    # swallowed warning.
    with pytest.raises(PublicDestinationError):
        publish_annotation_state(hub, state, manifest=manifest)


def test_publish_rejects_dry_run_and_missing_manifest() -> None:
    from daydream.training.adjudication.cli import handle_adjudicate

    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(
            ["import-local-observations", "--archive-root", "/tmp", "--state-dir", "/tmp/s",
             "--publish", "--dry-run"]
        )
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(
            ["import-local-observations", "--archive-root", "/tmp", "--state-dir", "/tmp/s", "--publish"]
        )
    assert exc.value.code == 2


def test_cli_import_persists_redacted_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The merge commits the redaction scan's payload, never the unredacted
    originals: a credential-bearing rubric_json (absolute local path) reaches
    the state archive only in its redacted form (M9)."""
    from daydream.archive.importer import REDACTED_PATH

    src = tmp_path / "src"
    head = hashlib.sha256("sess-1".encode()).hexdigest()
    base = hashlib.sha256(("base-" + "sess-1").encode()).hexdigest()
    upsert_run(
        src,
        make_manifest(
            session_id="sess-1",
            repo_slug="org/repo",
            head_sha=head,
            base_sha=base,
        ),
    )
    append_label_observation(
        src,
        "sess-1",
        labels=["accepted"],
        pr_state=None,
        labeler_version="980-rubric-r2",
        evidence_sha="e" * 64,
        rubric_json=json.dumps({"workdir": "/Users/k/proj/build", "note": "ok"}),
        valid_at=_VALID_AT,
        source="auto",
        observed_at=_OBSERVED,
    )

    state = tmp_path / "state"
    archive = tmp_path / "archive"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, archive_dir=archive, extra=[])]
    )
    assert rc == 0
    capsys.readouterr()
    conn = sqlite3.connect(f"file:{archive / 'index.db'}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT rubric_json FROM label_observations").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] is not None
    assert "/Users/k" not in rows[0][0]
    rubric = json.loads(rows[0][0])
    assert rubric["workdir"] == REDACTED_PATH
    assert rubric["note"] == "ok"


def test_cli_import_reports_full_source_inventory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The success message reports the full source row inventory (bucketed +
    deduped), not just the bucket sum (M7)."""
    root_a = tmp_path / "backup-a"
    root_b = tmp_path / "backup-b"
    _seed_session(root_a, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    _seed_session(root_b, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    state = tmp_path / "state"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(root_a, root_b, state_dir=state, extra=[])]
    )
    assert rc == 0
    captured = capsys.readouterr()
    report = json.loads((state / "import-report.json").read_text(encoding="utf-8"))
    total = sum(report["accounting"].values()) + report["deduped_count"]
    # Collapse Rich's word-wrapping so the message assertion is width-safe.
    assert f"{total} source row(s)" in " ".join(captured.out.split())


def test_cli_import_non_iso_stamp_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A hand-edited non-ISO observed_at aborts at the pre-write gate: exit 1
    and no state archive at all (no seeded runs, no partial appends)."""
    src = tmp_path / "src"
    head = hashlib.sha256("sess-1".encode()).hexdigest()
    base = hashlib.sha256(("base-" + "sess-1").encode()).hexdigest()
    upsert_run(
        src,
        make_manifest(
            session_id="sess-1",
            repo_slug="org/repo",
            head_sha=head,
            base_sha=base,
        ),
    )
    append_label_observation(
        src,
        "sess-1",
        labels=["accepted"],
        pr_state=None,
        labeler_version="980-rubric-r2",
        evidence_sha="e" * 64,
        valid_at=_VALID_AT,
        source="auto",
        observed_at=_OBSERVED,
    )
    # Corrupt the stamp in place (the trigger requires a hand-edited/corrupt
    # source db; writer-produced values are always ISO-8601).
    write = sqlite3.connect(src / "index.db")
    write.execute(
        "UPDATE label_observations SET observed_at = ? WHERE session_id = 'sess-1'",
        ("2026-04-30 00:00:00",),
    )
    write.commit()
    write.close()

    state = tmp_path / "state"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, extra=[])]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "observed_at" in captured.out + captured.err
    # Fail-closed before any state write: no archive, no seeded runs, no
    # partial appends, no placeholder success report.
    assert (state / "index.db").exists() is False
    assert not (tmp_path / "archive" / "index.db").exists()


def test_cli_missing_archive_root_exits_2() -> None:
    from daydream.training.adjudication.cli import handle_adjudicate

    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(["import-local-observations", "--state-dir", "/tmp/x"])
    assert exc.value.code == 2


def test_cli_unknown_subverb_exits_2() -> None:
    from daydream.training.adjudication.cli import handle_adjudicate

    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(["import-local-observations-typo", "--archive-root", "/tmp"])
    assert exc.value.code == 2


def test_cli_inventory_failure_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    state = tmp_path / "state"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(broken, state_dir=state, extra=[])]
    )
    assert rc == 1
    captured = capsys.readouterr()
    # The rich panel may elide the long tmp path, but the fail-closed reason
    # (naming the missing index.db) is always present; no placeholder success.
    # Assert the single token so Rich's fold (word-boundary wrapping) cannot
    # split the reason on an 80-col non-TTY console.
    assert "index.db" in captured.out + captured.err
    assert not state.exists()  # no placeholder success
    assert not (tmp_path / "archive" / "index.db").exists()  # no archive write either


def test_cli_import_links_against_hydrated_index_and_merges_into_archive_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty-literal identity maps are gone: the importer links sessions
    against the pinned hydrated index, validates exact finding identity
    against the projected findings, and appends into the hydrated
    --archive-dir index.db — never the state-dir index."""
    from daydream.archive.index import _get_connection

    src = tmp_path / "backup"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    index = tmp_path / "hydrated"
    conn = _get_connection(index)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES ('sess-1', '2026-01-01T00:00:00+00:00', 'deep', 'archive/sess-1')"
    )
    conn.commit()
    conn.close()
    # a materialized snapshot with one finding for sess-1 (project_findings shape)
    mat = _materialized_snapshot(tmp_path / "mat", session="sess-1", fingerprint="fp-1")

    state = tmp_path / "state"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, index_root=mat, archive_dir=index,
                                     extra=[])]
    )
    assert rc == 0
    # row landed in the hydrated archive, keyed to the Hub session id
    conn = _get_connection(index)
    rows = conn.execute(
        "SELECT session_id, source FROM label_observations"
    ).fetchall()
    conn.close()
    # Provenance survives the merge verbatim (source stays the source row's
    # own 'auto', never rewritten).
    assert [tuple(r) for r in rows] == [("sess-1", "auto")]
    # state-dir index.db was never created
    assert not (state / "index.db").exists()


def test_cli_import_report_shows_mapping_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The import report carries a per-session mapping summary (matched_by,
    validation outcome) so operators can audit identity resolution."""
    src = tmp_path / "backup"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    # identical derivative content on both sides -> links by session_id
    (src / "runs" / "sess-1").mkdir(parents=True)
    (src / "runs" / "sess-1" / "trajectory.json").write_text("{}", encoding="utf-8")
    mat = _materialized_snapshot(tmp_path / "mat", session="sess-1", fingerprint="fp-1")
    (mat / "runs" / "sess-1").mkdir(parents=True)
    (mat / "runs" / "sess-1" / "trajectory.json").write_text("{}", encoding="utf-8")

    state = tmp_path / "state"
    index = tmp_path / "hydrated"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, index_root=mat, archive_dir=index,
                                     extra=["--json"])]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    summary = report["identity_summary"]["sess-1"]
    assert summary["matched_by"] == "session_id"
    # The run-level evidence digest does not match the projected finding's
    # per-finding digest, so the session routes to the ambiguous bucket.
    assert summary["validation_outcome"] == "ambiguous"


def test_cli_import_missing_identity_flags_exit_2() -> None:
    from daydream.training.adjudication.cli import handle_adjudicate

    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(
            ["import-local-observations", "--archive-root", "/tmp", "--state-dir", "/tmp/s"]
        )
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        handle_adjudicate(
            ["import-local-observations", "--archive-root", "/tmp",
             "--index-root", "/tmp/idx", "--state-dir", "/tmp/s"]
        )
    assert exc.value.code == 2


def test_cli_import_unreadable_index_root_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing --index-root index fails closed: exit 1 via the derive
    failure path, no empty-literal fallback anywhere."""
    src = tmp_path / "backup"
    _seed_session(src, "sess-1", evidence_sha="e" * 64, labels=["accepted"])
    state = tmp_path / "state"
    missing = tmp_path / "no-such-index"
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, index_root=missing,
                                     archive_dir=tmp_path / "archive", extra=[])]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "no-such-index" in (captured.out + captured.err).replace("\n", " ") or \
        "index" in (captured.out + captured.err)
    assert not state.exists()
