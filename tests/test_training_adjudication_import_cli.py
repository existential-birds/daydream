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


def _import_args(*roots: Path, state_dir: Path, extra: list[str]) -> list[str]:
    argv: list[str] = ["import-local-observations"]
    for root in roots:
        argv += ["--archive-root", str(root)]
    argv += ["--state-dir", str(state_dir), *extra]
    return argv


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
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, extra=[])]
    )
    assert rc == 0
    capsys.readouterr()  # drain the human-readable run before the --json re-run
    # Read-only sources: byte-identical after a full (non-dry-run) import (M1).
    assert (src / "index.db").read_bytes() == before

    # Digest-stable report + hydrate-shaped ledger written into --state-dir.
    report = json.loads((state / "import-report.json").read_text(encoding="utf-8"))
    assert report["dry_run"] is False
    assert sum(report["accounting"].values()) == 2
    ledger = json.loads((state / "import-ledger.json").read_text(encoding="utf-8"))
    assert ledger["accounting"] == report["accounting"]
    assert {entry["session_id"] for entry in ledger["observations"]} == {"sess-1", "sess-2"}

    # The merge appended the imported observations into the state archive.
    conn = sqlite3.connect(f"file:{state / 'index.db'}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT session_id, evidence_sha FROM label_observations ORDER BY session_id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("sess-1", "e" * 64), ("sess-2", "f" * 64)]

    # Idempotent re-import (M4): identical sources, nothing new appended,
    # byte-identical report.
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, extra=["--json"])]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["merge"]["appended"] == 0
    # Digest-stable (S1): once the state archive has absorbed the import, an
    # identical re-import produces a byte-identical report.
    report_bytes = (state / "import-report.json").read_bytes()
    rc = cli._handle_corpus_command(
        ["adjudicate", *_import_args(src, state_dir=state, extra=[])]
    )
    assert rc == 0
    assert (state / "import-report.json").read_bytes() == report_bytes
    assert (src / "index.db").read_bytes() == before


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
    # resume anchor (AC5).
    for name in (
        "queue.json",
        "observations.jsonl",
        "preview-ledger.json",
        "preview-manifest.json",
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
    }
    for name in ("queue.json", "observations.jsonl", "preview-ledger.json"):
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
    _seed_publishable_state(tmp_path)
    state = tmp_path / "state"
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
    assert "no index.db" in captured.out + captured.err
    assert not state.exists()  # no placeholder success
