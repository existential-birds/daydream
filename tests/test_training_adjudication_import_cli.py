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
