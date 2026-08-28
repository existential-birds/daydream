"""Legacy bronze bundle sanitizer (issue #981 M14/M15/M19)."""
import json
from pathlib import Path

import pytest

from daydream.archive import sanitize
from daydream.archive import scan as scan_module


def _seed_bronze_bundle(archive_dir: Path, session_id: str, remote_url: str) -> Path:
    run_dir = archive_dir / "runs" / session_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"session_id": session_id, "git": {"remote_url": remote_url, "repo_slug": "o/r"}})
    )
    (run_dir / "diff.patch").write_text("+TOKEN=ghp_canaryfake123\n")
    return run_dir


def test_derivative_is_credential_free_and_source_untouched(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    src = _seed_bronze_bundle(archive_dir, "s1", "https://user:ghp_canaryfake123@github.com/o/r")
    sanitize.sanitize_bundle(src, archive_dir)
    # source byte-identical (M14)
    assert "ghp_canaryfake123" in (src / "manifest.json").read_text()
    # derivative exists under sanitized/ and is clean
    out_dir = archive_dir / "sanitized" / "s1"
    assert (out_dir / "manifest.json").exists()
    assert "ghp_canaryfake123" not in (out_dir / "manifest.json").read_text()
    assert "ghp_canaryfake123" not in (out_dir / "diff.patch").read_text()


def test_digest_is_stable_and_audit_record_links(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    src = _seed_bronze_bundle(archive_dir, "s1", "https://github.com/o/r")
    d1 = sanitize.sanitize_bundle(src, archive_dir)
    d2 = sanitize.sanitize_bundle(src, archive_dir)
    assert d1.derivative_digest == d2.derivative_digest  # M15: reproducible
    ledger = json.loads((archive_dir / "sanitized" / "audit.jsonl").read_text().splitlines()[-1])
    assert ledger["source"] == str(src)
    assert ledger["derivative_digest"] == d1.derivative_digest


def test_resume_skips_completed_items(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    for sid in ("a", "b"):
        _seed_bronze_bundle(archive_dir, sid, "https://github.com/o/r")
    sanitize.sanitize_archive(archive_dir)  # first pass completes both
    # delete derivative b and re-run: only b is re-processed (a untouched mtime)
    before = (archive_dir / "sanitized" / "a" / "manifest.json").stat().st_mtime_ns
    # simulate partial state (b's derivative dir exists but manifest missing)
    (archive_dir / "sanitized" / "b" / "manifest.json").unlink()
    sanitize.sanitize_archive(archive_dir)
    after = (archive_dir / "sanitized" / "a" / "manifest.json").stat().st_mtime_ns
    assert before == after  # M19: completed items not re-processed
    assert (archive_dir / "sanitized" / "b" / "manifest.json").exists()


def test_derivative_stays_quarantined_until_scan_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_dir = tmp_path / "archive"
    src = _seed_bronze_bundle(archive_dir, "s1", "https://github.com/o/r")
    # force the release scan to fail closed during sanitization
    monkeypatch.setattr(
        scan_module, "scan_run_dir", lambda _: scan_module.ScanResult(clean=False)
    )
    result = sanitize.sanitize_bundle(src, archive_dir)
    assert result.released is False  # M16
    assert (archive_dir / "quarantine" / "s1").is_dir()
    assert not (archive_dir / "sanitized" / "s1" / "manifest.json").exists()


def test_corpus_projection_reads_only_sanitized_paths(tmp_path: Path) -> None:
    from daydream.training.corpus import _build_record

    # M17: a projection row pointed at a bronze run_dir containing a dirty URL
    # resolves its inputs from the sanitized derivative when one exists.
    archive_dir = tmp_path / "archive"
    _seed_bronze_bundle(archive_dir, "s1", "https://user:ghp_canaryfake123@github.com/o/r")
    sanitize.sanitize_bundle(archive_dir / "runs" / "s1", archive_dir)
    row = {"archive_path": str(archive_dir / "runs" / "s1"), "session_id": "s1"}
    projected = _build_record(row, {}, None)  # existing corpus entrypoint
    assert projected is not None
    assert "ghp_canaryfake123" not in json.dumps(projected)


def test_corpus_projection_refuses_affected_bundle_without_derivative(tmp_path: Path) -> None:
    from daydream.training.corpus import _build_record

    # M17 fail-closed: affected bundle with no released derivative is skipped,
    # never read raw.
    archive_dir = tmp_path / "archive"
    _seed_bronze_bundle(archive_dir, "s1", "https://user:ghp_canaryfake123@github.com/o/r")
    row = {"archive_path": str(archive_dir / "runs" / "s1"), "session_id": "s1"}
    assert _build_record(row, {}, None) is None


def test_inventory_counts_by_category_without_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive_dir = tmp_path / "archive"
    _seed_bronze_bundle(archive_dir, "s1", "https://user:p@github.com/o/r")
    _seed_bronze_bundle(archive_dir, "s2", "https://x-access-token@github.com/o/r")
    _seed_bronze_bundle(archive_dir, "s3", "https://github.com/o/r")
    sanitize.report_inventory(archive_dir)
    out = capsys.readouterr().out
    assert "userinfo" in out and "2" in out  # two affected bundles
    assert "ghp" not in out and "user:p" not in out  # no values, ever
