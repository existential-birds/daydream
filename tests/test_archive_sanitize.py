"""Legacy bronze bundle sanitizer (issue #981 M14/M15/M19)."""
import json
from pathlib import Path

from daydream.archive import sanitize


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
