"""Legacy bronze bundle sanitizer (issue #981 M14/M15/M19)."""
import json
import re
from pathlib import Path

import pytest

from daydream.archive import sanitize, scan as scan_module
from daydream.training.corpus_projection.bundle import BundleError, load_curated_bundle


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

def test_text_file_token_only_userinfo_and_query_credentials_are_sanitized(tmp_path: Path,) -> None:
    archive_dir = tmp_path / "archive"
    run_dir = archive_dir / "runs" / "s1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"session_id": "s1", "git": {"remote_url": "https://github.com/o/r"}})
    )
    (run_dir / "review-output.md").write_text(
        "clone https://x-access-token@github.com/o/r.git\n"
        "fetch https://example.com/repo?token=abc123\n"
        "auth https://example.com/api?client=1&access_token=abcdef\n"
    )
    result = sanitize.sanitize_bundle(run_dir, archive_dir)
    assert result.released  # extended rules applied; release scan clean
    text = (archive_dir / "sanitized" / "s1" / "review-output.md").read_text()
    assert "x-access-token" not in text
    assert "token=abc123" not in text
    assert "access_token=abcdef" not in text

def test_unreadable_bundle_preserves_source_and_audits_failure(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    source = _seed_bronze_bundle(archive_dir, "bad", "https://github.com/o/r")
    payload = b"\x00\xff\xfe"
    (source / "binary.bin").write_bytes(payload)

    with pytest.raises(UnicodeDecodeError):
        sanitize.sanitize_bundle(source, archive_dir)

    assert (source / "binary.bin").read_bytes() == payload
    assert not (archive_dir / "sanitized" / "bad").exists()
    audit = json.loads((archive_dir / "sanitized" / "audit.jsonl").read_text())
    assert audit["session_id"] == "bad" and audit["status"] == "quarantined"
    assert audit["source"] == str(source)

def test_derivative_stays_quarantined_until_scan_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_dir = tmp_path / "archive"
    src = _seed_bronze_bundle(archive_dir, "s1", "https://github.com/o/r")
    # force the release scan to fail closed during sanitization
    monkeypatch.setattr(scan_module, "scan_run_dir", lambda _: scan_module.ScanResult(clean=False))
    result = sanitize.sanitize_bundle(src, archive_dir)
    assert result.released is False  # M16
    assert (archive_dir / "quarantine" / "s1").is_dir()
    assert not (archive_dir / "sanitized" / "s1" / "manifest.json").exists()

def test_advisory_only_derivative_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Release an advisory-only derivative using the real scanner and severity reducer.

    Swap only its rule: normal sanitization scrubs every built-in advisory shape,
    so content alone cannot reach this release-gate outcome.
    """
    advisory_rule = (
        re.compile(r"\bFEATURE_FLAG_OVERRIDE_KEY\b"),
        "env_var", scan_module.SEVERITY_ADVISORY,
    )
    monkeypatch.setattr(scan_module, "_RULES", (advisory_rule,))
    archive_dir = tmp_path / "archive"
    run_dir = archive_dir / "runs" / "s1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"session_id": "s1", "git": {"remote_url": "https://github.com/o/r"}})
    )
    (run_dir / "diff.patch").write_text('+FEATURE_FLAG_OVERRIDE_KEY = "override_flag"\n')

    result = sanitize.sanitize_bundle(run_dir, archive_dir)

    assert result.released is True
    assert result.status == "sanitized"
    assert not (archive_dir / "quarantine" / "s1").exists()
    assert (archive_dir / "sanitized" / "s1" / "manifest.json").is_file()
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "advisory" in out
    assert "diff.patch" in out
    assert "override_flag" not in out  # M11: never a matched value

def test_json_leaf_userinfo_shapes_are_sanitized_not_quarantined(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    run_dir = archive_dir / "runs" / "s1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"session_id": "s1", "git": {"remote_url": "https://github.com/o/r"}})
    )
    dsn = 'return f"postgresql://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:{cfg.DB_PORT}/x"'
    (run_dir / "trajectory.json").write_text(json.dumps({"steps": [
                    {"observation": dsn}, {"observation": "fetch https://example.com/repo?token=abc123"},
                    {"observation": "clone https://x-access-token@github.com/o/r.git"},
                ]
            }
        )
    )

    result = sanitize.sanitize_bundle(run_dir, archive_dir)

    assert result.released is True
    assert not (archive_dir / "quarantine" / "s1").exists()
    derivative = archive_dir / "sanitized" / "s1"
    assert scan_module.scan_run_dir(derivative).clean is True  # re-scans clean
    text = (derivative / "trajectory.json").read_text()
    assert "x-access-token" not in text  # the blocking shape is gone, not tolerated
    assert "token=abc123" not in text
    assert "{cfg.DB_PASSWORD}" not in text
    # M14: the source bundle is never modified.
    assert "x-access-token" in (run_dir / "trajectory.json").read_text()

def test_corpus_projection_admits_only_clean_batches(tmp_path: Path) -> None:
    """M17 successor: the projection layer's admission boundary refuses a
    bundle whose batch rows are not all ``admitted`` — a quarantined
    (secret-carrying) batch can never reach a projected record."""

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "_SUCCESS").write_text("ok\n")
    (bundle_dir / "curation-manifest.json").write_text(json.dumps({
            "schema_version": "1", "source_hub_commit": "0123456789abcdef0123456789abcdef01234567",
            "curation_id": "cur-0123456789abcdef", "sanitizer_version": "1", "hydration_index_schema_version": "1",
            "admission_policy_version": "1", "publication_prefix": "curated/cur-0123456789abcdef/",
            "batches": [{"session_id": "s1", "content_digest": "1" * 64, "status": "quarantined",
                    "reason_code": "secrets_scan_dirty", "artifact_relpath": "batches/s1", "artifact_digest": None,
                    "manifest_relpath": None,
                },
            ],
        })
    )
    with pytest.raises(BundleError):
        load_curated_bundle(bundle_dir)
