"""Diagnostic dumps preserve evidence, with explicit sanitization on request."""

import json
import re
import shutil
from pathlib import Path

import pytest

from daydream.archive import dump, scan


def _bundle(tmp_path: Path, patch: str) -> tuple[Path, Path]:
    source = tmp_path / "assembly"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"session_id": "session-1", "git": {"head_sha": "abc123"}}),)
    (source / "diff.patch").write_text(patch)
    stage = tmp_path / "late"
    stage.mkdir()
    return source, stage

def _contents(path: Path) -> dict[str, bytes]:
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_default_dump_preserves_credentials_and_binary_without_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, stage = _bundle(tmp_path, "+TOKEN=ghp_canaryfake123\n")
    (source / "evidence.bin").write_bytes(b"\xff\x00ghp_binarycanary")
    (stage / "unrelated.txt").write_bytes(b"prior destination")
    original = _contents(source)

    def unexpected_scan(_path: Path) -> scan.ScanResult:
        pytest.fail("raw dumps must not invoke the scanner")

    def unexpected_sanitize(_path: Path) -> None:
        pytest.fail("raw dumps must not invoke the sanitizer")

    monkeypatch.setattr(scan, "scan_run_dir", unexpected_scan)
    monkeypatch.setattr(dump, "sanitize_bundle_files", unexpected_sanitize)
    assert dump.publish_dump(source, stage, "session-1")
    assert _contents(stage) == {**original, "unrelated.txt": b"prior destination"}
    assert _contents(source) == original

@pytest.mark.parametrize("sanitize", [False, True])
@pytest.mark.parametrize("patch", ["+print('safe')\n", '+SORT_KEY = "created_at"\n'])
def test_accepted_dump_retains_exact_bundle_bytes(tmp_path: Path, patch: str, sanitize: bool) -> None:
    source, stage = _bundle(tmp_path, patch)
    original = _contents(source)
    assert dump.publish_dump(source, stage, "session-1", sanitize=sanitize)
    assert _contents(stage) == original
    assert _contents(source) == original

def test_blocking_dump_sanitizes_separate_copy_and_scans_sidecar(tmp_path: Path) -> None:
    source, stage = _bundle(tmp_path,
        '-dsn = "https://oldfixture@sentry.example/1"\n'
        '+dsn = "https://newfixture@sentry.example/1"\n'
        '+TOKEN=ghp_canaryfake123\n'
        '+-----BEGIN PRIVATE KEY-----\n+CANARYKEYMATERIAL\n+-----END PRIVATE KEY-----\n',
    )
    original = _contents(source)

    assert dump.publish_dump(source, stage, "session-1", sanitize=True)
    assert _contents(source) == original
    assert scan.scan_run_dir(stage).clean
    assert json.loads((stage / "manifest.json").read_text()) == json.loads(original["manifest.json"])
    assert json.loads((stage / "dump-sanitization.json").read_text()) == {
        "schema_version": 1, "session_id": "session-1", "sanitized": True,
    }
    exported = (stage / "diff.patch").read_text()
    for secret in ("oldfixture", "newfixture", "ghp_canaryfake123", "CANARYKEYMATERIAL"):
        assert secret not in exported
    assert list(tmp_path.glob(".dump-*")) == []

def test_residual_refusal_omits_stage_and_never_echoes_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source, stage = _bundle(tmp_path, "+opaque-canary-value\n")
    original = _contents(source)
    monkeypatch.setattr(
        scan, "_RULES", (*scan._RULES, (re.compile("opaque-canary-value"), "opaque", scan.SEVERITY_BLOCKING)),
    )
    assert not dump.publish_dump(source, stage, "session-1", sanitize=True)
    assert not stage.exists()
    assert _contents(source) == original
    assert list(tmp_path.glob(".dump-*")) == []
    output = capsys.readouterr().out
    assert "opaque-canary-value" not in output
    assert "Withholding" in output
    assert "diff.patch" in output

def test_sanitizer_failure_withholds_dump_without_exception_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source, stage = _bundle(tmp_path, "+ghp_canaryfake123\n")
    original = _contents(source)
    def fail_sanitization(_path: Path) -> None:
        raise ValueError("ghp_canaryfake123")
    monkeypatch.setattr(dump, "sanitize_bundle_files", fail_sanitization)
    assert not dump.publish_dump(source, stage, "session-1", sanitize=True)
    assert not stage.exists()
    assert _contents(source) == original
    assert list(tmp_path.glob(".dump-*")) == []
    assert "ghp_canaryfake123" not in capsys.readouterr().out

def test_scanner_error_withholds_dump(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source, stage = _bundle(tmp_path, "+print('safe')\n")
    (source / "unreadable.bin").write_bytes(b"\xff")
    assert not dump.publish_dump(source, stage, "session-1", sanitize=True)
    assert not stage.exists()
    assert (source / "unreadable.bin").read_bytes() == b"\xff"
    assert "scan_error" in capsys.readouterr().out
    assert list(tmp_path.glob(".dump-*")) == []

def test_raised_scanner_error_withholds_dump_without_exception_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source, stage = _bundle(tmp_path, "+print('safe')\n")
    def fail_scan(_path: Path) -> scan.ScanResult:
        raise ValueError("private-scanner-detail")
    monkeypatch.setattr(scan, "scan_run_dir", fail_scan)
    assert not dump.publish_dump(source, stage, "session-1", sanitize=True)
    assert not stage.exists()
    assert "private-scanner-detail" not in capsys.readouterr().out

@pytest.mark.parametrize("sanitize", [False, True])
@pytest.mark.parametrize("patch", ["+print('safe')\n", "+ghp_canaryfake123\n"])
def test_destination_publication_error_stays_fatal(tmp_path: Path, patch: str, sanitize: bool) -> None:
    source, stage = _bundle(tmp_path, patch)
    # A real destination filesystem conflict, after the release scan succeeds.
    (stage / "manifest.json" / "manifest.json").mkdir(parents=True)
    (stage / "manifest.json" / "keep.txt").write_text("prior evidence")
    with pytest.raises(shutil.Error):
        dump.publish_dump(source, stage, "session-1", sanitize=sanitize)
    assert (stage / "manifest.json" / "keep.txt").read_text() == "prior evidence"
    assert list(tmp_path.glob(".dump-*")) == []

def test_refusal_does_not_delete_unexpected_staged_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,) -> None:
    source, stage = _bundle(tmp_path, "+opaque-canary-value\n")
    (stage / "keep.txt").write_text("prior evidence")
    monkeypatch.setattr(
        scan, "_RULES", (*scan._RULES, (re.compile("opaque-canary-value"), "opaque", scan.SEVERITY_BLOCKING)),
    )
    with pytest.raises(OSError):
        dump.publish_dump(source, stage, "session-1", sanitize=True)
    assert (stage / "keep.txt").read_text() == "prior evidence"
    assert list(tmp_path.glob(".dump-*")) == []
