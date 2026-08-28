"""Fail-closed bundle secret scanner (issue #981 M9/M11/M13)."""
import json
from pathlib import Path

import pytest

from daydream.archive import scan


def _write_manifest(run_dir: Path, remote_url: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"git": {"remote_url": remote_url}}))


def test_scan_flags_credential_url_in_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "https://user:ghp_canaryfake123@github.com/o/r.git")
    result = scan.scan_run_dir(run_dir)
    assert not result.clean
    # M11: safe reporting only — field name + file name, never the value
    assert result.findings[0].path == "manifest.json"
    assert "git.remote_url" in result.findings[0].location
    assert "ghp_canaryfake123" not in result.summary()  # canary never echoed


def test_token_only_userinfo_flagged(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "https://x-access-token@github.com/o/r")
    assert not scan.scan_run_dir(run_dir).clean


def test_clean_bundle_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "https://github.com/o/r")
    (run_dir / "diff.patch").write_text("+++ b/f.py\n+print('hi')\n")
    assert scan.scan_run_dir(run_dir).clean


def test_canary_in_patch_file_flagged(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "https://github.com/o/r")
    (run_dir / "diff.patch").write_text("+TOKEN=ghp_canaryfake123\n")
    assert not scan.scan_run_dir(run_dir).clean


def test_scan_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "https://github.com/o/r")
    monkeypatch.setattr(scan, "_scan_text", lambda _t: (_ for _ in ()).throw(RuntimeError("boom")))
    result = scan.scan_run_dir(run_dir)
    assert not result.clean  # M-constraint: any scanner error blocks egress
