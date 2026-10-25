"""Fail-closed bundle secret scanner (issue #981 M9/M11/M13)."""

import json
from pathlib import Path

import pytest

from daydream.archive import scan


def _write_manifest(run_dir: Path, remote_url: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"git": {"remote_url": remote_url}}))


def test_scan_flags_credential_url_in_manifest(tmp_path: Path) -> None:
    """Scan one artifact tree before and after credential-bearing evidence is added.

    Exact per-file locations and categories distinguish every rule; an unrelated secret cannot
    mask a false positive in config/templates or a missed credential in another file.
    """
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "https://user:[REDACTED_PASSWORD]@github.com/o/r.git")
    config = run_dir / "config.py"
    config.write_text(
        "SORT_KEY = \"created_at\"\nPRIMARY_KEY = 'id'\nCACHE_KEY = 'v1'\nAUTH=none\n"
        "API_TOKEN=${API_TOKEN}\nPARTITION_KEY = event.get('pk')\n"
        'FEATURE_FLAG_OVERRIDE_KEY = "override_flag"\n'
        'return f"{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"\n'
        'DSN = f"postgresql://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"\n'
        'REMOTE = f"https://{cfg.GH_USER}:{cfg.GH_PASSWORD}@github.com/{o}/{r}.git"\n'
        'url = "https://{user}:{password}@registry.example.com/simple"\n'
    )
    initial = scan.scan_run_dir(run_dir)
    assert not initial.blocking
    assert all(f.path != "manifest.json" for f in initial.findings)
    # The overlapping URL rules produce one advisory finding for the template on line11.
    assert [(f.category, f.severity) for f in initial.findings if f.location == "line 11"] == [
        ("url_credential", scan.SEVERITY_ADVISORY)
    ]

    _write_manifest(run_dir, "https://user:ghp_canaryfake123@github.com/o/r.git")
    (run_dir / "secrets.txt").write_text(
        'token = "ghp_canaryfake123"\naws_key = "AKIAIOSFODNN7EXAMPLE"\n'
        'openai = "sk-abcdef1234567890"\n'
        'jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"\n'
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzk4OcQ+wgUjOtOOg=\n"
        "-----END RSA PRIVATE KEY-----\n"
        'url = "https://api.example.com/v1/x?token=s3cr3tvalue"\n'
        'remote = "https://x-access-token@github.com/o/r"\n'
        "deploy:s3cr3tvalue@host.example.com:/srv/app\n"
        "postgresql://real:s3cr3tvalue@db.example.com:5432/app\n"
        "https://user:s3cr3tvalue@example.com/x\n"
    )
    (run_dir / "diff.patch").write_text(
        "--- a/id_rsa\n"
        "+++ b/id_rsa\n"
        "@@ -0,0 +1,3 @@\n"
        "+-----BEGIN RSA PRIVATE KEY-----\n"
        "+MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzk4OcQ+wgUjOtOOg=\n"
        "+-----END RSA PRIVATE KEY-----\n"
    )
    result = scan.scan_run_dir(run_dir)
    assert not result.clean and result.blocking
    assert [(f.path, f.location, f.category) for f in result.findings if f.severity == scan.SEVERITY_BLOCKING] == [
        ("diff.patch", "line 4", "pem_key"),
        ("manifest.json", "git.remote_url (json)", "url_credential"),
        ("manifest.json", "git.remote_url (json)", "api_key"),
        ("secrets.txt", "line 1", "api_key"),
        ("secrets.txt", "line 2", "api_key"),
        ("secrets.txt", "line 3", "api_key"),
        ("secrets.txt", "line 4", "jwt"),
        ("secrets.txt", "line 8", "query_credential"),
        ("secrets.txt", "line 9", "url_credential"),
        ("secrets.txt", "line 10", "url_credential"),
        ("secrets.txt", "line 11", "url_credential"),
        ("secrets.txt", "line 12", "url_credential"),
        ("secrets.txt", "line 5", "pem_key"),
    ]
    assert [f for f in result.findings if f.path == "config.py"] == initial.findings
    assert all(len(f.digest) == 12 and set(f.digest) <= set("0123456789abcdef") for f in result.findings)
    summary = result.summary()
    assert "git.remote_url" not in summary  # report paths/categories without source fields or values
    assert all(
        secret not in summary
        for secret in (
            "ghp_canaryfake123",
            "AKIAIOSFODNN7EXAMPLE",
            "sk-abcdef1234567890",
            "eyJhbGciOiJIUzI1NiJ9",
            "MIIBOgIBAAJBAKj34",
            "x-access-token",
            "s3cr3tvalue",
        )
    )


def test_scan_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "https://github.com/o/r")
    monkeypatch.setattr(scan, "_scan_text", lambda _t: (_ for _ in ()).throw(RuntimeError("boom")))
    result = scan.scan_run_dir(run_dir)
    assert not result.clean  # M-constraint: any scanner error blocks egress
    assert result.blocking  # scan_error is always blocking


def test_dirty_result_without_findings_is_blocking() -> None:
    assert scan.ScanResult(clean=False).blocking
    assert not scan.ScanResult().blocking
