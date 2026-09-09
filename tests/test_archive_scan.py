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


def test_already_redacted_markers_skip(tmp_path: Path) -> None:
    """Marker-skip branch: a credential-shaped value whose match carries a
    [REDACTED_*] marker is already-safe sanitizer output, so it must not be
    flagged (scan.py skips such matches instead of re-reporting them)."""
    run_dir = tmp_path / "run"
    _write_manifest(
        run_dir, "https://user:[REDACTED_PASSWORD]@github.com/o/r.git"
    )
    assert scan.scan_run_dir(run_dir).clean


def test_scan_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    _write_manifest(run_dir, "https://github.com/o/r")
    monkeypatch.setattr(scan, "_scan_text", lambda _t: (_ for _ in ()).throw(RuntimeError("boom")))
    result = scan.scan_run_dir(run_dir)
    assert not result.clean  # M-constraint: any scanner error blocks egress
    assert result.blocking  # scan_error is always blocking


# --- V3: the tiered rule table (issue #1170) -------------------------------
#
# The scanner reuses redaction rules as publication rules. A redactor needs
# recall; a publication gate needs precision, because over-matching costs the
# whole run. These two corpora pin the split: ordinary config/template code
# may report, but must never refuse egress; real credential values must.

# Mirrors #455's exemplar (test_env_var_pattern_does_not_match_substring_lookalikes).
_FALSE_POSITIVES = (
    'SORT_KEY = "created_at"',
    "PRIMARY_KEY = 'id'",
    "CACHE_KEY = 'v1'",
    "AUTH=none",
    "API_TOKEN=${API_TOKEN}",
    "PARTITION_KEY = event.get('pk')",
    # The line from the issue's reproduction.
    'FEATURE_FLAG_OVERRIDE_KEY = "override_flag"',
    'return f"{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"',
    'DSN = f"postgresql://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"',
    # The two https-scheme templates. _URL_CREDENTIAL_PATTERN is a strict
    # subset of _TOKEN_ONLY_USERINFO_PATTERN, so a guard applied to only some
    # of the userinfo rules leaves these blocking through the wider rule.
    'REMOTE = f"https://{cfg.GH_USER}:{cfg.GH_PASSWORD}@github.com/{o}/{r}.git"',
    'url = "https://{user}:{password}@registry.example.com/simple"',
)

_TRUE_POSITIVES = (
    ('token = "ghp_canaryfake123"', "api_key"),
    ('aws_key = "AKIAIOSFODNN7EXAMPLE"', "api_key"),
    ('openai = "sk-abcdef1234567890"', "api_key"),
    (
        'jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"',
        "jwt",
    ),
    (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzk4OcQ+wgUjOtOOg=\n"
        "-----END RSA PRIVATE KEY-----",
        "pem_key",
    ),
    ('url = "https://api.example.com/v1/x?token=s3cr3tvalue"', "query_credential"),
    # Only _TOKEN_ONLY_USERINFO_PATTERN matches the single-token form, which is
    # why that rule is guarded rather than dropped.
    ('remote = "https://x-access-token@github.com/o/r"', "url_credential"),
    # SCP-only literals: the discrimination the subset shadowing would destroy.
    ("deploy:s3cr3tvalue@host.example.com:/srv/app", "url_credential"),
    ("postgresql://real:s3cr3tvalue@db.example.com:5432/app", "url_credential"),
)


@pytest.mark.parametrize("text", _FALSE_POSITIVES)
def test_ordinary_code_never_blocks_egress(text: str) -> None:
    """Config-code and DSN-template shapes are advisory-or-clean, never blocking."""
    findings = scan._scan_file("sample.py", text)
    blocking = [f for f in findings if f.severity == scan.SEVERITY_BLOCKING]
    assert blocking == [], [(f.category, f.location) for f in blocking]


@pytest.mark.parametrize(("text", "category"), _TRUE_POSITIVES)
def test_real_credential_shapes_block_egress(text: str, category: str) -> None:
    findings = scan._scan_file("sample.txt", text)
    blocking = {f.category for f in findings if f.severity == scan.SEVERITY_BLOCKING}
    assert category in blocking, [(f.category, f.severity) for f in findings]


def test_overlapping_userinfo_rules_dedupe_to_strictest_severity() -> None:
    """Rules 1 and 2 both fire on ``scheme://user:pass@`` with the same match.

    Without dedupe the finding is double-counted in ``summary()``; without the
    blocking-wins tie-break its severity would depend on rule iteration order.
    """
    literal = scan._scan_file("f.txt", "https://user:s3cr3tvalue@example.com/x")
    assert len(literal) == 1
    assert literal[0].category == "url_credential"
    assert literal[0].severity == scan.SEVERITY_BLOCKING

    templated = scan._scan_file("f.txt", 'u = "https://{user}:{password}@example.com/x"')
    assert len(templated) == 1
    assert templated[0].severity == scan.SEVERITY_ADVISORY


def test_dirty_result_without_findings_is_blocking() -> None:
    """Fail-closed tie-break: ``clean=False`` blocks even with no findings."""
    assert scan.ScanResult(clean=False).blocking
    assert not scan.ScanResult().blocking


# --- V4: multi-line PEM armor in a non-JSON file (issue #1170 D4) ----------


def test_multiline_pem_in_patch_file_blocks(tmp_path: Path) -> None:
    """``_PEM_KEY_PATTERN`` is DOTALL but the non-JSON pass is line-by-line, so
    real key armor in ``diff.patch`` was invisible while the same key inside a
    JSON string leaf was caught."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "diff.patch").write_text(
        "--- a/id_rsa\n"
        "+++ b/id_rsa\n"
        "@@ -0,0 +1,3 @@\n"
        "+-----BEGIN RSA PRIVATE KEY-----\n"
        "+MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzk4OcQ+wgUjOtOOg=\n"
        "+-----END RSA PRIVATE KEY-----\n"
    )
    result = scan.scan_run_dir(run_dir)
    assert result.blocking
    pem = [f for f in result.findings if f.category == "pem_key"]
    assert [(f.path, f.location, f.severity) for f in pem] == [
        ("diff.patch", "line 4", scan.SEVERITY_BLOCKING)
    ]
    assert "MIIBOgIBAAJBAKj34" not in result.summary()  # M11: value-free
