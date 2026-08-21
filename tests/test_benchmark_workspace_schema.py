import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from daydream.benchmark.schema import (
    BenchmarkManifest,
    PullRequestEntry,
    normalize_hostname,
)


def test_pyyaml_is_a_base_runtime_dependency():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    dev = data["dependency-groups"]["dev"]
    assert any(d == "pyyaml>=6.0" or d.startswith("pyyaml") for d in deps)
    assert "types-pyyaml>=6.0" in dev  # type stubs stay dev-only


def _valid_manifest():
    return {
        "schema_version": 1,
        "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
        "created_at": "2026-08-21T12:00:00Z",
        "source": {
            "provider": "github",
            "hostname": "github.com",
            "repository": "OWNER/REPO",
            "repository_id": None,
            "visibility": "unresolved",
        },
        "privacy": {
            "classification": "confidential",
            "reviewer_data": "source_snapshot",
            "reviewer_allowed_hosts": ["api.anthropic.com"],
            "judge_data": "finding_text_and_location_only",
            "judge_allowed_hosts": ["api.anthropic.com"],
            "archive": "disabled",
            "uploads": "disabled",
        },
        "pull_requests": [],
        "cases": [],
    }


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("api.anthropic.com", "api.anthropic.com"),
        ("API.Anthropic.COM", "api.anthropic.com"),
        ("https://api.anthropic.com", "api.anthropic.com"),
        ("api.anthropic.com:443", "api.anthropic.com"),
        ("user:pass@api.anthropic.com", "api.anthropic.com"),
        ("api.anthropic.com/path", "api.anthropic.com"),
    ],
)
def test_normalize_hostname(raw, expected):
    assert normalize_hostname(raw) == expected


@pytest.mark.parametrize("raw", ["", "*.anthropic.com", "not a host", "http://"])
def test_normalize_hostname_rejects_malformed(raw):
    with pytest.raises(ValueError):
        normalize_hostname(raw)


def test_manifest_accepts_valid_v1():
    m = BenchmarkManifest.model_validate(_valid_manifest())
    assert m.source.hostname == "github.com"
    assert m.source.repository_id is None
    assert m.source.visibility == "unresolved"


def test_manifest_rejects_unknown_field():
    base = _valid_manifest()
    base["bogus"] = True
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_manifest_rejects_non_github_hostname():
    base = _valid_manifest()
    base["source"]["hostname"] = "gitlab.com"
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_manifest_rejects_empty_host_allowlists():
    base = _valid_manifest()
    base["privacy"]["reviewer_allowed_hosts"] = []
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_manifest_rejects_bad_uuid():
    base = _valid_manifest()
    base["benchmark_id"] = "not-a-uuid"
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_manifest_rejects_non_utc_timestamp():
    base = _valid_manifest()
    base["created_at"] = "2026-08-21T12:00:00"  # no Z / offset
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_ledger_entry_valid_and_fetch_failed_error_shape():
    ok = PullRequestEntry(number=101, import_state="pending", requested_heads=["final"], case_ids=[])
    assert ok.import_file is None and ok.import_sha256 is None and ok.error is None

    failed = PullRequestEntry(
        number=102,
        import_state="fetch_failed",
        requested_heads=["final"],
        case_ids=[],
        error={"code": "E_AUTH", "message": "no access"},
    )
    assert failed.error["code"] == "E_AUTH"
