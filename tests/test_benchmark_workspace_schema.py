import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from daydream.benchmark.schema import (
    BenchmarkManifest,
    CaseDocument,
    PullRequestEntry,
    TransitionError,
    case_id_for,
    classify_validation,
    derive_finding_id,
    derive_gold_mode,
    derive_gold_status,
    derive_workspace_state,
    normalize_hostname,
    validate_case_transition,
    validate_pr_transition,
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


def _valid_case_dict():
    return {
        "schema_version": 1,
        "case_id": "pr-000101-0123456789ab",
        "pull_request": {"number": 101, "url": "https://github.com/O/R/pull/101", "title": "Fix cache"},
        "snapshot": {
            "status": "ready",
            "policy": "final_pr_head",
            "requested_head": "final",
            "original_base_sha": "0123456789abcdef0123456789abcdef01234567",
            "original_head_sha": "0123456789abcdef0123456789abcdef01234567",
            "base_tree_sha": "0000000000000000000000000000000000000001",
            "head_tree_sha": "0000000000000000000000000000000000000002",
            "diff_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bundle_file": "snapshots/pr-000101-0123456789ab.bundle",
            "bundle_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "error": None,
        },
        "source": {
            "import_file": "imports/pr-101.json",
            "import_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        },
        "curation": {
            "state": "ready",
            "snapshot_attested": True,
            "clean_attested": False,
            "gold_status": "findings",
            "findings": [
                {
                    "finding_id": _finding_id_for(
                        "Cache misses", "The cache layers never populate.", "high", "src/cache.py", 2, 2
                    ) or "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                    "title": "Cache misses",
                    "body": "The cache layers never populate.",
                    "severity": "high",
                    "location": {"path": "src/cache.py", "start_line": 2, "end_line": 2},
                    "provenance": {"kind": "edited", "source_ids": ["github:review_comment:1"]},
                }
            ],
            "exclusions": [],
            "case_exclusion": None,
        },
    }


def _finding_id_for(title, body, severity, path, start_line, end_line):
    return derive_finding_id(
        {
            "title": title,
            "body": body,
            "severity": severity,
            "location": {"path": path, "start_line": start_line, "end_line": end_line},
        }
    )


def _valid_case():
    return CaseDocument.model_validate(_valid_case_dict())


def test_case_id_derivation():
    assert case_id_for(101, "0123456789abcdef0123456789abcdef01234567") == "pr-000101-0123456789ab"


def test_ready_snapshot_valid():
    doc = _valid_case()
    assert doc.snapshot.status == "ready"
    assert doc.curation.state == "ready"


def test_ready_snapshot_rejects_missing_bundle_fields():
    # Re-validate from a raw dict so a missing required field is caught.
    raw = _valid_case_dict()
    raw["snapshot"].pop("bundle_file")
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_unreplayable_snapshot_requires_error_and_null_bundle():
    raw = _valid_case_dict()
    raw["snapshot"] = {
        "status": "unreplayable",
        "policy": "final_pr_head",
        "requested_head": "final",
        "original_base_sha": None,
        "original_head_sha": "0123456789abcdef0123456789abcdef01234567",
        "base_tree_sha": None,
        "head_tree_sha": None,
        "diff_sha256": None,
        "bundle_file": None,
        "bundle_sha256": None,
        "error": {"reason": "head_not_on_pr", "detail": "head sha not on PR"},
    }
    doc = CaseDocument.model_validate(raw)
    assert doc.snapshot.status == "unreplayable"


def test_unreplayable_with_ready_fields_rejected():
    raw = _valid_case_dict()
    raw["snapshot"]["status"] = "unreplayable"
    raw["snapshot"]["error"] = {"reason": "equal_trees", "detail": "no change"}
    raw["snapshot"]["bundle_file"] = "snapshots/pr-000101-0123456789ab.bundle"  # must be null
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)

def test_finding_title_and_body_limits():
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["title"] = "x" * 501
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)
    raw2 = _valid_case_dict()
    raw2["curation"]["findings"][0]["body"] = "y" * (8 * 1024 + 1)
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw2)


def test_finding_rejects_nul_in_title():
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["title"] = "bad\x00title"
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_finding_severity_enum():
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["severity"] = "critical"
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_finding_location_must_be_relative_and_ordered():
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["location"] = {"path": "/abs/path.py", "start_line": 2, "end_line": 2}
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)
    raw2 = _valid_case_dict()
    raw2["curation"]["findings"][0]["location"] = {"path": "src/cache.py", "start_line": 5, "end_line": 2}
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw2)


def test_finding_id_sha256_and_duplicate_rejection():
    f = _valid_case().curation.findings[0]
    expected = derive_finding_id(f)
    assert f.finding_id == expected
    # duplicate canonical finding in one case rejected
    raw = _valid_case_dict()
    raw["curation"]["findings"].append(raw["curation"]["findings"][0])
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_historical_daydream_marker_cannot_be_gold():
    from daydream.pr_review import finding_marker

    raw = _valid_case_dict()
    body = "looks fine"
    finding = {
        "title": "Daydream self-output",
        "body": body + finding_marker("a" * 64),
        "severity": None,
        "location": None,
        "provenance": {"kind": "historical", "source_ids": ["github:review_comment:1"]},
    }
    finding["finding_id"] = derive_finding_id(finding)  # canonical, so the marker guard is what rejects
    raw["curation"]["findings"][0] = finding
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_gold_status_and_mode_derived():
    case = _valid_case()  # ready, 1 finding, clean_attested=False
    assert derive_gold_status(case.curation) == "findings"
    assert derive_gold_mode(case.curation) == "historical"


@pytest.mark.parametrize(
    "frm,to",
    [("pending", "fetched"), ("pending", "fetch_failed"), ("fetch_failed", "fetched"), ("fetched", "fetched")],
)
def test_valid_pr_transitions(frm, to):
    validate_pr_transition(frm, to)  # must not raise


@pytest.mark.parametrize("frm,to", [("fetched", "pending"), ("pending", "draft")])
def test_invalid_pr_transition_rejected(frm, to):
    with pytest.raises(TransitionError):
        validate_pr_transition(frm, to)


@pytest.mark.parametrize(
    "frm,to",
    [
        ("draft", "ready"),
        ("draft", "excluded"),
        ("draft", "unreplayable"),
        ("ready", "stale"),
        ("ready", "draft"),
        ("stale", "ready"),
        ("stale", "excluded"),
        ("unreplayable", "excluded"),
        ("excluded", "draft"),
        ("excluded", "unreplayable"),
    ],
)
def test_valid_case_transitions(frm, to):
    validate_case_transition(frm, to)


@pytest.mark.parametrize("frm,to", [("ready", "excluded"), ("stale", "draft"), ("draft", "stale")])
def test_invalid_case_transition_rejected(frm, to):
    with pytest.raises(TransitionError):
        validate_case_transition(frm, to)


def test_derived_workspace_state_empty_vs_collecting():
    assert derive_workspace_state(pull_requests=[], cases=[]) == "empty"
    assert (
        derive_workspace_state(pull_requests=[{"number": 1, "import_state": "pending"}], cases=[])
        == "collecting"
    )
    assert (
        derive_workspace_state(
            pull_requests=[],
            cases=[
                {
                    "case_id": "pr-000001-0123456789ab",
                    "pr_number": 1,
                    "case_file": "x.yaml",
                    "curation_state": "ready",
                }
            ],
        )
        == "ready"
    )


def test_classify_validation_codes():
    assert classify_validation(ready=True, incomplete=False, corrupt=False) == 0
    assert classify_validation(ready=False, incomplete=True, corrupt=False) == 2
    assert classify_validation(ready=False, incomplete=False, corrupt=True) == 1
