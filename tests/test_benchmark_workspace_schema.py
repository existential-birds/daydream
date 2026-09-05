import hashlib
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import daydream.benchmark.schema as schema
from daydream.benchmark.schema import (
    BenchmarkManifest,
    CaseDocument,
    Curation,
    EvidenceRecord,
    Finding,
    ImportDocument,
    Provenance,
    PullRequestEntry,
    PullRequestMeta,
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


def test_pyyaml_is_a_base_runtime_dependency() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    dev = data["dependency-groups"]["dev"]
    assert any(d == "pyyaml>=6.0" or d.startswith("pyyaml") for d in deps)
    assert any(d == "types-pyyaml>=6.0" or d.startswith("types-pyyaml") for d in dev), (
        "type stubs stay dev-only"
    )


def _valid_manifest() -> dict[str, Any]:
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
def test_normalize_hostname(raw: Any, expected: Any) -> None:
    assert normalize_hostname(raw) == expected


@pytest.mark.parametrize("raw", ["", "*.anthropic.com", "not a host", "http://"])
def test_normalize_hostname_rejects_malformed(raw: Any) -> None:
    with pytest.raises(ValueError):
        normalize_hostname(raw)


def test_manifest_accepts_valid_v1() -> None:
    m = BenchmarkManifest.model_validate(_valid_manifest())
    assert m.source.hostname == "github.com"
    assert m.source.repository_id is None
    assert m.source.visibility == "unresolved"


def test_manifest_rejects_unknown_field() -> None:
    base = _valid_manifest()
    base["bogus"] = True
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_manifest_rejects_non_github_hostname() -> None:
    base = _valid_manifest()
    base["source"]["hostname"] = "gitlab.com"
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_source_repository_id_is_nonblank_opaque_string() -> None:
    from pydantic import ValidationError

    from daydream.benchmark.schema import Source

    s = Source(provider="github", hostname="github.com", repository="o/r",
               repository_id="R_kgDOABC123")
    assert s.repository_id == "R_kgDOABC123"
    # None sentinel (unresolved) unchanged
    assert Source(provider="github", hostname="github.com",
                  repository="o/r").repository_id is None
    # numeric-only and blank node ids are rejected as invalid
    for bad in ("5", "   ", "123456"):
        with pytest.raises(ValidationError):
            Source(provider="github", hostname="github.com",
                   repository="o/r", repository_id=bad)


def test_import_repository_id_is_opaque_string() -> None:
    from daydream.benchmark.schema import _ImportRepository

    r = _ImportRepository(id="R_kgDOABC123", name_with_owner="o/r", visibility="private")
    assert r.id == "R_kgDOABC123"
    # numeric-only ids (Pydantic would otherwise coerce int->str) must not model;
    # blank "" is the deliberate unresolved sentinel from _repository_block.
    with pytest.raises(ValidationError):
        _ImportRepository(id="5", name_with_owner="o/r", visibility="private")
    with pytest.raises(ValidationError):
        _ImportRepository(id="123456", name_with_owner="o/r", visibility="private")
    assert _ImportRepository(id="", name_with_owner="o/r", visibility="private").id == ""



def test_manifest_rejects_empty_host_allowlists() -> None:
    base = _valid_manifest()
    base["privacy"]["reviewer_allowed_hosts"] = []
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_privacy_classification_and_policies_are_literals() -> None:
    m = _valid_manifest()
    for key, bad in [("classification", "public"), ("reviewer_data", "everything"),
                     ("judge_data", "full"), ("archive", "enabled"), ("uploads", "enabled")]:
        raw = dict(m)
        raw["privacy"] = dict(m["privacy"])
        raw["privacy"][key] = bad
        with pytest.raises(ValidationError):
            BenchmarkManifest.model_validate(raw)


def test_snapshot_policy_is_literal() -> None:
    raw = _valid_case_dict()
    raw["snapshot"]["policy"] = "some_head"
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_reviewer_judge_hosts_stay_lists() -> None:
    m = BenchmarkManifest.model_validate(_valid_manifest())
    assert m.privacy.reviewer_allowed_hosts == ["api.anthropic.com"]
    assert m.privacy.judge_allowed_hosts == ["api.anthropic.com"]


def test_manifest_rejects_bad_uuid() -> None:
    base = _valid_manifest()
    base["benchmark_id"] = "not-a-uuid"
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_manifest_rejects_non_utc_timestamp() -> None:
    base = _valid_manifest()
    base["created_at"] = "2026-08-21T12:00:00"  # no Z / offset
    with pytest.raises(ValidationError):
        BenchmarkManifest.model_validate(base)


def test_ledger_entry_valid_and_fetch_failed_error_shape() -> None:
    ok = PullRequestEntry(number=101, import_state="pending", requested_heads=["final"], case_ids=[])
    assert ok.import_file is None and ok.import_sha256 is None and ok.error is None

    failed = PullRequestEntry(
        number=102,
        import_state="fetch_failed",
        requested_heads=["final"],
        case_ids=[],
        error={"code": "E_AUTH", "message": "no access"},
    )
    assert failed.error is not None
    assert failed.error["code"] == "E_AUTH"


def _evidence(kind: Any, db_id: Any, **kw: Any) -> Any:
    body = "see above"
    base = {
        "source_id": f"github:{kind}:{db_id}", "kind": kind, "database_id": db_id,
        "node_id": "N1", "author": {"login": "bot[bot]", "type": "Bot"},
        "body": body, "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z", "submitted_at": None, "is_bot": True,
        "url": "https://github.com/o/r/pull/1#discussion_r1",
    }
    base.update(kw)
    return base


def _valid_import_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repository": {"id": "R_kgDOABC123", "name_with_owner": "o/r", "visibility": "private"},
        "pull_request": {
            "number": 101,
            "url": "https://github.com/o/r/pull/101",
            "html_url": "https://github.com/o/r/pull/101",
            "title": "Fix cache",
            "body": "fixes the cache",
            "state": "open",
            "title_sha256": hashlib.sha256(b"Fix cache").hexdigest(),
            "body_sha256": hashlib.sha256("fixes the cache".encode()).hexdigest(),
            "base": {"ref": "main", "sha": "b" * 40},
            "head": {"ref": "feature/cache", "sha": "h" * 40},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "merged_at": None,
            "closed_at": None,
            "author": {"login": "alice", "type": "User"},
        },
        "evidence": [_evidence("inline_comment", 7)],
        "fetch": {"fetched_at": "2026-01-01T00:00:00Z", "etag": None, "payload_sha256": "a" * 64},
    }


def test_import_document_validates_and_forbids_unknown() -> None:
    doc = _valid_import_document()
    assert ImportDocument.model_validate(doc).pull_request.number == 101
    doc["bogus"] = True
    with pytest.raises(ValidationError):
        ImportDocument.model_validate(doc)


def test_import_pull_request_is_strict_submodel() -> None:
    doc = _valid_import_document()
    doc["pull_request"]["bogus"] = True
    with pytest.raises(ValidationError) as ei:
        ImportDocument.model_validate(doc)
    assert ei.value.errors()[0]["loc"][0] == "pull_request"


def test_case_pull_request_is_strict_submodel() -> None:
    # partial (missing a required field) rejected
    raw = _valid_case_dict()
    raw["pull_request"].pop("author")
    with pytest.raises(ValidationError) as ei:
        CaseDocument.model_validate(raw)
    assert ei.value.errors()[0]["loc"][0] == "pull_request"
    # unknown nested key rejected
    raw2 = _valid_case_dict()
    raw2["pull_request"]["bogus"] = 1
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw2)


def test_case_source_is_strict_submodel() -> None:
    raw = _valid_case_dict()
    raw["source"]["bogus"] = 1
    with pytest.raises(ValidationError) as ei:
        CaseDocument.model_validate(raw)
    assert ei.value.errors()[0]["loc"][0] == "source"


def test_import_and_case_pull_request_share_shape() -> None:
    doc = ImportDocument.model_validate(_valid_import_document())
    pr = doc.pull_request
    assert pr.number == 101 and pr.author.login == "alice" and pr.head.sha == "h" * 40


def test_pull_request_meta_accepts_full_field_set() -> None:
    m = PullRequestMeta.model_validate({
        "number": 101, "url": "https://github.com/o/r/pull/101",
        "html_url": "https://github.com/o/r/pull/101",
        "title": "Fix cache", "body": "fixes the cache\n\nsecond line",
        "state": "open",
        "title_sha256": hashlib.sha256(b"Fix cache").hexdigest(),
        "body_sha256": hashlib.sha256("fixes the cache\n\nsecond line".encode()).hexdigest(),
        "base": {"sha": "b" * 40, "ref": "main"},
        "head": {"sha": "a" * 40, "ref": "feature/cache"},
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "merged_at": None, "closed_at": None,
        "author": {"login": "alice", "type": "User"},
    })
    assert m.number == 101 and m.body == "fixes the cache\n\nsecond line"
    assert m.head.ref == "feature/cache"


def test_pull_request_meta_predate_reads_empty_and_validates() -> None:
    # predate import: lacks the additive body/digest/html_url/merged/closed fields
    m = PullRequestMeta.model_validate({
        "number": 101, "url": "https://github.com/o/r/pull/101",
        "title": "Fix cache", "state": "open",
        "base": {"ref": "main", "sha": "b" * 40},
        "head": {"sha": "a" * 40},                 # no head.ref (old import dropped it)
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "author": {"login": "alice", "type": "User"},
    })
    assert m.body == "" and m.title_sha256 == "" and m.body_sha256 == ""
    assert m.merged_at is None and m.closed_at is None and m.html_url == ""
    assert m.head.ref is None


def test_pull_request_meta_fails_closed_on_malformed_required() -> None:
    base = {
        "number": 101, "url": "u", "title": "t", "state": "open",
        "base": {"sha": "b" * 40}, "head": {"sha": "a" * 40},
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "author": {"login": "a", "type": "User"},
    }
    with pytest.raises(ValidationError):
        PullRequestMeta.model_validate(dict(base, number="abc"))
    with pytest.raises(ValidationError):
        PullRequestMeta.model_validate(dict(base, head={"ref": "x"}))  # no sha
    with pytest.raises(ValidationError):
        PullRequestMeta.model_validate(dict(base, bogus=1))            # extra forbid


def test_pull_request_meta_digests_are_64hex_and_match_body_when_present() -> None:
    full = {"number": 1, "url": "u", "title": "t", "state": "open",
            "base": {"sha": "b" * 40}, "head": {"sha": "a" * 40},
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
            "author": {"login": "a", "type": "User"}, "body": "hello"}
    with pytest.raises(ValidationError):
        PullRequestMeta.model_validate(dict(full, body_sha256="zz"))
    with pytest.raises(ValidationError):
        PullRequestMeta.model_validate(dict(full, body_sha256=hashlib.sha256(b"other").hexdigest()))


def test_evidence_requires_canonical_source_id_and_body_hash() -> None:
    e = _evidence("inline_comment", 7)
    e["source_id"] = "not-canonical"
    with pytest.raises(ValidationError):
        EvidenceRecord.model_validate(e)
    e2 = _evidence("inline_comment", 8)
    e2["body_sha256"] = "zz"
    with pytest.raises(ValidationError):
        EvidenceRecord.model_validate(e2)


def _anchor_payload(status: str, commit_id: str | None = "a" * 40) -> dict[str, Any]:
    """An AuthoringAnchor payload with data fields set only for the derived status."""
    if status == "derived":
        return {
            "version": 1,
            "status": status,
            "commit_id": commit_id,
            "path": "a.py",
            "start_line": 4,
            "end_line": 5,
        }
    return {
        "version": 1,
        "status": status,
        "commit_id": None,
        "path": None,
        "start_line": None,
        "end_line": None,
    }


def test_evidence_record_accepts_authoring_anchor_and_original_start_line() -> None:
    rec = EvidenceRecord.model_validate(_evidence("inline_comment", 7))
    payload = rec.model_dump(mode="json")
    payload["original_start_line"] = 4
    payload["authoring_anchor"] = {
        "version": 1,
        "status": "derived",
        "commit_id": "a" * 40,
        "path": "a.py",
        "start_line": 4,
        "end_line": 5,
    }
    parsed = schema.EvidenceRecord.model_validate(payload)
    assert parsed.original_start_line == 4
    assert parsed.authoring_anchor is not None
    assert parsed.authoring_anchor.path == "a.py"


def test_authoring_anchor_fail_closed_statuses_validate() -> None:
    for status in ("derived", "history-unavailable", "path-unavailable", "range-unavailable"):
        payload = _anchor_payload(status)  # commit/path/lines None unless status == "derived"
        anchor = schema.AuthoringAnchor.model_validate(payload)
        assert anchor.status == status
    with pytest.raises(ValidationError):
        schema.AuthoringAnchor.model_validate(_anchor_payload("derived", commit_id=None))
    with pytest.raises(ValidationError):
        schema.AuthoringAnchor.model_validate({"version": 1, "status": "guessed"})


def _valid_case_dict() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "case_id": "pr-000101-0123456789ab",
        "pull_request": {
            "number": 101,
            "url": "https://github.com/o/r/pull/101",
            "html_url": "https://github.com/o/r/pull/101",
            "title": "Fix cache",
            "body": "fix",
            "state": "open",
            "title_sha256": hashlib.sha256(b"Fix cache").hexdigest(),
            "body_sha256": hashlib.sha256("fix".encode()).hexdigest(),
            "base": {"ref": "main", "sha": "b" * 40},
            "head": {"ref": "feature/cache", "sha": "h" * 40},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "merged_at": None,
            "closed_at": None,
            "author": {"login": "alice", "type": "User"},
        },
        "snapshot": {
            "status": "ready",
            "policy": "final_pr_head",
            "requested_head": "final",
            "original_base_sha": "0123456789abcdef0123456789abcdef01234567",
            "requested_base_sha": "0123456789abcdef0123456789abcdef01234567",
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
            "import_sha256": "c" * 64,
        },
        "curation": {
            "state": "ready",
            "snapshot_attested": True,
            "clean_attested": False,
            "gold_status": "findings",
            "findings": [
                {
                    "finding_id": _finding_id_for(
                        "pr-000101-0123456789ab",
                        "Cache misses", "The cache layers never populate.", "high", "src/cache.py", 2, 2
                    ),
                    "title": "Cache misses",
                    "body": "The cache layers never populate.",
                    "severity": "high",
                    "location": {"path": "src/cache.py", "start_line": 2, "end_line": 2},
                    "provenance": {"kind": "edited", "source_ids": ["github:review_comment:1"]},
                }
            ],
            "exclusions": [],
            "case_exclusion": None,
            "task_spec_sha256": "d" * 64,
        },
    }


def _finding_id_for(
    case_id: Any,
    title: Any,
    body: Any,
    severity: Any,
    path: Any,
    start_line: Any,
    end_line: Any,
) -> Any:
    return derive_finding_id(
        {
            "title": title,
            "body": body,
            "severity": severity,
            "location": {"path": path, "start_line": start_line, "end_line": end_line},
        },
        case_id=case_id,
    )


def _valid_case() -> Any:
    return CaseDocument.model_validate(_valid_case_dict())


def test_case_id_derivation() -> None:
    assert case_id_for(101, "0123456789abcdef0123456789abcdef01234567") == "pr-000101-0123456789ab"


def test_ready_snapshot_valid() -> None:
    doc = _valid_case()
    assert doc.snapshot.status == "ready"
    assert doc.curation.state == "ready"


def test_ready_snapshot_rejects_missing_bundle_fields() -> None:
    # Re-validate from a raw dict so a missing required field is caught.
    raw = _valid_case_dict()
    raw["snapshot"].pop("bundle_file")
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_ready_snapshot_requires_requested_base_sha() -> None:
    # requested_base_sha is required on a ready snapshot (no back-compat).
    raw = _valid_case_dict()
    raw["snapshot"].pop("requested_base_sha")
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)
    # a populated requested_base_sha passes and is the selected base-branch tip.
    doc = _valid_case()
    assert doc.snapshot.requested_base_sha == "0123456789abcdef0123456789abcdef01234567"


def test_ready_requires_snapshot_attestation() -> None:
    raw = _valid_case_dict()
    raw["curation"].update({"state": "ready", "snapshot_attested": False})
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_stale_requires_snapshot_not_attested() -> None:
    raw = _valid_case_dict()
    raw["curation"].update({"state": "stale", "snapshot_attested": True,
                            "gold_status": None, "clean_attested": False})
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_unreplayable_curation_requires_unreplayable_snapshot() -> None:
    raw = _valid_case_dict()                       # snapshot.status == ready
    raw["curation"].update({"state": "unreplayable", "snapshot_attested": False,
                            "clean_attested": False, "gold_status": None, "findings": []})
    with pytest.raises(ValidationError):           # ready snapshot + unreplayable state
        CaseDocument.model_validate(raw)
    # a genuine unreplayable snapshot + unreplayable state loads
    raw["snapshot"] = {
        "status": "unreplayable", "policy": "final_pr_head", "requested_head": "final",
        "original_base_sha": None, "requested_base_sha": None,
        "original_head_sha": "0123456789abcdef0123456789abcdef01234567",
        "base_tree_sha": None, "head_tree_sha": None, "diff_sha256": None,
        "bundle_file": None, "bundle_sha256": None,
        "error": {"reason": "head_not_on_pr", "detail": "head sha not on PR"},
    }
    doc = CaseDocument.model_validate(raw)
    assert doc.curation.state == "unreplayable"


def test_unreplayable_snapshot_requires_error_and_null_bundle() -> None:
    raw = _valid_case_dict()
    raw["snapshot"] = {
        "status": "unreplayable",
        "policy": "final_pr_head",
        "requested_head": "final",
        "original_base_sha": None,
        "requested_base_sha": None,
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


def test_unreplayable_with_ready_fields_rejected() -> None:
    raw = _valid_case_dict()
    raw["snapshot"]["status"] = "unreplayable"
    raw["snapshot"]["error"] = {"reason": "equal_trees", "detail": "no change"}
    raw["snapshot"]["bundle_file"] = "snapshots/pr-000101-0123456789ab.bundle"  # must be null
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)

def test_finding_title_and_body_limits() -> None:
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["title"] = "x" * 501
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)
    raw2 = _valid_case_dict()
    raw2["curation"]["findings"][0]["body"] = "y" * (8 * 1024 + 1)
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw2)


def test_finding_rejects_nul_in_title() -> None:
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["title"] = "bad\x00title"
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_title_bound_is_unicode_characters_not_bytes() -> None:
    # "界" is 3 UTF-8 bytes; 500 chars = 1500 bytes. Passes under a char bound.
    title = "界" * 500
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["title"] = title
    raw["curation"]["findings"][0]["finding_id"] = derive_finding_id(
        {"title": title, "body": "The cache layers never populate.", "severity": "high",
         "location": {"path": "src/cache.py", "start_line": 2, "end_line": 2}},
        case_id=raw["case_id"],
    )
    doc = CaseDocument.model_validate(raw)          # 500 chars passes
    assert doc.curation.findings[0].title == title
    # 501 chars fails
    raw2 = _valid_case_dict()
    raw2["curation"]["findings"][0]["title"] = "界" * 501
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw2)


def test_finding_severity_enum() -> None:
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["severity"] = "critical"
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_finding_location_must_be_relative_and_ordered() -> None:
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["location"] = {"path": "/abs/path.py", "start_line": 2, "end_line": 2}
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)
    raw2 = _valid_case_dict()
    raw2["curation"]["findings"][0]["location"] = {"path": "src/cache.py", "start_line": 5, "end_line": 2}
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw2)


def test_finding_id_sha256_and_duplicate_rejection() -> None:
    f = _valid_case().curation.findings[0]
    expected = derive_finding_id(f, case_id="pr-000101-0123456789ab")
    assert f.finding_id == expected
    # duplicate canonical finding in one case rejected
    raw = _valid_case_dict()
    raw["curation"]["findings"].append(raw["curation"]["findings"][0])
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_finding_id_is_case_scoped() -> None:
    f = _valid_case().curation.findings[0]
    id1 = derive_finding_id(f, case_id="pr-000101-0123456789ab")
    id2 = derive_finding_id(f, case_id="pr-000102-0123456789ab")
    assert id1 != id2                      # identical content, different case -> different id


def test_v2_case_rejects_noncanonical_finding_id() -> None:
    raw = _valid_case_dict()
    raw["curation"]["findings"][0]["finding_id"] = "0" * 64   # wrong digest
    with pytest.raises(ValidationError) as ei:
        CaseDocument.model_validate(raw)
    assert "finding_id" in str(ei.value)


def test_v1_legacy_case_loads_without_digest_check() -> None:
    raw = _valid_case_dict()
    raw["schema_version"] = 1
    raw["curation"]["findings"][0]["finding_id"] = "e" * 64   # legacy id, not case-scoped
    doc = CaseDocument.model_validate(raw)                    # must load (digest gated on v2)
    assert doc.schema_version == 1


def test_historical_daydream_marker_cannot_be_gold() -> None:
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
    # canonical, so the marker guard is what rejects
    finding["finding_id"] = derive_finding_id(finding, case_id=raw["case_id"])
    raw["curation"]["findings"][0] = finding
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)


def test_legacy_ready_without_task_spec_digest_backfills_and_validates() -> None:
    """A pre-approval ready case is backfilled with its spec digest by _schema_ready.

    A ready curation persisted before the task-spec approval field existed
    carries no ``task_spec_sha256``; the strict-load preprocessor backfills the
    deterministic render digest so the legacy case validates (and later
    compiles) instead of surfacing as corrupt.
    """
    import daydream.benchmark.schema as schema
    from daydream.benchmark.harbor.build import ASSIGNMENT_TEXT, render_task_spec

    raw = _valid_case_dict()
    del raw["curation"]["task_spec_sha256"]                # legacy pre-approval workspace
    prepared = schema._schema_ready(raw)
    digest = prepared["curation"]["task_spec_sha256"]
    expected = hashlib.sha256(render_task_spec(raw, instruction=ASSIGNMENT_TEXT)).hexdigest()
    assert digest == expected
    assert CaseDocument.model_validate(prepared).curation.task_spec_sha256 == digest


def test_gold_status_and_mode_derived() -> None:
    case = _valid_case()  # ready, 1 finding, clean_attested=False
    assert derive_gold_status(case.curation) == "findings"
    assert derive_gold_mode(case.curation) == "historical"


@pytest.mark.parametrize("kinds,expected", [
    ([], "clean"),
    (["historical"], "historical"),
    (["edited"], "historical"),              # all-edited historical evidence stays historical
    (["authored"], "authored"),
    (["historical", "edited"], "historical"),
    (["authored", "historical"], "mixed"),
    (["authored", "edited"], "mixed"),
])
def test_gold_mode_truth_table(kinds: Any, expected: Any) -> None:
    findings = [
        Finding(finding_id="e" * 64, title=f"f{i}", body="b",
                provenance=Provenance(
                    kind=k,
                    source_ids=["github:review_comment:1"] if k in ("historical", "edited") else [],
                ))
        for i, k in enumerate(kinds)
    ]
    curation = Curation(state="draft", findings=findings)
    assert derive_gold_mode(curation) == expected


@pytest.mark.parametrize(
    "frm,to",
    [
        ("pending", "fetched"),
        ("pending", "fetch_failed"),
        ("fetch_failed", "fetched"),
        ("fetch_failed", "fetch_failed"),
        ("fetched", "fetched"),
    ],
)
def test_valid_pr_transitions(frm: Any, to: Any) -> None:
    validate_pr_transition(frm, to)  # must not raise


@pytest.mark.parametrize(
    "frm,to",
    [
        ("fetched", "pending"),
        ("fetched", "fetch_failed"),  # a fetched PR preserves linkage via latest_error
        ("pending", "draft"),
    ],
)
def test_invalid_pr_transition_rejected(frm: Any, to: Any) -> None:
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
def test_valid_case_transitions(frm: Any, to: Any) -> None:
    validate_case_transition(frm, to)


@pytest.mark.parametrize("frm,to", [("ready", "excluded"), ("stale", "draft"), ("draft", "stale")])
def test_invalid_case_transition_rejected(frm: Any, to: Any) -> None:
    with pytest.raises(TransitionError):
        validate_case_transition(frm, to)


def test_derived_workspace_state_empty_vs_collecting() -> None:
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


def test_classify_validation_codes() -> None:
    assert classify_validation(ready=True, incomplete=False, corrupt=False) == 0
    assert classify_validation(ready=False, incomplete=True, corrupt=False) == 2
    assert classify_validation(ready=False, incomplete=False, corrupt=True) == 1


def test_case_document_accepts_additive_prioritization_key() -> None:
    """Spike pin (task 0, plan #879): the new additive ``prioritization`` key
    loads under the strict CaseDocument schema (schema_version stays 2) and
    round-trips. If this ever fails in a way an optional-field default cannot
    fix, the additive-under-v2 Key Decision is unsound and must be re-routed to
    the spec before any prioritization task runs."""
    raw = _valid_case_dict()
    assert "prioritization" not in raw
    doc = CaseDocument.model_validate(raw)
    assert doc.prioritization is None  # absence tolerated (old case docs load)

    facts = {
        "extraction_version": 1,
        "head_sha": "a" * 40,
        "candidates": {
            "github:review_comment:1": {
                "commit_relation": "at_head",
                "anchor_delta": "unchanged",
            }
        },
        "non_candidates": {},
    }
    raw2 = _valid_case_dict()
    raw2["prioritization"] = facts
    doc2 = CaseDocument.model_validate(raw2)
    assert doc2.prioritization is not None
    assert doc2.prioritization.extraction_version == 1
    assert doc2.prioritization.head_sha == "a" * 40
    cand = doc2.prioritization.candidates["github:review_comment:1"]
    assert cand.commit_relation == "at_head" and cand.anchor_delta == "unchanged"
    assert doc2.prioritization.non_candidates == {}
    # extra="forbid" still holds inside the new subtree
    raw3 = _valid_case_dict()
    raw3["prioritization"] = dict(facts, bogus=1)
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw3)


def test_case_prioritization_facts_shape() -> None:
    from daydream.benchmark.schema import CaseDocument, PrioritizationFacts

    facts = PrioritizationFacts.model_validate(
        {
            "extraction_version": 1,
            "head_sha": "a" * 40,
            "candidates": {"gh:101:c1": {"commit_relation": "at_head", "anchor_delta": "changed"}},
            "non_candidates": {},
        }
    )
    assert facts.extraction_version == 1
    assert facts.candidates["gh:101:c1"].commit_relation == "at_head"
    # additive under v2: an absent key still loads as None
    doc = CaseDocument.model_validate(_valid_case_dict())
    assert doc.prioritization is None


def test_case_document_tolerates_absent_and_rejects_bad_prioritization() -> None:
    raw = _valid_case_dict()
    doc = CaseDocument.model_validate(raw)
    assert doc.prioritization is None  # absent key -> None, old docs load

    raw["prioritization"] = {
        "extraction_version": 1,
        "head_sha": "a" * 40,
        "candidates": {"x": {"commit_relation": "wat", "anchor_delta": "unchanged"}},
    }
    with pytest.raises(ValidationError):  # unknown enum value is a schema violation
        CaseDocument.model_validate(raw)


def test_pull_request_entry_fetched_allows_latest_error_only() -> None:
    # A fetched entry may carry latest_error (a failed refresh attempt on an
    # otherwise-intact import) but never `error` itself.
    base_entry = {
        "number": 101,
        "import_state": "pending",
        "requested_heads": ["final"],
        "case_ids": [],
    }
    valid = dict(
        base_entry,
        import_state="fetched",
        import_file="imports/pr-000101.json",
        import_sha256="a" * 64,
        latest_error={"code": "fetch", "message": "x"},
    )
    PullRequestEntry.model_validate(valid)                       # OK
    bad = dict(valid, error={"code": "fetch", "message": "x"})   # error still forbidden on fetched
    with pytest.raises(ValidationError):
        PullRequestEntry.model_validate(bad)
