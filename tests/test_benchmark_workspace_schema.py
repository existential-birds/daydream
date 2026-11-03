import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

import daydream.benchmark.schema as schema
from daydream.benchmark.harbor.build import (
    ASSIGNMENT_TEXT,
    render_task_spec,
)
from daydream.benchmark.schema import (
    BenchmarkManifest,
    CaseDocument,
    EvidenceRecord,
    PullRequestEntry,
    PullRequestMeta,
    Source,
    TransitionError,
    _ImportRepository,
    derive_finding_id,
    normalize_hostname,
    validate_pr_transition,
)
from daydream.pr_review import finding_marker


def _valid_manifest() -> dict[str, Any]:
    return {"schema_version": 1, "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
        "created_at": "2026-08-21T12:00:00Z",
        "source": {"provider": "github", "hostname": "github.com", "repository": "OWNER/REPO", "repository_id": None,
            "visibility": "unresolved",
        }, "privacy": {"classification": "confidential", "reviewer_data": "source_snapshot",
            "reviewer_allowed_hosts": ["api.anthropic.com"], "judge_data": "finding_text_and_location_only",
            "judge_allowed_hosts": ["api.anthropic.com"], "archive": "disabled", "uploads": "disabled",
        }, "pull_requests": [], "cases": [],
    }


def test_identity_fields_reject_malformed_hosts_and_numeric_node_ids() -> None:
    for host in ("", "not a host"):
        with pytest.raises(ValueError):
            normalize_hostname(host)
    s = Source(provider="github", hostname="github.com", repository="o/r", repository_id="R_kgDOABC123")
    assert s.repository_id == "R_kgDOABC123"
    assert Source(provider="github", hostname="github.com", repository="o/r").repository_id is None
    for bad in ("5", "   ", "123456"):
        with pytest.raises(ValidationError):
            Source(provider="github", hostname="github.com", repository="o/r", repository_id=bad)
    r = _ImportRepository(id="R_kgDOABC123", name_with_owner="o/r", visibility="private")
    assert r.id == "R_kgDOABC123"
    # Numeric-only ids (Pydantic would otherwise coerce int->str) must not model;
    # blank "" is the deliberate unresolved sentinel from _repository_block.
    for bad in ("5", "123456"):
        with pytest.raises(ValidationError):
            _ImportRepository(id=bad, name_with_owner="o/r", visibility="private")
    assert _ImportRepository(id="", name_with_owner="o/r", visibility="private").id == ""


def _evidence(kind: Any, db_id: Any, **kw: Any) -> Any:
    body = "see above"
    base = {"source_id": f"github:{kind}:{db_id}", "kind": kind, "database_id": db_id,
        "node_id": "N1", "author": {"login": "bot[bot]", "type": "Bot"},
        "body": body, "body_sha256": hashlib.sha256(body.encode()).hexdigest(), "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z", "submitted_at": None, "is_bot": True,
        "url": "https://github.com/o/r/pull/1#discussion_r1",
    }
    base.update(kw)
    return base


def test_pull_request_and_evidence_models_reject_noncanonical_input() -> None:
    ok = PullRequestEntry(number=101, import_state="pending", requested_heads=["final"], case_ids=[])
    assert ok.import_file is None and ok.import_sha256 is None and ok.error is None
    failed = PullRequestEntry(number=102, import_state="fetch_failed", requested_heads=["final"], case_ids=[],
        error={"code": "E_AUTH", "message": "no access"},
    )
    assert failed.error is not None and failed.error["code"] == "E_AUTH"
    # A fetched entry may carry latest_error (a failed refresh of an intact import), never `error`.
    fetched = {"number": 101, "import_state": "fetched", "requested_heads": ["final"], "case_ids": [],
               "import_file": "imports/pr-000101.json", "import_sha256": "a" * 64,
               "latest_error": {"code": "fetch", "message": "x"}}
    PullRequestEntry.model_validate(fetched)
    with pytest.raises(ValidationError):
        PullRequestEntry.model_validate(dict(fetched, error={"code": "fetch", "message": "x"}))
    with pytest.raises(TransitionError):
        validate_pr_transition("fetched", "pending")

    meta = {"number": 101, "url": "u", "title": "t", "state": "open", "base": {"sha": "b" * 40},
            "head": {"sha": "a" * 40}, "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
            "author": {"login": "a", "type": "User"}}
    for paths in (["a.py", "a.py"], [""], ["/absolute.py"], ["nul\x00.py"], [1]):
        with pytest.raises(ValidationError):
            PullRequestMeta.model_validate(dict(meta, changed_files=paths))
    for digest in ("zz", hashlib.sha256(b"other").hexdigest()):
        with pytest.raises(ValidationError):
            PullRequestMeta.model_validate(dict(meta, body="hello", body_sha256=digest))

    for field, value in (("source_id", "not-canonical"), ("body_sha256", "zz")):
        with pytest.raises(ValidationError):
            EvidenceRecord.model_validate(_evidence("inline_comment", 7, **{field: value}))


def _anchor_payload(status: str, commit_id: str | None = "a" * 40) -> dict[str, Any]:
    """An AuthoringAnchor payload with data fields set only for the derived status."""
    if status == "derived":
        return {"version": 1, "status": status, "commit_id": commit_id, "path": "a.py", "start_line": 4, "end_line": 5}
    return {"version": 1, "status": status, "commit_id": None, "path": None, "start_line": None, "end_line": None}


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
    return {"schema_version": 2, "case_id": "pr-000101-0123456789ab",
        "pull_request": {
            "number": 101, "url": "https://github.com/o/r/pull/101", "html_url": "https://github.com/o/r/pull/101",
            "title": "Fix cache", "body": "fix", "state": "open",
            "title_sha256": hashlib.sha256(b"Fix cache").hexdigest(),
            "body_sha256": hashlib.sha256("fix".encode()).hexdigest(),
            "base": {"ref": "main", "sha": "0123456789abcdef0123456789abcdef01234567"},
            "head": {"ref": "feature/cache", "sha": "h" * 40}, "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z", "merged_at": None, "closed_at": None,
            "author": {"login": "alice", "type": "User"},
        }, "snapshot": {
            "status": "ready", "base_resolution": "merge_base_v1", "policy": "final_pr_head", "requested_head": "final",
            "original_base_sha": "0123456789abcdef0123456789abcdef01234567",
            "requested_base_sha": "0123456789abcdef0123456789abcdef01234567",
            "original_head_sha": "0123456789abcdef0123456789abcdef01234567",
            "base_tree_sha": "0000000000000000000000000000000000000001",
            "head_tree_sha": "0000000000000000000000000000000000000002",
            "diff_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bundle_file": "snapshots/pr-000101-0123456789ab.bundle",
            "bundle_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "error": None,
        }, "source": {"import_file": "imports/pr-101.json", "import_sha256": "c" * 64},
        "curation": {"state": "ready", "snapshot_attested": True, "clean_attested": False, "gold_status": "findings",
            "findings": [{"finding_id": _finding_id_for("pr-000101-0123456789ab",
                        "Cache misses", "The cache layers never populate.", "high", "src/cache.py", 2, 2
                    ), "title": "Cache misses", "body": "The cache layers never populate.", "severity": "high",
                    "location": {"path": "src/cache.py", "start_line": 2, "end_line": 2},
                    "provenance": {"kind": "edited", "source_ids": ["github:review_comment:1"]},
                }
            ], "exclusions": [], "case_exclusion": None, "task_spec_sha256": "d" * 64,
        },
    }


@pytest.fixture
def raw() -> dict[str, Any]:
    """A fresh valid case dict; pytest rebuilds it per test."""
    return _valid_case_dict()


def _finding_id_for(case_id: Any, title: Any, body: Any, severity: Any, path: Any, start_line: Any, end_line: Any,
) -> Any:
    return derive_finding_id({"title": title, "body": body, "severity": severity,
            "location": {"path": path, "start_line": start_line, "end_line": end_line},
        }, case_id=case_id,
    )


def test_case_document_enforces_attestation_identity_and_marker_invariants() -> None:
    def rejected(mutate: Any) -> None:
        raw = _valid_case_dict()
        mutate(raw)
        with pytest.raises(ValidationError) as caught:
            CaseDocument.model_validate(raw)
        if raw["curation"]["findings"][0]["finding_id"] == "0" * 64:
            assert "finding_id" in str(caught.value)

    rejected(lambda raw: raw["curation"].update({"state": "ready", "snapshot_attested": False}))
    rejected(lambda raw: raw["curation"].update(
        {"state": "stale", "snapshot_attested": True, "gold_status": None, "clean_attested": False}))
    rejected(lambda raw: raw["snapshot"].update({"base_resolution": "merge_base_v1", "requested_base_sha": "c" * 40}))
    rejected(lambda raw: raw["curation"]["findings"][0].update(finding_id="0" * 64))

    def historical_marker(raw: dict[str, Any]) -> None:
        finding = {
            "title": "Daydream self-output", "body": "looks fine" + finding_marker("a" * 64), "severity": None,
            "location": None, "provenance": {"kind": "historical", "source_ids": ["github:review_comment:1"]},
        }
        # Canonical id, so the marker guard is what rejects.
        finding["finding_id"] = derive_finding_id(finding, case_id=raw["case_id"])
        raw["curation"]["findings"][0] = finding
    rejected(historical_marker)

    legacy = _valid_case_dict()
    legacy["schema_version"] = 1
    legacy["curation"]["findings"][0]["finding_id"] = "e" * 64   # legacy id: digest gated on v2
    assert CaseDocument.model_validate(legacy).schema_version == 1

    unapproved = _valid_case_dict()
    del unapproved["curation"]["task_spec_sha256"]                # legacy pre-approval workspace
    prepared = schema._schema_ready(unapproved)
    digest = prepared["curation"]["task_spec_sha256"]
    assert digest == hashlib.sha256(render_task_spec(unapproved, instruction=ASSIGNMENT_TEXT)).hexdigest()
    assert CaseDocument.model_validate(prepared).curation.task_spec_sha256 == digest


def test_manifest_collects_independent_field_errors() -> None:
    """One malformed manifest must report each independent trust-boundary failure."""
    raw = _valid_manifest()
    valid = BenchmarkManifest.model_validate(raw)
    assert valid.privacy.reviewer_allowed_hosts == ["api.anthropic.com"]
    assert valid.privacy.judge_allowed_hosts == ["api.anthropic.com"]
    raw.update(bogus=True, benchmark_id="not-a-uuid", created_at="2026-08-21T12:00:00")
    raw["source"]["hostname"] = "gitlab.com"
    raw["privacy"].update(
        reviewer_allowed_hosts=[],
        classification="public",
        reviewer_data="everything",
        judge_data="full",
        archive="enabled",
        uploads="enabled",
    )

    with pytest.raises(ValidationError) as rejected:
        BenchmarkManifest.model_validate(raw)

    assert {tuple(error["loc"]) for error in rejected.value.errors()} == {
        ("bogus",),
        ("benchmark_id",),
        ("created_at",),
        ("source", "hostname"),
        ("privacy", "reviewer_allowed_hosts"),
        ("privacy", "classification"),
        ("privacy", "reviewer_data"),
        ("privacy", "judge_data"),
        ("privacy", "archive"),
        ("privacy", "uploads"),
    }


def test_case_document_validates_a_heterogeneous_finding_population(raw: dict[str, Any]) -> None:
    """Batch admission preserves valid limits and identifies every bad finding independently."""
    finding = raw["curation"]["findings"][0]
    population = [
        {**finding, "title": "界" * 500, "body": "é" * 4096, "severity": "high"},
        {**finding, "title": "Medium concern", "severity": "medium"},
        {**finding, "title": "Low concern", "severity": "low"},
        {**finding, "title": "Unranked concern", "severity": None},
    ]
    for entry in population:
        entry["finding_id"] = derive_finding_id(entry, case_id=raw["case_id"])
    raw["curation"]["findings"] = population
    admitted = CaseDocument.model_validate(raw).curation.findings
    assert [entry.severity for entry in admitted] == ["high", "medium", "low", None]
    raw["curation"]["findings"] = [{**population[0], "provenance": {"kind": "authored", "source_ids": []}},
                                   population[1]]
    assert schema.derive_gold_mode(CaseDocument.model_validate(raw).curation) == "mixed"
    assert (admitted[0].title, admitted[0].body) == ("界" * 500, "é" * 4096)

    raw["curation"]["findings"] = [
        finding | changes
        for changes in [
            {"title": "界" * 501},
            {"body": "é" * 4096 + "x"},
            {"title": "bad\x00title"},
            {"location": {"path": "/abs/path.py", "start_line": 2, "end_line": 2}},
            {"location": {"path": "src/cache.py", "start_line": 5, "end_line": 2}},
            {"severity": "critical"},
        ]
    ]
    with pytest.raises(ValidationError) as rejected:
        CaseDocument.model_validate(raw)
    assert {tuple(error["loc"]) for error in rejected.value.errors()} == {
        ("curation", "findings", 0, "title"),
        ("curation", "findings", 1, "body"),
        ("curation", "findings", 2, "title"),
        ("curation", "findings", 3, "location", "path"),
        ("curation", "findings", 4, "location"),
        ("curation", "findings", 5, "severity"),
    }
