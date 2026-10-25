import hashlib
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

import daydream.benchmark.schema as schema
from daydream import severity
from daydream.benchmark.harbor.build import (
    ASSIGNMENT_TEXT,
    render_task_spec,
    task_spec_approval,
    task_spec_digest,
)
from daydream.benchmark.schema import (
    BenchmarkManifest,
    CaseDocument,
    Curation,
    EvidenceRecord,
    Finding,
    ImportDocument,
    PrioritizationFacts,
    Provenance,
    PullRequestEntry,
    PullRequestMeta,
    Source,
    TransitionError,
    _ImportRepository,
    classify_validation,
    derive_finding_id,
    derive_gold_mode,
    normalize_hostname,
    validate_case_transition,
    validate_pr_transition,
)
from daydream.pr_review import finding_marker


def test_pyyaml_is_a_base_runtime_dependency() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    dev = data["dependency-groups"]["dev"]
    assert any(d == "pyyaml>=6.0" or d.startswith("pyyaml") for d in deps)
    assert any(d == "types-pyyaml>=6.0" or d.startswith("types-pyyaml") for d in dev), ("type stubs stay dev-only"
    )


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



def test_hostname_normalization_and_rejection(subtests: pytest.Subtests) -> None:
    """Operator host forms canonicalize consistently; wildcard/blank inputs fail closed."""
    valid = [
        "api.anthropic.com",
        "API.Anthropic.COM",
        "https://api.anthropic.com",
        "api.anthropic.com:443",
        "user:pass@api.anthropic.com",
        "api.anthropic.com/path",
    ]
    for raw in valid:
        with subtests.test(raw=raw):
            assert normalize_hostname(raw) == "api.anthropic.com"
    for raw in ("", "*.anthropic.com", "not a host", "http://"):
        with subtests.test(invalid=raw):
            with pytest.raises(ValueError):
                normalize_hostname(raw)


def test_manifest_rejects_invalid_fields(subtests: pytest.Subtests) -> None:
    cases: Sequence[tuple[str | None, str, Any]] = [
        (None, "bogus", True),
        ("source", "hostname", "gitlab.com"),
        ("privacy", "reviewer_allowed_hosts", []),
        (None, "benchmark_id", "not-a-uuid"),
        (None, "created_at", "2026-08-21T12:00:00"),
    ]
    for case_index, (section, field, value) in enumerate(cases):
        with subtests.test(section=section, field=field, value=value):
            base = _valid_manifest()
            target = base if section is None else base[section]
            target[field] = value
            with pytest.raises(ValidationError):
                BenchmarkManifest.model_validate(base)


def test_source_repository_id_is_nonblank_opaque_string() -> None:
    s = Source(provider="github", hostname="github.com", repository="o/r", repository_id="R_kgDOABC123")
    assert s.repository_id == "R_kgDOABC123"
    # None sentinel (unresolved) unchanged
    assert Source(provider="github", hostname="github.com", repository="o/r").repository_id is None
    # numeric-only and blank node ids are rejected as invalid
    for bad in ("5", "   ", "123456"):
        with pytest.raises(ValidationError):
            Source(provider="github", hostname="github.com", repository="o/r", repository_id=bad)

def test_import_repository_id_is_opaque_string() -> None:
    r = _ImportRepository(id="R_kgDOABC123", name_with_owner="o/r", visibility="private")
    assert r.id == "R_kgDOABC123"
    # numeric-only ids (Pydantic would otherwise coerce int->str) must not model;
    # blank "" is the deliberate unresolved sentinel from _repository_block.
    with pytest.raises(ValidationError):
        _ImportRepository(id="5", name_with_owner="o/r", visibility="private")
    with pytest.raises(ValidationError):
        _ImportRepository(id="123456", name_with_owner="o/r", visibility="private")
    assert _ImportRepository(id="", name_with_owner="o/r", visibility="private").id == ""

def test_privacy_classification_and_policies_are_literals() -> None:
    m = _valid_manifest()
    for key, bad in [("classification", "public"), ("reviewer_data", "everything"),
                     ("judge_data", "full"), ("archive", "enabled"), ("uploads", "enabled")]:
        raw = dict(m)
        raw["privacy"] = dict(m["privacy"])
        raw["privacy"][key] = bad
        with pytest.raises(ValidationError):
            BenchmarkManifest.model_validate(raw)


def test_case_snapshot_guards_bind_readiness_policy_and_revision(subtests: pytest.Subtests) -> None:
    """Each row starts from valid authored evidence and isolates one readiness/provenance guard."""
    valid = CaseDocument.model_validate(_valid_case_dict())
    assert valid.snapshot.status == "ready"
    assert valid.snapshot.requested_base_sha == "0123456789abcdef0123456789abcdef01234567"
    absent = object()
    mutations: list[tuple[str, str, object]] = [
        ("snapshot", "policy", "some_head"),
        ("snapshot", "base_resolution", absent),
        ("snapshot", "base_resolution", "invented"),
        ("snapshot", "bundle_file", absent),
        ("snapshot", "requested_base_sha", absent),
        ("snapshot", "requested_base_sha", "c" * 40),
        ("curation", "snapshot_attested", False),
    ]
    for section, field, value in mutations:
        with subtests.test(section=section, field=field, value=value):
            raw = _valid_case_dict()
            if value is absent:
                raw[section].pop(field)
            else:
                raw[section][field] = value
            with pytest.raises(ValidationError):
                CaseDocument.model_validate(raw)
    for corruption in ("stale-attested", "unreplayable-with-ready-fields"):
        with subtests.test(corruption=corruption):
            raw = _valid_case_dict()
            if corruption == "stale-attested":
                raw["curation"].update(state="stale", snapshot_attested=True, gold_status=None, clean_attested=False)
            else:
                raw["snapshot"].update(
                    status="unreplayable",
                    error={"reason": "equal_trees", "detail": "no change"},
                    bundle_file="snapshots/pr-000101-0123456789ab.bundle",
                )
            with pytest.raises(ValidationError):
                CaseDocument.model_validate(raw)


def test_reviewer_judge_hosts_stay_lists() -> None:
    m = BenchmarkManifest.model_validate(_valid_manifest())
    assert m.privacy.reviewer_allowed_hosts == ["api.anthropic.com"]
    assert m.privacy.judge_allowed_hosts == ["api.anthropic.com"]

def test_ledger_entry_valid_and_fetch_failed_error_shape() -> None:
    ok = PullRequestEntry(number=101, import_state="pending", requested_heads=["final"], case_ids=[])
    assert ok.import_file is None and ok.import_sha256 is None and ok.error is None

    failed = PullRequestEntry(number=102, import_state="fetch_failed", requested_heads=["final"], case_ids=[],
        error={"code": "E_AUTH", "message": "no access"},
    )
    assert failed.error is not None
    assert failed.error["code"] == "E_AUTH"


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


def _valid_import_document() -> dict[str, Any]:
    return {
        "schema_version": 1, "repository": {"id": "R_kgDOABC123", "name_with_owner": "o/r", "visibility": "private"},
        "pull_request": {
            "number": 101, "url": "https://github.com/o/r/pull/101", "html_url": "https://github.com/o/r/pull/101",
            "title": "Fix cache", "body": "fixes the cache", "state": "open",
            "title_sha256": hashlib.sha256(b"Fix cache").hexdigest(),
            "body_sha256": hashlib.sha256("fixes the cache".encode()).hexdigest(),
            "base": {"ref": "main", "sha": "0123456789abcdef0123456789abcdef01234567"},
            "head": {"ref": "feature/cache", "sha": "h" * 40}, "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z", "merged_at": None, "closed_at": None,
            "author": {"login": "alice", "type": "User"},
        }, "evidence": [_evidence("inline_comment", 7)],
        "fetch": {"fetched_at": "2026-01-01T00:00:00Z", "etag": None, "payload_sha256": "a" * 64},
    }



def test_import_and_case_models_require_strict_nested_shapes(subtests: pytest.Subtests) -> None:
    """Top-level and nested extra keys stay forbidden; required PR metadata cannot disappear."""
    assert ImportDocument.model_validate(_valid_import_document()).pull_request.number == 101
    for kind, section, missing in (
        ("import", None, False),
        ("import", "pull_request", False),
        ("case", "pull_request", True),
        ("case", "pull_request", False),
        ("case", "source", False),
    ):
        with subtests.test(kind=kind, section=section, missing=missing):
            raw = _valid_import_document() if kind == "import" else _valid_case_dict()
            target = raw if section is None else raw[section]
            if missing:
                target.pop("author")
            else:
                target["bogus"] = 1
            with pytest.raises(ValidationError) as rejected:
                if kind == "import":
                    ImportDocument.model_validate(raw)
                else:
                    CaseDocument.model_validate(raw)
            if section is not None:
                assert rejected.value.errors()[0]["loc"][0] == section


def test_pull_request_meta_predate_reads_empty_and_validates() -> None:
    # predate import: lacks the additive body/digest/html_url/merged/closed fields
    m = PullRequestMeta.model_validate({
        "number": 101, "url": "https://github.com/o/r/pull/101", "title": "Fix cache", "state": "open",
        "base": {"ref": "main", "sha": "b" * 40},
        "head": {"sha": "a" * 40},                 # no head.ref (old import dropped it)
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "author": {"login": "alice", "type": "User"},
    })
    assert m.body == "" and m.title_sha256 == "" and m.body_sha256 == ""
    assert m.merged_at is None and m.closed_at is None and m.html_url == ""
    assert m.head.ref is None

def test_pull_request_changed_files_is_optional_but_canonical_when_present() -> None:
    raw = {
        "number": 101, "url": "u", "title": "t", "state": "open", "base": {"sha": "b" * 40}, "head": {"sha": "a" * 40},
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "author": {"login": "a", "type": "User"},
    }
    assert PullRequestMeta.model_validate(raw).changed_files is None
    assert PullRequestMeta.model_validate({**raw, "changed_files": []}).changed_files == []
    paths = ["a:b.py", r"dir\literal.py", "src/file with spaces.py"]
    assert PullRequestMeta.model_validate({**raw, "changed_files": paths}).changed_files == paths

def test_pull_request_changed_files_rejects_noncanonical_paths(subtests: pytest.Subtests) -> None:
    cases: Sequence[list[Any]] = [
        ["b.py", "a.py"],
        ["a.py", "a.py"],
        [""],
        ["/absolute.py"],
        ["../escape.py"],
        ["src/../escape.py"],
        ["src//empty.py"],
        ["nul\x00.py"],
        [1],
    ]
    for case_index, paths in enumerate(cases):
        with subtests.test(paths=paths):
            raw = {
                "number": 101,
                "url": "u",
                "title": "t",
                "state": "open",
                "base": {"sha": "b" * 40},
                "head": {"sha": "a" * 40},
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "author": {"login": "a", "type": "User"},
                "changed_files": paths,
            }
            with pytest.raises(ValidationError):
                PullRequestMeta.model_validate(raw)


def test_pull_request_meta_fails_closed_on_malformed_required() -> None:
    base = {
        "number": 101, "url": "u", "title": "t", "state": "open", "base": {"sha": "b" * 40}, "head": {"sha": "a" * 40},
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


def _valid_case() -> Any:
    return CaseDocument.model_validate(_valid_case_dict())


def _unreplayable_snapshot(
    *, reason: str, detail: str, policy: str = "final_pr_head", requested_head: str = "final",
    original_base_sha: str | None = None, requested_base_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "unreplayable", "policy": policy, "requested_head": requested_head,
        "original_base_sha": original_base_sha, "requested_base_sha": requested_base_sha,
        "original_head_sha": "0123456789abcdef0123456789abcdef01234567",
        "base_tree_sha": None, "head_tree_sha": None, "diff_sha256": None, "bundle_file": None, "bundle_sha256": None,
        "error": {"reason": reason, "detail": detail},
    }



def test_base_drift_is_the_only_new_snapshot_reason(raw: dict[str, Any]) -> None:
    raw["snapshot"] = _unreplayable_snapshot(
        policy="explicit_head", requested_head="a" * 40, original_base_sha="b" * 40,
        requested_base_sha="0123456789abcdef0123456789abcdef01234567",
        reason="base_drift", detail="one extra path",
    )
    raw["curation"].update({"state": "unreplayable", "snapshot_attested": False,
        "clean_attested": False, "gold_status": None, "findings": [], "task_spec_sha256": None,
    })
    assert CaseDocument.model_validate(raw).snapshot.status == "unreplayable"
    raw["snapshot"]["error"]["reason"] = "invented_reason"
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)






def test_unreplayable_snapshot_and_curation_states_must_agree(subtests: pytest.Subtests) -> None:
    """Unreplayable evidence cannot be ready, while explicit exclusion remains valid."""
    for starting_state in ("ready-snapshot", "unreplayable-snapshot"):
        with subtests.test(starting_state=starting_state):
            raw = _valid_case_dict()
            if starting_state == "ready-snapshot":
                raw["curation"].update(
                    state="unreplayable", snapshot_attested=False, clean_attested=False, gold_status=None, findings=[]
                )
                with pytest.raises(ValidationError):
                    CaseDocument.model_validate(raw)
                raw["snapshot"] = _unreplayable_snapshot(reason="head_not_on_pr", detail="head sha not on PR")
                parsed = CaseDocument.model_validate(raw)
                assert parsed.curation.state == "unreplayable" and parsed.snapshot.status == "unreplayable"
            else:
                raw["snapshot"] = _unreplayable_snapshot(
                    policy="explicit_head",
                    requested_head="a" * 40,
                    original_base_sha="b" * 40,
                    requested_base_sha="0123456789abcdef0123456789abcdef01234567",
                    reason="head_not_on_pr",
                    detail="not on PR",
                )
                with pytest.raises(ValidationError):
                    CaseDocument.model_validate(raw)
                raw["curation"].update(
                    state="excluded",
                    snapshot_attested=False,
                    clean_attested=False,
                    gold_status=None,
                    findings=[],
                    task_spec_sha256=None,
                    case_exclusion={"reason": "unreplayable", "note": None},
                )
                assert CaseDocument.model_validate(raw).curation.state == "excluded"
    raw = _valid_case_dict()
    raw["snapshot"] = _unreplayable_snapshot(reason="head_not_on_pr", detail="head sha not on PR")
    raw["curation"].update(
        state="unreplayable",
        snapshot_attested=False,
        clean_attested=False,
        gold_status=None,
        findings=[],
        task_spec_sha256=None,
    )
    assert CaseDocument.model_validate(raw).snapshot.status == "unreplayable"


def test_case_findings_reject_invalid_content_locations_and_identity(subtests: pytest.Subtests) -> None:
    """Authoring guards bound content, location and canonical identity independently."""
    mutations: list[tuple[str, object]] = [
        ("title", "x" * 501),
        ("body", "y" * (8 * 1024 + 1)),
        ("title", "bad\x00title"),
        ("location", {"path": "/abs/path.py", "start_line": 2, "end_line": 2}),
        ("location", {"path": "src/cache.py", "start_line": 5, "end_line": 2}),
        ("finding_id", "0" * 64),
    ]
    for field, value in mutations:
        with subtests.test(field=field, value=value):
            raw = _valid_case_dict()
            raw["curation"]["findings"][0][field] = value
            with pytest.raises(ValidationError) as rejected:
                CaseDocument.model_validate(raw)
            if field == "finding_id":
                assert "finding_id" in str(rejected.value)


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

def test_finding_severity_accepts_every_canonical_level_and_rejects_unknown() -> None:
    for level in severity.CANONICAL_LEVELS:
        raw = _valid_case_dict()
        finding = raw["curation"]["findings"][0]
        finding["severity"] = level
        finding["finding_id"] = derive_finding_id(finding, case_id=raw["case_id"])
        assert CaseDocument.model_validate(raw).curation.findings[0].severity == level

    raw = _valid_case_dict()
    finding = raw["curation"]["findings"][0]
    finding["severity"] = "critical"
    finding["finding_id"] = derive_finding_id(finding, case_id=raw["case_id"])
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)

def test_finding_severity_declares_no_inline_level_literal() -> None:
    """A second inline literal cannot follow a declaration change, so the field must
    be typed by the shared vocabulary alias (spec requirement 4)."""
    source = (Path(__file__).resolve().parents[1] / "daydream" / "benchmark" / "schema.py").read_text()
    assert "SeverityLevel" in source
    # Whitespace/quote-insensitive so an alternate spelling or re-wrap of the inline
    # literal cannot evade the guard.
    inline = re.search(r'Literal\s*\[\s*["\']high["\']\s*,\s*["\']medium["\']\s*,\s*["\']low["\']\s*\]', source)
    assert inline is None
    # And the resolved annotation is bound to the shared declaration, not a copy.
    annotation = Finding.model_fields["severity"].annotation
    literal = next(arg for arg in get_args(annotation) if get_args(arg))
    assert get_args(literal) == get_args(severity.SeverityLevel)



def test_case_finding_identity_is_case_scoped_unique_and_legacy_compatible() -> None:
    """Current IDs are canonical per case; legacy case documents retain their historical IDs."""
    finding = _valid_case().curation.findings[0]
    expected = derive_finding_id(finding, case_id="pr-000101-0123456789ab")
    assert finding.finding_id == expected
    assert expected != derive_finding_id(finding, case_id="pr-000102-0123456789ab")
    raw = _valid_case_dict()
    raw["curation"]["findings"].append(raw["curation"]["findings"][0])
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)
    legacy = _valid_case_dict()
    legacy["schema_version"] = 1
    legacy["curation"]["findings"][0]["finding_id"] = "e" * 64
    assert CaseDocument.model_validate(legacy).schema_version == 1




def test_historical_daydream_marker_cannot_be_gold(raw: dict[str, Any]) -> None:
    body = "looks fine"
    finding = {
        "title": "Daydream self-output", "body": body + finding_marker("a" * 64), "severity": None, "location": None,
        "provenance": {"kind": "historical", "source_ids": ["github:review_comment:1"]},
    }
    # canonical, so the marker guard is what rejects
    finding["finding_id"] = derive_finding_id(finding, case_id=raw["case_id"])
    raw["curation"]["findings"][0] = finding
    with pytest.raises(ValidationError):
        CaseDocument.model_validate(raw)

def test_legacy_ready_without_task_spec_digest_backfills_and_validates(raw: dict[str, Any]) -> None:
    del raw["curation"]["task_spec_sha256"]                # legacy pre-approval workspace
    prepared = schema._schema_ready(raw)
    digest = prepared["curation"]["task_spec_sha256"]
    expected = hashlib.sha256(render_task_spec(raw, instruction=ASSIGNMENT_TEXT)).hexdigest()
    assert digest == expected
    assert CaseDocument.model_validate(prepared).curation.task_spec_sha256 == digest

def test_task_spec_approval_reports_nonready_current_and_stale(raw: dict[str, Any]) -> None:
    raw["curation"]["state"] = "draft"
    assert task_spec_approval(raw).state == "not-required"
    raw = _valid_case_dict()
    digest = task_spec_digest(raw)
    raw["curation"]["task_spec_sha256"] = digest
    assert task_spec_approval(raw).state == "current"
    raw["curation"]["task_spec_sha256"] = "0" * 64
    approval = task_spec_approval(raw)
    assert approval.state == "stale"
    assert approval.current_sha256 == digest
    assert approval.approved_sha256 == "0" * 64

def test_gold_mode_truth_table(subtests: pytest.Subtests) -> None:
    cases: Sequence[tuple[Any, Any]] = [
        ([], "clean"),
        (["historical"], "historical"),
        (["edited"], "historical"),  # all-edited historical evidence stays historical
        (["authored"], "authored"),
        (["historical", "edited"], "historical"),
        (["authored", "historical"], "mixed"),
        (["authored", "edited"], "mixed"),
    ]
    for case_index, (kinds, expected) in enumerate(cases):
        with subtests.test(kinds=kinds, expected=expected):
            findings = [
                Finding(
                    finding_id="e" * 64,
                    title=f"f{i}",
                    body="b",
                    provenance=Provenance(
                        kind=k,
                        source_ids=["github:review_comment:1"] if k in ("historical", "edited") else [],
                    ),
                )
                for i, k in enumerate(kinds)
            ]
            curation = Curation(state="draft", findings=findings)
            assert derive_gold_mode(curation) == expected


def test_state_transitions_admit_only_the_declared_workflow_edges(subtests: pytest.Subtests) -> None:
    """Refresh preserves fetched linkage; case states cannot bypass required curation steps."""
    pr_rows = [
        ("pending", "fetched", True),
        ("pending", "fetch_failed", True),
        ("fetch_failed", "fetched", True),
        ("fetch_failed", "fetch_failed", True),
        ("fetched", "fetched", True),
        ("fetched", "pending", False),
        ("fetched", "fetch_failed", False),
        ("pending", "draft", False),
    ]
    for before, after, allowed in pr_rows:
        with subtests.test(entity="pull-request", before=before, after=after):
            if allowed:
                validate_pr_transition(before, after)
            else:
                with pytest.raises(TransitionError):
                    validate_pr_transition(before, after)
    case_rows = [
        ("draft", "ready", True),
        ("draft", "excluded", True),
        ("draft", "unreplayable", True),
        ("ready", "stale", True),
        ("ready", "draft", True),
        ("stale", "ready", True),
        ("stale", "excluded", True),
        ("unreplayable", "excluded", True),
        ("excluded", "draft", True),
        ("excluded", "unreplayable", True),
        ("ready", "excluded", False),
        ("stale", "draft", False),
        ("draft", "stale", False),
    ]
    for before, after, allowed in case_rows:
        with subtests.test(entity="case", before=before, after=after):
            if allowed:
                validate_case_transition(before, after)
            else:
                with pytest.raises(TransitionError):
                    validate_case_transition(before, after)


def test_classify_validation_codes() -> None:
    assert classify_validation(ready=True, corrupt=False) == 0
    assert classify_validation(ready=False, corrupt=False) == 2
    assert classify_validation(ready=False, corrupt=True) == 1

def test_case_document_accepts_additive_prioritization_key(raw: dict[str, Any]) -> None:
    """Spike pin (task 0, plan #879): the new additive ``prioritization`` key
    loads under the strict CaseDocument schema (schema_version stays 2) and
    round-trips. If this ever fails in a way an optional-field default cannot
    fix, the additive-under-v2 Key Decision is unsound and must be re-routed to
    the spec before any prioritization task runs."""
    assert "prioritization" not in raw
    doc = CaseDocument.model_validate(raw)
    assert doc.prioritization is None  # absence tolerated (old case docs load)

    facts = {"extraction_version": 1, "head_sha": "a" * 40,
        "candidates": {"github:review_comment:1": {"commit_relation": "at_head", "anchor_delta": "unchanged"}},
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
    facts = PrioritizationFacts.model_validate({"extraction_version": 1, "head_sha": "a" * 40,
            "candidates": {"gh:101:c1": {"commit_relation": "at_head", "anchor_delta": "changed"}},
            "non_candidates": {},
        }
    )
    assert facts.extraction_version == 1
    assert facts.candidates["gh:101:c1"].commit_relation == "at_head"
    # additive under v2: an absent key still loads as None
    doc = CaseDocument.model_validate(_valid_case_dict())
    assert doc.prioritization is None

def test_case_document_tolerates_absent_and_rejects_bad_prioritization(raw: dict[str, Any]) -> None:
    doc = CaseDocument.model_validate(raw)
    assert doc.prioritization is None  # absent key -> None, old docs load

    raw["prioritization"] = {"extraction_version": 1, "head_sha": "a" * 40,
        "candidates": {"x": {"commit_relation": "wat", "anchor_delta": "unchanged"}},
    }
    with pytest.raises(ValidationError):  # unknown enum value is a schema violation
        CaseDocument.model_validate(raw)

def test_pull_request_entry_fetched_allows_latest_error_only() -> None:
    # A fetched entry may carry latest_error (a failed refresh attempt on an
    # otherwise-intact import) but never `error` itself.
    base_entry = {"number": 101, "import_state": "pending", "requested_heads": ["final"], "case_ids": []}
    valid = dict(base_entry, import_state="fetched", import_file="imports/pr-000101.json", import_sha256="a" * 64,
        latest_error={"code": "fetch", "message": "x"},
    )
    PullRequestEntry.model_validate(valid)                       # OK
    bad = dict(valid, error={"code": "fetch", "message": "x"})   # error still forbidden on fetched
    with pytest.raises(ValidationError):
        PullRequestEntry.model_validate(bad)
