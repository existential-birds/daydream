"""Tests for the stdlib-only harbor verifier core module."""

import pytest

from daydream.benchmark.harbor.verifier_core import (
    CandidateFinding,
    GoldFinding,
    Verdict,
    VerifierError,
    derive_candidate_id,
    parse_candidate_finding,
    parse_gold_finding,
    retained_edges,
    validate_candidate_artifact,
    validate_gold_set,
)


def _cand(**overrides):
    base = {
        "candidate_id": "a" * 64,
        "title": "Cache key not scoped",
        "body": "Cache key omits the auth realm",
        "severity": "high",
        "path": "src/cache.py",
        "start_line": 42,
        "end_line": 42,
    }
    base.update(overrides)
    return base


def _gold(**overrides):
    base = {
        "finding_id": "a" * 64,
        "title": "Cache key not scoped",
        "body": "Cache key omits the auth realm",
        "severity": "high",
        "path": "src/cache.py",
        "start_line": 42,
        "end_line": 42,
    }
    base.update(overrides)
    return base


def test_candidate_parses_valid():
    f = parse_candidate_finding(_cand())
    assert isinstance(f, CandidateFinding)
    assert (f.title, f.severity, f.path, f.start_line, f.end_line) == (
        "Cache key not scoped",
        "high",
        "src/cache.py",
        42,
        42,
    )


def test_gold_parses_valid():
    g = parse_gold_finding(_gold())
    assert isinstance(g, GoldFinding) and g.finding_id == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("severity", "critical"),
        ("title", "   "),
        ("body", ""),
        ("path", "../etc/passwd"),
        ("path", "/abs/path"),
        ("start_line", 5),
    ],
)
def test_candidate_rejects_invalid(field, value):
    kw = _cand()
    if field == "start_line":
        kw["start_line"], kw["end_line"] = 5, 3  # end < start
    else:
        kw[field] = value
    with pytest.raises(VerifierError):
        parse_candidate_finding(kw)


def test_candidate_rejects_nul_and_oversize_title():
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(title="a\x00b"))
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(title="x" * 501))
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(body="y" * (8 * 1024 + 1)))


def test_candidate_rejects_bad_id_hex():
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(candidate_id="xyz"))


def test_gold_rejects_bad_id_hex():
    with pytest.raises(VerifierError):
        parse_gold_finding(_gold(finding_id="not-hex"))


def test_harbor_package_imports_stdlib_only():
    import daydream.benchmark.harbor.verifier_core as vc

    assert vc.MAX_GOLD_FINDINGS == 50
    assert vc.MAX_CANDIDATE_FINDINGS == 100
    assert vc.MAX_ARTIFACT_BYTES == 1_048_576
    assert vc.CONFIDENCE_THRESHOLD == 0.7
    assert issubclass(vc.VerifierError, Exception)

def test_candidate_id_is_64_lower_hex():
    cid = derive_candidate_id("case-x", _cand(), 0)
    assert len(cid) == 64 and all(ch in "0123456789abcdef" for ch in cid)


def test_candidate_id_scoped_by_case_key():
    a = derive_candidate_id("case-x", _cand(), 0)
    b = derive_candidate_id("case-y", _cand(), 0)
    assert a != b


def test_duplicate_content_gets_distinct_ordinals_and_ids():
    # two byte-identical findings, ordinals 0 and 1 → distinct IDs, both preserved
    first = derive_candidate_id("case-x", _cand(), 0)
    second = derive_candidate_id("case-x", _cand(), 1)
    assert first != second


def test_same_content_same_key_same_ordinal_is_stable():
    assert derive_candidate_id("case-x", _cand(), 0) == derive_candidate_id("case-x", _cand(), 0)


def test_null_fields_normalize_to_empty_string():
    # title/body/severity null → "" — the digest must be stable for a null vs "" field
    raw = _cand(title=None, body=None, severity=None)
    cid = derive_candidate_id("case-x", raw, 0)
    raw2 = _cand(title="", body="", severity=None)
    assert cid == derive_candidate_id("case-x", raw2, 0)


def _artifact(findings, **overrides):
    base = {
        "schema_version": 1,
        "case_id": "case-x",
        "base_ref": "base",
        "head_ref": "head",
        "findings": findings,
    }
    base.update(overrides)
    return base


def _valid_findings(n=1, key="case-x"):
    out = []
    groups: dict[tuple, int] = {}
    for i in range(n):
        base = _cand(title=f"f{i}", body=f"body{i}")
        canon = (
            f"f{i}",
            f"body{i}",
            base.get("severity") or "",
            base["path"],
            base["start_line"],
            base["end_line"],
        )
        ordinal = groups.get(canon, 0)
        groups[canon] = ordinal + 1
        base["candidate_id"] = derive_candidate_id(key, base, ordinal)
        out.append(base)
    return out


def test_artifact_accepts_and_returns_models():
    fs = validate_candidate_artifact(_artifact(_valid_findings(2)))
    assert len(fs) == 2 and all(isinstance(f, CandidateFinding) for f in fs)


def test_artifact_rejects_duplicate_candidate_id():
    dup = _valid_findings(2)
    dup[1] = dict(dup[0])  # same id, different title → not allowed
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(dup))


def test_artifact_rejects_mismatched_candidate_id():
    bad = _valid_findings(1)
    bad[0]["candidate_id"] = "f" * 64  # does not match derived id
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(bad))


def test_artifact_rejects_wrong_schema_version():
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact([], schema_version=2))


def test_artifact_rejects_missing_refs():
    art = _artifact([])
    del art["head_ref"]
    with pytest.raises(VerifierError):
        validate_candidate_artifact(art)


def test_artifact_rejects_over_one_mib():
    big = _valid_findings(1)
    big[0]["body"] = "x" * (1_048_576)  # artifact JSON > 1 MiB
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(big))


def test_artifact_rejects_over_100_findings():
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(_valid_findings(101)))

def test_gold_set_accepts_and_returns_models():
    gs = validate_gold_set([_gold(), _gold(finding_id="b" * 64, title="B")])
    assert len(gs) == 2 and all(isinstance(g, GoldFinding) for g in gs)


def test_gold_set_rejects_over_50():
    many = [_gold(finding_id=(hex(i)[2:].zfill(64)), title=f"f{i}") for i in range(51)]
    with pytest.raises(VerifierError):
        validate_gold_set(many)


def test_gold_set_rejects_invalid_member():
    with pytest.raises(VerifierError):
        validate_gold_set([_gold(severity="nope")])

def test_verdict_parses_and_validates():
    v = Verdict(gold_id="g", candidate_id="c", match=True, confidence=0.9, reasoning="same bug")
    assert (v.match, v.confidence) == (True, 0.9)


def test_verdict_rejects_out_of_range_confidence():
    with pytest.raises(VerifierError):
        Verdict("g", "c", True, 1.5, "")


def test_retained_edges_threshold():
    vs = [
        Verdict("g1", "c1", True, 0.9, "m"),  # keep
        Verdict("g2", "c2", True, 0.69, "m"),  # drop: below 0.7
        Verdict("g3", "c3", False, 0.95, "m"),  # drop: match false
        Verdict("g4", "c4", True, 0.7, "m"),  # keep: exactly 0.7
    ]
    kept = retained_edges(vs, ["g1", "g2", "g3", "g4"], ["c1", "c2", "c3", "c4"])
    assert {(v.gold_id, v.candidate_id) for v in kept} == {("g1", "c1"), ("g4", "c4")}
