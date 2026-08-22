"""Tests for the stdlib-only harbor verifier core module."""

import pytest

from daydream.benchmark.harbor.verifier_core import (
    CandidateFinding,
    GoldFinding,
    Verdict,
    VerifierError,
    derive_candidate_id,
    maximum_matching,
    parse_candidate_finding,
    parse_gold_finding,
    retained_edges,
    score_review,
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

def test_max_cardinality_beats_greedy():
    # A-1 (0.9), A-2 (0.8), B-1 (0.7) — greedy only matches A-1; max-cardinality matches 2.
    vs = [Verdict("A", "1", True, 0.9, ""), Verdict("A", "2", True, 0.8, ""), Verdict("B", "1", True, 0.7, "")]
    m = maximum_matching(vs, ["A", "B"], ["1", "2"])
    assert m == {("A", "2"), ("B", "1")}


def test_one_candidate_cannot_match_two_golds():
    vs = [Verdict("g1", "c1", True, 0.9, ""), Verdict("g2", "c1", True, 0.8, "")]
    m = maximum_matching(vs, ["g1", "g2"], ["c1"])
    assert len(m) == 1
    assert {c for _, c in m} == {"c1"}  # c1 used at most once


def test_one_gold_cannot_match_two_candidates():
    vs = [Verdict("g1", "c1", True, 0.9, ""), Verdict("g1", "c2", True, 0.8, "")]
    m = maximum_matching(vs, ["g1"], ["c1", "c2"])
    assert len(m) == 1 and {g for g, _ in m} == {"g1"}


def test_matching_is_deterministic_across_runs():
    # ties on confidence: ordering must still be stable run-to-run
    vs = [Verdict("g1", "c1", True, 0.8, ""), Verdict("g1", "c2", True, 0.8, ""),
          Verdict("g2", "c1", True, 0.8, ""), Verdict("g2", "c2", True, 0.8, "")]
    gold, cand = ["g1", "g2"], ["c1", "c2"]
    r1 = maximum_matching(vs, gold, cand)
    r2 = maximum_matching(vs, gold, cand)
    assert r1 == r2 == {("g1", "c1"), ("g2", "c2")}

def test_score_clean_zero_candidates():
    r = score_review([], _artifact([]), [])
    d = r.to_dict()
    assert d["reward"] == 1.0 and d["clean_task"] == 1 and d["clean_pass"] == 1
    assert (d["tp"], d["fp"], d["fn"]) == (0, 0, 0)
    assert (d["precision"], d["recall"], d["f1"]) == (1.0, 1.0, 1.0)  # zero-denom → 1.0


def test_score_clean_with_candidates():
    art = _artifact(_valid_findings(2))
    r = score_review([], art, [])
    assert (r.reward, r.fp, r.clean_pass) == (0.0, 2, 0)
    assert r.tp == 0 and r.fn == 0 and r.clean_task == 1


def test_score_gold_no_candidates():
    gold = [_gold(), _gold(finding_id="b" * 64, title="B")]
    r = score_review(gold, _artifact([]), [])
    assert (r.reward, r.fn) == (0.0, 2)
    assert r.fp == 0 and r.clean_task == 0


def test_score_f1_example():
    # 3 gold / 2 candidates, TP=2, FN=1 → precision 1.0, recall 0.6666666667, f1 0.8
    gold = [_gold(), _gold(finding_id="b" * 64, title="B"), _gold(finding_id="c" * 64, title="C")]
    cands = _valid_findings(2)
    art = _artifact(cands)
    vs = [Verdict("a" * 64, cands[0]["candidate_id"], True, 0.9, "same"),
          Verdict("b" * 64, cands[1]["candidate_id"], True, 0.8, "same")]
    r = score_review(gold, art, vs)
    assert (r.tp, r.fp, r.fn) == (2, 0, 1)
    assert r.precision == 1.0
    assert abs(r.recall - 0.6666666667) < 1e-9
    assert abs(r.f1 - 0.8) < 1e-9 and abs(r.reward - 0.8) < 1e-9


def test_score_malformed_artifact_is_verifier_error():
    art = {"schema_version": 1, "case_id": "c", "base_ref": "b", "head_ref": "h",
           "findings": [{"candidate_id": "not-hex"}]}
    r = score_review([], art, [])
    assert r.reward == 0.0 and r.verifier_error == 1

def test_empty_side_resolves_with_zero_verdicts():
    # clean/0 and N/0 both resolve deterministically with an EMPTY verdict set
    r0 = score_review([], _artifact([]), [])
    assert r0.reward == 1.0
    rn = score_review([_gold()], _artifact([]), [])
    assert rn.reward == 0.0 and rn.fn == 1
    # passing any verdict for an empty side must not change the result (ignored/not required)
    assert score_review([], _artifact([]), [Verdict("g", "c", True, 0.9, "")]).reward == 1.0


def test_reward_dict_is_numeric_only_with_all_keys():
    d = score_review([_gold()], _artifact(_valid_findings(1)), []).to_dict()
    assert set(d) == {"reward", "tp", "fp", "fn", "precision", "recall", "f1",
                      "gold_count", "candidate_count", "clean_task", "clean_pass", "verifier_error"}
    for k, v in d.items():
        assert isinstance(v, (int, float)) and not isinstance(v, bool)
