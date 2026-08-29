"""Tests for the stdlib-only harbor verifier core module."""
import hashlib
import json
from typing import Any

import pytest

from daydream.benchmark.harbor import verifier_core as vc
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
    reward_details,
    reward_details_to_json,
    reward_to_json,
    score_review,
    validate_candidate_artifact,
    validate_gold_set,
)


def _cand(**overrides: Any) -> Any:
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


def _gold(**overrides: Any) -> Any:
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


def test_candidate_parses_valid() -> None:
    f = parse_candidate_finding(_cand())
    assert isinstance(f, CandidateFinding)
    assert (f.title, f.severity, f.path, f.start_line, f.end_line) == (
        "Cache key not scoped",
        "high",
        "src/cache.py",
        42,
        42,
    )


def test_gold_parses_valid() -> None:
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
def test_candidate_rejects_invalid(field: str, value: Any) -> None:
    kw = _cand()
    if field == "start_line":
        kw["start_line"], kw["end_line"] = 5, 3  # end < start
    else:
        kw[field] = value
    with pytest.raises(VerifierError):
        parse_candidate_finding(kw)


def test_candidate_rejects_nul_and_oversize_title() -> None:
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(title="a\x00b"))
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(title="x" * 501))
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(body="y" * (8 * 1024 + 1)))


def test_candidate_rejects_bad_id_hex() -> None:
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(candidate_id="xyz"))


def test_gold_rejects_bad_id_hex() -> None:
    with pytest.raises(VerifierError):
        parse_gold_finding(_gold(finding_id="not-hex"))


def test_harbor_package_imports_stdlib_only() -> None:
    import daydream.benchmark.harbor.verifier_core as vc

    assert vc.MAX_GOLD_FINDINGS == 50
    assert vc.MAX_CANDIDATE_FINDINGS == 100
    assert vc.MAX_ARTIFACT_BYTES == 1_048_576
    assert vc.CONFIDENCE_THRESHOLD == 0.7
    assert issubclass(vc.VerifierError, Exception)

def test_candidate_id_is_64_lower_hex() -> None:
    cid = derive_candidate_id("case-x", _cand(), 0)
    assert len(cid) == 64 and all(ch in "0123456789abcdef" for ch in cid)


def test_candidate_id_scoped_by_case_key() -> None:
    a = derive_candidate_id("case-x", _cand(), 0)
    b = derive_candidate_id("case-y", _cand(), 0)
    assert a != b


def test_duplicate_content_gets_distinct_ordinals_and_ids() -> None:
    # two byte-identical findings, ordinals 0 and 1 → distinct IDs, both preserved
    first = derive_candidate_id("case-x", _cand(), 0)
    second = derive_candidate_id("case-x", _cand(), 1)
    assert first != second


def test_same_content_same_key_same_ordinal_is_stable() -> None:
    assert derive_candidate_id("case-x", _cand(), 0) == derive_candidate_id("case-x", _cand(), 0)


def test_null_fields_normalize_to_empty_string() -> None:
    # title/body/severity null → "" — the digest must be stable for a null vs "" field
    raw = _cand(title=None, body=None, severity=None)
    cid = derive_candidate_id("case-x", raw, 0)
    raw2 = _cand(title="", body="", severity=None)
    assert cid == derive_candidate_id("case-x", raw2, 0)


def _artifact(findings: Any, **overrides: Any) -> Any:
    base = {
        "schema_version": 1,
        "case_id": "case-x",
        "base_ref": "base",
        "head_ref": "head",
        "findings": findings,
    }
    base.update(overrides)
    return base


def _valid_findings(n: Any=1, key: Any="case-x") -> Any:
    out = []
    groups: dict[tuple[Any, ...], int] = {}
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


def test_artifact_accepts_and_returns_models() -> None:
    fs = validate_candidate_artifact(_artifact(_valid_findings(2)))
    assert len(fs) == 2 and all(isinstance(f, CandidateFinding) for f in fs)


def test_artifact_rejects_duplicate_candidate_id() -> None:
    dup = _valid_findings(2)
    dup[1] = dict(dup[0])  # same id, different title → not allowed
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(dup))


def test_artifact_rejects_mismatched_candidate_id() -> None:
    bad = _valid_findings(1)
    bad[0]["candidate_id"] = "f" * 64  # does not match derived id
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(bad))


def test_artifact_rejects_wrong_schema_version() -> None:
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact([], schema_version=2))


def test_artifact_rejects_missing_refs() -> None:
    art = _artifact([])
    del art["head_ref"]
    with pytest.raises(VerifierError):
        validate_candidate_artifact(art)


def test_artifact_rejects_over_one_mib() -> None:
    big = _valid_findings(1)
    big[0]["body"] = "x" * (1_048_576)  # artifact JSON > 1 MiB
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(big))


def test_artifact_rejects_over_100_findings() -> None:
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(_valid_findings(101)))

def test_gold_set_accepts_and_returns_models() -> None:
    g1 = _gold()
    g1["finding_id"] = _canonical_gold_id("case-x", g1)
    g2 = _gold(title="B")
    g2["finding_id"] = _canonical_gold_id("case-x", g2)
    gs = validate_gold_set([g1, g2], case_id="case-x")
    assert len(gs) == 2 and all(isinstance(g, GoldFinding) for g in gs)


def test_gold_finding_id_is_case_scoped() -> None:
    # gold finding_id must equal sha256(case_id, title, body, severity, path, start, end)
    g = _gold()
    g["finding_id"] = _canonical_gold_id("case-x", g)
    validate_gold_set([g], case_id="case-x")                       # case-scoped digest accepted
    legacy = _gold()
    legacy["finding_id"] = "f" * 64  # valid 64-hex, case_id-less digest -> rejected
    with pytest.raises(VerifierError):
        validate_gold_set([legacy], case_id="case-x")
    # a case-scoped digest under a different case_id is rejected
    g2 = _gold()
    g2["finding_id"] = _canonical_gold_id("case-y", g2)
    with pytest.raises(VerifierError):
        validate_gold_set([g2], case_id="case-x")


def test_gold_set_rejects_over_50() -> None:
    many = []
    for i in range(51):
        f = _gold(title=f"f{i}")
        f["finding_id"] = _canonical_gold_id("case-x", f)
        many.append(f)
    with pytest.raises(VerifierError):
        validate_gold_set(many, case_id="case-x")


def test_gold_set_rejects_invalid_member() -> None:
    with pytest.raises(VerifierError):
        validate_gold_set([_gold(severity="nope")], case_id="case-x")

def test_verdict_parses_and_validates() -> None:
    v = Verdict(gold_id="g", candidate_id="c", match=True, confidence=0.9, reasoning="same bug")
    assert (v.match, v.confidence) == (True, 0.9)


def test_verdict_rejects_out_of_range_confidence() -> None:
    with pytest.raises(VerifierError):
        Verdict("g", "c", True, 1.5, "")


def test_retained_edges_threshold() -> None:
    vs = [
        Verdict("g1", "c1", True, 0.9, "m"),  # keep
        Verdict("g2", "c2", True, 0.69, "m"),  # drop: below 0.7
        Verdict("g3", "c3", False, 0.95, "m"),  # drop: match false
        Verdict("g4", "c4", True, 0.7, "m"),  # keep: exactly 0.7
    ]
    kept = retained_edges(vs, ["g1", "g2", "g3", "g4"], ["c1", "c2", "c3", "c4"])
    assert {(v.gold_id, v.candidate_id) for v in kept} == {("g1", "c1"), ("g4", "c4")}

def test_max_cardinality_beats_greedy() -> None:
    # A-1 (0.9), A-2 (0.8), B-1 (0.7) — greedy only matches A-1; max-cardinality matches 2.
    vs = [Verdict("A", "1", True, 0.9, ""), Verdict("A", "2", True, 0.8, ""), Verdict("B", "1", True, 0.7, "")]
    m = maximum_matching(vs, ["A", "B"], ["1", "2"])
    assert m == {("A", "2"), ("B", "1")}


def test_one_candidate_cannot_match_two_golds() -> None:
    vs = [Verdict("g1", "c1", True, 0.9, ""), Verdict("g2", "c1", True, 0.8, "")]
    m = maximum_matching(vs, ["g1", "g2"], ["c1"])
    assert len(m) == 1
    assert {c for _, c in m} == {"c1"}  # c1 used at most once


def test_one_gold_cannot_match_two_candidates() -> None:
    vs = [Verdict("g1", "c1", True, 0.9, ""), Verdict("g1", "c2", True, 0.8, "")]
    m = maximum_matching(vs, ["g1"], ["c1", "c2"])
    assert len(m) == 1 and {g for g, _ in m} == {"g1"}


def test_matching_is_deterministic_across_runs() -> None:
    # ties on confidence: ordering must still be stable run-to-run
    vs = [Verdict("g1", "c1", True, 0.8, ""), Verdict("g1", "c2", True, 0.8, ""),
          Verdict("g2", "c1", True, 0.8, ""), Verdict("g2", "c2", True, 0.8, "")]
    gold, cand = ["g1", "g2"], ["c1", "c2"]
    r1 = maximum_matching(vs, gold, cand)
    r2 = maximum_matching(vs, gold, cand)
    assert r1 == r2 == {("g1", "c1"), ("g2", "c2")}

def test_score_clean_zero_candidates() -> None:
    r = score_review([], _artifact([]), [])
    d = r.to_dict()
    assert d["reward"] == 1.0 and d["clean_task"] == 1 and d["clean_pass"] == 1
    assert (d["tp"], d["fp"], d["fn"]) == (0, 0, 0)
    assert (d["precision"], d["recall"], d["f1"]) == (1.0, 1.0, 1.0)  # zero-denom → 1.0


def test_score_clean_with_candidates() -> None:
    art = _artifact(_valid_findings(2))
    r = score_review([], art, [])
    assert (r.reward, r.fp, r.clean_pass) == (0.0, 2, 0)
    assert r.tp == 0 and r.fn == 0 and r.clean_task == 1


def test_score_gold_no_candidates() -> None:
    gold = [_gold(), _gold(finding_id="b" * 64, title="B")]
    r = score_review(gold, _artifact([]), [])
    assert (r.reward, r.fn) == (0.0, 2)
    assert r.fp == 0 and r.clean_task == 0


def test_score_zero_match_reward_is_zero() -> None:
    # nonempty gold and nonempty candidates with zero matching edges → f1/reward 0.0
    gold = [_gold(), _gold(finding_id="b" * 64, title="B")]
    art = _artifact(_valid_findings(2))
    r = score_review(gold, art, [])
    assert (r.tp, r.fp, r.fn) == (0, 2, 2)
    assert r.precision == 0.0 and r.recall == 0.0
    assert r.f1 == 0.0 and r.reward == 0.0


def test_score_f1_example() -> None:
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


def test_score_malformed_artifact_is_scored_zero() -> None:
    art = {"schema_version": 1, "case_id": "c", "base_ref": "b", "head_ref": "h",
           "findings": [{"candidate_id": "not-hex"}]}
    r = score_review([], art, [])
    assert r.reward == 0.0 and r.verifier_error == 0   # invalid agent output is scored zero

def test_empty_side_resolves_with_zero_verdicts() -> None:
    # clean/0 and N/0 both resolve deterministically with an EMPTY verdict set
    r0 = score_review([], _artifact([]), [])
    assert r0.reward == 1.0
    rn = score_review([_gold()], _artifact([]), [])
    assert rn.reward == 0.0 and rn.fn == 1
    # passing any verdict for an empty side must not change the result (ignored/not required)
    assert score_review([], _artifact([]), [Verdict("g", "c", True, 0.9, "")]).reward == 1.0


def test_reward_dict_is_numeric_only_with_all_keys() -> None:
    d = score_review([_gold()], _artifact(_valid_findings(1)), []).to_dict()
    assert set(d) == EXPECTED_24_KEYS
    for k, v in d.items():
        assert isinstance(v, (int, float)) and not isinstance(v, bool)

def test_reward_to_json_is_numeric_only() -> None:
    art = _artifact(_valid_findings(1))
    cand = _valid_findings(1)[0]
    r = score_review([_gold()], art, [Verdict("a" * 64, cand["candidate_id"], True, 0.9, "same")])
    d = json.loads(reward_to_json(r))
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in d.values())


def test_reward_details_shape_and_no_source_leak() -> None:
    gold = [_gold(), _gold(finding_id="b" * 64, title="B")]
    cands = [
        parse_candidate_finding(_cand(candidate_id="c" * 64, title="f1")),
        parse_candidate_finding(_cand(candidate_id="d" * 64, title="f2")),
    ]
    vs = [Verdict("a" * 64, cands[0].candidate_id, True, 0.9, "same bug")]
    m = {("a" * 64, cands[0].candidate_id)}
    details = reward_details(gold, cands, vs, m)
    for key in ("verdicts", "matches", "unmatched_gold", "unmatched_candidates"):
        assert key in details
    assert details["unmatched_gold"] == ["b" * 64]
    assert details["unmatched_candidates"] == [cands[1].candidate_id]
    blob = reward_details_to_json(details)
    assert "same bug" in blob  # reasoning is kept
    assert "Cache key not scoped" not in blob  # finding body/title never leaks
    assert "f1" not in blob  # candidate content never leaks


def test_gold_finding_id_rejects_non_string() -> None:
    # fail-closed: a raw-dict gold whose finding_id is not a str raises
    # (reachable via _finding_id on the raw-dict path)
    gold = [{"finding_id": 123}]
    with pytest.raises(VerifierError, match="must be a string"):
        reward_details(gold, [], [], set())


def test_candidate_id_rejects_non_string() -> None:
    # fail-closed: reward_details rejects a raw candidate whose candidate_id is
    # not a str (reachable via _candidate_id on the raw-dict path)
    cands = [{"candidate_id": 456}]
    with pytest.raises(VerifierError, match="must be a string"):
        reward_details([_gold()], cands, [], set())


def test_artifact_rejects_unknown_top_level_key() -> None:
    art = _artifact(_valid_findings(1))
    art["smuggled"] = "x"
    with pytest.raises(VerifierError):
        validate_candidate_artifact(art)


def test_artifact_rejects_unknown_finding_key() -> None:
    fs = _valid_findings(1)
    fs[0]["smuggled"] = "x"
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(fs))


def test_gold_set_rejects_unknown_finding_key() -> None:
    g = _gold()
    g["provenance"] = {"kind": "authored", "source_ids": []}
    with pytest.raises(VerifierError):
        validate_gold_set([g], case_id="case-x")


def _canonical_gold_id(case_id: str, f: dict[str, Any]) -> str:
    payload = "\x1f".join([
        str(case_id or ""), str(f.get("title") or ""), str(f.get("body") or ""),
        str(f.get("severity") or ""), str(f.get("path") or ""),
        str(f.get("start_line") or ""), str(f.get("end_line") or ""),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_gold_set_rejects_non_canonical_finding_id() -> None:
    f = _gold()
    f["finding_id"] = "f" * 64  # valid 64-hex but not the canonical digest
    with pytest.raises(VerifierError):
        validate_gold_set([f], case_id="case-x")


def test_gold_set_rejects_duplicate_finding_ids() -> None:
    fid = _canonical_gold_id("case-x", _gold())
    with pytest.raises(VerifierError):
        validate_gold_set([
            {**_gold(), "finding_id": fid},
            {**_gold(), "finding_id": fid},
        ], case_id="case-x")


def test_null_location_normalizes_to_empty_in_canonical_tuple() -> None:
    locless = _cand(path=None, start_line=None, end_line=None)
    blank = _cand(path="", start_line=None, end_line=None)
    assert derive_candidate_id("case-x", locless, 0) == derive_candidate_id("case-x", blank, 0)


def test_locationless_canonical_digest_matches_schema_derive() -> None:
    from daydream.benchmark import schema
    raw = _gold(path=None, start_line=None, end_line=None)
    payload = "\x1f".join([
        str("case-x" or ""), str(raw["title"] or ""), str(raw["body"] or ""),
        str(raw["severity"] or ""), str(raw["path"] or ""),
        str(raw["start_line"] or ""), str(raw["end_line"] or ""),
    ])
    verif = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    authoring = schema.derive_finding_id({
        "title": raw["title"], "body": raw["body"], "severity": raw["severity"],
        "location": None,}, case_id="case-x")
    assert verif == authoring


def test_duplicate_locationless_get_distinct_occurrence_ids() -> None:
    locless = _cand(path=None, start_line=None, end_line=None)
    assert derive_candidate_id("case-x", locless, 0) != derive_candidate_id("case-x", locless, 1)


def test_locationless_gold_set_accepts_canonical_id() -> None:
    g = _gold(path=None, start_line=None, end_line=None)
    g["finding_id"] = _canonical_gold_id("case-x", g)
    gs = validate_gold_set([g], case_id="case-x")
    assert len(gs) == 1 and gs[0].path is None


def test_locationless_candidate_accepted() -> None:
    f = parse_candidate_finding(_cand(path=None, start_line=None, end_line=None))
    assert f.path is None and f.start_line is None and f.end_line is None


def test_locationless_gold_accepted() -> None:
    g = parse_gold_finding(_gold(path=None, start_line=None, end_line=None))
    assert g.path is None and g.start_line is None and g.end_line is None


@pytest.mark.parametrize("partial", [
    {"path": "src/a.py", "start_line": None, "end_line": None},
    {"path": None, "start_line": 1, "end_line": 1},
    {"path": "src/a.py", "start_line": 1, "end_line": None},
])
def test_partial_location_rejected(partial: Any) -> None:
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(**partial))
    with pytest.raises(VerifierError):
        parse_gold_finding(_gold(**partial))


def test_mixed_located_locationless_set_matching_is_id_keyed() -> None:
    g_loc = _gold(title="Located", path="src/a.py", start_line=1, end_line=1)
    g_loc["finding_id"] = _canonical_gold_id("case-x", g_loc)
    g_non = _gold(title="Locationless", path=None, start_line=None, end_line=None)
    g_non["finding_id"] = _canonical_gold_id("case-x", g_non)
    gold = validate_gold_set([g_loc, g_non], case_id="case-x")

    c_loc = _cand(title="Located", path="src/a.py", start_line=1, end_line=1)
    c_loc["candidate_id"] = derive_candidate_id("case-x", c_loc, 0)
    c_non = _cand(title="Locationless", path=None, start_line=None, end_line=None)
    c_non["candidate_id"] = derive_candidate_id("case-x", c_non, 0)
    art = _artifact([c_loc, c_non])

    vs = [
        Verdict(g_loc["finding_id"], c_loc["candidate_id"], True, 0.9, "same"),
        Verdict(g_non["finding_id"], c_non["candidate_id"], True, 0.9, "same"),
    ]
    r = score_review(gold, art, vs)
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)
    assert r.reward == 1.0 and r.verifier_error == 0


# ---------------------------------------------------------------------------
# location-tier + severity-distance helpers (issue #971, task 2)
# ---------------------------------------------------------------------------


def test_location_tier_classification() -> None:
    # exact: same path, distance 0 inside the range
    assert vc.location_tier("a.py", 10, 12, "a.py", 10, 12, 3) == "exact"
    # near: same path, within tolerance
    assert vc.location_tier("a.py", 10, 12, "a.py", 15, 15, 3) == "near"
    # file: same path, beyond tolerance
    assert vc.location_tier("a.py", 10, 12, "a.py", 100, 100, 3) == "file"
    # miss: different path
    assert vc.location_tier("a.py", 10, 12, "b.py", 10, 12, 3) == "miss"


def test_location_tier_spans_overlap_counts_exact() -> None:
    # multi-line candidate range overlapping the gold range at all -> exact
    assert vc.location_tier("a.py", 10, 20, "a.py", 18, 25, 3) == "exact"
    # non-overlapping span uses the nearer boundary distance
    assert vc.location_tier("a.py", 10, 20, "a.py", 22, 24, 3) == "near"
    assert vc.location_tier("a.py", 10, 20, "a.py", 30, 32, 3) == "file"


def test_severity_distance_and_credit() -> None:
    assert vc.severity_distance("high", "high") == 0
    assert vc.severity_distance("high", "low") == 2
    assert vc.severity_distance("low", "medium") == 1
    assert vc.severity_credit(0) == 1.0
    assert vc.severity_credit(1) == 0.5
    assert vc.severity_credit(2) == 0.0


def test_severity_distance_none_is_absent() -> None:
    assert vc.severity_distance(None, "high") is None
    assert vc.severity_distance("high", None) is None
    assert vc.severity_distance(None, None) is None


def test_severity_distance_unknown_raises() -> None:
    with pytest.raises(VerifierError):
        vc.severity_distance("critical", "high")
    with pytest.raises(VerifierError):
        vc.severity_distance("high", "info")


# ---------------------------------------------------------------------------
# Task 3: reported location/severity axes over matched pairs (issue #971)
# ---------------------------------------------------------------------------

EXPECTED_24_KEYS = {
    "reward", "tp", "fp", "fn", "precision", "recall", "f1",
    "gold_count", "candidate_count", "clean_task", "clean_pass", "verifier_error",
    "location_exact", "location_near", "location_file", "location_miss",
    "location_credit", "location_present",
    "severity_exact", "severity_within_1", "severity_mean_distance",
    "severity_credit", "severity_pairs", "severity_present",
}


def _axis_gold(**overrides: Any) -> Any:
    return _gold(path="src/a.py", start_line=10, end_line=12, **overrides)


def _axis_cand_id(**overrides: Any) -> Any:
    base = _cand()
    base.update(overrides)
    base["candidate_id"] = derive_candidate_id("case-x", base, 0)
    return base


def _axis_pair(gold_raw: Any, cand_raw: Any) -> tuple[list[Any], Any, list[Verdict]]:
    gold = [parse_gold_finding(gold_raw)]
    art = _artifact([cand_raw])
    vs = [Verdict(gold_raw["finding_id"], cand_raw["candidate_id"], True, 0.9, "same bug")]
    return gold, art, vs


def test_score_review_axes_reported_not_gating() -> None:
    # 1 gold finding at src/a.py:10-12, high severity; candidate matches content
    # but reports src/a.py:50 (beyond tolerance) with low severity.
    gold_raw = _axis_gold()
    cand_raw = _axis_cand_id(path="src/a.py", start_line=50, end_line=50, severity="low")
    reward = score_review(*_axis_pair(gold_raw, cand_raw))
    assert reward.reward > 0.0 and reward.tp == 1          # content match still gates tp (R4)
    assert reward.location_file == 1 and reward.location_present == 1
    assert reward.severity_exact == 0 and reward.severity_within_1 == 0
    assert reward.severity_present == 1 and reward.severity_credit == 0.0
    assert reward.location_exact == 0 and reward.location_near == 0 and reward.location_miss == 0
    assert reward.location_credit == 0.0 and reward.severity_mean_distance == 2.0


def test_score_review_axis_absent_never_imputes() -> None:
    # gold finding is locationless (path/start_line/end_line all None) and
    # severityless -> both axes absent for the pair
    gold_raw = _gold(path=None, start_line=None, end_line=None, severity=None)
    cand_raw = _cand()
    cand_raw["candidate_id"] = derive_candidate_id("case-x", cand_raw, 0)
    reward = score_review(*_axis_pair(gold_raw, cand_raw))
    assert reward.tp == 1 and reward.location_present == 0
    assert reward.location_exact == 0 and reward.location_near == 0   # no counts, not imputed
    assert reward.location_credit == 0.0
    assert reward.severity_present == 0 and reward.severity_mean_distance == 0.0


def test_score_review_locationless_candidate_side_absent() -> None:
    # candidate locationless, gold located -> location axis absent; severity
    # axis still scored independently (both sides have severity)
    gold_raw = _axis_gold()
    cand_raw = _cand(path=None, start_line=None, end_line=None)
    cand_raw["candidate_id"] = derive_candidate_id("case-x", cand_raw, 0)
    reward = score_review(*_axis_pair(gold_raw, cand_raw))
    assert reward.tp == 1 and reward.location_present == 0 and reward.location_exact == 0
    assert reward.location_credit == 0.0
    assert reward.severity_present == 1 and reward.severity_exact == 1


def test_score_review_location_tiers_and_severity_exact() -> None:
    gold_raw = _axis_gold()
    # near: within tolerance (distance 2); same severity -> exact
    near_cand = _axis_cand_id(path="src/a.py", start_line=14, end_line=14, severity="high")
    r_near = score_review(*_axis_pair(gold_raw, near_cand))
    assert r_near.location_near == 1 and r_near.location_present == 1
    assert r_near.location_credit == 1.0
    assert r_near.severity_exact == 1 and r_near.severity_within_1 == 1
    assert r_near.severity_mean_distance == 0.0 and r_near.severity_credit == 1.0

    # exact tier
    exact_cand = _axis_cand_id(path="src/a.py", start_line=11, end_line=11)
    r_exact = score_review(*_axis_pair(gold_raw, exact_cand))
    assert r_exact.location_exact == 1 and r_exact.location_credit == 1.0

    # miss: different path -> counted as miss, location still present
    miss_cand = _axis_cand_id(path="src/b.py", start_line=10, end_line=12)
    r_miss = score_review(*_axis_pair(gold_raw, miss_cand))
    assert r_miss.location_miss == 1 and r_miss.location_present == 1
    assert r_miss.location_exact == 0 and r_miss.location_credit == 0.0


def test_reward_to_dict_stays_numeric_only() -> None:
    gold_raw = _axis_gold()
    cand_raw = _axis_cand_id()
    reward = score_review(*_axis_pair(gold_raw, cand_raw))
    d = reward.to_dict()
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in d.values())
    assert set(d) == EXPECTED_24_KEYS


def test_reward_dict_early_returns_have_absent_axes() -> None:
    # clean pass: no tp pairs -> all axis fields at zero/absent defaults (A4)
    d = score_review([], _artifact([]), []).to_dict()
    assert set(d) == EXPECTED_24_KEYS
    assert d["location_present"] == 0 and d["severity_present"] == 0
    assert d["location_exact"] == 0 and d["severity_mean_distance"] == 0.0


def test_aggregate_metrics_pools_axis_keys() -> None:
    rows: list[dict[str, object] | None] = [
        {"verifier_error": 0, "reward": 1.0, "tp": 1, "fp": 0, "fn": 0,
         "clean_task": 1, "location_exact": 1, "location_near": 0, "location_file": 0,
         "location_miss": 0, "location_credit": 1.0, "location_present": 1,
         "severity_exact": 1, "severity_within_1": 1, "severity_mean_distance": 0.0,
         "severity_credit": 1.0, "severity_pairs": 1, "severity_present": 1},
        {"verifier_error": 0, "reward": 0.0, "tp": 1, "fp": 1, "fn": 0,
         "clean_task": 1, "location_file": 1, "location_credit": 0.0, "location_present": 1,
         "severity_present": 0, "severity_exact": 0, "severity_within_1": 0,
         "severity_mean_distance": 0.0, "severity_credit": 0.0,
         "location_exact": 0, "location_near": 0, "location_miss": 0},
    ]
    m = vc.aggregate_metrics(rows)
    assert m["location_exact_rate"] == 0.5 and m["location_pairs_scored"] == 2
    assert m["severity_pairs_scored"] == 1 and m["severity_credit"] == 1.0
    assert m["severity_mean_distance"] == 0.0


def test_aggregate_metrics_axis_rates_zero_when_no_pairs() -> None:
    m = vc.aggregate_metrics([])
    for tier in ("exact", "near", "file", "miss"):
        assert m[f"location_{tier}_rate"] == 0.0
    assert m["location_pairs_scored"] == 0 and m["severity_pairs_scored"] == 0
    assert m["severity_exact_rate"] == 0.0
    assert m["severity_within_1_rate"] == 0.0
    assert m["severity_mean_distance"] == 0.0 and m["severity_credit"] == 0.0
    assert m["total_location_exact"] == 0 and m["total_severity_exact"] == 0


def test_aggregate_metrics_pools_severity_counts_and_credit() -> None:
    rows: list[dict[str, object] | None] = [
        {"verifier_error": 0, "reward": 1.0, "tp": 1, "fp": 0, "fn": 0,
         "clean_task": 1, "location_present": 0,
         "location_exact": 0, "location_near": 0, "location_file": 0,
         "location_miss": 0, "location_credit": 0.0,
         "severity_exact": 1, "severity_within_1": 1,
         "severity_mean_distance": 0.0, "severity_credit": 1.0,
         "severity_pairs": 1, "severity_present": 1},
        {"verifier_error": 0, "reward": 0.5, "tp": 1, "fp": 0, "fn": 1,
         "clean_task": 0, "location_present": 0,
         "location_exact": 0, "location_near": 0, "location_file": 0,
         "location_miss": 0, "location_credit": 0.0,
         "severity_exact": 0, "severity_within_1": 1,
         "severity_mean_distance": 1.0, "severity_credit": 0.5,
         "severity_pairs": 1, "severity_present": 1},
    ]
    m = vc.aggregate_metrics(rows)
    assert m["severity_pairs_scored"] == 2
    assert m["total_severity_exact"] == 1 and m["total_severity_within_1"] == 2
    assert m["severity_exact_rate"] == 0.5 and m["severity_within_1_rate"] == 1.0
    assert m["severity_mean_distance"] == 0.5 and m["severity_credit"] == 0.75


def test_aggregate_metrics_multi_pair_severity_rates_bounded() -> None:
    # A single task with two severity-scored pairs: the per-pair numerators
    # must divide by the pooled pair count, never the per-task row count, so
    # the rates stay <= 1.0 (issue: pooled axis rates could exceed 1.0).
    rows: list[dict[str, object] | None] = [
        {"verifier_error": 0, "reward": 1.0, "tp": 2, "fp": 0, "fn": 0,
         "clean_task": 1, "location_present": 0,
         "location_exact": 0, "location_near": 0, "location_file": 0,
         "location_miss": 0, "location_credit": 0.0,
         "severity_exact": 2, "severity_within_1": 2,
         "severity_mean_distance": 0.0, "severity_credit": 1.0,
         "severity_pairs": 2, "severity_present": 1},
    ]
    m = vc.aggregate_metrics(rows)
    assert m["severity_pairs_scored"] == 2
    assert m["severity_exact_rate"] == 1.0
    assert m["severity_within_1_rate"] == 1.0
    assert m["severity_mean_distance"] == 0.0 and m["severity_credit"] == 1.0


def test_aggregate_metrics_severity_means_weight_by_pair_count() -> None:
    # Unequal per-task pair counts (1 vs 2) must pool to the per-pair mean,
    # weighting each task's reported mean by its pair count.
    rows: list[dict[str, object] | None] = [
        {"verifier_error": 0, "reward": 1.0, "tp": 1, "fp": 0, "fn": 0,
         "clean_task": 1, "location_present": 0,
         "location_exact": 0, "location_near": 0, "location_file": 0,
         "location_miss": 0, "location_credit": 0.0,
         "severity_exact": 1, "severity_within_1": 1,
         "severity_mean_distance": 0.0, "severity_credit": 1.0,
         "severity_pairs": 1, "severity_present": 1},
        {"verifier_error": 0, "reward": 0.5, "tp": 2, "fp": 0, "fn": 0,
         "clean_task": 0, "location_present": 0,
         "location_exact": 0, "location_near": 0, "location_file": 0,
         "location_miss": 0, "location_credit": 0.0,
         "severity_exact": 0, "severity_within_1": 2,
         "severity_mean_distance": 1.0, "severity_credit": 0.5,
         "severity_pairs": 2, "severity_present": 1},
    ]
    m = vc.aggregate_metrics(rows)
    assert m["severity_pairs_scored"] == 3
    assert m["severity_mean_distance"] == pytest.approx(2.0 / 3)
    assert m["severity_credit"] == pytest.approx(2.0 / 3)


def test_aggregate_metrics_location_credit_weights_by_pair_count() -> None:
    # Unequal per-task pair counts (1 vs 2) must pool to the per-pair mean,
    # weighting each task's reported credit by its pair count (the sum of its
    # tier counts) so location_credit agrees with the per-pair tier rates.
    rows: list[dict[str, object] | None] = [
        {"verifier_error": 0, "reward": 1.0, "tp": 1, "fp": 0, "fn": 0,
         "clean_task": 1, "location_exact": 1, "location_near": 0,
         "location_file": 0, "location_miss": 0, "location_credit": 1.0,
         "location_present": 1,
         "severity_present": 0, "severity_exact": 0, "severity_within_1": 0,
         "severity_mean_distance": 0.0, "severity_credit": 0.0,
         "severity_pairs": 0},
        {"verifier_error": 0, "reward": 0.5, "tp": 2, "fp": 0, "fn": 0,
         "clean_task": 0, "location_exact": 0, "location_near": 1,
         "location_file": 1, "location_miss": 0, "location_credit": 0.5,
         "location_present": 1,
         "severity_present": 0, "severity_exact": 0, "severity_within_1": 0,
         "severity_mean_distance": 0.0, "severity_credit": 0.0,
         "severity_pairs": 0},
    ]
    m = vc.aggregate_metrics(rows)
    assert m["location_pairs_scored"] == 3
    # unweighted mean of the per-task means would be 0.75; per-pair pooling
    # (1 pair at 1.0, 2 pairs at 0.5) is 2/3, matching the pair-level rates.
    assert m["location_credit"] == pytest.approx(2.0 / 3)


def test_aggregate_metrics_pre_axis_rows_default_to_zero_pairs() -> None:
    # An older row without any axis keys is a genuinely zero-pair row: it
    # contributes nothing to the pooled axes and never raises.
    m = vc.aggregate_metrics([
        {"verifier_error": 0, "reward": 1.0, "tp": 1, "fp": 0, "fn": 0, "clean_task": 1},
    ])
    assert m["location_pairs_scored"] == 0 and m["severity_pairs_scored"] == 0
    assert m["location_exact_rate"] == 0.0 and m["severity_credit"] == 0.0
