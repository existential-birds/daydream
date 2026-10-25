"""Tests for the stdlib-only harbor verifier core module."""

import hashlib
import json
from typing import Any

import pytest

from daydream.benchmark.harbor import verifier_core as vc
from daydream.benchmark.harbor.verifier_core import (
    Verdict,
    VerifierError,
    derive_candidate_id,
    maximum_matching,
    parse_candidate_finding,
    parse_gold_finding,
    retained_edges,
    reward_details,
    score_review,
    validate_candidate_artifact,
    validate_gold_set,
)


def _cand(**overrides: Any) -> Any:
    base = {"candidate_id": "a" * 64, "title": "Cache key not scoped", "body": "Cache key omits the auth realm",
        "severity": "high", "path": "src/cache.py", "start_line": 42, "end_line": 42,
    }
    base.update(overrides)
    return base


def _gold(**overrides: Any) -> Any:
    base = {"finding_id": "a" * 64, "title": "Cache key not scoped", "body": "Cache key omits the auth realm",
        "severity": "high", "path": "src/cache.py", "start_line": 42, "end_line": 42,
    }
    base.update(overrides)
    return base


def test_candidate_rejects_bad_id_hex() -> None:
    with pytest.raises(VerifierError):
        parse_candidate_finding(_cand(candidate_id="xyz"))


def test_gold_rejects_bad_id_hex() -> None:
    with pytest.raises(VerifierError):
        parse_gold_finding(_gold(finding_id="not-hex"))


def _artifact(findings: Any, **overrides: Any) -> Any:
    base = {"schema_version": 1, "case_id": "case-x", "base_ref": "base", "head_ref": "head", "findings": findings}
    base.update(overrides)
    return base


def _valid_findings(n: Any=1, key: Any="case-x") -> Any:
    out = []
    groups: dict[tuple[Any, ...], int] = {}
    for i in range(n):
        base = _cand(title=f"f{i}", body=f"body{i}")
        canon = (f"f{i}", f"body{i}", base.get("severity") or "", base["path"], base["start_line"], base["end_line"])
        ordinal = groups.get(canon, 0)
        groups[canon] = ordinal + 1
        base["candidate_id"] = derive_candidate_id(key, base, ordinal)
        out.append(base)
    return out


def test_artifact_rejects_duplicate_candidate_id() -> None:
    dup = _valid_findings(2)
    dup[1] = dict(dup[0])  # same id, different title → not allowed
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact(dup))


def test_artifact_rejects_wrong_schema_version() -> None:
    with pytest.raises(VerifierError):
        validate_candidate_artifact(_artifact([], schema_version=2))


def test_artifact_rejects_missing_refs() -> None:
    art = _artifact([])
    del art["head_ref"]
    with pytest.raises(VerifierError):
        validate_candidate_artifact(art)


def test_gold_finding_id_is_case_scoped() -> None:
    # gold finding_id must equal sha256(case_id, title, body, severity, path, start, end)
    g = _gold()
    g["finding_id"] = _canonical_gold_id("case-x", g)
    validate_gold_set([g], case_id="case-x")                       # case-scoped digest accepted
    legacy = _gold()
    legacy["finding_id"] = "f" * 64  # valid 64-hex, case_id-less digest -> rejected
    with pytest.raises(VerifierError):
        validate_gold_set([legacy], case_id="case-x")
    g2 = _gold()
    g2["finding_id"] = _canonical_gold_id("case-y", g2)
    with pytest.raises(VerifierError):
        validate_gold_set([g2], case_id="case-x")


def test_verdict_rejects_out_of_range_confidence() -> None:
    with pytest.raises(VerifierError):
        Verdict("g", "c", True, 1.5, "")


def test_mixed_review_matches_one_to_one_and_pools_reported_axes() -> None:
    # A greedy A->1 choice loses B. Other pairs independently exercise every
    # location tier, missing signals, severity distance, and rejected verdicts.
    gold = [
        _gold(finding_id="A", path="src/a.py", start_line=10, end_line=20),
        _gold(finding_id="B", path="src/a.py", start_line=10, end_line=20),
        _gold(finding_id="C", path="src/c.py", start_line=10, end_line=12),
        _gold(finding_id="D", path="src/d.py", severity=None),
        _gold(finding_id="E", path=None, start_line=None, end_line=None),
        _gold(finding_id="F", path="src/f.py"),
        _gold(finding_id="G"),
        _gold(finding_id="H"),
        _gold(finding_id="I"),
    ]
    candidates = [
        _cand(path="src/a.py", start_line=18, end_line=25),
        _cand(path="src/a.py", start_line=22, end_line=24, severity="medium"),
        _cand(path="src/c.py", start_line=100, end_line=100, severity="low"),
        _cand(path="src/other.py"),
        _cand(severity=None),
        _cand(path=None, start_line=None, end_line=None),
        _cand(title="below threshold"),
        _cand(title="explicit non-match"),
    ]
    for raw in candidates:
        raw["candidate_id"] = derive_candidate_id("case-x", raw, 0)
    ids = [raw["candidate_id"] for raw in candidates]
    verdicts = [
        Verdict("A", ids[0], True, 0.9, "same bug"),
        Verdict("A", ids[1], True, 0.8, "same bug"),
        Verdict("B", ids[0], True, 0.7, "same bug"),
        Verdict("C", ids[2], True, 0.9, "same bug"),
        Verdict("D", ids[3], True, 0.9, "same bug"),
        Verdict("E", ids[4], True, 0.9, "same bug"),
        Verdict("F", ids[5], True, 0.9, "same bug"),
        Verdict("G", ids[6], True, 0.69, "uncertain"),
        Verdict("H", ids[7], False, 0.95, "different bug"),
        Verdict("I", ids[0], True, 0.8, "same bug"),
    ]
    matches = maximum_matching(retained_edges(verdicts, list("ABCDEFGHI"), ids), list("ABCDEFGHI"), ids)
    assert matches == {("A", ids[1]), ("B", ids[0]), ("C", ids[2]), ("D", ids[3]), ("E", ids[4]), ("F", ids[5])}
    reward = score_review(gold, _artifact(candidates), verdicts)
    assert (reward.tp, reward.fp, reward.fn) == (6, 2, 3)
    assert reward.precision == 0.75 and reward.recall == 2 / 3
    assert reward.f1 == pytest.approx(12 / 17) and reward.reward == reward.f1
    assert (reward.location_exact, reward.location_near, reward.location_file, reward.location_miss) == (1, 1, 1, 1)
    assert reward.location_present == 1 and reward.location_credit == 0.5
    assert (reward.severity_pairs, reward.severity_exact, reward.severity_within_1) == (4, 2, 3)
    assert reward.severity_present == 1 and reward.severity_mean_distance == 0.75 and reward.severity_credit == 0.625

    # Unequal pair counts distinguish pooled metrics from averaging task means.
    short = score_review(gold[:2], _artifact(candidates[:2]), verdicts[:3])
    metrics = vc.aggregate_metrics(
        [
            dict(reward.to_dict()),
            dict(short.to_dict()),
            {"verifier_error": 0, "reward": 1.0, "tp": 1, "fp": 0, "fn": 0, "clean_task": 1, "clean_pass": 1},
            None,
            {"verifier_error": 1, "reward": 0.0, "tp": 50, "fp": 50, "fn": 50},
        ]
    )
    assert (metrics["task_count"], metrics["scored_task_count"], metrics["infra_error_task_count"]) == (5, 3, 2)
    assert (metrics["total_tp"], metrics["total_fp"], metrics["total_fn"]) == (9, 2, 3)
    assert metrics["micro_precision"] == 9 / 11 and metrics["micro_recall"] == 3 / 4
    assert metrics["micro_f1"] == pytest.approx(18 / 23)
    assert metrics["mean_task_score"] == pytest.approx(46 / 51)
    assert metrics["location_pairs_scored"] == 6 and metrics["location_credit"] == pytest.approx(2 / 3)
    assert metrics["location_exact_rate"] == 1 / 3 and metrics["location_near_rate"] == 1 / 3
    assert metrics["location_file_rate"] == 1 / 6 and metrics["location_miss_rate"] == 1 / 6
    assert metrics["severity_pairs_scored"] == 6 and metrics["total_severity_exact"] == 3
    assert metrics["total_severity_within_1"] == 5 and metrics["severity_exact_rate"] == 0.5
    assert metrics["severity_within_1_rate"] == 5 / 6
    assert metrics["severity_mean_distance"] == pytest.approx(2 / 3)
    assert metrics["severity_credit"] == pytest.approx(2 / 3)


def test_matching_is_deterministic_across_runs() -> None:
    # ties on confidence: ordering must still be stable run-to-run
    vs = [Verdict("g1", "c1", True, 0.8, ""), Verdict("g1", "c2", True, 0.8, ""),
          Verdict("g2", "c1", True, 0.8, ""), Verdict("g2", "c2", True, 0.8, "")]
    gold, cand = ["g1", "g2"], ["c1", "c2"]
    r1 = maximum_matching(vs, gold, cand)
    r2 = maximum_matching(vs, gold, cand)
    assert r1 == r2 == {("g1", "c1"), ("g2", "c2")}


def test_score_clean_with_candidates() -> None:
    art = _artifact(_valid_findings(2))
    r = score_review([], art, [])
    assert (r.reward, r.fp, r.clean_pass) == (0.0, 2, 0)
    assert r.tp == 0 and r.fn == 0 and r.clean_task == 1
    missed = score_review([_gold()], _artifact([]), [])
    corpus = vc.aggregate_metrics([dict(r.to_dict()), dict(missed.to_dict())])
    assert (corpus["total_tp"], corpus["total_fp"], corpus["total_fn"]) == (0, 2, 1)
    assert (corpus["micro_precision"], corpus["micro_recall"], corpus["micro_f1"]) == (0.0, 0.0, 0.0)
    nonclean = vc.aggregate_metrics([dict(missed.to_dict())])
    assert nonclean["clean_task_count"] == 0 and nonclean["clean_accuracy"] == 1.0


def test_score_gold_no_candidates() -> None:
    gold = [_gold(), _gold(finding_id="b" * 64, title="B")]
    r = score_review(gold, _artifact([]), [])
    assert (r.reward, r.fn) == (0.0, 2)
    assert r.fp == 0 and r.clean_task == 0


def test_score_zero_match_reward_is_zero() -> None:
    gold = [_gold(), _gold(finding_id="b" * 64, title="B")]
    art = _artifact(_valid_findings(2))
    r = score_review(gold, art, [])
    assert (r.tp, r.fp, r.fn) == (0, 2, 2)
    assert r.precision == 0.0 and r.recall == 0.0
    assert r.f1 == 0.0 and r.reward == 0.0


def test_score_malformed_artifact_is_scored_zero() -> None:
    art = {"schema_version": 1, "case_id": "c", "base_ref": "b", "head_ref": "h",
           "findings": [{"candidate_id": "not-hex"}]}
    r = score_review([], art, [])
    assert r.reward == 0.0 and r.verifier_error == 0  # invalid agent output is scored zero


def test_empty_side_resolves_with_zero_verdicts() -> None:
    r0 = score_review([], _artifact([]), [])
    assert r0.reward == 1.0
    rn = score_review([_gold()], _artifact([]), [])
    assert rn.reward == 0.0 and rn.fn == 1
    # passing any verdict for an empty side must not change the result (ignored/not required)
    assert score_review([], _artifact([]), [Verdict("g", "c", True, 0.9, "")]).reward == 1.0


def test_reward_details_shape_and_no_source_leak() -> None:
    gold = [_gold(), _gold(finding_id="b" * 64, title="B")]
    cands = [parse_candidate_finding(_cand(candidate_id="c" * 64, title="f1")),
        parse_candidate_finding(_cand(candidate_id="d" * 64, title="f2")),
    ]
    vs = [Verdict("a" * 64, cands[0].candidate_id, True, 0.9, "same bug")]
    m = {("a" * 64, cands[0].candidate_id)}
    details = reward_details(gold, cands, vs, m)
    for key in ("verdicts", "matches", "unmatched_gold", "unmatched_candidates"):
        assert key in details
    assert details["unmatched_gold"] == ["b" * 64]
    assert details["unmatched_candidates"] == [cands[1].candidate_id]
    blob = json.dumps(details)
    assert "same bug" in blob  # reasoning is kept
    assert "Cache key not scoped" not in blob  # finding body/title never leaks
    assert "f1" not in blob  # candidate content never leaks


def test_gold_finding_id_rejects_non_string() -> None:
    gold = [{"finding_id": 123}]
    with pytest.raises(VerifierError, match="must be a string"):
        reward_details(gold, [], [], set())


def test_candidate_id_rejects_non_string() -> None:
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


def test_gold_set_rejects_duplicate_finding_ids() -> None:
    fid = _canonical_gold_id("case-x", _gold())
    with pytest.raises(VerifierError):
        validate_gold_set([{**_gold(), "finding_id": fid}, {**_gold(), "finding_id": fid}], case_id="case-x")


def test_severity_distance_unknown_raises() -> None:
    with pytest.raises(VerifierError):
        vc.severity_distance("critical", "high")
    with pytest.raises(VerifierError):
        vc.severity_distance("high", "info")


EXPECTED_24_KEYS = {"reward", "tp", "fp", "fn", "precision", "recall", "f1",
    "gold_count", "candidate_count", "clean_task", "clean_pass", "verifier_error",
    "location_exact", "location_near", "location_file", "location_miss", "location_credit", "location_present",
    "severity_exact", "severity_within_1", "severity_mean_distance",
    "severity_credit", "severity_pairs", "severity_present",
}


def _axis_pair(gold_raw: Any, cand_raw: Any) -> tuple[list[Any], Any, list[Verdict]]:
    gold = [parse_gold_finding(gold_raw)]
    art = _artifact([cand_raw])
    vs = [Verdict(gold_raw["finding_id"], cand_raw["candidate_id"], True, 0.9, "same bug")]
    return gold, art, vs


def test_score_review_axis_absent_never_imputes() -> None:
    gold_raw = _gold(path=None, start_line=None, end_line=None, severity=None)
    cand_raw = _cand()
    cand_raw["candidate_id"] = derive_candidate_id("case-x", cand_raw, 0)
    reward = score_review(*_axis_pair(gold_raw, cand_raw))
    assert reward.tp == 1 and reward.location_present == 0
    assert reward.location_exact == 0 and reward.location_near == 0   # no counts, not imputed
    assert reward.location_credit == 0.0
    assert reward.severity_present == 0 and reward.severity_mean_distance == 0.0


def test_reward_dict_early_returns_have_absent_axes() -> None:
    d = score_review([], _artifact([]), []).to_dict()
    assert set(d) == EXPECTED_24_KEYS
    assert d["location_present"] == 0 and d["severity_present"] == 0
    assert d["location_exact"] == 0 and d["severity_mean_distance"] == 0.0


def test_aggregate_metrics_axis_rates_zero_when_no_pairs() -> None:
    m = vc.aggregate_metrics([])
    for tier in ("exact", "near", "file", "miss"):
        assert m[f"location_{tier}_rate"] == 0.0
    assert m["location_pairs_scored"] == 0 and m["severity_pairs_scored"] == 0
    assert m["severity_exact_rate"] == 0.0
    assert m["severity_within_1_rate"] == 0.0
    assert m["severity_mean_distance"] == 0.0 and m["severity_credit"] == 0.0
    assert m["total_location_exact"] == 0 and m["total_severity_exact"] == 0
