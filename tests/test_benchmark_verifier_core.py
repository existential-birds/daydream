"""Tests for the stdlib-only harbor verifier core module."""
import hashlib
import json
from typing import Any

import pytest

from daydream.benchmark import schema
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



def test_candidate_and_gold_identity_validation(subtests: pytest.Subtests) -> None:
    """Native parser and reward-detail boundaries both reject malformed identities."""
    for kind, raw in (("candidate", "xyz"), ("gold", "not-hex")):
        with subtests.test(boundary="parser", kind=kind):
            with pytest.raises(VerifierError):
                if kind == "candidate":
                    parse_candidate_finding(_cand(candidate_id=raw))
                else:
                    parse_gold_finding(_gold(finding_id=raw))
    for kind in ("candidate", "gold"):
        with subtests.test(boundary="reward-details", kind=kind):
            with pytest.raises(VerifierError, match="must be a string"):
                if kind == "candidate":
                    reward_details([_gold()], [{"candidate_id": 456}], [], set())
                else:
                    reward_details([{"finding_id": 123}], [], [], set())


def test_harbor_package_imports_stdlib_only() -> None:
    assert vc.MAX_GOLD_FINDINGS == 50
    assert vc.MAX_CANDIDATE_FINDINGS == 100
    assert vc.MAX_ARTIFACT_BYTES == 1_048_576
    assert vc.CONFIDENCE_THRESHOLD == 0.7
    assert issubclass(vc.VerifierError, Exception)


def test_candidate_identity_is_scoped_stable_and_normalized(subtests: pytest.Subtests) -> None:
    """Case/occurrence changes distinguish findings; absent text normalizes consistently."""
    normal = _cand()
    assert derive_candidate_id("case-x", normal, 0) == derive_candidate_id("case-x", normal, 0)
    assert derive_candidate_id("case-x", normal, 0) != derive_candidate_id("case-y", normal, 0)
    for name, left, right in (
        ("null-text", _cand(title=None, body=None, severity=None), _cand(title="", body="", severity=None)),
        (
            "null-location",
            _cand(path=None, start_line=None, end_line=None),
            _cand(path="", start_line=None, end_line=None),
        ),
    ):
        with subtests.test(normalization=name):
            assert derive_candidate_id("case-x", left, 0) == derive_candidate_id("case-x", right, 0)
    locationless = _cand(path=None, start_line=None, end_line=None)
    assert derive_candidate_id("case-x", locationless, 0) != derive_candidate_id("case-x", locationless, 1)


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



def test_candidate_artifact_validation_rejects_invalid_envelopes(subtests: pytest.Subtests) -> None:
    """Reject structural and identity corruption at the deployed artifact parser."""
    for corruption in ("duplicate-id", "schema-version", "missing-ref", "top-level-key", "finding-key"):
        with subtests.test(corruption=corruption):
            findings = _valid_findings(2 if corruption == "duplicate-id" else 1)
            artifact = _artifact(findings)
            if corruption == "duplicate-id":
                findings[1] = dict(findings[0])
            elif corruption == "schema-version":
                artifact["schema_version"] = 2
            elif corruption == "missing-ref":
                del artifact["head_ref"]
            elif corruption == "top-level-key":
                artifact["smuggled"] = "x"
            else:
                findings[0]["smuggled"] = "x"
            with pytest.raises(VerifierError):
                validate_candidate_artifact(artifact)


def test_gold_set_requires_case_bound_unique_content(subtests: pytest.Subtests) -> None:
    """Authored gold accepts canonical IDs and rejects incompatible or expanded records."""
    valid = _gold()
    valid["finding_id"] = _canonical_gold_id("case-x", valid)
    validate_gold_set([valid], case_id="case-x")
    for corruption in ("legacy-id", "other-case", "unknown-field", "duplicate-id"):
        with subtests.test(corruption=corruption):
            gold = _gold()
            gold["finding_id"] = _canonical_gold_id("case-y" if corruption == "other-case" else "case-x", gold)
            if corruption == "legacy-id":
                gold["finding_id"] = "f" * 64
            elif corruption == "unknown-field":
                gold["provenance"] = {"kind": "authored", "source_ids": []}
            records = [gold, dict(gold)] if corruption == "duplicate-id" else [gold]
            with pytest.raises(VerifierError):
                validate_gold_set(records, case_id="case-x")


def test_verdict_validation_and_edge_admission(subtests: pytest.Subtests) -> None:
    """Confidence validity, match flags and the inclusive edge threshold are independent guards."""
    valid = Verdict(gold_id="g", candidate_id="c", match=True, confidence=0.9, reasoning="same bug")
    assert (valid.match, valid.confidence) == (True, 0.9)
    with pytest.raises(VerifierError):
        Verdict("g", "c", True, 1.5, "")
    for confidence, match, expected in (
        (0.9, True, True),
        (0.69, True, False),
        (0.95, False, False),
        (0.7, True, True),
    ):
        with subtests.test(confidence=confidence, match=match):
            verdict = Verdict("g", "c", match, confidence, "m")
            kept = retained_edges([verdict], ["g"], ["c"])
            assert {(v.gold_id, v.candidate_id) for v in kept} == ({("g", "c")} if expected else set())


def test_matching_maximizes_cardinality_without_reusing_either_side(subtests: pytest.Subtests) -> None:
    """Maximum matching must beat greedy selection, retain exclusivity, and break ties stably."""
    scenarios = [
        (
            "greedy-trap",
            [Verdict("A", "1", True, 0.9, ""), Verdict("A", "2", True, 0.8, ""), Verdict("B", "1", True, 0.7, "")],
            ["A", "B"],
            ["1", "2"],
            {("A", "2"), ("B", "1")},
        ),
        (
            "shared-candidate",
            [Verdict("g1", "c1", True, 0.9, ""), Verdict("g2", "c1", True, 0.8, "")],
            ["g1", "g2"],
            ["c1"],
            {("g1", "c1")},
        ),
        (
            "shared-gold",
            [Verdict("g1", "c1", True, 0.9, ""), Verdict("g1", "c2", True, 0.8, "")],
            ["g1"],
            ["c1", "c2"],
            {("g1", "c1")},
        ),
        (
            "confidence-tie",
            [
                Verdict("g1", "c1", True, 0.8, ""),
                Verdict("g1", "c2", True, 0.8, ""),
                Verdict("g2", "c1", True, 0.8, ""),
                Verdict("g2", "c2", True, 0.8, ""),
            ],
            ["g1", "g2"],
            ["c1", "c2"],
            {("g1", "c1"), ("g2", "c2")},
        ),
    ]
    for name, verdicts, gold_ids, candidate_ids, expected in scenarios:
        with subtests.test(scenario=name):
            assert maximum_matching(verdicts, gold_ids, candidate_ids) == expected
            assert maximum_matching(verdicts, gold_ids, candidate_ids) == expected


def test_score_review_matches_content_counts_and_empty_sides(subtests: pytest.Subtests) -> None:
    """False positives/negatives, F1, malformed output, and clean passes share one score boundary."""
    gold = [_gold(), _gold(finding_id="b" * 64, title="B")]
    candidates = _valid_findings(2)
    matched_gold = [*gold, _gold(finding_id="c" * 64, title="C")]
    verdicts = [
        Verdict("a" * 64, candidates[0]["candidate_id"], True, 0.9, "same"),
        Verdict("b" * 64, candidates[1]["candidate_id"], True, 0.8, "same"),
    ]
    scenarios: list[tuple[str, list[Any], Any, list[Verdict], dict[str, Any]]] = [
        (
            "clean-false-positives",
            [],
            _artifact(candidates),
            [],
            {"reward": 0.0, "tp": 0, "fp": 2, "fn": 0, "clean_task": 1, "clean_pass": 0},
        ),
        ("missing-candidates", gold, _artifact([]), [], {"reward": 0.0, "fp": 0, "fn": 2, "clean_task": 0}),
        (
            "no-matches",
            gold,
            _artifact(candidates),
            [],
            {"tp": 0, "fp": 2, "fn": 2, "precision": 0.0, "recall": 0.0, "f1": 0.0, "reward": 0.0},
        ),
        (
            "partial-recall",
            matched_gold,
            _artifact(candidates),
            verdicts,
            {
                "tp": 2,
                "fp": 0,
                "fn": 1,
                "precision": 1.0,
                "recall": pytest.approx(2 / 3),
                "f1": pytest.approx(0.8),
                "reward": pytest.approx(0.8),
            },
        ),
        (
            "malformed-output",
            [],
            {
                "schema_version": 1,
                "case_id": "c",
                "base_ref": "b",
                "head_ref": "h",
                "findings": [{"candidate_id": "not-hex"}],
            },
            [],
            {"reward": 0.0, "verifier_error": 0},
        ),
        ("clean-empty", [], _artifact([]), [], {"reward": 1.0}),
        ("gold-empty", [_gold()], _artifact([]), [], {"reward": 0.0, "fn": 1}),
        ("empty-side-ignores-verdict", [], _artifact([]), [Verdict("g", "c", True, 0.9, "")], {"reward": 1.0}),
    ]
    for name, gold_rows, artifact, verdict_rows, expected in scenarios:
        with subtests.test(scenario=name):
            actual = score_review(gold_rows, artifact, verdict_rows).to_dict()
            assert {key: actual[key] for key in expected} == expected


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







def _canonical_gold_id(case_id: str, f: dict[str, Any]) -> str:
    payload = "\x1f".join([
        str(case_id or ""), str(f.get("title") or ""), str(f.get("body") or ""),
        str(f.get("severity") or ""), str(f.get("path") or ""),
        str(f.get("start_line") or ""), str(f.get("end_line") or ""),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




def test_locationless_canonical_digest_matches_schema_derive() -> None:
    raw = _gold(path=None, start_line=None, end_line=None)
    payload = "\x1f".join([
        str("case-x" or ""), str(raw["title"] or ""), str(raw["body"] or ""),
        str(raw["severity"] or ""), str(raw["path"] or ""), str(raw["start_line"] or ""), str(raw["end_line"] or ""),
    ])
    verif = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    authoring = schema.derive_finding_id({"title": raw["title"], "body": raw["body"], "severity": raw["severity"],
        "location": None,}, case_id="case-x")
    assert verif == authoring


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

    vs = [Verdict(g_loc["finding_id"], c_loc["candidate_id"], True, 0.9, "same"),
        Verdict(g_non["finding_id"], c_non["candidate_id"], True, 0.9, "same"),
    ]
    r = score_review(gold, art, vs)
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)
    assert r.reward == 1.0 and r.verifier_error == 0




def test_location_tier_uses_file_and_nearest_span_boundary(subtests: pytest.Subtests) -> None:
    """Overlap is exact; otherwise the nearer span boundary determines tolerance."""
    for gold_end, candidate_path, start, end, expected in (
        (12, "a.py", 10, 12, "exact"),
        (12, "a.py", 15, 15, "near"),
        (12, "a.py", 100, 100, "file"),
        (12, "b.py", 10, 12, "miss"),
        (20, "a.py", 18, 25, "exact"),
        (20, "a.py", 22, 24, "near"),
        (20, "a.py", 30, 32, "file"),
    ):
        with subtests.test(gold_end=gold_end, path=candidate_path, start=start, end=end):
            assert vc.location_tier("a.py", 10, gold_end, candidate_path, start, end, 3) == expected


def test_severity_distance_distinguishes_absent_invalid_and_ordered_levels(subtests: pytest.Subtests) -> None:
    """Only canonical pairs earn distance credit; absent inputs stay absent."""
    for left, right, expected in (
        ("high", "high", 0),
        ("high", "low", 2),
        ("low", "medium", 1),
        (None, "high", None),
        ("high", None, None),
        (None, None, None),
    ):
        with subtests.test(left=left, right=right):
            assert vc.severity_distance(left, right) == expected
    for distance, expected_credit in ((0, 1.0), (1, 0.5), (2, 0.0)):
        with subtests.test(distance=distance):
            assert vc.severity_credit(distance) == expected_credit
    for left, right in (("critical", "high"), ("high", "info")):
        with subtests.test(invalid=(left, right)):
            with pytest.raises(VerifierError):
                vc.severity_distance(left, right)


EXPECTED_24_KEYS = {"reward", "tp", "fp", "fn", "precision", "recall", "f1",
    "gold_count", "candidate_count", "clean_task", "clean_pass", "verifier_error",
    "location_exact", "location_near", "location_file", "location_miss", "location_credit", "location_present",
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


def _metric_row(**overrides: Any) -> dict[str, Any]:
    """Default all axes to absent, matching older rows without axis keys.

    A ``*_present == 0`` axis contributes no pairs; tests opt in explicitly.
    """
    row: dict[str, Any] = {
        "verifier_error": 0, "reward": 1.0, "tp": 0, "fp": 0, "fn": 0, "clean_task": 0, "location_present": 0,
        "location_exact": 0, "location_near": 0, "location_file": 0, "location_miss": 0, "location_credit": 0.0,
        "severity_present": 0, "severity_exact": 0, "severity_within_1": 0, "severity_mean_distance": 0.0,
        "severity_credit": 0.0, "severity_pairs": 0,
    }
    row.update(overrides)
    return row


def test_score_review_axes_reported_not_gating() -> None:
    # Content matches despite incorrect location and severity.
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
    gold_raw = _gold(path=None, start_line=None, end_line=None, severity=None)
    cand_raw = _cand()
    cand_raw["candidate_id"] = derive_candidate_id("case-x", cand_raw, 0)
    reward = score_review(*_axis_pair(gold_raw, cand_raw))
    assert reward.tp == 1 and reward.location_present == 0
    assert reward.location_exact == 0 and reward.location_near == 0   # no counts, not imputed
    assert reward.location_credit == 0.0
    assert reward.severity_present == 0 and reward.severity_mean_distance == 0.0

def test_score_review_locationless_candidate_side_absent() -> None:
    # Missing candidate location leaves the severity axis independently scoreable.
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

    exact_cand = _axis_cand_id(path="src/a.py", start_line=11, end_line=11)
    r_exact = score_review(*_axis_pair(gold_raw, exact_cand))
    assert r_exact.location_exact == 1 and r_exact.location_credit == 1.0

    miss_cand = _axis_cand_id(path="src/b.py", start_line=10, end_line=12)
    r_miss = score_review(*_axis_pair(gold_raw, miss_cand))
    assert r_miss.location_miss == 1 and r_miss.location_present == 1
    assert r_miss.location_exact == 0 and r_miss.location_credit == 0.0

def test_reward_dict_early_returns_have_absent_axes() -> None:
    d = score_review([], _artifact([]), []).to_dict()
    assert set(d) == EXPECTED_24_KEYS
    assert d["location_present"] == 0 and d["severity_present"] == 0
    assert d["location_exact"] == 0 and d["severity_mean_distance"] == 0.0


def test_aggregate_metrics_weights_axes_by_matched_pairs_and_preserves_absence(subtests: pytest.Subtests) -> None:
    """Pair counts own axis denominators; historical/missing axes never synthesize scored pairs."""
    scenarios: list[tuple[str, list[dict[str, object] | None], dict[str, Any]]] = [
        (
            "mixed-axes",
            [
                _metric_row(
                    tp=1,
                    clean_task=1,
                    location_present=1,
                    location_exact=1,
                    location_credit=1.0,
                    severity_present=1,
                    severity_exact=1,
                    severity_within_1=1,
                    severity_credit=1.0,
                    severity_pairs=1,
                ),
                _metric_row(reward=0.0, tp=1, fp=1, clean_task=1, location_present=1, location_file=1),
            ],
            {
                "location_exact_rate": 0.5,
                "location_pairs_scored": 2,
                "severity_pairs_scored": 1,
                "severity_credit": 1.0,
                "severity_mean_distance": 0.0,
            },
        ),
        (
            "absent",
            [],
            {
                "location_exact_rate": 0.0,
                "location_near_rate": 0.0,
                "location_file_rate": 0.0,
                "location_miss_rate": 0.0,
                "location_pairs_scored": 0,
                "severity_pairs_scored": 0,
                "severity_exact_rate": 0.0,
                "severity_within_1_rate": 0.0,
                "severity_mean_distance": 0.0,
                "severity_credit": 0.0,
                "total_location_exact": 0,
                "total_severity_exact": 0,
            },
        ),
        (
            "severity-credit",
            [
                _metric_row(
                    tp=1,
                    clean_task=1,
                    severity_present=1,
                    severity_exact=1,
                    severity_within_1=1,
                    severity_credit=1.0,
                    severity_pairs=1,
                ),
                _metric_row(
                    reward=0.5,
                    tp=1,
                    fn=1,
                    severity_present=1,
                    severity_within_1=1,
                    severity_mean_distance=1.0,
                    severity_credit=0.5,
                    severity_pairs=1,
                ),
            ],
            {
                "severity_pairs_scored": 2,
                "total_severity_exact": 1,
                "total_severity_within_1": 2,
                "severity_exact_rate": 0.5,
                "severity_within_1_rate": 1.0,
                "severity_mean_distance": 0.5,
                "severity_credit": 0.75,
            },
        ),
        (
            "multi-pair-bound",
            [
                _metric_row(
                    tp=2,
                    clean_task=1,
                    severity_present=1,
                    severity_exact=2,
                    severity_within_1=2,
                    severity_credit=1.0,
                    severity_pairs=2,
                )
            ],
            {
                "severity_pairs_scored": 2,
                "severity_exact_rate": 1.0,
                "severity_within_1_rate": 1.0,
                "severity_mean_distance": 0.0,
                "severity_credit": 1.0,
            },
        ),
        (
            "weighted-severity",
            [
                _metric_row(
                    tp=1,
                    clean_task=1,
                    severity_present=1,
                    severity_exact=1,
                    severity_within_1=1,
                    severity_credit=1.0,
                    severity_pairs=1,
                ),
                _metric_row(
                    reward=0.5,
                    tp=2,
                    severity_present=1,
                    severity_within_1=2,
                    severity_mean_distance=1.0,
                    severity_credit=0.5,
                    severity_pairs=2,
                ),
            ],
            {
                "severity_pairs_scored": 3,
                "severity_mean_distance": pytest.approx(2 / 3),
                "severity_credit": pytest.approx(2 / 3),
            },
        ),
        (
            "weighted-location",
            [
                _metric_row(tp=1, clean_task=1, location_present=1, location_exact=1, location_credit=1.0),
                _metric_row(
                    reward=0.5, tp=2, location_present=1, location_near=1, location_file=1, location_credit=0.5
                ),
            ],
            {"location_pairs_scored": 3, "location_credit": pytest.approx(2 / 3)},
        ),
        (
            "historical",
            [{"verifier_error": 0, "reward": 1.0, "tp": 1, "fp": 0, "fn": 0, "clean_task": 1}],
            {
                "location_pairs_scored": 0,
                "severity_pairs_scored": 0,
                "location_exact_rate": 0.0,
                "severity_credit": 0.0,
            },
        ),
    ]
    for name, rows, expected in scenarios:
        with subtests.test(scenario=name):
            actual = vc.aggregate_metrics(rows)
            assert {key: actual[key] for key in expected} == expected
