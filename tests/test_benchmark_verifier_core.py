"""Tests for the stdlib-only harbor verifier core module."""

import hashlib
from typing import Any

import pytest

from daydream.benchmark.harbor import verifier_core as vc
from daydream.benchmark.harbor.verifier_core import (
    Verdict,
    VerifierError,
    derive_candidate_id,
    maximum_matching,
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


def test_score_malformed_artifact_is_scored_zero() -> None:
    art = {"schema_version": 1, "case_id": "c", "base_ref": "b", "head_ref": "h",
           "findings": [{"candidate_id": "not-hex"}]}
    r = score_review([], art, [])
    assert r.reward == 0.0 and r.verifier_error == 0  # invalid agent output is scored zero


def test_empty_side_resolves_with_zero_verdicts() -> None:
    r0 = score_review([], _artifact([]), [])
    assert r0.reward == 1.0
    d = r0.to_dict()
    assert set(d) == EXPECTED_24_KEYS
    assert d["location_present"] == 0 and d["severity_present"] == 0
    assert d["location_exact"] == 0 and d["severity_mean_distance"] == 0.0
    rn = score_review([_gold(), _gold(finding_id="b" * 64, title="B")], _artifact([]), [])
    assert (rn.reward, rn.fn, rn.fp, rn.clean_task) == (0.0, 2, 0, 0)
    # passing any verdict for an empty side must not change the result (ignored/not required)
    assert score_review([], _artifact([]), [Verdict("g", "c", True, 0.9, "")]).reward == 1.0


def _canonical_gold_id(case_id: str, f: dict[str, Any]) -> str:
    payload = "\x1f".join([
        str(case_id or ""), str(f.get("title") or ""), str(f.get("body") or ""),
        str(f.get("severity") or ""), str(f.get("path") or ""),
        str(f.get("start_line") or ""), str(f.get("end_line") or ""),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_parsers_reject_malformed_ids_schemas_verdicts_and_severities() -> None:
    """Each public input boundary fails closed with VerifierError rather than scoring bad data."""
    fid = _canonical_gold_id("case-x", _gold())
    smuggled = _artifact(_valid_findings(1))
    smuggled["smuggled"] = "x"
    rejected: list[tuple[Any, ...]] = [
        (parse_gold_finding, _gold(finding_id="not-hex")),
        (validate_candidate_artifact, _artifact([], schema_version=2)),
        (validate_candidate_artifact, smuggled),
        (lambda: Verdict("g", "c", True, 1.5, ""),),
        (lambda: validate_gold_set([{**_gold(), "finding_id": fid}] * 2, case_id="case-x"),),
        (vc.severity_distance, "critical", "high"),
        (vc.severity_distance, "high", "info"),
    ]
    for call, *args in rejected:
        with pytest.raises(VerifierError):
            call(*args)
    with pytest.raises(VerifierError, match="must be a string"):
        reward_details([{"finding_id": 123}], [], [], set())
    with pytest.raises(VerifierError, match="must be a string"):
        reward_details([_gold()], [{"candidate_id": 456}], [], set())


EXPECTED_24_KEYS = {"reward", "tp", "fp", "fn", "precision", "recall", "f1",
    "gold_count", "candidate_count", "clean_task", "clean_pass", "verifier_error",
    "location_exact", "location_near", "location_file", "location_miss", "location_credit", "location_present",
    "severity_exact", "severity_within_1", "severity_mean_distance",
    "severity_credit", "severity_pairs", "severity_present",
}
