"""Tests for the stdlib-only harbor verifier core module."""

import pytest

from daydream.benchmark.harbor.verifier_core import (
    CandidateFinding,
    GoldFinding,
    VerifierError,
    parse_candidate_finding,
    parse_gold_finding,
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
