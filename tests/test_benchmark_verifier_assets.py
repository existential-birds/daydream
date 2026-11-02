"""Behavioral contract tests for the Harbor verifier scoring core and metric.

``verifier_core.aggregate_metrics`` is the canonical scorer: these tests pin
its observable aggregation contract (pooling, zero-denominator semantics,
absent-axis handling) plus the compiled metric's subprocess behavior, so
later canonicalization work must preserve behavior rather than source text.
"""

import importlib.util
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from daydream.benchmark.harbor import build, candidate, verifier_core, verifier_core as vc
from daydream.benchmark.harbor.build import _copy_assets
from daydream.hunk_index import range_distance as hunk_range_distance

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def copied_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _copy_assets(tmp_path)
    path = tmp_path / "tests" / "verifier_core.py"
    spec = importlib.util.spec_from_file_location("copied_validation_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _finding_content() -> dict[str, Any]:
    return {"title": "Cache key omits tenant", "body": "Two tenants can receive the same cached response.",
        "severity": "high", "path": "src/cache.py", "start_line": 3, "end_line": 5,
    }


def test_public_content_parser_rejects_invalid_content() -> None:
    """Over-cap, control-character, empty, absolute, non-integer and inverted fields fail closed."""
    for changes in ({"title": "x" * 501}, {"title": "a\x00b"}, {"body": ""}, {"path": ""}, {"path": "/src/cache.py"},
                    {"path": "src/ca\x00che.py"}, {"start_line": True}, {"start_line": 0}, {"end_line": "5"},
                    {"end_line": 2}):
        with pytest.raises(verifier_core.VerifierError):
            verifier_core.parse_finding_content(_finding_content() | changes)
    with pytest.raises(verifier_core.VerifierError):
        verifier_core.parse_finding_content([])


def test_host_built_artifacts_match_copied_verifier(copied_verifier: ModuleType) -> None:
    key = "case-validation-parity"
    content = {"title": "Tenant omitted", "body": "Tenant omitted\n\n**Severity:** high\n\nResponses cross tenants",
               "severity": "high", "path": "src/cache.py", "start_line": 3, "end_line": 3}
    candidate_artifact = candidate.build_candidate_artifact(
        key, [content | {"candidate_id": vc.derive_candidate_id(key, content, ordinal)} for ordinal in range(2)]
    )
    findings: list[dict[str, Any]] = [
        {"finding_id": "a" * 64, "title": "Cache key omits tenant", "body": "Wrong tenant",
         "severity": "high", "location": {"path": "src/cache.py", "start_line": 3, "end_line": 5}},
        {"finding_id": "b" * 64, "title": "Missing access check", "body": "No authorization",
         "severity": None, "location": None},
    ]
    gold = build.build_gold_list(findings, key=key)
    oracle = build.build_oracle_artifact(key, findings)
    for artifact in (candidate_artifact, oracle, candidate.build_candidate_artifact(key, [])):
        expected = [asdict(f) for f in verifier_core.validate_candidate_artifact(artifact)]
        assert [asdict(f) for f in copied_verifier.validate_candidate_artifact(artifact)] == expected
        assert [f["candidate_id"] for f in artifact["findings"]] == [f["candidate_id"] for f in expected]
    assert candidate_artifact["findings"][0]["candidate_id"] != candidate_artifact["findings"][1]["candidate_id"]
    other_key = "case-another-review"
    second_artifact = candidate.build_candidate_artifact(
        other_key,
        [content | {"candidate_id": vc.derive_candidate_id(other_key, content, ordinal)} for ordinal in range(2)],
    )
    copied_verifier.validate_candidate_artifact(second_artifact)
    second_gold = build.build_gold_list(findings, key=other_key)
    copied_verifier.validate_gold_set(second_gold, case_id=other_key)
    assert {finding["finding_id"] for finding in gold}.isdisjoint(finding["finding_id"] for finding in second_gold)
    assert {f["candidate_id"] for f in candidate_artifact["findings"]}.isdisjoint(
        f["candidate_id"] for f in second_artifact["findings"]
    )
    for entries in (gold, []):
        expected = [asdict(f) for f in verifier_core.validate_gold_set(entries, case_id=key)]
        assert [asdict(f) for f in copied_verifier.validate_gold_set(entries, case_id=key)] == expected

    # Mutate host-built artifacts at the input boundary: both deployed and host
    # validators must reject them with their own public exception category.
    located_gold = next(i for i, finding in enumerate(gold) if finding["path"] is not None)
    for changes in ({"body": "é" * 4096 + "x"}, {"path": None}, {"severity": "critical"}):
        bad_candidate = deepcopy(candidate_artifact)
        bad_candidate["findings"][0].update(changes)
        bad_gold = deepcopy(gold)
        bad_gold[located_gold].update(changes)
        for module in (verifier_core, copied_verifier):
            with pytest.raises(module.VerifierError, match="8 KiB|null|severity"):
                module.validate_candidate_artifact(bad_candidate)
            with pytest.raises(module.VerifierError, match="8 KiB|null|severity"):
                module.validate_gold_set(bad_gold, case_id=key)

    for module in (verifier_core, copied_verifier):
        wrong_id = deepcopy(candidate_artifact)
        wrong_id["findings"][0]["candidate_id"] = "0" * 64
        with pytest.raises(module.VerifierError, match="derived id"):
            module.validate_candidate_artifact(wrong_id)
        wrong_gold_id = deepcopy(gold)
        wrong_gold_id[0]["finding_id"] = "0" * 64
        with pytest.raises(module.VerifierError, match="canonical digest"):
            module.validate_gold_set(wrong_gold_id, case_id=key)
        too_many = deepcopy(candidate_artifact)
        too_many["findings"] = candidate_artifact["findings"] * 51
        with pytest.raises(module.VerifierError, match="100 findings"):
            module.validate_candidate_artifact(too_many)
        with pytest.raises(module.VerifierError, match="50 findings"):
            module.validate_gold_set(gold * 26, case_id=key)
        too_large = deepcopy(candidate_artifact)
        too_large["findings"][0]["body"] = "x" * module.MAX_ARTIFACT_BYTES
        with pytest.raises(module.VerifierError, match="1 MiB"):
            module.validate_candidate_artifact(too_large)


def test_range_distance_cannot_drift_from_hunk_index() -> None:
    assert vc.LOCATION_TOLERANCE >= 3  # below 3 measures the snapper, not the reviewer (R2)
    cases: list[tuple[int, int, int]] = [
        (1, 5, 10),   # below the range -> distance to the start boundary
        (4, 5, 10),   # one line below start
        (5, 5, 10),   # on the start boundary -> inside
        (7, 5, 10),   # inside the range
        (10, 5, 10),  # on the end boundary -> inside
        (11, 5, 10),  # one line above end
        (12, 5, 10),  # above the range -> distance to the end boundary
        (0, 0, 0),    # degenerate single-line range, on the line
        (2, 0, 0),    # degenerate range, below
        (0, 3, 3),    # degenerate range, above
    ]
    for line, start, end in cases:
        assert vc._range_distance(line, start, end) == hunk_range_distance(line, start, end)
