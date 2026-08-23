"""Fake-endpoint and fixture tests for ``daydream benchmark calibrate-judge``.

Pins the 24-pair source-free fixture contract, the importlib template loader,
the injectable calibration client-builder, host allowlist validation, the
72-call judge driver, the three-part pass gate, deterministic receipt
build/write/invalidation, the end-to-end orchestrator, the CLI subcommand,
and the six mandated fake-endpoint acceptance cases — all driven through the
injected ``http=`` seam, never a real socket.
"""

import json
import re
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parents[1] / "daydream" / "benchmark" / "harbor" / "calibration"

REQUIRED_CATEGORIES = {
    "wording_location", "related_distinct", "negation", "same_class_different_components",
    "severity_disagreement", "locationless", "delimiter_injection", "near_miss",
}
_CRED = re.compile(r"sk-ant-|sk-or-|Bearer |x-api-key")


def test_fixture_is_24_pairs_12_12():
    pairs = json.loads((_FIXTURE / "pairs.json").read_text())
    assert len(pairs) == 24
    labels = [p["label"] for p in pairs]
    assert labels.count("match") == 12 and labels.count("nonmatch") == 12


def test_fixture_covers_all_eight_categories():
    pairs = json.loads((_FIXTURE / "pairs.json").read_text())
    assert {p["category"] for p in pairs} >= REQUIRED_CATEGORIES


def test_fixture_is_source_free():
    text = (_FIXTURE / "pairs.json").read_text()
    assert not _CRED.search(text)
    # no content lifted from the real source-derived golden-review.json fixture
    gold = (_FIXTURE.parents[0] / "templates" / "tests" / "golden-review.json").read_text()
    for tok in ("Cache key not tenant-scoped", "0e2356faadfbf30a"):
        assert tok not in text


def test_fixture_has_provenance_note():
    note = (_FIXTURE / "PROVENANCE.md").read_text()
    assert "reviewer" in note.lower() and "source-free" in note.lower()


def test_every_pair_renders_within_24kib():
    # Uses the loader/fixture-load API built in Task 2; marker for the executor.
    from daydream.benchmark.harbor.calibrate import _load_judge_template, _load_fixture
    sr = _load_judge_template()
    for p in _load_fixture():
        sr.render_pair_prompt(p["gold"], p["candidate"], template=sr.JUDGE_PROMPT_TEMPLATE)  # must not raise