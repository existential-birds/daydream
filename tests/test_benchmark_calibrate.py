"""...
"""

import asyncio
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
    assert gold  # the real fixture is present to compare against
    for tok in ("Cache key tenant-scoped", "0e2356faadfbf30a"):
        assert tok not in text


def test_fixture_has_provenance_note():
    note = (_FIXTURE / "PROVENANCE.md").read_text()
    assert "reviewer" in note.lower() and "source-free" in note.lower()


def test_every_pair_renders_within_24kib():
    # Uses the loader/fixture-load API built in Task 2; marker for the executor.
    from daydream.benchmark.harbor.calibrate import _load_fixture, _load_judge_template
    sr = _load_judge_template()
    for p in _load_fixture():
        sr.render_pair_prompt(p["gold"], p["candidate"], template=sr.JUDGE_PROMPT_TEMPLATE)  # must not raise


def test_loader_resolves_sibling_verifier_core():
    from daydream.benchmark.harbor.calibrate import _load_judge_template
    sr = _load_judge_template()
    assert sr.__name__ == "score_review"
    assert sr.verifier_core.CONFIDENCE_THRESHOLD == 0.7


def test_client_builder_threads_http_seam():
    from daydream.benchmark.harbor.calibrate import _build_calibration_client
    calls = []

    class Fake:
        async def post(self, url, *, headers, json, timeout):
            calls.append(1)
            content = '{"match": false, "confidence": 0.2, "reasoning": "n"}'
            body = {"choices": [{"message": {"content": content}}]}
            return type("R", (), {"status_code": 200, "text": "ok",
                                  "json": lambda self: body})()
    env = {"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_MODEL": "m",
           "DAYDREAM_JUDGE_API_KEY": "k", "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9"}
    client = _build_calibration_client(env, http=Fake())
    raw = asyncio.run(client.complete_json(user="<p>"))
    assert calls == [1]                      # the fake seam was used, not a real socket
    assert raw == {"match": False, "confidence": 0.2, "reasoning": "n"}


def test_client_builder_fails_closed_without_model_or_key():
    from daydream.benchmark.harbor.calibrate import _build_calibration_client, _load_judge_template
    sr = _load_judge_template()
    with pytest.raises(sr.VerifierError):
        _build_calibration_client({"DAYDREAM_JUDGE_MODEL": "m"})  # no API key


def test_judge_host_resolved_from_env():
    from daydream.benchmark.harbor.calibrate import _judge_host_from_env
    assert _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "openai-compatible",
                                 "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9"}) == "127.0.0.1"
    assert _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "anthropic"}) == "api.anthropic.com"


def test_out_of_allowlist_host_rejected(tmp_path):
    from daydream.benchmark.harbor.calibrate import (
        _load_workspace_allowlist,
        _validate_workspace_host,
    )
    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    (ws / "benchmark.yaml").write_text(json.dumps({
        "schema_version": 1,
        "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
        "created_at": "2026-08-21T12:00:00Z",
        "source": {"provider": "github", "hostname": "github.com",
                    "repository": "OWNER/REPO", "repository_id": None,
                    "visibility": "unresolved"},
        "privacy": {"classification": "confidential",
                     "reviewer_data": "source_snapshot",
                     "reviewer_allowed_hosts": ["review.example"],
                     "judge_data": "finding_text_and_location_only",
                     "judge_allowed_hosts": ["127.0.0.1"],
                     "archive": "disabled", "uploads": "disabled"},
        "pull_requests": [],
        "cases": [],
    }))
    allow = _load_workspace_allowlist(ws)
    assert allow == ["127.0.0.1"]
    with pytest.raises(ValueError):
        _validate_workspace_host(allow, "evil.example")
    _validate_workspace_host(allow, "127.0.0.1")  # in-allowlist passes
