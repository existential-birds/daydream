"""...
"""

import asyncio
import hashlib
import json
import re
import stat
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parents[1] / "daydream" / "benchmark" / "harbor" / "calibration"

REQUIRED_CATEGORIES = {
    "wording_location", "related_distinct", "negation", "same_class_different_components",
    "severity_disagreement", "locationless", "delimiter_injection", "near_miss",
}
_CRED = re.compile(r"sk-ant-|sk-or-|Bearer |x-api-key")


def test_fixture_is_24_pairs_12_12():
    from daydream.benchmark.harbor.calibrate import _load_fixture
    pairs = _load_fixture()
    assert len(pairs) == 24
    labels = [p["label"] for p in pairs]
    assert labels.count("match") == 12 and labels.count("nonmatch") == 12


def test_fixture_covers_all_eight_categories():
    from daydream.benchmark.harbor.calibrate import _load_fixture
    pairs = _load_fixture()
    assert {p["category"] for p in pairs} >= REQUIRED_CATEGORIES


def test_fixture_is_source_free():
    text = (_FIXTURE / "pairs.json").read_text()
    assert not _CRED.search(text)
    # no content lifted from the real source-derived golden-review.json fixture
    gold = (_FIXTURE.parents[0] / "templates" / "tests" / "golden-review.json").read_text()
    assert gold  # the real fixture is present to compare against
    for tok in ("Cache key tenant-scoped", "0e2356faadfbf30a"):
        assert tok not in text


def test_fixture_provenance_declares_unverified_llm_origin():
    from daydream.benchmark.harbor.calibrate import _load_fixture, _load_provenance

    pairs = _load_fixture()
    prov = _load_provenance(pairs)
    assert prov["origin"] == "llm_generated"
    assert prov["human_reviewed"] is False
    assert prov["labels"] == "unverified"
    # Pair-attestable facts are derived from the passed pairs, not a fixed file.
    assert prov["class_balance"] == {"match": 12, "nonmatch": 12}
    assert prov["categories"] == 8
    assert len(pairs) == 24          # the 24 pairs survive the shape change


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


def test_judge_pairs_makes_exactly_72_calls():
    from daydream.benchmark.harbor.calibrate import (
        _judge_pairs,
        _load_fixture,
        _load_judge_template,
    )

    sr = _load_judge_template()
    pairs = _load_fixture()
    calls = []

    class Fake:
        async def post(self, url, *, headers, json, timeout):
            calls.append(1)
            content = '{"match": true, "confidence": 0.9, "reasoning": "x"}'
            body = {"choices": [{"message": {"content": content}}]}
            return type("R", (), {"status_code": 200, "text": "ok",
                                  "json": lambda self: body})()
    client = sr.OpenAIJudgeClient("k", "m", base_url="http://127.0.0.1:9", http=Fake())
    per_pair = _judge_pairs(sr, client, pairs, attempts=3)
    assert len(per_pair) == 24
    assert all(len(runs) == 3 for runs in per_pair)
    assert len(calls) == 72
    assert all(r.match is True and r.confidence == 0.9 for runs in per_pair for r in runs)


@pytest.fixture(scope="module")
def sr():
    from daydream.benchmark.harbor.calibrate import _load_judge_template
    return _load_judge_template()


def _v(sr, match, conf):
    return sr.verifier_core.Verdict(
        gold_id="g", candidate_id="c", match=match, confidence=conf, reasoning="x"
    )


def test_majority_and_stability(sr):
    from daydream.benchmark.harbor.calibrate import _majority_label, _per_pair_stable
    assert _majority_label([_v(sr, True, .9), _v(sr, True, .8), _v(sr, False, .6)]) is True
    assert _majority_label([_v(sr, False, .3), _v(sr, False, .2), _v(sr, True, .9)]) is False
    flips = [_v(sr, True, .9), _v(sr, True, .5), _v(sr, True, .9)]  # retained [T,F,T]
    stable = [_v(sr, True, .9), _v(sr, True, .9), _v(sr, True, .5)]  # retained [T,T,F]
    assert _per_pair_stable(flips, sr.verifier_core.CONFIDENCE_THRESHOLD) is False
    assert _per_pair_stable(stable, sr.verifier_core.CONFIDENCE_THRESHOLD) is True


def test_balanced_accuracy_and_confusion(sr):
    from daydream.benchmark.harbor.calibrate import _class_balanced_accuracy, _confusion_matrix
    assert _class_balanced_accuracy({"tp": 12, "fp": 0, "tn": 12, "fn": 0}) == pytest.approx(1.0)
    assert _class_balanced_accuracy({"tp": 9, "fp": 3, "tn": 12, "fn": 0}) == pytest.approx(0.9)
    assert _confusion_matrix([True, True, False, False],
                             [True, False, True, False]) == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_pass_gate_reports_instability_only(sr):
    from daydream.benchmark.harbor.calibrate import _pass_gate
    pairs = [{"label": "match"}, {"label": "match"}, {"label": "nonmatch"}]
    runs = [
        [_v(sr, True, .9), _v(sr, True, .5), _v(sr, True, .9)],
        [_v(sr, True, .9), _v(sr, True, .9), _v(sr, True, .9)],
        [_v(sr, False, .2), _v(sr, False, .2), _v(sr, False, .2)],
    ]
    passed, failures, _, _ = _pass_gate(
        pairs, runs, sr.verifier_core.CONFIDENCE_THRESHOLD
    )
    assert passed is False
    assert list(failures) == ["instability"]
    assert 0 in failures["instability"]


def _env():
    return {"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_MODEL": "m",
            "DAYDREAM_JUDGE_API_KEY": "k", "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9"}


def test_invalidation_inputs_and_determinism():
    from daydream.benchmark.harbor.calibrate import (
        _invalidation_inputs,
        _load_fixture,
        _load_judge_template,
        _serialize_inputs,
    )
    sr, pairs = _load_judge_template(), _load_fixture()
    a = _serialize_inputs(_invalidation_inputs(_env(), pairs, sr))
    b = _serialize_inputs(_invalidation_inputs(_env(), pairs, sr))
    assert a == b


def test_receipt_written_atomic_0600_and_current(tmp_path):
    from daydream.benchmark.harbor.calibrate import (
        _build_receipt,
        _invalidation_inputs,
        _load_fixture,
        _load_judge_template,
        _write_receipt,
        is_receipt_current,
    )
    sr, pairs = _load_judge_template(), _load_fixture()
    receipt = _build_receipt(sr, pairs, _env(), passed=True,
                             balanced_accuracy=0.9583,
                             confusion={"tp": 12, "fp": 0, "tn": 12, "fn": 0},
                             disagreements=[])
    p = _write_receipt(tmp_path, receipt)
    assert (tmp_path / "runtime" / "calibration-receipt.json").exists()
    assert stat.S_IMODE(
        (tmp_path / "runtime" / "calibration-receipt.json").stat().st_mode
    ) == 0o600
    assert is_receipt_current(p, _invalidation_inputs(_env(), pairs, sr)) is True


def test_receipt_invalidated_on_input_change(tmp_path):
    from daydream.benchmark.harbor.calibrate import (
        _build_receipt,
        _invalidation_inputs,
        _load_fixture,
        _load_judge_template,
        _write_receipt,
        is_receipt_current,
    )
    sr, pairs = _load_judge_template(), _load_fixture()
    _write_receipt(
        tmp_path,
        _build_receipt(sr, pairs, _env(), passed=True,
                       balanced_accuracy=0.9583,
                       confusion={"tp": 12, "fp": 0, "tn": 12, "fn": 0},
                       disagreements=[]),
    )
    changed = dict(_env())
    changed["DAYDREAM_JUDGE_MODEL"] = "other-model"
    assert is_receipt_current(
        tmp_path / "runtime" / "calibration-receipt.json",
        _invalidation_inputs(changed, pairs, sr),
    ) is False


def test_diagnostic_receipt_invalidates_on_fixture_content_change(tmp_path, monkeypatch):
    from daydream.benchmark.harbor import calibrate

    sr = calibrate._load_judge_template()
    pairs = calibrate._load_fixture()
    env = {"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_MODEL": "m",
           "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9", "DAYDREAM_JUDGE_API_KEY": "k"}
    receipt = calibrate._build_receipt(
        sr, pairs, env, passed=True, balanced_accuracy=1.0,
        confusion={"tp": 12, "fp": 0, "tn": 12, "fn": 0}, disagreements=[])
    path = calibrate._write_receipt(tmp_path, receipt)

    # Diagnostic receipts carry honest machine-readable provenance matching the fixture.
    assert receipt["provenance"]["origin"] == "llm_generated"
    assert receipt["provenance"]["human_reviewed"] is False
    assert receipt["provenance"]["labels"] == "unverified"
    # Full fixture content (provenance + all pairs) is part of the invalidation inputs.
    assert receipt["inputs"]["fixture_sha256"] == calibrate._fixture_sha256(pairs)
    assert receipt["inputs"]["fixture_provenance"]["origin"] == "llm_generated"

    assert calibrate.is_receipt_current(path, calibrate._invalidation_inputs(env, pairs, sr))

    # A provenance-block change (the fixture now claims human_reviewed=true)
    # must invalidate an existing diagnostic receipt, even though the ordered
    # gold/candidate/label triples (label_sha256) are unchanged.
    modified_provenance = dict(calibrate._load_provenance(pairs))
    modified_provenance["human_reviewed"] = True
    monkeypatch.setattr(calibrate, "_load_provenance", lambda pairs: modified_provenance)
    monkeypatch.setattr(
        calibrate,
        "_fixture_sha256",
        lambda pairs: hashlib.sha256(b"fixture-with-human-revised-provenance").hexdigest(),
    )
    assert not calibrate.is_receipt_current(
        path, calibrate._invalidation_inputs(env, pairs, sr))


def test_receipt_has_no_credentials_or_source():
    from daydream.benchmark.harbor.calibrate import _build_receipt, _load_fixture, _load_judge_template
    sr, pairs = _load_judge_template(), _load_fixture()
    receipt = _build_receipt(sr, pairs, _env(), passed=True,
                             balanced_accuracy=0.9583, confusion={"tp": 12, "fp": 0, "tn": 12, "fn": 0},
                             disagreements=[])
    text = json.dumps(receipt)
    assert "sk-ant-" not in text and "sk-or-" not in text and "Bearer " not in text
    assert "DAYDREAM_JUDGE_API_KEY" not in text and "Cache key not tenant-scoped" not in text


@pytest.fixture
def ws_factory():
    """Build a workspace with ``127.0.0.1`` on the judge host allowlist."""

    def _build(tmp_path):
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
        return ws

    return _build


def _scripted_http(responses):
    i = [0]
    _jsonlib = json  # module reference; ``json`` below is the keyword payload

    class Fake:
        async def post(self, url, *, headers, json, timeout):
            v = responses[i[0] % len(responses)]
            i[0] += 1
            body = {"choices": [{"message": {"content": _jsonlib.dumps(v)}}]}
            return type("R", (), {"status_code": 200, "text": "ok",
                                  "json": lambda self: body})()
    return Fake(), i


def _scripted_responses(pairs, *, mislabel_count=0):
    """Build the 72-response scripted verdict list (3 calls per pair).

    ``mislabel_count`` match pairs (from the front) are flipped to nonmatch.

    """
    responses = []
    wrong = 0
    for p in pairs:
        m = p["label"] == "match"
        if m and wrong < mislabel_count:
            m = False
            wrong += 1
        verdict = {"match": m, "confidence": 0.95 if m else 0.1, "reasoning": "x"}
        responses.extend([verdict] * 3)
    return responses


class TestCalibrateAcceptance:
    """The six acceptance-mandated fake-endpoint calibration cases.

    Each drives the full packaged judge path with an injected fake http client;
    none opens a socket or makes a paid call.
    """

    def test_run_calibration_pass_writes_receipt(self, tmp_path, ws_factory):
        from daydream.benchmark.harbor.calibrate import _load_fixture, run_calibration
        responses = _scripted_responses(_load_fixture())
        assert len(responses) == 72
        fake, counter = _scripted_http(responses)
        code = run_calibration(ws_factory(tmp_path), yes=True, env=_env(), http=fake)
        assert code == 0
        assert (tmp_path / "ws" / "runtime" / "calibration-receipt.json").exists()
        assert counter[0] == 72

    def test_accuracy_failure_no_receipt(self, tmp_path, ws_factory):
        from daydream.benchmark.harbor.calibrate import _load_fixture, run_calibration
        responses = _scripted_responses(_load_fixture(), mislabel_count=3)
        assert len(responses) == 72
        fake, counter = _scripted_http(responses)
        code = run_calibration(ws_factory(tmp_path), yes=True, env=_env(), http=fake)
        assert code == 1
        assert not (tmp_path / "ws" / "runtime" / "calibration-receipt.json").exists()
        assert counter[0] == 72

    def test_confirmation_refuses_before_any_call(self, tmp_path, ws_factory):
        from daydream.benchmark.harbor.calibrate import run_calibration
        fake, counter = _scripted_http([{"match": True, "confidence": 0.9, "reasoning": "x"}])
        code = run_calibration(
            ws_factory(tmp_path), yes=False, confirm=lambda _: False, env=_env(), http=fake
        )
        assert code == 1
        assert counter[0] == 0
        assert not (tmp_path / "ws" / "runtime" / "calibration-receipt.json").exists()

    def test_acceptance_instability_failure(self, tmp_path, ws_factory, capsys):
        from daydream.benchmark.harbor.calibrate import _load_fixture, run_calibration
        pairs = _load_fixture()
        responses = []
        for idx, p in enumerate(pairs):
            if idx == 0:  # first match pair: retained [T,F,T] -> unstable
                responses += [{"match": True, "confidence": 0.9, "reasoning": "x"},
                              {"match": True, "confidence": 0.5, "reasoning": "x"},
                              {"match": True, "confidence": 0.9, "reasoning": "x"}]
            else:
                m = p["label"] == "match"
                responses += [{"match": m, "confidence": 0.95 if m else 0.1,
                               "reasoning": "x"} for _ in range(3)]
        fake, counter = _scripted_http(responses)
        code = run_calibration(ws_factory(tmp_path), yes=True, env=_env(), http=fake)
        assert code == 1
        err = capsys.readouterr().err
        assert "instability" in err and "0" in err
        assert not (tmp_path / "ws" / "runtime" / "calibration-receipt.json").exists()

    def test_acceptance_invalidation_through_gate(self, tmp_path, ws_factory):
        from daydream.benchmark.harbor.calibrate import (
            _invalidation_inputs,
            _load_fixture,
            _load_judge_template,
            is_receipt_current,
            run_calibration,
        )
        pairs = _load_fixture()
        responses = _scripted_responses(pairs)
        fake, _ = _scripted_http(responses)
        assert run_calibration(ws_factory(tmp_path), yes=True, env=_env(), http=fake) == 0
        receipt = tmp_path / "ws" / "runtime" / "calibration-receipt.json"
        assert is_receipt_current(
            receipt, _invalidation_inputs(_env(), pairs, _load_judge_template())
        ) is True
        changed = dict(_env())
        changed["DAYDREAM_JUDGE_MODEL"] = "other"
        assert is_receipt_current(
            receipt, _invalidation_inputs(changed, pairs, _load_judge_template())
        ) is False

    def test_acceptance_zero_source_leakage(self, tmp_path, ws_factory):
        from daydream.benchmark.harbor.calibrate import _load_fixture, run_calibration
        responses = _scripted_responses(_load_fixture())
        fake, _ = _scripted_http(responses)
        assert run_calibration(ws_factory(tmp_path), yes=True, env=_env(), http=fake) == 0
        receipt = (tmp_path / "ws" / "runtime" / "calibration-receipt.json").read_text()
        assert not re.search(r"sk-ant-|sk-or-|Bearer |x-api-key", receipt)
        assert "Cache key not tenant-scoped" not in receipt
        assert "DAYDREAM_JUDGE_API_KEY" not in receipt


def test_calibrate_judge_subparser_and_flags():
    from daydream.benchmark.cli import _build_benchmark_parser
    args = _build_benchmark_parser().parse_args(["calibrate-judge", "/ws", "--yes"])
    assert args.subcommand == "calibrate-judge" and str(args.dir) == "/ws" and args.yes is True


def test_calibrate_judge_refuses_without_tty_or_yes(tmp_path, monkeypatch, capsys):
    from daydream.benchmark import cli
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    code = cli._handle_benchmark_command(["calibrate-judge", str(tmp_path)])
    assert code == 1
    assert "requires TTY confirmation or --yes" in capsys.readouterr().err


def test_calibrate_judge_handler_forwards_yes_and_dir(tmp_path, monkeypatch, capsys):
    import daydream.benchmark.harbor.calibrate as cal
    from daydream.benchmark import cli
    seen = {}

    def fake_run(workspace, *, yes, env, http):
        seen.update(ws=str(workspace), yes=yes)
        return 0
    monkeypatch.setattr(cal, "run_calibration", fake_run)
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    code = cli._handle_benchmark_command(["calibrate-judge", str(tmp_path), "--yes"])
    capsys.readouterr()
    assert code == 0 and seen == {"ws": str(tmp_path), "yes": True}
