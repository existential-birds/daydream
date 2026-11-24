"""Contracts for the OpenRouter Decisions (Jev / System 1) client and shadow-calibration harness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest

from daydream.config import DEFAULT_SYSTEM1_MODEL
from daydream.dataset import LocalRecordStore
from daydream.eval.system1_calibration import (
    CLAIM_TRIAGE_QUESTIONS,
    CROSS_ASSIGNMENT_QUESTIONS,
    TARGET_TRIAGE_QUESTIONS,
    CalibrationReport,
    ShadowDataset,
    compute_brier_and_ece,
    discover_run_dirs,
    load_shadow_cases_from_runs,
    load_shadow_cases_from_store,
    run_shadow_calibration,
)
from daydream.system1 import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionOutcome,
    DecisionQuestion,
    NoulQuestion,
    RefusalAnswer,
    ScoreAnswer,
    ScoreQuestion,
    System1Client,
    System1Config,
    fits_context_budget,
    injection_detected,
    parse_decisions_response,
    path_is_hard_denied,
    with_injection_guard,
)
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, EvidenceRecorder, default_trajectory_path
from tests.harness.dataset import run_record
from tests.harness.record_projection import projection_run
from tests.harness.scripts import cli_main


def test_system1_config_environment_parsing_and_validation(caplog: pytest.LogCaptureFixture) -> None:
    """Host env parsing defaults off, hides secrets from repr, degrades bad env vars, and validates constructors."""
    default = System1Config.from_environment({})
    assert default.enabled is False
    assert default.model == DEFAULT_SYSTEM1_MODEL
    assert default.api_key is None

    with caplog.at_level("WARNING"):
        parsed = System1Config.from_environment({
            "DAYDREAM_SYSTEM1_ENABLED": "true",
            "DAYDREAM_SYSTEM1_MODEL": "typesafe/jev-1.13",
            "DAYDREAM_SYSTEM1_ENDPOINT": "https://example.invalid/decisions",
            "OPENROUTER_API_KEY": "sk-or-secret-token-value",
            "DAYDREAM_SYSTEM1_TIMEOUT_S": "not-a-float",
            "DAYDREAM_SYSTEM1_BATCH_TIMEOUT_S": "-2",
            "DAYDREAM_SYSTEM1_MAX_CONCURRENCY": "0",
            "DAYDREAM_SYSTEM1_ZDR": "1",
            "DAYDREAM_SYSTEM1_DATA_COLLECTION": "invalid-policy",
        })
    assert parsed.enabled is True
    assert parsed.api_key == "sk-or-secret-token-value"
    assert "sk-or-secret-token-value" not in repr(parsed)
    assert parsed.zdr is True
    assert parsed.data_collection is None
    assert parsed.timeout_s == 5.0
    assert parsed.batch_timeout_s == 15.0
    assert parsed.max_concurrency == 8
    assert "invalid DAYDREAM_SYSTEM1_TIMEOUT_S" in caplog.text
    assert "non-positive DAYDREAM_SYSTEM1_BATCH_TIMEOUT_S" in caplog.text
    assert "non-positive DAYDREAM_SYSTEM1_MAX_CONCURRENCY" in caplog.text
    assert "invalid DAYDREAM_SYSTEM1_DATA_COLLECTION" in caplog.text

    allow_cfg = System1Config.from_environment({
        "DAYDREAM_SYSTEM1_DATA_COLLECTION": "deny",
        "DAYDREAM_SYSTEM1_MAX_CONCURRENCY": "bad",
    })
    assert allow_cfg.data_collection == "deny"
    assert allow_cfg.max_concurrency == 8
    assert System1Config.from_environment({"DAYDREAM_SYSTEM1_DATA_COLLECTION": "allow"}).data_collection == "allow"

    bad_configs: tuple[dict[str, Any], ...] = (
        {"model": "   "},
        {"endpoint": ""},
        {"timeout_s": 0.0},
        {"batch_timeout_s": float("nan")},
        {"timeout_s": True},
        {"max_concurrency": 0},
        {"max_request_tokens": True},
        {"data_collection": "other"},
    )
    for bad_kwargs in bad_configs:
        with pytest.raises(ValueError):
            System1Config(**bad_kwargs)

    with pytest.raises(ValueError):
        ChoiceQuestion(criteria={})
    with pytest.raises(ValueError):
        ChoiceQuestion(criteria={"": "blank key"})
    with pytest.raises(ValueError):
        ScoreQuestion(criteria=[])


async def test_openrouter_decisions_wire_roundtrip_and_recorder_billing(tmp_path: Path) -> None:
    """Wire payload matches OpenRouter /api/alpha/decisions, parses all primitives + refusal, and bills recorder."""
    captured: list[tuple[httpx.URL, dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append((request.url, dict(request.headers), body))
        if body["state"] == "no-cost-state":
            return httpx.Response(200, json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "is_bug": {"type": "noul", "noul": 0.25},
                    "refused_q": {"type": "refusal"},
                },
                "usage": {"input_tokens": 1000, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 200}},
            })
        return httpx.Response(200, json={
            "id": "gen-dec-1790015143-abc",
            "model": "typesafe/jev-1.13-20260917",
            "provider": "TypeSafe",
            "answers": {
                "is_bug": {"type": "noul", "noul": 0.96, "confidence": 0.92},
                "team": {
                    "type": "choice",
                    "choice": "payments",
                    "confidence": 0.75,
                    "probabilities": {"payments": 0.84, "frontend": 0.16},
                },
                "urgency": {
                    "type": "score",
                    "score": 1.99,
                    "confidence": 0.99,
                    "probabilities": {"0": 0.0, "1": 0.01, "2": 0.99},
                    "legend": {"0": "Low", "1": "Med", "2": "High"},
                },
            },
            "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
        })

    config = System1Config(
        enabled=True,
        api_key="sk-or-test-key",
        zdr=True,
        data_collection="deny",
    )
    client = System1Client(config, transport=httpx.MockTransport(handler))
    questions: dict[str, DecisionQuestion] = {
        "is_bug": NoulQuestion(instructions="Is defect?", true_criteria="Broken", false_criteria="Question"),
        "team": ChoiceQuestion(instructions="Owner?", criteria={"payments": "Billing", "frontend": "UI"}),
        "urgency": ScoreQuestion(instructions="Urgency?", criteria=["Low", "Med", "High"]),
    }

    recorder = EvidenceRecorder(
        path=default_trajectory_path(tmp_path / "evidence", "run-s1"),
        run_flow=DaydreamRunFlow.DEEP,
        target_dir=tmp_path / "evidence",
        agent_model_name="",
        session_id="run-s1",
    )
    async with recorder:
        outcome = await client.evaluate(
            {"ticket": "Double charge"},
            questions,
            phase=DaydreamPhase.REVIEW,
            session_id="session-1234",
        )
        fallback_cost_outcome = await client.evaluate(
            "no-cost-state",
            {"is_bug": NoulQuestion(), "refused_q": NoulQuestion(instructions="Maybe refused")},
            phase=DaydreamPhase.CHALLENGE,
        )

    assert outcome.ok is True
    assert outcome.response is not None
    assert outcome.response.id == "gen-dec-1790015143-abc"
    assert outcome.response.provider == "TypeSafe"
    assert outcome.noul("is_bug") == pytest.approx(0.96)
    assert outcome.noul("team") is None
    assert outcome.is_refused("is_bug") is False
    team = outcome.choice("team")
    assert isinstance(team, ChoiceAnswer) and team.choice == "payments" and team.confidence == pytest.approx(0.75)
    assert outcome.choice("is_bug") is None
    urgency = outcome.score("urgency")
    assert isinstance(urgency, ScoreAnswer) and urgency.score == pytest.approx(1.99)
    assert outcome.score("is_bug") is None

    # Verify exact wire payload & headers.
    url, headers, sent = captured[0]
    assert str(url) == "https://openrouter.ai/api/alpha/decisions"
    assert headers["authorization"] == "Bearer sk-or-test-key"
    assert sent["model"] == DEFAULT_SYSTEM1_MODEL
    assert sent["provider"] == {"zdr": True, "data_collection": "deny"}
    assert sent["session_id"] == "session-1234"
    assert sent["questions"]["is_bug"]["type"] == "noul"
    assert sent["questions"]["team"]["type"] == "choice"
    assert sent["questions"]["urgency"]["type"] == "score"

    # Second call synthesizes cost from DEFAULT_SYSTEM1_MODEL ($0.042 / 1M input) and parses RefusalAnswer.
    assert fallback_cost_outcome.ok is True
    assert fallback_cost_outcome.response is not None
    assert fallback_cost_outcome.response.usage.cached_tokens == 200
    assert fallback_cost_outcome.response.usage.cost_usd == pytest.approx(1000 * 0.042 / 1_000_000)
    assert isinstance(fallback_cost_outcome.response.answers["refused_q"], RefusalAnswer)
    assert fallback_cost_outcome.is_refused("refused_q") is True
    assert fallback_cost_outcome.noul("refused_q") is None

    # Recorder ledger accumulated both phases cleanly.
    review_share = recorder.usage.phases[DaydreamPhase.REVIEW.value].usage()
    challenge_share = recorder.usage.phases[DaydreamPhase.CHALLENGE.value].usage()
    assert review_share["prompt_tokens"] == 476
    assert review_share["completion_tokens"] == 70
    assert review_share["cost_usd"] == pytest.approx(0.000019992)
    assert challenge_share["prompt_tokens"] == 1000
    assert challenge_share["cached_tokens"] == 200
    assert challenge_share["cost_usd"] == pytest.approx(0.000042)


async def test_system1_context_ceilings_fail_open_and_cancellation() -> None:
    """Context limits, HTTP/transport/timeout/parse failures fail open; CancelledError propagates."""
    q = {"flag": NoulQuestion(instructions="Check")}

    # Empty questions is a programmer error.
    with pytest.raises(ValueError, match="at least one"):
        await System1Client(System1Config(enabled=True, api_key="k")).evaluate("s", {})

    # Disabled and missing key never touch transport.
    called = 0

    def counting_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called += 1
        return httpx.Response(200, json={})

    disabled = await System1Client(System1Config(enabled=False, api_key="k"),
                                   transport=httpx.MockTransport(counting_handler)).evaluate("s", q)
    assert disabled.ok is False and disabled.fallback_reason == "disabled"
    assert disabled.noul("flag") is None and disabled.choice("flag") is None
    assert disabled.score("flag") is None and disabled.is_refused("flag") is False

    no_key = await System1Client(System1Config(enabled=True, api_key=None),
                                 transport=httpx.MockTransport(counting_handler)).evaluate("s", q)
    assert no_key.ok is False and no_key.fallback_reason == "missing_api_key"

    # Context ceilings: per-question limit and total request limit.
    tight_cfg = System1Config(enabled=True, api_key="k", max_request_tokens=100, max_question_tokens=40)
    assert fits_context_budget("x" * 200, q, tight_cfg) is False
    over_q = await System1Client(tight_cfg, transport=httpx.MockTransport(counting_handler)).evaluate("x" * 200, q)
    assert over_q.ok is False and over_q.fallback_reason == "context_limit"

    many_q = {f"q{i}": NoulQuestion(instructions="a" * 60) for i in range(6)}
    assert fits_context_budget("ok", many_q, tight_cfg) is False
    assert called == 0

    # Zero/negative timeout override fails open immediately.
    neg_timeout = await System1Client(System1Config(enabled=True, api_key="k")).evaluate(
        "s", q, timeout_override_s=0.0,
    )
    assert neg_timeout.fallback_reason == "timeout"

    # HTTP 429, transport error, timeout, malformed response.
    def http_error(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": 429, "message": "Rate limited"}})

    res_http = await System1Client(System1Config(enabled=True, api_key="k"),
                                   transport=httpx.MockTransport(http_error)).evaluate("s", q)
    assert res_http.ok is False and res_http.fallback_reason == "http_error"

    def conn_error(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=req)

    res_net = await System1Client(System1Config(enabled=True, api_key="k"),
                                  transport=httpx.MockTransport(conn_error)).evaluate("s", q)
    assert res_net.ok is False and res_net.fallback_reason == "transport_error"

    async def slow_handler(_req: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5.0)
        return httpx.Response(200, json={})

    res_timeout = await System1Client(System1Config(enabled=True, api_key="k", timeout_s=0.05),
                                      transport=httpx.MockTransport(slow_handler)).evaluate("s", q)
    assert res_timeout.ok is False and res_timeout.fallback_reason == "timeout"

    res_bad = await System1Client(
        System1Config(enabled=True, api_key="k"),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"model": "x", "answers": {}, "usage": {}})),
    ).evaluate("s", q)
    assert res_bad.ok is False and res_bad.fallback_reason == "malformed_response"

    # CancelledError must propagate, never be converted into a fail-open outcome.
    async def cancel_handler(_req: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await System1Client(System1Config(enabled=True, api_key="k"),
                            transport=httpx.MockTransport(cancel_handler)).evaluate("s", q)


@pytest.mark.parametrize(
    ("payload", "questions", "match"),
    [
        ("not-a-dict", {"q": NoulQuestion()}, "JSON object"),
        ({"model": "", "answers": {}, "usage": {}}, {"q": NoulQuestion()}, "non-empty model"),
        ({"model": "m", "answers": "bad", "usage": {}}, {"q": NoulQuestion()}, "answers object"),
        ({"model": "m", "answers": {}, "usage": {}}, {"q": NoulQuestion()}, "missing requested answer"),
        ({"model": "m", "answers": {"q": "bad"}, "usage": {}}, {"q": NoulQuestion()}, "must be an object"),
        ({"model": "m", "answers": {"q": {"type": "choice"}}, "usage": {}},
         {"q": NoulQuestion()}, "expected type 'noul'"),
        ({"model": "m", "answers": {"q": {"type": "noul", "noul": True}}, "usage": {}},
         {"q": NoulQuestion()}, "number"),
        ({"model": "m", "answers": {"q": {"type": "noul", "noul": 1.5}}, "usage": {}},
         {"q": NoulQuestion()}, r"\[0, 1\]"),
        ({"model": "m", "answers": {"q": {"type": "noul", "noul": float("nan")}}, "usage": {}},
         {"q": NoulQuestion()}, "finite"),
        ({"model": "m", "answers": {"q": {"type": "noul"}}, "usage": {}},
         {"q": ChoiceQuestion(criteria={"a": "A"})}, "expected type 'choice'"),
        ({"model": "m", "answers": {"q": {"type": "choice", "choice": "z"}}, "usage": {}},
         {"q": ChoiceQuestion(criteria={"a": "A"})}, "declared criterion"),
        ({"model": "m", "answers": {"q": {"type": "choice", "choice": "a", "probabilities": "bad"}}, "usage": {}},
         {"q": ChoiceQuestion(criteria={"a": "A"})}, "must be an object"),
        ({"model": "m", "answers": {"q": {"type": "noul"}}, "usage": {}},
         {"q": ScoreQuestion(criteria=["0", "1"])}, "expected type 'score'"),
        ({"model": "m", "answers": {"q": {"type": "score", "score": 0.5, "legend": "bad"}}, "usage": {}},
         {"q": ScoreQuestion(criteria=["0", "1"])}, "legend must be an object"),
        ({"model": "m", "answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": "bad"},
         {"q": NoulQuestion()}, "usage must be an object"),
        ({"model": "m", "answers": {"q": {"type": "noul", "noul": 0.5}},
          "usage": {"input_tokens": -1, "output_tokens": 0}}, {"q": NoulQuestion()}, "non-negative integer"),
        ({"model": "m", "answers": {"q": {"type": "noul", "noul": 0.5}},
          "usage": {"input_tokens": 1, "output_tokens": 0, "cost": -0.1}}, {"q": NoulQuestion()}, "non-negative"),
    ],
)
def test_parse_decisions_response_rejects_malformed_payloads(
    payload: Any, questions: dict[str, Any], match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_decisions_response(payload, questions, request_model=DEFAULT_SYSTEM1_MODEL, latency_s=0.01)


async def test_evaluate_many_preserves_order_and_enforces_batch_deadline() -> None:
    """Concurrent fan-out preserves ordering and times out lagging items without losing fast siblings."""
    q = {"flag": NoulQuestion()}
    assert await System1Client(System1Config(enabled=True, api_key="k")).evaluate_many([]) == []
    disabled = await System1Client(System1Config(enabled=False, api_key="k")).evaluate_many([("a", q)])
    assert [o.fallback_reason for o in disabled] == ["disabled"]
    no_key = await System1Client(System1Config(enabled=True, api_key=None)).evaluate_many([("a", q)])
    assert [o.fallback_reason for o in no_key] == ["missing_api_key"]

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        idx = body["state"]["idx"]
        if idx == 1:
            await asyncio.sleep(0.3)
        return httpx.Response(200, json={
            "model": DEFAULT_SYSTEM1_MODEL,
            "answers": {"flag": {"type": "noul", "noul": idx / 10.0}},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })

    cfg = System1Config(enabled=True, api_key="k", timeout_s=1.0, batch_timeout_s=0.12, max_concurrency=1)
    client = System1Client(cfg, transport=httpx.MockTransport(handler))
    outcomes = await client.evaluate_many([({"idx": 0}, q), ({"idx": 1}, q), ({"idx": 2}, q)], record_usage=False)
    assert len(outcomes) == 3
    assert outcomes[0].ok is True and outcomes[0].noul("flag") == pytest.approx(0.0)
    assert outcomes[1].ok is False and outcomes[1].fallback_reason == "timeout"
    assert outcomes[2].ok is False and outcomes[2].fallback_reason == "timeout"


def test_safety_guards_injection_and_hard_deny_globs() -> None:
    """Injection guard fails closed on errors/refusals/high probs; hard-deny catches security-critical paths."""
    guarded = with_injection_guard({"ok": NoulQuestion()})
    assert "prompt_injection_attempt" in guarded

    assert injection_detected(DecisionOutcome(ok=False, fallback_reason="timeout")) is True
    refused = parse_decisions_response(
        {"model": "m", "answers": {"prompt_injection_attempt": {"type": "refusal"}},
         "usage": {"input_tokens": 1, "output_tokens": 0}},
        {"prompt_injection_attempt": guarded["prompt_injection_attempt"]},
        request_model="m",
        latency_s=0.01,
    )
    assert injection_detected(DecisionOutcome(ok=True, response=refused)) is True

    high = parse_decisions_response(
        {"model": "m", "answers": {"prompt_injection_attempt": {"type": "noul", "noul": 0.8}},
         "usage": {"input_tokens": 1, "output_tokens": 0}},
        {"prompt_injection_attempt": guarded["prompt_injection_attempt"]},
        request_model="m",
        latency_s=0.01,
    )
    assert injection_detected(DecisionOutcome(ok=True, response=high)) is True

    low = parse_decisions_response(
        {"model": "m", "answers": {"prompt_injection_attempt": {"type": "noul", "noul": 0.05}},
         "usage": {"input_tokens": 1, "output_tokens": 0}},
        {"prompt_injection_attempt": guarded["prompt_injection_attempt"]},
        request_model="m",
        latency_s=0.01,
    )
    assert injection_detected(DecisionOutcome(ok=True, response=low)) is False

    for denied in (
        "src/auth/login.py",
        "auth/tokens.py",
        "services\\crypto\\sign.py",
        ".github/workflows/release.yml",
        "Dockerfile",
        "deploy/Dockerfile.prod",
        "db/migrations/0042_init.sql",
        "infra/main.tf",
        "k8s/deployment.yaml",
        "./secrets/keys.json",
        "",
    ):
        assert path_is_hard_denied(denied) is True, denied

    for allowed in ("README.md", "daydream/ui/panel.py", "tests/test_pricing.py", "docs/runtime.md"):
        assert path_is_hard_denied(allowed) is False, allowed


def _seed_archived_run(run_dir: Path, *, run_id: str, decomposed: bool = True) -> Path:
    (run_dir / "deep").mkdir(parents=True, exist_ok=True)
    diff = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n+++ b/README.md\n@@ -1,1 +1,1 @@\n-old doc\n+new doc\n"
        "diff --git a/src/auth/login.py b/src/auth/login.py\n"
        "--- a/src/auth/login.py\n+++ b/src/auth/login.py\n@@ -1,1 +1,1 @@\n-check()\n+pass\n"
        "diff --git a/src/billing.py b/src/billing.py\n"
        "--- a/src/billing.py\n+++ b/src/billing.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 0\n"
        "diff --git a/assets/logo.png b/assets/logo.png\n"
        "Binary files a/assets/logo.png and b/assets/logo.png differ\n"
    )
    (run_dir / "diff.patch").write_text(diff, encoding="utf-8")
    confirmed_claim = {
        "uid": f"{run_id}:claim:1",
        "title": "Division by zero in billing",
        "disposition": "confirmed",
        "work_id": "work:cross" if decomposed else "work:1",
        "payload": {
            "file": "src/billing.py",
            "line": 1,
            "severity": "high",
            "claim": "x is zero",
            "trigger": "call charge()",
            "consequence": "ZeroDivisionError",
            "grounds": "line 1 sets x = 0",
        },
    }
    rejected_claim = {
        "uid": f"{run_id}:claim:2",
        "title": "Minor doc nit",
        "disposition": "rejected",
        "work_id": "work:1",
        "payload": {
            "file": "README.md",
            "line": 1,
            "severity": "low",
            "claim": "Word choice",
            "trigger": "read",
            "consequence": "none",
            "grounds": "stylistic",
        },
    }
    unresolved_claim = {
        "uid": f"{run_id}:claim:3",
        "title": "Unresolved",
        "disposition": "unresolved",
        "work_id": "work:1",
        "payload": {"file": "src/billing.py", "line": 1, "severity": "medium"},
    }
    sealed = {
        "run_id": run_id,
        "claims": [confirmed_claim, rejected_claim, unresolved_claim],
        "findings": [confirmed_claim],
        "work": [
            {"work_id": "work:1", "kind": "discovery", "targets": ["README.md", "src/billing.py"]},
            {"work_id": "work:cross", "kind": "cross_assignment", "targets": []},
        ],
        "settings": {"assignment_policy": {"decomposed": decomposed}},
    }
    (run_dir / "deep" / "committed-review.json").write_text(json.dumps(sealed), encoding="utf-8")
    return run_dir


def test_shadow_calibration_runs_store_metrics_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end shadow calibration over archived runs + LocalRecordStore computes ECE/Brier/thresholds & CLI works."""
    assert compute_brier_and_ece([]) == (None, None)

    archive_root = tmp_path / "archive"
    run1 = _seed_archived_run(archive_root / "run-1", run_id="run-1", decomposed=True)
    _seed_archived_run(archive_root / "run-2", run_id="run-2", decomposed=False)
    assert discover_run_dirs([archive_root, run1, tmp_path / "missing"]) == sorted([run1, archive_root / "run-2"])

    dataset = load_shadow_cases_from_runs([archive_root])
    assert dataset.run_ids == ("run-1", "run-2")
    # Binary logo.png is skipped; 3 text files per run = 6 targets.
    assert len(dataset.targets) == 6
    assert sum(t.hard_denied for t in dataset.targets) == 2  # src/auth/login.py in both runs
    # Unresolved claims excluded; 2 resolved claims per run = 4 claims.
    assert len(dataset.claims) == 4
    # Only run-1 was decomposed.
    assert len(dataset.cross_assignments) == 1 and dataset.cross_assignments[0].has_cross_finding is True

    # Also test LocalRecordStore snapshot extraction (validates available diff + skips unavailable runs).
    store = LocalRecordStore(tmp_path / "store")
    rec = projection_run("store-run-1", dispositions=("accepted",))
    rec["findings"]["value"]["items"][0]["file"] = "store-run-1.py"
    store.commit_run(rec)
    store.commit_run(run_record("store-run-no-diff"))
    snap = store.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    store_dataset = load_shadow_cases_from_store(store.root, snap["snapshot_id"])
    assert len(store_dataset.targets) == 1
    assert store_dataset.targets[0].path == "store-run-1.py"
    assert store_dataset.targets[0].has_confirmed_finding is True

    def shadow_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        qkeys = set(body["questions"])
        state = body["state"]
        if qkeys == set(TARGET_TRIAGE_QUESTIONS):
            path = state["path"]
            if path == "README.md":
                ans = {
                    "non_behavioral": {"type": "noul", "noul": 0.99},
                    "cross_file_contract": {"type": "noul", "noul": 0.01},
                    "risk_tier": {"type": "choice", "choice": "mechanical"},
                    "defect_risk": {"type": "score", "score": 0.1},
                    "prompt_injection_attempt": {"type": "noul", "noul": 0.02},
                }
            elif path == "src/auth/login.py":
                # Even with high non_behavioral, hard-deny protects auth path.
                ans = {
                    "non_behavioral": {"type": "noul", "noul": 0.99},
                    "cross_file_contract": {"type": "noul", "noul": 0.01},
                    "risk_tier": {"type": "choice", "choice": "high_critical"},
                    "defect_risk": {"type": "score", "score": 3.5},
                    "prompt_injection_attempt": {"type": "noul", "noul": 0.01},
                }
            else:
                ans = {
                    "non_behavioral": {"type": "noul", "noul": 0.05},
                    "cross_file_contract": {"type": "noul", "noul": 0.60},
                    "risk_tier": {"type": "choice", "choice": "high_critical"},
                    "defect_risk": {"type": "score", "score": 3.8},
                    "prompt_injection_attempt": {"type": "noul", "noul": 0.01},
                }
        elif qkeys == set(CLAIM_TRIAGE_QUESTIONS):
            is_nit = state["claim"]["title"] == "Minor doc nit"
            ans = {
                "obviously_invalid": {"type": "noul", "noul": 0.97 if is_nit else 0.04},
                "needs_repo_search": {"type": "noul", "noul": 0.05},
                "severity": {"type": "choice", "choice": "low" if is_nit else "high"},
                "prompt_injection_attempt": {"type": "noul", "noul": 0.01},
            }
        else:
            assert qkeys == set(CROSS_ASSIGNMENT_QUESTIONS)
            ans = {
                "cross_boundary_needed": {"type": "noul", "noul": 0.85},
                "prompt_injection_attempt": {"type": "noul", "noul": 0.01},
            }
        return httpx.Response(200, json={
            "model": DEFAULT_SYSTEM1_MODEL,
            "answers": ans,
            "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.0000042},
        })

    client = System1Client(System1Config(enabled=True, api_key="k"), transport=httpx.MockTransport(shadow_handler))
    report = anyio.run(run_shadow_calibration, dataset, client)
    assert isinstance(report, CalibrationReport)
    assert report.runs_scanned == 2
    assert report.total_calls == 11
    assert report.fallback_calls == 0
    assert report.total_input_tokens == 1100
    assert report.total_cost_usd == pytest.approx(11 * 0.0000042)
    assert report.p50_latency_ms is not None and report.p95_latency_ms is not None

    # Target gate recommends skipping README.md (2/6 = 1/3) while hard-denied auth & billing stay protected.
    assert report.target_skip.hard_denied_cases == 2
    assert report.target_skip.recommended_threshold is not None
    assert report.target_skip.recommended_gated_fraction == pytest.approx(2 / 6, abs=1e-5)
    assert report.target_skip.brier_score is not None and report.target_skip.ece is not None

    # Claim gate recommends fast-rejecting the 2 rejected doc nits while keeping 100% of confirmed claims.
    assert report.claim_fast_reject.recommended_threshold is not None
    assert report.claim_fast_reject.recommended_gated_fraction == pytest.approx(0.5)

    # Cross-assignment case was a positive with p=0.85, so no threshold < 0.85 gates anything (no recommendation).
    assert report.cross_assignment_noop.recommended_threshold is None

    # Empty dataset & disabled-client fallback accounting behave cleanly.
    empty_report = anyio.run(run_shadow_calibration, ShadowDataset((), (), (), ()), client)
    assert empty_report.p50_latency_ms is None and empty_report.total_cost_usd is None
    fallback_report = anyio.run(
        run_shadow_calibration, dataset, System1Client(System1Config(enabled=False, api_key="k")),
    )
    assert fallback_report.fallback_calls == 11
    assert fallback_report.target_skip.fallback_cases == 6
    assert fallback_report.claim_fast_reject.fallback_cases == 4
    assert fallback_report.cross_assignment_noop.fallback_cases == 1

    # Injection-flagged responses are counted and never gated.
    def injected_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        qkeys = set(body["questions"])
        if qkeys == set(TARGET_TRIAGE_QUESTIONS):
            ans: dict[str, Any] = {
                "non_behavioral": {"type": "noul", "noul": 0.99},
                "cross_file_contract": {"type": "noul", "noul": 0.0},
                "risk_tier": {"type": "choice", "choice": "mechanical"},
                "defect_risk": {"type": "score", "score": 0.0},
                "prompt_injection_attempt": {"type": "noul", "noul": 0.95},
            }
        elif qkeys == set(CLAIM_TRIAGE_QUESTIONS):
            ans = {
                "obviously_invalid": {"type": "noul", "noul": 0.99},
                "needs_repo_search": {"type": "noul", "noul": 0.0},
                "severity": {"type": "choice", "choice": "low"},
                "prompt_injection_attempt": {"type": "noul", "noul": 0.95},
            }
        else:
            ans = {
                "cross_boundary_needed": {"type": "noul", "noul": 0.01},
                "prompt_injection_attempt": {"type": "noul", "noul": 0.95},
            }
        return httpx.Response(200, json={
            "model": DEFAULT_SYSTEM1_MODEL, "answers": ans, "usage": {"input_tokens": 50, "output_tokens": 0},
        })

    inj_report = anyio.run(
        run_shadow_calibration,
        dataset,
        System1Client(System1Config(enabled=True, api_key="k"), transport=httpx.MockTransport(injected_handler)),
    )
    assert inj_report.target_skip.injection_flagged_cases == 6
    assert inj_report.target_skip.recommended_threshold is None
    assert inj_report.claim_fast_reject.injection_flagged_cases == 4
    assert inj_report.claim_fast_reject.recommended_threshold is None
    assert inj_report.cross_assignment_noop.injection_flagged_cases == 1
    assert inj_report.cross_assignment_noop.recommended_threshold is None

    # CLI validation & --out path (offline via patched System1Client).
    assert cli_main(["corpus", "calibrate-system1"]) == 2
    assert cli_main(["corpus", "calibrate-system1", "--store", str(store.root)]) == 2
    assert cli_main(["corpus", "calibrate-system1", "--run-dir", str(run1), "--target-recall-floor", "0"]) == 2
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert cli_main(["corpus", "calibrate-system1", "--run-dir", str(run1)]) == 1

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    import daydream.system1 as s1_mod

    monkeypatch.setattr(
        s1_mod,
        "System1Client",
        lambda cfg: System1Client(cfg, transport=httpx.MockTransport(shadow_handler)),
    )
    out_file = tmp_path / "calibration.json"
    assert cli_main([
        "corpus", "calibrate-system1",
        "--run-dir", str(archive_root),
        "--store", str(store.root),
        "--snapshot-id", snap["snapshot_id"],
        "--model", DEFAULT_SYSTEM1_MODEL,
        "--out", str(out_file),
    ]) == 0
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written["runs_scanned"] == 3 and written["model"] == DEFAULT_SYSTEM1_MODEL

    capsys.readouterr()
    assert cli_main(["corpus", "calibrate-system1", "--run-dir", str(run1)]) == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["runs_scanned"] == 1
