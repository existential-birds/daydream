import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import daydream.benchmark.harbor.calibrate as cal
from daydream.benchmark import cli
from daydream.benchmark.harbor.calibrate import (
    _judge_host_from_env,
    _load_fixture,
    _load_workspace_allowlist,
    _validate_workspace_host,
    run_calibration,
)

_FIXTURE = Path(__file__).parents[1] / "daydream" / "benchmark" / "harbor" / "calibration"

def _env() -> dict[str, Any]:
    return {"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_MODEL": "m",
            "DAYDREAM_JUDGE_API_KEY": "k", "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9"}


@pytest.fixture
def ws_factory() -> Any:
    """Build a workspace allowlisting localhost or the requested judge hosts."""

    def _build(tmp_path: Path, *, judge_allowed_hosts: tuple[str, ...] = ("127.0.0.1",)) -> Any:
        ws = tmp_path / "ws"
        (ws / "runtime").mkdir(parents=True)
        (ws / "benchmark.yaml").write_text(json.dumps({
            "schema_version": 1, "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
            "created_at": "2026-08-21T12:00:00Z", "source": {"provider": "github", "hostname": "github.com",
                        "repository": "OWNER/REPO", "repository_id": None, "visibility": "unresolved"},
            "privacy": {"classification": "confidential",
                        "reviewer_data": "source_snapshot", "reviewer_allowed_hosts": ["review.example"],
                        "judge_data": "finding_text_and_location_only",
                        "judge_allowed_hosts": list(judge_allowed_hosts), "archive": "disabled", "uploads": "disabled"},
            "pull_requests": [], "cases": [],
        }))
        return ws

    return _build


def _scripted_http(responses: Any) -> tuple[Any, ...]:
    i = [0]
    _jsonlib = json  # module reference; ``json`` below is the keyword payload

    class Fake:
        async def post(self, url: Any, *, headers: Any, json: Any, timeout: Any) -> Any:
            user = next(message["content"] for message in json["messages"] if message["role"] == "user")
            assert len(user.encode("utf-8")) <= 24 * 1024
            v = responses[i[0] % len(responses)]
            i[0] += 1
            body = {"choices": [{"message": {"content": _jsonlib.dumps(v)}}]}
            return type("R", (), {"status_code": 200, "text": "ok", "json": lambda self: body})()
    return Fake(), i


def _scripted_responses(pairs: Any, *, mislabel_count: Any=0) -> Any:
    """Repeat each verdict three times, flipping the first ``mislabel_count`` match pairs."""
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


def _scripted_cli_runner(responses: Any) -> tuple[Any, Any]:
    """Return CLI result envelopes in the judge's deterministic pair order."""

    class FakeProc:
        returncode = 0

        def __init__(self, payload: dict[str, str]) -> None:
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(json.dumps(payload).encode("utf-8"))
            self.stdout.feed_eof()

        async def wait(self) -> int:
            return self.returncode

    i = [0]

    async def runner(self: Any, argv: Any, env: Any) -> Any:
        payload = {"result": json.dumps(responses[i[0] % len(responses)])}
        i[0] += 1
        return FakeProc(payload)

    return runner, i


def test_acceptance_failures_publish_no_receipt(
    tmp_path: Path, ws_factory: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the packaged judge with injected HTTP responses and no sockets."""
    responses = _scripted_responses(_load_fixture(), mislabel_count=3)
    # Distinct nonmatch pairs12/13 yield false positives beside three false negatives.
    responses[36:42] = [{"match": True, "confidence": 0.95, "reasoning": "false positive"}] * 6
    assert len(responses) == 72
    fake, counter = _scripted_http(responses)
    accuracy = ws_factory(tmp_path / "accuracy")
    assert run_calibration(accuracy, yes=True, env=_env(), http=fake) == 1
    assert not (accuracy / "runtime" / "calibration-receipt.json").exists()
    assert counter[0] == 72
    errors = capsys.readouterr().err.splitlines()
    assert len(errors) == 2 and errors[0] == "calibrate-judge: FAIL (majority): [0, 1, 2, 12, 13]"
    prefix, value = errors[1].rsplit(": ", 1)
    assert prefix == "calibrate-judge: FAIL (balanced_accuracy)"
    assert float(value) == pytest.approx(19 / 24)

    responses = []
    for idx, p in enumerate(_load_fixture()):
        if idx == 0:  # first match pair: retained [T,F,T] -> unstable
            responses += [{"match": True, "confidence": 0.9, "reasoning": "x"},
                          {"match": True, "confidence": 0.5, "reasoning": "x"},
                          {"match": True, "confidence": 0.9, "reasoning": "x"}]
        elif idx == 1:  # T,T,F majority is True and a single retained-edge flip is stable.
            responses += [{"match": True, "confidence": 0.9, "reasoning": "x"}] * 2
            responses += [{"match": False, "confidence": 0.9, "reasoning": "x"}]
        elif idx == 12:  # F,F,T majority stays False for the first nonmatch pair.
            responses += [{"match": False, "confidence": 0.9, "reasoning": "x"}] * 2
            responses += [{"match": True, "confidence": 0.9, "reasoning": "x"}]
        else:
            m = p["label"] == "match"
            responses += [{"match": m, "confidence": 0.95 if m else 0.1, "reasoning": "x"} for _ in range(3)]
    fake, _ = _scripted_http(responses)
    unstable = ws_factory(tmp_path / "instability")
    assert run_calibration(unstable, yes=True, env=_env(), http=fake) == 1
    assert capsys.readouterr().err.splitlines() == ["calibrate-judge: FAIL (instability): [0]"]
    assert not (unstable / "runtime" / "calibration-receipt.json").exists()


def test_calibrate_judge_claude_cli_composed_path(tmp_path: Path, ws_factory: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake only the CLI subprocess; exercise OAuth admission through the real handler and judge."""

    ws = ws_factory(tmp_path, judge_allowed_hosts=("api.anthropic.com",))
    monkeypatch.delenv("DAYDREAM_JUDGE_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("DAYDREAM_JUDGE_PROVIDER", "claude-cli")
    monkeypatch.setenv("DAYDREAM_JUDGE_MODEL", "m")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-tok-sentinel")
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    sr = cal._load_judge_template()
    fake, counter = _scripted_cli_runner(_scripted_responses(cal._load_fixture()))
    monkeypatch.setattr(sr.ClaudeCliJudgeClient, "_default_runner", fake)
    code = cli._handle_benchmark_command(["calibrate-judge", str(ws), "--yes"])
    assert code == 0
    assert (ws / "runtime" / "calibration-receipt.json").exists()
    assert counter[0] == 72          # all 72 judge calls went through the fake runner

def test_calibration_refuses_unsafe_judge_selection_before_any_call(
    tmp_path: Path, ws_factory: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    assert _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "openai-compatible",
                                 "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9"}) == "127.0.0.1"
    assert _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "anthropic"}) == "api.anthropic.com"
    assert _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "claude-cli"}) == "api.anthropic.com"
    for environment, message in (
        ({}, "missing DAYDREAM_JUDGE_PROVIDER"),
        ({"DAYDREAM_JUDGE_PROVIDER": "bogus"}, "unsupported DAYDREAM_JUDGE_PROVIDER.*expected anthropic, "
                                               "openai-compatible, or claude-cli"),
        ({"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_API_KEY": "sk-or-key"},
         "missing DAYDREAM_JUDGE_BASE_URL"),
    ):
        with pytest.raises(ValueError, match=message):
            _judge_host_from_env(environment)
    local = ws_factory(tmp_path / "local")
    allow = _load_workspace_allowlist(local)
    assert allow == ["127.0.0.1"]
    with pytest.raises(ValueError):
        _validate_workspace_host(allow, "evil.example")
    _validate_workspace_host(allow, "127.0.0.1")
    assert run_calibration(local, yes=True, env={}) == 1
    assert "missing DAYDREAM_JUDGE_PROVIDER" in capsys.readouterr().err
    fake, counter = _scripted_http([{"match": True, "confidence": 0.9, "reasoning": "x"}])
    assert run_calibration(local, yes=False, confirm=lambda _: False, env=_env(), http=fake) == 1
    assert counter[0] == 0

    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    assert cli._handle_benchmark_command(["calibrate-judge", str(tmp_path)]) == 1
    assert "requires TTY confirmation or --yes" in capsys.readouterr().err
    # The real presence gate refuses before it can launch the Claude CLI.
    ws = ws_factory(tmp_path / "claude", judge_allowed_hosts=("api.anthropic.com",))
    monkeypatch.delenv("DAYDREAM_JUDGE_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("DAYDREAM_JUDGE_PROVIDER", "claude-cli")
    monkeypatch.setenv("DAYDREAM_JUDGE_MODEL", "m")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    assert cli._handle_benchmark_command(["calibrate-judge", str(ws), "--yes"]) == 1
    assert "missing CLAUDE_CODE_OAUTH_TOKEN" in capsys.readouterr().err
    for root in (local, ws):
        assert not (root / "runtime" / "calibration-receipt.json").exists()
