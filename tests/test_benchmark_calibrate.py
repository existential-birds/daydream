import json
import re
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import daydream.benchmark.harbor.calibrate as cal
from daydream.benchmark import cli
from daydream.benchmark.harbor import calibrate, verifier_core as canonical_vc
from daydream.benchmark.harbor.calibrate import (
    _build_calibration_client,
    _judge_host_from_env,
    _load_fixture,
    _load_judge_template,
    _load_provenance,
    _load_workspace_allowlist,
    _validate_workspace_host,
    run_calibration,
)

_FIXTURE = Path(__file__).parents[1] / "daydream" / "benchmark" / "harbor" / "calibration"

REQUIRED_CATEGORIES = {"wording_location", "related_distinct", "negation", "same_class_different_components",
    "severity_disagreement", "locationless", "delimiter_injection", "near_miss",
}
_CRED = re.compile(r"sk-ant-|sk-or-|Bearer |x-api-key")


def test_fixture_is_source_free() -> None:
    pairs = calibrate._load_fixture()
    text = json.dumps(pairs, ensure_ascii=False)
    raw_text = (_FIXTURE / "pairs.json").read_text(encoding="utf-8")
    assert not _CRED.search(raw_text)
    assert not _CRED.search(text)

    # Load the source-derived gold through the packaged judge's JSON boundary,
    # then check every meaningful title/body/digest value rather than a sample.
    sr = calibrate._load_judge_template()
    gold = sr._read_json(Path(sr.__file__).with_name("golden-review.json"))
    assert isinstance(gold, list) and gold
    source_values = {value
        for finding in gold
        for field in ("title", "body", "finding_id")
        if isinstance(finding, dict)
        and isinstance(value := finding.get(field), str)
        and len(value.strip()) > 1
    }
    assert source_values
    assert all(value not in text for value in source_values)

def test_fixture_provenance_declares_unverified_llm_origin() -> None:
    pairs = _load_fixture()
    prov = _load_provenance(pairs)
    assert prov["origin"] == "llm_generated"
    assert prov["human_reviewed"] is False
    assert prov["labels"] == "unverified"
    # Pair-attestable facts are derived from the passed pairs, not a fixed file.
    assert prov["class_balance"] == {"match": 12, "nonmatch": 12}
    assert prov["categories"] == 8
    assert len(pairs) == 24          # the 24 pairs survive the shape change
    labels = [p["label"] for p in pairs]
    assert labels.count("match") == 12 and labels.count("nonmatch") == 12
    assert {p["category"] for p in pairs} >= REQUIRED_CATEGORIES

def test_loader_resolves_sibling_verifier_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare-name collision must survive temporary template imports and cache hits."""
    foreign = ModuleType("score_review")
    monkeypatch.setattr(calibrate, "_TEMPLATE_CACHE", {})
    monkeypatch.setitem(sys.modules, "score_review", foreign)
    monkeypatch.delitem(sys.modules, "verifier_core", raising=False)
    sr = calibrate._load_judge_template()
    template_dir = (calibrate._TEMPLATES / "tests").resolve()
    assert Path(sr.__file__).resolve() == template_dir / "score_review.py"
    # The asset's bare import must resolve to the canonical host module.
    assert sr.verifier_core is canonical_vc
    assert calibrate._load_judge_template() is sr
    assert sys.modules["score_review"] is foreign and "verifier_core" not in sys.modules

    verdict = sr.parse_verdict({"match": True, "confidence": 0.9, "reasoning": "known verifier call"})
    assert isinstance(verdict, sr.verifier_core.Verdict)
    assert verdict.match is True and verdict.confidence == 0.9

def test_client_builder_fails_closed_without_model_or_key() -> None:
    sr = _load_judge_template()
    with pytest.raises(sr.VerifierError):
        _build_calibration_client({"DAYDREAM_JUDGE_MODEL": "m"})  # no API key

def test_judge_host_resolved_from_env() -> None:
    assert _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "openai-compatible",
                                 "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9"}) == "127.0.0.1"
    assert _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "anthropic"}) == "api.anthropic.com"
    assert _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "claude-cli"}) == "api.anthropic.com"
    with pytest.raises(ValueError, match="missing DAYDREAM_JUDGE_PROVIDER"):
        _judge_host_from_env({})
    with pytest.raises(ValueError, match="unsupported DAYDREAM_JUDGE_PROVIDER"):
        _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "bogus"})
    with pytest.raises(ValueError, match="expected anthropic, openai-compatible, or claude-cli"):
        _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "bogus"})
    with pytest.raises(ValueError, match="missing DAYDREAM_JUDGE_BASE_URL"):
        _judge_host_from_env({"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_API_KEY": "sk-or-key"})

def test_out_of_allowlist_host_rejected(tmp_path: Path, ws_factory: Any) -> None:
    ws = ws_factory(tmp_path)
    allow = _load_workspace_allowlist(ws)
    assert allow == ["127.0.0.1"]
    with pytest.raises(ValueError):
        _validate_workspace_host(allow, "evil.example")
    _validate_workspace_host(allow, "127.0.0.1")  # in-allowlist passes

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


def test_run_calibration_reports_missing_judge_provider(
    ws_factory: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_calibration(ws_factory(tmp_path), yes=True, env={}) == 1
    assert "missing DAYDREAM_JUDGE_PROVIDER" in capsys.readouterr().err


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
        stdout = ""

    i = [0]

    async def runner(self: Any, argv: Any, env: Any) -> Any:
        payload = {"result": json.dumps(responses[i[0] % len(responses)])}
        i[0] += 1
        FakeProc.stdout = json.dumps(payload)
        return FakeProc()

    return runner, i


class TestCalibrateAcceptance:
    """Exercise the packaged judge with injected HTTP responses and no sockets."""

    def test_run_calibration_pass_writes_receipt(self, tmp_path: Path, ws_factory: Any) -> None:
        responses = _scripted_responses(_load_fixture())
        assert len(responses) == 72
        fake, counter = _scripted_http(responses)
        workspace = ws_factory(tmp_path)
        env = {**_env(), "DAYDREAM_JUDGE_API_KEY": "sk-or-CALIBRATION_SECRET_SENTINEL"}
        code = run_calibration(workspace, yes=True, env=env, http=fake)
        assert code == 0
        assert (tmp_path / "ws" / "runtime" / "calibration-receipt.json").exists()
        assert counter[0] == 72
        receipt = (tmp_path / "ws" / "runtime" / "calibration-receipt.json").read_text()
        assert not re.search(r"sk-ant-|sk-or-|Bearer |x-api-key", receipt)
        assert "Cache key not tenant-scoped" not in receipt
        assert "DAYDREAM_JUDGE_API_KEY" not in receipt
        assert env["DAYDREAM_JUDGE_API_KEY"] not in receipt
        receipt_path = workspace / "runtime" / "calibration-receipt.json"
        assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
        first = json.loads(receipt)
        assert first["result"]["confusion_matrix"] == {"tp": 12, "fp": 0, "tn": 12, "fn": 0}
        assert first["result"]["balanced_accuracy"] == 1.0
        pairs = first["result"]["disagreements"]
        assert [pair["pair_index"] for pair in pairs] == list(range(24))
        assert [pair["per_run_retained"] for pair in pairs] == [[True] * 3] * 12 + [[False] * 3] * 12
        assert run_calibration(workspace, yes=True, env=env, http=fake) == 0
        second = json.loads(receipt_path.read_text())
        assert (second["inputs"], second["result"]) == (first["inputs"], first["result"])
        assert counter[0] == 144

    def test_accuracy_failure_no_receipt(
        self, tmp_path: Path, ws_factory: Any, capsys: pytest.CaptureFixture[str],
    ) -> None:
        responses = _scripted_responses(_load_fixture(), mislabel_count=3)
        # Distinct nonmatch pairs12/13 yield false positives beside three false negatives.
        responses[36:42] = [{"match": True, "confidence": 0.95, "reasoning": "false positive"}] * 6
        assert len(responses) == 72
        fake, counter = _scripted_http(responses)
        code = run_calibration(ws_factory(tmp_path), yes=True, env=_env(), http=fake)
        assert code == 1
        assert not (tmp_path / "ws" / "runtime" / "calibration-receipt.json").exists()
        assert counter[0] == 72
        errors = capsys.readouterr().err.splitlines()
        assert len(errors) == 2 and errors[0] == "calibrate-judge: FAIL (majority): [0, 1, 2, 12, 13]"
        prefix, value = errors[1].rsplit(": ", 1)
        assert prefix == "calibrate-judge: FAIL (balanced_accuracy)"
        assert float(value) == pytest.approx(19 / 24)

    def test_confirmation_refuses_before_any_call(self, tmp_path: Path, ws_factory: Any) -> None:
        fake, counter = _scripted_http([{"match": True, "confidence": 0.9, "reasoning": "x"}])
        code = run_calibration(ws_factory(tmp_path), yes=False, confirm=lambda _: False, env=_env(), http=fake)
        assert code == 1
        assert counter[0] == 0
        assert not (tmp_path / "ws" / "runtime" / "calibration-receipt.json").exists()

    def test_acceptance_instability_failure(self, tmp_path: Path, ws_factory: Any, capsys: pytest.CaptureFixture[str],
    ) -> None:
        pairs = _load_fixture()
        responses = []
        for idx, p in enumerate(pairs):
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
        code = run_calibration(ws_factory(tmp_path), yes=True, env=_env(), http=fake)
        assert code == 1
        err = capsys.readouterr().err
        assert err.splitlines() == ["calibrate-judge: FAIL (instability): [0]"]
        assert not (tmp_path / "ws" / "runtime" / "calibration-receipt.json").exists()

def test_calibrate_judge_refuses_without_tty_or_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    code = cli._handle_benchmark_command(["calibrate-judge", str(tmp_path)])
    assert code == 1
    assert "requires TTY confirmation or --yes" in capsys.readouterr().err

def test_calibrate_judge_handler_threads_claude_oauth_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(workspace: Any, *, yes: Any, env: Any, http: Any) -> int:
        seen["env"] = env
        return 0

    monkeypatch.setattr(cal, "run_calibration", fake_run)
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-tok-sentinel")
    monkeypatch.delenv("DAYDREAM_JUDGE_PROVIDER", raising=False)
    code = cli._handle_benchmark_command(["calibrate-judge", str(tmp_path), "--yes"])
    capsys.readouterr()
    assert code == 0
    assert seen["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-tok-sentinel"
    # Whitelisted but not exported -> None, not an ambient leak via dict(os.environ).
    assert seen["env"]["DAYDREAM_JUDGE_PROVIDER"] is None

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

def test_calibrate_judge_claude_cli_missing_token_fails_closed(
    tmp_path: Path, ws_factory: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the real presence gate before it can launch the Claude CLI."""

    ws = ws_factory(tmp_path, judge_allowed_hosts=("api.anthropic.com",))
    monkeypatch.delenv("DAYDREAM_JUDGE_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("DAYDREAM_JUDGE_PROVIDER", "claude-cli")
    monkeypatch.setenv("DAYDREAM_JUDGE_MODEL", "m")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(cli, "_is_interactive_tty", lambda: False)
    capsys.readouterr()
    code = cli._handle_benchmark_command(["calibrate-judge", str(ws), "--yes"])
    err = capsys.readouterr().err
    assert code == 1
    assert "missing CLAUDE_CODE_OAUTH_TOKEN" in err
    assert not (ws / "runtime" / "calibration-receipt.json").exists()
