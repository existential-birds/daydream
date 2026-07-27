"""Phase 4: the per-backend injection seam.

Each test asserts the artifact a backend's CLI actually reads — the env var it
resolves its base URL from, or the exact config file on disk — not that a
provisioning method was called.
"""

from __future__ import annotations

import json
import tomllib

import pytest
import verifiers.v1 as vf
from conftest import FakeRuntime

from daydream_review_v1.backends import INTERCEPT_PROVIDER, STRATEGIES, ClaudeStrategy, CodexStrategy, PiStrategy
from daydream_review_v1.harness import DaydreamReviewHarnessConfig

ENDPOINT = "http://127.0.0.1:54321/v1"
SECRET = "rollout-secret-token"
MODEL = "some-org/some-policy-model"
HOME = "/rollout"


async def test_claude_strategy_is_env_only() -> None:
    strategy = ClaudeStrategy(HOME)
    runtime = FakeRuntime()

    env = strategy.env(ENDPOINT, SECRET, fanout_concurrency=7)
    await strategy.provision(runtime, ENDPOINT, SECRET, MODEL)

    # The CLI appends /v1/messages itself, so the base URL must not already carry it.
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
    assert env["ANTHROPIC_API_KEY"] == SECRET
    assert env["CLAUDE_CONFIG_DIR"] == "/rollout/.claude"
    assert env["DAYDREAM_FANOUT_CONCURRENCY"] == "7"
    assert env["DISABLE_AUTOUPDATER"] == "1" and env["IS_SANDBOX"] == "1"
    assert runtime.writes == {}, "claude needs no provisioning file"
    assert runtime.commands == []


async def test_codex_strategy_writes_a_responses_provider_block() -> None:
    strategy = CodexStrategy(HOME)
    runtime = FakeRuntime()

    env = strategy.env(ENDPOINT, SECRET, fanout_concurrency=4)
    await strategy.provision(runtime, ENDPOINT, SECRET, MODEL)

    # CODEX_HOME, not HOME: a live rollout with only HOME moved had codex resolve
    # its config elsewhere, miss the provider block, and reach the provider
    # directly — real tokens billed, zero calls captured in the trace.
    assert env == {"CODEX_HOME": "/rollout/.codex", "CODEX_INTERCEPT_KEY": SECRET}
    written = runtime.writes["/rollout/.codex/config.toml"]
    config = tomllib.loads(written.decode())
    assert config["model_provider"] == INTERCEPT_PROVIDER
    provider = config["model_providers"][INTERCEPT_PROVIDER]
    assert provider["base_url"] == ENDPOINT
    assert provider["wire_api"] == "responses"
    assert provider["env_key"] == "CODEX_INTERCEPT_KEY"
    assert provider["requires_openai_auth"] is False


async def test_pi_strategy_installs_a_chat_completions_provider_extension() -> None:
    strategy = PiStrategy(HOME)
    runtime = FakeRuntime()

    env = strategy.env(ENDPOINT, SECRET, fanout_concurrency=9)
    await strategy.provision(runtime, ENDPOINT, SECRET, MODEL)

    assert env["PI_PROVIDER"] == INTERCEPT_PROVIDER
    assert env["VF_INTERCEPT_API_KEY"] == SECRET
    # pi has its own fan-out variable; the generic one would be ignored.
    assert env["DAYDREAM_PI_FANOUT_CONCURRENCY"] == "9"
    assert "DAYDREAM_FANOUT_CONCURRENCY" not in env
    # daydream only remaps PI_API_KEY for the built-in zai provider (pi.py:84),
    # so the extension must read its own variable instead.
    assert "PI_API_KEY" not in env

    source = runtime.writes[f"/rollout/.pi/extensions/{INTERCEPT_PROVIDER}/index.ts"].decode()
    marker = f'pi.registerProvider("{INTERCEPT_PROVIDER}", '
    assert marker in source
    provider = json.loads(source[source.index(marker) + len(marker) : source.rindex(");")])
    assert provider["baseUrl"] == ENDPOINT
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "$VF_INTERCEPT_API_KEY"
    # The policy model is declared per rollout, never hardcoded (SPEC C1).
    assert [model["id"] for model in provider["models"]] == [MODEL]

    assert any("pi install" in " ".join(argv) for argv in runtime.commands), (
        "a written extension that is never installed does nothing"
    )


async def test_pi_strategy_raises_when_install_fails() -> None:
    class FailingInstall(FakeRuntime):
        async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
            await super().run(argv, env)
            return vf.ProgramResult(exit_code=1, stdout="", stderr="no such extension")

    runtime = FailingInstall()
    with pytest.raises(RuntimeError, match="pi install"):
        await PiStrategy(HOME).provision(runtime, ENDPOINT, SECRET, MODEL)


async def test_every_strategy_honours_a_relocated_home() -> None:
    """The local smoke path has no /rollout to write into."""
    for name, factory in STRATEGIES.items():
        strategy = factory("/tmp/elsewhere")
        runtime = FakeRuntime()
        await strategy.provision(runtime, ENDPOINT, SECRET, MODEL)
        assert all(path.startswith("/tmp/elsewhere/") for path in runtime.writes), (
            f"{name} wrote outside the configured HOME: {sorted(runtime.writes)}"
        )
        env = strategy.env(ENDPOINT, SECRET, fanout_concurrency=4)
        assert all("/rollout" not in value for value in env.values()), f"{name} env pins /rollout: {env}"


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        DaydreamReviewHarnessConfig(backend="nope")
    message = str(excinfo.value)
    assert "nope" in message
    for name in STRATEGIES:
        assert name in message


def test_shipped_strategies_cover_every_daydream_backend() -> None:
    """daydream's --backend choices are claude|codex|pi (cli.py:238-244)."""
    assert set(STRATEGIES) == {"claude", "codex", "pi"}
