"""Endpoint injection, one strategy per daydream backend.

The interception server serves all three wire dialects at once and the ROUTE
selects the format (verifiers 0.2.1 ``dialects/__init__.py:14-16``,
``interception/server.py:186-190``): Chat Completions at ``/v1/chat/completions``,
Responses at ``/v1/responses``, Anthropic Messages at ``/v1/messages``. So the
question is never "which backend can this environment drive" but "how does this
backend's CLI learn a base URL and a key" — and every one of them answers with
env vars or a config file, which is why no daydream code changes to run any of
them under RL.

``claude`` is the day-one default only because its CLI is the one the base image
already carries and its injection needs no provisioning file. It is a row in this
table, not a design commitment.

A future ``osprey`` row is one class: env ``OSPREY_OPENAI_BASE_URL={endpoint}``
(osprey ``core/osprey-cli/src/config.rs:633``), landing when daydream grows
``--backend osprey``. No harness change.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import verifiers.v1 as vf

#: Default ``HOME`` inside the rollout image; where provisioning files are planted.
ROLLOUT_HOME = "/rollout"

#: Provider name the codex and pi strategies register for the interception endpoint.
INTERCEPT_PROVIDER = "vf-intercept"


@runtime_checkable
class BackendStrategy(Protocol):
    """How one daydream backend is pointed at the interception endpoint."""

    name: str
    """Value passed to ``daydream --backend``."""

    required_binaries: tuple[str, ...]
    """Binaries the rollout image must carry, beyond ``daydream`` itself."""

    def env(self, endpoint: str, secret: str, *, fanout_concurrency: int) -> dict[str, str]:
        """Environment the daydream process needs to reach the endpoint."""
        ...

    async def provision(self, runtime: vf.Runtime, endpoint: str, secret: str, model: str) -> None:
        """Write and install any config the CLI reads from disk."""
        ...


class ClaudeStrategy:
    """Anthropic Messages dialect, env only.

    The claude-agent-sdk spawns its CLI with the full parent environment
    (``subprocess_cli.py:491``), so the same two variables verifiers' own
    ``claude_code`` harness sets (``harnesses/claude_code/harness.py:63-71``)
    reach the CLI unchanged.
    """

    name = "claude"
    required_binaries: tuple[str, ...] = ("claude",)

    def __init__(self, home: str = ROLLOUT_HOME) -> None:
        self.home = home

    def env(self, endpoint: str, secret: str, *, fanout_concurrency: int) -> dict[str, str]:
        return {
            # The CLI appends /v1/messages itself, so the suffix must come off.
            "ANTHROPIC_BASE_URL": endpoint.removesuffix("/v1"),
            "ANTHROPIC_API_KEY": secret,
            "CLAUDE_CONFIG_DIR": f"{self.home}/.claude",
            "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
            "DAYDREAM_FANOUT_CONCURRENCY": str(fanout_concurrency),
        }

    async def provision(self, runtime: vf.Runtime, endpoint: str, secret: str, model: str) -> None:
        return None


def codex_provider_toml(endpoint: str) -> str:
    """``$HOME/.codex/config.toml`` naming the interception server as the provider.

    Mirrors the provider block verifiers' codex harness passes as ``-c``
    overrides (``harnesses/codex/harness.py:134-167``). daydream's codex backend
    forwards no provider flags of its own, so the file is the seam.
    """
    return (
        f'model_provider = "{INTERCEPT_PROVIDER}"\n'
        "\n"
        f"[model_providers.{INTERCEPT_PROVIDER}]\n"
        f'name = "{INTERCEPT_PROVIDER}"\n'
        f'base_url = "{endpoint}"\n'
        'env_key = "CODEX_INTERCEPT_KEY"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
    )


class CodexStrategy:
    """OpenAI Responses dialect, via a written provider block.

    daydream's codex backend passes no ``env=`` to ``create_subprocess_exec``
    (``daydream/backends/codex.py:198``), so the child inherits the harness's
    environment and reads ``$HOME/.codex/config.toml``.
    """

    name = "codex"
    required_binaries: tuple[str, ...] = ("codex",)

    def __init__(self, home: str = ROLLOUT_HOME) -> None:
        self.home = home

    @property
    def config_path(self) -> str:
        return f"{self.home}/.codex/config.toml"

    def env(self, endpoint: str, secret: str, *, fanout_concurrency: int) -> dict[str, str]:
        return {"CODEX_INTERCEPT_KEY": secret}

    async def provision(self, runtime: vf.Runtime, endpoint: str, secret: str, model: str) -> None:
        await runtime.write(self.config_path, codex_provider_toml(endpoint).encode())


def pi_extension_ts(endpoint: str, model: str, *, context_window: int, max_tokens: int) -> str:
    """A pi provider extension pointing at the interception endpoint.

    Shape follows pi's shipped provider extensions: a default-exported function
    taking the ``ExtensionAPI`` and calling ``registerProvider``, with ``apiKey``
    given as a ``$VAR`` reference rather than a literal. The key variable is the
    extension's own, NOT ``PI_API_KEY``: daydream only remaps that one for the
    built-in ``zai`` provider (``daydream/backends/pi.py:84``) and drops it with a
    warning for anything else.

    pi resolves a model id against the provider's declared catalogue, so the
    policy model is declared here per rollout. No id is hardcoded (C1) — it
    arrives as ``ctx.model``.
    """
    provider = {
        "name": INTERCEPT_PROVIDER,
        "baseUrl": endpoint,
        "apiKey": "$VF_INTERCEPT_API_KEY",
        "api": "openai-completions",
        "models": [
            {
                "id": model,
                "name": model,
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": context_window,
                "maxTokens": max_tokens,
            }
        ],
    }
    return (
        'import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";\n'
        "\n"
        "export default function (pi: ExtensionAPI) {\n"
        f'  pi.registerProvider("{INTERCEPT_PROVIDER}", {json.dumps(provider, indent=2)});\n'
        "}\n"
    )


class PiStrategy:
    """Chat Completions dialect, via an installed pi provider extension.

    pi is the one backend that already speaks an arbitrary OpenAI-compatible base
    URL, but the URL lives outside daydream in a TypeScript extension, so the
    rollout provisions and installs one. Two pi-specific facts the harness must
    not have to know: its fan-out hint has its own variable
    (``DAYDREAM_PI_FANOUT_CONCURRENCY``, ``daydream/backends/pi.py:178-198``), and
    it refuses subagents outright (``pi.py:560``), so an exploration pre-scan
    cannot run under it.
    """

    name = "pi"
    required_binaries: tuple[str, ...] = ("pi",)

    def __init__(
        self,
        home: str = ROLLOUT_HOME,
        *,
        context_window: int = 131072,
        max_tokens: int = 32768,
    ) -> None:
        self.home = home
        # pi's model catalogue wants both. They bound the request, not the policy:
        # no model id or parameter count is implied (SPEC C1).
        self.context_window = context_window
        self.max_tokens = max_tokens

    @property
    def extension_dir(self) -> str:
        return f"{self.home}/.pi/extensions/{INTERCEPT_PROVIDER}"

    def env(self, endpoint: str, secret: str, *, fanout_concurrency: int) -> dict[str, str]:
        return {
            "PI_PROVIDER": INTERCEPT_PROVIDER,
            "VF_INTERCEPT_API_KEY": secret,
            "DAYDREAM_PI_FANOUT_CONCURRENCY": str(fanout_concurrency),
        }

    async def provision(self, runtime: vf.Runtime, endpoint: str, secret: str, model: str) -> None:
        source = pi_extension_ts(
            endpoint, model, context_window=self.context_window, max_tokens=self.max_tokens
        )
        await runtime.write(f"{self.extension_dir}/index.ts", source.encode())
        result = await runtime.run(
            ["sh", "-c", f"pi install {shlex.quote(self.extension_dir)}"],
            {"HOME": self.home},
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"`pi install {self.extension_dir}` failed ({result.exit_code}): "
                f"{result.stdout}{result.stderr}"
            )


StrategyFactory = Callable[[str], BackendStrategy]
"""Builds a strategy for a given ``HOME``."""

#: Strategy factories keyed by the ``--backend`` value. Built per harness rather
#: than shared, so ``HOME`` can move — the local subprocess smoke path has no
#: ``/rollout`` to write to.
STRATEGIES: dict[str, StrategyFactory] = {
    ClaudeStrategy.name: ClaudeStrategy,
    CodexStrategy.name: CodexStrategy,
    PiStrategy.name: PiStrategy,
}
