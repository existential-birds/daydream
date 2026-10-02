"""Inject interception endpoints using each backend's native configuration.

The interception server selects the wire dialect by route: Chat Completions at
/v1/chat/completions, Responses at /v1/responses, Messages at /v1/messages.
Strategies provide CLI environment and provisioning; the harness is shared.
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

#: Conservative defaults for pi's required model-catalogue entries. Deliberately
#: modest: over-declaring a context window fails at rollout time, under-declaring
#: only truncates. Override per run via the harness's `pi_context_window` /
#: `pi_max_tokens`.
DEFAULT_PI_CONTEXT_WINDOW = 32768
DEFAULT_PI_MAX_TOKENS = 8192


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
    """Route Anthropic Messages through the inherited CLI environment.

    The CLI appends /v1/messages; its base URL must omit the /v1 suffix."""

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
    """Declare the interception provider in Codex config; Daydream passes no provider flags."""
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
    """Route Responses through an explicit CODEX_HOME provider configuration.

    HOME alone does not reliably control Codex configuration lookup. Pinning
    CODEX_HOME both captures requests and excludes a developer's stored login."""

    name = "codex"
    required_binaries: tuple[str, ...] = ("codex",)

    def __init__(self, home: str = ROLLOUT_HOME) -> None:
        self.home = home

    @property
    def codex_home(self) -> str:
        return f"{self.home}/.codex"

    @property
    def config_path(self) -> str:
        return f"{self.codex_home}/config.toml"

    def env(self, endpoint: str, secret: str, *, fanout_concurrency: int) -> dict[str, str]:
        return {
            "CODEX_HOME": self.codex_home,
            "CODEX_INTERCEPT_KEY": secret,
            "DAYDREAM_FANOUT_CONCURRENCY": str(fanout_concurrency),
        }

    async def provision(self, runtime: vf.Runtime, endpoint: str, secret: str, model: str) -> None:
        await runtime.write(self.config_path, codex_provider_toml(endpoint).encode())


def pi_extension_ts(endpoint: str, model: str, *, context_window: int, max_tokens: int) -> str:
    """Declare the rollout model in a Pi provider extension for Chat Completions.

    Use the extension's own key variable: Daydream remaps PI_API_KEY only for
    its built-in zai provider. Model identity comes from ctx.model; catalogue
    limits are configured capabilities of the actual endpoint."""
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
    """Install a Chat Completions provider extension and Pi's fan-out environment hint.

    Pi refuses subagents, so an exploration pre-scan cannot run under it."""

    name = "pi"
    required_binaries: tuple[str, ...] = ("pi",)

    def __init__(
        self,
        home: str = ROLLOUT_HOME,
        *,
        context_window: int = DEFAULT_PI_CONTEXT_WINDOW,
        max_tokens: int = DEFAULT_PI_MAX_TOKENS,
    ) -> None:
        self.home = home
        # pi's catalogue requires both. They are declared CAPABILITIES of the
        # policy endpoint, which this code cannot know, so they are harness
        # config (`pi_context_window` / `pi_max_tokens`) with a conservative
        # default rather than a baked-in assumption (SPEC C1). Set them to match
        # whatever the run's `seq_len` / `max_completion_tokens` actually allow —
        # a window declared larger than the endpoint serves fails at rollout time.
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
