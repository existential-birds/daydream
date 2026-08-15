"""In-process OpenAI-compatible JSON completer for the benchmark judge.

Any OpenAI Chat Completions-compatible endpoint — OpenAI direct, OpenRouter, or
a compatible gateway — can serve as the benchmark judge. ``OpenAIJsonCompleter``
implements the same ``complete_json`` seam as ``AnthropicJsonClient``, so the
shared direct-scoring pipeline in :mod:`daydream.benchmark.anthropic_score`
(extraction → dedup → per-pair evaluation) runs against it unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from daydream.benchmark.anthropic_score import (
    _AsyncHttpClient,
    _complete_json_with_http,
    _enter_shared_http,
    run_direct_scoring,
)
from daydream.benchmark.score import (
    _OPENROUTER_BASE_URL,
    _OPENROUTER_KEY_PREFIX,
    _TOOL,
    OPENAI_JUDGE_API_KEY_ENV,
    OPENAI_JUDGE_BASE_URL_ENV,
    BenchmarkStepError,
    DaydreamScores,
)

#: Chat Completions base URL for OpenAI's own API. OpenRouter is auto-routed when
#: the key carries the ``sk-or-`` prefix (see `_resolve_openai_base_url`).
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"


def _resolve_openai_base_url(api_key: str, base_url_env: str | None) -> str:
    """Resolve the Chat Completions base URL from the environment.

    An explicit ``OPENAI_BASE_URL`` always wins; an ``sk-or-`` OpenRouter key
    with no pin routes to OpenRouter (mirroring the martian route's auto-route,
    score.py); anything else defaults to OpenAI direct.
    """
    if base_url_env:
        return base_url_env
    if api_key.startswith(_OPENROUTER_KEY_PREFIX):
        return _OPENROUTER_BASE_URL
    return _OPENAI_DEFAULT_BASE_URL


def _openai_content(body: Any) -> str:
    """Extract ``choices[0].message.content`` from an OpenAI-compatible response body."""
    if not isinstance(body, dict):
        raise BenchmarkStepError("Judge response body was not an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BenchmarkStepError("Judge response missing a choices list")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise BenchmarkStepError("Judge response choice was not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise BenchmarkStepError("Judge response missing a message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise BenchmarkStepError("Judge response message content was not a string")
    text = content.strip()
    if not text:
        raise BenchmarkStepError("Judge response message content was empty")
    return text


@dataclass
class OpenAIJsonCompleter:
    """Small OpenAI Chat Completions client that returns strict parsed JSON objects."""

    api_key: str
    model: str
    base_url: str = _OPENAI_DEFAULT_BASE_URL
    http: _AsyncHttpClient | None = None

    async def complete_json(self, *, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        """POST a Chat Completions request and parse ``choices[0].message.content`` as JSON."""
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        url = self.base_url.rstrip("/") + _OPENAI_CHAT_COMPLETIONS_PATH

        if self.http is not None:
            return await _complete_json_with_http(
                self.http, url=url, payload=payload, headers=headers, content=_openai_content
            )
        async with httpx.AsyncClient() as http:
            return await _complete_json_with_http(
                http, url=url, payload=payload, headers=headers, content=_openai_content
            )


async def run_openai_scoring(
    benchmark_repo: Path,
    judge_model: str,
    *,
    golden_urls: Collection[str] | None = None,
    tool: str = _TOOL,
    client: OpenAIJsonCompleter | None = None,
) -> DaydreamScores:
    """Run extraction, dedup, and evaluation against an OpenAI-compatible endpoint.

    When ``client`` is omitted the completer is built from ``OPENAI_API_KEY`` /
    ``OPENAI_BASE_URL`` (with OpenRouter auto-routing for ``sk-or-`` keys). The
    pipeline is the shared in-process scorer with the ``openai-compatible`` leaf
    stamp, so results land in the same `results/<model>/evaluations.json` shape
    as ``anthropic-direct``.
    """
    async with AsyncExitStack() as stack:
        if client is None:
            api_key = os.environ.get(OPENAI_JUDGE_API_KEY_ENV)
            if not api_key:
                raise BenchmarkStepError(
                    f"{OPENAI_JUDGE_API_KEY_ENV} is not set; cannot run OpenAI-compatible scoring."
                )
            base_url = _resolve_openai_base_url(api_key, os.environ.get(OPENAI_JUDGE_BASE_URL_ENV))
            http = await _enter_shared_http(stack)
            client = OpenAIJsonCompleter(api_key=api_key, model=judge_model, base_url=base_url, http=http)

        return await run_direct_scoring(
            benchmark_repo,
            judge_model,
            golden_urls=golden_urls,
            tool=tool,
            client=client,
            judge_route="openai-compatible",
        )
