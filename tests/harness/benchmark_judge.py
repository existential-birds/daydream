"""Shared judge client fake and env for benchmark verifier tests."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


def judge_env() -> dict[str, Any]:
    return {"DAYDREAM_JUDGE_PROVIDER": "anthropic", "DAYDREAM_JUDGE_MODEL": "m", "DAYDREAM_JUDGE_API_KEY": "k",
        "DAYDREAM_JUDGE_BASE_URL": None,
    }


class MatchClient:
    async def complete_json(self, *, user: Any, system: Any, max_tokens: Any) -> dict[str, Any]:
        return {"match": True, "confidence": 1.0, "reasoning": "identical"}


class CliProcess:
    """Native process-shaped seam with a byte stream and settled exit status."""

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout.encode("utf-8"))
        self.stdout.feed_eof()
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        pass


def http_response(status: int, body: Any = None, *, text: str = "ok", **attrs: Any) -> httpx.Response:
    """Construct one native response; parsed JSON and byte content share its body."""
    if "content" in attrs:
        return httpx.Response(status, **attrs)
    if body is not None:
        return httpx.Response(status, json=body, **attrs)
    return httpx.Response(status, text=text, **attrs)
