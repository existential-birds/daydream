"""Direct Anthropic JSON client for benchmark judge steps."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from daydream.benchmark.score import BenchmarkStepError

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_REQUEST_TIMEOUT = 60.0
_MAX_RETRIES = 3


class _AsyncHttpClient(Protocol):
    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        ...


@dataclass
class AnthropicJsonClient:
    """Small Messages API client that returns strict parsed JSON objects."""

    api_key: str
    model: str
    http: _AsyncHttpClient | None = None

    async def complete_json(self, *, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        """POST a Messages API request and parse the first returned text block as JSON."""
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        if self.http is not None:
            return await _complete_json_with_http(self.http, payload=payload, headers=headers)
        async with httpx.AsyncClient() as http:
            return await _complete_json_with_http(http, payload=payload, headers=headers)


async def _complete_json_with_http(
    http: _AsyncHttpClient, *, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    for attempt in range(_MAX_RETRIES):
        try:
            try:
                response = await http.post(
                    _ANTHROPIC_MESSAGES_URL, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT
                )
            except Exception as exc:
                raise BenchmarkStepError(f"Anthropic Messages request failed: {exc}") from exc
            return _parse_json_response(response)
        except BenchmarkStepError:
            if attempt == _MAX_RETRIES - 1:
                raise
            await asyncio.sleep(2**attempt)
    raise BenchmarkStepError("Anthropic Messages request failed after retries")


def _parse_json_response(response: Any) -> dict[str, Any]:
    status_code = getattr(response, "status_code", None)
    if status_code is None or not 200 <= int(status_code) < 300:
        body = getattr(response, "text", "")
        raise BenchmarkStepError(f"Anthropic Messages request failed with HTTP {status_code}: {body}")

    try:
        body = response.json()
    except Exception as exc:
        raise BenchmarkStepError(f"Anthropic Messages response was not valid JSON: {exc}") from exc

    text = _first_text_block(body)
    try:
        parsed = json.loads(_strip_markdown_fences(text))
    except json.JSONDecodeError as exc:
        raise BenchmarkStepError(f"Anthropic Messages text block was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BenchmarkStepError("Anthropic Messages text block JSON was not an object")
    return parsed


def _first_text_block(body: Any) -> str:
    if not isinstance(body, dict):
        raise BenchmarkStepError("Anthropic Messages response body was not an object")
    content = body.get("content")
    if not isinstance(content, list):
        raise BenchmarkStepError("Anthropic Messages response missing content blocks")
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            text = block["text"].strip()
            if text:
                return text
    raise BenchmarkStepError("Anthropic Messages response contained no text block")


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
