"""Self-contained isolated Harbor judge verifier for the compiled-grade verifier image.

Stdlib + httpx only. Never imports daydream: this file must run unchanged
inside a compiled-grade verifier image that has no daydream wheel. It wires a
bounded per-pair judge prompt, two isolated external judge clients (Anthropic
Messages + OpenAI-compatible) behind one ``complete_json`` seam, strict verdict
parsing, a shared retry/redirect/timeout policy, a concurrency-10 runner, and
``run_verifier`` which writes ``reward.json`` / ``reward-details.json``
atomically.
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from pathlib import Path
from typing import Any, Protocol

import httpx
import verifier_core


class _AsyncHttpClient(Protocol):
    """An ``httpx.AsyncClient``-shaped seam for the injected fake clients."""

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        """POST ``json`` to ``url`` and return an httpx-like response object."""


VerifierError = verifier_core.VerifierError
_VERIFIER_THRESHOLD = verifier_core.CONFIDENCE_THRESHOLD

JUDGE_PROMPT_TEMPLATE = (
    Path(__file__).with_name("judge_prompt.md").read_text(encoding="utf-8")
)

_PROMPT_CAP_BYTES = 24 * 1024
_MAX_RETRIES = 3
_REQUEST_TIMEOUT = 60.0

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Truncate ``value`` to at most ``max_bytes`` on a UTF-8 boundary."""
    if len(value.encode("utf-8")) <= max_bytes:
        return value
    return value.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def _render_filled(
    template: str,
    gold: dict[str, Any],
    candidate: dict[str, Any],
    *,
    gold_body: str,
    candidate_body: str,
) -> str:
    return template.format(
        gold_title=gold.get("title", ""),
        gold_severity=str(gold.get("severity") or ""),
        gold_path=gold.get("path", ""),
        gold_start_line=gold.get("start_line", ""),
        gold_end_line=gold.get("end_line", ""),
        gold_body=gold_body,
        candidate_title=candidate.get("title", ""),
        candidate_severity=str(candidate.get("severity") or ""),
        candidate_path=candidate.get("path", ""),
        candidate_start_line=candidate.get("start_line", ""),
        candidate_end_line=candidate.get("end_line", ""),
        candidate_body=candidate_body,
    )


def render_pair_prompt(gold: dict, candidate: dict, *, template: str) -> str:
    """Render a bounded, untrusted-fenced prompt for one gold/candidate pair.

    If the filled template would exceed 24 KiB, the body fields (gold then
    candidate) are truncated on a UTF-8 boundary until the result fits — the
    only cap mechanism and it never fails the pair.
    """
    gold_body = gold.get("body", "") or ""
    candidate_body = candidate.get("body", "") or ""
    filled = _render_filled(
        template, gold, candidate, gold_body=gold_body, candidate_body=candidate_body
    )
    while len(filled.encode("utf-8")) > _PROMPT_CAP_BYTES:
        if gold_body:
            gold_body = _truncate_utf8(
                gold_body, max(0, len(gold_body.encode("utf-8")) // 2)
            )
        elif candidate_body:
            candidate_body = _truncate_utf8(
                candidate_body, max(0, len(candidate_body.encode("utf-8")) // 2)
            )
        else:
            break
        filled = _render_filled(
            template, gold, candidate, gold_body=gold_body, candidate_body=candidate_body
        )
    return filled


def parse_verdict(raw: object) -> verifier_core.Verdict:
    """Validate a raw judge verdict dict and return a ``verifier_core.Verdict``.

    ``gold_id``/``candidate_id`` are placeholders the caller (``judge_pairs``)
    stamps onto the returned verdict. Any violation — wrong type, out-of-range
    confidence, missing key, non-dict input — raises ``VerifierError``; never
    silently coerces a fallback value.
    """
    if not isinstance(raw, dict):
        raise VerifierError("verdict must be a JSON object")
    if "match" not in raw:
        raise VerifierError("verdict missing required field 'match'")
    match = raw["match"]
    if not isinstance(match, bool):
        raise VerifierError(f"verdict 'match' must be a boolean, got {match!r}")
    if "confidence" not in raw:
        raise VerifierError("verdict missing required field 'confidence'")
    confidence = raw["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VerifierError(
            f"verdict 'confidence' must be a number in [0,1], got {confidence!r}"
        )
    if not 0.0 <= confidence <= 1.0:
        raise VerifierError(f"verdict 'confidence' must be in [0,1], got {confidence!r}")
    if "reasoning" not in raw:
        raise VerifierError("verdict missing required field 'reasoning'")
    reasoning = raw["reasoning"]
    if not isinstance(reasoning, str):
        raise VerifierError(f"verdict 'reasoning' must be a string, got {reasoning!r}")
    return verifier_core.Verdict(
        gold_id="",
        candidate_id="",
        match=match,
        confidence=float(confidence),
        reasoning=reasoning,
    )


class _Retryable(Exception):
    """An internal marker: a request that should be retried (transport/5xx/429)."""


def _parse_json_response(response: Any, *, content: Any) -> dict[str, Any]:
    """Parse an httpx-like response through the ``content`` extraction callable."""
    status_code = getattr(response, "status_code", None)
    if status_code is None or not 200 <= int(status_code) < 300:
        body_text = getattr(response, "text", "")
        code = int(status_code) if status_code is not None else -1
        # 429 is retryable (rate limit); all other 4xx are terminal client errors.
        if 400 <= code < 500 and code != 429:
            raise VerifierError(f"Judge request failed with HTTP {status_code}: {body_text}")
        raise _Retryable(f"Judge request failed with HTTP {status_code}: {body_text}")
    try:
        parsed_body = response.json()
    except Exception as exc:
        raise VerifierError(f"Judge response was not valid JSON: {exc}") from exc
    text = content(parsed_body)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VerifierError(f"Judge text content was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise VerifierError("Judge text content JSON was not an object")
    return parsed


async def _complete_json_with_http(
    http: _AsyncHttpClient,
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    content: Any,
) -> dict[str, Any]:
    """Single retry/redirect/timeout policy shared by both judge clients.

    - Up to 3 attempts. Transport exceptions (including timeout) and HTTP
      429/5xx retry with exponential backoff (`2 ** attempt`); a terminal 4xx
      (non-429) and a malformed-JSON parse are not retried.
    - 3xx responses: never follow a redirect to a host outside the request
      URL's own host (allowlist); same-host redirects are followed once.
    - After 3 failed attempts, raise ``VerifierError`` — never a partial result.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            try:
                response = await http.post(
                    url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT
                )
            except Exception as exc:
                raise _Retryable(f"Judge request failed: {exc}") from exc
            status_code = getattr(response, "status_code", None)
            if status_code is not None and 300 <= int(status_code) < 400:
                headers = getattr(response, "headers", {}) or {}
                location = headers.get("location")
                if not location:
                    raise VerifierError("Judge request redirected without a Location")
                request_host = urllib.parse.urlparse(url).hostname
                redirect_host = urllib.parse.urlparse(location).hostname
                if redirect_host != request_host:
                    raise VerifierError(
                        "Judge request would redirect to a host outside the verifier allowlist"
                    )
                # Same-host redirect: follow once by re-issuing to the Location.
                response = await http.post(
                    location, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT
                )
            return _parse_json_response(response, content=content)
        except _Retryable:
            if attempt == _MAX_RETRIES - 1:
                raise VerifierError("Judge request failed after retries")
            await asyncio.sleep(2**attempt)
    raise VerifierError("Judge request failed after retries")


def _anthropic_text(body: dict[str, Any]) -> str:
    """Extract the first text block from an Anthropic Messages response body."""
    if not isinstance(body, dict):
        raise VerifierError("Judge response body was not an object")
    content = body.get("content")
    if not isinstance(content, list):
        raise VerifierError("Judge response missing content blocks")
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            text = block["text"].strip()
            if text:
                return text
    raise VerifierError("Judge response contained no text block")


class AnthropicJudgeClient:
    """Small Anthropic Messages API client returning strict parsed JSON verdicts."""

    def __init__(
        self, api_key: str, model: str, *, http: _AsyncHttpClient | None = None
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.http = http

    async def complete_json(
        self, *, user: str, system: str = "", max_tokens: int = 512
    ) -> dict[str, Any]:
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
            return await _complete_json_with_http(
                self.http,
                url=_ANTHROPIC_MESSAGES_URL,
                payload=payload,
                headers=headers,
                content=_anthropic_text,
            )
        async with httpx.AsyncClient() as http:
            return await _complete_json_with_http(
                http,
                url=_ANTHROPIC_MESSAGES_URL,
                payload=payload,
                headers=headers,
                content=_anthropic_text,
            )


def main() -> None:
    ...


if __name__ == "__main__":
    main()
