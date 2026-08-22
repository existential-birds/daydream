"""Self-contained isolated Harbor judge verifier for the compiled-grade verifier image.

Stdlib + httpx only. Never imports daydream: this file must run unchanged
inside a compiled-grade verifier image that has no daydream wheel. It wires a
bounded per-pair judge prompt, two isolated external judge clients (Anthropic
Messages + OpenAI-compatible) behind one ``complete_json`` seam, strict verdict
parsing, a shared retry/redirect/timeout policy, a concurrency-10 runner, and
``run_verifier`` which writes ``reward.json`` / ``reward-details.json``
atomically.

The external judge surface is fail-closed and bounded: exactly
``anthropic | openai-compatible`` providers are accepted, the judge host must
sit in the configured allowlist (``DAYDREAM_JUDGE_ALLOWED_HOSTS``; own-host
fallback when absent), redirects are bounded to ``_MAX_REDIRECTS``
credential-preserving same-origin hops whose targets are re-validated against
the allowlist, response/reasoning payloads are size-capped (rejected, never
truncated-and-accepted), and every failure path writes typed bounded (redacted)
diagnostics to ``reward.json`` / ``reward-details.json`` -- never a bare exit
and never an unbounded or credential-bearing error.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
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

JUDGE_PROMPT_TEMPLATE = (
    Path(__file__).with_name("judge_prompt.md").read_text(encoding="utf-8")
)

_PROMPT_CAP_BYTES = 24 * 1024
_MAX_RETRIES = 3
_REQUEST_TIMEOUT = 60.0

# Hardening caps: response/reasoning payloads are rejected -- never truncated
# and accepted -- above these sizes; redirects are bounded; diagnostics are
# bounded and redacted before they reach any artifact or log.
_RESPONSE_CAP_BYTES = 256 * 1024
_REASONING_CAP_BYTES = 32 * 1024
_MAX_REDIRECTS = 3
_ERROR_TEXT_CAP_BYTES = 4096
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

_ESCAPED_FINDING_TAGS = {
    "<gold_finding>": "&lt;gold_finding&gt;",
    "</gold_finding>": "&lt;/gold_finding&gt;",
    "<candidate_finding>": "&lt;candidate_finding&gt;",
    "</candidate_finding>": "&lt;/candidate_finding&gt;",
}

# Fixed marker rendered for each null location component of a locationless
# finding so the judge sees an explicit all-null location rather than empty,
# shape-ambiguous values. Reused across all six location fields.
_LOCATIONLESS_MARKER = "<none>"


def _escape_finding_delimiters(text: str) -> str:
    """Neutralize the ``<..._finding>`` block delimiters in untrusted text.

    Every untrusted scalar field (title, severity, path, start_line, end_line)
    as well as both bodies is escaped at render time. A literal closing tag
    inside one block would terminate its structural block early and leak the
    remainder into the other role's region; a literal opening tag could shift
    the boundary or synthesize extra blocks. Rewriting every delimiter to its
    entity form means injected text can never form a real structural delimiter.
    """
    escaped = text
    for delimiter, entity in _ESCAPED_FINDING_TAGS.items():
        escaped = escaped.replace(delimiter, entity)
    return escaped


_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


_REDACTION_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9]+"),
    re.compile(r"sk-or-[A-Za-z0-9]+"),
    re.compile(r"Bearer [A-Za-z0-9._~+/=-]+"),
    re.compile(r"x-api-key:?\s*\S+"),
    re.compile(r"[0-9a-fA-F]{32,}"),
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),
)


def _normalize_host(host: str | None) -> str:
    """Lowercase ``host`` and strip a trailing dot; never raises."""
    if host is None:
        return ""
    return host.strip().lower().rstrip(".")


def _effective_allowlist(base_url: str, env: dict[str, Any]) -> set[str]:
    """Resolve the judge-host allowlist from env; absent -> own-host fail-closed.

    A non-empty ``DAYDREAM_JUDGE_ALLOWED_HOSTS`` (whitespace/comma-separated)
    wins; otherwise the effective allowlist is exactly the resolved judge
    host of ``base_url`` so an unconfigured verifier can only ever reach its
    own judge endpoint -- never an arbitrary host.
    """
    raw = (env or {}).get("DAYDREAM_JUDGE_ALLOWED_HOSTS")
    if raw:
        hosts = {
            _normalize_host(host)
            for host in re.split(r"[\s,]+", str(raw))
            if host.strip()
        }
        if hosts:
            return hosts
    return {_normalize_host(urllib.parse.urlsplit(base_url).hostname)}


def _validate_base_url(url: str, allowlist: set[str]) -> str:
    """Validate ``url`` against the hardened judge-request contract.

    Rejects userinfo, any query string or fragment, non-HTTPS remote schemes
    (``http`` is permitted only to a loopback host), and any host outside
    ``allowlist``. Returns ``url`` unchanged; every rejection is a bounded
    ``VerifierError`` naming only the rejected form -- never URL content.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise VerifierError("judge URL must not contain userinfo")
    if parsed.query:
        raise VerifierError("judge URL must not contain a query string")
    if parsed.fragment:
        raise VerifierError("judge URL must not contain a fragment")
    host = _normalize_host(parsed.hostname)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and host in _LOOPBACK_HOSTS
    ):
        raise VerifierError("judge URL must use https (loopback http allowed)")
    if not host or host not in allowlist:
        raise VerifierError("judge host is not in the verifier allowlist")
    return url


def _resolve_redirect(request_url: str, location: str, allowlist: set[str]) -> str:
    """Resolve a redirect ``location`` against ``request_url`` and host-check it.

    A relative ``Location`` is resolved against the request URL first, then the
    resolved target is validated against ``allowlist`` exactly like an initial
    base URL -- a cross-host, out-of-allowlist, or malformed target is a
    terminal bounded ``VerifierError``.
    """
    resolved = urllib.parse.urljoin(request_url, location)
    return _validate_base_url(resolved, allowlist)


def _response_bytes(response: Any) -> bytes:
    """Return the response payload as bytes, preferring ``.content``."""
    content = getattr(response, "content", None)
    if content is not None:
        return bytes(content)
    text = getattr(response, "text", "")
    return str(text).encode("utf-8")


def _redact_text(text: str) -> str:
    """Replace credential-like content with ``<redacted>``."""
    for pattern in _REDACTION_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return text


def _bounded_error(text: object) -> str:
    """Redact ``text`` and bound it to ``_ERROR_TEXT_CAP_BYTES`` UTF-8 bytes.

    Redaction runs before the truncation so a credential in the first bytes can
    never survive a cut; truncation lands on a UTF-8 byte boundary. Empty input
    returns ``""``; any non-empty input stays non-empty.
    """
    if not text:
        return ""
    redacted = _redact_text(str(text))
    encoded = redacted.encode("utf-8")
    if len(encoded) <= _ERROR_TEXT_CAP_BYTES:
        return redacted
    return encoded[:_ERROR_TEXT_CAP_BYTES].decode("utf-8", errors="ignore")



def _render_filled(
    template: str,
    gold: dict[str, Any],
    candidate: dict[str, Any],
    *,
    gold_body: str,
    candidate_body: str,
    escape: bool = True,
) -> str:
    """Render one gold/candidate pair into ``template`` for a judge.

    Every untrusted scalar field and each body is passed through
    ``_escape_finding_delimiters`` so no literal delimiter can form a structural
    block. With ``escape=False`` the raw payload is returned instead -- the
    caller's pre-inflation budget yardstick.
    """

    def _field(value: object, none_marker: str = "") -> str:
        """Render a scalar field, optionally marking a ``None`` component.

        A locationless review finding (no file or line) renders its null
        location fields as ``none_marker`` (the fixed ``_LOCATIONLESS_MARKER``)
        so the judge sees an explicit all-null location rather than an empty,
        shape-ambiguous value. The marker is supplied by the caller -- never
        derived from untrusted input -- and is run through the same escaping
        path as every other field (respecting ``escape=False`` for the raw
        budget yardstick).
        """
        if value is None and none_marker:
            text = none_marker
        else:
            text = str(value or "")
        return _escape_finding_delimiters(text) if escape else text

    return template.format(
        gold_title=_field(gold.get("title")),
        gold_severity=_field(gold.get("severity")),
        gold_path=_field(gold.get("path"), _LOCATIONLESS_MARKER),
        gold_start_line=_field(gold.get("start_line"), _LOCATIONLESS_MARKER),
        gold_end_line=_field(gold.get("end_line"), _LOCATIONLESS_MARKER),
        gold_body=_field(gold_body),
        candidate_title=_field(candidate.get("title")),
        candidate_severity=_field(candidate.get("severity")),
        candidate_path=_field(candidate.get("path"), _LOCATIONLESS_MARKER),
        candidate_start_line=_field(candidate.get("start_line"), _LOCATIONLESS_MARKER),
        candidate_end_line=_field(candidate.get("end_line"), _LOCATIONLESS_MARKER),
        candidate_body=_field(candidate_body),
    )


def render_pair_prompt(gold: dict, candidate: dict, *, template: str) -> str:
    """Render a bounded, untrusted-fenced prompt for one gold/candidate pair.

    The untrusted finding fields are escaped by ``_render_filled`` so injected
    delimiters can never form structural blocks. The 24 KiB budget is checked
    against the raw, pre-escape payload — the same yardstick ``verifier_core``
    uses when it bounds each field — because the escaping fence can only
    inflate, and a pair the verifier accepts must not be voided whole by that
    inflation. A pair that exceeds the raw budget fails deterministically with
    ``VerifierError``: never truncate and never report a partial result.
    """
    gold_body = gold.get("body", "") or ""
    candidate_body = candidate.get("body", "") or ""
    # Budget against the raw, pre-escape payload. verifier_core binds each field
    # in raw bytes, and the escaping pass is a delimiter-fence that can only
    # inflate. Measuring the escaped pair makes a validator-legal pair (two dense
    # 8 KiB bodies) trip the cap and fail the whole task -- a budget incoherence.
    # Fencing inflates only when delimiters are present; such a pair is judged,
    # never voided. Truly oversized raw input still fails deterministically.
    raw = _render_filled(
        template,
        gold,
        candidate,
        gold_body=gold_body,
        candidate_body=candidate_body,
        escape=False,
    )
    if len(raw.encode("utf-8")) > _PROMPT_CAP_BYTES:
        raise VerifierError("rendered pair exceeds 24 KiB")
    return _render_filled(
        template,
        gold,
        candidate,
        gold_body=gold_body,
        candidate_body=candidate_body,
    )


def parse_verdict(raw: object) -> verifier_core.Verdict:
    """Validate a raw judge verdict dict and return a ``verifier_core.Verdict``.

    ``gold_id``/``candidate_id`` are placeholders the caller (``judge_pairs``)
    stamps onto the returned verdict. Any violation — wrong type, out-of-range
    confidence, missing key, unknown key, non-dict input — raises
    ``VerifierError``; never silently coerces a fallback value.
    """
    if not isinstance(raw, dict):
        raise VerifierError("verdict must be a JSON object")
    verifier_core.validate_exact_keys(raw, {"match", "confidence", "reasoning"}, "verdict")
    match = raw["match"]
    if not isinstance(match, bool):
        raise VerifierError(
            f"verdict 'match' must be a boolean, got {_bounded_error(repr(match))}"
        )
    confidence = raw["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VerifierError(
            f"verdict 'confidence' must be a number in [0,1], got {_bounded_error(repr(confidence))}"
        )
    if not 0.0 <= confidence <= 1.0:
        raise VerifierError(
            f"verdict 'confidence' must be in [0,1], got {_bounded_error(repr(confidence))}"
        )
    reasoning = raw["reasoning"]
    if not isinstance(reasoning, str):
        raise VerifierError(
            f"verdict 'reasoning' must be a string, got {_bounded_error(repr(reasoning))}"
        )
    # Reasoning is capped at 32 KiB and rejected -- never truncated-and-accepted
    # -- so an oversized/untrusted value can never bloat a diagnostic.
    if len(reasoning.encode("utf-8")) > _REASONING_CAP_BYTES:
        raise VerifierError("verdict reasoning exceeds 32 KiB")
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
    """Parse an httpx-like response through the ``content`` extraction callable.

    The raw body is size-capped (``_RESPONSE_CAP_BYTES``) BEFORE any status
    handling, so an oversized body is rejected regardless of status code -- a
    non-2xx (4xx/5xx) judge body must not bypass the cap. An over-cap body is a
    terminal ``VerifierError`` (rejected, never truncated-and-accepted).
    """
    body = _response_bytes(response)
    if len(body) > _RESPONSE_CAP_BYTES:
        raise VerifierError("judge response body exceeds 256 KiB")  # terminal, never truncated
    status_code = getattr(response, "status_code", None)
    if status_code is None or not 200 <= int(status_code) < 300:
        body_text = body.decode("utf-8", errors="replace")
        code = int(status_code) if status_code is not None else -1
        # 429 is retryable (rate limit); all other 4xx are terminal client errors.
        if 400 <= code < 500 and code != 429:
            raise VerifierError(
                f"Judge request failed with HTTP {status_code}: {_bounded_error(body_text)}"
            )
        raise _Retryable(
            f"Judge request failed with HTTP {status_code}: {_bounded_error(body_text)}"
        )
    try:
        parsed_body = response.json()
    except Exception as exc:
        raise VerifierError(
            f"Judge response was not valid JSON: {_bounded_error(str(exc))}"
        ) from exc
    text = content(parsed_body)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VerifierError(
            f"Judge text content was not valid JSON: {_bounded_error(str(exc))}"
        ) from exc
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
    allowlist: set[str],
) -> dict[str, Any]:
    """Single retry/redirect/timeout policy shared by both judge clients.

    - Up to 3 attempts. Transport exceptions (including timeout) and HTTP
      429/5xx retry with exponential backoff (`2 ** attempt`); a terminal 4xx
      (non-429) and a malformed-JSON parse are not retried.
    - 3xx responses: bounded credential-preserving redirects. At most
      ``_MAX_REDIRECTS`` hops per call; each next target is resolved against the
      request URL first (relative ``Location``) and must be inside the judge-host
      ``allowlist``. Configured auth headers are preserved across the hop and
      server-provided response headers are never replayed. A redirect-target
      rejection or an exhausted hop count is a terminal ``VerifierError``,
      never retried.
    - After 3 failed attempts, raise ``VerifierError`` — never a partial result.
    """
    current_url = url
    hop = 0
    for attempt in range(_MAX_RETRIES):
        while True:
            try:
                try:
                    response = await http.post(
                        current_url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT
                    )
                except Exception as exc:
                    raise _Retryable(f"Judge request failed: {exc}") from exc
                status_code = getattr(response, "status_code", None)
                if status_code is not None and 300 <= int(status_code) < 400:
                    if hop >= _MAX_REDIRECTS:
                        raise VerifierError("judge request exceeded maximum redirects")
                    response_headers = getattr(response, "headers", {}) or {}
                    location = response_headers.get("location")
                    if not location:
                        raise VerifierError("judge request redirected without a Location")
                    # Resolve relative Location against the request URL and
                    # fail closed on cross-host/out-of-allowlist targets. The
                    # configured ``headers`` are intentionally NOT replaced by
                    # the response headers, so auth survives the hop and server
                    # headers are never replayed.
                    current_url = _resolve_redirect(current_url, location, allowlist)
                    hop += 1
                    continue
                return _parse_json_response(response, content=content)
            except _Retryable:
                if attempt == _MAX_RETRIES - 1:
                    raise VerifierError("Judge request failed after retries")
                await asyncio.sleep(2**attempt)
                break
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
    """Small Anthropic Messages API client returning strict parsed JSON verdicts.

    Validates the initial Messages URL against the effective judge-host
    allowlist before any request (fail-closed); redirects are bounded and
    allowlist-checked inside the shared ``_complete_json_with_http`` policy.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        http: _AsyncHttpClient | None = None,
        allowlist: set[str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.http = http
        self.allowlist = allowlist

    async def complete_json(
        self, *, user: str, system: str = "", max_tokens: int = 512
    ) -> dict[str, Any]:
        effective = self.allowlist or _effective_allowlist(
            _ANTHROPIC_MESSAGES_URL, {}
        )
        # Fail closed before any request: a forced disallowed allowlist must
        # reject the initial URL here, never after a POST has been issued.
        _validate_base_url(_ANTHROPIC_MESSAGES_URL, effective)
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
                allowlist=effective,
            )
        async with httpx.AsyncClient() as http:
            return await _complete_json_with_http(
                http,
                url=_ANTHROPIC_MESSAGES_URL,
                payload=payload,
                headers=headers,
                content=_anthropic_text,
                allowlist=effective,
            )


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_KEY_PREFIX = "sk-or-"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_CHAT_COMPLETIONS_PATH = "/chat/completions"


def resolve_base_url(api_key: str, base_url_env: str | None) -> str:
    """Resolve the Chat Completions base URL from the environment.

    An explicit base URL always wins; an ``sk-or-`` OpenRouter key with no pin
    routes to OpenRouter; anything else defaults to OpenAI direct. The resolved
    URL is only a candidate: it is validated against the effective judge-host
    allowlist (scheme/host/form) at the client build and initial-request sites
    before any judge call.
    """
    if base_url_env:
        return base_url_env
    if api_key.startswith(_OPENROUTER_KEY_PREFIX):
        return _OPENROUTER_BASE_URL
    return _OPENAI_DEFAULT_BASE_URL


def _openai_content(body: dict[str, Any]) -> str:
    """Extract ``choices[0].message.content`` from an OpenAI-compatible response body."""
    if not isinstance(body, dict):
        raise VerifierError("Judge response body was not an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VerifierError("Judge response missing a choices list")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise VerifierError("Judge response choice was not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise VerifierError("Judge response missing a message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise VerifierError("Judge response message content was not a string")
    text = content.strip()
    if not text:
        raise VerifierError("Judge response message content was empty")
    return text


class OpenAIJudgeClient:
    """Small OpenAI-compatible Chat Completions client returning strict parsed verdicts.

    Validates the initial Chat Completions URL against the effective
    judge-host allowlist before any request (fail-closed); redirects are
    bounded and allowlist-checked inside the shared ``_complete_json_with_http``
    policy -- identical to the Anthropic client.
    """
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str,
        http: _AsyncHttpClient | None = None,
        allowlist: set[str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.http = http
        self.allowlist = allowlist

    async def complete_json(
        self, *, user: str, system: str = "", max_tokens: int = 512
    ) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + _CHAT_COMPLETIONS_PATH
        effective = self.allowlist or _effective_allowlist(
            self.base_url, {}
        )
        # Fail closed before any request: a base URL host outside the allowlist
        # is rejected here, never after a POST has been issued.
        _validate_base_url(url, effective)
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
        if self.http is not None:
            return await _complete_json_with_http(
                self.http,
                url=url,
                payload=payload,
                headers=headers,
                content=_openai_content,
                allowlist=effective,
            )
        async with httpx.AsyncClient() as http:
            return await _complete_json_with_http(
                http,
                url=url,
                payload=payload,
                headers=headers,
                content=_openai_content,
                allowlist=effective,
            )


_JUDGE_CONCURRENCY = 10
_MAX_PAIRS = 5000
_MAX_JUDGE_TOKENS = 512
_JUDGE_SYSTEM = (
    "You determine whether two code-review findings describe the same defect. "
    "Reply only with a single JSON object."
)


async def judge_pairs(
    gold: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    client: Any,
) -> list[verifier_core.Verdict]:
    """Return one Verdict per gold x candidate pair, at most 10 in flight.

    Enforces the fixed 5,000-pair cap before any judge call (fail-whole before
    judging). Verdicts are collected in gold-major, candidate-minor order
    regardless of completion order for deterministic ``reward-details`` output.
    """
    if len(gold) * len(candidates) > _MAX_PAIRS:
        raise VerifierError("pair count exceeds 5000")
    pairs = [(g, c) for g in gold for c in candidates]
    semaphore = asyncio.Semaphore(_JUDGE_CONCURRENCY)

    async def _judge(pair: tuple[dict[str, Any], dict[str, Any]]) -> verifier_core.Verdict:
        g, c = pair
        async with semaphore:
            raw = await client.complete_json(
                user=render_pair_prompt(g, c, template=JUDGE_PROMPT_TEMPLATE),
                system=_JUDGE_SYSTEM,
                max_tokens=_MAX_JUDGE_TOKENS,
            )
        verdict = parse_verdict(raw)
        return verifier_core.Verdict(
            gold_id=g.get("finding_id", ""),
            candidate_id=c.get("candidate_id", ""),
            match=verdict.match,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
        )

    return await asyncio.gather(*(_judge(pair) for pair in pairs))


_ENV_PROVIDER = "DAYDREAM_JUDGE_PROVIDER"
_ENV_MODEL = "DAYDREAM_JUDGE_MODEL"
_ENV_API_KEY = "DAYDREAM_JUDGE_API_KEY"
_ENV_BASE_URL = "DAYDREAM_JUDGE_BASE_URL"
_ENV_ALLOWED_HOSTS = "DAYDREAM_JUDGE_ALLOWED_HOSTS"
_ENV_ARTIFACT_PATH = "DAYDREAM_JUDGE_ARTIFACT_PATH"
_ENV_OUT_PATH = "DAYDREAM_JUDGE_OUT_PATH"
_DEFAULT_PROVIDER = "anthropic"


class _CountingClient:
    """Wraps a judge client to observe request counts and capture errors."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.requests = 0
        self.errors: list[str] = []

    async def complete_json(self, **kwargs: Any) -> Any:
        self.requests += 1
        try:
            return await self._inner.complete_json(**kwargs)
        except Exception as exc:
            self.errors.append(str(exc))
            raise


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise VerifierError(f"input file not found: {path}") from None
    except (json.JSONDecodeError, OSError) as exc:
        raise VerifierError(f"could not read {path}: {exc}") from exc


def _read_artifact_bytes(path: str | Path) -> dict[str, Any]:
    """Read the candidate artifact as raw bytes, size-checked and parsed in one step.

    The raw byte size is checked against ``verifier_core.MAX_ARTIFACT_BYTES``
    BEFORE any parse -- a whitespace-inflated payload over the cap fails on its
    raw size alone, never reaching the judge. A ``JSONDecodeError`` becomes a
    ``VerifierError`` naming only the path (never content).
    """
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        raise VerifierError(f"input file not found: {Path(path)}") from None
    except OSError as exc:
        raise VerifierError(f"could not read {Path(path)}: {exc}") from exc
    if len(raw) > verifier_core.MAX_ARTIFACT_BYTES:
        raise VerifierError("candidate artifact exceeds 1 MiB (raw bytes)")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise VerifierError(f"candidate artifact is not valid JSON: {Path(path)}") from None
    return parsed


def _read_gold_bytes(gold_path: Path, expected_sha256: str) -> list[Any]:
    """Read the gold set as raw bytes, digest-checked, parsed, and list-validated.

    Symmetric to ``_read_artifact_bytes`` for the trusted gold half: read the
    raw bytes, verify the sha256 against the compiler-rendered sentinel, parse
    the JSON, and assert the payload is a list. No raw-size cap applies -- gold
    is shipped, read-only task data while the candidate artifact is the
    attacker-controlled surface.
    """
    try:
        gold_bytes = gold_path.read_bytes()
    except FileNotFoundError:
        raise VerifierError(f"input file not found: {gold_path}") from None
    except OSError as exc:
        raise VerifierError(f"could not read {gold_path}: {exc}") from exc
    if hashlib.sha256(gold_bytes).hexdigest() != expected_sha256:
        raise VerifierError("gold digest mismatch")
    try:
        gold_raw = json.loads(gold_bytes)
    except json.JSONDecodeError:
        raise VerifierError(f"gold set is not valid JSON: {gold_path}") from None
    if not isinstance(gold_raw, list):
        raise VerifierError("gold set must be a JSON list")
    return gold_raw


def _load_verifier_metadata(gold_path: Path) -> dict[str, Any]:
    """Load the sibling immutable task-bound verifier metadata beside the gold file.

    Requires the ``{schema_version, case_id, base_ref, head_ref,
    template_version, gold_sha256}`` object the compiler renders per case; a
    missing/dict-violating field raises ``VerifierError``. The versioned shape
    is gated: ``schema_version`` must be 1 (parity with the candidate
    artifact's own gate in ``verifier_core``) and ``template_version`` must be
    a non-empty string, so a mismatched metadata schema fails the task whole
    rather than mis-binding it.
    """
    meta = _read_json(gold_path.parent / "verifier-metadata.json")
    if not isinstance(meta, dict):
        raise VerifierError("verifier metadata must be a JSON object")
    for field in (
        "schema_version",
        "case_id",
        "base_ref",
        "head_ref",
        "template_version",
        "gold_sha256",
    ):
        if field not in meta:
            raise VerifierError(f"verifier metadata missing required field {field}")
    if meta["schema_version"] != 1:
        raise VerifierError(
            f"unsupported verifier metadata schema_version {meta['schema_version']!r}"
        )
    if not isinstance(meta["template_version"], str) or not meta["template_version"].strip():
        raise VerifierError(
            "verifier metadata template_version must be a non-empty string"
        )
    return meta


def _atomic_write(out_dir: Path, filename: str, payload: str) -> None:
    """Write a file atomically via temp + rename so a crash never leaves a partial file."""
    tmp = out_dir / f".{filename}.{os.getpid()}.tmp"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, out_dir / filename)


def _error_details(provider: str, model: str, request_counts: dict[str, int], errors: list[str]) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "request_counts": request_counts,
        "errors": errors,
        "verdicts": [],
        "matches": [],
        "unmatched_gold": [],
        "unmatched_candidates": [],
    }


def _write_error_artifact(
    out_dir: str | Path,
    provider: str,
    model: str,
    request_counts: dict[str, int],
    errors: list[str],
    gold_count: int,
) -> verifier_core.Reward:
    """Write the fail-whole error artifacts and return the error reward.

    Every failure path -- validation, binding, digest, judging, exhausted
    retries, a missing client, or an unexpected runtime exception -- funnels
    through here so ``reward.json``/``reward-details.json`` are always written
    with typed bounded diagnostics, never a bare exit. ``errors`` must already
    be bounded/redacted before the call.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    error_reward = verifier_core.Reward(
        reward=0.0, gold_count=gold_count, verifier_error=1
    )
    details = _error_details(provider, model, request_counts, errors)
    _atomic_write(out_dir, "reward.json", verifier_core.reward_to_json(error_reward))
    _atomic_write(out_dir, "reward-details.json", json.dumps(details))
    return error_reward


def run_verifier(
    gold_path: str | Path,
    artifact_path: str | Path,
    out_dir: str | Path,
    *,
    client: Any,
    env: dict[str, Any],
) -> verifier_core.Reward:
    """Validate gold + the candidate artifact, judge all pairs, score, and write atomically.

    Validation order (issue #817): the candidate artifact is read as raw bytes
    (size-checked before parse), validated to its exact schema, then bound to
    the task's immutable ``verifier-metadata.json`` (case id + base/head refs);
    the gold is then read as raw bytes and its sha256 must match the
    compiler-rendered ``gold_sha256`` sentinel before it is parsed and validated
    (canonical/unique gold ids). Any ``VerifierError`` (validation, binding,
    digest, parsing, judging, exhausted retries, or a missing client) -- and
    any unexpected runtime exception -- becomes a typed bounded diagnostic that
    still writes ``Reward(reward=0, verifier_error=1)`` plus both output files
    atomically (temp + rename); the task fails whole, never reporting a
    partial score and never escaping to a bare exit. Error text is redacted and
    size-bounded before it reaches any artifact. Never emits source or diffs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = env or {}
    provider = env.get(_ENV_PROVIDER) or _DEFAULT_PROVIDER
    model = env.get(_ENV_MODEL) or ""
    request_counts: dict[str, int] = {"requests": 0}
    errors: list[str] = []
    gold_parsed: list[verifier_core.GoldFinding] = []
    detail_prefix = {"provider": provider, "model": model, "request_counts": request_counts, "errors": errors}

    try:
        if client is None:
            raise VerifierError("no judge client configured (missing DAYDREAM_JUDGE_*)")
        artifact_raw = _read_artifact_bytes(artifact_path)
        candidates = verifier_core.validate_candidate_artifact(artifact_raw)

        metadata = _load_verifier_metadata(Path(gold_path))
        for field in ("case_id", "base_ref", "head_ref"):
            if artifact_raw[field] != metadata[field]:
                raise VerifierError(f"candidate {field} does not match the bound task")

        gold_raw = _read_gold_bytes(Path(gold_path), metadata["gold_sha256"])
        gold_parsed = verifier_core.validate_gold_set(
            gold_raw, case_id=metadata.get("source_case_id")
        )

        verdicts: list[verifier_core.Verdict] = []
        matches: set[tuple[str, str]] = set()
        counting: _CountingClient | None = None

        if gold_parsed and artifact_raw.get("findings"):
            counting = _CountingClient(client)
            verdicts = asyncio.run(judge_pairs(gold_raw, artifact_raw["findings"], client=counting))
            cand_ids = [
                c.get("candidate_id", "") for c in artifact_raw["findings"]
            ]
            gold_ids = [g.finding_id for g in gold_parsed]
            retained = verifier_core.retained_edges(verdicts, gold_ids, cand_ids)
            matches = verifier_core.maximum_matching(retained, gold_ids, cand_ids)
            request_counts["requests"] = counting.requests
            if counting.errors:
                errors.extend(_bounded_error(str(e)) for e in counting.errors)

        reward = verifier_core.score_review(gold_parsed, artifact_raw, verdicts)
        inner = verifier_core.reward_details(gold_parsed, candidates, verdicts, matches)
        details = {**inner, **detail_prefix}
        _atomic_write(out_dir, "reward.json", verifier_core.reward_to_json(reward))
        _atomic_write(out_dir, "reward-details.json", json.dumps(details))
        return reward
    except VerifierError as exc:
        errors.insert(0, _bounded_error(str(exc)))
        return _write_error_artifact(
            out_dir, provider, model, request_counts, errors, len(gold_parsed)
        )
    except Exception as exc:
        # Unexpected runtime failures must not escape to a bare exit: they
        # become a typed bounded diagnostic that still writes both artifacts.
        errors.insert(0, _bounded_error(f"unexpected verifier failure: {exc}"))
        return _write_error_artifact(
            out_dir, provider, model, request_counts, errors, len(gold_parsed)
        )


def _build_client(env: dict[str, Any]) -> Any:
    """Build the judge client from the DAYDREAM_JUDGE_* env surface.

    Provider is fail-closed: exactly ``anthropic`` | ``openai-compatible`` is
    accepted (absent -> default ``anthropic``); anything else raises before any
    request. The provider's base URL is resolved and the initial request URL is
    validated against the effective judge-host allowlist at build time.
    """
    provider = env.get(_ENV_PROVIDER) or _DEFAULT_PROVIDER
    model = env.get(_ENV_MODEL)
    api_key = env.get(_ENV_API_KEY)
    if not model or not api_key:
        raise VerifierError("missing DAYDREAM_JUDGE_MODEL or DAYDREAM_JUDGE_API_KEY")
    if provider not in {"anthropic", "openai-compatible"}:
        raise VerifierError(
            f"unsupported DAYDREAM_JUDGE_PROVIDER '{provider}'; expected anthropic or openai-compatible"
        )
    if provider == "anthropic":
        allowlist = _effective_allowlist(_ANTHROPIC_MESSAGES_URL, env)
        # Fail-closed at build time, matching the openai-compatible branch: the
        # initial Messages URL is validated against the effective allowlist
        # before any request can be issued, so both providers share identical
        # validation timing.
        _validate_base_url(_ANTHROPIC_MESSAGES_URL, allowlist)
        return AnthropicJudgeClient(
            api_key,
            model,
            allowlist=allowlist,
        )
    base_url = resolve_base_url(api_key, env.get(_ENV_BASE_URL))
    allowlist = _effective_allowlist(base_url, env)
    initial_url = base_url.rstrip("/") + _CHAT_COMPLETIONS_PATH
    _validate_base_url(initial_url, allowlist)  # fail-closed before any request
    return OpenAIJudgeClient(api_key, model, base_url=base_url, allowlist=allowlist)


def _env_path(name: str, default: str) -> Path:
    """Return ``Path(os.environ[name])`` when set, else ``Path(default)``.

    The compiled-image defaults are unchanged; the overrides only relocate the
    artifact/out paths for isolated subprocess runs (e.g. the isolation test).
    """
    value = os.environ.get(name)
    return Path(value) if value else Path(default)


def _emit_reward(reward: verifier_core.Reward) -> int:
    """Print the reward payload JSON and return the verifier-error exit code.

    Shared by both ``main()`` terminal paths (a fail-closed client build
    rejection and the completed ``run_verifier``) so the two failure/success
    emissions cannot drift. ``Reward.to_dict()`` always exists (the compiled
    verifier_core twin is byte-identical); the fallback dict is a defensive
    guard that keeps the shape stable even if a duck-typed reward lacks it.
    """
    payload = (
        reward.to_dict()
        if hasattr(reward, "to_dict")
        else {
            "verifier_error": int(getattr(reward, "verifier_error", 0)),
            "reward": float(getattr(reward, "reward", 0.0)),
        }
    )
    print(json.dumps(payload))
    return 1 if getattr(reward, "verifier_error", 0) else 0


def main() -> int:
    """Compiled entry: resolve the §10 paths, read real env, judge, print reward JSON.

    Provider selection is fail-closed: an unsupported ``DAYDREAM_JUDGE_PROVIDER``
    or an out-of-allowlist judge host writes a typed bounded diagnostic artifact
    (``reward.json``/``reward-details.json``) instead of a barren ``client=None``
    exit with no provider reason. ``DAYDREAM_JUDGE_ARTIFACT_PATH`` /
    ``DAYDREAM_JUDGE_OUT_PATH`` relocate the compiled defaults for isolated
    subprocess runs; the compiled defaults ``/logs/artifacts/review.json`` and
    ``/logs/verifier`` are unchanged.
    """
    gold_path = Path(__file__).with_name("golden-review.json")
    artifact_path = _env_path(_ENV_ARTIFACT_PATH, "/logs/artifacts/review.json")
    out_dir = _env_path(_ENV_OUT_PATH, "/logs/verifier")
    env = {
        name: os.environ.get(name)
        for name in (
            _ENV_PROVIDER,
            _ENV_MODEL,
            _ENV_API_KEY,
            _ENV_BASE_URL,
            _ENV_ALLOWED_HOSTS,
        )
    }
    try:
        client = _build_client(env)
    except VerifierError as exc:
        if env.get(_ENV_MODEL) and env.get(_ENV_API_KEY):
            # Fail-closed provider/host rejection: a typed bounded diagnostic
            # artifact naming only the rejected form -- never a barren exit.
            provider = env.get(_ENV_PROVIDER) or _DEFAULT_PROVIDER
            model = env.get(_ENV_MODEL) or ""
            reward = _write_error_artifact(
                out_dir, provider, model, {"requests": 0}, [_bounded_error(str(exc))], 0
            )
            return _emit_reward(reward)
        # Missing MODEL/API_KEY keeps the compiled path: run_verifier emits its
        # own "no judge client configured" typed diagnostic.
        client = None
    reward = run_verifier(gold_path, artifact_path, out_dir, client=client, env=env)
    return _emit_reward(reward)


if __name__ == "__main__":
    sys.exit(main())
