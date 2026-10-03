"""Standalone Harbor judge verifier: stdlib plus httpx, with no Daydream dependency. Three
explicit provider adapters share bounded prompts, strict verdicts, retry/timeouts, and
concurrency limits. HTTP requests use validated host allowlists and bounded same-origin
redirects; oversized responses are rejected whole. Invalid agent candidates score zero.
Missing/unreadable inputs and infrastructure failures write only bounded, redacted
reward details and remain unscored.
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


def _terminate_proc(proc: Any) -> None:
    """Best-effort kill of a spawned CLI child; a no-op on the seam fakes.

    A hung or oversized child must never outlive the verifier, so every
    timeout/over-cap exit kills it. Kill failures are swallowed: the
    timeout/over-cap outcome the caller is recording must not be masked.
    """
    kill = getattr(proc, "kill", None) or getattr(proc, "terminate", None)
    if kill is not None:
        try:
            kill()
        except Exception:
            pass


async def _claude_cli_stdout(proc: Any) -> str:
    """Read bounded stdout and settle the child within one deadline; kill on timeout or
    overflow. Seam fakes may provide captured text/bytes or communicate() instead.
    """
    try:
        stream = getattr(proc, "stdout", None)
        if stream is None:
            communicate = getattr(proc, "communicate", None)
            if communicate is None:
                return ""
            stream, _stderr = await asyncio.wait_for(communicate(), timeout=_REQUEST_TIMEOUT)
        if isinstance(stream, (str, bytes)):
            raw = stream.encode("utf-8") if isinstance(stream, str) else stream
            if len(raw) > _RESPONSE_CAP_BYTES:
                raise VerifierError(
                    f"claude-cli judge output exceeds {_RESPONSE_CAP_BYTES // 1024} KiB"
                )
            return stream if isinstance(stream, str) else raw.decode("utf-8", errors="replace")
        read = getattr(stream, "read", None)
        if read is None:
            return ""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _REQUEST_TIMEOUT
        raw_output = bytearray()
        async with asyncio.timeout_at(deadline):
            while True:
                # Buffered reads may not yield: retain the explicit wall-clock check.
                if loop.time() >= deadline:
                    raise TimeoutError
                chunk = await read(_STDOUT_CHUNK_BYTES)
                if not chunk:
                    break
                raw_output.extend(chunk)
                if len(raw_output) > _RESPONSE_CAP_BYTES:
                    raise VerifierError(
                        f"claude-cli judge output exceeds {_RESPONSE_CAP_BYTES // 1024} KiB"
                    )
            # EOF can precede exit; include child settlement in the same deadline.
            wait = getattr(proc, "wait", None)
            if wait is not None:
                if loop.time() >= deadline:
                    raise TimeoutError
                await wait()
        return raw_output.decode("utf-8", errors="replace")
    except (TimeoutError, VerifierError):
        _terminate_proc(proc)
        raise


class _InputFileNotFound(verifier_core.VerifierError):
    """Missing or unreadable input is infrastructure failure, including an unavailable
    candidate artifact. Keep it separate from malformed agent content so mount/path
    failures cannot become numeric zero scores.
    """

JUDGE_PROMPT_TEMPLATE = (
    Path(__file__).with_name("judge_prompt.md").read_text(encoding="utf-8")
)

_PROMPT_CAP_BYTES = 24 * 1024
_MAX_RETRIES = 3
_REQUEST_TIMEOUT = 60.0

# Reject oversized payloads whole; bound redirects and sanitize diagnostics.
_RESPONSE_CAP_BYTES = 256 * 1024
# Incremental read chunk for _claude_cli_stdout: verifier memory stays bounded
# by _RESPONSE_CAP_BYTES even while a still-streaming child is mid-output.
_STDOUT_CHUNK_BYTES = 64 * 1024
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

# Explicit null-location markers distinguish absent locations from empty values.
_LOCATIONLESS_MARKER = "<none>"


def _escape_finding_delimiters(text: str) -> str:
    """Escape structural finding delimiters in every untrusted scalar and body. Embedded
    tags must never close, shift, or manufacture gold/candidate blocks.
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
    """Use the explicit whitespace/comma-separated judge allowlist, otherwise permit only
    the resolved judge host.
    """
    raw = (env or {}).get(_ENV_ALLOWED_HOSTS)
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
    """Require an allowed host, no userinfo/query/fragment, and HTTPS except loopback HTTP.
    Return the unchanged URL; bounded rejections never disclose URL content.
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
    """Resolve relative Location values against the request, then revalidate origin and
    allowlist. Malformed or forbidden targets fail terminally.
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


def _bounded_repr(value: object) -> str:
    """Redact and bound repr(value) through the shared error sanitizer."""
    return _bounded_error(repr(value))


def _render_filled(
    template: str,
    gold: dict[str, Any],
    candidate: dict[str, Any],
    *,
    gold_body: str,
    candidate_body: str,
    escape: bool = True,
) -> str:
    """Render a pair with all untrusted delimiters escaped; escape=False supplies the raw
    budget measurement.
    """

    def _field(value: object, none_marker: str = "") -> str:
        """Render a scalar with optional trusted null-location marker, applying the same
        escaping policy as other fields. escape=False retains the raw budget
        representation.
        """
        if value is None and none_marker:
            text = none_marker
        else:
            text = str(value or "")
        return _escape_finding_delimiters(text) if escape else text

    values = {}
    for side, finding, body in (("gold", gold, gold_body), ("candidate", candidate, candidate_body)):
        for name in ("title", "severity", "path", "start_line", "end_line", "body"):
            value = body if name == "body" else finding.get(name)
            marker = _LOCATIONLESS_MARKER if name in ("path", "start_line", "end_line") else ""
            values[f"{side}_{name}"] = _field(value, marker)
    return template.format(**values)


def render_pair_prompt(gold: dict[str, Any], candidate: dict[str, Any], *, template: str) -> str:
    """Render fenced findings after enforcing the 24 KiB raw, pre-escape budget. Escaping
    may inflate accepted content; oversize raw pairs fail without truncation or partial
    results.
    """
    gold_body = gold.get("body", "") or ""
    candidate_body = candidate.get("body", "") or ""
    # Measure raw bytes before delimiter escaping, which can inflate valid findings.
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
    """Strictly validate verdict keys/types/confidence without coercion. The caller stamps
    gold/candidate identity onto the returned placeholders.
    """
    if not isinstance(raw, dict):
        raise VerifierError("verdict must be a JSON object")
    verifier_core.validate_exact_keys(raw, {"match", "confidence", "reasoning"}, "verdict")
    match = raw["match"]
    if not isinstance(match, bool):
        raise VerifierError(
            f"verdict 'match' must be a boolean, got {_bounded_repr(match)}"
        )
    confidence = raw["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VerifierError(
            f"verdict 'confidence' must be a number in [0,1], got {_bounded_repr(confidence)}"
        )
    if not 0.0 <= confidence <= 1.0:
        raise VerifierError(
            f"verdict 'confidence' must be in [0,1], got {_bounded_repr(confidence)}"
        )
    reasoning = raw["reasoning"]
    if not isinstance(reasoning, str):
        raise VerifierError(
            f"verdict 'reasoning' must be a string, got {_bounded_repr(reasoning)}"
        )
    # Reasoning is capped at 32 KiB and rejected -- never truncated-and-accepted
    # -- so an oversized/untrusted value can never bloat a diagnostic.
    if len(reasoning.encode("utf-8")) > _REASONING_CAP_BYTES:
        raise VerifierError(
            f"verdict reasoning exceeds {_REASONING_CAP_BYTES // 1024} KiB"
        )
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
    """Enforce the raw response cap before status handling, including non-2xx bodies.
    Oversize bodies fail terminally and are never truncated into acceptance.
    """
    body = _response_bytes(response)
    if len(body) > _RESPONSE_CAP_BYTES:
        raise VerifierError(  # terminal, never truncated
            f"judge response body exceeds {_RESPONSE_CAP_BYTES // 1024} KiB"
        )
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
    error = parsed_body.get("error") if isinstance(parsed_body, dict) else None
    if isinstance(error, dict):
        raw_code = error.get("code")
        if isinstance(raw_code, int) and not isinstance(raw_code, bool):
            error_code = raw_code
        elif isinstance(raw_code, str):
            try:
                error_code = int(raw_code)
            except ValueError:
                error_code = -1
        else:
            error_code = -1
        message = _bounded_error(error.get("message") or "upstream judge error")
        if error_code == 429 or error_code >= 500:
            # OpenRouter HTTP-200 envelopes can contain retryable upstream 429/5xx errors.
            raise _Retryable(f"Judge upstream error {error_code}: {message}")
        raise VerifierError(f"Judge response error {error_code}: {message}")
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
    """Retry transport/429/5xx three times with exponential backoff; other failures are
    terminal. Bound redirects, validate each resolved target, and preserve configured
    auth headers. Exhaustion raises without partial output.
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
                    # Resolve allowlisted redirects; retain configured auth and discard server headers.
                    current_url = _resolve_redirect(current_url, location, allowlist)
                    hop += 1
                    continue
                return _parse_json_response(response, content=content)
            except _Retryable:
                if attempt < _MAX_RETRIES - 1:
                    # A retry remains: back off 2 ** attempt seconds, then the
                    # outer loop issues the next attempt.
                    await asyncio.sleep(2**attempt)
                break
    # Reaching here means the final attempt failed: this single raise is the
    # retry-exhaustion exit -- never a partial result.
    raise VerifierError("Judge request failed after retries")


async def _complete_json_via(http: _AsyncHttpClient | None, **kwargs: Any) -> dict[str, Any]:
    """Run one completion over an injected HTTP client, else a short-lived one."""
    if http is not None:
        return await _complete_json_with_http(http, **kwargs)
    async with httpx.AsyncClient() as created:
        return await _complete_json_with_http(created, **kwargs)


def _anthropic_text(body: dict[str, Any]) -> str:
    """Extract the first text block from an Anthropic Messages response body."""
    if not isinstance(body, dict):
        raise VerifierError("Judge response body was not an object")
    content = body.get("content")
    if not isinstance(content, list):
        raise VerifierError("Judge response missing content blocks")
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
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
        return await _complete_json_via(
            self.http,
            url=_ANTHROPIC_MESSAGES_URL,
            payload=payload,
            headers=headers,
            content=_anthropic_text,
            allowlist=effective,
        )


class ClaudeCliJudgeClient:
    """Run the pinned Claude CLI in noninteractive JSON print mode with one turn and no
    tools. OAuth stays in the environment, never argv; output has token and byte caps.
    Only timeouts retry, after killing the child. All other failures are terminal
    VerifierError classes without stderr or prompt disclosure. The injectable runner
    replaces subprocess creation for tests.
    """

    def __init__(self, model: str, *, runner: Any = None) -> None:
        self.model = model
        self.runner = runner

    def _default_runner(self, argv: list[str], env: dict[str, str]) -> Any:
        return asyncio.create_subprocess_exec(
            *argv,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            # DEVNULL keeps stderr private and prevents an undrained pipe from blocking stdout.
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def complete_json(
        self, *, user: str, system: str = "", max_tokens: int = 512
    ) -> dict[str, Any]:
        argv = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--max-turns",
            "1",
            # Untrusted findings reach a credentialed CLI: deny all tools and keep plan mode.
            "--permission-mode",
            "plan",
            "--allowedTools",
            "[]",
        ]
        if system:
            argv += ["--append-system-prompt", system]
        argv.append(user)
        env = dict(os.environ)
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        # CLI output tokens use an environment cap; byte/verdict caps reject rather than truncate.
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        last_error = "claude-cli judge failed (unknown)"
        for attempt in range(_MAX_RETRIES):
            proc = None
            try:
                proc = await asyncio.wait_for(
                    (self.runner or self._default_runner)(argv, env),
                    timeout=_REQUEST_TIMEOUT,
                )
                stdout = await _claude_cli_stdout(proc)
                rc = getattr(proc, "returncode", getattr(proc, "rc", 0))
                # Only timeouts retry; exit, empty/malformed output, and CLI errors are terminal.
                if rc != 0:
                    raise VerifierError(f"claude-cli judge failed (exit {rc})")
                if not stdout:
                    raise VerifierError("claude-cli judge failed (empty output)")
                try:
                    payload = json.loads(stdout)
                except (ValueError, TypeError):
                    payload = None
                if not isinstance(payload, dict):
                    raise VerifierError("claude-cli judge failed (malformed output)")
                if payload.get("is_error"):
                    raise VerifierError("claude-cli judge failed (cli reported error)")
                result = payload.get("result")
                if not isinstance(result, str) or not result.strip():
                    raise VerifierError("claude-cli judge failed (missing result)")
                parsed: dict[str, Any] = json.loads(result)
                # Strict validation via the shared parse_verdict;
                # return the validated dict (judge_pairs re-parses).
                parse_verdict(parsed)
                return parsed
            except (asyncio.TimeoutError, TimeoutError):
                # Kill timed-out children before retrying; spawn timeouts may have no child handle.
                _terminate_proc(proc)
                last_error = "claude-cli judge failed (timeout)"
            except ValueError:
                # Invalid JSON is terminal; malformed verdicts propagate their own VerifierError.
                raise VerifierError("claude-cli judge failed (invalid verdict)") from None
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
        raise VerifierError(last_error or "claude-cli judge failed (unknown)")


_CHAT_COMPLETIONS_PATH = "/chat/completions"


def resolve_base_url(base_url_env: str | None) -> str:
    """Resolve the Chat Completions base URL from the environment.

    A configured base URL is required. The resolved URL is validated against
    the effective judge-host allowlist (scheme/host/form) at the client build
    and initial-request sites before any judge call.
    """
    if not base_url_env:
        raise VerifierError("missing DAYDREAM_JUDGE_BASE_URL")
    return base_url_env


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
    """Strict OpenAI-compatible verdict client. Validate the initial URL before requests
    and share bounded redirect/allowlist policy with Anthropic.
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
        # OpenRouter models may require reasoning; exclude it from the bounded response.
        if (urllib.parse.urlsplit(self.base_url).hostname or "").lower() == "openrouter.ai":
            payload["reasoning"] = {"exclude": True}
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "verdict",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "match": {"type": "boolean"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "reasoning": {"type": "string"},
                        },
                        "required": ["match", "confidence", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        return await _complete_json_via(
            self.http,
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
_ENV_OAUTH_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"
_ENV_BASE_URL = "DAYDREAM_JUDGE_BASE_URL"
_ENV_ALLOWED_HOSTS = "DAYDREAM_JUDGE_ALLOWED_HOSTS"
_ENV_ARTIFACT_PATH = "DAYDREAM_JUDGE_ARTIFACT_PATH"
_ENV_OUT_PATH = "DAYDREAM_JUDGE_OUT_PATH"
_DEFAULT_PROVIDER = ""


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
    """Cap candidate raw bytes before JSON parsing, including whitespace inflation. Decode
    errors identify only the path.
    """
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        raise _InputFileNotFound(f"input file not found: {Path(path)}") from None
    except OSError as exc:
        # Missing and unreadable artifacts are unscored infrastructure failures.
        raise _InputFileNotFound(f"could not read {Path(path)}: {exc}") from exc
    if len(raw) > verifier_core.MAX_ARTIFACT_BYTES:
        raise VerifierError("candidate artifact exceeds 1 MiB (raw bytes)")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise VerifierError(f"candidate artifact is not valid JSON: {Path(path)}") from None
    if not isinstance(parsed, dict):
        raise VerifierError("candidate artifact must be a JSON object")
    return parsed


def _read_gold_bytes(gold_path: Path, expected_sha256: str) -> list[Any]:
    """Read trusted gold bytes, verify the compiler digest, then require a JSON list. Gold
    is shipped read-only task data and has no candidate-size cap.
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
    """Require sibling immutable task identity, refs, gold digest, schema 1, and nonempty
    template version. Missing or malformed metadata fails the task before binding.
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


def _write_reward_artifacts(
    out_dir: str | Path,
    provider: str,
    model: str,
    request_counts: dict[str, int],
    errors: list[str],
    gold_count: int,
    *,
    verifier_error: int,
) -> verifier_core.Reward:
    """Always atomically write bounded/redacted details. Agent candidate failures also
    write reward zero; infrastructure failures write no reward.json and remain unscored.
    Callers must sanitize errors first.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "provider": provider,
        "model": model,
        "request_counts": request_counts,
        "errors": errors,
        "verdicts": [],
        "matches": [],
        "unmatched_gold": [],
        "unmatched_candidates": [],
    }
    _atomic_write(out_dir, "reward-details.json", json.dumps(details))
    reward = verifier_core.Reward(
        reward=0.0, gold_count=gold_count, verifier_error=verifier_error
    )
    if verifier_error == 0:
        _atomic_write(out_dir, "reward.json", verifier_core.reward_to_json(reward))
    return reward


def run_verifier(
    gold_path: str | Path,
    artifact_path: str | Path,
    out_dir: str | Path,
    *,
    client: Any,
    env: dict[str, Any],
) -> verifier_core.Reward:
    """Validate candidate bytes/schema and immutable task binding, then digest-check and
    validate gold before judging all pairs. Malformed candidate content/binding scores
    zero; absent/unreadable files, metadata, gold, judge, credentials, exhausted
    retries, and unexpected failures remain unscored with details only. Never emit
    partial scores, source, diffs, or unbounded/unredacted errors.
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

    def fail(error: Exception | str, *, verifier_error: int = 1, gold_count: int = 0) -> verifier_core.Reward:
        errors.insert(0, _bounded_error(str(error)))
        return _write_reward_artifacts(
            out_dir, provider, model, request_counts, errors, gold_count,
            verifier_error=verifier_error,
        )

    try:
        if client is None:
            raise VerifierError("no judge client configured (missing DAYDREAM_JUDGE_*)")
        try:
            artifact_raw = _read_artifact_bytes(artifact_path)
            candidates = verifier_core.validate_candidate_artifact(artifact_raw)
        except VerifierError as exc:
            # Missing/unreadable files are infrastructure; malformed agent output scores zero.
            return fail(exc, verifier_error=int(isinstance(exc, _InputFileNotFound)))

        metadata = _load_verifier_metadata(Path(gold_path))

        try:
            for field in ("case_id", "base_ref", "head_ref"):
                if artifact_raw[field] != metadata[field]:
                    raise VerifierError(
                        f"candidate {field} does not match the bound task"
                    )
        except VerifierError as exc:
            # Binding zone: a candidate pointing at the wrong task is still the
            # agent's own output -- scored zero, not unscored.
            return fail(exc, verifier_error=0)

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
    except Exception as exc:
        error = exc if isinstance(exc, VerifierError) else f"unexpected verifier failure: {exc}"
        return fail(error, gold_count=len(gold_parsed))


def _build_client(env: dict[str, Any]) -> Any:
    """Require an explicit supported judge provider and validate its host before any
    request. HTTP providers require API credentials and a validated URL; claude-cli
    requires its OAuth token and allowlisted api.anthropic.com host.
    """
    provider = env.get(_ENV_PROVIDER) or ""
    model = env.get(_ENV_MODEL)
    api_key = env.get(_ENV_API_KEY)
    if provider == "claude-cli":
        # OAuth CLI judging uses the allowlisted Anthropic host, checked before any trial.
        oauth_token = env.get(_ENV_OAUTH_TOKEN)
        if not model:
            raise VerifierError("missing DAYDREAM_JUDGE_MODEL")
        if not oauth_token:
            raise VerifierError(
                "missing CLAUDE_CODE_OAUTH_TOKEN: required when DAYDREAM_JUDGE_PROVIDER is claude-cli"
            )
        _validate_base_url(
            _ANTHROPIC_MESSAGES_URL, _effective_allowlist(_ANTHROPIC_MESSAGES_URL, env)
        )
        return ClaudeCliJudgeClient(model)
    if not model or not api_key:
        raise VerifierError("missing DAYDREAM_JUDGE_MODEL or DAYDREAM_JUDGE_API_KEY")
    if provider not in {"anthropic", "openai-compatible"}:
        raise VerifierError(
            f"unsupported DAYDREAM_JUDGE_PROVIDER '{provider}'; "
            "expected anthropic, openai-compatible, or claude-cli"
        )
    if provider == "anthropic":
        allowlist = _effective_allowlist(_ANTHROPIC_MESSAGES_URL, env)
        # Validate the initial host before any request, as for OpenAI-compatible judging.
        _validate_base_url(_ANTHROPIC_MESSAGES_URL, allowlist)
        return AnthropicJudgeClient(
            api_key,
            model,
            allowlist=allowlist,
        )
    base_url = resolve_base_url(env.get(_ENV_BASE_URL))
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
    """Print the complete typed reward payload and return the verifier-error exit code."""
    payload = reward.to_dict()
    print(json.dumps(payload))
    return 1 if reward.verifier_error else 0


def main() -> int:
    """Resolve compiled paths/environment, run the judge, and print the typed result.
    Provider/host rejection writes bounded infrastructure diagnostics without numeric
    reward. ARTIFACT_PATH and OUT_PATH overrides support isolated runs; defaults remain
    /logs/artifacts/review.json and /logs/verifier.
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
            _ENV_OAUTH_TOKEN,
            _ENV_BASE_URL,
            _ENV_ALLOWED_HOSTS,
        )
    }
    try:
        client = _build_client(env)
    except VerifierError as exc:
        provider = env.get(_ENV_PROVIDER) or _DEFAULT_PROVIDER
        if env.get(_ENV_MODEL) and (env.get(_ENV_API_KEY) or provider == "claude-cli"):
            # Provider/host/OAuth rejection emits bounded unscored diagnostics; CLI needs no API key.
            model = env.get(_ENV_MODEL) or ""
            reward = _write_reward_artifacts(
                out_dir, provider, model, {"requests": 0}, [_bounded_error(str(exc))], 0,
                verifier_error=1,
            )
            return _emit_reward(reward)
        # Missing MODEL/API_KEY keeps the compiled path: run_verifier emits its
        # own "no judge client configured" typed diagnostic.
        client = None
    reward = run_verifier(gold_path, artifact_path, out_dir, client=client, env=env)
    return _emit_reward(reward)


if __name__ == "__main__":
    sys.exit(main())
