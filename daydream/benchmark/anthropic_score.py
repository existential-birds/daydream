"""Direct Anthropic JSON client for benchmark judge steps."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from daydream.benchmark.score import BenchmarkArtifactError, BenchmarkStepError, model_results_dir

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_REQUEST_TIMEOUT = 60.0
_MAX_RETRIES = 3
_MAX_EXTRACTION_TOKENS = 4096
_TOOL = "daydream"

_EXTRACTION_SYSTEM = "You extract code review issues from comments. Always respond with valid JSON."

_EXTRACTION_PROMPT = """You are analyzing an AI code review comment to extract individual issues mentioned.

The comment may discuss multiple distinct problems. Extract each separate issue as a standalone item.

Code Review Comment:
{comment}

Instructions:
- Extract each distinct code issue, bug, or concern mentioned
- Each issue should be a single, specific problem (not a general observation)
- Ignore meta-commentary like "I found 2 issues" - extract the actual issues
- Ignore sign-offs, greetings, or formatting instructions
- If the comment contains no actionable code review issues, return an empty list

Example input:
"Found several problems: 1) The getUserById function doesn't handle null input, which will cause a crash.
2) The cache key uses user.name but should use user.id for uniqueness.
Also, consider adding retry logic for the API call."

Example output:
{{"issues": [
  "getUserById function doesn't handle null input, causing potential crash",
  "Cache key uses user.name instead of user.id, breaking uniqueness",
  "Missing retry logic for API call"
]}}

Respond with ONLY a JSON object:
{{"issues": ["issue 1", "issue 2", ...]}}"""


class _AsyncHttpClient(Protocol):
    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        ...


class _AnthropicJsonCompleter(Protocol):
    async def complete_json(self, *, system: str, user: str, max_tokens: int) -> dict[str, Any]:
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


async def run_anthropic_extraction(
    benchmark_repo: Path,
    judge_model: str,
    *,
    tool: str = _TOOL,
    client: _AnthropicJsonCompleter,
) -> None:
    """Extract Martian-compatible candidate issues using direct Anthropic JSON calls."""
    benchmark_data_file = benchmark_repo / "results" / "benchmark_data.json"
    if not benchmark_data_file.exists():
        raise BenchmarkArtifactError(f"{benchmark_data_file} not found; cannot extract benchmark candidates.")

    data = json.loads(benchmark_data_file.read_text())
    if not isinstance(data, dict):
        raise BenchmarkStepError(f"{benchmark_data_file} must contain a JSON object.")

    results_dir = model_results_dir(benchmark_repo, judge_model)
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates_file = results_dir / "candidates.json"
    if candidates_file.exists():
        all_candidates = json.loads(candidates_file.read_text())
        if not isinstance(all_candidates, dict):
            raise BenchmarkStepError(f"{candidates_file} must contain a JSON object.")
    else:
        all_candidates = {}

    for golden_url, entry in data.items():
        reviews = entry.get("reviews", []) if isinstance(entry, dict) else []
        for review in reviews:
            if not isinstance(review, dict) or review.get("tool") != tool:
                continue
            existing_tools = all_candidates.get(golden_url)
            if isinstance(existing_tools, dict) and tool in existing_tools:
                continue

            all_text = _get_all_comment_text(review.get("review_comments", []))
            if not all_text or len(all_text.strip()) < 20:
                continue

            response = await client.complete_json(
                system=_EXTRACTION_SYSTEM,
                user=_EXTRACTION_PROMPT.format(comment=all_text),
                max_tokens=_MAX_EXTRACTION_TOKENS,
            )
            issues = _extract_issues(response)
            all_candidates.setdefault(golden_url, {})[tool] = [
                {"text": issue, "path": None, "line": None, "source": "extracted"} for issue in issues
            ]

    candidates_file.write_text(json.dumps(all_candidates, indent=2))


def _get_all_comment_text(review_comments: Any) -> str:
    if not isinstance(review_comments, list):
        return ""
    bodies = [comment["body"] for comment in review_comments if isinstance(comment, dict) and comment.get("body")]
    return "\n\n---\n\n".join(bodies)


def _extract_issues(response: dict[str, Any]) -> list[str]:
    if "issues" not in response:
        raise BenchmarkStepError("Anthropic extraction response missing required 'issues' key.")
    issues = response["issues"]
    if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
        raise BenchmarkStepError("Anthropic extraction response 'issues' must be a list of strings.")
    return issues


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
