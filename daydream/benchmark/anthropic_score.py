"""Direct Anthropic JSON client for benchmark judge steps."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from daydream.benchmark.judge import AnthropicFindingJudge, FindingJudge, JudgeVerdict
from daydream.benchmark.score import (
    ANTHROPIC_JUDGE_API_KEY_ENV,
    BenchmarkArtifactError,
    BenchmarkStepError,
    DaydreamScores,
    model_results_dir,
    parse_daydream_scores,
)


class _NonRetryableError(BenchmarkStepError):
    """Raised for HTTP 4xx (non-429) and JSON-parse failures that must not be retried."""


_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_REQUEST_TIMEOUT = 60.0
_MAX_RETRIES = 3
_MAX_EXTRACTION_TOKENS = 4096
_MAX_DEDUP_TOKENS = 4096
_MIN_DEDUP_CANDIDATES = 2
_JUDGE_CONCURRENCY = 10  # max parallel judge calls; mirrors CapacityLimiter convention in phases.py
_TOOL = "daydream"

_EXTRACTION_SYSTEM = "You extract code review issues from comments. Always respond with valid JSON."
_DEDUP_SYSTEM = "You group duplicate code review comments. Always respond with valid JSON only."

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

_DEDUP_PROMPT = """You are identifying duplicate code review comments.

Below is a numbered list of issues extracted from an AI tool's code review.
Some tools post the same issue in both a summary comment and an inline comment,
creating near-identical duplicates. Your job is to find those duplicates.

Two candidates are duplicates ONLY IF:
- They describe the same problem AND
- A single code change would fix both (i.e., they would be one bug report)

Two candidates are NOT duplicates if:
- They describe the same TYPE of bug but in different files, functions, or
  classes (e.g., "negative slicing in OptimizedCursorPaginator" vs "negative
  slicing in BasePaginator" are separate issues - fixing one does not fix
  the other)
- They describe related but distinct problems (e.g., "returns wrong type" vs
  "caller crashes because of wrong type" are separate issues)

When in doubt, keep candidates separate - it is better to leave a duplicate
ungrouped than to incorrectly merge two distinct issues.

Candidates:
{candidates}

Return ONLY a JSON object where each group is a list of 0-based indices.
Singletons (no duplicate) must still appear as single-element groups.

Example for 4 candidates where 0 and 2 are duplicates:
{{"groups": [[0, 2], [1], [3]]}}

Your response:"""


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


def _load_json_dict(path: Path, *, required: bool, missing_hint: str = "") -> dict[str, Any]:
    if not path.exists():
        if required:
            raise BenchmarkArtifactError(f"{path} not found; {missing_hint}")
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise BenchmarkStepError(f"{path} must contain a JSON object.")
    return data


async def run_anthropic_extraction(
    benchmark_repo: Path,
    judge_model: str,
    *,
    tool: str = _TOOL,
    client: _AnthropicJsonCompleter,
) -> None:
    """Extract Martian-compatible candidate issues using direct Anthropic JSON calls."""
    benchmark_data_file = benchmark_repo / "results" / "benchmark_data.json"
    data = _load_json_dict(benchmark_data_file, required=True, missing_hint="cannot extract benchmark candidates.")

    results_dir = model_results_dir(benchmark_repo, judge_model)
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates_file = results_dir / "candidates.json"
    all_candidates = _load_json_dict(candidates_file, required=False)

    for golden_url, entry in data.items():
        reviews = entry.get("reviews", []) if isinstance(entry, dict) else []
        for review in reviews:
            if not isinstance(review, dict) or review.get("tool") != tool:
                continue

            all_text = _get_all_comment_text(review.get("review_comments", []))
            if not all_text.strip():
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


async def run_anthropic_dedup(
    benchmark_repo: Path,
    judge_model: str,
    *,
    tool: str = _TOOL,
    client: _AnthropicJsonCompleter,
) -> None:
    """Write Martian-compatible dedup groups using direct Anthropic JSON calls."""
    results_dir = model_results_dir(benchmark_repo, judge_model)
    candidates_file = results_dir / "candidates.json"
    all_candidates = _load_json_dict(
        candidates_file, required=True, missing_hint="cannot deduplicate benchmark candidates."
    )

    groups_file = results_dir / "dedup_groups.json"
    all_groups = _load_json_dict(groups_file, required=False)

    for golden_url, tools in all_candidates.items():
        if not isinstance(tools, dict):
            continue
        candidates = tools.get(tool)
        if not isinstance(candidates, list) or len(candidates) < _MIN_DEDUP_CANDIDATES:
            continue

        texts = _candidate_texts(candidates)
        if len(texts) < _MIN_DEDUP_CANDIDATES:
            continue

        try:
            response = await client.complete_json(
                system=_DEDUP_SYSTEM,
                user=_DEDUP_PROMPT.format(candidates=_numbered_candidates(texts)),
                max_tokens=_MAX_DEDUP_TOKENS,
            )
        except Exception:
            response = None
        groups = _extract_dedup_groups(response, len(texts)) if isinstance(response, dict) else None
        if groups is None:
            groups = _singleton_groups(len(texts))
        all_groups.setdefault(golden_url, {})[tool] = groups

    groups_file.write_text(json.dumps(all_groups, indent=2))


async def run_anthropic_scoring(
    benchmark_repo: Path,
    judge_model: str,
    *,
    golden_urls: Collection[str] | None = None,
    tool: str = _TOOL,
    client: _AnthropicJsonCompleter | None = None,
) -> DaydreamScores:
    """Run direct Anthropic extraction, dedup, and evaluation for benchmark scores."""
    benchmark_repo = benchmark_repo.resolve()
    if client is None:
        api_key = os.environ.get(ANTHROPIC_JUDGE_API_KEY_ENV)
        if not api_key:
            raise BenchmarkStepError(
                f"{ANTHROPIC_JUDGE_API_KEY_ENV} is not set; cannot run Anthropic-direct scoring."
            )
        client = AnthropicJsonClient(api_key=api_key, model=judge_model)

    await run_anthropic_extraction(benchmark_repo, judge_model, tool=tool, client=client)
    await run_anthropic_dedup(benchmark_repo, judge_model, tool=tool, client=client)
    evals = await run_anthropic_evaluation(
        benchmark_repo,
        judge_model,
        golden_urls=golden_urls,
        tool=tool,
        client=client,
    )
    return parse_daydream_scores(evals, tool=tool, golden_urls=golden_urls)


async def run_anthropic_evaluation(
    benchmark_repo: Path,
    judge_model: str,
    *,
    golden_urls: Collection[str] | None = None,
    tool: str = _TOOL,
    client: _AnthropicJsonCompleter,
) -> dict[str, dict[str, Any]]:
    """Write Martian-compatible evaluations using direct Anthropic JSON calls.

    ``golden_urls``, when given, restricts evaluation to those PR URLs; entries
    already present in a resumable `evaluations.json` for other PRs are left
    untouched and unscored.
    """
    benchmark_data_file = benchmark_repo / "results" / "benchmark_data.json"
    data = _load_json_dict(benchmark_data_file, required=True, missing_hint="cannot evaluate benchmark candidates.")

    results_dir = model_results_dir(benchmark_repo, judge_model)
    candidates_file = results_dir / "candidates.json"
    all_candidates = _load_json_dict(candidates_file, required=False)

    groups_file = results_dir / "dedup_groups.json"
    all_dedup_groups = _load_json_dict(groups_file, required=False)

    evaluations_file = results_dir / "evaluations.json"
    evals = _load_json_dict(evaluations_file, required=False)

    judge = AnthropicFindingJudge(client)
    selected = set(golden_urls) if golden_urls is not None else None
    for golden_url, entry in data.items():
        if selected is not None and golden_url not in selected:
            continue
        if not isinstance(entry, dict):
            continue
        golden_comments = entry.get("golden_comments", [])
        if not isinstance(golden_comments, list):
            golden_comments = []
        reviews = entry.get("reviews", [])
        if not isinstance(reviews, list):
            continue

        for review in reviews:
            if not isinstance(review, dict) or review.get("tool") != tool:
                continue
            candidates = _get_candidates_for_review(review, all_candidates, golden_url)
            dedup_groups = _get_dedup_groups(all_dedup_groups, golden_url, tool)
            result = await _evaluate_review(judge, golden_comments, candidates, dedup_groups)
            result["tool"] = tool
            result["repo_name"] = review.get("repo_name")
            result["pr_url"] = review.get("pr_url")
            result["judge_route"] = "anthropic-direct"
            result["judge_model"] = judge_model

            evals.setdefault(golden_url, {})[tool] = result
            evaluations_file.write_text(json.dumps(evals, indent=2))

    evaluations_file.write_text(json.dumps(evals, indent=2))
    return evals


def _comment_bodies(review_comments: list[Any]) -> list[str]:
    return [comment["body"] for comment in review_comments if isinstance(comment, dict) and comment.get("body")]


def _get_all_comment_text(review_comments: Any) -> str:
    if not isinstance(review_comments, list):
        return ""
    return "\n\n---\n\n".join(_comment_bodies(review_comments))


def _get_candidates_for_review(review: dict[str, Any], all_candidates: dict[str, Any], golden_url: str) -> list[str]:
    tool = review["tool"]
    tools = all_candidates.get(golden_url)
    if isinstance(tools, dict) and isinstance(tools.get(tool), list):
        return _candidate_texts(tools[tool])

    review_comments = review.get("review_comments", [])
    if not isinstance(review_comments, list):
        return []
    return _comment_bodies(review_comments)


def _get_dedup_groups(all_dedup_groups: dict[str, Any], golden_url: str, tool: str) -> list[list[int]] | None:
    tools = all_dedup_groups.get(golden_url)
    if not isinstance(tools, dict):
        return None
    groups = tools.get(tool)
    if not isinstance(groups, list):
        return None
    parsed: list[list[int]] = []
    for group in groups:
        if not isinstance(group, list) or not all(isinstance(index, int) for index in group):
            return None
        parsed.append(group)
    return parsed


async def _evaluate_review(
    judge: FindingJudge,
    golden_comments: list[Any],
    candidates: list[str],
    dedup_groups: list[list[int]] | None,
) -> dict[str, Any]:
    golden = [
        comment
        for comment in golden_comments
        if isinstance(comment, dict) and isinstance(comment.get("comment"), str)
    ]

    if not golden:
        return {"skipped": True, "reason": "No golden comments"}

    if not candidates:
        return {
            "skipped": False,
            "true_positives": [],
            "false_positives": [],
            "false_negatives": [
                {"golden_comment": comment["comment"], "severity": comment.get("severity")} for comment in golden
            ],
            "errors": [],
            "total_candidates": 0,
            "total_golden": len(golden),
            "tp": 0,
            "fp": 0,
            "fn": len(golden),
            "errors_count": 0,
            "precision": 0.0,
            "recall": 0.0,
        }

    semaphore = asyncio.Semaphore(_JUDGE_CONCURRENCY)
    tasks = []
    task_meta = []
    for golden_comment in golden:
        for candidate in candidates:
            tasks.append(_judge_limited(semaphore, judge, golden_comment["comment"], candidate))
            task_meta.append(
                {
                    "golden": golden_comment["comment"],
                    "golden_severity": golden_comment.get("severity"),
                    "candidate": candidate,
                }
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    golden_matched = {
        comment["comment"]: {
            "severity": comment.get("severity"),
            "matched": False,
            "best_confidence": -1.0,  # sentinel: any valid confidence (including 0.0) beats this
            "matched_candidate": None,
        }
        for comment in golden
    }
    candidate_matched = dict.fromkeys(candidates, False)
    sibling_map = _build_sibling_map(candidates, dedup_groups)
    errors = []

    positives: list[tuple[Any, int, str, str, JudgeVerdict]] = []

    for index, result in enumerate(results):
        meta = task_meta[index]
        golden_text = meta["golden"]
        candidate = meta["candidate"]
        if isinstance(result, BaseException):
            errors.append({"golden": golden_text, "candidate": candidate, "error": str(result)})
            continue

        if result.match:
            positives.append((-result.confidence, index, golden_text, candidate, result))

    # One-to-one assignment: each candidate (with its dedup siblings) satisfies at most one golden.
    positives.sort(key=lambda item: (item[0], item[1]))
    used_groups: set[frozenset[str]] = set()
    for negated_confidence, _index, golden_text, candidate, result in positives:
        info = golden_matched[golden_text]
        siblings = sibling_map.get(candidate, set())
        group = frozenset({candidate} | siblings)
        if info["matched"] or group in used_groups:
            continue
        used_groups.add(group)
        info["matched"] = True
        info["best_confidence"] = -negated_confidence
        info["matched_candidate"] = candidate
        info["reasoning"] = result.reasoning
        candidate_matched[candidate] = True
        for sibling in siblings:
            candidate_matched[sibling] = True

    true_positives = []
    false_negatives = []
    for golden_text, info in golden_matched.items():
        if info["matched"]:
            true_positives.append(
                {
                    "golden_comment": golden_text,
                    "severity": info["severity"],
                    "matched_candidate": info["matched_candidate"],
                    "confidence": info["best_confidence"],
                    "reasoning": info.get("reasoning"),
                }
            )
        else:
            false_negatives.append({"golden_comment": golden_text, "severity": info["severity"]})

    false_positives = [{"candidate": candidate} for candidate, matched in candidate_matched.items() if not matched]
    total_candidates = len(candidates)
    total_golden = len(golden)
    tp_count = len(true_positives)
    predicted_count = tp_count + len(false_positives)
    precision = tp_count / predicted_count if predicted_count > 0 else 0.0
    recall = tp_count / total_golden if total_golden > 0 else 0.0

    return {
        "skipped": False,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "errors": errors,
        "total_candidates": total_candidates,
        "total_golden": total_golden,
        "tp": tp_count,
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "errors_count": len(errors),
        "precision": precision,
        "recall": recall,
    }


async def _judge_limited(
    semaphore: asyncio.Semaphore, judge: FindingJudge, golden_comment: str, candidate: str
) -> JudgeVerdict:
    """Bound judge concurrency; `JudgeError` propagates to the per-pair error record."""
    async with semaphore:
        return await judge.same_issue(golden_comment, candidate)


def _build_sibling_map(candidates: list[str], groups: list[list[int]] | None) -> dict[str, set[str]]:
    if not groups:
        return {}
    sibling_map: dict[str, set[str]] = {}
    for group in groups:
        group_texts = {candidates[index] for index in group if index < len(candidates)}
        for index in group:
            if index < len(candidates):
                sibling_map[candidates[index]] = group_texts - {candidates[index]}
    return sibling_map


def _extract_issues(response: dict[str, Any]) -> list[str]:
    if "issues" not in response:
        raise BenchmarkStepError("Anthropic extraction response missing required 'issues' key.")
    issues = response["issues"]
    if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
        raise BenchmarkStepError("Anthropic extraction response 'issues' must be a list of strings.")
    return issues


def _candidate_texts(candidates: list[Any]) -> list[str]:
    return [
        candidate["text"]
        for candidate in candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("text"), str) and candidate["text"].strip()
    ]


def _numbered_candidates(texts: list[str]) -> str:
    return "\n".join(f"{index}. {text}" for index, text in enumerate(texts))


def _extract_dedup_groups(response: dict[str, Any], n_candidates: int) -> list[list[int]] | None:
    groups = response.get("groups")
    if not isinstance(groups, list):
        return None

    seen: set[int] = set()
    parsed_groups: list[list[int]] = []
    for group in groups:
        if not isinstance(group, list):
            return None
        parsed_group: list[int] = []
        for idx in group:
            if not isinstance(idx, int) or idx < 0 or idx >= n_candidates or idx in seen:
                return None
            seen.add(idx)
            parsed_group.append(idx)
        parsed_groups.append(parsed_group)

    if seen != set(range(n_candidates)):
        return None
    return parsed_groups


def _singleton_groups(n_candidates: int) -> list[list[int]]:
    return [[index] for index in range(n_candidates)]


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
        except _NonRetryableError:
            raise  # 4xx (non-429) and malformed-JSON are permanent; never retry
        except BenchmarkStepError:
            if attempt == _MAX_RETRIES - 1:
                raise
            await asyncio.sleep(2**attempt)
    raise BenchmarkStepError("Anthropic Messages request failed after retries")


def _parse_json_response(response: Any) -> dict[str, Any]:
    status_code = getattr(response, "status_code", None)
    if status_code is None or not 200 <= int(status_code) < 300:
        body = getattr(response, "text", "")
        code = int(status_code) if status_code is not None else -1
        # 429 is retryable (rate-limit); all other 4xx are permanent client errors.
        if 400 <= code < 500 and code != 429:
            raise _NonRetryableError(f"Anthropic Messages request failed with HTTP {status_code}: {body}")
        raise BenchmarkStepError(f"Anthropic Messages request failed with HTTP {status_code}: {body}")

    try:
        body = response.json()
    except Exception as exc:
        raise _NonRetryableError(f"Anthropic Messages response was not valid JSON: {exc}") from exc

    text = _first_text_block(body)
    try:
        parsed = json.loads(_strip_markdown_fences(text))
    except json.JSONDecodeError as exc:
        raise _NonRetryableError(f"Anthropic Messages text block was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _NonRetryableError("Anthropic Messages text block JSON was not an object")
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
