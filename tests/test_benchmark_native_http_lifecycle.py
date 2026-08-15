"""Lifecycle contract tests: each native scoring invocation owns exactly one httpx.AsyncClient.

Tests are provider-parameterized over ``anthropic`` and ``openai`` so both the
``run_direct_scoring`` and ``run_openai_scoring`` entry points are covered.
"""

from __future__ import annotations

import json

import httpx
import pytest

from daydream.benchmark.anthropic_score import AnthropicJsonClient, run_direct_scoring
from daydream.benchmark.openai_score import OpenAIJsonCompleter, run_openai_scoring
from daydream.benchmark.score import (
    ANTHROPIC_JUDGE_API_KEY_ENV,
    OPENAI_JUDGE_API_KEY_ENV,
    BenchmarkStepError,  # noqa: F401  # used in Task-4 close-on-failure test
)
from tests.test_benchmark_anthropic_score import URL, FakeResponse, seed_benchmark_data


class TrackingAsyncClient:
    """Drop-in httpx.AsyncClient that records construction/close and routes phase responses."""

    instances: list[TrackingAsyncClient] = []  # overridden/reset by the provider fixture before each test
    envelope = lambda inner: {"content": [{"type": "text", "text": json.dumps(inner)}]}  # noqa: E731
    fail_with = None  # optional exception to raise on the first post

    def __init__(self):
        self.posts = []
        self.closed = False
        TrackingAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def post(self, url, *, headers, json, timeout):
        if TrackingAsyncClient.fail_with is not None:
            raise TrackingAsyncClient.fail_with
        self.posts.append((url, json))
        # Anthropic: system is a top-level key; OpenAI: system is in messages[0].content
        system = json.get("system", "")
        if not system:
            messages = json.get("messages", [])
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                system = messages[0].get("content", "")
        if "extract code review issues" in system:
            inner = {"issues": ["Bug one", "Bug two"]}
        elif "group duplicate code review comments" in system:
            inner = {"groups": [[0, 1]]}
        else:  # judge
            inner = {"reasoning": "same bug", "match": True, "confidence": 0.91}
        return FakeResponse(200, TrackingAsyncClient.envelope(inner))


@pytest.fixture(params=["anthropic", "openai"])
def provider(request, monkeypatch):
    TrackingAsyncClient.instances = []
    TrackingAsyncClient.fail_with = None
    if request.param == "anthropic":
        TrackingAsyncClient.envelope = lambda inner: {
            "content": [{"type": "text", "text": json.dumps(inner)}]
        }
        entry, key_env, model, completer = (
            run_direct_scoring,
            ANTHROPIC_JUDGE_API_KEY_ENV,
            "claude-opus-4-5-20251101",
            lambda t: AnthropicJsonClient(api_key="sk-test", model="claude-opus-4-5-20251101", http=t),
        )
    else:
        TrackingAsyncClient.envelope = lambda inner: {
            "choices": [{"message": {"content": json.dumps(inner)}}]
        }
        entry, key_env, model, completer = (
            run_openai_scoring,
            OPENAI_JUDGE_API_KEY_ENV,
            "gpt-5.6-luna",
            lambda t: OpenAIJsonCompleter(api_key="sk-test", model="gpt-5.6-luna", http=t),
        )
    monkeypatch.setenv(key_env, "sk-test")
    monkeypatch.setattr(httpx, "AsyncClient", TrackingAsyncClient)
    return entry, model, completer


@pytest.mark.asyncio
async def test_default_invocation_reuses_and_closes_one_client(provider, tmp_path):
    entry, model, _ = provider
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")

    scores = await entry(tmp_path, model, golden_urls=[URL], tool="daydream")

    assert scores.scored_pr_count == 1
    assert len(TrackingAsyncClient.instances) == 1, "expected one shared httpx.AsyncClient per invocation"
    assert TrackingAsyncClient.instances[0].closed is True, "shared client must be closed before return"


@pytest.mark.asyncio
async def test_one_client_serves_concurrent_judges(provider, tmp_path):
    entry, model, _ = provider
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")

    await entry(tmp_path, model, golden_urls=[URL], tool="daydream")

    client = TrackingAsyncClient.instances[0]
    assert len(TrackingAsyncClient.instances) == 1
    judge_systems = [p[1].get("system", "") for p in client.posts if "code review evaluator" in p[1].get("system", "")]
    assert len(judge_systems) == 2, "1 golden x 2 extracted candidates -> 2 concurrent judge posts on one client"
    assert client.closed is True
