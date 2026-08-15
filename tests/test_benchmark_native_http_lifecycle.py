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
    BenchmarkStepError,
    JudgeFailedError,
)
from tests.test_benchmark_anthropic_score import URL, FakeResponse, seed_benchmark_data


def _extract_system(payload):
    """System prompt: Anthropic puts it top-level; OpenAI puts it in messages[0].content."""
    system = payload.get("system", "")
    if not system:
        messages = payload.get("messages", [])
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            system = messages[0].get("content", "")
    return system


class TrackingAsyncClient:
    """Drop-in httpx.AsyncClient that records construction/close and routes phase responses."""

    instances: list[TrackingAsyncClient] = []  # overridden/reset by the provider fixture before each test
    fail_with = None  # optional exception to raise on judge posts (after extraction/dedup share the client)

    def __init__(self):
        self.posts = []
        self.closed = False
        TrackingAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def post(self, url, *, headers, json, timeout):
        system = _extract_system(json)
        if TrackingAsyncClient.fail_with is not None and "code review evaluator" in system:
            raise TrackingAsyncClient.fail_with
        self.posts.append((url, json))
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
    judge_systems = [_extract_system(p[1]) for p in client.posts if "code review evaluator" in _extract_system(p[1])]
    assert len(judge_systems) == 2, "1 golden x 2 extracted candidates -> 2 concurrent judge posts on one client"
    assert client.closed is True


@pytest.mark.asyncio
async def test_default_invocation_closes_client_on_failure(provider, tmp_path):
    entry, model, _ = provider
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")
    TrackingAsyncClient.fail_with = BenchmarkStepError("boom")

    with pytest.raises(JudgeFailedError):
        await entry(tmp_path, model, golden_urls=[URL], tool="daydream")

    assert len(TrackingAsyncClient.instances) == 1
    assert TrackingAsyncClient.instances[0].closed is True, "client must close when a phase raises"


@pytest.mark.asyncio
async def test_injected_transport_is_never_entered_or_closed(provider, tmp_path):
    entry, model, completer = provider
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")
    transport = TrackingAsyncClient()  # caller-owned; instances list is cleared below
    TrackingAsyncClient.instances = []  # clear so only scoring-code constructions are counted
    client = completer(transport)

    scores = await entry(tmp_path, model, golden_urls=[URL], tool="daydream", client=client)

    assert scores.scored_pr_count == 1
    assert transport.closed is False, "caller-injected transport must NOT be closed by scoring code"
    assert len(TrackingAsyncClient.instances) == 0, "no httpx.AsyncClient may be constructed on the injected path"
    assert len(transport.posts) > 0, "injected transport must still serve the pipeline"
