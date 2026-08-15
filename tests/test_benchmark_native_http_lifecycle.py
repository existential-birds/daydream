"""Lifecycle contract tests: each native scoring invocation owns exactly one httpx.AsyncClient.

Tests are provider-parameterized over ``anthropic`` and ``openai`` so both the
``run_direct_scoring`` and ``run_openai_scoring`` entry points are covered.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

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
    """Drop-in httpx.AsyncClient that records construction/close and routes phase responses.

    All tunables are instance state configured per test by the ``provider``
    fixture — the class itself carries nothing mutable, so no stale envelope or
    failure mode can leak into a test that constructs the double directly.
    """

    def __init__(self, envelope=None):
        self.posts = []
        self.closed = False
        self.fail_with = None
        # Neutral default: never silently post in a stale provider's envelope format.
        self.envelope = envelope if envelope is not None else self._neutral_envelope
        self._in_flight = 0
        self.max_in_flight = 0  # peak concurrent posts; proves judge fan-out is truly parallel

    @staticmethod
    def _neutral_envelope(inner):
        raise AssertionError(
            "TrackingAsyncClient has no envelope; construct it via the provider fixture "
            "or pass an envelope explicitly."
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def post(self, url, *, headers, json, timeout):
        system = _extract_system(json)
        if self.fail_with is not None and "code review evaluator" in system:
            raise self.fail_with
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0)  # yield so genuinely-concurrent callers overlap observably
            self.posts.append((url, json))
        finally:
            self._in_flight -= 1
        if "extract code review issues" in system:
            inner = {"issues": ["Bug one", "Bug two"]}
        elif "group duplicate code review comments" in system:
            inner = {"groups": [[0, 1]]}
        else:  # judge
            inner = {"reasoning": "same bug", "match": True, "confidence": 0.91}
        return FakeResponse(200, self.envelope(inner))


@pytest.fixture(params=["anthropic", "openai"])
def provider(request, monkeypatch):
    if request.param == "anthropic":

        def envelope(inner):
            return {"content": [{"type": "text", "text": json.dumps(inner)}]}

        entry, key_env, model, completer = (
            run_direct_scoring,
            ANTHROPIC_JUDGE_API_KEY_ENV,
            "claude-opus-4-5-20251101",
            lambda t: AnthropicJsonClient(api_key="sk-test", model="claude-opus-4-5-20251101", http=t),
        )
    else:

        def envelope(inner):
            return {"choices": [{"message": {"content": json.dumps(inner)}}]}

        entry, key_env, model, completer = (
            run_openai_scoring,
            OPENAI_JUDGE_API_KEY_ENV,
            "gpt-5.6-luna",
            lambda t: OpenAIJsonCompleter(api_key="sk-test", model="gpt-5.6-luna", http=t),
        )
    # Per-test mutable state: scoring code constructs clients via the factory below,
    # which registers them and binds this test's envelope/failure mode.
    tracker = SimpleNamespace(envelope=envelope, fail_with=None, instances=[])

    def make_client():
        client = TrackingAsyncClient(envelope=tracker.envelope)
        client.fail_with = tracker.fail_with
        tracker.instances.append(client)
        return client

    monkeypatch.setenv(key_env, "sk-test")
    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return entry, model, completer, tracker


@pytest.mark.asyncio
async def test_default_invocation_reuses_and_closes_one_client(provider, tmp_path):
    entry, model, _, tracker = provider
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")

    scores = await entry(tmp_path, model, golden_urls=[URL], tool="daydream")

    assert scores.scored_pr_count == 1
    assert len(tracker.instances) == 1, "expected one shared httpx.AsyncClient per invocation"
    assert tracker.instances[0].closed is True, "shared client must be closed before return"


@pytest.mark.asyncio
async def test_one_client_serves_concurrent_judges(provider, tmp_path):
    entry, model, _, tracker = provider
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")

    await entry(tmp_path, model, golden_urls=[URL], tool="daydream")

    client = tracker.instances[0]
    assert len(tracker.instances) == 1
    judge_systems = [_extract_system(p[1]) for p in client.posts if "code review evaluator" in _extract_system(p[1])]
    assert len(judge_systems) == 2, "1 golden x 2 extracted candidates -> 2 concurrent judge posts on one client"
    assert client.max_in_flight == 2, "judge posts must overlap on the shared client, not serialize"
    assert client.closed is True


@pytest.mark.asyncio
async def test_default_invocation_closes_client_on_failure(provider, tmp_path):
    entry, model, _, tracker = provider
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")
    tracker.fail_with = BenchmarkStepError("boom")

    with pytest.raises(JudgeFailedError):
        await entry(tmp_path, model, golden_urls=[URL], tool="daydream")

    assert len(tracker.instances) == 1
    assert tracker.instances[0].closed is True, "client must close when a phase raises"


@pytest.mark.asyncio
async def test_injected_transport_is_never_entered_or_closed(provider, tmp_path):
    entry, model, completer, tracker = provider
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")
    transport = TrackingAsyncClient(envelope=tracker.envelope)  # caller-owned; not registered by the fixture factory
    client = completer(transport)

    scores = await entry(tmp_path, model, golden_urls=[URL], tool="daydream", client=client)

    assert scores.scored_pr_count == 1
    assert transport.closed is False, "caller-injected transport must NOT be closed by scoring code"
    assert len(tracker.instances) == 0, "no httpx.AsyncClient may be constructed on the injected path"
    assert len(transport.posts) > 0, "injected transport must still serve the pipeline"
