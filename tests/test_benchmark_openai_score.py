import json

import pytest

from daydream.benchmark.openai_score import (
    OpenAIJsonCompleter,
    _resolve_openai_base_url,
    run_openai_scoring,
)
from daydream.benchmark.score import BenchmarkStepError, model_results_dir
from tests.test_benchmark_anthropic_score import URL, seed_benchmark_data

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeOpenAIJson:
    def __init__(self, responses):
        self._responses = list(responses)

    async def complete_json(self, *, system, user, max_tokens):
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_openai_json_completer_posts_chat_completions_and_parses_json():
    calls = []

    class FakeClient:
        async def post(self, url, *, headers, json, timeout):
            calls.append((url, headers, json, timeout))
            return FakeResponse(200, {"choices": [{"message": {"content": '{"issues":["x"]}'}}]})

    client = OpenAIJsonCompleter(api_key="sk-openai-x", model="gpt-5.6-luna", http=FakeClient())
    result = await client.complete_json(system="extract", user="return json", max_tokens=128)
    assert result == {"issues": ["x"]}
    url, headers, payload, _ = calls[0]
    assert url == OPENAI_CHAT_COMPLETIONS_URL
    assert headers["Authorization"] == "Bearer sk-openai-x"
    assert payload["model"] == "gpt-5.6-luna"
    assert payload["temperature"] == 0
    assert payload["messages"] == [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": "return json"},
    ]


@pytest.mark.asyncio
async def test_openai_json_completer_posts_to_custom_base_url():
    """An explicit base URL (e.g. OpenRouter) wins over the OpenAI default."""
    urls = []

    class FakeClient:
        async def post(self, url, *, headers, json, timeout):
            urls.append(url)
            return FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]})

    client = OpenAIJsonCompleter(
        api_key="sk-or-v1-x", model="openai/gpt-5.6-luna", base_url="https://openrouter.ai/api/v1", http=FakeClient()
    )
    await client.complete_json(system="s", user="u", max_tokens=1)
    assert urls == [OPENROUTER_CHAT_COMPLETIONS_URL]


def test_openai_base_url_resolution():
    assert _resolve_openai_base_url("sk-proj-x", None) == "https://api.openai.com/v1"
    # An sk-or- OpenRouter key with no pin auto-routes to OpenRouter.
    assert _resolve_openai_base_url("sk-or-v1-x", None) == "https://openrouter.ai/api/v1"
    # An explicit OPENAI_BASE_URL always wins over the auto-route.
    assert _resolve_openai_base_url("sk-or-v1-x", "https://gateway.example/v1") == "https://gateway.example/v1"


@pytest.mark.asyncio
async def test_openai_direct_scoring_stamps_route_on_evaluation_leaf(tmp_path):
    """The shared pipeline, driven by the OpenAI completer, stamps the openai route."""
    seed_benchmark_data(tmp_path, tool="daydream", body="candidate")
    client = FakeOpenAIJson(
        [
            {"issues": ["candidate"]},
            {"reasoning": "same bug", "match": True, "confidence": 0.91},
        ]
    )

    scores = await run_openai_scoring(tmp_path, "gpt-5.6-luna", golden_urls=[URL], tool="daydream", client=client)

    assert scores.scored_pr_count == 1
    assert scores.total_tp == 1 and scores.total_fp == 0 and scores.total_fn == 0
    evals = json.loads((model_results_dir(tmp_path, "gpt-5.6-luna") / "evaluations.json").read_text())
    leaf = evals[URL]["daydream"]
    assert leaf["judge_route"] == "openai-compatible"
    assert leaf["judge_model"] == "gpt-5.6-luna"
    assert leaf["tp"] == 1 and leaf["precision"] == 1.0 and leaf["recall"] == 1.0


@pytest.mark.asyncio
async def test_openai_scoring_requires_api_key_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BenchmarkStepError, match="OPENAI_API_KEY"):
        await run_openai_scoring(tmp_path, "gpt-5.6-luna")
