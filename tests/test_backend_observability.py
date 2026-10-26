"""Selected native SDK model usage retains provider billing authority."""

from __future__ import annotations

from pathlib import Path

import pytest
from claude_agent_sdk.types import ResultMessage

from daydream.backends import CostEvent, ResultEvent
from daydream.backends.claude import ClaudeBackend
from tests.harness.claude_sdk import scripted_client


async def test_claude_preserves_selected_native_model_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = ResultMessage(
        subtype="success", duration_ms=10, duration_api_ms=8, is_error=False, num_turns=1,
        session_id="session", model_usage={
            "provider-model": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 20,
                               "cacheCreationInputTokens": 30, "webSearchRequests": 0, "costUSD": 0.02,
                               "contextWindow": 1000, "maxOutputTokens": 100,
                               "canonicalModel": "actual-model", "provider": "bedrock"},
        },
    )
    monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", scripted_client([terminal]))
    events = [e async for e in ClaudeBackend("requested-model").execute(Path("/tmp"), "prompt")]
    cost = next(e for e in events if isinstance(e, CostEvent))
    assert cost.model_usage is not None
    usage = cost.model_usage["provider-model"]
    assert usage.model_name == "actual-model"
    assert usage.provider_name == "bedrock"
    assert (usage.input_tokens, usage.cached_tokens, usage.cache_creation_tokens) == (60, 20, 30)
    assert usage.cost_usd == 0.02
    result = next(e for e in events if isinstance(e, ResultEvent))
    assert result.model_name == "actual-model"
    assert result.provider_name == "bedrock"
