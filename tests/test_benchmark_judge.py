import pytest

from daydream.benchmark.judge import AnthropicFindingJudge, JudgeError, JudgeVerdict


class FakeCompleter:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def complete_json(self, *, system, user, max_tokens):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return self._response


@pytest.mark.asyncio
async def test_anthropic_judge_parses_verdict():
    client = FakeCompleter({"match": True, "confidence": 0.9, "reasoning": "same bug"})
    judge = AnthropicFindingJudge(client)

    verdict = await judge.same_issue("golden text", "candidate text")

    assert verdict == JudgeVerdict(match=True, confidence=0.9, reasoning="same bug")
    assert "golden text" in client.calls[0]["user"]
    assert "candidate text" in client.calls[0]["user"]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({"match": "yes", "confidence": 0.9, "reasoning": "x"}, "'match' must be a boolean"),
        ({"match": True, "confidence": -1, "reasoning": "x"}, "'confidence' must be between 0.0 and 1.0"),
        ({"match": True, "confidence": 2, "reasoning": "x"}, "'confidence' must be between 0.0 and 1.0"),
        ({"match": True, "confidence": float("nan"), "reasoning": "x"}, "'confidence' must be between 0.0 and 1.0"),
        ({"match": True, "confidence": float("inf"), "reasoning": "x"}, "'confidence' must be between 0.0 and 1.0"),
        ({"match": True, "confidence": True, "reasoning": "x"}, "'confidence' must be a number"),
    ],
    ids=["match-type", "negative", "above-one", "nan", "infinite", "boolean"],
)
@pytest.mark.asyncio
async def test_anthropic_judge_rejects_malformed_verdict(response, error):
    judge = AnthropicFindingJudge(FakeCompleter(response))

    with pytest.raises(JudgeError, match=error):
        await judge.same_issue("golden", "candidate")
