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


@pytest.mark.asyncio
async def test_anthropic_judge_rejects_malformed_verdict():
    judge = AnthropicFindingJudge(FakeCompleter({"match": "yes", "confidence": 0.9, "reasoning": "x"}))

    with pytest.raises(JudgeError, match="'match' must be a boolean"):
        await judge.same_issue("golden", "candidate")


@pytest.mark.parametrize("confidence", [-1, 2, float("nan"), float("inf")])
@pytest.mark.asyncio
async def test_anthropic_judge_rejects_out_of_range_confidence(confidence):
    judge = AnthropicFindingJudge(FakeCompleter({"match": True, "confidence": confidence, "reasoning": "x"}))

    with pytest.raises(JudgeError, match="'confidence' must be between 0.0 and 1.0"):
        await judge.same_issue("golden", "candidate")


@pytest.mark.asyncio
async def test_anthropic_judge_rejects_bool_confidence():
    judge = AnthropicFindingJudge(FakeCompleter({"match": True, "confidence": True, "reasoning": "x"}))

    with pytest.raises(JudgeError, match="'confidence' must be a number"):
        await judge.same_issue("golden", "candidate")
