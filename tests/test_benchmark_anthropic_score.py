import json

import pytest

from daydream.benchmark.anthropic_score import AnthropicJsonClient, run_anthropic_extraction
from daydream.benchmark.score import model_results_dir

URL = "https://x/pull/1"


class FakeAnthropicJson:
    def __init__(self, responses):
        self._responses = list(responses)

    async def complete_json(self, *, system, user, max_tokens):
        return self._responses.pop(0)


def seed_benchmark_data(tmp_path, *, tool, body):
    results = tmp_path / "results"
    results.mkdir()
    (results / "benchmark_data.json").write_text(
        json.dumps(
            {
                URL: {
                    "reviews": [
                        {
                            "tool": tool,
                            "repo_name": "repo",
                            "pr_url": URL,
                            "review_comments": [{"body": body}],
                        }
                    ]
                }
            }
        )
    )


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_anthropic_json_client_posts_messages_and_parses_json():
    calls = []

    class FakeClient:
        async def post(self, url, *, headers, json, timeout):
            calls.append((url, headers, json, timeout))
            return FakeResponse(200, {"content": [{"type": "text", "text": '{"issues":["x"]}'}]})

    client = AnthropicJsonClient(api_key="sk-ant-x", model="claude-opus-4-5-20251101", http=FakeClient())
    result = await client.complete_json(system="extract", user="return json", max_tokens=128)
    assert result == {"issues": ["x"]}
    assert calls[0][0] == "https://api.anthropic.com/v1/messages"
    assert calls[0][1]["x-api-key"] == "sk-ant-x"
    assert calls[0][2]["model"] == "claude-opus-4-5-20251101"
    assert calls[0][2]["temperature"] == 0


@pytest.mark.asyncio
async def test_direct_extraction_writes_martian_candidates(tmp_path):
    seed_benchmark_data(tmp_path, tool="daydream", body="Bug one.\n\nBug two in cache keys.")
    client = FakeAnthropicJson([{"issues": ["Bug one", "Bug two"]}])

    scores_dir = model_results_dir(tmp_path, "claude-opus-4-5-20251101")
    await run_anthropic_extraction(tmp_path, "claude-opus-4-5-20251101", tool="daydream", client=client)

    candidates = json.loads((scores_dir / "candidates.json").read_text())
    leaf = candidates[URL]["daydream"]
    assert [c["text"] for c in leaf] == ["Bug one", "Bug two"]
    assert all(c["source"] == "extracted" and c["path"] is None and c["line"] is None for c in leaf)
