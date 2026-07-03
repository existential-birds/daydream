import pytest

from daydream.benchmark.anthropic_score import AnthropicJsonClient


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
