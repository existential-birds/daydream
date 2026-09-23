"""Real CLI contract with an isolated local provider and no credentials.

This verifies native schema/tool compatibility, not model convergence or
provider behavior. Skipped when the native CLI is not installed.
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, CLINotFoundError
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from daydream.backends import BackendExecutionInput, ResultEvent, ToolStartEvent
from daydream.backends.claude import ClaudeBackend


@pytest.mark.asyncio
async def test_native_claude_keeps_only_schema_serialization_tool(tmp_path: Path) -> None:
    try:
        SubprocessCLITransport(prompt="", options=ClaudeAgentOptions())._find_cli()
    except CLINotFoundError:
        pytest.skip("requires installed or SDK-bundled Claude CLI")
    offered_tools: list[list[str]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            offered_tools.append([tool["name"] for tool in payload.get("tools", [])])
            message: dict[str, Any] = {
                "id": "msg_fixture", "type": "message", "role": "assistant",
                "model": "claude-haiku-4-5-20251001", "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 1},
            }
            events: list[dict[str, Any]] = [
                {"type": "message_start", "message": message},
                {"type": "content_block_start", "index": 0, "content_block": {
                    "type": "tool_use", "id": "tool_fixture", "name": "StructuredOutput", "input": {},
                }},
                {"type": "content_block_delta", "index": 0, "delta": {
                    "type": "input_json_delta", "partial_json": '{"findings":[]}',
                }},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {
                    "stop_reason": "tool_use", "stop_sequence": None,
                }, "usage": {"output_tokens": 10}},
                {"type": "message_stop"},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in events:
                self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # No inherited provider credentials, runtime settings, or real provider endpoint.
    execution = BackendExecutionInput.from_environment({
        "PATH": os.environ["PATH"], "HOME": str(tmp_path),
        "ANTHROPIC_API_KEY": "local-fixture-not-a-secret",
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }, backend="claude")
    schema = {
        "type": "object", "properties": {"findings": {"type": "array", "items": {"type": "string"}}},
        "required": ["findings"], "additionalProperties": False,
    }
    try:
        backend = ClaudeBackend(model="claude-haiku-4-5-20251001", execution_input=execution)
        async with asyncio.timeout(30):
            events = [event async for event in backend.execute(
                tmp_path, "Serialize exactly an empty findings array.", output_schema=schema,
                max_turns=3, read_only=True, persist_session=False, finalization=True,
            )]
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=1)
    assert offered_tools == [["StructuredOutput"]]
    assert not any(isinstance(event, ToolStartEvent) for event in events)
    assert next(event for event in events if isinstance(event, ResultEvent)).structured_output == {"findings": []}
