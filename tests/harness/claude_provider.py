"""Offline Anthropic-compatible streaming provider for the real bundled Claude CLI.

``LoopbackProvider`` records every ``/v1/messages`` request body and answers each
from the test's ``respond`` script with a tool use, text, or a non-retryable HTTP
error (retryable statuses would trigger the CLI's own backoff). ``claude_cli_env``
replaces the ambient Claude/Anthropic environment with a hermetic one pointing at
the provider; the ordinary SDK route merges ``os.environ``, so both routes see it.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class Turn:
    """One scripted model generation: a tool use or text, or an HTTP error."""

    call: tuple[str, str, dict[str, Any]] | None = None
    text: str = ""
    error_status: int | None = None


def tool(call_id: str, name: str, arguments: dict[str, Any]) -> Turn:
    return Turn(call=(call_id, name, arguments))


def _events(turn: Turn, index: int) -> bytes:
    if turn.call is not None:
        call_id, name, arguments = turn.call
        start = {"type": "tool_use", "id": call_id, "name": name, "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(arguments)}
    else:
        start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": turn.text}
    message: dict[str, Any] = {
        "id": f"msg_{index}", "type": "message", "role": "assistant", "model": MODEL, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}
    events: list[dict[str, Any]] = [
        {"type": "message_start", "message": message},
        {"type": "content_block_start", "index": 0, "content_block": start},
        {"type": "content_block_delta", "index": 0, "delta": delta},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn" if turn.call is None else "tool_use",
                                            "stop_sequence": None}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"}]
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


class LoopbackProvider:
    """A threaded loopback endpoint; scripting failures are re-raised by ``check``."""

    def __init__(self, respond: Callable[[dict[str, Any]], Turn]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.failures: list[BaseException] = []
        provider = self
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if "count_tokens" in self.path:
                    # A realistic estimate keeps the CLI's own output limits meaningful.
                    self._send(200, "application/json", json.dumps(
                        {"input_tokens": len(json.dumps(body)) // 3}).encode())
                    return
                with lock:
                    provider.requests.append(body)
                    index = len(provider.requests)
                try:
                    turn = respond(body)
                except BaseException as exc:  # noqa: BLE001 - surfaced by check()
                    provider.failures.append(exc)
                    turn = Turn(error_status=400)
                if turn.error_status is not None:
                    self._send(turn.error_status, "application/json", json.dumps({
                        "type": "error", "error": {"type": "invalid_request_error", "message": "fixture failure"},
                    }).encode())
                    return
                self._send(200, "text/event-stream", _events(turn, index))

            def _send(self, status: int, kind: str, data: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    @contextmanager
    def serving(self) -> Iterator[LoopbackProvider]:
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        try:
            yield self
        finally:
            self._server.shutdown()
            self._server.server_close()
            thread.join()

    def check(self) -> None:
        if self.failures:
            raise self.failures[0]


def claude_cli_env(monkeypatch: pytest.MonkeyPatch, provider: LoopbackProvider, home: Path) -> dict[str, str]:
    """Hermetic CLI environment: ambient Claude/Anthropic settings removed, fixture credential only."""
    for key in list(os.environ):
        if key.startswith(("ANTHROPIC", "CLAUDE")) and key != "CLAUDE_CONFIG_DIR":
            monkeypatch.delenv(key)
    home.mkdir(parents=True, exist_ok=True)
    environment = {
        "HOME": str(home), "ANTHROPIC_API_KEY": "loopback-fixture-key", "ANTHROPIC_BASE_URL": provider.url,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "IS_SANDBOX": "1",
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return dict(os.environ)
