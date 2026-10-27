"""Stdlib-only scripted provider shared by installed and packaged native journeys."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


@dataclass
class _NativeProvider:
    config: Path
    requests: list[dict[str, Any]] = field(default_factory=list)
    authorizations: list[str | None] = field(default_factory=list)
    turns: list[list[tuple[str, str, dict[str, Any]]]] = field(default_factory=list)
    choose: Callable[[dict[str, Any]], list[tuple[str, str, dict[str, Any]]]] | None = None
    text: str | None = None
    errors: dict[int, str] = field(default_factory=dict)
    finish_reasons: dict[int, str] = field(default_factory=dict)


@contextmanager
def native_provider_server(fixture: _NativeProvider) -> Iterator[str]:
    """Serve the scripted fixture on isolated loopback and join the worker."""
    class Provider(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fixture.requests.append(request)
            fixture.authorizations.append(self.headers.get("Authorization"))
            index = len(fixture.requests) - 1
            if index in fixture.errors:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": fixture.errors[index]}}).encode())
                return
            batch = (fixture.choose(request) if fixture.choose is not None else
                     fixture.turns[index] if index < len(fixture.turns) else [])
            delta: dict[str, Any] = {"role": "assistant", "content": fixture.text or f"Native reasoning {index}."}
            if batch:
                delta["tool_calls"] = [{"index": i, "id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}
                    for i, (call_id, name, arguments) in enumerate(batch)]
            finish_reason = fixture.finish_reasons.get(index, "tool_calls" if batch else "stop")
            chunks: list[dict[str, Any]] = [
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps({"id": f"host-output-{index}",
                    "object": "chat.completion.chunk", "created": 1, "model": request["model"], **chunk})
                    + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
