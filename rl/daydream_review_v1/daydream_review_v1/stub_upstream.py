"""A canned upstream for the local smoke rollout.

The interception server forwards to the eval client's upstream, so pointing that
at this server gives a full rollout — env injection, dialect, secret auth, trace
capture, scoring — with no provider account and no model quality. The replies are
deliberately useless; the rollout is expected to score near zero. What is being
proven is the plumbing.

    python -m daydream_review_v1.stub_upstream --port 8399
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

CANNED_REPLY = "No findings."


def _chat_completion(model: str, reply: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _response(model: str, reply: str) -> dict[str, Any]:
    return {
        "id": "resp-stub",
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg-stub",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": reply, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def _anthropic_message(model: str, reply: str) -> dict[str, Any]:
    return {
        "id": "msg-stub",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": reply}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class StubUpstreamHandler(BaseHTTPRequestHandler):
    """Answer a completion POST in the shape its own dialect requires.

    One canned payload for all three would not do: each dialect parses the
    upstream response with its own strict model, so a merged blob fails
    validation and the rollout exercises an error branch while still looking
    like it completed. Routing on the upstream path is what makes the smoke run
    prove the dialect rather than the retry logic.
    """

    reply: str = CANNED_REPLY

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            model = json.loads(raw).get("model") or "stub/canned"
        except (ValueError, TypeError):
            model = "stub/canned"

        if self.path.endswith("/messages"):
            payload = _anthropic_message(model, self.reply)
        elif self.path.endswith("/responses"):
            payload = _response(model, self.reply)
        else:
            payload = _chat_completion(model, self.reply)

        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port: int = 0) -> HTTPServer:
    """Bind a stub upstream on loopback. Caller owns ``serve_forever``/``shutdown``."""
    return HTTPServer(("127.0.0.1", port), StubUpstreamHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8399)
    args = parser.parse_args(argv)
    server = serve(args.port)
    print(f"stub upstream on http://127.0.0.1:{server.server_port}/v1")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
