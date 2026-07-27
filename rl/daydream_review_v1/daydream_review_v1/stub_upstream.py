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


class StubUpstreamHandler(BaseHTTPRequestHandler):
    """Answer any completion POST in a shape all three dialects can parse."""

    reply: str = CANNED_REPLY

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = json.dumps(
            {
                # Chat Completions / Responses
                "id": "stub-1",
                "object": "chat.completion",
                "created": 0,
                "model": "stub/canned",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.reply},
                        "finish_reason": "stop",
                    }
                ],
                # Anthropic Messages
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": self.reply}],
                "stop_reason": "end_turn",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens": 1,
                    "output_tokens": 1,
                },
            }
        ).encode()
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
