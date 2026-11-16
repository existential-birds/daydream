"""Minimal fake Responses API SSE server for probing codex exec locally (no real model)."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[2] if len(sys.argv) > 2 else None
counter = [0]

def sse(ev):
    return f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        if LOG:
            with open(LOG, "a") as f:
                f.write(("%s " % self.command) + (a[0] % a[1:]) + "\n")
    def do_GET(self):
        body = json.dumps({"models": [], "data": []}).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        counter[0] += 1
        rid = f"resp_{counter[0]}"
        if LOG:
            with open(LOG, "a") as f:
                f.write(f"POST {self.path} bytes={n}\n")
        import time; time.sleep(float(sys.argv[3]) if len(sys.argv) > 3 else 0)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        evs = [
            {"type": "response.created", "response": {"id": rid}},
            {"type": "response.output_item.done", "item": {"type": "message", "role": "assistant", "id": f"msg_{counter[0]}",
              "content": [{"type": "output_text", "text": f"pong {counter[0]}"}]}},
            {"type": "response.completed", "response": {"id": rid, "usage": {"input_tokens": 11, "input_tokens_details": {"cached_tokens": 0},
              "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 18}}},
        ]
        for ev in evs:
            self.wfile.write(sse(ev)); self.wfile.flush()

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
