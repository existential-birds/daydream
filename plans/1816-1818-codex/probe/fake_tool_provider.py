"""Fake Responses provider: dumps tool names from request 1, then calls the MCP tool named in argv[2]."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LOG = sys.argv[3]
state = {"n": 0}
def sse(ev): return f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b=b'{"models":[],"data":[]}'; self.send_response(200); self.send_header("content-length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        state["n"] += 1
        with open(LOG, "a") as f:
            f.write(json.dumps({"n": state["n"], "tools": [t.get("name") or t.get("type") for t in body.get("tools", [])],
                                "tool_objs": [t for t in body.get("tools", []) if "probe" in json.dumps(t)][:3],
                                "input_tail": body.get("input", [])[-2:]}) + "\n")
        self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
        rid = f"r{state['n']}"
        if not any(i.get("type") in ("function_call_output",) for i in body.get("input", [])) and sys.argv[2] != "none":
            name = sys.argv[2]
            ns = None
            if "/" in name: ns, name = name.split("/", 1)
            item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": name, "arguments": "{}"}
            if ns: item["namespace"] = ns
            out = [item]
        else:
            out = [{"type": "message", "role": "assistant", "id": "m"+rid, "content": [{"type": "output_text", "text": "done"}]}]
        evs = [{"type": "response.created", "response": {"id": rid}}] + [{"type": "response.output_item.done", "item": i} for i in out] + [
               {"type": "response.completed", "response": {"id": rid, "usage": {"input_tokens": 5, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 8}}}]
        for ev in evs: self.wfile.write(sse(ev)); self.wfile.flush()
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
