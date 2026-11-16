import sys
from daydream.backends import codex
from daydream.backends._transport import process_exit_message
P=sys.argv[1]
for v in ("v0_139_0","v0_169_0"):
    raw=[l.strip() for l in open(f"{P}/mcp_{v}.lines")]
    buf=[]
    for l in raw:
        if len(buf)>=codex._NON_JSON_EXCERPT_MAX_LINES: buf.pop(0)
        buf.append(codex._bounded_process_excerpt(l))
    msg=process_exit_message(display="Codex", returncode=1, lines=buf)
    print("=====",v,"raw lines", len(raw), "current message contains cause:", "required MCP servers failed to initialize" in msg, "len", len(msg))
    print(msg[:300])
    err=[l for l in raw if l.startswith("Error:")][0]
    i=err.find("required MCP servers failed to initialize")
    print("Error: line len", len(err), "cause offset", i)
    print("redacted+capped:", codex._bounded_process_excerpt(err))
