import sys
from daydream.backends import codex
from daydream.backends._transport import process_exit_message
P=sys.argv[1]
for v in ("v0_139_0","v0_153_4","v0_163_0","v0_169_0"):
    raw=[l.strip() for l in open(f"{P}/out2_{v}/mcp.stderr") if "PATH aliases" not in l]
    buf=[]; frames=False
    for l in raw:
        frames = frames or l.casefold() == "stack backtrace:"
        if frames or not l: continue
        if len(buf)>=codex._NON_JSON_EXCERPT_MAX_LINES: buf.pop(0)
        buf.append(codex._bounded_process_excerpt(l))
    msg=process_exit_message(display="Codex", returncode=1, lines=buf)
    print(v, "lines kept", len(buf), "cause kept:", "required MCP servers failed to initialize: daydream_probe" in msg, "len", len(msg))
