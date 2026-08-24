# Osprey backend

Daydream’s `osprey` backend is an additive subprocess adapter around Osprey’s
verified headless entry point:

```text
osprey agent --events-jsonl ... PROMPT
```

The adapter consumes JSONL protocol version 2 and translates the producer’s
text, thinking, tool, usage, turn, and terminal records into Daydream’s
existing backend event union. Osprey remains the only agent loop and the owner
of tool execution, Tool Search, MCP transport/lifecycle, hooks, approvals,
posture, result caps/spill, persistence, and telemetry. Daydream owns
cancellation of its child subprocesses. It does not add an MCP catalog or eager
deferred-tool schemas.

## Configuration boundary

The backend forwards only flags present in the current Osprey `agent` CLI.
Osprey configuration and provider credentials remain in the child process’s
normal environment/configuration; credentials are never placed in argv.

| Capability | Headless subprocess mapping |
| --- | --- |
| Prompt and model | Positional prompt; `--model` when a model override is supplied |
| Tool Search | Osprey’s effective `[agent].tool_search` config (`auto`, `on`, or `off`); the current CLI has no per-invocation `--tool-search` flag, so an explicit Daydream override fails closed |
| MCP | Included by Osprey’s effective filtered Tool Search catalog; no adapter-side registry |
| Read-only posture | `--read-only` |
| Headless approval | `--approval deny-untrusted`; interactive approval values fail before launch |
| Sandbox / roots | `--sandbox`, repeatable `--allowed-root` |
| Turn/stream limits | `--max-turns`, `--turn-timeout`, `--stream-idle-timeout-secs`, `--streaming-timeout-secs` |
| Result caps/spill | `--tool-result-cap`, `--tool-result-head`, `--tool-result-tail`, `--tool-result-max-lines`, `--tool-result-raw-dir` |
| Structured output | Writes the supplied JSON Schema to a temporary file and passes `--output-schema`; the file is removed after the subprocess ends |
| Persistence | Osprey sessions are persistent by default; `resume` and `fork` map to `--resume` and `--fork-from`. `persist_session=False` fails closed because this CLI has no ephemeral-session flag |
| Provider/base URL | Resolved by Osprey config/environment; the current CLI exposes no verified provider/base-url flags |

Omitting an option preserves Osprey’s own defaults. Unsupported options are
typed `OspreyUnsupportedOption` errors rather than silently dropped security
or policy settings.

## Stream and terminal semantics

The first nonblank line must be `{"event":"protocol","version":2}` followed by one
`session_start`. A successful stream ends with `session_end.outcome` equal to
`completed` or `terminal_tool`, then a Daydream `ResultEvent`. Usage is emitted
only when Osprey reports `usage_reported: true`; absent tokens, costs, cache
counts, and durations remain absent rather than becoming fabricated zeroes.

Other terminal outcomes (`failed`, `cancelled`, `max_continuations`,
`budget_expired`, and `actionless_timeout`) raise `OspreyTerminalError` with the
producer’s outcome when the subprocess exits successfully. A non-zero subprocess
exit is a backend failure even if the stream reported a terminal outcome. Missing
headers, malformed required fields, unknown event names, and truncated streams
are also explicit bounded backend failures.

## Identity and tool calls

The successful `ResultEvent` continuation retains Osprey’s provider, model,
session ID, terminal outcome, and exit code. Tool calls retain the producer’s
underlying `tool_name` and `tool_call_id`, including MCP calls resolved through
Tool Search. Daydream’s existing trajectory recorder continues to own the
top-level run identity.

Use this boundary rather than implementing another Python agent loop. Protocol
drift requires an explicit Osprey JSONL version change and a corresponding
adapter update.
