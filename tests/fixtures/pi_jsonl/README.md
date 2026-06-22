# Pi JSONL fixtures

Replay fixtures for `PiBackend`. Each file is a scripted `pi --mode json` stdout
stream (one JSON object per LF-delimited line) consumed by
`tests/harness/pi_replay.make_mock_process_from_fixture`.

Event vocabulary mirrors `pi-mono` `AgentSessionEvent` (see
`docs/plans/2026-06-21-pi-backend.md` §4):

- Line 0 — session header (`{"type":"session","id":"..."}`).
- `agent_start` / `agent_end` — agent lifecycle.
- `turn_start` / `turn_end` — turn lifecycle; `turn_end.message` carries the
  full `AssistantMessage` with `usage` (tokens + `cost.total` in USD) and
  `stopReason`.
- `message_start` / `message_end` — message lifecycle; `message_end.message`
  carries the full content blocks (text / thinking / toolCall).
- `tool_execution_start` / `tool_execution_end` — tool execution; the `end`
  event carries `result.content[].text` and `isError`.

Files:
- `simple_text.jsonl` — single text turn with usage.
- `tool_use.jsonl` — thinking + one `read` tool call/result + text.
- `structured_output.jsonl` — assistant text is JSON (parsed when
  `output_schema` is supplied).
- `multi_turn.jsonl` — two text turns (exercises per-turn `TurnEndEvent` +
  aggregate `CostEvent`).
- `error_turn.jsonl` — `turn_end.message.stopReason == "error"` → `PiError`.
