# Plan: Pi coding-agent backend with full feature parity + ATIF trajectories

**Date:** 2026-06-21
**Branch:** `feat/pi-backend` (worktree, branched from `origin/main`)
**Status:** Ready for implementation

## 1. Goal

Add a **`PiBackend`** to daydream as a third agent backend alongside `ClaudeBackend`
and `CodexBackend`, with **complete feature parity**: it implements the full `Backend`
protocol, emits the unified `AgentEvent` stream so the existing `TrajectoryRecorder`
produces valid ATIF v1.6 trajectories indistinguishable in shape from the other two
backends, and supports the **z.ai coding plan** (GLM models via Pi's first-class z.ai
provider).

Pi is the TypeScript coding agent (`pi` CLI, `@earendil-works/pi-coding-agent`). Like
Codex it is a **subprocess + JSONL** backend — this is the proven pattern already in the
repo (`daydream/backends/codex.py`). The `PiBackend` is a second instance of that
pattern. See `.beagle/concepts/multi-backend-architecture/design.md` for the design that
approves this shape.

## 2. The Backend protocol contract (do not change — implement against this)

`daydream/backends/__init__.py` defines:

```python
class Backend(Protocol):
    model: str
    def execute(self, cwd: Path, prompt: str, output_schema: dict | None = None,
                continuation: ContinuationToken | None = None,
                agents: dict | None = None, max_turns: int | None = None,
                read_only: bool = False) -> AsyncIterator[AgentEvent]: ...
    async def cancel(self) -> None: ...
    def format_skill_invocation(self, skill_key: str, args: str = "") -> str: ...
```

The 8 event types (union `AgentEvent`), all already consumed by the recorder:
`TextEvent`, `ThinkingEvent`, `ToolStartEvent`, `ToolResultEvent`, `CostEvent`,
`MetricsEvent`, `TurnEndEvent`, `ResultEvent`. Import them from `daydream.backends`.

`TrajectoryRecorder.invocation().observe(event)` builds ATIF `Step`s from the stream:
- `TextEvent`/`ThinkingEvent` → step message / reasoning_content
- `ToolStartEvent` → registers a tool_call + opens result routing
- `ToolResultEvent` → routes to host step by `id`
- `MetricsEvent` → per-step token/cost (keyed by `message_id`; empty string is fine —
  Codex uses `""`)
- `TurnEndEvent` → **closes the current Step** (one Step per turn)
- `CostEvent` → final totals + step metrics
- `ResultEvent` → finalize (structured output + continuation)

Order matters only in that a `ToolStartEvent` must precede its `ToolResultEvent`, and
`TurnEndEvent` closes the step. Within a turn, text/thinking/tool events may arrive in
any order.

## 3. Pi CLI invocation (the subprocess to spawn)

Pi is invoked as (JSON event-stream mode):

```
pi --mode json --model <MODEL> [--provider <PROVIDER>] [--api-key <KEY>] \
   [--system-prompt <PROMPT>] [--append-system-prompt <PROMPT>] \
   [--tools <COMMA_LIST>] [--exclude-tools <COMMA_LIST>] \
   [--session-id <ID>] [--thinking <LEVEL>] \
   --no-session  "<PROMPT>"
```

- `--mode json` → emits **every** session event as one JSON object per line on stdout
  (LF-delimited). This is the stream the backend parses.
- The prompt is the **positional** argument.
- `cwd` is the process working directory (set via `cwd=` on `create_subprocess_exec`).
- Built-in tools: `read`, `find`, `ls`, `grep`, `edit`, `bash`, `write`.
  Read-only subset: `read, find, ls, grep`. Mutating: `edit, bash, write`.
- `--session-id <ID>` resumes/uses an exact session (continuation). `--no-session`
  makes the run ephemeral (no saved session) — **mutually exclusive with continuation**.
- `--thinking <off|minimal|low|medium|high|xhigh>` controls reasoning emission.
- The first stdout line is a **session header** object from
  `session.sessionManager.getHeader()`; it carries the session id. Subsequent lines are
  `AgentSessionEvent`s.

### z.ai coding plan wiring

Pi's `openai-completions` provider **auto-detects z.ai**: any model whose `provider` is
`zai`/`zai-coding-cn` or whose `baseUrl` contains `api.z.ai` or `open.bigmodel.cn` gets
z.ai compatibility handling (verified in
`reference pi-mono/packages/ai/src/providers/openai-completions.ts` `detectCompat()`).

z.ai models are configured once in `~/.pi/models.json` (Pi's model registry):

```json
{
  "models": [
    {
      "id": "glm-5.2",
      "name": "GLM-4.6 (z.ai coding plan)",
      "api": "openai-completions",
      "provider": "zai",
      "baseUrl": "https://api.z.ai/api/paas/v4",
      "apiKey": "<Z_AI_API_KEY>"
    }
  ]
}
```

Then `daydream --backend pi --model glm-5.2` works. The backend passes `--model` and
optionally `--provider`/`--api-key` (from env, see §6); `baseUrl` resolution happens in
Pi from its models registry. **Do not** fabricate a base URL or cost table in daydream.

## 4. Pi JSON event vocabulary (authoritative — read from pi-mono source)

JSON mode emits the `AgentSessionEvent` union (a superset of `AgentEvent`). The relevant
members (all verified against `pi-mono/packages/agent/src/types.ts` and
`pi-mono/packages/coding-agent/src/core/agent-session.ts`):

### Agent lifecycle
- `{"type": "agent_start"}`
- `{"type": "agent_end", "messages": [...], "willRetry": false}` — final event.

### Turn lifecycle
- `{"type": "turn_start"}`
- `{"type": "turn_end", "message": <AssistantMessage>, "toolResults": [<ToolResultMessage>...]}`

### Message lifecycle
- `{"type": "message_start", "message": <Message>}`
- `{"type": "message_update", "message": <AssistantMessage>, "assistantMessageEvent": <AssistantMessageEvent>}`
- `{"type": "message_end", "message": <Message>}`

### Tool execution lifecycle (the authoritative tool-call/result source)
- `{"type": "tool_execution_start", "toolCallId": "<id>", "toolName": "<name>", "args": {...}}`
- `{"type": "tool_execution_update", "toolCallId": "...", "toolName": "...", "args": {...}, "partialResult": {...}}`
- `{"type": "tool_execution_end", "toolCallId": "<id>", "toolName": "<name>", "result": <AgentToolResult>, "isError": false}`

### Shapes

`AssistantMessage`:
```json
{ "role": "assistant",
  "content": [ {"type":"text","text":"..."} | {"type":"thinking","thinking":"..."} | {"type":"toolCall","id":"...","name":"...","arguments":{}} ],
  "api": "openai-completions", "provider": "zai", "model": "glm-5.2",
  "usage": <Usage>, "stopReason": "stop|length|toolUse|error|aborted",
  "errorMessage": "...", "timestamp": 1719000000000 }
```

`Usage`:
```json
{ "input": 100, "output": 50, "cacheRead": 10, "cacheWrite": 5,
  "totalTokens": 165,
  "cost": { "input": 0.0001, "output": 0.0002, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0003 } }
```
`cost.total` is in **USD**. This makes Pi a **strict metrics upgrade over Codex**
(Codex has `cost_usd=None` always).

`AgentToolResult` (the `result` on `tool_execution_end`):
```json
{ "content": [ {"type":"text","text":"..."} ],
  "details": <any>,
  "terminate": false }
```

`AssistantMessageEvent` (streaming deltas on `message_update`): `start`, `text_start`,
`text_delta` (`{delta, contentIndex, partial}`), `text_end`, plus thinking/tool
variants. Full message always available at `message_end` / `turn_end`.

### Session header (first line)
Shape from `sessionManager.getHeader()` — typically includes a session id field.
**Inspect the live header to confirm the exact field name** (read
`pi-mono/packages/agent/src/session/...` or the coding-agent session manager). Extract
the session id for the `ContinuationToken`. If the header has no stable id, derive the
session id from the `--session-id` value you passed (generate a UUID for fresh runs).

## 5. Event mapping (Pi JSONL → daydream AgentEvent)

Emit per turn, in this order, so streaming UX and Step shape are both correct:

| Pi event | daydream event(s) | Notes |
|---|---|---|
| session header (line 0) | — | capture `session_id` for continuation |
| `agent_start` | — | no-op |
| `message_end` (assistant) | `TextEvent` per text block; `ThinkingEvent` per thinking block | full text available here |
| `tool_execution_start` | `ToolStartEvent(id=toolCallId, name=toolName, input=args)` | |
| `tool_execution_end` | `ToolResultEvent(id=toolCallId, output=render(result), is_error=isError)` | render = join `result.content[*].text` |
| `turn_end` | `MetricsEvent(...)` then `TurnEndEvent(message_id="")` | see below |
| `agent_end` | `CostEvent(...)` then `ResultEvent(...)` | see below |

`turn_end` → `MetricsEvent`:
```python
MetricsEvent(
    message_id="",                       # Pi has no per-message id; "" like Codex
    prompt_tokens=usage["input"],
    completion_tokens=usage["output"],
    cached_tokens=usage.get("cacheRead"),
    cost_usd=usage["cost"]["total"],     # USD — Pi reports cost (unlike Codex)
    model_name=message.get("responseModel") or message.get("model") or self.model,
)
# then
TurnEndEvent(message_id="")
```

`agent_end` → `CostEvent` (aggregate of all turns) + `ResultEvent`:
```python
CostEvent(
    cost_usd=total_cost,            # sum of per-turn cost.total, or None
    input_tokens=total_input,
    output_tokens=total_output,
    cached_tokens=total_cache_read,
    model_name=self.model,
)
ResultEvent(
    structured_output=<parsed json from last assistant text if output_schema else None>,
    continuation=ContinuationToken(backend="pi", data={"session_id": session_id}),
)
```

### Structured output (`output_schema`)
Pi has no native `--output-schema`. **Emulate** exactly as the Codex fallback already
does: append a schema instruction to the prompt, and at `agent_end` attempt
`json.loads(last_assistant_text)`. Do not invent a wire-level schema mechanism.

### `read_only`
When `read_only=True`, restrict Pi's tools to the read-only subset by passing
`--tools read,find,ls,grep` (equivalently `--exclude-tools edit,bash,write`). Document
the semantics in the `execute()` docstring's `read_only` section (Pi with bash excluded
cannot run `git commit`, so it is cleaner than Codex's read-only sandbox).

### `max_turns`
Pi has no direct turn-count flag. If `max_turns` is set, document that it is not
enforced by Pi (mirror how the design doc treats pre-existing gaps — declare it). Do NOT
silently claim support.

### `agents` (subagent map)
Mirror Codex: `raise NotImplementedError("Pi backend does not support exploration
subagents; use --backend claude for exploration.")` if non-empty. (Production fan-out is
parallel `run_agent()` calls, not `agents=`.)

### `format_skill_invocation(skill_key, args)`
Pi has **no skill registry** (Beagle skills are Claude Code plugins). Use
**path-reference injection** (design doc §5): resolve the skill directory and return an
instruction string:

```
Read `<skill_dir>/SKILL.md` and follow it as your review methodology. Read its companion
files as it directs. <args>
```

Implement a small `_resolve_skill_dir(skill_key)` helper that searches the standard
Beagle/Claude plugin locations for a directory whose skill matches `skill_key`
(namespace:slug). If unresolvable, return the raw skill name as a hint (the review
proceeds; do NOT raise — match the design doc's degradation path). Keep this helper in
`pi.py`. The full Beagle skill-path resolver is tracked separately (design doc Phase 2);
this is a best-effort implementation sufficient for parity.

## 6. Configuration & env

In `daydream/config.py` add:
- `DEFAULT_PI_MODEL = "glm-5.2"`  (z.ai coding plan default)
- A `"pi"` tier in `PHASE_DEFAULT_MODELS` mirroring the codex tier (glm-5.2 across all
  phases) — with the same phase keys (`parse`, `fix`, `test`, `verify`, `exploration`,
  `per_stack_review`, `review`, `arbiter`, `wonder`, `envision`, `merge`, `intent`,
  `pr_feedback`).

In `daydream/backends/__init__.py`:
- Add `"pi"` to `create_backend()`:
  ```python
  if name == "pi":
      from daydream.backends.pi import PiBackend
      return PiBackend(model=model or DEFAULT_PI_MODEL)
  ```
- Update the docstring + error message: `Expected 'claude', 'codex', or 'pi'.`
- Optionally export `PiBackend` in `__all__`.

`PiBackend.__init__` reads optional provider/api-key overrides from env so the z.ai
coding plan key can be supplied without editing `~/.pi/models.json`:
- `PI_PROVIDER` (e.g. `"zai"`), `PI_API_KEY`, `PI_BASE_URL`, `PI_THINKING`.
Pass `--provider`/`--api-key` only when set. `baseUrl` is NOT a Pi CLI flag — it lives
in `~/.pi/models.json`; if `PI_BASE_URL` is set and no models.json entry exists, write a
temporary models.json override (best-effort; prefer documenting the models.json setup).

In `daydream/cli.py`:
- Update the `--backend`/`-b` help text to include `"pi"`.

## 7. Files to create / modify

**Create:**
- `daydream/backends/pi.py` — `PiBackend`, `PiError`, helpers (`_render_tool_result`,
  `_resolve_skill_dir`, `_extract_usage`).
- `tests/test_backend_pi.py` — unit tests mirroring `tests/test_backend_claude.py` and
  the codex event tests: mock the subprocess, drive `execute()`, assert the exact
  `AgentEvent` sequence (text, thinking, tool start/result, metrics, turn-end, cost,
  result). Include a multi-turn case and a structured-output case.
- `tests/fixtures/pi_jsonl/` — replay fixtures (canonical pi JSONL event sequences).
- `tests/harness/pi_replay.py` — a `make_mock_process(lines)` helper modeled on
  `tests/harness/codex_replay.py` that yields JSONL lines on stdout.

**Modify:**
- `daydream/backends/__init__.py` — `create_backend("pi")`, error message, `__all__`.
- `daydream/config.py` — `DEFAULT_PI_MODEL`, `"pi"` tier in `PHASE_DEFAULT_MODELS`.
- `daydream/cli.py` — `--backend` help text.
- `tests/contract/_loaders.py` — add `pi_loader(script, *, read_only=False)` that
  synthesizes Pi JSONL from the canonical script (translate turns →
  `message_end`/`tool_execution_*`/`turn_end`/`agent_end` lines) and drives
  `PiBackend.execute` with a mocked subprocess (mirror `codex_loader`).
- `tests/contract/test_backend_step_parity.py` — add a test asserting Pi produces the
  same Step shape as Claude/Codex against the canonical script. Parameterize or add
  `test_pi_produces_identical_steps` + `..._read_only`.
- `daydream/CLAUDE.md` — document `PiBackend` in the backends section, the `pi` CLI
  prerequisite, and z.ai setup.

## 8. Test plan (mandatory — real-path per the Testing Standard)

1. **Unit events test** (`tests/test_backend_pi.py`): mock subprocess returns a scripted
   JSONL stream; assert the exact `AgentEvent` list types + payloads. Cover: single-turn
   text+thinking, one tool call/result, multi-turn, structured-output parse, error turn.
2. **Parity contract test**: `pi_loader` + canonical script → recorded Steps must match
   Claude/Codex Step shape (`message`, `reasoning_content`, `tool_calls`,
   `observation.results`). This is the proof of ATIF trajectory parity.
3. **Replay/trajectory test**: drive `runner.run` (or recorder directly) with a Pi JSONL
   fixture, assert the written trajectory.json is valid ATIF v1.6 (validate with the
   vendored `daydream.atif.validator`) and contains the expected steps/metrics.
4. **cancel()** test: start a (mocked long-running) subprocess, call `cancel()`, assert
   SIGTERM→SIGKILL lifecycle.
5. **format_skill_invocation** test: path-reference output shape; unresolvable
   degradation.
6. **create_backend("pi")** test: factory returns a `PiBackend` with the right default
   model; unknown-backend error message includes `"pi"`.

All tests must mock the subprocess (no real `pi` binary required). Add one
**opt-in live smoke test** gated on `shutil.which("pi")` (skip if absent) — mirrors the
established pattern.

## 9. Verification (do this before declaring done)

```bash
make lint       # ruff — must be clean
make typecheck  # mypy — must be clean
make test       # full pytest suite — all green, including new pi tests
```
`make check` runs all three. The full suite is ~343+ tests; collection is slow, allow
time. **Do not bypass the pre-push hook or skip tests** (see CLAUDE.md
Non-Negotiable rules). Commit with Conventional Commits (`feat(backends): add pi
coding-agent backend with ATIF trajectory parity`).

## 10. Pitfalls

- **Event ordering for the recorder:** emit `message_end` text/thinking BEFORE the
  `tool_execution_*` events for the same turn (Pi emits them in that order naturally —
  consume in arrival order). Always emit `TurnEndEvent` at `turn_end` to close the Step.
- **`message_id=""` is correct** for Pi (no per-message id). The recorder handles it
  exactly as it does for Codex. Do not synthesize ids.
- **Do not fabricate cost.** If `usage.cost` is absent on a turn, emit `cost_usd=None`
  for that turn (never synthesize from a price table). Pi/z.ai DO report cost, but guard
  for its absence.
- **`--no-session` vs continuation are mutually exclusive.** Use `--no-session` for
  fresh non-resume runs; use `--session-id <id>` for resume (continuation). Never both.
- **Credential masking:** this repo's `pyproject.toml`/config may carry API keys in env.
  If editing any credential-bearing file, use Python/sed via terminal, not the patch
  tool. (General Hermes pitfall.)
- **Stdout line buffering:** read stdout line-by-line with `asyncio` `readline()`, as
  Codex does. Set a generous `_PI_STDOUT_LIMIT_BYTES` (10 MiB like Codex).
- **agent_end always fires** — ensure `CostEvent` + `ResultEvent` are emitted exactly
  once even if the stream ended mid-turn (guard with a finally/flag).
