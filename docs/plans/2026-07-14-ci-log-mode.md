# CI Log Mode — Delegation Spec

**Date:** 2026-07-14
**Issue:** Add `--log` flag to daydream that dumps raw agent events as plain text (suitable for CI log capture) instead of the Rich neon UI.

## Goal

Add a `--log` CLI flag to daydream. When set, bypass all Rich terminal UI (panels, spinners, colorization, phase banners) and instead emit raw agent event text as plain `print()` output to stdout. Default off — existing behavior unchanged.

## Benchmark isolation

This change touches only the output rendering layer in `run_agent()`. It does NOT affect:
- The ATIF trajectory recorder (recorder still runs normally)
- Any review/fix/test phases or pipeline logic
- Backend selection or model resolution
- Findings, scoring, or benchmark artifacts

**Cannot affect Martian benchmark scoring.** The benchmark scorer reads `merged-items.json` (written by the merge phase), not terminal output. The `--log` flag changes only stdout formatting.

## Contract: What `--log` mode must do

When `log_mode=True` is on `AgentState`, `run_agent()` must:

1. **Skip ALL Rich UI construction.** No `LiveToolPanelRegistry`, no `AgentTextRenderer`, no `console.print()`, no `print_thinking()`, no `print_cost()`, no `print_warning()`, no `print_error()`, no phase hero banners, no spinners.
2. **Emit raw event text to stdout via plain `print()`:**
   - `TextEvent`: `print(event.text, flush=True)` — the raw agent text, unmodified
   - `ThinkingEvent`: `print(f"[thinking] {event.text}", flush=True)` 
   - `ToolStartEvent`: `print(f"[tool:{event.name}] {_summarize_input(event.input)}", flush=True)`
   - `ToolResultEvent`: `print(f"[tool:{event.name} result] {_summarize_output(event.output)}", flush=True)` if event.is_error, prefix with `[tool:{event.name} ERROR]`
   - `CostEvent`: `print(f"[cost] ${event.cost_usd:.4f}" if event.cost_usd else "[cost] unknown", flush=True)`
   - `MetricsEvent`: `print(f"[metrics] prompt={event.prompt_tokens} completion={event.completion_tokens}", flush=True)`
   - `ResultEvent`: if structured_output, `print(f"[result] {json.dumps(event.structured_output)[:500]}", flush=True)`; otherwise skip
   - Wall budget abort: `print(f"[aborted] {reason}", flush=True)` instead of `print_warning`
   - Retry messages: `print(f"[retry] {msg}", flush=True)` instead of `print_warning`
3. **Trajectory recorder unaffected.** The `inv.observe(event)` calls inside `run_agent()` remain unchanged — the ATIF trajectory is still fully recorded.
4. **MissingSkillError** should still raise (doesn't print — that's for the caller).
5. **Backend errors** still raise (no change).

Helper for summarizing tool input (keep it simple — just the first 100 chars as a single line):

```python
def _summarize_input(input_data: dict[str, Any]) -> str:
    """One-line summary of tool input for log output."""
    if not input_data:
        return ""
    # For known tools, pick the most informative key
    if "command" in input_data:
        return input_data["command"][:200]
    if "path" in input_data:
        return f"{input_data['path']}" + (f" -> {input_data.get('new_path', '')}" if "new_path" in input_data else "")
    # Generic: first value that's a string
    for v in input_data.values():
        if isinstance(v, str):
            return v[:200]
    return str(input_data)[:200]
```

Helper for tool output:

```python
def _summarize_output(output: str) -> str:
    """One-line summary of tool output for log output."""
    if not output:
        return "(empty)"
    # Take first non-empty line or first 200 chars
    first_line = output.strip().split("\n")[0]
    return first_line[:200]
```

## File-by-file changes

| File | Change |
|------|--------|
| `daydream/agent.py` | Add `log_mode: bool = False` to `AgentState`. Add `set_log_mode()` / `get_log_mode()` functions. Add `_summarize_input()` / `_summarize_output()` helpers. In `run_agent()`, add `log_mode` guard: when `True`, use `print()` path instead of Rich UI path for every event type. |
| `daydream/runner.py` | Add `log_mode: bool = False` to `RunConfig` dataclass. In `run()`, call `set_log_mode(config.log_mode)` alongside existing `set_quiet_mode()`/`set_non_interactive()`/`set_assume()`. Re-export `set_log_mode` from agent imports. |
| `daydream/cli.py` | Add `--log` flag to `_build_main_parser()` (boolean, default False, `dest="log_mode"` — suppressed from default help, visible under `--help-all`). Add `log_mode=args.log_mode` to `RunConfig(...)` construction in `_parse_args()`. Also add to `_build_feedback_parser()` via `_add_shared_arguments` (but `_add_shared_arguments` currently only takes `full_help` — add the flag directly in each parser builder, or add it to `_add_shared_arguments` with suppressed help unless `full_help`). |
| `tests/` | New test file `tests/test_log_mode.py` with real-path tests. |

## Test plan

All tests use the real-path pattern: enter from `runner.run()` with real fs/git/event-loop, mock only the backend via `Backend` protocol.

1. **`test_log_mode_produces_plain_text`** — run daydream with `--log` against a mock backend that emits TextEvent("hello world"), verify stdout contains "hello world" and does NOT contain ANSI escape sequences or Rich markup.
2. **`test_log_mode_dumps_tool_events`** — mock backend emits ToolStartEvent + ToolResultEvent, verify stdout contains `[tool:bash]` and `[tool:bash result]` markers.
3. **`test_log_mode_dumps_cost`** — mock backend emits CostEvent($0.0042), verify stdout contains `[cost] $0.0042`.
4. **`test_log_mode_default_off`** — run daydream without `--log`, verify Rich-styled output (ANSI escapes present, or verify that `console` was used — the mock backend path through `run_agent` still hits the rich path).
5. **`test_log_mode_with_non_interactive`** — verify `--log --non-interactive` works (the two flags are orthogonal).
6. **`test_log_mode_trajectory_still_written`** — verify that with `--log`, a trajectory file is still produced (recorder unaffected).

## Verification commands

```bash
# In the daydream worktree:
make check

# Specific test run:
uv run pytest tests/test_log_mode.py -v

# Pre-push gate (the actual hook commands):
uv run ruff check daydream tests
uv run mypy daydream tests
uv run pytest -n auto -q
```

## Pitfalls

- **Do NOT change `create_console()` or the Rich theme.** The log mode bypasses console entirely — it's a separate code path, not a console mode toggle.
- **`log_mode` and `quiet_mode` are orthogonal.** `quiet_mode` controls whether tool panels display (the `LiveToolPanelRegistry`). `log_mode` completely bypasses all Rich UI. Keep them separate.
- **The `_state` module-level singleton pattern must be followed exactly.** Add `set_log_mode()` / `get_log_mode()` mirroring the existing `set_quiet_mode()` etc. Don't use ContextVars for this — it's process-global state.
- **`_summarize_input` must handle `dict` input, not raw text.** The `ToolStartEvent.input` field is a dict (e.g. `{"command": "make test", "description": "run tests"}`), not a string.
- **Flush after each print.** CI log buffering can reorder lines — use `print(..., flush=True)` for every event.
- **Do not add `log_mode` to `_add_shared_arguments`** — that function only takes `full_help`. Add the `--log` flag directly in `_build_main_parser()` and `_build_feedback_parser()`, suppressed under default help, shown under `--help-all`. Follow the existing `--dump-artifacts` pattern.
