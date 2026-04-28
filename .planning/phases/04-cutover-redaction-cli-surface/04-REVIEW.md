---
phase: 04-cutover-redaction-cli-surface
reviewed: 2026-04-28T00:00:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - daydream/agent.py
  - daydream/backends/codex.py
  - daydream/cli.py
  - daydream/deep/orchestrator.py
  - daydream/exploration_runner.py
  - daydream/phases.py
  - daydream/runner.py
  - daydream/trajectory.py
  - daydream/ui.py
  - tests/test_cutover_ast.py
  - tests/test_redaction.py
  - tests/test_trajectory.py
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 04: Code Review Report

**Reviewed:** 2026-04-28T00:00:00Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

The Phase 04 cutover successfully removes `_log_debug` machinery — the AST cutover guard test (`test_cutover_ast.py`) is comprehensive and the grep sweep confirms no stragglers in `daydream/` source. The redactor surface coverage in `redact_step()` is structurally sound, the `--trajectory` CLI flag wires through `RunConfig.trajectory_path` to all four run flows (normal/PR/TTT/deep) consistently, and the D-06 explicit-path fail-loud branch is correctly placed inside `__aexit__` with `raise SystemExit(2) from exc`.

However, there is one BLOCKER: `Redactor._redact_arguments` has an inverted conditional that produces malformed `ToolCall.arguments` whenever a non-string nested value contains a secret. This is uncovered by the existing tests (which only exercise string-valued tool arguments), and ToolCall.arguments accepting `dict[str, Any]` from the Pydantic side means the bug is silent in production — wrong types land in trajectories without any validation error.

A second-tier concern is the partial-trajectory writer: `write_partial()` only sees steps that have already been flushed via `Invocation.finish()` to `recorder.steps`. Steps accumulated inside an in-flight `Invocation` (the most likely state when SIGINT fires mid-`run_agent()`) are silently lost from the partial. The docstring claims "writes in-flight steps" but the implementation only writes already-completed invocations.

## Critical Issues

### CR-01: `Redactor._redact_arguments` inverted conditional corrupts non-string ToolCall arguments

**File:** `daydream/trajectory.py:199`
**Issue:** The non-string branch of `_redact_arguments` has an inverted ternary. The line currently reads:

```python
out[key] = json.loads(redacted) if redacted == serialized else redacted
```

This means: when no redaction happened (`redacted == serialized`), parse the JSON back to its original Python value (correct); when redaction DID happen (`redacted != serialized`), store the raw redacted JSON string (wrong — this leaks a serialized JSON string into a position that downstream consumers expect to be a structured Python value).

Trace example: `arguments = {"edits": [{"old": "echo sk-test-12345abcdef", "new": "echo redacted"}]}`. The value `[{"old": ..., ...}]` is non-string, so the else-branch runs:
- `serialized = '[{"old": "echo sk-test-12345abcdef", "new": "echo redacted"}]'`
- `redacted = '[{"old": "echo [REDACTED_API_KEY]", "new": "echo redacted"}]'`
- `redacted == serialized` is False → `out["edits"] = redacted` (a string, not a list)

Pydantic accepts this because `ToolCall.arguments: dict[str, Any]` is permissive — the bug is silent at write time and only surfaces when downstream consumers (Harbor, replay tooling) attempt to interpret the field's type. Existing tests cover only the string-value branch (`test_redactor_applies_to_tool_call_arguments` uses `{"command": "echo sk-..."}`), so this is uncovered by the test suite.

The Claude SDK's `ToolUseBlock.input` and Codex's `mcp_tool_call.arguments` both routinely contain nested dicts/lists (e.g., the Edit tool's `old_string` / `new_string` pairs, MultiEdit's edit list, MCP tool calls with structured arg envelopes), so this branch fires in real runs.

**Fix:**

```python
out: dict[str, Any] = {}
for key, val in arguments.items():
    try:
        if isinstance(val, str):
            out[key] = self._redact_text(val)
        else:
            serialized = json.dumps(val)
            redacted = self._redact_text(serialized)
            if redacted == serialized:
                # No redaction occurred — keep the original Python value.
                out[key] = val
            else:
                # Redaction happened — try to parse the redacted JSON back.
                # Replacement tokens like [REDACTED_API_KEY] are valid inside
                # JSON strings, so json.loads should round-trip; if not, fall
                # back to the [REDACTION_FAILED] sentinel per REDA-05.
                try:
                    out[key] = json.loads(redacted)
                except (json.JSONDecodeError, ValueError):
                    out[key] = "[REDACTION_FAILED]"
    except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
        out[key] = "[REDACTION_FAILED]"
```

Add a regression test exercising a nested-list value with an embedded secret:

```python
def test_redactor_preserves_structure_when_redacting_nested_args() -> None:
    call = ToolCall(
        tool_call_id="t1",
        function_name="MultiEdit",
        arguments={"edits": [{"old": "k=sk-test-12345abcdef", "new": "ok"}]},
    )
    step = _agent_step(tool_calls=[call])
    out = Redactor().redact_step(step)
    edits = out.tool_calls[0].arguments["edits"]
    assert isinstance(edits, list)
    assert isinstance(edits[0], dict)
    assert "[REDACTED_API_KEY]" in edits[0]["old"]
```

## Warnings

### WR-01: `write_partial()` silently drops steps from in-flight Invocations

**File:** `daydream/trajectory.py:675-702`
**Issue:** The docstring says "SIGINT/SIGTERM flush path — write in-flight steps to `<path>.partial`", and the public-facing description in `cli.py:_signal_handler` says the partial captures the in-flight run. But the implementation reads `self.steps` directly:

```python
def write_partial(self) -> None:
    if not self.steps:
        return
    try:
        trajectory = self._build_trajectory()
        ...
```

`Invocation` accumulates Steps in `Invocation.steps` and only flushes them to `recorder.steps` on `Invocation.finish()` (which runs in `_InvocationCM.__aexit__`). When SIGINT arrives during `run_agent()`, the active `Invocation` has not yet finished, so its accumulated Steps (text chunks, partial tool calls, MetricsEvents seen so far) never make it into the partial trajectory. In practice the partial is the trajectory as of the last completed phase, not "in-flight".

This is also why `_close_open_step()` is never called by `write_partial()` — even the open step buffer (`_open_step_dict` with text/thinking chunks) is silently lost.

**Fix:** Track the active `Invocation` on the recorder so `write_partial()` can finalize it before writing. Concretely:

```python
# In TrajectoryRecorder dataclass:
_active_invocation: Invocation | None = None

# In _InvocationCM.__aenter__:
self._invocation = Invocation(recorder=self._recorder, phase=self._phase)
self._recorder._active_invocation = self._invocation
return self._invocation

# In _InvocationCM.__aexit__:
if self._invocation is not None:
    self._invocation.finish()
    self._recorder._active_invocation = None
    self._invocation = None

# In write_partial:
def write_partial(self) -> None:
    # Finalize any in-flight invocation so its accumulated steps land on
    # self.steps before we serialize.
    if self._active_invocation is not None:
        try:
            self._active_invocation.finish()
            self._active_invocation = None
        except Exception:  # noqa: BLE001 - partial flush must never crash shutdown
            pass
    if not self.steps:
        return
    ...
```

Note that calling `finish()` from a signal handler is technically calling Python code from a non-async context, but it's pure synchronous data movement (no awaits) — same as the existing `write_partial` body — so it's safe.

### WR-02: `_signal_handler` uses `signal.signal()` not `loop.add_signal_handler`, so ContextVar lookup is unreliable

**File:** `daydream/cli.py:28-53` and `daydream/cli.py:56-59`
**Issue:** `_install_signal_handlers` installs the handler via `signal.signal(SIGINT, _signal_handler)`. With `anyio.run()` (asyncio backend), Python signal handlers fire between bytecode instructions in the main thread. When a signal arrives outside `Context.run()` (e.g., during sync code paths between awaits, or while asyncio is sleeping in `select`), `_RECORDER_VAR.get()` returns the main thread's outer-context value, which is `None` — because the recorder ContextVar is only `.set()` inside the asyncio task's context.

In practice this means SIGINT during the gap between two `await` points may produce a partial trajectory; SIGINT during a long `select`/syscall may not. The behavior is non-deterministic.

The standard fix is to use `loop.add_signal_handler()` (which schedules the handler inside the event loop's context) or to thread the recorder through a module-level reference that `_signal_handler` reads directly. Note that anyio offers `anyio.open_signal_receiver()` which integrates with the task group and would catch both signals reliably — but converting requires restructuring `cli.main()`.

**Fix:** The minimum-viable fix is to fall back to a module-level reference when the ContextVar lookup fails:

```python
# In trajectory.py:
_LAST_ACTIVE_RECORDER: TrajectoryRecorder | None = None

class TrajectoryRecorder:
    async def __aenter__(self) -> "TrajectoryRecorder":
        global _LAST_ACTIVE_RECORDER
        self._previous_token = _RECORDER_VAR.set(self)
        _LAST_ACTIVE_RECORDER = self
        return self

    async def __aexit__(self, ...):
        global _LAST_ACTIVE_RECORDER
        try:
            ...
        finally:
            if self._previous_token is not None:
                _RECORDER_VAR.reset(self._previous_token)
                self._previous_token = None
            if _LAST_ACTIVE_RECORDER is self:
                _LAST_ACTIVE_RECORDER = None

def get_recorder_for_signal_handler() -> TrajectoryRecorder | None:
    """Signal-safe recorder accessor. Falls back to module-level ref when
    the ContextVar is empty (signal fired outside the task's context)."""
    return _RECORDER_VAR.get() or _LAST_ACTIVE_RECORDER
```

Then update `_signal_handler` to call `get_recorder_for_signal_handler()` instead of `get_current_recorder()`.

If you choose not to fix this, update the docstring on `_signal_handler` to drop the misleading "ContextVar.get() works synchronously from any thread/context" claim.

### WR-03: Env-var redaction over-redacts identifiers like `MONKEY_PATCH=...` or `KEYBOARD_LAYOUT=...`

**File:** `daydream/trajectory.py:69-71`
**Issue:** `_ENV_VAR_PATTERN` is `r"\b([A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|...)[A-Z0-9_]*)\s*=\s*([^\s\n\r;]+)"`. The keyword list is checked as a substring, not a delimited word — so any identifier whose name happens to contain `KEY` (e.g., `MONKEY_PATCH`, `KEYBOARD_LAYOUT`, `LIST_OF_KEYS`, `TURNKEY_DEPLOY`) gets value-redacted. Same for `AUTHOR=...` matching `AUTH`, `TOKENIZED=...` matching `TOKEN`, and `CREDENTIALS_DIR=...` matching `CREDENTIALS`.

This produces over-redaction in real-world output where shell `env` dumps or commit-message bodies contain identifiers with these substrings. Trajectory readability suffers — the redacted artifact is harder to debug.

The negative test (`test_redactor_preserves_non_secret_env_vars`) uses `DEBUG=true` and `APP_NAME=myproject`, neither of which contains the keyword substrings, so it doesn't catch this.

**Fix:** Anchor the keyword match by requiring it to be at the END of the identifier (most secret env-vars do follow `*_KEY`, `*_TOKEN`, etc.) or wrap each keyword with a separator alternation. Concretely:

```python
_ENV_VAR_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:_KEY|_SECRET|_TOKEN|_PASSWORD|_PASSWD|"
    r"_CREDENTIAL|_CREDENTIALS|_API_?KEY|_AUTH)|"
    r"(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|API_?KEY|AUTH))"
    r"\s*=\s*([^\s\n\r;]+)"
)
```

Then add tests for the false positives:

```python
def test_redactor_preserves_keyword_substring_env_vars() -> None:
    out = Redactor().redact_step(_user_step(
        "MONKEY_PATCH=foo\nKEYBOARD_LAYOUT=qwerty\nAUTHOR=alice"
    ))
    assert "MONKEY_PATCH=foo" in out.message
    assert "KEYBOARD_LAYOUT=qwerty" in out.message
    assert "AUTHOR=alice" in out.message
```

### WR-04: `Redactor.redact_step` top-level fallback wipes the message but leaves other fields raw

**File:** `daydream/trajectory.py:252-254`
**Issue:** The top-level `except` in `redact_step` reads:

```python
except Exception as exc:  # noqa: BLE001 - REDA-05 redact-or-omit (top-level fallback)
    print_warning(_console, f"Redactor failure: {type(exc).__name__}: {exc}")
    return step.model_copy(update={"message": "[REDACTION_FAILED]"})
```

If the failure happens after redaction has partially mutated `updates` (e.g., `message` and `reasoning_content` are already in `updates` but `_redact_arguments` raises), the top-level handler discards `updates` entirely and returns a copy that only redacts `message`. This means `reasoning_content`, `tool_calls`, and `observation` pass through with their ORIGINAL (unredacted) values — exactly the leakage REDA-05 is designed to prevent.

**Fix:** Track the work-in-progress redacted fields and degrade only those that haven't been processed yet, OR redact every text-bearing field to `[REDACTION_FAILED]` on top-level failure:

```python
except Exception as exc:  # noqa: BLE001 - REDA-05 redact-or-omit (top-level fallback)
    print_warning(_console, f"Redactor failure: {type(exc).__name__}: {exc}")
    fallback: dict[str, Any] = {"message": "[REDACTION_FAILED]"}
    if step.reasoning_content is not None:
        fallback["reasoning_content"] = "[REDACTION_FAILED]"
    if step.tool_calls is not None:
        fallback["tool_calls"] = [
            tc.model_copy(update={
                "arguments": {k: "[REDACTION_FAILED]" for k in tc.arguments}
            })
            for tc in step.tool_calls
        ]
    if step.observation is not None:
        fallback["observation"] = step.observation.model_copy(update={
            "results": [
                r.model_copy(update={"content": "[REDACTION_FAILED]"})
                for r in step.observation.results
            ]
        })
    return step.model_copy(update=fallback)
```

## Info

### IN-01: `_LAST_ACTIVE_RECORDER` (or equivalent) duplicates per-test reset surface

**File:** `daydream/trajectory.py:287-294`
**Issue:** `_reset_recorder_for_tests()` only resets `_RECORDER_VAR`. If WR-02's fix introduces a fallback module-level reference, that fallback also needs to be cleared in this function and in the autouse `_reset_trajectory_recorder` fixture (`tests/conftest.py:125-141`). Mention as a paired-update obligation when WR-02 is fixed.

**Fix:** When implementing WR-02, also update `_reset_recorder_for_tests()`:

```python
def _reset_recorder_for_tests() -> None:
    global _LAST_ACTIVE_RECORDER
    _RECORDER_VAR.set(None)
    _LAST_ACTIVE_RECORDER = None
```

### IN-02: `_DIFF_HEADER_RE` and `_DIFF_BLOCK_SPLIT` parse the same git diff format independently

**File:** `daydream/exploration_runner.py:47` and `daydream/deep/orchestrator.py:81-86`
**Issue:** `count_changed_files()` (exploration_runner) and `_diff_changed_files()` (deep/orchestrator) both walk a unified diff to extract changed file paths. They use different regex strategies and live in different modules. Not a bug — both implementations work — but the duplication means a future change to git diff handling (e.g., supporting `diff --no-prefix`) has to be made twice and could drift.

**Fix:** Extract a shared `daydream/git_diff.py` helper module with `iter_changed_paths(diff_text: str) -> Iterator[str]` and have both call sites use it. Out of scope for Phase 04 unless explicitly desired.

### IN-03: `total_cost_usd or None` coerces zero-cost trajectories to None

**File:** `daydream/trajectory.py:649-654`
**Issue:** `_build_trajectory()` uses `self._final_totals["cost"] if self._final_totals["any_cost_seen"] else None` for cost — that's correct. But the same line above has `total_prompt_tokens=self._final_totals["prompt"] or None` etc., which falsey-coerces a literal 0 to None. A run that genuinely produced 0 prompt tokens (cached responses, sub-second tool-only turns) would have its FinalMetrics report `total_prompt_tokens: null` instead of `0`. The Harbor-side schema treats null as "unknown" rather than "zero", which slightly distorts aggregate stats.

**Fix:** Compare to None explicitly to distinguish "no metrics seen" (the initial `0` placeholder is the sentinel) from "real zero". Track `any_metrics_seen` similar to `any_cost_seen`:

```python
# In _accumulate_metrics:
if prompt_tokens is not None or completion_tokens is not None:
    self._final_totals["any_tokens_seen"] = True

# In _build_trajectory:
if self._final_totals["any_tokens_seen"]:
    total_prompt_tokens = self._final_totals["prompt"]
    total_completion_tokens = self._final_totals["completion"]
    total_cached_tokens = self._final_totals["cached"]
else:
    total_prompt_tokens = None
    total_completion_tokens = None
    total_cached_tokens = None
```

Pre-existing behavior, not Phase 04 specific — flagged because Phase 04 promotes ATIF to first-class output.

---

_Reviewed: 2026-04-28T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
