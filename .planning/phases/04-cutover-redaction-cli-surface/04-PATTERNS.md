# Phase 4: Cutover + Redaction + CLI Surface - Pattern Map

**Mapped:** 2026-04-28
**Files analyzed:** 7 modification areas + 2 new test artifacts
**Analogs found:** 9 / 9

## File Classification

| Modified File / Area | Role | Data Flow | Closest Analog | Match Quality |
|----------------------|------|-----------|----------------|---------------|
| `daydream/trajectory.py` (`Redactor._redact_text`) | service / utility | transform | `daydream/ui.py` `_FILE_PATH_PATTERN` etc. + `daydream/agent.py:detect_test_success` | role+flow match (compiled regex applied to raw string) |
| `daydream/cli.py` (`_parse_args` removes `--debug`, adds `--trajectory`) | controller / CLI | request-response | `daydream/cli.py` existing `--ignore-path` and `--max-iterations` argparse blocks | exact (same file, same idiom) |
| `daydream/cli.py` (`_signal_handler` partial flush) | controller / signal | event-driven | `daydream/cli.py` existing `_signal_handler` + `daydream/trajectory.py` `TrajectoryRecorder.__aexit__._write` | exact + role match |
| `daydream/phases.py` legacy `_log_debug` cutover sites (`[REVERT]`, `[PARSE_FALLBACK]`, `[TTT_REVIEW]`, `[TTT_PLAN]`) | service | transform | `daydream/phases.py:786` `print_warning(console, "Agent returned empty response; ...")` | exact (same file, same call shape) |
| `daydream/runner.py` `[PHASE2_ERROR]` cutover (lines 684, 740) + debug-init block (lines 480-491) | controller / orchestration | transform | `daydream/runner.py:685` `print_error(console, "Parse Failed", ...)` + the rest of `run()` setup-block | exact |
| `daydream/exploration_runner.py` `[PRE_SCAN]` removal (lines 227, 255, 281) | service / orchestration | transform | n/a — D-08 silently removes these (best-effort path) | (deletion only — no analog needed) |
| `daydream/agent.py` `_log_debug` definition + `AgentState.debug_log` field + `set_debug_log`/`get_debug_log` | model / utility | n/a (pure delete) | the matching `set_quiet_mode`/`get_quiet_mode` getter-setter pair next door | role match (same file pattern, opposite direction) |
| `daydream/backends/codex.py` lazy import + `_raw_log` (lines 35-40, 179, 183, 257, 298, 378) | service / logging | transform | n/a — D-08 silently removes; usage sites get hard-deleted | (deletion only) |
| `daydream/ui.py` `_ui_debug` proxy (lines 28-32) + 9 call sites | utility | transform | n/a — silent removal per D-08 | (deletion only) |
| `tests/test_cutover_ast.py` (or inline AST sweep) [NEW] | test | batch / transform | n/a in-repo (no AST analog); closest is `tests/test_cli.py` argparse-rejection pattern + `daydream/tree_sitter_index.py` Path-walking idiom | no analog (treat as net-new) |
| `tests/test_redaction.py` (or expanded `tests/test_trajectory.py`) | test | request-response | `tests/test_trajectory.py:test_redactor_is_passthrough` (lines 405-416) — same `Redactor()` direct-instantiation pattern | exact (same fixture, same file already imports `Redactor`) |

## Pattern Assignments

### `daydream/trajectory.py` — `Redactor._redact_text` regex implementation

**Existing surface (Phase 2 stub):** `daydream/trajectory.py:126-144`

```python
class Redactor:
    """No-op pass-through redactor (Phase 2 stub)."""

    def redact_step(self, step: Step) -> Step:
        return step
```

**Phase 4 fills in `_redact_text(s: str) -> str` and routes `redact_step` through it.** Per D-04 the pattern is "flat regex on serialized text" applied uniformly to `Step.message`, `Step.reasoning_content`, every `tool_calls[*].arguments` value (after `json.dumps`), and every `observation.results[*].content` string.

**Compiled-regex idiom — copy from `daydream/ui.py:985-1013`:**

```python
# Module-level compiled patterns; ALL_CAPS-with-_PATTERN suffix and leading underscore.
_FILE_PATH_PATTERN = re.compile(r"(\.?/?(?:[\w.-]+/)*[\w.-]+\.\w+)")
_LINE_NUMBER_PATTERN = re.compile(r"^(\s*)(\d+)([:\-\|])")
_ERROR_KEYWORDS = re.compile(r"\b(error|Error|ERROR|failed|Failed|FAILED|exception|...)\b")
```

**Multi-pattern dispatch idiom — copy from `daydream/agent.py:236-243` (multiple compiled finditers fed into the same string):**

```python
failed_counts = [
    int(match.group(1).replace(",", ""))
    for match in re.finditer(r"(\d[\d,]*)\s+(?:tests?\s+)?fail(?:ed|ures?)\b", output_lower)
]
passed_counts = [
    int(match.group(1).replace(",", ""))
    for match in re.finditer(r"(\d[\d,]*)\s+(?:tests?\s+)?passed\b", output_lower)
]
```

**Concrete redaction targets per D-01 / D-02 / D-03:**
- `[REDACTED_API_KEY]` for `sk-…`, `ghp_…`, `xoxb-…`, `AKIA…`
- `[REDACTED_JWT]` for `eyJ…` JWTs
- `[REDACTED_USER]` for `/Users/<name>/`, `/home/<name>/`, `C:\Users\<name>\`
- `[REDACTED_ENV_VAR]` for `KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL=<value>` lines

**Apply uniformly — `redact_step` walks every text-bearing field on the Step and runs `_redact_text` on the string form.** Targets are documented on the ATIF model fields:
- `daydream/atif/models/step.py:41` (`message: str | list[ContentPart]`)
- `daydream/atif/models/step.py:48` (`reasoning_content: str | None`)
- `daydream/atif/models/tool_call.py:19` (`arguments: dict[str, Any]` — serialize then redact, or walk values)
- `daydream/atif/models/observation_result.py:21` (`content: str | list[ContentPart] | None`)

**Step is immutable Pydantic — use `.model_copy(update=...)` (already in pyproject's `pydantic>=2.11.7`).**

**Boundary-catch idiom for "redact must never crash" (REDA-05) — copy from `Invocation.observe` at `daydream/trajectory.py:237-248`:**

```python
def observe(self, event: "AgentEvent") -> None:
    try:
        self._dispatch(event)
    except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
        print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")
```

The `redact_step` exception path follows the "redact-or-omit, never raw-pass-through" contract: on regex failure, replace the offending field with `"[REDACTION_FAILED]"` rather than letting the raw value through.

---

### `daydream/cli.py` — argparse changes (`--debug` removal + `--trajectory` add)

**Analog for `--trajectory <path>`:** existing `--ignore-path` block at `daydream/cli.py:234-241`:

```python
parser.add_argument(
    "--ignore-path",
    action="append",
    default=[],
    metavar="PATH",
    dest="ignore_paths",
    help="Exclude path from diff (repeatable, e.g. --ignore-path .planning --ignore-path vendor)",
)
```

**Phase 4 `--trajectory` follows the same shape (single Path, optional, no default):**

```python
parser.add_argument(
    "--trajectory",
    default=None,
    metavar="PATH",
    type=Path,
    dest="trajectory_path",
    help="Write ATIF v1.6 trajectory JSON to this path (default: <target>/.daydream/trajectory.json)",
)
```

**Hard reject for `--debug` (D-05):** Argparse's default behavior on an unknown flag IS the hard reject (`argparse.ArgumentParser.parse_args` raises `SystemExit(2)` with `unrecognized arguments: --debug`). Action: **delete** the existing block at `daydream/cli.py:145-150`:

```python
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Save debug log",
)
```

**Also remove from `RunConfig(...)` constructor at `daydream/cli.py:362-382`:**
- Remove `debug=args.debug,` (line 366).
- Add `trajectory_path=args.trajectory_path,` to the keyword arg list (existing `RunConfig.trajectory_path` field at `daydream/runner.py:113`).

**CLI test analog — `tests/test_cli.py:42-45` for the rejection assertion:**

```python
def test_invalid_backend_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--backend", "invalid"])
    with pytest.raises(SystemExit):
        _parse_args()
```

The Phase 4 hard-reject test mirrors this: `monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--debug"])` → `pytest.raises(SystemExit)`.

---

### `daydream/cli.py` — `_signal_handler` partial flush (D-07)

**Existing handler:** `daydream/cli.py:26-39`:

```python
def _signal_handler(signum: int, frame: object) -> None:
    """Handle termination signals by requesting shutdown."""
    signal_name = signal.Signals(signum).name
    set_shutdown_requested(True)

    panel = ShutdownPanel(console)
    set_shutdown_panel(panel)
    panel.start(f"Received {signal_name}, shutting down")

    if get_current_backends():
        panel.add_step("Terminating running agent(s)...")

    raise KeyboardInterrupt
```

**Trajectory-write analog inside `TrajectoryRecorder._write` at `daydream/trajectory.py:546-553`:**

```python
def _write(self) -> None:
    # Empty trajectory: skip — Pydantic Trajectory.steps has min_length=1.
    if not self.steps:
        return
    trajectory = self._build_trajectory()
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.path.write_text(json.dumps(trajectory.to_json_dict(), indent=2), encoding="utf-8")
```

**Phase 4 partial flush** — D-07 says SIGINT writes to `<path>.partial` with `extra.partial=true`. Implementation discretion (per D-08 in CONTEXT) on whether the hook lives in `_signal_handler`, in `TrajectoryRecorder.__aexit__`, or in `runner.py`. The recommended pattern is to **add a `TrajectoryRecorder.write_partial()` method** alongside the existing `_write` (mirrors its structure exactly):

```python
def write_partial(self) -> None:
    """SIGINT-flush path: write to .partial sibling with extra.partial=true."""
    if not self.steps:
        return
    trajectory = self._build_trajectory()
    # Pydantic Trajectory has model_copy(update=...) for adding to extra
    partial_path = self.path.with_suffix(self.path.suffix + ".partial")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    json_dict = trajectory.to_json_dict()
    json_dict.setdefault("extra", {})["partial"] = True
    partial_path.write_text(json.dumps(json_dict, indent=2), encoding="utf-8")
```

**Wiring in `_signal_handler`:** since signal handlers run on the main thread, the simplest correct approach is to read `get_current_recorder()` (it's a `ContextVar` — readable from any context) and call `write_partial()` synchronously before raising `KeyboardInterrupt`. The recorder's `__aexit__` should also become idempotent so the implicit second write on the way out doesn't clobber `<path>.partial` or write a normal `<path>` file when interrupted.

**Boundary-catch around the partial write:** wrap in `try/except Exception` like `__aexit__` at `daydream/trajectory.py:417-423`:

```python
try:
    self._write()
except Exception as exc:  # noqa: BLE001 - implicit write degrade-with-warning per D-11
    print_warning(_console, f"Trajectory write failed: {type(exc).__name__}: {exc}")
```

---

### `daydream/phases.py` — legacy `_log_debug` cutover

**Promotions (D-08):**

| Site | Current | Replacement |
|------|---------|-------------|
| `phases.py:497` `_log_debug(f"[REVERT] git clean removed:\n{clean_result.stdout}")` | `_log_debug` | **silent remove** (informational, not an error) |
| `phases.py:499` `_log_debug(f"[REVERT] failed: {type(e).__name__}: {e}")` | `_log_debug` | `print_warning(console, f"Revert failed: {type(e).__name__}: {e}")` |
| `phases.py:778-781` `[PARSE_FAIL]` block | `_log_debug` | **silent remove** (next line at 786 already calls `print_warning`) |
| `phases.py:785` `[PARSE_FALLBACK] empty result` | `_log_debug` | **silent remove** (next line at 786 already calls `print_warning`) |
| `phases.py:1238` `[TTT_REVIEW] unexpected result type` | `_log_debug` | `print_warning(console, f"TTT review returned unexpected result type: {type(result).__name__}")` |
| `phases.py:1372` `[TTT_PLAN] unexpected result type` | `_log_debug` | `print_warning(console, f"TTT plan returned unexpected result type: {type(result).__name__}")` (note: line 1373 already has a `print_warning` for the user-visible failure — the promoted line adds the `type(result).__name__` detail) |
| `phases.py:1482-1485` `debug = get_debug_log(); debug.write(...)` block | `get_debug_log` | **delete entire `if debug is not None:` block** — line 1486-1490 already prints the user-visible warning |

**Local style analog — `daydream/phases.py:786`:**

```python
print_warning(console, "Agent returned empty response; treating as no actionable issues")
```

```python
# phases.py:1373 (already in-place)
print_warning(console, "Failed to generate structured plan")
```

**Imports to clean up (`daydream/phases.py:10-16`):**

```python
from daydream.agent import (
    _log_debug,        # ← remove
    console,
    detect_test_success,
    get_debug_log,     # ← remove
    run_agent,
)
```

---

### `daydream/runner.py` — `[PHASE2_ERROR]` + debug-init removal

**Promotions (D-08):**

| Site | Current | Replacement |
|------|---------|-------------|
| `runner.py:684` `_log_debug(f"[PHASE2_ERROR] {exc}\n")` | `_log_debug` | `print_error(console, "Phase 2 Error", str(exc))` (paired with the existing `print_error(console, "Parse Failed", ...)` immediately following) |
| `runner.py:740` (same pattern in single-pass branch) | `_log_debug` | `print_error(console, "Phase 2 Error", str(exc))` |

**Debug-init block removal — delete entirely (`daydream/runner.py:479-491`):**

```python
# Set up debug logging if enabled
debug_log_path: Path | None = None
if config.debug:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log_path = target_dir / f".review-debug-{timestamp}.log"

with contextlib.ExitStack() as stack:
    if debug_log_path is not None:
        debug_log_file = stack.enter_context(open(debug_log_path, "w", encoding="utf-8"))
        set_debug_log(debug_log_file)
        stack.callback(set_debug_log, None)
        stack.callback(print_info, console, f"Debug log saved: {debug_log_path}")
        print_info(console, f"Debug log: {debug_log_path}")
```

**The outer `with contextlib.ExitStack() as stack:` becomes empty — the body still needs the same indentation, but `stack` becomes unused. Two options (Claude's discretion per D-08):**
1. Delete `contextlib.ExitStack` entirely and unindent the body (cleaner).
2. Leave `contextlib.ExitStack` for any other future stack usage (less code churn).

**Imports to clean up (`daydream/runner.py:11-18`):**

```python
from daydream.agent import (
    MissingSkillError,
    _log_debug,          # ← remove
    console,
    set_debug_log,       # ← remove
    set_model,
    set_quiet_mode,
)
```

**`RunConfig.debug` field removal (`daydream/runner.py:95`):** Delete line `debug: bool = False`. Update docstring at `runner.py:70` to remove the `debug:` line. `trajectory_path: Path | None = None` already exists at line 113 — Phase 4 just wires `--trajectory` to it.

**Local style analog — same file:**

```python
# runner.py:118 — print_error idiom
print_error(console, "Missing Skill", f"Skill '{skill_name}' is not available")

# runner.py:212 — print_error with title + detail
print_error(console, "Invalid PR config", "--pr and --bot are required for PR feedback mode.")

# runner.py:476 — print_error before return 1
print_error(console, "Missing Review File", str(e))
```

---

### `daydream/exploration_runner.py` — `[PRE_SCAN]` silent removal (D-08)

All three sites at lines 227, 255, 281 are silently removed per D-08 ("exploration is best-effort"). Also clean up the lazy import:

```python
# exploration_runner.py:221 — current
from daydream.agent import _log_debug, run_agent
```
becomes:
```python
from daydream.agent import run_agent
```

**No analog needed — pure deletion.**

---

### `daydream/agent.py` — `_log_debug`, `AgentState.debug_log`, getters/setters

**Pure-deletion targets:**

| Lines | What to delete |
|-------|----------------|
| `agent.py:11` | `from typing import TYPE_CHECKING, Any, TextIO` → drop `TextIO` |
| `agent.py:64` | `debug_log: TextIO | None = None` field |
| `agent.py:60` | docstring entry `debug_log: File handle for debug logging, or None to disable.` |
| `agent.py:106-126` | `set_debug_log()` and `get_debug_log()` |
| `agent.py:208-212` | `_log_debug()` definition |
| `agent.py:332, 333, 334, 342, 353, 372, 393, 403-406, 419-424, 427-431, 437, 451, 458, 466, 493, 499, 503-506, 511, 514` | every `_log_debug(...)` call inside `run_agent()` — silent remove (D-08, all are agent-event-mirroring redundant with trajectory recording) |

**Getter/setter pair to keep as the analog template (next door at `agent.py:129-149`):**

```python
def set_quiet_mode(quiet: bool) -> None:
    """Set quiet mode for agent output."""
    _state.quiet_mode = quiet


def get_quiet_mode() -> bool:
    """Get current quiet mode setting."""
    return _state.quiet_mode
```

**`reset_state()` at `agent.py:93-103`:** No code change required — `reset_state()` calls `_state = AgentState()` which auto-picks up the simplified dataclass. Docstring stays accurate.

**Note on `[EXECUTE_ERROR]` / `[EXECUTE_INIT_ERROR]` (D-08 PROMOTION targets):** lines 353, 493 carry these. The `raise` immediately after line 354 / 494 already propagates the exception to the caller, where `cli.main()`'s `print_error(console, "Fatal Error", str(e))` fallback at `cli.py:418` handles it. **Per D-08 these are explicitly listed as promotion targets.** Replacement (matches "promote to `print_error()`" contract, but keeps the exception flow intact):

```python
# agent.py:351-354 (current)
try:
    event_iter = backend.execute(cwd, prompt, output_schema, continuation, agents=agents, max_turns=max_turns)
except Exception as exc:
    _log_debug(f"[EXECUTE_INIT_ERROR] {type(exc).__name__}: {exc}\n")
    raise

# Phase 4
try:
    event_iter = backend.execute(cwd, prompt, output_schema, continuation, agents=agents, max_turns=max_turns)
except Exception as exc:
    print_error(console, "Backend Init Error", f"{type(exc).__name__}: {exc}")
    raise
```

Same shape for the wrapping `[EXECUTE_ERROR]` at line 492-494.

---

### `daydream/backends/codex.py` — lazy import + `_raw_log` removal

**Pure-deletion targets (D-08 silent removal):**
- `backends/codex.py:35-40`: `_raw_log` definition + lazy import
- `backends/codex.py:179, 183, 257, 298, 378`: every `_raw_log(...)` call

**No analog — straight delete + verify imports remain valid.**

---

### `daydream/ui.py` — `_ui_debug` removal

**Pure-deletion targets (D-08 silent removal):**
- `ui.py:28-32`: `_ui_debug` definition + lazy import
- `ui.py:611, 636, 2653, 2754, 2768, 2864, 2995, 3005`: every `_ui_debug(...)` call

**No analog — straight delete.**

---

### CUT-08 AST Sweep — new test (no in-repo analog)

**No existing AST-walking utility in the codebase.** The closest analogs are:

1. `daydream/tree_sitter_index.py` — uses tree-sitter (NOT `ast`), but demonstrates the "walk every source file in a tree" idiom.
2. `tests/test_cli.py` — demonstrates the parametrized-style argument-rejection assertion idiom.
3. `tests/conftest.py:125-141` — autouse fixture pattern for cross-cutting test invariants.

**Recommended structure (Claude's discretion per D-08):**

```python
# tests/test_cutover_ast.py (NEW)
"""CUT-08 AST sweep: assert no orphan _log_debug references survive cutover.

Walks the AST of every .py file under daydream/ and tests/ and rejects any
Name, Attribute, or ImportFrom node referencing the forbidden symbols.
This catches the lazy-import gotcha (Pitfall 13) that grep alone misses.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_NAMES = {"_log_debug", "_raw_log", "_ui_debug", "set_debug_log", "get_debug_log"}
FORBIDDEN_ATTRS = {"debug_log"}  # AgentState.debug_log

SOURCE_DIRS = [PROJECT_ROOT / "daydream", PROJECT_ROOT / "tests"]


def _all_py_files() -> list[Path]:
    return sorted(p for d in SOURCE_DIRS for p in d.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("py_file", _all_py_files(), ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_no_legacy_debug_logging_references(py_file: Path) -> None:
    """CUT-08: every .py file is free of forbidden debug-logging symbols."""
    # Self-exclude: this file references the forbidden names in string literals.
    if py_file.resolve() == Path(__file__).resolve():
        return

    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            pytest.fail(f"{py_file}:{node.lineno}: forbidden Name '{node.id}'")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES | FORBIDDEN_ATTRS:
            pytest.fail(f"{py_file}:{node.lineno}: forbidden Attribute '.{node.attr}'")
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_NAMES:
                    pytest.fail(f"{py_file}:{node.lineno}: forbidden import '{alias.name}'")
```

**Filesystem-walk idiom — copy from `tests/conftest.py:55+` (uses `Path.rglob` shape):** standard `Path.rglob("*.py")` with `"__pycache__"` skip.

**Self-exclusion technique:** the test file itself contains the forbidden literal strings — the `if py_file.resolve() == Path(__file__).resolve(): return` line skips it. Alternative: keep the strings out of `FORBIDDEN_*` constants and use a separate manifest.

---

### Redaction unit tests — extend `tests/test_trajectory.py`

**Existing minimal-stub test as analog (`tests/test_trajectory.py:405-416`):**

```python
def test_redactor_is_passthrough() -> None:
    """Redactor.redact_step returns the input unchanged (D-12 no-op)."""
    from daydream.atif import Step as AtifStep

    step = AtifStep(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message="hello",
    )
    out = Redactor().redact_step(step)
    assert out is step
```

**Phase 4 test pattern — same Redactor() direct-instantiation, replace `assert out is step` with assertions on the redacted text:**

```python
def test_redactor_scrubs_api_key() -> None:
    """REDA-01: sk-* tokens replaced with [REDACTED_API_KEY]."""
    step = Step(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message="OPENAI_API_KEY=sk-1234abcdef",
    )
    out = Redactor().redact_step(step)
    assert "sk-1234abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message or "[REDACTED_ENV_VAR]" in out.message


def test_redactor_scrubs_username_path() -> None:
    """REDA-02: /Users/<name>/ replaced with /Users/[REDACTED_USER]/."""
    step = Step(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message="Working in /Users/ka/github/project/src/app.py",
    )
    out = Redactor().redact_step(step)
    assert "/Users/ka" not in out.message
    assert "/Users/[REDACTED_USER]" in out.message
    # Project-relative tail must survive (D-02)
    assert "github/project/src/app.py" in out.message


def test_redactor_preserves_non_secret_env_vars() -> None:
    """REDA-03: DEBUG=true, APP_NAME=foo pass through unredacted."""
    step = Step(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message="DEBUG=true\nAPP_NAME=myproject",
    )
    out = Redactor().redact_step(step)
    assert "DEBUG=true" in out.message
    assert "APP_NAME=myproject" in out.message
    assert "[REDACTED" not in out.message
```

**Apply-to-all-surfaces test — covers `tool_calls[*].arguments` and `observation.results[*].content`:**

```python
def test_redactor_applies_to_tool_arguments_and_observation_content() -> None:
    """REDA-04: same patterns applied to ToolCall.arguments and ObservationResult.content."""
    from daydream.atif import ObservationResult, Observation, ToolCall

    step = Step(
        step_id=1,
        timestamp=now_iso(),
        source="agent",
        model_name="opus",
        message="ok",
        tool_calls=[ToolCall(
            tool_call_id="t1",
            function_name="Bash",
            arguments={"command": "echo sk-secretkey"},
        )],
        observation=Observation(results=[ObservationResult(
            source_call_id="t1",
            content="leaked path /Users/ka/.ssh/id_rsa",
        )]),
    )
    out = Redactor().redact_step(step)
    assert "sk-secretkey" not in str(out.tool_calls[0].arguments)
    assert "/Users/ka" not in str(out.observation.results[0].content)
```

---

## Shared Patterns

### Compiled-regex Module Constants
**Source:** `daydream/ui.py:985-1013`, `daydream/exploration_runner.py:47`, `daydream/backends/codex.py:31-32`
**Apply to:** `daydream/trajectory.py` Redactor implementation
**Idiom:** Module-level `_NAME_PATTERN = re.compile(r"...")` constants. Underscore prefix because they're private. Re-used across calls without recompilation.

### Boundary-Catch Around Recording / Side-Effect Code
**Source:** `daydream/trajectory.py:237-248` (`Invocation.observe`), `daydream/trajectory.py:417-423` (`__aexit__`), `daydream/trajectory.py:587-588` (`_ForkCM.__aexit__`)
**Apply to:** `Redactor.redact_step` (REDA-05 contract: never crash a run), partial-flush write path
**Idiom:**
```python
try:
    self._redact_text(...)
except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
    print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")
```
**Failure mode for Redactor specifically (REDA-05):** "redact-or-omit, never raw-pass-through" — replace the offending field value with `"[REDACTION_FAILED]"` rather than yielding the unredacted original.

### `print_warning` / `print_error` Call Style (UI helpers)
**Source:** `daydream/ui.py:1463-1505` (function defs), call sites throughout `daydream/phases.py:727, 786, 872, 896, 900, 904, 959, 1039, 1041, 1114, 1349, 1373, 1486` and `daydream/runner.py:118, 212, 476`
**Apply to:** All ~10 promoted log sites
**Idiom:**
```python
# Single-message warning (no title)
print_warning(console, "<single-line user-facing reason>")

# Two-part error (title + detail)
print_error(console, "<short title>", "<longer detail or exception text>")
```
**Quiet-mode contract (D-09):** `print_warning` and `print_error` themselves do NOT check `get_quiet_mode()` (verified by reading `ui.py:1463-1505`). Quiet handling is a call-site decision. D-09 means: don't wrap `print_error` in a `get_quiet_mode()` check; DO wrap `print_warning` in one if the call site is in a noisy hot path. None of the ~10 promoted sites in Phase 4 are in hot paths — direct unwrapped calls are correct.

### Argparse Add-Flag Pattern
**Source:** `daydream/cli.py:234-241` (`--ignore-path`), `daydream/cli.py:250-257` (`--max-iterations`)
**Apply to:** new `--trajectory` flag
**Idiom:** Single `parser.add_argument(...)` with explicit `default=None`, `metavar`, `dest` (when name has hyphens), and a `help=` string that documents the default.

### Argparse Hard-Reject Pattern
**Source:** Argparse's built-in error path (no in-repo analog — it's just default behavior).
**Apply to:** `--debug` removal (D-05)
**Idiom:** **Do nothing extra.** Deleting the `parser.add_argument("--debug", ...)` block IS the hard reject — argparse exits with code 2 and `error: unrecognized arguments: --debug` automatically. No mutual-exclusion code, no warning-and-continue branch, no deprecation shim.

### Test Isolation via Autouse Fixture
**Source:** `tests/conftest.py:125-141` (`_reset_trajectory_recorder`)
**Apply to:** No new fixture needed for Phase 4 — the existing autouse fixture already handles trajectory state. The `_log_debug` removal eliminates the need for a `set_debug_log(None)` reset (which never existed).

## No Analog Found

Files with no close match in the codebase:

| File / Area | Role | Reason |
|-------------|------|--------|
| `tests/test_cutover_ast.py` (CUT-08 AST sweep) | test | No existing `ast` module usage anywhere in the repo. Pattern is straightforward standard-library — no analog needed. Filesystem-walk idiom (`Path.rglob`) is the only borrowed pattern. |
| `Redactor` regex-table dispatch | utility | The compiled-regex pattern itself has analogs (see ui.py / exploration_runner.py); the **table-driven dispatch** of (pattern → token) pairs is novel here. Closest in-repo is `daydream/agent.py:251-271` (success/error pattern lists), which uses the exact-same `for pattern in PATTERNS: if re.search(...)` idiom. |

## Metadata

**Analog search scope:**
- `daydream/` (15 source files)
- `daydream/atif/` (vendored ATIF models)
- `tests/` (35 test files)
- `scripts/` (no relevant code)

**Files scanned (Read or Grep'd):** 14 source files + 4 test files

**Pattern extraction date:** 2026-04-28

---

## PATTERN MAPPING COMPLETE

**Phase:** 4 - Cutover + Redaction + CLI Surface
**Files classified:** 11 areas (7 modifications, 2 new tests, 2 cross-cutting test/utility patterns)
**Analogs found:** 9 / 9 areas have a clear in-repo analog (the AST sweep is the only "no analog" item, and standard-library `ast` is well-understood)

### Coverage
- Files with exact analog: 7
- Files with role-match analog: 2
- Files with no analog: 2 (AST sweep is novel; pure-deletion areas need none)

### Key Patterns Identified
- All compiled regex follows `_NAME_PATTERN = re.compile(...)` module-constant convention (ui.py / exploration_runner.py / agent.py)
- Trajectory-recording exception handling uses the `try/except Exception/print_warning(_console, ...)` boundary-catch (`# noqa: BLE001 — recording must never crash a run`) — applied uniformly to `Redactor.redact_step` and partial-write paths
- All UI output goes through `print_error(console, title, detail)` / `print_warning(console, message)` — no raw `print()`, no `console.print()` for promoted log sites
- Argparse hard-reject for `--debug` is FREE — just delete the `add_argument` block
- Step / ToolCall / ObservationResult are immutable Pydantic — Redactor uses `.model_copy(update=...)` to produce redacted copies
- AST sweep (CUT-08) is novel; no in-repo analog beyond stdlib `ast` and `Path.rglob` — self-exclusion of the test file itself via `__file__` comparison is the only non-obvious technique
