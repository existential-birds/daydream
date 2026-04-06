# Phase 2: Pre-scan Exploration - Research

**Researched:** 2026-04-06
**Domain:** Diff-driven parallel subagent exploration with tree-sitter import tracing
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Subagent Design**
- **D-01:** Three distinct specialist subagents with focused prompts and clear field ownership:
  - **pattern-scanner** → `conventions` + `guidelines` (reads CLAUDE.md, .coderabbit.yaml, house-style configs, infers conventions from code)
  - **dependency-tracer** → `dependencies` + contributes to `affected_files` (call-chain and import tracing beyond initial impact surface)
  - **test-mapper** → contributes to `affected_files` with `role="test"` (maps changed files to test coverage)
- **D-02:** Subagents read project guidelines directly as part of their own exploration — no pre-extraction step. Pattern-scanner is the primary owner of guideline reading (EXPL-04).
- **D-03:** Each subagent returns structured JSON matching its `ExplorationContext` fields. A merge function in `daydream/exploration.py` combines results into a single `ExplorationContext`.

**Diff Analysis & File Discovery**
- **D-04:** `detect_affected_files()` uses **tree-sitter** for AST-based import/dependency parsing. Regex-based parsing is **explicitly rejected**.
- **D-05:** Initial language support: **Python, TypeScript/JavaScript, Go, Rust**. Do not pull in the full `tree-sitter-languages` bundle.
- **D-06:** Files in unsupported languages → added with `role="modified"` but no dependency tracing. Dependency-tracer subagent investigates them via grep/read manually.
- **D-07:** Import tracing depth is **configurable via `RunConfig`, default = 1**.

**Scaling Thresholds**
- **D-08:** Tier metric is **changed file count** from the diff (not line count).
- **D-09:** Tiers:
  - **Skip** (0–1 files): empty `ExplorationContext`
  - **Single** (2–3 files): **dependency-tracer only**
  - **Parallel** (4+ files): all three subagents in parallel
- **D-10:** No global kill-switch flag. Automatic scaling only.

**Flow Integration**
- **D-11:** Wires into **both `--ttt` and normal review flows** in this phase.
- **D-12:** In `run_trust()`, exploration runs **before `phase_understand_intent`**.
- **D-13:** In `run()`, exploration runs **before `phase_review`**.
- **D-14:** `ExplorationContext` stored on **`RunConfig`** as `exploration_context: ExplorationContext | None = None`.

**UX**
- **D-15:** Live panel with per-subagent status via existing `LiveToolPanelRegistry` pattern.
- **D-16:** Dedicated **"EXPLORE" phase hero** via `print_phase_hero()` before the live panel.
- **D-17:** Skipped exploration prints a brief dim-text notice.

### Claude's Discretion
- Exact prompt text for each specialist subagent
- JSON schemas for each subagent's structured output (derived from `ExplorationContext` field types)
- Internal structure of the merge function
- Tree-sitter integration details (binding choice, grammar loading, parser caching)
- `detect_affected_files()` function signature and return shape beyond "list of files with role + dependency edges"
- Live panel row labels and intermediate status text
- Whether new code lives in extended `exploration.py` or new `exploration_runner.py`

### Deferred Ideas (OUT OF SCOPE)
- Global exploration force-on/off flag
- Two-hop / deeper default tracing
- Elixir + Swift grammars (coming soon, not now)
- Token budgets / cost caps for subagents
- Exploration result caching (ADVX-01, v2)
- Confidence scores, convention filtering, grounded recommendations (Phase 3)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXPL-01 | System maps impact surface from diff (affected files + transitive deps) | tree-sitter AST import parsing in `detect_affected_files()`; dependency-tracer subagent extends beyond immediate surface |
| EXPL-02 | System reads diff-adjacent files (touched + immediate imports/callers) | Default trace depth = 1 (D-07); tree-sitter resolves direct imports per file |
| EXPL-03 | System detects codebase conventions/patterns before review starts | pattern-scanner subagent (D-01) infers from code + reads house-style configs |
| EXPL-04 | Exploration subagents read project guidelines (CLAUDE.md, .coderabbit.yaml, etc.) | pattern-scanner is the explicit owner of guideline reading (D-02) |
| AGNT-01 | 3-5 parallel pre-scan subagents explore affected areas before review | Three subagents launched via `Backend.execute(agents=...)` in Parallel tier (D-09) |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- All commands via `uv run` / `uv sync`; lockfile committed
- Python 3.12 target; modern union syntax (`X | None`)
- ruff (`E`,`F`,`I`,`W`, line length 120) and mypy (`ignore_missing_imports = true`) MUST pass — `make check` is the gate
- pytest with `asyncio_mode = "auto"`; existing 50+ tests must keep passing
- All agent calls go through `Backend` protocol — no direct SDK calls in `phases.py`
- `@dataclass` for config/data types; `from __future__ import annotations` + `TYPE_CHECKING` for expensive imports
- Google-style docstrings; phase functions take `backend: Backend` first
- All terminal output via `daydream.ui` helpers — no `print()`
- Subprocess calls require `# noqa: S603` with justification comment
- GSD workflow enforcement: file edits only via GSD commands

## Summary

Phase 2 sits on top of an already-built foundation: `ExplorationContext`, `safe_explore()`, the `agents` kwarg on `Backend.execute()`, and `ClaudeBackend.execute()` are all in place from Phase 1. This phase delivers three things on top of that:

1. **`detect_affected_files()`** — a synchronous, tree-sitter-backed function that takes a git diff and returns the impact surface (changed files + 1-hop imports/importers) for Python/TS-JS/Go/Rust, with graceful pass-through for unsupported languages.
2. **A pre-scan orchestrator** — diff-size-tiered logic that launches 0, 1, or 3 subagents via `Backend.execute(agents=...)`, each producing structured JSON, merged into a single `ExplorationContext`.
3. **Flow wiring + UX** — `RunConfig.exploration_context` populated before `phase_review`/`phase_understand_intent`; "EXPLORE" phase hero plus a `LiveToolPanelRegistry`-driven multi-row status panel.

**Primary recommendation:** Use individual `tree-sitter-{python,typescript,go,rust}` grammar packages plus `tree-sitter` 0.25.x, register them in a single `LANGUAGES` dict for one-line future grammar additions, cache `Parser` instances per language at module level, and load them lazily under `TYPE_CHECKING` to keep cold-start cheap. Build `detect_affected_files()` as pure/sync (no async, no Backend) so it's trivially unit-testable.

**CRITICAL FOUNDATION DISCREPANCY (must address in plan):** The Phase 1 `Backend` protocol declares `agents: list[AgentDefinition] | None = None`, but the underlying SDK type is `agents: dict[str, AgentDefinition] | None` (verified in `.claude/skills/claude-agent-sdk/REFERENCE.md` line 57 and `AgentDefinition` definition at line 297 — `AgentDefinition` has no `name` field, the name lives in the dict key). The current Phase 1 protocol cannot be passed straight to `ClaudeSDKClient`. The planner MUST verify how `ClaudeBackend.execute()` currently consumes the `list` (likely converts to dict internally, or it's broken) and either keep the list-of-pairs convention or change the protocol to `dict[str, AgentDefinition]`. This is a Wave 0 verification step, not optional.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `tree-sitter` | 0.25.2 (2025-09-25) | Python bindings for the tree-sitter parsing C library | The canonical, official binding maintained by tree-sitter org. Pre-built wheels for all major platforms; no compilation needed. |
| `tree-sitter-python` | 0.25.0 (2025-09-11) | Python grammar | Official grammar package; ABI matches `tree-sitter` 0.25.x. |
| `tree-sitter-typescript` | 0.23.2 (2024-11-11) | TypeScript + TSX + JavaScript grammar | Official grammar; package exposes both `language_typescript()` and `language_tsx()`. |
| `tree-sitter-go` | 0.25.0 (2025-08-29) | Go grammar | Official grammar. |
| `tree-sitter-rust` | 0.24.2 (2026-03-27) | Rust grammar | Official grammar. |

**Versions verified:** Each version above was confirmed against `https://pypi.org/pypi/<pkg>/json` on 2026-04-06. All four grammar packages declare ABI 14/15 compatibility with `tree-sitter` 0.25.x — install together and let `uv lock` resolve.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Individual grammar packages | `tree-sitter-language-pack` 1.4.1 | One install, 248 languages, on-demand downloads. **Rejected:** D-05 explicitly says "do not pull in the full bundle." Bundle is large, brings transitive download behavior, and obscures which grammars are actually supported. Per-language packages give explicit, auditable surface. |
| Individual grammar packages | `tree-sitter-languages` (grantjenks fork) | **Rejected:** package is unmaintained and the README points to `tree-sitter-language-pack` as successor. |
| tree-sitter | Python `ast` module + per-language regex | **Rejected by D-04** ("regex is too unreliable"). `ast` only handles Python; can't unify the four languages under one model. |

**Installation:**
```bash
uv add tree-sitter tree-sitter-python tree-sitter-typescript tree-sitter-go tree-sitter-rust
```

This adds five entries to `[project].dependencies` in `pyproject.toml` and updates `uv.lock`. All five publish pre-built wheels for macOS/Linux/Windows on cpython 3.12 — no compiler toolchain required on developer machines or CI.

## Architecture Patterns

### Recommended Module Structure
```
daydream/
├── exploration.py           # Existing (Phase 1) — extended with merge_contexts() helper
├── exploration_runner.py    # NEW — orchestration, tier logic, subagent launching
├── tree_sitter_index.py     # NEW — detect_affected_files(), language registry, parser cache
└── prompts/
    └── exploration_subagents.py  # NEW — three subagent prompt builders + JSON schemas
```

Splitting `tree_sitter_index.py` from `exploration_runner.py` keeps the pure-AST code (sync, no Backend) separately testable from the async orchestration code (Backend-dependent, needs MockBackend in tests). Whether the planner collapses these is at Claude's discretion (D — context), but the test surface argues for the split.

### Pattern 1: Lazy Language Registry
**What:** A single module-level `dict` mapping a language id ("python", "typescript", "tsx", "javascript", "go", "rust") to a tuple of `(extensions, language_callable)`. Parsers cached per language.
**When to use:** Every entry in `detect_affected_files()` and any future grammar.
**Why:** D-05 requires Elixir/Swift to be addable as a one-line registration. A central registry is the only way to deliver that.
**Example:**
```python
# daydream/tree_sitter_index.py
# Source: tree-sitter docs https://tree-sitter.github.io/py-tree-sitter/
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from tree_sitter import Language, Parser

if TYPE_CHECKING:
    pass


def _python_lang() -> Language:
    import tree_sitter_python
    return Language(tree_sitter_python.language())


def _typescript_lang() -> Language:
    import tree_sitter_typescript
    return Language(tree_sitter_typescript.language_typescript())


def _tsx_lang() -> Language:
    import tree_sitter_typescript
    return Language(tree_sitter_typescript.language_tsx())


def _go_lang() -> Language:
    import tree_sitter_go
    return Language(tree_sitter_go.language())


def _rust_lang() -> Language:
    import tree_sitter_rust
    return Language(tree_sitter_rust.language())


# extension -> (language_id, factory)
LANGUAGES: dict[str, tuple[str, Callable[[], Language]]] = {
    ".py":  ("python",     _python_lang),
    ".ts":  ("typescript", _typescript_lang),
    ".tsx": ("tsx",        _tsx_lang),
    ".js":  ("javascript", _typescript_lang),  # TS grammar handles JS
    ".jsx": ("tsx",        _tsx_lang),
    ".go":  ("go",         _go_lang),
    ".rs":  ("rust",       _rust_lang),
}

_PARSER_CACHE: dict[str, Parser] = {}


def get_parser(language_id: str) -> Parser | None:
    if language_id in _PARSER_CACHE:
        return _PARSER_CACHE[language_id]
    for _, (lid, factory) in LANGUAGES.items():
        if lid == language_id:
            parser = Parser(factory())
            _PARSER_CACHE[language_id] = parser
            return parser
    return None
```
Adding Elixir later is one block + one row in `LANGUAGES`. No refactor required.

### Pattern 2: Tree-sitter Query for Imports
**What:** Use a tree-sitter `Query` against each language grammar's import nodes rather than walking the AST manually.
**Why:** Queries are declarative S-expressions that compile once per language and are robust to grammar version drift.
**Example (Python):**
```python
# Source: tree-sitter docs — Query API
PYTHON_IMPORT_QUERY = """
(import_statement name: (dotted_name) @import)
(import_from_statement module_name: (dotted_name) @import)
(import_from_statement module_name: (relative_import) @import)
"""
```
Per-language query strings live next to `LANGUAGES`. The dependency-tracer subagent does NOT need them — they're only consumed by `detect_affected_files()`.

### Pattern 3: Subagent as `AgentDefinition` Dict Entry
**What:** Each specialist subagent is a `dict[str, AgentDefinition]` entry passed to `Backend.execute(agents=...)`. The orchestrator's *prompt* tells the lead agent to delegate to all three in parallel and emit a structured JSON envelope containing the three sub-results.
**Why:** This is how `claude-agent-sdk` 0.1.52's `AgentDefinition` actually works — a named map of available specialists, not a list of tasks. The lead agent fans out via the SDK's built-in subagent tool calls.
**Verified against:** `.claude/skills/claude-agent-sdk/REFERENCE.md` line 57 (`agents: dict[str, AgentDefinition] | None = None`) and line 297 (`AgentDefinition` has `description`, `prompt`, `tools`, `model` — no `name` field).

### Pattern 4: Pure / Sync Diff Parser
**What:** `detect_affected_files(diff_text: str, repo_root: Path, depth: int = 1) -> list[FileInfo]` is synchronous, takes only data, and returns only data. No `Backend`, no `await`, no UI calls.
**Why:** Trivially unit-testable with diff fixtures; no async test scaffolding; can be called from both `run()` and `run_trust()` without context.

### Anti-Patterns to Avoid
- **Pre-loading all parsers at import time:** delays cold-start measurably; load lazily in `get_parser()`.
- **Walking the full AST manually:** tree-sitter Query is faster and grammar-version-resilient.
- **Hidden mutation of `RunConfig`:** mutate once at the orchestrator boundary, never inside subagent code paths.
- **Hand-rolled diff parsing for filename extraction:** use `git diff --name-status` from `_git_diff()` infrastructure or a focused wrapper — don't parse unified diff hunks just to get file names.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-language import parsing | Per-language regex / `ast` module | tree-sitter + per-grammar Query | Comments, multi-line imports, conditional imports, string literals containing `import` all break regex. Locked by D-04. |
| Subagent fan-out | `anyio.create_task_group()` calling `Backend.execute()` three times | Single `Backend.execute(agents={...})` call | The SDK's `AgentDefinition` mechanism IS the parallel-subagent primitive. Manual fan-out bypasses the lead-agent coordination the SDK provides and breaks the `Backend` abstraction. |
| Grammar bundling | Custom build of tree-sitter language libs | Official `tree-sitter-{lang}` PyPI wheels | Pre-compiled wheels exist for all four target languages on all target platforms. No reason to build from source. |
| File-list extraction from diff | Parse unified diff hunks | `git diff --name-status` (already used by `_git_diff()` in runner) | Existing infrastructure; no new failure modes. |
| Live multi-row status UI | Build a new Rich `Live` wrapper | Reuse `LiveToolPanelRegistry` from `daydream/ui.py` | D-15 explicitly requires reuse. The pattern is already proven for tool calls. |

**Key insight:** Almost every "build it ourselves" temptation in this phase is already solved either by the SDK (subagent fan-out), by the existing codebase (UI panel, diff helper), or by mature ecosystem tooling (tree-sitter grammars). The phase's real work is gluing these together, not reinventing them.

## Common Pitfalls

### Pitfall 1: ABI mismatch between `tree-sitter` and grammar packages
**What goes wrong:** `tree-sitter` 0.25 expects ABI 14/15; an older grammar package built against ABI 13 raises `ValueError: Incompatible Language version` at parser creation.
**Why it happens:** Grammar packages and the core binding version independently. Mixed versions in a lockfile drift over time.
**How to avoid:** Pin all four grammar packages to versions published *after* `tree-sitter` 0.25.0 (Sept 2025). Verified versions in the Standard Stack table satisfy this.
**Warning signs:** `Incompatible Language version` at first parser construction — fail loud during a smoke test, never silently.

### Pitfall 2: TypeScript grammar exposes two languages
**What goes wrong:** `tree_sitter_typescript.language()` does not exist; the package exports `language_typescript()` and `language_tsx()` separately.
**Why it happens:** The TS grammar is actually two grammars sharing infrastructure. JS files parse fine through `language_typescript()`; JSX/TSX files require `language_tsx()`.
**How to avoid:** Map `.ts/.js` → `language_typescript`, `.tsx/.jsx` → `language_tsx` in the registry (shown in Pattern 1).

### Pitfall 3: `Backend.execute(agents=...)` type mismatch with SDK
**What goes wrong:** Phase 1 declared `agents: list[AgentDefinition] | None`. SDK expects `dict[str, AgentDefinition] | None`. Whatever conversion `ClaudeBackend.execute()` is doing today is currently untested with real subagents (Phase 1 only verified the kwarg threads through).
**Why it happens:** The Phase 1 protocol was designed before any real subagent invocation existed.
**How to avoid:** Wave 0 task: read `daydream/backends/claude.py` end-to-end, identify how it currently maps the list to whatever it passes to `ClaudeAgentOptions`, and either (a) confirm the conversion is correct, or (b) change the protocol. The planner MUST address this before Wave 1 builds anything on top.
**Warning signs:** Subagents silently never run; `ClaudeAgentOptions(agents=...)` raises a TypeError; the lead agent reports it has no specialists available.

### Pitfall 4: Diff with deleted files
**What goes wrong:** `detect_affected_files()` tries to read a file that the diff deleted; raises `FileNotFoundError`.
**Why it happens:** Naive implementations open every changed path.
**How to avoid:** Inspect the `git diff --name-status` status letter — `D` (deleted) entries get `role="modified"` but skip the import parse step entirely. `A` (added) and `M` (modified) get parsed.

### Pitfall 5: Symbol search vs. file path search
**What goes wrong:** Resolving `from foo.bar import baz` to a file requires knowing whether `baz` is a module, a class, or a function. Naive resolution maps `foo.bar` to `foo/bar.py` but misses `foo/bar/__init__.py` and `foo/bar/baz.py`.
**Why it happens:** Python import resolution is a runtime concern, not a static one.
**How to avoid:** Resolve to *both* candidates and include any that exist on disk; document this as best-effort. For TS, look for `.ts`, `.tsx`, `.d.ts`, and `index.{ts,tsx}` siblings. The dependency-tracer subagent picks up any cases the static resolver misses.

### Pitfall 6: Subagent prompts that don't return JSON
**What goes wrong:** Subagent emits prose; merge function fails to parse.
**Why it happens:** No structured-output schema attached to the subagent invocation.
**How to avoid:** Each subagent prompt ends with an explicit "Return ONLY a JSON object matching this schema: ..." block, AND the orchestrator wraps the call with `output_schema=` enforcement at the Backend level (mirror the existing `FEEDBACK_SCHEMA` pattern in `daydream/phases.py`).

### Pitfall 7: Live panel rows that never close
**What goes wrong:** A subagent times out / errors, the Rich `Live` row spins forever.
**Why it happens:** The error path doesn't update the registry row.
**How to avoid:** Wrap each subagent's lifecycle in try/finally that always sets the row to a terminal state (`done` or `error`). `safe_explore()` covers the outer error path but not the per-row cleanup.

## Code Examples

### Tree-sitter import extraction (Python)
```python
# Source: https://tree-sitter.github.io/py-tree-sitter/ + tree-sitter-python README
from pathlib import Path
from tree_sitter import Query

def extract_python_imports(parser, source: bytes) -> list[str]:
    tree = parser.parse(source)
    query = Query(parser.language, PYTHON_IMPORT_QUERY)
    return [
        node.text.decode("utf-8")
        for node, _ in query.captures(tree.root_node).items()
        for node in _
    ]
```

### `git diff --name-status` parsing
```python
# Source: existing _git_diff() pattern in daydream/runner.py
import subprocess

def changed_files(repo_root: Path) -> list[tuple[str, str]]:
    result = subprocess.run(  # noqa: S603 — args are hardcoded
        ["git", "-C", str(repo_root), "diff", "--name-status", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    out: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            out.append((parts[0], parts[1]))  # (status, path)
    return out
```

### Subagent definition map
```python
# Source: .claude/skills/claude-agent-sdk/REFERENCE.md (lines 57, 297)
from claude_agent_sdk.types import AgentDefinition

EXPLORATION_AGENTS: dict[str, AgentDefinition] = {
    "pattern-scanner": AgentDefinition(
        description="Reads CLAUDE.md, .coderabbit.yaml, and code samples to detect conventions and guidelines.",
        prompt=PATTERN_SCANNER_PROMPT,  # ends with structured-output JSON contract
        tools=["Read", "Glob", "Grep"],
        model="inherit",
    ),
    "dependency-tracer": AgentDefinition(
        description="Traces call chains and imports beyond the initial impact surface.",
        prompt=DEPENDENCY_TRACER_PROMPT,
        tools=["Read", "Grep", "Glob"],
        model="inherit",
    ),
    "test-mapper": AgentDefinition(
        description="Maps changed source files to their test coverage.",
        prompt=TEST_MAPPER_PROMPT,
        tools=["Read", "Glob", "Grep"],
        model="inherit",
    ),
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `tree-sitter-languages` (grantjenks bundled wheels) | `tree-sitter-language-pack` OR per-grammar packages | 2024 — original package unmaintained | Per-grammar packages preferred when surface is small and explicit (D-05). |
| `Parser.set_language(lang)` | `Parser(language)` constructor | tree-sitter 0.22+ | Old style still works but is deprecated. New code uses constructor. |
| Manual subagent orchestration via multiple SDK clients | Single `ClaudeAgentOptions(agents=...)` dict | claude-agent-sdk 0.1.52 (Phase 1 already adopted) | Subagent coordination is now an SDK concern; the Backend protocol just passes the dict through. |

**Deprecated/outdated:**
- `tree-sitter-languages` (PyPI): unmaintained, do not use.
- `Language.build_library(...)`: removed in 0.22; pre-built wheels are the only supported install path now.

## Open Questions

1. **Does `ClaudeBackend.execute()` correctly pass `agents` to `ClaudeAgentOptions`?**
   - What we know: Phase 1 added the kwarg to the protocol as `list[AgentDefinition]`.
   - What's unclear: SDK actually wants `dict[str, AgentDefinition]`. Whatever conversion exists today is untested with real subagents.
   - Recommendation: Wave 0 verification task — read `daydream/backends/claude.py` end-to-end, write a smoke test that actually invokes a subagent through the protocol, fix the protocol if needed.

2. **How does the Codex backend handle `agents`?**
   - What we know: `CodexBackend` exists in `daydream/backends/codex.py`. Codex CLI does not have native subagents.
   - What's unclear: Whether Codex must raise `NotImplementedError`, fall back to sequential execution, or simulate via separate process spawns.
   - Recommendation: For this phase, **either** raise a clear `NotImplementedError("Codex backend does not support exploration subagents")` from `CodexBackend.execute()` when `agents` is non-None and route exploration only through Claude, **or** add a `Backend.supports_subagents` capability flag. Pick one in planning.

3. **Should `detect_affected_files()` handle non-git working trees?**
   - What we know: `_git_diff()` already assumes git. Daydream is currently git-only.
   - What's unclear: Whether this phase should add any defensive handling.
   - Recommendation: Out of scope. Existing failure mode (subprocess error → `safe_explore()` catches → empty context) is sufficient.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All | ✓ (project requires it) | 3.12+ | — |
| uv | Install/run | ✓ (project standard) | — | — |
| git CLI | `_git_diff()`, `detect_affected_files()` | ✓ (already required by Phase 0/1) | — | None — empty context via `safe_explore()` |
| `tree-sitter` 0.25.x wheels | `detect_affected_files()` | Will be added via `uv add` | 0.25.2 | None — phase cannot complete without it |
| `tree-sitter-{python,typescript,go,rust}` wheels | Per-language parsers | Will be added via `uv add` | (see Standard Stack) | Per-language: file gets `role="modified"` only (D-06) |
| `claude-agent-sdk` 0.1.52 | Subagent execution | ✓ (pinned in pyproject.toml) | 0.1.52 | None |
| `codex` CLI | CodexBackend (non-exploration paths) | Optional | — | Exploration may NotImplementedError on Codex (Open Question 2) |

**Missing dependencies with no fallback:**
- None blocking — all tree-sitter wheels are first-class PyPI packages with macOS/Linux/Windows wheels for cpython 3.12.

**Missing dependencies with fallback:**
- Per-language tree-sitter grammars: any individual file in an unsupported (or wheel-failed) language degrades to `role="modified"` with no AST trace, per D-06.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x + pytest-asyncio 0.24+ (`asyncio_mode = "auto"`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/test_exploration.py tests/test_tree_sitter_index.py -x` |
| Full suite command | `uv run pytest -v` |
| Lint/type gate | `make check` (ruff + mypy + pytest) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXPL-01 | `detect_affected_files()` returns changed files + 1-hop imports for a multi-file Python diff fixture | unit | `uv run pytest tests/test_tree_sitter_index.py::test_python_impact_surface -x` | ❌ Wave 0 |
| EXPL-01 | Same for TypeScript | unit | `uv run pytest tests/test_tree_sitter_index.py::test_typescript_impact_surface -x` | ❌ Wave 0 |
| EXPL-01 | Same for Go | unit | `uv run pytest tests/test_tree_sitter_index.py::test_go_impact_surface -x` | ❌ Wave 0 |
| EXPL-01 | Same for Rust | unit | `uv run pytest tests/test_tree_sitter_index.py::test_rust_impact_surface -x` | ❌ Wave 0 |
| EXPL-02 | Default trace depth = 1 (no two-hop traversal) | unit | `uv run pytest tests/test_tree_sitter_index.py::test_default_depth_is_one -x` | ❌ Wave 0 |
| EXPL-02 | `RunConfig.exploration_depth` overrides default | unit | `uv run pytest tests/test_runner.py::test_run_config_exploration_depth -x` | ❌ Wave 0 |
| EXPL-03 | pattern-scanner JSON merges into `ExplorationContext.conventions` | unit (mock subagent) | `uv run pytest tests/test_exploration.py::test_merge_pattern_scanner_result -x` | ❌ Wave 0 |
| EXPL-04 | pattern-scanner subagent prompt instructs reading CLAUDE.md / .coderabbit.yaml | unit (string assertion on prompt) | `uv run pytest tests/test_exploration_runner.py::test_pattern_scanner_prompt_includes_guideline_files -x` | ❌ Wave 0 |
| AGNT-01 | Parallel tier launches three subagents via `Backend.execute(agents=...)` | unit (MockBackend records call) | `uv run pytest tests/test_exploration_runner.py::test_parallel_tier_launches_three_agents -x` | ❌ Wave 0 |
| AGNT-01 | Single tier (2-3 files) launches dependency-tracer only | unit | `uv run pytest tests/test_exploration_runner.py::test_single_tier_dependency_tracer_only -x` | ❌ Wave 0 |
| AGNT-01 | Skip tier (0-1 files) launches no subagents | unit | `uv run pytest tests/test_exploration_runner.py::test_skip_tier_no_subagents -x` | ❌ Wave 0 |
| (cross-cutting) | `safe_explore()` catches a failing subagent and returns empty `ExplorationContext` | unit | `uv run pytest tests/test_exploration.py::test_safe_explore_swallows_failure -x` (already passes from Phase 1; re-verify) | ✅ |
| (cross-cutting) | `run()` and `run_trust()` populate `RunConfig.exploration_context` before calling downstream phases | integration (MockBackend, full runner) | `uv run pytest tests/test_integration.py::test_run_populates_exploration_context -x` | ❌ Wave 0 |
| (cross-cutting) | Backend protocol agents kwarg actually reaches `ClaudeAgentOptions` correctly (smoke) | integration (mock SDK) | `uv run pytest tests/test_backend_claude.py::test_execute_passes_agents_dict_to_options -x` | ❌ Wave 0 — see Open Question 1 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_exploration.py tests/test_tree_sitter_index.py tests/test_exploration_runner.py -x`
- **Per wave merge:** `uv run pytest -v` (full suite — must stay green; 50+ existing tests cannot regress)
- **Phase gate:** `make check` (lint + typecheck + full suite) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_tree_sitter_index.py` — covers EXPL-01, EXPL-02; needs language-specific diff fixtures (`tests/fixtures/diffs/*.diff`) and small source-tree fixtures
- [ ] `tests/test_exploration_runner.py` — covers AGNT-01, EXPL-04 prompt assertions; needs MockBackend extension that records `agents=` payloads
- [ ] Extend `tests/fixtures/` with `diffs/` subdir holding one fixture per supported language
- [ ] Extend existing `MockBackend` (in `tests/test_phases.py` or wherever it lives) to capture `agents` kwarg for assertion
- [ ] Verification test for `ClaudeBackend.execute()` actually wiring `agents` into `ClaudeAgentOptions` (Open Question 1)
- [ ] Tree-sitter grammar dependencies installed: `uv add tree-sitter tree-sitter-python tree-sitter-typescript tree-sitter-go tree-sitter-rust`

## Sources

### Primary (HIGH confidence)
- `.claude/skills/claude-agent-sdk/REFERENCE.md` (local) — `ClaudeAgentOptions.agents: dict[str, AgentDefinition]` (line 57); `AgentDefinition` fields (line 297)
- `daydream/exploration.py`, `daydream/backends/__init__.py` (local Phase 1 source) — actual current data model and protocol signatures
- PyPI `/pypi/<pkg>/json` API — verified versions and upload dates for `tree-sitter` 0.25.2, `tree-sitter-python` 0.25.0, `tree-sitter-typescript` 0.23.2, `tree-sitter-go` 0.25.0, `tree-sitter-rust` 0.24.2, `tree-sitter-language-pack` 1.4.1 (queried 2026-04-06)
- [py-tree-sitter docs](https://tree-sitter.github.io/py-tree-sitter/) — current Parser/Language/Query API
- [py-tree-sitter GitHub](https://github.com/tree-sitter/py-tree-sitter) — official binding repo

### Secondary (MEDIUM confidence)
- [tree-sitter-language-pack on PyPI](https://pypi.org/project/tree-sitter-language-pack/) — alternative bundled approach (rejected per D-05, documented for completeness)
- [Simon Willison's TIL: Using tree-sitter with Python](https://til.simonwillison.net/python/tree-sitter) — practical patterns for parser caching and Query usage
- [py-tree-sitter-languages](https://github.com/grantjenks/py-tree-sitter-languages) — confirms unmaintained status; points to language-pack as successor

### Tertiary (LOW confidence)
- None — all critical claims for this phase are backed by primary sources or local code.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified directly from PyPI on research date; ABI compatibility cross-checked
- Architecture: HIGH — patterns follow existing daydream conventions plus official tree-sitter idioms
- Pitfalls: HIGH for the SDK/tree-sitter mismatches (verified against local code and skill reference); MEDIUM for the Codex subagent question (open)
- Validation: HIGH — test framework and gating already established by project conventions

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (30 days; tree-sitter and SDK both stable, but the SDK type discrepancy makes a quick re-check worthwhile if planning slips)
