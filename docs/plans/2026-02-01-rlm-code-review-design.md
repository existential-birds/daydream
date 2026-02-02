# RLM Code Review Mode for Daydream

**Date**: 2026-02-01
**Status**: Draft
**Authors**: Collaborative design session

## Overview

Add Recursive Language Model (RLM) capabilities to Daydream for reviewing large codebases (1M+ tokens) that exceed standard context windows. Based on the [Recursive Language Models paper](https://arxiv.org/abs/...) by Zhang et al.

### Motivation

1. **Scale**: Current skill-based reviews hit context limits around 50-100K tokens. RLM enables 1M+ token monorepo reviews.
2. **Research**: Explore RLM as a technique for code review quality improvement.

### Key Insight from Paper

> "Long prompts should not be fed into the neural network directly but should instead be treated as part of the environment that the LLM can symbolically interact with."

The model writes Python code to explore, filter, and recursively sub-query portions of the codebase—never loading it all into context at once.

---

## Architecture

### Three-Process Model

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOST (Daydream CLI)                      │
│                                                                  │
│  ┌──────────────┐    spawn      ┌─────────────────────────────┐ │
│  │ Runner       │──────────────▶│ devcontainer exec -i        │ │
│  │              │               │ python -u -m daydream.rlm   │ │
│  │              │               └─────────────────────────────┘ │
│  │              │                        │                       │
│  │              │◀───── stdout ──────────┤ (JSON-RPC responses) │
│  │              │────── stdin ──────────▶│ (JSON-RPC requests)  │
│  │              │                        │                       │
│  │  llm_query() │◀────callback request───│                       │
│  │  handler     │────callback response──▶│                       │
│  └──────────────┘                        │                       │
│                                          ▼                       │
│                            ┌─────────────────────────────────┐   │
│                            │ stderr (logging only)           │   │
│                            └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

1. **Host process (Daydream CLI)** — Orchestrates everything, holds Claude SDK client, receives sub-LLM requests via IPC
2. **Devcontainer** — Sandboxed environment via [trailofbits/claude-code-devcontainer](https://github.com/trailofbits/claude-code-devcontainer)
3. **REPL process (inside container)** — Python process with codebase loaded, executes model-generated code

### Why Devcontainer?

- Model-generated Python executes in isolation
- Even malicious code can't escape the container
- Codebase mounted read-only
- `llm_query()` calls route back to host (host controls all API calls)

---

## CLI Interface

```bash
# RLM mode for large codebases
daydream /path/to/monorepo --rlm --python --typescript --go

# With timeout configuration
daydream /path/to/repo --rlm --python --rlm-timeout 600

# PR review mode
daydream /path/to/repo --rlm --python --pr 123
```

---

## REPL Environment API

### Context Metadata (provided at initialization)

```python
repo.total_tokens: int              # Total token count across all files
repo.file_count: int                # Number of files loaded
repo.largest_files: list[str]       # Top 10 files by size
repo.languages: list[str]           # ["python", "typescript", "go"]
```

### Core Data Structures

```python
repo.files: dict[str, str]           # {path: content} - all matching files
repo.structure: dict[str, FileInfo]  # {path: {functions, classes, imports}}
repo.services: dict[str, Service]    # {name: {root, files, dependencies}}
repo.changed_files: list[str]        # From git diff (for PR review mode)
```

### FileInfo Structure (from tree-sitter)

```python
@dataclass
class FileInfo:
    language: str                    # "python" | "typescript" | "go"
    functions: list[str]             # Function/method names
    classes: list[str]               # Class/struct/interface names
    imports: list[str]               # Import statements
    exports: list[str]               # Exported symbols (for TS/Go)
```

### Service Structure

```python
@dataclass
class Service:
    name: str
    root: str                        # e.g., "services/billing"
    files: list[str]                 # All files in this service
    dependencies: list[str]          # Other services this one imports from
```

### Query Functions

```python
def llm_query(prompt: str, model: str = "haiku") -> str:
    """Fresh-context sub-LLM call. Returns response text.

    Each call is stateless—no memory of previous queries.
    Batch information into calls (~100k chars per call).
    """

def llm_query_parallel(prompts: list[str], model: str = "haiku") -> list[str]:
    """Batch multiple independent queries for efficiency."""
```

### Termination Signals

```python
def FINAL(answer: str) -> None:
    """Signal task completion with direct answer."""

def FINAL_VAR(var_name: str) -> None:
    """Signal task completion, returning a REPL variable as output."""
```

### Convenience Helpers

```python
def files_containing(pattern: str) -> list[str]:
    """Grep-like regex search, returns matching paths."""

def files_importing(module: str) -> list[str]:
    """Find files that import a given module."""

def get_file_slice(path: str, start_line: int, end_line: int) -> str:
    """Get specific line range from a file."""
```

### Execution Feedback

- All `print()` output captured and returned to model
- Exceptions include full tracebacks
- Output truncated at 50k chars with message: `"[truncated - use llm_query to analyze large outputs]"`

---

## IPC Protocol

### Transport

- **Mechanism**: JSON-RPC 2.0 over stdin/stdout
- **Framing**: Newline-delimited JSON
- **Encoding**: UTF-8

### Launch Command

```bash
devcontainer exec --container-id $CONTAINER_ID -i \
  python -u -m daydream.rlm.repl
```

- `-i` keeps stdin attached (required)
- `-u` disables Python output buffering
- `--container-id` avoids container mismatch issues

### Message Flow

```
Host (Daydream)                    Devcontainer (REPL)
      │                                   │
      │──── exec request ────────────────▶│
      │     {"jsonrpc":"2.0",             │
      │      "method":"execute",          │
      │      "params":{"code":"..."},     │
      │      "id":1}                      │
      │                                   │
      │◀──── callback request ────────────│  (if llm_query called)
      │      {"jsonrpc":"2.0",            │
      │       "method":"llm_query",       │
      │       "params":{"prompt":"..."},  │
      │       "id":"cb-1"}                │
      │                                   │
      │──── callback response ───────────▶│
      │     {"jsonrpc":"2.0",             │
      │      "id":"cb-1",                 │
      │      "result":"LLM says..."}      │
      │                                   │
      │◀──── exec result ─────────────────│
      │      {"jsonrpc":"2.0",            │
      │       "id":1,                     │
      │       "result":{"output":"...",   │
      │                 "final":null}}    │
```

### Heartbeat

```python
# Host sends every 10 seconds:
{"jsonrpc":"2.0","method":"ping","id":"hb-42"}

# REPL responds within 5 seconds:
{"jsonrpc":"2.0","id":"hb-42","result":"pong"}
```

### Stream Separation

- `stdout` — Protocol messages only (JSON-RPC)
- `stderr` — Logging/debug output (never parsed)

---

## Error Handling

### Timeout Configuration

| Operation | Default Timeout | Configurable |
|-----------|----------------|--------------|
| REPL init | 60s | Yes |
| Code execution | 300s (5 min) | Yes |
| Heartbeat response | 5s | No |
| LLM sub-query | 60s | Yes |
| Container startup | 120s | Yes |

### Error Types

```python
class RLMError(Exception):
    """Base class for RLM errors."""

class REPLCrashError(RLMError):
    """REPL process exited unexpectedly."""

class REPLTimeoutError(RLMError):
    """Code execution exceeded timeout."""

class HeartbeatFailedError(RLMError):
    """REPL stopped responding to heartbeats."""

class ContainerError(RLMError):
    """Devcontainer failed to start or crashed."""
```

### Graceful Degradation

```python
async def run_rlm_review_with_fallback(cwd: Path, languages: list[str]) -> str:
    try:
        return await run_rlm_review(cwd, languages)
    except (REPLCrashError, HeartbeatFailedError, ContainerError) as e:
        console.print(f"[yellow]RLM mode failed: {e}[/yellow]")
        console.print("[yellow]Falling back to standard skill-based review...[/yellow]")
        return await run_standard_review(cwd, languages)
```

---

## Model Configuration

### Model Tiering (from paper)

| Role | Model | Purpose |
|------|-------|---------|
| Root LLM | Opus/Sonnet | Orchestration, code generation, synthesis |
| Sub-LLM | Haiku | Focused analysis on code snippets |

### Sub-LLM Calls

- **Fresh context every time** — No memory between calls
- **Cheap and parallelizable** — Batch aggressively
- **~100k chars per call** — Don't call per-file

---

## System Prompt

The system prompt is implemented in `daydream/prompts/review_system_prompt.py`.

Key elements (aligned with paper):

1. **Iterative framing** — "You will be queried iteratively until you provide a final answer"
2. **Output truncation explanation** — Why to use sub-LLM calls
3. **Probe-first pattern** — Print samples before loading content
4. **Concrete code examples** — Probing, batching, aggregation
5. **Strong batching warning** — "CRITICAL: Never call llm_query in a per-file loop"
6. **Buffer strategy** — "Use variables as buffers to build up your final answer"
7. **FINAL/FINAL_VAR clarification** — Plain text, not in code blocks
8. **Verification pattern** — Confirm critical findings with focused sub-query

---

## Testing Strategy

### Testing Pyramid

```
                    ┌─────────────────┐
                    │   E2E (local)   │  Real container + real LLM
                    │   Manual only   │  Run: make test-e2e-local
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │   Integration (CI + local)  │  Real container + mock LLM
              │   Mock at LLM boundary      │  Run: make test
              └──────────────┬──────────────┘
                             │
    ┌────────────────────────┴────────────────────────┐
    │              Unit tests (CI + local)            │
    │   REPL env, IPC, parsing, prompt generation     │
    └─────────────────────────────────────────────────┘
```

### Mock Boundary

```
┌─────────────────────────────────────────────────────────┐
│                    run_rlm_review()                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │              claude_client.send()  ◄── MOCK HERE  │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                              │
│  ┌───────────────────────▼───────────────────────────┐  │
│  │  Container + REPL + IPC + Environment (all real)  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Test Commands

```makefile
test:              # Unit + integration (CI-safe, mock LLM)
test-e2e-local:    # Real LLM (manual only, not in CI)
test-all:          # Everything including E2E
```

---

## File Structure

```
daydream/
├── daydream/
│   ├── cli.py                    # Add --rlm flag
│   ├── runner.py                 # Add RLM mode branch
│   ├── config.py                 # Add RLM constants
│   │
│   ├── prompts/                  # ✓ Created
│   │   ├── __init__.py
│   │   └── review_system_prompt.py
│   │
│   └── rlm/                      # NEW
│       ├── __init__.py
│       ├── runner.py             # run_rlm_review() orchestration
│       ├── container.py          # Devcontainer management
│       ├── repl.py               # REPL process (runs inside container)
│       ├── ipc.py                # JSON-RPC protocol helpers
│       ├── environment.py        # REPL namespace setup
│       └── parsing.py            # Tree-sitter parsing
│
├── tests/rlm/                    # NEW
│   ├── conftest.py
│   ├── test_environment.py
│   ├── test_ipc.py
│   ├── test_integration.py
│   └── test_e2e_local.py
│
└── .devcontainer/                # NEW (or use trailofbits template)
    ├── devcontainer.json
    └── Dockerfile
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `rlm/runner.py` | Main entry: `run_rlm_review()`, orchestration loop |
| `rlm/container.py` | Start/stop devcontainer, exec, health checks |
| `rlm/repl.py` | Python REPL process (runs inside container) |
| `rlm/ipc.py` | JSON-RPC encoding/decoding |
| `rlm/environment.py` | Build REPL namespace |
| `rlm/parsing.py` | Tree-sitter parsing for `repo.structure` |
| `prompts/review_system_prompt.py` | System prompt generation |

---

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Create `rlm/` module structure
- [ ] Implement `rlm/ipc.py` (JSON-RPC protocol)
- [ ] Implement `rlm/environment.py` (REPL namespace)
- [ ] Implement `rlm/repl.py` (REPL process)
- [ ] Add unit tests for above

### Phase 2: Container Integration
- [ ] Implement `rlm/container.py` (devcontainer management)
- [ ] Configure `.devcontainer/` or integrate trailofbits template
- [ ] Implement heartbeat mechanism
- [ ] Add integration tests with mock LLM

### Phase 3: Orchestration
- [ ] Implement `rlm/runner.py` (main loop)
- [ ] Add `--rlm` flag to CLI
- [ ] Implement graceful fallback
- [ ] Add tree-sitter parsing (`rlm/parsing.py`)

### Phase 4: Polish
- [ ] PR review mode (`--pr` integration)
- [ ] Model-specific prompt tuning (if needed)
- [ ] Local E2E test fixtures
- [ ] Documentation

---

## Open Questions

1. **Service detection**: How to automatically detect service boundaries? Directory conventions? Manifest files?

2. **Recursion depth**: Paper used depth=1 (sub-calls are LMs, not RLMs). Should we support deeper recursion?

3. **Cost tracking**: Should we surface token/cost estimates during RLM review?

4. **Incremental output**: Should findings stream to user as discovered, or only at FINAL?

---

## References

- [Recursive Language Models paper](https://arxiv.org/abs/...) - Zhang et al.
- [trailofbits/claude-code-devcontainer](https://github.com/trailofbits/claude-code-devcontainer) - Sandbox
- [Model Context Protocol](https://modelcontextprotocol.io/) - IPC patterns
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
