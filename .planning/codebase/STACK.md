# Technology Stack

**Analysis Date:** 2026-04-26

## Languages

**Primary:**
- Python 3.12 - All source code in `daydream/` package; `requires-python = ">=3.12"` declared in `pyproject.toml`

**Secondary:**
- None. This is a pure Python CLI tool.

## Runtime

**Environment:**
- Python 3.12 (minimum required per `pyproject.toml`)
- No `.python-version` file present; version pinned solely via `pyproject.toml`

**Package Manager:**
- uv — all commands run via `uv run`, `uv sync`
- Lockfile: `uv.lock` present and committed (revision 3)

## Frameworks

**Async I/O:**
- anyio 4.12.1 — `anyio.run()` is the process entry point in `daydream/cli.py`; `anyio.create_task_group()` and `anyio.CapacityLimiter` used for parallel fix invocations in `daydream/phases.py`

**Terminal UI:**
- rich 14.3.1 — All terminal output; `Live` panels, themed `Console`, progress indicators, Dracula-inspired theme configured in `daydream/ui.py`

**ASCII Art:**
- pyfiglet 1.0.4 — Phase name banners rendered via `daydream/ui.py`

**Static Code Analysis:**
- tree-sitter 0.25.2 — Core parsing engine used in `daydream/tree_sitter_index.py` for import resolution
- tree-sitter-python 0.25.0 — Python language grammar
- tree-sitter-typescript 0.23.2 — TypeScript and TSX grammars
- tree-sitter-go 0.25.0 — Go language grammar
- tree-sitter-rust 0.24.2 — Rust language grammar

**Build:**
- hatchling — Build backend declared in `pyproject.toml` `[build-system]`

**Testing:**
- pytest 9.0.3 with pytest-asyncio 1.3.0 — Run via `uv run pytest -v` (`make test`)
- `asyncio_mode = "auto"` configured in `pyproject.toml` `[tool.pytest.ini_options]`

## Key Dependencies

**Critical:**
- `claude-agent-sdk` 0.1.52 — Core Claude integration; `ClaudeSDKClient`, `ClaudeAgentOptions`, and all message types imported in `daydream/backends/claude.py`. Pinned to exact version in `pyproject.toml` (`==0.1.52`).
- `mcp` 1.26.0 — Pulled in transitively by `claude-agent-sdk`; MCP tool call events handled in `daydream/backends/codex.py`
- `anyio` 4.12.1 — Required for all async execution; entry point and task orchestration
- `rich` 14.3.1 — All user-facing terminal output; 3295-line `daydream/ui.py` depends entirely on it

**Infrastructure:**
- `httpx` 0.28.1 — HTTP client pulled in transitively by `claude-agent-sdk`/`mcp`
- `pydantic` — Pulled in transitively; used internally by `claude-agent-sdk` and `mcp`
- `python-dotenv` — Transitively included; not explicitly used in daydream source code
- `jsonschema` — Transitively included; JSON Schema validation used in claude-agent-sdk internals

## Configuration

**Environment:**
- No `.env` file; no direct environment variable loading in daydream source code
- Claude backend reads API keys from `~/.claude/settings.json` via `setting_sources=["user"]` in `ClaudeAgentOptions` (`daydream/backends/claude.py`, line 78)
- `CLAUDE_CONFIG_DIR` environment variable can override the default `~/.claude` directory (used in `daydream/deep/orchestrator.py`)
- Codex backend requires `codex` CLI on `$PATH`; credentials managed by the Codex CLI itself

**Build:**
- `pyproject.toml` — Single source of truth: project metadata, dependencies, tool configuration
- `[tool.mypy]` — `python_version = "3.12"`, `ignore_missing_imports = true`
- `[tool.ruff]` — `line-length = 120`, `target-version = "py312"`, rules: `E`, `F`, `I`, `W`
- `[tool.pytest.ini_options]` — `asyncio_mode = "auto"`, `asyncio_default_fixture_loop_scope = "function"`

## Platform Requirements

**Development:**
- Python 3.12+, uv package manager
- Beagle plugin installed in Claude Code (`~/.claude/settings.json`) — required for `beagle-*:review-*` skills
- `gh` (GitHub CLI) on `$PATH` — required for PR feedback and PR posting features
- `git` on `$PATH` — required for all diff and commit operations
- `codex` CLI on `$PATH` — required only when using `--backend codex`
- Pre-push hook at `scripts/hooks/pre-push` runs lint + typecheck + full test suite before every push

**Production:**
- No server deployment; purely a CLI tool executed locally
- Console script entrypoint: `daydream = "daydream.cli:main"` declared in `pyproject.toml`
- Can also run as module: `python -m daydream` via `daydream/__main__.py`

---

*Stack analysis: 2026-04-26*
