# Technology Stack

**Analysis Date:** 2026-04-05

## Languages

**Primary:**
- Python 3.12 - All source code in `daydream/` package

## Runtime

**Environment:**
- Python 3.12 (requires `>=3.12` per `pyproject.toml`)

**Package Manager:**
- uv (all commands via `uv run`, `uv sync`)
- Lockfile: `uv.lock` present and committed

## Frameworks

**Core:**
- anyio 4.0+ - Async I/O abstraction; `anyio.run()` is the entry point in `daydream/cli.py`, `anyio.create_task_group()` used for parallel fixes in `daydream/phases.py`

**Terminal UI:**
- rich 13.0+ - All terminal output: Live panels, themed console, progress indicators; configured in `daydream/ui.py`
- pyfiglet 1.0+ - ASCII art phase headers rendered via `daydream/ui.py`

**Build:**
- hatchling - Build backend declared in `pyproject.toml` `[build-system]`

## Key Dependencies

**Critical:**
- `claude-agent-sdk` 0.1.27+ - Core Claude integration; `ClaudeSDKClient`, `ClaudeAgentOptions`, and all message types imported in `daydream/backends/claude.py`
- `anyio` 4.0+ - Required for all async execution; used in `daydream/cli.py` (entry), `daydream/phases.py` (task groups, capacity limiting)

**Infrastructure:**
- `rich` 13.0+ - Terminal UI; `Console`, `Live`, panels, Dracula-inspired theme in `daydream/ui.py`
- `pyfiglet` 1.0+ - Phase name banners in `daydream/ui.py`

## Configuration

**Environment:**
- No `.env` file; no environment variable loading library used
- The Claude backend reads model/permission config via `ClaudeAgentOptions` in `daydream/backends/claude.py`
- `setting_sources=["user", "project", "local"]` means Claude reads `~/.claude/settings.json` for API keys and plugin config

**Build:**
- `pyproject.toml` - Single source of truth for project metadata, dependencies, tool config
- `[tool.mypy]` - `python_version = "3.12"`, `ignore_missing_imports = true`
- `[tool.ruff]` - `line-length = 120`, `target-version = "py312"`, rules: `E`, `F`, `I`, `W`
- `[tool.pytest.ini_options]` - `asyncio_mode = "auto"`, `asyncio_default_fixture_loop_scope = "function"`

## Development Tooling

**Linting:**
- ruff 0.9+ - Run via `uv run ruff check daydream` (`make lint`)

**Type Checking:**
- mypy 1.14+ - Run via `uv run mypy daydream` (`make typecheck`)

**Testing:**
- pytest 8.0+ with pytest-asyncio 0.24+ - Run via `uv run pytest -v` (`make test`)

**CI:**
- No CI pipeline file detected (no `.github/workflows/`, no `Makefile` CI target beyond local `make check`)
- Pre-push hook at `scripts/hooks/pre-push` runs lint + typecheck + full test suite before every push

## CLI Entry Point

- Console script: `daydream = "daydream.cli:main"` declared in `pyproject.toml`
- Can also run as module: `python -m daydream`

## Platform Requirements

**Development:**
- Python 3.12+, uv package manager
- Requires Beagle plugin installed in Claude Code (`~/.claude/settings.json`)
- `codex` CLI must be on PATH when using the Codex backend

**Production:**
- No server deployment; purely a CLI tool executed locally

---

*Stack analysis: 2026-04-05*
