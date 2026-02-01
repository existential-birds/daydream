# Daydream Repository Restructure Design

## Goal

Transform two Python scripts (`review_fix_loop.py`, `review_ui.py`) into a standalone CLI tool installable via `uv tool install`.

## Target User Experience

```bash
# Install
uvx daydream              # Run without installing
uv tool install daydream  # Install globally

# Usage
daydream                  # Interactive mode (current behavior)
```

## Repository Structure

```
daydream/
├── pyproject.toml              # Package config with [project.scripts]
├── README.md                   # Installation + usage docs
├── daydream/
│   ├── __init__.py             # Version + public exports
│   ├── __main__.py             # Entry: `python -m daydream`
│   ├── cli.py                  # CLI argument parsing + main()
│   ├── config.py               # Settings: REVIEW_SKILLS, defaults
│   ├── runner.py               # Core orchestration (main loop)
│   ├── phases.py               # Phase functions: review, parse, fix, test
│   ├── agent.py                # run_agent() + SDK interaction
│   └── ui.py                   # All UI components (from review_ui.py)
└── tests/                      # Optional: basic test stubs
```

## File Breakdown

### From `review_fix_loop.py` (~690 lines)

| Section | Target File | Contents |
|---------|-------------|----------|
| Lines 1-9 | `pyproject.toml` | Script metadata → project dependencies |
| Lines 24-63 | `cli.py` | Imports, console creation |
| Lines 69-76 | `config.py` | `REVIEW_SKILLS`, `REVIEW_OUTPUT_FILE` |
| Lines 79-127 | `agent.py` | `MissingSkillError`, signal handlers, globals |
| Lines 135-293 | `agent.py` | `_detect_test_success()`, `run_agent()`, `extract_json_from_output()` |
| Lines 343-528 | `phases.py` | `phase_review()`, `phase_parse_feedback()`, `phase_fix()`, `phase_test_and_heal()`, `phase_commit_push()` |
| Lines 531-551 | `cli.py` | `_print_missing_skill_error()` |
| Lines 555-689 | `runner.py` | `main()` orchestration logic |

### From `review_ui.py` (~2178 lines)

Kept as single `ui.py` file with all components:
- Color theme (Dracula-based)
- NeonConsole class
- Header/Phase/Menu components
- Tool call/result rendering
- AgentTextRenderer
- LiveToolPanel + Registry
- Progress indicators

## pyproject.toml

```toml
[project]
name = "daydream"
version = "0.1.0"
description = "Automated code review and fix loop using Claude"
requires-python = ">=3.10"
dependencies = [
    "claude-agent-sdk>=0.1.27",
    "anyio>=4.0",
    "rich>=13.0",
    "pyfiglet>=1.0",
]

[project.scripts]
daydream = "daydream.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Module Responsibilities

### `cli.py`
- Argument parsing (future: `--dir`, `--skill` flags)
- Signal handler installation
- Exception handling wrapper
- Entry point `main()`

### `config.py`
- `REVIEW_SKILLS` mapping
- `REVIEW_OUTPUT_FILE` constant
- Any future configuration

### `agent.py`
- `MissingSkillError` exception
- `run_agent()` - SDK client interaction
- `extract_json_from_output()` - JSON parsing
- `_detect_test_success()` - test result detection

### `phases.py`
- `phase_review()` - run review skill
- `phase_parse_feedback()` - extract issues
- `phase_fix()` - apply single fix
- `phase_test_and_heal()` - run tests with retry
- `phase_commit_push()` - commit workflow

### `runner.py`
- Main orchestration loop
- User prompts for configuration
- Phase sequencing
- Summary generation

### `ui.py`
- All Rich-based UI components
- Theme definitions
- Live panel management

## Migration Notes

1. Remove inline script metadata from Python files
2. Update all imports to use package structure
3. Move global state (`_current_client`, `_debug_log`, `_quiet_mode`) to appropriate modules
4. Keep `ui.py` as-is, just update import path
