# External Integrations

**Analysis Date:** 2026-04-26

## APIs & External Services

**Claude Agent SDK (Anthropic):**
- Used for all primary AI agent interactions — code review, fix application, intent analysis, plan generation
- SDK: `claude-agent-sdk==0.1.52` (PyPI)
- Client: `ClaudeSDKClient` from `claude_agent_sdk`, wrapped in `daydream/backends/claude.py`
- Auth: Claude Code reads credentials from `~/.claude/settings.json` via `setting_sources=["user"]`
- Permission mode: `bypassPermissions` — agents run with full tool access
- Tools granted: `["Read", "Write", "Edit", "Bash", "Glob", "Grep"]`
- Models supported: `"opus"` (default), `"sonnet"`, `"haiku"` — passed via `ClaudeAgentOptions.model`

**Codex CLI (OpenAI):**
- Optional alternative backend for agent execution
- Client: External `codex` subprocess, not a Python SDK; spawned via `asyncio.create_subprocess_exec` in `daydream/backends/codex.py`
- Auth: Managed by the `codex` CLI binary itself (not daydream)
- Event format: JSONL events over stdout (`thread.started`, `item.started`, `item.completed`, `turn.completed`, `turn.failed`)
- Default model: `"gpt-5.3-codex"` (passed via `--model` flag)
- Limitation: Does not support exploration subagents; raises `NotImplementedError` if agents are provided

**Beagle Plugin (Claude Code Plugin):**
- External plugin installed in Claude Code (`~/.claude/settings.json`)
- Provides review skills: `beagle-python:review-python`, `beagle-react:review-frontend`, `beagle-elixir:review-elixir`, `beagle-go:review-go`, `beagle-rust:review-rust`, `beagle-ios:review-ios`
- Skill mapping in `daydream/config.py` (`REVIEW_SKILLS`, `SKILL_MAP`)
- Plugin registry detected at `~/.claude/plugins/installed_plugins.json` (or `$CLAUDE_CONFIG_DIR/plugins/installed_plugins.json`)
- Detection logic in `daydream/deep/orchestrator.py:get_installed_skills()`
- Install command: `/install-plugin beagle@existential-birds` in Claude Code

## Data Storage

**Databases:**
- None. No database integration of any kind.

**File Storage:**
- Local filesystem only
- Review output written to `{target_dir}/.review-output.md` (constant `REVIEW_OUTPUT_FILE` in `daydream/config.py`)
- Deep-review artifacts written to `{target_dir}/.daydream/deep/` (paths defined in `daydream/deep/artifacts.py`)
- Debug logs written to timestamped files in target directory when `--debug` is enabled
- No cloud file storage, no S3, no object storage

**Caching:**
- None. No caching layer.

## Authentication & Identity

**Auth Provider:**
- No user authentication. Daydream is a local CLI tool with no user accounts.
- Claude credentials: Delegated entirely to Claude Code's `~/.claude/settings.json`
- GitHub credentials: Delegated entirely to the `gh` CLI binary's auth state

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, DataDog, or error reporting service.

**Logs:**
- Optional debug log written to a timestamped `.log` file in the target directory when `--debug` flag is passed
- Log format: prefixed entries (`[PROMPT]`, `[TEXT]`, `[TOOL_USE]`, `[TOOL_RESULT]`, `[COST]`, `[SCHEMA_OK]`, `[WARN]`, `[THINKING]`)
- Written via `_log_debug()` in `daydream/agent.py`; file handle stored in `AgentState.debug_log`
- All user-facing output goes to Rich `Console` via `daydream/ui.py` helpers (no stdout/stderr print statements)

## CI/CD & Deployment

**Hosting:**
- No server deployment. CLI tool distributed via PyPI or direct install.

**CI Pipeline:**
- GitHub Actions at `.github/workflows/ci.yml`
- Runs on push/PR to `main`/`master` branches
- Steps: checkout → `astral-sh/setup-uv@v4` → Python 3.12 → `uv sync` → ruff lint → mypy typecheck → pytest
- Uses uv cache via `enable-cache: true`

**Pre-push Hook:**
- Local gate at `scripts/hooks/pre-push`
- Installed via `make hooks` (symlinks to `.git/hooks/pre-push`)
- Runs: ruff lint → mypy typecheck → full pytest suite

## GitHub Integration

**How GitHub is accessed:**
- Via the `gh` GitHub CLI binary (subprocess calls), never via direct HTTP/REST API calls from Python
- Used in `daydream/pr_review.py` for PR listing, diff fetching, and posting reviews
- Used in `daydream/cli.py` for auto-detecting the current PR number (`gh pr view --json number`)
- Used in `daydream/phases.py` for fetching PR comments via a Beagle skill invocation

**Operations performed:**
- `gh pr list --head <branch>` — locate the open PR for the current branch
- `gh pr diff <num>` — fetch PR diff for line resolution
- `gh api POST /repos/<owner>/<repo>/pulls/<num>/reviews` — post review comments (inline + body)
- `gh pr view --json number` — auto-detect PR number from current branch

**Auth:** Delegated to `gh` CLI auth state; no tokens stored or accessed by daydream

**Error handling:** All `gh` subprocess calls wrapped in `try/except (subprocess.SubprocessError, OSError):`; failures warn and return, never raise

## Webhooks & Callbacks

**Incoming:**
- None. Daydream does not expose any HTTP endpoints.

**Outgoing:**
- None directly. GitHub review posting is done via `gh api` subprocess call, not an HTTP client.

## Environment Configuration

**Required for Claude backend (default):**
- `~/.claude/settings.json` must be present with valid Anthropic API credentials (managed by Claude Code)
- Beagle plugin must be installed in Claude Code for review skills to function

**Required for Codex backend (`--backend codex`):**
- `codex` CLI binary must be on `$PATH`
- Codex CLI handles its own auth (OpenAI API key)

**Required for PR features (`--pr`, `--post-review`):**
- `gh` CLI binary must be on `$PATH` and authenticated (`gh auth login`)

**Optional environment variables:**
- `CLAUDE_CONFIG_DIR` — Override the default `~/.claude` directory for Claude Code config (used in `daydream/deep/orchestrator.py`)

**No `.env` file:** Daydream does not load environment variables from any file. All configuration is via CLI flags, `~/.claude/settings.json`, and external tool auth.

---

*Integration audit: 2026-04-26*
