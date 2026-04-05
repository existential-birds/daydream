# External Integrations

**Analysis Date:** 2026-04-05

## Agent Backends

**Claude Agent SDK:**
- Purpose: Primary AI backend — streams review, parse, fix, and test interactions
- SDK/Client: `claude-agent-sdk` 0.1.27+, imported as `ClaudeSDKClient`, `ClaudeAgentOptions` in `daydream/backends/claude.py`
- Auth: Handled transparently by the SDK via `~/.claude/` credentials; no env var managed by daydream itself
- Permission mode: `bypassPermissions` — agents run with full file system access
- Setting sources: `["user", "project", "local"]` — reads `~/.claude/settings.json`
- Model: Defaults to `"opus"` (Claude claude-opus-4-6); overridable via `--model` flag

**Codex CLI (OpenAI):**
- Purpose: Alternative AI backend — spawns `codex exec --experimental-json` as subprocess
- Integration: Async subprocess via `asyncio.create_subprocess_exec` in `daydream/backends/codex.py`
- Auth: Handled by Codex CLI itself (not managed by daydream)
- Model: Defaults to `"gpt-5.3-codex"`; overridable via `--model` flag
- Protocol: JSONL events over stdout (stdin receives prompt text)
- Thread resumption: Codex supports multi-turn via `--resume <thread_id>` using `ContinuationToken`

## External CLI Tools

**Git:**
- Purpose: Diffing, branching, reverting, commit detection throughout `daydream/phases.py` and `daydream/runner.py`
- Called via: `subprocess.run(["git", ...])` — hardcoded arguments, no user input interpolated
- Operations used: `git diff`, `git log`, `git branch`, `git checkout .`, `git clean -fd`, `git rev-parse`, `git symbolic-ref`, `git status --porcelain`
- Required: Must be on PATH; target directory must be a git repo

**GitHub CLI (`gh`):**
- Purpose: Auto-detect current PR number in `daydream/cli.py:_auto_detect_pr_number()`
- Called via: `subprocess.run(["gh", "pr", "view", "--json", "number"])`
- Optional: Gracefully skips if `gh` is not installed (`FileNotFoundError` caught)
- Also used indirectly: The `beagle-core:fetch-pr-feedback` and `beagle-core:respond-pr-feedback` Beagle skills likely use `gh` internally (invoked via agent, not directly by daydream)

## Beagle Plugin Skills

Beagle is a Claude Code plugin (`beagle@existential-birds`) that provides specialized review and utility skills. Daydream invokes these via agent prompts formatted by `Backend.format_skill_invocation()`.

**Review Skills** (mapped in `daydream/config.py:REVIEW_SKILLS`):
- `beagle-python:review-python` — Python/FastAPI review
- `beagle-react:review-frontend` — React/TypeScript review
- `beagle-elixir:review-elixir` — Elixir/Phoenix review
- `beagle-go:review-go` — Go backend review
- `beagle-rust:review-rust` — Rust review
- `beagle-ios:review-ios` — iOS/SwiftUI review

**Core Skills** (invoked directly in `daydream/phases.py`):
- `beagle-core:commit-push` — Commit and push changes (used in `phase_commit_push`, `phase_commit_push_auto`, `phase_commit_iteration`)
- `beagle-core:fetch-pr-feedback` — Fetch bot comments from a GitHub PR (used in `phase_fetch_pr_feedback`)
- `beagle-core:respond-pr-feedback` — Post responses to PR comments (used in `phase_respond_pr_feedback`)

**Invocation syntax differs by backend:**
- Claude: `/{namespace:skill} [args]` (e.g., `/beagle-python:review-python`)
- Codex: `$skill-name [args]` (namespace stripped, e.g., `$review-python`)

**Installation requirement:**
- Plugin must be enabled in `~/.claude/settings.json` under `"enabledPlugins": { "beagle@existential-birds": true }`
- Missing skill raises `MissingSkillError` in `daydream/agent.py` (detected via regex on agent text output)

## Data Storage

**Databases:**
- None

**File Storage:**
- Local filesystem only
- `.review-output.md` — review results written to target directory root (configurable via `config.py:REVIEW_OUTPUT_FILE`)
- `.review-debug-<timestamp>.log` — debug log written to target directory when `--debug` is passed
- `.daydream/diff.patch` — git diff written here during `--trust-the-technology` flow (`daydream/runner.py`)
- Temp JSON schema files — written via `tempfile.NamedTemporaryFile` in `daydream/backends/codex.py`, deleted after use

**Caching:**
- None

## Authentication & Identity

**Auth Provider:**
- None managed by daydream
- Claude credentials: handled by `claude-agent-sdk` reading `~/.claude/`
- Codex credentials: handled by `codex` CLI
- GitHub: handled by `gh` CLI (for PR auto-detection and Beagle core skills)

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- Optional debug log file at `<target>/.review-debug-<timestamp>.log` when `--debug` is passed
- Codex backend logs raw JSONL events and warnings via `_raw_log()` in `daydream/backends/codex.py` (writes to debug log)
- All log writes go through `_log_debug()` in `daydream/agent.py`

## CI/CD & Deployment

**Hosting:**
- PyPI distribution (CLI tool installed by end users)
- No server deployment

**CI Pipeline:**
- No detected CI pipeline (no `.github/workflows/`)
- Pre-push hook at `scripts/hooks/pre-push` enforces lint + typecheck + tests locally

## Webhooks & Callbacks

**Incoming:**
- None (CLI tool, no server)

**Outgoing:**
- PR comment responses via `beagle-core:respond-pr-feedback` skill (writes to GitHub PR via `gh` CLI internally)

---

*Integration audit: 2026-04-05*
