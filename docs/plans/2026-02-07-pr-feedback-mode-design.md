# PR Feedback Mode

Automated loop that fetches PR review comments from GitHub, applies fixes in parallel, commits, pushes, and responds to the reviewer.

## CLI Interface

Triggered by `--pr` combined with `--bot`:

```
daydream /path/to/project --pr 123 --bot coderabbitai[bot]
daydream --pr 123 --bot github-actions[bot]    # target from cwd
daydream --pr --bot coderabbitai[bot]           # auto-detect PR from branch
```

Argument rules:

- `--pr` accepts an optional integer. If omitted, auto-detect from current branch via `gh pr view --json number`.
- `--bot` is required when `--pr` is used. No default value.
- `--pr` is mutually exclusive with `--review-only`, `--start-at`, and skill flags (`--python`, `--typescript`, `--elixir`).
- `--model` and `--debug` work as normal.

`RunConfig` gains two fields:

```python
pr_number: int | None = None
bot: str | None = None
```

## Execution Flow

```
phase_fetch_pr_feedback()
        |
phase_parse_feedback()          # existing, reused as-is
        |
phase_fix_parallel()            # concurrent agents, one per issue
        |
phase_commit_push_auto()        # no user prompt
        |
phase_respond_pr_feedback()     # reply + resolve threads
```

The `run()` function in `runner.py` branches early: if `config.pr_number` is set, call `run_pr_feedback()` instead of the existing phase sequence.

## New Phases

### phase_fetch_pr_feedback(target, pr_number, bot)

Invokes `run_agent()` with:

```
/beagle-core:fetch-pr-feedback --pr {pr_number} --bot {bot}
```

The skill fetches issue comments and line-specific review comments from GitHub, formats them as markdown, and internally calls `receive-feedback` to evaluate each item. Agent writes output to `.review-output.md` (reuses existing file convention).

### phase_parse_feedback()

Existing function, reused as-is. Reads `.review-output.md` and extracts structured JSON:

```json
{
  "issues": [
    {"id": 1, "description": "...", "file": "...", "line": 42}
  ]
}
```

### phase_fix_parallel(target, feedback_items)

Launches one `run_agent()` call per feedback item concurrently using `anyio.create_task_group()`. Each agent gets the same prompt format as the existing sequential `phase_fix()`.

Collects results as a list of `(item, success: bool, error: str | None)` tuples.

On completion:

- Prints a summary of which fixes succeeded and which failed.
- Alerts the user with a warning panel for each failed fix.
- Returns the results list.

If all fixes fail, the run aborts before commit/push/respond.

### phase_commit_push_auto(target)

No user prompt. Runs `run_agent()` with `/beagle-core:commit-push` directly. Only called if at least one fix succeeded.

### phase_respond_pr_feedback(target, pr_number, bot, results)

Filters to only successful fixes. Invokes `run_agent()` with:

```
/beagle-core:respond-pr-feedback --pr {pr_number} --bot {bot}
```

The skill handles posting replies and resolving threads automatically. Comments for failed fixes are left unanswered.

## Parallel Agent Execution

### Concurrency

Each parallel `run_agent()` call gets its own `AgentTextRenderer` and `LiveToolPanelRegistry` (already local variables). The `_current_client` singleton becomes `_current_clients: list[ClaudeSDKClient]` so the shutdown handler can terminate all running agents.

Each task in the `anyio.TaskGroup` catches exceptions individually and records failure in the results list. A failed agent does not cancel the others.

### Progress UI: ParallelFixPanel

A Rich `Live` display with a `Table` showing one row per agent:

```
┌─ Fixing 5 issues ─────────────────────────────────────────────┐
│  ◜ fix-1 │ src/auth.py:42     │ Reading file contents...      │
│  ◜ fix-2 │ src/api/routes.py  │ Applying edit to line 87      │
│  ✓ fix-3 │ src/models.py:15   │ Complete                      │
│  ◜ fix-4 │ lib/utils.py:203   │ Analyzing issue context...    │
│  ✗ fix-5 │ src/config.py:8    │ Failed: could not locate      │
└───────────────────────────────────────────────────────────────-┘
```

Each row shows:

- **Spinner/status icon** — animated while running, checkmark on success, X on failure.
- **Label** — `fix-N`.
- **File:line** — from the feedback item.
- **Latest output** — last meaningful line from the agent stream, truncated to terminal width.

Row updates come from different agent event types:

- `TextBlock`: last non-empty line of text.
- `ToolUseBlock`: tool name and first arg (e.g., "Edit src/auth.py").
- `ToolResultBlock`: brief "tool complete".

### Agent Callback

`run_agent()` gains an optional `progress_callback` parameter:

```python
async def run_agent(
    cwd: Path,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> str | Any
```

When `progress_callback` is set, the agent runs in quiet mode and routes status updates through the callback instead of printing to the console. All output still goes to the debug log.

## Error Handling

- If some fix agents fail: continue with partial success. Commit and push successful fixes, respond only to comments where fixes were applied, leave failed ones unanswered. Alert the user about failures.
- If all fix agents fail: abort the run before commit/push/respond.
- If commit/push fails: abort before responding.
- If respond fails: alert the user but exit successfully (fixes are already pushed).

## File Changes

| File | Change |
|------|--------|
| `cli.py` | Add `--pr` and `--bot` arguments, mutual exclusion rules, validation |
| `runner.py` | Add `pr_number`/`bot` to `RunConfig`, add `run_pr_feedback()` orchestrator |
| `phases.py` | Add `phase_fetch_pr_feedback()`, `phase_fix_parallel()`, `phase_commit_push_auto()`, `phase_respond_pr_feedback()` |
| `agent.py` | Add `progress_callback` to `run_agent()`, change `_current_client` to `_current_clients` list |
| `ui.py` | Add `ParallelFixPanel` class |
| `config.py` | No changes |

No new files. Estimated ~250-350 new lines across these files.
