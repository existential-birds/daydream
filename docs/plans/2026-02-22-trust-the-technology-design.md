# Design: `--trust-the-technology` Flag

**Date:** 2026-02-22
**Status:** Approved

## Overview

A new review flow triggered by `--trust-the-technology` (short: `--ttt`) that works with any technology stack. Instead of invoking Beagle review skills, it uses a three-phase conversational approach: understand the PR's intent, evaluate whether there's a better way, and generate an implementation plan for selected issues.

## Architecture

Three new phase functions + a `run_trust()` orchestrator, following the existing phase pattern.

```
cli.py (--trust-the-technology flag)
  → runner.py (run_trust() orchestrator)
    → phase_understand_intent()   — Agent 1: explore + present understanding
    → phase_alternative_review()  — Agent 2: evaluate + return numbered issues
    → phase_generate_plan()       — Agent 3: plan for selected issues → .daydream/
```

All phases use the standard `Backend.execute()` protocol — no skill invocations needed. Works with Claude, Codex, or any future backend.

## CLI & Config

**New flag:** `--trust-the-technology` / `--ttt` — boolean.

**RunConfig addition:**
```python
trust_the_technology: bool = False
```

**Mutual exclusions:**
- Excludes: `--python`, `--typescript`, `--elixir`, `--go`, `--skill`, `--review-only`, `--loop`, `--pr`
- Compatible with: `--backend`, `--model`, `--debug`, `--cleanup`

**Runner branching:** If `config.trust_the_technology` is True, call `run_trust(config)` and return early.

## Phase 1: Understand Intent (`phase_understand_intent`)

**Phase hero:** LISTEN

**Signature:**
```python
async def phase_understand_intent(
    backend: Backend,
    cwd: Path,
    diff: str,
    log: str,
    branch: str,
) -> str  # confirmed intent summary
```

**Behavior:**
1. Gather git context: `git diff main...HEAD`, `git log main..HEAD --oneline`, branch name
2. Agent prompt includes diff, log, branch name, and instructions to explore the codebase freely
3. Agent presents its understanding of the PR's intent
4. User confirms (`y`) or provides correction (any other input)
5. On correction: new agent invocation with original context + user's correction
6. Loop until user confirms
7. Returns the confirmed intent summary as a string

**Agent prompt core:**
> You have full access to explore the codebase. Examine the diff below and the codebase to understand the intent of these changes. Present your understanding concisely — what problem is being solved and how.

## Phase 2: Alternative Review (`phase_alternative_review`)

**Phase hero:** WONDER

**Signature:**
```python
async def phase_alternative_review(
    backend: Backend,
    cwd: Path,
    diff: str,
    intent_summary: str,
) -> list[dict]  # numbered issues
```

**Behavior:**
1. Fresh agent receives: confirmed intent summary + diff
2. Agent explores codebase and evaluates the implementation
3. Returns numbered issues via structured output

**Structured output schema:**
```json
{
  "issues": [
    {
      "id": 1,
      "title": "Brief title",
      "description": "What's wrong or could be better",
      "recommendation": "What you'd do instead",
      "severity": "high|medium|low",
      "files": ["path/to/relevant/file.py"]
    }
  ]
}
```

**Display:** Issues rendered as a numbered Rich table (ID, severity, title) with full details below.

**Agent prompt core:**
> Given this confirmed intent, explore the codebase and evaluate the implementation. Would you have done this differently? Return a numbered list of issues covering both architectural alternatives and incremental improvements.

## Phase 3: Generate Plan (`phase_generate_plan`)

**Phase hero:** ENVISION

**Signature:**
```python
async def phase_generate_plan(
    backend: Backend,
    cwd: Path,
    diff: str,
    intent_summary: str,
    issues: list[dict],
) -> Path | None  # path to plan file, or None if skipped
```

**Behavior:**
1. Display issues and prompt user: "Create an implementation plan? Enter issue numbers (e.g., 1,3,5) or 'all', or 'none' to skip:"
2. Parse selection — `none`/empty skips, `all` selects all, numbers select specific issues
3. Agent receives: intent summary + selected issues (full detail) + diff
4. Agent generates detailed implementation plan via structured output
5. Write plan as markdown to `.daydream/plan-{YYYY-MM-DD-HHmmss}.md`
6. Create `.daydream/` directory if it doesn't exist

**Structured output schema:**
```json
{
  "plan": {
    "summary": "Overall plan summary",
    "issues": [
      {
        "id": 1,
        "title": "Issue title",
        "changes": [
          {
            "file": "path/to/file.py",
            "description": "What to change and why",
            "action": "modify|create|delete"
          }
        ]
      }
    ]
  }
}
```

**Markdown output format:**
```markdown
# Implementation Plan
**Generated:** {timestamp}
**Branch:** {branch_name}

## Intent
{intent_summary}

## Plan Summary
{plan.summary}

## Issue {id}: {title}
**Severity:** {severity}
**Problem:** {description}
**Recommendation:** {recommendation}

### Changes
- **{action}** `{file}` — {description}
```

**Agent prompt core:**
> Create a detailed implementation plan for fixing these issues. For each issue, specify what files to change, what the change should be, and why. Make this actionable enough to hand to another developer or agent.

## Orchestration (`run_trust`)

```python
async def run_trust(config: RunConfig) -> int:
    target_dir = resolve_target(config)
    backend = _resolve_backend(config, "review")

    # Gather git context
    diff = git_diff(target_dir)
    log = git_log(target_dir)
    branch = git_branch(target_dir)

    # Phase 1: Understand intent
    intent_summary = await phase_understand_intent(backend, target_dir, diff, log, branch)

    # Phase 2: Alternative review
    issues = await phase_alternative_review(backend, target_dir, diff, intent_summary)

    if not issues:
        # No issues — implementation looks good
        return 0

    # Phase 3: Generate plan
    plan_path = await phase_generate_plan(backend, target_dir, diff, intent_summary, issues)

    if plan_path:
        print(f"Plan written to {plan_path}")

    return 0
```

## Error Handling

- Agent failures: print error, return exit code 1
- Shutdown signals: existing `ShutdownPanel` + `backend.cancel()` — no changes needed
- Empty diff: warn user and exit early
- No `.daydream/` directory: create it automatically

## Testing Strategy

- Mock backend returns predefined structured output for each phase
- Test intent confirmation loop (confirm on first try, correct then confirm)
- Test issue selection parsing (`all`, `none`, `1,3,5`, invalid input)
- Test plan markdown generation
- Test `.daydream/` directory creation
- Integration test: full `run_trust()` flow with mock backend
