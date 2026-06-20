---
name: resume-run
description: Resume the last daydream review run for the current repo. Locates the most recent archived run in ~/.daydream/archive/runs, determines which phase it reached and which findings were fixed / dispatched-but-not-run / never attempted, then proposes applying remaining fixes or disposing of erroneous, stale, or unrecoverable findings. Triggers on "resume the last run", "pick up where daydream stopped", "finish the daydream review", "what's left from the last run", "apply the remaining fixes".
---

# Resume the last daydream run

Pick up the most recent daydream review for THIS repo, figure out where it stopped, and drive the remaining findings to a decision: apply the fix or dispose it with a reason. Make no edits without explicit user confirmation.

## 1. Find the matching run

Resolve the archive root (this exact precedence; do NOT hardcode the home path):
1. `$DAYDREAM_ARCHIVE_DIR` if set, else `~/.daydream/archive`.
2. Runs live in `<root>/runs/<session-uuid>/`.

Identify the current repo: `git -C <cwd> rev-parse --show-toplevel` (absolute path) and `git rev-parse HEAD`, `git rev-parse --abbrev-ref HEAD`.

List runs newest-first and match. Each run's `manifest.json` records the target repo at `git.source_path` (absolute path) — that is the match field. Secondary signals in the same manifest: `git.repo_slug`, `git.branch`, `git.head_sha`.

```bash
ROOT="${DAYDREAM_ARCHIVE_DIR:-$HOME/.daydream/archive}/runs"
REPO="$(git rev-parse --show-toplevel)"
for d in $(ls -dt "$ROOT"/*/); do
  src=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]+'manifest.json'))['git']['source_path'])" "$d" 2>/dev/null)
  [ "$src" = "$REPO" ] && echo "$d" && break
done
```

Pick the newest match (prefer one whose `git.head_sha` equals the current HEAD; if the newest match is on a different branch/SHA, say so explicitly — its findings may be stale). If no run matches `source_path`, say "No archived daydream run found for this repo" and stop.

## 2. Read the run state

From the matched run dir read:
- `manifest.json` — `status` (`complete` / `partial` / `failed`), `run.flow`, `run.deep`, `run.backend`, `git.head_sha`, `git.base_branch`, `code_context.base_sha`, `code_context.changed_files`, `pr.number`, `metrics.grounding_rate`, `metrics.coverage_ratio`, `metrics.total_findings`.
- `deep/merged-items.json` — the findings: `items[]` each with `id`, `description`, `file`, `line`, `confidence` (HIGH/MEDIUM/LOW), `severity`, `lens`, `rationale`.
- `deep/recommendation-verdicts.json` — the arbiter's `verdicts[]`: `issue_id`, `verdict` (`consistent` / `inconsistent`), `evidence`, `unverified_assumptions`. A finding with no verdict row was not arbitrated; a finding may appear under more than one `id`/lens (dedupe by file+line+description).
- `review-output.md` — human-readable finding list (cross-check ids).
- `trajectories/fix-*.json` — one per finding the fix phase touched. The filename is the slugified target file (e.g. `fix-daydream-phases-py.json`).

### Tell whether a fix actually ran

Open each `trajectories/fix-*.json` and inspect `steps` / `final_metrics.total_steps`:
- **Never attempted** — no `fix-*.json` file exists for that finding's file.
- **Dispatched but not run** — the trajectory has only the single `source: "user"` prompt step (`total_steps == 1`, no `source: "assistant"` step, no tool calls). The fixer was given the prompt but produced no edits.
- **Applied** — multiple steps including `source: "assistant"` with tool calls (Edit/Write/Bash). Confirm against real git state below.

## 3. Cross-check against current code

Findings reference `file` + `line` captured at the run's `head_sha`. The repo may have moved since. For each finding: `git diff <run head_sha>..HEAD -- <file>` (and read the current file around the line). If the named line/symbol no longer exists or was already changed, the finding may be stale or already resolved — note it.

## 4. Present the summary

Output a tight status line: matched run id, `status`, flow, head_sha vs current HEAD, phase reached. Then a per-finding disposition table — one row per deduped finding:

| id | file:line | sev/conf | verdict | fix state | current-code check | proposed action |

`fix state` ∈ {applied, dispatched-not-run, never-attempted}. Keep it decisive, no hedging.

## 5. Propose dispositions, then confirm

For each finding, recommend ONE action:

**Apply the fix** when: verdict `consistent` (or unarbitrated but the code check confirms the issue still exists), finding still applies to current code, and fix state is `dispatched-not-run` or `never-attempted`.

**Dispose** (state the reason) when any of:
- verdict `inconsistent` — the arbiter found the rationale doesn't hold.
- `metrics.grounding_rate == 0.0` or `coverage_ratio == 0.0`, or a finding whose rationale cites only the diff — likely a diff-only guess, not grounded in the codebase. Treat with skepticism; verify against real code before acting.
- the named file/line/symbol no longer exists or was already fixed since the run's head_sha — no longer applies.
- the fix is too broken to recover (e.g. the finding misreads the contract; an in-code schema/type/comment documents the named behavior as intentional) — disposing is correct.

Present apply-vs-dispose recommendations and WAIT for the user to confirm before editing anything.

## 6. When applying

This repo's `CLAUDE.md` directives are binding:
- Fix at the root cause. Never bypass a hook/test/gate, never `--no-verify`, never paper over.
- Every behavior change ships a real-path test through the production entrypoint (`runner.run` / CLI) with real deps, mocking only the backend — assert observable outcomes, not that a function was called. Unit tests are supplementary.
- Run `make check` (lint + typecheck + tests) green before reporting done. State what you ran. "Committed" ≠ "verified."
- Anchor each edit to the finding's named site; justify any out-of-scope edit. If a finding conflicts with a documented in-code contract, the contract wins.
