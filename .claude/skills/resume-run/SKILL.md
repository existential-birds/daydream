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

Identify the current repo: `git -C <cwd> rev-parse --show-toplevel` (absolute path) and `git rev-parse HEAD` (used in step 3 to detect stale findings via `git diff <run head_sha>..HEAD`), `git rev-parse --abbrev-ref HEAD` (branch name, for display and secondary matching).

List runs newest-first and match. Each run's `manifest.json` records the target repo at `git.source_path` (absolute path) — that is the match field. Secondary signals in the same manifest: `git.repo_slug`, `git.branch`, `git.head_sha`.

```bash
ROOT="${DAYDREAM_ARCHIVE_DIR:-$HOME/.daydream/archive}/runs"
REPO="$(git rev-parse --show-toplevel)"
while IFS= read -r d; do
  src=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]+'manifest.json'))['git']['source_path'])" "$d")
  [ "$src" = "$REPO" ] && echo "$d" && break
done < <(python3 -c "
import os, sys
root = sys.argv[1]
dirs = [os.path.join(root, e) for e in os.listdir(root)
        if os.path.isdir(os.path.join(root, e))]
dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print('\n'.join(dirs))
" "$ROOT")
```

Pick the newest match (prefer one whose `git.head_sha` equals the current HEAD; if the newest match is on a different branch/SHA, say so explicitly — its findings may be stale). If no run matches `source_path`, say "No archived daydream run found for this repo" and stop.

## 2. Read the run state

From the matched run dir read:
- `manifest.json` — `status` (`complete` / `partial` / `failed`), `run.flow`, `run.deep`, `run.backend`, `git.head_sha`, `git.base_branch`, `code_context.base_sha`, `code_context.changed_files`, `pr.number`, `metrics.grounding_rate`, `metrics.coverage_ratio`, `metrics.total_findings`.
- `deep/merged-items.json` — the findings: `items[]` each with `id`, `description`, `file`, `line`, `confidence` (HIGH/MEDIUM/LOW), `severity`, `lens`, `rationale`.
- `deep/recommendation-verdicts.json` — the arbiter's `verdicts[]`: `issue_id`, `verdict` (`consistent` / `inconsistent`), `evidence`, `unverified_assumptions`. A finding with no verdict row was not arbitrated; a finding may appear under more than one `id`/lens (dedupe by file+line+description). **If this file is absent entirely** (the run stopped before the arbitration phase), treat every finding as unarbitrated — proceed to step 3 to cross-check against current code, and in step 5 apply the "unarbitrated but code check confirms the issue still exists" path for each finding.
- `review-output.md` — human-readable finding list (cross-check ids).
- `trajectories/fix-*.json` — one per finding the fix phase touched. The filename is `fix-<slug>.json` where `<slug>` is derived from the target file path (`file_key`) by the pipeline `_safe_descriptor("fix-" + fkey.replace("/", "-").replace("\\", "-"))`, which: (1) prepends `fix-` and replaces `/` and `\` with `-`, (2) lowercases the whole string, (3) replaces every character that is not `[a-z0-9-]` with `-`, (4) collapses consecutive `-` into one, (5) strips leading/trailing `-`. Example: `src/daydream/phases.py` → `fix-src-daydream-phases-py.json`; `My_Module.py` → `fix-my-module-py.json`. **Collision caveat**: paths that differ only in separator-like characters collapse to the same slug — e.g. `src/my.module/file.py` and `src/my-module/file.py` both yield `fix-src-my-module-file-py.json`. When a slug matches multiple candidate paths from the finding list, cross-reference the `items[].file` field in `deep/merged-items.json` against each candidate path to identify which finding(s) the trajectory actually covers; do not assume a unique mapping.

### Tell whether a fix actually ran

One trajectory file covers **all findings for the same target file** (findings are batched by file). A trajectory existing for a file does not mean every finding within that file was resolved — the agent may have addressed some findings and skipped or partially addressed others. Always verify each finding individually against real git state in step 3.

Open each `trajectories/fix-*.json` and inspect `steps` / `final_metrics.total_steps`:
- **Never attempted** — no `fix-*.json` file exists for that finding's file.
- **Dispatched but not run** — the trajectory has only the single `source: "user"` prompt step (`total_steps == 1`, no `source: "assistant"` step, no tool calls). The fixer was given the prompt but produced no edits.
- **Ran (unconfirmed — NOT yet "applied")** — multiple steps including `source: "assistant"` with tool calls (Edit/Write/Bash). **Stop here; do NOT record this as `applied`.** The final classification for each finding MUST come from the git-state check in step 3. After that check, classify as:
  - `applied` — tool calls ran AND `git diff` confirms the fix for this specific finding is present in current code.
  - `not-applied (reverted)` — tool calls ran AND `git diff` shows no net change for this specific finding (edits were reverted within the same trajectory).
  - `ran-not-addressed` — tool calls ran for this file's trajectory but the diff shows no edit was ever attempted for this specific finding (the agent skipped it within the batch). Use this instead of `not-applied (reverted)` when there is no evidence the finding was even touched.

  For files with multiple findings, each finding must be classified independently — the trajectory having run only means the agent executed for that file, not that every finding in the batch was addressed.

## 3. Cross-check against current code

Findings reference `file` + `line` captured at the run's `head_sha`. The repo may have moved since. For each finding: `git diff <run head_sha>..HEAD -- <file>` (and read the current file around the line). If the named line/symbol no longer exists or was already changed, the finding may be stale or already resolved — note it.

## 4. Present the summary

Output a tight status line: matched run id, `status`, flow, head_sha vs current HEAD, phase reached. Then a per-finding disposition table — one row per deduped finding:

| id | file:line | sev/conf | verdict | fix state | current-code check | proposed action |

`fix state` ∈ {applied, not-applied (reverted), ran-not-addressed, dispatched-not-run, never-attempted}. `applied` requires git-state confirmation from step 3 (the trajectory had tool calls AND `git diff` shows the change is present in current code for this specific finding). If tool calls appeared in the trajectory but the diff shows no net change for this finding and the agent did attempt it, use `not-applied (reverted)`. If the trajectory ran tool calls for the file but this specific finding was never touched (no diff evidence it was attempted), use `ran-not-addressed`. Keep it decisive, no hedging.

## 5. Propose dispositions, then confirm

For each finding, recommend ONE action:

**Apply the fix** when: verdict `consistent` (or unarbitrated but the code check confirms the issue still exists), finding still applies to current code, and fix state is `dispatched-not-run`, `ran-not-addressed`, or `never-attempted`.

**Dispose** (state the reason) when any of:
- verdict `inconsistent` — the arbiter found the rationale doesn't hold.
- a finding whose rationale cites only the diff and no codebase evidence — not grounded in the codebase. (Note: if the run's `metrics.grounding_rate == 0.0` or `coverage_ratio == 0.0`, flag this in the step 4 summary as a run-level signal, but evaluate each finding individually — a HIGH-confidence finding independently verified against current code in step 3 is not disposed solely because of a run-aggregate metric.)
- the named file/line/symbol no longer exists or was already fixed since the run's head_sha — no longer applies.
- the fix is too broken to recover (e.g. the finding misreads the contract; an in-code schema/type/comment documents the named behavior as intentional) — disposing is correct.

Present apply-vs-dispose recommendations and WAIT for the user to confirm before editing anything.

## 6. When applying

This repo's `CLAUDE.md` directives are binding:
- Fix at the root cause. Never bypass a hook/test/gate, never `--no-verify`, never paper over.
- Every behavior change ships a real-path test through the production entrypoint (`runner.run` / CLI) with real deps, mocking only the backend — assert observable outcomes, not that a function was called. Unit tests are supplementary.
- Run `make check` (lint + typecheck + tests) green before reporting done. State what you ran. "Committed" ≠ "verified."
- Anchor each edit to the finding's named site; justify any out-of-scope edit. If a finding conflicts with a documented in-code contract, the contract wins.
