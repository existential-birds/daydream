# Daydream: Worktree Isolation + Git Ops Refactor — Execution Plan

**Status:** Locked. Ready for execution in a fresh session.
**Date:** 2026-04-30
**Branch to create:** `feat/worktree-isolation` off `main`
**Closes:** GitHub issue #44 + multi-worktree silent-failure bug

---

## How to use this document

This is a self-contained orchestrator prompt. Paste it into a fresh Claude Code session in this repo and it will run end-to-end. It is structured so the main session **orchestrates only** — every implementation task fans out to a focused sub-agent that verifies its own work before reporting back.

The "Locked design decisions" section is settled. Do not relitigate.

---

## Your role: orchestrator only

You are the orchestrator for a multi-stage refactor. **You will not write code yourself.** You spawn sub-agents (via the Agent tool) for every implementation task, read their reports, decide the next stage, and continue until the whole job is complete.

Repo: `/Users/ka/github/existential-birds/daydream`
Branch: create `feat/worktree-isolation` off `main` before starting.
Run to completion. Do not stop unless the failure-recovery protocol below tells you to.

---

## Mission

Three-in-one refactor:

1. **Close GitHub issue #44** — unify all git/gh subprocess calls behind a single `daydream/git_ops.py` module so every call has explicit `repo: Path`, consistent error handling, and pre-flight worktree validation. Eliminates the "fatal: not a git repository" confabulation when daydream is run from an org-level directory.

2. **Solve the multi-worktree silent-failure bug** — daydream today runs `git diff origin/HEAD...HEAD` without verifying HEAD isn't the base branch. From a worktree with `main` checked out, this produces empty diffs and confusing "nothing to review" messages. Add a workspace abstraction with ephemeral worktree support so `daydream --branch feat/X` always reviews the server's version of feat/X regardless of local state.

3. **Consolidate the CLI** — replace `--python`/`--typescript`/`--ttt`/`--pr`/`--deep` with a single mental model: target = branch, output mode = `--comment | --review | (default loop)`, multi-stack (today's `--deep`) is the default. Old flags get one-release deprecation warnings.

---

## Locked design decisions

### CLI surface

```
daydream [--branch X] [--base Y] [<output-mode>] [<modifiers>]

Output modes (mutually exclusive, default is fix-loop):
  (none)           review + fix + test loop
  --comment        review + post inline PR comments + exit
  --review         review + terminal/markdown output + exit

Selection:
  --branch X       what to review (default: cwd's local HEAD)
  --base Y         what to compare against
                   default resolution order:
                     1. if --branch X has open PR → PR's base ref
                     2. origin/HEAD
                     3. main, master (local fallback)

Modifiers:
  --worktree       force ephemeral even when --branch is omitted
  --shallow        single-stack (skip multi-stack auto-detection)
  --copy <path>    add path to ephemeral copy list (repeatable)

Subcommand:
  daydream feedback <pr#>   ingest PR bot comments and fix (today's --pr)

Deprecated (one-release warning, then removed):
  --python, --typescript, --rust, --go, --elixir, --ruby, --java, --csharp
  --ttt, --pr <n>, --deep
  --review-only (renamed to --review)
```

### Worktree behavior

| Invocation | Worktree | Base default |
|---|---|---|
| `daydream` | In-place at cwd. | Auto (origin/HEAD → main → master). |
| `daydream --worktree` | Ephemeral at cwd's local HEAD. | Same. |
| `daydream --branch X` | **Always ephemeral**, detached at `origin/X` post-fetch. Even if X is currently checked out in cwd (warn). | PR's base if PR exists for X, else auto. |
| `daydream --comment` | Same rules as the row matching the other flags present. | Same. `--comment` always skips test phase. |
| `daydream feedback <pr#>` | Determine PR's branch; ephemeral if branch != current. | PR's base. |

Ephemeral worktrees live at `<repo>/.daydream/worktrees/<run-id>/` (run-id is `<UTC YYYYMMDDHHMMSS>-<hex>`). Always `git fetch origin` first. Cleanup with `git worktree remove --force` on exit, even on crash. Add `.daydream/worktrees/` to the project's `.gitignore` if missing.

### Branch-not-found semantics

`--branch feat/X` where neither `refs/heads/feat/X` nor `refs/remotes/origin/feat/X` exists post-fetch → error `"branch 'feat/X' not found locally or on origin. Push the branch or check the name."` Do not probe forks via `gh`.

### Stale-local handling

`--branch X` and X is currently checked out in cwd → ephemeral runs anyway; warn `"feat/X is checked out in cwd and is N commits behind origin/feat/X — reviewing origin/feat/X."` (or "0 commits" if up to date).

### Wrong-branch loud failure

When running default loop mode with no `--branch` and `current_branch == base_branch`, raise `WrongBranchError` with actionable text. This must NOT fire for `--comment` or `--review` (you might legitimately want a no-op review report from the base branch).

### `.env` copy mechanism

Only fires for ephemeral runs that execute the test phase (i.e., not `--comment`, not `--review`).

Three layers:

1. **Default copy list** — files matching `.env`, `.env.local`, `.env.*` if gitignored in the source worktree.
2. **`[tool.daydream.workspace] copy = [...]`** in target repo's `pyproject.toml` overrides the default list.
3. **`--copy <path>`** flag — additive, repeatable; combines with whichever of (1) or (2) is in effect.

Copy, never symlink. Files copied from the source worktree (the dir daydream was invoked from) into the ephemeral worktree before any agent runs.

### Base detection algorithm (port from codex)

Reference: `/Users/ka/github/reference_agents/codex/codex-rs/git-utils/src/branch.rs:15-48`. Port `merge_base_with_head` semantics into Python:

1. Resolve `repo_root` and `HEAD`.
2. Resolve `branch_ref` for the requested base (e.g., `main`).
3. Check if `<branch>@{upstream}` exists and if `git rev-list --left-right --count HEAD...upstream` shows the upstream is **ahead** (right count > 0). If so, prefer the upstream ref for the merge-base.
4. `git merge-base HEAD <preferred_ref>` and return the SHA.
5. Return `None` cleanly on missing HEAD, missing branch, or empty repo — caller decides how to surface.

### Refs-not-diffs to skills

Per project memory rule: `phase_commit_push`, `phase_commit_push_auto`, deep-mode commit, and `phase_commit_iteration` pass `--repo <path> --base <sha> --intent <json-path>` to `beagle-core:commit-push`. Skill re-derives diff from `--base..HEAD`. Old skill ignores unknown args, so daydream-side change can land first.

---

## Sub-agent orchestration protocol

### Spawning rules

- Every implementation task uses the Agent tool with `subagent_type="general-purpose"` unless a specialized agent fits better.
- Pass paths and self-contained context. Never paste file contents into agent prompts.
- End every agent prompt with: **"Verify your work by running the commands listed in the verification block. Do not return success unless every command in that block exits 0. Return a report of ≤200 words: what you changed, what verification you ran, and any deviation from the spec."**
- If an agent returns failure or runs out of context, do **not** finish its work yourself. Read its report, narrow the scope (e.g. split into two agents), and respawn.

### Verification protocol per agent

Every agent must, before returning, run from `/Users/ka/github/existential-birds/daydream`:

```bash
make lint        # ruff
make typecheck   # mypy
make test        # pytest -v
```

If the agent's changes touch files outside its own scope (e.g. broke an unrelated test), it must roll back, narrow scope, and report what blocked it.

### Failure-recovery protocol

- **First failure**: respawn the same agent with the failure mode included as additional context.
- **Second failure on the same task**: stop. Read the agent's reports, summarize the blocker, and tell the user. Do **not** push through with broken state.
- **Test regression in unrelated area**: stop. Surface to user. Likely indicates a hidden coupling we didn't account for.
- **Pre-push hook would fail**: same as above. The hook runs `make check`; if `make check` doesn't pass, do not commit, do not push.

### Commit hygiene

- Each stage commits separately with a descriptive message ending in `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- No `--no-verify`. If pre-push fails, stop and surface.
- Do not push to remote unless the user explicitly asks. PR creation happens at the very end, only on user request.

---

## Execution stages

### Stage 0: Setup

Spawn one agent:

```
Task: Prepare workspace for the worktree-isolation refactor.

Repo: /Users/ka/github/existential-birds/daydream
Steps:
1. Verify clean working tree. If dirty, stop and report.
2. Verify on main, up to date with origin. If not, stop and report.
3. Create branch feat/worktree-isolation off main.
4. Add .daydream/worktrees/ and .daydream/intents/ to .gitignore if missing.
5. Run `make check` and confirm baseline is green. Report any failing tests as a baseline blocker — do not proceed.

Verify:
- `git status` clean
- `git branch --show-current` returns feat/worktree-isolation
- `make check` passes

Return ≤150 words: confirmation, baseline test count, any anomalies.
```

If Stage 0 reports a baseline failure, stop and tell the user.

---

### Stage 1: Foundation (sequential — Agent 1.1 → Agent 1.2)

These two modules are the foundation everything else builds on. Run them sequentially because Agent 1.2 imports from Agent 1.1's module.

#### Agent 1.1: `daydream/git_ops.py`

```
Task: Create daydream/git_ops.py — the single git/gh subprocess wrapper module.

Repo: /Users/ka/github/existential-birds/daydream
Branch: feat/worktree-isolation (already created)

Read first:
- daydream/CLAUDE.md (project conventions)
- daydream/phases.py (see existing git calls around lines 478-617)
- daydream/pr_review.py (see _run helper around line 294)
- daydream/runner.py (lines 208-217, 673-680)
- daydream/tree_sitter_index.py (lines 311-318)
- daydream/archive/git_context.py (lines 56-95)
- daydream/eval/pr_feedback.py (lines 20-26)
- daydream/cli.py (lines 74-101)
- /Users/ka/github/reference_agents/codex/codex-rs/git-utils/src/branch.rs (reference for merge_base_with_head with upstream-ahead preference)

Write daydream/git_ops.py with:

Public API:
  # Errors
  class GitError(Exception): ...
  class NotAWorktreeError(GitError): ...
  class BranchNotFoundError(GitError): ...
  class WrongBranchError(GitError): ...

  # Pre-flight
  def assert_is_worktree(repo: Path) -> None
  def is_inside_worktree(repo: Path) -> bool

  # Read-only queries (all take repo: Path explicitly)
  def head_sha(repo: Path) -> str
  def current_branch(repo: Path) -> str | None  # None if detached
  def default_branch(repo: Path) -> str  # origin/HEAD → main → master
  def branch_exists(repo: Path, ref: str) -> bool  # checks local + origin/<ref>
  def merge_base(repo: Path, base: str, head: str = "HEAD") -> str | None
      # Port codex's algorithm: prefer upstream when ahead.
  def diff(repo: Path, base: str, head: str = "HEAD", *, exclude: list[str] | None = None) -> str
  def log(repo: Path, base: str, head: str = "HEAD") -> str
  def show(repo: Path, ref: str, path: str) -> bytes
  def grep(repo: Path, pattern: str) -> list[str]
  def status_porcelain(repo: Path) -> str
  def upstream_ahead_count(repo: Path, branch: str) -> int  # 0 if no upstream

  # Mutating
  def fetch(repo: Path, remote: str = "origin") -> None
  def checkout_paths(repo: Path, paths: list[Path]) -> None  # for `git checkout .`
  def clean_untracked(repo: Path) -> None  # for `git clean -fd`
  def worktree_add(repo: Path, path: Path, ref: str, *, detach: bool = True) -> None
  def worktree_remove(repo: Path, path: Path, *, force: bool = True) -> None

  # gh wrappers
  def gh_pr_view(repo: Path, pr: int) -> dict | None
  def gh_pr_list_for_branch(repo: Path, branch: str) -> list[dict]
  def gh_pr_diff(repo: Path, pr: int) -> str
  def gh_repo_view(repo: Path) -> tuple[str, str] | None  # (owner, name)
  def gh_api(repo: Path, endpoint: str, *, method: str = "GET", paginate: bool = False) -> Any

Conventions (match the project's style):
- Every subprocess.run call has `# noqa: S603 - arguments are not user-controlled` and `# noqa: S607` where applicable.
- Use `cwd=repo` (not `git -C`) for consistency.
- Use `text=True` and `check=False`; raise typed errors based on returncode.
- Reasonable timeouts: 5s for queries, 30s for fetch/diff, 60s for gh.
- Catch (subprocess.SubprocessError, OSError) at edges and convert to GitError.
- Type hints strict; Python 3.12 union syntax (str | None).

Write tests/test_git_ops.py with full coverage:
- Real git tempdirs (use pytest tmp_path); no mocks for subprocess.
- assert_is_worktree on: a worktree (passes), a non-repo dir (raises), an org dir with repo subdirs (raises).
- merge_base correctness on a 3-branch fixture (main, feat, feat-rebased) including the upstream-ahead case.
- branch_exists for local-only, origin-only, and missing branches.
- worktree_add + worktree_remove round trip.
- All error types reachable.

Verify:
- make lint
- make typecheck
- make test
- All new tests pass; existing 343 tests still pass.

Do NOT migrate any callsites yet — that's Stage 2.

Return ≤200 words.
```

#### Agent 1.2: `daydream/workspace.py`

```
Task: Create daydream/workspace.py — the WorkContext abstraction managing in-place vs ephemeral worktrees.

Repo: /Users/ka/github/existential-birds/daydream
Branch: feat/worktree-isolation

Read first:
- daydream/CLAUDE.md
- daydream/git_ops.py (just written by Agent 1.1)
- daydream/runner.py (RunConfig dataclass, _resolve_backend pattern)
- daydream/agent.py (singleton state pattern, reset_state for test isolation)
- /Users/ka/github/reference_agents/codex/codex-rs/exec/src/lib.rs around line 1773 (ReviewTarget pattern)

Write daydream/workspace.py with:

@dataclass(frozen=True)
class WorkContext:
    repo: Path             # the directory daydream operates on (cwd or ephemeral)
    source: Path           # original cwd; same as repo for in-place
    base_branch: str       # resolved base ref (e.g. "main" or "develop")
    base_sha: str          # merge-base SHA captured at entry
    head_branch: str | None  # None if detached or running on a SHA
    head_sha: str
    is_ephemeral: bool

    @property
    def is_in_place(self) -> bool: ...

@asynccontextmanager
async def open_workspace(
    source: Path,
    *,
    branch: str | None,
    base: str | None,
    force_ephemeral: bool,
    extra_copy: list[Path] | None = None,
    skip_tests: bool,  # True for --comment/--review; suppresses .env copy
) -> AsyncIterator[WorkContext]:
    """
    Resolution rules (locked design):
      - If branch is None and not force_ephemeral: in-place at source.
      - If branch is None and force_ephemeral: ephemeral at source's HEAD.
      - If branch is given: ephemeral, detached at origin/<branch> post-fetch.
        Warn if <branch> is currently checked out in source and is behind origin.
      - Always git_ops.fetch() in ephemeral mode.

    Base resolution:
      - If base is given: use it.
      - Else if branch has an open PR: use PR's baseRefName via gh_pr_list_for_branch.
      - Else: git_ops.default_branch().

    Errors:
      - source not a worktree → NotAWorktreeError
      - branch not found locally or on origin post-fetch → BranchNotFoundError
      - base not found → BranchNotFoundError

    On enter:
      1. assert_is_worktree(source)
      2. fetch() if ephemeral
      3. resolve branch, base
      4. if ephemeral: worktree_add(repo=source, path=<.daydream/worktrees/<run-id>>, ref=<resolved>)
         then copy .env files per copy_files_into_ephemeral()
      5. compute base_sha via git_ops.merge_base
      6. yield WorkContext

    On exit:
      - if ephemeral: worktree_remove (best-effort, even on exception)

def copy_files_into_ephemeral(
    source: Path, dest: Path, *, extra: list[Path] | None = None, skip: bool = False
) -> list[Path]:
    """
    skip=True → no-op (for --comment/--review).
    Otherwise:
      1. Read [tool.daydream.workspace] copy from source/pyproject.toml if present.
      2. Else use default list: .env, .env.local, .env.*.
      3. Append `extra` (from --copy flags).
      4. For each: if exists in source, copy to dest preserving relative path.
      5. Return list of files actually copied.
    """

Use `secrets.token_hex(4)` + timestamp for run-id. Run-id format: `<UTC YYYYMMDDHHMMSS>-<hex>`.

Tests in tests/test_workspace.py:
- In-place: yields source as repo, no fetch, no cleanup.
- Ephemeral with no branch: creates worktree at source's HEAD, fetches, cleans up.
- Ephemeral with branch (local exists, origin exists): uses origin/<branch>.
- Ephemeral with branch (only origin exists): fetches and uses origin/<branch>.
- Branch not found anywhere → BranchNotFoundError.
- copy_files_into_ephemeral with default list and pyproject override.
- skip=True suppresses copying.
- Cleanup runs even on exception inside the context.
- `git -C <ephemeral> rev-parse --is-inside-work-tree` returns true while context is open.

All tests use real git tempdirs with real fetches against a fixture remote (use a second tmp_path as the "origin" with `git init --bare` + push). No mocking of subprocess.

Verify:
- make lint, make typecheck, make test
- All previous tests still pass.

Return ≤200 words.
```

After Agent 1.2 returns success, commit:

```
git add daydream/git_ops.py daydream/workspace.py tests/test_git_ops.py tests/test_workspace.py .gitignore
git commit  # message: "feat(git): add git_ops and workspace foundations (#44)"
```

---

### Stage 2: Migrate existing callsites (3 parallel agents)

All three agents work on disjoint file sets. Spawn in parallel via a single message with three Agent tool calls.

#### Agent 2.1: Migrate phases.py + runner.py

```
Task: Migrate every git/gh subprocess call in daydream/phases.py and daydream/runner.py to use daydream/git_ops.py. No behavior change.

Branch: feat/worktree-isolation
Read first:
- daydream/git_ops.py (the new module)
- daydream/phases.py (look at lines 478-617 specifically; all _git_* helpers)
- daydream/runner.py (lines 208-217 and 673-680)

Replace the private helpers (_git_clean, _detect_default_branch, _git_diff, _git_log, _git_branch, _get_head_sha, status check) with direct calls to git_ops.* functions. Delete the now-dead helpers.

Constraints:
- No changes to public function signatures.
- No new types, no new behavior.
- Imports: `from daydream import git_ops`. Use `git_ops.diff(repo, base)` etc.
- All existing tests must pass without modification.

Verify:
- make lint, make typecheck, make test
- grep "subprocess" daydream/phases.py daydream/runner.py — should return zero git/gh hits (only non-git subprocess calls remain, if any).

Return ≤200 words listing which helpers were deleted and what replaced them.
```

#### Agent 2.2: Migrate pr_review.py + cli.py

```
Task: Migrate every git/gh subprocess call in daydream/pr_review.py and daydream/cli.py to use daydream/git_ops.py.

Branch: feat/worktree-isolation
Read first:
- daydream/git_ops.py
- daydream/pr_review.py (especially the _run helper at line 294 and all callsites)
- daydream/cli.py (lines 74-101 for gh calls)

Replace pr_review.py's _run helper and all its callsites with git_ops.* calls. Delete _run.
Replace cli.py's gh subprocess calls with git_ops.gh_* helpers.

Constraints:
- pr_review.py public functions keep their signatures.
- All existing pr_review tests pass without modification.

Verify:
- make lint, make typecheck, make test
- grep "subprocess" daydream/pr_review.py daydream/cli.py — git/gh hits should be zero.

Return ≤200 words.
```

#### Agent 2.3: Migrate tree_sitter_index.py + archive/git_context.py + eval/pr_feedback.py + deep/orchestrator.py

```
Task: Migrate remaining git/gh subprocess calls in:
- daydream/tree_sitter_index.py (line 311)
- daydream/archive/git_context.py (lines 56-95, the _run_git helper)
- daydream/eval/pr_feedback.py (lines 20-26, the _gh_api helper)
- daydream/deep/orchestrator.py (any git/gh call sites)

Branch: feat/worktree-isolation
Read first:
- daydream/git_ops.py
- The four files above.

For each file: replace direct subprocess calls and any private helpers with git_ops.* calls. Delete _run_git and _gh_api.

Constraints:
- No public-API changes.
- All existing tests pass.
- tree_sitter_index.py's `git -C` style → `git_ops.grep(repo, ...)`.

Verify:
- make lint, make typecheck, make test
- For each file, grep "subprocess" — only non-git/gh hits should remain.

Return ≤200 words.
```

After all three agents return success, commit:

```
git add -A
git commit  # "refactor(git): migrate all git/gh callsites to git_ops module (#44)"
```

If any agent reported a regression, stop and surface to the user.

---

### Stage 3: Wire WorkContext into phases (sequential)

#### Agent 3.1: Thread WorkContext through phases.py

```
Task: Replace `cwd: Path` with `work: WorkContext` in every phase function in daydream/phases.py. Update all callsites accordingly.

Branch: feat/worktree-isolation
Read first:
- daydream/workspace.py (the WorkContext dataclass)
- daydream/phases.py (every phase_* function)
- daydream/runner.py (the run() function and how it calls phases)
- daydream/deep/orchestrator.py (also calls phase functions)

Changes:
1. Every `phase_*` function: signature changes from `(backend: Backend, cwd: Path, ...)` to `(backend: Backend, work: WorkContext, ...)`.
2. Inside each phase: replace `cwd` with `work.repo` for filesystem operations; `work.base_branch` and `work.base_sha` for diff/commit anchoring.
3. Update `phase_commit_push` and `phase_commit_push_auto` to pass --repo, --base, --intent to the skill:
   - Write the intent file via a new helper write_intent_json(work, fixed_items) → Path. Place at work.repo / ".daydream" / "intents" / f"{run_id}.json" (gitignored).
   - skill_invocation arg string: f"--repo {work.repo} --base {work.base_sha} --intent {intent_path}"
   - Same change in deep/orchestrator.py's commit step.
4. Same arg-passing pattern for phase_commit_iteration if it exists.

Constraints:
- Backend is still first param.
- WorkContext is second param everywhere.
- No new behavior beyond the threaded refs.

Tests:
- Update mocks in test_phases.py to construct a WorkContext (use a small builder helper) instead of passing a raw Path.
- Add one test asserting phase_commit_push passes the expected --repo --base --intent string to the backend.

Verify:
- make lint, make typecheck, make test
- All 343+ existing tests pass.
- New commit-push test passes.

Return ≤200 words.
```

After Agent 3.1 returns success, commit:

```
git commit  # "feat(phases): thread WorkContext + pass refs to commit-push skill (#44)"
```

---

### Stage 4: CLI consolidation (sequential — Agent 4.1, then Agent 4.2)

#### Agent 4.1: New CLI surface in cli.py and runner.py

```
Task: Implement the new consolidated CLI surface.

Branch: feat/worktree-isolation
Read first:
- daydream/cli.py (current arg parsing in _parse_args)
- daydream/runner.py (RunConfig dataclass and the various run_* entry points: run, run_pr_feedback, run_trust)
- daydream/deep/orchestrator.py (run_deep entry)
- daydream/config.py (REVIEW_SKILLS, ReviewSkillChoice — these become internal-only after this change)
- daydream/workspace.py (open_workspace)

Implement:

1. New cli.py argument layout:
   - Positional [target] (default: ".") — directory path. Cannot be a number; if a number is given, error "did you mean: daydream feedback <pr#>?"
   - Subcommand `feedback <pr-number>` — replaces today's `--pr <n>`.
   - Output mode group (mutually exclusive): --comment, --review. Default = fix-loop.
   - Selection: --branch, --base.
   - Modifiers: --worktree, --shallow, --copy <path> (action="append").
   - Deprecated flags emit a stderr warning via warnings.warn(DeprecationWarning) but still map to behavior:
     - --python/--typescript/etc. → set internal forced_skill, imply --shallow
     - --ttt → maps to --comment (the closest replacement)
     - --pr <n> → "use `daydream feedback <n>` instead", then route to feedback flow
     - --deep → no-op (deep is now default); warn that the flag is unnecessary
     - --review-only → --review

2. New RunConfig fields:
   - branch: str | None
   - base: str | None
   - output_mode: Literal["loop", "comment", "review"]
   - force_worktree: bool
   - shallow: bool
   - extra_copy: list[Path]
   - forced_skill: str | None  # set by deprecated language flags

3. runner.run() rewrites:
   - Single entry point. open_workspace(...) at the top.
   - Dispatch on output_mode:
     - "loop" → existing flow (review → fix → test → loop), inside the workspace.
     - "comment" → review + post-comments (today's pr_review.post_review_comments path), no fix, no test.
     - "review" → review + write to terminal/markdown, exit.
   - When config.shallow is False (default) → use deep/orchestrator pipeline.
   - When config.shallow is True → today's single-stack path with forced_skill or auto-detect.
   - Delete the separate run_pr_feedback and run_trust top-level entry points; their bodies become small helpers called from run() based on output_mode.
   - feedback subcommand → separate run_feedback() that uses today's --pr behavior, also wrapped in open_workspace.

4. Backwards compat: any of the deprecated flags continues to work this release with a clear stderr warning naming the replacement.

Constraints:
- No skill changes. The agent prompts and skill names stay the same.
- Tests: any test that was relying on a removed entry point gets updated to use run() with the appropriate output_mode.

Verify:
- make lint, make typecheck, make test
- Manual: run `daydream --help` — output reflects new surface.
- Manual: run with each deprecated flag → confirm warning printed and behavior preserved.

Return ≤250 words listing exact deprecation mappings and any test updates.
```

After Agent 4.1 returns success, commit:

```
git commit  # "feat(cli): consolidate flags + introduce --branch/--base/--comment (#44)"
```

#### Agent 4.2: Loud branch validation + integration glue

```
Task: Add the loud branch validation that catches silent-failure cases, plus integration tests.

Branch: feat/worktree-isolation
Read first:
- daydream/workspace.py
- daydream/runner.py (post-Agent-4.1)
- daydream/git_ops.py

Implement:

1. In open_workspace's resolution path: if branch is None AND output_mode is "loop" AND current_branch == base_branch, raise WrongBranchError with a message like:
   "cwd is on the base branch '<base>' — there's nothing to review against itself.
    Either:
      • check out a feature branch in this worktree and re-run, or
      • run with --branch <feature-branch> to review the server's version, or
      • run with --worktree to force ephemeral isolation."
   This must NOT fire for --comment or --review (you might legitimately want a no-op review report from the base branch). Confirm via tests.

2. Ensure WrongBranchError is exported from git_ops.

3. Integration tests in tests/test_integration_workspace.py:
   - End-to-end: `daydream` from a worktree on main with no branch — fails with WrongBranchError, not a silent empty-diff.
   - End-to-end: `daydream --branch feat/X` with X only on origin — fetches, creates ephemeral, runs review, cleans up.
   - End-to-end: `daydream --branch feat/X` with X also currently checked out — warns about staleness, still uses origin/X.
   - End-to-end: `daydream --comment --branch feat/X` with no PR open — errors with "no open PR for branch X — push first or use --review".
   - End-to-end: `daydream --comment --branch feat/X` with an open PR — uses PR's base ref as default.
   - End-to-end: `daydream feedback <pr#>` — works as today's --pr flow did.
   - All use real git fixtures (bare origin + two clones), no mocks for git operations.
   - Mock only the Backend (use a small MockBackend that yields a canned review result).

Verify:
- make lint, make typecheck, make test
- Total test count: previous 343 + new tests from stages 1, 3, 4. Report final count.

Return ≤200 words.
```

After Agent 4.2 returns success, commit:

```
git commit  # "feat(workspace): loud branch validation + integration tests (#44)"
```

---

### Stage 5: Final integration verification

Spawn a verification-only agent (read-only):

```
Task: Verify the entire refactor is internally consistent and the daydream binary works end-to-end.

Branch: feat/worktree-isolation
This is verification-only. Do not write code.

Checks:
1. `make check` — full pass (lint + typecheck + test).
2. `grep -rn "subprocess" daydream/ --include="*.py" | grep -E "git|gh "` — should return zero hits outside daydream/git_ops.py.
3. `grep -rn "_run_git\|_gh_api\|_run\b" daydream/` — should return zero hits (all old helpers deleted).
4. `daydream --help` runs without error and shows the new flag surface.
5. `daydream --python /tmp/nonexistent-path 2>&1` — should print a deprecation warning and a "not a worktree" error (because the path doesn't exist), not a silent crash.
6. From a non-repo dir: `cd /tmp && daydream .` — should print NotAWorktreeError, not confabulate.
7. Spot-check 5 random tests from tests/test_phases.py and tests/test_pr_review.py for sanity (real git, no mocked subprocess).
8. Verify .gitignore contains .daydream/worktrees/ and .daydream/intents/.
9. Confirm all commits on the branch follow the expected message format.

Return ≤300 words: a checklist of which checks passed, which failed, and a final verdict (READY TO PR / NEEDS WORK).
```

If verification reports READY TO PR: surface to user with the branch name and the suggested PR title/body. **Do not push or create the PR yourself** — that's the user's call.

If verification reports NEEDS WORK: surface the failures to the user; do not attempt further fixes without instruction.

---

## Done criteria

- [ ] Branch `feat/worktree-isolation` exists with all stage commits.
- [ ] `make check` passes on the branch tip.
- [ ] No git/gh subprocess.run outside `daydream/git_ops.py`.
- [ ] No private `_run_git`, `_gh_api`, `_run` helpers in daydream source.
- [ ] WorkContext threaded through every `phase_*` function.
- [ ] `phase_commit_push`, `phase_commit_push_auto`, deep-mode commit pass `--repo --base --intent` to the skill.
- [ ] CLI surface matches the locked design; deprecated flags emit warnings.
- [ ] Wrong-branch silent failure replaced with a loud error.
- [ ] `--branch X` always uses `origin/X` post-fetch in an ephemeral worktree.
- [ ] `--comment` mode skips the test phase.
- [ ] `.env` copy mechanism (default list + pyproject override + --copy flag) wired into ephemeral path.
- [ ] Integration tests cover the multi-worktree scenario end-to-end.
- [ ] Total test count > 343 (previous baseline) with all passing.
- [ ] Issue #44 success criteria satisfied (re-read it before final report).

When all done criteria are met, surface a brief summary to the user with:
- Final test count.
- Files added: git_ops.py, workspace.py, test_git_ops.py, test_workspace.py, test_integration_workspace.py.
- Files heavily modified: phases.py, runner.py, cli.py, pr_review.py, deep/orchestrator.py.
- Any deviations from the locked design (should be zero; flag if not).
- Suggested PR title: `feat: unify git ops + worktree isolation (#44)`.
- Whether to also open a follow-up PR in beagle-core for the `commit-push` skill to accept the new --repo/--base/--intent flags (the daydream callsite passes them either way; the skill will currently ignore unknown args).

---

## Project rules to honor when implementing

From `CLAUDE.md` (project) and `MEMORY.md` (auto-memory):

- **Don't embed diffs in prompts.** Pass refs (file paths, base branch); let agents fetch via Read/Grep/Bash.
- **Module-bloat ban.** No `Step()`, `ToolCall()`, or `Trajectory()` construction inside `phases.py` or `ui.py` — all ATIF model construction stays in `daydream/trajectory.py`. (Keep new git/workspace modules similarly siloed.)
- **All AI calls go through `run_agent()` in `daydream/agent.py`** — never the SDK directly.
- **`Backend` is always the first parameter** in phase functions.
- **No `print()` in library code** — use `daydream/ui.py` helpers.
- **Pre-push hooks run lint + typecheck + full test suite.** Plan for this when migrating callsites.

---

## Final reminder

You are the orchestrator. Every code change goes through a sub-agent. Every sub-agent verifies its own work before reporting success. If anything fails twice, stop and surface to the user. Do not push or create PRs without explicit user instruction. Run to completion.
