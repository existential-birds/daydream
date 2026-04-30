# Worktree Isolation + Mode Consolidation — Brainstorm

**Status:** Active brainstorm. No code written yet. Pick up where the "Open questions to resume" section ends.
**Date started:** 2026-04-30
**Branch:** `brainstorm/worktree-isolation-and-mode-consolidation`

---

## How to use this doc

This is a self-contained continuation of a multi-machine brainstorming session. Read top-to-bottom. The "Resolved decisions" section locks settled questions; the "Open questions to resume" section is where the next session picks up.

Source-of-truth references in the codebase are line-stamped (`file.py:NNN`) so you can verify any claim before acting on it. The line numbers were captured on `main` at commit `cef922d` (release 0.14.0). If they've drifted, re-run the sub-agent prompts at the bottom of this doc.

---

## Two problems, one root cause

### Problem 1 — Issue #44 (existing, open)

> **Title:** Unify and harden git operations across daydream

GitHub: https://github.com/existential-birds/daydream/issues/44

Symptoms reported on the issue:
- Running `phase_commit_push` from a session rooted at an org/container dir (`/Users/ka/github/shelfspace-app/`, which holds multiple repo subdirs but isn't itself a repo) produced four parallel `git status`/`git diff`/`git log` calls all failing with `fatal: not a git repository`.
- The invoked agent **mis-narrated** the error as "directory doesn't exist" (it exists — it just isn't a repo).
- The agent then **confabulated** a worktree-related explanation ("this is a git worktree, .git is a file") that was factually wrong for the target.

Root cause per the issue: git calls are scattered with inconsistent conventions; `phase_commit_push` invokes `beagle-core:commit-push` with **zero args**, forcing the skill agent to re-discover state — and to confabulate when discovery fails.

Issue's own success criteria:
- [ ] Every `subprocess.run(["git", ...])` in daydream goes through one new module
- [ ] `phase_commit_push` and peers pass `--repo`/`--base`/`--intent` to the skill
- [ ] Running daydream from an org-level dir produces a clear, actionable error
- [ ] Commit messages reflect fix intent, not a fresh re-derivation
- [ ] Existing 50+ tests still pass; new tests cover not-a-worktree path

### Problem 2 — Worktree branch friction (new, raised in this session)

User report (paraphrased): "I keep a dedicated daydream-review worktree of repos. When I want to review a feature branch, I have to manually go to the *other* worktree (the one with the default branch) and update it before the review worktree can review the branch I want."

Investigation revealed this is worse than the user described — daydream has a **silent-failure bug** any time the worktree's checked-out branch is the same as the base branch:

- `_detect_default_branch()` (`phases.py:501`) reads `origin/HEAD` and returns `main`.
- `_git_diff()` (`phases.py:558`) runs `git diff main...HEAD` with **no check** that `HEAD != main`.
- If your review worktree has `main` checked out, the diff is empty.
- `runner.py:348` reports "No diff found — nothing to review" with no warning that you're sitting on the base branch.
- `--pr` mode (`pr_review.py:318`) calls `gh pr list --head <current_branch>`; on the wrong branch, it returns nothing and bails quietly.

Daydream has **zero** worktree-aware code: no `git worktree add`, no `git fetch`, no `git checkout`, no temp-checkout logic. It assumes you pre-positioned the right branch.

### The shared root cause

Both problems reduce to: **daydream trusts `cwd` as a proxy for "the right repo at the right branch" and never verifies it.** Issue #44 catches the failure when `cwd` isn't a repo at all (org dir). The worktree friction catches it when `cwd` is a repo but on the wrong branch.

---

## Codebase findings (sub-agent reports)

Three Explore agents mapped the codebase. Key facts captured here so a fresh session doesn't have to re-run them.

### Git subprocess inventory (18 callsites, 6+ files, 3 `_run` helpers)

| File | # calls | Notes |
|---|---|---|
| `daydream/phases.py` | 7 | `git checkout .`, `git clean -fd`, `git symbolic-ref`, `git rev-parse --verify`, `git diff base...HEAD`, `git log base..HEAD`, `git branch --show-current` — all use `cwd=Path` |
| `daydream/runner.py` | 2 | `git rev-parse HEAD` (line 208), `git status --porcelain` (line 673) |
| `daydream/pr_review.py` | 6 | Has its own `_run()` helper (line 294); mix of `git` and `gh pr` calls |
| `daydream/cli.py` | 2 | `gh pr view`, `gh repo view` |
| `daydream/tree_sitter_index.py` | 1 | Uses `git -C <repo>` (line 311) — the **only** call using `-C` instead of `cwd=` |
| `daydream/archive/git_context.py` | 5 | Has its own `_run_git()` helper (line 56) |
| `daydream/eval/pr_feedback.py` | 1 | Has its own `_gh_api()` helper (line 20) |

Indirect git in prompts: `daydream/prompts/exploration_subagents.py` instructs subagents to run `git diff {diff_ref} -- <file>` (3 occurrences).

Universal pattern: `subprocess.run()` with `capture_output=True`, hardcoded args, error wrapped in `try/(subprocess.SubprocessError, OSError)`. No shell injection risk; just inconsistent.

### Skill invocations (5 callsites)

`format_skill_invocation` signature: `(skill_key: str, args: str = "") -> str`. Returns `"/{skill_key}"` for Claude or `"/{skill_key} {args}"` when args non-empty.

| File:Line | Skill | Args today | Phase | Available context not passed |
|---|---|---|---|---|
| `phases.py:675` | dynamic | empty | `phase_review` | `cwd`, `diff_base` (incremental), `exclude` paths, base branch |
| `phases.py:916` | `beagle-core:commit-push` | **empty** | `phase_commit_push` | `cwd` (repo), would benefit from base SHA + fix intent |
| `phases.py:941` | `beagle-core:fetch-pr-feedback` | `--pr {n} --bot {bot}` | `phase_fetch_pr_feedback` | already passes args ✓ |
| `phases.py:1079` | `beagle-core:commit-push` | **empty** | `phase_commit_push_auto` | same as 916 |
| `phases.py:1111` | `beagle-core:respond-pr-feedback` | `--pr {n} --bot {bot}` | `phase_respond_pr_feedback` | already passes args ✓ |

**Base SHA is captured in `runner.py:663` via `_get_head_sha()` before `phase_commit_iteration()` runs. Stored in outer-scope `diff_base: str | None` at `runner.py:602`. Never threaded to `phase_commit_push` or `phase_commit_push_auto`.**

### Worktree / branch handling

- `target_dir` resolved from `config.target` at `runner.py:451-455`. Daydream does **not** verify `target_dir` is a git worktree.
- Base branch detection: `_detect_default_branch()` (`phases.py:501-540`) — primary method `git symbolic-ref refs/remotes/origin/HEAD`, fallback to `main`/`master` via `git rev-parse --verify`. **Never compares to `git branch --show-current`.**
- Diff: `_git_diff()` (`phases.py:558`) blindly runs `git diff {base_branch}...HEAD`. Empty when `HEAD == base_branch`.
- Branch detection per mode:
  - Normal: `_git_branch()` reads `git branch --show-current` (line 601)
  - `--pr`: `_current_branch()` (`pr_review.py:310`) + `gh pr list --head <branch>`
  - `--ttt`/`--deep`: same — current branch only
- Daydream **never** runs `git checkout` or `git fetch`. User must pre-position.
- Zero hits for `worktree`, `--bare`, `clone`, ephemeral checkout logic anywhere in the codebase.

---

## The 4-phase proposal

Each phase ships independently and is useful on its own. Recommended order: 1 → 2 → 4 → 3 (foundation, cheap fix, finish #44, then biggest design surface).

### Phase 1 — `daydream/git_ops.py` (foundation for #44)

One module wrapping every git/gh subprocess call. Every function takes an explicit `repo: Path`. No implicit cwd anywhere.

```python
# daydream/git_ops.py
class NotAWorktreeError(DaydreamError): ...
class WrongBranchError(DaydreamError): ...

def assert_is_worktree(repo: Path) -> None: ...
def head_sha(repo: Path) -> str: ...
def current_branch(repo: Path) -> str: ...
def default_branch(repo: Path) -> str: ...     # origin/HEAD with main/master fallback
def diff(repo: Path, base: str, head: str = "HEAD", *, exclude: list[str] = ...) -> str: ...
def log(repo: Path, base: str, head: str = "HEAD") -> str: ...
def show(repo: Path, ref: str, path: str) -> bytes: ...
def grep(repo: Path, pattern: str) -> list[str]: ...
def fetch(repo: Path, remote: str = "origin") -> None: ...
# gh helpers (gh_pr_list, gh_pr_diff, gh_repo_view, ...)
```

Migrate all 18 callsites + the three `_run`-style helpers behind it. Add a single `assert_is_worktree(repo)` pre-flight at run entry that raises `NotAWorktreeError` with actionable text — kills the org-dir confabulation case from issue #44 cleanly.

### Phase 2 — Loud branch validation (cheap fix for the silent-failure bug)

Before `phase_review` runs, call a new `WorkContext.validate()` helper that:

1. Asserts `repo` is a worktree (Phase 1).
2. Reads `current_branch(repo)` and `default_branch(repo)`.
3. If `current_branch == default_branch`, raises `WrongBranchError` with a message naming both branches and listing options.
4. For `--pr`, additionally verify a PR exists for the current branch before proceeding.

~50 LOC. Catches every silent failure identified by sub-agent #3 without needing the full ephemeral worktree machinery.

### Phase 3 — `--worktree` mode (the real answer to the user's question)

Add an opt-in flag that flips daydream into ephemeral-worktree mode. New module `daydream/workspace.py`:

```python
@dataclass
class WorkContext:
    repo: Path                 # the worktree daydream operates on
    base_sha: str              # captured at entry, threaded everywhere
    base_branch: str
    head_branch: str
    is_ephemeral: bool         # True when daydream created the worktree
    cleanup: Callable | None   # called in finally; None for in-place

@asynccontextmanager
async def open_workspace(target: Path, *, branch: str | None, ephemeral: bool) -> WorkContext:
    ...
```

Behavior matrix (subject to mode consolidation — see "Mode consolidation" below):

| Invocation | Behavior |
|---|---|
| `daydream .` (default) | In-place. Phase 2 validation. Errors loudly on wrong branch. |
| `daydream . --branch feat/X` | If `feat/X` is checked out in *this* worktree → in-place. Otherwise create ephemeral worktree at `.daydream/worktrees/<run-id>/`, `git fetch`, `git worktree add --detach <path> origin/feat/X`. Cleanup with `git worktree remove --force` on exit. |
| `daydream . --worktree` | Always ephemeral; uses current branch (post-fetch). |
| `daydream <pr#> --pr` | Implicit `--branch` derived from PR head ref. **Always ephemeral, no opt-out.** |

Composes with `git worktree`'s shared object DB: commits made by `phase_commit_push` in an ephemeral worktree are immediately visible in your other worktrees once pushed.

### Phase 4 — Pass refs to skills (closes #44's "harden the skill" item)

Once `WorkContext` exists, `phase_commit_push` becomes:

```python
intent_path = write_intent_json(work, fixed_items)
skill = backend.format_skill_invocation(
    "beagle-core:commit-push",
    f"--repo {work.repo} --base {work.base_sha} --intent {intent_path}",
)
```

Same treatment for `phase_commit_push_auto`, `phase_commit_iteration`, and any deep-mode commit calls. Per the project memory rule: **don't embed diffs in the prompt — pass refs.** The skill re-derives the diff from `--base..HEAD` cheaply.

The corresponding `beagle-core:commit-push` change is a separate PR in beagle-core, but it can land in either order — old skill ignores unknown args.

---

## Resolved decisions

These were settled during this session. Don't re-litigate without strong reason.

### Q1 — Where do ephemeral worktrees live?
**Resolved: inside the target repo at `.daydream/worktrees/<run-id>/`.**
Discoverable; cleaned up by hand if needed; consistent with the existing `.daydream/` artifact convention.

### Q2 — Auto-fetch policy in ephemeral mode?
**Resolved: always fetch.** If the user opted into isolation, they want fresh refs.

### Q3 — `--pr` mode and ephemeral?
**Resolved: ephemeral-only for PR review. No opt-out flag.**
PR review is brittle and error-prone in shared worktrees; eliminating the non-ephemeral path closes a whole class of "why is the PR review empty" bugs. Drops a config dimension we'd regret supporting.

### Q4 — Test phase + ephemeral interaction (.env problem)?
**Resolved: option A — `--pr`/`--comment` mode skips `phase_test_and_heal` entirely.**
PR review's job is "review and post comments," not "fix and test." Tests in PR review pull in `.env` complexity that doesn't pay for itself. Normal/`--worktree`/`--branch` modes (which do fix-and-test) keep tests.

Implication: `.env` copy config (next section) is dead in `--pr` path; only matters for opt-in ephemeral that runs tests.

### Copy config (resolved scope, two layers)

For ephemeral modes that *do* run tests (`--worktree`, `--branch X`-into-ephemeral), daydream needs to replicate the user's `.env` and similar gitignored config files into the ephemeral worktree.

**Layer 1 — Default copy list (zero config):** when entering ephemeral mode that runs tests, copy any file in the source worktree matching `.env`, `.env.local`, `.env.*` that is **gitignored**. Gitignored is the safety check — never copy tracked files (git provides them) and never copy untracked files the user might have for unrelated work.

**Layer 2 — Per-repo override:**
```toml
# pyproject.toml in the target repo
[tool.daydream.workspace]
copy = [".env", "instance/secrets.json", ".tool-versions"]
```
Overrides the default. Lives in the target repo so teammates inherit. If a listed file isn't gitignored, daydream warns but copies anyway.

**Dropped:** `--copy <path>` ad-hoc flag. Rare enough that "copy it yourself before running" is fine.

**Copy, not symlink** — symlinks would let an agent editing `.env` in the ephemeral worktree silently mutate the user's template. Copy is cheap (these files are tiny) and isolated.

The user's existing "dedicated review worktree" mental model is preserved: it becomes the **template** that ephemeral worktrees inherit from. The template's branch never changes (it can stay on `main` forever); ephemeral children handle the actual checkout.

---

## Mode consolidation discussion

The user said: "all of these modes are confusing, could we consolidate them? I mostly only use --deep and either stop at comments (I think this is pr mode) or fully loop and fix."

Critical clarification: **`--pr` is not "stop at comments."** It *ingests* existing PR comments (e.g. coderabbit's) and applies them as fixes. The "post my review as comments" flow is buried inside `--ttt` and `--deep` as a side-effect. The user has been confused about which flag does what, and the naming is the cause.

### What today's modes actually do

| Today's flag | What it really is |
|---|---|
| `--python` / `--typescript` / etc. | Single-stack review → fix → test loop |
| `--ttt` | Stack-agnostic alternative review, can post to PR |
| `--pr <n>` | Fetch existing bot comments, fix them, respond |
| `--deep` | Multi-stack review pipeline (TTT + per-stack + merge), can post to PR |
| `--review-only` | Modifier on any of the above |

Three orthogonal axes are tangled into one flag space:
1. **Source** — diff against base, or specific PR
2. **Depth** — single-stack vs multi-stack (deep)
3. **Output** — terminal, fix loop, or post-comments

Plus one genuinely separate workflow (ingest external bot feedback) that shouldn't share the same flag namespace.

### Proposed consolidation (NOT YET ACCEPTED — open question)

```bash
# Default: review → fix → test loop, multi-stack auto-detection
daydream                          # current branch
daydream <path>                   # specified repo
daydream <pr-number>              # auto-fetch PR's branch, ephemeral

# Output modes (mutually exclusive)
daydream --comment                # review → post as PR comments → exit  (today's "ttt"/"deep" with PR posting)
daydream --review                 # review → terminal/markdown → exit    (today's --review-only)
# default behavior is fix+test loop, no flag needed

# Scope modifiers
--branch X     # checkout in ephemeral worktree
--worktree     # force ephemeral
--shallow      # opt out of multi-stack (rare)

# Separate command for the third workflow
daydream feedback <pr#>           # today's --pr: ingest bot comments and fix
```

**Mapping the user's stated workflow:**
- "stop at comments" → `daydream --comment` or `daydream <pr#> --comment`
- "fully loop and fix" → `daydream` or `daydream <pr#>`

**Consequences:**
- Kills `--ttt`, `--pr` (as flag), `--deep`, all `--python`/`--typescript`/`--rust`/etc. language flags (deep auto-detects stacks via `daydream/deep/detection.py`), and `--review-only`-as-modifier.
- Five modes collapse to one default + two output flags.
- Multi-stack becomes the default; `--shallow` is the explicit opt-out for unusual single-stack control.
- Decision A becomes "**`--comment` skips tests**" — same logic, cleaner naming.
- Language-flag death: `--shallow --skill X` covers force-narrow cases; auto-detect handles the common path.

**Costs:**
- **Breaking CLI change.** Scripts, README examples, muscle memory using `--python`, `--ttt`, `--pr`, `--deep` all break. Ship a deprecation period: old flags stderr-warn for one release, map to new behavior.
- **Loss of explicit single-stack control on polyglot repos.** Today you can force `--python` to limit scope. New design uses `--shallow --skill X` — clumsier ladder.
- **`daydream <pr-number>` as a positional argument that changes meaning** based on whether it's a number or a path. That overload usually ages badly. Considered alternative: require `daydream pr <n>` as a subcommand. More typing, less ambiguity. **This is an open sub-question.**

---

## Open questions to resume

Pick up here in the next session.

1. **Lock or revise the mode consolidation.** Specifically:
   - (a) Accept the proposed consolidation as-is?
   - (b) Accept with `daydream pr <n>` as a subcommand instead of a positional-overload?
   - (c) Reject / propose alternative shape?
2. **Deprecation strategy for old flags.** One release with stderr warnings? Two? Hard break?
3. **`--shallow --skill X` vs auto-detection-only.** Worth the extra CLI surface or kill it?
4. **Implementation order.** Recommended in the proposal: 1 → 2 → 4 → 3, with mode-consolidation slotted as its own phase (probably last, since it depends on `WorkContext` from Phase 3 to make `--comment`/`--worktree` semantics clean). Confirm or revise.
5. **Should mode consolidation land *before* or *after* Phase 3?** Argument for before: design CLI once, build worktree machinery against final shape. Argument for after: ship value (Phase 1 + 2 + 4) without fighting CLI bikeshedding.
6. **The corresponding beagle-core change.** `beagle-core:commit-push` needs to accept `--repo`, `--base`, `--intent`. Separate PR in beagle-core repo. Coordinate timing or land daydream changes first (old skill ignores unknown args)?

After resolving the above, the next step is a real implementation plan (not yet drafted in this session). The work breaks into 5 phases (4 from #44 + mode consolidation); each lands as its own PR.

---

## Project rules to honor when implementing

From `CLAUDE.md` (project) and `MEMORY.md` (auto-memory) — relevant excerpts:

- **Don't embed diffs in prompts.** Pass refs (file paths, base branch); let agents fetch via Read/Grep/Bash. (`feedback_no_diff_in_prompt.md`)
- **Module-bloat ban.** No `Step()`, `ToolCall()`, or `Trajectory()` construction inside `phases.py` or `ui.py` — all ATIF model construction stays in `daydream/trajectory.py`. (Keep new git/workspace modules similarly siloed.)
- **All AI calls go through `run_agent()` in `daydream/agent.py` — never the SDK directly.**
- **`Backend` is always the first parameter** in phase functions.
- **No `print()` in library code** — use `daydream/ui.py` helpers.
- **Pre-push hooks run lint + typecheck + full test suite.** Plan for this when migrating 18 callsites.
- **GSD workflow enforcement:** start work through a GSD command (`/gsd-quick`, `/gsd-debug`, or `/gsd-execute-phase`) before file edits. This brainstorm doc sits outside that requirement.

---

## Sub-agent prompts (for re-running if line numbers drift)

If the codebase has moved on, re-run these three Explore agent prompts in parallel to refresh the inventory. They were the source for the "Codebase findings" section.

### Agent 1 — Map every git subprocess call

> Map every git subprocess call in the daydream codebase. I'm planning a refactor to centralize git operations into a single module (`daydream/git_ops.py`) and need a complete inventory.
>
> Repo root: /Users/ka/github/existential-birds/daydream
>
> For each callsite, report: File:line, command, CWD handling (`cwd=` vs `git -C` vs implicit), error handling, output usage.
>
> Pay attention to: `daydream/phases.py`, `daydream/runner.py`, `daydream/pr_review.py` (note its `_run` helper), `daydream/tree_sitter_index.py`, `daydream/exploration_runner.py`, `daydream/deep/orchestrator.py`. Also check `daydream/prompts/` and `daydream/exploration_subagents.py` for prompt strings that *instruct* a subagent to run git.
>
> Search breadth: very thorough. Return a structured table grouped by file. Aim for ≤400 words.

### Agent 2 — Map skill invocations and arg passing

> Map every callsite that invokes a skill via `backend.format_skill_invocation(...)`.
>
> For each: File:line, skill name, args passed (often empty), calling phase function, available context the caller already has but doesn't pass.
>
> Pay attention to: `daydream/phases.py` (`phase_commit_push`, `phase_commit_push_auto`, `phase_commit_iteration`, `phase_review`, `phase_fix`, `phase_fix_parallel`), `daydream/deep/orchestrator.py`, `daydream/runner.py`, `daydream/backends/__init__.py` for the `format_skill_invocation` signature.
>
> Also: does daydream currently capture a "base SHA before fixes ran" anywhere? Where? Is it threaded through to the commit phase?
>
> Search breadth: very thorough. Return a structured table. Aim for ≤400 words.

### Agent 3 — Investigate worktree handling and target dir

> I'm investigating a UX bug specific to multi-worktree workflows in daydream. The user keeps a dedicated "review" worktree of a repo. When they want to review branch X, they have to manually update a *different* worktree before daydream's review worktree can see the branch — or something equivalent.
>
> Investigate and report:
> 1. How does daydream identify "the repo"? Trace `target_dir`/`cwd`/`repo` from `cli.py:_parse_args` through `runner.RunConfig` into the phases.
> 2. How is the "base branch" detected? What happens if the worktree we're sitting in has the wrong branch checked out?
> 3. How is the "target branch to review" identified? For `--pr`, `--ttt`, `--deep`, normal mode. Does daydream ever run `git checkout` or `git fetch`?
> 4. What does the user have to do today to review branch `feat/X` if their daydream-review worktree has `main` checked out?
> 5. Does daydream support `git worktree add` natively? Search for `worktree`, `bare`, `clone`.
> 6. Where would isolation help most? List the phases that would be safer with an ephemeral worktree per run.
>
> Pay attention to: `daydream/cli.py`, `daydream/runner.py`, `daydream/phases.py`, `daydream/pr_review.py`, `daydream/deep/orchestrator.py`, `daydream/exploration_runner.py`.
>
> Search breadth: very thorough. Return a structured report with concrete file:line references. Aim for ≤500 words.
