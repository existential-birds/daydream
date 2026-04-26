# Concerns & Technical Debt

**Analysis Date:** 2026-04-26

This document tracks technical debt, fragile areas, and known concerns identified during codebase mapping. It is intended as a reference for prioritization, not a list of urgent bugs — the codebase is healthy overall (343 tests, ruff/mypy clean, CI green on push).

## Module Size & Complexity

**`daydream/ui.py` is 3470 lines.**
- Contains all Rich UI rendering: live panel registries (`LiveToolPanelRegistry`, `ParallelFixPanel`, `ShutdownPanel`), themed `Console`, phase heroes, summaries, cost displays, prompt helpers
- Risk: Single-module bottleneck for any UI change; difficult to navigate; hard to unit-test individual components
- Mitigation candidate: Split by concern — `ui/console.py`, `ui/panels.py`, `ui/heroes.py`, `ui/prompts.py` — but the existing flat structure is intentional and changes would touch every caller

**`daydream/phases.py` is 1552 lines.**
- Contains every `phase_*()` function plus inline `build_*_prompt()` helpers and JSON Schemas (`FEEDBACK_SCHEMA`, `ALTERNATIVE_REVIEW_SCHEMA`, `PLAN_SCHEMA`)
- Risk: Adding new phases or schemas grows this module without bound; cross-phase helpers and schemas are colocated with concrete phase logic
- Mitigation candidate: Move JSON Schemas to `daydream/schemas.py`; move prompt builders to `daydream/prompts/phase_prompts.py`

**`daydream/pr_review.py` is 925 lines.**
- Contains all GitHub PR posting logic — comment construction, `gh` CLI wrapping, dedup against existing comments, redrive support
- Risk: Subprocess-heavy module mixing transport, formatting, and dedup concerns

**`daydream/runner.py` is 781 lines.**
- Three top-level flows (`run`, `run_pr_feedback`, `run_trust`) plus helpers
- Acceptable size but each flow shares non-trivial bootstrap (backend resolution, exploration pre-scan); shared bootstrap could be extracted

## Tight Coupling / Circular-Risk Areas

**`daydream/deep/orchestrator.py` uses late imports to avoid circularity:**
- `from daydream.backends.codex import CodexBackend  # noqa: E402` (line 77)
- Inline `from daydream.phases import ...` calls inside `run_deep()` to break the runner→deep→phases→runner cycle
- Risk: Adding a new top-level import without checking the cycle can re-introduce import-time failures
- Detection: Already covered by `mypy` on import resolution + `pytest` collecting the modules

**`daydream/backends/__init__.py` re-exports from `claude.py`:**
- `from daydream.backends.claude import ClaudeBackend  # noqa: E402` is necessary because the `Backend` protocol must be defined before backend classes import it
- Risk: Adding a backend that imports the protocol at module top will fail unless ordering is preserved

## Broad Exception Catches

**Locations of `except Exception`:** (15 total per `grep`)
- `daydream/runner.py:248,266,273` — top-level flow boundaries
- `daydream/cli.py:411` — top-level CLI boundary (`sys.exit(1)`)
- `daydream/agent.py:335,437` — agent orchestration boundary
- `daydream/exploration.py:238`, `daydream/exploration_runner.py:225,250` — exploration is intentionally best-effort
- `daydream/tree_sitter_index.py:90,164` — parser failures must not crash the run
- `daydream/phases.py:1014,1465` — phase 1465 is annotated `# noqa: BLE001 -- intentionally broad for parallel isolation` (per-stack reviews must not fail the whole fan-out)

Most are intentional and annotated. Risk: A catch in `daydream/phases.py:1014` is inside a fix-iteration retry loop and could mask real errors during the fix attempt — worth a closer look if fix-loop reliability becomes an issue.

## Subprocess Surface

**Subprocess calls (16 sites):** all marked `# noqa: S603` with a justification, all use `capture_output=True`, `text=True`, `timeout=N`, `shell=False`. No `shell=True` anywhere. No `eval()` or `exec()` in source.

**Hot spots:**
- `daydream/phases.py` — 6 `subprocess.run` calls for git operations (diff, status, add, commit, push)
- `daydream/runner.py` — git status checks
- `daydream/pr_review.py` — `gh` CLI wrapping
- `daydream/tree_sitter_index.py:311` — `git ls-files` for repo-relative path enumeration
- `daydream/cli.py:57` — auto-detect PR number via `gh`

Risk areas:
- `daydream/phases.py:478,486,512,529,566,590,611` — git arguments are constructed from `target_dir` (caller-controlled, but `cwd=target_dir`, so injection requires already having access to the CWD); fine for a local CLI but worth re-reviewing if the tool ever runs against untrusted input
- `daydream/pr_review.py:299` — `gh` CLI args parsed from JSON output; argument list is hardcoded but content (PR number, repo) flows through unchecked

## Type Coverage Gaps

**`mypy` config:** `ignore_missing_imports = true`
- Required for `claude-agent-sdk`, `mcp`, and tree-sitter packages that ship without stubs
- Risk: Type errors in those external interaction layers (`daydream/backends/claude.py`, `daydream/backends/codex.py`, `daydream/tree_sitter_index.py`) won't surface from external types
- Mitigation: Internal type discipline is strict; the protocol abstraction (`Backend`) limits the blast radius

## Tree-Sitter Grammar Pinning

**`daydream/tree_sitter_index.py`** depends on four separate grammar packages:
- `tree-sitter-python`, `tree-sitter-typescript`, `tree-sitter-go`, `tree-sitter-rust`
- Each grammar version is pinned in `pyproject.toml`; bumping them can break query behavior silently
- Existing tests in `tests/test_tree_sitter_index.py` cover happy paths, but grammar regressions (e.g. node-type renames between versions) are easy to miss
- Risk: Medium — grammar bumps should be PR'd individually with the existing test suite + a manual smoke run

## Permission Mode

**`ClaudeAgentOptions(permission_mode="bypassPermissions")`** in `daydream/backends/claude.py:78`:
- All Claude tool calls run in bypass mode — no per-tool approval prompts
- Justification: Daydream is an automated review/fix loop; interactive permission prompts would break the workflow
- Risk: A malicious or broken review skill could cause unintended file writes; mitigated by the user explicitly opting into a target directory and the `--review-only` flag for read-only runs

**`setting_sources=["user"]`** — reads `~/.claude/settings.json`
- Reuses the user's normal Claude Code config (API keys, plugin install state)
- Risk: Beagle plugin must be installed under that profile; missing skills surface via `MissingSkillError`

## Singleton State

**`_state = AgentState()` module-level singleton in `daydream/agent.py`:**
- Holds: `debug_log`, `quiet_mode`, `model`, `shutdown_requested`, `current_backends`
- Accessed only via setters/getters; `reset_state()` provided for tests
- Risk: Anyone importing `_state` directly bypasses reset; tests that stick to setters/getters are safe
- Mitigation: Convention-only, not enforced by linting

## Deep-Mode Complexity

**`daydream/deep/orchestrator.py:run_deep()`** runs:
1. Exploration pre-scan
2. TTT intent + alternatives
3. Stack detection / per-stack fan-out (concurrent with `anyio.CapacityLimiter(4)`)
4. Cross-stack dedup pre-filter
5. Cross-stack merge
6. Optional fix gate

Risk: Long-running, multi-stage pipeline with many failure modes. Each stage writes artifacts under `target/.daydream/` so partial runs are recoverable, but error messages can be confusing when an early stage fails silently and a downstream stage tries to read a missing artifact. Recent commits (`b1b3322`, `8f14d8f`, `09954e4`) indicate ongoing reliability work in this area.

## Recently-Touched Areas (Hot Spots)

Per `git log` (recent commits):
- `8f14d8f` fix(deep): prevent agent write blocks and improve PR review formatting
- `b1b3322` fix(deep): write merge report to `.daydream/deep/` to avoid sandbox restrictions
- `09954e4` fix(deep): improve PR review line resolution and add max_turns control

Hot spots: `daydream/deep/orchestrator.py`, `daydream/pr_review.py`, deep merge artifact path handling. Active churn — new changes here should expect potential conflicts and run the full deep-mode integration suite (`tests/test_deep_*.py`).

## TODOs / FIXMEs

- `grep -rn 'TODO\|FIXME\|HACK\|XXX' daydream/` — **no matches.** No tracked code-level TODOs in source.

## Documentation Gaps

- `tests/fixtures/deep/` contains only a `README.md` — no recorded deep-mode artifacts yet for regression tests to replay
- `docs/plans/` has one design doc (`2026-02-21-go-review-design.md`); deep-mode and exploration system don't have committed design docs (relevant ADRs may live in `.planning/`)
- `daydream/prompts/review_system_prompt.py` defines `build_review_system_prompt()` but the function is not currently wired into the main flow — exploratory/unused module per `CLAUDE.md`

## Test Suite Concerns

- **No coverage tool configured** — no `pytest-cov`, no coverage gate in CI; relies on the file-per-module convention to keep coverage implicit
- `tests/test_runner.py` is only 20 lines — `daydream/runner.py` (781 lines) is exercised mostly through `tests/test_integration.py` (817 lines) and `tests/test_loop.py` (500 lines) rather than direct unit tests
- `tests/test_ui.py` is 77 lines — `daydream/ui.py` (3470 lines) has minimal direct test coverage; UI behavior is exercised indirectly through phase tests with stubbed print helpers

## Performance Considerations

- **No async I/O for git operations** — all `subprocess.run` calls are synchronous and block the event loop briefly. Acceptable in a CLI context where these calls are short and serialized
- **Per-stack fan-out caps at 4** — `anyio.CapacityLimiter(4)` in `phase_per_stack_reviews` and `phase_fix_parallel`. Hardcoded; could be exposed as `RunConfig` field if larger fan-outs become useful
- **Tree-sitter parsing is eager** — `daydream/tree_sitter_index.py` parses every file in the repo on pre-scan; scales linearly with repo size and could become the bottleneck for very large monorepos. No incremental parsing yet.

## Security Surface

**Inputs from external sources:**
- `gh` CLI output parsed in `daydream/pr_review.py` and `daydream/cli.py` — JSON-decoded; structure assumed
- `codex` CLI output parsed in `daydream/backends/codex.py` from JSONL stream — line-by-line JSON decode with narrow `except`
- Claude SDK message stream — typed by `claude-agent-sdk`

No credential storage in source; all credentials flow through `~/.claude/settings.json` (Claude) or the `codex` CLI's own config (Codex).

**No secrets management vulnerabilities identified.** A scan with the standard secret-pattern regex (`sk-...`, `ghp_...`, `eyJ...` etc.) returns no matches across the source tree.

---

*Concerns analysis: 2026-04-26*
