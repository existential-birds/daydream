# Implementation plan: #1677 bounded large-diff source navigation

Temporary handoff file for the `/goal` session. Do not include it in the PR.
Prepared read-only against `main` at 9d4d155ed01d6f164719bf674d756c6e2cbac7c7.

## Source and scope

- Repo: existential-birds/daydream. Issue #1677 "Review very large diffs through
  bounded source navigation" (no comments; body is current).
- Authority to read:
  - parent #1616;
  - accepted spec https://github.com/existential-birds/daydream/issues/1667#issuecomment-6305190550;
  - Source resolution https://github.com/existential-birds/daydream/issues/1668#issuecomment-6298352302;
  - broad-reviewer policy https://github.com/existential-birds/daydream/issues/1665#issuecomment-6304553345;
  - core decision https://github.com/existential-birds/daydream/issues/1665#issuecomment-6304647286.
- Reread AGENTS.md, CONTRIBUTING.md, Makefile and pyproject.toml.
- Read docs/runtime.md sections "Committed-review request/result", "Captured local
  source", "Pi host-tool lifetime and limits" and "Terminal review findings contract".
- Read docs/extensions.md sections "Stage-aware review builders (API 8)",
  "Invocation-scoped Pi host tools" and "Working artifact paths".
- Read docs/observability.md (Pi lifecycle and billing).
- Delivery order: #1677 is position 7 of 29. Predecessors are merged: #92; #1645
  via PR #1706 (88a3e47); #1619 via PR #1728 (d6c233b); #1676 via PR #1732 (9d4d155).
- Out of scope:
  - #1628 (adaptive splitting and cross-assignment investigation; next);
  - #1711 (queued-scope memory);
  - #1679/#1641 (CLI cutover and delivery);
  - paid model, judge or benchmark runs;
  - credential, provider or settings changes.
- **Target:** the fresh core `daydream/committed_review.py::run_committed_review`,
  both committed and `local=True`. Its only caller today is Harbor
  (`daydream/benchmark/harbor/entrypoint.py:183`).
- The ordinary CLI (`daydream . --backend pi [--review|--comment]`) still runs the
  legacy deep pipeline over the captured root (runner._dispatch → _run_loop_deep →
  deep.orchestrator.run_deep).
  - Do not add features to the deep pipeline, and add no legacy/candidate selector.
  - The deep path shares StageInputFactory, review_source_refs, host_tools*, the Pi
    extensions and HOSTED_PI_TOOLS, so preserve its behavior and tests.
  - Bump STAGED_REVIEW_CONTRACT=15 (`review_investigation.py:68`) deliberately, and
    only if deep cache inputs change.
- The issue's verification list names `tests/test_hunk_index.py`. It was deleted in
  #1696; its coverage now lives in
  `tests/deep_orchestrator/test_full_change_risk.py::test_dropped_sensitive_block_routes_from_full_captured_change`.

## Findings at 9d4d155 (verify before relying on them)

1. **Hard refusal.**
   - `Review.__init__` (committed_review.py:93-98) refuses plans with more than
     COMMITTED_ASSIGNMENT_LIMIT=512 parts (committed_review_model.py:26).
   - The same 512 appears as discovery `targets` maxItems, result `assignments`
     maxItems and TARGET_DECISIONS_MAX_ITEMS (review_result.py:107).
   - `StageInputFactory._parts` (review_stage_inputs.py:244) creates one part per file
     if the file renders within 24 KiB (`_EXACT_PATH_ASSIGNMENT_MAX_BYTES`). Otherwise
     it creates one part per hunk (`_split_file`), and oversized hunks become UTF-8
     continuation fragments (`_segments`).
   - Parts are never packed together, so any change with more than 512 files is
     refused before dispatch. `tests/test_committed_review.py:411-483` triggers this.
2. **Unbounded inline inventory.**
   - `_dispatch` (committed_review.py:665+) puts canonical_json of every assignment
     (source_refs, source_options, source_summary) into the user prompt. Estimates:
     about 1.6 KB for a one-hunk file, 0.5–1.2 MB for 300 files.
   - The only sanctioned input is a pointer to the whole private `diff.patch`
     (prompt_budget.py:27, 128 MiB). `hunk-index.json` is never named to the model, and
     no locator maps a target to a patch offset.
   - Deep-only `StageInputFactory.prepare()` and `_supporting_catalog` provide paged
     catalogs with explicit omission, but they are unused here.
   - Pi compaction can summarize away user-turn content. Only `--append-system-prompt`
     content survives (pi.py:975-990): `_INSTRUCTIONS`, the dispatch snapshot and
     REPOSITORY_SEARCH_GUIDANCE.
   - Deep review keeps its access guide in system instructions
     (`phases/review_prompts.py:20-44`) precisely so it survives compaction.
   - REPOSITORY_SEARCH_GUIDANCE (prompts/grounding.py:30-51) tells the model to run
     `daydream search`, which read-only Pi cannot run.
3. **All-or-nothing discovery.**
   - One submission must decide every target_id exactly once, together with claims
     (committed_review.py:176-180), guarded by `_discovered`.
   - The canonical result is capped at 128 KiB (`_MAX_RESULT_BYTES`, host_tools.py:27-29),
     which allows roughly 265 reviewed targets. The challenge-side check also counts
     discovery targets.
   - If discovery fails before submitting, nothing is admitted and earlier findings
     are lost.
   - Unfinished targets are implicit (assignments minus decisions), not typed per target.
4. **Capacity.**
   - Discovery gets `limits.tool_calls // 2`, without diff scaling
     (`review_scale_for_diff` is deep-only); challenge gets
     `budget.remaining_tool_calls`.
   - The enforcing ledger is `HostToolBinding.admit` (host_tools.py ~269), called via
     `/admit` from `pi_host_tools.ts` on every `tool_execution_start`. A second ledger
     is `ReviewInvestigationBudget.observed_tool_starts` (agent.py:743; `missing_starts`
     reconciliation in the finally block near 1001).
   - `GET /status` (host_tools_server.py:183) returns only `{ready, remaining_ms}`, and
     only to the extension. No model-visible tool-start count exists.
   - Prompts carry dispatch-time snapshots: agent.py:305-317, the retry snapshot at
     679-684, `render_pi_preamble` in pi.py:249-272, and committed `settings.limits` in
     the user prompt.
   - A one-time system-message warning fires at exactly 1 remaining start
     (`pi_structured_output.ts` `context_with_system`).
   - Recurring budget messages were removed deliberately in #1599, and tests assert they
     are absent.
   - Suspected leak: the over-limit start is charged before the check, so discovery may
     consume //2+1 starts and eat into the challenge reserve. Prove or refute it with a
     test.
5. **Native compaction.**
   - Daydream sets no compaction options.
   - Pi 1.4.0 auto-compacts when context tokens exceed contextWindow − reserveTokens.
     Defaults are reserveTokens 16384 and keepRecentTokens 20000
     (dist/core/compaction/compaction.js).
   - Compaction runs between turns. The summary request goes to the same provider with
     the system prompt "You are a context summarization assistant".
   - `compaction_start`/`compaction_end` are not in `_PI_EVENT_TYPES` (pi.py:121), so
     they are silently dropped, and summary usage is not billed.
   - Committed runs use `--no-session`. A retry starts a fresh conversation while the
     host ledger persists.
   - No whole-invocation context or output accounting exists. If maxTokens is too
     small for a submission, the run stops with `length` and nothing is admitted.
6. **Stock tools.**
   - The tool set is read,find,ls,grep,structured_output (pi.py:142 and :994-995,
     `HOSTED_PI_TOOLS` in hosted_review.py:19). The `pi_host_tools.ts` barrier checks
     this exact set, and docs/extensions.md:175-182 says "no additional model tool".
   - `read` returns at most 2000 lines or 50 KB. A single line over 50 KB returns only
     a "Use bash" hint, and bash is disabled. So very long multibyte lines are
     reachable only through bounded fragments.
   - `grep` returns 100 matches, cuts lines at 500 chars, caps output at 50 KB and
     respects .gitignore.
   - Tools are rooted at the capture but resolve absolute paths and `..` freely.
     Claim no confinement, and do not make navigation depend on reading private
     directories.
   - `CapturedSource.validate` rejects extra files in the capture, so navigation
     artifacts belong in the private stage-input session.
7. **Line numbers in large hunks.**
   - Continuation refs have a null source_range plus `fragment{offset,bytes,sha256}`,
     relative to the hunk body.
   - `_claim` (committed_review.py:126-132) requires the claim's line to fall inside a
     non-null range. So a claim in a large single-hunk target, such as a new
     20k-line file, cannot carry a line number.
   - Assess validating a line against the fragment's summarized complete-line span
     without changing canonical fragment identity. Otherwise document the limitation.
8. **Other risks.**
   - The committed path builds StageInputFactory on the event loop. The deep path
     builds it off-loop, and the docs warn that full checks on the loop starve
     host-tool timers.
   - `_segments` re-encodes the remaining text on every iteration, which is quadratic
     for multi-MB hunks.
   - Paths over 512 chars break the `_ID`/`_TEXT` schemas.
   - Inferred, not verified: `parse_hunks` may misread a removed line whose content
     starts with "-- " as a header.

## Required behavior (design decisions)

- **Population before bounding.**
  - Derive the inventory from the full captured patch before any prompt bounding.
  - Replace the 512 refusal with bounded navigation. Any remaining finite bound must
    come from real retained-output, transport or capture limits, be documented and
    versioned, and refuse before model work, never mid-run.
  - Use no decomposition threshold. ~20k lines must work with one broad discovery
    conversation plus host-scheduled fresh challenge.
- **Bounded catalog.**
  - Replace the inline inventory with a bounded initial catalog that is explicitly
    partial (status and omitted counts).
  - Put a stable, compact navigation policy and catalog root in system/developer
    instructions, so they survive compaction and retries.
  - Serve verbose attribution only through targeted lookup.
- **Targeted lookup (recommended design).**
  - Add one read-only host-backed lookup tool. Register it in the existing Pi
    extension and serve it from `PiHostToolServer` through the existing binding, so
    `/admit` charges it to both ledgers.
  - Look up by target_id or changed path, with an opaque continuation cursor.
  - Return a compact mapping plus the exact captured bytes of the part or fragment,
    paged, with explicit next/omitted/unavailable fields.
  - Serve only frozen canonical inputs and accepted `CapturedSource` bytes. Never live
    checkout bytes; follow no symlinks or submodules.
  - Deny lookups after authority closes. A lookup never mutates review state.
  - Each response also reports current host capacity, labelled as distinct from
    dispatch-time snapshots: remaining starts after charging this lookup, remaining
    time to the original deadline, and the protected reserves.
  - Adding a model tool is a contract change. Update HOSTED_PI_TOOLS, `--tools`, the
    extension's exact-set barrier and `selected_tools_count` (pi.py:1207-1211).
    Version the hosted policy `pi-read-only-v1` if its semantics change. Update docs
    and the changelog.
  - A file-based alternative (part files and catalog pages in the private stage-input
    directory) is acceptable only if it meets the targeted-lookup and
    current-capacity criteria.
- **Progressive admission.**
  - The single discovery conversation submits decisions for subsets of targets, plus
    claims, across multiple submissions.
  - Each submission is validated as a whole and installed through the same
    synchronous closure-first path, charged to the same ledger, and answered with a
    compact receipt rather than the whole retained state.
  - Completion requires every target to be reviewed. Unsubmitted targets stay typed
    incomplete per target, and earlier admissions survive later omission,
    unavailability or failure.
  - Discovery cannot consume protected challenge or finalization allowances.
  - Reassess TARGET_DECISIONS_MAX_BYTES (256 KiB per scope, 8 MiB per run) and the
    schema maxItems.
  - Version the result and terminal schemas deliberately (currently committed 1,
    immutable 2, local 3, terminal 2), and preserve the `planned_work_id` binding.
- **Whole-invocation preflight.**
  - Before model work, check that the instructions, a catalog page, the largest
    single tool result, a reserve and the largest bounded submission output all fit.
  - Take effective model limits from native evidence. For example, the extension's
    readiness handshake could report `ctx.model` contextWindow/maxTokens; verify this
    exists in Pi 1.4.0.
  - Fail before model work with an actionable message when the limits do not fit.
  - Never write models.json or settings, and never claim a setting is pinned without
    native evidence.
- **Compaction evidence.**
  - Retain compaction lifecycle events as native evidence instead of dropping them.
  - Bill summary usage where it is exposed, per docs/observability.md.
  - Compaction or a successful response is not completion evidence.
- **Search guidance.**
  - Fix the search guidance for read-only committed Pi so it names only tools the
    model can actually run.
  - Reuse `commands/search.py` limits only if a capture-bound search mode is added.
- **Preserve:**
  - the native Pi loop and its configuration/authentication, with nous /
    deepseek/deepseek-v4.1-flash;
  - no synthetic user turns and no polling-based enforcement;
  - restart-only stopped reviews;
  - extension API 8, unless deliberately versioned;
  - atomic findings+coverage publication and the terminal findings contract;
  - "correspondence is not a read receipt".

## Implementation lanes

Workers own disjoint files. Tell every worker that others are editing the codebase
and that they must accommodate those changes without reverting them.

- **Integrator (goal session).** Settles the shared contracts before parallel work:
  committed_review_model.py schemas and limits, the lookup request/response, the
  progressive-submission schema and ledger semantics. Also owns CHANGELOG
  `[Unreleased]`, the docs (docs/runtime.md, docs/extensions.md,
  docs/performance/captured-source.md) and final integration.
- **Review/source lane.**
  - committed_review.py: Review, planning, `_dispatch`, `_INSTRUCTIONS`.
  - review_stage_inputs.py: catalog, navigation index, lookup projection, off-loop
    construction.
  - review_source_refs.py, and the review_result.py budgets.
- **Native transport lane.**
  - host_tools.py, host_tools_server.py.
  - backends/pi_host_tools.ts, backends/pi_structured_output.ts.
  - backends/pi.py: events, tools, preflight.
  - hosted_review.py, and the agent.py snapshot labels.
- **Tests lane.** tests/test_committed_review.py, tests/test_host_tools.py,
  tests/test_pi_native_output.py, tests/harness/native_pi.py,
  tests/test_captured_source.py, and the tests/deep_orchestrator capacity tests.
- **Review.**
  - An independent read-only review of the integrated diff, using the `code-review`
    skill.
  - The `test-audit` skill (.agents/skills/test-audit/SKILL.md) in a separate
    fresh-context agent. Address its findings before publication.

## Validation

Run `uv lock --check` first, then `make install` and `make hooks`. Focused checks:

```bash
uv run pytest tests/test_deep_diff_population.py tests/deep_orchestrator/test_full_change_risk.py tests/test_prompt_budget.py tests/test_repository_search.py
uv run pytest tests/deep_orchestrator/test_review_capture_and_retry.py tests/deep_orchestrator/test_review_native_capacity.py tests/deep_orchestrator/test_review_stage_reservations.py
uv run pytest tests/test_committed_review.py tests/test_captured_source.py tests/test_host_tools.py tests/test_pi_native_output.py tests/test_backend_pi.py tests/test_pi_review_system.py -k 'not live_pi_smoke'
git diff --check
make check
```

- Run `make rl-check` only if rl/daydream_review changes. Never lower the coverage
  floor (fail_under 90; see docs/coverage.md).
- **Starting tests to extend:**
  - `test_oversized_hunk_assignments_reassemble_every_canonical_byte`
    (test_review_capture_and_retry.py:384);
  - `test_late_cross_file_judgment_fits_cumulative_native_budget` and
    `test_structure_keeps_bounded_interaction_review_and_patch_navigation`
    (test_review_native_capacity.py:50, :107);
  - `test_committed_review_native_challenge_uses_fresh_captured_source_context`
    (test_pi_native_output.py:134);
  - the ledger tests at test_pi_native_output.py:1448 and :1557;
  - `seed_feature_branch` in tests/harness/git_helpers.py.
- **Proof A: real temporary Git captures.**
  - Build ~20k changed lines in two shapes: more than 512 files, and one large file.
    Include a late target and a multibyte line over 50 KB.
  - Assert the inventory equals an independently computed full population.
  - Assert lookups reach late bytes exactly (the reassembled sha256 equals the
    canonical one), without aliasing.
  - Omit or fail late work. Assert that earlier admitted findings are retained, that
    unfinished targets are typed per target, and that the result is no approval.
- **Proof B: installed Pi with the loopback provider.**
  - Use a 96-start discovery allocation: `ReviewLimits(tool_calls=192)`, or
    `run_agent(tool_call_budget=96)` as in the test at :1557.
  - Trigger compaction deterministically after 52 completed starts:
    - set compaction.reserveTokens/keepRecentTokens and/or contextWindow in the
      isolated `PI_CODING_AGENT_DIR` (fixture at test_pi_native_output.py:59-79);
    - and/or add per-request usage control to `_NativeProvider`, keeping the default
      10/5/15 that :766 asserts.
  - Have `choose` answer the summarization request with plain text.
  - After compaction, request a target that was accessed earlier and whose locator now
    lies outside the retained detail.
  - Before the next charged start, assert host `remaining_tool_calls == 44`. Then
    assert that later lookups and submissions decrement it without any renewal.
  - Cover an unavailable selection: it keeps its unfinished responsibility, and prior
    admissions are retained.
  - Cover the suspected over-limit leak.
  - Scripted models prove software behavior, not review quality.
- **Environment notes.**
  - The native fixture calls `pytest.fail` when pi or node is missing; keep that
    explicit.
  - The preparation container had Node 22.23.4 (/opt/node22) and Pi 1.4.0. CI pins
    Node 24.18.1, and the hosted rows need it. Install it per CONTRIBUTING, or report
    it as unavailable.
  - Docker was unavailable, so actionlint skips; note that.
- Each new or changed test must name its regression, its owning seam and why existing
  coverage misses it. Report failures and skips honestly.

## Publication and review loop

- **Deadcode sweep.**
  - Before any PR, run the `daydream-deadcode-sweep` skill
    (.agents/skills/daydream-deadcode-sweep/SKILL.md), which CONTRIBUTING requires.
  - It was absent at preparation. If it is still missing, report that blocker before
    creating the PR.
- **Diff check.** Inspect and simplify the full base-to-head diff.
- **Commits and push.**
  - Stage related paths explicitly and make signed Conventional Commits.
  - Push with hooks enabled. Never use --no-verify or any other bypass.
- **Changelog.** Add a CHANGELOG `[Unreleased]` entry linking #1677. Assess SemVer
  under 0.y.z (the current version is 0.32.0).
- **PR.**
  - Use the `pr` skill with .github/PULL_REQUEST_TEMPLATE.md, and include "Closes
    #1677".
  - Open the PR ready for review after the checks pass.
  - In the PR, give a keep/replace/defer note for the 512 limit, the inline inventory
    and single-shot discovery.
  - The Test Plan lists actual test nodes, tests removed or consolidated, and the
    sweep outcome.
- **Daydream review.**
  - Verify the PR's base/head repositories, refs and SHAs.
  - From the head checkout, run
    `daydream . --backend pi --comment --non-interactive --capture-data`, installed as
    described in the README Quick start.
  - Keep nous / deepseek/deepseek-v4.1-flash, and make no credential changes.
  - Append `--base <verified SHA>` only if the default comparison base is wrong.
  - Record the command, the base/head SHAs, the merge-base and the result.
  - Missing credentials or a review failure remains a blocker until a successful
    review covers HEAD.
- **Feedback.**
  - Run `fetch-pr-feedback --include-author --include-self`, then `receive-feedback`,
    then `respond-pr-feedback`.
  - Cover every reviewer and bot, including `greptile-apps`.
  - Verify each finding against current code, fix it, rerun checks and push with
    hooks. Then reply and resolve the eligible threads.
  - Rerun Daydream after changes, and refresh feedback and required CI on the latest
    HEAD.
  - Wait for known queued bot runs. An empty fetch is not proof that review succeeded.
- **Completion.**
  - Merge-ready means:
    - all acceptance criteria are met;
    - local checks and required CI pass on the latest HEAD;
    - the reviews and the test-audit are done, with findings addressed;
    - the PR is open and ready for review;
    - the working tree is clean.
  - Report the PR URL, final HEAD, the checks that ran and that were skipped, and the
    feedback dispositions. Flag the owner, and stop before merging.
