# Handoff: existential-birds/daydream#1899, a machine-readable committed-review summary

This is preparation context from a read-only `prepare-implement` pass. It lives only on the
preparation branch `claude/great-gauss-l142e8` and must **not** be copied into the
implementation branch or PR. Read it with:

```bash
git fetch origin claude/great-gauss-l142e8
git show FETCH_HEAD:handoff/issue-1899-committed-review-summary.md
```

Paths and line numbers are as of base `dfde53da343ed62094f4e917fb8d7c0aeaab6b7b`. Recheck them
before editing. "Verified" means the preparing agents read the code; "recommended" is a design
choice the implementer may refine for a recorded reason, keeping the decisions in section 2.

## 1. Source, baseline and scope

- **Issue:** https://github.com/existential-birds/daydream/issues/1899. It has no comments or
  amendments, so the issue body is the acceptance source. Related context:
  - #1855 (closed): a manual Sonnet vs Opus measurement that collected these figures by hand.
  - #1900: trace-analysis labels; not a dependency.
  - code-review-benchmarks#30: the harness broke on a removed private import.
  - code-review-benchmarks#42 and #51: the hill-climbing loop; #51 step 7 reads this summary
    once it lands.
- **Baseline:**
  - `origin/main` = `dfde53da343ed62094f4e917fb8d7c0aeaab6b7b` (confirmed with `git ls-remote`
    on 2026-11-17).
  - The preparation checkout was on `claude/great-gauss-l142e8` at that same SHA with a clean
    worktree.
  - The local `refs/remotes/origin/main` was stale at `f636a1b7`; fetch before branching.
- **Acceptance criteria.** One small, schema-versioned summary per committed review, derived
  only from evidence the run already recorded. It contains:
  1. counts: assignments, claims proposed, claims the challenge rejected, findings delivered,
     and targets by decision (including not reviewed);
  2. the terminal analysis and pipeline states with their reason codes;
  3. budget stops and retries, by reason;
  4. per phase: the resolved model and reasoning effort, prompt, cached and completion tokens,
     the cost estimate and wall seconds;
  5. the review profile's name and digest, and the Daydream version.
  6. It is written beside the sealed review (`deep/review-summary.json` in the deep flow),
     offered on `FinalizedReview`, and documented in `docs/runtime.md`.
  7. The resolved per-phase model and effort get **public** names.
- **Constraints:**
  - No benchmark or Harbor dependency, and nothing about reference findings or scores.
  - No new model call and no change to what a review does: dispatch, verdicts, the sealed
    payload and terminal states stay byte-for-byte the same.
  - An unrecorded value is `null`, never `0`.
  - Follow the current-schema rules (section 4).
  - Keep the gate under its 2-minute budget by extending existing tests.

## 2. Decisions (made during preparation; carry them unless the code proves one impossible)

- **D1. Owner module.** Add `daydream/committed_review_summary.py`, in the `committed_review*`
  owner family. It holds:
  - the strict JSON schema (`strict_object` from `daydream/output_schema.py`, `schema_version`
    `{"const": 1}`);
  - a validator that refuses any other version, in the style of `validate_committed_result`
    at `committed_review_model.py:210-215`;
  - one builder.

  Do not name things `*_v1` (see `scripts/check-naming.sh`). Avoid confusion with the existing
  `review_summary: str` PR-comment parameter in `pr_review.py` and `reviews/rendering.py`.
- **D2. Shape (schema 1).** Every leaf that may be unrecorded is nullable:

  ```json
  {"schema_version": 1, "run_id": "<sealed run_id>", "result_schema_version": 10,
   "daydream_version": "0.36.0",
   "profile": {"name": "default", "source_kind": "default", "digest": "<sha256>"},
   "terminal": {"analysis_state": "complete", "pipeline_state": "completed", "reason_codes": []},
   "counts": {"assignments": 1,
              "targets": {"planned": 12, "reviewed": 11, "not_reviewed": 1},
              "claims": {"proposed": 3, "confirmed": 2, "rejected": 1, "unresolved": 0},
              "findings": {"confirmed": 2, "delivered": 1}},
   "budget_stops": {"host_tool_budget_exhaustion": 1},
   "retries": {"total": 1, "by_reason": {"RATE_LIMIT": 1}},
   "phases": {"review":  {"kinds": ["discovery", "follow_up", "cross_assignment"], "backend": "codex",
                          "model": "...", "reasoning_effort": "medium", "prompt_tokens": null,
                          "cached_tokens": null, "completion_tokens": null, "cost_usd": null,
                          "wall_seconds": 12.3},
              "arbiter": {"kinds": ["challenge"], "...": "same keys"}}}
  ```

  Field sources:

  | Field | Source |
  |---|---|
  | `counts.assignments` | `settings.assignment_policy.planned_assignments` (null when absent, for example no-diff) |
  | `targets.planned` | `len(sealed["assignments"])` (one entry per planned **target**) |
  | `targets.reviewed` / `targets.not_reviewed` | count of `terminal_result.stack_outcomes[].target_decisions[].status` |
  | `claims.proposed` | `len(claims)` |
  | `claims.confirmed` / `rejected` / `unresolved` | count of `claims[].disposition` (`rejected` = rejected by the challenge; the validator ties it to a current-revision challenge verdict) |
  | `findings.confirmed` | `len(sealed["findings"])` (after subsumption) |
  | `findings.delivered` | the items handed to the caller |
  | `terminal.*` | copied from `terminal_result` |
  | `profile.*` | `settings.profile_name` / `profile_source` / `profile_digest` |
  | `daydream_version` | `daydream.__version__` |

  On `findings.delivered`: in the deep flow an extension can rewrite the items, so
  `len(items)` may differ from the confirmed count; the console already prints both at
  `review_terminal.py:213-216`. On the explicit path, delivered equals confirmed.
- **D3. Phases use the config names users set** (`[phases.review]`, `[phases.arbiter]`) through
  `REVIEW_PHASES = (("discovery","review"),("challenge","arbiter"))` at
  `committed_review.py:1095`. Evidence labels non-challenge kinds `DaydreamPhase.REVIEW` and
  challenge `DaydreamPhase.CHALLENGE` (`committed_review.py:1642`). Map review to
  `REVIEW` + discovery/follow_up/cross_assignment, and arbiter to `CHALLENGE` + challenge.
  - `backend`, `model` and `reasoning_effort` are the **config-resolved** values from the
    public helpers (D7). They are null when nothing resolves; for example Pi defaults to
    `(None, None)` and the adapter's native default applies.
- **D4. Where it is written.** Every sealed review goes through `write_committed_review_output`
  (`committed_review_output.py:84-92`), so that writer also validates and atomically writes the
  summary to an **explicit `summary_path`**, after the sealed result.
  - **Deep flow** (`deep/review_terminal.py:225-226`): write `state.dd / "review-summary.json"`.
    Files inside `deep/` need no registration: `deep` is a public anchor
    (`artifacts/models.py:17`), the archive copies all of `deep/` (`archive/bundle.py:133-135`),
    and dataset capture uses explicit allowlists, so it ignores the file.
  - **Explicit `run_committed_review`, non-hosted** (`committed_review.py:1311`): write beside
    its top-level `.daydream/committed-review.json`, that is `.daydream/review-summary.json`,
    through the same session route.
  - **Hosted:** register a fourth `FINDINGS_OUTPUT` destination,
    `hosted.output_dir/"review-summary.json"`, next to lines 1320-1336, before recording
    starts (see the comment at 1317-1318). A sibling of the private `result_path` would not be
    delivered (`artifact_visibility.py:467`).
  - The explicit path seals **after** `async with recorder` exits (`committed_review.py:1380-1393`).
    There, read the `recorder` local; `run_usage()` returns `{}` at that point.
- **D5. Where it is offered.**
  - `FinalizedReview` gets a `summary` field. `deliver_finalized_review(result, items, *, usage,
    summary)` (`review_result.py:27-42`) deep-copies the summary like the other fields.
  - `CommittedReviewResult` (`committed_review_model.py:376-392`) exposes the same validated
    value for explicit-path callers.
  - A cancelled deep-flow run still writes the file but delivers nothing, unchanged from today
    (`review_terminal.py:226-234`, `tests/test_runner.py:518`).
  - A summary build or write failure follows the same failure handling as the sealed-result
    write. Do not invent a new partial state.
- **D6. Per-phase usage, time, retries and budget stops come from host-retained local evidence
  available at seal time, not from post-freeze reads.** The issue asks for the summary beside
  the sealed review and on `FinalizedReview`, both of which exist **before** freeze.
  - **Usage (recommended).** Add a per-phase `UsageLedger` map on `EvidenceRecorder`, charged at
    the existing charge sites (`trajectory/invocation.py:367-372` MetricsEvent deltas and
    `:448-450` reconciled usage) keyed by the invocation's phase. Fold it with forks under the **same** rule as
    the root ledger: only sealed forks with steps (`trajectory/scopes.py:107-116`).
    - Invariant to test: the per-phase sums equal the root ledger for every observed dimension.
    - Ledger keys today are `prompt_tokens`, `completion_tokens`, `cached_tokens` (a subset of
      prompt) and `cost_usd`. **Unobserved keys are omitted**, so the summary maps a missing
      key to null.
    - `trajectory/billing.py` owns usage; keep the summary a reader of it.
  - **Wall seconds.** Use the union of the phase's scoped `phase_start`/`phase_end` intervals.
    Every committed work item runs inside `phase_scope(phase, stage=f"committed-{kind}")`
    (`committed_review.py:1644`), which retains scoped events in memory
    (`recorder.py:368-369`, nested fork `phase_events`).
    - Expose a public accessor for `_recursive_phase_event_summaries` (`recorder.py:479-486`).
    - Extract the interval reduction from `compute_timing_summary` (`trajectory/timing.py:63-90`,
      `92-247`) so live and archived timing share one definition. Do not copy it.
    - Null when the phase has no interval.
  - **Retries by reason.** Today each attempt is its own invocation, with
    `host_dispatch.attempt` / `backend_lifecycle.attempt` starting at 1 (`agent.py:696-727`),
    but the **failure class that caused a retry is never recorded** (`agent.py:877`, admitted
    retry `continue` at about 965).
    - Record the bounded `FailureClass` value (`retry_policy.py:18-29`) on the retried
      attempt's invocation summary. This is evidence only, with no behaviour change.
    - Count committed-review invocations whose attempt is greater than 1, by that reason.
    - `total` is null only if attempt numbering is unavailable.
  - **Budget stops by reason.** Use the `ReasonCode` vocabulary `host_wall_budget_exhaustion`,
    `host_tool_budget_exhaustion` and `host_pipeline_budget_exhaustion`. Count:
    - (a) sealed `execution[].reason` entries with those codes: dispatch-level stops, mapped
      through `reason_for_budget` at `review_result.py:81-90`;
    - (b) host **schedule refusals**: refused follow-up rounds at `committed_review.py:1524-1527`
      and the refused challenge re-dispatch at `:1536-1538`.

    Today (b) reaches only the OTel trace (`observability/spans.py:341-348`), never local
    evidence or the sealed payload. Retain them in local evidence through the recorder at those
    two sites, mapping dimension `time` to wall and `tool_calls` to tool. Do **not** add sealed
    `execution` entries: host entries add global reasons and would change terminal states.
  - Document every new evidence fact in `docs/observability.md` (local evidence). Check that the
    trajectory view and the vendored ATIF export (`daydream/benchmarks/`) tolerate the
    additions; do not patch vendored code.
- **D7. Public resolved-phase API in `run_config.py`.**
  - Rename `_resolved_backend_name`, `_resolved_model` and `_resolved_reasoning_effort`
    (lines 109-148) to public names.
  - Add `resolved_phase_settings(config, phases) -> {phase: {backend, model, reasoning_effort}}`.
  - Replace the duplicate at `dataset_capture.py:178-183` (`provenance["resolved_phases"]`) with
    it. The summary uses it too.
  - Update every caller:
    - `runner.py:91-93,149-151` and `runner.py:199,275,308,360,370,387`;
    - `run_artifacts.py:32-33,256-258,316-329`;
    - `committed_review.py:96,1265`;
    - `deep/review_terminal.py:120,130`;
    - `tests/test_archive_data_capture.py:26,158`.
  - Fix the stale docstrings at `deep/settings.py:22` and `run_config.py:133-135` (the latter
    mentions latency routing, which no longer exists).
  - Keep no private aliases. The benchmark harness (`benchmarks/daydream/entrypoint.py`,
    `_configuration`) imports the private names and pins Daydream 0.36.0. Say in CHANGELOG and
    the PR that callers move to `resolved_phase_settings(config, sorted(PHASE_DEFAULT_MODELS[backend]))`.
  - vulture runs at `min_confidence=80`, so a public helper used only externally is not flagged.

## 3. What is in the sealed payload (verified)

- **Builder and validator.** `Review.seal` (`committed_review.py:843-932`) builds the payload,
  and `validate_committed_result` (`committed_review_model.py:210-339`) validates it.
- **Top-level keys:** `schema_version` (`RESULT_VERSIONS` committed 10, immutable 11, local 12,
  worktree 13 at `:128`), `run_id`, `analyzed_revision`, `planned_work_id`, `assignments`
  (targets), `claims`, `findings`, `work`, `terminal_result`, `settings`, `execution`, and
  optionally `immutable_input`.
- **`work`:** records with kind ∈ discovery/follow_up/cross_assignment/challenge and state ∈
  planned/open/accepted/closed/superseded.
- **`execution`:** one entry per dispatch with `assignment` (kind), `work_id`, `accepted`,
  `settled`, `cleanup_complete`, `reason`, `allocation`, `observed_starts` and `overshoot`.
  Host entries have `work_id: null`. Provider retries are **not** recorded here.
- **`settings`:** schema type is a bare `object`. It holds `profile_name`, `profile_digest`,
  `profile_source`, `limits`, `assignment_policy` and `native_contexts[kind] = {backend, phase,
  configuration_identity}`. The configuration identity is an opaque sha256, not the model or
  effort.
- **Daydream version:** not in the payload or the run snapshot. It appears only in evidence
  (`agent_version`, `service.version`).

## 4. Current-schema rules and patterns

- Older committed result versions are rejected with a restart instruction, never converted
  (`docs/runtime.md:573-579`).
- Unsupported schemas are rejected (`docs/runtime.md:16`).
- Archive indexes must be current or empty (`docs/runtime.md:1212-1213`, AGENTS.md).
- Patterns to copy:
  - nullable count `{"type":["integer","null"],"minimum":0}` (`committed_review_model.py:171`);
  - `validate_json_schema` (`json_utils.py:32`);
  - `atomic_write_json(path, payload, trailing_newline=True)`.

## 5. Tests: extend, do not add (AGENTS.md testing standard; `.agents/skills/test-audit/SKILL.md`)

Times were measured serially at base. All of these passed at base: 60 tests in 31.7s, 45 in
63.9s and 24 in 3.1s across the files below.

- **Deep flow, file, and `FinalizedReview` equality.**
  `tests/test_extension_seam_integration.py::test_extension_step_rewrites_items_before_the_review_is_sealed`
  (L34, 1.8s). It already has confirmed 2 and delivered 1. To produce a rejected claim, add a
  claim that a wrapped `_scripted_review` rejects (pattern at
  `tests/deep_orchestrator/test_review_completion.py:179-191`). `StubBackend` `review_verdict`
  applies to every claim.
- **Null-not-zero and resolved model/effort.**
  `tests/test_diagram_overlap.py::test_embedded_review_preserves_backend_dispatch_boundaries[diagram]`
  (L41, 1.3s). It uses Codex defaults and already asserts efforts `["medium","high"]`.
  `StubBackend` emits no usage, so tokens and cost must be null.
- **Recorded versus null on the explicit path.**
  `tests/test_committed_review.py::test_program_retains_work_and_reserves_challenge_after_discovery[cancelled]`
  (L463, 0.24s). Its challenge emits `MetricsEvent(prompt=17, completion=3, cached=None,
  cost=0.002)`, so arbiter should show 17, 3, null and 0.002, and review should be null.
- **Not-reviewed target and budget stop.**
  `tests/deep_orchestrator/test_review_completion.py::test_snapshot_boundaries[live]`
  (L148, 1.0s). Its second half produces `not_reviewed` plus
  `evidence_incomplete, host_tool_budget_exhaustion`. Assert from the file, because
  `ReviewRun` exposes no handler.
- **Pure derivation.** Use the `_challenge_case(..., "grounds")` sealed rejected claim inside
  `test_review_revision_admission_and_sealing_contracts` (`tests/test_committed_review.py:220`).
- **Cancellation.**
  `tests/test_runner.py::test_artifact_session_runner_cancellation_finalizes_then_propagates`
  (L518): the summary file exists with pipeline `cancelled`, and the handler is still not
  called.
- **Retries.** Extend the existing agent retry tests (`tests/test_agent_retry.py`,
  `tests/test_agent_budget.py:438`, `tests/test_retry_policy.py:35`) so the recorded failure
  class appears on the retried invocation. The 14-15s native tests at
  `tests/test_committed_review.py:2771` are not extension targets.
- **Per-phase ledger fold invariant.** Extend the existing billing/fork-fold test; find it by
  grepping `UsageLedger` / `fold` under `tests/`.
- **Timing extraction.** The existing timing tests must stay green unchanged.
- Optionally add `deep/review-summary.json` to the expected tuple at
  `tests/test_archive_data_capture.py:191`.
- Net test count must be neutral, and every changed test needs a credible regression.

## 6. Gate and conventions

- **Static and coverage gates:**
  - mypy `strict = true`; ruff E/F/I/W at 120 columns;
  - vulture `min_confidence=80` over `daydream tests` together, so check non-test callers;
  - branch coverage `fail_under = 90` (base measured about 90.59%); never lower it
    (`docs/coverage.md`).
- **Commands:**
  - `make install`, then `make hooks`: hooks were **not** installed in the preparation
    container.
  - The pre-push hook checks signatures and runs `make check`.
  - Commits must be signed. The container's git config uses SSH signing with
    `/home/claude/.ssh/commit_signing_key.pub`.
- **CHANGELOG:**
  - an `### Added` entry under `[Unreleased]` linking #1899 and the docs anchor;
  - a `### Changed` note for the renamed resolved-phase helpers.
  - This is a `0.y.z` minor addition. No version bump: release work is out of scope.
- **Docs:**
  - a new subsection in `docs/runtime.md` under "Committed-review request/result", covering
    the shape, null rule, locations, `FinalizedReview`/`CommittedReviewResult` and
    schema-version refusal;
  - update the hosted output lists at `docs/runtime.md:635-642,680-681` and the location
    paragraph at `:259-269`;
  - `docs/observability.md` for the new evidence facts.
  - Change AGENTS.md only if its contract text becomes stale.
- **PR requirements:**
  - The PR will exceed 5 files, so run the simplification sweep
    (`.agents/skills/daydream-deadcode-sweep/SKILL.md`, PR mode) before final validation.
  - Use `.github/PULL_REQUEST_TEMPLATE.md` via `.agents/skills/pr/SKILL.md`, and account for
    all checklist items.

## 7. Work lanes (disjoint ownership; others edit concurrently, never revert their changes)

1. **Lane C, config API.** It lands first because it is small and mechanical.
   - Owns: `run_config.py`, `dataset_capture.py`, `runner.py`, `run_artifacts.py`,
     `deep/settings.py`, `tests/test_archive_data_capture.py`.
   - Also renames the calls in `committed_review.py` and `deep/review_terminal.py`, touching
     only those call lines.
2. **Lane B, evidence.**
   - Owns: `trajectory/billing.py`, `trajectory/recorder.py`, `trajectory/invocation.py`,
     `trajectory/scopes.py`, `trajectory/timing.py`, `agent.py`, `docs/observability.md` and
     their tests.
   - Exposes a recorder-level reader that lane A calls: per-phase usage, phase wall seconds,
     retries by reason, and retained schedule budget stops.
3. **Lane A, summary contract and integration** (the one owner of shared contracts).
   - Owns: `committed_review_summary.py`, `committed_review_output.py`, `review_result.py`,
     `committed_review_model.py`, `committed_review.py`, `deep/review_terminal.py`,
     `docs/runtime.md`, `CHANGELOG.md`, and the integration test extensions.
   - Includes the schedule budget-stop call sites in `committed_review.py`, using lane B's
     recorder API.
   - Agree on the recorder reader's signature with lane B first.
4. **Independent read-only review** of the integrated diff with the `code-review` skill, at the
   default level, against this handoff and the issue.

## 8. Risks and open points

- The explicit path's top-level `.daydream/committed-review.json` already depends on ledger
  proof for a non-anchor name (the comment at `review_terminal.py:222-224`). The sibling
  summary shares that property. Verify that a second explicit run is not refused; if it is,
  report it rather than silently moving the file.
- The flow's `pipeline_state` argument is not passed to `seal`. A flow cancelled outside a
  dispatch can seal `completed`. Copy the sealed values; do not "fix" them here.
- `run_usage()` covers every phase so far (diagram, exploration). The summary is per review
  phase, so do not compare it with `FinalizedReview.usage`.
- A cost total can be partial when some invocations were unpriced (`pricing.py:202-219`); the
  ledger sums what it observed. Document that the summary inherits billing's semantics. Do not
  default it to 0 the way `eval/analyzer.py:87` and `pr_comment_renderer._aggregate` do.
- A fork whose seal failed drops out of the totals (`capture_complete:false`). The per-phase
  ledgers must behave identically.
- Codex can report two models in one invocation (#1796). The summary reports config-resolved
  values (D3), not served models.
- The committed program's own challenge re-dispatch after a failure (`committed_review.py`
  around 1545) is host scheduling, not a backend retry. It stays visible through work records,
  not `retries`.
- Daydream's Pi review loop (`daydream . --backend pi --comment --non-interactive
  --capture-data`) needs the Pi CLI, which may be missing in a cloud container. If so, report
  it as a blocker; never claim the review ran.

## 9. Publication and review loop (the full `prepare-implement` requirements)

- Stage only related paths explicitly. Make signed Conventional Commits, for example
  `feat(committed-review): write a schema-versioned review summary (#1899)`. Push with hooks
  enabled and never bypass them; fix failures, including ones that predate this work, or
  report the blocker.
- Open a ready-for-review PR (`Closes #1899`) after `make check` passes.
- Verify the PR's base/head repositories, refs and SHAs. Run
  `daydream . --backend pi --comment --non-interactive --capture-data` from the head checkout,
  appending `--base <verified base SHA>` if the default comparison base differs. Record the
  command, the base/head SHAs, the resolved merge-base and the result.
- Then run `fetch-pr-feedback --include-author --include-self`, then `receive-feedback`, then
  `respond-pr-feedback`. Cover every reviewer and bot, including `greptile-apps`, across
  inline threads, submitted reviews and general comments.
- Verify each finding against the code. Fix the confirmed ones, rerun the affected checks,
  commit and push, and reply with evidence. Rerun Daydream on the new HEAD.
- Recheck queued or in-progress bot runs. An empty fetch does not prove a review succeeded.
- **Done** means:
  - all acceptance criteria are met;
  - local checks and required CI pass on the latest HEAD;
  - the independent and Daydream reviews are done and their findings addressed;
  - the PR is open and ready for review, and the worktree is clean.

  Report the PR URL, final HEAD, checks and feedback dispositions. Stop before merging.
