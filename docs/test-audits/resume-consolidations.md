# Final resume consolidation and runtime evidence

Baseline: `6c48d368d529a89748c93e7bb61ae9ac6566936e`.
Collection at `3078954b`: **8,197 cases / 5,360 functions**, down **807 cases / 71 functions (8.96%)**.
All prior flow/shared keeper mappings remain applicable. These two additions remove two repeated real runner invocations; no production, fixture, transport, coverage, skip, workflow or worker setting changes.

## Validation and independent preservation review

Both plans below were reported before editing and independently approved.
The candidate's exact six selected cases pass. For the arbiter transfer, an isolated mutation of `_apply_adjudication_verdicts` ignores only arbiter descriptions: actual dispatch and completion marker assertions pass, then the transferred first-run `ARBITRATED:` delivered-report assertion fails. Exact production bytes restored; keeper passes. The control checkout was restored clean and removed normally. Final independent diff/AST review confirmed unrelated tests and the merge body unchanged, and both real resume calls retained.

Focused owners and shared consumers: precision, merge recovery, latency routing, review reuse, backend/stub/Git harness; **137 passed, five warnings, 44.04s pytest / 44.43s wall**, Python 3.12.13, n4, same locked dependencies, no coverage. Collection: 8,197 / 5,360. Before/after selected-run wall observations (37.51s / 8.63s) include startup/cleanup/load differences; retained call durations remain similar. Do not attribute that wall difference to these two removals.

## CI work bound and remaining goal

Temporary built-in complete duration reporting at `0866523dc6d2eb69bc10e2ce74924747dfc764b6` collected actual hosted phase work. [CI run 37566055439](https://github.com/existential-birds/daydream/actions/runs/37566055439) passed: **8,194 passed / five skipped / 72 warnings, 363.99s pytest**, Python 3.12.14, n4, unchanged branch coverage. Check job **432s (7m12s)**; RL job 123s, 192 passed / three skipped / 86.93s. Verbose reporting itself adds terminal output overhead, so this is diagnostic evidence rather than final concise-report performance.

`gh run view --log` returned a truncated duration block. The complete check-job raw log, fetched through the job logs API, contains **24,592 phase rows: 8,199 setup, 8,194 call, 8,199 teardown**. Printed durations sum to **1,262.80 worker-seconds**. Six phase/node pairs collide because GitHub masks distinct credential parameters; exclude all 12 affected rows from the lower bound rather than assume identities. For every remaining row, subtract 0.005s for rounding, clamp at zero, sum, and divide by four: **309.00375s minimum execution time**. This excludes collection, coverage/reporting and other gates. It rules out reaching 240s through queue scheduling alone on this measured workload; the two small removed runs do not close that gap. Temporary `--durations=0 --durations-min=0` is restored to `--durations=40` in the final change.

A further constant randomized registry-prefix proposal preserved source contracts and removed false numeric parses in an external mechanism probe, but actual 8,199-row autouse workload showed no improvement (38.55s current / 39.81s constant, same Python/n4/dependencies/machine/no coverage). It was rejected; exact source restored and control checkout removed. This is neither a suite speedup nor a production change.

Additional bounded discovery inventoried 125 high-cardinality groups / 1,589 cases; selected groups received complete owner/history audits and the expensive deterministic checks, handoffs, cache damage and fix isolation inputs. No large proven redundancy was established. Independent fault, authorization, path/privacy, transport, cancellation, persistence and shipped-byte contracts remain. Cheap table wrapper reduction is not a performance result. The under-four-minute objective remains unmet; PR remains draft.

## Exact candidate evidence recorded before editing

# Bounded remaining Git-heavy discovery

Read-only at9f62e468ca7a8acca139c99bd1299906aca90b31 while normal push hook runs. Baseline6c48d368d529a89748c93e7bb61ae9ac6566936e. No edits/tests/uv/dependency changes. Complete selected tests, fixtures and candidate merge production path read; this is bounded discovery, not a new complete module campaign.

## C1: one exact redundant cold merge row

Proposed affected node:
`tests/test_deep_merge_recovery.py::test_merge_salvage_and_resume_preserve_findings_and_failure_context[I could not produce a JSON item list for the merged cross-stack findings. The per-stack reviews completed, but no consolidated item list was emitted.-inspect]`

Named keeper:
`tests/test_deep_merge_recovery.py::test_merge_salvage_and_resume_preserve_findings_and_failure_context[I could not produce a JSON item list for the merged cross-stack findings. The per-stack reviews completed, but no consolidated item list was emitted.-fix]`

Exact parameter tuples are `(ARCHIVED_MERGE_STR, "inspect")` and `(ARCHIVED_MERGE_STR, "fix")` of the same declaration. Delete only the former tuple; no assertion/body/fixture/owner changes.

Both rows execute identical statements before the `if resume == "inspect": return` branch: silence UI; mute publication/healing/commit boundaries; install the same StubBackend with exploration off; set parse_severity=high and byte-identical reconstructed archived merge prose; call actual _run_deep with its identical default RunConfig; then assert nonzero controlled result, nonempty salvaged items without obsolete partial field, rendered merged report and per-stack records, coverage failed merge status, synthesis_failure reason, and exact CROSS_STACK_MERGE_ERR_MSG diagnostic. The removed row adds no assertion or stimulus. Keeper completes all these assertions before changing TTY/CI/prompt behavior for real fix resume, then proves PARTIAL warning/no re-review/no remerge and salvaged finding prompts.

Fixture chain: multi_stack_target -> _feature_repo performs actual unique init/config/add/basecommit/checkout/featurecommit for Python/React/Markdown; function-scoped monkeypatch/capsys/mute_side_effects and ordinary autouse environment/credential/artifact/archive/recorder isolation are identical for both rows. `_run_deep` calls actual runner.run with same config and target, start_at=review, cleanup=False, default precision/profile/cache options. Stub changes only backend external model execution. First-run assertions do not depend on the later resume argument. Failed healing/commit/publication are muted equally; this proposal does not claim those boundaries are proved here. Their genuine real Git/process/hook keepers remain retained in fix isolation/related regression/exhausted verdict suites.

Owners/callers: deep flow FlowStep cross-stack-merge -> _step_cross_stack_merge -> phase_cross_stack_merge -> real run_agent schema handling yields StructuredOutputFailure from unparseable TextEvent + ResultEvent(None); CrossStackMergeError -> coverage failure/lifecycle -> _salvage_merge_failure -> UID-based dedup and host normalized items/report -> Stop(1). Complete phase_cross_stack_merge and CrossStackMergeError read, merge stage/recovery/load-items code and registered flow callers read. Resume keeper later uses the same persisted artifact contract. No production/support deletion or extension contract is unlocked.

History: d31cb99d (#764/#361) pinned the archived error and folded standalone archive-shape variants into parameter siblings, preserving the exact error and PARTIAL resume warning. e698ba7d (#1440) combined old cold salvage and resume tests into the current six-row table and moved all cold diagnostic/item/report assertions into the common prefix. That strengthening makes the archive-only inspect row an exact subset of archive fix. Preserve the archived phase-level exception row too; it verifies structured stack context at the phase boundary, which this cold prefix does not assert.

Risk is low: one tuple removed, both cold input and every assertion retain their existing same-body real-run home. Collection becomes five rather than six rows, zero declarations removed. Focused validation if root authorizes: complete tests/test_deep_merge_recovery.py plus relevant review/resume/shared-harness consumers; final make check and independent row-preservation review. No tests run here. Existing final-full JUnit: removed row0.863s, keeper3.562s; aggregate testcase seconds are not wall time and not a CI speedup result. This tiny cut does not explain or achieve CI<240s.

## R: distinct remaining scenarios

- `test_ephemeral_failure_handoff_projects_public_refs_without_private_paths`: retain all seven destination/response rows. Default/custom-public/external trajectory destinations drive different private/public publication and exact partial path semantics. Clean/known-leaf/unknown-private authored bodies drive preserving structured model output versus fail-closed host fallback. Shared success label or same fake does not equate these inputs. Complete selected test and its generated reading stub assertions read; the real runner reads live trajectory partial bytes and retained fix edit, then validates durable handoff paths and archived/partial bytes. No matching keeper with all same destination/response inputs found.
- `test_legacy_cache_without_coverage_proof_recomputes_review`: retain all nine damage inputs legacy/corrupt/incomplete/head/diff/scope/origin/partial_payload/malformed_payload. They remove proof, corrupt typed proof/lineage, change snapshot/scope binding, or preserve digest while corrupting typed payload/partial semantics. ReuseCache + ReviewReuseUnit + lookup_reuse_entry gate these independently; helper-only negative tests cannot replace actual warm->tamper->recompute runner proof. One shared warm run plus nine fault runs would require restoring mutable store/payload/provenance state and serialize presently parallel work; no evidence of wall-time improvement or equivalent isolation. No cache/template proposal.
- Fix isolation: retain shell success/failure/timeout rows, publishing real checks/hooks row, staged+unstaged owner edits, three raw-index flag rows, and tracked-directory deletion. The expensive publish row starts with different owner tracked state and retains actual host checks/commit/push hooks/remote HEAD; it cannot replace muted-success dirty-owner restoration. Failure and cancellation have distinct publication/rollback outcomes. Exact raw index staged/intent-to-add/assume-unchanged and real replacement symlink/ignored owner modes are independent contracts. Full test file/FixIsolationRound read.
- Related regression: retain0600/0644 new related test modes. Same real runner convergence/fix/heal/archive proof also observes exact new-file permissions, scratch ownership/restoration, retained tree evidence keys, actual bare remote push and unavailable remote-CI handoff. Inputs and visible modes differ. Exhausted verdict rows unresolved/wrong_target/regressed retain distinct verifier/coordinator outcomes, actual host test and both commit/push hooks on publish, versus no tests/hooks/HEAD change for regressed. Full test file read. Histories0f564c47 (validated fixes after exhaustion), daffb6e2 (#1432 fix validation), e698ba7d (terminal coverage) substantiate retention.
- Merge recovery other rows retain generic inspect literal, generic/archived interactive fix, merge rerun, fix declined, public synthesis/projection resume, dedup UID collision/pre-filter/source attribution and coverage persistence failure. Inputs/real resume choices/fault boundaries differ. No additional exact matching expensive keeper established.

## Accounting

Only C1 proposed: one case, zero functions, no support/production changes, no changed coverage/skips/settings. No edits applied. No broader runtime reduction or percentage claim. Prior no-CI-wait/fake-gh import changes untouched.

# Precision/deterministic follow-up: one remaining exact real-run duplicate

Pinned baseline `6c48d368d529a89748c93e7bb61ae9ac6566936e`. Read-only investigation; no tests, profiles, checkout edits or mutation performed. Current precision module is unchanged from baseline. Current inventory: precision19declarations/52cases; deterministic9declarations/19cases.

## Selected candidate P-R1

**Consolidate one singleton and one genuine real runner invocation.** Exact removed node:

`tests/deep_orchestrator/test_precision_budgets_and_tiers.py::test_merge_resume_reruns_arbiter_when_marker_absent`

Named keeper:

`tests/deep_orchestrator/test_precision_budgets_and_tiers.py::test_merge_resume_skips_arbiter_when_marker_present`

This is a small verified opportunity, not the hypothesized30% quota or proof of target runtime closure. Do not include the cheap opt-in table or independent handoff/verification rows in the edit.

### Matching executed inputs and complete bodies

Both tests use the same fresh `multi_stack_target`, exact `_silence`, `_install_model_capturing_stubs(...merge_echo_records=True)`, `_prime_merge_resume_records(...python_severity='high')`, then `_run_deep(...start_at='merge')` with all other config arguments/defaults identical. No fixture-dependent scenario parameter differs. Original removed node makes one real run. Existing keeper already makes the same first real run, checks actual generated marker and nonempty arbiter calls, clears calls, then makes its genuine second merge-resume run and checks no arbiter calls. Its first run is the exact matching keeper; second run must stay intact.

| Removed regression/assertion | Existing keeper proof / exact transfer |
| --- | --- |
| An interrupted prior review has high unarbitrated records and no completion marker | Transfer `assert not (deep/'adjudication-complete.marker').exists()` immediately after priming, before first run. This is an input/positive-control requirement, not fabricated completion state. |
| Marker absence must dispatch arbiter and first merge resume succeeds | Existing first run asserts exit0 and `any('you are the arbiter'...)`. Same nonempty call discriminator covers original list truthiness; no weaker helper/mock invocation substitute. |
| Actual arbitration writes completion marker | Existing `is_file()` immediately after first real run. No stub/fixture supplies marker. |
| Arbiter-revised finding reaches delivered review output | Transfer exact `.review-output.md` read and `ARBITRATED:` assertion immediately after first marker/call assertions **before calls.clear and second run**. Unique claim must not be checked only after replay. |
| Existing keeper's separate valid-marker replay skips arbiter | Preserve its second actual runner invocation, call-log clear and exact empty arbiter-call assertion unchanged. Do not replace marker with fixture bytes or infer calls from the first run. |

Proposed keeper order: prime→assert marker absent→first actual run0→marker is_file/nonempty arbiter calls→read delivered report/assert revision→clear actual call log→second actual run0→empty arbiter calls. Only then delete old singleton. No production/shared-harness changes; precision module19→18declarations,52→51cases. Nothing parameterized is removed.

## Complete fixture, support and production evidence

Root `multi_stack_target` builds and commits real Python/React/README main+feature diffs through `_feature_repo` and real Git helpers. Root archive/artifact/private configuration/credential/trace/ContextVar/signal fixtures isolate actual per-test state; both tests share identical fixture chain. `_silence` only suppresses UI and responds to interaction (accept intent/decline later gates); it does not replace arbitration, record pool, persistence, report rendering or runner.

`_install_model_capturing_stubs` substitutes only backend construction and exploration availability. Every actual arbiter backend result reads the host-written arbiter input JSON, returns keep/identity verdicts with `ARBITRATED:` description. `merge_echo_records=True` reads actual rewritten per-stack envelope bytes to produce merge records. This is an external model boundary: it does not write the completion marker or delivered report.

`_prime_merge_resume_records` supplies required typed historical resume prerequisites: high/HIGH Python record plus React/generic/structure records. `_prime_merge_resume` creates current diff key through actual Git diff/base/head resolution; derives current planned stack scopes, stamps UIDs, writes intent/alternatives/envelopes and typed revision-bound review coverage. It never writes an adjudication marker. This is legitimate on-disk resume input, not a fixture supplying the output under test.

`_run_deep` builds real RunConfig(target,start_at='merge',cleanup=False,precision_mode=False,approve_on_clean=False,review_profile=None,review_cache_enabled=True) and calls production runner.run. Actual deep FlowStep order invokes `_step_arbiter` before cross-stack merge. `_step_arbiter` validates completion contract through `_completed_adjudication`, including coverage phase status, marker JSON, whole/group plan execution contract, current policy/profile/route/record digest. Missing marker cannot certify completion. Actual selected high/contested targets are sent through phase_arbiter_review; `_apply_adjudication_verdicts` calls revise_finding_fields on bound UIDs, pool.save persists results; completed arbitration writes actual JSON marker only after all coverage/groups complete. Cross-stack merge consumes resulting record paths, then delivered report is produced by the normal path. Current module read against baseline confirms no hidden extra keeper transfer has changed matching inputs.

Production callers: `_step_arbiter` is wired by deep orchestrator STEPS; runner dispatches that flow for default deep review and public `start_at='merge'`. Per-group replay/fault, policy binding/reuse and marker completeness have distinct sibling proofs in latency/reuse modules and remain outside this deletion. Public marker lifecycle, high-severity gate, typed persistence and report effect are retained at the same real runner boundary.

## History explains newly introduced redundancy

`ce7e5719` (#168/#175) introduced both regressions: interrupted high-severity on-disk review must still arbitrate; already complete review must not repeat expensive adjudication. Originally the marker-present test **prewrote an empty arbiter marker**, so its setup did not overlap the missing-marker real run.

`e698ba7d` (#1440) replaced the obsolete `arbiter-complete.marker` with certified `adjudication-complete.marker`, and changed marker-present keeper to obtain valid completion by a genuine first runner call. This intentional fix added the exact first-run scenario already covered by the absent-marker singleton, creating the consolidation opportunity. Preserve that real-produced input, which is stronger than the original prewritten marker.

`329a7224` (#1212) only split the deep suite; it did not add independent input/contract between these two nodes. Do not cite physical filename similarity as justification: matching first-run path and actual input are the evidence.

## Runtime evidence and limits

Read existing JUnit execution evidence, without new tests/profiling:

| Recorded suite artifact | Removed absent-marker node | Existing two-run keeper |
| --- | ---: | ---: |
| baseline.xml |1.733s|3.036s|
| baseline-focused.xml |1.261s|2.214s|
| candidate-focused.xml |1.238s|2.164s|

These testcase durations provide existing real executed cost for one redundant run; they are not a measured before/after for this proposed change, not an identical-environment latest-CI comparison, and not guaranteed suite-wall savings. Keep two-run keeper cost and validate current focused/full consumers after an approved edit. One removed case is the only count claim.

## Transfer experiment plan before final approval

The first-run report assertion is the unique transfer and should receive an isolated actual-owner mutation control:

1. Use isolated current checkout; add transferred assertions/remove singleton there; run exact keeper clean. Read actual first report and confirm no prewritten ARBITRATED text existed in primed record fixtures.
2. Mutate only actual `_apply_adjudication_verdicts` arbiter revision application to ignore the returned description while preserving bound keep/severity/other fields, actual dispatch and marker generation. The keeper must fail specifically at the newly transferred first-run report assertion; import/collection/other unrelated failure does not count. Restore exact owner bytes and rerun keeper.
3. If uncertainty remains in lifecycle transfer, two independent controls: `_completed_adjudication` wrongly returnsTrue for missing marker should fail at first-run dispatch/report; suppress actual marker write should fail first marker is_file; disable valid-marker short circuit should fail second-run empty-calls assertion. These are proposed controls, not executed proof.
4. No code/checkout edits while its tests run. Restore/verify exact bytes between controls, then independent final preservation review. Validate precise owner module and related latency/reuse/deep/shared consumers, then ordinary make check with unchanged workers/coverage. No hook bypass.

## Other high-cardinality findings retained

- All seven `test_runner_substantiates_blocking_check_claims` rows retained: inferred Makefile command versus explicit canonical recipe, true lint failure, undeclared gate, semantic-only regression, mixed semantic+green-check classification, and actual source-mutating check exercise different production commands/failure guards. Helper neighbors protect real missing-executable/timeout, untrusted shell authorization, multi-check completeness, absent classification, ignored owner state/chmod and symlink identity. History `daffb6e2` (#1432) hardened these separate contracts. No matching exact row can replace another. Recorded baseline7-run cost51.535s is not redundancy evidence.
- Seven `test_ephemeral_failure_handoff_projects_public_refs_without_private_paths` rows retained: default/custom public/private-runtime mapping versus external owned target, clean output, known private leaf fallback versus legal external evidence preservation, and unknown private path/test-output scrubbing. All read actual live partial bytes and authorized changed files through actual runner. History `023fc5b7` (#1161) introduced distinct artifact routing/privacy requirements. Baseline45.039s does not license cutting them.
- `test_opt_in_tiers_resolve_cli_then_file`21rows covers7tiers×3actual config fields, declaration-type guard and explicit exemptions. Could mechanically loop all3fields per tier, retaining all21 assertions in7wrappers, but all21cases total only0.023–0.063s across existing recorded runs. This does not justify expanding a runtime batch. History `c08e8ee5` (#1225/#1312) deliberately pinned truthiness versus sentinel behavior and future mirrored flag inventory. Retain now; do not represent wrapper count as performance.
- Precision low/medium selection, opt-in posting, tool versus wall/group budgets, successful unbudgeted group, carry-over wall accounting, per-file batching, infrastructure abort, interrupted partial diagnosis, distinct repair backend and forensic/default alternatives routes remain independently valuable. All current19declarations/52rows were read; only P-R1 selected.
