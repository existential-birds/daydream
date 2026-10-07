# Share identical cold runs without losing contract assertions

Pinned merged baseline: `6c48d368d529a89748c93e7bb61ae9ac6566936e`.
This follow-up starts at `1e0951928ba5a6eb8dd611d4d9dc5a6991b54bc8`.
Final collection: **8,196 cases / 5,357 distinct functions**, a net reduction of
**808 cases (8.97%) / 74 functions** against the merged baseline. This follow-up
removes two functions/cases and two cold real-runner invocations. Count reduction
is not a performance result. No production or harness change is included here.

## Exact affected nodes and named keepers

| Removed node | Keeper | Preserved regression checks |
| --- | --- | --- |
| `tests/test_improve_flow.py::test_rendered_plan_gives_a_literal_executor_no_room_to_guess[asyncio]` | `tests/test_improve_flow.py::test_reused_plan_publishes_its_stored_package_and_member_identities[asyncio]` | All 40 original assertions: complete recon/audit/vet/writer calls, actual nonpersist arguments, literal rendered executor instructions, real HEAD anchoring, scoped staging, no unauthorized publication, real author prompt and schema. |
| `tests/deep_orchestrator/test_review_reuse_path.py::test_exploration_provenance_is_recorded_in_the_store` | `tests/deep_orchestrator/test_review_reuse_path.py::test_store_directory_survives_a_fresh_run_and_is_readable_by_the_next` | All five provenance observation statements: actual published JSON exists, parses, contains exploration and has an allowed outcome. |

The complete first-run inputs match in each pair: fixture arguments, decorators,
backend installation and first awaited runner/config call. Improve explicitly
uses `n_findings=1` in both cases. Review reuse uses the same default backend with
optional exploration unavailable in both cases. Each keeper still creates a fresh
real temporary committed Git repository, runs the real event loop and production
artifact/cache writers, then performs its original second real run.

All Improve observation statements move unchanged immediately after the first
run, before `sidecar_path` is read or mutated; only `code` in the first exit
assertion becomes `first_code`. All eight original publication assertions and
stored identity mutations remain. Provenance reads move after the first success
and store-directory assertion, before either sentinel mkdir and before the warm
run. All four original persistence assertions remain. The redundant first-run
exit assertion from the provenance singleton already exists in the keeper.
Every other function AST and parameter row is unchanged. Earlier audit keeper
maps now point to the surviving publication node.

## Owners, fixtures and history

Complete affected test bodies, root fixture chains/configuration, external
backend scripts, production owners and callers were read before editing. Improve
runs through orchestrator/planning, durable PlanWriteSession/index reservation,
authoring prompt/schema and renderer; later publication reads the stored package
and member identities. Review runs through the real reuse-cache builder,
`_step_exploration`, atomic `ReuseCache.record`, manifested artifact publication
and artifact-session detachment/reseeding. No expected artifact is prewritten by
a fixture. No production export or new test seam is introduced.

Relevant history: Improve `9bacaa42` introduced literal executor coverage and
`77f6a7cf` introduced stored-identity publication; previous audit transfers
`06c79c49` and `1e095192` are preserved. Review `c93b41a2` introduced both store
and provenance bodies; `dd4a2a6e` and `e698ba7d` retain this exact input pair.
All reuse invalidation, corrupt/legacy/resume coverage, disabled-cache,
exploration-enabled, moved-HEAD and failure-recovery rows remain separate.

## Independent review and actual-owner controls

Independent pre-edit reviewers approved both exact transfers; a third reviewer
compared the final complete functions and earlier preservation ledgers. Evidence
is recorded under `/tmp/daydream-test-audit-6c48d368/` in
`improve-cold-prefix-evidence.md`, `reuse-prefix-ledger.md`,
`improve-prefix-independent-review.md`, `reuse-prefix-independent-review.md`
and `prefix-final-preservation-review.md`.

In an isolated checkout with the strengthened keepers:

1. Remove the renderer's `## Before you start` heading: real run succeeds, then
   the transferred cold artifact assertion fails before publication mutation.
2. Change authoring's actual `persist_session=False` to True: real run succeeds,
   complete phase marker check passes, then captured nonpersist assertion fails.
3. Suppress only unavailable-exploration provenance recording: real run succeeds,
   published records exist, then transferred nested read fails with
   `KeyError: 'exploration'` before sentinels or the warm run.

All three production files were restored byte-for-byte; both keepers then passed
in 3.06s. The control checkout was restored clean and removed. The mutation
runner initially rejected the correctly failing renderer log because its own
expected text included a variable name omitted by pytest's rendered expression;
that observer was corrected and all controls rerun. No control source is shipped.

## Comparable measurement and validation

Same Python 3.12.13, locked dependencies, n4, ten-case dispatch, macOS machine,
no coverage, explicit fresh basetemps, command settings and JUnit/duration output:

| Selection | Cases | Pytest | Wall |
| --- | ---: | ---: | ---: |
| Original two source nodes plus two keepers | 4 | 2.88s | 3.52s |
| Strengthened two keepers | 2 | 2.68s | 3.05s |

This single small-selection pair saves 0.47s wall; it is not a full-suite or
hosted-CI speedup. Owner and shared-harness consumers passed **750 tests / one
existing Unicode-filesystem skip**, in **79.90s pytest / 80.29s wall**. Selection:
all six Improve files, review-reuse path, reuse-store, exploration-cache/runner,
artifact-visibility and its integration file, harness stub/backend tests.
Lock check passed. Final full hook-enabled `make check`, coverage, skip accounting
and hosted CI measurements are recorded in PR #1497 after this commit.

The last measured hosted check before this follow-up was **6m08s**. The four-minute
goal remains unmet; these tiny proven consolidations do not establish a route to
it. The reviewed redaction micro-optimization is outside this test-only batch and
has only tens-of-milliseconds predicted savings per profiled flow. No production
change or deletion quota is justified by that proposal.
