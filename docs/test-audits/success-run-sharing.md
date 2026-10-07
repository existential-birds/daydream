# Matching Improve runs and unused snapshot setup

Baseline: `6c48d368d529a89748c93e7bb61ae9ac6566936e`; preceding green head: `0bfe3351c75031e3b39ca243160ab80aee3231a0`.
Collection: **8,198 cases / 5,359 functions**, down **806 cases (8.95%) / 72 functions**. These additions delete two repeated real runs and one function. No production, shared fixture producer, transport, coverage, skip, dependency, workflow or worker setting changes.

## Evidence recorded before editing

| Exact retired node in tests/test_improve_flow.py | Named keeper | Transferred observations |
| --- | --- | --- |
| `test_improve_model_calls_are_one_shot[asyncio]` | `test_rendered_plan_gives_a_literal_executor_no_room_to_guess[asyncio]` | Actual recon/audit/vet/plan-writer marker set is complete; every such call has persist_session=False. |
| `test_effort_and_focus_select_the_audited_categories_read_only[asyncio-standard-effort]` only | `test_improve_timing_completeness_preserves_p09_audit_isolation[asyncio]` | Persisted category inventory equals AUDIT_CATEGORIES; actual audit calls are nonempty and read-only. Quick-effort/security-focus rows remain unchanged. |

Both complete tests, keepers, fixture chain, backend constructor/recorder, production owners/callers and history were read, then independently reviewed before edits. S1 uses identical fresh committed monorepo, exact install_improve_stub(...,n_findings=1), unmodified backend and default real runner config. The prior n_findings=1 versus timing n_findings=None objection is resolved by the existing literal-executor keeper's exact one-finding input. Preserve all its shipped prompt/schema, Git revision, tool-path and rendered-plan assertions. Repo-scan remains excluded from the four-phase persistence assertion, exactly as before.

S2 uses the same unmodified default n_findings=None backend. Explicit standard/None equal the plain RunConfig dataclass defaults, with no supplied-field tracker; actual tier/category owners use those values directly. The timing keeper's absent registry override has no executed reader after3014107c/#902; its backend factory ignores that SDK environment. Native config/absence/populated/transport proofs remain. Preserve all timing child-dispatch, identity/enclosure, source HEAD/refs/index/status/config bytes, private audit-root confinement/removal, report/run marker and provenance assertions.

The actual backend execute method defaults persistence=True and records received flags, so False is not fabricated by the fixture. Real recon/audit/vet/authoring callers explicitly disable persistence; native Claude/session proofs remain independent. The audit artifact's categories_run setter writes planned categories, not necessarily successful executions: the transferred field check is accompanied by actual nonempty/read-only calls and existing successful24audit/eightvet dispatch proof. History9bacaa42/#291 introduced both contracts;836ade46/#1153 hardened ephemeral/native audit behavior. No fault, backend, cardinality, security, source ownership, CLI, description or retry variant is deleted merely because its success resembles these runs.

## Mutation controls and preservation review

In an isolated checkout, change only authoring._generate_once's persistence flag toTrue: real exit0 and all four phase markers remain, then the literal-executor keeper fails at the transferred persistence assertion. Restore exact owner bytes and rerun: pass. Separately truncate only audit.py's categories_run serialization to the first category while preserving dispatch: timing keeper fails at the new inventory assertion. Exact restoration passes. No stub mutation supplies the outcome. Control source hashes match original production; test files restored clean and the checkout removed normally.

Independent final whole-module AST comparison found only the authorized removed singleton/row and seven added transfer statements; every other declaration and all existing keeper assertions remain. No production export or test-only seam is introduced.

## Conditional real shard fixture

`tests/deep_orchestrator/test_review_completion.py::test_snapshot_boundaries` retains all nine exact rows: live,dirty,no-diff,dirty-no-diff,interactive,mismatch,base-tip,shards,pipeline-budget. Only shards references shard_many_python_target. Previously all nine built a second unused sibling Git repository. Replace the unconditional argument with FixtureRequest and retrieve the same real function-scoped producer only inside the existing shards branch.

All other eight rows keep their exact actual subject repository, PR/config/backend inputs, filesystem faults and delivered assertions. Shards retains real3Python+README committed changes, the same deep_shard_enabled/max-files configuration and all five complete scopes/exact file inventory checks. No cache, copied Git state, new producer or mocked filesystem is used. Existing root isolation still runs eagerly. Pytest's public call-phase lookup resolves the same fixture/dependencies and normal finalizers; no deadline/task/production run begins before retrieval. Its own setup cost moves into call phase for shards, so compare combined timings.

Complete helper/producer/caller/history review found no dependency or side effect beyond the unused sibling; production owners only consume the specified subject repo. e698ba7d/#1440 introduced the common signature;0d24691e/#1485 added snapshot/capture proofs without using the second repo. Independent pre-edit and final AST reviews approved only the argument/import/conditional lookup. Every table row, assertion and other declaration remains unchanged.

A temporary external pytest hook observed actual fixture construction, without replacing it: before9attempts across all nine rows, after1attempt only for shards; all worker exits0. All nine real tests pass on both sides. Source-derived removal: eight unused producers/80Git subprocesses plus unused file writes. No maintained helper-only test was added.

## Matched runtime and validation

| Same selection/settings | Before | After |
| --- | ---: | ---: |
| Improve selection |6 passed|4 passed|
| Improve pytest / wall |3.36s /3.99s|2.20s /2.56s|
| Snapshot selection |9 passed|9 passed|
| Snapshot pytest / wall |3.36s /3.71s|2.95s /3.31s|

Both pairs use Python3.12.13, n4, ten-case dispatch, same locked dependencies/macOS machine/no coverage and fresh explicit basetemp directories, identical command/plugin/duration settings within each pair. JUnit and phase rows reconcile the exact removed nodes; snapshot observer is identical in both runs. These single small-selection pairs are not whole-suite/CI savings or proof of the four-minute goal.

Full changed owners and related sharding/latency/Claude/backend harness consumers: **338 passed / one warning / no skips**,54.09s pytest /54.48s wall. Collection **8,198 /5,359**. Ordinary final make check/push and hosted results are recorded in the PR after actual completion; hooks must remain enabled.

The complete prior hosted probe's printed phase split is1,202.40worker-seconds call versus60.09setup and0.31teardown, with the independently reviewed309.00375s conservative four-worker bound. Fixture-only cleanup cannot close that measured execution gap. Existing source profiling points to actual production Git queries and trajectory redaction. No production optimization is authorized/shipped in this test batch; broader scope is pending user input. Goal remains unmet until actual final CI is under240s.
