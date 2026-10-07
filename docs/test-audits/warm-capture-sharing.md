# Remove duplicate warm reuse and offline capture executions

Pinned merged baseline: `6c48d368d529a89748c93e7bb61ae9ac6566936e`.
Immediate baseline: `ddcb09679733e9dbc37a9e5212c8041844adf81f`.
Final collection: **8,192 cases / 5,356 functions**, down **812 cases (9.02%) /
75 functions** against merged main. This follow-up removes four cases, one
function and seven actual runner calls. No production, support, skip, coverage,
worker, workflow or dependency change.

## C-W1: duplicate labels do not select different scenarios

Affected declaration:
`tests/deep_orchestrator/test_review_reuse_path.py::test_identical_rerun_restores_completed_units_and_current_run_evidence`.
Remove only its `[merge]`, `[companion]` and `[rebound]` rows; each has the exact
named keeper `[all-units]`. Retain `[independent]` and `[arbiter]`.

All six reads of `unit` occur in test-local conditions distinguishing only
independent/arbiter. None passes the label to config, backend, owner or helper.
All four default labels therefore execute an identical complete body, with the
same fresh committed multi-stack fixture, empty options and default stub. The
fixture chain does not consume their callspec labels. Separate literal-branch
AST projections independently reproduce the identical complete-body hash.

The body and all 17 assertions remain unchanged: genuine cold run, actual
canonical records/merge/intent/alternatives/dedup bytes and typed coverage reads,
backend-call reset, genuine warm run, no paid review calls, unchanged canonical
bytes and unfinished scopes, new run identity, equal analyzed revision/planned
scopes/stack outcomes, complete scopes and regular public report without partial
coverage. Thus every removed row's regeneration, byte loss, stale identity,
incomplete coverage and report failure remains caught by the same-input keeper.

Independent and arbiter inputs genuinely differ: custom alternatives policy,
explicit intent/wonder hit grounding, balanced routing, high findings, merge echo,
actual arbitration dispatch/hit/grounding and completion markers. They remain.
All nine cache-damage and six resume-damage rows remain: no equivalent independent
real-seed reset was proved for a shared sequential fault loop.

Full tests, fixtures, backend/profile/coverage support, reuse/review/merge/typed
artifact/session owners and callers were read. History `c93b41a2` introduced
standalone warm contracts; `e698ba7d` consolidated them into this matrix but left
four identical default labels. Its entire matrix AST is unchanged at merged main
and immediate baseline. This exact proof corrects the earlier retain-all matrix
assessment; labels alone did not preserve independent inputs.

## C-H1: offline row already performs the explicit-destination capture run

Remove:
`tests/test_dataset_hub_run.py::test_explicit_hub_destination_captures_evidence_without_archive_or_credentials`.

Named keeper:
`tests/test_dataset_hub_run.py::test_review_stays_successful_with_durable_failed_uploads[offline]`.
Keep all three offline/public/secret rows and all nine original keeper assertions.

Both use the same module config, fresh committed multi-stack repository, actual
local store and backend; review mode, archive/evaluation false, explicit private
hub destination, absent HF credentials/environment destination and cleanup false.
The fake's default private=True equals the offline row's explicit True; both
have fail_before=True, otherwise identical empty state/revision, and the same
external factory. The keeper's local credential string is not a producer input;
its secret-only file write does not execute for offline.

Four of the singleton's six assertions already have matching predicates in the
keeper: real exit0, nonnull store, one record and successful outcome. Move both
original-task and trajectory `status == "available"` predicates unchanged under
`failure == "offline"`, immediately after the captured count/outcome assertion,
before uploader construction/status inspection/reset/retry. Real snapshot/disk
readback remains; no fake supplies the record. These catch schema-valid missing
capture sections despite archive off, no credentials and failed external upload.

Preserve real durable failed receipt, value-free diagnostics, zero remote commits,
secret raw local evidence, public-destination refusal, external availability reset,
real retry and immutable local-record equality. Complete capture/store/schema,
scoring/hub/uploader/finalization/runner callers and fake were read. Both exact
bodies/config were introduced together at `9328e0d0` (#1493), remain unchanged,
and follow raw capture `0d24691e` and source-bound reply extension `6c48d368`.

## Review, controls and validation

Independent pre-edit reviewers approved both proposals. Final review projected
both complete modules: only three parameter strings, exact singleton deletion
and two offline assertions change; all other ASTs/decorators/rows remain exact.
Evidence under `/tmp/daydream-test-audit-6c48d368/`:
`reuse-supervision-followup-ledger.md`, `warm-matrix-independent-review.md`,
`capture-trace-sharing-ledger.md`, `capture-sharing-independent-review.md`,
`warm-capture-final-preservation-review.md` and its AST evidence JSON.

Two isolated actual capture serialization mutations independently omit
original_task and trajectory evidence. The real run/count/success assertions pass;
the transferred predicates fail on unavailable and unproduced respectively,
before uploader/retry. Exact source restoration then passes the keeper (0.94s).
The control checkout was restored clean and removed; no mutation is shipped.

Same Python3.12.13, locked dependencies, n4, ten-case dispatch, macOS machine,
no coverage, command settings and fresh explicit basetemps:

| Selection | Cases | Pytest | Wall |
| --- | ---: | ---: | ---: |
| Six warm rows plus capture singleton/offline keeper | 8 | 4.17s | 4.80s |
| Three retained warm rows plus strengthened offline keeper | 4 | 2.61s | 3.18s |

One small-selection pair saves1.62s wall; this is not a whole-suite or hosted-CI
performance result. Complete owner/shared consumers passed **225 / no skips /
eight warnings**, **53.11s pytest /53.49s wall**: reuse/supervision/store, dataset
hub-run/capture/store/commands/hub, archive data capture, record dataset journey,
corpus projection and trajectory phase events. The first focused command named
nonexistent `test_dataset.py` and ran zero tests; it was corrected to the actual
store/commands modules. Lockcheck and final collection passed. Ordinary full
make check, coverage/skips and hosted CI are recorded in PR #1497 after commit.

Static discovery catalogued334 root test modules,766 parametrized declarations
and813 decorators (792 literal/named tables). Conservative literal-condition
folding found C-W1; unknown, indirect and dynamic inputs remain outside that
proof. Expensive unused-fixture discovery found no material removable setup;
required WorkContext/fake-gh dependencies stay. This is additional bounded
redundancy evidence, not an exhaustive suite-wide completion claim.

The last hosted check was **5m52s**. The four-minute goal remains unmet; no
30% deletion quota, production cache, artifact copying or weaker fault/transport
proof is justified by these small reductions.
