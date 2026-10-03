# PR #1444 continuation

Baseline: `58cc88e4d7091b27afe1b2a6eec486504e255dfd`.
Continuation started at `b98a9dd51fde81df5dc4534a85ab29a986f6a127` on
`refactor/production-simplification`; PR #1444 was open and draft, working tree clean.

The original inventory and pinned Python 3.12 / scb-check 0.2.0 method reproduce
113,364 physical lines and 85,527 source lines at baseline. The continuation's
starting inventory reproduces 111,025 physical lines across 364 production files.
The latest integrated snapshot has 363 production files, 110,239 physical lines,
and 84,495 source lines (2.757% and 1.207% aggregate reductions). The starting
commit also reproduces its recorded 85,165 source lines. Targets remain at most
102,027 physical lines and 76,974 source lines: shortfalls are 8,212 physical and
7,521 source lines. The target is unmet; focused improvements are not completion.

## Decisions under integration

- Backend subprocess policy owns pipes and draining directly; redundant transport
  mode enums and synthetic exception conversion disappear. Pi's tool-call model
  owns argument validation.
- GitHub import uses the existing authenticated API boundary for REST and GraphQL,
  eliminating its independent parser and retry implementation while retaining
  bounded rate-limit recovery and read-only GraphQL timeout retries.
- Benchmark transactions derive manifest-last replacement order from their ordered
  target map. Normal completion and startup recovery share verification, with
  cleanup deferred until the complete journal can be recovered.
- Improve scope repair reconciles one normalized mapping; assembly enriches the
  detached mapping rather than projecting a second plan shape.
- Training adjudication retains the projector's source resolution rather than
  reconstructing and matching frozen projections. Eligibility and evidence
  freshness share one policy. A stale adjudication cannot authorize human-gold
  export; its accepted task disposition remains available.
- ArtifactSession owns freezing, publication, and rollback directly; the reciprocal
  free-function adapters and finalization module disappear. Exact lifecycle bodies
  retain validation and journal ordering. Concrete review consumers import their
  owners rather than routing through unrelated posting facades.
- Timing analysis consumes parsed payloads directly. Telemetry removes unused
  tool-name state, duplicate usage collection, and a responsibility-free exporter
  subclass. Quality analysis owns one parse-result mapping.

- Remote-CI receipt admission uses the producer's typed identities and one live
  outcome evaluator; archive no longer maintains its own terminal policy and
  identity models. Duplicate producers, partitions, polling limits, push binding,
  and no-CI evidence remain checked. Noncanonical persisted PR URLs are rejected.
- Plan landing shares rendering, text publication, durable entry construction,
  and main-index publication across normal and reanchored destinations, preserving
  their distinct outcome/error/cleanup order. An obsolete fingerprint-union
  helper disappears; runtime member coverage remains the one reservation policy.
- Telemetry binds its concrete invocation ledger rather than looking up the latest
  completed summary; an intervening sibling cannot change billing ownership.
  Generation metadata retains no raw choice content.
- Host judge imports use normal package ownership. The dynamic asset cache and
  process-wide module-registry rewrite disappear; compiled scripts retain their
  standalone sibling core import.

## Verification so far

Workers report focused backend, Improve, training, archive, and telemetry checks;
the coordinator owns final integrated checks and measurements. GitHub import and
curation focused checks passed (215 tests). Transaction checks passed (114 tests and a 70-test rerun including manifest
staging-order recovery). Review/extension/runner focused checks passed (390).
Artifact checks passed (228 plus one existing filesystem-dependent skip), and
runner/archive/trajectory/crash/layout integration passed (266). Ruff, both
Vulture scans, and mypy (748 files) passed in the integrated gate; the first full suite reached 90.46% coverage (86% floor), with 9,323 passing,
14 skipped, and three failing cases. The profile case had loaded an assertion
corrected during the run; its final six-test suite passes. Redaction exposed
excessive per-character Python work: a prefix filter preserves exact validation
and is about 10 times faster on the 200K-character case. Its 113 tests, 174
consumer/security tests, and 55,323 worker/reviewer differential cases pass.
The group-cap case correctly capped fixes, tested retained changes, and pushed;
remote CI was unavailable. Its unchanged isolated case and all 50 precision
cases pass at the configured parallel worker count. Its failure message now
includes the persisted CI diagnostic. No timing threshold or gate was weakened.
Signed checkpoint `3a79584d` passed the normal pre-commit and pre-push hooks,
including the full repository gate: 9,330 passed, 14 skipped, 90.07% coverage
against the unchanged 86% floor. The next batch adds 204 judge/import/workspace
checks, 251 agent/telemetry checks, 237 plan checks plus 21 focused member/real-flow
checks, and 340 remote-CI/archive checks. Independent review included 1,961
malformed-receipt differential cases with zero regressions. A broader archive run
exhausted disk space (Errno 28); only owned completed disposable test directories
were removed; the complete affected suite then passed (47 real workflow tests).
The next pre-push hook first stopped at a mixed event list inferred as `object`
in the new telemetry test. An explicit `list[AgentEvent]` preserves the closed
event union; full mypy (748 files) and Ruff pass. Signed checkpoint `0a54165e` then passed the normal hook retry and push:
9,340 passed, 14 skipped, 90.46% branch coverage; all required checks passed. The PR must remain draft while targets are unmet.

## Remaining work

The third coherent batch removes the HarvestServices protocol/factory and reciprocal
workflow forwarding. HarvestPass binds one frozen config, archive and acquisition
policy; callers cannot retarget archive or supply a conflicting dry-run flag.
FindingRecord retains immutable source, identity and adjudication, deriving distinct
consumer views instead of mutating/restoring conflict state or fingerprint-rejoining
frozen projections. RecordPool supplies already-admitted structural records and UIDs
to merge/salvage; repeated disk reconstruction and empty-merge proof disappear.
OspreyConfig owns native options; closed request fields define scalar argv order,
and the exact-base immutable request evidence excludes private paths/variables.
Logical agent model identity now passes the existing bounded privacy admission.
Readback binds accepted rows/count/hash to the same stable snapshot, correcting
LangSmith A/B/A acceptance with mismatched B rows.

Focused verification: 86 harvest/ownership/CLI tests plus 23 real CLI/cache/preview
checks; 227 finding-owner checks and a 960-case exact semantic differential;
396 merge/phase/cache/real-resume checks (two existing skips) plus 25 attribution
checks; 201 backend/protocol/runtime/real-visibility checks and 4,000 byte-identical
native argv differentials; 36 readback/replay checks including real loopback A/B/A.
Required mypy passes all 747 source files; Ruff passes. Independent reviews of each
owner and the integrated diff found no material issue. Full normal pre-push gate
is pending for this checkpoint. Provisional final batch measurement is 84,225
source / 109,892 physical / 362 files: 1.522% source / 3.063% physical reduction,
leaving 7,251 source / 7,865 physical lines above the targets. Exact owned inventory
and source counts include every new module; no upstream baseline advance exists.

Next assessed candidates include one captured annotation publication bundle.
RL scoring input caching was rejected: complete-caller accounting showed 1–6
source lines saved and extra derivative state. Internal curation class grouping
was also rejected because it would mostly rename stateless ownership. These
findings are not blockers; larger supported-workflow simplification remains.

Continue looking for substantial behavior-preserving architecture reduction;
current changes do not approach the aggregate target. Integrate worker edits,
remeasure with the same inventory, run required repository verification, obtain
an independent integrated review, and update the existing PR with honest metrics.
Inspected plan/diagram validation, artifact ownership, and isolated RL security
policies have distinct supported obligations; deleting those obligations is not
an acceptable route to the LOC target.

## Independent review

The read-only integrated review confirmed the inventory, physical count, hash,
target arithmetic, scope, substantive ownership changes, and meaningful fault
tests. It found no material regression or weakened gate in inspected changes.
Its identified artifact reciprocal-ownership issue was resolved and verified.
Unchanged extracted functions are explicitly classified as moved code, not
eliminated complexity. The production-target shortfall remains outstanding.
