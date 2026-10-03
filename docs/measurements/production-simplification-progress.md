# PR #1444 continuation

Baseline: `58cc88e4d7091b27afe1b2a6eec486504e255dfd`.
Continuation started at `b98a9dd51fde81df5dc4534a85ab29a986f6a127` on
`refactor/production-simplification`; PR #1444 was open and draft, working tree clean.

The original inventory and pinned Python 3.12 / scb-check 0.2.0 method reproduce
113,364 physical lines and 85,527 source lines at baseline. The continuation's
starting inventory reproduces 111,025 physical lines across 364 production files.
The latest pushed checkpoint `4d2967f626d6e83c754ae4ac7b74bd93e580476a` has
362 production files, 108,842 physical lines and 83,415 source lines
(3.989% and 2.469% aggregate reductions). Starting source LOC is 85,165.
Targets remain at most 102,027 physical and 76,974 source lines: shortfalls are
6,815 physical and 6,441 source lines. The objective remains unmet; useful
owner consolidation and passing checks do not establish completion.

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
owner and the integrated diff found no material issue. Signed checkpoint `7f8bc913` passed the normal pre-commit and pre-push hooks:
9,363 passed, 14 skipped, 90.47% branch coverage; every required check passed.
The committed batch measurement is 84,225
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

## Fourth batch integrated

- Configuration decode registration lives beside the frozen root field declarations;
  a separate name/policy registry disappears. Special/nested coercion and precedence
  remain explicit. Pinned source 268→221 (-47), physical 361→313 (-48); CC unchanged.
  108 real CLI/config tests, mypy and Ruff pass; 2,000 seeded old/new outcomes match.
  Independent flow review found no issue.
- The unused build_payload façade and its imported forwards disappear. Actual review
  submission still owns typed event/payload/approval rendering. Production/docs/registry
  reachability found only test callers. Existing golden payload and real submission
  checks now invoke those actual owners; no implementation moved to tests. Source
  680→654 (-26), physical 895→866 (-29). 173 checks, Ruff/mypy and flow review pass.
- Pi dispatch owns the one native event vocabulary. PiError admits omitted retry
  inference directly; the probe exception and mirrored failure facts disappear.
  Usage construction drops a throwaway mapping. Source 730→693 (-37), physical
  932→873 (-59); CC unchanged. 335 checks pass with one existing skip; 15,000 failure
  cases and 357 native replays are equivalent. Ruff/mypy and flow review pass.
- ReviewProfile/Pipeline frozen domain owners admit concrete field constraints and
  defaults; the manual parallel parser disappears. Arbiter selectors consume admitted
  Arbitration/Suppression policy rather than normalize it again. Three-file source
  1,338→1,241 (-97), physical 1,695→1,590 (-105); high CC 9→8, cognitive 9→7. 2,904
  exact profile/digest oracle outcomes and 198 affected checks pass; Ruff/mypy/Vulture
  and backend review pass.
- FinalAnnotationBundle binds immutable semantic bytes, captured identity/digests and
  canonical publication/reconstruction envelope. Receipt closure and repeated wire
  assembly disappear; hash-only dry-run remains separate from strict publication
  admission. The unsupported boolean-routed project_findings façade disappears, with
  callers using FindingRecord views directly. Five-file source 2,814→2,767 (-47),
  physical 3,383→3,331 (-52); high CC 18→16, cognitive 22→21. 338 checks and 30 exact
  remote/download receipt oracle outcomes pass. Backend review caught mutable-buffer
  admission; strict bytes-only admission and a no-Hub-access regression fixed it.
  Reviewer rechecked successfully. Source-deletion/replacement and corrupted-envelope
  checks preserve capture ownership and fail-before-stage behavior.

Current coherent batch net: 254 production source / 293 physical lines removed,
including all new owners and callers. Aggregate target remains far unmet. Full owned
snapshot /tmp/daydream-pr1444/fourth-final includes all 362 production files and the
unchanged inventory hash. Independent integrated review reproduced every owned byte and all pinned counts,
confirmed target arithmetic and found no material regression, hidden path or weakened
check. Normal signed commit/push verification is pending; workers are in read-only
assessment during that gate.

Next coherent opportunity: benchmark curation's callback mutation machinery and
raw/model reparse, using one concrete locked CaseEditor for the existing workflow.
Data owns curation/import callers after integration freeze lifts. Backend traces
prompt-selection/capture and structured telemetry ownership; flow traces finding
state/cache/persistence across review modes. None yet demonstrates a defensible
300-line net reduction. Unsupported private test seams are not automatic public
compatibility obligations; essential validation, security, diagnostics, durability,
wire formats, resume and supported extension behavior remain required.

Fourth integrated aggregate: 83971 source / 109599 physical; 1.819% source / 3.321% physical reduction. Remaining shortfall: 6997 source / 7572 physical lines. No inherited upstream change; PR base remains the recorded baseline.

Fourth checkpoint normal push initially stopped at mypy: one test-only legacy
payload import in test_training_labeler_signals.py was missed. The test now uses
the actual typed renderer and transport serializer; all footer assertions remain.
35 focused tests, Ruff/mypy and independent flow review pass. Ordinary signed amend
and full normal push gate retry follow; no hooks are bypassed.

Checkpoint 2a8b0b2330f8cdad2a9a27c9aff52274406e8577 is signed and pushed.
Normal commit and retry push hooks pass: 9,401 passed / 14 skipped / 204 warnings,
652.39s; 90.52% branch coverage against the unchanged 86% floor. Ruff, both
Vulture scans, mypy (747 files), actionlint, coverage artifact and naming pass.
Every committed production byte matches the independently reproduced fourth
snapshot. Mutation freeze lifted for the fifth curation/native-protocol batch.

## Fifth batch integrated, verification in progress

- CaseEditor owns each lazy lock/recover/fresh-read/mutate/validate/commit operation.
  The callback router, fifteen mutation closures, append adapter, and redundant
  one-item service forwards disappear. Raw repair before final strict validation
  remains supported; no lock spans an interactive editor or network operation.
  Four production files remove 65 source / 89 physical lines. 23 operation/error/
  wire-byte comparisons match; 101 final curation/TUI/CLI checks and 201 preceding
  callers pass. Independent flow review passes.
- Claude request evidence derives from the applied SDK options rather than parallel
  input facts. CodexDiagnostics owns early-once/final-on-change emission and detached
  consumer payloads. Native model-label privacy admission is shared with logical
  agent evidence. Four production files remove 36 source / 40 physical lines;
  355 checks pass with four existing skips, 277 native replay cases and 20,000
  diagnostic polls match. Actual OTLP private-label canaries and independent review
  pass; native SDK model selection is preserved.
- Harbor uses native httpx.Response and asyncio.Process contracts. Fake-only response
  and process adapters, a second child-kill site, and impossible retry fallbacks
  disappear. The bounded stdout reader owns EOF/exit/deadline/error/cancellation
  cleanup; stderr remains DEVNULL. One production file removes 26 source / 34
  physical lines; 143 checks pass including actual subprocess stdout, overflow,
  EOF-without-exit, and cancellation behavior. Independent backend/flow review passes.
- Archive portable wire is produced directly from captured native snapshot inputs;
  the mutable flat Manifest and reconstruction/projection bridge disappear. One
  canonical/legacy SQL projection serves native indexing and hydration. All five
  production files remove 71 source / 110 physical lines, including final typed
  SQL persistence ownership and its native caller. 156 exact wire/index/
  legacy/error comparisons match. Review caught subclass field leakage/deepcopy
  from an early asdict prototype; exact shallow approved-field projections and a
  real wire private-field/deepcopy-trap regression resolve it. Expanded 19-file integration suite passes 753 checks; final persistence-boundary
  rerun passes 442 checks. Mypy (27 files), Ruff and independent final review pass;
  caller fixtures now use native producer or literal wire.
- PhaseEvent wire policy lives on the exact base dataclass fields. Required-null and
  optional omission, duck enum values, order, and separately guarded metadata remain
  unchanged. Six source / six physical lines removed; high cognitive count rises
  0→1. 99 real lifecycle/OTLP checks and 5,000 exact wire/error comparisons pass;
  Ruff/mypy/Vulture and independent flow review pass.
- Quality has one scoped function-evidence traversal for body-only CC, identifier
  and assignment references, and wrapper flags. Repeated assignment subtree scans
  and separate single-use/wrapper function inventories disappear. 20 source / 26
  physical lines removed; high CC rises 8→9 while cognitive falls 11→10. 1,777
  inputs / 15,760 functions compare exactly and 153 checks pass, including actual
  deep fix-quality workflows. Independent backend review passes.
- Exploration's one native specialist inventory supplies dispatch descriptors,
  prompts, schemas, and model names. Duplicate dependency scheduling and the local
  task/limiter/fork workflow disappear into the existing fan-out owner. Deferred
  builders are consumed inside its task group, preserving prompt failure grouping
  and evidence before any model dispatch. 35 source / 34 physical lines removed including the existing fan-out cost.
  63 focused checks pass, including two prompt-failure/no-model/durable-dispatch
  cases; Ruff/mypy and independent backend review pass. No new production module is added.

Provisional complete owned snapshot: 83,709 source / 109,257 physical, 362 files,
unchanged inventory SHA256. Aggregate source reduction remains about 2.126%; physical
about 3.623%. Shortfall remains 6,735 source / 7,230 physical lines. These are dirty
checkout measurements and are not a claim that completion criteria are satisfied.

Rejected full-cost prototypes: native adjudication UID binding removes ordinal
translation but needs native-local-ID witness, unsharded selection-order preservation,
UID-less positional drops, and post-save coverage timing. Its eight-line gross saving
is consumed by necessary adapters; no repo edits. Combined retry accounting must retain
prepaid cancellable-backoff allowance versus completed-sleep telemetry, so its apparent
class deletion saves under twenty lines after callers; no new state owner was added.
A broader unreferenced-method scan found registered serializers, native overrides and
Pydantic validators rather than substantial dead methods. Negative assessments are
not blockers; source/physical targets remain outstanding and substantive work continues.

Final fifth fixed snapshot supersedes the provisional count: 83,712 source /
109,260 physical, 362 files, clone lines 2,244, high CC 493, high cognitive 605.
Aggregate source reduction 2.122%; physical reduction 3.620%. Remaining shortfall:
6,738 source / 7,233 physical. Batch removes 259 source / 339 physical lines;
continuation removes 1,453 source / 1,765 physical. Inventory unchanged and every
owned Python module/runtime template counted. Targets remain false. All production
owners are frozen for independent integrated review and ordinary signed commit/
push hooks; next read-side prototype remains outside the repository meanwhile.

Fifth independent integrated review reproduced the entire pinned SCB JSON and every owned
production path/byte; no unresolved material finding. Broader baseline-to-current review
was sampled, not an every-line new audit of the earlier draft. Signed checkpoint a187b7e7646702acbe5453a252304f622c80dd2b
is pushed with ordinary hooks: 9,430 passed / 14 existing skips / 203 warnings, 597.21s;
90.52% branch coverage, unchanged 86% floor. Lock/sync, Ruff, both Vulture scans, mypy
747 files, actionlint, coverage artifact and naming pass. Every committed production byte
matches fifth-final. PR remains draft with both targets false. Freeze lifts for sixth
native transport lifecycle, flow dispatch/recorder ownership, and hydrated read acquisition.

Sixth wave in progress (aggregate snapshot and required gate still outstanding):
- Native CliTransport teardown owns cancellation; its duplicate process list and
  standalone batch helper disappear. Captured exited roots close inherited pipes;
  pending native spawn owners remain registered. Process-group policy is unchanged.
  Two-file source 252→242 (-10), physical 363→349 (-14, one EOF blank).
  273 focused checks pass with one existing skip; broader native/OTLP checks,
  Ruff/mypy/Vulture and independent Flow review pass.
- Runner dispatch directly owns admitted native flow selection; recorder scope owns
  profile stamping. Forwarding helpers and re-entry projection disappear; nested
  runs preserve the outer recorder's profile. Three-file source 1,638→1,602 (-36),
  physical 2,149→2,080 (-69 including comments/EOF cleanup, no source credit).
  2,016 decision and 14 real model/recorder baseline comparisons match; 911 focused
  checks, Ruff/mypy/deadcode and independent backend review pass. High CC and
  cognitive counts each rise by one after consolidation.
- Hydrated adjudication reads capture one native connection's runs instead of
  reopening and reconstructing acquisition state. Preview parses/hashes one byte
  capture; JSONL authority, immutable SQL/WAL refusal, and transport-versus-gold
  admission remain distinct. Six-file source 1,920→1,870 (-50), physical
  2,490→2,408 (-82); 37 exact real hydration/harvest cases and 171 integrated
  checks pass, plus final affected checks and a meaningful byte-race regression.
  Ruff/mypy and independent Flow review pass; high CC rises by one.
- Stage0 drops V2Projection's stored by_split mirror and repeated outcome-row
  projection. One admitted projection captures labels in file order and training
  partitions in pinned lineage order; train precedes validation for seeded SGD,
  including valid records stored in other split files. No labels-file/model/gold
  gate or resume check is removed. Two-file source 577→564 (-13), physical
  772→749 (-23); 64 actual CPU pipeline/load/resume/failure-artifact cases match,
  40 focused checks and Ruff/mypy pass. Independent backend review passes.

Next positive full-cost candidates: concrete archive import classification plan
(~103 source after callers), and captured BackendExecutionInput replacing the sole
runner factory passthrough (20 source, fixes ignored effort override on embedded
runs). Public native commit/push flattening is being verified separately.
Rejected again after full cost: test recipe Pydantic admission needs JSON-list,
unresolved-value, and source/value coupling adapters, leaving the previous small
20–30 saving and extra ownership. No edit made. Backend event normalization,
HTTP/gRPC delivery, audit/vet versus deep arbitration, and rollback modes retain
materially different obligations; their superficial similarity is not evidence
that any supported path can be deleted. These are negative assessments, not
completion or blocking conditions. Both aggregate targets remain outstanding.

Sixth integrated fixed snapshot supersedes all per-owner estimates: 83,415 source /
108,842 physical, 362 files, unchanged inventory SHA256; clones 2,252, high CC 493,
high cognitive 604. This wave removes 297 source / 418 physical lines, counting all
callers and new model fields. Aggregate reduction: 2.469% source / 3.989% physical.
Shortfall remains 6,441 source / 6,815 physical. Target remains false; no module is
moved outside the baseline inventory and no test/documentation reduction is credited.
- ImportPlan owns classified merge rows, reason ledger, and identity summary, removing
  object-id membership rejoins and four standalone partition/accounting adapters.
  Whole three-file source 1,504→1,372 (-132), physical 1,854→1,694 (-160), high CC
  12→10, high cognitive 12→9. Six thousand exact plan/error comparisons and 216
  importer/adjudication/real-CLI/publication/atomic-write checks pass. Contradictory
  generations retain historical session/row precedence. Independent Flow review passes.
- Captured BackendExecutionInput replaces the sole runner factory closure, type,
  branch, and repeated forwarding. Embedded effort overrides now use distinct native
  cache entries instead of silently receiving the default effort. Native environment,
  audit/cwd and public agent/backend contracts are unchanged. Whole three-file source
  1,424→1,404 (-20), physical 1,904→1,880 (-24), thresholds unchanged. 229 checks,
  actual native Codex/Pi construction, Ruff/mypy/Vulture and Flow review pass.
- Public phase_commit_push directly owns its one supported workflow. Unused private
  commit-only/noninteractive modes, extra context binding, forwarding signature and
  intermediate allocation disappear. Public CommitPushResult type/export remains;
  hook/index/tree checks and exact remote receipt remain. Whole two-file source
  524→488 (-36), physical 637→591 (-46), complexity thresholds unchanged.
  380 expanded checks pass with two unchanged host skips; two additional no-op and
  real commit/push recorder checks and eleven complete native Git/bare-remote
  before/after cases pass. Independent backend review passes.

All sixth production/test owners are frozen for independent aggregate review and
ordinary signed commit/push hooks. GitHub PR base remains the recorded baseline,
so no inherited upstream reduction is included. Adjudication CLI error handling
remains local: its argparse/Rich command and stage policies deliberately catch
materially different exceptions; a dispatcher-wide catch would change errors or
add routing. The next GitHub refresh acquisition prototype remains outside repo.

The sixth ordinary pre-push gate passed lock/sync, Ruff, both Vulture scans, and
mypy, then stopped at one outdated deep resolver spy missing the newly captured
execution_input keyword (9,441 passed, 14 skipped, 90.52% coverage). The test now
forwards the native keyword and exercises ambient and captured execution inputs;
all original per-phase/exit assertions remain, with identity assertions added.
No production or measurement changes were needed. Focused variants pass; the
ordinary signed amend and full hook-enabled push retry are pending.

Signed checkpoint `4d2967f6` passed the normal amend and ordinary push retry:
9,443 tests passed, 14 skipped, 203 warnings, 90.52% branch coverage against the
unchanged 86% floor; lock/sync, Ruff, both Vulture scans, mypy (747 files),
actionlint, coverage artifact and naming all passed. Runtime 593.24 seconds.
All 362 committed production paths/bytes exactly match the independently measured
sixth snapshot. GitHub confirms this head, the recorded baseline base, and draft
status; the existing PR description now reports these results and shortfalls.

Seventh wave under integration: native current run-inventory replacement resolves
a demonstrated SQL/staged-directory disagreement after license/fixture exclusion;
immutable observation history remains independent and re-admission survives. It
adds seven source lines and receives no simplification credit. Harbor Objective
will retain the scorer's immutable native metric mapping instead of 35 scalar
mirrors and bidirectional aliases; producer/caller cost is being verified.
Artifact durable record admission is prototyped outside the checkout until native
reader/recovery costs and strict identity proofs justify integration. Rejected
whole-invocation, UI lifecycle, plan authoring and RL scoring mergers retain
distinct live obligations; these negative assessments are not a blocker or a
claim that the target has been achieved.

## Seventh integrated checkpoint (verification pending)

The complete fixed production inventory contains 362 files, 108,776 physical
lines and 83,349 source lines (4.047% physical, 2.547% source reductions).
The source and physical shortfalls remain 6,375 and 6,749 lines; the target is
not met. This wave removes 66 source / 66 physical lines in aggregate, counting
the stage-index correctness change’s added lines. High CC falls to 492; high
cognitive complexity remains 604 and clone LOC falls to 2,240.

- Harbor objectives retain one immutable captured metric mapping. The mirrored
  scalar fields and bidirectional cast/alias registry disappear. Six per-run
  JSON aliases remain at the final wire encoder; suite JSON stays canonical.
  Differential values/diagnostics match in 4,051 cases; object key order is not
  claimed byte-identical. Real objective/aggregate CLI cases preserve output,
  failure admission, and workspace bytes.
- Archive rebuild replaces the current run inventory transactionally, after
  normalizing all surviving derivatives. License/fixture rejection cannot leave
  a stale SQL run row. Independent historical observations remain intact; native
  re-admission and collision behavior remains checked. This fix adds seven
  source and fifteen physical lines, rather than receiving reduction credit.
- Artifact destination serialization uses the exact declared base vocabulary,
  preventing private subclass fields from reaching the durable wire. All seven
  added durable-corruption/private-subclass regression cases remain. The proposed
  typed-record/context rewrite was rejected: readable guard layout showed a net
  six source / eleven physical lines added, with more conditional routing and
  complexity. Existing loader, manifest and transient recovery owners are retained
  byte-for-byte; the one-line privacy fix receives zero LOC credit.
- Final annotation publication passes sealed bytes directly to the installed
  native Hub CommitOperationAdd. Its sibling temporary-directory allocation,
  file-copy mapping, and staging-parent threading disappear. Both commits, CAS,
  readback, privacy scans, and verified installation remain. Actual HF byte
  hashing and publication after source-parent deletion are exercised.
- Readback reconciliation uses the three concrete vendor metadata locations
  and shared identity/timing comparison rules. 12,000 native JSON cases preserve
  exact diagnostics/errors. Readable calls remain expanded; the final net is six
  source lines, not the larger packed prototype estimate.

Focused checks passed: 720 archive/workflow checks; 218 artifact/recovery checks
plus one existing skip (before record prototype rollback), with eight final
unchanged-native-owner/privacy checks; 357 annotation/Hub/CLI checks; Harbor objective/CLI
checks and six explicit real commands; 36 readback/replay checks. Independent
owner reviews passed. Full integrated normal-hook verification is pending.
The original baseline and inventory scope remain unchanged. A literal-prompt
duplication assessment found no repeated twelve-line production string blocks
to consolidate; no prompt behavior was removed for line credit. Broader review
evidence, corpus/checkpoint, and original extraction ownership assessments continue.
