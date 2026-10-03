# PR #1444 continuation

Repository: `existential-birds/daydream`; branch: `refactor/production-simplification`.
Continue this PR; do not create a replacement. Never bypass Git hooks or weaken
verification. Supported features, privacy, validation, recovery, and extension
contracts remain obligations. Internal signatures and module ownership may change.

## Reproducible inventory and current state

Baseline is `58cc88e4d7091b27afe1b2a6eec486504e255dfd`. Continuation began clean at
`b98a9dd51fde81df5dc4534a85ab29a986f6a127`. GitHub still reports the same PR base,
with no inherited upstream changes, and PR #1444 is open/draft. The latest pushed
checkpoint is signed `d6e99b67eaea6f9c15e588bf40f8ebc5e91384d0`.

Count every owned Python file under daydream/, scripts/, and the RL package,
including published runtime templates and all new modules. Exclude vendored ATIF
and RL tests exactly as the baseline method does. Physical LOC uses str.splitlines;
source LOC uses Python 3.12 and pinned scb-check 0.2.0 --report --include-all.
Complete JSON reports are required; exit 1 with a complete report denotes findings.
See production-simplification-reproduce.md and production-simplification.json.
Comments, docstrings, blanks, docs, and tests receive no source-reduction credit.

| State | Files | Physical | Source | High CC | High cognitive | Clone LOC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 341 | 113364 | 85527 | 504 | 624 | 2440 |
| Continuation start b98a9dd5 | 364 | 111025 | 85165 | 501 | 620 | 2375 |
| 3a79584d | 363 | 110393 | 84623 | 497 | 612 | 2336 |
| 0a54165e | 363 | 110239 | 84495 | 495 | 610 | 2332 |
| 7f8bc913 | 362 | 109892 | 84225 | 496 | 610 | 2290 |
| 2a8b0b23 | 362 | 109599 | 83971 | 493 | 607 | 2264 |
| a187b7e7 | 362 | 109260 | 83712 | 493 | 605 | 2244 |
| 4d2967f6 | 362 | 108842 | 83415 | 493 | 604 | 2252 |
| 25d1ac7c | 362 | 108776 | 83349 | 492 | 604 | 2240 |
| 2f35025b | 362 | 108612 | 83232 | 492 | 604 | 2240 |
| Latest pushed d6e99b67 | 362 | 108526 | 83157 | 492 | 603 | 2235 |
| Tenth frozen working snapshot | 362 | 108357 | 83028 | 491 | 603 | 2220 |

Latest pushed aggregate reduction is 4.268% physical and 2.771% source;
continuation removes 2499 physical /2008 source lines. Targets remain at most
102027 physical and 76974 source: pushed shortfalls are 6499 physical /6183 source.
The frozen next batch removes 169 physical /129 source lines. It is not yet
a pushed checkpoint. Its remaining shortfalls are 6330 physical /6054 source. The objective remains unmet; green gates and useful changes
do not establish completion. The production inventory hash remains
`76cc2585344c06d20088698af1466bab69e33436b8d4e3882a0511bd5590422f`.

## Architectural responsibilities consolidated

- Backend execution owns native subprocess pipes, draining, cancellation, and
  teardown. Transport mode enums, synthetic exit conversion, wide factory kwargs,
  and duplicate request projections disappear. Osprey owns one frozen config;
  Pi dispatch owns arguments and retry classification. Runner/FlowContext retain
  captured execution input, including distinct effort cache identity.
- GitHub import uses the existing authenticated REST/GraphQL API boundary for
  parsing, bounded rate-limit recovery, and read-only GraphQL timeout retries.
  PR evidence captures successful acquisition once, failed acquisition retries,
  and local per-path Git semantics remain for renames.
- ArtifactSession owns freezing, publication, and rollback. Reciprocal adapters
  and the finalization module disappear; manifest-last staging, durable journals,
  destination identity, external exchange, and recovery ordering remain. A rejected
  typed-record router is absent; exact-base private wire projection is retained.
- FindingRecord retains immutable source, adjudication, identity, and raw extras.
  Projector/source resolution, common adjudicated eligibility, fresh human-gold
  evidence, and a native ImportPlan replace state restoration, fingerprint rejoins,
  and adapter chains. Historical excluded rows remain observable.
- Hydration uses native persisted wire, one SQLite ownership boundary, ordered
  inventory, and one captured license policy for admission/history/curation binding.
  Discarded result models/counters and second policy parsing disappear. Stage index
  replacement fixes stale excluded runs; this correction added lines rather than
  earning reduction credit. Historical observations and collision checks remain.
- HarvestPass owns config, archive, acquisition, and workflow. The service protocol,
  factory, reciprocal forwarding, comment-login mirror, and replied-ID set disappear.
  Successful reviewer/resolution evidence uses the same captured comment population;
  standalone APIs and retryable failures preserve their acquisition obligations.
- FinalAnnotationBundle owns semantic bytes, identity, and integrity. Native byte
  mappings publish through HubCommitOperationAdd; staging path copies and repeated
  re-projection disappear. Legacy path-based upload APIs remain supported.
- Improve scope repair reconciles one mapping. The admitted authored plan remains
  the rendering/landing authority; duplicate numbering, expanded-command, STOP,
  selection, and landing projections disappear. Landing shares publication/index
  policy while retaining distinct drift cleanup. Successful diagnostic hashes and
  shape counts intentionally bind the retained admitted author object, with exact
  tests and extension documentation; failed diagnostic semantics remain.
- RecordPool owns admitted review records and UIDs through merge/salvage, eliminating
  disk mirrors and empty-proof reconstruction. Dedup captures normalized grams once;
  200 records still yield the same 19900 pairs with 200 normalization calls, formerly
  20100. Review coverage, incomplete evidence, resume, and custom builders remain.
- Scoring shares matching/reward/detail behavior and native admitted finding content.
  Runtime judge imports have normal package ownership; module registry rewriting and
  asset caches disappear. HarborObjective owns immutable canonical metrics rather
  than scalar mirrors. Standalone runtime artifacts and public reward APIs remain.
- Telemetry binds the actual invocation ledger, not the last completed sibling.
  Native SDK event evidence and billing stay separate from detached observer privacy
  projections. Response labels and preflight diagnostics now pass existing bounded
  identity/structured redaction policy. These security fixes added 29 source lines
  in the ninth batch and receive no reduction credit.
- Native configuration metadata replaces decoder registries. PhaseEvent owns one
  closed wire shape. Readback binds accepted rows/count/hash to one capture. Empty
  gRPC adapters, unused preflight result allocations, static UI ID mirrors, and
  responsibility-free helpers disappear without removing supported behavior.

## Verification and review record

Every applied owner was independently reviewed and byte-compared against its
reviewed prototype. Differential and actual workflow checks cover native subprocesses,
loopback OTLP, SQL/history/ledger, real Git/origin and hooks, CLI acquisition,
standalone verifier/wheel/runtime lock, plan landing/resume/custom builders, and
archive publication. Current owner reviews are complete; historical unchanged draft
areas have sampled broader review, not a claimed fresh exhaustive baseline review.
Before final completion, review the full integrated baseline diff independently.

All nine checkpoint commits and pushes succeeded through ordinary hooks. The full
hook suite includes locked sync/extras, Ruff, root and RL Vulture, root mypy,
parallel branch-coverage pytest with unchanged 86% floor, Docker actionlint,
coverage artifact, signatures and naming. Latest d6e99b67 passed 9529 tests,
14 existing skips, 200 warnings, 624.22 seconds, and 90.62% branch coverage;
root mypy checked 748 files. Earlier checkpoints passed respectively
9330, 9340, 9363, 9401, 9430, 9443, 9464, and 9471 tests with 14 existing skips.
Normal hook failures were fixed and retried: telemetry mixed event typing,
obsolete payload-labeler test seam, and execution-input resolver spy migration.
No bypass or weakened assertion/coverage threshold was used.

A broad archive run hit ENOSPC; only owned completed disposable test directories
were removed and all affected tests rerun successfully. A coordinator outside
proof accidentally recreated the shared virtual environment; locked extras were
restored immediately and the ninth full ordinary gate passed on Python 3.12.13.

## Tenth batch under integration

- FindingContent owns content validation and closed projection across five native
  host/runtime modules. Gold/candidate objects feed judge and scoring directly;
  parallel raw rows and generic object/dict readers disappear. Cost is -22 source
  /-31 physical; 2400 complete native differential outputs match and 373 actual
  verifier/Harbor/fresh-wheel checks pass. Seven multiply-invalid internal inputs
  keep rejection category with changed first error ordering. Oversized/body/privacy,
  malformed, concurrency, numeric, and no-judge assertions remain meaningful.
- Stage-3 scoring retains the native OutcomeModel admitted by its gate. Path-keyed
  global cache, repeated load, and per-task path mirror disappear. Valid A/B model
  replacement now produces distinct scores; already admitted tasks retain A while
  future loads re-admit B. Actual spawned worker transfer and intrinsic-only reuse
  regressions pass. Cost -9 source/-19 physical; 40 exact reward differentials,
  55 focused tests and required normal make rl-check:199 passed,2 existing skips,
  160.59 seconds, lock/sync/Ruff/mypy all pass.
- RFT native replay admission owns flat identity validation. One input-byte read
  binds sampled rows and result/header checksum; replacement cannot label A winners
  with B's hash. Cost -21 source/-21 physical, high CC -1;152 exact native cases,
  59 replay/projector/gate/stacks plus15 coordinator/lineage checks pass.
- Existing HTTP and gRPC transports are native SpanExporters. CompatSpanExporter
  and duplicate timeout/ledger ownership disappear; HTTP encoding retains an
  invocation-local deadline start. Actual concurrent slow/fast export proof formerly
  delivered both and now rejects the over-budget export with the bounded diagnostic.
  Cost -12 source/-20 physical;199 integrated delivery/runtime/native SDK checks pass.
- Git queries share exact NUL-delimited name admission. Quoted/trimmed readers and
  duplicate untracked enumeration disappear; strict/best-effort policies, two-dot
  range, 5/10-second timeouts, confinement, and first-seen order remain. Real Unicode,
  whitespace, tab/newline filenames now survive enumeration and generated-file
  recovery while user bytes remain. Cost -39 source/-46 physical across all three
  production callers;616 actual tests pass with4 existing skips. Review found and
  fixed the snapshot caller migration. An isolated setup failure was diagnosed by
  persisted CI receipt: gh's env python3 lacked pytest. Correct venv PATH restored
  the full suite; no CI timing threshold or behavior was changed.

- Oracle supervision retains its already admitted outcome and gold gate through
  receipt publication, eliminating a second result parse and two sole-caller private
  wrappers. Post-confirmation config/lock/wheel/current-identity reads and paid gates
  remain. Cost -26 source/-32 physical;144 complete native supervisor comparisons
  match,64 run/objective checks pass, and permission/write-failure cleanup assertions
  remain meaningful.

Ruff, scoped mypy, root Vulture, applied-byte reviews and diff checks pass for these
owners. Full final hook verification and committed-byte aggregate checks remain
required for the next checkpoint. Broader proposals stay outside the shared checkout.

## Remaining opportunities and rejected directions

Target shortfall requires substantive further work. Active coherent outside leads:
Immutable-head statement grounding
indexed once per source; curated-bundle checksum/manifest admission and per-file
publication capture; fix/test/verification lifecycle state ownership.

Reject extraction-only rewrites and new mode/callback frameworks that retain the
same responsibility. Complete caller costs matter, including models/adapters and
all new production files. No arbitrary candidate-size floor applies, but small
cuts must not be presented as massive simplification.

Rejected end states include: agent closure doubling nested complexity; typed artifact
router adding lines/routes; snapshot JSON cache adding a third representation;
payload-only snapshot serializing repeatedly (4.7x hash-read cost); full captured
Harbor input bypassing post-confirmation file revalidation; globally captured bundle
payloads adding large-trajectory memory retention; uniform strict legacy calibration
loader changing wire/admission semantics; generic exporter retry router merging real
HTTP/gRPC obligations; combining rollback/healing policies that differ on authority
and errors; generated schema replacements retaining repair/admission converters.
Distinct billing, terminal coverage, phase lifetimes, public extension dictionaries,
read-only capture, exact path, and crash-recovery policies remain essential.
