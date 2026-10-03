# PR #1444 continuation

Repository: `existential-birds/daydream`; branch: `refactor/production-simplification`.
Continue this PR; do not create a replacement. Never bypass Git hooks or weaken
verification. Supported features, privacy, validation, recovery, and extension
contracts remain obligations. Internal signatures and module ownership may change.

## Reproducible inventory and current state

Baseline is `58cc88e4d7091b27afe1b2a6eec486504e255dfd`. Continuation began clean at
`b98a9dd51fde81df5dc4534a85ab29a986f6a127`. GitHub still reports the same PR base,
with no inherited upstream changes, and PR #1444 is open/draft. The latest pushed
checkpoint is signed `5378e3dae53b14d0617594ac47965e1112e00a1f`.

Count every owned Python file under daydream/, scripts/, and rl/daydream_review/,
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
| d6e99b67 | 362 | 108526 | 83157 | 492 | 603 | 2235 |
| 3d994245 | 362 | 108357 | 83028 | 491 | 603 | 2220 |
| 95d1b315 | 362 | 108292 | 82966 | 491 | 602 | 2202 |
| Latest pushed 5378e3da | 362 | 107920 | 82698 | 490 | 601 | 2190 |

Latest pushed aggregate reduction is 4.802% physical and 3.308% source;
continuation removes 3105 physical /2467 source lines. Targets remain at most
102027 physical and 76974 source: pushed shortfalls are 5893 physical /5724 source.
The twelfth batch removes 372 physical /268 source lines. The objective remains unmet;
green gates and useful changes do not establish completion. The production inventory hash remains
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

All eleven checkpoint commits and pushes succeeded through ordinary hooks. The full
hook suite includes locked sync/extras, Ruff, root and RL Vulture, root mypy,
parallel branch-coverage pytest with unchanged 86% floor, Docker actionlint,
coverage artifact, signatures and naming. Latest 95d1b315 passed 9591 tests,
14 existing skips, 199 warnings, 604.38 seconds, and 90.68% branch coverage;
root mypy checked 748 files. Earlier checkpoints passed respectively
9330, 9340, 9363, 9401, 9430, 9443, 9464, 9471, 9529, and 9545 tests with 14 existing skips.
Normal hook failures were fixed and retried: telemetry mixed event typing,
obsolete payload-labeler test seam, and execution-input resolver spy migration.
No bypass or weakened assertion/coverage threshold was used.

A broad archive run hit ENOSPC; only owned completed disposable test directories
were removed and all affected tests rerun successfully. A coordinator outside
proof accidentally recreated the shared virtual environment; locked extras were
restored immediately and the ninth full ordinary gate passed on Python 3.12.13.

## Tenth batch pushed through normal hooks

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
  55 focused tests and required normal make rl-check:199 passed, 2 existing skips,
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
owners. The tenth normal hook suite passed; all 362 committed production paths and bytes
match its independently reproduced complete report. Fresh integrated checks remain
required after subsequent production changes. Broader proposals stay outside the shared checkout.

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

## Tenth checkpoint normal-hook result

Signed 3d994245 committed and pushed through ordinary hooks. Full lock/sync/extras,
Ruff, both Vulture scans, mypy 748, Docker actionlint, coverage artifact, signatures
and naming passed. Root: 9545 passed, 14 existing skips, 202 warnings, 602.28 seconds,
90.65% branch coverage against unchanged 86% floor. Required separate RL gate: 199
passed, 2 existing skips, 160.59 seconds, locked sync/Ruff/mypy passed. Two independent
reviews reproduced the full aggregate JSON and all 362 production paths/bytes; root
compared every committed Git production byte too. GitHub head matches 3d994245,
base remains 58cc88e4, PR #1444 remains open/draft. No bypasses or weakened gates.
Freeze released for independently reviewed next-owner work.

## Eleventh batch pushed through normal hooks

- StatementLines captures one native immutable-head membership index per source,
  replacing three per-point predicate APIs and language/line dispatch. Four
  production modules cost -13 source / -22 physical; 1213 point-query outcomes
  and 50 real Git captures match, 339 integrated checks pass. High complexity
  counts stay unchanged; total cognitive mass increases by 118. Independently
  reviewed and applied-byte closed.
- CuratedBundle admits content only for the existing admitted population, while
  retaining syntax admission for all rows and verification of every listed SHA.
  Native hydration legitimately publishes no excluded payload: the old loader
  incorrectly required it. Actual hydration/materialize/project workflows and
  malformed-path/tampered-listed-file regressions pass. Cost -2 source / -1
  physical; 155 integrated checks pass; independent review and bytes are clean.
- Snapshot consumers share one strict NUL name/status stream parser. Three
  traversal implementations disappear without a new DTO or mode router. Real
  anchor/ancestry/Unicode/whitespace rename outcomes match; malformed/truncated
  streams now consistently refuse rather than infer. Cost -19 source / -18
  physical; high cognitive count -1; 277 integrated tests pass. Reviewed and
  applied-byte closed.
- GitHub preflight returns its existing admitted PreflightLedger; acquisition
  and normalization retain that repository identity instead of rereading a
  replaceable manifest behind a catch-all fallback. The public CLI and legacy
  empty-ID schema remain. Cost -6 source / -9 physical; 144 complete native
  import documents/digests match and 264 integrated checks pass. Actual manifest
  replacement regression fails before and passes after; 0600 verification ledger
  remains. Backend review clean; applied bytes equal prototype.

- Native destination record declarations admit persisted field types, eliminating
  the extraction/type-check/construction mirror. Exact raw keys/index/list shape,
  enum-byte/integer-subclass rejection, path/manifest/identity/overlap guards and
  exact-base private projection remain. TypeAdapter is confined to persisted
  admission; ordinary stdlib dataclass constructors stay unchanged. Cost -11
  source / -10 physical; 261 complete native outcomes match and 253 integrated
  checks pass with one existing skip. Reviewed and exact applied bytes closed.
- Replay uses native protobuf evidence and captures admitted fixture/manifest
  bytes through probe and receipt, removing JSON/scalar re-decoding and later
  hash rereads. Cost -8 source / -10 physical; 65 native comparisons and 42 actual
  operator checks pass. Exact key/string/value marker admission intentionally
  rejects malformed/misbound markers formerly accepted by substring matching.
  Actual input replacement regression fails before and passes after. Reviewed
  and applied-byte closed.
- SQLite owns observation-to-run winner projection at both insertion boundaries.
  Duplicate Python cache mutation disappears; metadata upsert/readmission cannot
  erase the cached human winner while leaving history intact. Existing columns,
  no-history manifest defaults, unknown-session/dedup/collision/rollback and
  read-only clients remain. This correction costs -3 source / +5 physical;
  12 exact history differentials and 515 integrated checks pass. Independently
  reviewed and applied-byte closed.

Complete frozen eleventh inventory: 362 files, 108292 physical / 82966 source,
491 high CC / 602 high cognitive, clone LOC 2202; unchanged inventory hash.
This batch removes 65 physical / 62 source lines. Aggregate reductions are
4.474% physical / 2.994% source; continuation removes 2733 physical / 2199 source.
Shortfalls remain 6265 physical / 5992 source, target false. Both independent reviewers reproduced the entire raw JSON and every production
byte and closed the complete current integrated diff clean. Ordinary commit/push
hooks passed: 9591 tests, 14 existing skips, 199 warnings, 604.38 seconds, 90.68%
branch coverage with unchanged 86% floor, mypy 748, lock/extras/Ruff/both Vulture,
Docker actionlint/coverage artifact/signatures/naming. All 362 committed Git paths
and bytes equal the frozen snapshot; proof eleventh-committed-bytes.json. Same PR
head 95d1b315/base 58cc88e4 open/draft verified and description updated to these
exact metrics/gates. The objective remains unmet; no ready-for-review claim.

The full authored native-model prototype is rejected: all nine caller modules
cost +8 source / +63 physical after preserving schema descriptions, nullable
provider wire, integral float author hashes, redacted display projection, complete
failed diagnostics and raw repair. The earlier isolated schema -35 estimate earns
no credit. New cache evidence capture and annotation acquisition leads remain
outside the checkout while this checkpoint is frozen.

## Twelfth batch active

Freeze released only after the eleventh ordinary push, committed byte proof and
PR draft update succeeded. Approved reviewed cache receipt (-6 source/-31 physical)
and streaming annotation capture (+13 source/+14 physical) may apply with byte
guards and integrated checks. Cache receipts retain all bytes of one admitted hit
until restoration (sum of its payloads rather than maximum single file); unchanged
acceptance limits and this memory tradeoff must remain explicit. Annotation capture
retains only two required inputs and consumes all other checksum files without
retention; the earlier eager full-corpus payload map was rejected for memory cost.
Native manifest field admission is applied after independent review: -8 source /
-4 physical, 218 complete admission outcomes match, and 253 integrated checks pass
with one existing skip (82.49 seconds). Ordinary transient dataclass construction,
exact raw keys and integer guards, digest/path and file/directory policies remain.
Benchmark authoring paths now resolve fresh rather than reusing a stale containment
cache: -6 source / -10 physical. Native symlink replacement previously allowed an
actual transaction write outside the workspace; four regressions retain refusal,
recovery evidence and successful recovery after restoring containment.
The full coverage-report typed-shape proposal is rejected after caller cost: +6
source / +25 physical; tuple/slash-key strata projection and legacy collision
precedence remain distinct supported boundaries, so no new mapper is applied.
All candidate costs are provisional until whole inventory remeasurement; no target
completion or cosmetic/source-packing credit. Root owns integration/metrics/PR.

Twelfth active working-tree measurement (not a committed result): 362 production
files, 108261 physical / 82959 source, high CC 492 / high cognitive 602, clone LOC
2202. Complete pinned report and all-file snapshot: twelfth-active-scb.json and
twelfth-active-snapshot under /tmp/daydream-pr1444. This batch currently removes
31 physical / 7 source lines; source/physical shortfalls are 5985 / 6234. Target
remains false. No counting scope changes or credit for docs/tests/comments.
Workers are tracing full execution, telemetry/trajectory, and archive/benchmark/
training responsibilities; root owns Improve/artifacts plus integration.

Twelfth additional integrated owners (pending aggregate gates):

- Native benchmark scalar constraints remove manual validator wrappers, retaining
  cross-state/path/UTF-8/provenance checks and the shared source-ID pattern used
  by GitHub refresh. Full two-file cost -48 source / -61 physical; 2196 exact
  admission cases and 477 integrated checks pass. An initial missed refresh
  import was corrected before application; independent final caller closure and
  both applied-byte guards are clean.
- Fork receipts retain child identity/reference/dispatch links; child trajectory
  documents alone own invocation and phase facts. Ancestor copies, recursive
  flatteners and wrapper phase-interval inference disappear. Legacy readers and
  actual analyzer billing/timing remain. Cost -40 source / -44 physical; 48 native
  workflow comparisons and 300 integrated checks pass. Meaningful child document
  assertions replace mirror assertions; independent review and bytes are clean.
- DeepData replaces the generic descriptor/dynamic-import view with the same
  live public ctx.data dictionary, static types and admission of documented public
  extension fields. Concrete FixCycleState, retained snapshot and PushReceipt
  gates remain. Invalid public types now fail before publication; incidental
  private descriptor diagnostics disappear with the layer. Full 18-file cost
  -119 source / -181 physical; 469 complete native deep checks and 379 broader
  extension/Improve/diagram checks pass outside. Guarded integrated checks run;
  concurrent diff.py state/parser changes were explicitly merged and independently
  compared definition by definition.
- Hunk indexing owns native C-quoted paths and diff framing for deep, static
  import impact, exploration and posting. Duplicate path/status records/parsers
  disappear; posting still selects raw foreign blocks before decoding. Final
  cost -38 source / -43 physical, high CC -2 / cognitive -1; 350 focused native
  tests and eight actual Git graphs pass. Independent review found and resolved
  quoted internal separators, Unicode deletion delimiters, filename metadata
  substrings and filename hunk markers. True deletion fixtures use unique bytes;
  all final populations equal actual Git NUL names. Integrated checks run.
- Publication verification consumes the same confined scratch population that
  passed SHA checks; semantic rereads and per-batch download scans disappear.
  Remote SQLite payload remains separate from the scratch index. A native A/B
  replacement previously allowed a valid second population to evade its listed
  SHA checks. All identity/count/privacy/legacy omission policies remain; 469
  integrated checks and independent applied-byte review pass. Honest cost is
  zero source / +1 physical; this correction earns no deletion credit.

The native readback operation is applied and measured only -16 source /
-13 physical after complete transport/error policies, rather than its larger
initial estimate. Native HTTPX client owns transport/deadline, bounded native
failures replace discarded status/raw tuples and repeated error projections;
45 integrated checks and Ruff/mypy pass with independent exact-byte review. Existing shallow/deep registered execution already shares the
review spine; bypassing it would remove supported custom step insertion/replacement.
Native coverage DTO and full authored-plan variants remain rejected for positive
source cost. Production reference census finds only small unreferenced helpers
or documented/runtime callbacks, not an evidenced large dead-code inventory.
The architectural/LOC objective remains unmet; broader assessed opportunities
must eliminate responsibilities across callers rather than relocate them.

Frozen twelfth complete inventory: 362 files, 107920 physical / 82698 source,
490 high CC / 601 high cognitive, clone LOC 2190, inventory hash unchanged.
This batch removes 372 physical / 268 source lines. Aggregate reductions are
4.802% physical / 3.308% source; continuation removes 3105 physical / 2467 source.
Shortfalls remain 5893 physical / 5724 source, target false. Backend independently
reproduced the full raw JSON, every production byte and complete 55-file current
integrated diff; second reproduction/review runs. Applied root framing350 (37.59s),
DeepData474 (91.92s), publication469 (107.85s), readback45 (8.93s), fork300 (67.79s),
scalar477, cache74 and manifest253/one existing skip checks pass. Scoped static
checks and both whole Vulture scans pass. Ordinary integrated repository gates,
signed commit/push, committed-byte proof and same-PR draft update remain pending.
Annotation consumption retains its final loop payload reference until admission
finishes; this minor memory hygiene observation is deferred to the next batch and
receives no deletion credit. It never creates a full-corpus payload map. Cache
receipt retention still costs the sum of one entry's admitted payload bytes.

## Twelfth checkpoint complete; thirteenth active

Signed 5378e3dae53b14d0617594ac47965e1112e00a1f passed the ordinary commit/push
hooks with locked dependency sync/extras, Ruff, both whole Vulture scans, mypy
(748 files), 9623 tests /14 existing skips /201 warnings in 612.62 seconds,
90.71% branch coverage against unchanged 86%, Docker actionlint, coverage artifact,
signatures and naming. No hook or verification bypass. Both independent full
report reproductions and current integrated reviews closed without material
findings. All 362 committed Git paths/bytes equal the frozen report snapshot and
live checkout; proof /tmp/daydream-pr1444/twelfth-committed-bytes.json. Same PR
head 5378e3da/base 58cc88e4 open/draft and description rewritten to actual evidence.
The pending-gate notes above record the pre-commit state, superseded by this result.

Fresh outside prototypes remain uncredited until integrated and remeasured:
- Review posting operation eliminates ClassifiedReviewPlan and its parallel four
  finding collections. One native ClassifiedIssues capture precedes all external
  writes. Full caller cost -36 source /-43 physical; 256 complete mutation/order/
  outcome comparisons match and 216 real workflow checks pass. Independent review
  runs; transport capability, authorization and privacy boundaries remain.
- Captured Harbor compatibility identity removes a second wheel/lock observation
  and parallel receipt/objective constructors: provisional -22 source /-43 physical.
  Actual A/B identity attribution is corrected; the separate confirmed-task tree
  launch replacement race remains unresolved, not claimed fixed.
- Full native diagram ownership and quality grammar ownership are being costed
  across all declarations/callers. Local schema/query deletion estimates earn no
  credit; opaque-string counting, generated code and moved source are forbidden.

Next checkpoint should accumulate several reviewed coherent changes before full
repository gates. Additional field/type owner or CLI boundary opportunities must
remove repeated representations/validation rather than add decoder registries.

Thirteenth applied owners (not yet aggregate-gated or committed):
- Native review submission operation removes ClassifiedReviewPlan and captures
  detached findings before ordered writes: -36 source /-43 physical, 256 exact
  mutation/outcome cases, 216 integrated workflows, Ruff/mypy2, independent review.
- Harbor compatibility identity is one captured execution owner through receipt
  and objective attribution: -22 source /-43 physical, 100 exact cases, two old
  replacement-race failures corrected, 103 integrated workflows and static checks.
  Mutable task-tree launch race remains separately unresolved; no claim of sealing it.
- Native quality Query replaces seven hand-coded grammar decoders: -63 source /
  -41 physical, 2685 complete metrics exact, paired 2.43s versus 3.04s after fixing
  the uncached prototype's 4.3x regression. 172 integrated workflows, Ruff/mypy2
  and independent exact-byte closure; no opaque grammar packing or scope change.
- Hunk indexing owns one nearest-boundary selection, preserving first positive
  ties/last containing ranges and distinct posting/demotion policy: -14 source /
  -14 physical, 12004 exact check/snap/citation/demotion outputs, 180 integrated
  workflows, Ruff/mypy3 and independent review. A malformed later tuple can now
  refuse after an earlier direct hit; actual parsed typed-pair input is unchanged.
- Annotation verified-file loop releases its final byte buffer immediately:
  zero source/physical credit, 84 corpus workflows and static checks pass.

Native test-input owners are applied with exact independently reviewed bytes:
- Persisted TestRecipe field declarations now own admission; four nested manual
  constructors disappear. PackageResolution captures selected inputs once for
  interpreter and digest, eliminating four separate readers. Full combined cost
  -49 source /-56 physical; 2075 recipe and 336 package complete outcomes match.
  Two actual open/replace races fail before and pass after; 352 integrated checks
  /3 existing skips pass in 99.71s. Ruff/mypy2 and independent applied closure pass.
- Native grounded diagram types own admitted specs across authoring/grounding/
  report/render/posting: -97 source /-95 physical across eight inventoried files.
  An independent review caught privileged raw stored-artifact acceptance drift;
  concrete raw-wire checks restore it after the same head acquisition. 702 full
  authoring and 1013 immutable-head posting outcomes match; 289 outside checks
  plus static checks pass. Independent applied-byte closure is clean; integrated
  workflow rerun settles separately. Original -109 estimate is superseded.

Second thirteenth complete working-tree snapshot: 362 files, 107628 physical /
82417 source, 487 high CC /599 high cognitive, clone LOC 2165, functions3860.
Inventory hash remains 76cc2585344c06d20088698af1466bab69e33436b8d4e3882a0511bd5590422f.
This uncommitted batch removes 292 physical /281 source; aggregate reductions
are 5.061% physical /3.634% source. Required shortfalls remain 5601 physical /
5443 source, target false. Snapshot/meta/raw pinned report are under
/tmp/daydream-pr1444/thirteenth-second-*; PR artifact remains the committed
5378 measurement until integrated gates and committed-byte proof. Native gold
publication admission and remaining quality grammar capture remain outside
pending complete proof/static/independent review. No provisional credit.

Thirteenth additional applied owners, all independently reviewed with guarded
prototype-to-checkout byte comparison and focused integrated checks:
- CaseDocument native gold admission now serves curation mutation and compilation;
  stale historical read/status/refresh remains supported. The global read-time gate
  was rejected. Complete cost -34 source/-59 physical; 541 integrated checks pass.
  Forged source provenance and content are refused before staged publication.
- Native quality guard queries remove four grammar walkers: -21 source/-18 physical,
  8062 complete metric comparisons and 192 integrated checks pass. High CC drops
  six to four and high cognitive eight to five, while total cognitive mass rises;
  no blanket cognitive-complexity improvement is claimed.
- Native tree-sitter kind captures remove three private kind sets and classifiers
  while preserving exported default/custom classification: -23 source/-27 physical,
  307 same-tree complete comparisons, 120 ordering cases and 221 integrated checks.
  Native capture ordering remains unchanged; separately parsed trees were an invalid
  ordering comparator. Total cognitive mass increases slightly.
- Adjudication Argparse actions own command admission, removing its independent
  subverb routing table and twelve argv adapters. Captured JSONL and manifest bytes
  now remain the emitted authority, fixing parse-A/publish-B substitution races.
  The obsolete independently acquiring reviewer bridge disappears; actual harvest
  qualification coverage remains. Combined -44 source/-53 physical, 800 exact CLI
  outcomes, 323 integrated checks and static checks pass. The byte-capture correction
  itself adds one source line and is included in the net cost.
- Native Improve block publication directly owns index construction, deleting a
  one-use wrapper: -14 source/-17 physical; 27 unchanged real-Git checks pass.
- Native Argparse Action owns the full-help constructor, deleting its pass-through
  initializer: -12 source/-13 physical; 100 exact parser outcomes and 175 integrated
  CLI checks pass.
- Grounded diagram integrated rerun also passes 289 checks in 63.72 seconds.

Third thirteenth complete working snapshot: 362 files, 107441 physical/82269 source,
483 high CC/594 high cognitive, clone LOC2126 and functions3851. Full pinned JSON
report and snapshot/meta: /tmp/daydream-pr1444/thirteenth-third-*. Inventory remains
76cc2585344c06d20088698af1466bab69e33436b8d4e3882a0511bd5590422f. Batch net reduction
is 479 physical/429 source; aggregate reductions 5.224% physical/3.809% source.
Required shortfalls remain 5414 physical/5295 source: objective false. No full
integrated gate or committed measurement is yet claimed for this batch. Negative
private invocation (+12 source/+16 physical) and captured-source DTO (+36 source/
+45 physical) prototypes remain unapplied; forwarding/lazy-view owners must remove
more than their native capture, reset and public-adapter obligations cost.

## Higher-level redesign requested

User steering: simplify how Daydream works, rather than continue isolated helper
cuts. Local outside prototypes are paused and uncredited. The reviewed native
local-preview SHA admission applied before that steering removes the invented
local Hub client: -10 source/-23 physical, 83 integrated checks; high cognitive
one to two is an explicit tradeoff. Third aggregate measurement predates that
application. Hydration native producer/ledger prototype remains outside pending
review of failure order and late-receipt substitution semantics.

Higher-level working model is request -> acquired evidence -> admitted findings
-> authorized repair -> verified publication. Built-in Deep/shallow/diagram and
Improve already share the registry flow engine. Replacing that engine would
mostly relocate existing policy. Three coherent axes are under assessment:
- Training carries one admitted FindingRecord population from acquisition through
  materialization/adjudication and corpus projection. Session-shaped intermediate
  state and immediate rubric JSON serialization/parsing can disappear, while
  existing formats remain serializers at supported boundaries. Benchmark case
  authoring and maintainer label harvesting retain their genuinely distinct rules.
- Repair owns retry decisions and evidence-after-mutation across fix verdicts,
  test healing and publication. A new owner only helps if states/decisions actually
  disappear; three loops currently encode different supported obligations.
- Completed run evidence owns publication as directly as possible. Source artifact
  detach, external live streaming and crash recovery are actual contracts.

A universal run-owned repository was considered as a broader execution model,
but cloning alone cannot remove source/sibling restoration: all four fixing
backends do not currently prove root confinement, and shell commands can still
address source paths. Improve excludes untracked/ignored files whereas fix/tests
need protected drafts/support files; hooks can mutate the verified tree. Any
implementation must prove these obligations rather than delete them. Likewise a
mandatory single ATIF/raw-event journal would conflate privacy-redacted telemetry,
returned model content and capped finalization evidence, or add unbounded capture.
No such replacement is applied or credited. Clarification about proposing public
capability retirement is pending; original supported-behavior constraints govern
independent implementation meanwhile.

User answered: also propose retiring capabilities. The concrete proposal is at
production-simplification-retirement-proposals.md; no removal authorized yet.
Recommended Improve exclusive cone28files7184source8818physical. Alternative
Benchmark exclusive cone29files10174source13374physical. Pi is required by the
checked-in online-RL renderer configuration, so removing it is not recommended
with training retained. Shared service discovery, historical wire readers and
independent Git preparation are not counted as retired. Awaiting user scope
choice before any feature deletion; existing authorized verification continues.

Fourth complete working snapshot includes native local-preview admission:
362files107418physical82259source,483highCC595highcognitive,2126cloneLOC,3845functions.
Inventory hash unchanged; batch removes502physical439source. Aggregate reductions
5.245%physical3.821%source; shortfalls5391physical5285source, objective false.
/tmp/daydream-pr1444/thirteenth-fourth-* holds full pinned snapshot/meta/report.
Ordinary make check started on this unchanged integrated production checkout;
no result is yet claimed. Fresh independent integrated review/reproduction run.

Fourth measurement independently reproduced by both Flow and Data: complete pinned
SCB JSON, all 362 inventory paths and every production byte match. Data's independent
cross-owner review found no material issue and ran 318 checks successfully (21.16s).
The first ordinary integrated `make check` stopped at mypy: a submission test used
an unexported facade annotation. It now imports the actual native result type from
reviews.models; focused mypy passes. Full ordinary check restarted; no full result
is claimed until it settles. No production bytes or measured totals changed.

User scope decision: KEEP BOTH Improve and Benchmark; continue internal redesign.
Retirement remains proposal-only and receives zero measurement credit. An isolated
detached candidate under /tmp/daydream-pr1444/improve-retirement-candidate explored
complete Improve callers/config/registration closure; it remains unapplied and is
not a supported replacement. No shared production capability was deleted.

Full baseline independent review is now partitioned across all changed production
paths and companion tests, with own-authored hunks assigned to another reviewer.
Backend found profile native admission retaining private raw values in its exception
cause. Three actual parser-to-traceback canaries failed before the fix; suppressing
the ValidationError cause retains the bounded public error/source and all admission.
Focused profile checks pass 45 tests; independent exact fix review is clean. The
full check started before this one-line fix, so a subsequent normal gate must cover
the final integrated bytes. Source/physical totals are unchanged; frozen production
byte proofs will be refreshed before commit.

Full ordinary integrated check before the later privacy fix completed successfully:
9696 passed, 14 existing skips, 199 warnings, 831.61s; coverage90.78% against unchanged
86% floor. Locked dependencies/all extras, Ruff, both Vulture scans, mypy750, Docker
actionlint, coverage artifact and naming passed. Later native profile privacy fix:
45 focused checks3.27s, independent exact review clean. Final hook gate remains due.

Full-baseline independent review exposed lost archive design rationale and newly
packed finding/adjudication/dedupe fields. Restored eight meaningful privacy/session/
column-projection explanations (+41physical), and expanded introduced multi-field
wire mappings/guards (+32source/+32physical). Full module ASTs are identical to HEAD
for these formatting corrections; independent review closed them clean. The seven
original archive/hub explanations are copied exactly; RunColumn is adapted to the
native SQL winner triggers. Pre-existing compact mappings in local_history.py are
unchanged from baseline and earn no PR reduction credit; avoid unrelated churn.

Sixth complete working snapshot:362files107491physical82291source,483highCC595highCog,
2126cloneLOC3845functions; inventory hash unchanged. Batch net -429physical/-407source
from5378, including all corrective costs. Source target short5317, physicalshort5464;
objectivefalse. Full pinned snapshot/meta/report: /tmp/daydream-pr1444/thirteenth-sixth-*.
Measurement artifact now represents these provisional working bytes; committed PR
is still5378 until ordinary commit/push and committed-byte proof complete.

Additional repository-wide simplification discovery uses the local deadcode-sweep
skill. Durable state: .git/daydream-deadcode-sweep/runs/pr1444-internal-redesign/.
1024 Git-tracked paths partitioned into ten disjoint native scout slices, all owned
hashes and initially unvisited statuses recorded. Data/Flow scouts start slices1/2;
Backend starts slice3 after full baseline review closes. Remaining slices run in
waves using existing agents. This is INCOMPLETE discovery, not a completed sweep.
No new sweep finding may be applied before every slice has substantive triage.
Existing outside native finding/repair prototypes are paused, partial or modest
and uncredited. Keep Improve/Benchmark and all supported capabilities throughout.

Full baseline review also restored hostname normalization, working-directory schema
spellings and PrioritizationFacts hash-exclusion invariants (+13physical), plus the
import evidence-anchor fallback rationale and readable hydrated/finding projection
(+7physical/+5source). Import executable AST unchanged. Seventh complete working
snapshot:362files107511physical82296source;483highCC595highCog2126cloneLOC3845functions.
Inventory unchanged; batch5378→working -409physical/-402source. Target remains false:
physical short5484/source short5322. Full pinned seventh snapshot/meta/report under
/tmp/daydream-pr1444/thirteenth-seventh-*. These are provisional working bytes;
committed PR remains5378 pending ordinary final hooks and exact committed proof.

Independent full-baseline review exposed a previously moved test double: original
counted archive/hydrate_client.py→tests/harness/hub.py. Restored the verified current
FakeHub bytes to the original counted path and removed the harness copy; migrated
seven test/fixture imports only. No implementation or assertion changed. All135
focused native consumers pass3.16s; Backend independently closes exact bytes/callers.
No moved-out inventory reduction credit remains. Fresh eighth full measurement:
363files107691physical82441source,483CC595Cog2139cloneLOC3858functions; inventorySHA
036bd5e167a222b8e1986c2f4e77a2733d8ebf9e92b75daa468ad226993c92fe.
Aggregate5.004%physical3.608%source; targets remain false (short5664physical5467source).
Batch5378→working now -229physical/-257source, after all corrective costs. Full pinned
eighth snapshot/meta/report outside/tmp/daydream-pr1444. Existing full gate preceded
profile privacy and relocation corrections; final ordinary hook gate remains due.

Full-baseline reviewers Flow/Data have closed assigned scopes. Backend closed142
production and91 companion test/fixture paths; five own-authored boundaries/tests
are explicitly qualified pending Data complement (flows/engine,deep/orchestrator,
diagram_grounding/evidence,reviews/diagrams,artifacts/ledger). No self-approval is
accepted. Resolved findings: profile private cause, lost design rationale, packed
wire formatting and moved fixture credit. Supplemental restored FakeHub reviewed.

Local sweep planned triage slices1/2 complete (126/96 unique substantive records
reconciled); Data complement→slice4,Flow slice5,Backend slice3. One slice1 later
import-only delta awaits owner recheck; all ten passes/global ranking still due.
No new sweep candidate applied. Larger native ownership hypotheses remain partial
with explicit caller/format obligations; modest estimates receive no target credit.
