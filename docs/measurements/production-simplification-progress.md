# PR #1444 continuation

Baseline: `58cc88e4d7091b27afe1b2a6eec486504e255dfd`.
Continuation started at `b98a9dd51fde81df5dc4534a85ab29a986f6a127` on
`refactor/production-simplification`; PR #1444 was open and draft, working tree clean.

The original inventory and pinned Python 3.12 / scb-check 0.2.0 method reproduce
113,364 physical lines and 85,527 source lines at baseline. The continuation's
starting inventory reproduces 111,025 physical lines across 364 production files.
The latest integrated snapshot has 363 production files, 110,393 physical lines,
and 84,623 source lines (2.621% and 1.057% aggregate reductions). The starting
commit also reproduces its recorded 85,165 source lines. Targets remain at most
102,027 physical lines and 76,974 source lines: shortfalls are 8,366 physical and
7,649 source lines. The target is unmet; focused improvements are not completion.

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
The normal signed commit and pre-push gate are required; see PR #1444 for the
final hook-enabled push outcome. The PR must remain draft while targets are unmet.

## Remaining work

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
