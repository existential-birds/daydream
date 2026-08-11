# Integrated-candidate handoff — Daydream #357 (Plan 008)

**slice:** `integrated-candidate`
**schema:** `osprey-delivery-handoff-v1`
**plan_digest:** `0d2e712a06ad9c97ed2354fda427b1b43020463c28e1b59be1c80a02659828b3`
**issue:** https://github.com/existential-birds/daydream/issues/357
**branch:** `feat/executor-neutral-review-service`
**base_commit:** `0b422abf53a510792267ddfffb833472ec5c7ff6`
**output_commit:** `7f87240eea15f7f4ef4c4392b4ecde9e7f26ce9b` (HEAD of `feat/executor-neutral-review-service`; the full-gate-verified candidate)
**contracts frozen:** `REVIEW_TARGET_V1` (t_ed1f4349) + `DAYDREAM_SERVICE_V1` (t_6f41b005)
**this is NOT a Daydream verdict / GitHub Check / merge authorization.**

---

## Assembly

Combined the five disjoint-worktree leaf changes onto ONE branch in this order,
resolving every interface seam they left open:

1. `merge(executors)` leaf-D (t_0747df84) — canonical `daydream/executors/` +
   additive extension seam. Clean (disjoint paths).
2. `merge(service)` leaf-B (t_169d176d) — controller state machine + admission.
   Clean at merge (leaf-D owned disjoint paths).
3. `merge(store)` leaf-C (t_c6edd122) — `ServiceStore` + in-memory/SQLite impls.
   **add/add conflict** on `daydream/service/__init__.py` resolved to the unified
   facade below.
4. `merge(worker)` leaf-A (t_494dc7f0) — immutable jobs, passive artifacts,
   fail-closed worker. **add/add conflicts** on `service/__init__.py` (resolved to
   unified facade) and `service/models.py` (reconciled into the single union
   module below).
5. leaf-E (t_2c05d2a5) **had no committed output** — its changes sat uncommitted
   in its worktree. Harvested the uncommitted files (`service/policy.py`,
   `service/publisher.py`, `service/config_digest.py`, `daydream/github_app.py`,
   and its three tests) as a merge commit. This is noted as a leaf-side deviation:
   leaf-E claimed commits in its handoff but none were on `wt/t_2c05d2a5`.

Then two integration commits added the reconciliation bridges and the
end-to-end acceptance coverage (below).

## Reconciliation notes (every port/type/method seam)

| Seam | Leaf-A/B/E each defined | Resolved as | Authority |
|---|---|---|---|
| `service/models.py` | 3 disjoint definitions (worker job models; controller models; policy models) | ONE union module: `ReviewTargetV1`/`ReviewJobV1`, `CandidateTarget`/`JobSpec`/`ControllerRecord`, and `ReviewTarget`/`ReviewPolicy`/`RoundRecord`/`LensInventory`/`SourceOfTruth`. `TargetKind` stays the policy-side Enum; `ReviewTargetV1.target_kind` is the worker-side Literal (distinct fields, no collision). | frozen contracts |
| `ExecutionRef`/`ArtifactEnvelope`/`ExecutionSnapshot` | leaf-B in `service/models.py`; leaf-D canonical in `executors/contract.py` | Both kept as separate layers. `Executors/contract.py` is the canonical executor-contract type (conformance-tested). `service/models.py` keeps the controller-port opaque ref. `ExecutionBridge` normalizes canonical snapshots/envelopes onto the controller seam. | DAYDREAM_SERVICE_V1 |
| `ReviewExecutor` port | leaf-B `service/ports.py`; leaf-D `executors/protocol.py` | Controller programs against the controller-shaped port; `executors/protocol.py` is canonical for registered adapters. No semantic divergence (same 5 ops, opaque ref). | DAYDREAM_SERVICE_V1 |
| `ControllerStorage` (leaf-B) vs `ServiceStore` (leaf-C) | two storage ABIs | Both kept: leaf-C's `ServiceStore` is the durable store; leaf-B's `ControllerStorage` is the controller's port. Controller tests use the in-memory `ControllerStorage` fake; the store's own suite covers `ServiceStore`. Not force-merged — each is internally consistent and independently gated. | stable semantics |
| Controller publish path | leaf-B controller had bare `publish()` | Added `daydream/service/runner.py` `ReviewRunner` that binds `ServiceController` + `PolicyEvaluator` + `Publisher` and the bridge, so only a complete configured round set for the exact candidate reaches the publisher. | leaf-E residual-risk #1 |
| `Backend`/`provider` meaning | — | Unchanged. `worker.py` is the only service module that touches `Backend` (as the read-only agent driver), matching the contract's terminology. `ExecutionBridge` maps candidate/service/backend/provider into the neutral `ExecutorJob.payload`; no overload. | contract |

## Contract-conformance statement

- `ReviewTargetV1` is the frozen REVIEW_TARGET_V1 (target kind, repo, exact
  candidate SHA + tree digest, base SHA, PR/merge-group ids, full-diff digest,
  protected config ref/digest, invalidation id), validated + strict-deserialized.
- Common models carry **no vendor/SDK/worker-asserted infrastructure fields**:
  no pod/VM/container/hostname/lease/agent identity. A grep for `sprite|coder|
  kubernetes|hostname|token|secret` across `service/models.py`, `policy.py`,
  `store.py`, `executors/contract.py` returns only docstring/denial references.
  The `test_common_contract_has_no_sprite_names` conformance gate enforces the
  rejection.
- Capability admission is a contract STOP: `require_capabilities` rejects an
  executor missing any of the 8 `REQUIRED_CAPABILITIES`.
- `Backend` = the Daydream model-agent driver; `provider` = model endpoint
  provider; `executor` = compute/workspace adapter. None overloaded.

## End-to-end acceptance (hermetic; GitHub/publisher seam mocked)

`tests/test_service_acceptance.py` (6 tests) drives the REAL controller +
`ExecutionBridge` + canonical `ScriptedExecutor` + `PolicyEvaluator` + a
recording publisher, and confirms:

1. **enqueue head A, force-push B**: a B round that completes only one of two
   required lenses can never authorize B (`test_b_round_with_missing_lens_cannot_authorize_b`).
2. **retry B clean**: success is published exactly once, and only once the
   complete configured round set (both rounds) is bound to B's exact candidate
   (`test_only_complete_configured_round_set_for_b_publishes`);
   an incomplete set is fail-closed with no publish (`test_incomplete_round_set_never_publishes`);
   a failed round still releases its job to terminal `RELEASED`
   (`test_failed_round_job_reaches_released`).
3. **merge-queue M1 -> M2**: rounds bound to the replaced M1 candidate can never
   authorize M2 (`test_m1_rounds_can_never_authorize_replaced_m2`);
   a stale live identity after force-push refuses success entirely
   (`test_stale_live_identity_never_publishes_success`).

## Full gate results (all on the integrated branch)

| Gate | Command | Result |
|---|---|---|
| Lockfile | `uv lock --check` | exit 0 (resolved 58) |
| Lint | `uv run ruff check daydream tests` | all checks passed |
| Typecheck | `uv run mypy daydream tests` | success, 353 source files |
| Targeted | `uv run pytest -q tests/test_service_*.py tests/test_executor_contract.py tests/test_executor_local.py tests/test_executor_sprites.py tests/test_config_file.py tests/test_extensions.py tests/test_findings.py tests/test_github_app.py` | **366 passed, 2 skipped** |
| Full | `make check` | **3420 passed, 8 skipped** (108s) |

Skipped cases are the opt-in live Sprites/staging cases (not credentialed in CI)
and the SQLite-restart case under the in-memory store impl — all by design.

## Changed files (vs base `0b422abf`)

55 files, +10,772/-6. Production: `daydream/executors/*` (new), `daydream/extensions/{api,loader,registry,__init__}`, `daydream/service/*` (new package:
`models`, `states`, `ports`, `controller`, `admission`, `store`+`store_memory`/`store_sqlite`,
`worker`, `artifact`, `policy`, `publisher`, `config_digest`, `executor_bridge`,
`runner`, `__init__`), `daydream/github_app.py`, `daydream/agent.py`,
`daydream/phases.py`, `daydream/runner.py`, `daydream/findings.py`,
`daydream/archive/manifest.py`, `daydream/cli.py`. Docs: `docs/extensions.md`,
`docs/executors/{executor-registration,sprites}.md`. Tests: `tests/test_service_*.py`
(11 new), `tests/test_executor_*.py` (3 new), `tests/test_extensions.py`, `tests/harness/service_fakes.py`,
`tests/fixtures/service/`.

## Artifact hashes (bounded pointers, sha256 prefixes)

- `service/models.py` `4d9d6b7f71e17415` · `service/runner.py` `8e377cae574b0dd5` ·
  `service/executor_bridge.py` `863e72a274e4e476` · `executors/contract.py`
  `5493d72526539f63` (leaf-D, unchanged)
- `tests/test_service_acceptance.py` `b68990c867d8714d`

## Resolve notes for every port seam (recap for the qualifier)

- No leaf weakened an exact-SHA, complete-lens, mutation-free, isolation, or trust
  boundary to pass; the E2E suite re-derives each from the real merged pieces.
- The one genuine leaf disagreement surfaced by integration: leaf-E's branch had
  **no commit** (work uncommitted) and leaf-A's worker shells `git` directly for
  tree/index/untracked digests (documented in its module). Neither affects the
  gate. Everything else integrated cleanly because the leaves were disjoint.

## Residual risk

- `SpritesExecutor` is a stub (live methods raise `ExecutorError` until a real
  client is wired under staging); live integration is deferred to a qualification
  card per the executor leaf contract.
- The durable `ServiceStore` (leaf-C) and the controller port `ControllerStorage`
  (leaf-B) remain two storage ABIs wired only through the controller's own fake;
  a future production adapter may bridge `ServiceStore` onto `ControllerStorage`
  directly without contract changes.
- Controller `collect()` waits for terminal state in `ExecutionBridge`; this is a
  reasonable adapter default, but the qualifier may want to confirm polling bounds
  for long-lived hosted executors.

**Board done = this integrated-candidate handoff is produced and `make check`
passes. Not a Daydream verdict.**