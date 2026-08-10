# Executor-conformance leaf handoff — Daydream #357 (Plan 008, Step 4)

**slice:** `executor-conformance`
**schema:** `osprey-delivery-handoff-v1`
**plan_digest:** `0d2e712a06ad9c97ed2354fda427b1b43020463c28e1b59be1c80a02659828b3`
**issue:** https://github.com/existential-birds/daydream/issues/357
**base_commit:** `0b422abf53a510792267ddfffb833472ec5c7ff6`
**leaf_worktree:** `/Users/ka/github/existential-birds/daydream/.worktrees/t_0747df84` (branch `wt/t_0747df84`)

## Drift check

`git diff --stat 0b422abf5..HEAD -- daydream/extensions daydream/executors tests/test_executor_contract.py tests/test_executor_local.py tests/test_executor_sprites.py`
→ empty. Dead clean at planned commit; no in-scope file changed by another leaf.
No semantic mismatch — this leaf began work and completed on-plan.

## Contracts consumed (frozen)

- `REVIEW_TARGET_V1` (t_ed1f4349): ReviewTarget distinguishes `pr_head|merge_group`,
  carries repo, exact candidate SHA+tree, base SHA, PR/merge-group ids, full-diff digest,
  protected config-source digest, invalidation id.
- `DAYDREAM_SERVICE_V1` (t_6f41b005): neutral executor port + opaque ExecutionRef;
  no vendor fields in common models; capability admission is contract-verified.

## What this leaf added

New public-core package `daydream/executors/`:
- `contract.py` — neutral DAYDREAM_SERVICE_V1 models, frozen: `ExecutionRef` (opaque
  handle + attempt_id), `ExecutionSnapshot` (lifecycle only: NO pod/VM/lease/workspace/
  hostname identity), `ArtifactEnvelope` (bounded outcome/lenses/content-hash, no raw bytes,
  no adapter identity), `ExecutorJob`, `ExecutorCapability` + `REQUIRED_CAPABILITIES`
  (every capability a merge-authorizing executor must prove), `ExecutionStatus`/`Outcome`,
  the neutral `ExecutorError` hierarchy, and `map_vendor_error` (adapter-internal vendor
  exception → neutral error; vendor type never leaks). `require_capabilities` is capability
  admission, a contract STOP when a required capability is undeclared.
- `protocol.py` — the `ReviewExecutor` port: `start/job->ref`, `inspect/ref->snapshot`,
  `cancel`, `collect/ref->envelope`, `release(ref, disposition)`. `is_review_executor` cheap
  structural check (registry seam). Deliberately mirrors the Backend seam's async-port shape
  but is a SEPARATE concern: `Backend` drives model-agent turns, `ReviewExecutor` owns
  execution/workspace lifecycle. **The meaning of `Backend` is unchanged.**
- `local.py` — `LocalExecutor`, hermetic reference adapter: real per-execution on-disk
  workspace, time-based asyncio lifecycle, durable on-disk state so a fresh instance over
  the same root reconciles a prior ref (`restart_reconciliation` against real storage).
  Deterministic `release` cleanup order (artifacts → state → workspace dir).
- `scripted.py` — `ScriptedExecutor`, the SECOND structurally different hermetic adapter:
  in-memory store, step-based lifecycle (each `inspect` advances one step), no I/O, no
  clock, no asyncio task. Shares the caller-owned store across instances to model restart
  reconciliation. Proves the conformance suite carries no filesystem/timing/store assumption
  into the contract (suite has no adapter-typed branching — it polls `inspect` via `settle`).
- `sprites.py` — optional reference hosted-integration adapter, strictly adapter-scoped.
  Every Sprite name/SDK/API type lives in this module; none appears in common models
  (enforced by `test_common_contract_has_no_sprite_names`). Live execution is opt-in via
  `DAYDREAM_SPRITES_STAGING=1` + explicit connection; hermetic path refuses loudly with
  `ExecutorError`. Stub lifecycle until live staging is wired; quarantine-on-ambiguous-cleanup
  and one-exclusive-clean-execution policies documented.

Extension seam `daydream/extensions/{api.py,registry.py,loader.py,__init__.py}`:
- additive registration: `Registry.register_executor(name, executor)`
  enforces structural conformance + capability admission at registration (weak/duplicate/
  out-of-range-service-version rejected with `ExtensionError`); `register_publisher` names a
  trusted publisher (credential-safety is the publisher leaf's contract; not capability-admitted
  here). Resolve via `executor()/executor_if_registered()/executor_names()` and
  `publisher()/publisher_if_registered()/publisher_names()`.
- `DAYDREAM_SERVICE_V1`/`MIN_SUPPORTED_DAYDREAM_SERVICE_V1` re-exported from
  `daydream.extensions.api` and the `daydream.extensions` facade, additive — does NOT bump
  `EXTENSION_API_VERSION` (stays 5, asserted). `loader.py` docstring documents that
  `build_registry` builds the same Registry a fork mutates, so `register()` may call the
  new seam.
- `daydream/cli.py` `ext validate` now reports registered executor/publisher counts + names.
- Docs: `docs/extensions.md` seam section; new `docs/executors/executor-registration.md`
  (generic Coder/Kubernetes/external-package guide: port, required capabilities, registration,
  how to run the same conformance suite, behavioural obligations, authoring checklist) and
  `docs/executors/sprites.md` (binding live behaviours to verify first; SKIPPED live gate).

Common conformance suite `tests/test_executor_contract.py` (parametrized over BOTH hermetic
adapters, same assertions run twice; no adapter-typed branching):
- capability admission (declared covers required; a partial adapter is rejected)
- opaque handles (typed `ExecutionRef`; never parsed)
- start/inspect cancel/collect/release lifecycle
- idempotency (repeat start with same identity binds same execution; distinct attempts distinct)
- restart reconciliation (fresh instance over same durable backing sees a prior ref)
- cancel → cancelled outcome; collect non-terminal raises; release-then-collect raises
- vendor-error mapping (simulated SDK exception → neutral `INFRA_ERROR`, never raw)
- vendor-neutrality gate (`test_common_models_carry_no_vendor_fields` asserts snapshots/
  envelopes/refs carry no infra identity beyond the opaque handle)

Real-path Local tests `tests/test_executor_local.py`: genuine workspace dir creation,
deterministic release cleanup ordering (workspace fully removed, shared root survives),
fresh-instance restart reconciliation, vendor-error mapping, cancelled-collection,
uniform unknown-ref error.

Optional Sprites tests `tests/test_executor_sprites.py` — live cases SKIPPED:
kind/capability declaration, live gate defaults off, hermetic call refuses without staging,
the live-lifecycle case is `@skipif(not DAYDREAM_SPRITES_STAGING)` and never runs in CI,
and vendor-name isolation (no `sprite` in any common-model field; contract module does not
import the adapter).

Extension-seam tests `tests/test_extensions.py`: register+resolve, name isolation, duplicate
rejection, non-conformant rejection, weak-capability admission failure, out-of-range service
version rejection, unresolved/-if_registered, publisher register+resolve+duplicate+unknown,
and the additive-contract guard (`EXTENSION_API_VERSION == 5`; seam rides current version).

## Changed files

- `daydream/executors/` (new): `__init__.py`, `contract.py`, `protocol.py`, `local.py`,
  `scripted.py`, `sprites.py`.
- `daydream/extensions/{__init__.py,api.py,registry.py,loader.py}` — additive seam (+executors,
  +publishers, +version constants), no `EXTENSION_API_VERSION` bump.
- `daydream/cli.py` — `ext validate` reports executor/publisher counts + names.
- `docs/extensions.md` — executor/publisher seam section.
- `docs/executors/` (new): `executor-registration.md`, `sprites.md`.
- `tests/test_executor_contract.py` (new), `tests/test_executor_local.py` (new),
  `tests/test_executor_sprites.py` (new), `tests/test_extensions.py` (new).

## Verification results

- `uv lock --check` → exit 0.
- `uv run pytest -q tests/test_executor_contract.py tests/test_executor_local.py
  tests/test_executor_sprites.py tests/test_extensions.py` → **53 passed, 1 skipped** (the
  skipped is the live Sprites staging case, gated off).
- `uv run ruff check daydream tests` → All checks passed.
- `uv run mypy daydream tests` → Success, no issues in 325 source files.
- Live staging cases not run: they are opt-in and separately credentialed, per contract.
- NOTE: full worktree suite not re-run (sibling leaves own unrelated paths); the targeted gate
  listed in the plan is green.

## Artifact hashes (bounded pointers, sha256 prefixes)

- `contract.py` `5493d72526539f63`
- `protocol.py` `7e7644927c567e24`
- `local.py` `86badbea083e0155`
- `scripted.py` `76b0dab424067fea`
- `sprites.py` `88a316cf3409be3d`
- `executors/__init__.py` `69617262e497fb11`
- `docs/executors/executor-registration.md` `003b62d7d0761984`
- `docs/executors/sprites.md` `8a777105dd6342a0`
- `test_executor_contract.py` `3c42f05cfe131307`
- `test_executor_local.py` `c93ddd0ec58b04e4`
- `test_executor_sprites.py` `af01b9d73e73e3d9`
- `test_extensions.py` `82ebe529b3de9ffc`

## Retry notes

Prior run crashed mid-leaf (pid died ~53m in) leaving a partial worktree; all four gates were
already green on re-verify. No classified infrastructure failure was retried. Standard
pre-push hook is not installed on sprite clones (they are hook-free); no hook bypass occurred.

## Residual risk / integrator notes

- This leaf is the executor modularity + conformance slice only. It does NOT wire the executor
  registry into a running controller state machine, nor bind publishers — those are sibling/
  later leaves. Integration (t_ce22c871) must consume `ReviewExecutor`, `ExecutionRef`,
  `ArtifactEnvelope` and the registry `executor()/publisher()` lookups into the controller's
  evaluated→publishing→passed transitions.
- `SpritesExecutor` is a stub: live methods raise `ExecutorError` until a real Sprite client
  is wired up under explicit staging. The adapter module is genuinely adapter-scoped and the
  live-integration work is deferred to a qualification card that verifies lifecycle/exec/Task/
  checkpoint/disk/Connector/delete against first-party docs BEFORE hardening the live path.
- `ScriptedExecutor` and the local `inspect().started_at_iso` fields use "now" snapshots rather
  than a persisted monotonic start; the conformance suite does not assert timestamp stability,
  so this is intentionally not pinning clock semantics for the fakes.
- `tests/test_extensions.py` is a NEW file this leaf owns (the sibling publisher leaf noted the
  plan's `test_extensions.py` did not exist as `test_extensions_loader/registry.py`; this leaf's
  verify line names this new seam-test file and it passes).
- The common contract stays frozen: no adapter may weaken required capabilities or require a
  common-schema change. This leaf introduces only additive extension-surface changes and does
  not change `Backend`.

**Board "done" = this implementation handoff is complete; it is not a Daydream verdict,
GitHub Check, approval, or merge authorization.**