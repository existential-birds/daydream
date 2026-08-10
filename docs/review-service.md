# Review Service — operating and design manual

This document is the self-contained operating/design reference for Daydream's
**executor-neutral review service** (`daydream.service` + `daydream.executors`),
introduced as the consolidated plan-008 candidate (issue #357). It describes the
service as it exists in the code today. It complements the two adapter docs under
[`docs/executors/`](executors/executor-registration.md): this page is the service
as a whole; those pages are the "how do I write an adapter" guides.

> **Scope warning.** The service is a library plus a runner hook — it is not yet
> a long-running network daemon. There is **no HTTP/network listener** in this
> codebase; "port" throughout this document means a `typing.Protocol` seam
> (`ControllerStorage`, `ReviewExecutor`, `Publisher`), not a TCP port. Where a
> component is a stub or not yet live it is called out explicitly.

---

## 1. What the service is for

The review service makes a durable, **fail-closed** decision about whether an
*exact* candidate commit may be authorized. It is executor-neutral: the core
(`daydream.service`) knows nothing about *where* a review runs (local process,
hosted VM, Sprites, Kubernetes, ...). Compute/workspace lifecycle is behind an
opaque `ExecutionRef` produced by a registered `ReviewExecutor` adapter
(`daydream.executors`).

The flow, end to end, is:

```
forge event
   -> immutable ReviewTargetV1 / ReviewJobV1        (the exact candidate + job)
   -> ServiceController                             (durable state machine)
        -> ReviewExecutor.start / inspect /          (compute/workspace, opaque ref)
             collect / cancel / release
        -> ServiceStore                              (transactional, crash-safe)
   -> PolicyEvaluator                                (fail-closed, protected policy)
   -> Publisher (GitHubChecksPublisher)              (the ONLY checks-write holder)
        -> revalidates live candidate identity before success
```

The composition that ties the leaves together is `daydream/service/runner.py`
(`ReviewRunner`): enqueue -> dispatch -> collect -> evaluate each round, aggregate
the configured round set, ask the `PolicyEvaluator`, and only publish the exact
decision. A stale/replaced candidate or an incomplete round set can never publish.

## 2. Terminology (used consistently in this doc and the code)

- **backend** — the Daydream model-agent driver (`claude` / `codex` / `pi`).
  Meaning is unchanged from the rest of Daydream.
- **provider / model** — the model endpoint provider and the concrete model id.
- **executor** — the compute/workspace adapter (`local`, `scripted`, `sprites`,
  or a future `coder` / `kubernetes`). This is the *new* seam.
- **publisher** — the external writer that durably records a decision (GitHub
  Checks). Distinct from an executor.

None of these overload `daydream.Backend`. The executor seam is a separate
concern from the agent driver.

## 3. Config source and precedence

There are two distinct configuration surfaces, and it is important not to blur
them.

### 3.1 Backend / run configuration (how the worker's agent is picked)

The service-mode worker resolves its model backend through the **normal Daydream
precedence chain** (`runner._resolve_backend`): CLI `--model`/`--backend` >
config-file per-phase override > config-file global > backend default. The
service-mode hook `runner.run_service` resolves a backend with the `"review"`
phase key and hands it to the fail-closed worker as a read-only agent turn
(`phase_service_review`, `read_only=True`).

These keys **do not affect merge authorization** — see the policy surface below.

### 3.2 Merge-authorizing policy (the trust boundary)

Merge-authorizing policy is *never* taken from the PR head or an ambient
unpinned override. `daydream/service/config_digest.py` resolves it from a
**protected source** only, in this preference order (`resolve_policy_source`):

1. **Explicit protected per-service source** — a controller-owned, pinned,
   digest-bound source (`protected_source_config` + ref + sha). Wins because it
   is explicitly designated as the protected per-service config.
2. **Protected base/default-branch snapshot** — `base_config` read at
   `base_sha` (`refs/heads/main`).
3. **Never the PR head.**

An *ambient* (unpinned / PR-controlled) file may re-expose the protected policy
for development but can **never lower its strength**: ambient values only apply
for keys the protected effective config did not already set (`_merge_ambient_without_weakening`).
If no protected source is available at all, resolution fails closed with
`ProtectedPolicyError` rather than guessing.

### 3.3 Effective-config digest

Every round and every published Check is bound to one immutable effective policy
by the **canonical effective-config digest** (`policy_digest`). It:

- reduces the policy to its merge-authorizing fields (a fixed `_NON_AUTHORIZING_KEYS`
  set is excluded — model, budgets, quality gates, supervisor, phases, improve,
  bench, etc. never influence authorization);
- canonicalises it into an order-independent SHA-256 digest (list-of-strings
  values such as the lens inventory sort; nested dicts recurse).

Any change to an authorizing field (round count, backend, provider, model, lens
policy, executor, publisher, Check name, budgets) changes the digest. The
`PolicyEvaluator` refuses to evaluate unless a round's digest matches the
target's protected config digest (`policy.source.digest == target.config_source.digest`),
so a round or Check bound to an old digest can never be mistaken for current
policy.

## 4. Ports

The controller programs against narrow `typing.Protocol` seams so the storage
and executor implementations can be swapped without touching the state machine.

### 4.1 `ControllerStorage` (`daydream/service/ports.py`)

The durable, transactional store seam. All methods are async. Exceptions:
`StoreConflict` means "the row moved under you — re-read and retry"; a normal
not-found returns `None`. Operations: `insert`, `load`, `transition` (CAS),
`bind_execution`, `bind_artifacts`, `mark_superseded`, `bump_retry`.

### 4.2 `ReviewExecutor` (`daydream/executors/protocol.py`, canonical DAYDREAM_SERVICE_V1)

The compute/workspace adapter. Exactly five methods, all async:

```
start(job: ExecutorJob) -> ExecutionRef        # begin; idempotent per identity
inspect(ref: ExecutionRef) -> ExecutionSnapshot
cancel(ref: ExecutionRef) -> None              # strong cancel
collect(ref: ExecutionRef) -> ArtifactEnvelope # terminal only
release(ref, disposition: str) -> None         # deterministic cleanup
```

`ExecutionRef` is `executor_kind + adapter_version + opaque_handle + attempt_id`.
The controller **never parses `opaque_handle`** — it stores it and, on restart,
feeds it back to the originating executor via `inspect`. Every adapter must
declare `kind`, `adapter_version`, and `capabilities`.

The controller-facing surface (`daydream/service/ports.py`) is a normalized
variant; the single seam that maps canonical adapters onto it is
`daydream/service/executor_bridge.py` (`ExecutionBridge`). It keeps the
controller's ref and the canonical ref as separate values, normalizes
snapshots/envelopes, and polls a terminal status (bounded) before collecting for
step-based adapters.

### 4.3 `Publisher` (`daydream/service/publisher.py`)

A narrow, trust-neutral port: `publish(PublishRequest) -> PublishReceipt`.
`PublishRequest` binds an immutable `external_id`, a `conclusion`, a **bounded
non-secret summary**, repo, exact `target_sha`, `check_name`, and the full
`ReviewTarget`. `publish` raises `PublishError` on any failure, and a caller must
treat any exception as **not-published** (a retry is a fresh explicit call — it
never flips a failure to success).

## 5. Capabilities and capability admission

Every merge-authorizing executor must prove **all** of these `REQUIRED_CAPABILITIES`
(`daydream/executors/contract.py`):

| Capability | Meaning |
|---|---|
| `exclusive_workspace` | each execution in its own isolated workspace; no shared writable state |
| `no_ambient_credentials` | worker env carries no ambient forge / secret-manager / executor-control credentials |
| `source_read_only` | reviewed source staged read-only and proven unchanged |
| `bounded_egress` | outbound network is bounded |
| `durable_execution_identity` | an execution identity survives across calls |
| `strong_cancel` | `cancel` interrupts promptly; `inspect` then reports `cancelled` |
| `deterministic_release` | `release` removes resources in a fixed order; ref gone after |
| `restart_reconciliation` | a fresh adapter can `inspect` a stored opaque ref it previously created |

Admission is a **contract STOP**, enforced at two points:

- `require_capabilities(...)` / `AdmissionController.admit_executor` — rejects an
  executor missing any required capability before it is admitted.
- `Registry.register_executor(name, executor)` — enforces structural conformance
  + capability admission **at registration**, so a weak executor is rejected
  loudly the moment a fork registers it. `daydream ext validate` reports
  registered executor and publisher names.

A missing capability is never papered over: an executor that cannot prove one is
not eligible for merge-authorizing execution.

## 6. State machine

`daydream/service/states.py` is the pure, deterministic controller state machine:

```
queued -> starting -> running -> collecting -> evaluated -> publishing -> passed
   (INFRA / CANCEL are legal from every active state)  \-> failed
   passing through infra_error / cancelled / released
```

Rules:

- Only `ServiceEvent`s move a job. Re-delivering an event whose effect is already
  present is an **idempotent no-op** (controller restarts and duplicate delivery
  are safe). A reordered/stale/superseded event raises `InvalidTransition`.
- `INFRA` and `CANCEL` are legal from every active (non-terminal) state — the
  service fails closed on missing coverage and cannot be wedged open.
- `RELEASE` is the final transition from any terminal state and is itself
  idempotent; once released no event except a duplicate `RELEASE` moves the job.

The durable store shares this vocabulary (`daydream/service/store.py` +
`store_sqlite.py`: `queued ... passed|failed|infra_error|cancelled -> released`).

## 7. Controller behaviour

`daydream/service/controller.py` (`ServiceController`) drives each job through
the state machine and persists every transition through the storage port. Key
behaviours:

- **Opaque refs bound separately from artifacts.** It binds the `ExecutionRef`
  (via `bind_execution`) *separately* from collected artifact hashes
  (`bind_artifacts`), so a late/stale artifact for a superseded or cancelled job
  is rejected without disturbing the live execution reference.
- **Admission before dispatch.** `dispatch` checks `AdmissionController.can_start`
  first; if any bucket is saturated the job stays queued and an `AdmissionBackoff`
  is returned (retry, never a hard failure, never unbounded Pi fan-out).
- **Supersession / invalidation.** A job superseded by a newer candidate head
  (`supersede`, invalidation id bump) is cancelled and any later artifact
  rejected; a fresh candidate is admitted through its own idempotent enqueue.
- **Restart reconciliation.** `reconcile_restart` hands each stored opaque ref
  back to the registered executor's `inspect` (never parses the handle) and
  aligns neutral state. A vanished execution fails closed to infra.
- **Infra retry routing.** Only a *classified infrastructure failure* is retried
  (`retry_infra`), and only within a bounded per-scope retry budget; on exhaustion
  the job is routed to an operator (`ROUTE_TO_OPERATOR`) instead of thrashing the
  same worker.

### Concurrency / fan-out bounds

`daydream/service/admission.py` (`Budgets` / `AdmissionController`) provides
fleet/global, per-service, per-backend, and per-model-provider concurrency caps,
each with a `"*"` wildcard. A single `ReviewExecutor.start` consumes one slot in
every applicable bucket; `can_start` returns a human denial reason when any
bucket is saturated. Infra-failure retries are budgeted per (scope, key), default
fail-closed (0) in the absence of configuration.

## 8. Storage

`daydream/service/store.py` defines the transactional storage port (`ServiceStore`)
and its neutral models (`JobRecord`, `AttemptRecord`, `RecoverableAttempt`). It
deliberately contains **no implementation** and no vendor/executor fields; the
controller binds an opaque `ExecutionRef` to an attempt and the store persists and
returns it verbatim, never parsing the handle.

Two implementations:

- `daydream/service/store_memory.py` (`InMemoryServiceStore`) — the hermetic
  conformance double for tests. **Process-local and ephemeral; never use in
  production.** Correctness invariants (exactly one claimant wins a CAS race, one
  lease owner, append-only attempt history, idempotent no-ops) match SQLite.
- `daydream/service/store_sqlite.py` (`SqliteServiceStore`) — the **production**
  store. Crash-safe, concurrent, persisted to a SQLite file.

SQLite specifics:

- **Stack:** Python stdlib `sqlite3` (bundled; no pip install / no extra dep / no
  background service). `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=10000`.
- **Migration:** a tiny versioned migrator driven by `PRAGMA user_version`
  (`SCHEMA_VERSION`); re-open is a no-op. Never edit applied migrations in place —
  add a new step and bump `SCHEMA_VERSION`.
- **Error mapping:** `sqlite3.IntegrityError` on duplicate `idempotency_key` ->
  `IdempotencyError` (never a silent dup); unknown job -> `JobNotFoundError`;
  owned-transition mismatch -> `StateConflictError`; locked/busy ->
  `StoreError` (transient infra, retriable). Writes serialize through a
  process-local lock plus `BEGIN IMMEDIATE`; readers do not block writers (WAL).
- **Leases/heartbeats/recovery:** the store gives the controller compare-and-set
  claims, leases that expire on deadline, append-only attempt history, idempotent
  event handling, and restart recovery (`recoverable()` returns every non-terminal
  job needing post-restart reconciliation). Do not use `:memory:` — it does not
  survive a restart, defeating the store's purpose.

**Note on two ABIs:** the controller's port `ControllerStorage` (leaf-B) and the
durable `ServiceStore` (leaf-C) are two storage ABIs wired today only through the
controller's own in-memory fake. A future production adapter may bridge
`ServiceStore` onto `ControllerStorage` directly without contract changes; until
then, production persistence is `SqliteServiceStore`.

## 9. The fail-closed worker and artifacts

`daydream/service/worker.py` (`run_service_review`) runs one immutable
`ReviewJobV1` against a detached checkout through a strictly **read-only** backend
turn and returns a strictly-passive `WorkerArtifactV1`. Every invariant is a hard
gate, never a soft warning:

1. **Pre-flight:** the checkout is detached at the exact candidate SHA, carries
   the exact candidate tree digest, is pristine (no staged/unstaged/untracked
   drift), and its full `base_sha..HEAD` diff matches `full_diff_digest`. Any
   mismatch is `git_preflight_failed`.
2. **Lens inventory:** every required lens must be present before dispatch; a gap
   is `lens_unavailable` and the backend never runs.
3. **Read-only run:** `phase_service_review` runs `run_agent(read_only=True)`.
   Budget/supervisor aborts are `infra_error` (`budget_exhausted`/`tool_vetoed`),
   never `clean`.
4. **Every lens must complete**; missing after dispatch is `incomplete_lenses`
   (infra) unless the turn was cancelled.
5. **Mutation check:** the git head/tree/index/tracked/untracked state must be
   byte-identical before and after the turn (`GitSnapshot`, five digests); any
   change is `mutation_detected` — the worker-side proof the read-only run stayed
   read-only.
6. **Findings:** blocking findings (high/medium) keep `terminal="findings"` even
   when the process exited 0; a missing-free run with no blocking findings is
   `terminal="clean"`.
7. **Process/UI/parse loss** is `infra_error`, never `clean`.

`WorkerArtifactV1` (`daydream/service/artifact.py`) is frozen, validated, and
STRICTLY passive: it carries no Sprite/Coder/pod/VM/lease, no executor kind, no
opaque handle, and no attempt binding (those live in the controller's separate
`ExecutionRef`). Findings are bounded (`MAX_FINDINGS = 200`), strict, and
homogeneous; hashes are digests and never infrastructure pointers (URLs are
rejected).

Exit codes (`terminal_exit_code`): `clean`/`findings` = 0, `infra_error` = 1,
`cancelled` = 2 (so a controller can tell "cancelled" from "broken").

The immutable job/target models (`ReviewJobV1` / `ReviewTargetV1`) and the policy
models attribute this page's contracts. `from_dict` enforces
`additionalProperties=False` semantics at every nesting level.

## 10. Least privilege

The trust boundary is designed so no single component holds enough authority to
fabricate success:

- **No ambient credentials.** The worker environment carries no ambient forge,
  secret-manager, or executor-control credentials (`no_ambient_credentials` is a
  required capability). Only the `Publisher` holds external write authority.
- **Publisher-only write path.** `GitHubChecksPublisher` is **the ONLY holder of
  Checks-write authority** in a service run. Workers never receive publisher or
  repository-write credentials. It publishes only after the `PolicyEvaluator`
  returns `SUCCESS`, and before any `success` it **revalidates live candidate
  identity** (current PR head for `pr_head`; a caller-supplied resolver for
  `merge_group`) — a changed head/replaced candidate means the reviewed candidate
  is stale and the publisher refuses (release fails closed, nothing is written).
  `external_id` is always bound to the immutable job id, so a check can never be
  favourably reused across candidates. A PR approval is not an authorization
  primitive.
- **No worker-asserted infrastructure identity** in common models. A conformance
  gate (`test_common_contract_has_no_sprite_names`) greps the common models for
  sprite/coder/kubernetes/hostname/token/secret and rejects any occurrence.
- **Source read-only + mutation proof** (see §9) — an executor cannot mutate the
  reviewed source and still claim `clean`.
- **Fail-closed evaluation.** `PolicyEvaluator` returns `SUCCESS` only for the
  complete, configured set of independent full rounds, each bound to this exact
  candidate with full lens coverage, complete artifacts, `CLEAN` outcome, zero
  findings, and distinct attempt ids. Findings are findings even when the process
  exited zero; a single clean round is never sufficient when the policy requires
  more.

## 11. Adapter and publisher authoring

The full how-to lives in [`docs/executors/executor-registration.md`](executors/executor-registration.md);
key contract points:

- Implement `ReviewExecutor` (the 5 methods + `kind`/`adapter_version`/`capabilities`),
  declaring **all** required capabilities. `require_capabilities` enforces the
  subset check; a partial adapter is rejected at registration.
- Keep every vendor name / SDK type / handle / credential **inside the adapter**
  module. Clean up ambiguity by *quarantine*: if you cannot tell whether an
  execution was cleaned up, do not `release` it — surface `INFRA_ERROR` instead.
  Use one exclusive clean execution per attempt; export artifacts before any
  reset.
- Register through the `daydream_ext` `register(registry)` entrypoint:
  `r.register_executor("name", MyExecutor())` and optionally
  `r.register_publisher("name", MyPublisher())`. The seam is additive and does not
  change the meaning of `Backend`; registered names resolve via
  `executor(name)` / `publisher(name)`.
- Pass the **same** common conformance suite
  (`tests/test_executor_contract.py`, parametrized over `LocalExecutor` and
  `ScriptedExecutor`): capability admission, opaque handles, full
  start/inspect/cancel/collect/release lifecycle, idempotency, restart
  reconciliation, deterministic cleanup, vendor-error mapping, and
  vendor-neutrality. Add your adapter as another parametrized case; it must
  **not** weaken required capabilities or require a common schema change.
- Qualify live via a separately-credentialed staging environment **never in the
  hermetic gate**. Document adapter specifics under `docs/executors/`.

Built-in adapters:

- `LocalExecutor` (`daydream/executors/local.py`) — real filesystem workspace +
  time-based async lifecycle; persists state to disk so a fresh instance can
  reconcile a prior ref. **Development/test infrastructure only** — it does not
  sandbox ambient credentials, source writes, or egress, and must not be used to
  merge-authorize untrusted code.
- `ScriptedExecutor` (`daydream/executors/scripted.py`) — in-memory, step-based
  conformance adapter.
- `SpritesExecutor` (`daydream/executors/sprites.py`) — optional hosted adapter,
  a **stub** today: live lifecycle methods raise `ExecutorError` until real
  staging is wired (`DAYDREAM_SPRITES_STAGING=1` + an explicit Sprite connection).
  Hermetic tests never invoke it. Behaviour requirements (quarantine, exec/session
  kill, tasks, checkpoint/export-before-reset, disk bounds, connectors,
  one-exclusive-clean-execution-per-attempt) are in
  [`docs/executors/sprites.md`](executors/sprites.md).

## 12. Rollback and recovery

The service is designed to make "un-publishing" unnecessary and recovery safe:

- **Detached exact-SHA checkouts.** Every execution runs on a detached checkout at
  the exact candidate SHA; the worker re-verifies SHA, tree, diff digest, and
  pristine-ness before the agent runs and again after.
- **Durable monotonic store.** Transitions persist through the transactional
  store; a restart reconciles non-terminal jobs against live executions via
  `inspect`, never by trusting in-memory state. Idempotent event handling makes
  duplicate delivery a no-op.
- **Supersession over rollback.** A force-push or replaced merge-group candidate
  bumps the invalidation id, cancels the superseded job, and rejects its late
  artifacts — older rounds can never authorize the new head. `bind_artifacts`
  records `blocked` and the completed-lens set so evaluation can fail closed on a
  blocking finding or incomplete coverage regardless of process exit code.
- **Rollback of a bad publish is not needed by design**: a stale target fails at
  the publisher's live-identity revalidation *before* any success is written, and
  release is deterministic per adapter (with quarantine on ambiguity).

## 13. Credential rotation

Credentials live only where the trust boundary requires them:

- **GitHub App** (`daydream/github_app.py`): provided by the environment vars
  `DAYDREAM_APP_ID` (int) and `DAYDREAM_APP_PRIVATE_KEY` (PEM **content**, not a
  path). Both are required together; a partially-set pair is a hard
  misconfiguration error. The module mints a short-lived RS256 JWT and exchanges
  it for a scoped installation access token; the installation token is short-lived
  and GitHub-expiry-bounded. **Rotation** = replace the env vars and restart the
  service process (the JWT/token are never read as credentials onto argv; the
  token is injected into the child's environment at subprocess time via the
  `gh` shim). Without configured App credentials, ambient `gh` identity is used
  only for read-only posting paths.
- **Executors / publishers** carry **no** shared secret in common models. Live
  hosted adapters are separately credentialed (e.g. `DAYDREAM_SPRITES_STAGING=1` +
  an explicit Sprite connection for Sprites) and must be rotated at their own
  secret manager; the service never forwards an executor credential into a worker.

There is no bundled secrets vault in the service today; the boundary holds because
only the publisher holds write authority and only the environment-credentialed
GitHub App path can mint publication tokens.

## 14. Observability

Current surface (honest about what exists today):

- **Structured worker artifacts.** `WorkerArtifactV1` carries the terminal,
  process outcome, completed/missing lenses, bounded findings, and ISO-8601 UTC
  timestamps — the primary audit record of each turn.
- **Python `logging`.** Service modules (`daydream/service/worker.py`, store, ...)
  log warnings on every fail-closed event (`git_preflight_failed`,
  `mutation_detected`, `incomplete_lenses`, `state_capture_failed`, CAS conflicts,
  retry routing). `DAYDREAM_GH_TIMEOUT_SECONDS` / `DAYDREAM_GH_RETRIES` govern the
  `gh` git-op subprocesses.
- **CLI/runner rendering.** The service-mode hook (`runner.run_service`) renders
  the terminal outcome via the Rich console (`_print_service_outcome`) and returns
  the terminal exit code (0/1/2) for the caller.
- **Diagnostics snapshot.** `AdmissionController.in_flight()` returns a
  `InFlightSnapshot` of concurrent execution counts per bucket for debugging the
  fan-out. `Registry` exposes `executor_names()` / `publisher_names()` and
  `daydream ext validate` reports registered executors/publishers.

There is no Prometheus/metrics endpoint and no hosted trace collector wired into
the service today; the controller record's `trigger_ref` and `store_fields` are the
durable causal/debug markers the store leaf retains.

## 15. Planned / deferred (as of this candidate)

- **Sprites live path** — the adapter is a stub; live Sprite lifecycle is deferred
  to a separate qualification card and must never run in the hermetic gate
  ([`docs/executors/sprites.md`](executors/sprites.md)).
- **`ControllerStorage` ↔ `ServiceStore` bridge** — today two ABIs wired through
  the controller's in-memory fake; a production adapter may bridge them directly.
- **`ExecutionBridge.collect` polls to terminal** (bounded) — a reasonable adapter
  default; operators may want to tune polling bounds for long-lived hosted
  executors.

---

**Related docs:** [`docs/executors/executor-registration.md`](executors/executor-registration.md) ·
[`docs/executors/sprites.md`](executors/sprites.md) · `docs/extensions.md`
(DAYDREAM_SERVICE_V1 seam).