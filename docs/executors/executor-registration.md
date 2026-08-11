# Registering an executor with Daydream (DAYDREAM_SERVICE_V1)

This document is the *generic* guide for any public external package that wants
to add a compute/workspace executor — Coder, Kubernetes, local-production, or a
future adapter — to the Daydream review service, and run the same conformance
suite the built-in adapters do. It is intentionally vendor-neutral; Sprite
specifics live in [`sprites.md`](sprites.md).

## The seam

Executors plug into the versioned `DAYDREAM_SERVICE_V1` seam through the
existing `daydream_ext` extension registry — the same `register(registry)`
entrypoint a fork already uses for flows, skills, and prompts. There is *no
separate plugin mechanism* and the seam does **not** change the meaning of
`Backend` (the model-agent driver). `executor` means the compute/workspace
adapter (Sprites, Coder, Kubernetes, local, ...); `backend` stays the Daydream
agent driver; `provider` stays the Pi/model endpoint provider.

## The port you implement

```python
class ReviewExecutor:
    kind: str
    adapter_version: int
    capabilities: frozenset[ExecutorCapability]

    async def start(self, job: ExecutorJob) -> ExecutionRef: ...
    async def inspect(self, ref: ExecutionRef) -> ExecutionSnapshot: ...
    async def cancel(self, ref: ExecutionRef) -> None: ...
    async def collect(self, ref: ExecutionRef) -> ArtifactEnvelope: ...
    async def release(self, ref: ExecutionRef, disposition: str) -> None: ...
```

`ExecutionRef` is `executor_kind + adapter_version + opaque_handle +
attempt_id`. The `opaque_handle` is yours to define and is never parsed by the
controller — the controller stores it and feeds it back to *your* executor.
`ExecutionSnapshot` and `ArtifactEnvelope` are neutral models: they must not
carry your pod/VM/workspace/lease/hostname identity, your SDK objects, or your
capability fields. Vendor types and handles stay inside your adapter.

### Required capabilities

Missing a required capability fails capability admission (a contract STOP
condition). Every merge-authorizing executor must declare **all** of:

`exclusive_workspace`, `no_ambient_credentials`, `source_read_only`,
`bounded_egress`, `durable_execution_identity`, `strong_cancel`,
`deterministic_release`, `restart_reconciliation`.

Declare them as the adapter's `capabilities` frozenset. `require_capabilities`
(exported from `daydream.executors`) enforces the admission subset check.

## Registering

```python
from daydream.extensions import ReviewExecutor

def register(r):
    r.register_executor("coder", MyCoderExecutor())
    r.register_publisher("github-checks", MyPublisher())  # publisher seam, optional
```

`register_executor` enforces the structural conformance check and capability
admission *at registration* — a weak executor is rejected loudly. `daydream ext
validate` reports registered executor/publisher names.

## Running the conformance suite

The common conformance suite lives in `tests/test_executor_contract.py` and is
already parametrized over the two built-in hermetic adapters (`LocalExecutor`,
`ScriptedExecutor`). To qualify a new adapter, add it as another
parametrized case in that suite so it runs the *same* contract assertions. The
suite drives every job through its `_job()` helper — identity via
`attempt`/`key`, plus a `**payload` dict (e.g. `lenses`, `outcome`). Your
adapter must honour that payload: it is not free to ignore it, because the
suite asserts on the jobs it drives through it.

1. capability admission (declares all required capabilities; a partial adapter
   is rejected);
2. opaque handles (`start` returns a typed `ExecutionRef`; never parsed);
3. `start` / `inspect` / `cancel` / `collect` / `release` lifecycle;
4. idempotency (repeat `start` with the same identity binds the same execution);
5. restart reconciliation (a fresh instance over the same durable backing sees
   a prior ref);
6. cleanup ordering / deterministic release;
7. vendor-error mapping (a simulated SDK exception must surface as the
   neutral `INFRA_ERROR` status/outcome, never leak as a raw vendor type);
8. vendor-neutrality (no non-neutral field may appear in a common model).

You may run the suite yourself with:

```bash
uv run pytest -q tests/test_executor_contract.py
```

A new adapter must **not** weaken required capabilities or require a common
schema change. If your adapter cannot prove a required capability, it is not
eligible for merge-authorizing execution — that is a STOP condition, not
something to paper over.

## Behavioural notes every adapter must honour

- **Exclusive workspace**: each execution runs in its own isolated workspace;
  no two executions share writable state.
- **No ambient credentials**: the worker environment carries no ambient forge,
  secret-manager, or executor-control credentials.
- **Source read-only**: the reviewed source is staged read-only and proven
  unchanged (the service compares initial/final Git tree).
- **Strong cancel**: `cancel` interrupts the execution promptly; `inspect`
  thereafter reports `cancelled`.
- **Deterministic release**: `release(ref, disposition)` always removes the
  execution's resources in a fixed order; after release the ref is gone.
- **Restart reconciliation**: on controller restart, the adapter can
  `inspect` a stored opaque reference it previously created (its durable
  backing survives).
- **Quarantine on ambiguous cleanup**: if you cannot tell whether an execution
  was cleaned up, do NOT release it — surface `INFRA_ERROR` so the controller
  never mistakes an ambiguous resource for a released one.

## Authoring checklist

1. Implement `ReviewExecutor` with all required capabilities.
2. Keep every vendor name/SDK type/handle inside the adapter module.
3. Register through `daydream_ext`'s `register()`.
4. Add your adapter as a parametrized case in `tests/test_executor_contract.py`
   (or a sibling that reuses the same assertions) and make it hermetic.
5. Add adapter-scoped docs under `docs/executors/`.
6. Add an SDK pin + install command + error/idempotency mapping in that doc.
7. Qualify via live staging separately (never in the hermetic gate).