"""Durable, neutral review-service core (Plan 008, consolidated).

This single package is the integrated home of the exact-candidate review service
that Plan 008 split across leaves. It is deliberately free of any execution
adapter, provider, or worker-asserted infrastructure identity: no Sprites, Coder,
pod/VM/lease, or backend credential ever appears in a common model.

Submodules (contracts, all neutral):

- ``models`` — the frozen immutable data: exact ``ReviewTargetV1`` /
  ``ReviewJobV1`` (worker/job side), ``ReviewTarget`` / ``ReviewPolicy`` /
  ``RoundRecord`` / ``LensInventory`` / ``SourceOfTruth`` (policy side), and the
  controller's ``JobSpec`` / ``ControllerRecord``. The service defines its own
  distinct execution types (opaque ``ExecutionRef``, ``ArtifactEnvelope``,
  ``ExecutionSnapshot``) in ``models``; the canonical executor contract types of
  the same names live in :mod:`daydream.executors.contract` and are not
  re-exported here.
- ``states`` — the deterministic neutral state machine (queued -> starting ->
  running -> collecting -> evaluated -> publishing -> passed -> released;
  ``failed`` exits ``evaluated``, and ``infra_error`` / ``cancelled`` are legal
  side exits from every active state).
- ``ports`` — the ``ControllerStorage`` + ``ReviewExecutor`` Protocols the
  controller programs against (implementations live in consuming modules).
- ``controller`` — the durable ``ServiceController`` driving jobs through the
  state machine, binding the opaque execution reference and rejecting late /
  stale / superseded artifacts.
- ``admission`` — fleet/global + per-service/backend/provider concurrency caps
  and bounded infra-only retry budgets.
- ``store`` (+ ``store_memory``/``store_sqlite``) — the transactional store port
  and its in-memory conformance double + production SQLite implementation.
- ``worker`` / ``artifact`` — the fail-closed service-mode review worker and its
  strictly-passive ``WorkerArtifactV1`` envelope.
- ``policy`` / ``config_digest`` — the fail-closed ``PolicyEvaluator`` and the
  protected-source rule + canonical effective-config digest.
- ``publisher`` — the public publisher port (the trusted GitHub Checks writer is
  in :mod:`daydream.github_app`).
- ``executor_bridge`` — the single seam normalizing a canonical
  ``ReviewExecutor`` adapter onto the controller-facing surface.
- ``runner`` — the composition entry point: ``ReviewRunner`` drives one job per
  execution through controller, policy, and publisher.

The meaning of ``Backend`` (the Daydream model-agent driver) is unchanged; the
review-service executor seam is a separate concern. Importing this package never
pulls the phase/agent stack in.
"""
