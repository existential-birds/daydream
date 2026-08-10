# Sprites executor adapter (reference hosted integration)

`SpritesExecutor` (`daydream/executors/sprites.py`) is an **optional** reference
hosted-executor integration. It is kept strictly adapter-scoped: every Sprite
name, Sprite SDK/API type, checkpoint/Task/Connector detail, and credential
lives in the adapter module and never appears in the common
`DAYDREAM_SERVICE_V1` models of `daydream.executors.contract` (enforced by
`tests/test_executor_sprites.py`).

## Status and gating

- **Hermetic**: the adapter refuses to run without live staging. In the
  hermetic gate it exposes `kind`, `adapter_version`, and required capabilities,
  and raises `ExecutorError` on any lifecycle call. The common conformance
  suite does NOT run against it (it needs live backing); it runs against the
  two built-in hermetic adapters (`LocalExecutor`, `ScriptedExecutor`).
- **Live**: execution requires explicit opt-in
  (`DAYDREAM_SPRITES_STAGING=1`) **and** an explicit Sprite connection. Live
  tests are skipped unless that gate is set — they never run in CI or the
  hermetic suite.

## Behaviour requirements (binding, once live)

Before hardening the live path, each of these Sprite behaviours must be
verified against first-party documentation on a separately credentialed staging
environment:

- **Lifecycle**: Sprite creation, readiness, and teardown map to the neutral
  lifecycle (`starting -> running -> ... -> evaluated`); no Sprite state type
  leaks into a common `ExecutionSnapshot`.
- **Exec / session kill**: the exec mechanism used by review workers, and the
  session-kill path used for `cancel`, settle promptly and deterministically.
- **Tasks**: if Tasks are used, their lifecycle/outcome maps to neutral
  `evaluated` / `infra_error`; a lost or ambiguous Task is `INFRA_ERROR`, never
  assumed clean.
- **Checkpoint/export**: export the reviewed artifacts *before* any reset; do
  NOT assume cross-Sprite checkpoint cloning.
- **Disk bounds**: enforce disk and retention bounds so a Sprite cannot grow
  unbounded.
- **Connectors**: if a Connector is used, its egress is bounded and its
  credentials are never ambient in the worker.
- **Deletion/quarantine**: deletion is deterministic; on ambiguous cleanup the
  adapter quarantines — it refuses `release` and surfaces `INFRA_ERROR` rather
  than guessing that resources were cleaned.
- **One exclusive clean execution per attempt**: an attempt runs on one clean,
  exclusive Sprite; never two attempts sharing a dirty Sprite.

## Conformance

A Sprites adapter, once live-capable, still implements the `ReviewExecutor`
port and must pass the same common conformance contract
(`tests/test_executor_contract.py`) via its mocked/recorded path before any
merge-authorizing use. See [`executor-registration.md`](executor-registration.md)
for the generic steps.
