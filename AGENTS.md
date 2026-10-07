# Repository guidance

## Validation and Git

- Run `make check`; also run `make rl-check` for changes in `rl/daydream_review/`.
- Use signed Conventional Commits and stage explicit paths, excluding unrelated changes.
- **Never bypass Git hooks**, including `--no-verify` or hook-disabling environment variables,
  even for pre-existing failures. Fix the failure on this branch and retry normally, or report
  the blocker. A commit or push is complete only when the hook-enabled command succeeds.
- Dependency changes include both `pyproject.toml` and its affected `uv.lock`.
  Read the Makefile and configuration for current commands, settings, and exemptions.

## Testing

Every changed user-visible behavior needs a test through `runner.run` or the CLI using real
Git worktrees, filesystem, and event loop. Mock only external network/API boundaries via
`Backend` / `create_backend`; assert exit status, written files, retained fixes, and trajectory state.
Unit tests supplement this proof. See `tests/deep_orchestrator/test_fix_gate_cleanup_and_precision.py`.
Terminal findings retain typed coverage, snapshot binding, and atomic publication per the
[README contract](README.md#terminal-review-findings-contract); inject faults at actual filesystem operations.

## Architecture

Paths below are relative to `daydream/`.

- `cli.py` / `commands/` own process lifecycle and dispatch; `runner.py` owns workspace lifecycle,
  the per-run extension registry, and flow dispatch. `flows/engine.py` executes steps;
  `deep/` and `improve/` assemble flows.
- Route agent calls, including capacity-limited fan-out, through `agent.run_agent()` and shared
  `Backend` / `AgentEvent` contracts, not SDKs. Fork `TrajectoryRecorder` for sibling trajectories.
- Git/GitHub subprocesses belong in `git_ops/`; ATIF construction belongs in `trajectory/`.
  `atif/` is vendored Harbor code: re-vendor wholesale, retain licensing, avoid local patches
  and a Harbor runtime dependency for its models.
- SDK changes preserve cancellation-safe teardown. Truncated, failed, or malformed results retain
  admitted evidence and incomplete coverage. Fix/verify calls never use `max_turns`:
  exhaustion can discard a valid fix.
- Identity is host-assigned: `uid` identifies records, `item_uid` identifies merged findings,
  and `source_uids` identify derivation. Use `deep/records.py` accessors; display integer IDs
  and content fingerprints are not durable handles.
- Resolve the private artifact owner once; pass it and the active session through workspace,
  recorder, and flow composition. Generate files through that session, join all writers before
  freezing archive/evaluation/publication evidence, and preserve confined backend read roots.
- Diagrams begin as evidence-bearing JSON specs validated by `deep/diagram_grounding/`
  before deterministic Mermaid rendering.

## Subsystem contracts

Read the relevant contract before editing its subsystem; update it when behavior changes:

- CLI/configuration/outputs: [README.md](README.md). Setup/signing/gates: [CONTRIBUTING.md](CONTRIBUTING.md).
- Phases/flows/prompts/exporters: [extensions](docs/extensions.md). Fork changes belong in `daydream_ext`;
  supported API versions come from `extensions/api.py`.
- Tracing/billing/privacy: [observability](docs/observability.md). Runs own OTel providers;
  operator settings choose destinations/credentials, not target file configuration.
- Benchmark isolation/credentials: [benchmark](docs/benchmark.md); `daydream/benchmark/harbor/env_policy.py`
  alone declares allowed and scrubbed environment names.
- Corpus/training: [training launch](docs/training-launch.md) and [calibration](docs/calibration.md).
  Harvest/projection consume frozen records/snapshots; keep source-bound reply captures separate
  from semantic evidence and scan complete records for secrets on publication.
- Coverage: [coverage](docs/coverage.md); follow the ratchet without lowering the floor.
- RL: [RL README](rl/daydream_review/README.md).
