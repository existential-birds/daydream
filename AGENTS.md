# Repository guidance

Daydream reviews code changes, applies fixes, validates them, and records agent
interactions as ATIF trajectories. Its corpus pipeline turns those records into
training datasets.

## Validation and Git

- Use `make check` for the full root quality gate. The pre-push hook verifies
  commit signatures and runs it. Run `make rl-check` as well when changing
  `rl/daydream_review/`; that standalone project's full gate is separate.
- Keep commits signed, use Conventional Commits, and stage explicit paths so
  unrelated working-tree changes stay out of the commit.
- **Never bypass Git hooks.** Never use `--no-verify`, hook-disabling environment
  variables, or any other mechanism that skips commit, push, or repository
  verification hooks, even for unrelated or pre-existing failures. Fix the
  failure on the current branch and retry with hooks enabled, or stop and report
  the blocker. A commit or push is complete only after the ordinary hook-enabled
  command succeeds.
- When changing dependencies, update the affected `uv.lock` alongside
  `pyproject.toml`. Check the Makefile and project configuration for current
  commands, tool settings, and exemptions.

## Testing standard

Every changed user-visible behavior needs a real-path test entering through
`runner.run` or the CLI with a real temporary Git worktree, filesystem, and event
loop. Mock external network/API boundaries through `Backend` / `create_backend`;
assert observable outcomes such as exit status, written files, retained fixes,
and trajectory state. Unit tests supplement this coverage.

Use `tests/deep_orchestrator/test_fix_gate_cleanup_and_precision.py` as an
exemplar. For terminal findings, preserve typed coverage, snapshot binding, and
atomic publication as defined in
[README.md](README.md#terminal-review-findings-contract); inject filesystem
faults at the actual read/write/install operations.

## Architecture

- `cli.py` / `commands/` own process lifecycle and command dispatch;
  `runner.py` owns workspace lifecycle, the per-run extension registry, and flow
  dispatch. `flows/engine.py` executes registered steps; `deep/` and `improve/`
  assemble their respective flows. Paths here are relative to `daydream/`.
- Route agent calls through `agent.run_agent()`, including fan-out under the
  effective backend capacity limiter. Phases use the shared `Backend` /
  `AgentEvent` contracts rather than calling SDKs directly. Fork the
  `TrajectoryRecorder` for sibling agent trajectories.
- Keep Git/GitHub subprocess operations in `git_ops/` and ATIF construction in
  `trajectory/`. `daydream/atif/` is vendored Harbor code: update it by wholesale
  re-vendoring, preserving licensing; avoid local patches and a Harbor runtime
  dependency for ATIF models.
- Preserve cancellation-safe backend teardown when changing SDK dependencies.
  Truncated, failed, or malformed agent results must retain admitted evidence
  and mark incomplete coverage rather than become a clean review. Do not add
  `max_turns` to fix/verify calls: exhaustion can discard a valid fix.
- Finding identity is host-assigned: record `uid`, merged finding `item_uid`, and
  derivation `source_uids` answer different questions. Human-facing integer `id`
  is renumbered for display; content fingerprints identify similar defects.
  Neither is a durable record handle. Use the accessors in `deep/records.py`.
- Resolve the private artifact owner once and pass the owner and active session
  through workspace, recorder, and flow composition. Route generated files
  through that session and join all writers before freezing evidence for
  archive, evaluation, or publication. Preserve confined backend read roots.
- Diagrams start as evidence-bearing JSON specs; `deep/diagram_grounding/`
  validates what may be drawn before deterministic Mermaid rendering.

## Task-specific references

Read the relevant contract before changing its subsystem, and update it when
the behavior changes:

- CLI/configuration and review outputs: [README.md](README.md).
- Development setup, signing, and gate commands:
  [CONTRIBUTING.md](CONTRIBUTING.md).
- Extension phases, flows, prompts, and exporters:
  [docs/extensions.md](docs/extensions.md). Fork customizations belong in
  `daydream_ext`; take supported API versions from `extensions/api.py`.
- Tracing, billing ownership, and export privacy:
  [docs/observability.md](docs/observability.md). Runs own their OTel providers;
  operator settings select destinations and credentials, not target file config.
- Benchmark isolation and credentials: [docs/benchmark.md](docs/benchmark.md).
  `daydream/benchmark/harbor/env_policy.py` is the single declaration of allowed
  and scrubbed environment names.
- Corpus/training: [docs/training-launch.md](docs/training-launch.md) and
  [docs/calibration.md](docs/calibration.md). Harvest/projection consume frozen
  records and snapshots; preserve source-bound reply captures separately from
  semantic evidence and retain complete-record secret scanning on publication.
- Coverage changes: [docs/coverage.md](docs/coverage.md); follow the ratchet
  procedure rather than lowering the floor.
- RL package: [rl/daydream_review/README.md](rl/daydream_review/README.md).
