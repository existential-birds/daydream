# Agent guidance

AGENTS.md owns repository guidance for coding agents. Keep cross-cutting rules
here and subsystem detail in the linked documents; update the relevant owner when
its contract changes. Prefer links to executable configuration over copied inventories.

Before contribution work, read [CONTRIBUTING.md](CONTRIBUTING.md), which owns the
shared `prepare-implement` → goal prompt → execution workflow, validation,
SemVer/changelog policy, and PR template/checklist requirements. Read
[Makefile](Makefile) and [pyproject.toml](pyproject.toml) for current commands,
settings, and exemptions; use [README.md](README.md) for the product surface.

## Never bypass Git hooks

Run every commit and push with hooks enabled. Never use `--no-verify`,
hook-disabling environment variables, configuration, or another mechanism that
skips verification, even for pre-existing failures. Fix the failure on the
current branch and retry normally, or report the blocker. Sign commits and stage
only related paths explicitly. A commit or push is complete only after the
normal hook-enabled command succeeds.

## Validation

```bash
make check      # lockcheck + install + lint + deadcode + typecheck + test + actionlint + coverage-report + check-naming (the gate)
```

The pre-push hook checks signatures and runs this gate. Follow
[CONTRIBUTING.md's setup and focused checks](CONTRIBUTING.md#everyday-commands-and-the-required-gate).
Run `make rl-check` separately when changing `rl/daydream_review`; it is outside
`make check`. Follow [the coverage ratchet](docs/coverage.md) without lowering it.

## Testing standard (mandatory)

Choose the test boundary that owns the observable behavior or independent contract.
Each new or changed test must identify a credible regression and explain why existing
coverage misses it. Prefer extending existing tests and fixtures; preserve real
filesystem, Git, and event-loop behavior when those are part of the contract.
Unit, integration, and end-to-end tests are valid choices according to the behavior
and risk. Use the [test-audit skill](.agents/skills/test-audit/SKILL.md) for test
changes and address its findings. This section is authoritative when a skill
repeats conflicting or older testing requirements.
For fix-loop changes, assert retained fixes and trajectory state.
Inject terminal-finding faults at actual filesystem operations; preserve typed
coverage, snapshot binding, and atomic publication under the
[terminal findings contract](docs/runtime.md#terminal-review-findings-contract).

## Architecture

Daydream reviews diffs, applies authorized fixes, validates them, and records agent
interactions as ATIF v1.7 trajectories. Captured records feed the bitemporal corpus
pipeline. Deep multi-stack review is the default; shallow is single-stack, and
review/comment modes stop before fixes. Improve currently requires Claude for
every model phase because it needs the snapshot-root tool boundary.

```text
cli.py / commands/ → runner.py → deep/orchestrator.py → flows/engine.py
                              → improve/orchestrator.py → flows/engine.py
                              → custom extension flows → flows/engine.py
flow steps → phases/ → agent.run_agent() → Backend.execute()
```

These boundaries matter across subsystems; module docstrings carry finer detail:

| Owner | Responsibility |
|-------|----------------|
| `runner.py`, `run_config.py`, `run_artifacts.py`, `run_snapshot.py` | Workspace/registry dispatch, resolved configuration, recorder lifecycle, frozen run identity |
| `flows/`, `extensions/` | Ordered/gated flow execution and the per-run extension registry |
| `deep/`, `improve/` | Flow assembly and domain steps; `deep/state.py` validates the shared `FlowContext.data` mapping |
| `agent.py`, `agent_output.py`, `agent_retry.py` | Backend lifecycle/budgets, host result selection/validation, retry policy |
| `committed_review*` | Explicit committed-review request/result, semantic claim revisions, mandatory fresh challenge and sealed output |
| `backends/` | Native transport and the common `Backend` / `AgentEvent` contract |
| `trajectory/`, `observability/` | Recorded evidence, generation billing ownership, and per-run tracing |
| `artifact_visibility.py`, `artifacts/`, `workspace.py` | Private artifact ownership, finalization/recovery, and operational worktrees |
| `git_ops/` | Single Git/GitHub subprocess boundary |
| `reviews/`, `pr_review.py`, `pr_comment_renderer.py` | Finding placement/publication and pure comment rendering |
| `dataset*.py`, `training/`, `archive/`, `eval/` | Frozen records/Hub exchange, corpus projection, diagnostic archives, deterministic analysis |

### Runtime invariants

- Review/exploration search guidance is shared in `prompts/grounding.py` and retained
  through custom prompt composition. Use relevant roots and tracked plus nonignored
  untracked file sets; named ignored-file expansion requires a concrete reason.
  `daydream search` owns the bounded no-rg fallback in `daydream/commands/search.py`.

- `git_ops.repository` resolves embedded provenance and App bootstrap locally (explicit `pr_repo`,
  captured `GH_REPO`, then target `origin`), admitting only unambiguous github.com identities.
  The App session binds token minting, refresh, and GitHub routing to that identity.
  CLI provenance uses `gh repo view` after session resolution, preserving gh's selected
  upstream for forks, supported hosts, and SSH aliases with that session's authentication.
  Explicit `pr_repo` wins; local provenance is a fallback when gh discovery is unavailable.
  Embedded runs resolve provenance locally without implicit remote repository discovery.
  CLI-requested branch PR discovery uses session auth on the original checkout before detached workspace creation.
  Direct runner callers retain flow-owned PR discovery.
- Call agents through `agent.run_agent()`, never directly through a backend/SDK
  from phases. Fan-out uses parallel calls under
  `anyio.CapacityLimiter(effective_fanout_concurrency(ceiling, backend))` and
  `TrajectoryRecorder.fork()` for sibling trajectories, not SDK `agents=`.
- The committed-review callable owns assignments, admitted claim revisions,
  challenge obligations and derived coverage. Every proposed publication claim
  needs fresh counterevidence-seeking challenge against captured source in a
  distinct native context. Semantic revisions renew that obligation; absent or
  failed verdicts retain unresolved concerns. CLI route/default cutover and
  remaining flow retirement have separate owners; read
  [its runtime contract](docs/runtime.md#committed-review-requestresult).
- Keep changing budgets, pacing, and progress state authoritative on the host.
  Do not send orchestration nudges as user messages, including SDK custom messages
  that serialize as user turns. Put stable policy in system/developer instructions;
  when a model decision needs current state, provide an existing bounded read-only
  lookup and explain when to use it. Host enforcement must not depend on polling.
  Preserve genuine user turns and the native assistant/tool conversation history.
- Keep ATIF construction in `trajectory/`. Use the vendored models without adding
  Harbor to core runtime dependencies; the benchmark extra is separate. Follow
  [the vendoring policy](daydream/atif/NOTICE): re-vendor wholesale, no local patches.
- Resolve the private artifact owner once and pass the same owner/session through
  workspace, recorder, and flows. Route generated files through that session;
  join writers before freezing evidence, then archive/evaluate/publish. Preserve
  competing bytes on conflicts. Use sanctioned exact inputs or bounded inline
  contents without broadening backend read roots or reconstructing public paths.
- Host-assigned `uid` identifies a pre-merge record, `item_uid` a merged item,
  `source_uids` provenance, and integer `id` a renumberable display position.
  Preserve identities through rewriting; content fingerprints answer similarity,
  not identity. Load stack records in stack-name order for reproducibility.
- Compute latency risk, sharding inputs, file/stack counts, diff keys, and coverage
  from the full captured patch before bounding `DeepState.diff`. Routing evidence
  is diagnostic; missing/corrupt routing records must not block execution/resume.
  `forensic` is a latency profile with an unsharded arbiter; per-stack review
  sharding is independent.
- Model-owned merge receives all retained language and structural records; its
  validated output is authoritative without structural reappend. Whole-file
  structural privileges require retained structural source UIDs. Host bypass and
  failure salvage retain their separate folding policy.
- Preserve admitted findings and truthful incomplete coverage on review failure.
  Validate submitted source references atomically against frozen side/range/fragment identities and retain per-target
  attribution through file folding, cache and resume. Correspondence is not proof of understanding, reads or checks;
  optional supporting-file relevance metadata cannot veto valid primary decisions.
  Canonical test readiness gates accepted fixes before verify/FIX work; model
  prose or unresolved command candidates cannot authorize tests or publication.
- Archive indexes must be empty or use the current complete schema. Preserve and
  reject unsupported indexes; use a fresh archive directory instead of migrating.
  Annotation history belongs in LocalRecordStore observations.

### Backend changes

Read the protocols in [backends/__init__.py](daydream/backends/__init__.py) and the
adapter being changed, plus [lifecycle evidence](docs/observability.md#codex-lifecycle-evidence)
and the relevant contracts below.

- Attempt cancellation closes only that invocation's stream; backend-wide
  cancellation belongs to run shutdown. Preserve terminal metrics on errors.
- Codex success requires the latest public turn to complete, stdout EOF, and an
  independently observed zero exit. Preserve bounded terminal drain, process reap,
  and cleanup; repeated terminal events cannot extend the tail allowance or charge
  usage twice. Postterminal expiry prohibits replay of a possibly completed mutation.
- Preserve Claude's dangerous-command/background-Bash guards for ordinary turns
  and its deny-by-default snapshot-root guard for Improve, which permits no Bash.
  Background Bash cannot report after the host stops consuming the completed turn.
  Keep the SDK pin at least `0.2.111` for cancellation-safe subprocess teardown.
- Do not add `max_turns` to fix or verify: exhaustion raises `MaxTurnsError` and
  causes the fix group to be reverted. Use enforced wall/tool budgets; exploration
  is the bounded-turn caller. Read defaults from `config.py` and `agent_retry.py`.
- Adapters emit native events; `agent_output.py` selects final-turn output and
  `ReviewInvestigation` admits staged domain results before presentation. Retain
  raw events as evidence. Report transport gaps only from public evidence; never
  invent missing tool calls or invocation-wide coverage warnings.

## Read before changing a subsystem

Read the relevant row before editing its implementation or contracts. Keep detailed
rules in these owners rather than expanding this file with defaults and inventories.

| Change | Required reference |
|--------|--------------------|
| Flow order, extension phases/prompts, configuration | [Extension contract](docs/extensions.md), [configuration](daydream/config.py); defaults in `config.py`, resolution in `run_config.py` / `runner._resolve_backend()` |
| Staged review, Pi output, budgets/retries, sharding | [Stage-aware builders](docs/extensions.md#stage-aware-review-builders-api-8), [review assignments](daydream/review_stage_inputs.py), [retry recovery](daydream/agent_retry.py) |
| Latency routing or diff bounding | [Latency routing](daydream/deep/latency.py); `deep/latency.py` and `deep/routing_record.py` own selection and evidence |
| Record identity or merge/adjudication | [Finding identity](docs/extensions.md#finding-identity), [built-in flow steps](docs/extensions.md#built-in-flows-and-steps); `deep/records.py` owns identities |
| Fix, verify, test, commit, remote CI | [Flow gates](docs/extensions.md#built-in-flows-and-steps), [validation contracts](docs/runtime.md#validation-and-repair), [repair jobs](docs/extensions.md#repair-jobs-test-step) |
| Artifacts, sanctioned inputs, recovery | [Working artifact paths](docs/extensions.md#working-artifact-paths), [artifact contracts](docs/runtime.md#artifacts-and-recovery), [terminal findings](docs/runtime.md#terminal-review-findings-contract) |
| Diagrams | [Diagram contract](docs/runtime.md#diagrams), [diagram-only flow](docs/extensions.md#diagram); `deep/diagram_grounding/` is the drawing authority, models supply JSON specs and the host renders Mermaid |
| Tracing, timing, billing | [Observability](docs/observability.md); tracing is operator-controlled and owns a per-run provider, never a global provider or `Traceloop.init()` |
| Improve audit/plans | [Improve implementation](daydream/improve/), [Improve flow](docs/extensions.md#improve); `services.py` owns shared service discovery |
| Capture, harvest, adjudication, dataset publication, corpus projection | [Research](README.md#research), [record dataset workflow](docs/runbooks/record-dataset.md), [calibration](docs/calibration.md); preserve immutable evidence, repository provenance, and C5 benchmark isolation |
| Training launches or RL | [Training launch](docs/training-launch.md), [RL package](rl/daydream_review/README.md) |
| Benchmarks | [Benchmark runbook](docs/benchmark.md); `daydream/benchmark/harbor/env_policy.py` owns environment allow/scrub policy |
| Bot/workflow publication or CI policy | [Packaged workflows](daydream/templates/workflows/README.md), [self-hosted bot](docs/self-hosted-bot-setup.md), [CI spend policy](.github/workflows/README.md#ci-spend-policy) |
