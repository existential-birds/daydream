# Product simplification proposals for PR #1444

Status: proposals only. The user chose **keep both Improve and Benchmark; continue
internal redesign**. No capability removal is approved or applied. The original
behavior-preservation criteria govern implementation. An isolated Improve retirement
candidate was prepared under /tmp for feasibility assessment and remains unapplied.

Daydream's README defines its core as automated code review, authorized fixes,
host verification and structured trajectories, then an open-weight review model
trained from those trajectories and evaluated on held-out PRs. A simpler product
can retain that loop instead of combining it with an independent repository audit
and implementation-plan management product. Existing built-in review modes already
share one registry engine; another engine merger would mostly relocate policy.

## Measured options

Counts use the unchanged production inventory and Python3.12/scb-check0.2.0 method.
These are complete exclusive feature cones, not final integrated retirement diffs.
They include their production modules and runtime templates and give no credit for
docs/tests/comments. Shared helpers and additional caller cleanup receive no credit.

| Proposal | Exclusive files | Source LOC | Physical LOC | Product effect |
| --- | ---: | ---: | ---: | --- |
| Retire Improve (recommended) | 28 | 7184 | 8818 | Remove repository-wide audit, request-to-plan authoring, plan board/index/re-anchor/pruning and plan-to-GitHub-issue publication. Keep review/fix/test, training and benchmark. |
| Retire standalone Benchmark product | 29 | 10174 | 13374 | Remove benchmark workspace/import/repair, human gold authoring/TUI, Harbor task compilation/package/runtime/oracle/judge calibration and aggregate objectives. Keep review/fix/test, archive/feedback training, Stage0, SFT/RFT and online RL. |
| Retire Osprey backend | 1 direct driver | 651 | 759 | Remove native Osprey provider/session/tool protocol integration. Other backend and observer machinery remains. Insufficient alone for the LOC target. |
| Retire offline reward calibration | 2 | 715 | 880 | Remove calibrate-reward grid/bootstrap/marginal analysis reports; retain core scoring, gate and training defaults. Insufficient alone. |

Retiring Pi is not recommended with online RL retained. Tracked rl/train/rl.toml
and rl/daydream_review/README.md explicitly use Pi because the pinned training
renderer accepts Chat Completions rather than Codex Responses/Claude Messages.
Pi and Osprey together remove only1344 direct source lines and would also remove
the supported online-training backend unless it were replaced. They are active
capabilities, not unused adapters.

## Recommended scope: retire Improve

The exclusive cone is the27 modules in daydream/improve/ plus
daydream/commands/improve.py. Complete pinned report and inventory:
/tmp/daydream-pr1444/improve-retirement-assessment/exclusive-scb.json.
The cone has37 high-CC and44 high-cognitive functions.

Implementation must remove Improve verb/subverb routing, built-in phases/prompts
and registered flow, runner audit setup/preflight, Improve-only configuration and
effort policy, and dependent advisory-plan machinery. The removed command should
fail explicitly rather than be interpreted as a repository target. This is
feature retirement, not a claim that supported behavior was preserved unchanged.
The PR must explain the new supported scope and migration.

Keep shared service discovery and its documented legacy service_roots fallback
for diagrams; keep independent Git snapshot preparation used by Codex and fixer
isolation. Keep historical Improve archive/trajectory wire decoding and existing
operator artifacts. Removing a producer does not authorize deleting stored plans,
worktrees, source files or historical evidence. Public extension hooks belonging
to Improve are intentionally retired; surviving named review/fix/diagram/custom
flows and their meaningful mutation, coverage, security and hook guarantees remain.

README, configuration documentation, extension inventory and bot issue-permission
explanation require updates. Retire tests whose sole capability no longer exists
and update mixed tests around the retained behaviors; do not skip tests, lower the
coverage floor or weaken guard verification. Run focused native workflows and all
ordinary repository hooks. Obtain independent review of the integrated baseline
diff, then remeasure every remaining/new production file and update the same PR.

The exclusive-cone counts above are historical feasibility evidence, not an
implemented retirement diff. The user chose to retain both capabilities. Current
internal-redesign totals and shortfalls are in production-simplification.json;
no hypothetical cone subtraction contributes to those measurements. Any future
scope decision would require complete caller/compatibility verification and a new
measurement including every retained and new production module.

## Alternative: retire standalone Benchmark

Measured inventory/report:
/tmp/daydream-data-retirement-proposals/benchmark_product-inventory.json
/tmp/daydream-data-retirement-proposals/benchmark_product-scb.json.
Only CLI dispatch imports that product from outside its cone. The optional Harbor
dependency can be removed with a fresh locked resolution after checking all shared
transitives; GitOps, Pydantic, YAML, HTTPX and ATIF remain. Generated benchmark
workspaces/bundles would become historical artifacts rather than maintained live
authoring/runtime formats. This option meets the arithmetic shortfall but gives
up the held-out benchmark product described as part of the open-model goal.
