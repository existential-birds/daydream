# Phase Labels — Display Decision (Task 4 input)

## 1. Phase keys in use

Single source of truth: `DaydreamPhase(str, Enum)` at
`daydream/trajectory.py:141-158`. All call sites pass an enum member; no
free-form strings. The enum `.value` (snake_case string) is what lands in
`Step.extra['daydream_phase']` and is what the per-phase Markdown table will
group rows by.

| Enum member | `.value` (raw key) | Call sites (file:line) |
|---|---|---|
| `REVIEW` | `review` | `daydream/phases.py:642` |
| `PARSE` | `parse` | `daydream/phases.py:696` |
| `FIX` | `fix` | `daydream/phases.py:746`, `808`, `867`, `961`, `973`, `1014`, `1049` |
| `TEST` | `test` | `daydream/phases.py:781` |
| `INTENT` | `intent` | `daydream/phases.py:1127` |
| `ALTERNATIVES` | `alternatives` | `daydream/phases.py:1193` |
| `PLAN` | `plan` | `daydream/phases.py:1331` |
| `PR_FEEDBACK` | `pr_feedback` | `daydream/phases.py:897`, `1084` |
| `DEEP` | `deep` | `daydream/phases.py:1443`, `1457`, `1519` |
| `EXPLORATION` | `exploration` | `daydream/exploration_runner.py:250`, `278` |

Indirect call sites (forwarders, not new keys): `daydream/agent.py:260` (`run_agent`
phase param), `daydream/agent.py:324` (`recorder.invocation(phase=phase)`),
`daydream/trajectory.py:691` (`invocation()`), `daydream/trajectory.py:745`
(`create_dispatch_step()`), `daydream/trajectory.py:954` (`Invocation` ctor).

`daydream/runner.py` and `daydream/deep/orchestrator.py` contain **zero**
`phase=` invocations — they delegate exclusively to `phases.py` /
`exploration_runner.py`.

## 2. Decision

**(b) Explicit dict mapping**, defined in the new PR-comment renderer module
(Task 4). Reasons:

- Naive `str.title()` / `replace("_", " ").title()` produces wrong casing for
  the two compound keys: `pr_feedback` → "Pr Feedback" (should be **PR
  Feedback**) and the test phase wants the verb form **Test & Heal** (the
  function is `phase_test_and_heal`; the enum was shortened to `test` but the
  user-facing column should reflect the actual phase semantics).
- Explicit dict is one screen of code, trivially reviewable, and gives us one
  obvious file to edit when a new `DaydreamPhase` member is added.
- A transform function would still need PR-specific overrides, so it
  collapses to a dict with extra ceremony.

## 3. Mapping table

```python
PHASE_LABELS: dict[DaydreamPhase, str] = {
    DaydreamPhase.REVIEW:       "Review",
    DaydreamPhase.PARSE:        "Parse Feedback",
    DaydreamPhase.FIX:          "Fix",
    DaydreamPhase.TEST:         "Test & Heal",
    DaydreamPhase.INTENT:       "Understand Intent",
    DaydreamPhase.ALTERNATIVES: "Alternatives",
    DaydreamPhase.PLAN:         "Plan",
    DaydreamPhase.PR_FEEDBACK:  "PR Feedback",
    DaydreamPhase.DEEP:         "Deep Review",
    DaydreamPhase.EXPLORATION:  "Exploration",
}
```

### Guardrails for Task 4

- Place `PHASE_LABELS` next to the table renderer (not in `trajectory.py` —
  display concern, not recording concern).
- Lookup with `.get(phase, phase.value.replace("_", " ").title())` so an
  unmapped future enum member never crashes the PR-comment build; instead it
  degrades to a snake-cased Title Case fallback.
- Add a unit test that iterates `DaydreamPhase` and asserts every member has
  an entry — fails CI the moment someone adds a new phase without updating
  the label map.
- Phase **column ordering** in the per-phase table is a separate concern
  (M2 spec); recommend ordering by first-seen invocation timestamp rather
  than enum declaration order, so flow-specific tables (`--ttt`, `--pr`,
  `--deep`) read in execution order.
