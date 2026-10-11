# Full-change risk population evidence (#1517)

This is a deterministic reconstruction from the frozen input named in
[issue #1517](https://github.com/existential-birds/daydream/issues/1517), not a
new model run. The original `scripts/review_1516_reporting_probe.py`,
`docs/measurements/review-1516-reporting.json`, and investigation document were
absent from the workspace and available Git history. Searching the local
Daydream, GitHub, and Codex trees did not recover them. The issue's original
probe command was therefore **not run**.

## Input verification

The single-object capture was loaded with `json.load`. Its SHA-256 is
`ed579ec396b6dd62ed206a0867dd68e20d3bfe7184cb4ed290dc3c5539161825`.
Documents were read from `trajectories.value.documents` and joined by
`trajectory_id` through `documents[0].extra.subtrajectories`, not document
order. The patch SHA-256 is
`634e4990c9c80918c09e7eaa322a27971e5de72754f1d572f75f8f505f927c53`.

The following diff is byte-identical to that frozen patch (no byte differences):

```sh
git diff --no-ext-diff --no-textconv \
  1a7073f2c49b49b76b8d81a93680c868abd3d031 \
  0176602ef4bf7c92ab7de23068e631b66b4d3a52
```

## Actual-function reconstruction

`bound_deep_diff`, `iter_diff_blocks`, `diff_signals`, `summarize_risk`, and
`route_for` were replayed on the verified patch, using the captured full-change
file/stack counts (68 files, 17 stacks).

| Population | UTF-8 bytes | Lines | File blocks |
|---|---:|---:|---:|
| Captured full patch | 926,308 | 4,765 | 68 |
| Bounded prompt diff, including marker | 14,523 | 211 | 9 |
| Retained patch, excluding marker | 11,865 | — | 9 |
| Truncation marker | 2,658 | — | 0 |
| Dropped patch | — | — | 59 |

The bounded input reproduces the captured `risk.signals` and scores. Both
inputs match security and migration lexical categories. Full input also
matches concurrency and interface; persistence remains false. Size score is
1 for bounded input and 2 for full input; breadth score stays 1. These are
surface signals, not demonstrated defects, and size/breadth scores are
audit-only.

Both select the same balanced route: wonder `high`, arbiter `high`, sharding
enabled, maximum three targets per group. Selected policy differs from
execution: captured wonder was **folded**, and captured arbiter selected no
targets (no groups, execution reported unsharded). There was no independent
wonder invocation. The original review reported no confirmed findings; this
reconstruction establishes neither a missed defect nor a model recall benefit.

## Corrected boundary

The public-runner regression uses a real two-file Git repository with a
password-shaped changed line only in the later oversized block. The old
bounded-input boundary selects balanced wonder `medium`; full-input routing
selects `high`. The production path must publish full-change signals and the
snapshot-bound `diff_population` counts, retain complete disk evidence, and
keep existing sanctioned prompt transport bounds.

The latency-profile corpus is hand-built; no corpus rebuild command was found
or run. Historical fixtures are unchanged.
