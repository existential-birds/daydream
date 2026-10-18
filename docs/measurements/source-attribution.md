# Source attribution: historical paired evidence

The [issue #1518 comparison](https://github.com/existential-birds/daydream/issues/1518#issuecomment-6252430113)
observed 14.7% lower aggregate runner time and 5.7% fewer gross tokens with bounded
source references, with no observed quality regression on its small corpus. These
are descriptive results from six pairs, not an established causal or repeatable
speedup, population quality equivalence, or observed billing savings.

## Frozen comparison

The final experiment ran on 2026-10-18, before the implementation was rebased:

| Identity | Value |
| --- | --- |
| Baseline commit | `ad3409b16c69ed5341b8a052d1a8702ae29e9e85` |
| Treatment commit | `e93febdff316e6d7ebd3530f63e41ba215a71641` |
| Experiment manifest SHA-256 | `272831623911e064ecac37bd88e1f6097c404c5721bb0fc8df54dfda36d81a7f` |
| Corpus manifest SHA-256 | `fef8a6b0f83288decce4fea4fc06d7f2bec00c1ea375ca6dbb3414c070cc3c95` |
| Common task-context SHA-256 | `2991235d03ace7d4e9d8a131e3c7b33a97690b38537a3952c68d169205aa61fe` |

Three self-contained Python Git cases each changed two files:

- **Clean rename:** a replacement-heavy validation/module rename with Unicode and
  caller context, preserving behavior under an explicit built-in-input contract.
- **Deleted guard:** removal of required blank-actor rejection, beside a clean
  display refactor. An independent check passed at base and failed at head.
- **Boundary UTF-8:** an inclusive-quota comparison regression, beside a clean
  label refactor and a quote-only replacement containing 4,400 copies of `界`.
  The long line exercised split fragments; the seeded defect was elsewhere in
  the same file. An independent check passed at base and failed at head.

Each case ran twice per arm: 12 sequential production `runner.run` invocations,
six pairs. Arm order alternated within each case's repetitions. Each invocation
used a fresh pinned fixture copy and separate artifacts. Ground truth stayed
outside model-supplied inputs; task context specified intended compatibility and
the inclusive quota requirement equally for both arms.

Both arms used subscription-authenticated Codex, `gpt-6.1-sol`, medium ordinary
reasoning effort, normal low-effort delivery finalization, balanced profile and
fanout 1. Ordinary intent, alternatives, review flags and production lifecycle
safeguards remained enabled. Structural review, arbitration and suppression were
disabled equally; merge used the single-stack host path. The final experiment had
a fixed 12-run sample and eight-native-call ceiling per invocation. Earlier
artificial time/token stops were removed before this final attempt; it retained
ordinary production deadlines. All final invocations completed without cutoff,
using 52 native calls, 26 per arm.

## Time and tokens

Times are runner seconds, excluding adjudication and waiting between invocations.
Gross tokens are input plus output; cached input is a subset. Mirrored native
metrics and cost events were reconciled rather than added twice.

| Case / repetition | Baseline tokens / seconds | Treatment tokens / seconds | Treatment minus baseline |
| --- | ---: | ---: | ---: |
| Clean rename 1 | 339,958 / 68.625 | 317,688 / 70.481 | −22,270 / +1.856s |
| Clean rename 2 | 340,042 / 62.244 | 293,721 / 84.233 | −46,321 / +21.989s |
| Deleted guard 1 | 317,304 / 76.610 | 297,570 / 67.916 | −19,734 / −8.694s |
| Deleted guard 2 | 253,873 / 71.369 | 258,097 / 72.169 | +4,224 / +0.799s |
| Boundary UTF-8 1 | 849,493 / 325.956 | 853,599 / 213.392 | +4,106 / −112.564s |
| Boundary UTF-8 2 | 771,594 / 199.604 | 689,169 / 177.975 | −82,425 / −21.629s |
| **Total** | **2,872,264 / 804.409** | **2,709,844 / 686.166** | **−162,420 / −118.243s** |

Treatment was faster in three of six pairs. The median paired time difference was
−3.947s, while the per-arm run-time medians were 73.990s baseline and 78.201s
treatment. One boundary pair contributed 95.2% of the net time reduction; both
clean pairs were slower. Repetitions of three purposively selected tiny cases are
correlated, and native execution and caching vary. The aggregate cannot establish
the expected latency effect on ordinary repositories.

| Token component | Baseline | Treatment |
| --- | ---: | ---: |
| Input | 2,855,757 | 2,690,626 |
| Cached input | 2,062,080 | 1,853,824 |
| Uncached input (input minus cached) | 793,677 | 836,802 |
| Output | 16,507 | 19,218 |

Despite lower gross tokens, uncached input increased 5.4% and output increased
16.4%. Gross-token savings therefore do not establish API-cost savings. Execution
used existing subscription access; API-price estimates are not observed billing.

## Quality and the shipping decision

Independent adjudication scored explanations, candidates, every required
continuation, frozen reference correspondence, actual check results and terminal
persistence. Blinding was partial because structured references reveal treatment.
Both arms achieved 12/12 faithful complete-file judgments, admitted 20/20 required
part decisions and detected 4/4 seeded opportunities. Neither had a confirmed
false positive, false-complete outcome, direct side/range/offset misattribution,
or fabricated/unresolved check claim. Treatment submitted 20/20 valid typed
references, including ten independently checked exact UTF-8 fragment references.
Valid references establish correspondence/existence, not reviewer understanding.

The absolute pilot thresholds passed. Its separate, predeclared shipping-benefit
gate required at least one additional faithful file judgment or one eliminated
baseline false-complete outcome. That gate **failed to demonstrate benefit**:
both arms tied at perfect observed scores. The issue was consequently closed
without merging. Renewed user direction reopened the shipping decision to consider
useful attribution guarantees and promising efficiency evidence. This does not
retroactively change the old gate or turn a tie into proven quality equivalence.

Two earlier interrupted attempts are excluded from the six-pair analysis and
remain disclosed in the issue comment: one baseline exceeded its original token
cap; another was interrupted by an external process guard. The final fixed sample
started after separately approved limit and guard repairs. No outcome-selected
replacement, tuning or sample expansion was used within that final sample.

## Reproducibility and current validation

The original corpus, harness, manifests, raw records and adjudications are private,
untracked task artifacts, not fresh-checkout runnable repository assets. The
self-contained cases above were controls, not an exact replay of the unavailable
historical PR #1516 capture. Reproducing new model observations also requires the
native backend and suitable access; their outputs and latency are nondeterministic.
This report records the historical revisions, not measurements of a rebased or
subsequently repaired implementation.

During recovery, read-only checks independently matched the three identity hashes
above, all 12 frozen source snapshots and three patches against corpus Git objects,
all 12 final adjudication digests, and their 134 bound evidence files. No evidence
file was missing or mismatched. Result files had complete usage streams, exit 0
and no cutoff; summing them reproduced the table totals.

With the separately supplied original artifact directory as `EVIDENCE_ROOT`, its
top-level identities can be checked without executing the private harness:

```sh
shasum -a 256 \
  "$EVIDENCE_ROOT/pilot-v3-repaired-20261018/frozen-manifest.json" \
  "$EVIDENCE_ROOT/pilot-corpus/manifest.json" \
  "$EVIDENCE_ROOT/pilot-corpus/task-context.txt"
```

Current deterministic tests and `make check` must validate the current PR revision
separately. Historical contract checks and model outcomes cannot substitute for
that gate. A new comparison would need its own fixed inputs and plan; these six
pairs alone support only the observed savings and absence of observed regression.
