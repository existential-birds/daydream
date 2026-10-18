# Native Pi successful-input convergence investigation

Investigation for [#1560](https://github.com/existential-birds/daydream/issues/1560),
2026-10-15. **No runtime policy change is selected.** The frozen record establishes
a successful-input reread loop, but the current baseline completed both historical
assignments. Eight matched synthetic reviews retained quality with either input
transport and showed no convergence gain from inlining the assignment. Elapsed
comparisons reversed direction on the second repeat. These results do not support
a new reread guard, evidence-summary mechanism, or production default.

The accompanying [sanitized measurements](measurements/pi-successful-input-convergence.json)
contain input/result hashes, per-invocation prompt growth, admission/coverage outcomes,
read classifications and costs. Raw historical source, captured prompts, tool output,
reasoning and private runtime logs remain operator-local.

## What the frozen evidence establishes

Capture SHA-256:
`e31451388fe831ba965eacdb8d58ed24de3b82bdebf11f217944ece9dfc460b7`.
The capture has one physical JSONL line. Scope documents are joined to the root's
`extra.subtrajectories` by trajectory ID. Producer:
`1c6fb39d33db6c57d9cf9a89e294a1268b95616a`.

| Scope | Calls / exact distinct calls | Exact duplicate excess | Last recorded prompt tokens | Native submissions |
| --- | ---: | ---: | ---: | ---: |
| React #0 | 31 / 11 | 20 | 93,257 | 0 |
| Structure | 46 / 21 | 25 | 72,680 | 0 |
| React #2 | 29 / 24 | 5 | 51,642 | 1 |

React #0 received the complete 2,652-character assignment and complete
17,646-character test response at its first tool-result boundary. The same
assignment result occurs six times; the same full test response occurs six times.
Eight further test reads return two repeated ranges: the 6,277-character prefix
has a continuation footer, and the 11,423-character suffix reaches the end.
The 14 test reads must not be described as 14 identical full reads. Result hashes
and exact argument hashes distinguish these cases in the measurements.

The assignment result SHA-256 is
`c0521d7e87279db6` (prefix); the full test-result SHA-256 is
`f69a97e6c4ba29c2` (prefix). Complete identities are in the JSON. React #0's
recorded prompt grows from 15,671 to 93,257 tokens. Repeated successful bytes are
captured; the model's later statement that it lacks the diff does not establish
that transport lost those bytes. The capture contains no original provider
request bodies and establishes no context reset or compaction.

The initial captured host stages explicitly grant 48 remaining calls and a 48-call
hard invocation allowance for all three scopes. React #0 phase-end metadata records
31 starts and 17 calls remaining; its native first-dispatch wall allowance is
479.996587 seconds after setup. The JSON includes these captured budget values and
precise lifecycle pointers, rather than inferring the original tool cap from current defaults.

The successful React #2 scope also repeats five exact calls. Repetition alone
therefore does not establish a failed review, unnecessary investigation, or
semantic completion. The failed scopes submitted no structured result; their
speculative reasoning is not admitted evidence.

## Native request/context path

The historical production route inspected for this investigation was:

1. `StageInputFactory.prepare` captures the revision-bound assignment and supporting
   inputs. `ReviewInvestigation` revalidates them and invokes `agent.run_agent`.
2. `PiBackend.execute` supplies the prompt through an `@file`, the review system
   instruction through an attachment, and the native submission schema in an
   attempt-local packet. Each stage in these experiments uses a fresh native session.
3. The packaged `pi_structured_output.ts` context hook replaces only its own
   `daydream_native_tool_budget` custom message and retains all other messages.
4. Installed Pi's SDK applies the extension context transform, restores system
   messages, converts custom messages to user messages and passes ordinary user,
   assistant and tool-result messages into the OpenAI-completions request path.
5. Native `structured_output` must settle successfully before host admission.
   A valid submission remains separate from correctness of its reviewer judgments.

A real installed-CLI loopback replay exercised four requests through this route.
The original user assignment survived every request. A 3,667-byte source response
survived first once, then twice, alongside a failed read and a later partial-range
result. Message counts were 3, 5, 8 and 10, with exactly one live budget note showing
8, 7, 5 and 4 remaining calls. The final native submission succeeded. This tests
ordinary request-time retention; it does not induce native compaction or establish
what an earlier remote service received.

At the time, `tests/test_pi_native_output.py::test_native_pi_live_budget_counts_batch_failures_and_replaces_generation_note`
checked prompt retention, every preceding tool-result identity including failures,
and exact successful source bytes across ample, submission-only, exhausted and
unbounded native note modes. The scripted provider previously checked note updates
and final output without detecting silently discarded history. No runtime repair
was claimed: the strengthened test passed against the then-current context owner.

Issue #1595 subsequently removes those recurring custom budget messages and
delivers the one-shot final-submission warning and distinct missing-submission
recovery as system instructions. The measurements above describe the earlier
mechanics. They do not establish history loss, a provider change, or a performance
benefit from that prompt-policy cleanup. The current native regression retains
the request-history checks while asserting the absence of recurring messages.

## Fixed comparison protocol

All 12 live scopes used `nous` / `deepseek/deepseek-v4.1-flash`, medium effort,
installed Pi 1.0.4 and Node v26.5.0 on the operator's Mac. The selected native API
was OpenAI completions. No provider, credential or security configuration was changed.
Trial Daydream producer was `86b797588218295c1a6ec848d688c313346f8ca7`.
The inspected context/prompt owners are unchanged from the frozen producer; the
Pi adapter difference is the already-merged temporary-capacity classifier.

Each scope kept **480 seconds and 48 observed tool starts shared across every
stage**. No stage or retry renewed the allowance. The experiment disabled host
transport retries equally for both variants and observed no provider failures.
No dataset recorder or uploader was activated. The trials used read-only native
file tools and the existing structured-admission lifecycle.

The intervention used the existing `build_per_stack_prompt(inline_diff=...)`
seam to put the current bounded assignment diff in the initial stage prompt,
while retaining its exact artifact pointer. Baseline used the existing pointer
transport. Both variants preserved full tool history, ordinary dependency/range
reads and unresolved-check/submission policy. No tool quota or inferred-completion
rule was added. No model-authored summary was introduced.

Synthetic reviews used the complete `ReviewInvestigation.run` lifecycle with three
fixed singleton first-pass batches and normal candidate handling. Fixing batches
excluded planner effects from this experiment. The outer runner's routing,
synthesis and repository-wide fanout were outside these trials.

The synthetic snapshots contain cache, pagination and name-formatting changes.
The clean API documents `None` as the cache-miss sentinel. The defect snapshot adds
two independently adjudicated regressions: batch cache reads bypass expiry, and
one-based pagination shifts every page's start by one. A hidden executable oracle
outside model-visible roots passes all five clean checks and fails exactly those
two checks on the defect snapshot. It also covers live falsey values, invalid page
arguments and empty/normalized names. Independent source and output adjudication
confirmed the ground truth, changed-side locations and finding dispositions.

The two repeats reverse order within each condition: clean baseline/inline then
inline/baseline; defect inline/baseline then baseline/inline. Snapshots, model,
effort, batch assignments and scope allowances stay fixed within each pair.

## Complete staged quality controls

Every row completed all three assigned stages, admitted every assigned decision,
submitted three native results and left no unfinished work. Every row has zero
within-invocation duplicate exact calls and zero repeated full-file responses.

| Snapshot / repeat | Baseline seconds / calls | Inline seconds / calls | Independently adjudicated quality, both variants |
| --- | ---: | ---: | --- |
| Clean / 1 | 75.141 / 17 | 16.541 / 13 | 0 findings, 0 false positives |
| Clean / 2 | 26.541 / 14 | 28.139 / 16 | 0 findings, 0 false positives |
| Defect / 1 | 73.326 / 12 | 27.016 / 16 | 2/2 seeded defects, 0 false positives |
| Defect / 2 | 53.249 / 13 | 65.793 / 10 | 2/2 seeded defects, 0 false positives |

Across the eight controls, all 24 target decisions were admitted and all eight
seed instances were found. Both variants completed 4/4 scopes. The second repeat
reverses the time advantage in each snapshot. Four equal-argument reads in one
inline run occur across fresh stages; they are not within-invocation duplicates.
Fresh-stage context gathering must remain separate from unproductive repetition.

These controls establish preserved quality for these two small snapshots. They do
not reproduce the original loop and provide no evidence that the intervention
improves useful completed work at the fixed deadline. A smaller read count or one
faster run does not pass that gate.

## Historical assignments on the current native path

The frozen source head was `cf338a312114454460ceb692bf02d48c6657d9aa`, with
base `302f96dd53d4c9b5b835b7d9cda5204fc52fa458`. Its 32,551-byte Git diff has
SHA-256 `0de25b682d45e3a41deffccd153a866bd8e5f6ebe51e23e2c623d5a692e2c02a`,
matching the capture's declared diff identity. Only tracked Git content was
materialized; operator working files and datasets were preserved. Captured
intent and the complete available exploration-summary/affected-file contexts
were supplied unchanged as retained supporting data.

The original captured stage JSON is damaged by redaction, and some captured
result hashes differ from raw Git blobs. Current production prompts and access
paths were regenerated against exact Git content. These are matched current
comparisons of the historical assignments, not byte-identical replay of the
original requests. The frozen record does not identify its native CLI version.

| Assignment | Baseline seconds / calls | Inline seconds / calls | Outcome, both variants |
| --- | ---: | ---: | --- |
| Formerly failing React #0, one test-file target | 9.767 / 5 | 11.684 / 4 | Complete, one admitted native result, 0 findings |
| Previously successful React #2, three targets | 67.043 / 24 | 41.251 / 16 | Complete, one admitted native result, 0 findings |

No historical trial repeated an exact call within its invocation. The baseline
converged on the small single-stage case. These unseeded assignments have no
comprehensive independent defect oracle, so their recall is unknown. Their empty
results are admission/completion observations, not proof that the snapshots have
no defects. One pair per task cannot establish absence of an intermittent loop.

## Decision and validation

Keep production context delivery and review/submission defaults unchanged. The
native replay gives no evidence for ordinary history loss in the tested path,
and the live baseline did not reproduce the incident. The evaluated intervention
preserved the controlled quality checks but demonstrated no convergence benefit
and no consistent elapsed advantage. The historical incident remains valid
observed evidence; this investigation makes no claim that it was fixed.

All 12 trials completed 28 admitted stages and 160 calls in 495.491 seconds of
summed sequential trial elapsed time. The native CLI returned zero dollar totals;
actual billed cost is unknown. No token-throughput or upstream-routing analysis
was performed. No production timeout, cumulative limit, retry policy, admission rule or
cancellation behavior was changed.

Focused checks completed on the investigation diff:

- Native loopback budget/context test: all four modes passed in 3.07 seconds.
- Independent hidden oracle and source/output adjudication: clean 5/5 checks;
  seeded exactly two expected failed checks, all eight live seed instances found.
- `uv lock --check`, Ruff for the changed test, mypy for the changed test and
  `git diff --check` passed.

The full contribution gate and latest-head PR reviews belong to the PR's Test Plan.
The operator retains exact private manifests, scripts and raw experiment evidence
for replay with authorized access. The public native-context regression is runnable
with the installed Pi CLI and a synthetic localhost provider:

```bash
uv run pytest tests/test_pi_native_output.py
```

A future intervention should first reproduce an at-risk invocation with preserved
input identities, then repeat matched complete/seeded controls at unchanged limits.
Necessary partial/range/dependency reads, contradictory evidence, admitted prior
findings and truthful incomplete coverage remain part of the acceptance gate.
