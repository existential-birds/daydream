# Cloud review rerun: completion improved, reconsideration persists

Compared baseline run `b3272dd6-cd46-4041-9903-5209cb95fb2a` with rerun `19a7f7b7-ed26-4582-b752-9d42c500f559` on 2026-10-11. Three read-only agents independently examined timing/coverage, substantive behavior, and finding quality. No production changes, live model calls or GitHub writes in this inspection.

## Inputs and comparison boundary

Baseline capture `~/.daydream/dataset/runs/aba364fae2578aa3c66d08b224ab19823bd6c8cdaf2c5ea4d135e7834ba6eef0.jsonl`, SHA256 `39129a9f606046a498e0d708f9316b5ac257f7be9bb7115ed41cf8ed4b9fdf02`.

Rerun capture `~/.daydream/dataset/runs/7b364efec2d9365c8a0ea0b3710ed4861c24f8759560cbc97084f8f97ed891a5.jsonl`, SHA256 `404c0cf804cb4ac14fcf92e34edf0bf6a9fe354544c4d1770e8cbe13db132bae`.

Both target cloud PR #28, HEAD `38ca72ba7a87c4058877a744c5c128db3bc56d66`, merge base `3cc00976f40e5a9b4ddbcc55d7811320d8b6d898`, identical diff key `f3a8efa742459d532c32db03267b2a2cdf249034198c5d0fe9328f9a3d087d4f`, full diff and 39 changed paths. Pi / nous / deepseek-v4.1-flash, default profile, balanced latency route match. Producer commit `70ab7478` matches; baseline clean, rerun dirty with authorized effort/prompt adjustments. Rerun records per_stack_review reasoning_effort=medium; baseline records null. Independent native thinking-setting value is unavailable. Effort and instruction removal changed together.

Review timing ends at consolidated merge, excluding subsequent optional diagram. Whole-run times include preparation/cache effects and are reported separately.

## Completion and timing

| Measurement | Baseline | Rerun |
| --- | ---: | ---: |
| Fanout critical path | 960.118 s | 732.185 s |
| React elapsed | 960.030 s | 732.050 s |
| Generic elapsed | 407.481 s | 222.100 s |
| Structure elapsed | 387.545 s | 250.094 s |
| Review through merge | 960.181 s | 806.402 s |
| Whole run | 1248.686 s | 870.477 s |
| Review tool starts | 168 | 212 |
| Review generations | 64 | 97 |
| Review output tokens | 71779 | 125460 |
| React admitted batches | 3 / 8 | 8 / 8 |
| React decided units | 12 / 25 | 25 / 25 |
| React observed tool starts | 64 / 96 | 96 / 96 |

Rerun generic/react/structure terminal coverage is complete, partial_evidence=false, with no diagnostics/reason codes. All reviewer stages have attempt=1 and successful admitted submissions. Baseline React timed out in batch four; four later batches never started. Both captures report top-level success, which must not be confused with baseline complete coverage.

Fanout decreases 227.933 seconds (23.74%) while completing more assigned work. Rerun adds 44.613 seconds suppression and 29.540 seconds actual merge where baseline had noops. Rerun uses cached preparation: baseline exploration and intent actual calls cost about 163 seconds. Diagram durations (122.841 -> 61.023 seconds) are excluded from review measurements.

## What improved and what did not

Real correctness review occurred: grants/bootstrap, migration constraints, route origin/session gates, auth continuation, setup states, GitHub identity/permission narrowing/pagination and fixture behavior. React now submits every planned batch, including two ordered installation-test parts.

Reconsideration persists. Fourth React stage still assumes a planted/introduced bug and repeatedly promises submission before reopening settled UPSERT/session/token/pagination questions. Fifth stage emits 2831 reasoning words after saying it has enough before more checks, then 1216 further words after its last import check before an empty submission. Structure also repeatedly debates the same expired-intent candidate, albeit ultimately reports a grounded defect.

Comparable first three React batches: reasoning words 12587 -> 11445 (~9% lower), emitted reasoning characters 93632 -> 85547 (-8.63%), output tokens 27009 -> 25326 (-6.23%), wall time 653.246 -> 240.139 seconds (-63.24%). Individual reasoning lengths move in both directions. Batch one has virtually unchanged reasoning and slightly more output tokens while running roughly three times faster. Observed effective output rate rises ~41.35 -> 105.46 tokens/second. Provider service conditions/effort dynamics are not isolated. This is evidence for successful completion, not a demonstrated cure for looping or causal speedup from either individual edit.

## Findings independently checked

Four confirmed admitted candidates, no open/unresolved candidates, produce three final findings:

1. **Functional defect confirmed:** an expired unconsumed intent retaining an installation_id projects as incomplete and offers Refresh, but refresh rejects expired intents and the server action redirects to /error. Source chain: installations.ts:68, :178-179, :233/:41; setup/page.tsx:94-103; setup/actions.ts. Existing-installation continuation deferral complicates remediation but does not make the unusable Refresh action work. The findings agent executed unchanged source functions with external auth/database/provider boundaries stubbed: projection returned incomplete with refresh guidance; Refresh threw Installation intent unavailable. This is source execution, not browser E2E.
2. **Sleep-flakiness finding is false as described:** provider expiry is floor(issuance time/1000)+lifetime; admission converts to ms. A 1.1-second wait starts after sign-in for a 1-second lifetime, so added delay makes expiry more certain. The claimed delayed scheduling leaves the token still valid argument is reversed. A 4000-case arithmetic control found no such valid-after-wait case. Different clock-jump or premature-setup expiry triggers are unproven here.
3. **Low-impact documentation omission is factual:** feature README:22 omits the conditional configuration spec named in installations.md:85-86 and tests/installation-configuration.spec.ts:5. Exhaustiveness of Source owners is not formally guaranteed; adding the actual spec is a reasonable correction.

## Material decision-consistency failure

Rerun React trajectory `:dispatch:1:fork:1`, step 51 admits sleep candidate react:candidate:1 in installations.spec.ts. Step 55 correctly recognizes that slower CI leaves the token expired, then defers to a purported host closed decision and emits HIGH-confidence medium react:2 in session-lifecycle.spec.ts. Host admission accepted earlier model judgment; it did not independently confirm it.

Root suppression step 6 rejects the installation sibling (react:1) as not confirmable as a real race. Root merge step 9 still publishes the session sibling (react:2), citing the rejected sibling as host already confirmed. The medium finding bypasses the suppression selection that rejected its low-severity sibling. This is a false authority/consistency failure across decisions, not merely excess reasoning. Closed judgments must retain their actual provenance and must not prevent source-supported correction.

## Limits and follow-up

Complete coverage records admitted model decisions; it does not independently certify recall or a clean repository. This is one matched-snapshot before/after pair with joint edits and cached preparation. No claim of improved semantic quality is justified. Existing #1522 retains attributed dependency notes; this inspection strengthens the need to preserve attribution rather than elevate notes/closed candidates into host verification. No additional issue was created during this inspection.

## PR disposition after the code investigation

The subsequent PR fixes truthful per-attempt Pi wall allowances (#1521) and reviewer-origin attribution/source-supported contradiction guidance (#1523), alongside the medium per-stack default and neutral review framing. Contract 10 invalidates earlier complete caches. Deterministic tests cover actual request bytes, preparation/retry deadline consumption, unbounded custom flows, provenance, retained findings/incomplete coverage and cache invalidation. These checks do not establish model obedience or semantic-quality improvement.

Supporting-dependency note relevance stays in #1522. Bounded rejected-adjudication context for the existing merge turn is separately specified in #1524. Source replay confirmed that suppression selected/dropped the correct UID; the downstream issue is missing adjudication grounds rather than incorrect UID binding or a reason to propagate semantic rejection automatically.
