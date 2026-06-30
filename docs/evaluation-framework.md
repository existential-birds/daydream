# Evaluating Daydream vs. Greptile

## Goal

Determine whether Daydream provides materially better pre-merge review than Greptile, with sufficient rigor that the conclusion is defensible to engineers, leadership, and external observers.

## Two Evaluation Arms

### Arm 1 — Quantitative: Martian Code Review Benchmark

- **Daydream only** (Greptile already scored on the public leaderboard) [1].
- Martian measures precision, recall, and F1 against golden comments on 50 offline PRs (Sentry, Grafana, Cal.com, Discourse, Keycloak) and an online mode scoring 5,400+ real bot-reviewed PRs [1].
- Report: overall F1, precision, recall, and high-severity-filtered F1.
- **Limitations**: ground truth is developer action (not objective correctness), LLM judge introduces variance, offline set is small and potentially leaked, gold sets understate precision for high-recall tools [1].

### Arm 2 — Qualitative: Blinded Engineer Review on Internal PRs

- Both tools run on identical PR snapshots with equivalent context access.
- All comments anonymized, normalized, and randomly ordered.
- Engineers rate blinded comments; a human gold baseline enables recall measurement.
- Methodology informed by Atlassian's RovoDev deployment study [9] and SWR-Bench's PR-centric evaluation design [5].

---

## Evaluation Dimensions

### Substantive Review Quality

| Dimension | Question |
|---|---|
| Intent | Does the code match the ticket/PR description? |
| Cross-service dependencies | Downstream/upstream impact? Mono vs roman-api? |
| Maintainability | Abstraction where reuse was better? Maintainability regression? |
| Toothless tests | Do tests truly exercise changed paths? Overmocking? Missing edge/negative/concurrency tests? |
| Silent contract breaks | Value unchanged, behavior changed? |
| Incomplete enumeration | New value added, other branches/handlers not updated? |
| Swallowed errors | Errors caught too generally, or discarded needlessly? |
| Security | Injection, authz bypass, secrets handling, unsafe deserialization, tenant isolation? |
| Performance / regression | N+1 queries, unbounded loops, memory leaks, missing pagination, blocking calls on hot path? |
| Concurrency | Race conditions, deadlocks, shared mutable state, idempotency, retry safety? |
| Data / migration safety | Backward compat, nullable columns, destructive migrations, rollback safety? |
| API contract validation | Breaking changes to public endpoints, versioning, deprecation, schema changes? |
| Observability | Missing metrics/logs/traces, silent failure modes, unclear error messages? |
| Documentation drift | Comments/docs/runbooks/changelog updated to match the diff? |

Security dimensions informed by SeRe dataset categories [11]. Test adequacy informed by CR-Bench defect taxonomy [14].

### Meta-Quality

| Dimension | Question |
|---|---|
| Precision | Were issues real or hallucinations? (adjudicated TP / all findings) |
| Recall / false negatives | What real issues were missed? (requires gold baseline) |
| Calibration | Are severity rankings in line with human expectations? Does the tool distinguish "must fix" from "consider"? |
| Groundedness | Are real code paths cited, or generic advice? All factual claims supported by available code? |
| Actionability | Can the engineer fix the issue directly from the comment? Informed by c-CRAB's review-to-fix methodology [6]. |
| Silence | Does the agent know when to say nothing? (correct silence on clean PRs vs inappropriate silence on buggy ones) |
| Fix correctness | When a fix is suggested, does it compile/pass tests/introduce regressions? Informed by c-CRAB [6] and SWE-bench [20]. |
| Reproducibility | Same PR reviewed N times; do findings materially differ? |

---

## Methodology

### Human Gold Baseline

The single most important methodological requirement: **a human gold baseline is needed to measure recall, not just precision.** This gap was identified across multiple benchmark studies [1][5][9].

- Subset of 20-40 PRs receives independent human review by 2-3 engineers **before** seeing tool output.
- Time-boxed: 20-30 min per normal PR, 45-60 min for large PRs (prevents impossibly thorough baseline).
- Gold issues deduplicated, severity-labeled, and adjudicated by a senior reviewer.
- Tool-discovered real issues absent from the initial gold set may be added after adjudication ("gold expansion") [1].
- Report gold expansion rate: percentage of initial FPs later accepted as real.

### Blinding Protocol

- Tool identity hidden from all raters.
- Comments normalized to identical formatting; product-specific phrasing removed.
- Comments randomly ordered per PR.
- Raters cite code evidence for TP labels (prevents anchoring on tool output).

### PR Selection

- **Sample size**: 50-100 internal PRs for the qualitative arm. Minimum 30 for directional signal.
- **Stratify** across: repo/service, language, PR size, PR type (feature, bugfix, refactor, migration, config).
- **Include clean PRs** (no known issues) to test noise/silence behavior. This follows SWR-Bench's design of including 500 clean PRs alongside 500 change-PRs [5].
- Document exclusion criteria before evaluation.
- Use paired design: both tools review the same PRs.

### Comment Scoring

- Split multi-issue comments into atomic findings before scoring.
- Each finding rated by minimum 2 engineers; disagreements go to senior adjudicator.
- 3 raters for high-severity findings or disputed cases.

**Per-comment rubric** (5-point anchored Likert, 4 required + optional):

| Dimension | 1 | 5 |
|---|---|---|
| Correctness | Clearly wrong | Definitely correct |
| Actionability | Cannot act on it | Precise fix/test suggestion |
| Severity calibration | Severely misstated | Excellent prioritization |
| Overall usefulness | Harmful/wasted time | Should block or strongly influence merge |

Rubric design informed by DeepCRCEval's finding that less than 10% of benchmark comments are high quality for automation, and that BLEU/text-similarity metrics poorly capture review usefulness [10].

Plus binary labels: *Would you want this posted? Would you change code because of it? Duplicate? Category?*

**Per-PR rubric** (after viewing all comments):

- Did the tool catch the most important issues? (yes/no + free text)
- Was comment volume appropriate? (1-5)
- Would you enable this tool on this repo? (yes/no)
- Estimated reviewer time saved/cost? (minutes)

### Inter-Rater Reliability

Use **Krippendorff's alpha** (handles ordinal scales, missing ratings, multiple raters) [19].

| Alpha | Interpretation |
|---|---|
| >= 0.80 | Strong / reliable |
| 0.67-0.79 | Acceptable for tentative conclusions |
| 0.60-0.66 | Weak; use with caution |
| < 0.60 | Revise rubric or retrain raters |

Also report raw agreement percentage. Disagreement itself is signal — ambiguous comments may be poorly grounded.

**How alpha works.** Alpha measures how much observed disagreement exceeds what you'd expect by chance:

```
alpha = 1 - (D_o / D_e)
```

- **D_o** (observed disagreement): computed from actual rater pairs who disagree on the same item.
- **D_e** (expected disagreement): derived from the **marginal distribution** of your own data — the frequency histogram of each score value, pooled across all raters and items.

**What "marginals" means concretely.** The marginal distribution is the proportion of all ratings that received each score. For example, with 250 total ratings across 100 comments:

```
Score 1:   8 ratings  → p_1 = 0.032
Score 2:  22 ratings  → p_2 = 0.088
Score 3: 105 ratings  → p_3 = 0.420
Score 4:  80 ratings  → p_4 = 0.320
Score 5:  35 ratings  → p_5 = 0.140
```

Expected disagreement is then:

```
D_e = Σ_c Σ_c'  p_c * p_c' * δ²(c, c')
```

where δ²(c, c') is the distance metric (see below). In plain terms: for every possible pair of scores, multiply the probability of drawing each score independently times the squared distance between them. D_e is not specified or estimated separately — it falls directly out of the score distribution in your data. Skewed distributions (most ratings cluster at 3) produce low D_e, so alpha demands higher observed agreement to register as reliable.

**Ordinal vs nominal alpha.** The distance metric δ² determines whether alpha uses the ordering of your scale:

| Level | δ²(c, c') | When to use |
|---|---|---|
| **Ordinal** | (rank_c - rank_c')² | Likert scales. Penalizes a 1-vs-5 disagreement more than 3-vs-4. **Use this for the per-comment rubric.** |
| **Nominal** | 0 if same, 1 if different | Categorical labels with no ordering (e.g., issue category = security/performance/maintainability). Any mismatch is equally bad. |
| **Ratio** | ((c - c') / (c + c'))² | Continuous physical measurements. Not applicable here. |

For the per-comment rubric (correctness, actionability, etc. on 1-5 scales), use **ordinal** alpha. For binary labels (posted: yes/no), use **nominal** alpha. Report ordinal alpha as the primary IRR statistic.

**Key difference from Cohen's kappa.** Cohen's kappa computes expected agreement from the marginals of two specific raters. Krippendorff's alpha pools across all raters into a single marginal distribution, making it suitable for designs with 2-3 raters per item, uneven assignment, and missing ratings — exactly the comment-scoring design above.

### Statistical Reporting

#### The core question: is the difference real?

Every metric we compute (precision, recall, F1, usefulness scores) is an estimate based on a finite sample of PRs. The question statistics answers is: **if we ran this evaluation again on a different set of PRs, would we see the same ranking, or could the result be a fluke?**

There are three things that can go wrong:

1. **The difference is within noise** (not enough PRs to tell). The tools might actually be tied.
2. **We tested so many dimensions that one looks good by accident** (the multiple comparisons problem).
3. **Our sample of PRs isn't representative** (selection bias, covered in the bias checklist above).

The sections below address each.

#### Confidence intervals: how precise is our estimate?

**In plain terms:** A confidence interval (CI) is the range of values that our true score could plausibly take, given the data we have. If Daydream's precision is 65% with a 95% CI of [58%, 72%], we're confident the true precision is somewhere in that range. If Greptile's CI is [45%, 76%], the two ranges overlap so much that we can't declare a winner.

**For ML researchers:** Use **PR-level bootstrap resampling** (1,000-10,000 iterations). On each iteration, sample PRs with replacement, recompute precision/recall/F1 for both tools on that resampled set, and record the difference. The 2.5th and 97.5th percentiles of the difference distribution give a 95% CI.

**Why PR-level, not comment-level:** Comments within the same PR are not independent — a tool that misses one dependency issue on a PR tends to miss related issues too. Treating comments as independent inflates the apparent sample size and produces artificially tight confidence intervals. PR-level bootstrap respects the clustering structure.

**What the CI tells you:** If the 95% CI for the difference (Daydream - Greptile) excludes zero, the difference is statistically significant. If it includes zero, the tools may be equivalent on that metric. Report CIs for all primary metrics, not just point estimates.

#### Minimum detectable effect: what difference is worth caring about?

Not every statistically significant difference matters operationally. A 2-point precision difference on 10,000 comments may be "significant" but useless in practice.

**Practical thresholds for this evaluation:**

| Metric | Meaningful difference | Inconclusive below |
|---|---|---|
| Precision | >= 10 percentage points | < 10 pp |
| Recall | >= 10 percentage points | < 10 pp |
| F1 | >= 0.05 absolute (0.10 = clearly meaningful) | < 0.05 |
| High-severity recall | >= 5-10 percentage points | < 5 pp |
| Latency | >= 25-30% reduction | < 25% |
| Cost per review | >= 25-50% difference | < 25% |
| Engineer usefulness (1-5) | >= 0.5 points | < 0.3 |
| FPs per PR | >= 0.3-0.5 fewer | < 0.3 |

State in the evaluation: *"We treat differences below 5 F1 points or below 10 percentage points in precision/recall as inconclusive unless confidence intervals are tight and operational impact is clear."*

#### Sample size: how many PRs do we need?

**In plain terms:** The smaller the difference you're trying to detect, the more data you need. If Daydream is 30% better than Greptile, 20 PRs will make it obvious. If it's 5% better, you might need hundreds.

**For ML researchers:** Statistical power is the probability of detecting a real difference of a given size. At 80% power and alpha=0.05:

| Difference to detect | Comments needed per tool | Realistic PR count |
|---|---|---|
| 5 percentage points (precision/recall) | ~1,500+ | 500+ PRs |
| 10 percentage points | ~400 | 75-150 PRs |
| 15 percentage points | ~170 | 30-50 PRs |

**Paired design advantage:** Because both tools review the **same** PRs, PR-level variation (some PRs are harder, some easier) cancels out. This effectively doubles your statistical power compared to testing on different PR sets. The numbers above already reflect this advantage.

**What this means for the evaluation:**

| Evaluation scope | PR count | What you can claim |
|---|---|---|
| Pilot / exploratory | 20-30 | Directional signal only; "Daydream appears better on X" |
| Comparative (recommended v1) | 50-100 | Defensible per-metric comparison with CIs |
| Publication-grade | 200+ | Strong claims, multi-repo, category-level analysis |

#### Multiple comparisons: the "testing everything" problem

**In plain terms:** If you test 20 different metrics, one will look good purely by chance — just like if you flip 20 coins, one might come up heads 5 times in a row. The more things you measure, the more likely one is a false positive.

**For ML researchers:** With 15+ evaluation dimensions, the family-wise error rate inflates rapidly. Mitigation:

1. **Predefine primary metrics** before looking at data:
   - High-severity F1
   - Overall precision
   - Overall recall
   - Engineer-rated usefulness
2. **Treat all other dimensions as secondary/exploratory** — report them, but don't declare victory on them.
3. **Apply Holm-Bonferroni correction** to primary metrics if formal hypothesis testing is needed. This adjusts the significance threshold downward based on the number of tests performed.
4. **Prefer effect sizes and confidence intervals over p-values.** A p-value tells you "is there a difference?" An effect size with CI tells you "how big is the difference, and how sure are we?" — which is the more useful question for tool selection.

#### Putting it together: what to report

For each primary metric, report a table like this:

```
Metric              Daydream (95% CI)    Greptile (95% CI)    Difference (95% CI)    Significant?
Precision           68% [61, 75]         54% [47, 61]         +14 pp [+4, +24]       Yes
Recall              52% [44, 60]         38% [31, 45]         +14 pp [+3, +25]       Yes
High-sev recall     61% [50, 72]         42% [32, 53]         +19 pp [+4, +34]       Yes
F1                  0.59 [0.52, 0.66]    0.45 [0.39, 0.51]    +0.14 [+0.04, +0.24]   Yes
Usefulness (1-5)    3.8 [3.5, 4.1]       2.9 [2.6, 3.2]       +0.9 [+0.5, +1.3]      Yes
```

(Illustrative numbers, not real results.)

### Bias Mitigation Checklist

| Bias | Mitigation |
|---|---|
| Confirmation bias | Blind tool identity; pre-register criteria |
| Anchoring on tool output | Gold review before seeing AI comments |
| Cherry-picking PRs | Stratified random sample from fixed time window |
| Survivorship bias | Sample PRs first, then run tools; score silence |
| Brand/prestige bias | Normalize formatting; remove product phrasing |
| Evaluator fatigue | 20-30 min sessions; randomize order; track position effects |
| Incomplete gold set | Allow "new valid issue" adjudication; track gold expansion [1] |
| Tool config unfairness | Document config policy; both default or both tuned equally |

---

## Operational Metrics

Operational metrics informed by Atlassian RovoDev's production deployment metrics (code resolution rate, PR cycle time, human comment reduction) [9].

| Metric | Definition |
|---|---|
| Time-to-feedback | P50/P90 latency from PR open to first useful comment |
| Cost per review | API + compute + indexing cost per PR |
| Cost per true positive | (tool cost + estimated human triage cost) / TPs |
| Comment volume | Median comments per flagged PR |
| Noise rate | FPs / total findings = 1 - precision |
| FPs per PR | Absolute false-positive burden |
| Repeatability | Jaccard similarity of findings across repeated runs on same PR |

---

## Greptile Context

- Martian leaderboard: F1 49.9%, precision 71.7%, recall 38.3% — high precision, lower recall [1]. MorphLLM reports slightly different figures (66.2% precision, 40.4% recall, 50.2% F1) due to methodology differences [17].
- Architecture: codegraph + multi-modal retrieval (not simple RAG) + TREX (runs code in sandbox, generates logs/screenshots/API traces) [2][3].
- Known weakness: signal-to-noise historically poor (19% address rate, improved to 55% via embedding-based filtering using ChromaDB on Cloudflare) [2]. Non-deterministic. Pricing complaints on $1/review overage [2].
- Market context: best-in-class F1 on Martian is Gemini at 59.5% [1]. Category ARR ~$420M, 133% YoY growth, 44% of teams use AI code review on some PRs [18]. Entire category is mediocre in absolute terms.
- Independence principle: Greptile's core thesis is that "the tool that generates the code should not be the same tool that reviews it" [2].

---

## Additional Benchmarks (Optional for v1)

| Benchmark | Why | N | Source |
|---|---|---|---|
| SWR-Bench | Clean PRs enable false-positive/noise measurement | 1,000 PRs | [5] |
| c-CRAB | Tests actionability via executable tests — does the review guide a correct fix? | varies | [6] |
| Qodo PR-Review-Bench | Injected bugs in multi-language PRs | 100 PRs, 580 issues | [7] |
| AACR-Bench | Multilingual repo-level context; labels diff/file/repo context requirements | 200 PRs, 10 languages | [8] |
| CR-Bench / CR-Evaluator | Defect-focused review with signal-to-noise ratio metric | 584 tasks (174 verified) | [14] |
| SWE-PRBench | Matching human reviewer findings under controlled context settings | 350 PRs | [13] |
| CodeFuse-CR-Bench | End-to-end repo-level Python review with multi-dimensional scoring | 601 instances, 70 projects | [15] |
| SeRe | Security-specific code review evaluation | 6,732 instances, 5 languages | [11] |

---

## References

[1] Martian Code Review Bench. `codereview.withmartian.com`. Repo: `github.com/withmartian/code-review-benchmark`.

[2] Greptile Blog Posts (v2, v3, v4, TREX announcement). `greptile.com/blog`. 2024-2026.

[3] Hatchet Case Study: Greptile's Workflow Infrastructure. `hatchet.run` / Greptile engineering blog. 2025.

[4] Greptile AI Code Review Benchmark. `greptile.com/benchmarks`. 2025.

[5] SWR-Bench: "Benchmarking and Studying the LLM-based Code Review." arXiv:2509.01494. 2025.

[6] c-CRAB: "Code Review Agent Benchmark." arXiv:2603.23448. Repo: `github.com/c-CRAB-Benchmark`. 2026.

[7] Qodo PR-Review-Bench / Code Review Benchmark 1.0. HF dataset: `Qodo/PR-Review-Bench`. GitHub: `agentic-review-benchmarks`. 2025-2026.

[8] AACR-Bench: "Repository-level Automated Code Review." arXiv:2601.19494. Repo: `github.com/alibaba/aacr-bench`. HF: `Alibaba-Aone/aacr-bench`. 2026.

[9] Atlassian RovoDev: "LLM-based Code Reviewer in Enterprise Workflows." arXiv:2601.01129. 2026.

[10] DeepCRCEval: "Evaluating Code Review Comment Quality." arXiv:2412.18291. FASE 2025.

[11] SeRe: "Security-Related Code Review Dataset." arXiv:2601.01042. ICSE 2026.

[12] CodeReviewer: "Automating Code Review Activities by Large-Scale Pre-training." ESEC/FSE 2022. Repo: `github.com/microsoft/CodeBERT/tree/master/CodeReviewer`.

[13] SWE-PRBench: "Evaluating AI Code Review Against Human Reviewer Findings." arXiv:2603.26130. HF: `foundry-ai/swe-prbench`. 2026.

[14] CR-Bench / CR-Evaluator: "Benchmarking AI Code Review Agents for Defect Detection." arXiv:2603.11078. 2026.

[15] CodeFuse-CR-Bench: "End-to-End Repository-Level Code Review Evaluation." arXiv:2509.14856. 2025.

[16] CodeReviewQA: "Can LLMs Understand Code Review Comments?" arXiv:2503.16167. 2025.

[17] MorphLLM: "AI Code Review Tool Comparison." `morphllm.com`. 2026.

[18] IdeaPlan: "AI Code Review Market Report." `ideaplan.io`. 2026.

[19] Krippendorff, K. *Content Analysis: An Introduction to Its Methodology.* 4th ed. SAGE Publications, 2018. ISBN: 978-1506395653.

[20] SWE-bench: "Can Language Models Resolve Real-World GitHub Issues?" Jimenez et al. ICLR 2024. Repo: `github.com/swe-bench/SWE-bench`. `swebench.com`.
