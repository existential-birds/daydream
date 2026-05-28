# Research Plan — Posterior False-Positive Axis Validation

**Date:** 2026-05-28
**Branch:** `feat/reward-posterior-axis`
**Caller:** User-invoked, post-PR-description review

## Research question

Validate or challenge the design choices in `daydream/training/reward.py`'s
posterior false-positive penalty axis. Specifically: is the reduction shape
(subtractive term in a clipped weighted mean) right; is `w_fp = 0.3 < w_grounding = 0.4`
defensible; is the `rejected/contested/accepted → 1.0/0.5/0.0` mapping right;
does "byte-identical on absent labels" introduce training bias; and is the
`REWARD_VERSION` + tunable `RewardWeights` versioning pattern accepted practice?

## Subtopics

1. **Reduction shape: subtractive penalty vs gates, separate heads, GRPO advantages.**
   Establish what RLHF/RLAIF practice prefers for combining a delayed posterior
   correctness signal with capture-time intrinsic credits in code-review style tasks.
   Sources to seek: GRPO papers (DeepSeek-Math, DeepSeek-R1), constrained-RL with
   Lagrangian penalties, multi-objective RLHF (Safe-RLHF, Rewarded Soups), RULER /
   judge-based reward construction.

2. **Verify arXiv:2509.15557 and weight-ordering claim.** The PR description and
   reward.py docstring cite arXiv:2509.15557 for a specific weight ordering
   (`w_b=1.0 > w_a=0.5 > w_s=0.3`). Establish: does this paper exist; does it say
   what's claimed; what is the actual recommended ordering of penalty vs credit
   weights in its composite reward. If it doesn't exist or doesn't say this, flag it.

3. **Maintainer accept/reject as posterior reward — hard-coded vs calibrated mapping.**
   Surface what is published or shipped on using PR-merge / PR-accept signals as a
   reward in code-review agent training (Sweep, CodeRabbit-style flows, Cursor's
   composer, AlphaCode/AlphaEvolve, SWE-bench Verified leaderboards). Specifically:
   do they use hard ternary {1, 0.5, 0} or do they calibrate against base rates?
   Is there published research on the prevalence and treatment of "contested"
   reviews as a middle class?

4. **Bias from "byte-identical on absent labels."** Whether structural-zero treatment
   of a missing posterior axis introduces selection bias when not-all trajectories
   get labeled (only PR-posted ones do), and what off-policy / delayed-reward
   correction techniques are standard. Sources: delayed-reward RL, off-policy
   correction (Munos / Retrace, importance-sampling correction), selection bias
   in learned reward models.

5. **`REWARD_VERSION` + tunable weights pattern.** Whether stamping a version
   string to a default-config and letting callers override at the function boundary
   is accepted practice in eval / reward frameworks (lm-evaluation-harness, HELM,
   AlpacaEval, OpenAI evals, RewardBench). Whether the right pattern would instead
   pin the version to the weights tuple (so any override invalidates the stamp).

## What each subtopic should establish

For each subtopic, the findings file should answer:

- **Established norms.** What does the published literature or production practice say?
- **Direct evidence for/against the PR's choice.** Concrete quotes with citations.
- **Strongest alternative.** If there's a competing approach, name it and cite it.
- **Verdict.** "Approach is supported", "Approach is contested", "Approach is unsupported",
  or "Insufficient evidence" — chosen on evidence, not vibe.

## Synthesis approach

The synthesizer will produce `report.md` with:

- **TL;DR** — 3-5 bullets, one per subtopic verdict, with the single most-load-bearing citation.
- **Findings** — organized by subtopic; every claim footnoted; the arXiv:2509.15557
  verification gets its own callout because the PR description leans on it.
- **Gaps & Limitations** — including any subagent that returned `status: empty` or `failed`.
- **Sources** — numbered bibliography.

A separate **`recommendation.md`** sibling file (outside the standard report shape) will
distill the report into "keep / tune / restructure" guidance and a concrete change list,
so the user can act on it without re-reading the full report.

## Budget

- 5 parallel subagents (one per subtopic) — exceeds the default-3 cap because all five
  questions are independent and the user explicitly asked for parallel fan-out.
- 4-6 web searches per subagent.
- 1 verification fetch for the arXiv reference (subtopic 2).

## Output paths

- `docs/research/2026-05-28-reward-posterior-axis-validation/plan.md` (this file)
- `.../findings/reduction-shape.md`
- `.../findings/arxiv-2509-15557-and-weight-ordering.md`
- `.../findings/posterior-accept-reject-mapping.md`
- `.../findings/absent-label-bias.md`
- `.../findings/reward-versioning-pattern.md`
- `.../report.md`
- `.../recommendation.md`
