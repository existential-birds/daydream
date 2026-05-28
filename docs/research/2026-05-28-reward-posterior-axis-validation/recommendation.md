# Recommendation — Posterior False-Positive Axis (PR #115)

**Date:** 2026-05-28
**Verdict on PR #115 in its current form:** **Do not merge as-is. Five corrections must land in this same PR before it can ship.** No follow-up issues, no `Owner: Phase X` deferrals — see CLAUDE.md §5 "Forbidden Rationalizations" #4.

The research (see `report.md`) does not invalidate having a posterior axis. It does invalidate:

1. One specific citation that is misquoted in two ways.
2. One safety-critical guarantee ("byte-identical on absent labels") whose meaning at the *training* layer is the opposite of what the PR description claims.
3. One runtime affordance (`weights=` at the call boundary) whose guard is docstring-only.
4. One numeric choice (`w_fp = 0.3`) presented as derived but actually borrowed by reference order-of-magnitude from a domain-mismatched paper.
5. One structural choice (subtractive scalarization of a constraint-style signal) that the field actively recommends against for posterior signals.

All five sit on the same surface (`reward.py`, `harvest.py`, the harvest schema, and the PR description). They belong together. Do them now.

## Required corrections

### C1. Correct the arXiv:2509.15557 citation

**What's wrong** (`findings/arxiv-2509-15557-and-weight-ordering.md`):

- "improper format … maximum penalty regardless of correctness" framing **is not in the paper**. The paper subtracts a fixed scalar `λ_s`; it does not gate.
- `w_b=1.0 > w_a=0.5 > w_s=0.3` is **one credit weight above two penalty weights of different kinds**, not a "smallest credit > any penalty" rule.
- Domain: USMLE-style medical MCQA on Llama 3.2-3B / Qwen2.5-3B; penalties target surface-form CoT-tag pathologies. Not transferable to a semantic posterior PR outcome.

**Fix:**

- In `reward.py` module docstring: drop the "improper format → maximum penalty regardless of correctness" attribution. Cite an actual gating precedent for the format gate (DeepSeek-R1's accuracy + format rule reward) or mark it as an internal design choice.
- In `reward.py` module docstring: rewrite the `w_fp = 0.3` defense as "matches the empirical penalty-weight magnitude in arXiv:2509.15557's medical-QA composite-verifiable-reward setup, borrowed by order-of-magnitude reference." Be explicit that the number is unjustified for this domain and pin it to a recalibration commitment that lands in this same PR.
- In PR description: same corrections.

### C2. Add a runtime guard on non-canonical weights

**What's wrong** (`findings/reward-versioning-pattern.md`):

`score_trajectory(weights=...)` accepts arbitrary `RewardWeights` and returns a float that looks identical to a canonical score. The only safeguard is a docstring sentence. lm-evaluation-harness, OpenAI Evals, and AlpacaEval all keep weights inside the versioned config precisely so non-default invocations cannot be confused with canonical scores. MLflow/W&B compensate by hashing actual params into the run identity.

**Fix:**

- Add `RewardWeights.is_default: bool` (default `False`; set `True` only on `DEFAULT_WEIGHTS`).
- Stamp the `RewardBreakdown.reward_version` (already exists) as `REWARD_VERSION` for defaults and `f"{REWARD_VERSION}+custom-{sha256(weights)[:8]}"` for overrides. Plumb it through to `reward_json`.
- Make `harvest.build_annotation` assert `breakdown.reward_version == REWARD_VERSION` before writing to the canonical column. Fail loudly on non-canonical writes.

### C3. Replace "byte-identical on absent labels" with the honest training contract

**What's wrong** (`findings/absent-label-bias.md`):

The PR description's "byte-identical on absent labels" is true at the *scoring* layer and structurally identical to **MNAR zero-imputation** at the *training* layer. When `--comment` is selectively invoked on PRs the user expects to land, `p(label_observed | trajectory_quality)` is correlated with quality. Berrevoets 2022: "no imputation at all also leads to biased estimates, as missingness determined by treatment introduces bias in covariates." InstructGPT BT loss, DeepSeek-V3 rule reward, and statistical rejection-sampling FT all *drop unlabeled rows from the loss*, never zero-substitute them into a composite.

**Fix:**

- Add `RewardBreakdown.has_posterior: bool` (or equivalent) — explicitly true when `pr_feedback` was supplied and mapped, false otherwise.
- Plumb that into the harvest schema so `payload.has_posterior` (or a new column / field name aligned with the existing schema) is queryable.
- Document the contract: aggregate-level use (advantage normalization, RM fitting, group-relative comparisons) **must** filter to `has_posterior=True` rows. Add a brief docstring section to `reward.py` and a one-line section to the README composite formula write-up.
- Audit current downstream callers in `daydream/training/` for any aggregate that would mix labeled and unlabeled rows. If any exist, fix them (or assert against the population mix) in this same PR.

### C4. Per-maintainer base-rate normalization for `fp_penalty`

**What's wrong** (`findings/posterior-accept-reject-mapping.md`):

Raw scalars deteriorate above 0.8 and post-hoc calibration recovers ~3 points on RewardBench. A maintainer with a 95% reject rate dominates gradients vs one with 80% accept. The DPO-with-ties literature models ties as a *probability* parameterized by a learnable tie-tendency `θ`, not a fixed midpoint.

**Fix (minimal viable, ships in this PR):**

- Add `outcome_prior: float | None = None` to `score_trajectory`. When provided, compute `fp_penalty = max(0.0, observed − prior)` — penalize only the surprise component.
- In `harvest.build_annotation`, look up the maintainer's rolling accept rate from the harvest corpus and pass it as `outcome_prior`. If insufficient history (define a threshold — e.g., < 10 prior labeled outcomes for this maintainer), pass `None` and document that the row is uncalibrated.
- Add `outcome_prior: float | None` and `outcome_prior_n: int` to `RewardBreakdown` so the audit trail is on the row.
- Replace the hard `contested → 0.5` midpoint with the empirical fraction of `contested → eventual-merge` from the maintainer's history (if available; else fall back to 0.5 with an audit flag).

### C5. Restructure the posterior axis to a separate field on RewardBreakdown OR commit a defensible scalarization

**What's wrong** (`findings/reduction-shape.md`):

Safe-RLHF documents "safety compensation" pathology for fixed-weight scalarization of constraint-style signals: violations on some inputs can be compensated by extreme over-caution on others. RULER beats hand-crafted multi-component rewards on 3/4 agentic tasks. A maintainer-rejection signal is conceptually a constraint, not a credit axis. The PR's defense — `w_fp` small + format gate strict — mitigates but does not eliminate this.

**Fix — pick one and commit:**

- **(Option A, structurally correct.)** Promote the posterior axis to a separate field on `RewardBreakdown` (e.g., `posterior_cost: float | None`) emitted alongside the intrinsic `composite`. Do not subtract it inside the composite. Downstream training callers receive both signals and the responsibility for combining them lives at the training-time aggregate (where IPW or constrained-RL can be applied properly). The composite stays a pure intrinsic score; the posterior axis is a sibling field.
- **(Option B, scalarized — only if A is rejected.)** Keep the subtractive shape, but justify the choice on the record (it's defensible as a small MORL scalarization, DeepSeek-R1 precedent) and explicitly document the residual risks (safety compensation, reward-hacking instability) and the mitigations relied on (`w_fp` small, format gate strict, `outcome_prior` debias from C4, drop-unlabeled-from-aggregates from C3).

**Recommend A.** It composes cleanly with C3 (drop-unlabeled-from-aggregates is exactly the kind of decision that belongs at the training-time aggregate, not in the reducer) and C4 (per-maintainer prior is more straightforward when the posterior is a separate field with its own audit metadata). It does not require restructuring `score_trajectory`'s caller contract — only adding a sibling field to the returned breakdown.

If you choose B, get explicit user sign-off — the research evidence pushes toward A.

## What this changes in the PR scope

The PR's stated motivation — "make reward react to maintainer outcomes; make weights overridable so calibration is a no-fork change" — is preserved. What changes:

- The reducer emits two scalars (`composite` for the intrinsic axis, `posterior_cost` for the posterior axis) rather than folding the second into the first.
- Override path is hash-stamped, not docstring-guarded.
- Posterior axis carries calibration metadata (`outcome_prior`, `outcome_prior_n`, `has_posterior`) so downstream training can make calibrated decisions without re-deriving them.
- Citation hygiene matches the actual referenced literature.

`REWARD_VERSION` bumps from `2026.05.27-1` to the next version (`2026.05.28-1` or similar) — the composite shape changed materially, more than the original PR's bump captured.

## Out-of-scope

Empirical recalibration of `LEN_TAU`, `LEN_SCALE`, `uncertain`, `w_fp` against the bootstrap corpus stays out of scope — that's a numerics task that needs labeled data this PR is helping to produce. The PR ships the *interface* and *contract* corrections; the *numbers* land later when the corpus has grown. This is the only honest deferral.

## Citation footnotes

See `report.md` Sources section for full citations. Each finding (`findings/<subtopic>.md`) carries its own footnoted citations.
