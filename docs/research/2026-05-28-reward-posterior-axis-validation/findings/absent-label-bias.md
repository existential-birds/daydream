---
subtopic: absent-label-bias
status: ok
searches_run: 7
fetches_run: 6
---

# Absent-Label Bias: Selection Bias When Posterior Feedback Is Missing

## Context

Daydream's `reward.py` leaves the posterior false-positive axis absent (`fp_penalty`
stays `None`) when no `pr_feedback` label is supplied: the result is a plain
`RewardBreakdown` with no posterior fields, and the composite is computed from the
intrinsic axes only (`composite = credit − w_len · ramp`). The posterior penalty is
never folded into the composite — when present it is a sibling field on
`PosteriorBreakdown`, never subtracted. So absent feedback is a *drop*, not a
zero-substitution: the unlabeled row simply carries no posterior axis. Trajectories
posted to a PR can carry labels (rejected / contested / accepted); trajectories from
`--review`, `--shallow`, or never-posted runs cannot. The "byte-identical on absent
label" guarantee thus *mixes two populations whose only difference is whether
labeling was possible*. Subtopic question: is this a benign back-compat shim, or
does it inject selection bias once the posterior axis feeds the policy gradient?

## 1. Selection-Bias Risk — Theoretically Real

Daydream already does the right thing *per row* — it drops the axis rather than
imputing zero. But the moment the posterior axis is aggregated or compared across
rows (advantage normalization, reward-model fitting, any pooled posterior statistic),
the labeled subset is no longer a random sample of the corpus, and the same
selection-bias failure modes that afflict **zero-imputation of a missing
feature/reward** apply to the *which-rows-have-the-axis* question:

- **Sparsity bias (Yi et al., 2019).** Zero imputation creates a "variable sparsity
  problem (VSP), which describes a phenomenon where the output of a predictive
  model largely varies with respect to the rate of missingness in the given input"
  and "adversarially affects the model performance." The model learns to read the
  *missingness rate* as signal rather than the underlying covariate.
  (`https://arxiv.org/abs/1906.00150`)

- **Treatment-effect bias (Berrevoets et al., 2022).** "Naively imputing all data
  leads to poor performing treatment effects models, as the act of imputation
  effectively removes information necessary to provide unbiased estimates. However,
  no imputation at all also leads to biased estimates, as missingness determined by
  treatment introduces bias in covariates." (`https://arxiv.org/abs/2202.02096`)
  Daydream's case is exactly this: the "treatment" (was-this-PR-posted) determines
  which trajectories can have labels — classic MNAR.

- **Off-policy evaluation under MNAR rewards (Saito et al., 2025).** When "rewards
  are missing not at random," naive estimators "may suffer from significant bias";
  the standard fix is dual propensity scoring on both logging and reward-observation
  probabilities. (`https://arxiv.org/abs/2502.08993`)

The Daydream contract avoids the worst form (it never imputes a zero penalty), but
the missingness is still MNAR: posting is *not* random with respect to the policy's
quality (a "good" review run is more likely to be posted in the first place). So any
posterior statistic pooled over the labeled subset is theoretically biased, not
benign — the bias has just moved from "what value gets imputed" to "which rows the
axis exists on."

## 2. Standard Mitigations

Three established families:

1. **Inverse-propensity weighting (IPW) / CausalRM.** Train propensity model
   `p(label_observed | trajectory)`; reweight labeled rows by `1/p`. CausalRM:
   "uses propensity scores — the probability of a user providing feedback for a
   given response — to reweight training samples … yields a loss function that
   eliminates user preference bias." (`https://arxiv.org/abs/2603.18736`)

2. **Doubly-robust estimators.** Combine IPW with an outcome model; bias survives
   only if *both* models are wrong. Standard in MNAR recommendation literature
   (Wang et al., ICML 2019).

3. **Drop-if-absent (rejection sampling on label availability).** Train the
   posterior-axis head *only* on the labeled subset; the intrinsic-only score is
   the reward for unlabeled rows. This is what InstructGPT-style RM training does
   implicitly — the RM is trained on pairs that *exist*; unpaired completions are
   simply not part of the loss.

4. **Missing-indicator method.** Add a binary "label_observed" feature so the
   model can condition on the missingness mechanism rather than confounding it
   with the signal value.

5. **Separate heads, joint loss.** Treat intrinsic and posterior as multi-task;
   each loss term is averaged over its own observed subset; no implicit zero.

## 3. Practice in RLHF / DPO / Recent RMs

- **InstructGPT / classical Bradley-Terry RM.** Training is on observed pairs
  only — there is no "score-zero for absent comparison." The RLHF book is explicit:
  "Training a preference reward model requires pairs of chosen and rejected
  completions." Unpaired data does not contribute to the BT loss; it does not
  contribute *zero* either. (`https://rlhfbook.com/c/05-reward-models`)

- **DeepSeek-V3 / R1.** Used rule-based reward (accuracy + format) with no neural
  RM. Where rules don't apply, the sample isn't graded on that axis — there is no
  structural-zero substitution into a composite. (`https://arxiv.org/abs/2412.19437`)

- **Rejection-sampling fine-tuning (RFT / Statistical RS).** Discards samples
  below the labeling threshold rather than imputing them at zero — the canonical
  "drop, don't zero" pattern. (`https://arxiv.org/abs/2309.06657`)

- **Reward Selection under Limited Feedback (Zhang et al., 2025).** Frames the
  *which-to-label* question explicitly: "Which samples should be labeled to
  maximize policy performance? We formalize this problem of reward selection for
  reinforcement learning from limited feedback (RLLF)." The framing presumes
  unlabeled rows are excluded, not zero-substituted. (`https://arxiv.org/abs/2510.00144`)

Daydream's per-row treatment (drop the axis, keep the intrinsic composite) is in
fact the same "drop, don't zero" pattern these systems use — the open question is
purely whether the labeled subset is then pooled into cross-row aggregates without
correcting for its non-random membership. The closest cautionary analogue —
zero-imputed implicit feedback in recsys — is precisely the case the propensity-score
/ doubly-robust literature exists to *fix*, and it is the failure mode Daydream
inherits if it pools the labeled subset as though it were representative.

## 4. Concrete Recommendation for Daydream

Ranked by rigor vs. implementation cost:

**(b) Exclude unlabeled rows from posterior-axis aggregates (recommended baseline).**
The per-row composite is already intrinsic-only when the label is absent (the
posterior axis is simply dropped), so the remaining work is to flag the row as
"intrinsic-only" and *exclude it from any aggregate that compares to labeled rows*
(e.g., advantage normalization, reward-model fitting, pooled posterior statistics).
This matches InstructGPT practice and removes the bias entirely. Cheap: one extra
column in the training corpus schema.

**(d) Add a missingness indicator + IPW reweight (if mixed training is required).**
If labeled coverage is < 50%, fit a logistic `p(label_observed | features
available at capture time)`; weight the posterior-axis contribution of labeled
rows by `1/p`. Justified when posting probability correlates with policy quality
(which it almost certainly does — daydream is run in `--comment` mode more often
on PRs the user expects to land). (`https://arxiv.org/abs/2603.18736`)

**(c) Separate heads per signal (cleanest long-term).** Intrinsic head trained on
all rows; posterior head trained on labeled rows only. Compose at inference, not
in the training reward. Eliminates the mixed-population question entirely.

**(a) Pool the labeled subset as-is (current Daydream default if no exclusion is
applied) — NOT recommended.** Daydream already drops the axis per row rather than
imputing zero, but pooling the labeled rows into a shared aggregate without an
exclusion flag is "benign" only if `p(label_observed) ⊥ trajectory_quality`, which
is empirically false for code review (`--comment` is selectively run on PRs the user
cares about).

**Backfill is a complement, not a fix.** Re-running daydream on already-posted
PRs to harvest labels reduces the *fraction* of absent labels but does not
correct the *mechanism* — non-posted PRs remain systematically excluded.

## 5. Verdict

`contested` — The "byte-identical" guarantee is a clean back-compat contract, and
the per-row treatment is already the preferred "drop, don't zero" form: the composite
stays intrinsic-only and no zero penalty is imputed. The residual risk is at the
aggregate level — the labeled subset on which the posterior axis exists is MNAR.
Theory (Berrevoets 2022, Yi 2019, Saito 2025) and RLHF practice (InstructGPT,
DPO, RFT) both prefer "drop or reweight" over treating an unrepresentative labeled
subset as representative. The bias is small if posting is near-random w.r.t.
trajectory quality and large if not. For Daydream's use case (selectively-posted PR
reviews), it is more likely large. Recommend the exclude-from-aggregates mitigation
as a near-term fix; consider IPW or separate heads if the labeled fraction stays low.

## 6. Strongest Single Citation

Berrevoets, J. et al. (2022), "To Impute or not to Impute? Missing Data in
Treatment Effect Estimation":

> "Naively imputing all data leads to poor performing treatment effects models,
> as the act of imputation effectively removes information necessary to provide
> unbiased estimates. However, no imputation at all also leads to biased
> estimates, as missingness determined by treatment introduces bias in
> covariates."

(`https://arxiv.org/abs/2202.02096`)

Direct mapping: "treatment" → was-this-trajectory-posted; "covariates" →
intrinsic features; the impute-vs-drop choice IS the bias-vs-information tradeoff
Daydream faces. It already drops rather than imputes per row; the same tradeoff
re-emerges whenever the dropped-from rows are pooled with the labeled ones.

## Sources

- [Why Not to Use Zero Imputation? — Yi et al., NeurIPS 2019](https://arxiv.org/abs/1906.00150)
- [To Impute or not to Impute? — Berrevoets et al., 2022](https://arxiv.org/abs/2202.02096)
- [CausalRM — propensity-score RM debiasing](https://arxiv.org/abs/2603.18736)
- [Off-Policy Evaluation with MNAR Rewards — Saito et al., 2025](https://arxiv.org/abs/2502.08993)
- [Reward Selection under Limited Feedback (RLLF)](https://arxiv.org/abs/2510.00144)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Statistical Rejection Sampling for Preference Optimization](https://arxiv.org/abs/2309.06657)
- [RLHF Book — Reward Models chapter](https://rlhfbook.com/c/05-reward-models)
- [Doubly Robust Joint Learning for MNAR Recommendation — Wang et al., ICML 2019](http://proceedings.mlr.press/v97/wang19n/wang19n.pdf)
