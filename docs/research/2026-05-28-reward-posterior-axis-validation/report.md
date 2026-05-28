# Report — Posterior False-Positive Axis Validation

**Date:** 2026-05-28
**PR under review:** #115 — `feat(training): add posterior reject-penalty axis to reward (#88)`
**Branch:** `feat/reward-posterior-axis`

## TL;DR

- **The arXiv:2509.15557 citation in the PR description and `reward.py` docstring is misquoted.** The paper exists (Tarek & Beheshti, 2025) and the numeric values `w_b=1.0, w_a=0.5, w_s=0.3` are verbatim — but (i) the "improper format → maximum penalty regardless of correctness" framing is **not in the paper** (the paper subtracts penalties, it does not gate); (ii) the ordering pairs ONE credit weight with TWO penalty weights of different kinds, **not** a "smallest credit > penalty" rule; (iii) the domain is medical USMLE-QA on 3B chat models with surface-form CoT-tag penalties — generalizing to posterior PR-outcome penalties in code review is not defensible from this paper alone.[^arxiv]
- **The subtractive-composite shape is *defensible* for an intrinsic axis but the *wrong shape* for a posterior maintainer-rejection signal.** RULER head-to-head beats hand-crafted multi-component rewards on 3/4 agentic tasks[^ruler]; Safe-RLHF explicitly argues that constraint-style signals (reject = "must not violate") belong in a separate cost head with a Lagrange multiplier, and that fixed-weight scalarization causes "safety compensation" pathology.[^saferlhf]
- **Treating "rejected = 1.0 / contested = 0.5 / accepted = 0.0" is structurally OK but uncalibrated.** Keeping "contested" as its own class is supported by the DPO-with-ties literature[^dpoties], but the specific `0.5` midpoint is a stopgap; per-maintainer / per-repo base-rate normalization is a documented mitigation against raw-scalar deterioration in reward calibration[^calibration].
- **"Byte-identical on absent labels" is textbook MNAR zero-imputation and biases the policy gradient.** When `--comment` is run selectively on PRs the user expects to land, posting is *not* random with respect to trajectory quality. Theory (Berrevoets 2022[^berrevoets]; Yi 2019[^yi]) and RLHF practice (InstructGPT BT loss[^rlhfbook]; rejection-sampling FT[^rft]) both prefer "drop or reweight" over "zero-substitute."
- **`REWARD_VERSION` as a default-pin is supported; the call-boundary override guarded only by a docstring is not.** lm-evaluation-harness and OpenAI Evals[^openai-evals] both put weights inside the versioned config so any change forces a rename; MLflow/W&B compensate by hashing actual params into run identity. Daydream's hybrid is the weakest combination of the two traditions.

**Overall: keep the axis, fix the citation, and ship one runtime guard. Defer the structural changes (separate head, IPW, RULER) to a tracked follow-up so this PR can land.**

## Findings

### Subtopic 1 — Reduction shape: subtractive penalty vs. gates, separate heads, GRPO advantages

DeepSeek-R1's rule-based reward composition is the canonical minimal-scalar baseline: "Each response receives a scalar reward based on factors like accuracy, formatting, and language consistency"[^deepseek-r1]. Daydream's clipped-subtractive shape is in the same family. Two design principles in the published shaping literature support the choice: rewards should be *bounded*, and they should exhibit "rapid initial growth followed by gradual convergence"[^shaping]. The `clip(..., 0, 1)` floor satisfies the first; the small `w_fp = 0.3` keeps the second.

The counter-evidence is concentrated on the *posterior* axis specifically:

- **RULER beats hand-crafted rewards in 3/4 agentic tasks**[^ruler] and reduces implementation time by 2-3× — a direct head-to-head substitute for hand-weighted axes.
- **Safe-RLHF documents "safety compensation" pathology** where fixed-weight scalarization lets violations on some inputs be masked by over-caution on others[^saferlhf]. A `w_fp = 0.3` subtraction is exactly this shape; an FP-heavy slice of the corpus can be compensated by inflated grounding scores elsewhere.
- **GRPO pushes KL into the loss rather than the reward**[^grpo] — the canonical statement that "shape less, regularize at the trainer more."

**Verdict: contested.** The shape is defensible as a small MORL scalarization; for the *posterior* axis specifically, the better pattern is a separate cost head with a Lagrangian or a single LLM-judge ranking over groups.

### Subtopic 2 — arXiv:2509.15557 verification

**Paper exists. Citation is misquoted.**

| Claim in PR / docstring | What paper actually says |
| --- | --- |
| "improper format … maximum penalty regardless of correctness" justifies the format gate flooring composite to 0 | Not in paper. Paper subtracts a fixed scalar `λ_s` when preamble word count exceeds `τ_preamble`. **A correct answer with a format violation still earns positive `R_binary` credit.** No gate, no floor. |
| `w_b=1.0 > w_a=0.5 > w_s=0.3` ordering establishes a "penalty weight strictly below smallest credit weight" rule | Paper has **one** credit weight and **two** penalty weights of different kinds (`P_answer` = answer leak, `P_structural` = preamble length). The ordering is `1 credit > 2 different penalties`, not "smallest credit > any penalty." |
| Applicable to code-review reward design | Domain is medical USMLE-style multiple-choice QA on Llama 3.2-3B / Qwen2.5-3B. Penalties target surface-form CoT-tag pathologies (cosine similarity to leak phrases, preamble word count). Mechanics do not map onto a semantic posterior PR outcome. |

The paper also offers no theoretical or ablation-based justification for the 1.0/0.5/0.3 ordering — they are presented as tuned hyperparameters. No broader survey of composite-reward design papers (ENCORE, RLMR, PPO-driven adaptive filtering) corroborates the "penalty below smallest credit weight" rule as a published norm[^rlmr].

**Verdict: citation-misquoted with domain-mismatch overlay.** Either drop the citation or rewrite the docstring to (a) describe the paper accurately as a *subtractive* composite (not a gate), (b) acknowledge the weight ordering is a tuned hyperparameter choice in a medical-QA paper, and (c) cite Daydream-specific reasoning or an ablation for why `w_fp = 0.3` here.

### Subtopic 3 — Posterior accept/reject mapping

**No published reference mapping for maintainer accept/reject as RL reward exists.** Most disclosed code-agent RL pipelines use **test-pass outcomes** as the ORM, not maintainer outcome:

> "1 — LLM's generated patch passes a selected sample of tests (Pass2Pass and Fail2Pass) within a time limit … 0 — We assign no reward if the LLM's code fails on at least one test case or times out." (DeepSWE)[^deepswe]

CodeReviewer's per-reviewer experience-weighted loss[^reviewerexp] and Calibrating Cheap Signals in Peer Review[^cheap-signals] both establish that **per-reviewer weighting exists in the SE literature**, but neither prescribes normalizing accept/reject reward by the maintainer's empirical accept rate.

**Where the literature does speak directly is on "contested" as a middle class.** DPO-with-ties is unambiguous:

> "many potentially useful, and expensively collected, preference judgments are discarded simply because they are ties … explicitly labeled ties can be added to datasets for these DPO variants without the degradation in task performance observed when tied pairs are presented to standard DPO."[^dpoties]

This *supports* keeping `contested` as its own class but flags that the encoding matters: DPO-ties models ties as a *probability* parameterized by a learnable tie-tendency `θ`, not a fixed midpoint.

**Verdict: insufficient-evidence leaning contested.** Keeping `contested` is supported; the specific `1.0/0.5/0.0` scalars are not anchored; the missing per-maintainer base-rate normalization is a documented failure mode in the calibration literature ("preference scores deteriorate when scores exceed 0.8 … post-hoc calibration gives 3.11 average performance gain across 33 reward models on RewardBench")[^calibration].

### Subtopic 4 — "Byte-identical on absent labels" — selection bias

Daydream's "byte-identical when no `pr_feedback`" guarantee is structurally identical to **zero-imputation of a missing reward feature** under MNAR ("missing not at random"). This is a known anti-pattern:

> "Naively imputing all data leads to poor performing treatment effects models, as the act of imputation effectively removes information necessary to provide unbiased estimates. However, no imputation at all also leads to biased estimates, as missingness determined by treatment introduces bias in covariates."[^berrevoets]

The "treatment" maps directly: *was-this-trajectory-posted*. Daydream is run in `--comment` mode selectively on PRs the user expects to land, so `p(label_observed | trajectory_quality)` is correlated with quality — exactly the case the propensity-score / doubly-robust literature exists to fix[^causalrm][^mnar-recsys].

Standard RLHF practice never mixes a structural zero into a composite:

- **InstructGPT / Bradley-Terry RM** trains on observed pairs only — "unpaired data does not contribute to the BT loss; it does not contribute zero either"[^rlhfbook].
- **DeepSeek-V3 / R1** use rule-based reward where rules apply; samples without an applicable rule are not graded on that axis, not zero-substituted[^deepseek-v3].
- **Rejection-sampling fine-tuning** drops samples below the labeling threshold — the canonical "drop, don't zero" pattern[^rft].

**Verdict: contested.** The "byte-identical" guarantee is a clean back-compat contract for *scoring*. As a *training* signal it is biased. The fix is small: flag the row as "intrinsic-only" and exclude it from any aggregate that compares against labeled rows.

### Subtopic 5 — `REWARD_VERSION` + tunable weights pattern

The **default-pin half** of the pattern is well-supported:

> "In general, running the same eval name against the same model should always give similar results so that others can reproduce it. Therefore, when you change your eval, you should bump the version." (OpenAI Evals)[^openai-evals]

> "if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. Task versions start at 0, and each time a breaking change is made, the version is incremented by one." (lm-evaluation-harness)[^lm-eval-harness]

The **call-boundary override gated only by a docstring** is not standard. lm-eval-harness, OpenAI Evals, AlpacaEval all keep weights inside the versioned config so any change forces a rename (new YAML, new annotator config). MLflow / W&B[^mlflow] compensate for runtime parameter variation by hashing actual params into the run identity. Daydream's hybrid (runtime overrides + single human stamp + docstring guard) is the weakest combination.

Concrete failure mode: nothing on the produced score (a float) carries the weights that produced it. The storage layer (corpus / trajectory / harvest output) has no in-band signal it can refuse on. A future sensitivity sweep can trivially write the override result into the same column as canonical scores.

**Verdict: contested.** Cheap fix: have `score_trajectory` return a `(score, version_stamp)` pair where the stamp is `REWARD_VERSION` for defaults and `f"{REWARD_VERSION}+custom-{hash}"` for overrides. The storage layer can then assert canonical == default at write time.

## Gaps & Limitations

- **No primary-source data on actual maintainer accept rates.** Per-maintainer base-rate normalization is recommended, but we have no Daydream-internal numbers on how skewed those rates actually are. The bias magnitude in (Subtopic 4) is therefore qualitative.
- **CodeRabbit / Cursor composer / Copilot Workspace training details are proprietary.** Subtopic 3 surfaced no disclosed mapping from those systems to triangulate against.
- **RULER (Subtopic 1) is benchmarked on 4 tasks**; none of them are code review. The "RULER beats hand-crafted" claim transfers by structural similarity, not direct evidence.
- **The KD2 drown-out guard** referenced in the PR description is cited in the docstring as a self-evident property of the ordering, not against an external source. No external source was found that uses this term; it appears to be Daydream-internal terminology.

## Sources

[^arxiv]: Tarek, M.F.B. & Beheshti, R., *Reward Hacking Mitigation using Verifiable Composite Rewards*, arXiv:2509.15557, September 2025. <https://arxiv.org/abs/2509.15557> / <https://arxiv.org/html/2509.15557v1>. Composite reward: "R_total(g) = w_b R_binary(g) - w_a P_answer(g) - w_s P_structural(g)"; hyperparameters: "w_b=1.0, w_a=0.5, w_s=0.3"; structural penalty: "P_structural(g) = {λ_s if |T_preamble| > τ_preamble; 0 otherwise}." Task: USMLE-style medical QA on Llama 3.2-3B-Instruct / Qwen2.5-3B-Instruct.

[^ruler]: OpenPipe ART, *RULER: Relative Universal LLM-Elicited Rewards*. <https://art.openpipe.ai/fundamentals/ruler>. "In 3 out of 4 tasks, models trained with RULER slightly outperform those trained with hand-crafted reward functions."

[^saferlhf]: Dai et al., *Safe RLHF: Safe Reinforcement Learning from Human Feedback*, ICLR 2024. <https://proceedings.iclr.cc/paper_files/paper/2024/file/dd1577afd396928ed64216f3f1fd5556-Paper-Conference.pdf>. "This decoupling is essential to avoid policy collapse to trivial refusals (when cost is overemphasized) or unsafe reward maximization (when reward dominates) … A notable pathology in average-constrained Safe RLHF is 'safety compensation,' where violations for some inputs can be compensated by extreme over-caution for others."

[^deepseek-r1]: DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948. <https://arxiv.org/pdf/2501.12948>. "Each response receives a scalar reward based on factors like accuracy, formatting, and language consistency … We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero, because we find that the neural reward model may suffer from reward hacking."

[^shaping]: *Reward Shaping to Mitigate Reward Hacking in RLHF*, arXiv:2502.18770. <https://arxiv.org/html/2502.18770v3>. "two key design principles: the RL reward should be bounded, and the RL reward benefits from rapid initial growth followed by gradual convergence."

[^grpo]: Wolfe, C.R., *Group Relative Policy Optimization (GRPO)*. <https://cameronrwolfe.substack.com/p/grpo>. "rewards are normalized by subtracting the group mean and dividing by the group standard deviation, yielding a relative (or whitened) advantage for each output … The KL penalty is applied directly in the loss rather than shaping the reward."

[^dpoties]: *On Extending Direct Preference Optimization to Accommodate Ties*, arXiv:2409.17431. <https://arxiv.org/html/2409.17431>. "many potentially useful, and expensively collected, preference judgments are discarded simply because they are ties … explicitly labeled ties can be added to datasets for these DPO variants without the degradation in task performance observed when tied pairs are presented to standard DPO."

[^deepswe]: Together AI, *DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL*. <https://www.together.ai/blog/deepswe>. "1 — LLM's generated patch passes a selected sample of tests (Pass2Pass and Fail2Pass) within a time limit … 0 — We assign no reward if the LLM's code fails on at least one test case or times out."

[^reviewerexp]: *Leveraging Reviewer Experience in Code Review Comment Generation*, arXiv:2409.10959. <https://arxiv.org/pdf/2409.10959>. "experience-aware loss functions that weight the model's loss function based on reviewers' project ownership … allows experienced reviewers' code reviews to yield larger influence."

[^cheap-signals]: *Calibrating "Cheap Signals" in Peer Review without a Prior*, arXiv:2312.07269. <https://arxiv.org/pdf/2312.07269>. "papers receiving identical quality in a clean setting may obtain different acceptance rates when reviews are noisy."

[^calibration]: *Post-hoc Reward Calibration: A Case Study on Length Bias*, OpenReview. <https://openreview.net/pdf?id=Iu8RytBaji>. "preference scores initially calibrate well with win rates but deteriorate when scores exceed 0.8 … 3.11 average performance gain across 33 reward models on RewardBench."

[^berrevoets]: Berrevoets, J. et al., *To Impute or not to Impute? Missing Data in Treatment Effect Estimation*, 2022. <https://arxiv.org/abs/2202.02096>. "Naively imputing all data leads to poor performing treatment effects models … no imputation at all also leads to biased estimates, as missingness determined by treatment introduces bias in covariates."

[^yi]: Yi et al., *Why Not to Use Zero Imputation? Correcting Sparsity Bias in Training Neural Networks*, NeurIPS 2019. <https://arxiv.org/abs/1906.00150>. Documents the "variable sparsity problem" caused by zero-imputation.

[^causalrm]: *CausalRM: Disentangling Confounders for Robust Reward Modeling*, arXiv:2603.18736. <https://arxiv.org/abs/2603.18736>. "uses propensity scores — the probability of a user providing feedback for a given response — to reweight training samples … yields a loss function that eliminates user preference bias."

[^mnar-recsys]: Saito et al., *Off-Policy Evaluation with MNAR Rewards*, arXiv:2502.08993. <https://arxiv.org/abs/2502.08993>. "rewards are missing not at random … naive estimators may suffer from significant bias"; the standard fix is dual propensity scoring.

[^rlhfbook]: Lambert, N., *RLHF Book — Chapter 5: Reward Models*. <https://rlhfbook.com/c/05-reward-models>. "Training a preference reward model requires pairs of chosen and rejected completions."

[^deepseek-v3]: DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437. <https://arxiv.org/abs/2412.19437>. Rule-based reward usage; no structural-zero substitution.

[^rft]: *Statistical Rejection Sampling Improves Preference Optimization*, arXiv:2309.06657. <https://arxiv.org/abs/2309.06657>. Discards samples rather than imputing them at zero — the canonical "drop, don't zero" pattern.

[^openai-evals]: OpenAI, *Building an Eval*. <https://github.com/openai/evals/blob/main/docs/build-eval.md>. "In general, running the same eval name against the same model should always give similar results so that others can reproduce it. Therefore, when you change your eval, you should bump the version."

[^lm-eval-harness]: EleutherAI, *lm-evaluation-harness Task Guide*. <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md>. "if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons."

[^mlflow]: MLflow Documentation, *Tracking Experiments*. Convention: hash actual params into run identity.

[^rlmr]: *RLMR: Reinforcement Learning with Mixed Rewards*, arXiv:2508.18642. <https://arxiv.org/html/2508.18642v1>. "writing quality scores and constraint verification signals operate on different scales and distributions, making it difficult to determine appropriate weighting coefficients" — i.e., fixed-weight ordering rules are explicitly discouraged.
