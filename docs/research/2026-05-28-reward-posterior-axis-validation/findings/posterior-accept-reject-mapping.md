---
subtopic: posterior-accept-reject-mapping
status: ok
searches_run: 7
fetches_run: 4
---

# Posterior accept/reject mapping — hard-coded ternary vs calibrated

Daydream's `reward.py` uses a fixed ternary mapping: `accepted=0.0`, `contested=0.5`, `rejected=1.0`. This maps to the raw false-positive penalty; `w_fp=0.3` is a documented training-time combination weight that is **not** applied inside the intrinsic composite (the posterior is a sibling axis, not a subtracted term). This note checks that scheme against published practice.

## 1. Practice in code-review agent training

Most disclosed code-agent RL pipelines do **not** use maintainer accept/reject as the reward at all — they use **test-pass outcomes** (an Outcome Reward Model). DeepSWE is the canonical example:

> "1 — LLM's generated patch passes a selected sample of tests (Pass2Pass and Fail2Pass) within a time limit … 0 — We assign no reward if the LLM's code fails on at least one test case or times out." ([DeepSWE blog](https://www.together.ai/blog/deepswe))

SWE-Gym / Kimi-Dev / R2E-Gym all follow the same sparse-binary test-based ORM. CodeReviewer (Li et al., 2022) pre-trains on review comments and quality estimation but does **not** publicly describe a per-reviewer accept/reject reward calibration — its tasks are *generation* and *quality estimation*, not RL from maintainer outcomes ([arXiv:2203.09095](https://arxiv.org/abs/2203.09095)). CodeRabbit advertises "learning from team feedback" but discloses no scheme ([docs.coderabbit.ai](https://docs.coderabbit.ai/)).

**Implication:** there is no published reference mapping to anchor Daydream's `1.0/0.5/0.0`. The ternary scheme is a design choice, not a borrowed convention.

## 2. Hard ternary vs calibrated mapping

The closest published guidance is the RLHF reward-shaping/calibration literature. Post-hoc reward calibration finds substantial bias in raw reward scalars:

> "preference scores initially calibrate well with win rates but deteriorate when scores exceed 0.8" — and post-hoc calibration using "local average reward to estimate bias terms" gives "3.11 average performance gain across 33 reward models on RewardBench" ([Reward Shaping to Mitigate Reward Hacking, arXiv:2502.18770](https://arxiv.org/html/2502.18770v3); [Post-hoc Reward Calibration, OpenReview](https://openreview.net/pdf?id=Iu8RytBaji)).

This is evidence **against** raw fixed scalars and **for** per-rater / base-rate normalization — but it concerns reward-model outputs, not maintainer label codes. No source directly endorses or rejects `{0, 0.5, 1}` for `{accept, contested, reject}`.

## 3. "Contested" as a middle class

This is the cleanest finding. DPO/Bradley-Terry literature directly addresses ambiguous middle labels:

> "many potentially useful, and expensively collected, preference judgments are discarded simply because they are ties" — and when ties are naively added to standard DPO, "task performance degrades significantly … the frontier shifts down and to the left." Modeling ties explicitly via Rao-Kupper or Davidson extensions gives "regularization without performance degradation" ([Extending DPO to Accommodate Ties, arXiv:2409.17431](https://arxiv.org/html/2409.17431)).

> "humans often perceive two responses as ties when their rewards have very little difference" and ignoring ties "causes significant bias in preference strength measurement" ([Reward Learning From Preference With Ties, arXiv:2410.05328](https://ar5iv.labs.arxiv.org/html/2410.05328)).

The literature **supports keeping a tie/contested class** rather than dropping it — but it models the tie as a *probability* over outcomes parameterized by θ, **not** as a fixed midpoint scalar. Daydream's `0.5` is a defensible point estimate of the tie outcome's penalty mass (Rao-Kupper assumes humans randomly label ties 50-50), but it skips the tie-tendency parameter θ.

## 4. Base-rate normalization

No retrieved source describes per-maintainer or per-repo base-rate normalization of accept/reject as a reward. The closest analogue is peer-review calibration:

> "papers receiving identical quality in a clean setting may obtain different acceptance rates when reviews are noisy" ([Calibrating Cheap Signals in Peer Review, arXiv:2312.07269](https://arxiv.org/pdf/2312.07269)).

…and reviewer-experience-weighted losses in code-review comment generation:

> "experience-aware loss functions that weight the model's loss function based on reviewers' project ownership … allows experienced reviewers' code reviews to yield larger influence" ([Leveraging Reviewer Experience, arXiv:2409.10959](https://arxiv.org/pdf/2409.10959)).

Both establish that **per-reviewer weighting exists in the SE literature**, but neither prescribes normalizing the accept/reject reward by the maintainer's base accept rate. This is a real gap in the unnormalized scheme: a 95%-reject maintainer's signal will dominate gradients vs an 80%-accept one.

## 5. Verdict

**`insufficient-evidence` leaning `contested`.**

- *Supported* aspects: keeping `contested` as its own class (not dropping it) is endorsed by the DPO-with-ties literature.
- *Contested* aspects: (a) the specific `1.0 / 0.5 / 0.0` scalars are not anchored in any cited published scheme — the DPO-ties work models ties as a *probability* with a learnable tendency parameter θ, not a fixed midpoint penalty; (b) absence of per-maintainer base-rate normalization is a documented failure mode in the RLHF calibration literature (raw scalars deteriorate, post-hoc calibration recovers ~3 points on RewardBench).
- *Unsupported* aspect of the critique: there is no published evidence that a learned mapping outperforms a hand-set ternary mapping *for the specific case of bot-PR-comment outcomes*; everyone in disclosed code-agent RL uses test-pass signals instead.

**Practical implication:** the ternary is not wrong, but it's a stopgap. The two highest-leverage upgrades would be (a) per-maintainer base-rate normalization (subtract or divide by the maintainer's empirical accept rate when deriving the posterior axis) and (b) replacing `0.5` for `contested` with the empirical fraction of contested → eventual-merge in the harvested corpus.

## 6. Strongest single citation

> "many potentially useful, and expensively collected, preference judgments are discarded simply because they are ties … explicitly labeled ties can be added to datasets for these DPO variants without the degradation in task performance observed when tied pairs are presented to standard DPO."
> — [On Extending Direct Preference Optimization to Accommodate Ties, arXiv:2409.17431](https://arxiv.org/html/2409.17431)

This directly supports Daydream's choice to keep `contested` as a real class rather than drop it — while implicitly warning that *how* the tie is encoded matters more than whether it appears.
