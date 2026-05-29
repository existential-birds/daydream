---
subtopic: reduction-shape
status: ok
searches_run: 6
fetches_run: 3
---

# Reduction shape: subtractive penalty vs. gates, separate heads, GRPO advantages

## Scope

The Daydream PR `feat/reward-posterior-axis` reduces multi-axis reward to one scalar via
`composite = round(clip(credit − w_len·len_norm, 0, 1), 4)` with `format_valid=False` flooring to 0, where `credit` is the renormalized weighted mean over the *present* credit axes only. The posterior false-positive penalty is **not** folded into this composite: it is carried as a sibling axis (`posterior_cost` on `PosteriorBreakdown`) rather than subtracted inside the clipped mean. This document evaluates that shape — and the question of where the posterior axis *should* live — against (a) GRPO group-relative advantages, (b) Safe-RLHF Lagrangian constraints, (c) Rewarded Soups / MORLHF linear scalarization, and (d) RULER / generalist judge composition.

## 1. Established norm

Modern RLHF/RLAIF for agentic code work splits the question into **two layers**: a *reward emission* layer (often a scalar with rule-based or judge-based components) and an *advantage/credit assignment* layer that handles normalization, KL regularization, and constraints. DeepSeek-R1's rule-based reward is the dominant "minimal scalar" baseline — only **accuracy + format** components, deliberately avoiding neural reward models and hand-balanced composites because of reward-hacking concerns.[^1] GRPO then handles cross-trajectory normalization at the trainer (advantage = `(r_i − mean) / std` over the group); KL regularization is applied in the loss, *not* by shaping the reward.[^2] When safety/cost is involved, the dominant published pattern is Safe-RLHF: a **separate cost head** with a **dynamic Lagrange multiplier**, not a fixed subtractive penalty inside a clipped composite.[^3] Multi-objective fine-tuning either trains separate models and interpolates weights (Rewarded Soups) or sweeps weighted-sum scalarizations to trace the Pareto front (MORLHF).[^4]

## 2. Direct evidence FOR a subtractive composite reward

A subtractive shape with format/correctness components exists in published RL setups, but the published examples keep it deliberately minimal. DeepSeek-R1 emits a scalar reward that aggregates rule-based components: "Each response receives a scalar reward based on factors like accuracy, formatting, and language consistency."[^1] Notably this is a *small* composite (2–3 rule-based terms), with the language penalty added only after observing language-mixing reward-hacking, and the authors explicitly warn that even this caused "slight degradation in DeepSeek-R1's performance" because the alignment tax of the extra term cannot be cleanly disentangled.[^1] The composite-with-clipping shape itself is also endorsed at the design-principle level by reward-shaping work: bounded rewards prevent runaway exploitation, and rapid-initial-growth-then-convergence shapes outperform unbounded penalties.[^5]

For weighted-sum scalarization more broadly, MORL treats "weights as design parameters dependent on the expertise and preference of the person performing the learning,"[^6] which is exactly what Daydream's `w_len` is inside the composite — a legitimate, if hand-tuned, MORL slice. (`w_fp` is *not* part of this reduction; it is held back as a sibling-axis combination weight — see §4.)

## 3. Direct evidence AGAINST (alternatives preferred)

Three strands of evidence push back on the subtractive-composite shape.

**(a) RULER / judge-as-reward beats hand-crafted multi-component rewards in the agentic regime.** "In 3 out of 4 tasks, models trained with RULER slightly outperform those trained with hand-crafted reward functions"[^7], and the framework can "reduce implementation time by 2–3x compared to hand-crafted rewards"[^8]. For a code-review trajectory setting like Daydream — where the "credit" axes (correctness, grounding) are themselves judge-scored — substituting a single LLM-judge ranking over trajectory groups is a documented stronger baseline than hand-weighting axes.

**(b) Safe-RLHF argues constraint-style decoupling beats fixed subtractive penalties when one axis is a "must-not-violate" signal.** A posterior false-positive penalty driven by maintainer rejection is conceptually a *constraint*, not a reward axis. "This decoupling is essential to avoid policy collapse to trivial refusals (when cost is overemphasized) or unsafe reward maximization (when reward dominates) … Empirically, methods without explicit cost models fail to balance trade-offs and exhibit significantly worse safety metrics."[^3] The same paper notes that fixed-weight scalarization causes "safety compensation" pathology where violations on some inputs are masked by over-caution on others[^3] — which is precisely why the PR's choice to keep `w_fp` *out* of the composite matters: were the FP penalty subtracted inside the clipped mean, an FP-heavy slice of the data could be compensated by inflated grounding scores elsewhere. The risk is deferred, not eliminated, to wherever the sibling axis is later combined under `w_fp` at training time.

**(c) Reward hacking literature warns hand-tuned weights are unstable under policy drift.** "The model learns to optimize the reward … by sounding correct and confident, rather than being correct"[^9], and the standard mitigation is *advantage-sign robustness* / *unified regularization* applied inside the trainer — not heavier reward shaping.[^10] GRPO's design choice to push KL into the loss rather than the reward[^2] is the canonical statement of this principle.

## 4. Verdict

**`contested`.**

The clipped composite shape is *defensible* as a small, bounded MORL scalarization (DeepSeek-R1 precedent[^1], MORL weighted-sum tradition[^6], bounded-reward design principle[^5]). And for the specific axis being added in this PR — a **posterior maintainer-rejection signal** — the field's strong consensus is exactly the shape the PR adopts: this is a constraint, not a reward axis, and should live separately rather than be subtracted inside a clipped weighted mean. The PR keeps the posterior as a sibling field (`posterior_cost` on `PosteriorBreakdown`) and computes the composite from intrinsic axes only, which is the decoupling Safe-RLHF prescribes[^3] and which leaves the door open to group-relative judge ranking over trajectories (RULER[^7][^8]). Because `w_fp` is never applied inside the composite, the *fixed-subtractive* failure modes — safety-compensation pathology[^3] and reward-hacking instability from a hand-tuned penalty weight[^9] — do not arise at the reduction step; cross-trajectory normalization remains the trainer's job, as GRPO prescribes.[^2]

The residual question the verdict turns on is therefore not the composite's shape but the posterior's *downstream* combination: `w_fp` survives as a documented training-time combination weight (pending recalibration), so the constraint-vs-fixed-weight tension reappears wherever the sibling axis is eventually folded into the trainer signal. The field consensus (Safe-RLHF[^3]) favors a dynamic Lagrange multiplier over a fixed `w_fp` at that point.

## 5. Strongest single citation

> "In 3 out of 4 tasks, models trained with RULER slightly outperform those trained with hand-crafted reward functions."
> — OpenPipe ART, *RULER: Relative Universal LLM-Elicited Rewards*, https://art.openpipe.ai/fundamentals/ruler

This is the strongest single piece of evidence because it is a controlled head-to-head between (a) a hand-crafted multi-component reward composition, exactly the shape Daydream uses, and (b) an LLM-judge group-relative alternative, in the *agent training* regime that Daydream targets. The result is not unanimous (1 of 4 tasks goes the other way), which is why the verdict is `contested` rather than `unsupported`.

---

## Footnotes

[^1]: DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948, https://arxiv.org/pdf/2501.12948 — "The reward system consists of two types of rewards: accuracy rewards and format rewards … Each response receives a scalar reward based on factors like accuracy, formatting, and language consistency … We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero, because we find that the neural reward model may suffer from reward hacking" (paraphrased from the paper and corroborated in Aman's AI Journal primer https://aman.ai/primers/ai/deepseek-R1/). Language-consistency reward was added after observing language-mixing reward-hacking; the paper notes "we observe that such alignment results in a slight degradation in the model's performance."

[^2]: Cameron R. Wolfe, *Group Relative Policy Optimization (GRPO)*, https://cameronrwolfe.substack.com/p/grpo — "rewards are normalized by subtracting the group mean and dividing by the group standard deviation, yielding a relative (or whitened) advantage for each output … The KL penalty is applied directly in the loss rather than shaping the reward." Also Emergent Mind, *Guide: Group Relative Policy Optimization (GRPO)*, https://www.emergentmind.com/topics/guide-grpo.

[^3]: Dai et al., *Safe RLHF: Safe Reinforcement Learning from Human Feedback*, ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/dd1577afd396928ed64216f3f1fd5556-Paper-Conference.pdf; summary at Emergent Mind https://www.emergentmind.com/topics/safe-reinforcement-learning-from-human-feedback-safe-rlhf — "This decoupling is essential to avoid policy collapse to trivial refusals (when cost is overemphasized) or unsafe reward maximization (when reward dominates) … Empirically, methods without explicit cost models fail to balance trade-offs and exhibit significantly worse safety metrics … A notable pathology in average-constrained Safe RLHF is 'safety compensation,' where violations for some inputs can be compensated by extreme over-caution for others."

[^4]: Ramé et al., *Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards*, NeurIPS 2023, https://arxiv.org/abs/2306.04488 — "Rewarded soup (RS) is an efficient and flexible multi-policy strategy … first specializing multiple networks independently (one for each proxy reward) and then interpolating their weights linearly … MORL requires multiple trainings on different linear weightings over the rewards (1 − μ) × R₁ + μ × R₂."

[^5]: *Reward Shaping to Mitigate Reward Hacking in RLHF*, arXiv:2502.18770, https://arxiv.org/html/2502.18770v3 — "two key design principles: the RL reward should be bounded, and the RL reward benefits from rapid initial growth followed by gradual convergence." This is the published basis for the `clip(..., 0, 1)` choice.

[^6]: *Predicting optimal value functions by interpolating reward functions in scalarized multi-objective reinforcement learning*, arXiv:1909.05004, https://arxiv.org/pdf/1909.05004 — "A common approach for defining a reward function for multi-objective reinforcement learning (MORL) problems is the weighted sum of the multiple objectives, with the weights treated as design parameters dependent on the expertise and preference of the person performing the learning."

[^7]: OpenPipe ART, *RULER: Relative Universal LLM-Elicited Rewards*, https://art.openpipe.ai/fundamentals/ruler — "In 3 out of 4 tasks, models trained with RULER slightly outperform those trained with hand-crafted reward functions."

[^8]: ibid. — "Can be applied to a wide variety of RL tasks without modification" and (from Vanita.AI summary, https://vanitaai.com/agent-reinforcement-trainer-art-llm-rl/) "reduce implementation time by 2-3x compared to hand-crafted rewards."

[^9]: Rohan Paul, *Reward Hacking in RLHF*, https://www.rohan-paul.com/p/reward-hacking-in-rlhf — "The model learns to optimize the reward (human approval) by sounding correct and confident, rather than being correct."

[^10]: *Mitigating Reward Hacking in RLHF via Advantage Sign Robustness*, arXiv:2604.02986, https://arxiv.org/pdf/2604.02986; *Unifying Stable Optimization and Reference Regularization in RLHF*, arXiv:2602.11523, https://arxiv.org/pdf/2602.11523 — both argue mitigation belongs at the trainer level (advantage / KL regularization) rather than via heavier hand-shaped reward terms.
