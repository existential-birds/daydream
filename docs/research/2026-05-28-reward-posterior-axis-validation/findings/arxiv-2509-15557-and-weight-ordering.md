---
subtopic: arxiv-2509-15557-and-weight-ordering
status: ok
arxiv_verified: true
searches_run: 4
fetches_run: 3
---

# arXiv:2509.15557 verification and weight-ordering claim audit

## 1. arXiv verification

**Verified — paper exists.**

- **Title:** *Reward Hacking Mitigation using Verifiable Composite Rewards*
- **Authors:** Mirza Farhan Bin Tarek, Rahmatollah Beheshti
- **arXiv ID:** 2509.15557 (v1, September 2025)
- **Venue:** Also appears in ACM DL (10.1145/3765612.3767230)
- **URLs fetched:**
  - https://arxiv.org/abs/2509.15557 (abstract page — 200 OK)
  - https://www.arxiv.org/pdf/2509.15557 (PDF — 200 OK)
  - https://arxiv.org/html/2509.15557v1 (HTML render — 200 OK)

No redirect; no 404. Title/authors/abstract all consistent across pages.

## 2. "Improper format → maximum penalty regardless of correctness" claim

**Claim NOT found in paper.**

Direct fetches of the HTML render (`arxiv.org/html/2509.15557v1`) searching for the phrases "improper format," "maximum penalty," and "regardless of correctness" returned no matches. The paper's penalty structure is *additive subtraction*, not a *gate*:

> "R_total(g) = w_b R_binary(g) - w_a P_answer(g) - w_s P_structural(g)"

`P_structural` is a **fixed scalar penalty** applied when preamble word count exceeds a threshold, not a multiplicative floor:

> "P_structural(g) = {λ_s if |T_preamble| > τ_preamble; 0 otherwise} … A fixed penalty λ_s is applied in such cases."

Nothing in the paper says a format violation overrides or floors the correctness component. Format and answer-leak penalties are subtracted from `w_b R_binary`; in principle a correct answer with a format violation still earns positive `R_binary` credit (`R_binary` is undocumented in our excerpts but is a verifiable-reward term).

**The daydream docstring's framing — `format_valid=False` as a dominating gate that floors composite to 0 — is NOT what this paper does.** The paper *subtracts* penalties; it does not *gate*.

## 3. Weight-ordering claim (w_b=1.0 > w_a=0.5 > w_s=0.3)

**Numerics verified verbatim. Justification absent.**

Verbatim quote from §experiments:

> "we train the LLM with RLVR and our proposed composite reward model for one epoch with 1,000 samples from the dataset. The hyperparameters for the reward model were chosen as w_b=1.0, w_a=0.5, w_s=0.3."

What the symbols denote (from the composite reward equation):

- **w_b** — weight on `R_binary`, the **verifiable correctness reward** (credit weight).
- **w_a** — weight on `P_answer`, penalty for **answer-leak inside the reasoning block** (cosine similarity to leak phrases above threshold τ_answer). A penalty weight.
- **w_s** — weight on `P_structural`, penalty for **preamble exceeding word-count threshold τ_preamble**. A penalty weight.

**Critical mismatch with the daydream citation:** in the paper, **w_b is the credit weight and w_a, w_s are BOTH penalty weights.** The paper is **not** establishing a "credit > penalty" ordering — it's stating one credit weight is larger than two penalty weights. There is no "smallest credit weight > penalty weight" rule to import.

Additionally, the paper offers **no theoretical or ablation-based justification** for the chosen ordering. The values are presented as tuned hyperparameters, with no defense of why w_a > w_s, no sensitivity analysis, and no claim that this ordering generalizes.

## 4. Domain match

- **Task:** Medical multiple-choice question answering (USMLE-style).
- **Models fine-tuned:** Llama 3.2-3B-Instruct and Qwen2.5-3B-Instruct.
- **Setup:** RLVR (Reinforcement Learning from Verifiable Rewards) over 1 epoch, 1,000 samples.
- **Penalty targets:** Two specific medical-QA reward-hacking pathologies — (i) answer leaking inside the `<think>` block, (ii) prose outside the reasoning tags ("structural non-compliance"). Both are surface-form pathologies of a CoT-tagged chat completion.

**Generalization to code-review reward design is weak.** The paper's penalties target chat-format violations in a tagged-CoT medical-QA setup with binary correctness. Daydream's `reward.py` deals with code-review trajectories where "false positive" is a *semantic* judgment about a posterior PR outcome, not a *syntactic* word-count or cosine-similarity check. The penalty mechanics (fixed λ if word count exceeds threshold) don't map onto reward-posterior-axis design.

## 5. Independent corroboration

Broader literature on penalty-vs-credit weight ordering in composite reward functions:

- **No published "penalty weight strictly below the smallest credit weight" norm found.** Searches across composite-reward design papers (ENCORE, RLMR, PPO-driven adaptive filtering, biomechanical RL guidelines) surface a different consensus: weights are *task-tuned hyperparameters*, often dynamic/adaptive, with no universal ordering rule.
- One source notes: "writing quality scores and constraint verification signals operate on different scales and distributions, making it difficult to determine appropriate weighting coefficients" — i.e., fixed-weight ordering rules are explicitly discouraged in recent work (e.g., RLMR, arXiv:2508.18642).
- The closest published heuristic is "correctness has the highest weight" (gated weighted sum), which 2509.15557 does follow — but this is a *credit dominance* rule, not a *penalty-below-credit* rule.

The "w_fp = 0.3 must sit strictly below the credit weights" framing in the daydream PR is **idiosyncratic to this single paper's hyperparameter table**, not a published norm.

## 6. Verdict

**`citation-misquoted`** (with a domain-mismatch overlay).

Specifically:

1. The paper exists and the numeric weights (1.0 / 0.5 / 0.3) are verbatim — that part is sound.
2. **The "improper format → maximum penalty regardless of correctness" claim is not in the paper.** The paper subtracts penalties; it does not gate. Citing it to justify a format-gate flooring composite to 0 is a misquote.
3. **The "w_b > w_a > w_s" ordering is being misread.** The paper orders ONE credit weight above TWO penalty weights of different kinds; it does not establish a "smallest credit weight > any penalty weight" rule. Using it to justify w_fp=0.3 sitting "strictly below the credit weights" reads structure into the table that the authors didn't claim.
4. **Domain is medical USMLE-style QA on 3B chat models with surface-form penalties** — generalizing to code-review reward design where the penalized event is a semantic posterior PR outcome is not defensible from this paper alone.

**Recommendation:** either drop the citation, or rewrite the docstring to (a) accurately describe the paper as a *subtractive* composite reward (not a gate), (b) acknowledge the weight ordering is a tuned hyperparameter choice in a medical-QA paper, not a derived rule, and (c) cite the daydream-specific reasoning (or an ablation) for why w_fp=0.3 here.

## Sources

- [arXiv:2509.15557 abstract](https://arxiv.org/abs/2509.15557)
- [arXiv:2509.15557 HTML](https://arxiv.org/html/2509.15557v1)
- [arXiv:2509.15557 PDF](https://www.arxiv.org/pdf/2509.15557)
- [RLMR: Reinforcement Learning with Mixed Rewards (arXiv:2508.18642)](https://arxiv.org/html/2508.18642v1)
- [ENCORE entropy-guided reward composition (arXiv:2503.20995)](https://arxiv.org/pdf/2503.20995)
- [Composite Reward Design in PPO-Driven Adaptive Filtering (arXiv:2506.06323)](https://arxiv.org/html/2506.06323v1)
