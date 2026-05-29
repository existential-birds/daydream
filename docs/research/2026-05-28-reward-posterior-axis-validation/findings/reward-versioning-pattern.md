---
subtopic: reward-versioning-pattern
status: ok
searches_run: 6
fetches_run: 2
---

# REWARD_VERSION as a default-only stamp + tunable weights at call boundary

## 1. Survey of eval-framework versioning

Established eval frameworks treat the version field as a stamp on the **scoring code/config itself**, and require a bump whenever the scoring code changes. None of the public docs surveyed expose a "pass overrides at call time without bumping the stamp" affordance.

**lm-evaluation-harness (EleutherAI).** Task-level VERSION pinned to the YAML/code; replication requires both the config and the codebase commit.

> "Most tasks should include a `version` key in this field that is used to denote the version of the yaml config."
> "These YAML configuration files, along with the current codebase commit hash, are intended to be shareable such that providing the YAML config enables another researcher to precisely replicate the evaluation setup used by another"
> Earlier guidance also states: "if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. Task versions start at 0, and each time a breaking change is made, the version is incremented by one."
> Source: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md

**OpenAI Evals.** Version is part of the eval identifier and explicitly tied to "same eval name should give similar results":

> "The naming convention for evals is in the form `<eval_name>.<split>.<version>`."
> "In general, running the same eval name against the same model should always give similar results so that others can reproduce it. Therefore, when you change your eval, you should bump the version."
> Source: https://github.com/openai/evals/blob/main/docs/build-eval.md

**HELM (Stanford CRFM).** Reproducibility achieved by releasing the full prompt/output trace per (model, scenario), not by a tunable weight knob — the scenario *is* the contract.

**AlpacaEval.** Annotators (judge) are config files; changing the judge means a new `annotators_config` (e.g. `weighted_alpaca_eval_gpt4_turbo`), i.e. a new named artifact rather than runtime overrides on a "default" judge.

**RewardBench.** Curated dataset + code; reproducibility is the leaderboard contract, with private-model scores explicitly excluded from the paper because they are not reproducible.

## 2. Default-pin vs config-hash patterns

Two patterns appear in the wild:

- **Default-pin (lm-eval-harness, OpenAI Evals, AlpacaEval).** Version is a human-assigned token attached to one canonical config; non-canonical runs are expected to be renamed (new YAML, new annotator config). Bug-bump discipline is enforced by code review / docs convention, not by the runtime.
- **Config-hash / provenance (MLflow, W&B, Hydra-style configs).** Every run logs its full hyperparameter set + code commit; identity is the (params_hash, code_hash) pair, not a human-assigned stamp. MLflow logs hyperparameters via `mlflow.log_params()` and dataset hashes are explicitly recommended for reproducibility.

The default-pin pattern catches the **"scoring code drifted"** bug well (it's the whole point of `lm-eval-harness`'s bump rule). It does **not** catch the **"caller passed custom weights and stored the result"** bug — that is a class the surveyed eval frameworks largely sidestep by not exposing weight overrides at the call boundary at all. Frameworks that do allow runtime hyperparameters (MLflow/W&B) compensate by hashing the *actual* params used into the run identity.

## 3. Risk of "non-canonical" scores leaking

The Daydream pattern (one version string + a `RewardWeights` dataclass overridable at the call boundary, with a docstring saying "don't store non-default scores as canonical") is **load-bearing on the docstring**. Concretely:

- Nothing on the produced score (a float) carries the weights that produced it.
- The storage layer (corpus / trajectory / harvest output) has no in-band signal it can refuse on.
- A future caller doing a sensitivity sweep can trivially write the override result into the same column as canonical scores; the only guard is reviewer attention.

This matches a known anti-pattern in the MLOps survey results: identity-by-name without identity-by-content lets divergent runs share a label. Whereas `lm-eval-harness` and OpenAI Evals dodge this by making weights part of the YAML (so any change forces a rename), Daydream has chosen to keep weights as a Python-level parameter — a flexibility the surveyed frameworks deliberately did not offer.

## 4. Concrete recommendation

**Strengthen, don't replace.** Two minimally invasive options, ordered by effort:

1. **(Cheap, recommended.)** Make `RewardWeights` carry an explicit `is_default: bool` flag set by `DEFAULT_WEIGHTS` only, and have `score_trajectory` return a `(score, version_stamp)` tuple where `version_stamp` is `REWARD_VERSION` for defaults and `f"{REWARD_VERSION}+custom-{hash}"` for overrides (config-hash pattern, suffix form). Storage code can then assert `version_stamp == REWARD_VERSION` before persisting to the canonical column. This converts the docstring-only guard into a runtime contract and matches the OpenAI Evals convention of putting the variant into the identifier.
2. **(More invasive.)** Forbid call-boundary overrides entirely; require sensitivity sweeps to go through a separate `score_trajectory_experimental(weights=...)` entrypoint that returns a distinct type (`ExperimentalScore`) that the storage layer cannot accept. This is what `lm-eval-harness` effectively does by making weight changes a config-file change.

Either change preserves the analysis-time use case while removing the "looks identical to a canonical score" failure mode.

## 5. Verdict

**contested.** The default-pin half of the pattern is well-supported by `lm-evaluation-harness` and OpenAI Evals. The "expose weight overrides at the call boundary, gated only by a docstring" half is **not** standard in surveyed eval frameworks — they keep weights inside the versioned config so overrides force a rename. The MLOps tradition (MLflow/W&B) handles runtime parameter variation but compensates by hashing params into run identity. Daydream's hybrid (runtime overrides + a single human stamp + docstring guard) is the weakest combination of the two traditions.

## 6. Strongest single citation

OpenAI Evals build guide, which most directly captures the principle Daydream's pattern bends:

> "In general, running the same eval name against the same model should always give similar results so that others can reproduce it. Therefore, when you change your eval, you should bump the version."
> — https://github.com/openai/evals/blob/main/docs/build-eval.md

## Sources

- https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
- https://github.com/openai/evals/blob/main/docs/build-eval.md
- https://github.com/stanford-crfm/helm
- https://github.com/tatsu-lab/alpaca_eval
- https://github.com/allenai/reward-bench
- https://arxiv.org/abs/2403.13787
- https://www.dailydoseofds.com/mlops-crash-course-part-4/
- https://gerben-oostra.medium.com/semantic-versioning-for-ml-models-8315d03907bf
