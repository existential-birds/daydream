# daydream
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Daydream is a code-review agent that produces structured training data from its own runs. It reviews diffs using stack-specific [Beagle](https://github.com/existential-birds/beagle) skills, applies fixes, validates via test suite, and records every agent interaction as an [ATIF v1.6](https://www.harborframework.com/docs/agents/trajectory-format) trajectory. A bitemporal corpus pipeline then scores, labels, and projects those trajectories into JSONL datasets for SFT and RL fine-tuning.

The goal is an open-weight code-review model (Qwen2.5-Coder-7B, QLoRA) trained on daydream's own trajectory archive, independently benchmarked against leading commercial code-review bots on a held-out PR replay corpus. See [Milestone 1](https://github.com/existential-birds/daydream/issues/86) for the current training roadmap.

![demo](https://github.com/user-attachments/assets/60a80645-36de-410e-afa7-7a96efef3f57)

## Architecture

### Review Pipeline

Two modes: deep (default) and shallow (`--shallow`).

**Deep review** runs a five-stage pipeline:

1. **Exploration pre-scan**: tree-sitter import resolution and convention detection across the changed files
2. **Intent analysis**: understands the diff and commit history to build context
3. **Alternative review**: identifies potential improvements as numbered findings
4. **Per-stack reviews**: parallel Beagle skill invocations, one per detected stack (Python, TypeScript, Go, Rust, Elixir, iOS)
5. **Cross-stack merge**: deduplicates and synthesizes per-stack findings into a unified report

After merge, an optional fix gate applies fixes one-by-one and validates with the project's test suite.

**Shallow review** (`--shallow`) runs a single-skill loop: review → parse → fix → test. Useful for single-stack projects or when you want to force a specific Beagle skill.

### Trajectory Recording

Every run produces an ATIF v1.6 trajectory at `<target>/.daydream/runs/<id>/trajectory.json` capturing prompts, responses, tool calls, observations, and per-step token/cost metrics. Parallel fan-outs (per-stack reviews, parallel fixes) produce sibling trajectories via `recorder.fork()` under `.daydream/runs/<id>/trajectories/`. Sensitive content (API keys, JWTs, URL credentials, `.env` values) is redacted before writing. Interrupted runs flush a `.partial` file with `extra.partial=true`.

### Corpus Pipeline

The training data pipeline converts archived trajectories into fine-tuning datasets through three stages:

**Harvest** (`daydream harvest`): Walks the archive index, assembles bronze signals (recommendation-verifier verdicts, per-stack finding records, grounding rate, review length), scores an intrinsic reward per run, derives the outcome label, and appends one bitemporal annotation. Re-running appends a new generation rather than overwriting, so older `as_of` pins still resolve their original scores after a reward-version bump.

**Reward scoring** (`daydream/training/reward.py`): Pure composite reducer over capture-time signals. Formula:

- **correctness** (w=0.6): mean over per-finding verifier verdicts (`consistent→1.0`, `uncertain→0.5`, `contradicts→0.0`)
- **grounding** (w=0.4): `grounding_rate ∈ [0,1]` passed through
- **format_valid**: dominating gate; `False` floors composite to 0.0 (an internal design choice, in the spirit of the DeepSeek-R1 accuracy+format reward — arXiv:2501.12948)
- **length**: bounded saturating penalty (w=0.2), subtracted from credit mean

Composite = `round(clip(credit − w_len·len_norm, 0, 1), 4)` — a **pure intrinsic** score, where credit is the weighted mean over present credit axes, renormalized. Missing signals become `None`, never imputed as 0.

The posterior false-positive axis is **not** folded into the composite. When a mapped maintainer accept/reject label is supplied, `score_trajectory` returns a `PosteriorBreakdown` (a subclass of `RewardBreakdown`) carrying `posterior_cost` as a **sibling field** of `composite` — `posterior_cost = max(0.0, observed_penalty − outcome_prior)`, the calibrated surprise above the reviewers' mean observed penalty on the `[0,1]` penalty scale (`rejected→1.0`, `contested→0.5`, `accepted→0.0` — see `_FP_PENALTY_MAP` in `daydream/training/reward.py`; `outcome_prior` defaults to the `0.5` max-entropy midpoint when uncalibrated). The `composite` field is identical whether or not the posterior is present. `w_fp` (default 0.3) survives on `RewardWeights` as a documented training-time combination weight pending recalibration (#114) — it is **never** applied inside the composite.

`has_posterior` is the population discriminator: intrinsic-only runs return a plain `RewardBreakdown` (no posterior fields), labeled runs return a `PosteriorBreakdown`. The two populations require different treatment, so aggregate consumers in `training/` must filter on `has_posterior` (the `runs` index mirrors it as an `INTEGER` (0/1) column) rather than mixing labeled and unlabeled rows (C3).

**Build corpus** (`daydream build-corpus`): Projects the `as_of`-pinned silver annotations into JSONL training records. Filters by outcome label (default: `accepted` only), reward threshold, stack stratification, exclusion list (benchmark repos), and copyleft license opt-in. Writes a `lineage.json` manifest with content-addressed `trajectory_set_hash`, labeler/reward versions, and the `as_of` pin for byte-for-byte reproducibility. A temporal-leakage guard drops annotations whose `valid_at` (e.g., PR merge timestamp) is posterior to the `as_of` pin.

### Archive

Unless `--no-archive` is passed, each run is archived to `~/.daydream/archive/runs/{session_id}/` with manifest, trajectory, review output, evaluation results, and deep artifacts. A SQLite index at `~/.daydream/archive/index.db` supports querying by repo, backend, cost, grounding rate, and outcome labels.

### Training Roadmap

The [Milestone 1 epic](https://github.com/existential-birds/daydream/issues/86) tracks an open-weight code-review model (Qwen2.5-Coder-7B-Instruct, QLoRA rank 32/alpha 64, 4-bit) trained via a staged recipe:

1. **RFT**: rejection-filter the corpus using the composite reward as a threshold
2. **Span-segmented SFT**: SAD-style segment-specific losses on ATIF REASON/ACT spans (per arXiv:2505.13820)
3. **KTO**: preference-train on PR-comment accept/reject labels (per arXiv:2402.01306), with synthetic-accept balancing for label imbalance

Target bar for the trained model on a held-out PR replay benchmark: Scoring follows [Martian's Code Review Bench](https://github.com/withmartian/code-review-benchmark), evaluated on its five-repo set (Sentry, Grafana, Cal.com, Discourse, Keycloak) plus additional OSS:

| Metric | Target |
|--------|--------|
| Precision (offline real PRs) | ≥50% |
| F1 (Martian-bench scoring) | ≥51% |
| Addressed comments per PR | ≥1.5 |
| False positives per 50-PR run | ≤4 |

If the recipe misses precision, F1, or addressed-comments targets, [Milestone 2](https://github.com/existential-birds/daydream/issues/102) triggers automatically with GRPO + composite verifiable reward.

## Quickstart

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and [Claude Code](https://claude.ai/code) CLI.

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv sync
```

Install the [Beagle](https://github.com/existential-birds/beagle) plugin (provides the stack-specific review skills):

```bash
claude plugin marketplace add https://github.com/existential-birds/beagle
claude plugin install beagle
```

Optional: [GitHub CLI](https://cli.github.com/) (`gh`) for PR feedback and `--comment` mode. [Codex CLI](https://openai.com/codex) for `--backend codex`.

Run a review:

```bash
daydream /path/to/project                     # deep multi-stack review-fix-test
daydream --shallow /path/to/project           # single-stack loop
daydream --review /path/to/project            # report only, no fixes
daydream --comment --branch feat/x /path/to/project  # post inline PR comments
daydream feedback 42 --bot "<bot-login>[bot]" /path/to/project  # fix bot PR comments
```

To update: `git pull && uv sync`

## CLI Reference

### Output Modes

| Flag | Behavior |
|------|----------|
| _(default)_ | Deep multi-stack review → fix → test loop |
| `--shallow` | Single-stack review → parse → fix → test |
| `--review` | Write report to terminal/markdown, then exit |
| `--comment` | Post inline PR comments, then exit |
| `--comment --plan` | Post comments + implementation plan |

### Corpus Commands

```bash
daydream harvest                              # annotate all archived runs (reward + label)
daydream harvest --dry-run                    # preview without writing
daydream build-corpus --out /path/to/out.jsonl  # project labeled runs to JSONL
daydream build-corpus --out out.jsonl --min-reward 0.5 --label accepted --label mixed
daydream build-corpus --out out.jsonl --as-of 2026-05-01T00:00:00Z  # pinned snapshot
daydream label <session_id> --accepted        # manual outcome label override
```

### Common Options

```bash
daydream -s python /path/to/project           # force a specific Beagle skill
daydream --backend codex /path/to/project     # use Codex instead of Claude
daydream --review-model claude-opus-4-6 /path/to/project
daydream --start-at fix /path/to/project      # resume from a specific phase
daydream --loop --max-iterations 3 /path/to/project
daydream --trajectory /tmp/run.json /path/to/project
daydream --ignore-path vendor /path/to/project
daydream --worktree /path/to/project          # force ephemeral worktree
```

Per-phase backend and model overrides: `--review-backend`, `--fix-backend`, `--test-backend`, `--review-model`, `--parse-model`, `--fix-model`, `--test-model`, `--exploration-model`. Run `daydream --help` for the full option list and per-backend model defaults.

## Output Files

| Path | Description |
|------|-------------|
| `.daydream/runs/<id>/trajectory.json` | ATIF v1.6 trajectory (customize with `--trajectory`) |
| `.daydream/runs/<id>/trajectories/` | Forked sub-trajectories from parallel fan-outs |
| `.daydream/diff.patch` | Unified diff captured at run start |
| `.daydream/deep/` | Deep pipeline artifacts: intent, per-stack reviews, merged report |
| `.review-output.md` | Review findings (removed with `--cleanup`) |
| `~/.daydream/archive/runs/<id>/` | Archived run: manifest, trajectory, review output, evaluation, deep artifacts |
| `~/.daydream/archive/index.db` | SQLite index for cross-project querying |

## Development

```bash
make install    # install dependencies
make hooks      # install git hooks
make lint       # ruff linter
make typecheck  # mypy
make test       # pytest
make check      # all CI checks
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
