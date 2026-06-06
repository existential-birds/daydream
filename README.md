# daydream
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/existential-birds/daydream)

Daydream is a code-review agent that produces structured training data from its own runs. It reviews diffs using stack-specific [Beagle](https://github.com/existential-birds/beagle) skills, applies fixes, validates via test suite, and records every agent interaction as an [ATIF v1.6](https://www.harborframework.com/docs/agents/trajectory-format) trajectory. A bitemporal corpus pipeline then scores, labels, and projects those trajectories into JSONL datasets for SFT and RL fine-tuning.

The goal is an open-weight code-review model (Qwen2.5-Coder-7B, QLoRA) trained on daydream's own trajectory archive, benchmarked against commercial code-review bots on a held-out PR replay corpus.

![demo](https://github.com/user-attachments/assets/60a80645-36de-410e-afa7-7a96efef3f57)

## Architecture

### Review Pipeline

Two modes: deep (default) and shallow (`--shallow`).

**Deep review** runs a five-stage pipeline:

1. **Exploration pre-scan**: tree-sitter import resolution and convention detection
2. **Intent analysis**: understands the diff and commit history
3. **Alternative review**: identifies improvements as numbered findings
4. **Per-stack reviews**: parallel Beagle skill invocations, one per detected stack (Python, TypeScript, Go, Rust, Elixir, iOS)
5. **Cross-stack merge**: deduplicates per-stack findings into a unified report

After merge, an optional fix gate applies fixes one-by-one and validates with the project's test suite.

**Shallow review** (`--shallow`) runs a single-skill loop: review → parse → fix → test. Useful for single-stack projects or when you want to force a specific Beagle skill.

### Trajectory Recording

Every run produces an [ATIF v1.6](https://www.harborframework.com/docs/agents/trajectory-format) trajectory at `<target>/.daydream/runs/<id>/trajectory.json` capturing prompts, responses, tool calls, and per-step token/cost metrics. Parallel fan-outs fork sibling trajectories under `.daydream/runs/<id>/trajectories/`; secrets are redacted before writing and interrupted runs flush a `.partial` file.

### Corpus Pipeline

The training data pipeline converts archived trajectories into fine-tuning datasets through three stages:

**Harvest** (`daydream corpus harvest`): Walks the archive index, assembles bronze signals (verifier verdicts, finding records, grounding rate, review length), scores an intrinsic reward, derives the outcome label, and appends a bitemporal annotation. Re-running appends a new generation rather than overwriting, so older `as_of` pins still resolve their original scores.

**Reward scoring** (`daydream/training/reward.py`): a pure composite over capture-time signals:

- **correctness** (w=0.6): mean over per-finding verifier verdicts (`consistent→1.0`, `uncertain→0.5`, `contradicts→0.0`)
- **grounding** (w=0.4): `grounding_rate ∈ [0,1]`
- **format_valid**: dominating gate — `False` floors the composite to 0.0 (after DeepSeek-R1, arXiv:2501.12948)
- **length**: bounded saturating penalty (w=0.2)

Composite = `round(clip(credit − w_len·len_norm, 0, 1), 4)`. Missing signals are `None`, never imputed as 0. When a maintainer accept/reject label is supplied, scoring also returns a posterior false-positive cost as a sibling field — it never alters the composite. See `reward.py` for the posterior breakdown and weight details.

**Build corpus** (`daydream corpus build`): Projects `as_of`-pinned silver annotations into JSONL training records, filtered by outcome label, reward threshold, stack, exclusion list, and license. Writes a `lineage.json` manifest (content-addressed `trajectory_set_hash`, labeler/reward versions, `as_of` pin) for byte-for-byte reproducibility, with a temporal-leakage guard that drops annotations newer than the pin.

### Archive

Unless `--no-archive` is passed, each run is archived to `~/.daydream/archive/runs/{session_id}/` with manifest, trajectory, review output, evaluation results, and deep artifacts. A SQLite index at `~/.daydream/archive/index.db` supports querying by repo, backend, cost, grounding rate, and outcome labels.

### Training Roadmap

The [Milestone 1 epic](https://github.com/existential-birds/daydream/issues/86) tracks an open-weight code-review model (Qwen2.5-Coder-7B-Instruct, QLoRA rank 32/alpha 64, 4-bit) trained via a staged recipe:

1. **RFT**: rejection-filter the corpus by composite-reward threshold
2. **Span-segmented SFT**: SAD-style losses on ATIF REASON/ACT spans (arXiv:2505.13820)
3. **KTO**: preference-train on PR-comment accept/reject labels (arXiv:2402.01306), with synthetic-accept balancing

Targets follow [Martian's Code Review Benchmark](https://github.com/withmartian/code-review-benchmark), scored on its five-repo set (Sentry, Grafana, Cal.com, Discourse, Keycloak) plus additional OSS:

| Metric | Target |
|--------|--------|
| Precision (offline real PRs) | ≥50% |
| F1 (Martian-bench scoring) | ≥51% |
| Addressed comments per PR | ≥1.5 |
| False positives per 50-PR run | ≤4 |

Missing the precision, F1, or addressed-comments targets auto-triggers [Milestone 2](https://github.com/existential-birds/daydream/issues/102) (GRPO + composite verifiable reward).

### Benchmarking

`daydream bench` scores deep-review findings against [Martian's Code Review Benchmark](https://github.com/withmartian/code-review-benchmark) offline set and writes per-PR precision/recall into `results/<model>/evaluations.json`. See the [benchmark runbook](docs/benchmark.md) for the full setup-to-result sequence.

## Quickstart

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and [Claude Code](https://claude.ai/code) CLI.

```bash
git clone https://github.com/existential-birds/daydream.git
cd daydream
uv sync
```

Install the [Beagle](https://github.com/existential-birds/beagle) plugin:

```bash
claude plugin marketplace add https://github.com/existential-birds/beagle
claude plugin install beagle
```

Optional: [GitHub CLI](https://cli.github.com/) (`gh`) for PR feedback and `--comment` mode. [Codex CLI](https://openai.com/codex) for `--backend codex`.

### Golden paths

Two near-zero-flag entry points cover the common cases:

```bash
daydream /path/to/project            # review → fix → test (deep multi-stack)
daydream --comment /path/to/project  # review → post inline PR comments, then exit
```

`daydream /path` is the default verb; `daydream review /path` is identical. The
remaining surface is opt-in:

```bash
daydream --review /path/to/project            # write a report to terminal/markdown, no fixes
daydream --shallow /path/to/project           # single-stack review → parse → fix → test loop
daydream --yes /path/to/project               # auto-apply fixes without prompting
daydream --loop /path/to/project              # repeat review-fix-test until clean (or 5 rounds)
daydream feedback 42 --bot "<bot-login>[bot]" /path/to/project  # fix bot PR comments
```

Run `daydream --help` for the common flags and `daydream --help-all` for the full
advanced surface (`--start-at`, `--ignore-path`, `--worktree`, `--trajectory`, …).

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

The data-pipeline verbs live under the `corpus` namespace:

```bash
daydream corpus harvest                              # annotate all archived runs (reward + label)
daydream corpus harvest --dry-run
daydream corpus build --out /path/to/out.jsonl       # project labeled runs to JSONL
daydream corpus build --out out.jsonl --min-reward 0.5 --include-all-labels
daydream corpus build --out out.jsonl --as-of 2026-05-01T00:00:00Z  # pinned snapshot
daydream corpus label <session_id> --outcome accepted  # manual outcome label override
```

### Common Options

```bash
daydream -s python /path/to/project           # force a specific Beagle skill
daydream --backend codex /path/to/project     # override backend for this run
daydream --model claude-opus-4-8 /path/to/project  # overrides ALL phases (beats config-file overrides)
daydream --loop 3 /path/to/project            # repeat up to 3 review-fix-test rounds
daydream --yes /path/to/project               # auto-apply fixes without prompting
```

Advanced flags (hidden from `--help`, shown by `--help-all`, all still parse):

```bash
daydream --start-at fix /path/to/project      # resume from a specific phase
daydream --trajectory /tmp/run.json /path/to/project
daydream --ignore-path vendor /path/to/project
daydream --worktree /path/to/project          # force ephemeral worktree
daydream --non-interactive /path/to/project   # run unattended; take every prompt's safe default
```

`--non-interactive` takes each prompt's safe default: on test failure it writes a `handoff.md` and exits non-zero instead of looping, otherwise it declines fixes and exits 0. It is orthogonal to `--yes`: `--non-interactive` controls *whether* daydream may block on stdin, while `--yes` pre-decides every yes/no gate as "yes". A non-TTY or CI environment (`CI` set) auto-enables non-interactive mode without the flag.

Per-phase model and backend overrides are no longer CLI flags — set them in the config file (see [Configuration](#configuration)).

## Configuration

Per-phase model/backend selection and global defaults live in a config file, read from the **target repo root** at two sources, merged per-key (the dotfile wins on scalar conflicts):

- `pyproject.toml` under `[tool.daydream]` (lower precedence)
- `.daydream.toml` at the repo root, using bare top-level keys (higher precedence)

```toml
# pyproject.toml  →  [tool.daydream]
[tool.daydream]
model = "claude-opus-4-8"     # global default across phases
backend = "claude"            # global default backend

[tool.daydream.phases.fix]    # per-phase override
backend = "codex"
model = "gpt-5.5"

[tool.daydream.phases.review]
model = "claude-opus-4-8"
```

```toml
# .daydream.toml  (top-level keys; no [tool.daydream] prefix)
model = "claude-opus-4-8"

[phases.fix]
backend = "codex"
```

Phase names: `exploration`, `review`, `parse`, `fix`, `test`, `verify`, `merge` (plus
`intent`, `wonder`, `envision`, `pr_feedback`). Resolution precedence, highest first:

**CLI > config file (phase, then global) > built-in per-backend default.**

So `--model` beats a `[tool.daydream.phases.*]` override, which beats the
per-backend table in `daydream/config.py`. The same order applies to backend
selection via `--backend` / config / the `claude` fallback. (There is no
environment-variable tier — `DAYDREAM_MODEL`/`DAYDREAM_BACKEND` are not read.)

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
make install
make hooks      # install git hooks
make lint       # ruff linter
make typecheck  # mypy
make test       # pytest
make check      # all CI checks
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
